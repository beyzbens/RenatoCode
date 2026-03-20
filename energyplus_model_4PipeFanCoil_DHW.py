import os
import numpy as np
from gym import spaces
from gym_energyplus.envs.energyplus_model import EnergyPlusModel

class EnergyPlusModel4PipeFanCoilDHW(EnergyPlusModel):
    # ----------------------------------------------------------------
    # EnergyPlus output variables - MATCHED EXACTLY TO YOUR IDF
    # ----------------------------------------------------------------
    OBS_VARIABLES = [
        ("Site Outdoor Air Drybulb Temperature", "Environment"),          # [0]
        ("Zone Mean Air Temperature", "GF_Living_Room"),                   # [1]
        ("Zone Mean Air Temperature", "GF_Kitchen"),                       # [2]
        ("Zone Mean Air Temperature", "UF_Bedroom"),                       # [3]
        ("Zone Mean Air Temperature", "UF_Bathroom"),                      # [4]
        ("Zone Mean Air Temperature", "UF_Storage_Room"),                  # [5]
        ("Water Heater Tank Temperature", "BufferTank"),                  # [6]
        ("Water Heater Tank Temperature", "DHW_Tank"),                     # [7]
        ("Facility Total Electricity Demand Rate", "Whole Building"),        # [8] (Matched to IDF)
        ("Facility Total HVAC Electricity Demand Rate", "Whole Building"),   # [9] (Matched to IDF)
        ("Water Heater Heating Energy", "BufferTank"),                     # [10]
        ("Water Heater Heating Energy", "DHW_Tank"),                       # [11]
        ("Site Outdoor Air Relative Humidity", "Environment"),             # [12]
    ]

    # --- ACTUATORS: Matched to your IDF Schedule Names ---
    ACTUATOR_CONFIG = {
        'heating_setpoint': [("Schedule:Constant", "Schedule Value", "HTG_Setpoint_Sch")],
        'cooling_setpoint': [("Schedule:Constant", "Schedule Value", "CLG_Setpoint_Sch")],
        'hw_supply_temp': [("Schedule:Compact", "Schedule Value", "HW_SupplyTemp_Sch")],
        'dhw_setpoint': [("Schedule:Compact", "Schedule Value", "DHW_TSET_SCH")],
    }

    DHW_MIN_TEMP = 45.0
    DHW_NOMINAL_TEMP = 55.0

    def __init__(self, model_file, log_dir, config=None, verbose=False):
        self.reward_low_limit = -10000.
        if config is None: config = {}
        
        # External Context Signals
        self.grid_excess_input = 0.5
        self.electricity_price_input = 0.15

        # --- YENİ ESNEKLİK (DEADBAND) AYARLARI ---
        # 20°C - 26°C arası deadband için merkez 23, tolerans 3 olmalı.
        self.temperature_center = config.get('temp_center', 23.0) 
        self.temperature_tolerance = config.get('temp_tolerance', 3.0) 
        self.dhw_target = config.get('dhw_target', self.DHW_NOMINAL_TEMP)
        
        self.w_comfort = config.get('w_comfort', 2.0)
        self.w_grid_flex = config.get('w_grid_flex', 1.0)
        self.w_energy_cost = config.get('w_energy_cost', 0.5)
        
        self.hvac_max_power_w = 21000.0
        self.action = np.zeros(4)
        self.action_prev = np.zeros(4)
        
        super(EnergyPlusModel4PipeFanCoilDHW, self).__init__(model_file, log_dir, verbose)

    def get_obs_variables(self): return self.OBS_VARIABLES
    def get_actuator_config(self): return self.ACTUATOR_CONFIG

    def setup_spaces(self):
        # Actions: [HTG_SP, CLG_SP, HW_TEMP, DHW_SP]
        # HTG_SP'nin üst sınırını 26.0, CLG_SP'nin alt sınırını 20.0 yaptık!
        self.action_space = spaces.Box(
            low=np.array([15.0, 20.0, 30.0, 45.0], dtype=np.float32),
            high=np.array([26.0, 28.0, 80.0, 65.0], dtype=np.float32),
        )
        # Observations: 13 variables
        self.observation_space = spaces.Box(
            low=np.array([-10, 5, 5, 5, 5, 5, 10, 10, 0, 0, -1e6, -1e6, 0], dtype=np.float32),
            high=np.array([45, 40, 40, 40, 40, 40, 95, 70, 5e5, 5e5, 1e9, 1e9, 100], dtype=np.float32),
        )

    def set_raw_state(self, raw_state):
        if raw_state is not None:
            self.raw_state = raw_state
        else:
            self.raw_state = np.zeros(len(self.OBS_VARIABLES), dtype=np.float32)

    def apply_action(self, ep_state, action, actuator_handles, api):
        htg_sp = float(action[0])
        clg_sp = float(action[1])

        # --- SAFETY GUARD: Prevent Setpoint Overlap (Avoids FATAL error) ---
        if htg_sp >= (clg_sp - 0.5):
            clg_sp = htg_sp + 0.5
        # ------------------------------------------------------------------

        mapping = {
            'heating_setpoint': htg_sp,
            'cooling_setpoint': clg_sp,
            'hw_supply_temp': float(action[2]),
            'dhw_setpoint': float(action[3]),
        }
        for group, value in mapping.items():
            for h in actuator_handles.get(group, []):
                if h != -1:
                    api.exchange.set_actuator_value(ep_state, h, value)

    def format_state(self, raw_state):
        zone_temps = raw_state[1:6]
        avg_zone_temp = np.mean(zone_temps)
        return np.array([
            raw_state[0],                  # [0] outdoor temp
            raw_state[1], raw_state[2],    # [1,2] living room, kitchen
            raw_state[3], raw_state[4],    # [3,4] bedroom, bathroom
            raw_state[6],                  # [5] buffer tank temp
            raw_state[7],                  # [6] DHW tank temp
            raw_state[8],                  # [7] total electric power
            raw_state[9],                  # [8] HVAC electric power
            self.grid_excess_input,        # [9] Grid context (Python Side)
            self.electricity_price_input,  # [10] Cost context (Python Side)
            raw_state[12],                 # [11] outdoor RH
            avg_zone_temp,                 # [12] comfort metric
        ], dtype=np.float32)

    def compute_reward(self):
        rew, _ = self._compute_reward()
        return rew

    def _compute_reward(self, raw_state=None):
        st = raw_state if raw_state is not None else self.raw_state
        zone_temps = st[1:6]
        total_elec = st[8]
        
        # ---------------------------------------------------------
        # 1. KONFOR CEZASI (Deadband / Ölü Bant Mantığı: 20°C - 26°C)
        # ---------------------------------------------------------
        r_comfort = 0.0
        lower_bound = self.temperature_center - self.temperature_tolerance # 23 - 3 = 20°C
        upper_bound = self.temperature_center + self.temperature_tolerance # 23 + 3 = 26°C
        
        for zt in zone_temps:
            if zt < lower_bound:
                r_comfort -= (lower_bound - zt) ** 2  # Üşüme cezası (karesel)
            elif zt > upper_bound:
                r_comfort -= (zt - upper_bound) ** 2  # Terleme cezası (karesel)
        
        # 5 odaya bölerek ortalama konfor cezasını buluyoruz
        r_comfort = r_comfort / 5.0 

        # ---------------------------------------------------------
        # 2. ŞEBEKE ESNEKLİĞİ ÖDÜLÜ/CEZASI (Grid Flexibility)
        # grid_excess_input varsayımı: 
        #   Pozitif (+) = Şebekede fazla enerji var, tüket! (Ödül)
        #   Negatif (-) = Şebeke sıkışık, tüketimi kıs! (Ceza)
        # ---------------------------------------------------------
        # Tüketimi (Watt) maksimum HVAC kapasitesine bölerek 0-1 arasına normalize ediyoruz
        normalized_power = total_elec / self.hvac_max_power_w 
        
        r_grid = normalized_power * self.grid_excess_input 

        # ---------------------------------------------------------
        # 3. GENEL ENERJİ TASARRUFU (Sürtünme Kuvveti)
        # ---------------------------------------------------------
        r_cost = -normalized_power * self.electricity_price_input

        # === TOPLAM REWARD ===
        rew = (r_comfort * self.w_comfort) + (r_grid * self.w_grid_flex) + (r_cost * self.w_energy_cost)
        
        return rew, {'comfort': r_comfort, 'grid_flex': r_grid, 'cost': r_cost}

    def read_episode(self, ep): pass
    def plot_episode(self, ep): pass
    def dump_timesteps(self, **kwargs): pass
    def dump_episodes(self, **kwargs): pass