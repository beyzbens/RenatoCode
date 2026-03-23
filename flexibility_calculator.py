"""
Building Flexibility Calculator with CEM Optimization.

=== WHAT IS FLEXIBILITY? ===

Building power flexibility at time t is defined as:

    flex_up(t)   = P_max(t) - P_baseline(t)    [kW]
    flex_down(t) = P_baseline(t) - P_min(t)     [kW]

where:
    P_baseline = power at normal setpoints (htg=20, clg=26, hw=50, dhw=55)
    P_max      = maximum power achievable while ALL zones stay within
                 comfort limits [T_min, T_max]
    P_min      = minimum power achievable under same comfort constraints

=== WHY CEM INSTEAD OF GRID SEARCH? ===

Grid search:
    Try ALL combinations of setpoints on a fixed grid.
    Example: 10 × 10 × 10 × 10 = 10,000 evaluations
    Wastes time on clearly bad combinations (e.g., htg=24 AND clg=22)

CEM (Cross-Entropy Method):
    Iteratively focuses on promising regions of the setpoint space.
    Typically finds better solutions in ~1000 evaluations (5 iters × 200).

CEM procedure for finding P_max:
    Initialize: μ = center of action space, σ = wide spread
    For iteration = 1 to M:
        1) Sample N candidates from N(μ, σ²)
        2) Evaluate each: predict power + check comfort constraints
        3) Select top-K "elites" (highest power that satisfies constraints)
        4) Update: μ_new = mean(elites), σ_new = std(elites)
    Return: best elite found across all iterations

=== UNCERTAINTY-AWARE FLEXIBILITY ===

Because we use an ensemble model, each prediction has an uncertainty σ.
We report flexibility at different confidence levels:

    flex_up_mean = P_max_expected - P_baseline
    flex_up_conservative = (P_max_expected - β·σ_P) - P_baseline

where β controls conservatism:
    β = 0:   use mean prediction (optimistic, 50% confidence)
    β = 1:   ~68% confidence (1 standard deviation)
    β = 1.96: ~95% confidence (widely used in engineering)
    β = 2.58: ~99% confidence (very conservative)

For GAMS input, we provide both mean and conservative estimates,
letting the optimizer choose the appropriate risk level.

=== DURATION ESTIMATION ===

Duration = how many timesteps the flexibility can be sustained.

Method: Forward simulation with the surrogate model.
    Starting from current state, apply the flexibility setpoints
    and check at each step whether comfort is still satisfied.
    Stop when any zone exits the comfort band.

    Duration_up = number of steps until comfort violation under P_max setpoints
    Duration_down = number of steps until comfort violation under P_min setpoints

Duration in hours = Duration_steps × (timestep_minutes / 60)
"""

import numpy as np
import importlib
import importlib.util
from pathlib import Path


def _load_surrogate_class():
    module_names = ('thermal_surrogate', 'agent.thermal_surrogate')
    for module_name in module_names:
        parent_name = module_name.split('.')[0]
        if '.' in module_name and importlib.util.find_spec(parent_name) is None:
            continue
        if importlib.util.find_spec(module_name) is not None:
            module = importlib.import_module(module_name)
            return module.EnsembleThermalSurrogate

    base_dir = Path(__file__).resolve().parent
    candidate_paths = (
        base_dir / 'thermal_surrogate.py',
        base_dir / 'agent' / 'thermal_surrogate.py',
    )

    for module_path in candidate_paths:
        if module_path.exists():
            spec = importlib.util.spec_from_file_location(
                'loaded_thermal_surrogate', module_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module.EnsembleThermalSurrogate

    raise ModuleNotFoundError(
        'Could not locate thermal_surrogate.py. Checked importable modules '
        '"thermal_surrogate" and "agent.thermal_surrogate", plus relative '
        'paths "<project>/thermal_surrogate.py" and '
        '"<project>/agent/thermal_surrogate.py".')


EnsembleThermalSurrogate = _load_surrogate_class()


class CEMFlexibilityOptimizer:
    """Find max/min power setpoints using Cross-Entropy Method.

    === CEM PARAMETERS ===

    num_candidates = 200:
        Number of random setpoint combinations sampled per iteration.
        200 balances speed and coverage. More candidates = better solution
        but slower. For a 4D action space, 200 is sufficient.

    num_elites = 20:
        Top 10% of candidates used to update the distribution.
        Too few (5) → noisy updates, too many (100) → slow convergence.
        Rule of thumb: 10-20% of num_candidates.

    num_iterations = 5:
        CEM convergence rounds. After 5 iterations, the distribution
        has typically converged. More iterations give diminishing returns.

    alpha = 0.3:
        Momentum for distribution update:
            μ_new = α·mean(elites) + (1-α)·μ_old
        Prevents oscillation. 0.3 means 70% of previous distribution
        is retained, smooth convergence.

    confidence_beta = 1.96:
        Uncertainty penalty for conservative estimate.
        1.96 corresponds to 95% confidence interval:
            P(X > μ - 1.96σ) = 97.5%
        This means "we're 97.5% confident the real power is at least this much"
    """

    def __init__(self, surrogate: EnsembleThermalSurrogate,
                 comfort_min=20.0, comfort_max=26.0,
                 dhw_min=45.0, dhw_max=65.0,
                 action_low=np.array([15.0, 22.0, 30.0, 45.0]),
                 action_high=np.array([24.0, 28.0, 80.0, 65.0]),
                 num_candidates=200, num_elites=20,
                 num_iterations=5, alpha=0.3,
                 confidence_beta=1.96):
        self.surrogate = surrogate
        self.comfort_min = comfort_min
        self.comfort_max = comfort_max
        self.dhw_min = dhw_min
        self.dhw_max = dhw_max
        self.action_low = action_low
        self.action_high = action_high
        self.action_dim = len(action_low)
        self.num_candidates = num_candidates
        self.num_elites = num_elites
        self.num_iterations = num_iterations
        self.alpha = alpha
        self.confidence_beta = confidence_beta

    def _check_comfort(self, zone_temps, dhw_T, zone_temp_std=None):
        """Check comfort constraints, optionally using uncertainty.

        Without uncertainty: zone_T must be in [comfort_min, comfort_max]
        With uncertainty (conservative): zone_T + β·σ must be in bounds

        This means: we require the WORST CASE (at given confidence)
        to still be within comfort limits.
        """
        for i, zt in enumerate(zone_temps):
            low = self.comfort_min
            high = self.comfort_max
            if zone_temp_std is not None:
                # Conservative: account for prediction uncertainty
                margin = self.confidence_beta * zone_temp_std[i]
                if (zt - margin) < low or (zt + margin) > high:
                    return False
            else:
                if zt < low or zt > high:
                    return False
        if dhw_T < self.dhw_min:
            return False
        return True

    def find_max_power(self, outdoor_T, zone_temps, buffer_T, dhw_T, hour):
        """Find setpoints that MAXIMIZE power while respecting comfort.

        Used for upward flexibility (thermal charging).
        Higher hw_T and dhw_sp → more power consumption → more storage.
        """
        return self._cem_optimize(
            outdoor_T, zone_temps, buffer_T, dhw_T, hour,
            maximize=True)

    def find_min_power(self, outdoor_T, zone_temps, buffer_T, dhw_T, hour):
        """Find setpoints that MINIMIZE power while respecting comfort.

        Used for downward flexibility (load shedding).
        """
        return self._cem_optimize(
            outdoor_T, zone_temps, buffer_T, dhw_T, hour,
            maximize=False)

    def _cem_optimize(self, outdoor_T, zone_temps, buffer_T, dhw_T, hour,
                      maximize=True):
        """Core CEM optimization loop.

        === ALGORITHM ===

        1) Initialize Gaussian distribution over setpoints:
              μ = center of action bounds
              σ = (high - low) / 4   (covers ~95% of valid range)

        2) For each CEM iteration:
            a) Sample N setpoint candidates from N(μ, diag(σ²))
            b) Clip to valid bounds
            c) Evaluate all candidates with ensemble surrogate
            d) Filter: keep only those satisfying comfort constraints
            e) Sort by power (descending for max, ascending for min)
            f) Select top-K elites
            g) Update distribution:
                  μ = α·mean(elites) + (1-α)·μ_old
                  σ = α·std(elites) + (1-α)·σ_old

        3) Return best feasible candidate found
        """
        mu = (self.action_low + self.action_high) / 2.0
        sigma = (self.action_high - self.action_low) / 4.0

        best_power = -np.inf if maximize else np.inf
        best_setpoints = None
        best_power_std = 0.0
        # Track best without strict uncertainty check as fallback
        best_relaxed_power = -np.inf if maximize else np.inf
        best_relaxed_setpoints = None
        best_relaxed_std = 0.0

        for iteration in range(self.num_iterations):
            noise = np.random.randn(self.num_candidates, self.action_dim)
            candidates = mu + noise * sigma
            candidates = np.clip(candidates, self.action_low, self.action_high)

            for c in candidates:
                if c[1] <= c[0] + 1.0:
                    c[1] = c[0] + 1.5

            inputs = np.array([
                EnsembleThermalSurrogate.build_features(
                    outdoor_T, zone_temps, buffer_T, dhw_T, hour,
                    c[0], c[1], c[2], c[3])
                for c in candidates
            ])
            preds, unc = self.surrogate.predict(inputs)

            feasible_strict = []
            feasible_relaxed = []
            for i in range(self.num_candidates):
                next_zones = preds['next_zone_temps'][i]
                next_dhw = preds['next_dhw_T'][i]
                zone_std = unc['zone_temp_std'][i]
                power = preds['total_power'][i]
                power_std = unc['power_std'][i]

                # Relaxed check (mean only, no uncertainty margin)
                if self._check_comfort(next_zones, next_dhw, None):
                    feasible_relaxed.append((i, power, power_std, candidates[i]))
                    # Strict check (with uncertainty margin)
                    if self._check_comfort(next_zones, next_dhw, zone_std):
                        feasible_strict.append((i, power, power_std, candidates[i]))

            # Update from strict feasible set
            if feasible_strict:
                feasible_strict.sort(key=lambda x: x[1], reverse=maximize)
                top = feasible_strict[0]
                if (maximize and top[1] > best_power) or \
                   (not maximize and top[1] < best_power):
                    best_power = top[1]
                    best_setpoints = top[3].copy()
                    best_power_std = top[2]

                elites = feasible_strict[:min(self.num_elites, len(feasible_strict))]
                elite_actions = np.array([e[3] for e in elites])
                new_mu = elite_actions.mean(axis=0)
                new_sigma = elite_actions.std(axis=0) + 1e-4
                mu = self.alpha * new_mu + (1 - self.alpha) * mu
                sigma = self.alpha * new_sigma + (1 - self.alpha) * sigma

            # Also track relaxed best as fallback
            if feasible_relaxed:
                feasible_relaxed.sort(key=lambda x: x[1], reverse=maximize)
                top_r = feasible_relaxed[0]
                if (maximize and top_r[1] > best_relaxed_power) or \
                   (not maximize and top_r[1] < best_relaxed_power):
                    best_relaxed_power = top_r[1]
                    best_relaxed_setpoints = top_r[3].copy()
                    best_relaxed_std = top_r[2]

                if not feasible_strict:
                    elites = feasible_relaxed[:min(self.num_elites, len(feasible_relaxed))]
                    elite_actions = np.array([e[3] for e in elites])
                    new_mu = elite_actions.mean(axis=0)
                    new_sigma = elite_actions.std(axis=0) + 1e-4
                    mu = self.alpha * new_mu + (1 - self.alpha) * mu
                    sigma = self.alpha * new_sigma + (1 - self.alpha) * sigma

        # Use strict result if available, otherwise fallback to relaxed
        if best_setpoints is None and best_relaxed_setpoints is not None:
            best_power = best_relaxed_power
            best_setpoints = best_relaxed_setpoints
            best_power_std = best_relaxed_std

        final_power = best_power if best_setpoints is not None else 0.0
        final_std = best_power_std

        if maximize:
            conservative = final_power - self.confidence_beta * final_std
        else:
            conservative = final_power + self.confidence_beta * final_std
        if best_setpoints is None:
            conservative = 0.0

        return {
            'power_W': final_power,
            'power_std_W': final_std,
            'setpoints': best_setpoints,
            'power_conservative_W': conservative,
        }

    def estimate_duration(self, outdoor_T, zone_temps, buffer_T, dhw_T,
                          hour, setpoints, max_steps=16):
        """Forward-simulate to estimate how long flexibility is sustainable.

        === METHOD ===

        Starting from current state, repeatedly apply the given setpoints
        and advance one timestep using the surrogate model. At each step,
        check if comfort constraints are still satisfied.

        The simulation uses MEAN predictions (not sampled) because we want
        the expected duration, not a single stochastic trajectory.

        max_steps = 16 → at 15 min/step, this is 4 hours maximum.
        Most thermal storage can be sustained for 2-4 hours, so 16 is enough.
        """
        if setpoints is None:
            return 0

        htg, clg, hw, dhw_sp = setpoints
        zt = list(zone_temps)
        bt = buffer_T
        dt = dhw_T
        h = hour

        for step in range(max_steps):
            result = self.surrogate.predict_power(
                outdoor_T, zt, bt, dt, h, htg, clg, hw, dhw_sp)

            zt = list(result['next_zone_temps'])
            bt = result['next_buffer_T']
            dt = result['next_dhw_T']
            h = (h + 0.25) % 24.0  # advance 15 minutes

            if not self._check_comfort(zt, dt):
                return step

        return max_steps


class FlexibilityCalculator:
    """Main calculator: computes flexibility profiles for GAMS export.

    === OUTPUT FORMAT ===

    For each condition (hour × outdoor_T × storage state), produces:

    flex_up_kW:   Maximum additional power (kW) the building can absorb
                  while maintaining comfort. This is the "charging" capacity.

    flex_down_kW: Maximum power (kW) the building can reduce while
                  maintaining comfort. This is the "load shedding" capacity.

    flex_up_conservative_kW:   flex_up with 95% confidence (accounts for
                               model uncertainty). Use this in GAMS for
                               reliable results.

    duration_up_hours:  How long flex_up can be sustained (hours)
    duration_down_hours: How long flex_down can be sustained (hours)
    """

    def __init__(self, surrogate: EnsembleThermalSurrogate,
                 comfort_min=20.0, comfort_max=26.0,
                 baseline_htg=20.0, baseline_clg=26.0,
                 baseline_hw=50.0, baseline_dhw=55.0,
                 confidence_beta=1.96):
        self.surrogate = surrogate
        self.baseline = (baseline_htg, baseline_clg, baseline_hw, baseline_dhw)

        self.cem_optimizer = CEMFlexibilityOptimizer(
            surrogate=surrogate,
            comfort_min=comfort_min,
            comfort_max=comfort_max,
            confidence_beta=confidence_beta,
        )

    def calculate_single(self, outdoor_T, zone_temps, buffer_T, dhw_T, hour):
        """Calculate flexibility for a single condition."""
        # Baseline power
        base = self.surrogate.predict_power(
            outdoor_T, zone_temps, buffer_T, dhw_T, hour, *self.baseline)
        baseline_W = base['total_power']

        # CEM: find max power (upward flex)
        up = self.cem_optimizer.find_max_power(
            outdoor_T, zone_temps, buffer_T, dhw_T, hour)

        # CEM: find min power (downward flex)
        down = self.cem_optimizer.find_min_power(
            outdoor_T, zone_temps, buffer_T, dhw_T, hour)

        # Duration estimation
        dur_up = self.cem_optimizer.estimate_duration(
            outdoor_T, zone_temps, buffer_T, dhw_T, hour, up['setpoints'])
        dur_down = self.cem_optimizer.estimate_duration(
            outdoor_T, zone_temps, buffer_T, dhw_T, hour, down['setpoints'])

        flex_up = max(0, up['power_W'] - baseline_W)
        flex_down = max(0, baseline_W - down['power_W'])
        flex_up_cons = max(0, up['power_conservative_W'] - baseline_W)
        flex_down_cons = max(0, baseline_W - down['power_conservative_W'])

        return {
            'baseline_power_kW': baseline_W / 1000.0,
            'flex_up_kW': flex_up / 1000.0,
            'flex_down_kW': flex_down / 1000.0,
            'flex_up_conservative_kW': flex_up_cons / 1000.0,
            'flex_down_conservative_kW': flex_down_cons / 1000.0,
            'flex_up_power_std_kW': up['power_std_W'] / 1000.0,
            'flex_down_power_std_kW': down['power_std_W'] / 1000.0,
            'duration_up_hours': dur_up * 0.25,
            'duration_down_hours': dur_down * 0.25,
            'flex_up_setpoints': up['setpoints'],
            'flex_down_setpoints': down['setpoints'],
            'outdoor_T': outdoor_T,
            'avg_zone_T': np.mean(zone_temps),
            'buffer_T': buffer_T,
            'dhw_T': dhw_T,
            'hour': hour,
        }

    def calculate_profile(self, conditions, verbose=True):
        """Calculate flexibility for a list of conditions."""
        results = []
        for i, c in enumerate(conditions):
            r = self.calculate_single(
                c['outdoor_T'], c['zone_temps'],
                c['buffer_T'], c['dhw_T'], c['hour'])
            results.append(r)

            if verbose and (i + 1) % 50 == 0:
                print('  {}/{} conditions evaluated | '
                      'last: flex_up={:.2f}kW, flex_down={:.2f}kW'.format(
                          i + 1, len(conditions),
                          r['flex_up_kW'], r['flex_down_kW']))

        if verbose:
            up_vals = [r['flex_up_kW'] for r in results]
            down_vals = [r['flex_down_kW'] for r in results]
            print('\n  Flexibility Summary:')
            print('    Upward:   avg={:.2f} kW, max={:.2f} kW'.format(
                np.mean(up_vals), np.max(up_vals)))
            print('    Downward: avg={:.2f} kW, max={:.2f} kW'.format(
                np.mean(down_vals), np.max(down_vals)))

        return results

    @staticmethod
    def export_gams_csv(results, output_path):
        """Export as CSV formatted for GAMS import."""
        import csv
        with open(output_path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow([
                'hour', 'outdoor_T_C', 'avg_zone_T_C', 'buffer_T_C', 'dhw_T_C',
                'baseline_kW',
                'flex_up_kW', 'flex_down_kW',
                'flex_up_95pct_kW', 'flex_down_95pct_kW',
                'flex_up_std_kW', 'flex_down_std_kW',
                'duration_up_h', 'duration_down_h',
            ])
            for r in results:
                w.writerow([
                    '{:.1f}'.format(r['hour']),
                    '{:.1f}'.format(r['outdoor_T']),
                    '{:.1f}'.format(r['avg_zone_T']),
                    '{:.1f}'.format(r['buffer_T']),
                    '{:.1f}'.format(r['dhw_T']),
                    '{:.3f}'.format(r['baseline_power_kW']),
                    '{:.3f}'.format(r['flex_up_kW']),
                    '{:.3f}'.format(r['flex_down_kW']),
                    '{:.3f}'.format(r['flex_up_conservative_kW']),
                    '{:.3f}'.format(r['flex_down_conservative_kW']),
                    '{:.3f}'.format(r['flex_up_power_std_kW']),
                    '{:.3f}'.format(r['flex_down_power_std_kW']),
                    '{:.2f}'.format(r['duration_up_hours']),
                    '{:.2f}'.format(r['duration_down_hours']),
                ])
        print('Exported {} rows to {}'.format(len(results), output_path))
