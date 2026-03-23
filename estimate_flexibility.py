"""
Building Flexibility Estimation Pipeline.

=== COMPLETE WORKFLOW ===

Step 1: DATA COLLECTION (~30-60 min)
    Run EnergyPlus with varied setpoint strategies to collect
    training data covering the building's operational envelope.

    Strategies (4 types, cycled across episodes):
      - Baseline:   fixed normal setpoints (reference behavior)
      - Random:     uniformly random setpoints (maximum exploration)
      - Gaussian:   small perturbations around baseline (local behavior)
      - Sweep:      systematically increasing setpoints (monotonic response)

    Each 15-minute timestep records:
      Input:  (outdoor_T, zone_temps, tank_temps, hour, setpoints)
      Output: (next_zone_temps, next_tank_temps, hvac_power, total_power)

Step 2: MODEL TRAINING (~5-10 min)
    Train an ensemble of 5 probabilistic MLPs on the collected data.
    The ensemble learns to predict the building's thermal response
    and provides uncertainty estimates.

Step 3: FLEXIBILITY CALCULATION (~5-15 min)
    For representative conditions (outdoor temperature × hour × storage state):
      - Use CEM optimization to find maximum comfort-feasible power
      - Use CEM optimization to find minimum comfort-feasible power
      - Estimate duration of sustainability
      - Report both mean and 95%-confidence estimates

Step 4: GAMS EXPORT
    Write results to CSV with columns:
      hour, outdoor_T, baseline_kW, flex_up_kW, flex_down_kW,
      flex_up_95pct_kW, flex_down_95pct_kW, duration_up_h, duration_down_h

=== USAGE ===

    python estimate_flexibility.py run \
        --model_file EnergyPlus/Model/renato17_4PipeFanCoil_DHW.idf \
        --weather_file weather/porto.epw \
        --comfort_min 20.0 --comfort_max 26.0

    python estimate_flexibility.py flex \
        --surrogate_path runs/flexibility/surrogate.pt \
        --output_csv flexibility_for_gams.csv
"""

import os
import sys
import json
import numpy as np
import pprint
import importlib.util
import inspect
from argparse import ArgumentParser
from pathlib import Path

ep_path = r'C:\EnergyPlusV25-2-0' 
os.environ['PATH'] = ep_path + os.pathsep + os.environ.get('PATH', '')

from gym_energyplus.envs.energyplus_env import EnergyPlusEnv
from gym_energyplus.wrappers import EnergyPlusSplitEpisodeWrapper
from flexibility_calculator import FlexibilityCalculator


def _load_surrogate_class():
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
        'Could not locate thermal_surrogate.py. Expected either '
        '"<project>/thermal_surrogate.py" or '
        '"<project>/agent/thermal_surrogate.py".')


EnsembleThermalSurrogate = _load_surrogate_class()


# ====================================================================
# Step 1: Data Collection
# ====================================================================

def collect_training_data(model_file, weather_file, config,
                          num_episodes=365, verbose=True):
    """Run EnergyPlus with varied setpoints to collect training data.

    === DATA COLLECTION STRATEGY ===

    We need data that covers the full operational range of the building.
    Using only normal setpoints would give biased data (only near 20-26°C).

    Four strategies ensure good coverage:

    1) Baseline (25% of episodes):
       Fixed setpoints: htg=20, clg=26, hw=50, dhw=55
       Purpose: learn nominal building behavior

    2) Random (25%):
       Uniformly random setpoints within full action bounds
       Purpose: explore extreme operating conditions

    3) Gaussian perturbation (25%):
       Small random noise around baseline (±15% of range)
       Purpose: learn sensitivity to small setpoint changes

    4) Linear sweep (25%):
       Setpoints increase linearly over the episode
       Purpose: learn transient/ramp behavior

    Total data per episode: ~96 timesteps (1 day at 15-min intervals)
    Total data for 365 episodes: ~35,040 samples
    """
    env = EnergyPlusEnv(
        model_file=model_file,
        weather_file=weather_file,
        config=config,
        verbose=False,
    )
    env = EnergyPlusSplitEpisodeWrapper(env, max_steps=96)

    action_low = env.action_space.low
    action_high = env.action_space.high
    action_center = (action_low + action_high) / 2.0
    action_range = action_high - action_low

    all_inputs = []
    all_outputs = []
    step_global = 0

    if verbose:
        print('  Observation space: {} dims'.format(
            env.observation_space.shape[0]))
        print('  Action space: {} dims, low={}, high={}'.format(
            env.action_space.shape[0], action_low, action_high))

    for ep in range(num_episodes):
        obs = env.reset()

        if verbose and ep == 0:
            print('  First observation ({}): {}'.format(len(obs), obs))

        done = False
        strategy = ep % 4
        step_in_ep = 0

        while not done:
            if strategy == 0:
                action = action_center.copy()
            elif strategy == 1:
                action = np.random.uniform(action_low, action_high)
            elif strategy == 2:
                noise = np.random.normal(0, 0.25, size=action_low.shape)
                action = np.clip(
                    action_center + noise * action_range,
                    action_low, action_high)
            else:
                phase = step_in_ep / 96.0
                action = action_low + phase * action_range
                action = np.clip(action, action_low, action_high)

            next_obs, reward, done, info = env.step(action)

            hour = (step_global % 96) * 0.25

            # obs is the formatted state from ep_model.format_state()
            # Format: [outdoor, living, kitchen, bedroom, bathroom,
            #          buffer_T, dhw_T, total_W, hvac_W,
            #          grid_excess, elec_price, outdoor_RH, avg_zone]
            obs_dim = len(obs)
            if obs_dim >= 13:
                outdoor_T = obs[0]
                zone_temps = obs[1:5]
                buffer_T = obs[5]
                dhw_T = obs[6]
                total_W = obs[7]
                hvac_W = obs[8]

                next_zone_temps = next_obs[1:5]
                next_buffer_T = next_obs[5]
                next_dhw_T = next_obs[6]
                next_hvac_W = next_obs[8]
                next_total_W = next_obs[7]
            else:
                # Fallback for different observation formats
                outdoor_T = obs[0]
                zone_temps = obs[1:min(5, obs_dim)]
                while len(zone_temps) < 4:
                    zone_temps = np.append(zone_temps, zone_temps[-1])
                buffer_T = obs[5] if obs_dim > 5 else 50.0
                dhw_T = obs[6] if obs_dim > 6 else 55.0
                total_W = obs[7] if obs_dim > 7 else 0.0
                hvac_W = obs[8] if obs_dim > 8 else 0.0

                next_zone_temps = next_obs[1:min(5, len(next_obs))]
                while len(next_zone_temps) < 4:
                    next_zone_temps = np.append(next_zone_temps, next_zone_temps[-1])
                next_buffer_T = next_obs[5] if len(next_obs) > 5 else 50.0
                next_dhw_T = next_obs[6] if len(next_obs) > 6 else 55.0
                next_hvac_W = next_obs[8] if len(next_obs) > 8 else 0.0
                next_total_W = next_obs[7] if len(next_obs) > 7 else 0.0

            inp = EnsembleThermalSurrogate.build_features(
                outdoor_T=outdoor_T,
                zone_temps=zone_temps,
                buffer_T=buffer_T,
                dhw_T=dhw_T,
                hour=hour,
                htg_sp=action[0], clg_sp=action[1],
                hw_T=action[2], dhw_sp=action[3])

            out = EnsembleThermalSurrogate.build_targets(
                next_zone_temps=next_zone_temps,
                next_buffer_T=next_buffer_T,
                next_dhw_T=next_dhw_T,
                hvac_power_W=next_hvac_W,
                total_power_W=next_total_W)

            all_inputs.append(inp)
            all_outputs.append(out)

            if verbose and step_global == 0:
                print('  First input features: {}'.format(inp))
                print('  First output targets: {}'.format(out))

            obs = next_obs
            step_global += 1
            step_in_ep += 1

            if info.get('true_done', False):
                break

        if verbose and (ep + 1) % 10 == 0:
            print('  Episode {}/{} [{}]: {} total samples'.format(
                ep + 1, num_episodes,
                ['baseline', 'random', 'gaussian', 'sweep'][strategy],
                len(all_inputs)))

    env.close()

    inputs = np.array(all_inputs, dtype=np.float32)
    outputs = np.array(all_outputs, dtype=np.float32)

    if verbose:
        print('\n  Data collection complete: {} samples'.format(len(inputs)))

    return inputs, outputs


# ====================================================================
# Step 2: Train Ensemble Surrogate
# ====================================================================

def train_surrogate(inputs, outputs, ensemble_size=5, epochs=150,
                    hidden_dim=256, learning_rate=5e-4,
                    power_loss_weight=3.0, hvac_loss_weight=2.0,
                    verbose=True):
    """Train ensemble thermal surrogate model."""
    if verbose:
        print('\n  Training Ensemble Surrogate ({} members)...'.format(
            ensemble_size))
        print('  Samples: {}, Input dim: {}, Output dim: {}'.format(
            len(inputs), inputs.shape[1], outputs.shape[1]))
        print('  hidden_dim={}, learning_rate={}, hvac_loss_weight={}, total_power_loss_weight={}'.format(
            hidden_dim, learning_rate, hvac_loss_weight, power_loss_weight))

    output_weights = np.ones(outputs.shape[1], dtype=np.float32)
    output_weights[6] = hvac_loss_weight
    output_weights[7] = power_loss_weight
    init_signature = inspect.signature(EnsembleThermalSurrogate.__init__)
    init_params = init_signature.parameters
    surrogate_kwargs = {}

    if 'ensemble_size' in init_params:
        surrogate_kwargs['ensemble_size'] = ensemble_size
    if 'hidden_dim' in init_params:
        surrogate_kwargs['hidden_dim'] = hidden_dim
    if 'learning_rate' in init_params:
        surrogate_kwargs['learning_rate'] = learning_rate
    if 'output_weights' in init_params:
        surrogate_kwargs['output_weights'] = output_weights
    elif verbose:
        print('  Warning: loaded EnsembleThermalSurrogate does not support '
              'output_weights; continuing without weighted loss.')

    surrogate = EnsembleThermalSurrogate(**surrogate_kwargs)
    surrogate.fit(inputs, outputs, epochs=epochs, verbose=verbose)

    return surrogate


def save_evaluation_report(surrogate, output_dir, verbose=True):
    """Write train/val/test metrics with residual and coverage stats."""
    report_path = os.path.join(output_dir, 'evaluation_metrics.json')
    if not hasattr(surrogate, 'evaluate_splits'):
        evaluation = {
            'available': False,
            'reason': 'Loaded surrogate implementation does not provide '
                      'evaluate_splits().',
        }
        with open(report_path, 'w') as f:
            json.dump(evaluation, f, indent=2)
        if verbose:
            print('\n  Evaluation summary skipped: loaded surrogate '
                  'implementation does not provide evaluate_splits().')
            print('  Detailed evaluation JSON: {}'.format(report_path))
        return evaluation

    evaluation = surrogate.evaluate_splits()
    with open(report_path, 'w') as f:
        json.dump(evaluation, f, indent=2)

    if verbose:
        print('\n  Evaluation summary:')
        for split_name in ('train', 'val', 'test'):
            split_metrics = evaluation[split_name]
            total_power = split_metrics['outputs']['total_power']
            print(
                '    {:>5} | n={} | total_power R²={:.4f}, MAE={:.2f}, RMSE={:.2f}, '
                'residual μ={:.2f}, σ={:.2f}, cov@1σ={:.3f}, cov@2σ={:.3f}'.format(
                    split_name,
                    split_metrics['num_samples'],
                    total_power['r2'],
                    total_power['mae'],
                    total_power['rmse'],
                    total_power['residual_mean'],
                    total_power['residual_std'],
                    total_power['coverage_1sigma'],
                    total_power['coverage_2sigma'],
                )
            )
        print('  Detailed evaluation JSON: {}'.format(report_path))

    return evaluation


# ====================================================================
# Step 3: Generate Representative Conditions
# ====================================================================

def generate_conditions():
    """Generate conditions covering a typical year in Porto, Portugal.

    === CONDITION SPACE ===

    We evaluate flexibility at the cross-product of:
      - Outdoor temperatures:  4 seasons × 4 temps = 16 values
      - Hours of day:          8 representative hours = 8 values
      - Buffer tank states:    3 levels (low/mid/high) = 3 values
      - DHW tank states:       2 levels (low/high) = 2 values

    Total: 16 × 8 × 3 × 2 = 768 conditions

    Zone temperatures are estimated from outdoor temperature
    (assuming building has been at quasi-steady-state).
    """
    conditions = []

    seasons = {
        'winter': [5, 8, 10, 12],
        'spring': [12, 15, 18, 20],
        'summer': [20, 24, 28, 30],
        'autumn': [15, 18, 20, 22],
    }
    hours = [0, 3, 6, 9, 12, 15, 18, 21]
    buffer_temps = [40, 55, 70]
    dhw_temps = [48, 55]

    for season_name, out_temps in seasons.items():
        for out_T in out_temps:
            for hour in hours:
                for buf_T in buffer_temps:
                    for dhw_T in dhw_temps:
                        zone_base = 22.0
                        if out_T < 10:
                            zone_base = 20.5
                        elif out_T > 25:
                            zone_base = 24.0

                        conditions.append({
                            'outdoor_T': float(out_T),
                            'zone_temps': [
                                zone_base,
                                zone_base - 0.5,
                                zone_base + 0.3,
                                zone_base + 0.5,
                            ],
                            'buffer_T': float(buf_T),
                            'dhw_T': float(dhw_T),
                            'hour': float(hour),
                            'season': season_name,
                        })

    return conditions


# ====================================================================
# Main Pipeline
# ====================================================================

def run_full_pipeline(model_file, weather_file, config, output_dir):
    """Run complete pipeline: collect → train → calculate → export."""
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    # --- Step 1: Collect ---
    print('\n' + '='*60)
    print('  STEP 1: Data Collection from EnergyPlus')
    print('='*60)
    inputs, outputs = collect_training_data(
        model_file, weather_file, config,
        num_episodes=config.get('num_episodes', 365))
    np.save(os.path.join(output_dir, 'train_inputs.npy'), inputs)
    np.save(os.path.join(output_dir, 'train_outputs.npy'), outputs)

    # --- Step 2: Train ---
    print('\n' + '='*60)
    print('  STEP 2: Ensemble Surrogate Training')
    print('='*60)
    surrogate = train_surrogate(
        inputs, outputs,
        ensemble_size=config.get('ensemble_size', 5),
        epochs=config.get('train_epochs', 150),
        hidden_dim=config.get('hidden_dim', 256),
        learning_rate=config.get('learning_rate', 5e-4),
        hvac_loss_weight=config.get('hvac_loss_weight', 2.0),
        power_loss_weight=config.get('power_loss_weight', 3.0))
    surrogate.save(os.path.join(output_dir, 'surrogate.pt'))
    evaluation = save_evaluation_report(surrogate, output_dir)

    # --- Step 3: Calculate ---
    print('\n' + '='*60)
    print('  STEP 3: Flexibility Calculation (CEM + Uncertainty)')
    print('='*60)
    conditions = generate_conditions()
    calculator = FlexibilityCalculator(
        surrogate,
        comfort_min=config.get('comfort_min', 20.0),
        comfort_max=config.get('comfort_max', 26.0),
        confidence_beta=config.get('confidence_beta', 1.96),
    )
    results = calculator.calculate_profile(conditions, verbose=True)

    # --- Step 4: Export ---
    print('\n' + '='*60)
    print('  STEP 4: GAMS Export')
    print('='*60)
    csv_path = os.path.join(output_dir, 'flexibility_for_gams.csv')
    FlexibilityCalculator.export_gams_csv(results, csv_path)

    # Summary
    summary = {
        'num_conditions': len(results),
        'num_training_samples': len(inputs),
        'ensemble_size': config.get('ensemble_size', 5),
        'hidden_dim': config.get('hidden_dim', 256),
        'learning_rate': config.get('learning_rate', 5e-4),
        'confidence_level': '95%' if config.get('confidence_beta', 1.96) == 1.96 else 'custom',
        'comfort_band': [config.get('comfort_min', 20.0), config.get('comfort_max', 26.0)],
        'evaluation_available': bool(evaluation.get('available', True)),
        'avg_flex_up_kW': float(np.mean([r['flex_up_kW'] for r in results])),
        'max_flex_up_kW': float(np.max([r['flex_up_kW'] for r in results])),
        'avg_flex_down_kW': float(np.mean([r['flex_down_kW'] for r in results])),
        'max_flex_down_kW': float(np.max([r['flex_down_kW'] for r in results])),
        'avg_flex_up_95pct_kW': float(np.mean([r['flex_up_conservative_kW'] for r in results])),
        'avg_flex_down_95pct_kW': float(np.mean([r['flex_down_conservative_kW'] for r in results])),
        'avg_duration_up_h': float(np.mean([r['duration_up_hours'] for r in results])),
        'avg_duration_down_h': float(np.mean([r['duration_down_hours'] for r in results])),
    }
    if evaluation.get('available', True):
        summary.update({
            'train_total_power_r2': evaluation['train']['outputs']['total_power']['r2'],
            'val_total_power_r2': evaluation['val']['outputs']['total_power']['r2'],
            'test_total_power_r2': evaluation['test']['outputs']['total_power']['r2'],
            'test_total_power_mae_W': evaluation['test']['outputs']['total_power']['mae'],
            'test_total_power_rmse_W': evaluation['test']['outputs']['total_power']['rmse'],
            'test_total_power_coverage_1sigma': evaluation['test']['outputs']['total_power']['coverage_1sigma'],
            'test_total_power_coverage_2sigma': evaluation['test']['outputs']['total_power']['coverage_2sigma'],
        })
    with open(os.path.join(output_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print('\n' + '='*60)
    print('  COMPLETE')
    print('='*60)
    print('  GAMS CSV:  {}'.format(csv_path))
    print('  Model:     {}'.format(os.path.join(output_dir, 'surrogate.pt')))
    print('\n  Flexibility Summary:')
    for k, v in summary.items():
        if isinstance(v, float):
            print('    {}: {:.3f}'.format(k, v))
        else:
            print('    {}: {}'.format(k, v))


# ====================================================================
# CLI
# ====================================================================

def make_parser():
    p = ArgumentParser(
        description='Building Flexibility Estimation for GAMS')
    sub = p.add_subparsers(dest='command')

    run_p = sub.add_parser('run', help='Full pipeline')
    run_p.add_argument('--model_file', type=str, required=True)
    run_p.add_argument('--weather_file', type=str, required=True)
    run_p.add_argument('--output_dir', type=str, default='runs/flexibility')
    # YENİ HALİ: default=365 yapıldı
    run_p.add_argument('--num_episodes', type=int, default=365,
                       help='EnergyPlus data collection episodes (more=better model)')
    run_p.add_argument('--ensemble_size', type=int, default=5,
                       help='Number of ensemble members (5=standard)')
    run_p.add_argument('--hidden_dim', type=int, default=256,
                       help='Hidden width of each MLP block')
    run_p.add_argument('--learning_rate', type=float, default=5e-4)
    run_p.add_argument('--hvac_loss_weight', type=float, default=2.0,
                       help='Relative loss weight for HVAC power output')
    run_p.add_argument('--power_loss_weight', type=float, default=3.0,
                       help='Relative loss weight for total power output')
    run_p.add_argument('--train_epochs', type=int, default=150)
    run_p.add_argument('--comfort_min', type=float, default=20.0)
    run_p.add_argument('--comfort_max', type=float, default=26.0)
    run_p.add_argument('--confidence_beta', type=float, default=1.96,
                       help='1.96=95%% confidence, 1.0=68%%, 2.58=99%%')
    run_p.add_argument('--temp_center', type=float, default=22.0)
    run_p.add_argument('--dhw_target', type=float, default=55.0)

    flex_p = sub.add_parser('flex', help='Flexibility from pre-trained model')
    flex_p.add_argument('--surrogate_path', type=str, required=True)
    flex_p.add_argument('--output_csv', type=str, default='flexibility_for_gams.csv')
    flex_p.add_argument('--comfort_min', type=float, default=20.0)
    flex_p.add_argument('--comfort_max', type=float, default=26.0)
    flex_p.add_argument('--confidence_beta', type=float, default=1.96)

    return p


if __name__ == '__main__':
    parser = make_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == 'run':
        config = {
            'temp_center': 23.0,       
            'temp_tolerance': 3.0,     
            'dhw_target': args.dhw_target,
            'hvac_max_power': 21000,
            # YENİ HALİ: fallback değeri de 365 yapıldı
            'num_episodes': args.num_episodes,
            'ensemble_size': args.ensemble_size,
            'hidden_dim': args.hidden_dim,
            'learning_rate': args.learning_rate,
            'hvac_loss_weight': args.hvac_loss_weight,
            'power_loss_weight': args.power_loss_weight,
            'train_epochs': args.train_epochs,
            'comfort_min': 20.0,      
            'comfort_max': 26.0,       
            'confidence_beta': args.confidence_beta,
        }
        run_full_pipeline(
            args.model_file, args.weather_file, config, args.output_dir)

    elif args.command == 'flex':
        surrogate = EnsembleThermalSurrogate()
        surrogate.load(args.surrogate_path)
        print('Loaded ensemble surrogate from {}'.format(args.surrogate_path))
        conditions = generate_conditions()
        calculator = FlexibilityCalculator(
            surrogate,
            comfort_min=args.comfort_min,
            comfort_max=args.comfort_max,
            confidence_beta=args.confidence_beta)
        results = calculator.calculate_profile(conditions)
        FlexibilityCalculator.export_gams_csv(results, args.output_csv)
