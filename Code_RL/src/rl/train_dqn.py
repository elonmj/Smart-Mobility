"""
DQN Training Script for Traffic Signal Control

Implements training loop with the baseline DQN algorithm as specified
in the design document.
"""

import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.logger import configure
import json
is_kaggle = 'KAGGLE_KERNEL_RUN_TYPE' in os.environ

# Import custom callbacks with intelligent checkpoint rotation
import sys
sys.path.append(os.path.dirname(__file__))
from callbacks import RotatingCheckpointCallback, TrainingProgressCallback

from endpoint.client import create_endpoint_client, EndpointConfig
from signals.controller import create_signal_controller
from env.traffic_signal_env import TrafficSignalEnv, EnvironmentConfig
from utils.config import (
    load_configs, load_config, validate_config_consistency, setup_logging,
    ExperimentTracker, save_training_results, RLConfigBuilder
)


def create_environment_pydantic(rl_config: 'RLConfigBuilder', use_mock: bool = False) -> TrafficSignalEnv:
    """
    Create traffic signal environment from Pydantic configs (NEW).
    
    Args:
        rl_config: RLConfigBuilder instance with Pydantic configs
        use_mock: If True, use mock simulator instead of real ARZ
    
    Returns:
        TrafficSignalEnv instance
    """
    # Create endpoint client
    endpoint_params = rl_config.endpoint_params.copy()
    if use_mock:
        endpoint_params["protocol"] = "mock"
    
    endpoint_config = EndpointConfig(**endpoint_params)
    endpoint_client = create_endpoint_client(endpoint_config)
    
    # Create signal controller
    signal_controller = create_signal_controller(rl_config.signal_params)
    
    # Create environment config from RL parameters
    env_params = rl_config.rl_env_params
    env_config = EnvironmentConfig(
        dt_decision=env_params["dt_decision"],
        episode_length=env_params["episode_length"],
        max_steps=env_params["max_steps"],
        rho_max_motorcycles=env_params["normalization"]["rho_max_motorcycles"],
        rho_max_cars=env_params["normalization"]["rho_max_cars"],
        v_free_motorcycles=env_params["normalization"]["v_free_motorcycles"],
        v_free_cars=env_params["normalization"]["v_free_cars"],
        queue_max=env_params["normalization"]["queue_max"],
        phase_time_max=env_params["normalization"]["phase_time_max"],
        w_wait_time=env_params["reward"]["w_wait_time"],
        w_queue_length=env_params["reward"]["w_queue_length"],
        w_stops=env_params["reward"]["w_stops"],
        w_switch_penalty=env_params["reward"]["w_switch_penalty"],
        w_throughput=env_params["reward"]["w_throughput"],
        reward_clip=tuple(env_params["reward"]["reward_clip"]),
        stop_speed_threshold=env_params["reward"]["stop_speed_threshold"],
        ewma_alpha=env_params["observation"]["ewma_alpha"],
        include_phase_timing=env_params["observation"]["include_phase_timing"],
        include_queues=env_params["observation"]["include_queues"]
    )
    
    # Extract branch IDs from ARZ config (if available)
    # For now, use default branch IDs - this will be updated when we integrate
    # with full network configuration
    branch_ids = ["branch_0", "branch_1", "branch_2", "branch_3"]
    
    # Create environment
    env = TrafficSignalEnv(
        endpoint_client=endpoint_client,
        signal_controller=signal_controller,
        config=env_config,
        branch_ids=branch_ids
    )
    
    return env


def create_environment(configs: dict, use_mock: bool = False) -> TrafficSignalEnv:
    """
    Create traffic signal environment from configs (LEGACY - YAML-based).
    
    DEPRECATED: Use create_environment_pydantic() for new code.
    This function is preserved for backward compatibility.
    """
    
    # Create endpoint client
    # Handle nested config structure
    endpoint_params = configs["endpoint"]
    if "endpoint" in endpoint_params:
        endpoint_params = endpoint_params["endpoint"]
    
    # Filter unknown keys to match EndpointConfig dataclass
    allowed_keys = {
        "protocol", "host", "port", "base_url", "dt_sim", "timeout", "max_retries", "retry_backoff"
    }
    endpoint_params_filtered = {k: v for k, v in endpoint_params.items() if k in allowed_keys}
    
    endpoint_config = EndpointConfig(**endpoint_params_filtered)
    if use_mock:
        endpoint_config.protocol = "mock"
    
    endpoint_client = create_endpoint_client(endpoint_config)
    
    # Create signal controller
    # Handle nested config structure
    signals_params = configs["signals"]
    if "signals" in signals_params:
        signals_config = signals_params
    else:
        signals_config = {"signals": signals_params}
    
    signal_controller = create_signal_controller(signals_config)
    
    # Create environment config
    env_config_data = configs["env"]["environment"]
    env_config = EnvironmentConfig(
        dt_decision=env_config_data["dt_decision"],
        episode_length=env_config_data["episode_length"],
        max_steps=env_config_data["max_steps"],
        rho_max_motorcycles=env_config_data["normalization"]["rho_max_motorcycles"],
        rho_max_cars=env_config_data["normalization"]["rho_max_cars"],
        v_free_motorcycles=env_config_data["normalization"]["v_free_motorcycles"],
        v_free_cars=env_config_data["normalization"]["v_free_cars"],
        queue_max=env_config_data["normalization"]["queue_max"],
        phase_time_max=env_config_data["normalization"]["phase_time_max"],
        w_wait_time=env_config_data["reward"]["w_wait_time"],
        w_queue_length=env_config_data["reward"]["w_queue_length"],
        w_stops=env_config_data["reward"]["w_stops"],
        w_switch_penalty=env_config_data["reward"]["w_switch_penalty"],
        w_throughput=env_config_data["reward"]["w_throughput"],
        reward_clip=tuple(env_config_data["reward"]["reward_clip"]),
        stop_speed_threshold=env_config_data["reward"]["stop_speed_threshold"],
        ewma_alpha=env_config_data["observation"]["ewma_alpha"],
        include_phase_timing=env_config_data["observation"]["include_phase_timing"],
        include_queues=env_config_data["observation"]["include_queues"]
    )
    
    # Get branch IDs
    branch_ids = [branch["id"] for branch in configs["network"]["network"]["branches"]]
    
    # Create environment
    env = TrafficSignalEnv(
        endpoint_client=endpoint_client,
        signal_controller=signal_controller,
        config=env_config,
        branch_ids=branch_ids
    )
    
    return env


def find_latest_checkpoint(checkpoint_dir: str, name_prefix: str) -> tuple[str, int]:
    """
    Find the latest checkpoint file in the directory.
    
    Returns:
        tuple: (checkpoint_path, num_timesteps) or (None, 0) if no checkpoint found
    """
    if not os.path.exists(checkpoint_dir):
        return None, 0
    
    # List all checkpoint files
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith(name_prefix) and f.endswith('.zip')]
    
    if not checkpoint_files:
        return None, 0
    
    # Extract timestep numbers from filenames
    # Format: {name_prefix}_{timesteps}_steps.zip
    checkpoints_with_steps = []
    for fname in checkpoint_files:
        try:
            # Extract number between last underscore and "_steps.zip"
            parts = fname.replace('.zip', '').split('_')
            # Find "steps" index and get the number before it
            if 'steps' in parts:
                steps_idx = parts.index('steps')
                if steps_idx > 0:
                    num_steps = int(parts[steps_idx - 1])
                    checkpoints_with_steps.append((fname, num_steps))
        except (ValueError, IndexError):
            continue
    
    if not checkpoints_with_steps:
        return None, 0
    
    # Get checkpoint with most timesteps
    latest_checkpoint = max(checkpoints_with_steps, key=lambda x: x[1])
    checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint[0])
    
    return checkpoint_path, latest_checkpoint[1]


def create_custom_dqn_policy():
    """Create custom DQN policy network"""
    return "MlpPolicy"  # Use built-in MLP policy


def train_dqn_agent(
    env: TrafficSignalEnv,
    total_timesteps: int = 100000,
    learning_rate: float = 1e-3,
    buffer_size: int = 50000,
    learning_starts: int = 1000,
    batch_size: int = 32,
    tau: float = 1.0,
    gamma: float = 0.99,
    train_freq: int = 4,
    gradient_steps: int = 1,
    target_update_interval: int = 1000,
    exploration_fraction: float = 0.1,
    exploration_initial_eps: float = 1.0,
    exploration_final_eps: float = 0.05,
    seed: int = 42,
    output_dir: str = "results",
    experiment_name: str = "dqn_baseline",
    resume_training: bool = True,
    checkpoint_freq: int = None,  # Will be set adaptively if None
    max_checkpoints_to_keep: int = 2  # Keep only 2 most recent checkpoints
) -> DQN:
    """
    Train DQN agent on traffic signal environment.
    
    Args:
        resume_training: If True, automatically resume from latest checkpoint if available
        checkpoint_freq: Frequency (in timesteps) to save checkpoints.
                        If None, will be set adaptively:
                        - Quick test (<5000 timesteps): every 100 steps
                        - Small run (<20000 timesteps): every 500 steps  
                        - Production run (>=20000 timesteps): every 1000 steps
        max_checkpoints_to_keep: Maximum number of checkpoints to keep (default: 2)
                                 Older checkpoints are automatically deleted to save disk space.
                                 Recommended: 2-3 for Kaggle (20GB limit)
    """
    
    # Setup output directory
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Adaptive checkpoint frequency based on total_timesteps
    if checkpoint_freq is None:
        if total_timesteps < 5000:
            checkpoint_freq = 100  # Quick test: save every 100 steps
            print(f"⚙️  Quick test mode: checkpoint every {checkpoint_freq} steps")
        elif total_timesteps < 20000:
            checkpoint_freq = 500  # Small run: save every 500 steps
            print(f"⚙️  Small run mode: checkpoint every {checkpoint_freq} steps")
        else:
            checkpoint_freq = 1000  # Production: save every 1000 steps
            print(f"⚙️  Production mode: checkpoint every {checkpoint_freq} steps")
    else:
        print(f"⚙️  Manual checkpoint frequency: every {checkpoint_freq} steps")
    
    # Configure logger
    sb3_logger = configure(output_dir, ["csv", "tensorboard"])
    
    # Check for existing checkpoint
    checkpoint_path, completed_timesteps = find_latest_checkpoint(checkpoint_dir, f"{experiment_name}_checkpoint")
    
    if resume_training and checkpoint_path and completed_timesteps > 0:
        print(f"🔄 RESUMING TRAINING from checkpoint: {checkpoint_path}")
        print(f"   ✓ Already completed: {completed_timesteps:,} timesteps")
        print(f"   ✓ Remaining: {total_timesteps - completed_timesteps:,} timesteps")
        
        # Load the model from checkpoint
        model = DQN.load(
            checkpoint_path,
            env=env,
            verbose=1
        )
        model.set_logger(sb3_logger)
        
        # Calculate remaining timesteps
        remaining_timesteps = max(0, total_timesteps - completed_timesteps)
        
        if remaining_timesteps == 0:
            print("✅ Training already completed!")
            return model
            
    else:
        print(f"🆕 STARTING NEW TRAINING: {total_timesteps:,} timesteps")
        
        # Create new DQN model
        model = DQN(
            policy=create_custom_dqn_policy(),
            env=env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            target_update_interval=target_update_interval,
            exploration_fraction=exploration_fraction,
            exploration_initial_eps=exploration_initial_eps,
            exploration_final_eps=exploration_final_eps,
            seed=seed,
            verbose=1
        )
        
        model.set_logger(sb3_logger)
        remaining_timesteps = total_timesteps
        completed_timesteps = 0
    
    # Setup callbacks with 3-level checkpoint strategy
    
    # 1. BEST MODEL: Saved automatically when evaluation improves
    #    - Used for final deployment and thesis results
    #    - Never deleted, only updated when better performance achieved
    best_model_dir = os.path.join(output_dir, "best_model")
    os.makedirs(best_model_dir, exist_ok=True)
    
    eval_callback = EvalCallback(
        eval_env=env,
        best_model_save_path=best_model_dir,
        log_path=os.path.join(output_dir, "eval"),
        eval_freq=max(checkpoint_freq, 1000),  # Evaluate at least as often as checkpoints
        n_eval_episodes=5 if total_timesteps < 10000 else 10,
        deterministic=True,
        render=False,
        verbose=1
    )
    print(f"📊 Evaluation: every {eval_callback.eval_freq:,} steps, saving BEST model to {best_model_dir}")
    
    # 2. LATEST CHECKPOINTS: For resuming interrupted training
    #    - Keeps only 2 most recent (rotating deletion)
    #    - Includes replay buffer for exact resume
    checkpoint_callback = RotatingCheckpointCallback(
        save_freq=checkpoint_freq,
        save_path=checkpoint_dir,
        name_prefix=f"{experiment_name}_checkpoint",
        max_checkpoints=max_checkpoints_to_keep,  # Automatic rotation
        save_replay_buffer=True,  # CRITICAL: Needed for proper DQN resume
        save_vecnormalize=True,
        verbose=1
    )
    print(f"💾 Checkpoints: every {checkpoint_freq:,} steps, keeping {max_checkpoints_to_keep} most recent in {checkpoint_dir}")
    
    # 3. PROGRESS TRACKING: For monitoring on Kaggle
    progress_callback = TrainingProgressCallback(
        total_timesteps=remaining_timesteps,
        log_freq=checkpoint_freq,  # Log at same frequency as checkpoints
        verbose=1
    )
    
    callbacks = [eval_callback, checkpoint_callback, progress_callback]
    
    print(f"\n{'='*70}")
    print(f" TRAINING STRATEGY:")
    print(f"   - Resume from: {'Latest checkpoint' if checkpoint_path else 'Scratch (new training)'}")
    print(f"   - Total timesteps: {total_timesteps:,}")
    print(f"   - Remaining timesteps: {remaining_timesteps:,}")
    print(f"   - Checkpoint strategy: Keep {max_checkpoints_to_keep} latest + 1 best")
    print(f"   - Checkpoint freq: {checkpoint_freq:,} steps")
    print(f"   - Evaluation freq: {eval_callback.eval_freq:,} steps")
    print(f"{'='*70}\n")
    
    start_time = time.time()
    
    # Train the model
    model.learn(
        total_timesteps=remaining_timesteps,
        callback=callbacks,
        progress_bar=True,
        reset_num_timesteps=False  # CRITICAL: Don't reset timestep counter when resuming
    )
    
    training_time = time.time() - start_time
    print(f"✅ Training completed in {training_time:.1f} seconds ({training_time/60:.1f} minutes)")
    
    # Save final model
    final_model_path = os.path.join(output_dir, f"{experiment_name}_final")
    model.save(final_model_path)
    print(f"💾 Final model saved: {final_model_path}.zip")
    
    # Save training metadata with checkpoint strategy explanation
    metadata = {
        "total_timesteps": total_timesteps,
        "completed_timesteps": completed_timesteps + remaining_timesteps,
        "training_time_seconds": training_time,
        "resumed_from_checkpoint": checkpoint_path is not None,
        "latest_checkpoint_path": checkpoint_path,
        "final_model_path": final_model_path + ".zip",
        "best_model_path": os.path.join(output_dir, "best_model", "best_model.zip"),
        "checkpoint_strategy": {
            "description": "3-level checkpoint system",
            "levels": {
                "latest": {
                    "purpose": "Resume interrupted training",
                    "count": max_checkpoints_to_keep,
                    "frequency_steps": checkpoint_freq,
                    "location": checkpoint_dir,
                    "includes_replay_buffer": True
                },
                "best": {
                    "purpose": "Final evaluation and deployment",
                    "count": 1,
                    "selection_criterion": "Highest mean evaluation reward",
                    "location": os.path.join(output_dir, "best_model"),
                    "note": "Never deleted, only updated on improvement"
                },
                "final": {
                    "purpose": "State at training completion",
                    "location": final_model_path + ".zip",
                    "note": "May not be the best model if performance degraded"
                }
            },
            "recommendation": "Use 'best_model.zip' for thesis results and deployment"
        }
    }
    
    metadata_path = os.path.join(output_dir, f"{experiment_name}_training_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"📁 CHECKPOINT SUMMARY:")
    print(f"   Latest checkpoint: {os.path.join(checkpoint_dir, '...')}")
    print(f"   Best model: {metadata['best_model_path']}")
    print(f"   Final model: {metadata['final_model_path']}")
    print(f"   Metadata: {metadata_path}")
    print(f"\n   ⚠️  IMPORTANT:")
    print(f"   - For RESUME: Use latest checkpoint automatically detected")
    print(f"   - For THESIS RESULTS: Use best_model.zip")
    print(f"   - For DEPLOYMENT: Use best_model.zip")
    print(f"{'='*70}\n")
    
    return model


def evaluate_agent(
    model: DQN,
    env: TrafficSignalEnv,
    n_episodes: int = 10,
    deterministic: bool = True
) -> dict:
    """Evaluate trained agent"""
    
    episode_rewards = []
    episode_summaries = []
    
    print(f"Evaluating agent over {n_episodes} episodes...")
    
    for episode in range(n_episodes):
        obs, info = env.reset(seed=42 + episode)
        episode_reward = 0
        done = False
        step_count = 0
        
        while not done:
            action, _states = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            step_count += 1
            done = terminated or truncated
        
        # Get episode summary
        summary = env.get_episode_summary()
        summary["episode_reward"] = episode_reward
        summary["steps"] = step_count
        
        episode_rewards.append(episode_reward)
        episode_summaries.append(summary)
        
        print(f"Episode {episode + 1}: Reward={episode_reward:.2f}, "
              f"Steps={step_count}, Switches={summary.get('phase_switches', 0)}")
    
    # Calculate statistics
    eval_results = {
        "n_episodes": n_episodes,
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "min_reward": np.min(episode_rewards),
        "max_reward": np.max(episode_rewards),
        "episode_summaries": episode_summaries
    }
    
    print(f"Evaluation Results:")
    print(f"  Mean Reward: {eval_results['mean_reward']:.2f} ± {eval_results['std_reward']:.2f}")
    print(f"  Reward Range: [{eval_results['min_reward']:.2f}, {eval_results['max_reward']:.2f}]")
    
    # Calculate performance metrics
    if episode_summaries:
        avg_queue = np.mean([ep.get('avg_total_queue_length', 0) for ep in episode_summaries])
        avg_throughput = np.mean([ep.get('avg_total_throughput', 0) for ep in episode_summaries])
        avg_switches = np.mean([ep.get('phase_switches', 0) for ep in episode_summaries])
        
        print(f"  Avg Queue Length: {avg_queue:.1f}")
        print(f"  Avg Throughput: {avg_throughput:.1f}")
        print(f"  Avg Phase Switches: {avg_switches:.1f}")
        
        eval_results.update({
            "avg_queue_length": avg_queue,
            "avg_throughput": avg_throughput,
            "avg_phase_switches": avg_switches
        })
    
    return eval_results


def run_baseline_comparison(env: TrafficSignalEnv, n_episodes: int = 10) -> dict:
    """Run fixed-time baseline for comparison"""
    
    print(f"Running fixed-time baseline over {n_episodes} episodes...")
    
    episode_summaries = []
    
    for episode in range(n_episodes):
        obs, info = env.reset(seed=42 + episode)
        done = False
        step_count = 0
        
        # Fixed-time control: switch every 60 seconds (6 steps @ 10s intervals)
        steps_per_phase = 6
        
        while not done:
            action = 1 if (step_count % steps_per_phase == 0 and step_count > 0) else 0
            obs, reward, terminated, truncated, info = env.step(action)
            step_count += 1
            done = terminated or truncated
        
        summary = env.get_episode_summary()
        summary["steps"] = step_count
        episode_summaries.append(summary)
    
    # Calculate baseline metrics
    baseline_results = {
        "n_episodes": n_episodes,
        "episode_summaries": episode_summaries
    }
    
    if episode_summaries:
        avg_queue = np.mean([ep.get('avg_total_queue_length', 0) for ep in episode_summaries])
        avg_throughput = np.mean([ep.get('avg_total_throughput', 0) for ep in episode_summaries])
        avg_switches = np.mean([ep.get('phase_switches', 0) for ep in episode_summaries])
        
        print(f"Baseline Results:")
        print(f"  Avg Queue Length: {avg_queue:.1f}")
        print(f"  Avg Throughput: {avg_throughput:.1f}")
        print(f"  Avg Phase Switches: {avg_switches:.1f}")
        
        baseline_results.update({
            "avg_queue_length": avg_queue,
            "avg_throughput": avg_throughput,
            "avg_phase_switches": avg_switches
        })
    
    return baseline_results


def main():
    """Main training function"""
    if is_kaggle:
        with open(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'config.json'), 'r') as f:
            config = json.load(f)
        config_dir = "configs"
        config_set = "lagos"
        output_dir = '/kaggle/working/'
        experiment_name = "dqn_baseline"
        timesteps = config['timesteps']
        eval_episodes = 10
        use_mock = config['use_mock']
        seed = 42
        no_baseline = False
        import subprocess
        subprocess.run(['pip', 'install', '-r', os.path.join(os.path.dirname(__file__), '..', '..', '..', 'requirements_rl.txt')])
    else:
        parser = argparse.ArgumentParser(description="Train DQN agent for traffic signal control")
        parser.add_argument("--config-dir", type=str, default="configs", help="Configuration directory")
        parser.add_argument("--config", type=str, default="default", help="Configuration set (default, lagos)")
        parser.add_argument("--output-dir", type=str, default="results", help="Output directory")
        parser.add_argument("--experiment-name", type=str, default="dqn_baseline", help="Experiment name")
        parser.add_argument("--timesteps", type=int, default=100000, help="Training timesteps")
        parser.add_argument("--eval-episodes", type=int, default=10, help="Evaluation episodes")
        parser.add_argument("--use-mock", action="store_true", help="Use mock ARZ simulator")
        parser.add_argument("--seed", type=int, default=42, help="Random seed")
        parser.add_argument("--no-baseline", action="store_true", help="Skip baseline comparison")
        parser.add_argument("--use-pydantic", action="store_true", help="Use Pydantic configs instead of YAML")
        parser.add_argument("--scenario", type=str, default="lagos", help="Scenario for Pydantic mode (simple, lagos, riemann)")
        parser.add_argument("--grid-size", type=int, default=200, help="Grid size N for Pydantic mode")
        
        args = parser.parse_args()
        config_dir = args.config_dir
        config_set = args.config
        output_dir = args.output_dir
        experiment_name = args.experiment_name
        timesteps = args.timesteps
        eval_episodes = args.eval_episodes
        use_mock = args.use_mock
        seed = args.seed
        no_baseline = args.no_baseline
        use_pydantic = args.use_pydantic
        scenario = args.scenario
        grid_size = args.grid_size
    
    # Setup logging
    setup_logging(level="INFO")
    
    # Set random seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Create environment using Pydantic or YAML configs
    if not is_kaggle and use_pydantic:
        # NEW: Use Pydantic configuration system
        print(f"✨ Using Pydantic configuration system (scenario: {scenario})")
        print(f"   Grid size: N={grid_size}")
        print(f"   Episode length: {timesteps * 15.0 / 1000.0:.0f}s")  # Approximate
        
        rl_config = RLConfigBuilder.for_training(
            scenario=scenario,
            N=grid_size,
            episode_length=3600.0,  # 1 hour episodes
            device="cpu",  # Will auto-detect GPU in future
            dt_decision=15.0
        )
        
        print("Creating environment with Pydantic configs...")
        env = create_environment_pydantic(rl_config, use_mock=use_mock)
        
        configs = rl_config.to_legacy_dict()  # For experiment tracking
        
    else:
        # LEGACY: Use YAML configuration system
        print(f"⚠️  Using legacy YAML configuration system")
        if not is_kaggle:
            print(f"   💡 Tip: Use --use-pydantic flag to use new Pydantic configs")
        
        # Load configurations based on selected config set
        print(f"Loading {config_set} configurations...")
        if config_set == "lagos":
            # Load Lagos-specific configurations
            configs = {}
            configs["endpoint"] = load_config(os.path.join(config_dir, "endpoint.yaml"))
            configs["network"] = load_config(os.path.join(config_dir, "network_real.yaml"))
            configs["env"] = load_config(os.path.join(config_dir, "env_lagos.yaml"))
            configs["signals"] = load_config(os.path.join(config_dir, "signals_lagos.yaml"))
            print("   ✓ Using Lagos Victoria Island configuration set")
        else:
            # Load default configurations
            configs = load_configs(config_dir)
            print("   ✓ Using default configuration set")
        
        if not validate_config_consistency(configs):
            print("Configuration validation failed!")
            return 1
        
        print("Creating environment with YAML configs...")
        env = create_environment(configs, use_mock=use_mock)
    
    # Initialize experiment tracker
    tracker = ExperimentTracker(output_dir)
    
    # Start experiment
    experiment_config = {
        "algorithm": "DQN",
        "timesteps": timesteps,
        "seed": seed,
        "use_mock": use_mock,
        "configs": configs
    }
    
    tracker.start_experiment(
        name=experiment_name,
        config=experiment_config,
        description="Baseline DQN training for traffic signal control"
    )
    
    try:
        # Train agent
        print("Training DQN agent...")
        model = train_dqn_agent(
            env=env,
            total_timesteps=timesteps,
            seed=seed,
            output_dir=output_dir,
            experiment_name=experiment_name
        )
        
        # Evaluate agent
        print("Evaluating trained agent...")
        eval_results = evaluate_agent(model, env, n_episodes=eval_episodes)
        
        results = {"evaluation": eval_results}
        
        # Run baseline comparison
        if not no_baseline:
            baseline_results = run_baseline_comparison(env, n_episodes=eval_episodes)
            results["baseline"] = baseline_results
            
            # Compare performance
            if baseline_results.get("avg_queue_length") and eval_results.get("avg_queue_length"):
                queue_improvement = (
                    (baseline_results["avg_queue_length"] - eval_results["avg_queue_length"]) 
                    / baseline_results["avg_queue_length"] * 100
                )
                
                throughput_improvement = (
                    (eval_results["avg_throughput"] - baseline_results["avg_throughput"])
                    / baseline_results["avg_throughput"] * 100
                )
                
                print(f"\nPerformance Comparison:")
                print(f"  Queue Length Improvement: {queue_improvement:.1f}%")
                print(f"  Throughput Improvement: {throughput_improvement:.1f}%")
                
                results["comparison"] = {
                    "queue_improvement_pct": queue_improvement,
                    "throughput_improvement_pct": throughput_improvement
                }
        
        # Save results
        results_file = os.path.join(output_dir, f"{experiment_name}_results.json")
        save_training_results(results, results_file)
        
        # Finish experiment
        tracker.finish_experiment(results)
        
        print(f"Training completed successfully!")
        print(f"Results saved to: {results_file}")
        
        return 0
        
    except Exception as e:
        print(f"Training failed: {str(e)}")
        if tracker.current_experiment:
            tracker.current_experiment["status"] = "failed"
            tracker.current_experiment["error"] = str(e)
            tracker.finish_experiment()
        return 1
    
    finally:
        env.close()


if __name__ == "__main__":
    exit(main())
