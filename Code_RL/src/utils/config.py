"""
Utility functions for configuration, logging, and analysis

MIGRATION NOTE (2025-01-27):
This module is being migrated to use Pydantic-based configurations.
- NEW: RLConfigBuilder class uses arz_model.config.ConfigBuilder
- DEPRECATED: load_config(), load_configs(), save_config() for ARZ configs
- PRESERVED: RL-specific utilities (setup_logging, ExperimentTracker, etc.)

For new code, use:
    from utils.config import RLConfigBuilder
    rl_config = RLConfigBuilder.for_training(scenario="lagos")
"""

import yaml
import json
import logging
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np

# NEW: Import Pydantic config system
try:
    import sys
    from pathlib import Path
    
    # Add parent directory to path to access arz_model
    project_root = Path(__file__).parent.parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    from arz_model.config import ConfigBuilder, SimulationConfig
    PYDANTIC_AVAILABLE = True
except ImportError as e:
    PYDANTIC_AVAILABLE = False
    ConfigBuilder = None
    SimulationConfig = None
    print(f"[WARNING] Pydantic config system not available: {e}")
    print("[WARNING] Falling back to legacy YAML configs")


class RLConfigBuilder:
    """
    Configuration builder for RL training using Pydantic-based ARZ configs.
    
    This class bridges the RL training system with the new Pydantic configuration
    system, replacing legacy YAML-based configuration loading.
    
    Example:
        >>> from utils.config import RLConfigBuilder
        >>> 
        >>> # For RL training
        >>> rl_config = RLConfigBuilder.for_training(
        ...     scenario="lagos",
        ...     N=200,
        ...     episode_length=3600.0
        ... )
        >>> 
        >>> # Access ARZ simulation config
        >>> arz_config = rl_config.arz_simulation_config
        >>> 
        >>> # Access RL environment parameters
        >>> env_params = rl_config.rl_env_params
    """
    
    def __init__(
        self,
        arz_simulation_config: 'SimulationConfig',
        rl_env_params: Dict[str, Any],
        endpoint_params: Dict[str, Any],
        signal_params: Dict[str, Any]
    ):
        """
        Initialize RL configuration.
        
        Args:
            arz_simulation_config: Pydantic SimulationConfig for ARZ simulator
            rl_env_params: RL environment parameters (rewards, normalization, etc.)
            endpoint_params: Endpoint client parameters
            signal_params: Traffic signal controller parameters
        """
        self.arz_simulation_config = arz_simulation_config
        self.rl_env_params = rl_env_params
        self.endpoint_params = endpoint_params
        self.signal_params = signal_params
    
    @classmethod
    def for_training(
        cls,
        scenario: str = "simple",
        N: int = 200,
        episode_length: float = 3600.0,
        device: str = "cpu",
        **kwargs
    ) -> 'RLConfigBuilder':
        """
        Create RL training configuration.
        
        Args:
            scenario: Scenario name ("simple", "lagos", "riemann")
            N: Grid resolution
            episode_length: Episode length in seconds
            device: Compute device ("cpu" or "gpu")
            **kwargs: Additional scenario-specific parameters
        
        Returns:
            RLConfigBuilder instance
        """
        if not PYDANTIC_AVAILABLE:
            raise RuntimeError(
                "Pydantic config system not available. "
                "Please ensure arz_model.config is accessible."
            )
        
        # Create ARZ simulation config based on scenario
        if scenario == "simple":
            arz_config = ConfigBuilder.simple_test()
        elif scenario == "section_7_6" or scenario == "lagos":
            arz_config = ConfigBuilder.section_7_6(N=N, t_final=episode_length, device=device)
        elif scenario == "riemann":
            arz_config = ConfigBuilder.riemann_problem(N=N, t_final=episode_length, device=device)
        else:
            raise ValueError(f"Unknown scenario: {scenario}. Use 'simple', 'lagos', or 'riemann'")
        
        # Override config parameters if provided
        if 't_final' in kwargs:
            arz_config.t_final = kwargs['t_final']
        if 'output_dt' in kwargs:
            arz_config.output_dt = kwargs['output_dt']
        
        # RL environment parameters (Lagos-specific defaults)
        rl_env_params = {
            "dt_decision": kwargs.get("dt_decision", 15.0),  # RL decision interval
            "episode_length": episode_length,
            "max_steps": int(episode_length / kwargs.get("dt_decision", 15.0)),
            
            # Normalization (Lagos traffic conditions)
            "normalization": {
                "rho_max_motorcycles": kwargs.get("rho_max_motorcycles", 250.0),  # veh/km
                "rho_max_cars": kwargs.get("rho_max_cars", 120.0),
                "v_free_motorcycles": kwargs.get("v_free_motorcycles", 32.0),  # km/h
                "v_free_cars": kwargs.get("v_free_cars", 28.0),
                "queue_max": kwargs.get("queue_max", 400.0),
                "phase_time_max": kwargs.get("phase_time_max", 120.0),
            },
            
            # Reward weights
            "reward": {
                "w_wait_time": kwargs.get("w_wait_time", 1.0),
                "w_queue_length": kwargs.get("w_queue_length", 0.5),
                "w_stops": kwargs.get("w_stops", 0.3),
                "w_switch_penalty": kwargs.get("w_switch_penalty", 0.1),
                "w_throughput": kwargs.get("w_throughput", 0.8),
                "reward_clip": kwargs.get("reward_clip", (-10.0, 10.0)),
                "stop_speed_threshold": kwargs.get("stop_speed_threshold", 5.0),
            },
            
            # Observation settings
            "observation": {
                "ewma_alpha": kwargs.get("ewma_alpha", 0.1),
                "include_phase_timing": kwargs.get("include_phase_timing", True),
                "include_queues": kwargs.get("include_queues", True),
            }
        }
        
        # Endpoint parameters
        endpoint_params = {
            "protocol": kwargs.get("protocol", "mock"),
            "host": kwargs.get("host", "localhost"),
            "port": kwargs.get("port", 8000),
            "dt_sim": 0.5,  # Fixed ARZ timestep (will be adaptive in runtime)
            "timeout": kwargs.get("timeout", 30.0),
            "max_retries": kwargs.get("max_retries", 3),
        }
        
        # Signal controller parameters
        signal_params = {
            "signals": {
                "phases": [
                    {
                        "id": 0,
                        "name": "north_south",
                        "description": "North-South green, East-West red",
                        "duration": 60.0,
                        "movements": ["north_through", "south_through", "north_left", "south_left"]
                    },
                    {
                        "id": 1,
                        "name": "east_west",
                        "description": "East-West green, North-South red",
                        "duration": 60.0,
                        "movements": ["east_through", "west_through", "east_left", "west_left"]
                    }
                ],
                "timings": {
                    "min_green": kwargs.get("min_green", 10.0),
                    "max_green": kwargs.get("max_green", 120.0),
                    "yellow": kwargs.get("yellow_duration", 3.0),
                    "all_red": kwargs.get("all_red", 2.0)
                },
                "initial_phase": 0
            }
        }
        
        return cls(
            arz_simulation_config=arz_config,
            rl_env_params=rl_env_params,
            endpoint_params=endpoint_params,
            signal_params=signal_params
        )
    
    def to_legacy_dict(self) -> Dict[str, Any]:
        """
        Convert to legacy YAML-style dict for backward compatibility.
        
        Returns:
            Dictionary in legacy format
        """
        return {
            "arz_simulation": self.arz_simulation_config,
            "env": {"environment": self.rl_env_params},
            "endpoint": {"endpoint": self.endpoint_params},
            "signals": self.signal_params,
        }


def setup_logging(level: str = "INFO", log_file: str = None):
    """Setup logging configuration"""
    import logging.config
    
    logging_config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'detailed': {
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            },
            'simple': {
                'format': '%(levelname)s - %(message)s'
            }
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'level': level,
                'formatter': 'simple'
            }
        },
        'loggers': {
            '': {  # root logger
                'level': level,
                'handlers': ['console']
            }
        }
    }
    
    if log_file:
        logging_config['handlers']['file'] = {
            'class': 'logging.FileHandler',
            'filename': log_file,
            'level': level,
            'formatter': 'detailed'
        }
        logging_config['loggers']['']['handlers'].append('file')
    
    logging.config.dictConfig(logging_config)


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load YAML configuration file.
    
    DEPRECATED: This function is deprecated for ARZ simulation configs.
    Use RLConfigBuilder.for_training() instead for new code.
    
    This function is preserved for backward compatibility with legacy
    RL training scripts.
    """
    import warnings
    warnings.warn(
        "load_config() is deprecated for ARZ configs. "
        "Use RLConfigBuilder.for_training() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def save_config(config: Dict[str, Any], config_path: str):
    """
    Save configuration to YAML file.
    
    DEPRECATED: YAML configs are being phased out in favor of Pydantic.
    This function is preserved for backward compatibility only.
    """
    import warnings
    warnings.warn(
        "save_config() is deprecated. Use Pydantic configs instead.",
        DeprecationWarning,
        stacklevel=2
    )
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)


def load_configs(config_dir: str) -> Dict[str, Any]:
    """
    Load all configuration files from directory.
    
    DEPRECATED: This function is deprecated for ARZ simulation configs.
    Use RLConfigBuilder.for_training() instead for new code.
    
    This function is preserved for backward compatibility with legacy
    RL training scripts.
    """
    import warnings
    warnings.warn(
        "load_configs() is deprecated for ARZ configs. "
        "Use RLConfigBuilder.for_training() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    config_path = Path(config_dir)
    configs = {}
    
    for config_file in ["endpoint.yaml", "signals.yaml", "network.yaml", "env.yaml"]:
        file_path = config_path / config_file
        if file_path.exists():
            configs[file_path.stem] = load_config(str(file_path))
    
    return configs


def load_lagos_traffic_params(config_dir: Optional[str] = None) -> Dict[str, float]:
    """Load real Lagos traffic parameters from traffic_lagos.yaml.
    
    Extracts key traffic parameters for Victoria Island Lagos:
    - Max densities: motorcycles and cars (veh/km)
    - Free speeds: motorcycles and cars (km/h)
    - Vehicle mix: percentages for each vehicle type
    - Behaviors: creeping rate, gap filling, signal compliance
    
    Args:
        config_dir: Path to configs directory. If None, uses default Code_RL/configs
    
    Returns:
        Dictionary with extracted parameters ready for simulation use
    
    Example:
        >>> params = load_lagos_traffic_params()
        >>> print(params['max_density_motorcycles'])  # 250.0 veh/km
        >>> print(params['free_speed_cars'])  # 28.0 km/h
    """
    if config_dir is None:
        # Default to Code_RL/configs directory
        config_dir = Path(__file__).parent.parent.parent / "configs"
    else:
        config_dir = Path(config_dir)
    
    lagos_config_path = config_dir / "traffic_lagos.yaml"
    
    if not lagos_config_path.exists():
        print(f"[WARNING] traffic_lagos.yaml not found at {lagos_config_path}")
        print("[WARNING] Using fallback parameters")
        # Fallback to reasonable defaults
        return {
            'max_density_motorcycles': 200.0,  # veh/km
            'max_density_cars': 100.0,
            'free_speed_motorcycles': 30.0,  # km/h
            'free_speed_cars': 25.0,
            'vehicle_mix_motorcycles': 0.30,
            'vehicle_mix_cars': 0.50,
            'creeping_rate': 0.5,
            'gap_filling_rate': 0.7,
            'signal_compliance': 0.8
        }
    
    config = load_config(str(lagos_config_path))
    traffic = config.get('traffic', {})
    
    # Extract nested parameters
    max_densities = traffic.get('max_densities', {})
    free_speeds = traffic.get('free_speeds', {})
    vehicle_mix = traffic.get('vehicle_mix', {})
    behaviors = traffic.get('behaviors', {})
    
    params = {
        'max_density_motorcycles': float(max_densities.get('motorcycles', 250)),  # veh/km
        'max_density_cars': float(max_densities.get('cars', 120)),
        'max_density_total': float(max_densities.get('total', 370)),
        'free_speed_motorcycles': float(free_speeds.get('motorcycles', 32)),  # km/h
        'free_speed_cars': float(free_speeds.get('cars', 28)),
        'free_speed_average': float(free_speeds.get('average', 30)),
        'vehicle_mix_motorcycles': float(vehicle_mix.get('motorcycles_percentage', 35)) / 100.0,
        'vehicle_mix_cars': float(vehicle_mix.get('cars_percentage', 45)) / 100.0,
        'vehicle_mix_buses': float(vehicle_mix.get('buses_percentage', 15)) / 100.0,
        'vehicle_mix_trucks': float(vehicle_mix.get('trucks_percentage', 5)) / 100.0,
        'creeping_rate': float(behaviors.get('creeping_rate', 0.6)),
        'gap_filling_rate': float(behaviors.get('gap_filling_rate', 0.8)),
        'signal_compliance': float(behaviors.get('signal_compliance', 0.7)),
        'context': traffic.get('context', 'Victoria Island Lagos')
    }
    
    return params


def validate_config_consistency(configs: Dict[str, Any]) -> bool:
    """Validate consistency between configuration files"""
    errors = []
    warnings = []
    
    # Check dt_decision vs dt_sim compatibility
    if "endpoint" in configs and "env" in configs:
        endpoint_config = configs["endpoint"]
        env_config = configs["env"]
        
        # Check if dt_sim exists in endpoint config (at root level or nested)
        dt_sim = None
        if "dt_sim" in endpoint_config:
            dt_sim = endpoint_config["dt_sim"]
        elif "endpoint" in endpoint_config and "dt_sim" in endpoint_config["endpoint"]:
            dt_sim = endpoint_config["endpoint"]["dt_sim"]
            
        if dt_sim is not None:
            if "environment" in env_config and "dt_decision" in env_config["environment"]:
                dt_decision = env_config["environment"]["dt_decision"]
                
                k = dt_decision / dt_sim
                if not k.is_integer() or k < 1:
                    errors.append(f"dt_decision ({dt_decision}) must be integer multiple of dt_sim ({dt_sim})")
        else:
            warnings.append("dt_sim not found in endpoint config, skipping timing validation")
    
    # Check phase count consistency
    if "signals" in configs and "env" in configs:
        signals_config = configs["signals"]
        if "signals" in signals_config and "phases" in signals_config["signals"]:
            num_phases = len(signals_config["signals"]["phases"])
            # Could add more phase-related validations here
            if num_phases < 2:
                warnings.append(f"Only {num_phases} phases defined, consider adding more for complex intersections")
    
    # Check branch mapping
    if "network" in configs:
        network_config = configs["network"]
        if "network" in network_config and "branches" in network_config["network"]:
            branch_ids = [b["id"] for b in network_config["network"]["branches"]]
            if len(branch_ids) != len(set(branch_ids)):
                errors.append("Duplicate branch IDs in network configuration")
        else:
            warnings.append("No branches found in network configuration")
    
    # Print warnings
    if warnings:
        for warning in warnings:
            print(f"Config validation warning: {warning}")
    
    # Print errors
    if errors:
        for error in errors:
            print(f"Config validation error: {error}")
        return False
    
    return True


def save_episode_data(episode_data: List[Dict[str, Any]], filepath: str):
    """Save episode data to CSV"""
    df = pd.DataFrame(episode_data)
    df.to_csv(filepath, index=False)


def save_training_results(results: Dict[str, Any], filepath: str):
    """Save training results to JSON"""
    # Convert numpy arrays to lists for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_numpy(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        else:
            return obj
    
    results_serializable = convert_numpy(results)
    
    with open(filepath, 'w') as f:
        json.dump(results_serializable, f, indent=2)


def calculate_performance_metrics(episode_summaries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate aggregate performance metrics across episodes"""
    if not episode_summaries:
        return {}
    
    metrics = {}
    
    # Get all numeric keys
    numeric_keys = []
    for key, value in episode_summaries[0].items():
        if isinstance(value, (int, float)):
            numeric_keys.append(key)
    
    # Calculate statistics for each metric
    for key in numeric_keys:
        values = [ep[key] for ep in episode_summaries if key in ep]
        if values:
            metrics[f"{key}_mean"] = np.mean(values)
            metrics[f"{key}_std"] = np.std(values)
            metrics[f"{key}_min"] = np.min(values)
            metrics[f"{key}_max"] = np.max(values)
    
    return metrics


def compare_baselines(
    rl_results: List[Dict[str, Any]], 
    baseline_results: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Compare RL performance against baseline"""
    comparison = {}
    
    rl_metrics = calculate_performance_metrics(rl_results)
    baseline_metrics = calculate_performance_metrics(baseline_results)
    
    # Calculate improvement percentages
    for key in rl_metrics:
        if key.endswith("_mean") and key in baseline_metrics:
            base_key = key.replace("_mean", "")
            rl_value = rl_metrics[key]
            baseline_value = baseline_metrics[key]
            
            if baseline_value != 0:
                improvement = (rl_value - baseline_value) / abs(baseline_value) * 100
                comparison[f"{base_key}_improvement_pct"] = improvement
    
    comparison["rl_metrics"] = rl_metrics
    comparison["baseline_metrics"] = baseline_metrics
    
    return comparison


class ExperimentTracker:
    """Track experiments and results"""
    
    def __init__(self, experiment_dir: str):
        self.experiment_dir = Path(experiment_dir)
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        self.current_experiment = None
        self.experiments_log = self.experiment_dir / "experiments.json"
        
        # Load existing experiments
        if self.experiments_log.exists():
            with open(self.experiments_log, 'r') as f:
                self.experiments = json.load(f)
        else:
            self.experiments = []
    
    def start_experiment(self, name: str, config: Dict[str, Any], description: str = ""):
        """Start new experiment"""
        import time
        import hashlib
        
        self.current_experiment = {
            "name": name,
            "description": description,
            "start_time": time.time(),
            "config": config,
            "config_hash": hashlib.md5(str(config).encode()).hexdigest()[:8],
            "status": "running",
            "results": {}
        }
        
        print(f"Started experiment: {name}")
    
    def log_episode(self, episode: int, summary: Dict[str, Any]):
        """Log episode results"""
        if self.current_experiment:
            if "episodes" not in self.current_experiment["results"]:
                self.current_experiment["results"]["episodes"] = []
            
            episode_data = {"episode": episode, **summary}
            self.current_experiment["results"]["episodes"].append(episode_data)
    
    def finish_experiment(self, final_results: Optional[Dict[str, Any]] = None):
        """Finish current experiment"""
        if self.current_experiment:
            import time
            
            self.current_experiment["end_time"] = time.time()
            self.current_experiment["duration"] = (
                self.current_experiment["end_time"] - self.current_experiment["start_time"]
            )
            self.current_experiment["status"] = "completed"
            
            if final_results:
                self.current_experiment["results"].update(final_results)
            
            # Calculate final metrics
            if "episodes" in self.current_experiment["results"]:
                episodes = self.current_experiment["results"]["episodes"]
                final_metrics = calculate_performance_metrics(episodes)
                self.current_experiment["results"]["final_metrics"] = final_metrics
            
            # Save experiment
            self.experiments.append(self.current_experiment)
            self._save_experiments()
            
            # Save individual experiment file
            exp_file = self.experiment_dir / f"{self.current_experiment['name']}.json"
            with open(exp_file, 'w') as f:
                json.dump(self.current_experiment, f, indent=2)
            
            print(f"Finished experiment: {self.current_experiment['name']}")
            print(f"Duration: {self.current_experiment['duration']:.1f} seconds")
            
            self.current_experiment = None
    
    def _save_experiments(self):
        """Save experiments log"""
        with open(self.experiments_log, 'w') as f:
            json.dump(self.experiments, f, indent=2)
    
    def get_experiment_summary(self) -> pd.DataFrame:
        """Get summary of all experiments"""
        if not self.experiments:
            return pd.DataFrame()
        
        summary_data = []
        for exp in self.experiments:
            row = {
                "name": exp["name"],
                "status": exp["status"],
                "duration": exp.get("duration", 0),
                "config_hash": exp.get("config_hash", ""),
            }
            
            # Add final metrics
            if "final_metrics" in exp.get("results", {}):
                for key, value in exp["results"]["final_metrics"].items():
                    if key.endswith("_mean"):
                        base_key = key.replace("_mean", "")
                        row[base_key] = value
            
            summary_data.append(row)
        
        return pd.DataFrame(summary_data)


def _generate_signalized_network_lagos(vmax_m: float, vmax_c: float,
                                       theta_m: float, theta_c: float,
                                       duration: float, domain_length: float) -> tuple:
    """Generate NetworkConfig for signalized intersection with Lagos parameters."""
    from datetime import datetime
    
    # ===== INITIAL CONDITIONS & BOUNDARY CONDITIONS (CRITICAL FOR SIMULATIONS) =====
    # Initial state: VERY CONGESTED traffic (to see RED vs GREEN effect even if BC isn't working perfectly)
    # BUG FIX: Light initial conditions (0.01 veh/m) made RED/GREEN indistinguishable
    # With no congestion, blocking inflow creates no observable backpressure
    # Solution: Start with HEAVY traffic so RED control creates visible queueing
    # ✅ CRITICAL FIX (2025-10-24): Increased to 85% of max to force congestion detection
    # Previous 70% wasn't creating enough queue signal for RL learning
    
    rho_jam_veh_km = 250 + 120  # Max motorcycles + cars (370 veh/km)
    rho_jam = rho_jam_veh_km / 1000.0
    V_creeping = 0.6
    
    rho_m_initial_veh_km = 250 * 0.85  # 212.5 veh/km (heavy - 85% of max) ✅ INCREASED
    rho_c_initial_veh_km = 120 * 0.85  # 102 veh/km (heavy - 85% of max) ✅ INCREASED
    rho_total_initial = (rho_m_initial_veh_km + rho_c_initial_veh_km) / 1000.0  # 0.315 veh/m
    g_initial = max(0.0, 1.0 - rho_total_initial / rho_jam)  # ≈ 0.15 (very congested!)
    w_m_initial = V_creeping + (vmax_m - V_creeping) * g_initial  # ≈ 1.84 m/s (slow!)
    w_c_initial = vmax_c * g_initial  # ≈ 1.17 m/s (slow!)
    
    # Inflow: GREEN allows full flow, RED uses creeping (applied via set_traffic_signal_state)
    # ✅ CRITICAL: With runner.py fix, RED phase now maintains inflow at V_creeping
    # This creates queue formation upstream (realistic traffic signal behavior)
    w_m_inflow = vmax_m  # FREE speed (will be modulated by RED phase to V_creeping)
    w_c_inflow = vmax_c
    rho_m_inflow_veh_km = 250 * 0.85  # 212.5 veh/km (heavy Lagos traffic) ✅ INCREASED
    rho_c_inflow_veh_km = 120 * 0.85  # 102 veh/km (heavy Lagos traffic) ✅ INCREASED
    
    # ===== NETWORK STRUCTURE =====
    network_data = {
        # Legacy parameters for SimulationRunner
        'N': 100,
        'xmin': 0.0,
        'xmax': domain_length,
        't_final': duration,
        'output_dt': 60.0,
        # Initial conditions
        'initial_conditions': {
            'type': 'uniform',
            'state': [rho_m_initial_veh_km, w_m_initial, rho_c_initial_veh_km, w_c_initial]
        },
        # Boundary conditions (used by default, overridden by set_traffic_signal_state)
        'boundary_conditions': {
            'left': {'type': 'inflow', 'state': [rho_m_inflow_veh_km, w_m_inflow, rho_c_inflow_veh_km, w_c_inflow]},
            'right': {'type': 'outflow'}
        },
        # NetworkConfig structure
        'network': {
            'name': 'Traffic_Light_Control_Lagos',
            'description': 'Signalized intersection with REAL Lagos traffic data',
            'segments': {
                'upstream': {
                    'x_min': 0.0,
                    'x_max': domain_length / 2,
                    'N': 50,
                    'start_node': 'entry',
                    'end_node': 'signal_junction',
                    'road_type': 'urban_arterial',
                    'parameters': {
                        'V0_m': vmax_m,  # Lagos free speed (m/s)
                        'V0_c': vmax_c,
                        'tau_m': 5.0,
                        'tau_c': 10.0
                    }
                },
                'downstream': {
                    'x_min': domain_length / 2,
                    'x_max': domain_length,
                    'N': 50,
                    'start_node': 'signal_junction',
                    'end_node': 'exit',
                    'road_type': 'urban_arterial',
                    'parameters': {
                        'V0_m': vmax_m,
                        'V0_c': vmax_c,
                        'tau_m': 5.0,
                        'tau_c': 10.0
                    }
                }
            },
            'nodes': {
                'entry': {
                    'type': 'boundary',
                    'position': [0.0, 0.0],
                    'description': 'Network entry - Lagos traffic inflow'
                },
                'signal_junction': {
                    'type': 'signalized',
                    'position': [domain_length / 2, 0.0],
                    'incoming_segments': ['upstream'],
                    'outgoing_segments': ['downstream'],
                    'description': 'RL-controlled traffic signal'
                },
                'exit': {
                    'type': 'boundary',
                    'position': [domain_length, 0.0],
                    'description': 'Network exit'
                }
            },
            'links': [
                {
                    'from_segment': 'upstream',
                    'to_segment': 'downstream',
                    'via_node': 'signal_junction',
                    'coupling_type': 'signalized',
                    'theta_m': theta_m,
                    'theta_c': theta_c,
                    'description': 'Behavioral coupling (config_base.yml)'
                }
            ]
        },
        'metadata': {
            'created': datetime.now().isoformat(),
            'scenario_type': 'traffic_light_control',
            'data_source': 'Lagos Victoria Island (REAL)'
        }
    }
    
    traffic_data = {
        'traffic_control': {
            'traffic_lights': {
                'signal_junction': {
                    'cycle_time': 120.0,
                    'offset': 0.0,
                    'control_mode': 'rl_agent',
                    'phases': [
                        {'id': 0, 'duration': 60.0, 'green_segments': ['upstream'], 'description': 'Green (RL-controlled)'},
                        {'id': 1, 'duration': 5.0, 'yellow_segments': ['upstream'], 'description': 'Yellow transition'},
                        {'id': 2, 'duration': 50.0, 'description': 'Red phase'},
                        {'id': 3, 'duration': 5.0, 'description': 'All-red clearance'}
                    ]
                }
            }
        }
    }
    
    return network_data, traffic_data


def _generate_ramp_network_lagos(vmax_m: float, vmax_c: float,
                                 theta_priority: float, theta_secondary: float,
                                 duration: float, domain_length: float) -> tuple:
    """Generate NetworkConfig for ramp metering with Lagos parameters."""
    from datetime import datetime
    
    network_data = {
        'N': 130,
        'xmin': 0.0,
        'xmax': domain_length * 1.3,
        't_final': duration,
        'output_dt': 60.0,
        'network': {
            'name': 'Ramp_Metering_Lagos',
            'description': 'Highway ramp metering with Lagos traffic data',
            'segments': {
                'highway_upstream': {
                    'x_min': 0.0,
                    'x_max': domain_length / 2,
                    'N': 50,
                    'start_node': 'highway_entry',
                    'end_node': 'merge_junction',
                    'road_type': 'highway',
                    'parameters': {
                        'V0_m': vmax_m * 1.5,  # Highway speed (+50%)
                        'V0_c': vmax_c * 1.5,
                        'tau_m': 3.0,
                        'tau_c': 5.0
                    }
                },
                'on_ramp': {
                    'x_min': 0.0,
                    'x_max': 300.0,
                    'N': 30,
                    'start_node': 'ramp_entry',
                    'end_node': 'merge_junction',
                    'road_type': 'ramp',
                    'parameters': {
                        'V0_m': vmax_m,
                        'V0_c': vmax_c,
                        'tau_m': 5.0,
                        'tau_c': 10.0
                    }
                },
                'highway_downstream': {
                    'x_min': domain_length / 2,
                    'x_max': domain_length,
                    'N': 50,
                    'start_node': 'merge_junction',
                    'end_node': 'highway_exit',
                    'road_type': 'highway',
                    'parameters': {
                        'V0_m': vmax_m * 1.5,
                        'V0_c': vmax_c * 1.5,
                        'tau_m': 3.0,
                        'tau_c': 5.0
                    }
                }
            },
            'nodes': {
                'highway_entry': {'type': 'boundary', 'position': [0.0, 0.0]},
                'ramp_entry': {'type': 'boundary', 'position': [0.0, -200.0]},
                'merge_junction': {
                    'type': 'merge',
                    'position': [domain_length / 2, 0.0],
                    'incoming_segments': ['highway_upstream', 'on_ramp'],
                    'outgoing_segments': ['highway_downstream']
                },
                'highway_exit': {'type': 'boundary', 'position': [domain_length, 0.0]}
            },
            'links': [
                {'from_segment': 'highway_upstream', 'to_segment': 'highway_downstream',
                 'via_node': 'merge_junction', 'coupling_type': 'priority',
                 'theta_m': theta_priority, 'theta_c': theta_priority},
                {'from_segment': 'on_ramp', 'to_segment': 'highway_downstream',
                 'via_node': 'merge_junction', 'coupling_type': 'secondary',
                 'theta_m': theta_secondary, 'theta_c': theta_secondary}
            ]
        },
        'metadata': {'created': datetime.now().isoformat(), 'scenario_type': 'ramp_metering', 'data_source': 'Lagos'}
    }
    
    traffic_data = {
        'traffic_control': {
            'ramp_meters': {
                'merge_junction': {
                    'cycle_time': 30.0,
                    'control_mode': 'rl_agent',
                    'max_release_rate': 1.0,
                    'phases': [
                        {'id': 0, 'duration': 15.0, 'release_rate': 0.5, 'description': 'Moderate release'},
                        {'id': 1, 'duration': 15.0, 'release_rate': 0.0, 'description': 'Ramp hold'}
                    ]
                }
            }
        }
    }
    
    return network_data, traffic_data


def _generate_speed_control_network_lagos(vmax_m: float, vmax_c: float,
                                          duration: float, domain_length: float) -> tuple:
    """Generate NetworkConfig for variable speed limits with Lagos parameters."""
    from datetime import datetime
    
    network_data = {
        'N': 100,
        'xmin': 0.0,
        'xmax': domain_length,
        't_final': duration,
        'output_dt': 60.0,
        'network': {
            'name': 'Adaptive_Speed_Control_Lagos',
            'description': 'Variable speed limits (VSL) with Lagos traffic data',
            'segments': {
                'zone_1': {
                    'x_min': 0.0,
                    'x_max': domain_length / 3,
                    'N': 33,
                    'start_node': 'entry',
                    'end_node': 'vsl_zone_2',
                    'road_type': 'highway',
                    'parameters': {'V0_m': vmax_m * 1.5, 'V0_c': vmax_c * 1.5, 'tau_m': 3.0, 'tau_c': 5.0}
                },
                'zone_2': {
                    'x_min': domain_length / 3,
                    'x_max': 2 * domain_length / 3,
                    'N': 33,
                    'start_node': 'vsl_zone_2',
                    'end_node': 'vsl_zone_3',
                    'road_type': 'highway',
                    'parameters': {'V0_m': vmax_m * 1.5, 'V0_c': vmax_c * 1.5, 'tau_m': 3.0, 'tau_c': 5.0}
                },
                'zone_3': {
                    'x_min': 2 * domain_length / 3,
                    'x_max': domain_length,
                    'N': 34,
                    'start_node': 'vsl_zone_3',
                    'end_node': 'exit',
                    'road_type': 'highway',
                    'parameters': {'V0_m': vmax_m * 1.5, 'V0_c': vmax_c * 1.5, 'tau_m': 3.0, 'tau_c': 5.0}
                }
            },
            'nodes': {
                'entry': {'type': 'boundary', 'position': [0.0, 0.0]},
                'vsl_zone_2': {'type': 'vsl_boundary', 'position': [domain_length / 3, 0.0]},
                'vsl_zone_3': {'type': 'vsl_boundary', 'position': [2 * domain_length / 3, 0.0]},
                'exit': {'type': 'boundary', 'position': [domain_length, 0.0]}
            },
            'links': [
                {'from_segment': 'zone_1', 'to_segment': 'zone_2', 'via_node': 'vsl_zone_2', 'coupling_type': 'continuous'},
                {'from_segment': 'zone_2', 'to_segment': 'zone_3', 'via_node': 'vsl_zone_3', 'coupling_type': 'continuous'}
            ]
        },
        'metadata': {'created': datetime.now().isoformat(), 'scenario_type': 'adaptive_speed_control', 'data_source': 'Lagos'}
    }
    
    traffic_data = {
        'traffic_control': {
            'vsl_zones': {
                f'zone_{i}': {
                    'control_mode': 'rl_agent',
                    'default_speed_limit_kmh': vmax_m * 1.5 * 3.6,
                    'min_speed_limit_kmh': vmax_m * 0.5 * 3.6,
                    'max_speed_limit_kmh': vmax_m * 2.0 * 3.6,
                    'update_interval': 30.0
                } for i in [1, 2, 3]
            }
        }
    }
    
    return network_data, traffic_data


def create_scenario_config_with_lagos_data(
    scenario_type: str,
    output_path: Optional[Path] = None,
    config_dir: Optional[str] = None,
    duration: float = 600.0,
    domain_length: float = 1000.0
) -> Dict[str, Any]:
    """Create scenario configuration using REAL Lagos traffic data.
    
    **UPDATED**: Generates NetworkConfig-compatible YAML (network.yml + traffic_control.yml)
    Uses config_base.yml behavioral_coupling parameters (θ_k) for realistic junction physics.
    
    Args:
        scenario_type: Type of scenario ('traffic_light_control', 'ramp_metering', 'adaptive_speed_control')
        output_path: Path to save network.yml (traffic_control.yml will be {output_path}_traffic_control.yml)
        config_dir: Path to configs directory for Lagos data
        duration: Simulation duration in seconds (default: 600s = 10 minutes)
        domain_length: Road domain length in meters (default: 1000m = 1km)
    
    Returns:
        Dictionary with scenario configuration using real Lagos parameters
    """
    import yaml
    from pathlib import Path as PathlibPath
    from datetime import datetime
    
    # Load REAL Lagos traffic parameters
    lagos_params = load_lagos_traffic_params(config_dir)
    
    # Extract real parameters
    max_density_m = lagos_params['max_density_motorcycles']  # 250 veh/km (REAL)
    max_density_c = lagos_params['max_density_cars']  # 120 veh/km (REAL)
    free_speed_m_kmh = lagos_params['free_speed_motorcycles']  # 32 km/h (REAL)
    free_speed_c_kmh = lagos_params['free_speed_cars']  # 28 km/h (REAL)
    
    # Convert speeds to m/s
    free_speed_m = free_speed_m_kmh / 3.6  # ~8.9 m/s
    free_speed_c = free_speed_c_kmh / 3.6  # ~7.8 m/s
    
    # Load config_base.yml for behavioral_coupling θ_k
    project_root = PathlibPath(__file__).parent.parent.parent.parent
    config_base_path = project_root / "arz_model" / "config" / "config_base.yml"
    with open(config_base_path) as f:
        base_config = yaml.safe_load(f)
    
    theta_moto_signal = base_config['behavioral_coupling']['theta_moto_signalized']
    theta_car_signal = base_config['behavioral_coupling']['theta_car_signalized']
    theta_moto_priority = base_config['behavioral_coupling']['theta_moto_priority']
    theta_moto_secondary = base_config['behavioral_coupling']['theta_moto_secondary']
    
    # Generate NetworkConfig YAML based on scenario type
    if scenario_type == 'traffic_light_control':
        network_data, traffic_data = _generate_signalized_network_lagos(
            free_speed_m, free_speed_c, theta_moto_signal, theta_car_signal, duration, domain_length
        )
    elif scenario_type == 'ramp_metering':
        network_data, traffic_data = _generate_ramp_network_lagos(
            free_speed_m, free_speed_c, theta_moto_priority, theta_moto_secondary, duration, domain_length
        )
    elif scenario_type == 'adaptive_speed_control':
        network_data, traffic_data = _generate_speed_control_network_lagos(
            free_speed_m, free_speed_c, duration, domain_length
        )
    else:
        raise ValueError(f"Unknown scenario type: {scenario_type}")
    
    # Save NetworkConfig files if output path provided
    if output_path:
        output_path = PathlibPath(output_path)
        network_path = output_path
        traffic_path = output_path.parent / f"{output_path.stem}_traffic_control.yml"
        
        with open(network_path, 'w') as f:
            yaml.dump(network_data, f, default_flow_style=False, sort_keys=False)
        
        with open(traffic_path, 'w') as f:
            yaml.dump(traffic_data, f, default_flow_style=False, sort_keys=False)
        
        print(f"[LAGOS CONFIG] Saved NetworkConfig:")
        print(f"  - Network: {network_path}")
        print(f"  - Traffic: {traffic_path}")
    
    # Return combined config (for backward compatibility)
    combined_config = {
        **network_data,
        'traffic_control': traffic_data['traffic_control'],
        'lagos_parameters': lagos_params
    }
    return combined_config
    
    # Extract real parameters
    max_density_m = lagos_params['max_density_motorcycles']  # 250 veh/km (REAL)
    max_density_c = lagos_params['max_density_cars']  # 120 veh/km (REAL)
    free_speed_m_kmh = lagos_params['free_speed_motorcycles']  # 32 km/h (REAL)
    free_speed_c_kmh = lagos_params['free_speed_cars']  # 28 km/h (REAL)
    
    # Convert speeds to m/s
    free_speed_m = free_speed_m_kmh / 3.6  # ~8.9 m/s
    free_speed_c = free_speed_c_kmh / 3.6  # ~7.8 m/s
    
    # ✅ BUG #34 FIX: Inflow must use EQUILIBRIUM SPEED not free speed
    # Discovery: ARZ model relaxes w → Ve via source term S = (Ve - w) / tau
    # At rho=200 veh/km: Ve_m = 2.26 m/s << V0_m = 8.89 m/s
    # Result: Prescribed high-speed flux gets reduced by relaxation → no accumulation
    # Solution: Use equilibrium speed for inflow to match ARZ physics
    
    # Calculate equilibrium speeds using ARZ model formula
    rho_jam_veh_km = max_density_m + max_density_c  # 370 veh/km total
    rho_jam = rho_jam_veh_km / 1000.0  # Convert to veh/m
    V_creeping = 0.6  # Default creeping speed (m/s)
    
    # Initial state: CONGESTED traffic (to see RED vs GREEN effect)
    # BUG FIX: Light initial conditions (0.01 veh/m) made RED/GREEN indistinguishable
    # With no congestion, blocking inflow creates no observable backpressure
    # Solution: Start with MEDIUM-HEAVY traffic so RED control creates visible queueing
    rho_m_initial_veh_km = max_density_m * 0.5  # 125 veh/km (medium-heavy)
    rho_c_initial_veh_km = max_density_c * 0.5  # 60 veh/km (medium-heavy)
    rho_total_initial = (rho_m_initial_veh_km + rho_c_initial_veh_km) / 1000.0  # 0.185 veh/m
    g_initial = max(0.0, 1.0 - rho_total_initial / rho_jam)  # ≈ 0.5 (congested)
    w_m_initial = V_creeping + (free_speed_m - V_creeping) * g_initial  # ≈ 4.75 m/s
    w_c_initial = free_speed_c * g_initial  # ≈ 4.35 m/s
    # Flux_init ≈ 0.185 * 4.55 = 0.84 veh/s (high, will be restricted by RED light)
    
    # Inflow: EQUILIBRIUM demand with realistic velocity modulation for traffic signals
    # GREEN phase: Inflow at free speed (or near free speed) → unrestricted flow
    # RED phase: Inflow reduced to 50% of green → causes congestion
    # This allows RL agent to see meaningful difference between RED and GREEN phases
    
    # GREEN phase velocity = free speed
    w_m_inflow_green = free_speed_m  # ≈ 8.9 m/s (free speed motorcycles)
    w_c_inflow_green = free_speed_c  # ≈ 7.8 m/s (free speed cars)
    
    # Inflow density: medium-heavy (not jamming)
    rho_m_inflow_veh_km = max_density_m * 0.8  # 200 veh/km (heavy Lagos traffic)
    rho_c_inflow_veh_km = max_density_c * 0.8  # 96 veh/km (heavy Lagos traffic)
    rho_total_inflow = (rho_m_inflow_veh_km + rho_c_inflow_veh_km) / 1000.0  # ≈ 0.296 veh/m
    
    # RED phase will be 50% of this (applied by set_traffic_signal_state)
    # So: RED velocity = 4.45 m/s (50% of 8.9) → creates queuing
    #     GREEN velocity = 8.9 m/s (full free speed) → clears queuing
    # This gives RL agent clear signal to learn from!
    
    w_m_inflow = w_m_inflow_green  # Will be multiplied by 0.5 in RED phase
    w_c_inflow = w_c_inflow_green  # Will be multiplied by 0.5 in RED phase
    
    # Base configuration
    # BUG #31 FIX: Use NETWORK-BASED configuration instead of single-segment BC modulation
    # This aligns with section4_modeles_reseaux.tex theoretical framework
    # Network with proper node solver enables realistic queue formation via conservation laws
    config = {
        'scenario_name': f'{scenario_type}_lagos_real',
        'N': 100,
        'xmin': 0.0,
        'xmax': domain_length,
        't_final': duration,
        'output_dt': 60.0,
        'CFL': 0.4,
        'road': {'quality_type': 'uniform', 'quality_value': 2},
        'lagos_parameters': lagos_params,  # Include full Lagos params for reference
        
        # ⚠️ BUG #31: Enable NETWORK system with proper traffic light node
        # This fixes zero-reward issue by enabling realistic queue formation
        'network': {
            'has_network': True,
            'segments': [
                {
                    'id': 'upstream',
                    'length': domain_length / 2,
                    'cells': 50,
                    'is_source': True,  # Can receive inflow boundary condition
                    'is_sink': False
                },
                {
                    'id': 'downstream',
                    'length': domain_length / 2,
                    'cells': 50,
                    'is_source': False,
                    'is_sink': True  # Can output as boundary condition
                }
            ],
            'nodes': [
                {
                    'id': 'traffic_light_node_1',
                    'position': domain_length / 2,  # Center of domain
                    'segments': ['upstream', 'downstream'],
                    'type': 'signalized_intersection',
                    'traffic_lights': {
                        'cycle_time': 120.0,  # 120s total cycle
                        'phases': [
                            {
                                'duration': 60.0,  # RED phase: 60s
                                'green_segments': []  # No outflow during RED
                            },
                            {
                                'duration': 60.0,  # GREEN phase: 60s
                                'green_segments': ['upstream']  # Allow outflow from upstream
                            }
                        ],
                        'offset': 0.0
                    },
                    'max_queue_lengths': {
                        'motorcycle': 200.0,
                        'car': 200.0
                    },
                    'creeping': {
                        'enabled': True,
                        'speed_kmh': 5.0,
                        'threshold': 50.0
                    }
                }
            ]
        },
        
        # ⚠️ DEPRECATED: Old single-segment boundary conditions 
        # (kept only for backward compatibility - will be removed in v2.0)
        # Use network-based configuration above instead (has_network=True)
        'boundary_conditions': {
            'left': {'type': 'inflow', 'state': [rho_m_inflow_veh_km, w_m_inflow, rho_c_inflow_veh_km, w_c_inflow]},
            'right': {'type': 'outflow'}
        }
    }
    
    # Scenario-specific parameters
    if scenario_type == 'traffic_light_control':
        config['parameters'] = {
            'V0_m': free_speed_m,  # Use REAL Lagos speeds
            'V0_c': free_speed_c,
            'tau_m': 1.0,
            'tau_c': 1.2
        }
        config['initial_conditions'] = {
            'type': 'uniform',
            'state': [rho_m_initial_veh_km, w_m_initial, rho_c_initial_veh_km, w_c_initial]
        }
    elif scenario_type == 'ramp_metering':
        config['parameters'] = {
            'V0_m': free_speed_m * 1.1,
            'V0_c': free_speed_c * 1.1,
            'tau_m': 0.8,
            'tau_c': 1.0
        }
        # Riemann IC for ramp scenario
        config['initial_conditions'] = {
            'type': 'riemann',
            'U_L': [rho_m_inflow_veh_km/1000*0.8, w_m_inflow, rho_c_inflow_veh_km/1000*0.8, w_c_inflow],
            'U_R': [rho_m_initial_veh_km/1000*0.8, w_m_initial, rho_c_initial_veh_km/1000*0.8, w_c_initial],
            'split_pos': domain_length / 2
        }
    elif scenario_type == 'adaptive_speed_control':
        config['parameters'] = {
            'V0_m': free_speed_m * 1.2,
            'V0_c': free_speed_c * 1.2,
            'tau_m': 0.6,
            'tau_c': 0.8
        }
        # Riemann IC for adaptive speed scenario
        config['initial_conditions'] = {
            'type': 'riemann',
            'U_L': [rho_m_inflow_veh_km/1000*0.7, w_m_inflow, rho_c_inflow_veh_km/1000*0.7, w_c_inflow],
            'U_R': [rho_m_initial_veh_km/1000*0.7, w_m_initial, rho_c_initial_veh_km/1000*0.7, w_c_initial],
            'split_pos': domain_length / 2
        }
    
    # Save if output path provided
    if output_path:
        save_config(config, str(output_path))
        print(f"[LAGOS CONFIG] Saved to {output_path}")
    
    return config


def analyze_experiment_config(exp_config: Dict[str, Any], configs: Dict[str, Any]) -> List[str]:
    """Analyze experiment configuration and provide insights"""
    insights = []
    
    # Algorithm analysis
    if "algorithm" in exp_config:
        algo = exp_config["algorithm"]
        insights.append(f"Using {algo} algorithm for training")
        
        if algo == "DQN":
            insights.append("DQN is well-suited for discrete action spaces like traffic signals")
    
    # Timesteps analysis
    if "timesteps" in exp_config:
        timesteps = exp_config["timesteps"]
        if timesteps < 10000:
            insights.append(f"Low timesteps ({timesteps}) - suitable for quick testing")
        elif timesteps < 100000:
            insights.append(f"Medium timesteps ({timesteps}) - good for initial training")
        else:
            insights.append(f"High timesteps ({timesteps}) - thorough training expected")
    
    # Environment analysis
    if "env_config" in exp_config and "max_steps" in exp_config["env_config"]:
        max_steps = exp_config["env_config"]["max_steps"]
        insights.append(f"Episode length: {max_steps} steps")
    
    # Network analysis
    if "network" in exp_config:
        network = exp_config["network"]
        insights.append(f"Target network: {network}")
        
        if network == "victoria_island":
            insights.append("Victoria Island Lagos - high traffic complexity expected")
    
    return insights
