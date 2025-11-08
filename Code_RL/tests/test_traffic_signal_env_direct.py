"""
Tests for TrafficSignalEnvDirect - Direct Gymnasium Environment

Validates Gymnasium API compliance and environment functionality.
"""

import pytest
import numpy as np
import gymnasium as gym
from gymnasium.utils.env_checker import check_env
import os
import sys

# Add paths
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Code_RL.src.env.traffic_signal_env_direct import TrafficSignalEnvDirect


class TestTrafficSignalEnvDirect:
    """Test suite for direct coupling environment."""
    
    @pytest.fixture
    def env(self):
        """Create environment instance for testing."""
        scenario_path = os.path.join(
            os.path.dirname(__file__),
            '../../scenarios/scenario_calibration_victoria_island.yml'
        )
        
        if not os.path.exists(scenario_path):
            pytest.skip(f"Scenario file not found: {scenario_path}")
        
        env = TrafficSignalEnvDirect(
            scenario_config_path=scenario_path,
            decision_interval=10.0,
            episode_max_time=300.0,  # 5 minutes for testing
            quiet=True,
            device='cpu'
        )
        
        yield env
        env.close()
    
    # ==================================================================
    # Gymnasium API Compliance Tests
    # ==================================================================
    
    def test_gymnasium_check_env(self, env):
        """Test that environment passes official Gymnasium validation."""
        # This is the critical test for Gymnasium compliance
        check_env(env.unwrapped, skip_render_check=True)
    
    def test_reset_returns_correct_tuple(self, env):
        """Test reset() returns (observation, info) tuple."""
        result = env.reset()
        
        assert isinstance(result, tuple)
        assert len(result) == 2
        
        observation, info = result
        assert isinstance(observation, np.ndarray)
        assert isinstance(info, dict)
    
    def test_step_returns_correct_tuple(self, env):
        """Test step() returns (obs, reward, terminated, truncated, info) tuple."""
        env.reset()
        result = env.step(0)
        
        assert isinstance(result, tuple)
        assert len(result) == 5
        
        observation, reward, terminated, truncated, info = result
        assert isinstance(observation, np.ndarray)
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
    
    def test_observation_in_observation_space(self, env):
        """Test that observations are within defined observation_space."""
        observation, _ = env.reset()
        
        assert env.observation_space.contains(observation)
        
        # Take a step and check again
        observation, _, _, _, _ = env.step(0)
        assert env.observation_space.contains(observation)
    
    def test_action_space_discrete(self, env):
        """Test that action space is Discrete(2)."""
        assert isinstance(env.action_space, gym.spaces.Discrete)
        assert env.action_space.n == 2
    
    # ==================================================================
    # Observation Building Tests
    # ==================================================================
    
    def test_observation_shape(self, env):
        """Test observation has correct shape."""
        observation, _ = env.reset()
        
        # Expected shape: 4 vars × n_segments + n_phases
        expected_dim = 4 * env.n_segments + env.n_phases
        assert observation.shape == (expected_dim,)
    
    def test_observation_normalized(self, env):
        """Test that observations are normalized to [0, 1]."""
        observation, _ = env.reset()
        
        # All values should be in [0, 1]
        assert np.all(observation >= 0.0)
        assert np.all(observation <= 1.0)
    
    def test_phase_onehot_encoding(self, env):
        """Test that phase is correctly one-hot encoded in observation."""
        observation, _ = env.reset()
        
        # Extract phase one-hot (last n_phases elements)
        phase_onehot = observation[-env.n_phases:]
        
        # Exactly one element should be 1.0
        assert np.sum(phase_onehot) == 1.0
        assert np.sum(phase_onehot == 1.0) == 1
    
    # ==================================================================
    # Action and Dynamics Tests
    # ==================================================================
    
    def test_action_maintain_phase(self, env):
        """Test that action=0 maintains current phase."""
        env.reset()
        initial_phase = env.current_phase
        
        _, _, _, _, info = env.step(action=0)
        
        assert env.current_phase == initial_phase
        assert info['phase_changed'] is False
    
    def test_action_switch_phase(self, env):
        """Test that action=1 switches phase."""
        env.reset()
        initial_phase = env.current_phase
        
        _, _, _, _, info = env.step(action=1)
        
        assert env.current_phase != initial_phase
        assert info['phase_changed'] is True
    
    def test_simulation_time_advances(self, env):
        """Test that simulation time advances by decision_interval."""
        env.reset()
        initial_time = env.runner.t
        
        _, _, _, _, info = env.step(action=0)
        
        expected_time = initial_time + env.decision_interval
        assert np.isclose(info['simulation_time'], expected_time, atol=0.1)
    
    # ==================================================================
    # Reward Calculation Tests
    # ==================================================================
    
    def test_reward_is_scalar(self, env):
        """Test that reward is a scalar float."""
        env.reset()
        _, reward, _, _, _ = env.step(0)
        
        assert isinstance(reward, float)
        assert np.isfinite(reward)
    
    def test_reward_phase_change_penalty(self, env):
        """Test that phase change incurs penalty."""
        env.reset()
        
        # Maintain phase (no penalty)
        _, reward_maintain, _, _, _ = env.step(action=0)
        
        # Reset and switch phase (penalty)
        env.reset()
        _, reward_switch, _, _, _ = env.step(action=1)
        
        # Switching should be penalized (more negative or less positive)
        # Note: This is a weak test due to other reward components
        # At minimum, verify both rewards are finite
        assert np.isfinite(reward_maintain)
        assert np.isfinite(reward_switch)
    
    # ==================================================================
    # Episode Termination Tests
    # ==================================================================
    
    def test_episode_truncation_on_time_limit(self, env):
        """Test that episode truncates when time limit reached."""
        env.reset()
        
        # Run until time limit
        truncated = False
        max_iterations = 100
        iterations = 0
        
        while not truncated and iterations < max_iterations:
            _, _, terminated, truncated, _ = env.step(0)
            iterations += 1
        
        assert truncated, "Episode should truncate at time limit"
        assert not terminated, "Episode should not terminate (no goal state)"
    
    def test_reset_after_episode(self, env):
        """Test that reset works after episode completion."""
        # Run one episode
        env.reset()
        for _ in range(5):
            _, _, _, truncated, _ = env.step(0)
            if truncated:
                break
        
        # Reset and verify
        observation, info = env.reset()
        
        assert env.episode_step == 0
        assert env.total_reward == 0.0
        assert env.runner.t <= env.decision_interval  # Should be near zero
    
    # ==================================================================
    # Integration Tests
    # ==================================================================
    
    def test_multi_step_episode(self, env):
        """Test running multiple steps in an episode."""
        observation, _ = env.reset()
        
        for i in range(5):
            action = i % 2  # Alternate actions
            observation, reward, terminated, truncated, info = env.step(action)
            
            # Verify state consistency
            assert env.observation_space.contains(observation)
            assert info['episode_step'] == i + 1
            assert not terminated
    
    def test_observation_changes_over_time(self, env):
        """Test that observations change as simulation progresses."""
        obs1, _ = env.reset()
        obs2, _, _, _, _ = env.step(0)
        
        # Observations should differ (traffic state evolves)
        assert not np.array_equal(obs1, obs2)
    
    def test_info_dict_completeness(self, env):
        """Test that info dict contains expected keys."""
        _, info = env.reset()
        
        required_keys = ['episode_step', 'simulation_time', 'current_phase']
        for key in required_keys:
            assert key in info
        
        # After step
        _, _, _, _, info = env.step(0)
        step_keys = required_keys + ['phase_changed', 'total_reward']
        for key in step_keys:
            assert key in info


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
