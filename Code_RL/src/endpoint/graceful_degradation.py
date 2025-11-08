"""
Graceful Degradation System for RL-Simulator Communication

This module provides graceful degradation mechanisms to handle simulator unavailability
scenarios, ensuring RL training can continue with reduced functionality.
"""

import asyncio
import time
import logging
from typing import Dict, Any, Optional, Union, List, Callable
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import json
import pickle
import numpy as np

logger = logging.getLogger(__name__)


class DegradationLevel(Enum):
    """Levels of service degradation"""
    NORMAL = "normal"              # Full functionality
    REDUCED = "reduced"            # Some features disabled
    CACHED = "cached"              # Using cached/historical data  
    SIMULATED = "simulated"        # Using simplified simulation
    OFFLINE = "offline"            # No external communication


@dataclass
class DegradationConfig:
    """Configuration for graceful degradation"""
    
    # Degradation triggers
    max_failures_before_degradation: int = 3
    failure_window_seconds: float = 60.0
    
    # Degradation timeouts
    cached_mode_timeout_seconds: float = 300.0      # 5 minutes
    simulated_mode_timeout_seconds: float = 600.0   # 10 minutes
    offline_mode_timeout_seconds: float = 1800.0    # 30 minutes
    
    # Cache settings
    enable_state_caching: bool = True
    max_cached_states: int = 1000
    cache_ttl_seconds: float = 3600.0               # 1 hour
    
    # Simplified simulation settings
    enable_simplified_simulation: bool = True
    simplified_state_noise: float = 0.1
    simplified_reward_scaling: float = 0.8
    
    # Recovery settings
    enable_auto_recovery: bool = True
    recovery_check_interval_seconds: float = 30.0
    min_successful_checks_for_recovery: int = 3


@dataclass
class DegradationState:
    """Current degradation state"""
    
    level: DegradationLevel = DegradationLevel.NORMAL
    start_time: float = 0.0
    failure_count: int = 0
    last_failure_time: float = 0.0
    reason: Optional[str] = None
    auto_recovery_enabled: bool = True


class StateCache:
    """Cache for traffic states to enable cached operation"""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: float = 3600.0):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.access_times: Dict[str, float] = {}
        self.insertion_order: List[str] = []
    
    def put(self, key: str, state: Dict[str, Any]):
        """Store state in cache"""
        current_time = time.time()
        
        # Remove if already exists
        if key in self.cache:
            self.insertion_order.remove(key)
        
        # Store state
        self.cache[key] = state.copy()
        self.access_times[key] = current_time
        self.insertion_order.append(key)
        
        # Cleanup if over limit
        while len(self.cache) > self.max_size:
            oldest_key = self.insertion_order.pop(0)
            del self.cache[oldest_key]
            del self.access_times[oldest_key]
        
        # Cleanup expired entries
        self._cleanup_expired()
    
    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Get state from cache"""
        if key not in self.cache:
            return None
        
        # Check TTL
        if time.time() - self.access_times[key] > self.ttl_seconds:
            self._remove(key)
            return None
        
        # Update access time
        self.access_times[key] = time.time()
        return self.cache[key].copy()
    
    def get_recent_states(self, count: int = 10) -> List[Dict[str, Any]]:
        """Get most recent states"""
        recent_keys = self.insertion_order[-count:]
        states = []
        
        for key in recent_keys:
            state = self.get(key)
            if state:
                states.append(state)
        
        return states
    
    def get_similar_state(self, target_state: Dict[str, Any], similarity_threshold: float = 0.9) -> Optional[Dict[str, Any]]:
        """Get cached state similar to target state"""
        # This is a simplified similarity check
        # In practice, you might want more sophisticated matching
        
        if not target_state or not self.cache:
            return None
        
        # Try to find states with similar structure
        for state in self.cache.values():
            if self._states_similar(target_state, state, similarity_threshold):
                return state.copy()
        
        return None
    
    def _states_similar(self, state1: Dict[str, Any], state2: Dict[str, Any], threshold: float) -> bool:
        """Check if two states are similar"""
        # Simple similarity check based on keys
        keys1 = set(state1.keys())
        keys2 = set(state2.keys())
        
        if not keys1 or not keys2:
            return False
        
        intersection = keys1.intersection(keys2)
        union = keys1.union(keys2)
        
        similarity = len(intersection) / len(union)
        return similarity >= threshold
    
    def _remove(self, key: str):
        """Remove key from cache"""
        if key in self.cache:
            del self.cache[key]
            del self.access_times[key]
            self.insertion_order.remove(key)
    
    def _cleanup_expired(self):
        """Remove expired entries"""
        current_time = time.time()
        expired_keys = [
            key for key, access_time in self.access_times.items()
            if current_time - access_time > self.ttl_seconds
        ]
        
        for key in expired_keys:
            self._remove(key)
    
    def clear(self):
        """Clear all cached data"""
        self.cache.clear()
        self.access_times.clear()
        self.insertion_order.clear()
    
    def size(self) -> int:
        """Get current cache size"""
        return len(self.cache)


class SimplifiedSimulator:
    """Simplified simulator for degraded operation"""
    
    def __init__(self, config: DegradationConfig):
        self.config = config
        self.current_state: Optional[Dict[str, Any]] = None
        self.step_count = 0
        self.random_state = np.random.RandomState(42)  # Deterministic for reproducibility
    
    def initialize(self, initial_state: Optional[Dict[str, Any]] = None):
        """Initialize simplified simulator"""
        if initial_state:
            self.current_state = initial_state.copy()
        else:
            self.current_state = self._create_default_state()
        
        self.step_count = 0
        logger.info("Simplified simulator initialized")
    
    def step(self, action: Dict[str, Any], dt: float = 1.0) -> Dict[str, Any]:
        """Simulate one step"""
        if not self.current_state:
            self.initialize()
        
        # Apply simplified dynamics
        next_state = self._apply_simple_dynamics(self.current_state, action, dt)
        
        # Add noise for realism
        next_state = self._add_noise(next_state)
        
        self.current_state = next_state
        self.step_count += 1
        
        return {
            "state": next_state,
            "reward": self._calculate_simplified_reward(next_state, action),
            "done": False,
            "info": {"simplified": True, "step": self.step_count}
        }
    
    def _create_default_state(self) -> Dict[str, Any]:
        """Create default state for initialization"""
        return {
            "timestamp": time.time(),
            "branches": {
                f"branch_{i}": {
                    "rho_m": 0.3 + self.random_state.random() * 0.4,
                    "rho_c": 0.2 + self.random_state.random() * 0.3,
                    "v_m": 30.0 + self.random_state.random() * 20.0,
                    "v_c": 25.0 + self.random_state.random() * 15.0,
                    "queue_len": self.random_state.randint(0, 10),
                    "flow": 100.0 + self.random_state.random() * 200.0
                }
                for i in range(4)  # 4-way intersection
            },
            "signals": {
                "main": {"phase": 1, "remaining_time": 30.0},
                "cross": {"phase": 2, "remaining_time": 60.0}
            }
        }
    
    def _apply_simple_dynamics(self, state: Dict[str, Any], action: Dict[str, Any], dt: float) -> Dict[str, Any]:
        """Apply simplified traffic dynamics"""
        next_state = state.copy()
        
        # Update signal timings
        for signal_id, signal_data in next_state.get("signals", {}).items():
            signal_data["remaining_time"] = max(0, signal_data["remaining_time"] - dt)
            
            # Handle signal switching from action
            if action.get("signal_action") == 1 and signal_data["remaining_time"] <= 5.0:
                signal_data["phase"] = 3 - signal_data["phase"]  # Toggle between 1 and 2
                signal_data["remaining_time"] = 30.0
        
        # Update traffic densities (simplified)
        for branch_id, branch_data in next_state.get("branches", {}).items():
            # Simplified traffic flow
            current_phase = next_state["signals"]["main"]["phase"]
            is_green = (current_phase == 1 and branch_id in ["branch_0", "branch_2"]) or \
                      (current_phase == 2 and branch_id in ["branch_1", "branch_3"])
            
            if is_green:
                # Vehicles can pass, reduce density
                branch_data["rho_m"] = max(0.1, branch_data["rho_m"] - 0.05 * dt)
                branch_data["queue_len"] = max(0, branch_data["queue_len"] - int(2 * dt))
                branch_data["flow"] += 10 * dt
            else:
                # Red light, increase density
                branch_data["rho_m"] = min(0.9, branch_data["rho_m"] + 0.03 * dt)
                branch_data["queue_len"] += int(1 * dt)
                branch_data["flow"] = max(0, branch_data["flow"] - 5 * dt)
            
            # Update velocities based on density
            max_velocity = 50.0
            branch_data["v_m"] = max_velocity * (1 - branch_data["rho_m"])
            branch_data["v_c"] = max_velocity * (1 - branch_data["rho_c"]) * 0.8
        
        next_state["timestamp"] = time.time()
        return next_state
    
    def _add_noise(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Add noise to state for realism"""
        noisy_state = state.copy()
        noise_level = self.config.simplified_state_noise
        
        for branch_id, branch_data in noisy_state.get("branches", {}).items():
            for key in ["rho_m", "rho_c", "v_m", "v_c", "flow"]:
                if key in branch_data:
                    noise = self.random_state.normal(0, noise_level)
                    branch_data[key] = max(0, branch_data[key] * (1 + noise))
        
        return noisy_state
    
    def _calculate_simplified_reward(self, state: Dict[str, Any], action: Dict[str, Any]) -> float:
        """Calculate simplified reward"""
        total_wait_time = 0.0
        total_throughput = 0.0
        
        for branch_data in state.get("branches", {}).values():
            # Penalty for queues and low velocity
            total_wait_time += branch_data.get("queue_len", 0) * 2.0
            if branch_data.get("v_m", 0) < 10.0:
                total_wait_time += 5.0
            
            # Reward for throughput
            total_throughput += branch_data.get("flow", 0)
        
        # Switch penalty
        switch_penalty = 1.0 if action.get("signal_action") == 1 else 0.0
        
        reward = (total_throughput * 0.01 - total_wait_time * 0.1 - switch_penalty) * self.config.simplified_reward_scaling
        
        return np.clip(reward, -10.0, 10.0)


class GracefulDegradationManager:
    """Manager for graceful degradation of RL-simulator communication"""
    
    def __init__(self, config: DegradationConfig = None):
        self.config = config or DegradationConfig()
        self.state = DegradationState()
        self.cache = StateCache(self.config.max_cached_states, self.config.cache_ttl_seconds)
        self.simplified_simulator = SimplifiedSimulator(self.config)
        
        # Callbacks for different modes
        self._mode_callbacks: Dict[DegradationLevel, List[Callable]] = {
            level: [] for level in DegradationLevel
        }
        
        # Recovery monitoring
        self._recovery_task: Optional[asyncio.Task] = None
        self._recovery_check_callback: Optional[Callable] = None
        
        # Statistics
        self.stats = {
            "normal_operations": 0,
            "degraded_operations": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "simulated_steps": 0,
            "degradation_events": 0
        }
    
    def add_mode_callback(self, level: DegradationLevel, callback: Callable):
        """Add callback for degradation level changes"""
        self._mode_callbacks[level].append(callback)
    
    def set_recovery_check_callback(self, callback: Callable):
        """Set callback for checking if service has recovered"""
        self._recovery_check_callback = callback
    
    async def record_failure(self, error: Exception, context: str = ""):
        """Record a failure and potentially trigger degradation"""
        current_time = time.time()
        
        # Check if this failure is within the failure window
        if (current_time - self.state.last_failure_time) <= self.config.failure_window_seconds:
            self.state.failure_count += 1
        else:
            self.state.failure_count = 1
        
        self.state.last_failure_time = current_time
        
        logger.warning(f"Failure recorded: {error} (context: {context}). Count: {self.state.failure_count}")
        
        # Check if degradation is needed
        if (self.state.failure_count >= self.config.max_failures_before_degradation and 
            self.state.level == DegradationLevel.NORMAL):
            await self._trigger_degradation(f"Multiple failures: {error}")
    
    async def record_success(self):
        """Record a successful operation"""
        if self.state.level == DegradationLevel.NORMAL:
            self.stats["normal_operations"] += 1
        else:
            self.stats["degraded_operations"] += 1
    
    async def _trigger_degradation(self, reason: str):
        """Trigger degradation to next level"""
        old_level = self.state.level
        
        # Determine next degradation level
        if self.state.level == DegradationLevel.NORMAL:
            if self.config.enable_state_caching and self.cache.size() > 0:
                new_level = DegradationLevel.CACHED
            elif self.config.enable_simplified_simulation:
                new_level = DegradationLevel.SIMULATED
            else:
                new_level = DegradationLevel.OFFLINE
        elif self.state.level == DegradationLevel.CACHED:
            if self.config.enable_simplified_simulation:
                new_level = DegradationLevel.SIMULATED
            else:
                new_level = DegradationLevel.OFFLINE
        elif self.state.level == DegradationLevel.SIMULATED:
            new_level = DegradationLevel.OFFLINE
        else:
            new_level = DegradationLevel.OFFLINE
        
        await self._set_degradation_level(new_level, reason)
        
        logger.warning(f"Degradation triggered: {old_level.value} -> {new_level.value}. Reason: {reason}")
        self.stats["degradation_events"] += 1
    
    async def _set_degradation_level(self, level: DegradationLevel, reason: str):
        """Set degradation level and notify callbacks"""
        old_level = self.state.level
        self.state.level = level
        self.state.start_time = time.time()
        self.state.reason = reason
        
        # Execute callbacks for new level
        for callback in self._mode_callbacks[level]:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(old_level, level, reason)
                else:
                    callback(old_level, level, reason)
            except Exception as e:
                logger.error(f"Error in degradation callback: {e}")
        
        # Start recovery monitoring if enabled
        if (self.config.enable_auto_recovery and 
            level != DegradationLevel.NORMAL and 
            (self._recovery_task is None or self._recovery_task.done())):
            self._recovery_task = asyncio.create_task(self._recovery_monitoring())
    
    async def _recovery_monitoring(self):
        """Monitor for service recovery"""
        consecutive_successes = 0
        
        while self.state.level != DegradationLevel.NORMAL:
            try:
                await asyncio.sleep(self.config.recovery_check_interval_seconds)
                
                # Check if we've been degraded too long
                degraded_time = time.time() - self.state.start_time
                max_time = self._get_max_degradation_time()
                
                if degraded_time > max_time:
                    logger.warning(f"Degradation timeout reached for {self.state.level.value}")
                    await self._trigger_degradation("Timeout")
                    consecutive_successes = 0
                    continue
                
                # Try recovery check
                if self._recovery_check_callback:
                    try:
                        recovery_result = await self._recovery_check_callback()
                        if recovery_result:
                            consecutive_successes += 1
                            logger.info(f"Recovery check successful: {consecutive_successes}/{self.config.min_successful_checks_for_recovery}")
                        else:
                            consecutive_successes = 0
                    except Exception:
                        consecutive_successes = 0
                
                # Check if enough successes for recovery
                if consecutive_successes >= self.config.min_successful_checks_for_recovery:
                    await self._recover_to_normal()
                    break
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in recovery monitoring: {e}")
                await asyncio.sleep(5.0)
    
    def _get_max_degradation_time(self) -> float:
        """Get maximum time for current degradation level"""
        if self.state.level == DegradationLevel.CACHED:
            return self.config.cached_mode_timeout_seconds
        elif self.state.level == DegradationLevel.SIMULATED:
            return self.config.simulated_mode_timeout_seconds
        elif self.state.level == DegradationLevel.OFFLINE:
            return self.config.offline_mode_timeout_seconds
        else:
            return float('inf')
    
    async def _recover_to_normal(self):
        """Recover to normal operation"""
        logger.info(f"Recovering from {self.state.level.value} to normal operation")
        await self._set_degradation_level(DegradationLevel.NORMAL, "Recovered")
        
        # Reset failure count
        self.state.failure_count = 0
    
    async def force_degradation_level(self, level: DegradationLevel, reason: str = "Manual"):
        """Manually force degradation level"""
        await self._set_degradation_level(level, reason)
    
    async def force_recovery(self):
        """Manually force recovery to normal"""
        await self._recover_to_normal()
    
    def cache_state(self, key: str, state: Dict[str, Any]):
        """Cache a state for later use"""
        if self.config.enable_state_caching:
            self.cache.put(key, state)
    
    def get_cached_state(self, key: str) -> Optional[Dict[str, Any]]:
        """Get cached state"""
        if self.config.enable_state_caching:
            state = self.cache.get(key)
            if state:
                self.stats["cache_hits"] += 1
            else:
                self.stats["cache_misses"] += 1
            return state
        return None
    
    def get_recent_cached_states(self, count: int = 10) -> List[Dict[str, Any]]:
        """Get recent cached states"""
        return self.cache.get_recent_states(count)
    
    async def execute_degraded_operation(self, operation_type: str, **kwargs) -> Dict[str, Any]:
        """Execute operation in degraded mode"""
        if self.state.level == DegradationLevel.NORMAL:
            raise ValueError("Not in degraded mode")
        
        if self.state.level == DegradationLevel.CACHED:
            return await self._execute_cached_operation(operation_type, **kwargs)
        elif self.state.level == DegradationLevel.SIMULATED:
            return await self._execute_simulated_operation(operation_type, **kwargs)
        elif self.state.level == DegradationLevel.OFFLINE:
            return await self._execute_offline_operation(operation_type, **kwargs)
        else:
            raise ValueError(f"Unknown degradation level: {self.state.level}")
    
    async def _execute_cached_operation(self, operation_type: str, **kwargs) -> Dict[str, Any]:
        """Execute operation using cached data"""
        if operation_type == "reset":
            # Return most recent cached state
            recent_states = self.cache.get_recent_states(1)
            if recent_states:
                return recent_states[0]
            else:
                # Fall back to simulated if no cache
                return await self._execute_simulated_operation(operation_type, **kwargs)
        
        elif operation_type == "step":
            # Try to find similar cached state
            current_state = kwargs.get("current_state")
            if current_state:
                similar_state = self.cache.get_similar_state(current_state)
                if similar_state:
                    return similar_state
            
            # Fall back to simulated
            return await self._execute_simulated_operation(operation_type, **kwargs)
        
        else:
            return {"error": f"Unsupported operation in cached mode: {operation_type}"}
    
    async def _execute_simulated_operation(self, operation_type: str, **kwargs) -> Dict[str, Any]:
        """Execute operation using simplified simulation"""
        if operation_type == "reset":
            initial_state = kwargs.get("initial_state")
            self.simplified_simulator.initialize(initial_state)
            self.stats["simulated_steps"] += 1
            return self.simplified_simulator.current_state
        
        elif operation_type == "step":
            action = kwargs.get("action", {})
            dt = kwargs.get("dt", 1.0)
            result = self.simplified_simulator.step(action, dt)
            self.stats["simulated_steps"] += 1
            return result
        
        else:
            return {"error": f"Unsupported operation in simulated mode: {operation_type}"}
    
    async def _execute_offline_operation(self, operation_type: str, **kwargs) -> Dict[str, Any]:
        """Execute operation in offline mode"""
        return {
            "error": "Service unavailable",
            "mode": "offline",
            "operation": operation_type,
            "degraded": True
        }
    
    def get_degradation_status(self) -> Dict[str, Any]:
        """Get current degradation status"""
        degraded_time = time.time() - self.state.start_time if self.state.start_time > 0 else 0
        
        return {
            "level": self.state.level.value,
            "start_time": self.state.start_time,
            "degraded_duration_seconds": degraded_time,
            "failure_count": self.state.failure_count,
            "last_failure_time": self.state.last_failure_time,
            "reason": self.state.reason,
            "auto_recovery_enabled": self.state.auto_recovery_enabled,
            "cache_size": self.cache.size(),
            "statistics": self.stats.copy()
        }
    
    async def stop_monitoring(self):
        """Stop degradation monitoring"""
        if self._recovery_task and not self._recovery_task.done():
            self._recovery_task.cancel()
            try:
                await self._recovery_task
            except asyncio.CancelledError:
                pass


# Factory function
def create_degradation_manager(config: DegradationConfig = None) -> GracefulDegradationManager:
    """Create graceful degradation manager"""
    return GracefulDegradationManager(config)