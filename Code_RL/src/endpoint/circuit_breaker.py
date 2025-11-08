"""
Circuit Breaker Pattern Implementation for RL-Simulator Communication

This module implements circuit breaker patterns to prevent cascade failures
during network issues and provide graceful degradation of service quality.
"""

import asyncio
import time
import logging
from typing import Dict, Any, Optional, Callable, Union, List
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import statistics

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"        # Normal operation
    OPEN = "open"           # Blocking requests
    HALF_OPEN = "half_open" # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker"""
    
    # Failure thresholds
    failure_threshold: int = 5          # Failures before opening
    success_threshold: int = 3          # Successes to close from half-open
    timeout_threshold: float = 30.0     # Timeout duration in seconds
    
    # Time windows
    failure_window_seconds: float = 60.0   # Rolling window for failure counting
    recovery_timeout_seconds: float = 30.0  # Time before half-open attempt
    half_open_timeout_seconds: float = 10.0 # Max time in half-open state
    
    # Monitoring
    monitor_interval_seconds: float = 5.0   # Health check interval
    enable_metrics: bool = True
    
    # Failure detection
    failure_rate_threshold: float = 0.5     # 50% failure rate
    min_requests_for_rate: int = 10         # Minimum requests to calculate rate
    
    # Response time monitoring
    slow_response_threshold_ms: float = 5000.0  # 5 second threshold
    slow_response_failure_count: int = 3        # Treat slow responses as failures


@dataclass
class RequestResult:
    """Result of a request execution"""
    success: bool
    duration: float
    timestamp: float
    error: Optional[str] = None
    status_code: Optional[int] = None


class CircuitBreakerException(Exception):
    """Exception raised when circuit breaker is open"""
    def __init__(self, message: str, circuit_state: CircuitState):
        super().__init__(message)
        self.circuit_state = circuit_state


class CircuitBreakerMetrics:
    """Metrics collection for circuit breaker"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.requests: List[RequestResult] = []
        self.state_changes: List[Dict[str, Any]] = []
        self._lock = asyncio.Lock()
    
    async def record_request(self, result: RequestResult):
        """Record request result"""
        async with self._lock:
            self.requests.append(result)
            if len(self.requests) > self.window_size:
                self.requests = self.requests[-self.window_size:]
    
    async def record_state_change(self, old_state: CircuitState, new_state: CircuitState, reason: str):
        """Record state change"""
        async with self._lock:
            self.state_changes.append({
                "timestamp": time.time(),
                "old_state": old_state.value,
                "new_state": new_state.value,
                "reason": reason
            })
            if len(self.state_changes) > 50:
                self.state_changes = self.state_changes[-50:]
    
    async def get_stats(self, window_seconds: float = 60.0) -> Dict[str, Any]:
        """Get circuit breaker statistics"""
        async with self._lock:
            now = time.time()
            recent_requests = [
                r for r in self.requests 
                if now - r.timestamp <= window_seconds
            ]
            
            if not recent_requests:
                return {
                    "total_requests": 0,
                    "success_rate": 0.0,
                    "failure_rate": 0.0,
                    "avg_response_time": 0.0,
                    "state_changes": len(self.state_changes)
                }
            
            successful = [r for r in recent_requests if r.success]
            failed = [r for r in recent_requests if not r.success]
            durations = [r.duration for r in recent_requests]
            
            return {
                "total_requests": len(recent_requests),
                "successful_requests": len(successful),
                "failed_requests": len(failed),
                "success_rate": len(successful) / len(recent_requests),
                "failure_rate": len(failed) / len(recent_requests),
                "avg_response_time": statistics.mean(durations) if durations else 0.0,
                "p95_response_time": sorted(durations)[int(len(durations) * 0.95)] if durations else 0.0,
                "state_changes": len(self.state_changes),
                "recent_state_changes": self.state_changes[-5:] if self.state_changes else []
            }


class CircuitBreaker:
    """Circuit breaker implementation for fault tolerance"""
    
    def __init__(self, name: str, config: CircuitBreakerConfig):
        self.name = name
        self.config = config
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0.0
        self.state_change_time = time.time()
        
        # Metrics
        self.metrics = CircuitBreakerMetrics() if config.enable_metrics else None
        
        # Health monitoring
        self._monitor_task: Optional[asyncio.Task] = None
        self._health_check_callback: Optional[Callable] = None
        
        logger.info(f"Initialized circuit breaker '{name}' with config: {config}")
    
    async def execute(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection"""
        # Check if circuit is open
        if self.state == CircuitState.OPEN:
            if not self._should_attempt_reset():
                raise CircuitBreakerException(
                    f"Circuit breaker '{self.name}' is OPEN",
                    self.state
                )
            else:
                await self._transition_to_half_open()
        
        start_time = time.time()
        
        try:
            # Execute the function
            result = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
            
            # Record success
            duration = time.time() - start_time
            await self._record_success(duration)
            
            return result
            
        except Exception as e:
            # Record failure
            duration = time.time() - start_time
            await self._record_failure(duration, str(e))
            
            # Re-raise the original exception
            raise
    
    async def _record_success(self, duration: float):
        """Record successful request"""
        if self.metrics:
            result = RequestResult(
                success=True,
                duration=duration,
                timestamp=time.time()
            )
            await self.metrics.record_request(result)
        
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                await self._transition_to_closed()
        elif self.state == CircuitState.CLOSED:
            # Reset failure count on success
            self.failure_count = 0
    
    async def _record_failure(self, duration: float, error: str):
        """Record failed request"""
        if self.metrics:
            result = RequestResult(
                success=False,
                duration=duration,
                timestamp=time.time(),
                error=error
            )
            await self.metrics.record_request(result)
        
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        # Check if we should open the circuit
        if self.state == CircuitState.CLOSED:
            if self._should_open_circuit():
                await self._transition_to_open()
        elif self.state == CircuitState.HALF_OPEN:
            # Any failure in half-open state should open the circuit
            await self._transition_to_open()
    
    def _should_open_circuit(self) -> bool:
        """Determine if circuit should be opened"""
        # Check failure count threshold
        if self.failure_count >= self.config.failure_threshold:
            return True
        
        # Check failure rate if we have enough requests
        if self.metrics:
            # Get recent stats
            stats = asyncio.create_task(
                self.metrics.get_stats(self.config.failure_window_seconds)
            )
            try:
                recent_stats = asyncio.get_event_loop().run_until_complete(stats)
                if (recent_stats["total_requests"] >= self.config.min_requests_for_rate and
                    recent_stats["failure_rate"] >= self.config.failure_rate_threshold):
                    return True
            except:
                pass
        
        return False
    
    def _should_attempt_reset(self) -> bool:
        """Determine if we should attempt to reset from open state"""
        return (time.time() - self.state_change_time) >= self.config.recovery_timeout_seconds
    
    async def _transition_to_open(self):
        """Transition circuit to open state"""
        old_state = self.state
        self.state = CircuitState.OPEN
        self.state_change_time = time.time()
        
        if self.metrics:
            await self.metrics.record_state_change(
                old_state, 
                CircuitState.OPEN, 
                f"Failure threshold reached: {self.failure_count} failures"
            )
        
        logger.warning(f"Circuit breaker '{self.name}' opened due to failures")
    
    async def _transition_to_half_open(self):
        """Transition circuit to half-open state"""
        old_state = self.state
        self.state = CircuitState.HALF_OPEN
        self.state_change_time = time.time()
        self.success_count = 0
        
        if self.metrics:
            await self.metrics.record_state_change(
                old_state,
                CircuitState.HALF_OPEN,
                "Attempting recovery"
            )
        
        logger.info(f"Circuit breaker '{self.name}' moved to half-open for testing")
    
    async def _transition_to_closed(self):
        """Transition circuit to closed state"""
        old_state = self.state
        self.state = CircuitState.CLOSED
        self.state_change_time = time.time()
        self.failure_count = 0
        self.success_count = 0
        
        if self.metrics:
            await self.metrics.record_state_change(
                old_state,
                CircuitState.CLOSED,
                f"Service recovered: {self.success_count} successful requests"
            )
        
        logger.info(f"Circuit breaker '{self.name}' closed - service recovered")
    
    def is_closed(self) -> bool:
        """Check if circuit is closed (normal operation)"""
        return self.state == CircuitState.CLOSED
    
    def is_open(self) -> bool:
        """Check if circuit is open (blocking requests)"""
        return self.state == CircuitState.OPEN
    
    def is_half_open(self) -> bool:
        """Check if circuit is half-open (testing recovery)"""
        return self.state == CircuitState.HALF_OPEN
    
    async def get_status(self) -> Dict[str, Any]:
        """Get current circuit breaker status"""
        stats = {}
        if self.metrics:
            stats = await self.metrics.get_stats()
        
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "last_failure_time": self.last_failure_time,
            "state_change_time": self.state_change_time,
            "time_in_current_state": time.time() - self.state_change_time,
            "statistics": stats
        }
    
    def set_health_check(self, callback: Callable):
        """Set health check callback for monitoring"""
        self._health_check_callback = callback
    
    async def start_monitoring(self):
        """Start health monitoring task"""
        if self._monitor_task is None or self._monitor_task.done():
            self._monitor_task = asyncio.create_task(self._monitor_loop())
            logger.info(f"Started monitoring for circuit breaker '{self.name}'")
    
    async def stop_monitoring(self):
        """Stop health monitoring task"""
        if self._monitor_task and not self._monitor_task.done():
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
            logger.info(f"Stopped monitoring for circuit breaker '{self.name}'")
    
    async def _monitor_loop(self):
        """Health monitoring loop"""
        while True:
            try:
                await asyncio.sleep(self.config.monitor_interval_seconds)
                
                # Perform health check if callback is set
                if self._health_check_callback:
                    try:
                        await self._health_check_callback()
                    except Exception as e:
                        await self._record_failure(0.0, f"Health check failed: {e}")
                
                # Check for half-open timeout
                if (self.state == CircuitState.HALF_OPEN and 
                    time.time() - self.state_change_time >= self.config.half_open_timeout_seconds):
                    await self._transition_to_open()
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in circuit breaker monitor: {e}")


class CircuitBreakerManager:
    """Manager for multiple circuit breakers"""
    
    def __init__(self):
        self.breakers: Dict[str, CircuitBreaker] = {}
    
    def create_breaker(self, name: str, config: CircuitBreakerConfig) -> CircuitBreaker:
        """Create new circuit breaker"""
        if name in self.breakers:
            raise ValueError(f"Circuit breaker '{name}' already exists")
        
        breaker = CircuitBreaker(name, config)
        self.breakers[name] = breaker
        return breaker
    
    def get_breaker(self, name: str) -> Optional[CircuitBreaker]:
        """Get circuit breaker by name"""
        return self.breakers.get(name)
    
    async def get_all_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all circuit breakers"""
        status = {}
        for name, breaker in self.breakers.items():
            status[name] = await breaker.get_status()
        return status
    
    async def start_all_monitoring(self):
        """Start monitoring for all circuit breakers"""
        for breaker in self.breakers.values():
            await breaker.start_monitoring()
    
    async def stop_all_monitoring(self):
        """Stop monitoring for all circuit breakers"""
        for breaker in self.breakers.values():
            await breaker.stop_monitoring()
    
    def remove_breaker(self, name: str):
        """Remove circuit breaker"""
        if name in self.breakers:
            asyncio.create_task(self.breakers[name].stop_monitoring())
            del self.breakers[name]


# Decorator for easy circuit breaker application
def circuit_breaker(name: str, config: CircuitBreakerConfig = None):
    """Decorator to apply circuit breaker to a function"""
    if config is None:
        config = CircuitBreakerConfig()
    
    def decorator(func):
        breaker = CircuitBreaker(name, config)
        
        async def async_wrapper(*args, **kwargs):
            return await breaker.execute(func, *args, **kwargs)
        
        def sync_wrapper(*args, **kwargs):
            return asyncio.run(breaker.execute(func, *args, **kwargs))
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


# Global circuit breaker manager instance
_global_manager = CircuitBreakerManager()


def get_global_manager() -> CircuitBreakerManager:
    """Get global circuit breaker manager"""
    return _global_manager


# Convenience functions
def create_endpoint_circuit_breaker(endpoint_name: str, **config_kwargs) -> CircuitBreaker:
    """Create circuit breaker for endpoint with sensible defaults"""
    config = CircuitBreakerConfig(
        failure_threshold=5,
        success_threshold=3,
        recovery_timeout_seconds=30.0,
        failure_rate_threshold=0.5,
        **config_kwargs
    )
    
    return _global_manager.create_breaker(f"endpoint_{endpoint_name}", config)


def create_websocket_circuit_breaker(**config_kwargs) -> CircuitBreaker:
    """Create circuit breaker for WebSocket with real-time optimized settings"""
    config = CircuitBreakerConfig(
        failure_threshold=3,          # More sensitive for real-time
        success_threshold=2,          # Faster recovery
        recovery_timeout_seconds=10.0, # Quicker retry
        failure_rate_threshold=0.3,   # Lower threshold
        slow_response_threshold_ms=1000.0,  # 1 second for real-time
        **config_kwargs
    )
    
    return _global_manager.create_breaker("websocket", config)