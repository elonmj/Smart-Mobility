"""
Health Check Monitoring and Automatic Recovery for RL-Simulator Communication

This module provides comprehensive health monitoring with automatic recovery
mechanisms to ensure reliable RL training under network instability.
"""

import asyncio
import time
import logging
from typing import Dict, Any, Optional, Callable, List, Union
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import statistics

from .circuit_breaker import CircuitBreaker, CircuitBreakerConfig, CircuitBreakerManager

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health check status levels"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class RecoveryAction(Enum):
    """Available recovery actions"""
    NONE = "none"
    RETRY = "retry"
    RECONNECT = "reconnect"
    FALLBACK_PROTOCOL = "fallback_protocol"
    CIRCUIT_BREAK = "circuit_break"
    ALERT = "alert"


@dataclass
class HealthCheckConfig:
    """Configuration for health checks"""
    
    # Check intervals
    check_interval_seconds: float = 30.0
    fast_check_interval_seconds: float = 5.0    # During degraded state
    
    # Thresholds
    response_time_warning_ms: float = 1000.0    # 1 second warning
    response_time_critical_ms: float = 5000.0   # 5 second critical
    
    # Health determination
    healthy_checks_required: int = 3            # Consecutive checks to be healthy
    unhealthy_checks_required: int = 2          # Consecutive checks to be unhealthy
    
    # Recovery settings
    enable_auto_recovery: bool = True
    max_recovery_attempts: int = 3
    recovery_backoff_seconds: float = 10.0
    
    # Monitoring
    keep_history_hours: int = 24
    enable_detailed_logging: bool = True


@dataclass
class HealthCheckResult:
    """Result of a health check"""
    
    timestamp: float
    status: HealthStatus
    response_time_ms: float
    error: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    recovery_suggested: Optional[RecoveryAction] = None


class HealthChecker(ABC):
    """Abstract base class for health checkers"""
    
    @abstractmethod
    async def check_health(self) -> HealthCheckResult:
        """Perform health check"""
        pass
    
    @abstractmethod
    def get_check_name(self) -> str:
        """Get name of health check"""
        pass


class EndpointHealthChecker(HealthChecker):
    """Health checker for HTTP/WebSocket endpoints"""
    
    def __init__(self, name: str, client, config: HealthCheckConfig):
        self.name = name
        self.client = client
        self.config = config
    
    async def check_health(self) -> HealthCheckResult:
        """Check endpoint health"""
        start_time = time.time()
        
        try:
            # Perform health check request
            if hasattr(self.client, 'health'):
                if asyncio.iscoroutinefunction(self.client.health):
                    health_data = await self.client.health()
                else:
                    health_data = self.client.health()
            else:
                # Fallback to simple connectivity test
                health_data = {"status": "unknown", "method": "fallback"}
            
            response_time_ms = (time.time() - start_time) * 1000
            
            # Determine health status based on response time and content
            status = self._determine_status(response_time_ms, health_data)
            
            return HealthCheckResult(
                timestamp=time.time(),
                status=status,
                response_time_ms=response_time_ms,
                details=health_data,
                recovery_suggested=self._suggest_recovery(status, response_time_ms)
            )
            
        except Exception as e:
            response_time_ms = (time.time() - start_time) * 1000
            
            return HealthCheckResult(
                timestamp=time.time(),
                status=HealthStatus.CRITICAL,
                response_time_ms=response_time_ms,
                error=str(e),
                recovery_suggested=RecoveryAction.RECONNECT
            )
    
    def _determine_status(self, response_time_ms: float, health_data: Dict[str, Any]) -> HealthStatus:
        """Determine health status from response"""
        # Check response time thresholds
        if response_time_ms >= self.config.response_time_critical_ms:
            return HealthStatus.CRITICAL
        elif response_time_ms >= self.config.response_time_warning_ms:
            return HealthStatus.DEGRADED
        
        # Check health data content
        if isinstance(health_data, dict):
            status = health_data.get("status", "").lower()
            if status == "healthy":
                return HealthStatus.HEALTHY
            elif status in ["degraded", "warning"]:
                return HealthStatus.DEGRADED
            elif status in ["unhealthy", "error"]:
                return HealthStatus.UNHEALTHY
        
        return HealthStatus.HEALTHY
    
    def _suggest_recovery(self, status: HealthStatus, response_time_ms: float) -> RecoveryAction:
        """Suggest recovery action based on health status"""
        if status == HealthStatus.CRITICAL:
            return RecoveryAction.RECONNECT
        elif status == HealthStatus.UNHEALTHY:
            return RecoveryAction.RETRY
        elif status == HealthStatus.DEGRADED and response_time_ms > self.config.response_time_warning_ms:
            return RecoveryAction.FALLBACK_PROTOCOL
        
        return RecoveryAction.NONE
    
    def get_check_name(self) -> str:
        return f"endpoint_{self.name}"


class ComprehensiveHealthChecker:
    """Comprehensive health checker for multiple components"""
    
    def __init__(self, config: HealthCheckConfig):
        self.config = config
        self.checkers: List[HealthChecker] = []
        self.history: List[Dict[str, HealthCheckResult]] = []
        self.current_status = HealthStatus.UNKNOWN
        self.consecutive_status_count = 0
        
        # Recovery tracking
        self.recovery_attempts = 0
        self.last_recovery_time = 0.0
        
        # Monitoring task
        self._monitor_task: Optional[asyncio.Task] = None
        self._recovery_callbacks: Dict[RecoveryAction, List[Callable]] = {
            action: [] for action in RecoveryAction
        }
    
    def add_checker(self, checker: HealthChecker):
        """Add health checker"""
        self.checkers.append(checker)
        logger.info(f"Added health checker: {checker.get_check_name()}")
    
    def add_recovery_callback(self, action: RecoveryAction, callback: Callable):
        """Add recovery callback for specific action"""
        self._recovery_callbacks[action].append(callback)
    
    async def check_all_health(self) -> Dict[str, HealthCheckResult]:
        """Check health of all components"""
        results = {}
        
        for checker in self.checkers:
            try:
                result = await checker.check_health()
                results[checker.get_check_name()] = result
            except Exception as e:
                logger.error(f"Health check failed for {checker.get_check_name()}: {e}")
                results[checker.get_check_name()] = HealthCheckResult(
                    timestamp=time.time(),
                    status=HealthStatus.CRITICAL,
                    response_time_ms=0.0,
                    error=str(e)
                )
        
        # Store in history
        self.history.append(results)
        
        # Cleanup old history
        cutoff_time = time.time() - (self.config.keep_history_hours * 3600)
        self.history = [
            h for h in self.history 
            if any(r.timestamp >= cutoff_time for r in h.values())
        ]
        
        # Update overall status
        await self._update_overall_status(results)
        
        return results
    
    async def _update_overall_status(self, results: Dict[str, HealthCheckResult]):
        """Update overall health status"""
        if not results:
            return
        
        # Determine worst status among all checks
        worst_status = HealthStatus.HEALTHY
        for result in results.values():
            if result.status.value == "critical":
                worst_status = HealthStatus.CRITICAL
                break
            elif result.status.value == "unhealthy":
                worst_status = HealthStatus.UNHEALTHY
            elif result.status.value == "degraded" and worst_status == HealthStatus.HEALTHY:
                worst_status = HealthStatus.DEGRADED
        
        # Track consecutive status
        if worst_status == self.current_status:
            self.consecutive_status_count += 1
        else:
            self.consecutive_status_count = 1
            old_status = self.current_status
            self.current_status = worst_status
            
            if self.config.enable_detailed_logging:
                logger.info(f"Health status changed: {old_status.value} -> {worst_status.value}")
        
        # Trigger recovery if needed
        if self.config.enable_auto_recovery:
            await self._attempt_recovery(results)
    
    async def _attempt_recovery(self, results: Dict[str, HealthCheckResult]):
        """Attempt automatic recovery based on health results"""
        # Check if recovery is needed
        if self.current_status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED]:
            self.recovery_attempts = 0
            return
        
        # Check recovery limits
        if self.recovery_attempts >= self.config.max_recovery_attempts:
            logger.warning("Maximum recovery attempts reached")
            return
        
        # Check recovery cooldown
        time_since_last_recovery = time.time() - self.last_recovery_time
        if time_since_last_recovery < self.config.recovery_backoff_seconds:
            return
        
        # Collect suggested recovery actions
        suggested_actions = set()
        for result in results.values():
            if result.recovery_suggested:
                suggested_actions.add(result.recovery_suggested)
        
        # Execute recovery actions
        for action in suggested_actions:
            if action != RecoveryAction.NONE:
                await self._execute_recovery_action(action)
                self.recovery_attempts += 1
                self.last_recovery_time = time.time()
                break  # Execute one action at a time
    
    async def _execute_recovery_action(self, action: RecoveryAction):
        """Execute recovery action"""
        logger.info(f"Executing recovery action: {action.value}")
        
        try:
            # Execute registered callbacks
            for callback in self._recovery_callbacks[action]:
                if asyncio.iscoroutinefunction(callback):
                    await callback()
                else:
                    callback()
                    
        except Exception as e:
            logger.error(f"Recovery action {action.value} failed: {e}")
    
    async def start_monitoring(self):
        """Start health monitoring"""
        if self._monitor_task is None or self._monitor_task.done():
            self._monitor_task = asyncio.create_task(self._monitor_loop())
            logger.info("Started health monitoring")
    
    async def stop_monitoring(self):
        """Stop health monitoring"""
        if self._monitor_task and not self._monitor_task.done():
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
            logger.info("Stopped health monitoring")
    
    async def _monitor_loop(self):
        """Main monitoring loop"""
        while True:
            try:
                # Determine check interval based on current status
                if self.current_status in [HealthStatus.DEGRADED, HealthStatus.UNHEALTHY, HealthStatus.CRITICAL]:
                    interval = self.config.fast_check_interval_seconds
                else:
                    interval = self.config.check_interval_seconds
                
                await asyncio.sleep(interval)
                
                # Perform health checks
                await self.check_all_health()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health monitoring: {e}")
                await asyncio.sleep(5.0)  # Short delay before retry
    
    def get_overall_status(self) -> Dict[str, Any]:
        """Get overall health status"""
        recent_results = self.history[-10:] if self.history else []
        
        return {
            "current_status": self.current_status.value,
            "consecutive_status_count": self.consecutive_status_count,
            "recovery_attempts": self.recovery_attempts,
            "last_recovery_time": self.last_recovery_time,
            "checkers_count": len(self.checkers),
            "recent_checks_count": len(recent_results),
            "monitoring_active": self._monitor_task is not None and not self._monitor_task.done()
        }
    
    def get_health_history(self, hours: int = 1) -> List[Dict[str, Any]]:
        """Get health history for specified hours"""
        cutoff_time = time.time() - (hours * 3600)
        
        filtered_history = []
        for check_results in self.history:
            # Find the timestamp from any result in this check round
            if check_results:
                first_result = next(iter(check_results.values()))
                if first_result.timestamp >= cutoff_time:
                    summary = {
                        "timestamp": first_result.timestamp,
                        "results": {
                            name: {
                                "status": result.status.value,
                                "response_time_ms": result.response_time_ms,
                                "error": result.error
                            }
                            for name, result in check_results.items()
                        }
                    }
                    filtered_history.append(summary)
        
        return filtered_history
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics from health checks"""
        if not self.history:
            return {}
        
        # Collect metrics from recent history
        recent_results = self.history[-50:]  # Last 50 check rounds
        
        metrics = {}
        for checker_name in {name for results in recent_results for name in results.keys()}:
            checker_results = [
                results[checker_name] for results in recent_results 
                if checker_name in results
            ]
            
            if checker_results:
                response_times = [r.response_time_ms for r in checker_results]
                error_count = sum(1 for r in checker_results if r.error)
                
                metrics[checker_name] = {
                    "check_count": len(checker_results),
                    "error_count": error_count,
                    "error_rate": error_count / len(checker_results),
                    "avg_response_time_ms": statistics.mean(response_times),
                    "p95_response_time_ms": sorted(response_times)[int(len(response_times) * 0.95)] if response_times else 0,
                    "min_response_time_ms": min(response_times) if response_times else 0,
                    "max_response_time_ms": max(response_times) if response_times else 0
                }
        
        return metrics


class HealthAwareClient:
    """Client wrapper with integrated health monitoring"""
    
    def __init__(self, client, health_config: HealthCheckConfig = None):
        self.client = client
        self.health_config = health_config or HealthCheckConfig()
        self.health_checker = ComprehensiveHealthChecker(self.health_config)
        
        # Add health checker for the client
        endpoint_checker = EndpointHealthChecker("primary", client, self.health_config)
        self.health_checker.add_checker(endpoint_checker)
        
        # Recovery callbacks
        self._setup_recovery_callbacks()
        
        # Circuit breaker integration
        self.circuit_breaker: Optional[CircuitBreaker] = None
    
    def _setup_recovery_callbacks(self):
        """Setup recovery callbacks"""
        self.health_checker.add_recovery_callback(
            RecoveryAction.RECONNECT, 
            self._recover_reconnect
        )
        self.health_checker.add_recovery_callback(
            RecoveryAction.RETRY,
            self._recover_retry
        )
    
    async def _recover_reconnect(self):
        """Recovery action: reconnect"""
        logger.info("Attempting to reconnect client")
        try:
            if hasattr(self.client, 'disconnect'):
                await self.client.disconnect()
            if hasattr(self.client, 'connect'):
                await self.client.connect()
        except Exception as e:
            logger.error(f"Reconnection failed: {e}")
    
    async def _recover_retry(self):
        """Recovery action: retry last operation"""
        # This could be implemented to retry the last failed operation
        # For now, just log the action
        logger.info("Recovery retry action triggered")
    
    def enable_circuit_breaker(self, config: CircuitBreakerConfig = None):
        """Enable circuit breaker protection"""
        if config is None:
            config = CircuitBreakerConfig()
        
        self.circuit_breaker = CircuitBreaker("health_aware_client", config)
        
        # Set health check callback for circuit breaker
        self.circuit_breaker.set_health_check(self._circuit_breaker_health_check)
    
    async def _circuit_breaker_health_check(self):
        """Health check for circuit breaker"""
        results = await self.health_checker.check_all_health()
        
        # If any critical issues, raise exception to trigger circuit breaker
        for result in results.values():
            if result.status == HealthStatus.CRITICAL:
                raise Exception(f"Critical health issue: {result.error}")
    
    async def execute_with_health_protection(self, func: Callable, *args, **kwargs):
        """Execute function with health monitoring and circuit breaker protection"""
        if self.circuit_breaker:
            return await self.circuit_breaker.execute(func, *args, **kwargs)
        else:
            return await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
    
    async def start_health_monitoring(self):
        """Start health monitoring"""
        await self.health_checker.start_monitoring()
        if self.circuit_breaker:
            await self.circuit_breaker.start_monitoring()
    
    async def stop_health_monitoring(self):
        """Stop health monitoring"""
        await self.health_checker.stop_monitoring()
        if self.circuit_breaker:
            await self.circuit_breaker.stop_monitoring()
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status"""
        health_status = self.health_checker.get_overall_status()
        
        if self.circuit_breaker:
            circuit_status = await self.circuit_breaker.get_status()
            health_status["circuit_breaker"] = circuit_status
        
        return health_status
    
    # Delegate client methods with health protection
    async def reset(self, scenario: Optional[str] = None, seed: Optional[int] = None):
        return await self.execute_with_health_protection(self.client.reset, scenario, seed)
    
    async def step(self, dt: float, repeat_k: int = 1):
        return await self.execute_with_health_protection(self.client.step, dt, repeat_k)
    
    async def set_signal(self, signal_plan: Dict[str, Any]) -> bool:
        return await self.execute_with_health_protection(self.client.set_signal, signal_plan)
    
    async def get_metrics(self) -> Dict[str, Any]:
        metrics = await self.execute_with_health_protection(self.client.get_metrics)
        
        # Add health metrics
        health_metrics = self.health_checker.get_performance_metrics()
        metrics["health_metrics"] = health_metrics
        
        return metrics


# Factory function
def create_health_aware_client(client, health_config: HealthCheckConfig = None, enable_circuit_breaker: bool = True):
    """Create health-aware client wrapper"""
    health_client = HealthAwareClient(client, health_config)
    
    if enable_circuit_breaker:
        health_client.enable_circuit_breaker()
    
    return health_client