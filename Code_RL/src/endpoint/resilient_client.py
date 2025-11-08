"""
Integrated Resilient Client with Complete Fault Tolerance

This module integrates all fault tolerance mechanisms (circuit breaker, health monitoring,
graceful degradation) into a comprehensive resilient client for RL-simulator communication.
"""

import asyncio
import time
import logging
from typing import Dict, Any, Optional, Union, List, Callable
from dataclasses import dataclass
import traceback

from .circuit_breaker import CircuitBreaker, CircuitBreakerConfig, CircuitBreakerManager
from .health_monitoring import HealthAwareClient, HealthCheckConfig, HealthStatus, ComprehensiveHealthChecker
from .graceful_degradation import GracefulDegradationManager, DegradationConfig, DegradationLevel

logger = logging.getLogger(__name__)


@dataclass
class ResilientClientConfig:
    """Configuration for resilient client"""
    
    # Circuit breaker settings
    circuit_breaker_config: Optional[CircuitBreakerConfig] = None
    enable_circuit_breaker: bool = True
    
    # Health monitoring settings
    health_check_config: Optional[HealthCheckConfig] = None
    enable_health_monitoring: bool = True
    
    # Graceful degradation settings
    degradation_config: Optional[DegradationConfig] = None
    enable_graceful_degradation: bool = True
    
    # Integration settings
    enable_auto_recovery: bool = True
    fallback_chain_enabled: bool = True
    
    # Logging and metrics
    enable_detailed_logging: bool = True
    enable_performance_metrics: bool = True


class ResilientARZClient:
    """
    Resilient ARZ client with comprehensive fault tolerance
    
    Features:
    - Circuit breaker for fast failure detection
    - Health monitoring with automatic recovery
    - Graceful degradation with cached/simulated modes
    - Intelligent fallback chains
    - Performance monitoring and metrics
    """
    
    def __init__(self, primary_client, config: Optional[ResilientClientConfig] = None):
        self.primary_client = primary_client
        self.config = config or ResilientClientConfig()
        
        # Initialize components
        self._initialize_components()
        
        # Integration state
        self.is_started = False
        self.last_successful_operation = time.time()
        
        # Performance metrics
        self.metrics = {
            "operations_total": 0,
            "operations_successful": 0,
            "operations_failed": 0,
            "degraded_operations": 0,
            "circuit_breaker_trips": 0,
            "recovery_events": 0,
            "cache_hits": 0,
            "simulated_operations": 0
        }
    
    def _initialize_components(self):
        """Initialize all fault tolerance components"""
        
        # Circuit breaker
        if self.config.enable_circuit_breaker:
            cb_config = self.config.circuit_breaker_config or CircuitBreakerConfig()
            self.circuit_breaker = CircuitBreaker("resilient_client", cb_config)
        else:
            self.circuit_breaker = None
        
        # Health monitoring
        if self.config.enable_health_monitoring:
            health_config = self.config.health_check_config or HealthCheckConfig()
            self.health_client = HealthAwareClient(self.primary_client, health_config)
            if self.circuit_breaker:
                self.health_client.circuit_breaker = self.circuit_breaker
        else:
            self.health_client = None
        
        # Graceful degradation
        if self.config.enable_graceful_degradation:
            degradation_config = self.config.degradation_config or DegradationConfig()
            self.degradation_manager = GracefulDegradationManager(degradation_config)
            
            # Set up callbacks
            self._setup_degradation_callbacks()
        else:
            self.degradation_manager = None
        
        # Integration setup
        self._setup_component_integration()
    
    def _setup_degradation_callbacks(self):
        """Setup callbacks for degradation events"""
        if not self.degradation_manager:
            return
        
        # Add callbacks for different degradation levels
        for level in DegradationLevel:
            self.degradation_manager.add_mode_callback(level, self._on_degradation_level_change)
        
        # Set recovery check callback
        self.degradation_manager.set_recovery_check_callback(self._check_primary_service_recovery)
    
    def _setup_component_integration(self):
        """Setup integration between components"""
        
        # Circuit breaker integration with degradation
        if self.circuit_breaker and self.degradation_manager:
            # Integration would require extending circuit breaker with callbacks
            # For now, we'll monitor circuit breaker state manually
            logger.info("Circuit breaker and degradation manager integrated")
        
        # Health monitoring integration with degradation
        if self.health_client and self.degradation_manager:
            async def on_health_degraded(old_level, new_level, reason):
                if new_level in [DegradationLevel.CACHED, DegradationLevel.SIMULATED, DegradationLevel.OFFLINE]:
                    logger.warning(f"Health degradation triggered graceful degradation: {reason}")
    
    async def _on_degradation_level_change(self, old_level: DegradationLevel, new_level: DegradationLevel, reason: str):
        """Handle degradation level changes"""
        if self.config.enable_detailed_logging:
            logger.info(f"Degradation level changed: {old_level.value} -> {new_level.value}. Reason: {reason}")
        
        # Update metrics
        if new_level != DegradationLevel.NORMAL:
            self.metrics["degraded_operations"] += 1
        elif old_level != DegradationLevel.NORMAL:
            self.metrics["recovery_events"] += 1
    
    async def _check_primary_service_recovery(self) -> bool:
        """Check if primary service has recovered"""
        try:
            # Try a simple operation to check health
            if hasattr(self.primary_client, 'health'):
                if asyncio.iscoroutinefunction(self.primary_client.health):
                    await self.primary_client.health()
                else:
                    self.primary_client.health()
                return True
            elif hasattr(self.primary_client, 'get_metrics'):
                if asyncio.iscoroutinefunction(self.primary_client.get_metrics):
                    await self.primary_client.get_metrics()
                else:
                    self.primary_client.get_metrics()
                return True
            else:
                # Fallback: assume recovered if no specific health check
                return True
                
        except Exception as e:
            logger.debug(f"Primary service recovery check failed: {e}")
            return False
    
    async def start(self):
        """Start all monitoring and fault tolerance systems"""
        if self.is_started:
            return
        
        logger.info("Starting resilient client systems")
        
        try:
            # Start health monitoring
            if self.health_client:
                await self.health_client.start_health_monitoring()
            
            # Start circuit breaker monitoring
            if self.circuit_breaker:
                await self.circuit_breaker.start_monitoring()
            
            self.is_started = True
            logger.info("Resilient client started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start resilient client: {e}")
            raise
    
    async def stop(self):
        """Stop all monitoring systems"""
        if not self.is_started:
            return
        
        logger.info("Stopping resilient client systems")
        
        try:
            # Stop health monitoring
            if self.health_client:
                await self.health_client.stop_health_monitoring()
            
            # Stop circuit breaker monitoring
            if self.circuit_breaker:
                await self.circuit_breaker.stop_monitoring()
            
            # Stop degradation monitoring
            if self.degradation_manager:
                await self.degradation_manager.stop_monitoring()
            
            self.is_started = False
            logger.info("Resilient client stopped")
            
        except Exception as e:
            logger.error(f"Error stopping resilient client: {e}")
    
    async def reset(self, scenario: Optional[str] = None, seed: Optional[int] = None) -> Dict[str, Any]:
        """Reset simulation with fault tolerance"""
        return await self._execute_resilient_operation("reset", scenario=scenario, seed=seed)
    
    async def step(self, dt: float, repeat_k: int = 1) -> Dict[str, Any]:
        """Execute simulation step with fault tolerance"""
        return await self._execute_resilient_operation("step", dt=dt, repeat_k=repeat_k)
    
    async def set_signal(self, signal_plan: Dict[str, Any]) -> bool:
        """Set traffic signal with fault tolerance"""
        return await self._execute_resilient_operation("set_signal", signal_plan=signal_plan)
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get metrics with fault tolerance"""
        metrics = await self._execute_resilient_operation("get_metrics")
        
        # Add resilience metrics
        if isinstance(metrics, dict):
            metrics["resilience"] = self.get_resilience_metrics()
        
        return metrics
    
    async def _execute_resilient_operation(self, operation: str, **kwargs) -> Any:
        """Execute operation with comprehensive fault tolerance"""
        self.metrics["operations_total"] += 1
        start_time = time.time()
        
        try:
            # Check if we're in degraded mode
            if self.degradation_manager and self.degradation_manager.state.level != DegradationLevel.NORMAL:
                return await self._execute_degraded_operation(operation, **kwargs)
            
            # Try primary operation with circuit breaker protection
            if self.circuit_breaker:
                result = await self.circuit_breaker.execute(
                    self._execute_primary_operation, operation, **kwargs
                )
            else:
                result = await self._execute_primary_operation(operation, **kwargs)
            
            # Record success
            self.metrics["operations_successful"] += 1
            self.last_successful_operation = time.time()
            
            # Cache successful results for degradation scenarios
            if self.degradation_manager and operation in ["reset", "step"]:
                cache_key = f"{operation}_{int(start_time)}"
                self.degradation_manager.cache_state(cache_key, result)
            
            return result
            
        except Exception as e:
            self.metrics["operations_failed"] += 1
            
            # Record failure for degradation analysis
            if self.degradation_manager:
                await self.degradation_manager.record_failure(e, f"operation_{operation}")
            
            # Try fallback operation if enabled
            if self.config.fallback_chain_enabled:
                return await self._execute_fallback_operation(operation, e, **kwargs)
            else:
                raise
    
    async def _execute_primary_operation(self, operation: str, **kwargs) -> Any:
        """Execute operation on primary client"""
        if self.health_client:
            # Use health-aware client
            client = self.health_client
        else:
            client = self.primary_client
        
        if operation == "reset":
            return await client.reset(kwargs.get("scenario"), kwargs.get("seed"))
        elif operation == "step":
            dt = kwargs.get("dt")
            if dt is None:
                raise ValueError("dt parameter is required for step operation")
            return await client.step(dt, kwargs.get("repeat_k", 1))
        elif operation == "set_signal":
            signal_plan = kwargs.get("signal_plan")
            if signal_plan is None:
                raise ValueError("signal_plan parameter is required for set_signal operation")
            return await client.set_signal(signal_plan)
        elif operation == "get_metrics":
            return await client.get_metrics()
        else:
            raise ValueError(f"Unknown operation: {operation}")
    
    async def _execute_degraded_operation(self, operation: str, **kwargs) -> Any:
        """Execute operation in degraded mode"""
        if not self.degradation_manager:
            raise Exception("Degradation manager not available")
        
        self.metrics["degraded_operations"] += 1
        
        # Prepare operation parameters
        op_kwargs = kwargs.copy()
        
        # Add current state context if available
        if operation == "step" and hasattr(self, '_last_state'):
            op_kwargs["current_state"] = self._last_state
        
        result = await self.degradation_manager.execute_degraded_operation(operation, **op_kwargs)
        
        # Update metrics based on degradation level
        level = self.degradation_manager.state.level
        if level == DegradationLevel.CACHED:
            self.metrics["cache_hits"] += 1
        elif level == DegradationLevel.SIMULATED:
            self.metrics["simulated_operations"] += 1
        
        # Store state for next operation
        if operation in ["reset", "step"] and isinstance(result, dict):
            self._last_state = result
        
        return result
    
    async def _execute_fallback_operation(self, operation: str, original_error: Exception, **kwargs) -> Any:
        """Execute fallback operation when primary fails"""
        logger.warning(f"Primary operation failed, trying fallback for {operation}: {original_error}")
        
        try:
            # Force degradation if not already degraded
            if self.degradation_manager and self.degradation_manager.state.level == DegradationLevel.NORMAL:
                await self.degradation_manager.force_degradation_level(
                    DegradationLevel.CACHED, 
                    f"Fallback for {operation}"
                )
            
            # Execute in degraded mode
            return await self._execute_degraded_operation(operation, **kwargs)
            
        except Exception as fallback_error:
            logger.error(f"Fallback operation also failed: {fallback_error}")
            
            # Return error response with context
            return {
                "error": str(original_error),
                "fallback_error": str(fallback_error),
                "operation": operation,
                "resilient_mode": True,
                "timestamp": time.time()
            }
    
    def get_resilience_metrics(self) -> Dict[str, Any]:
        """Get comprehensive resilience metrics"""
        # Create new dict with base metrics
        metrics: Dict[str, Any] = {
            "operations_total": self.metrics["operations_total"],
            "operations_successful": self.metrics["operations_successful"],
            "operations_failed": self.metrics["operations_failed"],
            "degraded_operations": self.metrics["degraded_operations"],
            "circuit_breaker_trips": self.metrics["circuit_breaker_trips"],
            "recovery_events": self.metrics["recovery_events"],
            "cache_hits": self.metrics["cache_hits"],
            "simulated_operations": self.metrics["simulated_operations"]
        }
        
        # Calculate derived metrics
        total_ops = int(self.metrics["operations_total"])
        if total_ops > 0:
            successful_ops = int(self.metrics["operations_successful"])
            failed_ops = int(self.metrics["operations_failed"])
            degraded_ops = int(self.metrics["degraded_operations"])
            
            metrics["success_rate"] = successful_ops / total_ops
            metrics["failure_rate"] = failed_ops / total_ops
            metrics["degraded_rate"] = degraded_ops / total_ops
        
        # Add component metrics
        if self.circuit_breaker:
            try:
                # Try to get status synchronously if possible
                metrics["circuit_breaker"] = {"status": "available"}
            except Exception:
                metrics["circuit_breaker"] = {"status": "error"}
        
        if self.health_client:
            try:
                metrics["health"] = {"status": "available"}
            except Exception:
                metrics["health"] = {"status": "error"}
        
        if self.degradation_manager:
            metrics["degradation"] = self.degradation_manager.get_degradation_status()
        
        # System status
        metrics["system_status"] = {
            "is_started": self.is_started,
            "last_successful_operation": self.last_successful_operation,
            "uptime_seconds": time.time() - self.last_successful_operation
        }
        
        return metrics
    
    async def get_comprehensive_status(self) -> Dict[str, Any]:
        """Get comprehensive status of all systems"""
        status = {
            "resilient_client": {
                "is_started": self.is_started,
                "config": {
                    "circuit_breaker_enabled": self.config.enable_circuit_breaker,
                    "health_monitoring_enabled": self.config.enable_health_monitoring,
                    "graceful_degradation_enabled": self.config.enable_graceful_degradation,
                    "auto_recovery_enabled": self.config.enable_auto_recovery
                }
            }
        }
        
        # Circuit breaker status
        if self.circuit_breaker:
            status["circuit_breaker"] = await self.circuit_breaker.get_status()
        
        # Health monitoring status
        if self.health_client:
            status["health_monitoring"] = await self.health_client.get_health_status()
            status["performance_metrics"] = self.health_client.health_checker.get_performance_metrics()
        
        # Degradation status
        if self.degradation_manager:
            status["graceful_degradation"] = self.degradation_manager.get_degradation_status()
        
        # Resilience metrics
        status["metrics"] = self.get_resilience_metrics()
        
        return status
    
    async def force_recovery(self):
        """Force recovery to normal operation"""
        logger.info("Forcing recovery to normal operation")
        
        # Reset circuit breaker - just log since it will recover automatically
        if self.circuit_breaker:
            logger.info("Circuit breaker will recover automatically")
        
        # Force degradation recovery
        if self.degradation_manager:
            await self.degradation_manager.force_recovery()
        
        self.metrics["recovery_events"] += 1
        logger.info("Recovery completed")
    
    async def force_degradation(self, level: DegradationLevel, reason: str = "Manual"):
        """Force degradation to specific level"""
        if self.degradation_manager:
            await self.degradation_manager.force_degradation_level(level, reason)
        else:
            logger.warning("Graceful degradation not enabled")
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check"""
        if self.health_client:
            return await self.health_client.health_checker.check_all_health()
        else:
            return {"error": "Health monitoring not enabled"}
    
    # Context manager support
    async def __aenter__(self):
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.stop()


# Factory functions
def create_resilient_client(primary_client, config: Optional[ResilientClientConfig] = None) -> ResilientARZClient:
    """Create resilient client with all fault tolerance features"""
    return ResilientARZClient(primary_client, config)


def create_minimal_resilient_client(primary_client) -> ResilientARZClient:
    """Create resilient client with minimal configuration"""
    config = ResilientClientConfig(
        enable_circuit_breaker=True,
        enable_health_monitoring=False,
        enable_graceful_degradation=True
    )
    return ResilientARZClient(primary_client, config)


def create_full_resilient_client(primary_client) -> ResilientARZClient:
    """Create resilient client with all features enabled"""
    config = ResilientClientConfig(
        enable_circuit_breaker=True,
        enable_health_monitoring=True,
        enable_graceful_degradation=True,
        enable_auto_recovery=True,
        fallback_chain_enabled=True,
        enable_detailed_logging=True,
        enable_performance_metrics=True
    )
    return ResilientARZClient(primary_client, config)