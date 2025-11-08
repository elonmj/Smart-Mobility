"""
Enhanced HTTP Client with Connection Pooling and Performance Optimization

This module extends the existing HTTPEndpointClient with advanced connection management,
session optimization, and performance monitoring capabilities.
"""

import time
import logging
import asyncio
import threading
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass, field
from urllib3.util.retry import Retry
from urllib3.poolmanager import PoolManager
import requests
from requests.adapters import HTTPAdapter
import concurrent.futures

from .client import (
    ARZEndpointClient, SimulationState, EndpointConfig, 
    EndpointError, TimeoutError, InvalidCommandError
)

logger = logging.getLogger(__name__)


@dataclass
class OptimizedEndpointConfig(EndpointConfig):
    """Enhanced configuration with performance optimization settings"""
    
    # Connection pooling settings
    pool_connections: int = 10
    pool_maxsize: int = 20
    pool_block: bool = False
    
    # Retry strategy
    retry_total: int = 3
    retry_backoff_factor: float = 0.3
    retry_status_forcelist: List[int] = field(default_factory=lambda: [500, 502, 503, 504])
    
    # Connection settings
    connect_timeout: float = 10.0
    read_timeout: float = 30.0
    keep_alive: bool = True
    
    # Performance monitoring
    enable_metrics: bool = True
    metrics_window_size: int = 100
    
    # Session optimization
    tcp_keepalive: bool = True
    tcp_keepidle: int = 600
    tcp_keepintvl: int = 60
    tcp_keepcnt: int = 9


@dataclass
class RequestMetrics:
    """Metrics for HTTP request performance"""
    timestamp: float
    endpoint: str
    method: str
    duration: float
    status_code: int
    retry_count: int = 0
    error: Optional[str] = None


class MetricsCollector:
    """Collects and analyzes HTTP request performance metrics"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.metrics: List[RequestMetrics] = []
        self._lock = threading.Lock()
    
    def record_request(self, metrics: RequestMetrics):
        """Record a request's performance metrics"""
        with self._lock:
            self.metrics.append(metrics)
            # Keep only the latest N requests
            if len(self.metrics) > self.window_size:
                self.metrics = self.metrics[-self.window_size:]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get aggregated performance statistics"""
        with self._lock:
            if not self.metrics:
                return {}
            
            durations = [m.duration for m in self.metrics]
            recent_errors = [m for m in self.metrics if m.error is not None]
            
            return {
                "total_requests": len(self.metrics),
                "avg_duration": sum(durations) / len(durations),
                "min_duration": min(durations),
                "max_duration": max(durations),
                "p95_duration": sorted(durations)[int(len(durations) * 0.95)],
                "error_rate": len(recent_errors) / len(self.metrics),
                "total_retries": sum(m.retry_count for m in self.metrics),
                "window_size": len(self.metrics)
            }
    
    def get_endpoint_stats(self, endpoint: str) -> Dict[str, Any]:
        """Get statistics for a specific endpoint"""
        with self._lock:
            endpoint_metrics = [m for m in self.metrics if m.endpoint == endpoint]
            if not endpoint_metrics:
                return {}
            
            durations = [m.duration for m in endpoint_metrics]
            errors = [m for m in endpoint_metrics if m.error is not None]
            
            return {
                "endpoint": endpoint,
                "request_count": len(endpoint_metrics),
                "avg_duration": sum(durations) / len(durations),
                "error_rate": len(errors) / len(endpoint_metrics) if endpoint_metrics else 0,
                "total_retries": sum(m.retry_count for m in endpoint_metrics)
            }


class OptimizedHTTPEndpointClient(ARZEndpointClient):
    """HTTP client with connection pooling, session optimization, and performance monitoring"""
    
    def __init__(self, config: OptimizedEndpointConfig):
        self.config = config
        self.base_url = f"http://{config.host}:{config.port}{config.base_url}"
        
        # Initialize metrics collector
        self.metrics_collector = MetricsCollector(config.metrics_window_size) if config.enable_metrics else None
        
        # Setup optimized session with connection pooling
        self.session = self._create_optimized_session()
        
        logger.info(f"Initialized optimized HTTP client with pool_size={config.pool_maxsize}, "
                   f"keep_alive={config.keep_alive}, metrics_enabled={config.enable_metrics}")
    
    def _create_optimized_session(self) -> requests.Session:
        """Create session with connection pooling and retry strategy"""
        session = requests.Session()
        
        # Configure retry strategy
        retry_strategy = Retry(
            total=self.config.retry_total,
            backoff_factor=self.config.retry_backoff_factor,
            status_forcelist=self.config.retry_status_forcelist,
            raise_on_status=False
        )
        
        # Configure HTTP adapter with connection pooling
        adapter = HTTPAdapter(
            pool_connections=self.config.pool_connections,
            pool_maxsize=self.config.pool_maxsize,
            pool_block=self.config.pool_block,
            max_retries=retry_strategy
        )
        
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        # Set session headers
        session.headers.update({
            "Content-Type": "application/json",
            "Connection": "keep-alive" if self.config.keep_alive else "close",
            "User-Agent": "ARZ-RL-Client/1.0"
        })
        
        return session
    
    def _make_request(self, method: str, endpoint: str, data: Optional[Dict] = None) -> Dict[str, Any]:
        """Make HTTP request with performance monitoring and optimization"""
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        start_time = time.time()
        retry_count = 0
        error_msg = None
        status_code = 0
        
        try:
            # Set timeouts as tuple (connect, read)
            timeout = (self.config.connect_timeout, self.config.read_timeout)
            
            if method.upper() == "GET":
                response = self.session.get(url, timeout=timeout, params=data)
            elif method.upper() == "POST":
                response = self.session.post(url, timeout=timeout, json=data)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            status_code = response.status_code
            
            # Handle HTTP errors
            if response.status_code >= 400:
                if response.status_code == 400:
                    raise InvalidCommandError(f"Invalid command: {response.text}")
                else:
                    raise EndpointError(f"HTTP error {response.status_code}: {response.text}")
            
            result = response.json()
            
        except requests.exceptions.Timeout as e:
            error_msg = f"Timeout: {str(e)}"
            raise TimeoutError(f"Request to {url} timed out")
            
        except requests.exceptions.RequestException as e:
            error_msg = f"Request failed: {str(e)}"
            raise EndpointError(error_msg)
            
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            raise EndpointError(error_msg)
            
        finally:
            # Record metrics if enabled
            if self.metrics_collector:
                duration = time.time() - start_time
                metrics = RequestMetrics(
                    timestamp=start_time,
                    endpoint=endpoint,
                    method=method.upper(),
                    duration=duration,
                    status_code=status_code,
                    retry_count=retry_count,
                    error=error_msg
                )
                self.metrics_collector.record_request(metrics)
        
        return result
    
    def reset(self, scenario: Optional[str] = None, seed: Optional[int] = None) -> Tuple[SimulationState, float]:
        """Reset ARZ simulation with optimized HTTP client"""
        data = {
            "scenario": scenario or "default",
            "seed": seed,
            "dt_sim": self.config.dt_sim
        }
        
        logger.info(f"Resetting simulation with scenario={scenario}, seed={seed}")
        result = self._make_request("POST", "reset", data)
        
        state = SimulationState(
            timestamp=result["timestamp"],
            branches=result["branches"],
            phase_id=result.get("phase_id")
        )
        
        return state, result["timestamp"]
    
    def set_signal(self, signal_plan: Dict[str, Any]) -> bool:
        """Set traffic signal plan with optimized HTTP client"""
        logger.debug(f"Setting signal plan: {signal_plan}")
        result = self._make_request("POST", "signals", signal_plan)
        return result.get("success", False)
    
    def step(self, dt: float, repeat_k: int = 1) -> Tuple[SimulationState, float]:
        """Advance simulation with optimized HTTP client"""
        data = {
            "dt": dt,
            "repeat": repeat_k
        }
        
        result = self._make_request("POST", "step", data)
        
        state = SimulationState(
            timestamp=result["timestamp"],
            branches=result["branches"],
            phase_id=result.get("phase_id")
        )
        
        return state, result["timestamp"]
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get simulation metrics with performance data"""
        sim_metrics = self._make_request("GET", "metrics")
        
        if self.metrics_collector:
            performance_stats = self.metrics_collector.get_stats()
            sim_metrics["http_performance"] = performance_stats
        
        return sim_metrics
    
    def health(self) -> Dict[str, Any]:
        """Enhanced health check with connection pool status"""
        start_time = time.time()
        result = self._make_request("GET", "health")
        latency = time.time() - start_time
        
        result["latency"] = latency
        
        # Add connection pool information
        if hasattr(self.session, 'adapters'):
            for prefix, adapter in self.session.adapters.items():
                if hasattr(adapter, 'poolmanager'):
                    pool_manager = adapter.poolmanager
                    result[f"connection_pool_{prefix}"] = {
                        "num_pools": len(pool_manager.pools),
                        "pool_maxsize": adapter.config.get("pool_maxsize", "unknown"),
                        "pool_connections": adapter.config.get("pool_connections", "unknown")
                    }
        
        # Add performance metrics if available
        if self.metrics_collector:
            result["performance_stats"] = self.metrics_collector.get_stats()
        
        return result
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get detailed performance statistics"""
        if not self.metrics_collector:
            return {"error": "Performance metrics not enabled"}
        
        stats = self.metrics_collector.get_stats()
        
        # Add endpoint-specific stats
        endpoints = ["reset", "step", "signals", "metrics", "health"]
        endpoint_stats = {}
        for endpoint in endpoints:
            endpoint_stats[endpoint] = self.metrics_collector.get_endpoint_stats(endpoint)
        
        stats["endpoints"] = endpoint_stats
        return stats
    
    def close(self):
        """Clean up resources"""
        if self.session:
            self.session.close()
            logger.info("HTTP session closed")


# Factory function to create optimized client
def create_optimized_client(config: OptimizedEndpointConfig) -> OptimizedHTTPEndpointClient:
    """Create an optimized HTTP endpoint client"""
    return OptimizedHTTPEndpointClient(config)