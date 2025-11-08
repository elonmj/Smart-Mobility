"""
Asynchronous HTTP Client for Non-blocking RL Environment Communication

This module provides async/await support for HTTP communication with the ARZ simulator,
preventing RL training loops from blocking during network I/O operations.
"""

import asyncio
import aiohttp
import time
import logging
from typing import Dict, Any, Optional, List, Tuple, AsyncContextManager
from dataclasses import dataclass, field
import concurrent.futures
from contextlib import asynccontextmanager

from .client import SimulationState, EndpointError, TimeoutError, InvalidCommandError
from .optimized_client import OptimizedEndpointConfig, RequestMetrics, MetricsCollector

logger = logging.getLogger(__name__)


@dataclass
class AsyncEndpointConfig(OptimizedEndpointConfig):
    """Configuration for async HTTP client"""
    
    # Async-specific settings
    connector_limit: int = 20
    connector_limit_per_host: int = 10
    tcp_connector_ttl_dns_cache: int = 300
    tcp_connector_use_dns_cache: bool = True
    
    # Async timeout settings
    total_timeout: float = 60.0
    sock_connect_timeout: float = 10.0
    sock_read_timeout: float = 30.0
    
    # Concurrent request limits
    max_concurrent_requests: int = 5
    request_semaphore_timeout: float = 30.0


class AsyncMetricsCollector(MetricsCollector):
    """Thread-safe async metrics collector"""
    
    def __init__(self, window_size: int = 100):
        super().__init__(window_size)
        self._async_lock = asyncio.Lock()
    
    async def record_request_async(self, metrics: RequestMetrics):
        """Async version of record_request"""
        async with self._async_lock:
            self.metrics.append(metrics)
            if len(self.metrics) > self.window_size:
                self.metrics = self.metrics[-self.window_size:]
    
    async def get_stats_async(self) -> Dict[str, Any]:
        """Async version of get_stats"""
        async with self._async_lock:
            return self.get_stats()


class AsyncARZEndpointClient:
    """Async HTTP client for ARZ simulator communication"""
    
    def __init__(self, config: AsyncEndpointConfig):
        self.config = config
        self.base_url = f"http://{config.host}:{config.port}{config.base_url}"
        self.session: Optional[aiohttp.ClientSession] = None
        self.metrics_collector = AsyncMetricsCollector(config.metrics_window_size) if config.enable_metrics else None
        
        # Semaphore to limit concurrent requests
        self._request_semaphore = asyncio.Semaphore(config.max_concurrent_requests)
        
        logger.info(f"Initialized async HTTP client with max_concurrent={config.max_concurrent_requests}")
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()
    
    async def connect(self):
        """Initialize the async HTTP session"""
        if self.session and not self.session.closed:
            return
        
        # Configure TCP connector
        connector = aiohttp.TCPConnector(
            limit=self.config.connector_limit,
            limit_per_host=self.config.connector_limit_per_host,
            ttl_dns_cache=self.config.tcp_connector_ttl_dns_cache,
            use_dns_cache=self.config.tcp_connector_use_dns_cache,
            keepalive_timeout=self.config.tcp_keepidle,
            enable_cleanup_closed=True
        )
        
        # Configure timeouts
        timeout = aiohttp.ClientTimeout(
            total=self.config.total_timeout,
            connect=self.config.sock_connect_timeout,
            sock_read=self.config.sock_read_timeout
        )
        
        # Create session
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={
                "Content-Type": "application/json",
                "Connection": "keep-alive",
                "User-Agent": "ARZ-RL-AsyncClient/1.0"
            }
        )
        
        logger.info("Async HTTP session initialized")
    
    async def _make_request(self, method: str, endpoint: str, data: Optional[Dict] = None) -> Dict[str, Any]:
        """Make async HTTP request with performance monitoring"""
        if not self.session:
            await self.connect()
        
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        start_time = time.time()
        error_msg = None
        status_code = 0
        
        # Acquire semaphore to limit concurrent requests
        try:
            await asyncio.wait_for(
                self._request_semaphore.acquire(),
                timeout=self.config.request_semaphore_timeout
            )
        except asyncio.TimeoutError:
            raise TimeoutError(f"Timeout acquiring request semaphore for {endpoint}")
        
        try:
            async with self.session.request(
                method,
                url,
                json=data if method.upper() == "POST" else None,
                params=data if method.upper() == "GET" else None
            ) as response:
                status_code = response.status
                
                # Handle HTTP errors
                if response.status >= 400:
                    response_text = await response.text()
                    if response.status == 400:
                        raise InvalidCommandError(f"Invalid command: {response_text}")
                    else:
                        raise EndpointError(f"HTTP error {response.status}: {response_text}")
                
                result = await response.json()
                
        except asyncio.TimeoutError as e:
            error_msg = f"Async timeout: {str(e)}"
            raise TimeoutError(f"Async request to {url} timed out")
            
        except aiohttp.ClientError as e:
            error_msg = f"Async client error: {str(e)}"
            raise EndpointError(error_msg)
            
        except Exception as e:
            error_msg = f"Unexpected async error: {str(e)}"
            raise EndpointError(error_msg)
            
        finally:
            # Release semaphore
            self._request_semaphore.release()
            
            # Record metrics if enabled
            if self.metrics_collector:
                duration = time.time() - start_time
                metrics = RequestMetrics(
                    timestamp=start_time,
                    endpoint=endpoint,
                    method=method.upper(),
                    duration=duration,
                    status_code=status_code,
                    error=error_msg
                )
                # Use asyncio.create_task to avoid blocking
                asyncio.create_task(self.metrics_collector.record_request_async(metrics))
        
        return result
    
    async def reset(self, scenario: Optional[str] = None, seed: Optional[int] = None) -> Tuple[SimulationState, float]:
        """Async reset ARZ simulation"""
        data = {
            "scenario": scenario or "default",
            "seed": seed,
            "dt_sim": self.config.dt_sim
        }
        
        logger.info(f"Async resetting simulation with scenario={scenario}, seed={seed}")
        result = await self._make_request("POST", "reset", data)
        
        state = SimulationState(
            timestamp=result["timestamp"],
            branches=result["branches"],
            phase_id=result.get("phase_id")
        )
        
        return state, result["timestamp"]
    
    async def set_signal(self, signal_plan: Dict[str, Any]) -> bool:
        """Async set traffic signal plan"""
        logger.debug(f"Async setting signal plan: {signal_plan}")
        result = await self._make_request("POST", "signals", signal_plan)
        return result.get("success", False)
    
    async def step(self, dt: float, repeat_k: int = 1) -> Tuple[SimulationState, float]:
        """Async advance simulation"""
        data = {
            "dt": dt,
            "repeat": repeat_k
        }
        
        result = await self._make_request("POST", "step", data)
        
        state = SimulationState(
            timestamp=result["timestamp"],
            branches=result["branches"],
            phase_id=result.get("phase_id")
        )
        
        return state, result["timestamp"]
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Async get simulation metrics"""
        sim_metrics = await self._make_request("GET", "metrics")
        
        if self.metrics_collector:
            performance_stats = await self.metrics_collector.get_stats_async()
            sim_metrics["http_performance"] = performance_stats
        
        return sim_metrics
    
    async def health(self) -> Dict[str, Any]:
        """Async health check"""
        start_time = time.time()
        result = await self._make_request("GET", "health")
        latency = time.time() - start_time
        
        result["latency"] = latency
        
        # Add connection info
        if self.session and hasattr(self.session, 'connector'):
            connector = self.session.connector
            result["connection_pool"] = {
                "limit": connector.limit,
                "limit_per_host": connector.limit_per_host,
                "connections_count": len(connector._conns)
            }
        
        return result
    
    async def close(self):
        """Close async session"""
        if self.session and not self.session.closed:
            await self.session.close()
            logger.info("Async HTTP session closed")


class HybridEndpointClient:
    """Hybrid client that supports both sync and async operations"""
    
    def __init__(self, config: AsyncEndpointConfig):
        self.config = config
        self.async_client = AsyncARZEndpointClient(config)
        self._event_loop = None
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
    
    def _get_or_create_event_loop(self):
        """Get or create event loop for sync operations"""
        try:
            loop = asyncio.get_running_loop()
            return loop
        except RuntimeError:
            # No running loop, create one
            if self._event_loop is None or self._event_loop.is_closed():
                self._event_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self._event_loop)
            return self._event_loop
    
    def _run_async(self, coro):
        """Run async function in sync context"""
        loop = self._get_or_create_event_loop()
        
        if loop.is_running():
            # We're already in an async context, use executor
            future = concurrent.futures.Future()
            
            async def _run():
                try:
                    result = await coro
                    future.set_result(result)
                except Exception as e:
                    future.set_exception(e)
            
            asyncio.create_task(_run())
            return future.result(timeout=self.config.total_timeout)
        else:
            # No running loop, we can use run_until_complete
            return loop.run_until_complete(coro)
    
    async def async_reset(self, scenario: Optional[str] = None, seed: Optional[int] = None) -> Tuple[SimulationState, float]:
        """Async reset (preferred for async contexts)"""
        return await self.async_client.reset(scenario, seed)
    
    def reset(self, scenario: Optional[str] = None, seed: Optional[int] = None) -> Tuple[SimulationState, float]:
        """Sync reset (for compatibility)"""
        return self._run_async(self.async_client.reset(scenario, seed))
    
    async def async_step(self, dt: float, repeat_k: int = 1) -> Tuple[SimulationState, float]:
        """Async step (preferred for async contexts)"""
        return await self.async_client.step(dt, repeat_k)
    
    def step(self, dt: float, repeat_k: int = 1) -> Tuple[SimulationState, float]:
        """Sync step (for compatibility)"""
        return self._run_async(self.async_client.step(dt, repeat_k))
    
    async def async_set_signal(self, signal_plan: Dict[str, Any]) -> bool:
        """Async set signal (preferred for async contexts)"""
        return await self.async_client.set_signal(signal_plan)
    
    def set_signal(self, signal_plan: Dict[str, Any]) -> bool:
        """Sync set signal (for compatibility)"""
        return self._run_async(self.async_client.set_signal(signal_plan))
    
    async def async_get_metrics(self) -> Dict[str, Any]:
        """Async get metrics (preferred for async contexts)"""
        return await self.async_client.get_metrics()
    
    def get_metrics(self) -> Dict[str, Any]:
        """Sync get metrics (for compatibility)"""
        return self._run_async(self.async_client.get_metrics())
    
    async def async_health(self) -> Dict[str, Any]:
        """Async health check (preferred for async contexts)"""
        return await self.async_client.health()
    
    def health(self) -> Dict[str, Any]:
        """Sync health check (for compatibility)"""
        return self._run_async(self.async_client.health())
    
    def close(self):
        """Close both sync and async resources"""
        # Close async client
        if self.async_client:
            if self._event_loop and not self._event_loop.is_closed():
                self._event_loop.run_until_complete(self.async_client.close())
        
        # Close executor
        if self._executor:
            self._executor.shutdown(wait=True)
        
        # Close event loop
        if self._event_loop and not self._event_loop.is_closed():
            self._event_loop.close()


# Context manager for async client
@asynccontextmanager
async def async_endpoint_client(config: AsyncEndpointConfig) -> AsyncContextManager[AsyncARZEndpointClient]:
    """Async context manager for ARZ endpoint client"""
    client = AsyncARZEndpointClient(config)
    try:
        await client.connect()
        yield client
    finally:
        await client.close()


# Factory function for hybrid client
def create_hybrid_client(config: AsyncEndpointConfig) -> HybridEndpointClient:
    """Create a hybrid client supporting both sync and async operations"""
    return HybridEndpointClient(config)