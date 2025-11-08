"""
Performance Benchmarking and Monitoring Tools for RL-Simulator Communication

This module provides comprehensive tools for benchmarking HTTP client performance,
monitoring real-time metrics, and generating performance reports.
"""

import time
import asyncio
import logging
import statistics
import json
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import concurrent.futures
from pathlib import Path

from .client import ARZEndpointClient, EndpointConfig
from .optimized_client import OptimizedHTTPEndpointClient, OptimizedEndpointConfig
from .async_client import AsyncARZEndpointClient, AsyncEndpointConfig, async_endpoint_client

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkConfig:
    """Configuration for performance benchmarks"""
    
    # Test parameters
    duration_seconds: float = 60.0
    num_requests: Optional[int] = None  # If set, overrides duration
    concurrent_requests: int = 1
    warmup_requests: int = 10
    
    # Request patterns
    reset_probability: float = 0.05
    signal_probability: float = 0.10
    step_probability: float = 0.70
    metrics_probability: float = 0.10
    health_probability: float = 0.05
    
    # Output configuration
    save_results: bool = True
    output_dir: str = "benchmark_results"
    include_detailed_logs: bool = False


@dataclass
class BenchmarkResult:
    """Results from a performance benchmark"""
    
    # Test configuration
    config: BenchmarkConfig
    client_type: str
    timestamp: datetime
    
    # Overall statistics
    total_requests: int
    total_duration: float
    requests_per_second: float
    
    # Latency statistics
    avg_latency: float
    min_latency: float
    max_latency: float
    p50_latency: float
    p95_latency: float
    p99_latency: float
    
    # Error statistics
    total_errors: int
    error_rate: float
    timeout_errors: int
    connection_errors: int
    
    # Endpoint-specific stats
    endpoint_stats: Dict[str, Dict[str, Any]]
    
    # Resource utilization
    peak_memory_mb: Optional[float] = None
    cpu_usage_percent: Optional[float] = None


class PerformanceBenchmark:
    """Performance benchmarking tool for endpoint clients"""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.results: List[Dict[str, Any]] = []
        self.start_time: Optional[float] = None
        self.endpoint_counters = {
            "reset": 0, "step": 0, "signals": 0, "metrics": 0, "health": 0
        }
    
    def _select_endpoint(self) -> str:
        """Select endpoint based on configured probabilities"""
        import random
        
        rand = random.random()
        cumulative = 0.0
        
        endpoints = [
            ("reset", self.config.reset_probability),
            ("signals", self.config.signal_probability),
            ("step", self.config.step_probability),
            ("metrics", self.config.metrics_probability),
            ("health", self.config.health_probability)
        ]
        
        for endpoint, probability in endpoints:
            cumulative += probability
            if rand <= cumulative:
                return endpoint
        
        return "step"  # Default fallback
    
    async def _make_benchmark_request(self, client: Union[ARZEndpointClient, AsyncARZEndpointClient], endpoint: str) -> Dict[str, Any]:
        """Make a single benchmark request"""
        start_time = time.time()
        error = None
        
        try:
            if endpoint == "reset":
                if hasattr(client, 'async_reset'):
                    await client.async_reset()
                else:
                    client.reset()
                    
            elif endpoint == "step":
                if hasattr(client, 'async_step'):
                    await client.async_step(0.5, 1)
                else:
                    client.step(0.5, 1)
                    
            elif endpoint == "signals":
                signal_plan = {"phase_id": 1, "duration": 30}
                if hasattr(client, 'async_set_signal'):
                    await client.async_set_signal(signal_plan)
                else:
                    client.set_signal(signal_plan)
                    
            elif endpoint == "metrics":
                if hasattr(client, 'async_get_metrics'):
                    await client.async_get_metrics()
                else:
                    client.get_metrics()
                    
            elif endpoint == "health":
                if hasattr(client, 'async_health'):
                    await client.async_health()
                else:
                    client.health()
                    
        except Exception as e:
            error = str(e)
        
        duration = time.time() - start_time
        self.endpoint_counters[endpoint] += 1
        
        return {
            "timestamp": start_time,
            "endpoint": endpoint,
            "duration": duration,
            "error": error
        }
    
    async def _run_concurrent_benchmark(self, client: Union[ARZEndpointClient, AsyncARZEndpointClient], semaphore: asyncio.Semaphore) -> List[Dict[str, Any]]:
        """Run benchmark requests with concurrency control"""
        results = []
        self.start_time = time.time()
        request_count = 0
        
        # Warmup requests
        for _ in range(self.config.warmup_requests):
            async with semaphore:
                endpoint = self._select_endpoint()
                await self._make_benchmark_request(client, endpoint)
        
        # Reset counters after warmup
        self.endpoint_counters = {k: 0 for k in self.endpoint_counters}
        
        # Main benchmark loop
        while True:
            current_time = time.time()
            
            # Check termination conditions
            if self.config.num_requests and request_count >= self.config.num_requests:
                break
            if not self.config.num_requests and (current_time - self.start_time) >= self.config.duration_seconds:
                break
            
            async with semaphore:
                endpoint = self._select_endpoint()
                result = await self._make_benchmark_request(client, endpoint)
                results.append(result)
                request_count += 1
        
        return results
    
    async def benchmark_async_client(self, config: AsyncEndpointConfig) -> BenchmarkResult:
        """Benchmark async client performance"""
        logger.info(f"Starting async client benchmark: {self.config.concurrent_requests} concurrent requests")
        
        semaphore = asyncio.Semaphore(self.config.concurrent_requests)
        results = []
        
        async with async_endpoint_client(config) as client:
            # Run concurrent benchmark tasks
            tasks = [
                self._run_concurrent_benchmark(client, semaphore)
                for _ in range(self.config.concurrent_requests)
            ]
            
            task_results = await asyncio.gather(*tasks)
            for task_result in task_results:
                results.extend(task_result)
        
        return self._analyze_results(results, "AsyncClient")
    
    def benchmark_sync_client(self, client: ARZEndpointClient) -> BenchmarkResult:
        """Benchmark synchronous client performance"""
        logger.info(f"Starting sync client benchmark")
        
        results = []
        self.start_time = time.time()
        request_count = 0
        
        # Warmup
        for _ in range(self.config.warmup_requests):
            endpoint = self._select_endpoint()
            asyncio.run(self._make_benchmark_request(client, endpoint))
        
        # Reset counters
        self.endpoint_counters = {k: 0 for k in self.endpoint_counters}
        
        # Main benchmark
        while True:
            current_time = time.time()
            
            if self.config.num_requests and request_count >= self.config.num_requests:
                break
            if not self.config.num_requests and (current_time - self.start_time) >= self.config.duration_seconds:
                break
            
            endpoint = self._select_endpoint()
            result = asyncio.run(self._make_benchmark_request(client, endpoint))
            results.append(result)
            request_count += 1
        
        return self._analyze_results(results, "SyncClient")
    
    def _analyze_results(self, results: List[Dict[str, Any]], client_type: str) -> BenchmarkResult:
        """Analyze benchmark results and generate statistics"""
        if not results:
            raise ValueError("No benchmark results to analyze")
        
        # Filter successful requests for latency analysis
        successful_results = [r for r in results if r["error"] is None]
        durations = [r["duration"] for r in successful_results]
        
        # Calculate overall statistics
        total_duration = time.time() - self.start_time if self.start_time else 0.0
        total_requests = len(results)
        requests_per_second = total_requests / total_duration if total_duration > 0 else 0.0
        
        # Calculate latency statistics
        if durations:
            avg_latency = statistics.mean(durations)
            min_latency = min(durations)
            max_latency = max(durations)
            p50_latency = statistics.median(durations)
            p95_latency = self._percentile(durations, 95)
            p99_latency = self._percentile(durations, 99)
        else:
            avg_latency = min_latency = max_latency = p50_latency = p95_latency = p99_latency = 0.0
        
        # Calculate error statistics
        errors = [r for r in results if r["error"] is not None]
        total_errors = len(errors)
        error_rate = total_errors / total_requests if total_requests > 0 else 0.0
        
        timeout_errors = len([e for e in errors if "timeout" in e["error"].lower()])
        connection_errors = len([e for e in errors if "connection" in e["error"].lower()])
        
        # Calculate endpoint-specific statistics
        endpoint_stats = {}
        for endpoint in self.endpoint_counters:
            endpoint_results = [r for r in results if r["endpoint"] == endpoint]
            endpoint_successes = [r for r in endpoint_results if r["error"] is None]
            endpoint_durations = [r["duration"] for r in endpoint_successes]
            
            if endpoint_results:
                endpoint_stats[endpoint] = {
                    "total_requests": len(endpoint_results),
                    "successful_requests": len(endpoint_successes),
                    "error_rate": (len(endpoint_results) - len(endpoint_successes)) / len(endpoint_results),
                    "avg_latency": statistics.mean(endpoint_durations) if endpoint_durations else 0.0,
                    "p95_latency": self._percentile(endpoint_durations, 95) if endpoint_durations else 0.0
                }
        
        result = BenchmarkResult(
            config=self.config,
            client_type=client_type,
            timestamp=datetime.now(),
            total_requests=total_requests,
            total_duration=total_duration,
            requests_per_second=requests_per_second,
            avg_latency=avg_latency,
            min_latency=min_latency,
            max_latency=max_latency,
            p50_latency=p50_latency,
            p95_latency=p95_latency,
            p99_latency=p99_latency,
            total_errors=total_errors,
            error_rate=error_rate,
            timeout_errors=timeout_errors,
            connection_errors=connection_errors,
            endpoint_stats=endpoint_stats
        )
        
        if self.config.save_results:
            self._save_results(result, results if self.config.include_detailed_logs else None)
        
        return result
    
    @staticmethod
    def _percentile(data: List[float], percentile: float) -> float:
        """Calculate percentile of a dataset"""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        index = int((percentile / 100.0) * len(sorted_data))
        return sorted_data[min(index, len(sorted_data) - 1)]
    
    def _save_results(self, result: BenchmarkResult, detailed_logs: Optional[List[Dict[str, Any]]] = None):
        """Save benchmark results to file"""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(exist_ok=True)
        
        timestamp_str = result.timestamp.strftime("%Y%m%d_%H%M%S")
        filename = f"benchmark_{result.client_type}_{timestamp_str}.json"
        
        output_data = {
            "summary": asdict(result),
            "detailed_logs": detailed_logs
        }
        
        output_path = output_dir / filename
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2, default=str)
        
        logger.info(f"Benchmark results saved to {output_path}")


class RealTimeMonitor:
    """Real-time performance monitoring for RL training"""
    
    def __init__(self, window_size: int = 100, update_interval: float = 5.0):
        self.window_size = window_size
        self.update_interval = update_interval
        self.metrics_history: List[Dict[str, Any]] = []
        self.monitoring = False
        self._monitor_task: Optional[asyncio.Task] = None
    
    async def start_monitoring(self, client: Union[ARZEndpointClient, AsyncARZEndpointClient]):
        """Start real-time monitoring"""
        if self.monitoring:
            return
        
        self.monitoring = True
        self._monitor_task = asyncio.create_task(self._monitor_loop(client))
        logger.info("Real-time monitoring started")
    
    async def stop_monitoring(self):
        """Stop real-time monitoring"""
        self.monitoring = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        logger.info("Real-time monitoring stopped")
    
    async def _monitor_loop(self, client: Union[ARZEndpointClient, AsyncARZEndpointClient]):
        """Main monitoring loop"""
        while self.monitoring:
            try:
                # Collect metrics
                start_time = time.time()
                
                if hasattr(client, 'async_health'):
                    health_data = await client.async_health()
                else:
                    health_data = client.health()
                
                collection_time = time.time() - start_time
                
                # Store metrics
                metric_entry = {
                    "timestamp": time.time(),
                    "collection_time": collection_time,
                    "health_data": health_data
                }
                
                self.metrics_history.append(metric_entry)
                
                # Maintain window size
                if len(self.metrics_history) > self.window_size:
                    self.metrics_history = self.metrics_history[-self.window_size:]
                
                await asyncio.sleep(self.update_interval)
                
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                await asyncio.sleep(self.update_interval)
    
    def get_current_stats(self) -> Dict[str, Any]:
        """Get current performance statistics"""
        if not self.metrics_history:
            return {}
        
        recent_metrics = self.metrics_history[-10:]  # Last 10 entries
        collection_times = [m["collection_time"] for m in recent_metrics]
        latencies = [m["health_data"].get("latency", 0) for m in recent_metrics if "health_data" in m]
        
        return {
            "monitoring_window_size": len(self.metrics_history),
            "avg_collection_time": statistics.mean(collection_times) if collection_times else 0,
            "avg_health_latency": statistics.mean(latencies) if latencies else 0,
            "last_update": self.metrics_history[-1]["timestamp"] if self.metrics_history else 0
        }


# Convenience functions for common benchmarking scenarios
async def quick_benchmark_comparison(
    sync_config: OptimizedEndpointConfig,
    async_config: AsyncEndpointConfig,
    duration: float = 30.0
) -> Dict[str, BenchmarkResult]:
    """Quick comparison between sync and async clients"""
    
    benchmark_config = BenchmarkConfig(
        duration_seconds=duration,
        concurrent_requests=5,
        save_results=True
    )
    
    benchmark = PerformanceBenchmark(benchmark_config)
    
    # Benchmark sync client
    from .optimized_client import OptimizedHTTPEndpointClient
    sync_client = OptimizedHTTPEndpointClient(sync_config)
    sync_result = benchmark.benchmark_sync_client(sync_client)
    sync_client.close()
    
    # Reset benchmark for async test
    benchmark = PerformanceBenchmark(benchmark_config)
    async_result = await benchmark.benchmark_async_client(async_config)
    
    return {
        "sync": sync_result,
        "async": async_result
    }


def benchmark_client_configurations(
    configs: List[Tuple[str, Union[OptimizedEndpointConfig, AsyncEndpointConfig]]],
    duration: float = 60.0
) -> Dict[str, BenchmarkResult]:
    """Benchmark multiple client configurations"""
    
    results = {}
    benchmark_config = BenchmarkConfig(
        duration_seconds=duration,
        save_results=True
    )
    
    for config_name, endpoint_config in configs:
        logger.info(f"Benchmarking configuration: {config_name}")
        
        benchmark = PerformanceBenchmark(benchmark_config)
        
        if isinstance(endpoint_config, AsyncEndpointConfig):
            result = asyncio.run(benchmark.benchmark_async_client(endpoint_config))
        else:
            from .optimized_client import OptimizedHTTPEndpointClient
            client = OptimizedHTTPEndpointClient(endpoint_config)
            result = benchmark.benchmark_sync_client(client)
            client.close()
        
        results[config_name] = result
    
    return results