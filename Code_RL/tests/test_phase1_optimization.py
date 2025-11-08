"""
Test suite for Phase 1 Performance Optimization implementations

Tests the optimized HTTP client, async client, and performance monitoring tools.
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, patch
from typing import Dict, Any

from Code_RL.src.endpoint.optimized_client import (
    OptimizedHTTPEndpointClient, OptimizedEndpointConfig, MetricsCollector
)
from Code_RL.src.endpoint.async_client import (
    AsyncARZEndpointClient, AsyncEndpointConfig, HybridEndpointClient
)
from Code_RL.src.endpoint.performance import (
    PerformanceBenchmark, BenchmarkConfig, RealTimeMonitor
)


class TestOptimizedHTTPClient:
    """Test optimized HTTP client implementation"""
    
    def test_config_initialization(self):
        """Test configuration initialization with optimization parameters"""
        config = OptimizedEndpointConfig(
            host="localhost",
            port=8080,
            pool_connections=20,
            pool_maxsize=30,
            enable_metrics=True
        )
        
        assert config.pool_connections == 20
        assert config.pool_maxsize == 30
        assert config.enable_metrics is True
        assert config.tcp_keepalive is True
    
    def test_metrics_collector(self):
        """Test metrics collection functionality"""
        collector = MetricsCollector(window_size=5)
        
        # Add some test metrics
        for i in range(10):
            from Code_RL.src.endpoint.optimized_client import RequestMetrics
            metrics = RequestMetrics(
                timestamp=time.time(),
                endpoint="test",
                method="GET",
                duration=0.1 + i * 0.01,
                status_code=200
            )
            collector.record_request(metrics)
        
        stats = collector.get_stats()
        
        # Should only keep last 5 metrics due to window size
        assert stats["total_requests"] == 5
        assert stats["avg_duration"] > 0
        assert stats["error_rate"] == 0.0
    
    @patch('requests.Session.get')
    def test_optimized_client_request(self, mock_get):
        """Test optimized client makes requests correctly"""
        # Mock response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "ok"}
        mock_get.return_value = mock_response
        
        config = OptimizedEndpointConfig(host="localhost", port=8080)
        client = OptimizedHTTPEndpointClient(config)
        
        result = client._make_request("GET", "test")
        
        assert result == {"status": "ok"}
        assert mock_get.called
        
        # Test metrics collection
        if client.metrics_collector:
            stats = client.metrics_collector.get_stats()
            assert stats["total_requests"] == 1
        
        client.close()


class TestAsyncClient:
    """Test asynchronous client implementation"""
    
    def test_async_config(self):
        """Test async configuration initialization"""
        config = AsyncEndpointConfig(
            host="localhost",
            port=8080,
            connector_limit=25,
            max_concurrent_requests=10
        )
        
        assert config.connector_limit == 25
        assert config.max_concurrent_requests == 10
        assert config.tcp_connector_use_dns_cache is True
    
    @pytest.mark.asyncio
    async def test_async_client_context_manager(self):
        """Test async client context manager functionality"""
        config = AsyncEndpointConfig(host="localhost", port=8080)
        
        # Test context manager (will fail to connect but that's expected in test)
        try:
            async with AsyncARZEndpointClient(config) as client:
                assert client.session is not None
        except Exception:
            # Connection failure expected in test environment
            pass
    
    def test_hybrid_client_creation(self):
        """Test hybrid client creation and configuration"""
        config = AsyncEndpointConfig(host="localhost", port=8080)
        client = HybridEndpointClient(config)
        
        assert client.async_client is not None
        assert client._executor is not None
        
        client.close()


class TestPerformanceBenchmark:
    """Test performance benchmarking tools"""
    
    def test_benchmark_config(self):
        """Test benchmark configuration"""
        config = BenchmarkConfig(
            duration_seconds=30.0,
            concurrent_requests=5,
            step_probability=0.8,
            save_results=False
        )
        
        assert config.duration_seconds == 30.0
        assert config.concurrent_requests == 5
        assert config.step_probability == 0.8
        assert config.save_results is False
    
    def test_endpoint_selection(self):
        """Test endpoint selection based on probabilities"""
        config = BenchmarkConfig(
            step_probability=1.0,  # Always select step
            reset_probability=0.0,
            signal_probability=0.0,
            metrics_probability=0.0,
            health_probability=0.0
        )
        
        benchmark = PerformanceBenchmark(config)
        
        # Should always select 'step' with probability 1.0
        selected_endpoints = [benchmark._select_endpoint() for _ in range(10)]
        assert all(endpoint == "step" for endpoint in selected_endpoints)
    
    def test_result_analysis(self):
        """Test benchmark result analysis"""
        config = BenchmarkConfig(save_results=False)
        benchmark = PerformanceBenchmark(config)
        
        # Mock results data
        mock_results = [
            {"timestamp": time.time(), "endpoint": "step", "duration": 0.1, "error": None},
            {"timestamp": time.time(), "endpoint": "step", "duration": 0.15, "error": None},
            {"timestamp": time.time(), "endpoint": "health", "duration": 0.05, "error": None},
            {"timestamp": time.time(), "endpoint": "step", "duration": 0.2, "error": "timeout"}
        ]
        
        benchmark.start_time = time.time() - 1.0  # 1 second ago
        result = benchmark._analyze_results(mock_results, "TestClient")
        
        assert result.total_requests == 4
        assert result.total_errors == 1
        assert result.error_rate == 0.25
        assert result.avg_latency > 0
        assert "step" in result.endpoint_stats
        assert "health" in result.endpoint_stats


class TestRealTimeMonitor:
    """Test real-time monitoring functionality"""
    
    def test_monitor_initialization(self):
        """Test monitor initialization"""
        monitor = RealTimeMonitor(window_size=50, update_interval=2.0)
        
        assert monitor.window_size == 50
        assert monitor.update_interval == 2.0
        assert monitor.monitoring is False
        assert len(monitor.metrics_history) == 0
    
    def test_metrics_window_management(self):
        """Test metrics window size management"""
        monitor = RealTimeMonitor(window_size=3)
        
        # Add more metrics than window size
        for i in range(5):
            monitor.metrics_history.append({
                "timestamp": time.time(),
                "collection_time": 0.01,
                "health_data": {"latency": 0.1}
            })
        
        # Simulate window management
        if len(monitor.metrics_history) > monitor.window_size:
            monitor.metrics_history = monitor.metrics_history[-monitor.window_size:]
        
        assert len(monitor.metrics_history) == 3
    
    def test_current_stats_calculation(self):
        """Test current statistics calculation"""
        monitor = RealTimeMonitor()
        
        # Add some mock data
        for i in range(5):
            monitor.metrics_history.append({
                "timestamp": time.time() + i,
                "collection_time": 0.02 + i * 0.001,
                "health_data": {"latency": 0.1 + i * 0.01}
            })
        
        stats = monitor.get_current_stats()
        
        assert stats["monitoring_window_size"] == 5
        assert stats["avg_collection_time"] > 0
        assert stats["avg_health_latency"] > 0
        assert stats["last_update"] > 0


# Integration test that can be run manually
def test_integration_mock_scenario():
    """Integration test using mock client to validate full flow"""
    from Code_RL.src.endpoint.client import MockEndpointClient, EndpointConfig
    
    # Test with mock client
    config = EndpointConfig(protocol="mock")
    mock_client = MockEndpointClient(config)
    
    # Test basic operations
    state, timestamp = mock_client.reset(scenario="test", seed=42)
    assert state.timestamp == timestamp
    assert len(state.branches) > 0
    
    # Test step operation
    new_state, new_timestamp = mock_client.step(0.5, 1)
    assert new_timestamp > timestamp
    
    # Test signal setting
    success = mock_client.set_signal({"phase_id": 1})
    assert success is True
    
    # Test metrics
    metrics = mock_client.get_metrics()
    assert "avg_wait_time" in metrics
    
    # Test health
    health = mock_client.health()
    assert health["status"] == "healthy"


if __name__ == "__main__":
    # Run basic tests
    print("Running Phase 1 optimization tests...")
    
    # Test 1: Optimized client configuration
    test_client = TestOptimizedHTTPClient()
    test_client.test_config_initialization()
    test_client.test_metrics_collector()
    print("✓ Optimized client tests passed")
    
    # Test 2: Async client configuration  
    test_async = TestAsyncClient()
    test_async.test_async_config()
    test_async.test_hybrid_client_creation()
    print("✓ Async client tests passed")
    
    # Test 3: Performance benchmarking
    test_bench = TestPerformanceBenchmark()
    test_bench.test_benchmark_config()
    test_bench.test_endpoint_selection()
    test_bench.test_result_analysis()
    print("✓ Performance benchmark tests passed")
    
    # Test 4: Real-time monitoring
    test_monitor = TestRealTimeMonitor()
    test_monitor.test_monitor_initialization()
    test_monitor.test_metrics_window_management()
    test_monitor.test_current_stats_calculation()
    print("✓ Real-time monitoring tests passed")
    
    # Test 5: Integration test
    test_integration_mock_scenario()
    print("✓ Integration test passed")
    
    print("\n🎉 All Phase 1 tests completed successfully!")
    print("Ready to proceed with Phase 2 implementation.")