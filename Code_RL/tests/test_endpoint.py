"""
Unit tests for the endpoint client
"""

import pytest
import time
from unittest.mock import Mock, patch
from src.endpoint.client import (
    HTTPEndpointClient, MockEndpointClient, EndpointConfig,
    create_endpoint_client, SimulationState, EndpointError, TimeoutError
)


class TestEndpointConfig:
    def test_default_config(self):
        config = EndpointConfig()
        assert config.protocol == "http"
        assert config.host == "localhost"
        assert config.port == 8080
        assert config.dt_sim == 0.5
    
    def test_custom_config(self):
        config = EndpointConfig(
            protocol="mock",
            host="127.0.0.1",
            port=9000,
            dt_sim=1.0
        )
        assert config.protocol == "mock"
        assert config.host == "127.0.0.1"
        assert config.port == 9000
        assert config.dt_sim == 1.0


class TestMockEndpointClient:
    def setup_method(self):
        self.config = EndpointConfig(protocol="mock")
        self.client = MockEndpointClient(self.config)
    
    def test_initialization(self):
        assert self.client.current_time == 0.0
        assert self.client.phase_id == 0
        assert len(self.client.branches) == 8
    
    def test_reset(self):
        state, timestamp = self.client.reset(scenario="test", seed=123)
        
        assert isinstance(state, SimulationState)
        assert state.timestamp == 0.0
        assert timestamp == 0.0
        assert len(state.branches) == 8
        
        # Check branch data structure
        for branch_id, data in state.branches.items():
            assert "rho_m" in data
            assert "v_m" in data
            assert "rho_c" in data
            assert "v_c" in data
            assert "queue_len" in data
            assert "flow" in data
            assert data["rho_m"] >= 0
            assert data["v_m"] >= 0
    
    def test_set_signal(self):
        signal_plan = {"phase_id": 1}
        result = self.client.set_signal(signal_plan)
        
        assert result is True
        assert self.client.phase_id == 1
    
    def test_step(self):
        # Reset first
        self.client.reset()
        
        state, timestamp = self.client.step(dt=1.0, repeat_k=2)
        
        assert isinstance(state, SimulationState)
        assert timestamp == 2.0  # dt * repeat_k
        assert self.client.current_time == 2.0
    
    def test_get_metrics(self):
        metrics = self.client.get_metrics()
        
        assert isinstance(metrics, dict)
        assert "avg_wait_time" in metrics
        assert "max_queue_length" in metrics
        assert "throughput" in metrics
        assert "num_stops" in metrics
    
    def test_health(self):
        health = self.client.health()
        
        assert isinstance(health, dict)
        assert health["status"] == "healthy"
        assert "latency" in health
        assert "timestamp" in health


class TestHTTPEndpointClient:
    def setup_method(self):
        self.config = EndpointConfig(protocol="http")
        self.client = HTTPEndpointClient(self.config)
    
    @patch('requests.Session.post')
    def test_successful_reset(self, mock_post):
        # Mock successful response
        mock_response = Mock()
        mock_response.json.return_value = {
            "timestamp": 0.0,
            "branches": {
                "north_in": {
                    "rho_m": 100.0, "v_m": 30.0, "rho_c": 50.0, 
                    "v_c": 40.0, "queue_len": 25.0, "flow": 150.0
                }
            },
            "phase_id": 0
        }
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response
        
        state, timestamp = self.client.reset(scenario="test", seed=42)
        
        assert isinstance(state, SimulationState)
        assert state.timestamp == 0.0
        assert timestamp == 0.0
        assert "north_in" in state.branches
        
        # Verify API call
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert "reset" in call_args[0][0]
        assert call_args[1]["json"]["scenario"] == "test"
        assert call_args[1]["json"]["seed"] == 42
    
    @patch('requests.Session.post')
    def test_timeout_handling(self, mock_post):
        import requests
        mock_post.side_effect = requests.exceptions.Timeout()
        
        with pytest.raises(TimeoutError):
            self.client.reset()
    
    @patch('requests.Session.post')
    def test_http_error_handling(self, mock_post):
        import requests
        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.text = "Bad Request"
        error = requests.exceptions.HTTPError()
        error.response = mock_response
        mock_post.side_effect = error
        
        with pytest.raises(Exception):  # Should raise InvalidCommandError
            self.client.reset()


class TestEndpointFactory:
    def test_create_http_client(self):
        config = EndpointConfig(protocol="http")
        client = create_endpoint_client(config)
        assert isinstance(client, HTTPEndpointClient)
    
    def test_create_mock_client(self):
        config = EndpointConfig(protocol="mock")
        client = create_endpoint_client(config)
        assert isinstance(client, MockEndpointClient)
    
    def test_unsupported_protocol(self):
        config = EndpointConfig(protocol="unsupported")
        with pytest.raises(ValueError):
            create_endpoint_client(config)


if __name__ == "__main__":
    pytest.main([__file__])
