"""
ARZ Simulator Endpoint Client

Provides interface to external ARZ traffic simulator for the RL environment.
Implements the contract specified in the design document.
"""

import time
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
import requests
import json

logger = logging.getLogger(__name__)


@dataclass
class SimulationState:
    """State returned by ARZ simulator"""
    timestamp: float
    branches: Dict[str, Dict[str, Any]]  # branch_id -> {rho_m, v_m, rho_c, v_c, queue_len, flow}
    phase_id: Optional[int] = None
    

@dataclass 
class EndpointConfig:
    """Configuration for ARZ endpoint"""
    protocol: str = "http"
    host: str = "localhost" 
    port: int = 8080
    base_url: str = "/api/v1/arz"
    dt_sim: float = 0.5
    timeout: float = 30.0
    max_retries: int = 3
    retry_backoff: float = 1.0
    

class EndpointError(Exception):
    """Base exception for endpoint errors"""
    pass


class TimeoutError(EndpointError):
    """Timeout during endpoint communication"""
    pass


class InvalidCommandError(EndpointError):
    """Invalid command sent to endpoint"""
    pass


class ARZEndpointClient(ABC):
    """Abstract base class for ARZ simulator clients"""
    
    @abstractmethod
    def reset(self, scenario: Optional[str] = None, seed: Optional[int] = None) -> Tuple[SimulationState, float]:
        """Reset simulation and return initial state"""
        pass
    
    @abstractmethod
    def set_signal(self, signal_plan: Dict[str, Any]) -> bool:
        """Set traffic signal configuration"""
        pass
    
    @abstractmethod
    def step(self, dt: float, repeat_k: int = 1) -> Tuple[SimulationState, float]:
        """Advance simulation by dt*repeat_k seconds"""
        pass
    
    @abstractmethod
    def get_metrics(self) -> Dict[str, Any]:
        """Get aggregated metrics (if not included in state)"""
        pass
    
    @abstractmethod
    def health(self) -> Dict[str, Any]:
        """Check endpoint health and latency"""
        pass


class HTTPEndpointClient(ARZEndpointClient):
    """HTTP-based ARZ simulator client"""
    
    def __init__(self, config: EndpointConfig):
        self.config = config
        self.base_url = f"http://{config.host}:{config.port}{config.base_url}"
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})
        
    def _make_request(self, method: str, endpoint: str, data: Optional[Dict] = None) -> Dict[str, Any]:
        """Make HTTP request with retry logic"""
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        
        for attempt in range(self.config.max_retries + 1):
            try:
                if method.upper() == "GET":
                    response = self.session.get(url, timeout=self.config.timeout, params=data)
                elif method.upper() == "POST":
                    response = self.session.post(url, timeout=self.config.timeout, json=data)
                else:
                    raise ValueError(f"Unsupported HTTP method: {method}")
                
                response.raise_for_status()
                return response.json()
                
            except requests.exceptions.Timeout:
                if attempt == self.config.max_retries:
                    raise TimeoutError(f"Request to {url} timed out after {self.config.max_retries} retries")
                time.sleep(self.config.retry_backoff * (attempt + 1))
                
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 400:
                    raise InvalidCommandError(f"Invalid command: {e.response.text}")
                raise EndpointError(f"HTTP error {e.response.status_code}: {e.response.text}")
                
            except requests.exceptions.RequestException as e:
                if attempt == self.config.max_retries:
                    raise EndpointError(f"Request failed: {str(e)}")
                time.sleep(self.config.retry_backoff * (attempt + 1))
    
    def reset(self, scenario: Optional[str] = None, seed: Optional[int] = None) -> Tuple[SimulationState, float]:
        """Reset ARZ simulation"""
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
        """Set traffic signal plan"""
        logger.debug(f"Setting signal plan: {signal_plan}")
        result = self._make_request("POST", "signals", signal_plan)
        return result.get("success", False)
    
    def step(self, dt: float, repeat_k: int = 1) -> Tuple[SimulationState, float]:
        """Advance simulation"""
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
        """Get aggregated metrics"""
        return self._make_request("GET", "metrics")
    
    def health(self) -> Dict[str, Any]:
        """Check endpoint health"""
        start_time = time.time()
        result = self._make_request("GET", "health")
        latency = time.time() - start_time
        
        result["latency"] = latency
        return result


class MockEndpointClient(ARZEndpointClient):
    """Mock ARZ client for testing"""
    
    def __init__(self, config: EndpointConfig):
        self.config = config
        self.current_time = 0.0
        self.phase_id = 0
        self.branches = [
            "north_in", "north_out", "south_in", "south_out",
            "east_in", "east_out", "west_in", "west_out"
        ]
        
    def _generate_mock_state(self) -> SimulationState:
        """Generate realistic mock traffic state"""
        import numpy as np
        
        branches_data = {}
        for branch_id in self.branches:
            # Generate realistic traffic densities and velocities
            # Motorcycles: higher density, moderate speed
            rho_m = np.random.uniform(50, 200)  # veh/km
            v_m = np.random.uniform(20, 35)     # km/h
            
            # Cars: lower density, variable speed  
            rho_c = np.random.uniform(20, 100)  # veh/km
            v_c = np.random.uniform(15, 45)     # km/h
            
            # Queue length (correlated with density)
            queue_len = np.random.uniform(0, (rho_m + rho_c) * 0.8)
            
            # Flow (approximate)
            flow = (rho_m * v_m + rho_c * v_c) * 0.01
            
            branches_data[branch_id] = {
                "rho_m": rho_m,
                "v_m": v_m, 
                "rho_c": rho_c,
                "v_c": v_c,
                "queue_len": queue_len,
                "flow": flow
            }
        
        return SimulationState(
            timestamp=self.current_time,
            branches=branches_data,
            phase_id=self.phase_id
        )
    
    def reset(self, scenario: Optional[str] = None, seed: Optional[int] = None) -> Tuple[SimulationState, float]:
        """Reset mock simulation"""
        if seed:
            import numpy as np
            np.random.seed(seed)
            
        self.current_time = 0.0
        self.phase_id = 0
        state = self._generate_mock_state()
        
        logger.info(f"Mock reset: scenario={scenario}, seed={seed}")
        return state, self.current_time
    
    def set_signal(self, signal_plan: Dict[str, Any]) -> bool:
        """Set signal plan in mock"""
        self.phase_id = signal_plan.get("phase_id", self.phase_id)
        return True
    
    def step(self, dt: float, repeat_k: int = 1) -> Tuple[SimulationState, float]:
        """Advance mock simulation"""
        self.current_time += dt * repeat_k
        state = self._generate_mock_state()
        return state, self.current_time
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get mock metrics"""
        return {
            "avg_wait_time": 45.5,
            "max_queue_length": 120.3, 
            "throughput": 850.2,
            "num_stops": 23
        }
    
    def health(self) -> Dict[str, Any]:
        """Mock health check"""
        return {
            "status": "healthy",
            "latency": 0.001,
            "timestamp": time.time()
        }


def create_endpoint_client(config: EndpointConfig) -> ARZEndpointClient:
    """Factory function to create appropriate endpoint client"""
    if config.protocol == "http":
        return HTTPEndpointClient(config)
    elif config.protocol == "mock":
        return MockEndpointClient(config)
    else:
        raise ValueError(f"Unsupported protocol: {config.protocol}")
