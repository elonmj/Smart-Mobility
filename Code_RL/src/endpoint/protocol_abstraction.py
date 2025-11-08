"""
Protocol Abstraction Layer for RL-Simulator Communication

This module provides a unified interface that can transparently switch between
HTTP, WebSocket, and other communication protocols based on configuration and
performance requirements.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, Union, Type
from dataclasses import dataclass
from enum import Enum

from .client import SimulationState, ARZEndpointClient
from .optimized_client import OptimizedHTTPEndpointClient, OptimizedEndpointConfig
from .async_client import AsyncARZEndpointClient, AsyncEndpointConfig, HybridEndpointClient
from .websocket_client import WebSocketARZClient, WebSocketConfig

logger = logging.getLogger(__name__)


class ProtocolType(Enum):
    """Supported communication protocols"""
    HTTP_SYNC = "http_sync"
    HTTP_ASYNC = "http_async"
    HTTP_HYBRID = "http_hybrid"
    WEBSOCKET = "websocket"
    AUTO = "auto"  # Automatic selection based on requirements


class PerformanceProfile(Enum):
    """Performance profile for protocol selection"""
    LOW_LATENCY = "low_latency"      # <50ms: WebSocket preferred
    BALANCED = "balanced"            # <200ms: HTTP async preferred
    HIGH_THROUGHPUT = "throughput"   # HTTP with connection pooling
    RELIABLE = "reliable"            # HTTP sync with retries
    DEVELOPMENT = "development"      # Mock or local protocols


@dataclass
class ProtocolSelectionCriteria:
    """Criteria for automatic protocol selection"""
    max_latency_ms: float = 200.0
    min_throughput_rps: float = 10.0
    reliability_requirement: float = 0.99  # 99% success rate
    concurrent_agents: int = 1
    real_time_requirement: bool = False
    development_mode: bool = False


@dataclass
class UnifiedConfig:
    """Unified configuration supporting all protocols"""
    
    # Protocol selection
    protocol_type: ProtocolType = ProtocolType.AUTO
    performance_profile: PerformanceProfile = PerformanceProfile.BALANCED
    selection_criteria: ProtocolSelectionCriteria = None
    
    # Common connection settings
    host: str = "localhost"
    http_port: int = 8080
    websocket_port: int = 8081
    ssl_enabled: bool = False
    
    # Common timeout settings
    connect_timeout: float = 10.0
    response_timeout: float = 30.0
    
    # Performance settings
    enable_metrics: bool = True
    metrics_window_size: int = 100
    
    # Fallback configuration
    enable_fallback: bool = True
    fallback_protocols: list = None  # List of ProtocolType for fallback chain
    
    # Protocol-specific configs (will be generated automatically)
    http_config: Optional[OptimizedEndpointConfig] = None
    async_config: Optional[AsyncEndpointConfig] = None
    websocket_config: Optional[WebSocketConfig] = None
    
    def __post_init__(self):
        if self.selection_criteria is None:
            self.selection_criteria = ProtocolSelectionCriteria()
        
        if self.fallback_protocols is None:
            self.fallback_protocols = [
                ProtocolType.HTTP_ASYNC,
                ProtocolType.HTTP_SYNC,
                ProtocolType.WEBSOCKET
            ]


class ProtocolAdapter(ABC):
    """Abstract base class for protocol adapters"""
    
    @abstractmethod
    async def connect(self) -> bool:
        """Establish connection"""
        pass
    
    @abstractmethod
    async def disconnect(self):
        """Close connection"""
        pass
    
    @abstractmethod
    async def reset(self, scenario: Optional[str] = None, seed: Optional[int] = None) -> Tuple[SimulationState, float]:
        """Reset simulation"""
        pass
    
    @abstractmethod
    async def step(self, dt: float, repeat_k: int = 1) -> Tuple[SimulationState, float]:
        """Advance simulation"""
        pass
    
    @abstractmethod
    async def set_signal(self, signal_plan: Dict[str, Any]) -> bool:
        """Set traffic signal"""
        pass
    
    @abstractmethod
    async def get_metrics(self) -> Dict[str, Any]:
        """Get simulation metrics"""
        pass
    
    @abstractmethod
    async def health(self) -> Dict[str, Any]:
        """Health check"""
        pass
    
    @abstractmethod
    def get_protocol_info(self) -> Dict[str, Any]:
        """Get protocol-specific information"""
        pass


class HTTPSyncAdapter(ProtocolAdapter):
    """Adapter for synchronous HTTP client"""
    
    def __init__(self, config: OptimizedEndpointConfig):
        self.config = config
        self.client = OptimizedHTTPEndpointClient(config)
        self.executor = None
    
    async def connect(self) -> bool:
        """HTTP doesn't require explicit connection"""
        return True
    
    async def disconnect(self):
        """Close HTTP client"""
        if self.client:
            self.client.close()
    
    async def _run_sync(self, func, *args, **kwargs):
        """Run synchronous function in executor"""
        if self.executor is None:
            import concurrent.futures
            self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, func, *args, **kwargs)
    
    async def reset(self, scenario: Optional[str] = None, seed: Optional[int] = None) -> Tuple[SimulationState, float]:
        return await self._run_sync(self.client.reset, scenario, seed)
    
    async def step(self, dt: float, repeat_k: int = 1) -> Tuple[SimulationState, float]:
        return await self._run_sync(self.client.step, dt, repeat_k)
    
    async def set_signal(self, signal_plan: Dict[str, Any]) -> bool:
        return await self._run_sync(self.client.set_signal, signal_plan)
    
    async def get_metrics(self) -> Dict[str, Any]:
        return await self._run_sync(self.client.get_metrics)
    
    async def health(self) -> Dict[str, Any]:
        return await self._run_sync(self.client.health)
    
    def get_protocol_info(self) -> Dict[str, Any]:
        return {
            "protocol": "HTTP/1.1",
            "type": "synchronous",
            "connection_pooling": True,
            "base_url": self.client.base_url
        }


class HTTPAsyncAdapter(ProtocolAdapter):
    """Adapter for asynchronous HTTP client"""
    
    def __init__(self, config: AsyncEndpointConfig):
        self.config = config
        self.client = AsyncARZEndpointClient(config)
    
    async def connect(self) -> bool:
        await self.client.connect()
        return True
    
    async def disconnect(self):
        await self.client.close()
    
    async def reset(self, scenario: Optional[str] = None, seed: Optional[int] = None) -> Tuple[SimulationState, float]:
        return await self.client.reset(scenario, seed)
    
    async def step(self, dt: float, repeat_k: int = 1) -> Tuple[SimulationState, float]:
        return await self.client.step(dt, repeat_k)
    
    async def set_signal(self, signal_plan: Dict[str, Any]) -> bool:
        return await self.client.set_signal(signal_plan)
    
    async def get_metrics(self) -> Dict[str, Any]:
        return await self.client.get_metrics()
    
    async def health(self) -> Dict[str, Any]:
        return await self.client.health()
    
    def get_protocol_info(self) -> Dict[str, Any]:
        return {
            "protocol": "HTTP/1.1",
            "type": "asynchronous", 
            "connection_pooling": True,
            "concurrent_requests": self.config.max_concurrent_requests,
            "base_url": self.client.base_url
        }


class WebSocketAdapter(ProtocolAdapter):
    """Adapter for WebSocket client"""
    
    def __init__(self, config: WebSocketConfig):
        self.config = config
        self.client = WebSocketARZClient(config)
    
    async def connect(self) -> bool:
        return await self.client.connect()
    
    async def disconnect(self):
        await self.client.disconnect()
    
    async def reset(self, scenario: Optional[str] = None, seed: Optional[int] = None) -> Tuple[SimulationState, float]:
        return await self.client.reset(scenario, seed)
    
    async def step(self, dt: float, repeat_k: int = 1) -> Tuple[SimulationState, float]:
        return await self.client.step(dt, repeat_k)
    
    async def set_signal(self, signal_plan: Dict[str, Any]) -> bool:
        return await self.client.set_signal(signal_plan)
    
    async def get_metrics(self) -> Dict[str, Any]:
        return await self.client.get_metrics()
    
    async def health(self) -> Dict[str, Any]:
        return await self.client.health()
    
    def get_protocol_info(self) -> Dict[str, Any]:
        return {
            "protocol": "WebSocket",
            "type": "bidirectional",
            "compression": self.config.compression,
            "heartbeat": self.config.enable_heartbeat,
            "url": self.client.url
        }


class ProtocolSelector:
    """Intelligent protocol selector based on requirements"""
    
    @staticmethod
    def select_protocol(config: UnifiedConfig) -> ProtocolType:
        """Select optimal protocol based on configuration and criteria"""
        
        if config.protocol_type != ProtocolType.AUTO:
            return config.protocol_type
        
        criteria = config.selection_criteria
        profile = config.performance_profile
        
        # Development mode - prefer HTTP sync for simplicity
        if criteria.development_mode or profile == PerformanceProfile.DEVELOPMENT:
            return ProtocolType.HTTP_SYNC
        
        # Real-time requirement - prefer WebSocket
        if criteria.real_time_requirement or profile == PerformanceProfile.LOW_LATENCY:
            if criteria.max_latency_ms < 50:
                return ProtocolType.WEBSOCKET
        
        # High concurrency - prefer async HTTP
        if criteria.concurrent_agents > 5 or profile == PerformanceProfile.HIGH_THROUGHPUT:
            return ProtocolType.HTTP_ASYNC
        
        # Reliability requirement - prefer HTTP sync with retries
        if criteria.reliability_requirement > 0.99 or profile == PerformanceProfile.RELIABLE:
            return ProtocolType.HTTP_SYNC
        
        # Balanced default - async HTTP
        return ProtocolType.HTTP_ASYNC
    
    @staticmethod
    def generate_protocol_configs(config: UnifiedConfig) -> Dict[str, Any]:
        """Generate protocol-specific configurations"""
        configs = {}
        
        # HTTP Sync Configuration
        configs['http_sync'] = OptimizedEndpointConfig(
            host=config.host,
            port=config.http_port,
            timeout=config.response_timeout,
            connect_timeout=config.connect_timeout,
            enable_metrics=config.enable_metrics,
            metrics_window_size=config.metrics_window_size
        )
        
        # HTTP Async Configuration
        configs['http_async'] = AsyncEndpointConfig(
            host=config.host,
            port=config.http_port,
            total_timeout=config.response_timeout,
            sock_connect_timeout=config.connect_timeout,
            enable_metrics=config.enable_metrics,
            metrics_window_size=config.metrics_window_size,
            max_concurrent_requests=max(5, config.selection_criteria.concurrent_agents)
        )
        
        # WebSocket Configuration
        configs['websocket'] = WebSocketConfig(
            host=config.host,
            port=config.websocket_port,
            protocol="wss" if config.ssl_enabled else "ws",
            connect_timeout=config.connect_timeout,
            response_timeout=config.response_timeout,
            compression="deflate" if config.selection_criteria.max_latency_ms > 100 else None
        )
        
        return configs


class UnifiedARZClient:
    """Unified client that abstracts protocol details"""
    
    def __init__(self, config: UnifiedConfig):
        self.config = config
        self.current_adapter: Optional[ProtocolAdapter] = None
        self.current_protocol: Optional[ProtocolType] = None
        self.protocol_configs = ProtocolSelector.generate_protocol_configs(config)
        
        # Performance tracking
        self.connection_attempts = {}
        self.performance_history = {}
        
        logger.info(f"Initialized unified client with profile: {config.performance_profile}")
    
    async def connect(self) -> bool:
        """Connect using optimal protocol"""
        selected_protocol = ProtocolSelector.select_protocol(self.config)
        
        if await self._connect_protocol(selected_protocol):
            return True
        
        # Try fallback protocols if enabled
        if self.config.enable_fallback:
            for fallback_protocol in self.config.fallback_protocols:
                if fallback_protocol != selected_protocol:
                    logger.warning(f"Trying fallback protocol: {fallback_protocol}")
                    if await self._connect_protocol(fallback_protocol):
                        return True
        
        logger.error("All protocol connection attempts failed")
        return False
    
    async def _connect_protocol(self, protocol_type: ProtocolType) -> bool:
        """Connect using specific protocol"""
        try:
            # Disconnect current adapter if needed
            if self.current_adapter:
                await self.current_adapter.disconnect()
            
            # Create new adapter
            adapter = self._create_adapter(protocol_type)
            
            # Attempt connection
            if await adapter.connect():
                self.current_adapter = adapter
                self.current_protocol = protocol_type
                
                self.connection_attempts[protocol_type.value] = self.connection_attempts.get(protocol_type.value, 0) + 1
                
                logger.info(f"Successfully connected using {protocol_type.value}")
                return True
            
        except Exception as e:
            logger.error(f"Failed to connect using {protocol_type.value}: {e}")
        
        return False
    
    def _create_adapter(self, protocol_type: ProtocolType) -> ProtocolAdapter:
        """Create protocol adapter instance"""
        if protocol_type == ProtocolType.HTTP_SYNC:
            return HTTPSyncAdapter(self.protocol_configs['http_sync'])
        elif protocol_type == ProtocolType.HTTP_ASYNC:
            return HTTPAsyncAdapter(self.protocol_configs['http_async'])
        elif protocol_type == ProtocolType.WEBSOCKET:
            return WebSocketAdapter(self.protocol_configs['websocket'])
        else:
            raise ValueError(f"Unsupported protocol type: {protocol_type}")
    
    async def disconnect(self):
        """Disconnect current adapter"""
        if self.current_adapter:
            await self.current_adapter.disconnect()
            self.current_adapter = None
            self.current_protocol = None
    
    async def reset(self, scenario: Optional[str] = None, seed: Optional[int] = None) -> Tuple[SimulationState, float]:
        """Reset simulation using current protocol"""
        self._ensure_connected()
        return await self.current_adapter.reset(scenario, seed)
    
    async def step(self, dt: float, repeat_k: int = 1) -> Tuple[SimulationState, float]:
        """Advance simulation using current protocol"""
        self._ensure_connected()
        return await self.current_adapter.step(dt, repeat_k)
    
    async def set_signal(self, signal_plan: Dict[str, Any]) -> bool:
        """Set traffic signal using current protocol"""
        self._ensure_connected()
        return await self.current_adapter.set_signal(signal_plan)
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get simulation metrics using current protocol"""
        self._ensure_connected()
        metrics = await self.current_adapter.get_metrics()
        
        # Add protocol abstraction metrics
        metrics["protocol_info"] = self.get_connection_info()
        
        return metrics
    
    async def health(self) -> Dict[str, Any]:
        """Health check using current protocol"""
        self._ensure_connected()
        health = await self.current_adapter.health()
        
        # Add connection info
        health["protocol_info"] = self.get_connection_info()
        
        return health
    
    def _ensure_connected(self):
        """Ensure client is connected"""
        if not self.current_adapter:
            raise ConnectionError("Client not connected. Call connect() first.")
    
    def get_connection_info(self) -> Dict[str, Any]:
        """Get current connection information"""
        if not self.current_adapter:
            return {"status": "disconnected"}
        
        protocol_info = self.current_adapter.get_protocol_info()
        protocol_info.update({
            "current_protocol": self.current_protocol.value if self.current_protocol else None,
            "connection_attempts": self.connection_attempts,
            "performance_profile": self.config.performance_profile.value
        })
        
        return protocol_info
    
    async def switch_protocol(self, new_protocol: ProtocolType) -> bool:
        """Manually switch to a different protocol"""
        logger.info(f"Switching protocol from {self.current_protocol} to {new_protocol}")
        
        # Disconnect current
        if self.current_adapter:
            await self.current_adapter.disconnect()
        
        # Connect with new protocol
        return await self._connect_protocol(new_protocol)
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.disconnect()


# Factory functions
def create_unified_client(
    performance_profile: PerformanceProfile = PerformanceProfile.BALANCED,
    host: str = "localhost",
    **kwargs
) -> UnifiedARZClient:
    """Create unified client with simplified configuration"""
    
    config = UnifiedConfig(
        performance_profile=performance_profile,
        host=host,
        **kwargs
    )
    
    return UnifiedARZClient(config)


def create_low_latency_client(host: str = "localhost", **kwargs) -> UnifiedARZClient:
    """Create client optimized for low latency (WebSocket preferred)"""
    criteria = ProtocolSelectionCriteria(
        max_latency_ms=50.0,
        real_time_requirement=True
    )
    
    config = UnifiedConfig(
        performance_profile=PerformanceProfile.LOW_LATENCY,
        selection_criteria=criteria,
        host=host,
        **kwargs
    )
    
    return UnifiedARZClient(config)


def create_development_client(host: str = "localhost", **kwargs) -> UnifiedARZClient:
    """Create client for development (HTTP sync preferred)"""
    criteria = ProtocolSelectionCriteria(
        development_mode=True,
        reliability_requirement=0.95
    )
    
    config = UnifiedConfig(
        performance_profile=PerformanceProfile.DEVELOPMENT,
        selection_criteria=criteria,
        host=host,
        **kwargs
    )
    
    return UnifiedARZClient(config)