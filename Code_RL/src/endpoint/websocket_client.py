"""
WebSocket Client Implementation for Low-Latency RL-Simulator Communication

This module provides WebSocket-based communication for real-time scenarios
where HTTP request/response latency is too high for effective RL training.
"""

import asyncio
import websockets
import json
import time
import logging
from typing import Dict, Any, Optional, Callable, Union, List
from dataclasses import dataclass, field
from enum import Enum
import concurrent.futures
from contextlib import asynccontextmanager

from .client import SimulationState, EndpointError, TimeoutError, InvalidCommandError
from .optimized_client import RequestMetrics

logger = logging.getLogger(__name__)


class MessageType(Enum):
    """WebSocket message types for RL-simulator communication"""
    RESET = "reset"
    STEP = "step"
    SET_SIGNAL = "set_signal"
    GET_METRICS = "get_metrics"
    HEALTH = "health"
    STATE_UPDATE = "state_update"
    ERROR = "error"
    HEARTBEAT = "heartbeat"


@dataclass
class WebSocketConfig:
    """Configuration for WebSocket client"""
    
    # Connection settings
    host: str = "localhost"
    port: int = 8081
    path: str = "/ws/arz"
    protocol: str = "ws"  # or "wss" for SSL
    
    # WebSocket settings
    ping_interval: float = 20.0
    ping_timeout: float = 10.0
    close_timeout: float = 10.0
    max_size: Optional[int] = 2**20  # 1MB max message size
    
    # Performance settings
    compression: Optional[str] = "deflate"  # None, "deflate", or "gzip"
    enable_heartbeat: bool = True
    heartbeat_interval: float = 30.0
    
    # Timeout settings
    connect_timeout: float = 10.0
    response_timeout: float = 30.0
    
    # Retry settings
    max_reconnect_attempts: int = 5
    reconnect_delay: float = 1.0
    reconnect_backoff: float = 2.0
    
    # Message handling
    message_queue_size: int = 1000
    enable_message_logging: bool = False


@dataclass
class WebSocketMessage:
    """WebSocket message structure"""
    type: MessageType
    data: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    request_id: Optional[str] = None


class WebSocketMetricsCollector:
    """Metrics collector for WebSocket communication"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.metrics: List[RequestMetrics] = []
        self.connection_events: List[Dict[str, Any]] = []
        self._lock = asyncio.Lock()
    
    async def record_message(self, message_type: str, duration: float, success: bool = True, error: Optional[str] = None):
        """Record WebSocket message performance"""
        async with self._lock:
            metrics = RequestMetrics(
                timestamp=time.time(),
                endpoint=message_type,
                method="WS",
                duration=duration,
                status_code=200 if success else 500,
                error=error
            )
            
            self.metrics.append(metrics)
            if len(self.metrics) > self.window_size:
                self.metrics = self.metrics[-self.window_size:]
    
    async def record_connection_event(self, event_type: str, details: Dict[str, Any] = None):
        """Record connection events (connect, disconnect, reconnect)"""
        async with self._lock:
            event = {
                "timestamp": time.time(),
                "event_type": event_type,
                "details": details or {}
            }
            
            self.connection_events.append(event)
            if len(self.connection_events) > self.window_size:
                self.connection_events = self.connection_events[-self.window_size:]
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive WebSocket statistics"""
        async with self._lock:
            if not self.metrics:
                return {"websocket_stats": "no_data"}
            
            durations = [m.duration for m in self.metrics]
            successful = [m for m in self.metrics if m.status_code == 200]
            errors = [m for m in self.metrics if m.status_code != 200]
            
            recent_connections = self.connection_events[-10:] if self.connection_events else []
            
            return {
                "total_messages": len(self.metrics),
                "successful_messages": len(successful),
                "error_rate": len(errors) / len(self.metrics) if self.metrics else 0,
                "avg_latency": sum(durations) / len(durations) if durations else 0,
                "min_latency": min(durations) if durations else 0,
                "max_latency": max(durations) if durations else 0,
                "p95_latency": sorted(durations)[int(len(durations) * 0.95)] if durations else 0,
                "recent_connections": recent_connections,
                "connection_events_count": len(self.connection_events)
            }


class WebSocketARZClient:
    """WebSocket client for ARZ simulator communication"""
    
    def __init__(self, config: WebSocketConfig):
        self.config = config
        self.websocket: Optional[websockets.WebSocketServerProtocol] = None
        self.connected = False
        self.reconnecting = False
        
        # Message handling
        self.pending_requests: Dict[str, asyncio.Future] = {}
        self.message_handlers: Dict[MessageType, Callable] = {}
        self.next_request_id = 1
        
        # Tasks
        self._receive_task: Optional[asyncio.Task] = None
        self._heartbeat_task: Optional[asyncio.Task] = None
        
        # Metrics
        self.metrics_collector = WebSocketMetricsCollector()
        
        # Build WebSocket URL
        protocol = "wss" if self.config.protocol == "wss" else "ws"
        self.url = f"{protocol}://{config.host}:{config.port}{config.path}"
        
        logger.info(f"Initialized WebSocket client for {self.url}")
    
    def _generate_request_id(self) -> str:
        """Generate unique request ID"""
        request_id = str(self.next_request_id)
        self.next_request_id += 1
        return request_id
    
    async def connect(self) -> bool:
        """Establish WebSocket connection"""
        if self.connected:
            return True
        
        try:
            logger.info(f"Connecting to WebSocket: {self.url}")
            
            # Connection parameters
            connect_kwargs = {
                "ping_interval": self.config.ping_interval,
                "ping_timeout": self.config.ping_timeout,
                "close_timeout": self.config.close_timeout,
                "max_size": self.config.max_size,
            }
            
            # Add compression if configured
            if self.config.compression:
                connect_kwargs["compression"] = self.config.compression
            
            # Connect with timeout
            self.websocket = await asyncio.wait_for(
                websockets.connect(self.url, **connect_kwargs),
                timeout=self.config.connect_timeout
            )
            
            self.connected = True
            logger.info("WebSocket connection established")
            
            # Start message handling tasks
            self._receive_task = asyncio.create_task(self._receive_loop())
            
            if self.config.enable_heartbeat:
                self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
            
            await self.metrics_collector.record_connection_event("connected", {"url": self.url})
            return True
            
        except Exception as e:
            logger.error(f"WebSocket connection failed: {e}")
            await self.metrics_collector.record_connection_event("connection_failed", {"error": str(e)})
            return False
    
    async def disconnect(self):
        """Close WebSocket connection"""
        if not self.connected:
            return
        
        logger.info("Disconnecting WebSocket")
        self.connected = False
        
        # Cancel tasks
        if self._receive_task:
            self._receive_task.cancel()
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
        
        # Close WebSocket
        if self.websocket:
            await self.websocket.close()
            self.websocket = None
        
        # Cancel pending requests
        for future in self.pending_requests.values():
            if not future.done():
                future.cancel()
        self.pending_requests.clear()
        
        await self.metrics_collector.record_connection_event("disconnected")
        logger.info("WebSocket disconnected")
    
    async def _receive_loop(self):
        """Main receive loop for handling incoming messages"""
        try:
            while self.connected and self.websocket:
                try:
                    message_str = await self.websocket.recv()
                    
                    if self.config.enable_message_logging:
                        logger.debug(f"Received: {message_str}")
                    
                    # Parse message
                    message_data = json.loads(message_str)
                    message = WebSocketMessage(
                        type=MessageType(message_data.get("type")),
                        data=message_data.get("data", {}),
                        timestamp=message_data.get("timestamp", time.time()),
                        request_id=message_data.get("request_id")
                    )
                    
                    await self._handle_message(message)
                    
                except websockets.exceptions.ConnectionClosed:
                    logger.warning("WebSocket connection closed")
                    self.connected = False
                    break
                    
                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON received: {e}")
                    continue
                    
                except Exception as e:
                    logger.error(f"Error in receive loop: {e}")
                    continue
                    
        except asyncio.CancelledError:
            logger.debug("Receive loop cancelled")
        except Exception as e:
            logger.error(f"Receive loop error: {e}")
            self.connected = False
    
    async def _handle_message(self, message: WebSocketMessage):
        """Handle incoming WebSocket message"""
        # Handle response to pending request
        if message.request_id and message.request_id in self.pending_requests:
            future = self.pending_requests.pop(message.request_id)
            if not future.done():
                if message.type == MessageType.ERROR:
                    future.set_exception(EndpointError(message.data.get("error", "Unknown error")))
                else:
                    future.set_result(message.data)
        
        # Handle server-initiated messages
        elif message.type in self.message_handlers:
            try:
                await self.message_handlers[message.type](message)
            except Exception as e:
                logger.error(f"Error handling {message.type}: {e}")
    
    async def _heartbeat_loop(self):
        """Send periodic heartbeat messages"""
        try:
            while self.connected:
                await asyncio.sleep(self.config.heartbeat_interval)
                
                if self.connected:
                    try:
                        await self._send_message(MessageType.HEARTBEAT, {})
                    except Exception as e:
                        logger.warning(f"Heartbeat failed: {e}")
                        
        except asyncio.CancelledError:
            logger.debug("Heartbeat loop cancelled")
    
    async def _send_message(self, message_type: MessageType, data: Dict[str, Any], expect_response: bool = False) -> Optional[Dict[str, Any]]:
        """Send WebSocket message and optionally wait for response"""
        if not self.connected or not self.websocket:
            raise EndpointError("WebSocket not connected")
        
        start_time = time.time()
        request_id = self._generate_request_id() if expect_response else None
        
        message = {
            "type": message_type.value,
            "data": data,
            "timestamp": time.time(),
            "request_id": request_id
        }
        
        try:
            message_str = json.dumps(message)
            
            if self.config.enable_message_logging:
                logger.debug(f"Sending: {message_str}")
            
            # Send message
            await self.websocket.send(message_str)
            
            # Wait for response if expected
            if expect_response and request_id:
                future = asyncio.Future()
                self.pending_requests[request_id] = future
                
                try:
                    response = await asyncio.wait_for(future, timeout=self.config.response_timeout)
                    await self.metrics_collector.record_message(message_type.value, time.time() - start_time, True)
                    return response
                    
                except asyncio.TimeoutError:
                    self.pending_requests.pop(request_id, None)
                    await self.metrics_collector.record_message(message_type.value, time.time() - start_time, False, "timeout")
                    raise TimeoutError(f"WebSocket request {message_type.value} timed out")
            else:
                await self.metrics_collector.record_message(message_type.value, time.time() - start_time, True)
                return None
                
        except websockets.exceptions.ConnectionClosed:
            await self.metrics_collector.record_message(message_type.value, time.time() - start_time, False, "connection_closed")
            raise EndpointError("WebSocket connection closed")
            
        except Exception as e:
            await self.metrics_collector.record_message(message_type.value, time.time() - start_time, False, str(e))
            raise EndpointError(f"WebSocket send error: {e}")
    
    async def reset(self, scenario: Optional[str] = None, seed: Optional[int] = None) -> tuple[SimulationState, float]:
        """Reset simulation via WebSocket"""
        data = {
            "scenario": scenario or "default",
            "seed": seed,
            "dt_sim": 0.5  # Default simulation timestep
        }
        
        logger.info(f"WebSocket reset: scenario={scenario}, seed={seed}")
        response = await self._send_message(MessageType.RESET, data, expect_response=True)
        
        state = SimulationState(
            timestamp=response["timestamp"],
            branches=response["branches"],
            phase_id=response.get("phase_id")
        )
        
        return state, response["timestamp"]
    
    async def step(self, dt: float, repeat_k: int = 1) -> tuple[SimulationState, float]:
        """Advance simulation via WebSocket"""
        data = {
            "dt": dt,
            "repeat": repeat_k
        }
        
        response = await self._send_message(MessageType.STEP, data, expect_response=True)
        
        state = SimulationState(
            timestamp=response["timestamp"],
            branches=response["branches"],
            phase_id=response.get("phase_id")
        )
        
        return state, response["timestamp"]
    
    async def set_signal(self, signal_plan: Dict[str, Any]) -> bool:
        """Set traffic signal via WebSocket"""
        logger.debug(f"WebSocket set signal: {signal_plan}")
        response = await self._send_message(MessageType.SET_SIGNAL, signal_plan, expect_response=True)
        return response.get("success", False)
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get simulation metrics via WebSocket"""
        response = await self._send_message(MessageType.GET_METRICS, {}, expect_response=True)
        
        # Add WebSocket performance metrics
        ws_stats = await self.metrics_collector.get_stats()
        response["websocket_performance"] = ws_stats
        
        return response
    
    async def health(self) -> Dict[str, Any]:
        """Health check via WebSocket"""
        start_time = time.time()
        response = await self._send_message(MessageType.HEALTH, {}, expect_response=True)
        latency = time.time() - start_time
        
        response["websocket_latency"] = latency
        response["websocket_connected"] = self.connected
        
        return response
    
    async def reconnect(self) -> bool:
        """Attempt to reconnect WebSocket"""
        if self.reconnecting:
            return False
        
        self.reconnecting = True
        attempt = 0
        delay = self.config.reconnect_delay
        
        try:
            while attempt < self.config.max_reconnect_attempts:
                attempt += 1
                logger.info(f"Reconnection attempt {attempt}/{self.config.max_reconnect_attempts}")
                
                await self.disconnect()
                await asyncio.sleep(delay)
                
                if await self.connect():
                    logger.info("WebSocket reconnection successful")
                    return True
                
                delay *= self.config.reconnect_backoff
                await self.metrics_collector.record_connection_event("reconnect_failed", {"attempt": attempt})
            
            logger.error("WebSocket reconnection failed after all attempts")
            return False
            
        finally:
            self.reconnecting = False
    
    def add_message_handler(self, message_type: MessageType, handler: Callable):
        """Add handler for server-initiated messages"""
        self.message_handlers[message_type] = handler
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.disconnect()


# Context manager for WebSocket client
@asynccontextmanager
async def websocket_client(config: WebSocketConfig):
    """Async context manager for WebSocket ARZ client"""
    client = WebSocketARZClient(config)
    try:
        if await client.connect():
            yield client
        else:
            raise EndpointError("Failed to establish WebSocket connection")
    finally:
        await client.disconnect()


# Factory function
def create_websocket_client(config: WebSocketConfig) -> WebSocketARZClient:
    """Create WebSocket client instance"""
    return WebSocketARZClient(config)