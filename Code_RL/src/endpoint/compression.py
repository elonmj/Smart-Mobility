"""
Compression Support for Large Traffic State Payloads

This module provides compression and decompression capabilities for 
RL-simulator communication to reduce bandwidth usage and improve
performance for large traffic state data.
"""

import gzip
import zlib
import brotli
import json
import time
import logging
from typing import Dict, Any, Optional, Union, Callable
from dataclasses import dataclass
from enum import Enum
import asyncio
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class CompressionType(Enum):
    """Supported compression algorithms"""
    NONE = "none"
    GZIP = "gzip"
    DEFLATE = "deflate" 
    BROTLI = "brotli"
    LZ4 = "lz4"  # Fast compression for real-time scenarios


@dataclass
class CompressionConfig:
    """Configuration for compression settings"""
    
    # Compression algorithm
    algorithm: CompressionType = CompressionType.GZIP
    
    # Compression levels (1-9, algorithm dependent)
    compression_level: int = 6  # Balanced compression/speed
    
    # Thresholds
    min_size_bytes: int = 1024  # Only compress payloads > 1KB
    max_size_bytes: int = 10 * 1024 * 1024  # 10MB max
    
    # Performance settings
    enable_async: bool = True
    chunk_size: int = 8192
    
    # Content-specific settings
    json_compression: bool = True
    binary_compression: bool = True
    
    # Adaptive compression
    enable_adaptive: bool = True
    target_compression_ratio: float = 0.3  # Target 70% size reduction
    max_compression_time_ms: float = 50.0  # Max 50ms compression time


@dataclass
class CompressionMetrics:
    """Metrics for compression performance"""
    
    original_size: int
    compressed_size: int
    compression_ratio: float
    compression_time: float
    decompression_time: float
    algorithm: CompressionType
    success: bool = True
    error: Optional[str] = None


class CompressionAlgorithm(ABC):
    """Abstract base class for compression algorithms"""
    
    @abstractmethod
    def compress(self, data: bytes, level: int = 6) -> bytes:
        """Compress data"""
        pass
    
    @abstractmethod
    def decompress(self, data: bytes) -> bytes:
        """Decompress data"""
        pass
    
    @abstractmethod
    def get_default_level(self) -> int:
        """Get default compression level"""
        pass
    
    @abstractmethod
    def get_max_level(self) -> int:
        """Get maximum compression level"""
        pass


class GzipCompression(CompressionAlgorithm):
    """GZIP compression implementation"""
    
    def compress(self, data: bytes, level: int = 6) -> bytes:
        return gzip.compress(data, compresslevel=level)
    
    def decompress(self, data: bytes) -> bytes:
        return gzip.decompress(data)
    
    def get_default_level(self) -> int:
        return 6
    
    def get_max_level(self) -> int:
        return 9


class DeflateCompression(CompressionAlgorithm):
    """DEFLATE compression implementation"""
    
    def compress(self, data: bytes, level: int = 6) -> bytes:
        return zlib.compress(data, level=level)
    
    def decompress(self, data: bytes) -> bytes:
        return zlib.decompress(data)
    
    def get_default_level(self) -> int:
        return 6
    
    def get_max_level(self) -> int:
        return 9


class BrotliCompression(CompressionAlgorithm):
    """Brotli compression implementation (high compression ratio)"""
    
    def compress(self, data: bytes, level: int = 6) -> bytes:
        # Brotli quality range is 0-11
        quality = min(level, 11)
        return brotli.compress(data, quality=quality)
    
    def decompress(self, data: bytes) -> bytes:
        return brotli.decompress(data)
    
    def get_default_level(self) -> int:
        return 6
    
    def get_max_level(self) -> int:
        return 11


class LZ4Compression(CompressionAlgorithm):
    """LZ4 compression implementation (fast compression)"""
    
    def __init__(self):
        try:
            import lz4.frame
            self.lz4 = lz4.frame
        except ImportError:
            logger.warning("LZ4 not available, falling back to GZIP")
            self.lz4 = None
    
    def compress(self, data: bytes, level: int = 6) -> bytes:
        if self.lz4 is None:
            # Fallback to gzip
            return gzip.compress(data, compresslevel=level)
        
        # LZ4 compression level mapping (0-16)
        compression_level = min(level * 2, 16)
        return self.lz4.compress(data, compression_level=compression_level)
    
    def decompress(self, data: bytes) -> bytes:
        if self.lz4 is None:
            return gzip.decompress(data)
        
        return self.lz4.decompress(data)
    
    def get_default_level(self) -> int:
        return 1  # LZ4 prioritizes speed
    
    def get_max_level(self) -> int:
        return 8  # Reasonable max for real-time usage


class AdaptiveCompressor:
    """Adaptive compressor that selects optimal algorithm based on content and performance"""
    
    def __init__(self, config: CompressionConfig):
        self.config = config
        self.algorithms = {
            CompressionType.GZIP: GzipCompression(),
            CompressionType.DEFLATE: DeflateCompression(),
            CompressionType.BROTLI: BrotliCompression(),
            CompressionType.LZ4: LZ4Compression()
        }
        
        # Performance history for adaptive selection
        self.compression_history: Dict[CompressionType, list] = {
            comp_type: [] for comp_type in CompressionType if comp_type != CompressionType.NONE
        }
    
    def _should_compress(self, data_size: int) -> bool:
        """Determine if data should be compressed"""
        return (
            self.config.min_size_bytes <= data_size <= self.config.max_size_bytes
        )
    
    def _select_algorithm(self, data: bytes, content_type: str = "json") -> CompressionType:
        """Select optimal compression algorithm"""
        if not self.config.enable_adaptive:
            return self.config.algorithm
        
        data_size = len(data)
        
        # For small data, use fast compression
        if data_size < 4096:  # 4KB
            return CompressionType.LZ4
        
        # For JSON data, GZIP is usually optimal
        if content_type == "json" and data_size < 50000:  # 50KB
            return CompressionType.GZIP
        
        # For large data, use high-ratio compression
        if data_size > 100000:  # 100KB
            return CompressionType.BROTLI
        
        # Default balanced choice
        return CompressionType.GZIP
    
    def compress_data(self, data: Union[str, bytes, Dict[str, Any]], content_type: str = "json") -> tuple[bytes, CompressionMetrics]:
        """Compress data with performance tracking"""
        start_time = time.time()
        
        # Convert to bytes if needed
        if isinstance(data, dict):
            data_bytes = json.dumps(data, separators=(',', ':')).encode('utf-8')
            content_type = "json"
        elif isinstance(data, str):
            data_bytes = data.encode('utf-8')
        else:
            data_bytes = data
        
        original_size = len(data_bytes)
        
        # Check if compression is worthwhile
        if not self._should_compress(original_size):
            metrics = CompressionMetrics(
                original_size=original_size,
                compressed_size=original_size,
                compression_ratio=1.0,
                compression_time=0.0,
                decompression_time=0.0,
                algorithm=CompressionType.NONE
            )
            return data_bytes, metrics
        
        # Select algorithm
        algorithm = self._select_algorithm(data_bytes, content_type)
        compressor = self.algorithms[algorithm]
        
        try:
            # Perform compression
            compressed_data = compressor.compress(data_bytes, self.config.compression_level)
            compression_time = time.time() - start_time
            
            # Check if compression was beneficial
            compression_ratio = len(compressed_data) / original_size
            
            if compression_ratio > 0.95:  # Less than 5% reduction
                # Compression not beneficial, return original
                metrics = CompressionMetrics(
                    original_size=original_size,
                    compressed_size=original_size,
                    compression_ratio=1.0,
                    compression_time=compression_time,
                    decompression_time=0.0,
                    algorithm=CompressionType.NONE
                )
                return data_bytes, metrics
            
            # Successful compression
            metrics = CompressionMetrics(
                original_size=original_size,
                compressed_size=len(compressed_data),
                compression_ratio=compression_ratio,
                compression_time=compression_time,
                decompression_time=0.0,
                algorithm=algorithm
            )
            
            # Update performance history
            self._update_performance_history(algorithm, metrics)
            
            return compressed_data, metrics
            
        except Exception as e:
            logger.error(f"Compression failed with {algorithm}: {e}")
            
            # Return original data on compression failure
            metrics = CompressionMetrics(
                original_size=original_size,
                compressed_size=original_size,
                compression_ratio=1.0,
                compression_time=time.time() - start_time,
                decompression_time=0.0,
                algorithm=CompressionType.NONE,
                success=False,
                error=str(e)
            )
            return data_bytes, metrics
    
    def decompress_data(self, compressed_data: bytes, algorithm: CompressionType) -> tuple[bytes, float]:
        """Decompress data"""
        if algorithm == CompressionType.NONE:
            return compressed_data, 0.0
        
        start_time = time.time()
        
        try:
            compressor = self.algorithms[algorithm]
            decompressed_data = compressor.decompress(compressed_data)
            decompression_time = time.time() - start_time
            
            return decompressed_data, decompression_time
            
        except Exception as e:
            logger.error(f"Decompression failed with {algorithm}: {e}")
            raise
    
    def _update_performance_history(self, algorithm: CompressionType, metrics: CompressionMetrics):
        """Update performance history for adaptive algorithm selection"""
        history = self.compression_history[algorithm]
        history.append({
            "timestamp": time.time(),
            "compression_ratio": metrics.compression_ratio,
            "compression_time": metrics.compression_time,
            "size": metrics.original_size
        })
        
        # Keep only recent history (last 100 entries)
        if len(history) > 100:
            self.compression_history[algorithm] = history[-100:]
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get compression performance statistics"""
        stats = {}
        
        for algorithm, history in self.compression_history.items():
            if not history:
                continue
            
            ratios = [h["compression_ratio"] for h in history]
            times = [h["compression_time"] for h in history]
            
            stats[algorithm.value] = {
                "sample_count": len(history),
                "avg_compression_ratio": sum(ratios) / len(ratios),
                "avg_compression_time": sum(times) / len(times),
                "best_compression_ratio": min(ratios),
                "worst_compression_ratio": max(ratios)
            }
        
        return stats


class CompressedMessageHandler:
    """Handler for compressed messages in communication protocols"""
    
    def __init__(self, config: CompressionConfig):
        self.config = config
        self.compressor = AdaptiveCompressor(config)
        self.executor = None
    
    async def compress_message_async(self, message: Dict[str, Any]) -> tuple[bytes, Dict[str, Any]]:
        """Compress message asynchronously"""
        if not self.config.enable_async:
            return self._compress_message_sync(message)
        
        # Use thread pool for CPU-intensive compression
        if self.executor is None:
            import concurrent.futures
            self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            self.executor,
            self._compress_message_sync,
            message
        )
        
        return result
    
    def _compress_message_sync(self, message: Dict[str, Any]) -> tuple[bytes, Dict[str, Any]]:
        """Compress message synchronously"""
        compressed_data, metrics = self.compressor.compress_data(message, "json")
        
        # Create compression metadata
        compression_info = {
            "algorithm": metrics.algorithm.value,
            "original_size": metrics.original_size,
            "compressed_size": metrics.compressed_size,
            "compression_ratio": metrics.compression_ratio,
            "compression_time": metrics.compression_time
        }
        
        return compressed_data, compression_info
    
    async def decompress_message_async(self, compressed_data: bytes, algorithm: str) -> Dict[str, Any]:
        """Decompress message asynchronously"""
        if not self.config.enable_async:
            return self._decompress_message_sync(compressed_data, algorithm)
        
        if self.executor is None:
            import concurrent.futures
            self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            self.executor,
            self._decompress_message_sync,
            compressed_data,
            algorithm
        )
        
        return result
    
    def _decompress_message_sync(self, compressed_data: bytes, algorithm: str) -> Dict[str, Any]:
        """Decompress message synchronously"""
        algorithm_enum = CompressionType(algorithm)
        decompressed_data, _ = self.compressor.decompress_data(compressed_data, algorithm_enum)
        
        # Parse JSON
        message = json.loads(decompressed_data.decode('utf-8'))
        return message
    
    def get_compression_stats(self) -> Dict[str, Any]:
        """Get compression performance statistics"""
        return self.compressor.get_performance_stats()
    
    def close(self):
        """Clean up resources"""
        if self.executor:
            self.executor.shutdown(wait=True)


# Utility functions for integration with existing clients
def create_compressed_http_headers(compression_type: CompressionType) -> Dict[str, str]:
    """Create HTTP headers for compressed content"""
    headers = {}
    
    if compression_type == CompressionType.GZIP:
        headers["Content-Encoding"] = "gzip"
        headers["Accept-Encoding"] = "gzip, deflate"
    elif compression_type == CompressionType.DEFLATE:
        headers["Content-Encoding"] = "deflate"
        headers["Accept-Encoding"] = "gzip, deflate"
    elif compression_type == CompressionType.BROTLI:
        headers["Content-Encoding"] = "br"
        headers["Accept-Encoding"] = "gzip, deflate, br"
    
    return headers


def compress_json_payload(payload: Dict[str, Any], config: CompressionConfig = None) -> tuple[bytes, str]:
    """Compress JSON payload and return compressed data with algorithm"""
    if config is None:
        config = CompressionConfig()
    
    compressor = AdaptiveCompressor(config)
    compressed_data, metrics = compressor.compress_data(payload, "json")
    
    return compressed_data, metrics.algorithm.value


def decompress_json_payload(compressed_data: bytes, algorithm: str) -> Dict[str, Any]:
    """Decompress JSON payload"""
    config = CompressionConfig()
    compressor = AdaptiveCompressor(config)
    
    algorithm_enum = CompressionType(algorithm)
    decompressed_data, _ = compressor.decompress_data(compressed_data, algorithm_enum)
    
    return json.loads(decompressed_data.decode('utf-8'))


# Performance benchmarking for compression algorithms
def benchmark_compression_algorithms(test_data: Dict[str, Any], iterations: int = 100) -> Dict[str, Any]:
    """Benchmark different compression algorithms"""
    results = {}
    
    test_payload = json.dumps(test_data, separators=(',', ':')).encode('utf-8')
    original_size = len(test_payload)
    
    algorithms = [
        CompressionType.GZIP,
        CompressionType.DEFLATE,
        CompressionType.BROTLI,
        CompressionType.LZ4
    ]
    
    for algorithm in algorithms:
        if algorithm == CompressionType.NONE:
            continue
        
        config = CompressionConfig(algorithm=algorithm)
        compressor = AdaptiveCompressor(config)
        
        compression_times = []
        decompression_times = []
        compression_ratios = []
        
        for _ in range(iterations):
            # Test compression
            compressed_data, metrics = compressor.compress_data(test_payload)
            compression_times.append(metrics.compression_time)
            compression_ratios.append(metrics.compression_ratio)
            
            # Test decompression
            start_time = time.time()
            decompressed_data, _ = compressor.decompress_data(compressed_data, algorithm)
            decompression_times.append(time.time() - start_time)
        
        results[algorithm.value] = {
            "avg_compression_time": sum(compression_times) / len(compression_times),
            "avg_decompression_time": sum(decompression_times) / len(decompression_times),
            "avg_compression_ratio": sum(compression_ratios) / len(compression_ratios),
            "original_size": original_size,
            "avg_compressed_size": int(original_size * sum(compression_ratios) / len(compression_ratios))
        }
    
    return results