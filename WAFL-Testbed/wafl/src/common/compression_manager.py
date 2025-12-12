import logging
import struct
import time
import zlib

try:
    import lz4.frame
except ImportError:
    lz4 = None


class CompressionManager:
    """
    Manages adaptive compression of model data.
    Supports: None, LZ4, Zlib, and Adaptive (auto-selects best method).
    """

    def __init__(self, initial_method: str = "zlib"):
        """Initialize compression manager.

        Args:
            initial_method: Initial compression method. Options:
                - "none": No compression
                - "lz4": LZ4 fast compression
                - "zlib": Zlib high compression
                - "adaptive": Automatically select best method based on bandwidth
        """
        self.adaptive_enabled = initial_method == "adaptive"
        # If adaptive, start with zlib as default, will be updated based on bandwidth
        self.method = "zlib" if self.adaptive_enabled else initial_method
        self.initial_method = initial_method
        self.logger = logging.getLogger("CompressionManager")
        self.last_adjustment = time.time()

        # Stats for adaptation
        self.bandwidth_est = 10.0 * 1024 * 1024  # Start with 10 MB/s estimate
        self.history = []

        # Track bytes for metrics
        self.total_bytes_original = 0
        self.total_bytes_compressed = 0
        self.total_compression_time_ms = 0

        self.methods = ["none", "lz4", "zlib"]
        if lz4 is None:
            self.methods.remove("lz4")
            self.logger.warning("LZ4 not installed, disabling LZ4 compression.")

        mode_str = "Adaptive" if self.adaptive_enabled else self.method
        self.logger.info(f"CompressionManager initialized: mode={mode_str}")

    def compress(self, data: bytes) -> bytes:
        """
        Compresses data using current method and records metrics.
        Input data is expected to be pickled bytes (except for float16 which needs raw tensor handling,
        but here we assume we get bytes. For float16 we might need to handle it before pickling or
        re-pickle. To keep it simple, we'll apply float16 quantization on the bytes if possible
        or assume the caller handles it.
        Actually, Float16 quantization must happen on the Tensor BEFORE serialization.
        So 'compress' here strictly deals with byte-level compression.
        We will handle LZ4 and Zlib here. Float16 logic needs to be in ModelSharingUtils.
        """
        start_t = time.time()
        out = data
        method_used = self.method
        original_size = len(data)

        try:
            if self.method == "zlib":
                out = zlib.compress(data, level=6)
            elif self.method == "lz4" and lz4:
                out = lz4.frame.compress(data)
            elif self.method == "none":
                out = data
            # float16 is handled at model level, not bytes level.

            comp_time = time.time() - start_t
            compressed_size = len(out)
            compression_ratio = compressed_size / original_size if original_size > 0 else 1.0

            # Record compression metrics for adaptation
            self.history.append(
                {
                    "method": method_used,
                    "original_size": original_size,
                    "compressed_size": compressed_size,
                    "compression_time": comp_time,
                    "compression_ratio": compression_ratio,
                    "timestamp": time.time(),
                }
            )

            # Update cumulative stats for reporting
            self.total_bytes_original += original_size
            self.total_bytes_compressed += compressed_size
            self.total_compression_time_ms += comp_time * 1000

            # Keep history limited to last 100 entries
            if len(self.history) > 100:
                self.history = self.history[-100:]

            # Add header to identify method
            # Format: [MethodID:1b][Data]
            # 0: None, 1: Zlib, 2: LZ4
            mid = 0
            if self.method == "zlib":
                mid = 1
            elif self.method == "lz4":
                mid = 2

            out = struct.pack("B", mid) + out

            return out
        except Exception as e:
            self.logger.error(f"Compression failed: {e}")
            return b"\x00" + data  # Fallback to none

    def decompress(self, data: bytes) -> bytes:
        """
        Decompresses data.
        """
        try:
            if not data:
                return b""
            mid = data[0]
            payload = data[1:]

            if mid == 0:  # None
                return payload
            elif mid == 1:  # Zlib
                return zlib.decompress(payload)
            elif mid == 2:  # LZ4
                if lz4:
                    return lz4.frame.decompress(payload)
                else:
                    raise ValueError("LZ4 data received but module not available")
            else:
                return payload
        except Exception as e:
            self.logger.error(f"Decompression failed: {e}")
            return data

    def update_strategy(self, transfer_time: float, data_size: int, fec_redundancy: float = 1.0):
        """
        Updates compression strategy based on:
        T_est = T_comp + (Size_comp * R) / BW

        Only applies when adaptive mode is enabled.
        """
        if transfer_time <= 0:
            return

        # Update Bandwidth Estimate (EMA)
        current_bw = data_size / transfer_time
        alpha = 0.3
        self.bandwidth_est = (alpha * current_bw) + ((1 - alpha) * self.bandwidth_est)

        # Only adjust method if adaptive mode is enabled
        if not self.adaptive_enabled:
            return

        current_time = time.time()
        if current_time - self.last_adjustment < 10.0:  # Adapt every 10s
            return

        self.last_adjustment = current_time

        # Evaluate candidates
        # We need estimates for Compression Ratio (CR) and Compression Speed (CS) for each method.
        # These are rough heuristics or learned values.
        # Method: (CR, Speed MB/s)
        # None: (1.0, Inf)
        # LZ4: (0.5, 500)
        # Zlib: (0.3, 50)

        candidates = {
            "none": {"cr": 1.0, "speed": 999999},
            "zlib": {"cr": 0.3, "speed": 50 * 1024 * 1024},
        }
        if lz4:
            candidates["lz4"] = {"cr": 0.5, "speed": 500 * 1024 * 1024}

        best_method = self.method
        min_est_time = float("inf")

        original_size = data_size  # Approximation

        for m, stats in candidates.items():
            t_comp = original_size / stats["speed"]
            size_comp = original_size * stats["cr"]
            t_net = (size_comp * fec_redundancy) / self.bandwidth_est
            t_total = t_comp + t_net

            if t_total < min_est_time:
                min_est_time = t_total
                best_method = m

        if best_method != self.method:
            self.logger.info(f"🔄 Switching compression: {self.method} -> {best_method} (BW: {self.bandwidth_est / 1024 / 1024:.2f} MB/s)")
            self.method = best_method

    def get_stats(self) -> dict:
        """Get compression statistics for logging.

        Returns:
            Dictionary with compression stats for MetricsLogger
        """
        return {
            "method": self.method if not self.adaptive_enabled else f"adaptive-{self.method}",
            "total_bytes_original": self.total_bytes_original,
            "total_bytes_compressed": self.total_bytes_compressed,
            "total_compression_time_ms": self.total_compression_time_ms,
            "compression_ratio": (self.total_bytes_compressed / self.total_bytes_original if self.total_bytes_original > 0 else 1.0),
            "bandwidth_est_mbps": self.bandwidth_est / (1024 * 1024),
        }

    def get_last_compression_stats(self) -> dict:
        """Get stats from the last compression operation.

        Returns:
            Dictionary with last compression stats or empty dict if no history
        """
        if not self.history:
            return {
                "ratio": 1.0,
                "time_ms": 0,
                "original_size": 0,
                "compressed_size": 0,
            }
        last = self.history[-1]
        return {
            "ratio": last["compression_ratio"],
            "time_ms": last["compression_time"] * 1000,
            "original_size": last["original_size"],
            "compressed_size": last["compressed_size"],
        }
