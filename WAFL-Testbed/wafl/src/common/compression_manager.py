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
    Supports: None, LZ4, Float16, Zlib.
    """

    def __init__(self, initial_method: str = "zlib"):
        self.method = initial_method
        self.logger = logging.getLogger("CompressionManager")
        self.last_adjustment = time.time()

        # Stats for adaptation
        self.bandwidth_est = 10.0 * 1024 * 1024  # Start with 10 MB/s estimate
        self.history = []

        self.methods = ["none", "lz4", "float16", "zlib"]
        if lz4 is None:
            self.methods.remove("lz4")
            self.logger.warning("LZ4 not installed, disabling LZ4 compression.")

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
        """
        if transfer_time <= 0:
            return

        # Update Bandwidth Estimate (EMA)
        current_bw = data_size / transfer_time
        alpha = 0.3
        self.bandwidth_est = (alpha * current_bw) + ((1 - alpha) * self.bandwidth_est)

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
