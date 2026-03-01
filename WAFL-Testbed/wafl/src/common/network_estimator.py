"""
Network state estimator.

Estimates network conditions from observed measurements for dynamic parameter tuning.
"""

import logging
import math
import threading
from collections import deque
from dataclasses import dataclass
from typing import Optional


@dataclass
class NetworkMetrics:
    """Network quality metrics."""

    packet_loss_rate: float = 0.0  # Packet loss rate (0.0-1.0)
    rtt_ms: float = 100.0  # Round-trip time (ms)
    bandwidth_mbps: float = 10.0  # Bandwidth (Mbps)
    jitter_ms: float = 0.0  # Jitter (ms)

    def get_quality_level(self) -> str:
        """Return the network quality level."""
        # User defined reference: Ex(2%,5ms), Good(5%,15ms), Fair(8%,30ms), Poor(12%,50ms)
        # Thresholds set strictly based on definition (RTT roughly 2x one-way delay)
        if self.packet_loss_rate <= 0.02 and self.rtt_ms <= 10:
            return "excellent"
        elif self.packet_loss_rate <= 0.05 and self.rtt_ms <= 30:
            return "good"
        elif self.packet_loss_rate <= 0.08 and self.rtt_ms <= 60:
            return "fair"
        else:
            return "poor"


class EstimatorState:
    """Estimator internal state for a single context (global or per-peer)"""

    def __init__(self, window_size: int, alpha: float):
        self.window_size = window_size
        self.alpha = alpha

        # Measurements
        self._packet_sent = deque(maxlen=window_size)
        self._packet_success = deque(maxlen=window_size)
        self._rtt_samples = deque(maxlen=window_size)
        self._bandwidth_samples = deque(maxlen=window_size)

        # Smoothed metrics
        self._smoothed_loss_rate = 0.0
        self._smoothed_rtt = 100.0
        self._smoothed_bandwidth = 10.0  # Optimistic start for faster initial convergence

    def seed(self, loss_rate: float, rtt_ms: float, bandwidth_mbps: float):
        self._smoothed_loss_rate = loss_rate
        self._smoothed_rtt = rtt_ms
        self._smoothed_bandwidth = bandwidth_mbps

        self._packet_sent.clear()
        self._packet_success.clear()
        seed_n = max(5, min(20, self.window_size // 5))
        for _ in range(seed_n):
            self._packet_sent.append(1)
            self._packet_success.append(int(round((1.0 - loss_rate))))
            self._rtt_samples.append(rtt_ms)
            self._bandwidth_samples.append(bandwidth_mbps)

    def record_packet_result(self, success: bool):
        self._packet_sent.append(1)
        self._packet_success.append(1 if success else 0)
        self._update_loss_rate()

    def record_loss_rate_sample(self, loss_rate: float):
        self._smoothed_loss_rate = self.alpha * loss_rate + (1 - self.alpha) * self._smoothed_loss_rate

    def record_rtt(self, rtt_ms: float):
        if rtt_ms <= 0:
            return
        self._rtt_samples.append(rtt_ms)
        self._update_rtt()

    def record_transfer(self, bandwidth_mbps: float):
        self._bandwidth_samples.append(bandwidth_mbps)
        self._update_bandwidth()

    def _update_loss_rate(self):
        if len(self._packet_sent) < 5:
            return
        recent_loss = 1.0 - (sum(self._packet_success) / len(self._packet_sent))
        self._smoothed_loss_rate = self.alpha * recent_loss + (1 - self.alpha) * self._smoothed_loss_rate

    def _update_rtt(self):
        if not self._rtt_samples:
            return
        recent_rtt = sum(self._rtt_samples) / len(self._rtt_samples)
        self._smoothed_rtt = self.alpha * recent_rtt + (1 - self.alpha) * self._smoothed_rtt

    def _update_bandwidth(self):
        if not self._bandwidth_samples:
            return
        recent_bw = sum(self._bandwidth_samples) / len(self._bandwidth_samples)
        self._smoothed_bandwidth = self.alpha * recent_bw + (1 - self.alpha) * self._smoothed_bandwidth

    def get_metrics(self) -> NetworkMetrics:
        jitter = 0.0
        if len(self._rtt_samples) >= 2:
            mean_rtt = sum(self._rtt_samples) / len(self._rtt_samples)
            variance = sum((x - mean_rtt) ** 2 for x in self._rtt_samples) / len(self._rtt_samples)
            jitter = variance**0.5

        return NetworkMetrics(
            packet_loss_rate=self._smoothed_loss_rate,
            rtt_ms=self._smoothed_rtt,
            bandwidth_mbps=self._smoothed_bandwidth,
            jitter_ms=jitter,
        )

    def get_sample_count(self) -> int:
        """Return the number of bandwidth samples collected."""
        return len(self._bandwidth_samples)


class NetworkEstimator:
    """Network state estimator using observed measurements, handling global and per-peer states."""

    WINDOW_SIZE = 100
    ALPHA = 0.3  # Faster convergence to measured values

    def __init__(self):
        self.logger = logging.getLogger("NetworkEstimator")
        self._lock = threading.Lock()

        # State management
        self._global_state = EstimatorState(self.WINDOW_SIZE, self.ALPHA)
        self._peer_states = {}  # type: dict[str, EstimatorState]

    def _get_state(self, peer_ip: Optional[str] = None) -> EstimatorState:
        """Get the appropriate state object."""
        if peer_ip is None:
            return self._global_state

        if peer_ip not in self._peer_states:
            self._peer_states[peer_ip] = EstimatorState(self.WINDOW_SIZE, self.ALPHA)

            # Seed from global state (Measurement-based initialization)
            # This ensures that we don't start with "Poor" defaults if we already know
            # the environment is "Excellent" from other peers.
            global_metrics = self._global_state.get_metrics()
            self._peer_states[peer_ip].seed(global_metrics.packet_loss_rate, global_metrics.rtt_ms, global_metrics.bandwidth_mbps)
            self.logger.debug(f"🌱 Seeded peer {peer_ip} from global metrics: q={global_metrics.get_quality_level()}")

        return self._peer_states[peer_ip]

    def seed_metrics(self, *, loss_rate: float, rtt_ms: float, bandwidth_mbps: float) -> None:
        """Seed the global estimator state."""
        loss_rate = max(0.0, min(1.0, float(loss_rate)))
        rtt_ms = max(1.0, float(rtt_ms))
        bandwidth_mbps = max(0.01, float(bandwidth_mbps))

        with self._lock:
            self._global_state.seed(loss_rate, rtt_ms, bandwidth_mbps)
            self._peer_states.clear()  # Reset peers context

        self.logger.info(f"📌 Seeded global network metrics: loss={loss_rate * 100:.1f}%, RTT={rtt_ms:.1f}ms, BW={bandwidth_mbps:.2f}Mbps")

    def record_packet_result(self, success: bool, peer_ip: Optional[str] = None) -> None:
        with self._lock:
            self._global_state.record_packet_result(success)
            if peer_ip:
                self._get_state(peer_ip).record_packet_result(success)

    def record_loss_rate_sample(self, loss_rate: float, peer_ip: Optional[str] = None) -> None:
        loss_rate = max(0.0, min(1.0, float(loss_rate)))
        with self._lock:
            self._global_state.record_loss_rate_sample(loss_rate)
            if peer_ip:
                self._get_state(peer_ip).record_loss_rate_sample(loss_rate)

    def record_rtt(self, rtt_ms: float, peer_ip: Optional[str] = None) -> None:
        with self._lock:
            self._global_state.record_rtt(rtt_ms)
            if peer_ip:
                self._get_state(peer_ip).record_rtt(rtt_ms)

    def record_transfer(self, bytes_transferred: int, duration_s: float, peer_ip: Optional[str] = None) -> None:
        if duration_s <= 0:
            return
        bandwidth_mbps = (bytes_transferred * 8) / (duration_s * 1_000_000)
        with self._lock:
            self._global_state.record_transfer(bandwidth_mbps)
            if peer_ip:
                self._get_state(peer_ip).record_transfer(bandwidth_mbps)

    def get_metrics(self, peer_ip: Optional[str] = None) -> NetworkMetrics:
        with self._lock:
            return self._get_state(peer_ip).get_metrics()

    def get_sample_count(self, peer_ip: Optional[str] = None) -> int:
        """Return the number of bandwidth samples for a peer or global."""
        with self._lock:
            return self._get_state(peer_ip).get_sample_count()

    def get_recommended_fec_parity(self, k: int, peer_ip: Optional[str] = None) -> int:
        metrics = self.get_metrics(peer_ip)
        loss = metrics.packet_loss_rate

        # Linear redundancy calculation with safety factor
        # Linear: parity = ceil(k * loss_rate * safety_factor)
        # Factor 2.5 means:
        # Loss 1% -> +2.5% (min 1)
        # Loss 5% -> +12.5% (k=16 -> +2)
        # Loss 20% -> +50% (k=16 -> +8)
        safety_factor = 2.5
        parity = math.ceil(k * loss * safety_factor)

        # Always ensure at least 1 parity packet (min 1/k redundancy)
        # to allow recovery of single packet loss without retransmission
        parity = max(1, parity)

        # Cap at zfec limit (k + m <= 256)
        if k + parity > 256:
            parity = 256 - k

        return int(parity)

    def get_recommended_window_size(self, peer_ip: Optional[str] = None) -> int:
        metrics = self.get_metrics(peer_ip)

        bandwidth_bytes_per_sec = metrics.bandwidth_mbps * 1_000_000 / 8
        rtt_sec = metrics.rtt_ms / 1000
        bdp_bytes = bandwidth_bytes_per_sec * rtt_sec

        packet_size = 1400
        window = int(bdp_bytes / packet_size)
        final_window = max(4, min(4096, window))

        return final_window

    def get_recommended_aging_limit(self, peer_ip: Optional[str] = None) -> float:
        metrics = self.get_metrics(peer_ip)
        aging_limit = max(0.5, (metrics.rtt_ms / 1000) * 5)
        return min(10.0, aging_limit)

    def get_recommended_pacing_delay(self, peer_ip: Optional[str] = None) -> float:
        metrics = self.get_metrics(peer_ip)
        bandwidth = metrics.bandwidth_mbps

        # Pure Bandwidth-Based Pacing
        # Do not penalize for loss (which is often random/interference in this context).
        # We rely on FEC to handle loss, and Pacing to handle bottleneck capacity.

        if bandwidth <= 0:
            # Default safe pacing if unknown
            return 0.002  # 2ms

        packet_bits = 1400 * 8
        packet_time = packet_bits / (bandwidth * 1_000_000)

        # Pacing at 50% of packet serialization time to avoid bursting
        # If BW is 100Mbps, packet_time ~ 0.11ms -> Pacing 0.05ms
        # rsleep/time.sleep resolution is poor (1ms+), so accumulation might be needed in sender.
        # But here we just return the recommended delay.
        pacing_delay = packet_time * 0.5

        return max(0.0, min(0.01, pacing_delay))


# Global singleton instance
_global_estimator: Optional[NetworkEstimator] = None
_estimator_lock = threading.Lock()


def get_network_estimator() -> NetworkEstimator:
    """Get the global NetworkEstimator instance."""
    global _global_estimator
    with _estimator_lock:
        if _global_estimator is None:
            _global_estimator = NetworkEstimator()
        return _global_estimator
