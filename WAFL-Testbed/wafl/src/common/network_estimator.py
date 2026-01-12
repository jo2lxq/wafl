"""
ネットワーク状態推定器．

実測データからネットワーク条件を推定し，動的パラメータ調整に使用する．
"""

import logging
import threading
from collections import deque
from dataclasses import dataclass
from typing import Optional


@dataclass
class NetworkMetrics:
    """ネットワークメトリクス"""

    packet_loss_rate: float = 0.0  # パケットロス率 (0.0-1.0)
    rtt_ms: float = 100.0  # RTT (ms)
    bandwidth_mbps: float = 10.0  # 帯域 (Mbps)
    jitter_ms: float = 0.0  # ジッター (ms)

    def get_quality_level(self) -> str:
        """ネットワーク品質レベルを返す"""
        if self.packet_loss_rate < 0.02 and self.rtt_ms < 20:
            return "excellent"
        elif self.packet_loss_rate < 0.05 and self.rtt_ms < 50:
            return "good"
        elif self.packet_loss_rate < 0.10 and self.rtt_ms < 100:
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
        self._smoothed_bandwidth = 5.0  # Balanced start

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


class NetworkEstimator:
    """Network state estimator using observed measurements, handling global and per-peer states."""

    WINDOW_SIZE = 100
    ALPHA = 0.2

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

    def get_recommended_fec_parity(self, k: int, peer_ip: Optional[str] = None) -> int:
        metrics = self.get_metrics(peer_ip)
        loss = metrics.packet_loss_rate

        if loss < 0.002:
            parity = 1
        elif loss < 0.005:
            parity = max(1, int(k * 0.0625))
        elif loss < 0.01:
            parity = max(1, int(k * 0.0625))
        elif loss < 0.02:
            parity = max(2, int(k * 0.125))
        elif loss < 0.05:
            parity = max(4, int(k * 0.25))
        elif loss < 0.10:
            parity = max(8, int(k * 0.50))
        elif loss < 0.20:
            parity = max(12, int(k * 0.75))
        else:
            parity = max(16, int(k * 1.0))

        if k + parity > 256:
            parity = 256 - k

        return parity

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
        loss = metrics.packet_loss_rate
        bandwidth = metrics.bandwidth_mbps

        if loss < 0.005:
            loss_based = 0.0
        elif loss < 0.02:
            loss_based = 0.0001
        elif loss < 0.05:
            loss_based = 0.0005
        else:
            loss_based = 0.001 + (0.001 * loss * 10)

        if bandwidth <= 0:
            bw_based = 0.0
        else:
            packet_bits = 1400 * 8
            packet_time = packet_bits / (bandwidth * 1_000_000)
            bw_based = packet_time * 0.5

        pacing = max(loss_based, bw_based)
        return max(0.0, min(0.02, pacing))


# グローバルインスタンス (シングルトン)
_global_estimator: Optional[NetworkEstimator] = None
_estimator_lock = threading.Lock()


def get_network_estimator() -> NetworkEstimator:
    """グローバル NetworkEstimator を取得"""
    global _global_estimator
    with _estimator_lock:
        if _global_estimator is None:
            _global_estimator = NetworkEstimator()
        return _global_estimator
