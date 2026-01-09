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


class NetworkEstimator:
    """ネットワーク状態を実測値から推定するクラス"""

    WINDOW_SIZE = 100  # 直近の測定数

    def __init__(self):
        self.logger = logging.getLogger("NetworkEstimator")
        self._lock = threading.Lock()

        # 測定履歴 (最新 WINDOW_SIZE 件)
        self._packet_sent = deque(maxlen=self.WINDOW_SIZE)
        self._packet_success = deque(maxlen=self.WINDOW_SIZE)
        self._rtt_samples = deque(maxlen=self.WINDOW_SIZE)
        self._bandwidth_samples = deque(maxlen=self.WINDOW_SIZE)

        # 平滑化されたメトリクス
        self._smoothed_loss_rate = 0.0
        self._smoothed_rtt = 100.0
        self._smoothed_bandwidth = 10.0

        # EMA (指数移動平均) 係数
        self._alpha = 0.2  # 新しい値の重み

    def record_packet_result(self, success: bool) -> None:
        """パケット送信結果を記録"""
        with self._lock:
            self._packet_sent.append(1)
            self._packet_success.append(1 if success else 0)
            self._update_loss_rate()

    def record_rtt(self, rtt_ms: float) -> None:
        """RTT を記録"""
        if rtt_ms <= 0:
            return
        with self._lock:
            self._rtt_samples.append(rtt_ms)
            self._update_rtt()

    def record_transfer(self, bytes_transferred: int, duration_s: float) -> None:
        """転送速度を記録"""
        if duration_s <= 0:
            return
        bandwidth_mbps = (bytes_transferred * 8) / (duration_s * 1_000_000)
        with self._lock:
            self._bandwidth_samples.append(bandwidth_mbps)
            self._update_bandwidth()

    def _update_loss_rate(self) -> None:
        """パケットロス率を更新 (EMA)"""
        if len(self._packet_sent) < 5:
            return
        recent_loss = 1.0 - (sum(self._packet_success) / len(self._packet_sent))
        self._smoothed_loss_rate = self._alpha * recent_loss + (1 - self._alpha) * self._smoothed_loss_rate

    def _update_rtt(self) -> None:
        """RTT を更新 (EMA)"""
        if not self._rtt_samples:
            return
        recent_rtt = sum(self._rtt_samples) / len(self._rtt_samples)
        self._smoothed_rtt = self._alpha * recent_rtt + (1 - self._alpha) * self._smoothed_rtt

    def _update_bandwidth(self) -> None:
        """帯域を更新 (EMA)"""
        if not self._bandwidth_samples:
            return
        recent_bw = sum(self._bandwidth_samples) / len(self._bandwidth_samples)
        self._smoothed_bandwidth = self._alpha * recent_bw + (1 - self._alpha) * self._smoothed_bandwidth

    def get_metrics(self) -> NetworkMetrics:
        """現在のネットワークメトリクスを取得"""
        with self._lock:
            # ジッターは RTT の標準偏差で推定
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

    def get_recommended_fec_parity(self, k: int) -> int:
        """推奨 FEC parity 値を計算

        パケットロス率に基づいて冗長性を動的に決定する．

        Args:
            k: データパケット数

        Returns:
            推奨 parity 値
        """
        metrics = self.get_metrics()
        loss = metrics.packet_loss_rate

        # パケットロス率に応じた冗長性
        # 目標: k 個のパケットで復元できる確率を 99% 以上に
        if loss < 0.02:
            # excellent: 25% 冗長性
            parity_ratio = 0.25
        elif loss < 0.05:
            # good: 40% 冗長性
            parity_ratio = 0.40
        elif loss < 0.10:
            # fair: 60% 冗長性
            parity_ratio = 0.60
        elif loss < 0.20:
            # poor: 85% 冗長性
            parity_ratio = 0.85
        else:
            # very poor: 100% 冗長性
            parity_ratio = 1.0

        parity = max(2, int(k * parity_ratio))

        # zfec の制約: k + parity <= 256
        if k + parity > 256:
            parity = 256 - k

        return parity

    def get_recommended_window_size(self) -> int:
        """推奨ウィンドウサイズを計算

        帯域遅延積 (BDP) に基づいてウィンドウサイズを決定．
        """
        metrics = self.get_metrics()

        # BDP = 帯域 (bytes/s) × RTT (s)
        bandwidth_bytes_per_sec = metrics.bandwidth_mbps * 1_000_000 / 8
        rtt_sec = metrics.rtt_ms / 1000
        bdp_bytes = bandwidth_bytes_per_sec * rtt_sec

        # パケットサイズで割ってウィンドウサイズを計算
        packet_size = 1400
        window = int(bdp_bytes / packet_size)

        # 制約: 4 <= window <= 128
        return max(4, min(128, window))

    def get_recommended_aging_limit(self) -> float:
        """推奨 Aging 制限を計算

        RTT に基づいて動的に寿命を調整．
        """
        metrics = self.get_metrics()

        # 寿命 = max(0.5s, RTT * 5)
        aging_limit = max(0.5, (metrics.rtt_ms / 1000) * 5)

        # 上限: 10秒
        return min(10.0, aging_limit)


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
