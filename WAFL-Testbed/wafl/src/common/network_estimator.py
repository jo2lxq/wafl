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
        self._smoothed_bandwidth = 5.0  # Balanced start (Was 10.0 -> 2.0 -> 5.0)

        # EMA (指数移動平均) 係数
        self._alpha = 0.2  # 新しい値の重み

    def seed_metrics(self, *, loss_rate: float, rtt_ms: float, bandwidth_mbps: float) -> None:
        """静的なネットワーク条件で推定器を初期化する。

        ctrl 側が把握している rate/delay/loss をノード側に反映し、
        起動直後から pacing/FEC/window の推奨値を現実の条件に合わせる。

        Args:
            loss_rate: パケットロス率 (0.0-1.0)
            rtt_ms: RTT (ms)
            bandwidth_mbps: 帯域 (Mbps)
        """
        # Sanitize
        loss_rate = max(0.0, min(1.0, float(loss_rate)))
        rtt_ms = max(1.0, float(rtt_ms))
        bandwidth_mbps = max(0.01, float(bandwidth_mbps))

        with self._lock:
            self._smoothed_loss_rate = loss_rate
            self._smoothed_rtt = rtt_ms
            self._smoothed_bandwidth = bandwidth_mbps

            # Seed short histories so get_metrics() jitter computation won't misbehave
            self._packet_sent.clear()
            self._packet_success.clear()
            # Seed with WINDOW_SIZE/10 samples to avoid overweighting a single point
            seed_n = max(5, min(20, self.WINDOW_SIZE // 5))
            for _ in range(seed_n):
                self._packet_sent.append(1)
                self._packet_success.append(int(round((1.0 - loss_rate))))
                self._rtt_samples.append(rtt_ms)
                self._bandwidth_samples.append(bandwidth_mbps)

        self.logger.info(f"📌 Seeded network metrics: loss={loss_rate * 100:.1f}%, RTT={rtt_ms:.1f}ms, BW={bandwidth_mbps:.2f}Mbps")

    def record_packet_result(self, success: bool) -> None:
        """パケット送信結果を記録"""
        with self._lock:
            self._packet_sent.append(1)
            self._packet_success.append(1 if success else 0)
            self._update_loss_rate()

    def record_loss_rate_sample(self, loss_rate: float) -> None:
        """パケットロス率の観測値を直接取り込む。

        FEC 統計など、パケット単位の成功/失敗とは異なる集計から得られた
        loss 観測値を推定器にフィードバックするために使う。

        Args:
            loss_rate: パケットロス率 (0.0-1.0)
        """
        loss_rate = max(0.0, min(1.0, float(loss_rate)))
        with self._lock:
            self._smoothed_loss_rate = self._alpha * loss_rate + (1 - self._alpha) * self._smoothed_loss_rate

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
        NACK ベースの動的調整を考慮し、最小限の冗長性で高い信頼性を実現する。

        Args:
            k: データパケット数

        Returns:
            推奨 parity 値
        """
        metrics = self.get_metrics()
        loss = metrics.packet_loss_rate

        # パケットロス率に応じた冗長性
        # 目標: k 個のパケットで復元できる確率を 99% 以上に
        # NOTE: parity=0 は NACK 再送のみに依存するため、最小 parity=1 を維持
        if loss < 0.002:
            # excellent (nearly perfect < 0.2%): 最小 FEC (1パケットのみ)
            parity = 1
        elif loss < 0.005:
            # excellent (very low loss < 0.5%): 6.25% 冗長性 (1/16)
            parity_ratio = 0.0625
            parity = max(1, int(k * parity_ratio))
        elif loss < 0.01:
            # excellent/good boundary: 6.25% 冗長性
            parity_ratio = 0.0625
            parity = max(1, int(k * parity_ratio))
        elif loss < 0.02:
            # good: 12.5% 冗長性
            parity_ratio = 0.125
            parity = max(2, int(k * parity_ratio))
        elif loss < 0.05:
            # good/fair: 25% 冗長性
            parity_ratio = 0.25
            parity = max(4, int(k * parity_ratio))
        elif loss < 0.10:
            # fair: 50% 冗長性
            parity_ratio = 0.50
            parity = max(8, int(k * parity_ratio))
        elif loss < 0.20:
            # poor: 75% 冗長性
            parity_ratio = 0.75
            parity = max(12, int(k * parity_ratio))
        else:
            # very poor: 100% 冗長性
            parity_ratio = 1.0
            parity = max(16, int(k * parity_ratio))

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

        # 制約: 4 <= window <= 4096 (Was 128)
        final_window = max(4, min(4096, window))

        # ログ出力 (計算過程を可視化)
        self.logger.info(f"📈 Window size calc: BW={metrics.bandwidth_mbps:.2f}Mbps, RTT={metrics.rtt_ms:.1f}ms, BDP={bdp_bytes / 1024:.1f}KB -> window={final_window}")

        return final_window

    def get_recommended_aging_limit(self) -> float:
        """推奨 Aging 制限を計算

        RTT に基づいて動的に寿命を調整．
        """
        metrics = self.get_metrics()

        # 寿命 = max(0.5s, RTT * 5)
        aging_limit = max(0.5, (metrics.rtt_ms / 1000) * 5)

        # 上限: 10秒
        return min(10.0, aging_limit)

    def get_recommended_pacing_delay(self) -> float:
        """推奨パケットペーシング遅延を計算 (秒)

        パケットロス率と帯域幅に基づいて決定．
        バースト転送による輻輳を回避する．
        """
        metrics = self.get_metrics()
        loss = metrics.packet_loss_rate
        bandwidth = metrics.bandwidth_mbps

        # Base pacing calculation:
        # Prevent saturating the link instantaneously.
        # Use a small sleep if loss rate is high.

        # Loss-based baseline (keeps behavior similar to current implementation)
        if loss < 0.005:
            loss_based = 0.0
        elif loss < 0.02:
            loss_based = 0.0001
        elif loss < 0.05:
            loss_based = 0.0005
        else:
            loss_based = 0.001 + (0.001 * loss * 10)  # Max ~2ms at loss~0.1

        # Bandwidth-based pacing (avoid bursting far above rate limit)
        # Approx packet time on the wire (assume ~1400B UDP packet)
        # Use a conservative fraction to allow some kernel buffering but reduce bloat.
        if bandwidth <= 0:
            bw_based = 0.0
        else:
            packet_bits = 1400 * 8
            packet_time = packet_bits / (bandwidth * 1_000_000)
            bw_based = packet_time * 0.5

        pacing = max(loss_based, bw_based)
        # Clamp for safety
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
