"""
E-RUDP (Extended RUDP) プロトコル実装．

RUDP を継承し，不安定なネットワーク（高パケットロス・変動する遅延）に特化した
拡張機能を提供する．
- データパケットのエージング (Aging): 寿命切れパケットの破棄
- 動的再送制御: RTT に基づく RTO と最大再送回数の動的計算
"""

import logging
import time
from math import ceil

from .rudp_protocol import (
    DEFAULT_MAX_RETRIES,
    DEFAULT_TIMEOUT,
    DEFAULT_WINDOW_SIZE,
    MAX_PAYLOAD_SIZE,
    Packet,
    PacketFlags,
    RUDPSocket,
    SendBufferEntry,
)

# =============================================================================
# 定数定義
# =============================================================================

DEFAULT_AGING_LIMIT = 0.5  # デフォルトパケット寿命 (500ms)
DEFAULT_MAX_CUMULATIVE_ACKS = 3  # 累積 ACK の最大数
MIN_RTO = 0.05  # 最小 RTO (50ms)
MAX_RTO = 5.0  # 最大 RTO (5秒)


# =============================================================================
# ERUDPSocket クラス
# =============================================================================


class ERUDPSocket(RUDPSocket):
    """
    E-RUDP (Extended RUDP) ソケットクラス．

    RUDPSocket を継承し，以下の拡張機能を追加する:
    1. データパケットのエージング (Aging): 一定時間経過したパケットを再送せず破棄
    2. 動的再送制御: RTT に基づいて RTO と最大再送回数を動的に計算
    """

    def __init__(
        self,
        timeout: float = DEFAULT_TIMEOUT,
        window_size: int = DEFAULT_WINDOW_SIZE,
        max_retries: int = DEFAULT_MAX_RETRIES,
        aging_limit: float = DEFAULT_AGING_LIMIT,
        max_cumulative_acks: int = DEFAULT_MAX_CUMULATIVE_ACKS,
    ):
        """
        ERUDPSocket を初期化する．

        Args:
            timeout: 接続/受信タイムアウト (秒)
            window_size: スライディングウィンドウサイズ
            max_retries: 最大再送回数（動的計算のフォールバック用）
            aging_limit: パケット寿命 (秒)．0.5 = 500ms
            max_cumulative_acks: 累積 ACK の最大数
        """
        super().__init__(timeout=timeout, window_size=window_size, max_retries=max_retries)
        self.logger = logging.getLogger("ERUDPSocket")

        # E-RUDP 固有設定（aging_limit は動的に決定するためフォールバック値として保持）
        self._base_aging_limit = aging_limit
        self._max_cumulative_acks = max_cumulative_acks

        # E-RUDP 統計
        self._stats["aged_packets"] = 0

    def send(self, data: bytes) -> int:
        """
        データを送信する（E-RUDP 拡張版）．

        Aging 機能により，送信バッファへの追加時にパケット生成時刻を記録する．

        Args:
            data: 送信するバイト列

        Returns:
            送信したバイト数

        Raises:
            RuntimeError: 接続が確立されていない場合
        """
        from .rudp_protocol import ConnectionState

        if self._state != ConnectionState.ESTABLISHED:
            raise RuntimeError("Connection not established")

        total_sent = 0
        offset = 0

        while offset < len(data):
            # ウィンドウに空きがあるまで待機
            while self._send_seq - self._send_base >= self._window_size:
                self._ack_event.wait(timeout=0.1)
                self._ack_event.clear()
                if not self._running:
                    raise RuntimeError("Connection closed")

            chunk = data[offset : offset + MAX_PAYLOAD_SIZE]
            packet = Packet(
                flags=PacketFlags.DATA,
                seq_num=self._send_seq,
                ack_num=self._recv_seq,
                payload=chunk,
                created_at=time.time(),  # パケット生成時刻を記録
            )

            with self._lock:
                self._send_buffer[self._send_seq] = SendBufferEntry(
                    packet=packet,
                    sent_time=time.time(),
                )
                self._send_seq += 1

            self._send_packet(packet)
            self._stats["bytes_sent"] += len(chunk)
            offset += len(chunk)
            total_sent += len(chunk)

        return total_sent

    def _check_retransmissions(self) -> None:
        """
        再送タイムアウトをチェックする（E-RUDP 拡張版）．

        Aging と動的再送制御を適用する．
        """

        current_time = time.time()

        with self._lock:
            aged_seqs = []

            for seq_num, entry in list(self._send_buffer.items()):
                if entry.acked:
                    continue

                packet = entry.packet
                packet_age = current_time - packet.created_at
                aging_limit = self._network_estimator.get_recommended_aging_limit()

                # 早期タイムアウト: aging limit の 70% でスキップ（スループット維持）
                if packet_age > aging_limit * 0.7:
                    aged_seqs.append(seq_num)
                    self._stats["aged_packets"] += 1
                    self.logger.debug(f"Packet seq {seq_num} early-aged (age: {packet_age:.3f}s > 70% of {aging_limit}s)")
                    continue

                # Aging チェック: パケット寿命が切れているか
                if self._check_packet_aging(packet, current_time):
                    aged_seqs.append(seq_num)
                    self._stats["aged_packets"] += 1
                    self.logger.debug(f"Packet seq {seq_num} aged out (age: {current_time - packet.created_at:.3f}s > limit: {self._aging_limit}s)")
                    continue

                # RTO を動的に計算
                rto = self._calculate_dynamic_rto()

                elapsed = current_time - entry.sent_time
                if elapsed > rto:
                    # 動的最大再送回数を計算
                    max_retries = self._calculate_dynamic_max_retries(packet, current_time)

                    if entry.retries >= max_retries:
                        # 最大再送回数到達 → このパケットは Aged として扱う
                        aged_seqs.append(seq_num)
                        self._stats["aged_packets"] += 1
                        self._stats["max_retries_reached"] += 1
                        self.logger.warning(f"Max retries ({max_retries}) reached for seq {seq_num}, marking as aged")
                        continue

                    # 再送
                    self._send_packet(entry.packet)
                    entry.sent_time = current_time
                    entry.retries += 1
                    self._stats["retransmissions"] += 1
                    self.logger.debug(f"Retransmit seq {seq_num} (retry {entry.retries}/{max_retries}, RTO={rto:.3f}s)")

            # Aged パケットを送信バッファから削除
            for seq_num in aged_seqs:
                self._send_buffer.pop(seq_num, None)

            # 送信ウィンドウ基底を更新（Aged パケットをスキップ）
            while self._send_base not in self._send_buffer and self._send_base < self._send_seq:
                self._send_base += 1
                self._ack_event.set()

    def _check_packet_aging(self, packet: Packet, current_time: float) -> bool:
        """
        パケットが寿命切れかどうかをチェックする（動的 Aging）．

        Args:
            packet: チェック対象のパケット
            current_time: 現在時刻

        Returns:
            寿命切れの場合 True
        """
        age = current_time - packet.created_at
        # 実測値から動的に寿命を決定
        adaptive_limit = self._network_estimator.get_recommended_aging_limit()
        return age > adaptive_limit

    def _calculate_dynamic_rto(self) -> float:
        """
        動的 RTO を計算する．

        累積 ACK の遅延を考慮した RTO 計算式:
        RTO = SRTT + 4 * RTTVAR + MaxCumulativeAcks * InterPacketDelay

        Returns:
            計算された RTO (秒)
        """
        if self._srtt == 0.0:
            # RTT 測定前はデフォルト値を使用
            return max(MIN_RTO, min(MAX_RTO, 0.5))

        # 基本 RTO (RFC 6298)
        base_rto = self._srtt + 4 * self._rttvar

        # 累積 ACK による追加遅延
        # InterPacketDelay は RTT を基準に推定（簡略化）
        inter_packet_delay = self._srtt * 0.1  # RTT の 10%
        ack_delay = self._max_cumulative_acks * inter_packet_delay

        rto = base_rto + ack_delay

        return max(MIN_RTO, min(MAX_RTO, rto))

    def _calculate_dynamic_max_retries(self, packet: Packet, current_time: float) -> int:
        """
        動的最大再送回数を計算する．

        パケットの寿命が尽きるまでに何回再送できるかを計算する:
        MaxRetries = ceil((aging_limit - elapsed) / RTO)

        Args:
            packet: 対象パケット
            current_time: 現在時刻

        Returns:
            計算された最大再送回数
        """
        elapsed = current_time - packet.created_at
        remaining_life = self._aging_limit - elapsed

        if remaining_life <= 0:
            return 0

        rto = self._calculate_dynamic_rto()
        if rto <= 0:
            return self._max_retries

        dynamic_retries = ceil(remaining_life / rto)

        # 最低 1 回，最大はベース設定値
        return max(1, min(dynamic_retries, self._max_retries))

    def get_stats(self) -> dict:
        """
        統計情報を取得する（E-RUDP 拡張版）．

        Returns:
            統計情報の辞書（aged_packets を含む）
        """
        stats = super().get_stats()
        with self._lock:
            stats["aged_packets"] = self._stats["aged_packets"]
        return stats
