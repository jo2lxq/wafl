"""
RUDP (Reliable UDP) プロトコル実装．

UDP の低遅延性と TCP の信頼性を両立する RUDP プロトコルの基底実装を提供する．
Selective Repeat ARQ によるパケットの信頼性確保と，スライディングウィンドウによる
効率的なデータ転送を実現する．
"""

import binascii
import logging
import math
import queue
import socket
import struct
import threading
import time
from dataclasses import dataclass, field
from enum import IntFlag
from typing import Dict, List, Optional, Tuple

import zfec

from .network_estimator import get_network_estimator

# =============================================================================
# 定数定義
# =============================================================================

# パケットヘッダサイズ: Flags(1) + HeaderLen(1) + SeqNum(4) + AckNum(4) + Checksum(4) + Reserved(6) = 20 bytes
HEADER_SIZE = 20
MAX_PACKET_SIZE = 1400  # MTU を考慮した安全なサイズ
MAX_PAYLOAD_SIZE = MAX_PACKET_SIZE - HEADER_SIZE

# デフォルト設定
DEFAULT_WINDOW_SIZE = 16
DEFAULT_TIMEOUT = 5.0
DEFAULT_MAX_RETRIES = 10
DEFAULT_RTO = 0.1  # 初期再送タイムアウト (秒) - ロス耐性向上
MAX_CUMULATIVE_ACKS = 3  # 累積 ACK の最大数


class PacketFlags(IntFlag):
    """
    パケットタイプ制御フラグ．

    複数のフラグを同時に設定可能（例: SYN | ACK で SYN+ACK パケット）．
    """

    NONE = 0x00
    SYN = 0x80  # 接続開始要求
    ACK = 0x40  # 確認応答
    EAK = 0x20  # 拡張確認応答（順序外受信時）
    RST = 0x10  # 強制切断
    NUL = 0x08  # キープアライブ
    FIN = 0x04  # 切断要求
    DATA = 0x02  # データパケット


class ConnectionState:
    """
    RUDP コネクションの状態．

    TCP ライクな状態遷移を管理する．
    """

    CLOSED = "CLOSED"
    LISTEN = "LISTEN"
    SYN_SENT = "SYN_SENT"
    SYN_RCVD = "SYN_RCVD"
    ESTABLISHED = "ESTABLISHED"
    FIN_WAIT_1 = "FIN_WAIT_1"
    FIN_WAIT_2 = "FIN_WAIT_2"
    CLOSE_WAIT = "CLOSE_WAIT"
    LAST_ACK = "LAST_ACK"
    TIME_WAIT = "TIME_WAIT"


# =============================================================================
# Packet クラス
# =============================================================================


@dataclass
class Packet:
    """
    RUDP パケットを表すデータクラス．

    パケット構造 (20 バイトヘッダ):
        | Flags (1B) | HeaderLen (1B) | SeqNum (4B) | AckNum (4B) |
        | Checksum (4B) | Reserved (6B) | Payload (可変) |

    Attributes:
        flags: パケットタイプ制御フラグ
        seq_num: シーケンス番号 (32bit)
        ack_num: 確認応答番号 (32bit)
        payload: ペイロードデータ
        checksum: CRC-32 チェックサム
        created_at: パケット生成時刻 (E-RUDP の Aging 用)
    """

    flags: int = PacketFlags.NONE
    seq_num: int = 0
    ack_num: int = 0
    payload: bytes = b""
    checksum: int = 0
    created_at: float = field(default_factory=time.time)
    # EAK 用: 順序外で受信したシーケンス番号のリスト
    eak_list: List[int] = field(default_factory=list)

    @staticmethod
    def calculate_checksum(data: bytes) -> int:
        """
        CRC-32 チェックサムを計算する．

        Args:
            data: チェックサム計算対象のバイト列

        Returns:
            32bit CRC-32 チェックサム値
        """
        return binascii.crc32(data) & 0xFFFFFFFF

    def pack(self) -> bytes:
        """
        パケットをバイト列にシリアライズする．

        Returns:
            シリアライズされたパケットのバイト列
        """
        header_len = HEADER_SIZE

        # EAK リストをペイロードに追加（EAK フラグが立っている場合）
        eak_payload = b""
        if self.flags & PacketFlags.EAK and self.eak_list:
            # EAK リストを 4 バイト整数の連続として格納
            eak_payload = struct.pack(f"!{len(self.eak_list)}I", *self.eak_list)

        full_payload = eak_payload + self.payload

        # チェックサム計算用のヘッダ（checksum フィールドは 0）
        header_for_checksum = struct.pack(
            "!BBII6s",
            self.flags,
            header_len,
            self.seq_num,
            self.ack_num,
            b"\x00" * 6,  # Reserved
        )

        # チェックサム計算（ヘッダ + ペイロード）
        checksum = self.calculate_checksum(header_for_checksum + full_payload)

        # 最終ヘッダ（チェックサムを含む）
        header = struct.pack(
            "!BBIIII2s",
            self.flags,
            header_len,
            self.seq_num,
            self.ack_num,
            checksum,
            0,  # Reserved (最初の 4 バイト)
            b"\x00" * 2,  # Reserved (残り 2 バイト)
        )

        return header + full_payload

    @classmethod
    def unpack(cls, data: bytes) -> Optional["Packet"]:
        """
        バイト列からパケットをデシリアライズする．

        Args:
            data: パケットのバイト列

        Returns:
            デシリアライズされた Packet オブジェクト，または検証失敗時は None
        """
        if len(data) < HEADER_SIZE:
            return None

        try:
            # ヘッダ解析
            flags, header_len, seq_num, ack_num, checksum, reserved1, reserved2 = struct.unpack("!BBIIII2s", data[:HEADER_SIZE])

            payload = data[HEADER_SIZE:]

            # チェックサム検証用のヘッダ
            header_for_checksum = struct.pack(
                "!BBII6s",
                flags,
                header_len,
                seq_num,
                ack_num,
                b"\x00" * 6,
            )

            expected_checksum = cls.calculate_checksum(header_for_checksum + payload)
            if checksum != expected_checksum:
                return None  # チェックサム不一致

            # EAK リストの解析
            eak_list = []
            actual_payload = payload
            if flags & PacketFlags.EAK:
                # EAK パケットの場合，ペイロードの先頭に EAK リストが含まれる
                # EAK リストの長さは seq_num フィールドで示される（暫定仕様）
                # TODO: より良い EAK リスト長の伝達方法を検討
                pass  # 現時点では EAK リストは受信側で再構築

            return cls(
                flags=flags,
                seq_num=seq_num,
                ack_num=ack_num,
                payload=actual_payload,
                checksum=checksum,
                eak_list=eak_list,
            )
        except struct.error:
            return None

    def verify_checksum(self) -> bool:
        """
        パケットのチェックサムを検証する．

        Returns:
            チェックサムが正しければ True，そうでなければ False
        """
        header_for_checksum = struct.pack(
            "!BBII6s",
            self.flags,
            HEADER_SIZE,
            self.seq_num,
            self.ack_num,
            b"\x00" * 6,
        )
        expected = self.calculate_checksum(header_for_checksum + self.payload)
        return self.checksum == expected

    def is_syn(self) -> bool:
        return bool(self.flags & PacketFlags.SYN)

    def is_ack(self) -> bool:
        return bool(self.flags & PacketFlags.ACK)

    def is_syn_ack(self) -> bool:
        return self.is_syn() and self.is_ack()

    def is_fin(self) -> bool:
        return bool(self.flags & PacketFlags.FIN)

    def is_rst(self) -> bool:
        return bool(self.flags & PacketFlags.RST)

    def is_data(self) -> bool:
        return bool(self.flags & PacketFlags.DATA)

    def is_eak(self) -> bool:
        return bool(self.flags & PacketFlags.EAK)

    def is_nul(self) -> bool:
        return bool(self.flags & PacketFlags.NUL)


# =============================================================================
# 送信バッファエントリ
# =============================================================================


@dataclass
class SendBufferEntry:
    """
    送信バッファのエントリ．

    再送管理に必要な情報を保持する．
    """

    packet: Packet
    sent_time: float
    last_retransmit_time: float = 0.0
    retries: int = 0
    acked: bool = False


# =============================================================================
# RUDPSocket クラス
# =============================================================================


class RUDPSocket:
    """
    RUDP ソケットクラス．

    TCP ライクな接続指向の通信を UDP 上で実現する．
    Selective Repeat ARQ とスライディングウィンドウによる信頼性確保を行う．
    """

    def __init__(
        self,
        timeout: float = DEFAULT_TIMEOUT,
        window_size: int = DEFAULT_WINDOW_SIZE,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ):
        """
        RUDPSocket を初期化する．

        Args:
            timeout: 接続/受信タイムアウト (秒)
            window_size: スライディングウィンドウサイズ
            max_retries: 最大再送回数
        """
        self.logger = logging.getLogger("RUDPSocket")

        # ソケット設定
        self._sock: Optional[socket.socket] = None
        self._timeout = timeout
        self._window_size = window_size
        self._max_retries = max_retries

        # 接続状態
        self._state = ConnectionState.CLOSED
        self._peer_addr: Optional[Tuple[str, int]] = None
        self._local_addr: Optional[Tuple[str, int]] = None

        # シーケンス番号管理
        self._send_seq = 0  # 次に送信するシーケンス番号
        self._recv_seq = 0  # 次に受信を期待するシーケンス番号
        self._send_base = 0  # 送信ウィンドウの基底

        # バッファ
        self._send_buffer: Dict[int, SendBufferEntry] = {}  # seq_num -> entry
        self._recv_buffer: Dict[int, Packet] = {}  # seq_num -> packet

        # RTT 計測
        self._srtt = 0.0  # 平滑化 RTT
        self._rttvar = 0.0  # RTT 分散
        self._rto = DEFAULT_RTO  # 再送タイムアウト

        # スレッド制御
        self._running = False
        self._lock = threading.RLock()
        self._receiver_thread: Optional[threading.Thread] = None
        self._timer_thread: Optional[threading.Thread] = None
        self._recv_queue: queue.Queue = queue.Queue()
        self._ack_event = threading.Event()

        # 累積 ACK 管理
        self._pending_acks: List[int] = []
        self._ack_timer: Optional[float] = None

        # 接続受付キュー（サーバ用）
        self._accept_queue: queue.Queue = queue.Queue()
        self._pending_connections: Dict[Tuple[str, int], dict] = {}  # ペンディング接続情報
        self._established_connections: Dict[Tuple[str, int], "RUDPSocket"] = {}  # 確立済み接続
        self._parent_sock: Optional[socket.socket] = None  # 親ソケット（子接続がデータ送信に使用）

        # 統計情報
        self._stats = {
            "retransmissions": 0,
            "acks_sent": 0,
            "acks_received": 0,
            "eaks_sent": 0,
            "eaks_received": 0,
            "packets_sent": 0,
            "packets_received": 0,
            "bytes_sent": 0,
            "bytes_received": 0,
            "connect_time_ms": 0.0,
            "avg_rtt_ms": 0.0,
            "max_retries_reached": 0,
            "fec_recoveries": 0,
            "nacks_sent": 0,
        }

        # Pacing control
        self._pacing_counter = 0

        # Fast Retransmit (RFC 5681)
        self._dup_ack_count = 0
        self._last_ack_num = 0

        # ネットワーク推定器
        self._network_estimator = get_network_estimator()

        # FEC 設定
        self._fec_enabled = True
        self._fec_k = 16  # データブロック数
        self._fec_parity = self._network_estimator.get_recommended_fec_parity(self._fec_k)
        self._fec_m = self._fec_k + self._fec_parity
        self._fec_encoder = zfec.Encoder(self._fec_k, self._fec_m)
        self._fec_decoder = zfec.Decoder(self._fec_k, self._fec_m)

        # FEC 受信バッファ
        self._fec_recv_buffer: Dict[int, Dict[int, bytes]] = {}  # msg_id -> {block_idx: data}
        self._fec_recv_meta: Dict[int, dict] = {}  # msg_id -> metadata

    # =========================================================================
    # ソケット API
    # =========================================================================

    def bind(self, address: Tuple[str, int]) -> None:
        """
        ソケットをアドレスにバインドする．

        Args:
            address: (ip, port) のタプル
        """
        if self._sock is not None:
            raise RuntimeError("Socket already bound")

        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 8 * 1024 * 1024)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1024 * 1024)
        self._sock.bind(address)
        self._local_addr = address
        self.logger.debug(f"Bound to {address}")

    def listen(self, backlog: int = 5) -> None:
        """
        接続待機状態に入る．

        Args:
            backlog: 接続待機キューの最大長（現時点では未使用）
        """
        if self._sock is None:
            raise RuntimeError("Socket not bound")

        self._state = ConnectionState.LISTEN
        self._start_threads()
        self.logger.info(f"Listening on {self._local_addr}")

    def accept(self, timeout: Optional[float] = None) -> Tuple["RUDPSocket", Tuple[str, int]]:
        """
        接続を受け付ける．

        Args:
            timeout: タイムアウト (秒)．None の場合はブロック．

        Returns:
            新しい RUDPSocket とピアアドレスのタプル

        Raises:
            TimeoutError: タイムアウト時
        """
        if self._state != ConnectionState.LISTEN:
            raise RuntimeError("Socket not in LISTEN state")

        try:
            new_sock, addr = self._accept_queue.get(timeout=timeout or self._timeout)
            return new_sock, addr
        except queue.Empty:
            raise TimeoutError("Accept timed out")

    def connect(self, address: Tuple[str, int]) -> None:
        """
        ピアに接続する．

        Args:
            address: 接続先の (ip, port)

        Raises:
            TimeoutError: 接続タイムアウト時
            ConnectionRefusedError: 接続拒否時
        """
        if self._sock is None:
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 8 * 1024 * 1024)
            self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1024 * 1024)
            # エフェメラルポートにバインド
            self._sock.bind(("", 0))
            self._local_addr = self._sock.getsockname()

        self._peer_addr = address
        self._state = ConnectionState.SYN_SENT

        # SYN 送信
        connect_start = time.time()
        syn_packet = Packet(flags=PacketFlags.SYN, seq_num=self._send_seq)
        self._send_packet(syn_packet)
        self._send_seq += 1

        # SYN+ACK 待機 (高速リトライ: 100ms timeout, max 20 retries = 2s)
        handshake_timeout = 0.1  # 100ms per attempt
        self._sock.settimeout(handshake_timeout)
        retries = 0

        while retries < self._max_retries:
            try:
                data, addr = self._sock.recvfrom(MAX_PACKET_SIZE)
                if addr != self._peer_addr:
                    continue

                response = Packet.unpack(data)
                if response is None:
                    continue

                if response.is_syn_ack():
                    # ACK 送信
                    self._recv_seq = response.seq_num + 1
                    ack_packet = Packet(
                        flags=PacketFlags.ACK,
                        seq_num=self._send_seq,
                        ack_num=self._recv_seq,
                    )
                    self._send_packet(ack_packet)
                    self._state = ConnectionState.ESTABLISHED
                    self._stats["connect_time_ms"] = (time.time() - connect_start) * 1000
                    self._start_threads()
                    self.logger.info(f"Connected to {address}")
                    return

                if response.is_rst():
                    raise ConnectionRefusedError("Connection refused")

            except socket.timeout:
                retries += 1
                self._stats["retransmissions"] += 1
                # SYN 再送
                syn_packet = Packet(flags=PacketFlags.SYN, seq_num=self._send_seq - 1)
                self._send_packet(syn_packet)

        self._state = ConnectionState.CLOSED
        raise TimeoutError("Connection timed out")

    def send(self, data: bytes) -> int:
        """
        データを送信する．

        Args:
            data: 送信するバイト列

        Returns:
            送信したバイト数

        Raises:
            RuntimeError: 接続が確立されていない場合
        """
        if self._state != ConnectionState.ESTABLISHED:
            raise RuntimeError("Connection not established")

        total_sent = 0
        offset = 0

        while offset < len(data):
            # ウィンドウに空きがあるまで待機
            while self._send_seq - self._send_base >= self._window_size:
                self._ack_event.wait(timeout=0.1)
                self._ack_event.clear()
                # タイムアウトチェック
                if not self._running:
                    raise RuntimeError("Connection closed")

            chunk = data[offset : offset + MAX_PAYLOAD_SIZE]
            packet = Packet(
                flags=PacketFlags.DATA,
                seq_num=self._send_seq,
                ack_num=self._recv_seq,
                payload=chunk,
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

            # Simple Pacing removed (Moved to _send_packet)

        return total_sent

    def send_with_fec(self, data: bytes) -> int:
        """
        FEC エンコードしてデータを送信する．

        Args:
            data: 送信するデータ

        Returns:
            送信したバイト数
        """
        if self._state != ConnectionState.ESTABLISHED:
            raise RuntimeError("Connection not established")

        if not self._fec_enabled:
            return self.send(data)

        # FEC パラメータを動的に更新
        self._update_fec_params()

        # ウィンドウサイズを動的に更新
        self._update_window_size()

        # データを k 個のブロックに分割
        block_size = math.ceil(len(data) / self._fec_k)
        padded_data = data + b"\x00" * (block_size * self._fec_k - len(data))

        blocks = [padded_data[i * block_size : (i + 1) * block_size] for i in range(self._fec_k)]

        # FEC エンコード
        encoded_blocks = self._fec_encoder.encode(blocks)

        # 各ブロックを送信
        msg_id = int(time.time() * 1000) % (2**32)
        total_sent = 0

        for block_idx, block in enumerate(encoded_blocks):
            # ペイロード: msg_id(4) + block_idx(2) + k(1) + m(1) + original_len(4) + block
            header = struct.pack("!IHBBI", msg_id, block_idx, self._fec_k, self._fec_m, len(data))
            payload = header + block

            # ウィンドウに空きがあるまで待機
            while self._send_seq - self._send_base >= self._window_size:
                self._ack_event.wait(timeout=0.1)
                self._ack_event.clear()
                if not self._running:
                    raise RuntimeError("Connection closed")

            packet = Packet(
                flags=PacketFlags.DATA,
                seq_num=self._send_seq,
                ack_num=self._recv_seq,
                payload=payload,
            )

            with self._lock:
                self._send_buffer[self._send_seq] = SendBufferEntry(
                    packet=packet,
                    sent_time=time.time(),
                )
                self._send_seq += 1

            self._send_packet(packet)
            self._stats["bytes_sent"] += len(payload)
            total_sent += len(block)

            # 送信結果を記録
            self._network_estimator.record_packet_result(True)

        return total_sent

    def _update_fec_params(self) -> None:
        """ネットワーク状態に基づいて FEC パラメータを更新"""
        new_parity = self._network_estimator.get_recommended_fec_parity(self._fec_k)

        if new_parity != self._fec_parity:
            old_parity = self._fec_parity
            self._fec_parity = new_parity
            self._fec_m = self._fec_k + self._fec_parity

            # エンコーダ/デコーダを再作成
            self._fec_encoder = zfec.Encoder(self._fec_k, self._fec_m)
            self._fec_decoder = zfec.Decoder(self._fec_k, self._fec_m)

            self.logger.info(f"FEC params updated: parity {old_parity} -> {new_parity} (m={self._fec_m}, redundancy={self._fec_parity / self._fec_m:.1%})")

    def _update_window_size(self) -> None:
        """実測値に基づいてウィンドウサイズを動的更新"""
        new_window = self._network_estimator.get_recommended_window_size()
        if new_window != self._window_size:
            old_window = self._window_size
            self._window_size = new_window
            self.logger.debug(f"Window size updated: {old_window} -> {new_window}")

    def _send_nack(self, missing_seqs: List[int]) -> None:
        """NACK パケットを送信（欠損シーケンス番号を通知）"""
        if not missing_seqs or self._peer_addr is None:
            return

        # NACK ペイロード: 欠損シーケンス番号のリスト (最大16個)
        payload = b"".join(struct.pack("!I", seq) for seq in missing_seqs[:16])

        nack_packet = Packet(
            flags=PacketFlags.EAK,  # EAK フラグを NACK として使用
            seq_num=self._recv_seq,
            ack_num=self._recv_seq,
            payload=payload,
        )
        self._send_packet(nack_packet)
        self._stats["nacks_sent"] += 1
        self.logger.debug(f"Sent NACK for seqs: {missing_seqs[:16]}")

    def recv(self, bufsize: int = 65535, timeout: Optional[float] = None) -> bytes:
        """
        データを受信する．

        ストリーム指向の受信を行い、要求されたバイト数を返す。
        内部でパケットペイロードをバッファリングする。

        Args:
            bufsize: 要求するバイト数
            timeout: タイムアウト (秒)

        Returns:
            受信したバイト列（要求されたバイト数、または利用可能なデータ）

        Raises:
            TimeoutError: タイムアウト時
            RuntimeError: 接続が閉じられた場合
        """
        if self._state not in (ConnectionState.ESTABLISHED, ConnectionState.CLOSE_WAIT):
            raise RuntimeError("Connection not established")

        # 内部バッファの初期化（最初の呼び出し時）
        if not hasattr(self, "_stream_buffer"):
            self._stream_buffer = b""

        effective_timeout = timeout if timeout is not None else self._timeout
        deadline = time.time() + effective_timeout

        # 要求されたバイト数を満たすまでデータを収集
        while len(self._stream_buffer) < bufsize:
            remaining_time = deadline - time.time()
            if remaining_time <= 0:
                # タイムアウト: 部分データを返す（またはエラー）
                if len(self._stream_buffer) > 0:
                    result = self._stream_buffer[:bufsize]
                    self._stream_buffer = self._stream_buffer[bufsize:]
                    return result
                raise TimeoutError("Recv timed out")

            try:
                # キューから1パケット分のペイロードを取得
                payload = self._recv_queue.get(timeout=min(remaining_time, 0.1))
                self._stream_buffer += payload
            except queue.Empty:
                # まだタイムアウトしていない場合は続行
                continue

        # 要求されたバイト数を返す
        result = self._stream_buffer[:bufsize]
        self._stream_buffer = self._stream_buffer[bufsize:]
        return result

    def close(self) -> None:
        """
        接続を閉じる．
        """
        if self._state == ConnectionState.CLOSED:
            return

        if self._state == ConnectionState.ESTABLISHED:
            # FIN 送信
            fin_packet = Packet(
                flags=PacketFlags.FIN,
                seq_num=self._send_seq,
                ack_num=self._recv_seq,
            )
            self._send_packet(fin_packet)
            self._state = ConnectionState.FIN_WAIT_1

            # FIN+ACK 待機（簡略化: タイムアウトで強制終了）
            time.sleep(0.5)

        self._running = False
        self._state = ConnectionState.CLOSED

        if self._receiver_thread and self._receiver_thread.is_alive():
            self._receiver_thread.join(timeout=1.0)
        if self._timer_thread and self._timer_thread.is_alive():
            self._timer_thread.join(timeout=1.0)

        if self._sock:
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None

        self.logger.debug("Connection closed")

    def get_stats(self) -> dict:
        """
        統計情報を取得する．

        Returns:
            統計情報の辞書
        """
        with self._lock:
            stats = self._stats.copy()
            stats["avg_rtt_ms"] = self._srtt * 1000
            return stats

    # =========================================================================
    # 内部メソッド
    # =========================================================================

    def _start_threads(self) -> None:
        """受信スレッドとタイマースレッドを開始する．"""
        # すでにスレッドが動作していれば何もしない
        # ただし、子ソケットの場合は _running=True で初期化されている可能性があるため
        # スレッドが未作成であることを確認する
        if self._receiver_thread is not None or self._timer_thread is not None:
            return

        self._running = True

        # 受信スレッドは、自前のソケットを持つ場合のみ開始
        if self._sock is not None:
            self._sock.settimeout(0.5)
            self._receiver_thread = threading.Thread(
                target=self._receiver_loop,
                daemon=True,
                name=f"RUDP-Receiver-{self._local_addr}",
            )
            self._receiver_thread.start()

        # タイマースレッドは常に開始（再送制御のため）
        self._timer_thread = threading.Thread(
            target=self._timer_loop,
            daemon=True,
            name=f"RUDP-Timer-{self._local_addr}",
        )
        self._timer_thread.start()

    def _send_packet(self, packet: Packet) -> None:
        """パケットを送信する．"""
        if self._peer_addr is None:
            return

        # 送信ソケットを決定: _sock がある場合はそれを使用、なければ _parent_sock を使用
        send_sock = self._sock if self._sock is not None else self._parent_sock
        if send_sock is None:
            return

        data = packet.pack()
        try:
            send_sock.sendto(data, self._peer_addr)
            self._stats["packets_sent"] += 1

            if packet.is_ack():
                self._stats["acks_sent"] += 1
            if packet.is_eak():
                self._stats["eaks_sent"] += 1

            # Pacing disabled: Exponential backoff handles congestion
            # if packet.is_data():
            #     with self._lock:
            #         self._pacing_counter += 1
            #         if self._pacing_counter % 2 == 0:
            #             time.sleep(0.002)

        except Exception as e:
            self.logger.error(f"Send error: {e}")

    def _receiver_loop(self) -> None:
        """受信スレッドのメインループ．"""
        while self._running:
            try:
                data, addr = self._sock.recvfrom(MAX_PACKET_SIZE)

                # LISTEN 状態の場合は接続要求を処理
                if self._state == ConnectionState.LISTEN:
                    self._handle_incoming_connection(data, addr)
                    continue

                # それ以外はピアからのパケットのみ処理
                if addr != self._peer_addr:
                    continue

                packet = Packet.unpack(data)
                if packet is None:
                    continue

                self._stats["packets_received"] += 1
                self._handle_packet(packet)

            except socket.timeout:
                continue
            except Exception as e:
                if self._running:
                    self.logger.error(f"Receiver error: {e}")

    def _handle_incoming_connection(self, data: bytes, addr: Tuple[str, int]) -> None:
        """接続要求を処理する（サーバ側）．

        新しいソケットを作成せず、元の LISTEN ソケットを再利用して通信する。
        これにより、ポート競合の問題を回避する。
        """
        packet = Packet.unpack(data)
        if packet is None:
            return

        if packet.is_syn() and not packet.is_ack():
            # 新規接続要求
            if addr in self._pending_connections:
                # 重複 SYN: SYN+ACK を再送
                pending_info = self._pending_connections[addr]
                syn_ack = Packet(
                    flags=PacketFlags.SYN | PacketFlags.ACK,
                    seq_num=pending_info["send_seq"],
                    ack_num=pending_info["recv_seq"],
                )
                self._sock.sendto(syn_ack.pack(), addr)
                return

            # 新しい接続情報を記録（新しいソケットは作成しない）
            recv_seq = packet.seq_num + 1
            send_seq = 0

            # SYN+ACK を元のソケットから直接送信
            syn_ack = Packet(
                flags=PacketFlags.SYN | PacketFlags.ACK,
                seq_num=send_seq,
                ack_num=recv_seq,
            )
            self._sock.sendto(syn_ack.pack(), addr)

            # ペンディング接続として記録
            self._pending_connections[addr] = {
                "send_seq": send_seq + 1,
                "recv_seq": recv_seq,
                "state": ConnectionState.SYN_RCVD,
            }
            self.logger.debug(f"SYN received from {addr}, sent SYN+ACK")

        elif packet.is_ack() and not packet.is_syn():
            # ACK パケット処理
            if addr in self._pending_connections:
                # ハンドシェイク完了 ACK
                pending_info = self._pending_connections.pop(addr)

                # 接続用の新しいソケットを作成
                # 注意: 新しいソケットは _receiver_loop を開始しない
                # 代わりに、LISTEN ソケットがデータを転送する
                new_sock = RUDPSocket(
                    timeout=self._timeout,
                    window_size=self._window_size,
                    max_retries=self._max_retries,
                )
                # 新しいソケットはソケットオブジェクトを持たない（データは転送される）
                new_sock._sock = None
                new_sock._local_addr = self._local_addr
                new_sock._peer_addr = addr
                new_sock._send_seq = pending_info["send_seq"]
                new_sock._recv_seq = pending_info["recv_seq"]
                new_sock._state = ConnectionState.ESTABLISHED
                new_sock._running = True  # 重要: send/recv が動作するために必要
                # 重要: 元のLISTENソケットへの参照を保持（データ送信用）
                new_sock._parent_sock = self._sock
                # スレッドを開始（タイマースレッドのみ起動される）
                new_sock._start_threads()

                # 確立済み接続として登録（データ転送用）
                self._established_connections[addr] = new_sock

                self._accept_queue.put((new_sock, addr))
                self.logger.info(f"Connection established with {addr}")

            elif addr in self._established_connections:
                # 確立済み接続への ACK（DATA への応答）
                conn = self._established_connections[addr]
                conn._stats["packets_received"] += 1
                conn._handle_ack(packet)
                conn._ack_event.set()  # 送信待機を解除

        # DATA パケットを確立済み接続に転送
        elif packet.is_data():
            if addr in self._established_connections:
                conn = self._established_connections[addr]
                conn._handle_data_from_parent(packet)
            elif addr in self._pending_connections:
                # ハンドシェイク完了前のデータは無視（または後で処理）
                pass

        # 確立済み接続への ACK/EAK/FIN パケット転送
        else:
            if addr in self._established_connections:
                conn = self._established_connections[addr]
                # ACK パケットを転送（送信確認用）
                if packet.is_ack():
                    conn._stats["packets_received"] += 1
                    conn._handle_ack(packet)
                    conn._ack_event.set()  # 送信待機を解除
                # EAK パケットを転送
                if packet.is_eak():
                    conn._stats["packets_received"] += 1
                    conn._handle_eak(packet)
                # FIN パケットを転送
                if packet.is_fin():
                    conn._stats["packets_received"] += 1
                    conn._handle_fin(packet)

    def _handle_packet(self, packet: Packet) -> None:
        """受信パケットを処理する．"""
        # ACK 処理
        if packet.is_ack():
            self._handle_ack(packet)

        # EAK 処理
        if packet.is_eak():
            self._handle_eak(packet)

        # DATA 処理
        if packet.is_data():
            self._handle_data(packet)

        # FIN 処理
        if packet.is_fin():
            self._handle_fin(packet)

        # RST 処理
        if packet.is_rst():
            self._running = False
            self._state = ConnectionState.CLOSED

    def _handle_ack(self, packet: Packet) -> None:
        """ACK を処理する．"""
        ack_num = packet.ack_num
        self._stats["acks_received"] += 1

        with self._lock:
            # Fast Retransmit: 同じ ACK 番号が 3 回来たら即座に再送
            if ack_num == self._last_ack_num:
                self._dup_ack_count += 1
                if self._dup_ack_count >= 3:
                    # 3 重複 ACK: 欠落パケットを即座に再送
                    if self._send_base in self._send_buffer:
                        entry = self._send_buffer[self._send_base]
                        self._send_packet(entry.packet)
                        entry.sent_time = time.time()
                        entry.retries += 1
                        self._stats["retransmissions"] += 1
                        self.logger.debug(f"Fast retransmit seq {self._send_base}")
                    self._dup_ack_count = 0
            else:
                self._dup_ack_count = 0
                self._last_ack_num = ack_num

            # 累積 ACK: ack_num までのパケットを確認
            acked_seqs = [seq for seq in self._send_buffer if seq < ack_num]
            for seq in acked_seqs:
                entry = self._send_buffer.pop(seq, None)
                if entry and not entry.acked:
                    # RTT 更新
                    rtt = time.time() - entry.sent_time
                    self._update_rtt(rtt)

            # 送信ウィンドウ基底を更新
            if acked_seqs:
                self._send_base = max(self._send_base, max(acked_seqs) + 1)
                self._ack_event.set()

    def _handle_eak(self, packet: Packet) -> None:
        """EAK (拡張 ACK) を処理する．"""
        self._stats["eaks_received"] += 1
        # EAK に含まれる欠落シーケンス番号のパケットを再送
        # 現時点では簡略化: EAK を受け取ったら直前のパケットを再送
        with self._lock:
            if self._send_base in self._send_buffer:
                entry = self._send_buffer[self._send_base]

                # Fast Retransmit Dampening: 短期間の連続再送を防ぐ
                current_time = time.time()
                if current_time - entry.last_retransmit_time < 0.1:  # 100ms
                    return

                self._send_packet(entry.packet)
                # entry.sent_time = current_time  # RTO タイマーはリセットしない（RFC 6298）
                entry.last_retransmit_time = current_time
                entry.retries += 1
                self._stats["retransmissions"] += 1

    def _handle_data_from_parent(self, packet: Packet) -> None:
        """
        親ソケットから転送された DATA パケットを処理する．

        Args:
            packet: 受信したパケット
        """
        self._stats["packets_received"] += 1
        self._handle_data(packet)

    def _handle_data(self, packet: Packet) -> None:
        """データパケットを処理する．"""
        seq_num = packet.seq_num

        with self._lock:
            if seq_num == self._recv_seq:
                # 期待通りのシーケンス番号
                self._recv_queue.put(packet.payload)
                self._stats["bytes_received"] += len(packet.payload)
                self._recv_seq += 1

                # バッファ内の連続パケットも処理
                while self._recv_seq in self._recv_buffer:
                    buffered = self._recv_buffer.pop(self._recv_seq)
                    self._recv_queue.put(buffered.payload)
                    self._stats["bytes_received"] += len(buffered.payload)
                    self._recv_seq += 1

                # ACK 送信
                self._send_ack()

            elif seq_num > self._recv_seq:
                # 順序外パケット: バッファに保存
                if seq_num not in self._recv_buffer:
                    self._recv_buffer[seq_num] = packet

                # EAK 送信（欠落を通知）
                self._send_eak()

            # seq_num < self._recv_seq の場合は重複パケット: ACK を再送して送信側に通知
            else:
                self._send_ack()

    def _handle_fin(self, packet: Packet) -> None:
        """FIN を処理する．"""
        # ACK 送信
        ack_packet = Packet(
            flags=PacketFlags.ACK | PacketFlags.FIN,
            seq_num=self._send_seq,
            ack_num=packet.seq_num + 1,
        )
        self._send_packet(ack_packet)

        if self._state == ConnectionState.ESTABLISHED:
            self._state = ConnectionState.CLOSE_WAIT
        elif self._state == ConnectionState.FIN_WAIT_1:
            self._state = ConnectionState.TIME_WAIT
            # TIME_WAIT 後に CLOSED へ遷移
            threading.Timer(1.0, self._close_after_time_wait).start()

    def _close_after_time_wait(self) -> None:
        """TIME_WAIT 後にソケットを閉じる．"""
        self._running = False
        self._state = ConnectionState.CLOSED

    def _send_ack(self) -> None:
        """ACK を送信する．"""
        ack_packet = Packet(
            flags=PacketFlags.ACK,
            seq_num=self._send_seq,
            ack_num=self._recv_seq,
        )
        self._send_packet(ack_packet)

    def _send_eak(self) -> None:
        """EAK (拡張 ACK) を送信する．"""
        # 欠落しているシーケンス番号を収集
        missing = []
        for i in range(self._recv_seq, max(self._recv_buffer.keys()) + 1 if self._recv_buffer else self._recv_seq):
            if i not in self._recv_buffer and i != self._recv_seq:
                missing.append(i)

        eak_packet = Packet(
            flags=PacketFlags.ACK | PacketFlags.EAK,
            seq_num=self._send_seq,
            ack_num=self._recv_seq,
            eak_list=missing[:10],  # 最大 10 個まで
        )
        self._send_packet(eak_packet)
        self._stats["eaks_sent"] += 1

    def _timer_loop(self) -> None:
        """タイマースレッドのメインループ．"""
        while self._running:
            time.sleep(0.025)  # 25ms 間隔でチェック（高速回復用）
            self._check_retransmissions()

    def _check_retransmissions(self) -> None:
        """再送タイムアウトをチェックする．"""
        current_time = time.time()

        with self._lock:
            for seq_num, entry in list(self._send_buffer.items()):
                if entry.acked:
                    continue

                elapsed = current_time - entry.sent_time
                if elapsed > self._rto:
                    if entry.retries >= self._max_retries:
                        # 最大再送回数到達
                        self._stats["max_retries_reached"] += 1
                        self.logger.warning(f"Max retries reached for seq {seq_num}")
                        self._running = False
                        self._state = ConnectionState.CLOSED
                        return

                    # 再送
                    self._send_packet(entry.packet)
                    entry.sent_time = current_time
                    entry.retries += 1
                    self._stats["retransmissions"] += 1
                    self.logger.debug(f"Retransmit seq {seq_num} (retry {entry.retries})")

                    # RFC 6298: Exponential Backoff - RTO を 2 倍にする
                    self._rto = min(10.0, self._rto * 2)

    def _update_rtt(self, rtt: float) -> None:
        """RTT を更新する（RFC 6298）．"""
        if self._srtt == 0.0:
            # 初回
            self._srtt = rtt
            self._rttvar = rtt / 2
        else:
            # 更新
            alpha = 0.125
            beta = 0.25
            self._rttvar = (1 - beta) * self._rttvar + beta * abs(self._srtt - rtt)
            self._srtt = (1 - alpha) * self._srtt + alpha * rtt

        # RTO 計算（最小 0.025 秒，最大 10 秒 - 高速回復用）
        self._rto = max(0.025, min(10.0, self._srtt + 4 * self._rttvar))
