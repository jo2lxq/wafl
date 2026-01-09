"""
RUDP/E-RUDP ベースのモデル共有クラス．

WAFL のモデル共有に特化した RUDP/E-RUDP ラッパーを提供する．
既存の UDPModelSharing と同様のインターフェースを持ち，
ModelSharingUtils から透過的に利用可能．
"""

import logging
import struct
import threading
import time
from typing import Callable, Dict, Optional

from .erudp_protocol import DEFAULT_AGING_LIMIT, ERUDPSocket
from .network_estimator import get_network_estimator
from .rudp_connection_pool import RUDPConnectionPool
from .rudp_protocol import (
    DEFAULT_MAX_RETRIES,
    DEFAULT_TIMEOUT,
    DEFAULT_WINDOW_SIZE,
    RUDPSocket,
)

# =============================================================================
# 定数定義
# =============================================================================

# メッセージタイプ
MSG_TYPE_MDLREQ = 0x01  # モデル要求
MSG_TYPE_MODEL = 0x02  # モデルデータ
MSG_TYPE_ACK = 0x03  # 確認応答

# メッセージヘッダサイズ: type(1) + length(4) = 5 bytes
MSG_HEADER_SIZE = 5


# =============================================================================
# RUDPModelSharing クラス
# =============================================================================


class RUDPModelSharing:
    """
    RUDP/E-RUDP ベースのモデル共有クラス．

    UDPModelSharing と同様のインターフェースを提供し，
    RUDP または E-RUDP を用いてモデルデータを信頼性高く転送する．
    """

    def __init__(
        self,
        ip: str,
        port: int,
        mode: str = "rudp",
        timeout: float = DEFAULT_TIMEOUT,
        aging_limit: float = DEFAULT_AGING_LIMIT,
        window_size: int = DEFAULT_WINDOW_SIZE,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ):
        """
        RUDPModelSharing を初期化する．

        Args:
            ip: ローカル IP アドレス
            port: ローカルポート番号
            mode: "rudp" または "erudp"
            timeout: 接続/受信タイムアウト (秒)
            aging_limit: E-RUDP のパケット寿命 (秒)
            window_size: スライディングウィンドウサイズ
            max_retries: 最大再送回数
        """
        self.logger = logging.getLogger("RUDPModelSharing")
        self.ip = ip
        self.port = port
        self.mode = mode.lower()
        self.timeout = timeout
        self.aging_limit = aging_limit
        self.window_size = window_size
        self.max_retries = max_retries

        self.running = False
        self.callback: Optional[Callable[[bytes, str], None]] = None
        self.mdlreq_callback: Optional[Callable[[str], None]] = None

        # サーバソケット
        self._server_socket: Optional[RUDPSocket] = None
        self._listener_thread: Optional[threading.Thread] = None

        # 受信済みモデルバッファ
        self._received_models: Dict[str, bytes] = {}
        self._received_models_lock = threading.Lock()

        # 統計情報
        self.stats = {
            "sent_models": 0,
            "sent_failed": 0,
            "received_models": 0,
            "bytes_sent": 0,
            "bytes_received": 0,
            "timeout_models": 0,
            "retransmissions": 0,
            "acks_sent": 0,
            "acks_received": 0,
            "eaks_sent": 0,
            "eaks_received": 0,
            "aged_packets": 0,
            "connect_time_ms": 0.0,
            "avg_rtt_ms": 0.0,
            "max_retries_reached": 0,
            "nacks_sent": 0,  # 追加
            "fec_recoveries": 0,  # 追加
        }

        self.logger.info(f"RUDPModelSharing initialized: mode={self.mode}, port={self.port}, timeout={self.timeout}s")
        if self.mode == "erudp":
            self.logger.info("  E-RUDP aging is dynamically adjusted")

        # ネットワーク推定器
        self._network_estimator = get_network_estimator()

        # 接続プール
        self._connection_pool = RUDPConnectionPool(
            socket_factory=self._create_socket,
            timeout=timeout,
        )

    def _create_socket(self) -> RUDPSocket:
        """モードに応じたソケットを作成する．"""
        if self.mode == "erudp":
            return ERUDPSocket(
                timeout=self.timeout,
                window_size=self.window_size,
                max_retries=self.max_retries,
                aging_limit=self.aging_limit,
            )
        else:
            return RUDPSocket(
                timeout=self.timeout,
                window_size=self.window_size,
                max_retries=self.max_retries,
            )

    def start_listener(self, callback: Callable[[bytes, str], None]) -> None:
        """
        モデル受信リスナーを開始する．

        Args:
            callback: モデル受信時に呼び出されるコールバック関数
                      callback(data: bytes, source_ip: str)
        """
        self.callback = callback
        self.running = True

        # サーバソケット作成
        self._server_socket = self._create_socket()
        self._server_socket.bind(("0.0.0.0", self.port))
        self._server_socket.listen()

        # リスナースレッド開始
        self._listener_thread = threading.Thread(
            target=self._listener_loop,
            daemon=True,
            name=f"RUDPModelSharing-Listener-{self.port}",
        )
        self._listener_thread.start()

        self.logger.info(f"RUDP Listener started on port {self.port}")

    def _listener_loop(self) -> None:
        """リスナーのメインループ．"""
        while self.running:
            try:
                # 接続を受け付け
                client_sock, addr = self._server_socket.accept(timeout=2.0)
                peer_ip = addr[0]

                # 接続ごとにハンドラスレッドを起動
                handler_thread = threading.Thread(
                    target=self._handle_connection,
                    args=(client_sock, peer_ip),
                    daemon=True,
                    name=f"RUDPModelSharing-Handler-{peer_ip}",
                )
                handler_thread.start()

            except TimeoutError:
                continue
            except Exception as e:
                if self.running:
                    self.logger.error(f"Listener error: {e}")

    def _handle_connection(self, sock: RUDPSocket, peer_ip: str) -> None:
        """接続を処理する．"""
        try:
            # メッセージヘッダを受信
            header_data = sock.recv(MSG_HEADER_SIZE, timeout=self.timeout)
            if len(header_data) < MSG_HEADER_SIZE:
                return

            msg_type, msg_length = struct.unpack("!BI", header_data)

            if msg_type == MSG_TYPE_MDLREQ:
                # モデル要求を処理
                self._handle_mdlreq(sock, peer_ip)

            elif msg_type == MSG_TYPE_MODEL:
                # モデルデータを受信
                self._handle_model_data(sock, peer_ip, msg_length)

        except Exception as e:
            self.logger.error(f"Connection handler error: {e}")
        finally:
            try:
                sock.close()
            except Exception:
                pass

    def _handle_mdlreq(self, sock: RUDPSocket, peer_ip: str) -> None:
        """モデル要求を処理する．

        MDLREQ を受信したら、同じ接続でモデルを直接送り返す。
        """
        self.logger.info(f"📬 RUDP MDLREQ received from {peer_ip}")

        if self.mdlreq_callback:
            try:
                # コールバックからモデルデータを取得
                model_data = self.mdlreq_callback(peer_ip)

                if model_data is not None and len(model_data) > 0:
                    # 同じ接続でモデルを直接送信
                    import struct

                    header = struct.pack("!BI", MSG_TYPE_MODEL, len(model_data))
                    self.logger.debug(f"📤 Starting to send model ({len(model_data)} bytes) to {peer_ip}")
                    try:
                        sock.send(header + model_data)
                        self.stats["sent_models"] += 1
                        self.stats["bytes_sent"] += len(model_data)
                        self.logger.info(f"📡 Sent model via RUDP ({len(model_data)} bytes) to {peer_ip} on same connection")
                    except Exception as send_error:
                        self.logger.error(f"❌ Failed to send model to {peer_ip}: {send_error}")
                else:
                    self.logger.warning(f"No model data returned from callback for {peer_ip}")

            except Exception as e:
                self.logger.error(f"MDLREQ callback error: {e}")

    def _handle_model_data(self, sock: RUDPSocket, peer_ip: str, msg_length: int) -> None:
        """モデルデータを受信する．"""
        try:
            # モデルデータを受信
            data = b""
            remaining = msg_length

            while remaining > 0:
                chunk = sock.recv(min(remaining, 65535), timeout=self.timeout)
                if not chunk:
                    break
                data += chunk
                remaining -= len(chunk)

            if len(data) == msg_length:
                self.logger.info(f"📦 Received model from {peer_ip} ({len(data)} bytes)")
                self.stats["received_models"] += 1
                self.stats["bytes_received"] += len(data)

                # ソケット統計を集約
                sock_stats = sock.get_stats()
                self._aggregate_stats(sock_stats)

                # コールバック呼び出し
                if self.callback:
                    self.callback(data, peer_ip)
            else:
                self.logger.warning(f"Incomplete model data: {len(data)}/{msg_length}")
                self.stats["timeout_models"] += 1

        except TimeoutError:
            self.logger.warning(f"Model receive timeout from {peer_ip}")
            self.stats["timeout_models"] += 1

    def set_mdlreq_callback(self, callback: Callable[[str], None]) -> None:
        """
        MDLREQ コールバックを設定する．

        Args:
            callback: モデル要求時に呼び出されるコールバック関数
                      callback(requester_ip: str)
        """
        self.mdlreq_callback = callback
        self.logger.debug("MDLREQ callback registered")

    def send_mdlreq(self, target_ip: str, target_port: int, requester_ip: str) -> bool:
        """
        モデル要求を送信し、モデルを受信する（接続プール使用）．

        Args:
            target_ip: 送信先 IP アドレス
            target_port: 送信先ポート番号
            requester_ip: 要求元 IP アドレス（現時点では未使用）

        Returns:
            成功した場合 True
        """
        address = (target_ip, target_port)
        sock = None
        success = False  # 成功フラグ初期化
        try:
            # 接続プールから接続を取得
            sock = self._connection_pool.get_connection(address)
            if sock is None:
                self.logger.error(f"Failed to get connection to {target_ip}:{target_port}")
                self.stats["timeout_models"] += 1  # 接続失敗もカウント
                self._network_estimator.record_packet_result(False)
                return False

            # MDLREQ メッセージを送信
            header = struct.pack("!BI", MSG_TYPE_MDLREQ, 0)
            sock.send(header)
            self.logger.debug(f"📤 Sent RUDP MDLREQ to {target_ip}:{target_port}")

            # 同じ接続でモデルを受信
            response_header = sock.recv(MSG_HEADER_SIZE, timeout=self.timeout)
            if len(response_header) < MSG_HEADER_SIZE:
                self.logger.warning(f"Incomplete header from {target_ip}")
                self.stats["timeout_models"] += 1  # ヘッダ不完全もカウント
                self._network_estimator.record_packet_result(False)
                return False

            msg_type, msg_length = struct.unpack("!BI", response_header)

            if msg_type != MSG_TYPE_MODEL:
                self.logger.warning(f"Unexpected message type {msg_type} from {target_ip}")
                self.stats["timeout_models"] += 1  # 予期しないメッセージもカウント
                return False

            # モデルデータを受信
            data = b""
            remaining = msg_length

            while remaining > 0:
                chunk = sock.recv(min(remaining, 65535), timeout=self.timeout)
                if not chunk:
                    break
                data += chunk
                remaining -= len(chunk)

            if len(data) == msg_length:
                self.logger.info(f"📦 Received model from {target_ip} ({len(data)} bytes) via MDLREQ response")
                self.stats["received_models"] += 1
                self.stats["bytes_received"] += len(data)

                # ソケット統計を集約
                sock_stats = sock.get_stats()
                self._aggregate_stats(sock_stats)

                # RTT を記録
                if sock_stats.get("avg_rtt_ms", 0) > 0:
                    self._network_estimator.record_rtt(sock_stats["avg_rtt_ms"])

                # 受信バッファに格納
                with self._received_models_lock:
                    self._received_models[target_ip] = data

                # コールバック呼び出し
                if self.callback:
                    self.callback(data, target_ip)

                self._network_estimator.record_packet_result(True)
                success = True  # 成功フラグ
                return True
            else:
                self.logger.warning(f"Incomplete model data: {len(data)}/{msg_length}")
                self.stats["timeout_models"] += 1
                self._network_estimator.record_packet_result(False)
                return False

        except TimeoutError:
            self.logger.warning(f"Model receive timeout from {target_ip}")
            self.stats["timeout_models"] += 1
            self._network_estimator.record_packet_result(False)
            return False
        except Exception as e:
            self.logger.error(f"Failed to send MDLREQ: {e}")
            self.stats["timeout_models"] += 1  # 例外もカウント
            self._network_estimator.record_packet_result(False)
            return False
        finally:
            if sock:
                # 成功した場合はプールに戻し、失敗した場合は破棄する
                if success:
                    self._connection_pool.release_connection(address)
                else:
                    self.logger.warning(f"Closing failed connection to {target_ip}:{target_port}")
                    self._connection_pool.close_connection(address)

    def send_model(self, model_data: bytes, target_ip: str, target_port: int) -> bool:
        """
        モデルデータを送信する（接続プール + FEC 使用）．

        Args:
            model_data: シリアライズされたモデルデータ
            target_ip: 送信先 IP アドレス
            target_port: 送信先ポート番号

        Returns:
            成功した場合 True
        """
        address = (target_ip, target_port)
        sock = None
        try:
            # 接続プールから接続を取得
            connect_start = time.time()
            sock = self._connection_pool.get_connection(address)
            if sock is None:
                self.logger.error(f"Failed to get connection to {target_ip}:{target_port}")
                self.stats["sent_failed"] += 1
                self._network_estimator.record_packet_result(False)
                return False
            connect_time = (time.time() - connect_start) * 1000

            # モデルデータメッセージを送信
            header = struct.pack("!BI", MSG_TYPE_MODEL, len(model_data))

            # FEC 送信を使用（利用可能な場合）
            if hasattr(sock, "send_with_fec"):
                sock.send(header)
                sock.send_with_fec(model_data)
            else:
                sock.send(header + model_data)

            # ソケット統計を集約
            sock_stats = sock.get_stats()
            self._aggregate_stats(sock_stats)
            self.stats["connect_time_ms"] = connect_time

            # RTT を記録
            if sock_stats.get("avg_rtt_ms", 0) > 0:
                self._network_estimator.record_rtt(sock_stats["avg_rtt_ms"])

            self.stats["sent_models"] += 1
            self.stats["bytes_sent"] += len(model_data)
            self._network_estimator.record_packet_result(True)

            self.logger.info(f"📡 Sent model via RUDP ({len(model_data)} bytes) to {target_ip}:{target_port} (connect: {connect_time:.1f}ms)")
            return True

        except Exception as e:
            self.logger.error(f"Failed to send model: {e}")
            self.stats["sent_failed"] += 1
            self._network_estimator.record_packet_result(False)
            return False
        finally:
            if sock:
                self._connection_pool.release_connection(address)

    def _aggregate_stats(self, sock_stats: dict) -> None:
        """ソケット統計を集約する．"""
        self.stats["retransmissions"] += sock_stats.get("retransmissions", 0)
        self.stats["acks_sent"] += sock_stats.get("acks_sent", 0)
        self.stats["acks_received"] += sock_stats.get("acks_received", 0)
        self.stats["eaks_sent"] += sock_stats.get("eaks_sent", 0)
        self.stats["eaks_received"] += sock_stats.get("eaks_received", 0)
        self.stats["aged_packets"] += sock_stats.get("aged_packets", 0)
        self.stats["max_retries_reached"] += sock_stats.get("max_retries_reached", 0)

        # RTT は平均を更新
        if sock_stats.get("avg_rtt_ms", 0) > 0:
            if self.stats["avg_rtt_ms"] == 0:
                self.stats["avg_rtt_ms"] = sock_stats["avg_rtt_ms"]
            else:
                # 指数移動平均
                self.stats["avg_rtt_ms"] = 0.8 * self.stats["avg_rtt_ms"] + 0.2 * sock_stats["avg_rtt_ms"]

    def get_stats(self) -> dict:
        """
        統計情報を取得する．

        Returns:
            統計情報の辞書
        """
        return self.stats.copy()

    def get_survival_rate(self) -> float:
        """
        モデル受信成功率を取得する．

        Returns:
            成功率 (0.0 - 1.0)
        """
        received = self.stats.get("received_models", 0)
        timeout = self.stats.get("timeout_models", 0)
        total = received + timeout

        if total == 0:
            return 1.0
        return received / total

    def stop(self) -> None:
        """リスナーを停止する．"""
        self.running = False

        if self._server_socket:
            try:
                self._server_socket.close()
            except Exception:
                pass

        if self._listener_thread and self._listener_thread.is_alive():
            self._listener_thread.join(timeout=2.0)

        self.logger.info("RUDPModelSharing stopped")
