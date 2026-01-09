"""
RUDP 接続プール．

接続を再利用してハンドシェイクオーバーヘッドを削減する．
"""

import logging
import threading
import time
from typing import Dict, Optional, Tuple

from .rudp_protocol import ConnectionState, RUDPSocket


class RUDPConnectionPool:
    """RUDP 接続プールクラス

    各ピアへの接続を保持し，再利用することでハンドシェイクを削減．
    """

    # 接続のアイドルタイムアウト (10分)
    IDLE_TIMEOUT = 600
    # 接続の最大保持数
    MAX_CONNECTIONS = 100

    def __init__(
        self,
        socket_factory,
        timeout: float = 5.0,
    ):
        """
        Args:
            socket_factory: ソケット作成関数 (引数なしで RUDPSocket を返す)
            timeout: 接続タイムアウト
        """
        self.logger = logging.getLogger("RUDPConnectionPool")
        self._socket_factory = socket_factory
        self._timeout = timeout

        # 接続プール: {(ip, port): {'socket': RUDPSocket, 'last_used': float}}
        self._connections: Dict[Tuple[str, int], dict] = {}
        self._lock = threading.Lock()

        # クリーンアップスレッド
        self._running = True
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_loop,
            daemon=True,
            name="RUDPConnectionPool-Cleanup",
        )
        self._cleanup_thread.start()

        self.logger.info("RUDPConnectionPool initialized")

    def get_connection(self, address: Tuple[str, int]) -> Optional[RUDPSocket]:
        """接続を取得（既存があれば再利用，なければ新規作成）

        Args:
            address: (ip, port) のタプル

        Returns:
            接続済みの RUDPSocket，または接続失敗時は None
        """
        with self._lock:
            # 既存接続を確認
            if address in self._connections:
                conn_info = self._connections[address]
                sock = conn_info["socket"]

                # 接続が生きているか確認
                if sock._state == ConnectionState.ESTABLISHED:
                    conn_info["last_used"] = time.time()
                    self.logger.debug(f"Reusing connection to {address}")
                    return sock
                else:
                    # 接続が切れている場合は削除
                    self.logger.debug(f"Stale connection to {address}, removing")
                    del self._connections[address]

        # 新規接続を作成
        sock = self._socket_factory()
        try:
            sock.connect(address)
            with self._lock:
                # プールが満杯の場合は最も古い接続を削除
                if len(self._connections) >= self.MAX_CONNECTIONS:
                    self._evict_oldest()

                self._connections[address] = {
                    "socket": sock,
                    "last_used": time.time(),
                }
            self.logger.info(f"New connection to {address} established")
            return sock
        except Exception as e:
            self.logger.error(f"Failed to connect to {address}: {e}")
            try:
                sock.close()
            except Exception:
                pass
            return None

    def release_connection(self, address: Tuple[str, int]) -> None:
        """接続を解放（プールに戻す）

        現在の実装では接続はプールに残り続けるため，
        このメソッドは last_used を更新するだけ．
        """
        with self._lock:
            if address in self._connections:
                self._connections[address]["last_used"] = time.time()

    def close_connection(self, address: Tuple[str, int]) -> None:
        """接続を明示的に閉じる"""
        with self._lock:
            if address in self._connections:
                conn_info = self._connections.pop(address)
                try:
                    conn_info["socket"].close()
                except Exception:
                    pass
                self.logger.debug(f"Connection to {address} closed")

    def _evict_oldest(self) -> None:
        """最も古い接続を削除（ロック保持中に呼び出すこと）"""
        if not self._connections:
            return

        oldest_addr = min(self._connections.keys(), key=lambda a: self._connections[a]["last_used"])
        conn_info = self._connections.pop(oldest_addr)
        try:
            conn_info["socket"].close()
        except Exception:
            pass
        self.logger.debug(f"Evicted oldest connection to {oldest_addr}")

    def _cleanup_loop(self) -> None:
        """アイドル接続をクリーンアップするスレッド"""
        while self._running:
            time.sleep(60)  # 1分ごとにチェック

            current_time = time.time()
            to_close = []

            with self._lock:
                for addr, conn_info in list(self._connections.items()):
                    idle_time = current_time - conn_info["last_used"]
                    if idle_time > self.IDLE_TIMEOUT:
                        to_close.append((addr, conn_info["socket"]))
                        del self._connections[addr]

            for addr, sock in to_close:
                try:
                    sock.close()
                    self.logger.debug(f"Closed idle connection to {addr}")
                except Exception:
                    pass

    def close_all(self) -> None:
        """全接続を閉じる"""
        self._running = False
        with self._lock:
            for addr, conn_info in list(self._connections.items()):
                try:
                    conn_info["socket"].close()
                except Exception:
                    pass
            self._connections.clear()
        self.logger.info("All connections closed")

    def get_stats(self) -> dict:
        """統計情報を取得"""
        with self._lock:
            return {
                "active_connections": len(self._connections),
                "addresses": list(self._connections.keys()),
            }
