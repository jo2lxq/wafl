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

                # 接続が生きているか確認（ESTABLISHED 状態のみ再利用可能）
                # CLOSE_WAIT 状態は相手側から FIN を受信済みで送信不可のため再利用しない
                is_healthy = sock._state == ConnectionState.ESTABLISHED and sock._running and sock.get_stats().get("max_retries_reached", 0) == 0

                if is_healthy:
                    conn_info["last_used"] = time.time()
                    self.logger.debug(f"Reusing connection to {address} (state={sock._state})")
                    return sock
                else:
                    # 接続が不健全な場合は削除して新規作成
                    self.logger.info(f"Unhealthy connection to {address}, recreating (state={sock._state}, running={sock._running})")
                    try:
                        sock.close()
                    except Exception:
                        pass
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

    def close_unhealthy(self, addresses: list[Tuple[str, int]]) -> int:
        """指定したアドレス群について、不健全な接続のみを閉じる。

        epoch 境界などで「使えるなら使い回し、使えないなら閉じる」を実現するために利用。

        Args:
            addresses: [(ip, port), ...]

        Returns:
            閉じた接続数
        """
        to_close: list[Tuple[Tuple[str, int], RUDPSocket, str]] = []
        with self._lock:
            for addr in addresses:
                if addr not in self._connections:
                    continue
                conn_info = self._connections[addr]
                sock: RUDPSocket = conn_info["socket"]

                is_healthy = sock._state == ConnectionState.ESTABLISHED and sock._running and sock.get_stats().get("max_retries_reached", 0) == 0

                if not is_healthy:
                    reason = f"unhealthy (state={sock._state}, running={sock._running})"
                    to_close.append((addr, sock, reason))
                    del self._connections[addr]

        for addr, sock, reason in to_close:
            try:
                sock.close()
            except Exception:
                pass
            self.logger.debug(f"Closed connection to {addr}: {reason}")

        return len(to_close)

    def prune_for_next_epoch(self, keep_addresses: set[Tuple[str, int]]) -> dict:
        """次 epoch で接続しないノードの接続を閉じ、併せて不健全な接続を閉じる。

        「使える」= 次 epoch でも同じノードに接続する（= keep_addresses に含まれる）こと。
        keep_addresses に含まれない接続は、健全であっても次 epoch で不要なので閉じる。

        Args:
            keep_addresses: 次 epoch で接続を維持したい (ip, port) の集合

        Returns:
            dict: {"kept": int, "closed_unhealthy": int, "closed_unneeded": int, "before": int, "after": int}
        """
        SAMPLE_SIZE = 5
        to_close: list[Tuple[Tuple[str, int], RUDPSocket, str]] = []
        kept_sample: list[str] = []
        closed_unhealthy_sample: list[str] = []
        closed_unneeded_sample: list[str] = []

        with self._lock:
            before = len(self._connections)
            for addr, conn_info in list(self._connections.items()):
                sock: RUDPSocket = conn_info["socket"]
                is_healthy = sock._state == ConnectionState.ESTABLISHED and sock._running and sock.get_stats().get("max_retries_reached", 0) == 0

                if not is_healthy:
                    reason = f"unhealthy (state={sock._state}, running={sock._running})"
                    to_close.append((addr, sock, reason))
                    if len(closed_unhealthy_sample) < SAMPLE_SIZE:
                        closed_unhealthy_sample.append(f"{addr[0]}:{addr[1]}")
                    del self._connections[addr]
                    continue

                if addr not in keep_addresses:
                    to_close.append((addr, sock, "unneeded for next epoch"))
                    if len(closed_unneeded_sample) < SAMPLE_SIZE:
                        closed_unneeded_sample.append(f"{addr[0]}:{addr[1]}")
                    del self._connections[addr]

            after = len(self._connections)
            for addr in list(self._connections.keys())[:SAMPLE_SIZE]:
                kept_sample.append(f"{addr[0]}:{addr[1]}")

        closed_unhealthy = sum(1 for _, _, r in to_close if r.startswith("unhealthy"))
        closed_unneeded = len(to_close) - closed_unhealthy

        for addr, sock, reason in to_close:
            try:
                sock.close()
            except Exception:
                pass
            self.logger.debug(f"Pruned connection to {addr}: {reason}")

        return {
            "before": before,
            "after": after,
            "kept": after,
            "closed_unhealthy": closed_unhealthy,
            "closed_unneeded": closed_unneeded,
            "kept_sample": kept_sample,
            "closed_unhealthy_sample": closed_unhealthy_sample,
            "closed_unneeded_sample": closed_unneeded_sample,
        }

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
        """アイドル接続および不健全な接続をクリーンアップするスレッド"""
        while self._running:
            time.sleep(5)  # 5秒ごとにチェック (監視強化)

            current_time = time.time()
            to_close = []

            with self._lock:
                for addr, conn_info in list(self._connections.items()):
                    sock = conn_info["socket"]
                    idle_time = current_time - conn_info["last_used"]

                    # 接続健全性チェック
                    is_healthy = sock._state == ConnectionState.ESTABLISHED and sock._running and sock.get_stats().get("max_retries_reached", 0) == 0

                    should_close = False
                    reason = ""

                    if not is_healthy:
                        should_close = True
                        reason = f"unhealthy (state={sock._state})"
                    elif idle_time > self.IDLE_TIMEOUT:
                        should_close = True
                        reason = f"idle ({idle_time:.1f}s)"

                    if should_close:
                        to_close.append((addr, sock, reason))
                        del self._connections[addr]

            for addr, sock, reason in to_close:
                try:
                    sock.close()
                    self.logger.debug(f"Closed connection to {addr}: {reason}")
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

    def warm_up(self, addresses: list) -> int:
        """
        【廃止】プリウォーミング機能は廃止された．

        この関数は後方互換性のために残されているが，何も行わない．
        接続は必要時にオンデマンドで確立される．

        Args:
            addresses: [(ip, port), ...] のリスト（無視される）

        Returns:
            常に 0
        """
        self.logger.debug("warm_up() is deprecated and does nothing")
        return 0
