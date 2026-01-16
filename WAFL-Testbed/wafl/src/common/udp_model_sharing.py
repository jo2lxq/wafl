import collections
import logging
import math
import socket
import struct
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Empty, Queue

import zfec

from .network_estimator import get_network_estimator


class UDPModelSharing:
    """
    Handles UDP-based model sharing with Forward Error Correction (FEC).
    """

    DEFAULT_MAX_PACKET_SIZE = 1400  # Safe MTU size
    # Header: type(1), timestamp(8), model_seq(4), chunk_idx(4), total_chunks(4), block_idx(4), original_len(4), k(1), m(1), pad(1) = 32 bytes
    HEADER_SIZE = 32

    # Packet Types
    PTYPE_DATA = 0
    PTYPE_NACK = 1
    PTYPE_MCAST = 2
    PTYPE_ACK = 3
    PTYPE_END = 4  # End of Transfer (Fast NACK trigger)
    PTYPE_MDLREQ = 5  # Model Request (UDP-based MDLREQ)
    PTYPE_ABORT = 6  # Abort signal (receiver timed out)

    MAX_RETRIES = 10  # Max NACK attempts
    BATCH_SIZE = 16  # Number of packets to send before sleeping (Batch Pacing)

    def __init__(
        self,
        ip: str,
        port: int,
        fec_m: int,
        timeout: float,
        inter_packet_timeout: float,
        max_packet_size: int = None,
        fast_mode: bool = False,
        # Ablation Study 用パラメータ
        fec_enabled: bool = True,
        fec_mode: str = "adaptive",  # "adaptive" | "fixed"
        fec_fixed_redundancy: float = 0.0625,  # 固定冗長率 (6.25%)
        nack_enabled: bool = True,
        dynamic_mode: bool = False,  # Dynamic モードフラグ
    ):
        self.ip = ip
        self.port = port

        # Packet sizing
        if max_packet_size is None:
            max_packet_size = self.DEFAULT_MAX_PACKET_SIZE
        # Clamp to a conservative range to avoid fragmentation on common paths
        max_packet_size = int(max(800, min(1472, max_packet_size)))
        self.MAX_PACKET_SIZE = max_packet_size
        self.PAYLOAD_SIZE = self.MAX_PACKET_SIZE - self.HEADER_SIZE

        # Base FEC Settings
        # k: data packets per chunk
        self.k = fec_m if fec_m > 0 else 16  # Default k=16 if not specified

        # Ablation Study: FEC 制御パラメータ
        self.fec_enabled = fec_enabled
        self.fec_mode = fec_mode
        self.fec_fixed_redundancy = fec_fixed_redundancy

        # Ablation Study: NACK 制御パラメータ
        self.nack_enabled = nack_enabled

        # Network Estimator for dynamic parameters
        self._network_estimator = get_network_estimator()

        # Adaptive Timeout default (will be adjusted per peer transfer potentially, but hard to do per-packet)
        self.timeout = max(timeout, 2.0)
        self.inter_packet_timeout = max(inter_packet_timeout, 0.1)

        self.logger = logging.getLogger("UDPModelSharing")

        # Fast mode: disables NACK retries, discards incomplete models immediately
        self.fast_mode = fast_mode

        # バッチサイズ動的化: Fast モードまたは NACK 無効時に適用
        if self.fast_mode or not self.nack_enabled:
            # Adaptive Batch Size: Adjust based on estimated bandwidth
            metrics = self._network_estimator.get_metrics()
            bandwidth = metrics.bandwidth_mbps
            if bandwidth < 2.0:  # Poor (~1Mbit)
                self.BATCH_SIZE = 16
            elif bandwidth < 10.0:  # Fair (~5Mbit)
                self.BATCH_SIZE = 32
            else:  # Good/Excellent
                self.BATCH_SIZE = 64
            self.logger.info(f"🚀 Dynamic batch size: BATCH_SIZE={self.BATCH_SIZE} (bandwidth={bandwidth:.1f}Mbps)")
        elif dynamic_mode:
            # Dynamic モード: NetworkEstimator の品質レベルに基づく調整
            metrics = self._network_estimator.get_metrics()
            quality = metrics.get_quality_level()
            if quality == "poor":
                self.BATCH_SIZE = 16
            elif quality == "fair":
                self.BATCH_SIZE = 32
            elif quality == "good":
                self.BATCH_SIZE = 48
            else:  # excellent
                self.BATCH_SIZE = 64
            self.logger.info(f"⚖️ Dynamic mode batch size: BATCH_SIZE={self.BATCH_SIZE} (quality={quality})")

        self.MAX_FEC_M = 256  # zfec limit

        # Encoder Cache: (k, m) -> zfec.Encoder
        self._encoder_cache = {}

        # FEC and Survival Rate Statistics
        self.stats = {
            "sent_models": 0,
            "sent_failed": 0,
            "received_models": 0,
            "fec_recovery_success": 0,
            "fec_recovery_fail": 0,
            "total_chunks_received": 0,
            "chunks_decoded": 0,
            "failed_chunks": 0,
            "timeout_models": 0,
            "bytes_sent": 0,
            "bytes_received": 0,
            "app_bytes_sent": 0,
            "fec_encode_time_ms": 0.0,
            "fec_decode_time_ms": 0.0,
            # Dynamic parameter tracking (sums for averaging)
            "sum_parity": 0,
            "sum_pacing_delay": 0.0,
            "total_transfers": 0,
            # Fast mode statistics
            "fast_mode_discarded": 0,
            "aborted_sends": 0,  # Sends aborted by receiver timeout
            # Ablation Study: NACK 無効時の破棄カウント
            "nack_disabled_discards": 0,
        }

        # Early abort tracking: {target_ip: True} when receiver sends ABORT
        self.abort_peers = {}
        self.abort_lock = threading.Lock()

        # Monotonically increasing model sequence number
        self.model_seq = 0
        self.seq_lock = threading.Lock()

        # Cache for ARQ (Retransmission)
        # {model_seq: {'chunks': chunks_data, 'timestamp': ts, 'k': k, 'm': m, 'encoder': encoder}}
        self.sent_models_cache = {}
        self.sent_cache_lock = threading.Lock()

        # Multicast Settings
        self.mcast_ttl = 2

        # Log initialization with Ablation parameters
        ablation_info = f"fec_enabled={self.fec_enabled}, fec_mode={self.fec_mode}, nack_enabled={self.nack_enabled}"
        self.logger.info(f"UDPModelSharing initialized: max_packet={self.MAX_PACKET_SIZE}B, k={self.k}, {ablation_info}")

        # Encode cache (single-entry)
        # Keyed by (id(model_data), k, m, payload_size).
        self._last_encoded_model_id = None
        self._last_encoded_params = None
        self._last_encoded_chunks = None
        self._encode_cache_hits = 0
        self._encode_cache_misses = 0

        # NACK Thread Pool
        self.nack_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="NACK")

        # Per-peer worker threads
        self.peer_queues = {}  # {peer_ip: Queue}
        self.peer_workers = {}  # {peer_ip: Thread}
        self.peer_workers_lock = threading.Lock()
        self.callback = None
        self.mdlreq_callback = None
        self.mdlreq_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="MDLREQ")
        self.encode_executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="FEC-Encode")
        self.stats_lock = threading.Lock()

        # Activity tracking for smart retransmission
        self.peer_last_activity = {}  # {peer_ip: timestamp}

    def get_last_peer_activity(self, peer_ip: str) -> float:
        """Return the timestamp of the last packet received from this peer."""
        return self.peer_last_activity.get(peer_ip, 0.0)

    def _get_encoder(self, k: int, m: int):
        """Get or create cached encoder."""
        if (k, m) not in self._encoder_cache:
            self._encoder_cache[(k, m)] = zfec.Encoder(k, m)
        return self._encoder_cache[(k, m)]

    def _get_params_for_peer(self, peer_ip: str) -> tuple[int, int, float]:
        """Get FEC parity, total m, and pacing delay for a specific peer."""
        # Ablation Study: FEC 無効時は冗長度 0
        if not self.fec_enabled:
            pacing = self._network_estimator.get_recommended_pacing_delay(peer_ip)
            return 0, self.k, pacing  # m = k (NO-FEC mode)

        # Ablation Study: FEC 固定モード
        if self.fec_mode == "fixed":
            parity = max(1, int(self.k * self.fec_fixed_redundancy))
            self.logger.debug(f"🎯 Fixed FEC: redundancy={self.fec_fixed_redundancy * 100:.1f}%, parity={parity} (k={self.k})")
        else:
            # 適応モード（既存ロジック）
            parity = self._network_estimator.get_recommended_fec_parity(self.k, peer_ip)

        # Fast / Ablation (FEC有効) モード: ロス率×2 + 最低15% + 安全マージン
        # FEC で確実に回復できるようにするため、追加マージンを設ける
        if self.fast_mode or (self.fec_enabled and not self.nack_enabled):
            metrics = self._network_estimator.get_metrics(peer_ip)
            loss_rate = metrics.packet_loss_rate

            # FEC parity: ロス率×2 + 最低15% + 安全マージン
            base_parity = int(self.k * loss_rate * 2.0)  # ロス率の2倍
            min_parity = int(self.k * 0.15)  # 最低15%
            parity = max(base_parity, min_parity) + 1  # +1 安全マージン

            # 高ロス補正 (5%以上で+1)
            if loss_rate > 0.05:
                parity += 1

            self.logger.debug(f"🎯 FEC (loss×2 + min15% + margin): loss={loss_rate * 100:.1f}%, parity={parity} (k={self.k})")

        m = self.k + parity

        if self.MAX_FEC_M:
            m = min(m, self.MAX_FEC_M)
            parity = m - self.k

        pacing = self._network_estimator.get_recommended_pacing_delay(peer_ip)
        return parity, m, pacing

    def send_mdlreq(self, target_ip: str, target_port: int, requester_ip: str) -> bool:
        """Send a UDP MDLREQ (Model Request) to a peer."""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.settimeout(0.2)

            requester_bytes = requester_ip.encode("utf-8")
            requester_bytes = requester_bytes[:32].ljust(32, b"\0")

            req_ts = time.time()
            header = struct.pack("!BdIIIIIBBB", self.PTYPE_MDLREQ, req_ts, 0, 0, 0, 0, 0, 0, 0, 0)
            packet = header + requester_bytes

            sock.sendto(packet, (target_ip, target_port))

            # RTT probe
            try:
                data, _ = sock.recvfrom(self.MAX_PACKET_SIZE)
                if len(data) >= self.HEADER_SIZE:
                    ack_header = data[: self.HEADER_SIZE]
                    ptype, echoed_ts, model_seq, *_ = struct.unpack("!BdIIIIIBBB", ack_header)
                    if ptype == self.PTYPE_ACK and abs(echoed_ts - req_ts) < 0.5:
                        rtt_sec = time.time() - echoed_ts
                        if 0 < rtt_sec < 5.0:
                            self._network_estimator.record_rtt(rtt_sec * 1000, target_ip)
            except socket.timeout:
                pass
            sock.close()

            self.logger.debug(f"📬 Sent UDP MDLREQ to {target_ip}:{target_port} from {requester_ip}")
            return True

        except Exception as e:
            self.logger.error(f"❌ Failed to send UDP MDLREQ: {e}")
            return False

    def send_abort(self, target_ip: str, target_port: int) -> None:
        """Send ABORT signal to sender (receiver timed out, stop sending)."""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            # Use minimal packet: just the packet type byte
            header = struct.pack("!BdIIIIIBBB", self.PTYPE_ABORT, time.time(), 0, 0, 0, 0, 0, 0, 0, 0)
            sock.sendto(header, (target_ip, target_port))
            sock.close()
            self.logger.debug(f"🛑 Sent ABORT to {target_ip}:{target_port}")
        except Exception as e:
            self.logger.warning(f"Failed to send ABORT: {e}")

    def send_model(self, model_data: bytes, target_ip: str, target_port: int) -> bool:
        """
        Sends serialized model data via UDP with FEC and Pacing using Per-Peer Optimization.
        """
        try:
            transfer_start = time.time()

            # --- Per-Peer Parameter Selection ---
            parity, m, pacing_delay = self._get_params_for_peer(target_ip)

            # Update accumulation stats for dynamic parameter analysis
            self.stats["sum_parity"] += parity
            self.stats["sum_pacing_delay"] += pacing_delay
            self.stats["total_transfers"] += 1

            # Get encoder for this specific configuration
            encoder = None
            if parity > 0:
                encoder = self._get_encoder(self.k, m)

            metrics = self._network_estimator.get_metrics(target_ip)
            self.logger.debug(f"🎯 sending to {target_ip}: k={self.k}, parity={parity}, pacing={pacing_delay * 1000:.1f}ms (Loss: {metrics.packet_loss_rate:.1%})")

            # ====== FEC BYPASS MODE (parity=0) ======
            if parity == 0:
                # Still pass pacing_delay
                return self._send_model_no_fec(model_data, target_ip, target_port, pacing_delay)

            # ====== FEC MODE ======
            chunk_size = self.PAYLOAD_SIZE * self.k
            total_chunks = math.ceil(len(model_data) / chunk_size)

            # Pre-calculate chunks
            chunks_data = []
            for chunk_idx in range(total_chunks):
                start = chunk_idx * chunk_size
                end = min(start + chunk_size, len(model_data))
                chunk = model_data[start:end]

                block_size = math.ceil(len(chunk) / self.k)
                chunk_padded = chunk + b"\0" * (block_size * self.k - len(chunk))

                blocks_in = [chunk_padded[i * block_size : (i + 1) * block_size] for i in range(self.k)]
                chunks_data.append((chunk_idx, blocks_in, len(chunk)))

            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4 * 1024 * 1024)

            if self._is_multicast(target_ip):
                sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, self.mcast_ttl)

            current_seq = 0
            with self.seq_lock:
                self.model_seq += 1
                current_seq = self.model_seq

            actual_bytes_sent = 0
            fec_encode_time_ms = 0.0

            # Cache key includes 'm' which might vary per peer
            cache_key = (id(model_data), int(self.k), int(m), int(self.PAYLOAD_SIZE))
            use_cache = self._last_encoded_model_id == cache_key[0] and self._last_encoded_params == cache_key[1:] and self._last_encoded_chunks is not None

            if use_cache:
                self._encode_cache_hits += 1
            else:
                self._encode_cache_misses += 1

            encoded_chunks = None
            if use_cache:
                encoded_chunks = self._last_encoded_chunks
            else:
                # Define task properly binding local encoder variable (though zfec encoder is essentially stateless functional)
                def _encode_task(chunk_idx: int, blocks_in: list[bytes], original_len: int):
                    t0 = time.time()
                    blocks_out = encoder.encode(blocks_in)
                    return chunk_idx, original_len, blocks_out, (time.time() - t0) * 1000

                encoded_by_idx = {}
                encoded_by_idx = {}
                # Use persistent executor
                future_to_chunk = {}
                for chunk_idx, blocks_in, original_len in chunks_data:
                    future = self.encode_executor.submit(_encode_task, chunk_idx, blocks_in, original_len)
                    future_to_chunk[future] = chunk_idx

                packets_sent_in_batch = 0
                for future in as_completed(future_to_chunk):
                    chunk_idx, original_len, blocks_out, enc_ms = future.result()
                    fec_encode_time_ms += float(enc_ms)
                    encoded_by_idx[chunk_idx] = (original_len, blocks_out)

                    for block_idx, block in enumerate(blocks_out):
                        header = struct.pack("!BdIIIIIBBB", self.PTYPE_DATA, time.time(), current_seq, chunk_idx, total_chunks, block_idx, original_len, self.k, m, 0)
                        packet = header + block
                        sock.sendto(packet, (target_ip, target_port))
                        actual_bytes_sent += len(packet)
                        packets_sent_in_batch += 1

                        if pacing_delay > 0 and packets_sent_in_batch >= self.BATCH_SIZE:
                            time.sleep(pacing_delay * self.BATCH_SIZE)
                            packets_sent_in_batch = 0

                encoded_chunks = [(i, encoded_by_idx[i][0], encoded_by_idx[i][1]) for i in range(total_chunks) if i in encoded_by_idx]
                self._last_encoded_model_id = cache_key[0]
                self._last_encoded_params = cache_key[1:]
                self._last_encoded_chunks = encoded_chunks

            if use_cache:
                packets_sent_in_batch = 0
                for chunk_idx, original_len, blocks_out in encoded_chunks:
                    for block_idx, block in enumerate(blocks_out):
                        header = struct.pack("!BdIIIIIBBB", self.PTYPE_DATA, time.time(), current_seq, chunk_idx, total_chunks, block_idx, original_len, self.k, m, 0)
                        packet = header + block
                        sock.sendto(packet, (target_ip, target_port))
                        actual_bytes_sent += len(packet)
                        packets_sent_in_batch += 1

                        if pacing_delay > 0 and packets_sent_in_batch >= self.BATCH_SIZE:
                            time.sleep(pacing_delay * self.BATCH_SIZE)
                            packets_sent_in_batch = 0

            self.stats["fec_encode_time_ms"] += float(fec_encode_time_ms)

            # Send END packet
            end_header = struct.pack("!BdIIIIIBBB", self.PTYPE_END, time.time(), current_seq, 0, 0, 0, 0, self.k, m, 0)
            sock.sendto(end_header, (target_ip, target_port))

            sock.close()

            transfer_duration = time.time() - transfer_start
            if transfer_duration > 0 and actual_bytes_sent > 0:
                self._network_estimator.record_transfer(actual_bytes_sent, transfer_duration, peer_ip=target_ip)

            total_packets = total_chunks * m
            fec_overhead_ratio = (actual_bytes_sent / len(model_data) - 1) * 100 if len(model_data) > 0 else 0
            self.logger.info(f"📡 Sent model via UDP ({len(model_data)}B -> {actual_bytes_sent}B, {total_chunks} chunks, {total_packets} pkts, +{fec_overhead_ratio:.1f}%) to {target_ip}")
            self.stats["sent_models"] += 1
            self.stats["bytes_sent"] += actual_bytes_sent
            self.stats["app_bytes_sent"] += len(model_data)

            # Cache for ARQ
            with self.sent_cache_lock:
                self.sent_models_cache[current_seq] = {
                    "chunks": chunks_data,
                    "timestamp": time.time(),
                    "k": self.k,
                    "m": m,
                    "encoder": encoder,
                }
                now = time.time()
                expired_keys = [k for k, v in self.sent_models_cache.items() if now - v["timestamp"] > 120]
                for k in expired_keys:
                    del self.sent_models_cache[k]
                while len(self.sent_models_cache) > 100:
                    oldest = min(self.sent_models_cache.keys())
                    del self.sent_models_cache[oldest]

            # self._network_estimator.record_packet_result(True, peer_ip=target_ip)  <-- REMOVED: Sending != Success
            return True

        except Exception as e:
            self.logger.error(f"❌ Failed to send model via UDP: {e}")
            self.stats["sent_failed"] += 1
            # self._network_estimator.record_packet_result(False, peer_ip=target_ip) <-- REMOVED: Sending error is local, not network loss
            return False

    def _send_model_no_fec(self, model_data: bytes, target_ip: str, target_port: int, pacing_delay: float) -> bool:
        """
        Send model without FEC encoding (zero overhead mode).
        """
        try:
            transfer_start = time.time()
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4 * 1024 * 1024)

            with self.seq_lock:
                self.model_seq += 1
                current_seq = self.model_seq

            chunk_size = self.PAYLOAD_SIZE * self.k
            total_chunks = math.ceil(len(model_data) / chunk_size)
            actual_bytes_sent = 0

            packets_sent_in_batch = 0

            for chunk_idx in range(total_chunks):
                chunk_start = chunk_idx * chunk_size
                chunk_end = min(chunk_start + chunk_size, len(model_data))
                chunk = model_data[chunk_start:chunk_end]

                block_size = math.ceil(len(chunk) / self.k)
                chunk_padded = chunk + b"\0" * (block_size * self.k - len(chunk))

                for block_idx in range(self.k):
                    block = chunk_padded[block_idx * block_size : (block_idx + 1) * block_size]

                    header = struct.pack(
                        "!BdIIIIIBBB",
                        self.PTYPE_DATA,
                        time.time(),
                        current_seq,
                        chunk_idx,
                        total_chunks,
                        block_idx,
                        len(chunk),
                        self.k,
                        self.k,  # m = k
                        0,
                    )
                    packet = header + block
                    sock.sendto(packet, (target_ip, target_port))
                    actual_bytes_sent += len(packet)
                    packets_sent_in_batch += 1

                    if pacing_delay > 0 and packets_sent_in_batch >= self.BATCH_SIZE:
                        time.sleep(pacing_delay * self.BATCH_SIZE)
                        packets_sent_in_batch = 0

            end_header = struct.pack("!BdIIIIIBBB", self.PTYPE_END, time.time(), current_seq, 0, total_chunks, 0, 0, self.k, self.k, 0)
            sock.sendto(end_header, (target_ip, target_port))

            sock.close()

            transfer_duration = time.time() - transfer_start
            if transfer_duration > 0 and actual_bytes_sent > 0:
                self._network_estimator.record_transfer(actual_bytes_sent, transfer_duration, peer_ip=target_ip)

            self.logger.info(f"📡 Sent model via UDP (NO-FEC) ({len(model_data)} bytes, {total_chunks} chunks) to {target_ip}")
            self.stats["sent_models"] += 1
            self.stats["bytes_sent"] += actual_bytes_sent
            self.stats["app_bytes_sent"] += len(model_data)
            # self._network_estimator.record_packet_result(True, peer_ip=target_ip) <-- REMOVED
            return True

        except Exception as e:
            self.logger.error(f"❌ Failed to send model via UDP (NO-FEC): {e}")
            self.stats["sent_failed"] += 1
            # self._network_estimator.record_packet_result(False, peer_ip=target_ip) <-- REMOVED
            return False

    def get_survival_rate(self) -> float:
        """Calculate packet survival rate."""
        received = self.stats.get("received_models", 0)
        timeout = self.stats.get("timeout_models", 0)
        total = received + timeout
        if total == 0:
            return 1.0
        return received / total

    def start_listener(self, callback):
        self.running = True
        self.callback = callback
        self.thread = threading.Thread(target=self._listener_loop, args=(callback,), daemon=True)
        self.thread.start()

    def set_mdlreq_callback(self, callback):
        self.mdlreq_callback = callback

    def _get_or_create_peer_worker(self, peer_ip):
        with self.peer_workers_lock:
            if peer_ip not in self.peer_workers:
                q = Queue(maxsize=10000)
                self.peer_queues[peer_ip] = q
                t = threading.Thread(target=self._peer_worker_loop, args=(peer_ip, q), daemon=True, name=f"PeerWorker-{peer_ip}")
                t.start()
                self.peer_workers[peer_ip] = t
            return self.peer_queues[peer_ip]

    def _listener_loop(self, callback):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(("0.0.0.0", self.port))
        self.sock.settimeout(2.0)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 16 * 1024 * 1024)

        self.logger.info(f"UDP Listener started on port {self.port}")

        while self.running:
            try:
                data, addr = self.sock.recvfrom(self.MAX_PACKET_SIZE + 100)

                if len(data) < self.HEADER_SIZE:
                    continue

                header = data[: self.HEADER_SIZE]
                ptype, timestamp, model_seq, chunk_idx, total_chunks, block_idx, original_len, sender_k, _, _ = struct.unpack("!BdIIIIIBBB", header)

                peer_ip = addr[0]

                if ptype == self.PTYPE_NACK:
                    payload = data[self.HEADER_SIZE :]
                    missing_chunks = []
                    for i in range(0, len(payload), 4):
                        if i + 4 <= len(payload):
                            missing_chunks.append(struct.unpack("!I", payload[i : i + 4])[0])
                    self.logger.info(f"🔁 NACK received from {peer_ip} for model {model_seq}")
                    self._handle_nack(peer_ip, model_seq, missing_chunks)
                    continue

                if ptype == self.PTYPE_ACK:
                    self._handle_ack(peer_ip, model_seq, timestamp)
                    continue

                if ptype in [self.PTYPE_DATA, self.PTYPE_MCAST, self.PTYPE_END]:
                    # Update activity tracker
                    self.peer_last_activity[peer_ip] = time.time()

                    peer_queue = self._get_or_create_peer_worker(peer_ip)
                    try:
                        peer_queue.put_nowait((data, addr, timestamp))
                    except Exception:
                        pass
                    continue

                if ptype == self.PTYPE_MDLREQ:
                    payload = data[self.HEADER_SIZE :]
                    requester_ip = payload.rstrip(b"\0").decode("utf-8")
                    self.logger.info(f"📬 UDP MDLREQ from {peer_ip}, requester: {requester_ip}")

                    try:
                        ack_header = struct.pack("!BdIIIIIBBB", self.PTYPE_ACK, timestamp, model_seq, 0, 0, 0, 0, 0, 0, 0)
                        self.sock.sendto(ack_header, addr)
                    except Exception:
                        pass

                    if hasattr(self, "mdlreq_callback") and self.mdlreq_callback:
                        try:
                            self.mdlreq_executor.submit(self.mdlreq_callback, requester_ip)
                        except Exception:
                            pass
                    continue

                # Handle ABORT signal (receiver timed out, stop sending to them)
                if ptype == self.PTYPE_ABORT:
                    with self.abort_lock:
                        self.abort_peers[peer_ip] = True
                    self.logger.info(f"🛑 ABORT received from {peer_ip}, stopping sends")
                    continue

            except socket.timeout:
                continue
            except Exception as e:
                self.logger.error(f"Error in UDP dispatcher: {e}")

    def _peer_worker_loop(self, peer_ip, queue):
        incoming_models = {}
        completed_models = set()
        completed_history = collections.deque(maxlen=100)

        self.logger.info(f"🧵 Peer worker started for {peer_ip}")

        while self.running:
            try:
                try:
                    data, addr, _ = queue.get(timeout=2.0)
                except Empty:
                    self._check_peer_timeouts(peer_ip, incoming_models)
                    continue

                if len(data) < self.HEADER_SIZE:
                    continue

                header = data[: self.HEADER_SIZE]
                payload = data[self.HEADER_SIZE :]
                ptype, timestamp, model_seq, chunk_idx, total_chunks, block_idx, original_len, sender_k, sender_m, _ = struct.unpack("!BdIIIIIBBB", header)

                if ptype == self.PTYPE_END:
                    key = model_seq
                    if key in completed_models:
                        continue
                    if key not in incoming_models:
                        incoming_models[key] = {
                            "chunks": {},
                            "meta": {},
                            "last_update": time.time(),
                            "first_packet_time": time.time(),
                            "sender_k": sender_k,
                            "sender_m": sender_m,
                            "retries": 0,
                        }
                    state = incoming_models[key]
                    if self._is_model_complete(state):
                        continue
                    # Fast mode: discard incomplete models without NACK
                    if self.fast_mode:
                        self.logger.info(f"🚀 Fast mode: Discarding incomplete model {model_seq} from {peer_ip}")
                        with self.stats_lock:
                            self.stats["fast_mode_discarded"] += 1
                        completed_models.add(key)
                        completed_history.append(key)
                        if key in incoming_models:
                            del incoming_models[key]

                        # Notify listener to stop waiting
                        if self.callback:
                            try:
                                self.callback(b"FAST_DISCARD", peer_ip)
                            except Exception:
                                pass
                        continue
                    # NACK 制御（Ablation Study）
                    if self.nack_enabled:
                        # NACK 有効: 不足チャンクの再送を要求
                        missing = self._identify_missing_chunks(state)
                        if missing:
                            self._send_nack(peer_ip, model_seq, missing)
                            state["retries"] += 1
                            state["last_update"] = time.time()
                    else:
                        # NACK 無効: 不完全モデルを破棄
                        self.logger.info(f"🔕 NACK disabled: Discarding incomplete model {model_seq} from {peer_ip}")
                        with self.stats_lock:
                            self.stats["nack_disabled_discards"] += 1
                        completed_models.add(key)
                        completed_history.append(key)
                        if key in incoming_models:
                            del incoming_models[key]

                        # Notify listener to stop waiting
                        if self.callback:
                            try:
                                self.callback(b"NACK_DISABLED_DISCARD", peer_ip)
                            except Exception:
                                pass
                    continue

                if ptype not in [self.PTYPE_DATA, self.PTYPE_MCAST]:
                    continue

                key = model_seq
                if key in completed_models:
                    continue

                if key not in incoming_models:
                    incoming_models[key] = {
                        "chunks": {},
                        "meta": {},
                        "last_update": time.time(),
                        "first_packet_time": time.time(),
                        "sender_k": sender_k,
                        "sender_m": sender_m,
                        "retries": 0,
                    }

                state = incoming_models[key]
                state["last_update"] = time.time()

                if chunk_idx not in state["chunks"]:
                    state["chunks"][chunk_idx] = {}
                    state["meta"][chunk_idx] = {"total": total_chunks, "len": original_len}

                if block_idx not in state["chunks"][chunk_idx]:
                    state["chunks"][chunk_idx][block_idx] = payload
                    with self.stats_lock:
                        self.stats["total_chunks_received"] += 1
                        self.stats["bytes_received"] += len(data)

                if self._is_model_complete(state):
                    model_data = self._reassemble_model(state)
                    if model_data is not None:
                        with self.stats_lock:
                            self.stats["received_models"] += 1

                        try:
                            duration = time.time() - float(state.get("first_packet_time", time.time()))
                            if duration > 0:
                                self._network_estimator.record_transfer(len(model_data), duration, peer_ip=peer_ip)

                            # Calculate and record Observed Loss Rate (Corrected Logic)
                            sender_k = state.get("sender_k", 16)
                            sender_m = state.get("sender_m", 0)
                            if sender_m == 0:
                                sender_m = sender_k  # For NO-FEC, m=k effectively

                            total_chunks = state.get("meta", {}).get(0, {}).get("total", 0)
                            if total_chunks == 0:
                                # Safe fallback if meta not fully populated
                                total_chunks = max(state.get("chunks", {}).keys(), default=-1) + 1

                            expected_packets = total_chunks * sender_m
                            received_packets = 0
                            for c_idx, blocks in state.get("chunks", {}).items():
                                received_packets += len(blocks)

                            if expected_packets > 0:
                                loss_rate = 1.0 - (received_packets / expected_packets)
                                loss_rate = max(0.0, loss_rate)
                                self._network_estimator.record_loss_rate_sample(loss_rate, peer_ip=peer_ip)
                                self.logger.debug(f"📉 Observed Loss from {peer_ip}: {loss_rate:.1%} ({received_packets}/{expected_packets})")

                        except Exception:
                            pass

                        if self.callback:
                            try:
                                self.callback(model_data, peer_ip)
                            except Exception:
                                pass

                        completed_models.add(key)
                        completed_history.append(key)
                        del incoming_models[key]

            except Exception as e:
                self.logger.error(f"Error in peer worker {peer_ip}: {e}")

    def _check_peer_timeouts(self, peer_ip, incoming_models):
        current_time = time.time()
        to_remove = []
        for key, state in incoming_models.items():
            # Use aggressive timeout in queue loop? No, use self.timeout relative to START
            # Actually, standard UDP timeout logic is time since last packet or total duration?
            # Impl uses time since FIRST packet.
            time_since_first = current_time - state.get("first_packet_time", current_time)
            if time_since_first > self.timeout:
                to_remove.append(key)
        for key in to_remove:
            with self.stats_lock:
                self.stats["timeout_models"] += 1
                if self.fast_mode:
                    self.stats["fast_mode_discarded"] += 1
            if self.fast_mode:
                self.logger.debug(f"🚀 Fast mode: Timeout discarding model {key} from {peer_ip}")
                # Notify listener to stop waiting
                if self.callback:
                    try:
                        self.callback(b"FAST_DISCARD", peer_ip)
                    except Exception:
                        pass
            del incoming_models[key]

    def _is_model_complete(self, state):
        chunks = state["chunks"]
        meta = state["meta"]
        sender_k = state.get("sender_k", self.k)

        if not meta:
            return False
        first_meta = next(iter(meta.values()))
        total_chunks = first_meta["total"]

        if len(chunks) < total_chunks:
            return False

        for i in range(total_chunks):
            if i not in chunks:
                return False
            if len(chunks[i]) < sender_k:
                return False
        return True

    def _reassemble_model(self, state):
        try:
            chunks = state["chunks"]
            meta = state["meta"]
            sender_k = state.get("sender_k", self.k)
            sender_m = state.get("sender_m", self.k)  # Default to k if m missing? NO, default to m?

            full_data = b""
            sorted_chunk_indices = sorted(chunks.keys())
            chunks_recovered_with_fec = 0

            # NO-FEC
            if sender_k == sender_m:
                for i in sorted_chunk_indices:
                    blocks_map = chunks[i]
                    sorted_blocks = [blocks_map[b_idx] for b_idx in sorted(blocks_map.keys())]
                    chunk_data = b"".join(sorted_blocks)
                    original_len = meta[i]["len"]
                    full_data += chunk_data[:original_len]
                return full_data

            # FEC
            decoder = zfec.Decoder(sender_k, sender_m)
            fec_decode_start = time.time()

            for i in sorted_chunk_indices:
                blocks_map = chunks[i]
                blocks = []
                block_nums = []
                for b_idx, b_data in blocks_map.items():
                    blocks.append(b_data)
                    block_nums.append(b_idx)
                    if len(blocks) == sender_k:
                        break

                if len(blocks_map) < sender_m and len(blocks_map) >= sender_k:
                    chunks_recovered_with_fec += 1

                decoded = decoder.decode(blocks, block_nums)
                chunk_data = b"".join(decoded)
                original_len = meta[i]["len"]
                full_data += chunk_data[:original_len]

            fec_decode_time_ms = (time.time() - fec_decode_start) * 1000
            with self.stats_lock:
                self.stats["fec_decode_time_ms"] += fec_decode_time_ms

            if chunks_recovered_with_fec > 0:
                with self.stats_lock:
                    self.stats["fec_recovery_success"] += chunks_recovered_with_fec

            return full_data

        except Exception as e:
            self.logger.error(f"Reassembly failed: {e}")
            with self.stats_lock:
                self.stats["fec_recovery_fail"] += 1
            return None

    def _identify_missing_chunks(self, state):
        chunks = state["chunks"]
        meta = state["meta"]
        if not meta:
            return []
        first_meta = next(iter(meta.values()))
        total_chunks = first_meta["total"]
        sender_k = state.get("sender_k", self.k)
        missing = []
        for i in range(total_chunks):
            if i not in chunks or len(chunks[i]) < sender_k:
                missing.append(i)
        return missing

    def _send_nack(self, target_ip, model_seq, missing_chunks):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            payload = b"".join([struct.pack("!I", c) for c in missing_chunks])
            header = struct.pack("!BdIIIIIBBB", self.PTYPE_NACK, time.time(), model_seq, 0, 0, 0, 0, 0, 0, 0)
            sock.sendto(header + payload, (target_ip, self.port))
            sock.close()
        except Exception:
            pass

    def _handle_nack(self, target_ip, model_seq, missing_chunks):
        cache = None
        with self.sent_cache_lock:
            cache = self.sent_models_cache.get(model_seq)
        if not cache:
            return

        chunks_data = cache["chunks"]
        encoder = cache["encoder"]
        # encoder might be None if it was NO-FEC? No, ARQ cache always has encoder if available...
        # Wait, if mode was NO-FEC, encoder is None. But NACK implies failure.
        # If NO-FEC used, we need to resend using NO-FEC or switch to FEC?
        # Current logic: reuse established encoder. If None, handle gracefully?
        # Actually my send_model stores encoder=None if parity=0.
        # If parity=0, chunks_data contains raw chunks? No, contains (chunk_idx, blocks_in, len).
        # blocks_in is split chunks. We can resend blocks.
        # If encoder is None, just resend existing blocks in cache?

        needed = [c for c in chunks_data if c[0] in missing_chunks]
        if not needed:
            return

        self.nack_executor.submit(self._resend_worker, target_ip, needed, cache["m"], cache["k"], model_seq, encoder, len(chunks_data))

    def _resend_worker(self, target_ip, chunks_to_send, m, k, model_seq, encoder, total_chunks):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1024 * 1024)

            for chunk_idx, blocks_in, original_len in chunks_to_send:
                if encoder:
                    blocks_out = encoder.encode(blocks_in)
                    limit = len(blocks_out)
                else:
                    # NO-FEC case: blocks_in IS the blocks to send (k blocks)
                    # Just resend all of them? Or just missing ones?
                    # Receiver NACKs chunks, not blocks. So we resend whole chunk.
                    blocks_out = blocks_in
                    limit = len(blocks_out)

                for block_idx in range(limit):
                    block = blocks_out[block_idx]
                    header = struct.pack("!BdIIIIIBBB", self.PTYPE_DATA, time.time(), model_seq, chunk_idx, total_chunks, block_idx, original_len, k, m, 0)
                    sock.sendto(header + block, (target_ip, self.port))

            sock.close()
            self.logger.info(f"🔄 Resent {len(chunks_to_send)} chunks to {target_ip}")
        except Exception:
            pass

    def _handle_ack(self, target_ip, model_seq, echoed_ts):
        now = time.time()
        if echoed_ts is None or echoed_ts <= 0 or now - echoed_ts > 5.0:
            return
        rtt = (now - echoed_ts) * 1000
        self._network_estimator.record_rtt(rtt, target_ip)

    def join_multicast_group(self, group_ip):
        try:
            target_sock = getattr(self, "sock", None)
            if target_sock:
                mreq = struct.pack("4s4s", socket.inet_aton(group_ip), socket.inet_aton("0.0.0.0"))
                target_sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
        except Exception:
            pass

    def _is_multicast(self, ip_str):
        try:
            first_byte = int(ip_str.split(".")[0])
            return 224 <= first_byte <= 239
        except Exception:
            return False
