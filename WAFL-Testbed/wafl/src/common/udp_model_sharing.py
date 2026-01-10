import collections
import logging
import math
import socket
import struct
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from queue import Empty, Queue

import zfec

from .network_estimator import get_network_estimator


class UDPModelSharing:
    """
    Handles UDP-based model sharing with Forward Error Correction (FEC).
    """

    MAX_PACKET_SIZE = 1400  # Safe MTU size
    # Header: type(1), timestamp(8), model_seq(4), chunk_idx(4), total_chunks(4), block_idx(4), original_len(4), k(1), m(1), pad(1) = 32 bytes
    HEADER_SIZE = 32
    PAYLOAD_SIZE = MAX_PACKET_SIZE - HEADER_SIZE

    # Packet Types
    PTYPE_DATA = 0
    PTYPE_NACK = 1
    PTYPE_MCAST = 2
    PTYPE_ACK = 3
    PTYPE_END = 4  # End of Transfer (Fast NACK trigger)
    PTYPE_MDLREQ = 5  # Model Request (UDP-based MDLREQ)

    MAX_RETRIES = 10  # Max NACK attempts (Increased for resilience)
    BATCH_SIZE = 16  # Number of packets to send before sleeping (Batch Pacing)

    def __init__(self, ip: str, port: int, fec_m: int, timeout: float, inter_packet_timeout: float):
        self.ip = ip
        self.port = port

        # FEC Settings
        # k: data packets per chunk
        # parity: redundant packets per chunk
        # m = k + parity (total packets per chunk)
        self.k = fec_m if fec_m > 0 else 16  # Default k=16 if not specified

        # ネットワーク推定器から動的に parity を決定
        self._network_estimator = get_network_estimator()
        self.parity = self._network_estimator.get_recommended_fec_parity(self.k)
        self.m = self.k + self.parity

        # Pacing Control
        self.pacing_delay = 0.0

        # Adaptive Timeout: Use RTT-based values if available
        metrics = self._network_estimator.get_metrics()
        rtt_sec = metrics.rtt_ms / 1000 if metrics.rtt_ms > 0 else 0.1
        self.timeout = max(timeout, rtt_sec * 10)  # At least 10x RTT
        self.inter_packet_timeout = max(inter_packet_timeout, rtt_sec * 3)  # At least 3x RTT

        self.logger = logging.getLogger("UDPModelSharing")

        # FEC encoder/decoder (handle parity=0 case)
        if self.parity > 0:
            self.encoder = zfec.Encoder(self.k, self.m)
            self.decoder = zfec.Decoder(self.k, self.m)
        else:
            self.encoder = None  # No FEC
            self.decoder = None

        # FEC and Survival Rate Statistics
        self.stats = {
            "sent_models": 0,
            "sent_failed": 0,
            "received_models": 0,
            "fec_recovery_success": 0,
            "fec_recovery_fail": 0,
            "total_chunks_received": 0,
            "failed_chunks": 0,  # Chunks that failed due to timeout (insufficient packets)
            "timeout_models": 0,  # Models that failed due to timeout
            "bytes_sent": 0,
            "bytes_received": 0,
            "fec_encode_time_ms": 0.0,  # FEC エンコード処理時間（累積）
            "fec_decode_time_ms": 0.0,  # FEC デコード処理時間（累積）
        }

        # Monotonically increasing model sequence number with thread lock
        self.model_seq = 0
        self.seq_lock = threading.Lock()

        # Cache for ARQ (Retransmission)
        # {model_seq: {'chunks': chunks_data, 'timestamp': ts, 'k': k, 'm': m}}
        self.sent_models_cache = {}
        self.sent_cache_lock = threading.Lock()

        # RTT and Congestion Control
        self.min_rtt = float("inf")
        self.smoothed_rtt = 0.0
        self.rtt_var = 0.0
        self.rtt_learning = True

        # Multicast Settings
        self.mcast_ttl = 2

        self.logger.info(f"UDPModelSharing initialized: k={self.k}, m={self.m} (parity={self.parity}), timeout={self.timeout}s, pacing={self.pacing_delay * 1000:.2f}ms")

        # NACK Thread Pool (persistent, avoids thread creation overhead)
        self.nack_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="NACK")

        # Per-peer worker threads and queues for parallel reception
        # This eliminates single-thread bottleneck when receiving from multiple peers
        self.peer_queues = {}  # {peer_ip: Queue}
        self.peer_workers = {}  # {peer_ip: Thread}
        self.peer_workers_lock = threading.Lock()
        self.callback = None  # Will be set by start_listener
        self.mdlreq_callback = None  # Callback for MDLREQ handling
        self.mdlreq_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="MDLREQ")  # Thread pool for MDLREQ callbacks
        self.stats_lock = threading.Lock()  # Protect stats from concurrent worker access

    def update_network_params(self, new_k: int, new_parity: int = None, new_pacing: float = None) -> None:
        """
        Update network parameters (FEC and Pacing) dynamically based on network conditions.

        Args:
            new_k: New k value (number of data packets per chunk)
            new_parity: Optional new parity value.
            new_pacing: Optional new pacing delay in seconds.
        """
        # --- Update Pacing ---
        if new_pacing is not None:
            # Sanity check: between 0 and 0.1s
            new_pacing = max(0.0, min(0.1, new_pacing))
            old_pacing = self.pacing_delay
            if abs(new_pacing - old_pacing) > 1e-6:
                self.pacing_delay = new_pacing
                self.logger.info(f"⏱️ Pacing updated: {old_pacing * 1000:.2f}ms -> {self.pacing_delay * 1000:.2f}ms")

        # --- Update FEC ---
        # Ensure k is at least 1 (zfec requires k >= 1)
        new_k = max(1, new_k)
        # Upper limit: k > 200 is impractical
        new_k = min(200, new_k)

        if new_parity is None:
            new_parity = max(1, math.ceil(new_k * 0.1))

        new_parity = max(1, new_parity)

        # Validate total m (zfec max 256)
        if new_k + new_parity > 256:
            new_parity = 256 - new_k

        if new_k == self.k and new_parity == self.parity:
            return  # No change needed

        old_k, old_m, old_parity = self.k, self.m, self.parity
        self.k = new_k
        self.parity = new_parity
        self.m = new_k + new_parity

        # Reinitialize encoder/decoder with new parameters
        self.encoder = zfec.Encoder(self.k, self.m)
        self.decoder = zfec.Decoder(self.k, self.m)

        self.logger.info(f"🔄 FEC params updated: k={old_k}→{self.k}, parity={old_parity}→{self.parity} (m={old_m}→{self.m}, redundancy={self.parity / self.m:.1%})")

    def send_mdlreq(self, target_ip: str, target_port: int, requester_ip: str) -> bool:
        """
        Send a UDP MDLREQ (Model Request) to a peer.
        This allows requesting a model without TCP handshake.

        Args:
            target_ip: IP address of the peer to request model from
            target_port: Port of the peer
            requester_ip: IP address of this node (who is requesting)

        Returns:
            True if request was sent successfully, False otherwise
        """
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.settimeout(1.0)

            # Pack MDLREQ packet: type(1), timestamp(8), requester_ip as bytes (padded)
            # We encode the requester IP as a string in the payload
            requester_bytes = requester_ip.encode("utf-8")
            requester_bytes = requester_bytes[:32].ljust(32, b"\0")  # Pad or truncate to 32 bytes

            # Header: PTYPE_MDLREQ, timestamp, 0, 0, 0, 0, 0, 0, 0, 0
            header = struct.pack("!BdIIIIIBBB", self.PTYPE_MDLREQ, time.time(), 0, 0, 0, 0, 0, 0, 0, 0)
            packet = header + requester_bytes

            sock.sendto(packet, (target_ip, target_port))
            sock.close()

            self.logger.debug(f"📬 Sent UDP MDLREQ to {target_ip}:{target_port} from {requester_ip}")
            return True

        except Exception as e:
            self.logger.error(f"❌ Failed to send UDP MDLREQ: {e}")
            return False

    def send_model(self, model_data: bytes, target_ip: str, target_port: int) -> bool:
        """
        Sends serialized model data via UDP with FEC and Pacing.
        """
        try:
            # 送信前に FEC パラメータを動的更新
            new_parity = self._network_estimator.get_recommended_fec_parity(self.k)
            if new_parity != self.parity:
                old_parity = self.parity
                self.parity = new_parity
                self.m = self.k + self.parity
                if self.parity > 0:
                    self.encoder = zfec.Encoder(self.k, self.m)
                    self.decoder = zfec.Decoder(self.k, self.m)
                else:
                    self.encoder = None
                    self.decoder = None
                self.logger.info(f"🔄 FEC params updated: parity {old_parity} -> {self.parity} (m={self.m})")

            # Adaptive Timeout & Pacing: Update based on current RTT and Loss
            metrics = self._network_estimator.get_metrics()
            if metrics.rtt_ms > 0:
                rtt_sec = metrics.rtt_ms / 1000
                self.inter_packet_timeout = max(0.1, rtt_sec * 3)

            # Update pacing delay
            self.pacing_delay = self._network_estimator.get_recommended_pacing_delay()

            # ====== FEC BYPASS MODE (parity=0) ======
            if self.parity == 0:
                return self._send_model_no_fec(model_data, target_ip, target_port)

            # Strategy: Split data into chunks of size (PAYLOAD_SIZE * k).
            # Each chunk is encoded into m packets of size PAYLOAD_SIZE.

            chunk_size = self.PAYLOAD_SIZE * self.k
            total_chunks = math.ceil(len(model_data) / chunk_size)

            # Pre-calculate chunks
            chunks_data = []
            for chunk_idx in range(total_chunks):
                start = chunk_idx * chunk_size
                end = min(start + chunk_size, len(model_data))
                chunk = model_data[start:end]

                # Pad chunk to be exactly divisible by k
                block_size = math.ceil(len(chunk) / self.k)
                chunk_padded = chunk + b"\0" * (block_size * self.k - len(chunk))

                # Split into k blocks
                blocks_in = [chunk_padded[i * block_size : (i + 1) * block_size] for i in range(self.k)]
                chunks_data.append((chunk_idx, blocks_in, len(chunk)))

            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

            # Increase socket buffer to avoid local drops during pacing sleep if any
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4 * 1024 * 1024)  # 4MB (Was 1MB)

            # Multicast TTL if target is multicast
            if self._is_multicast(target_ip):
                sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, self.mcast_ttl)

            # Increment model sequence for this transmission (Thread-Safe)
            current_seq = 0
            with self.seq_lock:
                self.model_seq += 1
                current_seq = self.model_seq

            # Track actual bytes sent (including FEC redundancy and headers)
            actual_bytes_sent = 0
            fec_encode_start = time.time()  # FEC エンコード時間計測開始

            # Parallel Encoding with Pipeline Parallelism
            # Send packets as soon as encoding completes (don't wait for all chunks)
            from concurrent.futures import ThreadPoolExecutor, as_completed

            num_workers = min(8, total_chunks) if total_chunks > 0 else 1  # 8 workers (Was 4)

            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                # Submit all encoding tasks with chunk metadata
                future_to_chunk = {}
                for chunk_idx, blocks_in, original_len in chunks_data:
                    future = executor.submit(self.encoder.encode, blocks_in)
                    future_to_chunk[future] = (chunk_idx, original_len)

                # Send loop with BATCH PACING and Pipeline Parallelism
                # Use as_completed to send packets as soon as encoding finishes
                packets_sent_in_batch = 0
                for future in as_completed(future_to_chunk):
                    chunk_idx, original_len = future_to_chunk[future]
                    blocks_out = future.result()

                    for block_idx, block in enumerate(blocks_out):
                        # Header: type(1), timestamp(8), model_seq(4), chunk_idx(4), total_chunks(4), block_idx(4), original_len(4), k(1), m(1), pad(1)
                        # Padding last byte with 0
                        header = struct.pack("!BdIIIIIBBB", self.PTYPE_DATA, time.time(), current_seq, chunk_idx, total_chunks, block_idx, original_len, self.k, self.m, 0)
                        packet = header + block
                        sock.sendto(packet, (target_ip, target_port))
                        actual_bytes_sent += len(packet)
                        packets_sent_in_batch += 1

                        # BATCH PACING: Sleep after every BATCH_SIZE packets
                        if self.pacing_delay > 0 and packets_sent_in_batch >= self.BATCH_SIZE:
                            time.sleep(self.pacing_delay * self.BATCH_SIZE)
                            packets_sent_in_batch = 0

            # FEC エンコード時間計測終了（送信時間を除く純粋なエンコード時間は並列処理のため概算）
            fec_encode_time_ms = (time.time() - fec_encode_start) * 1000
            self.stats["fec_encode_time_ms"] += fec_encode_time_ms

            # Send END packet for Fast NACK trigger
            # Header: PTYPE_END, ts, seq, 0, 0, 0, 0, 0, 0, 0
            end_header = struct.pack("!BdIIIIIBBB", self.PTYPE_END, time.time(), current_seq, 0, 0, 0, 0, self.k, self.m, 0)
            sock.sendto(end_header, (target_ip, target_port))

            # ====== PROACTIVE REDUNDANCY ======
            # For high-loss conditions (loss > 1%), send extra parity packets proactively
            # This reduces NACK round-trip latency
            loss_rate = metrics.packet_loss_rate if metrics.packet_loss_rate > 0 else 0.0
            if loss_rate > 0.01 and self.parity > 0:
                import random

                extra_redundancy = min(self.parity, 4)  # Send up to 4 extra packets per chunk
                for chunk_idx, blocks_in, original_len in chunks_data:
                    # Re-encode this chunk to get parity blocks
                    blocks_out = self.encoder.encode(blocks_in)
                    # Send random parity blocks (indices k to m-1)
                    parity_indices = list(range(self.k, self.m))
                    random.shuffle(parity_indices)
                    for block_idx in parity_indices[:extra_redundancy]:
                        block = blocks_out[block_idx]
                        header = struct.pack("!BdIIIIIBBB", self.PTYPE_DATA, time.time(), current_seq, chunk_idx, total_chunks, block_idx, original_len, self.k, self.m, 0)
                        packet = header + block
                        sock.sendto(packet, (target_ip, target_port))
                        actual_bytes_sent += len(packet)
                self.logger.debug(f"📦 Proactive redundancy: sent {extra_redundancy} extra blocks/chunk for {total_chunks} chunks")

            sock.close()

            total_packets = total_chunks * self.m
            fec_overhead_ratio = (actual_bytes_sent / len(model_data) - 1) * 100 if len(model_data) > 0 else 0
            self.logger.info(f"📡 Sent model via UDP ({len(model_data)} payload bytes, {actual_bytes_sent} total bytes, {total_chunks} chunks, {total_packets} packets, +{fec_overhead_ratio:.1f}% overhead) to {target_ip}:{target_port}")
            self.stats["sent_models"] += 1
            self.stats["bytes_sent"] += actual_bytes_sent  # Count actual bytes sent, not just payload

            # Cache for ARQ
            with self.sent_cache_lock:
                self.sent_models_cache[current_seq] = {
                    "chunks": chunks_data,
                    "timestamp": time.time(),
                    "k": self.k,
                    "m": self.m,
                    "encoder": self.encoder,  # safe to share? Encoder is stateless? No, configured with k,m.
                }
                # Cleanup old cache: Remove entries older than 120s OR if cache exceeds 100 entries
                now = time.time()
                expired_keys = [k for k, v in self.sent_models_cache.items() if now - v["timestamp"] > 120]
                for k in expired_keys:
                    del self.sent_models_cache[k]
                # If still too large (>100), remove oldest
                while len(self.sent_models_cache) > 100:
                    oldest = min(self.sent_models_cache.keys())
                    del self.sent_models_cache[oldest]

            # 送信成功を記録
            self._network_estimator.record_packet_result(True)
            return True

        except Exception as e:
            self.logger.error(f"❌ Failed to send model via UDP: {e}")
            self.stats["sent_failed"] += 1
            # 送信失敗を記録
            self._network_estimator.record_packet_result(False)
            return False

    def _send_model_no_fec(self, model_data: bytes, target_ip: str, target_port: int) -> bool:
        """
        Send model without FEC encoding (zero overhead mode).
        Used when network conditions are excellent (loss < 1%).

        IMPORTANT: This mode still uses the same packet structure as FEC mode
        but with k=m (no redundancy). Receiver handles k=m case by skipping FEC decode.
        """
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4 * 1024 * 1024)  # 4MB (Was 1MB)

            # Increment model sequence
            with self.seq_lock:
                self.model_seq += 1
                current_seq = self.model_seq

            # Use same chunk structure as FEC mode for compatibility
            # Split data into chunks of size (PAYLOAD_SIZE * k), then split each chunk into k blocks
            chunk_size = self.PAYLOAD_SIZE * self.k
            total_chunks = math.ceil(len(model_data) / chunk_size)
            actual_bytes_sent = 0

            # Batch pacing counter
            packets_sent_in_batch = 0

            for chunk_idx in range(total_chunks):
                chunk_start = chunk_idx * chunk_size
                chunk_end = min(chunk_start + chunk_size, len(model_data))
                chunk = model_data[chunk_start:chunk_end]

                # Split chunk into k blocks (same as FEC mode)
                block_size = math.ceil(len(chunk) / self.k)
                chunk_padded = chunk + b"\0" * (block_size * self.k - len(chunk))

                for block_idx in range(self.k):
                    block = chunk_padded[block_idx * block_size : (block_idx + 1) * block_size]

                    # Header: type, timestamp, model_seq, chunk_idx, total_chunks, block_idx, original_len, k, m, pad
                    header = struct.pack(
                        "!BdIIIIIBBB",
                        self.PTYPE_DATA,
                        time.time(),
                        current_seq,
                        chunk_idx,  # Correct chunk index
                        total_chunks,  # Correct total chunks
                        block_idx,  # Block index within chunk
                        len(chunk),  # Original chunk length
                        self.k,
                        self.k,  # m = k (no parity)
                        0,
                    )
                    packet = header + block
                    sock.sendto(packet, (target_ip, target_port))
                    actual_bytes_sent += len(packet)
                    packets_sent_in_batch += 1

                    # BATCH PACING: Sleep after every BATCH_SIZE packets
                    if self.pacing_delay > 0 and packets_sent_in_batch >= self.BATCH_SIZE:
                        time.sleep(self.pacing_delay * self.BATCH_SIZE)
                        packets_sent_in_batch = 0

            # Send END packet
            end_header = struct.pack("!BdIIIIIBBB", self.PTYPE_END, time.time(), current_seq, 0, total_chunks, 0, 0, self.k, self.k, 0)
            sock.sendto(end_header, (target_ip, target_port))

            sock.close()

            self.logger.info(f"📡 Sent model via UDP (NO-FEC) ({len(model_data)} bytes, {total_chunks} chunks, {total_chunks * self.k} packets) to {target_ip}:{target_port}")
            self.stats["sent_models"] += 1
            self.stats["bytes_sent"] += actual_bytes_sent
            self._network_estimator.record_packet_result(True)
            return True

        except Exception as e:
            self.logger.error(f"❌ Failed to send model via UDP (NO-FEC): {e}")
            self.stats["sent_failed"] += 1
            self._network_estimator.record_packet_result(False)
            return False

    def get_survival_rate(self) -> float:
        """Calculate packet survival rate.

        This measures the ratio of models that were successfully received
        (with or without FEC recovery) vs total models attempted.
        Includes timeout failures in the calculation.
        """
        received = self.stats.get("received_models", 0)
        timeout = self.stats.get("timeout_models", 0)
        total = received + timeout

        if total == 0:
            return 1.0  # No data yet

        return received / total

    def start_listener(self, callback):
        """
        Starts a thread to listen for UDP packets and reassemble models.
        callback(data: bytes, source_ip: str)
        """
        self.running = True
        self.callback = callback  # Save for peer workers
        self.thread = threading.Thread(target=self._listener_loop, args=(callback,), daemon=True)
        self.thread.start()

    def set_mdlreq_callback(self, callback):
        """
        Set a callback function for handling UDP MDLREQ packets.
        callback(requester_ip: str) - called when a MDLREQ is received
        """
        self.mdlreq_callback = callback
        self.logger.debug("MDLREQ callback registered")

    def _get_or_create_peer_worker(self, peer_ip):
        """
        Get or create a dedicated worker thread and queue for a peer.
        This ensures each peer's packets are processed independently.
        """
        with self.peer_workers_lock:
            if peer_ip not in self.peer_workers:
                q = Queue(maxsize=10000)
                self.peer_queues[peer_ip] = q
                t = threading.Thread(target=self._peer_worker_loop, args=(peer_ip, q), daemon=True, name=f"PeerWorker-{peer_ip}")
                t.start()
                self.peer_workers[peer_ip] = t
                self.logger.debug(f"🧵 Created worker thread for peer {peer_ip}")
            return self.peer_queues[peer_ip]

    def _listener_loop(self, callback):
        """
        Main listener loop - acts as a dispatcher.
        DATA/MCAST packets are routed to per-peer worker queues.
        NACK/ACK packets are handled inline (they're small and time-critical).
        """
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(("0.0.0.0", self.port))
        self.sock.settimeout(2.0)

        # Increase receive buffer to 16MB to avoid OS-level drops
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 16 * 1024 * 1024)  # 16MB (Was 8MB)

        self.logger.info(f"UDP Listener (dispatcher) started on port {self.port}")

        while self.running:
            try:
                data, addr = self.sock.recvfrom(self.MAX_PACKET_SIZE + 100)

                if len(data) < self.HEADER_SIZE:
                    continue

                header = data[: self.HEADER_SIZE]
                ptype, timestamp, model_seq, chunk_idx, total_chunks, block_idx, original_len, sender_k, sender_m, _ = struct.unpack("!BdIIIIIBBB", header)

                peer_ip = addr[0]

                # NACK handling - time critical, handle inline
                if ptype == self.PTYPE_NACK:
                    payload = data[self.HEADER_SIZE :]
                    missing_chunks = []
                    for i in range(0, len(payload), 4):
                        if i + 4 <= len(payload):
                            missing_chunks.append(struct.unpack("!I", payload[i : i + 4])[0])
                    self.logger.info(f"🔁 NACK received from {peer_ip} for model {model_seq}, missing chunks: {len(missing_chunks)}")
                    self._handle_nack(peer_ip, model_seq, missing_chunks)
                    continue

                # ACK handling - inline
                if ptype == self.PTYPE_ACK:
                    self._handle_ack(peer_ip, model_seq, timestamp)
                    continue

                # DATA/MCAST/END packets - route to per-peer worker
                if ptype in [self.PTYPE_DATA, self.PTYPE_MCAST, self.PTYPE_END]:
                    peer_queue = self._get_or_create_peer_worker(peer_ip)
                    try:
                        peer_queue.put_nowait((data, addr, timestamp))
                    except Exception as e:
                        # Queue full - drop packet (will be recovered via NACK)
                        self.logger.error(f"Queue full for {peer_ip}: {e}")
                        pass
                    continue

                # MDLREQ handling - model request received
                if ptype == self.PTYPE_MDLREQ:
                    # Extract requester IP from payload
                    payload = data[self.HEADER_SIZE :]
                    requester_ip = payload.rstrip(b"\0").decode("utf-8")
                    self.logger.info(f"📬 UDP MDLREQ received from {peer_ip}, requester: {requester_ip}")

                    # Trigger MDLREQ callback if set
                    if hasattr(self, "mdlreq_callback") and self.mdlreq_callback:
                        try:
                            # Use thread pool to avoid blocking the listener
                            self.mdlreq_executor.submit(self.mdlreq_callback, requester_ip)
                        except Exception as e:
                            self.logger.error(f"Error submitting MDLREQ callback: {e}")
                    continue

            except socket.timeout:
                continue
            except Exception as e:
                self.logger.error(f"Error in UDP dispatcher: {e}")

    def _peer_worker_loop(self, peer_ip, queue):
        """
        Per-peer worker thread - handles FEC decoding and model reassembly.
        Each peer has its own isolated state, eliminating contention.
        """
        # Per-peer state (completely isolated from other peers)
        incoming_models = {}  # model_seq -> state
        completed_models = set()
        completed_history = collections.deque(maxlen=100)

        self.logger.info(f"🧵 Peer worker started for {peer_ip}")

        while self.running:
            try:
                # Get packet from queue with timeout
                try:
                    data, addr, _ = queue.get(timeout=2.0)
                except Empty:
                    # Check for timeouts during idle
                    self._check_peer_timeouts(peer_ip, incoming_models)
                    continue

                if len(data) < self.HEADER_SIZE:
                    continue

                header = data[: self.HEADER_SIZE]
                payload = data[self.HEADER_SIZE :]
                ptype, timestamp, model_seq, chunk_idx, total_chunks, block_idx, original_len, sender_k, sender_m, _ = struct.unpack("!BdIIIIIBBB", header)

                # Handle END packet - triggers fast NACK
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
                            "last_ts": timestamp,
                        }

                    state = incoming_models[key]
                    if self._is_model_complete(state):
                        continue

                    missing = self._identify_missing_chunks(state)
                    if missing:
                        self._send_nack(peer_ip, model_seq, missing)
                        state["retries"] += 1
                        state["last_update"] = time.time()
                    continue

                # Handle DATA/MCAST packets
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
                        "last_ts": timestamp,
                    }

                state = incoming_models[key]
                state["last_update"] = time.time()
                state["last_ts"] = timestamp

                if chunk_idx not in state["chunks"]:
                    state["chunks"][chunk_idx] = {}
                    state["meta"][chunk_idx] = {"total": total_chunks, "len": original_len}

                if block_idx not in state["chunks"][chunk_idx]:
                    state["chunks"][chunk_idx][block_idx] = payload
                    with self.stats_lock:
                        self.stats["total_chunks_received"] += 1
                        self.stats["bytes_received"] += len(data)

                # Check if model is complete
                if self._is_model_complete(state):
                    model_data = self._reassemble_model(state)
                    if model_data is not None:
                        with self.stats_lock:
                            self.stats["received_models"] += 1

                        # Send ACK
                        self._send_ack(peer_ip, model_seq, state["last_ts"])

                        # Call callback
                        if self.callback:
                            try:
                                self.callback(model_data, peer_ip)
                            except Exception as e:
                                self.logger.error(f"Callback error: {e}")

                        # Mark as completed
                        completed_models.add(key)
                        completed_history.append(key)
                        if len(completed_history) > 50:
                            old_key = completed_history.popleft()
                            if old_key in completed_models:
                                completed_models.discard(old_key)
                        del incoming_models[key]

            except Exception as e:
                self.logger.error(f"Error in peer worker {peer_ip}: {e}")

    def _check_peer_timeouts(self, peer_ip, incoming_models):
        """Check for model completion timeouts in a peer's state."""
        current_time = time.time()
        to_remove = []
        for key, state in incoming_models.items():
            time_since_first = current_time - state.get("first_packet_time", current_time)
            if time_since_first > self.timeout:
                to_remove.append(key)
        for key in to_remove:
            with self.stats_lock:
                self.stats["timeout_models"] += 1
            del incoming_models[key]

    def _is_model_complete(self, state):
        chunks = state["chunks"]
        meta = state["meta"]
        sender_k = state.get("sender_k", self.k)  # Use sender's k

        if not meta:
            return False

        # We need to know total chunks.
        # We can get it from any meta entry (assuming consistency)
        first_meta = next(iter(meta.values()))
        total_chunks = first_meta["total"]

        if len(chunks) < total_chunks:
            return False

        for i in range(total_chunks):
            if i not in chunks:
                return False
            if len(chunks[i]) < sender_k:  # Use sender's k
                return False

        return True

    def _reassemble_model(self, state):
        try:
            chunks = state["chunks"]
            meta = state["meta"]
            sender_k = state.get("sender_k", self.k)
            sender_m = state.get("sender_m", self.m)

            full_data = b""
            chunks_recovered_with_fec = 0

            sorted_chunk_indices = sorted(chunks.keys())

            # ====== NO-FEC MODE (k=m): Skip FEC decode ======
            if sender_k == sender_m:
                self.logger.debug(f"NO-FEC mode detected (k=m={sender_k}), skipping FEC decode")
                for i in sorted_chunk_indices:
                    blocks_map = chunks[i]
                    # Sort blocks by index and concatenate
                    sorted_blocks = [blocks_map[b_idx] for b_idx in sorted(blocks_map.keys())]
                    chunk_data = b"".join(sorted_blocks)
                    original_len = meta[i]["len"]
                    chunk_data = chunk_data[:original_len]
                    full_data += chunk_data

                self.logger.debug(f"✅ Reassembled model (NO-FEC) ({len(full_data)} bytes)")
                return full_data

            # ====== FEC MODE (k<m): Use FEC decode ======
            # Create a decoder with sender's FEC params
            decoder = zfec.Decoder(sender_k, sender_m)
            fec_decode_start = time.time()  # FEC デコード時間計測開始

            for i in sorted_chunk_indices:
                blocks_map = chunks[i]
                blocks = []
                block_nums = []
                for b_idx, b_data in blocks_map.items():
                    blocks.append(b_data)
                    block_nums.append(b_idx)
                    if len(blocks) == sender_k:  # Use sender's k
                        break

                # Check if we're using FEC for recovery (missing some blocks but still able to decode)
                received_blocks = len(blocks_map)
                if received_blocks < sender_m and received_blocks >= sender_k:
                    chunks_recovered_with_fec += 1

                decoded = decoder.decode(blocks, block_nums)  # Use dynamic decoder
                chunk_data = b"".join(decoded)
                original_len = meta[i]["len"]
                chunk_data = chunk_data[:original_len]
                full_data += chunk_data

            # FEC デコード時間計測終了
            fec_decode_time_ms = (time.time() - fec_decode_start) * 1000
            with self.stats_lock:
                self.stats["fec_decode_time_ms"] += fec_decode_time_ms

            # Stats are tracked by _peer_worker_loop with proper locking
            if chunks_recovered_with_fec > 0:
                with self.stats_lock:
                    self.stats["fec_recovery_success"] += chunks_recovered_with_fec
                self.logger.info(f"✅ Reassembled model ({len(full_data)} bytes, k={sender_k}, m={sender_m}) - FEC recovered {chunks_recovered_with_fec}/{len(sorted_chunk_indices)} chunks, decode={fec_decode_time_ms:.1f}ms")
            else:
                self.logger.debug(f"✅ Reassembled model ({len(full_data)} bytes, k={sender_k}, m={sender_m}) - no FEC recovery needed, decode={fec_decode_time_ms:.1f}ms")

            return full_data
        except Exception as e:
            self.logger.error(f"❌ Reassembly failed: {e}")
            with self.stats_lock:
                self.stats["fec_recovery_fail"] += 1
            return None

    def _identify_missing_chunks(self, state):
        """Identify which chunks are missing (or insufficient packets)."""
        chunks = state["chunks"]
        meta = state["meta"]
        if not meta:
            return []  # Can't know what's missing

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
            # Payload: list of 4-byte integers
            payload = b"".join([struct.pack("!I", c) for c in missing_chunks])

            # Header needs to match format: type, ts, seq, ...
            # Most fields are 0/irrelevant for NACK except Type and Seq
            header = struct.pack("!BdIIIIIBBB", self.PTYPE_NACK, time.time(), model_seq, 0, 0, 0, 0, 0, 0, 0)

            sock.sendto(header + payload, (target_ip, self.port))
            sock.close()
        except Exception as e:
            self.logger.error(f"Failed to send NACK: {e}")

    def _handle_nack(self, target_ip, model_seq, missing_chunks):
        """Retransmit requested chunks."""
        cache = None
        with self.sent_cache_lock:
            cache = self.sent_models_cache.get(model_seq)

        if not cache:
            self.logger.warning(f"⚠️ NACK for unknown/expired model {model_seq} from {target_ip}")
            return

        chunks_data = cache["chunks"]
        encoder = cache["encoder"]  # Might need to recreate if not stateless safe? zfec encoder is light.
        # k, m are in cache

        # Filter needed chunks
        needed = [c for c in chunks_data if c[0] in missing_chunks]
        if not needed:
            return

        # DISABLED: NACK-based pacing adjustment (now using fixed pacing)
        # old_pacing = self.pacing_delay
        # self.pacing_delay = min(0.1, self.pacing_delay * 1.25)
        # if self.pacing_delay > old_pacing:
        #     self.logger.debug(f"📉 NACK received, slowing pacing: {old_pacing * 1000:.2f}ms -> {self.pacing_delay * 1000:.2f}ms")

        # Encode and Send using NACK thread pool (avoids thread creation overhead)
        total_chunks = len(chunks_data)
        self.nack_executor.submit(self._resend_worker, target_ip, needed, self.m, self.k, model_seq, encoder, total_chunks)

    def _resend_worker(self, target_ip, chunks_to_send, m, k, model_seq, encoder, total_chunks):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1024 * 1024)

            for chunk_idx, blocks_in, original_len in chunks_to_send:
                blocks_out = encoder.encode(blocks_in)
                # Resend ALL m packets (full FEC protection) to maximize recovery chance
                # Under high loss, even retransmissions get dropped, so max parity is needed
                limit = len(blocks_out)  # m = k + parity

                for block_idx in range(limit):
                    block = blocks_out[block_idx]
                    header = struct.pack("!BdIIIIIBBB", self.PTYPE_DATA, time.time(), model_seq, chunk_idx, total_chunks, block_idx, original_len, k, m, 0)
                    sock.sendto(header + block, (target_ip, self.port))
                    if self.pacing_delay > 0:
                        time.sleep(self.pacing_delay)
            sock.close()
            self.logger.info(f"🔄 Resent {len(chunks_to_send)} chunks to {target_ip} (Seq {model_seq})")
        except Exception as e:
            self.logger.error(f"Resend worker failed: {e}")

    def _send_ack(self, target_ip, model_seq, echo_ts):
        """Send ACK to let sender measure RTT."""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            # Encode echo_ts in header timestamp field
            header = struct.pack("!BdIIIIIBBB", self.PTYPE_ACK, echo_ts, model_seq, 0, 0, 0, 0, 0, 0, 0)
            sock.sendto(header, (target_ip, self.port))
            sock.close()
        except Exception:
            pass

    def _handle_ack(self, target_ip, model_seq, echoed_ts):
        """Update RTT metrics and Pacing."""
        now = time.time()
        rtt = now - echoed_ts
        if rtt < 0:
            return  # Clock skew or error

        # RTT を NetworkEstimator に記録
        self._network_estimator.record_rtt(rtt * 1000)  # ms 単位

        # RFC 6298 RTT estimation
        if self.min_rtt == float("inf"):
            self.min_rtt = rtt
            self.smoothed_rtt = rtt
            self.rtt_var = rtt / 2
        else:
            self.min_rtt = min(self.min_rtt, rtt)
            self.rtt_var = 0.75 * self.rtt_var + 0.25 * abs(self.smoothed_rtt - rtt)
            self.smoothed_rtt = 0.875 * self.smoothed_rtt + 0.125 * rtt

        # DISABLED: RTT-based pacing adjustment (now using fixed pacing)
        # Keep RTT measurement for observation only
        # threshold = self.min_rtt * 1.5
        # if self.smoothed_rtt > threshold:
        #     # Congestion detected
        #     self.pacing_delay = min(0.1, self.pacing_delay * 1.05)
        #     self.logger.debug(f"📉 Pacing slowed to {self.pacing_delay * 1000:.2f}ms (RTT {self.smoothed_rtt * 1000:.1f}ms > {threshold * 1000:.1f}ms)")
        # elif self.smoothed_rtt < self.min_rtt * 1.1:
        #     # Link clear - speed up very aggressively
        #     self.pacing_delay = max(0.0001, self.pacing_delay * 0.75)

    def join_multicast_group(self, group_ip):
        """Join a UDP multicast group."""
        try:
            # membership is socket-specific. Apply to listener socket if it exists.
            target_sock = getattr(self, "sock", None)
            if target_sock:
                # Use 4s4s to ensure 8 bytes (IPv4) without architectural padding issues
                mreq = struct.pack("4s4s", socket.inet_aton(group_ip), socket.inet_aton("0.0.0.0"))
                target_sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
                self.logger.info(f"✅ Joined multicast group {group_ip} on listener socket")
            else:
                self.logger.warning(f"⚠️ Listener socket not ready, multicast join postponed for {group_ip}")
        except Exception as e:
            self.logger.error(f"Failed to join multicast {group_ip}: {e}")

    def _is_multicast(self, ip_str):
        try:
            first_byte = int(ip_str.split(".")[0])
            return 224 <= first_byte <= 239
        except Exception:
            return False
