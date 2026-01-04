import collections
import logging
import math
import socket
import struct
import threading
import time

import zfec


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

    MAX_RETRIES = 10  # Max NACK attempts (Increased for resilience)

    def __init__(self, ip: str, port: int, fec_m: int = 8, timeout: float = 3.0, inter_packet_timeout: float = 0.5):
        self.ip = ip
        self.port = port

        # FEC Settings
        # k: data packets per chunk
        # parity: redundant packets per chunk
        # m = k + parity (total packets per chunk)
        self.k = fec_m
        # Default parity: 50% redundancy for conservative startup (will be adjusted by Adaptive FEC)
        self.parity = max(4, math.ceil(self.k * 0.5))
        self.m = self.k + self.parity

        # Pacing Control
        # Default pacing: 0.1ms per packet (~112Mbps) - Aggressive start for low-latency
        self.pacing_delay = 0.0001

        self.timeout = timeout  # Model completion timeout
        self.inter_packet_timeout = inter_packet_timeout  # Time between packets (Default 0.2s by caller preferred)
        self.logger = logging.getLogger("UDPModelSharing")
        self.encoder = zfec.Encoder(self.k, self.m)
        self.decoder = zfec.Decoder(self.k, self.m)

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

    def send_model(self, model_data: bytes, target_ip: str, target_port: int) -> bool:
        """
        Sends serialized model data via UDP with FEC and Pacing.
        """
        try:
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
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1024 * 1024)  # 1MB

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

            # Parallel Encoding using ThreadPoolExecutor
            # zfec releases GIL during encoding so this should speed up CPU-bound tasks
            from concurrent.futures import ThreadPoolExecutor

            num_workers = min(4, total_chunks) if total_chunks > 0 else 1

            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                # Submit all encoding tasks
                futures = []
                for chunk_idx, blocks_in, original_len in chunks_data:
                    futures.append(executor.submit(self.encoder.encode, blocks_in))

                # Send loop with pacing
                for i, future in enumerate(futures):
                    chunk_idx, _, original_len = chunks_data[i]
                    blocks_out = future.result()

                    for block_idx, block in enumerate(blocks_out):
                        # Header: type(1), timestamp(8), model_seq(4), chunk_idx(4), total_chunks(4), block_idx(4), original_len(4), k(1), m(1), pad(1)
                        # Padding last byte with 0
                        header = struct.pack("!BdIIIIIBBB", self.PTYPE_DATA, time.time(), current_seq, chunk_idx, total_chunks, block_idx, original_len, self.k, self.m, 0)
                        packet = header + block
                        sock.sendto(packet, (target_ip, target_port))
                        actual_bytes_sent += len(packet)

                        # PACING: Sleep to control burst rate
                        if self.pacing_delay > 0:
                            time.sleep(self.pacing_delay)

            # Send END packet for Fast NACK trigger
            # Header: PTYPE_END, ts, seq, 0, 0, 0, 0, 0, 0, 0
            end_header = struct.pack("!BdIIIIIBBB", self.PTYPE_END, time.time(), current_seq, 0, 0, 0, 0, self.k, self.m, 0)
            sock.sendto(end_header, (target_ip, target_port))

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

            return True

        except Exception as e:
            self.logger.error(f"❌ Failed to send model via UDP: {e}")
            self.stats["sent_failed"] += 1
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
        self.thread = threading.Thread(target=self._listener_loop, args=(callback,), daemon=True)
        self.thread.start()

    def _listener_loop(self, callback):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(("0.0.0.0", self.port))
        self.sock.settimeout(2.0)  # Reduced for faster cleanup checks

        # Increase receive buffer to avoid OS-level drops
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 2 * 1024 * 1024)  # 2MB

        # State: (source_ip, model_seq) -> {chunk_idx: {block_idx: data}, meta: {}}
        incoming_models = {}

        # Track completed (IP, model_seq) pairs to ignore late FEC packets
        # set of (IP, model_seq) tuples
        completed_models = set()
        completed_history = collections.deque(maxlen=1000)  # For cleaning up completed_models set

        self.logger.info(f"UDP Listener started on port {self.port}")

        while self.running:
            try:
                data, addr = self.sock.recvfrom(self.MAX_PACKET_SIZE + 100)

                if len(data) < self.HEADER_SIZE:  # Updated: 22-byte header
                    continue

                header = data[: self.HEADER_SIZE]
                payload = data[self.HEADER_SIZE :]

                # Parse header with k and m (new format 32 bytes)
                ptype, timestamp, model_seq, chunk_idx, total_chunks, block_idx, original_len, sender_k, sender_m, _ = struct.unpack("!BdIIIIIBBB", header)

                if ptype == self.PTYPE_NACK:
                    # Payload contains sequence of missing chunk indices (4 bytes each)
                    missing_chunks = []
                    for i in range(0, len(payload), 4):
                        if i + 4 <= len(payload):
                            missing_chunks.append(struct.unpack("!I", payload[i : i + 4])[0])

                    self.logger.info(f"🔁 NACK received from {addr[0]} for model {model_seq}, missing chunks: {len(missing_chunks)}")
                    self._handle_nack(addr[0], model_seq, missing_chunks)
                    continue

                elif ptype == self.PTYPE_END:
                    # Fast NACK Trigger: Sender finished transmission
                    # Immediately check if we have everything
                    key = (addr[0], model_seq)

                    if key in completed_models:
                        continue

                    if key not in incoming_models:
                        # Even if no data packets arrived, we can now start a state from END packet
                        incoming_models[key] = {
                            "chunks": {},
                            "meta": {},  # Will be initialized below
                            "last_update": time.time(),
                            "first_packet_time": time.time(),
                            "sender_k": sender_k,
                            "sender_m": sender_m,
                            "retries": 0,
                            "last_ts": timestamp,
                        }

                    state = incoming_models[key]
                    state["last_update"] = time.time()

                    # Ensure total_chunks is known in meta for at least one entry (for _identify_missing_chunks)
                    if 0 not in state["meta"]:  # Use chunk 0 as placeholder meta if none received
                        state["meta"][0] = {"total": total_chunks, "len": original_len}

                    if self._is_model_complete(state):
                        continue

                    missing = self._identify_missing_chunks(state)
                    if missing:
                        self._send_nack(addr[0], model_seq, missing)
                        state["retries"] += 1
                        state["last_update"] = time.time()
                        self.logger.info(f"⚡ Fast NACK triggered by END for {addr[0]} m-{model_seq}, missing {len(missing)} chunks")
                    continue

                if ptype == self.PTYPE_ACK:
                    # Echoed timestamp in header is used to calculate RTT
                    self._handle_ack(addr[0], model_seq, timestamp)
                    continue

                # PTYPE_DATA or PTYPE_MCAST
                if ptype not in [self.PTYPE_DATA, self.PTYPE_MCAST]:
                    self.logger.warning(f"Unknown packet type {ptype} from {addr[0]}")
                    continue

                source_ip = addr[0]
                # Use (IP, model_seq) as unique key for each model transmission
                model_key = (source_ip, model_seq)

                # Skip packets from already completed models
                if model_key in completed_models:
                    continue

                if model_key not in incoming_models:
                    incoming_models[model_key] = {
                        "chunks": {},
                        "meta": {},
                        "last_update": time.time(),
                        "first_packet_time": time.time(),  # Track when first packet arrived
                        "sender_k": sender_k,  # Store sender's FEC params
                        "sender_m": sender_m,
                        "retries": 0,  # NACK retry count
                        "last_ts": timestamp,  # Store timestamp of last packet for ACK
                    }

                # Update last timestamp
                incoming_models[model_key]["last_ts"] = timestamp

                model_state = incoming_models[model_key]
                model_state["last_update"] = time.time()

                if chunk_idx not in model_state["chunks"]:
                    model_state["chunks"][chunk_idx] = {}
                    model_state["meta"][chunk_idx] = {
                        "total": total_chunks,
                        "len": original_len,
                    }

                model_state["chunks"][chunk_idx][block_idx] = payload

                # Check if model is complete (early completion when k packets received)
                if self._is_model_complete(model_state):
                    full_data = self._reassemble_model(model_state)
                    if full_data:
                        callback(full_data, source_ip)
                        # Send ACK with RTT info
                        last_ts = model_state.get("last_ts", time.time())
                        self._send_ack(source_ip, model_seq, last_ts)

                    del incoming_models[model_key]
                    completed_models.add(model_key)
                    completed_history.append(model_key)
                    if len(completed_history) >= 1000:
                        # Sliding window cleanup
                        old_key = completed_history.popleft()
                        completed_models.discard(old_key)

                # Cleanup models based on timeouts
                current_time = time.time()
                to_remove = []
                for key, state in incoming_models.items():
                    time_since_last_packet = current_time - state["last_update"]
                    time_since_first_packet = current_time - state.get("first_packet_time", current_time)

                    # Inter-packet timeout: no packet received for too long
                    # This enables fast failure detection when packets stop arriving
                    if time_since_last_packet > self.inter_packet_timeout:
                        if state["retries"] < self.MAX_RETRIES:
                            # Try NACK (Fast Retransmit on silence)
                            missing = self._identify_missing_chunks(state)
                            if missing:
                                self._send_nack(key[0], key[1], missing)
                                state["retries"] += 1
                                state["last_update"] = current_time  # Reset timer to allow retransmission arrival
                                self.logger.info(f"🔁 Sending NACK (Inter-packet) to {key[0]} for model {key[1]}, requesting {len(missing)} chunks (Retry {state['retries']})")
                            else:
                                to_remove.append(key)
                        else:
                            to_remove.append(key)

                    # Model completion timeout: total time exceeded
                    elif time_since_first_packet > self.timeout:
                        to_remove.append(key)

                for key in to_remove:
                    # Remove without counting as timeout_models (handled at higher level)
                    # Note: stats['timeout_models'] should theoretically be incremented here if we want strictly accurate UDP stats internal
                    # but UDPModelSharing.get_survival_rate uses it.
                    self.stats["timeout_models"] += 1
                    del incoming_models[key]

            except socket.timeout:
                # Socket timeout - check for inter_packet timeout on all pending models
                current_time = time.time()
                to_remove = []
                for key, state in incoming_models.items():
                    time_since_last_packet = current_time - state["last_update"]
                    if time_since_last_packet > self.inter_packet_timeout:
                        to_remove.append(key)
                for key in to_remove:
                    self.stats["timeout_models"] += 1  # Count for accurate survival rate
                    del incoming_models[key]
                continue
            except Exception as e:
                self.logger.error(f"Error in UDP listener: {e}")

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

            # Create a decoder with sender's FEC params
            decoder = zfec.Decoder(sender_k, sender_m)

            full_data = b""
            chunks_recovered_with_fec = 0

            sorted_chunk_indices = sorted(chunks.keys())
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

            self.stats["received_models"] += 1
            self.stats["bytes_received"] += len(full_data)
            self.stats["total_chunks_received"] += len(sorted_chunk_indices)

            if chunks_recovered_with_fec > 0:
                self.stats["fec_recovery_success"] += chunks_recovered_with_fec
                self.logger.info(f"✅ Reassembled model ({len(full_data)} bytes, k={sender_k}, m={sender_m}) - FEC recovered {chunks_recovered_with_fec}/{len(sorted_chunk_indices)} chunks")
            else:
                self.logger.info(f"✅ Reassembled model ({len(full_data)} bytes, k={sender_k}, m={sender_m}) - no FEC recovery needed")

            return full_data
        except Exception as e:
            self.logger.error(f"❌ Reassembly failed: {e}")
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

        # NACK indicates congestion/loss -> Slow down pacing
        old_pacing = self.pacing_delay
        self.pacing_delay = min(0.1, self.pacing_delay * 1.25)
        if self.pacing_delay > old_pacing:
            self.logger.debug(f"📉 NACK received, slowing pacing: {old_pacing * 1000:.2f}ms -> {self.pacing_delay * 1000:.2f}ms")

        # Encode and Send (Simple synchronous send for retransmission or use executor?)
        # Use executor to avoid blocking listener
        total_chunks = len(chunks_data)
        threading.Thread(target=self._resend_worker, args=(target_ip, needed, self.m, self.k, model_seq, encoder, total_chunks)).start()

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

        # RFC 6298 RTT estimation
        if self.min_rtt == float("inf"):
            self.min_rtt = rtt
            self.smoothed_rtt = rtt
            self.rtt_var = rtt / 2
        else:
            self.min_rtt = min(self.min_rtt, rtt)
            self.rtt_var = 0.75 * self.rtt_var + 0.25 * abs(self.smoothed_rtt - rtt)
            self.smoothed_rtt = 0.875 * self.smoothed_rtt + 0.125 * rtt

        # Pacing Adjustment (BBR-like simple logic)
        # If RTT is inflating (queued), slow down (increase pacing)
        # If RTT is close to min, speed up (decrease pacing)

        threshold = self.min_rtt * 1.5
        if self.smoothed_rtt > threshold:
            # Congestion detected
            self.pacing_delay = min(0.1, self.pacing_delay * 1.05)
            self.logger.debug(f"📉 Pacing slowed to {self.pacing_delay * 1000:.2f}ms (RTT {self.smoothed_rtt * 1000:.1f}ms > {threshold * 1000:.1f}ms)")
        elif self.smoothed_rtt < self.min_rtt * 1.1:
            # Link clear - speed up very aggressively
            self.pacing_delay = max(0.0001, self.pacing_delay * 0.75)
            # self.logger.debug(f"📈 Pacing sped up to {self.pacing_delay*1000:.2f}ms")

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
