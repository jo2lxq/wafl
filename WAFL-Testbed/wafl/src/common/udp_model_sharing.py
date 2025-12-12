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
    HEADER_SIZE = 16  # 4 bytes seq, 4 bytes total_packets, 4 bytes k, 4 bytes m
    PAYLOAD_SIZE = MAX_PACKET_SIZE - HEADER_SIZE

    def __init__(self, ip: str, port: int, fec_m: int = 8, timeout: float = 5.0):
        self.ip = ip
        self.port = port
        # Research Plan: "1 XOR redundant packet for M data chunks"
        # So k = M, m = M + 1
        self.k = fec_m
        self.m = fec_m + 1
        self.timeout = timeout
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
            "failed_chunks": 0,
            "bytes_sent": 0,
            "bytes_received": 0,
        }

        self.logger.info(f"UDPModelSharing initialized: k={self.k}, m={self.m} (FEC redundancy={1.0 / self.m:.1%})")

    def send_model(self, model_data: bytes, target_ip: str, target_port: int) -> bool:
        """
        Sends serialized model data via UDP with FEC.
        """
        try:
            # Pad data to be multiple of k
            pad_len = (self.k - (len(model_data) % self.k)) % self.k
            padded_data = model_data + b"\0" * pad_len

            # Encode
            blocks = self.encoder.encode(padded_data)

            # Prepare packets
            # packets = []
            # num_blocks = len(blocks)

            # We need to split blocks into chunks if they are too large,
            # but zfec blocks are usually (len / k).
            # If len is large, block size is large.
            # We need to fragment the blocks or fragment the data BEFORE encoding?
            # zfec encodes the whole buffer into k blocks of size len/k.
            # If len/k > MTU, we have a problem.
            # So we should chunk the data first, then encode each chunk?
            # Or just use zfec on chunks.

            # Strategy: Split data into chunks of size (PAYLOAD_SIZE * k).
            # Each chunk is encoded into m packets of size PAYLOAD_SIZE.

            chunk_size = self.PAYLOAD_SIZE * self.k
            total_chunks = math.ceil(len(model_data) / chunk_size)

            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

            for chunk_idx in range(total_chunks):
                start = chunk_idx * chunk_size
                end = min(start + chunk_size, len(model_data))
                chunk = model_data[start:end]

                # Pad chunk
                chunk_pad = (self.k - (len(chunk) % self.k)) % self.k
                chunk_padded = chunk + b"\0" * chunk_pad

                blocks = self.encoder.encode(chunk_padded)

                for block_idx, block in enumerate(blocks):
                    # Header: chunk_id (4), total_chunks (4), block_idx (4), original_len (4)
                    # We need to know original length to strip padding.
                    header = struct.pack("!IIII", chunk_idx, total_chunks, block_idx, len(chunk))
                    packet = header + block
                    sock.sendto(packet, (target_ip, target_port))
                    # Pacing
                    time.sleep(0.0001)
            sock.close()

            total_packets = total_chunks * self.m
            self.logger.info(f"📡 Sent model via UDP ({len(model_data)} bytes, {total_chunks} chunks, {total_packets} packets with FEC) to {target_ip}:{target_port}")
            self.stats["sent_models"] += 1
            self.stats["bytes_sent"] += len(model_data)
            return True

        except Exception as e:
            self.logger.error(f"❌ Failed to send model via UDP: {e}")
            self.stats["sent_failed"] += 1
            return False

    def get_survival_rate(self) -> float:
        """Calculate model survival rate (successful receptions / total reception attempts).

        This measures how well FEC helps recover from packet loss.
        A rate of 1.0 means all models were successfully received/recovered.
        """
        # Reception-based survival rate
        total_receive_attempts = self.stats["received_models"] + self.stats["fec_recovery_fail"]
        if total_receive_attempts == 0:
            # No reception attempts - check if any sends occurred
            if self.stats["sent_models"] > 0:
                return 1.0  # Sent but not received yet (likely still in transit)
            return 1.0  # No activity
        return self.stats["received_models"] / total_receive_attempts

    def start_listener(self, callback):
        """
        Starts a thread to listen for UDP packets and reassemble models.
        callback(data: bytes, source_ip: str)
        """
        self.running = True
        self.thread = threading.Thread(target=self._listener_loop, args=(callback,), daemon=True)
        self.thread.start()

    def _listener_loop(self, callback):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(("0.0.0.0", self.port))
        sock.settimeout(1.0)

        # State: source_addr -> {chunk_idx: {block_idx: data}, meta: {}}
        incoming_models = {}

        self.logger.info(f"UDP Listener started on port {self.port}")

        while self.running:
            try:
                data, addr = sock.recvfrom(self.MAX_PACKET_SIZE + 100)

                if len(data) < 16:
                    continue

                header = data[:16]
                payload = data[16:]

                chunk_idx, total_chunks, block_idx, original_len = struct.unpack("!IIII", header)

                source_key = addr

                if source_key not in incoming_models:
                    incoming_models[source_key] = {
                        "chunks": {},
                        "meta": {},
                        "last_update": time.time(),
                    }

                model_state = incoming_models[source_key]
                model_state["last_update"] = time.time()

                if chunk_idx not in model_state["chunks"]:
                    model_state["chunks"][chunk_idx] = {}
                    model_state["meta"][chunk_idx] = {
                        "total": total_chunks,
                        "len": original_len,
                    }

                model_state["chunks"][chunk_idx][block_idx] = payload

                # Check if model is complete
                if self._is_model_complete(model_state):
                    full_data = self._reassemble_model(model_state)
                    if full_data:
                        callback(full_data, addr[0])
                    del incoming_models[source_key]

                # Cleanup old partial models
                current_time = time.time()
                to_remove = []
                for key, state in incoming_models.items():
                    if current_time - state["last_update"] > self.timeout:
                        to_remove.append(key)
                for key in to_remove:
                    del incoming_models[key]

            except socket.timeout:
                continue
            except Exception as e:
                self.logger.error(f"Error in UDP listener: {e}")

    def _is_model_complete(self, state):
        chunks = state["chunks"]
        meta = state["meta"]

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
            if len(chunks[i]) < self.k:
                return False

        return True

    def _reassemble_model(self, state):
        try:
            chunks = state["chunks"]
            meta = state["meta"]
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
                    if len(blocks) == self.k:
                        break

                # Check if we're using FEC for recovery (missing some blocks but still able to decode)
                received_blocks = len(blocks_map)
                if received_blocks < self.m and received_blocks >= self.k:
                    chunks_recovered_with_fec += 1

                decoded = self.decoder.decode(blocks, block_nums)
                chunk_data = b"".join(decoded)
                original_len = meta[i]["len"]
                chunk_data = chunk_data[:original_len]
                full_data += chunk_data

            self.stats["received_models"] += 1
            self.stats["bytes_received"] += len(full_data)

            if chunks_recovered_with_fec > 0:
                self.stats["fec_recovery_success"] += chunks_recovered_with_fec
                self.logger.info(f"✅ Reassembled model ({len(full_data)} bytes) - FEC recovered {chunks_recovered_with_fec}/{len(sorted_chunk_indices)} chunks")
            else:
                self.logger.info(f"✅ Reassembled model ({len(full_data)} bytes) - no FEC recovery needed")

            return full_data
        except Exception as e:
            self.logger.error(f"❌ Reassembly failed: {e}")
            self.stats["fec_recovery_fail"] += 1
            return None
