import json
import os
import threading
import time


class MetricsLogger:
    def __init__(self, experiment_id, agent_id, log_dir="./results"):
        self.agent_id = agent_id
        self.log_file = os.path.join(log_dir, experiment_id, f"metrics_{agent_id}.jsonl")
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        self.lock = threading.Lock()

    def log(self, event_type, **kwargs):
        entry = {
            "timestamp": time.time(),
            "node": self.agent_id,
            "type": event_type,
        }
        entry.update(kwargs)

        json_line = json.dumps(entry)

        with self.lock:
            with open(self.log_file, "a") as f:
                f.write(json_line + "\n")


# Singleton or global instance management could be added here if needed
