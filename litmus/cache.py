"""Tiny JSONL response cache so reruns are near-free."""
import os
import json
import hashlib
import threading


class Cache:
    def __init__(self, path):
        self.path = os.path.expanduser(path)
        self._d = {}
        self._lock = threading.Lock()
        d = os.path.dirname(self.path) or "."
        os.makedirs(d, exist_ok=True)
        try:
            os.chmod(d, 0o700)  # prompts + responses can be sensitive
        except OSError:
            pass
        if os.path.exists(self.path):
            with open(self.path, encoding="utf-8") as fh:
                for line in fh:
                    try:
                        o = json.loads(line)
                        self._d[o["k"]] = o["v"]
                    except Exception:
                        pass

    @staticmethod
    def key(*parts):
        return hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()

    def get(self, k):
        return self._d.get(k)

    def put(self, k, v):
        with self._lock:
            self._d[k] = v
            # Single os.write of pre-serialized bytes keeps appends robust across
            # processes; 0600 so prompts/responses aren't world-readable.
            line = (json.dumps({"k": k, "v": v}, ensure_ascii=False) + "\n").encode("utf-8")
            fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            try:
                os.write(fd, line)
            finally:
                os.close(fd)
