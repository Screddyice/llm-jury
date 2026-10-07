"""Cross-process model reservations for two independent verified local tasks.

Slot file locks are the source of ownership, so a crash releases a reservation
without trusting a timeout or a reused PID. The coordinator makes selection,
aggregate memory admission, and publication one atomic operation. No task text,
candidate code, or credentials enter the reservation files.
"""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import sys
import time

from . import memguard


def choose_model(models, preferred, busy):
    occupied = set(busy)
    if preferred in models and preferred not in occupied:
        return preferred
    for model in models:
        if model not in occupied:
            return model
    return None


class LocalScheduler:
    def __init__(self, host, num_ctx=8192, wait_seconds=30):
        self.host = host.rstrip("/")
        self.num_ctx = num_ctx or memguard.SERVER_DEFAULT_CTX
        self.wait_seconds = wait_seconds
        lock = Path(os.environ.get("LLMJURY_LOCAL_LOCK") or
                    Path.home() / ".cache/llmjury/local-compute.lock")
        self.root = lock.with_name(lock.name + ".tasks")
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)

    def require_available(self):
        exclusive, owner = memguard.exclusive_compute(self.host)
        if exclusive:
            raise RuntimeError("exclusive 27B compute is active: " + owner)

    @contextmanager
    def reserve(self, models, preferred=None):
        """Reserve one idle model, or yield None for verifier-gated fallback."""
        import fcntl
        candidates = list(dict.fromkeys(models))
        # Ollama treats the optional :latest suffix as the same model lane.
        canonical = lambda model: model.removesuffix(":latest")
        deadline = time.monotonic() + self.wait_seconds
        owned = None
        selected = None
        try:
            while candidates:
                self.require_available()
                with (self.root / "coordinator.lock").open("a+") as coordinator:
                    fcntl.flock(coordinator, fcntl.LOCK_EX)
                    active, free = [], []
                    try:
                        for index in range(2):
                            handle = (self.root / f"slot-{index}.json").open("a+")
                            try:
                                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                                free.append(handle)
                            except BlockingIOError:
                                try:
                                    handle.seek(0)
                                    row = json.load(handle)
                                    if (not isinstance(row, dict) or
                                            not isinstance(row.get("model"), str) or
                                            not row["model"] or
                                            type(row.get("num_ctx")) is not int or
                                            row["num_ctx"] <= 0 or
                                            row.get("host") != self.host):
                                        raise ValueError("invalid or incompatible reservation")
                                    active.append(row)
                                except (ValueError, OSError) as error:
                                    raise RuntimeError("cannot read active local reservation") from error
                                finally:
                                    handle.close()
                        busy = [canonical(row["model"]) for row in active]
                        choices = list(candidates)
                        refusal = None
                        while free and choices:
                            name = choose_model([canonical(m) for m in choices],
                                                canonical(preferred) if preferred else None,
                                                busy)
                            if name is None:
                                break
                            model = next(m for m in choices if canonical(m) == name)
                            union = [row["model"] for row in active] + [model]
                            context = max([self.num_ctx] + [row["num_ctx"] for row in active])
                            report = memguard.check(union, host=self.host, num_ctx=context)
                            if report.terminal:
                                raise RuntimeError(report.message())
                            if report.ok:
                                owned = free.pop(0)
                                owned.seek(0)
                                owned.truncate()
                                json.dump({"model": model, "num_ctx": self.num_ctx,
                                           "host": self.host, "pid": os.getpid()}, owned)
                                owned.flush()
                                selected = model
                                break
                            choices.remove(model)
                            refusal = report
                        if not selected and refusal is not None:
                            sys.stderr.write("[llmjury] local scheduler refused admission: "
                                             + refusal.message() + "\n")
                            candidates = []
                    finally:
                        for handle in free:
                            handle.close()
                if selected or not candidates:
                    break
                if time.monotonic() >= deadline:
                    sys.stderr.write("[llmjury] local model lanes busy; using configured fallback\n")
                    break
                time.sleep(0.05)
            if selected:
                sys.stderr.write(f"[llmjury] local scheduler assigned {selected}\n")
            yield selected
        finally:
            if owned is not None:
                # Retain the model metadata until flock releases it. Readers holding
                # the coordinator ignore stale content in every unlocked slot.
                owned.close()
