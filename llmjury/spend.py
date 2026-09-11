"""Append-only ledger of what llm-jury's metered backends actually cost.

Why this exists: llm-jury's frontier ladder spends real money on OpenRouter,
and none of it appears in a Claude or Codex transcript because llm-jury runs in
its own process. The weekly model-economics report therefore could not see it.

Why a ledger and not an API call: the consumer previously read
``OPENROUTER_API_KEY`` out of ``~/.llmjury/.env`` and queried the account. That
was removed on 2026-08-26 for two good reasons — one system should not hold
another's credential, and an account-wide total cannot be attributed to a
client anyway. A ledger written by the process that made the call is both
key-free and precisely attributed.

Contract, one JSON object per line:

    {"ts": ISO-8601 UTC, "backend": str, "model": str,
     "prompt_tokens": int, "completion_tokens": int, "cost_usd": float}

Recording is best-effort by design. A billing side-effect that can fail the
solve it was measuring is worse than no ledger at all, so every error here is
swallowed.
"""
import json
import os
from datetime import datetime, timezone

DEFAULT_LEDGER = os.path.expanduser("~/.llmjury/spend.jsonl")


def ledger_path():
    return os.environ.get("LLMJURY_SPEND_LEDGER", DEFAULT_LEDGER)


def record(backend, model, usage):
    """Append one call's cost. Never raises.

    `usage` is the provider's usage block. Falsy means nothing was measured,
    and we write nothing: a zero we invented would claim the call was free.
    """
    if not usage:
        return
    try:
        path = ledger_path()
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        row = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "backend": backend,
            "model": model,
            "prompt_tokens": int(usage.get("prompt_tokens") or 0),
            "completion_tokens": int(usage.get("completion_tokens") or 0),
            "cost_usd": float(usage.get("cost") or 0.0),
        }
        # O_APPEND with one write per line keeps concurrent council members from
        # interleaving inside a record.
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")
    except Exception:
        # Deliberately silent: see the module docstring.
        pass
