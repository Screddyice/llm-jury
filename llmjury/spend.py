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

# What an escalation WOULD have cost on OpenRouter, per million tokens, when the
# Codex CLI served it instead. Override with LLMJURY_AVOIDED_{INPUT,OUTPUT}_PER_MTOK.
#
# Defaults are deepseek-v4-pro's rates: the ladder's middle rung, and the tier a
# Codex escalation most often stands in for. The first rung is cheaper and the
# Anthropic rung is far dearer, so this is a deliberately conservative middle --
# an avoided-cost figure should under-claim rather than flatter itself.
AVOIDED_INPUT_PER_MTOK = float(os.environ.get("LLMJURY_AVOIDED_INPUT_PER_MTOK", 0.40))
AVOIDED_OUTPUT_PER_MTOK = float(os.environ.get("LLMJURY_AVOIDED_OUTPUT_PER_MTOK", 1.60))
# The Codex CLI returns text, not token counts, so tokens are estimated from
# characters. Four is the usual English rule of thumb and is close enough for a
# figure already labelled an estimate.
CHARS_PER_TOKEN = 4.0


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


def record_subscription(backend, model, prompt="", completion=""):
    """Record an escalation a subscription served, and what it avoided paying.

    `--frontier-backend codex` authenticates from the Codex CLI's own session,
    so the call is covered by a subscription already paid for and no metered
    request is made. Nothing therefore reached this ledger, and a report reading
    it saw OpenRouter usage of zero -- true, but indistinguishable from "the
    frontier ladder never ran" when what actually happened is "it ran for free".

    `cost_usd` is 0.0 because nothing was billed. `avoided_usd` is an ESTIMATE of
    the OpenRouter charge that did not occur, flagged with `estimated: true` so a
    consumer can present it as the projection it is rather than as measured
    spend. Never raises, for the reason in the module docstring.
    """
    try:
        path = ledger_path()
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        in_tok = len(prompt or "") / CHARS_PER_TOKEN
        out_tok = len(completion or "") / CHARS_PER_TOKEN
        avoided = (in_tok * AVOIDED_INPUT_PER_MTOK
                   + out_tok * AVOIDED_OUTPUT_PER_MTOK) / 1e6
        row = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "backend": backend,
            "model": model,
            "billing": "subscription",
            "prompt_tokens": int(in_tok),
            "completion_tokens": int(out_tok),
            "cost_usd": 0.0,
            "avoided_usd": round(avoided, 6),
            "estimated": True,
        }
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")
    except Exception:
        pass
