"""Small local analyst for comparing council candidates.

The analyst ranks candidates and records disagreements. It never accepts code;
the independent verifier remains the only acceptance gate.
"""
import json
import re

ANALYST_PROMPT = """You are the local council analyst. Compare candidate Python solutions for the task below.
Use the verifier_pass flags as evidence, but do not invent a pass. Return JSON only:
{{"order":[candidate indexes from strongest to weakest],"consensus":"short statement",
"conflicts":"short statement","gaps":"short statement"}}
Rank candidates by correctness, coverage, and simplicity. Include every index once.

TASK:
{task}

CANDIDATES:
{candidates}
"""

def prompt(task, candidates):
    rows = [{"index": i, "model": model, "verifier_pass": bool(passed),
             "code": (text or "")[:12000]}
            for i, (model, text, passed) in enumerate(candidates)]
    return ANALYST_PROMPT.format(task=task[:12000], candidates=json.dumps(rows, ensure_ascii=False))

def parse(raw, count):
    """Return a safe ranking and compact observations from analyst JSON."""
    if not raw:
        return list(range(count)), None
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return list(range(count)), None
    try:
        data = json.loads(match.group(0))
    except (TypeError, ValueError):
        return list(range(count)), None
    order = data.get("order") if isinstance(data, dict) else None
    if not isinstance(order, list):
        return list(range(count)), None
    try:
        order = [int(index) for index in order]
    except (TypeError, ValueError):
        return list(range(count)), None
    if sorted(order) != list(range(count)):
        return list(range(count)), None
    summary = {key: str(data.get(key, ""))[:500]
               for key in ("consensus", "conflicts", "gaps")}
    return order, summary
