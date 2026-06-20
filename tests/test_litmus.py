"""Offline tests — no backend, no network. Covers the two confirmed extract_code bugs
and the per-line output comparison. Run: `python tests/test_litmus.py` (or pytest)."""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from litmus.verifiers import (  # noqa: E402
    extract_code, _same_output, FunctionalCodeVerifier, StdioCodeVerifier,
)


def test_extract_prefers_program_over_trailing_example():
    # chatty model: real solution first, then a usage-example block
    text = ("Here you go:\n```python\ndef add(a, b):\n    return a + b\n```\n"
            "Example:\n```python\nadd(2, 3)\n```")
    assert extract_code(text) == "def add(a, b):\n    return a + b"


def test_extract_handles_truncated_fence():
    text = "```python\ndef add(a, b):\n    return a + b"  # no closing fence
    code = extract_code(text)
    assert "```" not in code
    assert "def add" in code


def test_extract_none_when_no_code():
    assert extract_code("I can't solve this.") is None
    assert extract_code("") is None
    assert extract_code(None) is None


def test_same_output_ignores_trailing_whitespace_per_line():
    assert _same_output("3\n1 2 3 \nDONE\n", "3\n1 2 3\nDONE\n")
    assert _same_output("hi\n", "hi")
    assert not _same_output("3\n1 2 4\n", "3\n1 2 3\n")


def test_functional_verifier_pass_and_fail():
    tests = "def check(c):\n    assert c(2, 3) == 5\n    assert c(-1, 1) == 0\n"
    good = "```python\ndef add(a, b):\n    return a + b\n```"
    bad = "```python\ndef add(a, b):\n    return a - b\n```"
    assert FunctionalCodeVerifier(tests, "add").verify(good)
    assert not FunctionalCodeVerifier(tests, "add").verify(bad)


def test_stdio_verifier_pass_and_fail():
    cases = [{"input": "2 3\n", "output": "5\n"}]
    good = "```python\na, b = map(int, input().split())\nprint(a + b)\n```"
    bad = "```python\na, b = map(int, input().split())\nprint(a - b)\n```"
    assert StdioCodeVerifier(cases).verify(good)
    assert not StdioCodeVerifier(cases).verify(bad)


def test_sandbox_blocks_secret_read():
    # The scrubbed env should keep a secret out of the child process.
    os.environ["LITMUS_TEST_SECRET"] = "topsecret"
    try:
        cases = [{"input": "", "output": "none"}]
        # prints the secret if it leaks, else "none"
        prog = ("```python\nimport os\nprint(os.environ.get('LITMUS_TEST_SECRET', 'none'))\n```")
        assert StdioCodeVerifier(cases).verify(prog)  # passes only because secret is absent
    finally:
        del os.environ["LITMUS_TEST_SECRET"]


class _FakeBackend:
    """Deterministic backend for offline engine tests — no network, no models."""
    name = "ollama"

    def __init__(self, responses):
        self.responses = responses  # {model: [text, ...]}

    def complete(self, model, prompt, n=1, temperature=0.7, max_tokens=4000):
        r = self.responses.get(model, [""])
        return [r[i % len(r)] for i in range(n)]


_TESTS = "def check(c):\n    assert c(2, 3) == 5\n"
_GOOD = "```python\ndef add(a, b):\n    return a + b\n```"
_BAD = "```python\ndef add(a, b):\n    return a - b\n```"


def test_engine_single_when_best_solves():
    from litmus.engine import Engine
    from litmus.verifiers import FunctionalCodeVerifier
    eng = Engine(_FakeBackend({"best": [_GOOD]}), panel=["best", "other"], best="best", k=2)
    r = eng.solve("add", FunctionalCodeVerifier(_TESTS, "add"))
    assert r.verified and r.stage == "single" and r.model == "best"


def test_engine_escalates_to_council():
    from litmus.engine import Engine
    from litmus.verifiers import FunctionalCodeVerifier
    # best always wrong; the other council member is correct -> must escalate and pick it
    eng = Engine(_FakeBackend({"best": [_BAD], "other": [_GOOD]}),
                 panel=["best", "other"], best="best", k=2)
    r = eng.solve("add", FunctionalCodeVerifier(_TESTS, "add"))
    assert r.verified and r.stage == "council" and r.model == "other"


def test_engine_unverified_when_nobody_solves():
    from litmus.engine import Engine
    from litmus.verifiers import FunctionalCodeVerifier
    eng = Engine(_FakeBackend({"best": [_BAD], "other": [_BAD]}),
                 panel=["best", "other"], best="best", k=2)
    r = eng.solve("add", FunctionalCodeVerifier(_TESTS, "add"))
    assert not r.verified and r.stage == "unverified"


def test_functional_verifier_accepts_body_only():
    # the documented form: just the body of check(), no `def check(candidate):`
    body = "assert candidate(2, 3) == 5\nassert candidate(-1, 1) == 0\n"
    assert FunctionalCodeVerifier(body, "add").verify(_GOOD)
    assert not FunctionalCodeVerifier(body, "add").verify(_BAD)


def test_functional_verifier_from_cases():
    v = FunctionalCodeVerifier.from_cases("add", [((2, 3), 5), ((-1, 1), 0), ((0, 0), 0)])
    assert v.verify(_GOOD)
    assert not v.verify(_BAD)
    # single non-tuple arg convenience (for 1-arg functions)
    sq = "```python\ndef square(x):\n    return x * x\n```"
    assert FunctionalCodeVerifier.from_cases("square", [(5, 25), (3, 9)]).verify(sq)


def test_demo_backend_escalates_and_verifies():
    from litmus.engine import Engine
    from litmus.backends import DemoBackend
    r = Engine(DemoBackend(), k=2).solve(
        "add", FunctionalCodeVerifier.from_cases("add", [((2, 3), 5)]))
    assert r.verified and r.stage == "council" and r.model == "demo-council"


def test_timeout_is_bounded():
    import time
    slow = "```python\ndef f():\n    import time\n    time.sleep(30)\n```"
    t = time.time()
    ok = FunctionalCodeVerifier("def check(c):\n    c()\n", "f", timeout=2).verify(slow)
    assert not ok and (time.time() - t) < 10  # reaped near the timeout, not after 30s


if __name__ == "__main__":
    tests = sorted((k, v) for k, v in globals().items()
                   if k.startswith("test_") and callable(v))
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print("PASS", name)
        except AssertionError as e:
            failed += 1
            print("FAIL", name, e)
        except Exception as e:
            failed += 1
            print("ERROR", name, repr(e))
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
