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
