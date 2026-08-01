"""Offline tests — no backend, no network. Covers the two confirmed extract_code bugs
and the per-line output comparison. Run: `python tests/test_llmjury.py` (or pytest)."""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import shutil  # noqa: E402
import subprocess  # noqa: E402
import json  # noqa: E402
import tempfile  # noqa: E402
from pathlib import Path  # noqa: E402

from llmjury import verifiers  # noqa: E402
from llmjury.verifiers import (  # noqa: E402
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


def test_functional_verifier_rejects_non_python_oracle_before_generation():
    """A malformed oracle must fail before any paid or long-running model call."""
    invalid = 'const cursor: Cursor = { id: "first" };\n'
    try:
        FunctionalCodeVerifier(invalid, "solve")
        assert False, "TypeScript passed as --tests must be rejected"
    except ValueError as error:
        assert "valid Python" in str(error)


def test_cli_rejects_invalid_tests_before_constructing_backend():
    from types import SimpleNamespace
    from unittest.mock import patch
    from llmjury.cli import cmd_solve

    with tempfile.TemporaryDirectory() as tmp:
        task = Path(tmp) / "task.txt"
        tests = Path(tmp) / "tests.ts"
        task.write_text("implement solve", encoding="utf-8")
        tests.write_text("type Cursor = { id: string }\n", encoding="utf-8")
        args = SimpleNamespace(
            task=str(task), tests=str(tests), cases=None, entry_point=None,
        )
        with patch(
            "llmjury.cli._backend",
            side_effect=AssertionError("backend constructed before verifier validation"),
        ) as backend:
            try:
                cmd_solve(args)
                assert False, "invalid tests must stop the CLI"
            except SystemExit as error:
                assert "invalid --tests verifier" in str(error)
            backend.assert_not_called()


def test_stdio_verifier_pass_and_fail():
    cases = [{"input": "2 3\n", "output": "5\n"}]
    good = "```python\na, b = map(int, input().split())\nprint(a + b)\n```"
    bad = "```python\na, b = map(int, input().split())\nprint(a - b)\n```"
    assert StdioCodeVerifier(cases).verify(good)
    assert not StdioCodeVerifier(cases).verify(bad)


def test_sandbox_blocks_secret_read():
    # The scrubbed env should keep a secret out of the child process.
    os.environ["LLMJURY_TEST_SECRET"] = "topsecret"
    try:
        cases = [{"input": "", "output": "none"}]
        # prints the secret if it leaks, else "none"
        prog = ("```python\nimport os\nprint(os.environ.get('LLMJURY_TEST_SECRET', 'none'))\n```")
        assert StdioCodeVerifier(cases).verify(prog)  # passes only because secret is absent
    finally:
        del os.environ["LLMJURY_TEST_SECRET"]


def _docker_reachable():
    if not shutil.which("docker"):
        return False
    try:
        return subprocess.run(["docker", "info"], stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL, timeout=20).returncode == 0
    except Exception:
        return False


def test_container_cuts_network_when_docker_available():
    # With a daemon reachable, code runs in a --network none container: a benign solution
    # still verifies, but one that tries to phone out during verification must fail.
    # Skipped (returns) where Docker is unavailable, e.g. CI without a daemon — so this
    # never turns CI red, it only adds coverage where a sandbox can actually be built.
    if not _docker_reachable():
        return
    os.environ["LLMJURY_SANDBOX"] = "docker"
    verifiers._provisioned = None                 # force a fresh resolve under the env
    try:
        good = "```python\ndef add(a, b):\n    return a + b\n```"
        net_test = ("import socket\nsocket.setdefaulttimeout(5)\n"
                    "socket.create_connection(('1.1.1.1', 53))\nassert candidate(2, 3) == 5\n")
        assert FunctionalCodeVerifier.from_cases("add", [((2, 3), 5)]).verify(good)
        assert not FunctionalCodeVerifier(net_test, "add").verify(good)
    finally:
        os.environ.pop("LLMJURY_SANDBOX", None)
        verifiers._provisioned = None


class _FakeBackend:
    """Deterministic backend for offline engine tests — no network, no models."""
    name = "ollama"

    def __init__(self, responses):
        self.responses = responses  # {model: [text, ...]}
        self.calls = []

    def complete(self, model, prompt, n=1, temperature=0.7, max_tokens=4000):
        self.calls.append((model, prompt, n, temperature, max_tokens))
        r = self.responses.get(model, [""])
        return [r[i % len(r)] for i in range(n)]


_TESTS = "def check(c):\n    assert c(2, 3) == 5\n"
_GOOD = "```python\ndef add(a, b):\n    return a + b\n```"
_BAD = "```python\ndef add(a, b):\n    return a - b\n```"


def test_engine_single_when_best_solves():
    from llmjury.engine import Engine
    from llmjury.verifiers import FunctionalCodeVerifier
    eng = Engine(_FakeBackend({"best": [_GOOD]}), panel=["best", "other"], best="best", k=2)
    r = eng.solve("add", FunctionalCodeVerifier(_TESTS, "add"))
    assert r.verified and r.stage == "single" and r.model == "best"


def test_engine_escalates_to_council():
    from llmjury.engine import Engine
    from llmjury.verifiers import FunctionalCodeVerifier
    # best always wrong; the other council member is correct -> must escalate and pick it
    eng = Engine(_FakeBackend({"best": [_BAD], "other": [_GOOD]}),
                 panel=["best", "other"], best="best", k=2)
    r = eng.solve("add", FunctionalCodeVerifier(_TESTS, "add"))
    assert r.verified and r.stage == "council" and r.model == "other"


def test_engine_unverified_when_nobody_solves():
    from llmjury.engine import Engine
    from llmjury.verifiers import FunctionalCodeVerifier
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


def test_cli_func_cases_conversion():
    # --cases + --entry-point: JSON cases become (args_tuple, expected) pairs for a call.
    from llmjury.cli import _func_cases
    assert _func_cases([{"args": [2, 3], "expected": 5}]) == [((2, 3), 5)]
    # input/output are accepted as aliases for args/expected
    assert _func_cases([{"input": [2, 3], "output": 5}]) == [((2, 3), 5)]
    # a scalar (non-list) arg is wrapped into a 1-tuple (1-arg functions)
    assert _func_cases([{"args": 5, "expected": 25}]) == [((5,), 25)]
    # JSON types are preserved verbatim (strings stay strings)
    assert _func_cases([{"args": ["a", "b"], "expected": "ab"}]) == [(("a", "b"), "ab")]


def test_cli_func_cases_roundtrip_verifies():
    # The CLI conversion feeds from_cases, which must verify a correct add and reject a wrong one.
    from llmjury.cli import _func_cases
    cases = _func_cases([{"args": [2, 3], "expected": 5}, {"args": [-1, 1], "expected": 0}])
    v = FunctionalCodeVerifier.from_cases("add", cases)
    assert v.verify(_GOOD)
    assert not v.verify(_BAD)


def test_demo_backend_escalates_and_verifies():
    from llmjury.engine import Engine
    from llmjury.backends import DemoBackend
    r = Engine(DemoBackend(), k=2).solve(
        "add", FunctionalCodeVerifier.from_cases("add", [((2, 3), 5)]))
    assert r.verified and r.stage == "council" and r.model == "demo-council"


def test_ollama_backend_disables_thinking_by_default():
    from unittest.mock import patch
    from llmjury.backends import OllamaBackend

    requests = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def read(self):
            return b'{"message":{"content":"answer"}}'

    def urlopen(request, timeout):
        requests.append((request, timeout))
        return Response()

    with patch("llmjury.backends.urllib.request.urlopen", side_effect=urlopen):
        assert OllamaBackend().complete("qwen3:8b", "solve", n=1) == ["answer"]

    body = json.loads(requests[0][0].data)
    assert body["think"] is False


def test_openrouter_timeout_is_reported_once_without_hidden_retries():
    import contextlib
    import io
    import urllib.error
    from unittest.mock import patch
    from llmjury.backends import OpenRouterBackend

    backend = OpenRouterBackend(api_key="test", request_timeout=7)
    timeout_errors = [
        TimeoutError("provider stopped responding"),
        urllib.error.URLError(TimeoutError("provider stopped responding")),
    ]

    for timeout_error in timeout_errors:
        calls = 0

        def timeout(*_args, **_kwargs):
            nonlocal calls
            calls += 1
            raise timeout_error

        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr), \
                patch("llmjury.backends.urllib.request.urlopen", side_effect=timeout):
            assert backend.complete("test/model", "solve", n=1) == [""]

        assert calls == 1
        assert "timed out after 7s" in stderr.getvalue()


def test_openrouter_rejects_non_positive_timeout():
    from llmjury.backends import OpenRouterBackend

    try:
        OpenRouterBackend(api_key="test", request_timeout=0)
        assert False, "zero timeout must be rejected"
    except ValueError as error:
        assert "greater than zero" in str(error)


def test_ollama_cache_separates_thinking_modes():
    from unittest.mock import patch
    from llmjury.backends import OllamaBackend

    calls = []

    class Response:
        def __init__(self, content):
            self.content = content

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def read(self):
            return json.dumps({"message": {"content": self.content}}).encode()

    def urlopen(request, timeout):
        body = json.loads(request.data)
        calls.append(body)
        return Response("reasoned" if body["think"] else "direct")

    with tempfile.TemporaryDirectory() as tmp, \
            patch("llmjury.backends.urllib.request.urlopen", side_effect=urlopen):
        cache = str(Path(tmp) / "responses.jsonl")
        direct = OllamaBackend(cache_path=cache, think=False)
        assert direct.complete("qwen3:8b", "solve", n=1) == ["direct"]
        reasoned = OllamaBackend(cache_path=cache, think=True)
        assert reasoned.complete("qwen3:8b", "solve", n=1) == ["reasoned"]

    assert len(calls) == 2


def test_codex_backend_runs_ephemeral_read_only_generation():
    from llmjury.backends import CodexBackend

    calls = []

    def runner(cmd, **kw):
        calls.append((cmd, kw))
        return subprocess.CompletedProcess(cmd, 0, stdout=_GOOD + "\n", stderr="")

    backend = CodexBackend(runner=runner)
    assert backend.complete("gpt-test", "write add", n=1) == [_GOOD]
    cmd, kw = calls[0]
    assert cmd[:2] == ["codex", "exec"]
    assert "--ephemeral" in cmd and "read-only" in cmd
    assert "--ignore-user-config" in cmd and "--ignore-rules" in cmd
    assert cmd.count("--disable") == 2
    assert "shell_tool" in cmd and "unified_exec" in cmd
    assert cmd[cmd.index("--model") + 1] == "gpt-test"
    assert 'model_reasoning_effort="low"' in cmd
    assert cmd[-1] == "write add"
    assert kw["timeout"] == 600 and not kw["check"]


def test_cli_backend_builds_codex_provider():
    from unittest.mock import patch
    from llmjury.cli import _backend

    with patch("llmjury.backends.shutil.which", return_value="/usr/local/bin/codex"):
        assert _backend("codex").name == "codex"


def test_cli_backend_can_enable_ollama_thinking():
    from llmjury.cli import _backend

    backend = _backend("ollama", think=True)
    assert backend.think is True


def test_cli_auto_frontier_resolves_open_source_ladder():
    from llmjury.cli import _frontier_models
    from llmjury.panels import AUTO_FRONTIER, OPEN_SOURCE_FRONTIER
    assert _frontier_models("auto", "openrouter") == AUTO_FRONTIER
    assert _frontier_models("open", "openrouter") == OPEN_SOURCE_FRONTIER
    assert _frontier_models("custom/model", "openrouter") == "custom/model"
    try:
        _frontier_models("auto", "codex")
        assert False, "auto must reject non-OpenRouter frontier backends"
    except ValueError:
        pass


def test_auto_frontier_tries_open_weight_before_the_paid_top_tier():
    """Cost ordering is the whole point: the proprietary tier must be last."""
    from llmjury.panels import AUTO_FRONTIER, OPEN_SOURCE_FRONTIER, TOP_FRONTIER
    assert AUTO_FRONTIER[:len(OPEN_SOURCE_FRONTIER)] == OPEN_SOURCE_FRONTIER
    assert AUTO_FRONTIER[-1] == TOP_FRONTIER
    assert TOP_FRONTIER not in OPEN_SOURCE_FRONTIER


def test_cli_named_frontier_aliases_resolve_to_slugs():
    from llmjury.cli import _frontier_models
    assert _frontier_models("opus", "openrouter") == ["anthropic/claude-opus-5"]
    assert _frontier_models("fable", "openrouter") == ["anthropic/claude-fable-5"]
    try:
        _frontier_models("opus", "codex")
        assert False, "named ladders must reject non-OpenRouter frontier backends"
    except ValueError:
        pass


def test_openrouter_never_returns_thinking_as_an_answer():
    """A truncated reasoning model leaves content empty and reasoning full of prose.

    Passing that through as a candidate burns a verifier pass on text that was
    never an answer; only reasoning that actually carries code is recoverable.
    """
    from llmjury.backends import _answer_from_reasoning
    assert _answer_from_reasoning("The user is asking for a function.") == ""
    assert _answer_from_reasoning("") == ""
    assert _answer_from_reasoning(None) == ""
    stranded = "```python\ndef add(a, b):\n    return a + b\n```"
    assert _answer_from_reasoning(stranded) == stranded


def test_frontier_tier_gets_token_headroom_for_reasoning():
    """Reasoning tokens count against max_tokens, so the last tier needs more."""
    from llmjury.engine import Engine
    from llmjury.verifiers import FunctionalCodeVerifier
    council = _FakeBackend({"local": [_BAD]})
    frontier = _FakeBackend({"top": [_GOOD]})
    eng = Engine(council, panel=["local"], best="local", k=1, max_tokens=4000,
                 frontier="top", frontier_backend=frontier)
    assert eng.frontier_max_tokens == 8000
    eng.solve("add", FunctionalCodeVerifier.from_cases("add", [((2, 3), 5)]))
    # the council keeps its own budget; only the frontier call is widened
    assert council.calls[0][-1] == 4000
    assert frontier.calls[0][-1] == 8000

    explicit = Engine(council, panel=["local"], best="local", k=1,
                      max_tokens=4000, frontier="top", frontier_backend=frontier,
                      frontier_max_tokens=32000)
    assert explicit.frontier_max_tokens == 32000


def test_engine_frontier_escalation():
    from llmjury.engine import Engine
    # local council fails everything; a separate frontier backend solves it on the last tier
    council = _FakeBackend({"best": [_BAD], "other": [_BAD]})
    frontier_bk = _FakeBackend({"front": [_GOOD]})
    eng = Engine(council, panel=["best", "other"], best="best", k=2,
                 frontier="front", frontier_backend=frontier_bk)
    r = eng.solve("add", FunctionalCodeVerifier.from_cases("add", [((2, 3), 5)]))
    assert r.verified and r.stage == "frontier" and r.model == "front"


def test_engine_frontier_ladder_is_verifier_gated():
    from llmjury.engine import Engine
    from llmjury.verifiers import FunctionalCodeVerifier
    # The local council and cheap cloud tier fail; the strong final tier passes.
    # Ordering matters: later models must not run after a verified result.
    local = _FakeBackend({"local": [_BAD]})
    frontier = _FakeBackend({"flash": [_BAD], "pro": [_GOOD], "unused": [_GOOD]})
    r = Engine(local, panel=["local"], best="local", k=1,
               frontier=["flash", "pro", "unused"], frontier_backend=frontier) \
        .solve("implement add", FunctionalCodeVerifier.from_cases("add", [((2, 3), 5)]))
    assert r.verified and r.stage == "frontier" and r.model == "pro"
    assert [call[0] for call in frontier.calls] == ["flash", "pro"]


def test_timeout_is_bounded():
    import time
    slow = "```python\ndef f():\n    import time\n    time.sleep(30)\n```"
    t = time.time()
    ok = FunctionalCodeVerifier("def check(c):\n    c()\n", "f", timeout=2).verify(slow)
    assert not ok and (time.time() - t) < 10  # reaped near the timeout, not after 30s


def test_engine_routes_brain_panelist_to_its_own_backend():
    # A2: an extra panelist ("jarvis-brain") generates through its OWN backend via
    # `route`, not the shared council backend. Council members are all wrong; only
    # the routed brain solves -> proves routing works and the brain joins the council.
    from llmjury.engine import Engine
    from llmjury.verifiers import FunctionalCodeVerifier
    council = _FakeBackend({"best": [_BAD], "other": [_BAD]})       # no "jarvis-brain" key
    brain = _FakeBackend({"jarvis-brain": [_GOOD]})
    eng = Engine(council, panel=["best", "other", "jarvis-brain"], best="best", k=2,
                 route={"jarvis-brain": brain})
    r = eng.solve("add", FunctionalCodeVerifier(_TESTS, "add"))
    assert r.verified and r.stage == "council" and r.model == "jarvis-brain"


def test_engine_route_empty_uses_shared_backend():
    # Regression: with no route, every panelist uses the shared backend (unchanged path).
    from llmjury.engine import Engine
    from llmjury.verifiers import FunctionalCodeVerifier
    eng = Engine(_FakeBackend({"best": [_GOOD]}), panel=["best"], best="best", k=1)
    r = eng.solve("add", FunctionalCodeVerifier(_TESTS, "add"))
    assert r.verified and r.model == "best"


class _NeverVerifies:
    """Sandbox-free verifier so timing tests measure decode, not docker startup."""
    def verify(self, text):
        return False


class _VerifiesGood:
    def verify(self, text):
        return bool(text) and "a + b" in text


def test_engine_stage_decodes_concurrently():
    # Every sample takes ~0.25s to "decode". Serially, 3 models x k=2 = 1.5s of
    # decode; with per-sample futures the stages overlap internally:
    # ~0.25s (stage 1) + ~0.25s (stage 2, both panelists at once) + overhead.
    import time
    from llmjury.backends import Backend
    from llmjury.engine import Engine

    class _SlowBad(Backend):
        name = "slow"

        def _one(self, model, prompt, temperature, max_tokens):
            time.sleep(0.25)
            return _BAD

    eng = Engine(_SlowBad(), panel=["best", "p1", "p2"], best="best", k=2)
    t = time.time()
    r = eng.solve("add", _NeverVerifies())
    wall = time.time() - t
    assert not r.verified and r.attempts == 6
    assert wall < 1.2, f"council decode looks serialized: {wall:.2f}s for 6x0.25s samples"


def test_engine_early_exit_does_not_wait_for_slow_samples():
    # First sample verifies instantly; its k-1 siblings hang. solve() must return
    # on the verified sample without joining the stragglers.
    import time
    import threading
    from llmjury.backends import Backend
    from llmjury.engine import Engine

    class _FirstFastRestSlow(Backend):
        name = "mixed"

        def __init__(self, **kw):
            super().__init__(**kw)
            self._lock = threading.Lock()
            self._calls = 0

        def _one(self, model, prompt, temperature, max_tokens):
            with self._lock:
                self._calls += 1
                first = self._calls == 1
            if not first:
                time.sleep(1.5)
            return _GOOD

    eng = Engine(_FirstFastRestSlow(), panel=["best"], best="best", k=3)
    t = time.time()
    r = eng.solve("add", _VerifiesGood())
    wall = time.time() - t
    assert r.verified and r.stage == "single"
    assert wall < 1.0, f"early exit waited for abandoned samples: {wall:.2f}s"


def test_codex_delegator_uses_workspace_write_and_repo_rules():
    from llmjury.delegation import CodexDelegator

    calls = []

    def runner(cmd, **kw):
        calls.append((cmd, kw))
        output = Path(cmd[cmd.index("--output-last-message") + 1])
        output.write_text(json.dumps({
            "status": "completed",
            "summary": "Implemented the bounded change.",
            "changed_files": ["llmjury/example.py"],
            "tests": ["python tests/test_llmjury.py (pass)"],
            "blockers": [],
        }), encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    with tempfile.TemporaryDirectory() as workspace:
        result = CodexDelegator(runner=runner).delegate(
            "Implement the parser and run its tests.", workspace, model="gpt-test")

    cmd, kwargs = calls[0]
    assert result.status == "completed" and result.returncode == 0
    assert result.changed_files == ["llmjury/example.py"]
    assert cmd[:3] == ["codex", "exec", "--ephemeral"]
    assert cmd[cmd.index("--sandbox") + 1] == "workspace-write"
    assert "--ignore-user-config" not in cmd and "--ignore-rules" not in cmd
    assert "--output-schema" in cmd and "--output-last-message" in cmd
    assert "shell_environment_policy.inherit=core" in cmd
    assert "gpt-test" in cmd
    assert kwargs["timeout"] == 1800
    prompt = cmd[-1]
    assert "Implement the parser" in prompt
    assert "AGENTS.md" in prompt and "--backend ollama" in prompt


def test_codex_delegator_reports_invalid_handoff_as_blocked():
    from llmjury.delegation import CodexDelegator

    def runner(cmd, **kw):
        output = Path(cmd[cmd.index("--output-last-message") + 1])
        output.write_text(json.dumps({
            "status": "surprise", "summary": "invalid", "changed_files": [],
            "tests": [], "blockers": [],
        }), encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 2, "", "authentication failed")

    with tempfile.TemporaryDirectory() as workspace:
        result = CodexDelegator(runner=runner).delegate("Do the thing", workspace)
    assert result.status == "blocked" and result.returncode == 2
    assert "authentication failed" in result.blockers[0]


def test_codex_delegator_nonzero_exit_overrides_completed_handoff():
    from llmjury.delegation import CodexDelegator

    def runner(cmd, **kw):
        output = Path(cmd[cmd.index("--output-last-message") + 1])
        output.write_text(json.dumps({
            "status": "completed", "summary": "claimed success", "changed_files": [],
            "tests": [], "blockers": [],
        }), encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 3, "", "")

    with tempfile.TemporaryDirectory() as workspace:
        result = CodexDelegator(runner=runner).delegate("Do the thing", workspace)
    assert result.status == "blocked" and result.returncode == 3
    assert result.blockers == ["Codex exited with status 3."]


def test_validate_handoff_requires_exact_typed_schema():
    from llmjury.delegation import validate_handoff

    valid = {
        "status": "completed", "summary": "done", "changed_files": ["a.py"],
        "tests": ["pytest (pass)"], "blockers": [],
    }
    assert validate_handoff(valid)
    assert not validate_handoff({**valid, "status": "done"})
    assert not validate_handoff({**valid, "changed_files": [1]})
    assert not validate_handoff({**valid, "extra": True})
    assert not validate_handoff(None)


def test_baked_system_warnings_flag_contaminated_tags():
    from llmjury.backends import baked_system_warnings

    sizes = {"phi4": 0, "gemma3:12b-fable": 186191, "tiny-persona": 1500}
    msgs = baked_system_warnings(list(sizes), sizes.get)
    # only the tag over the threshold is flagged, and the message is actionable
    assert len(msgs) == 1
    assert "gemma3:12b-fable" in msgs[0]
    assert "182 KB" in msgs[0]
    assert "pristine" in msgs[0]
    # a probe failure (None) must not produce a false warning
    assert baked_system_warnings(["phi4"], lambda m: None) == []
    # duplicates are reported once
    assert len(baked_system_warnings(
        ["fat", "fat"], lambda m: 50_000)) == 1


def test_install_claude_skill_is_idempotent_and_refuses_overwrite():
    from llmjury.claude_integration import SKILL, install_claude_skill

    with tempfile.TemporaryDirectory() as project:
        path, changed = install_claude_skill("project", project)
        assert changed and path.read_text(encoding="utf-8") == SKILL
        same_path, changed = install_claude_skill("project", project)
        assert same_path == path and not changed
        path.write_text("custom skill", encoding="utf-8")
        try:
            install_claude_skill("project", project)
            assert False, "a custom skill must not be overwritten without --force"
        except FileExistsError:
            pass
        _, changed = install_claude_skill("project", project, force=True)
        assert changed and path.read_text(encoding="utf-8") == SKILL


def test_install_claude_fusion_agent_is_router_independent():
    from llmjury.claude_integration import AGENT, install_claude_agent

    with tempfile.TemporaryDirectory() as project:
        path, changed = install_claude_agent("project", project)
        assert changed and path.read_text(encoding="utf-8") == AGENT
        assert path.name == "llm-jury-fusion.md"
        assert path.parent.name == "agents"
        frontmatter = AGENT.split("---")[1]
        assert "name: llm-jury-fusion" in frontmatter
        assert "model:" not in frontmatter, \
            "a model pin would break sessions without a local router (Claude desktop app)"
        assert "llmjury solve" in AGENT
        assert "--backend ollama" in AGENT
        same_path, changed = install_claude_agent("project", project)
        assert same_path == path and not changed
        path.write_text("custom agent", encoding="utf-8")
        try:
            install_claude_agent("project", project)
            assert False, "a custom agent must not be overwritten without --force"
        except FileExistsError:
            pass
        _, changed = install_claude_agent("project", project, force=True)
        assert changed and path.read_text(encoding="utf-8") == AGENT


def test_claude_planner_is_read_only_and_parses_structured_envelope():
    from llmjury.planning import ClaudePlanner

    calls = []
    plan = {
        "status": "planned", "summary": "Two bounded steps.",
        "steps": [{
            "id": "step-1", "objective": "Implement parser",
            "acceptance": "focused tests pass", "files": ["parser.py"],
        }],
        "risks": ["Preserve compatibility"], "questions": [],
    }

    def runner(cmd, **kw):
        calls.append((cmd, kw))
        return subprocess.CompletedProcess(
            cmd, 0, json.dumps({"type": "result", "structured_output": plan}), "")

    with tempfile.TemporaryDirectory() as workspace:
        result = ClaudePlanner(runner=runner).plan(
            "Plan the parser change", workspace, model="claude-test")
        expected_workspace = str(Path(workspace).resolve())

    cmd, kwargs = calls[0]
    assert result.status == "planned" and result.steps[0]["id"] == "step-1"
    assert kwargs["cwd"] == expected_workspace and kwargs["timeout"] == 900
    assert cmd[cmd.index("--permission-mode") + 1] == "plan"
    assert cmd[cmd.index("--tools") + 1] == "Read,Glob,Grep"
    assert "--json-schema" in cmd and "--no-session-persistence" in cmd
    assert "claude-test" in cmd and "dynamically replan" in cmd[-1]


def test_validate_plan_rejects_malformed_steps():
    from llmjury.planning import validate_plan

    valid = {
        "status": "planned", "summary": "ok",
        "steps": [{"id": "1", "objective": "do", "acceptance": "pass", "files": []}],
        "risks": [], "questions": [],
    }
    assert validate_plan(valid)
    assert not validate_plan({**valid, "status": "done"})
    assert not validate_plan({**valid, "steps": [{"id": "1"}]})
    assert not validate_plan({**valid, "risks": [1]})


def test_install_codex_skill_is_idempotent_and_refuses_overwrite():
    from llmjury.codex_integration import SKILL, install_codex_skill

    old_home = os.environ.get("CODEX_HOME")
    with tempfile.TemporaryDirectory() as codex_home:
        os.environ["CODEX_HOME"] = codex_home
        try:
            path, changed = install_codex_skill()
            assert changed and path.read_text(encoding="utf-8") == SKILL
            same_path, changed = install_codex_skill()
            assert same_path == path and not changed
            path.write_text(SKILL.replace("version: 2", "version: 1"), encoding="utf-8")
            _, changed = install_codex_skill()
            assert changed and path.read_text(encoding="utf-8") == SKILL
            path.write_text("custom skill", encoding="utf-8")
            try:
                install_codex_skill()
                assert False, "a custom skill must not be overwritten without --force"
            except FileExistsError:
                pass
            _, changed = install_codex_skill(force=True)
            assert changed and path.read_text(encoding="utf-8") == SKILL
        finally:
            if old_home is None:
                os.environ.pop("CODEX_HOME", None)
            else:
                os.environ["CODEX_HOME"] = old_home


def test_codex_skill_drives_verified_native_app_workflow():
    from llmjury.codex_integration import SKILL

    assert "Codex app" in SKILL
    assert "llmjury solve --task" in SKILL
    assert "--backend ollama" in SKILL
    assert '"verified": true' in SKILL
    assert "Do not integrate" in SKILL
    assert "llmjury plan" in SKILL


def test_grok_backend_runs_isolated_toolless_generation():
    from llmjury.backends import GrokBackend

    calls = []

    def runner(cmd, **kw):
        calls.append((cmd, kw))
        return subprocess.CompletedProcess(cmd, 0, stdout=_GOOD + "\n", stderr="")

    backend = GrokBackend(runner=runner)
    assert backend.complete("grok-test", "write add", n=1) == [_GOOD]
    cmd, kw = calls[0]
    assert cmd[0] == "grok"
    # A generator, not an agent: no repo tools, no child sessions, no memory,
    # no planning, and a single turn — the verifier is what judges the output.
    assert cmd[cmd.index("--sandbox") + 1] == "read-only"
    assert cmd[cmd.index("--tools") + 1] == ""
    assert "--no-subagents" in cmd and "--no-memory" in cmd and "--no-plan" in cmd
    assert cmd[cmd.index("--max-turns") + 1] == "1"
    assert cmd[cmd.index("--permission-mode") + 1] == "dontAsk"
    assert cmd[cmd.index("--model") + 1] == "grok-test"
    assert cmd[cmd.index("--reasoning-effort") + 1] == "low"
    # The prompt is passed as the -p value and never leaks into an argv slot
    # that Grok would read as a flag.
    assert cmd[-2:] == ["-p", "write add"]
    assert kw["timeout"] == 600 and not kw["check"]


def test_grok_backend_isolates_cwd_from_the_repository():
    from llmjury.backends import GrokBackend

    seen = {}

    def runner(cmd, **kw):
        workdir = cmd[cmd.index("--cwd") + 1]
        seen["workdir"] = workdir
        seen["empty"] = os.listdir(workdir) == []
        return subprocess.CompletedProcess(cmd, 0, stdout=_GOOD, stderr="")

    GrokBackend(runner=runner).complete("grok-test", "write add", n=1)
    assert seen["empty"], "Grok must generate in an empty dir, not the repo"
    assert not os.path.exists(seen["workdir"]), "the temp workdir must be cleaned up"


def test_grok_backend_reports_failures_without_raising():
    from llmjury.backends import GrokBackend

    def failing(cmd, **kw):
        return subprocess.CompletedProcess(cmd, 3, stdout="", stderr="not authenticated\n")

    assert GrokBackend(runner=failing).complete("grok-test", "write add", n=1) == [""]

    def timing_out(cmd, **kw):
        raise subprocess.TimeoutExpired(cmd, 600)

    assert GrokBackend(runner=timing_out).complete("grok-test", "write add", n=1) == [""]


def test_cli_backend_builds_grok_provider():
    from unittest.mock import patch
    from llmjury.cli import _backend

    with patch("llmjury.backends.shutil.which", return_value="/opt/homebrew/bin/grok"):
        assert _backend("grok").name == "grok"


def test_grok_is_a_single_provider_panel_not_a_council():
    from llmjury.panels import GROK_BEST, GROK_PANEL, default_panel

    assert GROK_PANEL == [GROK_BEST]
    assert default_panel("grok") == (GROK_BEST, GROK_PANEL)


def test_named_frontier_ladders_still_reject_the_grok_backend():
    from llmjury.cli import _frontier_models

    # `auto`/`opus`/`fable` are OpenRouter slugs; Grok resolves models through its
    # own registry, so the ladder must not silently hand it a slug it cannot serve.
    for alias in ("auto", "open", "opus", "fable"):
        try:
            _frontier_models(alias, "grok")
            assert False, f"{alias} must reject the grok frontier backend"
        except ValueError:
            pass
    assert _frontier_models("grok-4.5", "grok") == "grok-4.5"


def test_install_grok_skill_is_idempotent_and_refuses_overwrite():
    from llmjury.grok_integration import SKILL, install_grok_skill

    old_home = os.environ.get("GROK_HOME")
    with tempfile.TemporaryDirectory() as grok_home:
        os.environ["GROK_HOME"] = grok_home
        try:
            path, changed = install_grok_skill()
            assert changed and path.read_text(encoding="utf-8") == SKILL
            same_path, changed = install_grok_skill()
            assert same_path == path and not changed
            path.write_text("custom skill", encoding="utf-8")
            try:
                install_grok_skill()
                assert False, "a custom skill must not be overwritten without --force"
            except FileExistsError:
                pass
            _, changed = install_grok_skill(force=True)
            assert changed and path.read_text(encoding="utf-8") == SKILL
        finally:
            if old_home is None:
                os.environ.pop("GROK_HOME", None)
            else:
                os.environ["GROK_HOME"] = old_home


def test_grok_skill_keeps_the_council_local_and_escalation_on_subscription_auth():
    from llmjury.grok_integration import SKILL

    assert "Grok CLI" in SKILL
    assert "llmjury solve --task" in SKILL
    assert "--backend ollama" in SKILL
    assert "--frontier-backend grok" in SKILL
    assert '"verified": true' in SKILL
    assert "Do not integrate" in SKILL


def test_grok_agent_frontmatter_description_is_quoted_for_strict_yaml():
    """Grok parses agent frontmatter as strict YAML; Claude Code does not.

    An unquoted `description:` containing a colon-space is silently dropped by
    Grok — the agent never appears in `grok inspect` and cannot be spawned. The
    shipped agent must stay loadable in both hosts.
    """
    from llmjury.claude_integration import AGENT

    line = [l for l in AGENT.split("\n") if l.startswith("description:")][0]
    value = line[len("description:"):].strip()
    if ": " in value:
        assert value.startswith('"') and value.endswith('"'), (
            "a description containing ': ' must be a quoted YAML scalar or Grok "
            "will silently drop the agent")


# --- memory preflight -------------------------------------------------------------
#
# Resident size measured with `ollama ps` on a 36 GB M-series Mac at num_ctx 8192 x
# OLLAMA_NUM_PARALLEL 4 (32768 KV cells). (tag, disk GB, resident GB).
MEASURED = [
    ("granite4.1:3b", 2.1, 4.6),
    ("phi4-mini:3.8b", 2.5, 5.7),
    ("qwen3.5:4b", 3.4, 5.9),
    ("llama3.1:8b", 4.9, 9.0),
    ("qwen3:8b", 5.2, 7.9),
    ("gemma3:12b", 8.1, 11.0),
    ("phi4", 9.1, 14.0),
]
HOST_36GB = 38654705664          # hw.memsize on the machine that panicked


def _fake_ollama(monkeypatched, sizes_gb, loaded=None, simulator=(False, 0)):
    """Point memguard at a synthetic host. Returns a restore callable.

    Also stubs the simulator probe: without that, every test here would inherit
    the REAL machine's simulator state, and a booted iPhone on the developer's
    desk would fail the whole suite.
    """
    from llmjury import memguard
    saved = (memguard.disk_sizes, memguard.loaded_bytes, memguard.total_ram_bytes,
             memguard.simulator_stack)
    loaded = loaded or {}
    memguard.disk_sizes = lambda host: {t: int(g * 1e9) for t, g in sizes_gb.items()}
    memguard.loaded_bytes = lambda host: (sum(loaded.values()), dict(loaded))
    memguard.total_ram_bytes = lambda: monkeypatched
    memguard.simulator_stack = lambda: simulator

    def restore():
        (memguard.disk_sizes, memguard.loaded_bytes, memguard.total_ram_bytes,
         memguard.simulator_stack) = saved
    return restore


def test_memguard_never_underestimates_measured_resident():
    """Over-estimating wastes a run; under-estimating panics the host."""
    from llmjury import memguard
    cells = 8192 * 4
    for tag, disk_gb, resident_gb in MEASURED:
        estimate = memguard.estimate_resident(int(disk_gb * 1e9), cells)
        assert estimate >= resident_gb * 1e9, (
            f"{tag}: estimated {estimate / 1e9:.1f} GB < measured {resident_gb} GB")


def test_memguard_refuses_the_panel_that_panicked_the_host():
    """Regression for 2026-07-31: phi4 + gemma3:12b + llama3.1:8b on a 36 GB Mac."""
    from llmjury import memguard
    restore = _fake_ollama(HOST_36GB, {t: g for t, g, _ in MEASURED})
    try:
        report = memguard.check(["phi4", "gemma3:12b", "llama3.1:8b"],
                                num_ctx=8192, parallel=4)
        assert not report.ok, "the panel that took the machine down must be refused"
        assert report.projected > report.budget
        assert "smaller panel" in report.hint()
    finally:
        restore()


def test_memguard_allows_the_shipped_panel_at_its_documented_parallelism():
    """The default panel is specified at OLLAMA_NUM_PARALLEL=2, not Ollama's stock 4.

    KV is charged num_ctx x slots, so parallelism is part of a panel's spec. Pinning
    the supported configuration here keeps the default honest about what it needs.
    """
    from llmjury import memguard, panels
    restore = _fake_ollama(HOST_36GB, {t: g for t, g, _ in MEASURED})
    try:
        report = memguard.check(panels.LOCAL_PANEL, num_ctx=8192, parallel=2)
        assert report.ok, f"default panel must fit a 36 GB host at 2 slots: {report.message()}"
    finally:
        restore()


def test_reproduce_pins_num_ctx_on_the_ollama_backend():
    """Leaving num_ctx unset means the SERVER's default, not a small one.

    Ollama sizes KV as num_ctx x OLLAMA_NUM_PARALLEL at load. This path once passed
    nothing, so a host tuned to 32k for coding-agent use loaded llama3.1:8b at 13.0 GB
    and phi4-mini:3.8b at 8.9 GB against memguard estimates of 8.0 and 4.8 — 1.6-1.9x —
    and the third panelist would have taken it past physical RAM. Benchmark prompts are
    tiny, so the large context bought nothing.
    """
    from llmjury.benchmarks import reproduce
    be = reproduce._backend("ollama")
    assert be.num_ctx == reproduce.DEFAULT_NUM_CTX, (
        "reproduce must pin num_ctx explicitly; None inherits the server default")
    assert reproduce._backend("ollama", num_ctx=2048).num_ctx == 2048


def test_local_panel_mirrors_cloud_panel_lineages():
    """The local council is meant to be the benchmarked council, run locally.

    CLOUD_PANEL is what the published numbers were measured with. LOCAL_PANEL mirrors
    its *lineages* so those numbers describe something reproducible off-cloud. Exact
    tags cannot match (phi-4 is 12.7 GiB locally and no 3-model panel holding it fits a
    36 GiB host), so a panelist may be swapped for a smaller model from the SAME lab —
    never for a different lab, which would silently drop a lineage from the council and
    make the benchmark a weaker description of local behaviour.
    """
    from llmjury import panels
    lineage = {
        "microsoft/phi-4": "microsoft", "phi4": "microsoft", "phi4-mini:3.8b": "microsoft",
        "google/gemma-3-12b-it": "google", "gemma3:12b": "google",
        "meta-llama/llama-3.1-8b-instruct": "meta", "llama3.1:8b": "meta",
        "granite4.1:3b": "ibm", "qwen3:8b": "alibaba", "qwen3.5:4b": "alibaba",
    }
    missing = [m for m in panels.CLOUD_PANEL + panels.LOCAL_PANEL if m not in lineage]
    assert not missing, f"unmapped model, extend the lineage table: {missing}"
    cloud = sorted(lineage[m] for m in panels.CLOUD_PANEL)
    local = sorted(lineage[m] for m in panels.LOCAL_PANEL)
    assert local == cloud, (
        f"local council must mirror the benchmarked lineages {cloud}, got {local}. "
        "Substitute within a lab (phi-4 -> phi4-mini), do not swap labs.")


def test_memguard_refuses_the_shipped_panel_at_stock_parallelism():
    """The other half of the contract: an untuned host is REFUSED, never panicked.

    Requiring OLLAMA_NUM_PARALLEL=2 is only safe because the preflight runs before any
    model loads, so a stock 4-slot host gets an actionable error naming a smaller
    panel. If this ever starts passing, the default silently became a crash risk for
    anyone who never tuned Ollama.
    """
    from llmjury import memguard, panels
    restore = _fake_ollama(HOST_36GB, {t: g for t, g, _ in MEASURED})
    try:
        report = memguard.check(panels.LOCAL_PANEL, num_ctx=8192, parallel=4)
        assert not report.ok, "default panel at 4 slots must refuse, not silently fit"
        assert "smaller panel" in report.hint()
    finally:
        restore()


def test_memguard_counts_kv_per_parallel_slot():
    """KV is charged num_ctx x slots, which is what made a '3 GB' model cost 7.5 GB."""
    from llmjury import memguard
    restore = _fake_ollama(HOST_36GB, {t: g for t, g, _ in MEASURED})
    try:
        one = memguard.check(["llama3.1:8b"], num_ctx=8192, parallel=1)
        four = memguard.check(["llama3.1:8b"], num_ctx=8192, parallel=4)
        assert four.projected > one.projected
    finally:
        restore()


def test_memguard_charges_models_already_resident():
    """A model another session is holding is still occupying the same RAM."""
    from llmjury import memguard
    sizes = {t: g for t, g, _ in MEASURED}
    restore = _fake_ollama(HOST_36GB, sizes, loaded={"qwen3.5:4b-64k": int(7.5e9)})
    try:
        report = memguard.check(["llama3.1:8b"], num_ctx=8192, parallel=4)
        assert report.resident == int(7.5e9)
        assert report.projected > memguard.estimate_resident(int(4.9e9), 8192 * 4)
        assert "unload" in report.hint()
    finally:
        restore()


def test_memguard_counts_a_repeated_tag_once():
    """cli builds the probe as [best] + panel, and best is usually in the panel."""
    from llmjury import memguard
    restore = _fake_ollama(HOST_36GB, {t: g for t, g, _ in MEASURED})
    try:
        once = memguard.check(["phi4", "gemma3:12b"], num_ctx=8192, parallel=4)
        twice = memguard.check(["phi4", "phi4", "gemma3:12b"], num_ctx=8192, parallel=4)
        assert once.projected == twice.projected, "Ollama loads a repeated tag once"
        assert len(twice.per_model) == 2
    finally:
        restore()


def test_memguard_skips_rather_than_blocks_when_it_cannot_tell():
    """The guard exists to stop a known-bad run, not to invent new failures."""
    from llmjury import memguard
    saved = (memguard.disk_sizes, memguard.total_ram_bytes, memguard.simulator_stack)
    try:
        memguard.simulator_stack = lambda: (False, 0)
        memguard.total_ram_bytes = lambda: 0
        assert memguard.check(["phi4"]).ok
        memguard.total_ram_bytes = lambda: HOST_36GB
        memguard.disk_sizes = lambda host: None          # ollama unreachable
        report = memguard.check(["phi4"])
        assert report.ok and report.skipped
    finally:
        memguard.disk_sizes, memguard.total_ram_bytes, memguard.simulator_stack = saved


def test_memguard_refuses_a_local_panel_while_a_simulator_is_booted():
    """Regression for 2026-08-01: council + CoreSimulator stack is the co-residency
    that takes a 36 GB host down. The stack measured 17.6 GB across 282 processes,
    none of which appear in Ollama's numbers, so the refusal must come before any
    RAM arithmetic -- even a panel that would otherwise fit is refused."""
    from llmjury import memguard
    restore = _fake_ollama(HOST_36GB, {"phi4-mini:3.8b": 2.5},
                           simulator=(True, int(17.6 * memguard.GB)))
    try:
        report = memguard.check(["phi4-mini:3.8b"], num_ctx=8192, parallel=2)
        assert not report.ok
        assert report.simulator
        assert "Simulator" in report.message()
        assert "17.6 GB" in report.message()
        assert "simctl shutdown" in report.hint()
        assert memguard.SIMULATOR_OVERRIDE_ENV in report.hint()
    finally:
        restore()


def test_memguard_simulator_override_env_allows_coresidency():
    """The escape hatch is explicit: with the env var set, the check falls through
    to the normal RAM arithmetic instead of the simulator refusal. (Plain
    os.environ handling, not pytest's monkeypatch -- CI runs these functions
    through the zero-dependency runner, where fixtures do not exist.)"""
    from llmjury import memguard
    restore = _fake_ollama(HOST_36GB, {"phi4-mini:3.8b": 2.5},
                           simulator=(True, int(17.6 * 1e9)))
    env = memguard.SIMULATOR_OVERRIDE_ENV
    saved = os.environ.get(env)
    try:
        os.environ[env] = "1"
        report = memguard.check(["phi4-mini:3.8b"], num_ctx=8192, parallel=2)
        assert report.ok and not report.simulator
        os.environ[env] = "0"
        assert not memguard.check(["phi4-mini:3.8b"], num_ctx=8192, parallel=2).ok
    finally:
        if saved is None:
            os.environ.pop(env, None)
        else:
            os.environ[env] = saved
        restore()


def test_memguard_simulator_probe_fails_open():
    """A host where pgrep is missing or errors must not start refusing panels."""
    import subprocess as sp
    from llmjury import memguard
    saved = sp.run
    try:
        def boom(*a, **k):
            raise OSError("no pgrep on this host")
        sp.run = boom
        assert memguard.simulator_stack() == (False, 0)
    finally:
        sp.run = saved


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
