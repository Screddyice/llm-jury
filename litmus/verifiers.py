"""Verifiers — the heart of Litmus.

A verifier checks a candidate WITHOUT being told the answer, so it can SELECT the
right one from many attempts. For code that's just: run it against tests and see if
it passes. The verifier is what makes a council beat a single model — it keeps the
correct attempt and discards the wrong ones.

SECURITY: these verifiers EXECUTE model-generated code. That is the whole point, and
it is inherently dangerous. v0.1 hardens execution with a scrubbed environment, an
isolated temp working directory, and POSIX resource limits (CPU / file size / no core
dumps). This is NOT a real sandbox. Do not run untrusted tasks on a machine with
secrets; use a container or VM for real isolation.
"""
import os
import re
import sys
import signal
import textwrap
import subprocess
import tempfile

_MAX_OUTPUT = 1 << 20  # cap captured stdout/stderr at 1 MiB (a print-bomb can't OOM the parent)

# ---------------------------------------------------------------- code extraction
_PROG = re.compile(r"(?:^|\n)\s*(?:def |class |import |from |if |for |while |return |print\(|input\()")


def _looks_like_program(s):
    return bool(s and _PROG.search(s))


def extract_code(text):
    """Pull the model's solution out of its reply.

    Robust to two failure modes confirmed in the wild: (1) chatty models that put a
    trailing usage-example block after the real solution — we take the longest block
    that *looks like a program*, not blindly the last; (2) truncated/unterminated
    ``` fences from max_tokens cutoffs — we slice from a lone opening fence and strip
    stray fence markers.
    """
    if not text:
        return None
    blocks = re.findall(r"```(?:[A-Za-z0-9_+-]*)\s*\n?(.*?)```", text, re.DOTALL)
    prog = [b for b in blocks if _looks_like_program(b)]
    if prog:
        return max(prog, key=len).strip()
    if blocks:
        return max(blocks, key=len).strip()
    # No closed fence — handle a lone opening fence (common on truncation).
    m = re.search(r"```(?:[A-Za-z0-9_+-]*)\s*\n(.*)$", text, re.DOTALL)
    if m and _looks_like_program(m.group(1)):
        return m.group(1).strip()
    if _looks_like_program(text):
        return "\n".join(l for l in text.splitlines() if not l.strip().startswith("```")).strip()
    return None


# ---------------------------------------------------------------- sandboxing
def _safe_env(workdir):
    """An allowlist environment — the child gets none of the parent's secrets
    (API keys, tokens). HOME points inside the sandbox so `~` can't reach real files.
    """
    keep = ("PATH", "LANG", "LC_ALL", "SYSTEMROOT", "TMPDIR")
    env = {k: os.environ[k] for k in keep if k in os.environ}
    env["HOME"] = workdir
    env["PYTHONIOENCODING"] = "utf-8"
    env.setdefault("PATH", os.defpath)
    return env


def _limits(cpu_seconds):
    """A preexec_fn applying POSIX resource limits. Returns None on non-POSIX.

    Deliberately omits RLIMIT_NPROC (per-uid — can wedge the user's whole session)
    and RLIMIT_AS (fragile on macOS, where Python reserves huge virtual space). The
    CPU limit reaps runaway loops; the wall-clock timeout backstops it.
    """
    if not hasattr(os, "fork"):
        return None
    try:
        import resource
    except Exception:
        return None

    def apply():
        for what, soft, hard in (
            (resource.RLIMIT_CPU, cpu_seconds, cpu_seconds + 1),
            (resource.RLIMIT_FSIZE, 10 * 1024 * 1024, 10 * 1024 * 1024),
            (resource.RLIMIT_CORE, 0, 0),
        ):
            try:
                resource.setrlimit(what, (soft, hard))
            except Exception:
                pass
    return apply


def _run(args, stdin, timeout, workdir):
    # start_new_session puts the child in its own process group so a timeout can reap
    # the WHOLE tree (forked grandchildren), not just the direct child.
    p = subprocess.Popen(
        args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, cwd=workdir, env=_safe_env(workdir),
        preexec_fn=_limits(timeout), start_new_session=hasattr(os, "setsid"))
    try:
        out, err = p.communicate(input=stdin, timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            if hasattr(os, "killpg"):
                os.killpg(os.getpgid(p.pid), signal.SIGKILL)
            else:
                p.kill()
        except (ProcessLookupError, OSError):
            pass
        p.wait()
        raise
    return subprocess.CompletedProcess(args, p.returncode, out[:_MAX_OUTPUT], err[:_MAX_OUTPUT])


def _same_output(got, want):
    """Competitive-judge semantics: ignore trailing whitespace per line and trailing
    blank lines."""
    def norm(s):
        lines = [l.rstrip() for l in (s or "").replace("\r\n", "\n").split("\n")]
        while lines and lines[-1] == "":
            lines.pop()
        return lines
    return norm(got) == norm(want)


# ---------------------------------------------------------------- verifiers
class Verifier:
    def verify(self, candidate_text) -> bool:
        raise NotImplementedError


class FunctionalCodeVerifier(Verifier):
    """HumanEval-style. The candidate defines `entry_point`; `test` is EITHER a full
    `def check(candidate): ...` OR just its body (assert statements). Passes iff
    `check(entry_point)` runs without raising.
    """
    def __init__(self, test, entry_point, header="", timeout=20):
        self.test = test
        self.entry_point = entry_point
        self.header = header
        self.timeout = timeout

    @classmethod
    def from_cases(cls, entry_point, cases, **kw):
        """Build a verifier from (args, expected) pairs — no check() string to learn:

            FunctionalCodeVerifier.from_cases("add", [((2, 3), 5), ((-1, 1), 0)])

        `args` may be a tuple of positional args, or a single value.
        """
        lines = []
        for args, expected in cases:
            if not isinstance(args, tuple):
                args = (args,)
            lines.append(f"assert candidate({', '.join(map(repr, args))}) == {expected!r}")
        return cls("\n".join(lines), entry_point, **kw)

    def verify(self, candidate_text):
        code = extract_code(candidate_text)
        if not code:
            return False
        # Accept a full `def check(candidate): ...` OR just its body (assert lines).
        test = self.test if re.search(r"(?m)^\s*def\s+check\s*\(", self.test or "") \
            else "def check(candidate):\n" + textwrap.indent(self.test or "pass", "    ")
        program = (
            f"{self.header}\n{code}\n\n{test}\n\n"
            f"check({self.entry_point})\nprint('LITMUS_OK')"
        )
        try:
            with tempfile.TemporaryDirectory(prefix="litmus-") as wd:
                p = _run([sys.executable, "-"], program, self.timeout, wd)
            return p.returncode == 0 and "LITMUS_OK" in p.stdout
        except Exception:
            return False


class StdioCodeVerifier(Verifier):
    """Competitive-programming style. The candidate is a full program; for each
    {"input", "output"} case we run it with `input` on stdin and compare stdout
    (per-line, ignoring trailing whitespace). Passes iff every case matches.
    """
    def __init__(self, cases, timeout=10):
        self.cases = cases
        self.timeout = timeout

    def verify(self, candidate_text):
        code = extract_code(candidate_text)
        if not code:
            return False
        try:
            with tempfile.TemporaryDirectory(prefix="litmus-") as wd:
                path = os.path.join(wd, "sol.py")
                with open(path, "w", encoding="utf-8") as f:
                    f.write(code)
                for c in self.cases:
                    try:
                        p = _run([sys.executable, path], c.get("input", ""), self.timeout, wd)
                    except Exception:
                        return False
                    if p.returncode != 0 or not _same_output(p.stdout, c.get("output", "")):
                        return False
            return True
        except Exception:
            return False
