"""Model backends: Codex, local Ollama, and cloud OpenRouter, one interface.

A backend turns (model, prompt) into text samples. Diversity across samples comes
from temperature > 0; we make one request per sample and run them concurrently.
Stdlib only — no `requests`, no SDKs.
"""
import os
import sys
import json
import time
import urllib.request
import urllib.error
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor

from .cache import Cache

_RETRYABLE = (408, 429, 500, 502, 503, 520, 524)
_HTTP_MSG = {401: "invalid API key", 402: "insufficient credits", 403: "forbidden"}


class Backend:
    name = "backend"

    def __init__(self, cache_path=None, max_workers=8):
        self.cache = Cache(cache_path) if cache_path else None
        self.max_workers = max_workers

    def _one(self, model, prompt, temperature, max_tokens):
        raise NotImplementedError

    def _sample(self, model, prompt, temperature, max_tokens, i):
        """One cached sample — the unit of work `complete` and `submit` share."""
        ck = self.cache.key(self.name, model, temperature, max_tokens, i, prompt) if self.cache else None
        if ck is not None:
            hit = self.cache.get(ck)
            if hit is not None:
                return hit
        txt = self._one(model, prompt, temperature, max_tokens)
        if ck is not None:
            self.cache.put(ck, txt)
        return txt

    def complete(self, model, prompt, n=1, temperature=0.7, max_tokens=4000):
        """Return a list of n text samples for (model, prompt), generated concurrently."""
        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            return list(ex.map(
                lambda i: self._sample(model, prompt, temperature, max_tokens, i), range(n)))

    def submit(self, ex, model, prompt, n=1, temperature=0.7, max_tokens=4000):
        """Queue n samples on a caller-owned executor; returns one Future per sample.

        This is what lets the engine interleave decoding across models and verify
        samples in completion order instead of blocking on a whole batch — the
        backend (e.g. Ollama with OLLAMA_NUM_PARALLEL > 1) sees every request at
        once and can batch them into a single decode pass.
        """
        return [ex.submit(self._sample, model, prompt, temperature, max_tokens, i)
                for i in range(n)]


class OpenRouterBackend(Backend):
    name = "openrouter"
    API = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(self, api_key=None, **kw):
        super().__init__(**kw)
        self.key = api_key or os.environ.get("OPENROUTER_API_KEY") or self._from_env_file()
        if not self.key:
            raise RuntimeError(
                "No OpenRouter API key. Set OPENROUTER_API_KEY, pass api_key=, "
                "or put it in ~/.llmjury/.env — or use the local Ollama backend.")

    @staticmethod
    def _from_env_file():
        p = os.path.expanduser("~/.llmjury/.env")
        if os.path.exists(p):
            with open(p, encoding="utf-8") as fh:
                for line in fh:
                    s = line.strip()
                    if s.startswith("export "):
                        s = s[7:]
                    if s.startswith("OPENROUTER_API_KEY="):
                        return s.split("=", 1)[1].strip().strip('"').strip("'")
        return None

    def _one(self, model, prompt, temperature, max_tokens):
        body = json.dumps({
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }).encode()
        h = {
            "Authorization": f"Bearer {self.key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/ajsai47/llm-jury",
            "X-Title": "llmjury",
        }
        for attempt in range(4):
            try:
                req = urllib.request.Request(self.API, data=body, headers=h, method="POST")
                with urllib.request.urlopen(req, timeout=300) as r:
                    d = json.load(r)
                # OpenRouter can return HTTP 200 with an error body (rate-limit, moderation, ...).
                if isinstance(d, dict) and d.get("error"):
                    sys.stderr.write(f"[llmjury] openrouter {model}: {d['error']}\n")
                    return ""
                choices = d.get("choices") or []
                if not choices:
                    sys.stderr.write(f"[llmjury] openrouter {model}: empty response\n")
                    return ""
                m = choices[0].get("message", {}) or {}
                return m.get("content") or m.get("reasoning") or ""
            except urllib.error.HTTPError as e:
                if e.code in _RETRYABLE:
                    time.sleep(3 * (attempt + 1))
                    continue
                sys.stderr.write(f"[llmjury] openrouter {model}: {_HTTP_MSG.get(e.code, f'HTTP {e.code}')}\n")
                return ""
            except urllib.error.URLError as e:
                sys.stderr.write(f"[llmjury] cannot reach OpenRouter ({e.reason}) — check your connection\n")
                return ""
            except Exception:
                time.sleep(3 * (attempt + 1))
        return ""


class CodexBackend(Backend):
    """Generate candidates through the authenticated Codex CLI.

    Codex is run as an ephemeral, read-only generator with shell tools disabled in an
    empty working directory. This deliberately avoids repo tools and repo instructions:
    LLM-Jury supplies the task and independently verifies the returned code itself.
    """
    name = "codex"

    def __init__(self, executable="codex", timeout=600, reasoning_effort="low",
                 runner=None, **kw):
        super().__init__(**kw)
        self.executable = executable
        self.timeout = timeout
        self.reasoning_effort = reasoning_effort
        self.runner = runner or subprocess.run
        if runner is None and not shutil.which(executable):
            raise RuntimeError(
                "Codex CLI not found. Install and authenticate Codex, or use "
                "--backend ollama/openrouter.")

    def _one(self, model, prompt, temperature, max_tokens):
        # Codex owns its sampling and output budget. Keeping the Backend signature
        # lets it participate in the same verified escalation ladder.
        with tempfile.TemporaryDirectory(prefix="llmjury-codex-") as workdir:
            cmd = [
                self.executable, "exec", "--ephemeral", "--sandbox", "read-only",
                "--skip-git-repo-check", "--ignore-user-config", "--ignore-rules",
                "--disable", "shell_tool", "--disable", "unified_exec",
                "--color", "never", "--cd", workdir,
            ]
            if model:
                cmd.extend(["--model", model])
            if self.reasoning_effort:
                cmd.extend(["-c", f'model_reasoning_effort="{self.reasoning_effort}"'])
            cmd.append(prompt)
            try:
                completed = self.runner(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                    timeout=self.timeout, check=False,
                )
            except subprocess.TimeoutExpired:
                sys.stderr.write(
                    f"[llmjury] codex {model or '(configured default)'} timed out "
                    f"after {self.timeout}s\n")
                return ""
            except OSError as e:
                sys.stderr.write(f"[llmjury] cannot run Codex CLI: {e}\n")
                return ""
            if completed.returncode != 0:
                detail = (completed.stderr or "").strip().splitlines()
                suffix = f": {detail[-1]}" if detail else ""
                sys.stderr.write(
                    f"[llmjury] codex {model or '(configured default)'} exited "
                    f"{completed.returncode}{suffix}\n")
                return ""
            return (completed.stdout or "").strip()


class DemoBackend(Backend):
    """Offline, canned backend so `llmjury demo` runs with no API key and no Ollama —
    while still exercising the REAL generate->verify->select->escalate pipeline and the
    REAL code-executing verifier. `demo-weak` returns a wrong answer and `demo-council` a
    right one, so the demo actually shows the council recovering what one model missed.
    """
    name = "demo"
    _RESP = {
        "demo-weak": "```python\ndef add(a, b):\n    return a - b  # subtly wrong\n```",
        "demo-council": "```python\ndef add(a, b):\n    return a + b\n```",
    }

    def _one(self, model, prompt, temperature, max_tokens):
        return self._RESP.get(model, "")


class OllamaBackend(Backend):
    name = "ollama"

    def __init__(self, host="http://localhost:11434", num_ctx=None, **kw):
        super().__init__(**kw)
        self.host = host.rstrip("/")
        # Per-request context cap. Ollama sizes a model's KV cache as
        # num_ctx x OLLAMA_NUM_PARALLEL at load, so a server tuned with a big
        # default context (e.g. 32k for coding-agent use) burns GPU memory on
        # jury runs whose prompts are tiny. A lean num_ctx keeps parallel
        # decode slots cheap enough that the whole council fits in memory.
        self.num_ctx = num_ctx

    def _one(self, model, prompt, temperature, max_tokens):
        options = {"temperature": temperature, "num_predict": max_tokens}
        if self.num_ctx:
            options["num_ctx"] = self.num_ctx
        body = json.dumps({
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": options,
        }).encode()
        for attempt in range(3):
            try:
                req = urllib.request.Request(
                    self.host + "/api/chat", data=body,
                    headers={"Content-Type": "application/json"}, method="POST")
                with urllib.request.urlopen(req, timeout=600) as r:
                    d = json.load(r)
                if isinstance(d, dict) and d.get("error"):
                    sys.stderr.write(f"[llmjury] ollama {model}: {d['error']}\n")
                    return ""
                return d.get("message", {}).get("content", "") or ""
            except urllib.error.HTTPError as e:
                sys.stderr.write(
                    f"[llmjury] ollama {model} HTTP {e.code} (pulled it? try: ollama pull {model})\n")
                return ""
            except urllib.error.URLError as e:
                # Connection refused = Ollama isn't running. Don't burn retries silently.
                sys.stderr.write(
                    f"[llmjury] cannot reach Ollama at {self.host} — is `ollama serve` running? ({e.reason})\n")
                return ""
            except Exception:
                time.sleep(2 * (attempt + 1))
        return ""


class OpenAICompatBackend(Backend):
    """A generic OpenAI-compatible /chat/completions endpoint.

    Covers any server that speaks the OpenAI chat shape: an `mlx_lm.server`, vLLM,
    LM Studio, llama.cpp's server, etc. The motivating case is bringing Shawn's
    fine-tuned MLX brain (Qwen3.5-4B + LoRA, served by mlx_lm.server on
    127.0.0.1:8801) into the council as an opt-in panelist — without a GGUF
    conversion (the qwen3.5 GGUF path is broken). Pair with the engine's per-model
    `route` so only this one panelist hits the custom endpoint.
    """
    name = "openai-compat"

    def __init__(self, base_url, api_key=None, **kw):
        super().__init__(**kw)
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.key = api_key or os.environ.get("OPENAI_COMPAT_API_KEY")

    def _one(self, model, prompt, temperature, max_tokens):
        body = json.dumps({
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }).encode()
        h = {"Content-Type": "application/json"}
        if self.key:
            h["Authorization"] = f"Bearer {self.key}"
        for attempt in range(3):
            try:
                req = urllib.request.Request(self.url, data=body, headers=h, method="POST")
                with urllib.request.urlopen(req, timeout=600) as r:
                    d = json.load(r)
                if isinstance(d, dict) and d.get("error"):
                    sys.stderr.write(f"[llmjury] openai-compat {model}: {d['error']}\n")
                    return ""
                choices = d.get("choices") or []
                if not choices:
                    sys.stderr.write(f"[llmjury] openai-compat {model}: empty response\n")
                    return ""
                m = choices[0].get("message", {}) or {}
                return m.get("content") or ""
            except urllib.error.HTTPError as e:
                sys.stderr.write(
                    f"[llmjury] openai-compat {model} HTTP {e.code} at {self.url} "
                    "(is the server up and the model loaded?)\n")
                return ""
            except urllib.error.URLError as e:
                sys.stderr.write(
                    f"[llmjury] cannot reach {self.url} ({e.reason}) — is the endpoint serving?\n")
                return ""
            except Exception:
                time.sleep(2 * (attempt + 1))
        return ""
