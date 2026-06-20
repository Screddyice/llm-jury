"""Model backends: local (Ollama) and cloud (OpenRouter), one interface.

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

    def complete(self, model, prompt, n=1, temperature=0.7, max_tokens=4000):
        """Return a list of n text samples for (model, prompt), generated concurrently."""
        def call(i):
            ck = self.cache.key(self.name, model, temperature, max_tokens, i, prompt) if self.cache else None
            if ck is not None:
                hit = self.cache.get(ck)
                if hit is not None:
                    return hit
            txt = self._one(model, prompt, temperature, max_tokens)
            if ck is not None:
                self.cache.put(ck, txt)
            return txt

        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            return list(ex.map(call, range(n)))


class OpenRouterBackend(Backend):
    name = "openrouter"
    API = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(self, api_key=None, **kw):
        super().__init__(**kw)
        self.key = api_key or os.environ.get("OPENROUTER_API_KEY") or self._from_env_file()
        if not self.key:
            raise RuntimeError(
                "No OpenRouter API key. Set OPENROUTER_API_KEY, pass api_key=, "
                "or put it in ~/.litmus/.env — or use the local Ollama backend.")

    @staticmethod
    def _from_env_file():
        p = os.path.expanduser("~/.litmus/.env")
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
            "HTTP-Referer": "https://github.com/ajsai47/litmus",
            "X-Title": "litmus",
        }
        for attempt in range(4):
            try:
                req = urllib.request.Request(self.API, data=body, headers=h, method="POST")
                with urllib.request.urlopen(req, timeout=300) as r:
                    d = json.load(r)
                # OpenRouter can return HTTP 200 with an error body (rate-limit, moderation, ...).
                if isinstance(d, dict) and d.get("error"):
                    sys.stderr.write(f"[litmus] openrouter {model}: {d['error']}\n")
                    return ""
                choices = d.get("choices") or []
                if not choices:
                    sys.stderr.write(f"[litmus] openrouter {model}: empty response\n")
                    return ""
                m = choices[0].get("message", {}) or {}
                return m.get("content") or m.get("reasoning") or ""
            except urllib.error.HTTPError as e:
                if e.code in _RETRYABLE:
                    time.sleep(3 * (attempt + 1))
                    continue
                sys.stderr.write(f"[litmus] openrouter {model}: {_HTTP_MSG.get(e.code, f'HTTP {e.code}')}\n")
                return ""
            except urllib.error.URLError as e:
                sys.stderr.write(f"[litmus] cannot reach OpenRouter ({e.reason}) — check your connection\n")
                return ""
            except Exception:
                time.sleep(3 * (attempt + 1))
        return ""


class DemoBackend(Backend):
    """Offline, canned backend so `litmus demo` runs with no API key and no Ollama —
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

    def __init__(self, host="http://localhost:11434", **kw):
        super().__init__(**kw)
        self.host = host.rstrip("/")

    def _one(self, model, prompt, temperature, max_tokens):
        body = json.dumps({
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }).encode()
        for attempt in range(3):
            try:
                req = urllib.request.Request(
                    self.host + "/api/chat", data=body,
                    headers={"Content-Type": "application/json"}, method="POST")
                with urllib.request.urlopen(req, timeout=600) as r:
                    d = json.load(r)
                if isinstance(d, dict) and d.get("error"):
                    sys.stderr.write(f"[litmus] ollama {model}: {d['error']}\n")
                    return ""
                return d.get("message", {}).get("content", "") or ""
            except urllib.error.HTTPError as e:
                sys.stderr.write(
                    f"[litmus] ollama {model} HTTP {e.code} (pulled it? try: ollama pull {model})\n")
                return ""
            except urllib.error.URLError as e:
                # Connection refused = Ollama isn't running. Don't burn retries silently.
                sys.stderr.write(
                    f"[litmus] cannot reach Ollama at {self.host} — is `ollama serve` running? ({e.reason})\n")
                return ""
            except Exception:
                time.sleep(2 * (attempt + 1))
        return ""
