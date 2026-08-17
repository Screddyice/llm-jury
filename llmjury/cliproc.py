"""Shared CLI subprocess execution for backends, delegation, and planning.

Three call sites (CodexBackend, CodexDelegator, ClaudePlanner) run an
external CLI the same way: injected runner, captured text output, a timeout, no
check, and the same two failure modes. This module centralizes only that
mechanics — every caller keeps its own command construction and its own error
messages, so migrating a site changes no observable behavior.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass


@dataclass
class CliRunOutcome:
    """One of three mutually exclusive outcomes of a CLI run.

    - ``completed`` is the ``CompletedProcess`` when the runner returned.
    - ``timed_out`` is True when the runner raised ``subprocess.TimeoutExpired``.
    - ``os_error`` is the ``OSError`` when the CLI could not be started.
    """

    completed: subprocess.CompletedProcess | None = None
    timed_out: bool = False
    os_error: OSError | None = None


def run_cli(runner, cmd, timeout, cwd=None):
    """Invoke ``runner`` (``subprocess.run``-shaped) and classify the outcome.

    Always passes ``stdout=PIPE, stderr=PIPE, text=True, check=False`` plus the
    given ``timeout``; ``cwd`` is forwarded only when set, so call sites that
    never passed one keep an identical runner invocation (injected test runners
    see the same kwargs as before).
    """
    kwargs = dict(
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        timeout=timeout, check=False,
    )
    if cwd is not None:
        kwargs["cwd"] = cwd
    try:
        completed = runner(cmd, **kwargs)
    except subprocess.TimeoutExpired:
        return CliRunOutcome(timed_out=True)
    except OSError as exc:
        return CliRunOutcome(os_error=exc)
    return CliRunOutcome(completed=completed)
