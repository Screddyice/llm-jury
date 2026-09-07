"""Memory admission tests. No model inference or live host probes."""
import contextlib
import io
import os
import sys
import unittest
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llmjury import memguard


class MemoryAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        for name, value in {
            "router_failover": (False, ""),
            "exclusive_compute": (False, ""),
            "simulator_stack": (False, 0),
            "total_ram_bytes": 36 * memguard.GB,
            "disk_sizes": {"small:latest": 3 * memguard.GB},
            "loaded_bytes": (0, {}),
        }.items():
            self.stack.enter_context(patch.object(memguard, name, return_value=value))

    def check(self, available=20, pressure=1, cache=1024, loaded=None):
        if loaded is not None:
            self.stack.enter_context(patch.object(memguard, "loaded_bytes", return_value=loaded))
        with patch.object(memguard, "host_memory", return_value=(available * memguard.GB, pressure), create=True), patch.dict(os.environ, {"LLMJURY_PROMPT_CACHE_MIB": str(cache)}):
            return memguard.check(["small"], num_ctx=8192, parallel=1)

    def test_refuses_when_desktop_has_no_headroom(self):
        self.assertFalse(self.check(available=3).ok)

    def test_refuses_warning_pressure_even_if_free_metric_is_high(self):
        self.assertFalse(self.check(pressure=2).ok)

    def test_accounts_for_prompt_cache_outside_ollama_ps(self):
        report = self.check(cache=8192)
        self.assertGreater(report.projected, 12 * memguard.GB)

    def test_resident_model_still_reserves_its_unreported_cache(self):
        report = self.check(available=3, loaded=(4 * memguard.GB, {"small:latest": 4 * memguard.GB}), cache=8192)
        self.assertFalse(report.ok)

    def test_allows_small_work_with_headroom(self):
        self.assertTrue(self.check().ok)

    def test_unknown_host_pressure_refuses_local_admission(self):
        with patch.object(memguard, "host_memory", return_value=(None, None), create=True):
            self.assertFalse(memguard.check(["small"], parallel=1).ok)

    def test_check_honors_exclusive_ownership_for_library_callers(self):
        with patch.object(memguard, "exclusive_compute", return_value=(True, "Qwen lease")):
            report = memguard.check(["small"], parallel=1)
            self.assertFalse(report.ok)
            self.assertTrue(report.terminal)

    def test_unlimited_cache_refuses_instead_of_assuming_zero(self):
        self.assertFalse(self.check(cache=-1).ok)

    def test_preflight_cli_stops_without_creating_provider(self):
        from llmjury import cli
        with patch.object(memguard, "host_memory", return_value=(memguard.GB, 2), create=True), patch.object(cli, "_backend", side_effect=AssertionError("must not create provider")), patch.object(sys, "argv", ["llmjury", "preflight", "--models", "small", "--num-ctx", "8192"]), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as result:
                cli.main()
            self.assertEqual(result.exception.code, 1)

    def test_unreadable_residency_refuses_local_admission(self):
        self.assertFalse(self.check(loaded=(None, {})).ok)


class HostProbeTests(unittest.TestCase):
    def test_missing_or_invalid_residency_is_not_an_empty_server(self):
        for payload in ({}, {"models": None}, {"models": [{"name": "other", "size": -1}]}, {"models": ["invalid"]}):
            with self.subTest(payload=payload), patch.object(memguard, "_get_json", return_value=payload):
                self.assertEqual(memguard.loaded_bytes("http://localhost"), (None, {}))

    def test_darwin_probe_uses_query_only_and_reports_warning(self):
        replies = [SimpleNamespace(stdout="2\n"), SimpleNamespace(stdout="System-wide memory free percentage: 25%\n")]
        with patch.object(memguard.platform, "system", return_value="Darwin"), patch.object(memguard, "total_ram_bytes", return_value=36 * memguard.GB), patch.object(memguard.subprocess, "run", side_effect=replies) as probe:
            self.assertEqual(memguard.host_memory(), (9 * memguard.GB, 2))
            self.assertEqual(probe.call_args_list[1].args[0], ["/usr/bin/memory_pressure", "-Q"])

    def test_bad_probe_never_becomes_zero_pressure(self):
        with patch.object(memguard.platform, "system", return_value="Darwin"), patch.object(memguard.subprocess, "run", side_effect=OSError("unavailable")):
            self.assertEqual(memguard.host_memory(), (None, None))

    def test_local_lock_refuses_second_owner_and_releases_on_exit(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"LLMJURY_LOCAL_LOCK": directory + "/compute.lock"}):
            with memguard.local_compute_lock():
                with self.assertRaises(RuntimeError):
                    with memguard.local_compute_lock():
                        self.fail("second owner admitted")
            with memguard.local_compute_lock():
                pass

    def test_local_benchmark_respects_the_same_compute_lock(self):
        from llmjury import cli
        from llmjury.benchmarks import reproduce
        args = SimpleNamespace(which="humaneval", backend="ollama", n=1, k=1, pace=0, num_ctx=8192)
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"LLMJURY_LOCAL_LOCK": directory + "/compute.lock"}), patch.object(reproduce, "run", side_effect=AssertionError("benchmark started despite existing owner")):
            with memguard.local_compute_lock():
                with self.assertRaises(SystemExit) as result:
                    cli.cmd_reproduce(args)
                self.assertIn("another local council or review", str(result.exception))


if __name__ == "__main__":
    unittest.main()
