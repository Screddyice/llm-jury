"""The Codex cloud helper that lets a Codex task ask Cursor for a second opinion."""

import os
from pathlib import Path
import subprocess

import pytest

CLOUD = Path(__file__).resolve().parents[1] / "deploy" / "codex-cloud"
FAKE_AGENT = """#!/bin/sh
printf '%s\\n' "$@" > "$ARGS_FILE"
echo "Cursor says: guard the empty list."
"""


@pytest.fixture
def fake(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    agent = bin_dir / "cursor-agent"
    agent.write_text(FAKE_AGENT)
    agent.chmod(0o755)
    env = {"PATH": f"{bin_dir}:/usr/bin:/bin", "HOME": str(tmp_path / "home"),
           "ARGS_FILE": str(tmp_path / "args"), "CURSOR_API_KEY": "test-key",
           "LLMJURY_SHARED_BIN": str(tmp_path / "shared-bin")}
    return tmp_path, env


def consult(env, *args):
    return subprocess.run([str(CLOUD / "cursor-consult"), *args], env=env,
                          capture_output=True, text=True, timeout=30)


def test_consult_sends_the_question_and_files_read_only(fake):
    tmp, env = fake
    source = tmp / "stats.py"
    source.write_text("def avg(xs):\n    return sum(xs) / len(xs)\n")
    result = consult(env, "Is avg safe for an empty list?", str(source))
    assert result.returncode == 0, result.stderr
    assert "guard the empty list" in result.stdout
    args = (tmp / "args").read_text()
    assert "--mode\nask\n" in args and "--print" in args
    assert "--model\ncomposer-2.5\n" in args
    assert "Is avg safe for an empty list?" in args and "sum(xs) / len(xs)" in args


def test_consult_fails_loudly_without_a_key(fake):
    _, env = fake
    del env["CURSOR_API_KEY"]
    result = consult(env, "question")
    assert result.returncode == 2 and "CURSOR_API_KEY" in result.stderr


def test_consult_fails_loudly_without_the_cli(fake):
    tmp, env = fake
    (tmp / "bin" / "cursor-agent").unlink()
    result = consult(env, "question")
    assert result.returncode == 3 and "setup" in result.stderr


def test_setup_installs_the_helper_and_instructions_once(fake):
    tmp, env = fake
    for _ in range(2):
        result = subprocess.run(["bash", str(CLOUD / "setup.sh")], env=env,
                                capture_output=True, text=True, timeout=60)
        assert result.returncode == 0, result.stderr
    home = Path(env["HOME"])
    helper = home / ".local" / "bin" / "cursor-consult"
    assert helper.is_file() and os.access(helper, os.X_OK)
    instructions = (home / ".codex" / "AGENTS.md").read_text()
    assert instructions.count("cursor-consult") >= 1
    assert instructions.count("<!-- llmjury:cursor-consult -->") == 1


def test_consult_reads_a_file_named_like_an_option(fake):
    tmp, env = fake
    (tmp / "--help").write_text("print('real file')\n")
    result = subprocess.run([str(CLOUD / "cursor-consult"), "Check this", "--help"], env=env,
                            cwd=tmp, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert "real file" in (tmp / "args").read_text()


def test_consult_caps_the_combined_context(fake):
    tmp, env = fake
    files = []
    for n in range(9):
        path = tmp / f"big{n}.txt"
        path.write_text("x" * 20000)
        files.append(str(path))
    env["CURSOR_CONSULT_CONTEXT_LIMIT"] = "16000"
    result = consult(env, "Review these", *files)
    assert result.returncode == 0, result.stderr
    assert len((tmp / "args").read_text()) < 20000


def test_setup_puts_the_helper_on_the_default_path(fake):
    tmp, env = fake
    shared = tmp / "shared-bin"
    shared.mkdir()
    result = subprocess.run(["bash", str(CLOUD / "setup.sh")], env=env,
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert os.access(shared / "cursor-consult", os.X_OK)


def test_setup_leaves_another_cursor_consult_alone(fake):
    tmp, env = fake
    shared = tmp / "shared-bin"
    shared.mkdir()
    other = shared / "cursor-consult"
    other.write_text("#!/bin/sh\necho someone else\n")
    for _ in range(2):
        result = subprocess.run(["bash", str(CLOUD / "setup.sh")], env=env,
                                capture_output=True, text=True, timeout=60)
        assert result.returncode == 0, result.stderr
    assert not other.is_symlink() and "someone else" in other.read_text()
    assert "not ours" in result.stderr
