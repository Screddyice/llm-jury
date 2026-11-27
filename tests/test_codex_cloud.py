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
           "LLMJURY_SHARED_BIN": str(tmp_path / "shared-bin"),
           "LLMJURY_FALLBACK_HOME": str(tmp_path / "fallback-home")}
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


def test_setup_accepts_a_link_made_by_a_concurrent_run(fake):
    tmp, env = fake
    shared = tmp / "shared-bin"
    shared.mkdir()
    # An `ln` that loses the race: the other run's link appears, then ours fails.
    racing_ln = tmp / "bin" / "ln"
    racing_ln.write_text('#!/bin/sh\n/bin/ln "$@"\necho "ln: File exists" >&2\nexit 1\n')
    racing_ln.chmod(0o755)
    result = subprocess.run(["bash", str(CLOUD / "setup.sh")], env=env,
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert (shared / "cursor-consult").is_symlink()


def test_setup_reports_the_real_link_error(fake):
    tmp, env = fake
    (tmp / "shared-bin").mkdir()
    failing_ln = tmp / "bin" / "ln"
    failing_ln.write_text('#!/bin/sh\necho "ln: Read-only file system" >&2\nexit 1\n')
    failing_ln.chmod(0o755)
    result = subprocess.run(["bash", str(CLOUD / "setup.sh")], env=env,
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert "Read-only file system" in result.stderr
    assert "not ours" not in result.stderr
    assert os.access(Path(env["HOME"]) / ".local" / "bin" / "cursor-consult", os.X_OK)


@pytest.mark.skipif(os.geteuid() == 0, reason="root can write to a 0555 folder")
def test_setup_and_consult_work_with_a_read_only_home(fake):
    tmp, env = fake
    home = Path(env["HOME"])
    home.mkdir()
    home.chmod(0o555)
    fallback = tmp / "fallback-home"
    env["LLMJURY_FALLBACK_HOME"] = str(fallback)
    (tmp / "bin" / "cursor-agent").write_text(
        '#!/bin/sh\nprintf "%s\\n" "$HOME" > "$ARGS_FILE.home"\nprintf "%s\\n" "$@" > "$ARGS_FILE"\necho ok\n')
    try:
        result = subprocess.run(["bash", str(CLOUD / "setup.sh")], env=env,
                                capture_output=True, text=True, timeout=60)
        assert result.returncode == 0, result.stderr
        helper = fallback / ".local" / "bin" / "cursor-consult"
        assert os.access(helper, os.X_OK)
        assert str(helper) in result.stdout
        run = subprocess.run([str(helper), "question"], env=env,
                             capture_output=True, text=True, timeout=30)
        assert run.returncode == 0, run.stderr
        assert (tmp / "args.home").read_text().strip() == str(fallback)
    finally:
        home.chmod(0o755)


@pytest.mark.skipif(os.geteuid() == 0, reason="root can write to a 0555 folder")
def test_setup_avoids_a_fallback_folder_others_can_write(fake):
    tmp, env = fake
    home = Path(env["HOME"])
    home.mkdir()
    home.chmod(0o555)
    hostile = Path(env["LLMJURY_FALLBACK_HOME"])
    hostile.mkdir()
    hostile.chmod(0o777)
    env["TMPDIR"] = str(tmp)
    try:
        result = subprocess.run(["bash", str(CLOUD / "setup.sh")], env=env,
                                capture_output=True, text=True, timeout=60)
        assert result.returncode == 0, result.stderr
        assert not (hostile / ".local").exists()
        installed = Path(result.stdout.strip().splitlines()[-1].split(" at ", 1)[1])
        private = installed.parents[2]
        assert private.parent == tmp and private.stat().st_mode & 0o777 == 0o700
        run = subprocess.run([str(installed), "question"], env=env,
                             capture_output=True, text=True, timeout=30)
        assert run.returncode == 0, run.stderr
    finally:
        home.chmod(0o755)


@pytest.mark.skipif(os.geteuid() == 0, reason="root can write to a 0555 folder")
def test_consult_through_a_link_still_finds_its_home(fake):
    tmp, env = fake
    home = Path(env["HOME"])
    home.mkdir()
    home.chmod(0o555)
    (tmp / "bin" / "cursor-agent").write_text(
        '#!/bin/sh\nprintf "%s\\n" "$HOME" > "$ARGS_FILE.home"\necho ok\n')
    try:
        result = subprocess.run(["bash", str(CLOUD / "setup.sh")], env=env,
                                capture_output=True, text=True, timeout=60)
        assert result.returncode == 0, result.stderr
        private = Path(env["LLMJURY_FALLBACK_HOME"])
        link = tmp / "elsewhere" / "cursor-consult"
        link.parent.mkdir()
        link.symlink_to(private / ".local" / "bin" / "cursor-consult")
        run = subprocess.run([str(link), "question"], env=env,
                             capture_output=True, text=True, timeout=30)
        assert run.returncode == 0, run.stderr
        assert (tmp / "args.home").read_text().strip() == str(private)
    finally:
        home.chmod(0o755)
