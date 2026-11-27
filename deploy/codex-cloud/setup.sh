#!/usr/bin/env bash
# Codex cloud environment setup: install the Cursor CLI and the cursor-consult helper,
# and tell Codex when to use it. Safe to run more than once.
#   curl -fsSL https://raw.githubusercontent.com/Screddyice/llm-jury/main/deploy/codex-cloud/setup.sh | bash
set -euo pipefail

raw="${LLMJURY_RAW:-https://raw.githubusercontent.com/Screddyice/llm-jury/main/deploy/codex-cloud}"
bin="$HOME/.local/bin"
mkdir -p "$bin" "$HOME/.codex"
export PATH="$bin:$PATH"

if ! command -v cursor-agent >/dev/null 2>&1 && ! command -v agent >/dev/null 2>&1; then
  curl -fsS https://cursor.com/install | bash
fi

here="$(cd "$(dirname "${BASH_SOURCE[0]:-.}")" 2>/dev/null && pwd || true)"
if [ -n "$here" ] && [ -f "$here/cursor-consult" ]; then
  install -m 755 "$here/cursor-consult" "$bin/cursor-consult"
else
  curl -fsSL "$raw/cursor-consult" -o "$bin/cursor-consult"
  chmod 755 "$bin/cursor-consult"
fi

# Non-interactive task shells skip ~/.bashrc, so also link the helper onto the default PATH.
shared="${LLMJURY_SHARED_BIN:-/usr/local/bin}"
link="$shared/cursor-consult"
if [ -d "$shared" ] && [ -w "$shared" ]; then
  if [ ! -e "$link" ] && [ ! -L "$link" ]; then
    ln -s "$bin/cursor-consult" "$link"
  elif [ "$(readlink "$link" 2>/dev/null)" != "$bin/cursor-consult" ]; then
    echo "setup: $link already exists and is not ours; leaving it. Use ~/.local/bin/cursor-consult." >&2
  fi
fi

if ! grep -qs 'HOME/.local/bin' "$HOME/.bashrc"; then
  # shellcheck disable=SC2016  # write a literal $HOME for later shells
  echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.bashrc"
fi

marker="<!-- llmjury:cursor-consult -->"
if ! grep -qsF "$marker" "$HOME/.codex/AGENTS.md"; then
  cat >> "$HOME/.codex/AGENTS.md" <<NOTE

$marker
## Second opinion from Cursor
Before a material implementation decision, when stuck on a non-obvious failure, and
before you finish a non-trivial change, ask Cursor for a read-only second opinion:

    ~/.local/bin/cursor-consult "one precise question" path/to/changed_file [more files]

Weigh its answer against the code and tests; it is advisory. If cursor-consult fails
or is unavailable, say so in your summary and continue with your own checks. Never
pass secrets or credential files to it.
NOTE
fi

echo "cursor-consult installed at $bin/cursor-consult"
