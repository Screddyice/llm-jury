#!/usr/bin/env bash
# LLM-Jury release-readiness gate (fast). Exits 0 only when the repo is release-ready:
# required files present, working tree clean, and everything pushed to origin/main.
# (Build/install is verified separately during release prep, not on every gate run.)
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO" 2>/dev/null || { echo "FAIL: cannot cd $REPO"; exit 1; }

for f in CHANGELOG.md CONTRIBUTING.md LICENSE README.md pyproject.toml; do
  [ -f "$f" ] || { echo "FAIL: missing required file: $f"; exit 1; }
done
ls .github/workflows/*.yml >/dev/null 2>&1 || { echo "FAIL: no CI workflow in .github/workflows/"; exit 1; }

if [ -n "$(git status --porcelain)" ]; then
  echo "FAIL: uncommitted changes:"; git status --short; exit 1
fi
git fetch origin -q 2>/dev/null || true
UNPUSHED="$(git log origin/main..HEAD --oneline 2>/dev/null)"
if [ -n "$UNPUSHED" ]; then echo "FAIL: unpushed commits:"; echo "$UNPUSHED"; exit 1; fi

echo "RELEASE-READY: required files present, tree clean, pushed to origin/main."
