#!/usr/bin/env bash
# install-git-hooks.sh — point git at the versioned .githooks directory.
set -euo pipefail
cd "$(dirname "$0")/.."
git config core.hooksPath .githooks
chmod +x .githooks/commit-msg .githooks/pre-push
echo "git hooks installed (core.hooksPath=.githooks)"
