#!/bin/sh
# skill-issue hook launcher. Finds a Python 3 and runs the standard-library-only hook client.
# It must never break a prompt: if no Python is found, it exits 0 and prints nothing.
ROOT="${CLAUDE_PLUGIN_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
HOOK="$ROOT/src/skillissue/hook.py"
SI_HOME="${SKILL_ISSUE_HOME:-$HOME/.skill-issue}"

run() {
  # -I: ignore PYTHON* env vars and user site; -S: skip site import (faster; the hook is stdlib only)
  exec "$1" -I -S "$HOOK" "$2"
}

# 1. explicit override, 2. the plugin's own venv, 3. the Python the daemon last ran with
for py in "$SKILL_ISSUE_HOOK_PYTHON" \
          "${CLAUDE_PLUGIN_DATA:+$CLAUDE_PLUGIN_DATA/venv/bin/python}" \
          "${CLAUDE_PLUGIN_DATA:+$CLAUDE_PLUGIN_DATA/venv/Scripts/python.exe}"; do
  [ -n "$py" ] && [ -x "$py" ] && run "$py" "$1"
done
if [ -f "$SI_HOME/python.txt" ]; then
  py=$(cat "$SI_HOME/python.txt")
  [ -n "$py" ] && [ -x "$py" ] && run "$py" "$1"
fi
# 4. whatever is on PATH, checked first (Windows ships a python3 stub that does not run)
for py in python3 python; do
  if command -v "$py" >/dev/null 2>&1 && "$py" -S -c "" >/dev/null 2>&1; then
    run "$py" "$1"
  fi
done
exit 0
