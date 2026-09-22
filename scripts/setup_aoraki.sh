#!/usr/bin/env bash
set -e

echo "== Aoraki Python setup =="

# Prefer versions visible on the user's Aoraki environment.
for MOD in   python/3.10.20-pmymmy   python/3.10.13-7ad3v37   python/3.10.8-modgmbk
do
  if module spider "$MOD" >/dev/null 2>&1; then
    echo "Found $MOD"
    echo "Attempting to load $MOD ..."
    if module load "$MOD" 2>/dev/null; then
      break
    fi
  fi
done

echo "Python:"
python3 --version
which python3

python3 - <<'PY'
import sys
assert sys.version_info >= (3,10), "Python >= 3.10 is required."
print("Python version OK")
PY

rm -rf .venv
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

echo
echo "Environment ready."
echo "Activate later with: source .venv/bin/activate"
