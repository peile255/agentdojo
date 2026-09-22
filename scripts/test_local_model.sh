#!/usr/bin/env bash
set -e
source .venv/bin/activate
python -m src.check_local_qwen --config configs/aoraki_qwen.yaml
