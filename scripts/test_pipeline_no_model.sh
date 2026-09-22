#!/usr/bin/env bash
set -e
source .venv/bin/activate
python -m src.make_fixture
python -m src.prepare_collab_seeds --config configs/aoraki_qwen.yaml
pytest -q
