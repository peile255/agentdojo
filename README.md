# AgentDojo + qwen36_27b_local on Aoraki

This version is designed for the Otago Aoraki cluster and uses **no paid API**.

Model:
```text
qwen36_27b_local
```

Public benchmark source:
```text
AgentDojo
```

## Important design

AgentDojo supplies the public benchmark tasks / prompt-injection traces.

The collaborative LLM layer uses only the local model:

```text
qwen36_27b_local
```

No OpenAI / Anthropic / Google paid API key is required.

---

## 1. Aoraki Python requirement

AgentDojo requires Python >= 3.10.

Your Aoraki default Python may be 3.9, so first inspect:

```bash
module avail python
module spider python
```

Then load one of the Python 3.10 modules available on Aoraki.

Example:

```bash
module spider python/3.10.13-7ad3v37
```

Follow any prerequisite-module instructions shown by `module spider`, then:

```bash
module load python/3.10.13-7ad3v37
```

Check:

```bash
python3 --version
which python3
```

Only continue if Python >= 3.10.

---

## 2. Create environment

```bash
cd ~/agentdojo_experiment

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

Test:

```bash
python -c "import yaml, pandas, agentdojo; print('ENV_OK')"
```

---

## 3. Local Qwen backends

This project supports two free/local inference modes.

### Mode A: OpenAI-compatible LOCAL endpoint

Recommended if Aoraki exposes `qwen36_27b_local` through vLLM, SGLang, TGI gateway, or another local HTTP service.

Configure:

```yaml
local_model:
  backend: openai_compatible
  model: qwen36_27b_local
  base_url: http://127.0.0.1:8000/v1
```

Or export:

```bash
export QWEN_BACKEND=openai_compatible
export QWEN_MODEL=qwen36_27b_local
export QWEN_BASE_URL=http://127.0.0.1:8000/v1
```

Test:

```bash
python -m src.check_local_qwen --config configs/aoraki_qwen.yaml
```

Expected response should contain:

```text
LOCAL_QWEN_OK
```

### Mode B: local Transformers model path

If the model is stored directly on Aoraki:

```yaml
local_model:
  backend: transformers
  model: qwen36_27b_local
  model_path: /actual/aoraki/model/path
```

Install:

```bash
pip install -r requirements_transformers.txt
```

Then:

```bash
export QWEN_BACKEND=transformers
export QWEN_MODEL_PATH=/actual/aoraki/model/path

python -m src.check_local_qwen --config configs/aoraki_qwen.yaml
```

A 27B-class model normally requires substantial GPU memory; request the GPU resources required by the actual Aoraki deployment.

---

## 4. Prepare AgentDojo-derived seeds

Place your real AgentDojo run artifacts under:

```text
agentdojo_runs/
```

Then:

```bash
python -m src.prepare_collab_seeds \
  --config configs/aoraki_qwen.yaml
```

Output:

```text
data/agentdojo_collab_seeds.jsonl
```

Check:

```bash
wc -l data/agentdojo_collab_seeds.jsonl
head -1 data/agentdojo_collab_seeds.jsonl
```

---

## 5. Test parser without calling any model

```bash
bash scripts/test_pipeline_no_model.sh
```

This uses only a synthetic fixture for code testing.

Do not report fixture results in the paper.

---

## 6. Pilot experiment

Once qwen36_27b_local is confirmed:

```bash
sbatch slurm/02_run_collab_pilot.sbatch
```

Check queue:

```bash
squeue -u $USER
```

Check logs:

```bash
tail -f logs/pilot-<JOBID>.out
```

Pilot uses:

```text
--limit 2
```

Only two AgentDojo-derived seeds are used.

---

## 7. Full experiment

After the pilot succeeds:

```bash
sbatch slurm/03_run_collab_full.sbatch
```

Results:

```text
results/collab_raw.jsonl
results/collab_summary.csv
figures/rpr_by_architecture.png
```

---

## 8. Architectures

The project evaluates:

1. Independent Voting
2. Sequential Collaboration
3. Peer Debate
4. Hierarchical Collaboration
5. Safety Monitor

---

## 9. Metrics

Current automated outputs include:

- Utility
- RPR: Risk Propagation Rate
- CR: Consensus Robustness
- UCR: Unsafe Consensus Rate
- F_SU: Safety–Utility harmonic score

For the final paper, add matched benign/attack runs to compute:

```text
TSD = Safety_benign - Safety_adversarial
```

and add round-level logging for a stronger REC estimate.

---

## 10. Critical note about qwen36_27b_local

The project cannot know how Otago Research Computing exposes the local model.

You must determine whether `qwen36_27b_local` is:

- an OpenAI-compatible local endpoint,
- a local HuggingFace model path,
- or a cluster-specific wrapper/command.

If it is a cluster-specific wrapper rather than HTTP or Transformers, only
`src/local_qwen.py` needs to be extended.

The rest of the experimental pipeline remains unchanged.

---

## 11. Recommended first commands on Aoraki

```bash
cd ~/agentdojo_experiment

module spider python/3.10.13-7ad3v37
```

Load the prerequisite shown by Aoraki, then:

```bash
module load python/3.10.13-7ad3v37

python3 --version
python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt

python -c "import yaml, pandas, agentdojo; print('ENV_OK')"
```

Then determine the local-Qwen access method and run:

```bash
python -m src.check_local_qwen --config configs/aoraki_qwen.yaml
```
