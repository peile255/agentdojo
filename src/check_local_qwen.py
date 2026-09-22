import argparse
import yaml
from .local_qwen import LocalQwen

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/aoraki_qwen.yaml")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    llm = LocalQwen(cfg["local_model"])
    out = llm.generate(
        "You are a concise test assistant.",
        "Reply exactly with: LOCAL_QWEN_OK"
    )
    print(out)

if __name__ == "__main__":
    main()
