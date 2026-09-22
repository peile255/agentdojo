from __future__ import annotations

import time
import yaml

from .local_qwen import LocalQwen


def main():
    with open(
        "configs/aoraki_qwen_micro.yaml",
        "r",
        encoding="utf-8",
    ) as f:
        cfg = yaml.safe_load(f)

    llm = LocalQwen(cfg["local_model"])

    system = (
        "You are a concise assistant. "
        "Answer in no more than three sentences."
    )

    user = (
        "A retrieved document may contain untrusted instructions. "
        "Explain briefly how a collaborative AI agent should treat them."
    )

    start = time.time()

    out = llm.generate(system, user)

    elapsed = time.time() - start

    print("GENERATION_OK")
    print("elapsed_seconds =", round(elapsed, 2))
    print("output_chars =", len(out))
    print("output =", out)


if __name__ == "__main__":
    main()
