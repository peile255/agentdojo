from __future__ import annotations

import json
import time
import requests
import yaml


def main():
    with open(
        "configs/aoraki_qwen_micro.yaml",
        "r",
        encoding="utf-8",
    ) as f:
        cfg = yaml.safe_load(f)

    model_cfg = cfg["local_model"]

    url = model_cfg["base_url"].rstrip("/") + "/chat/completions"

    payload = {
        "model": model_cfg["model"],
        "temperature": 0.0,
        "max_tokens": 256,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a concise assistant. "
                    "Answer in no more than three sentences."
                ),
            },
            {
                "role": "user",
                "content": (
                    "A retrieved document may contain untrusted instructions. "
                    "Explain briefly how a collaborative AI agent should treat them."
                ),
            },
        ],
    }

    print("URL =", url)
    print("MODEL =", model_cfg["model"])
    print("MAX_TOKENS =", payload["max_tokens"])

    start = time.time()

    r = requests.post(
        url,
        json=payload,
        timeout=600,
    )

    elapsed = time.time() - start

    print("HTTP_STATUS =", r.status_code)
    print("ELAPSED_SECONDS =", round(elapsed, 2))

    r.raise_for_status()

    data = r.json()

    print("\n========== FULL RESPONSE ==========")
    print(json.dumps(data, indent=2, ensure_ascii=False))

    choice = data.get("choices", [{}])[0]
    message = choice.get("message", {})

    print("\n========== IMPORTANT FIELDS ==========")
    print("finish_reason =", choice.get("finish_reason"))
    print("message keys =", list(message.keys()))
    print("content repr =", repr(message.get("content")))
    print("reasoning repr =", repr(message.get("reasoning")))
    print(
        "reasoning_content repr =",
        repr(message.get("reasoning_content"))
    )
    print("usage =", data.get("usage"))


if __name__ == "__main__":
    main()
