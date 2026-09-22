from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import yaml

from .agentdojo_adapter import extract_candidate_records


def numeric_suffix(value):
    if value is None:
        return -1
    m = re.search(r"(\d+)$", str(value))
    return int(m.group(1)) if m else -1


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default="configs/aoraki_qwen.yaml",
    )

    parser.add_argument(
        "--logdir",
        default=None,
    )

    parser.add_argument(
        "--output",
        default="data/agentdojo_collab_seeds.jsonl",
    )

    parser.add_argument(
        "--only-compromised",
        action="store_true",
        help=(
            "Keep only AgentDojo runs where the injection goal "
            "was successfully executed (security=True)."
        ),
    )

    parser.add_argument(
        "--include-utility-failures",
        action="store_true",
        help="Also keep attacked records with utility=False.",
    )

    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    logdir = args.logdir or cfg["agentdojo"]["logdir"]

    records = extract_candidate_records(logdir)

    selected = []

    for record in records:
        # Adapter already restricts this to real attacked records,
        # but keep the condition explicit for reproducibility.
        if not record.get("attack_exposed", False):
            continue

        if (
            not args.include_utility_failures
            and record.get("utility") is not True
        ):
            continue

        if (
            args.only_compromised
            and record.get("attack_success") is not True
        ):
            continue

        selected.append(record)

    selected.sort(
        key=lambda r: (
            str(r.get("suite_name", "")),
            numeric_suffix(r.get("user_task_id")),
            numeric_suffix(r.get("injection_task_id")),
        )
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", encoding="utf-8") as f:
        for i, record in enumerate(selected):
            row = {
                "seed_id": f"agentdojo_seed_{i:06d}",
                **record,
            }

            f.write(
                json.dumps(row, ensure_ascii=False)
                + "\n"
            )

    attack_successes = sum(
        r.get("attack_success") is True
        for r in selected
    )

    attack_failures = sum(
        r.get("attack_success") is False
        for r in selected
    )

    compromised = sum(
        r.get("source_compromised") is True
        for r in selected
    )

    utility_success = sum(
        r.get("utility") is True
        for r in selected
    )

    print("AgentDojo collaborative-seed preparation complete")
    print("logdir =", logdir)
    print("attacked records found =", len(records))
    print("selected attack-exposed seeds =", len(selected))
    print("attack_success =", attack_successes)
    print("attack_failure =", attack_failures)
    print("source_compromised =", compromised)
    print("utility_success =", utility_success)
    print("output =", out)


if __name__ == "__main__":
    main()
