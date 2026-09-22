from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    args = parser.parse_args()
    rows = [
        json.loads(line)
        for line in Path(args.input).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != 176:
        raise SystemExit(f"FAILED: expected 176 rows, found {len(rows)}")
    keys = {
        (r["seed_id"], r["topology"], r["attack_condition"], r["defense_prompt"])
        for r in rows
    }
    if len(keys) != 176:
        raise SystemExit(f"FAILED: expected 176 unique identities, found {len(keys)}")
    invalid = [
        r["row_id"] for r in rows
        if r["task_mode"] == "action"
        and not isinstance(r["verified_execution_completed"], bool)
    ]
    if invalid:
        raise SystemExit(f"FAILED: missing action execution labels: {invalid[:5]}")
    counts = Counter(
        (r["topology"], r["attack_condition"], r["defense_prompt"], r["utility_v2"])
        for r in rows
    )
    print("=" * 72)
    print("UTILITY V2 VALIDATION")
    print("=" * 72)
    print("rows: 176")
    print("unique identities: 176")
    print("status: PASSED")
    print("\nUTILITY V2 COUNTS")
    for key, count in sorted(counts.items()):
        print(key, count)


if __name__ == "__main__":
    main()

