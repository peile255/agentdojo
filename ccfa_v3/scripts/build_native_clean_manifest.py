from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"Invalid JSON at {path}:{line_number}: {error}"
            ) from error
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a deduplicated manifest for matched native clean traces."
    )
    parser.add_argument(
        "--seeds",
        type=Path,
        default=Path("data/agentdojo_compromised_all.jsonl"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("ccfa_v3/native_clean/native_clean_manifest.tsv"),
    )
    parser.add_argument(
        "--pair-map",
        type=Path,
        default=Path("ccfa_v3/native_clean/native_clean_pair_map.json"),
    )
    args = parser.parse_args()

    rows = read_jsonl(args.seeds)
    if not rows:
        raise SystemExit(f"ERROR: no seeds found in {args.seeds}")

    required = {"seed_id", "suite_name", "user_task_id"}
    seen_seed_ids = set()
    tasks: OrderedDict[tuple[str, str], list[str]] = OrderedDict()

    for index, row in enumerate(rows, start=1):
        missing = required - set(row)
        if missing:
            raise SystemExit(
                f"ERROR: seed row {index} missing fields: {sorted(missing)}"
            )

        seed_id = str(row["seed_id"])
        if seed_id in seen_seed_ids:
            raise SystemExit(f"ERROR: duplicate seed_id: {seed_id}")
        seen_seed_ids.add(seed_id)

        suite = str(row["suite_name"])
        user_task = str(row["user_task_id"])
        tasks.setdefault((suite, user_task), []).append(seed_id)

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.pair_map.parent.mkdir(parents=True, exist_ok=True)

    manifest_text = "".join(
        f"{suite}\t{user_task}\n" for suite, user_task in tasks
    )
    args.manifest.write_text(manifest_text, encoding="utf-8")

    pair_map = {
        "source_seed_file": str(args.seeds),
        "seed_count": len(rows),
        "unique_task_count": len(tasks),
        "tasks": [
            {
                "suite_name": suite,
                "user_task_id": user_task,
                "seed_ids": seed_ids,
                "expected_trace": (
                    "agentdojo_runs/ccfa_native_clean/local/"
                    f"{suite}/{user_task}/none/none.json"
                ),
            }
            for (suite, user_task), seed_ids in tasks.items()
        ],
    }
    args.pair_map.write_text(
        json.dumps(pair_map, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"compromised seeds: {len(rows)}")
    print(f"unique clean tasks: {len(tasks)}")
    print(f"manifest: {args.manifest}")
    print(f"pair map: {args.pair_map}")


if __name__ == "__main__":
    main()
