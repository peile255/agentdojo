from __future__ import annotations

import argparse
import json
from pathlib import Path


FILES = {
    "clean_standard": ("clean", False),
    "clean_defended": ("clean", True),
    "attack_standard": ("compromised", False),
    "attack_defended": ("compromised", True),
}

TOPOLOGIES = {
    "sequential",
    "peer_debate",
    "hierarchical",
    "safety_monitor",
}


def load(path: Path) -> list[dict]:
    rows = []

    if not path.exists():
        raise FileNotFoundError(path)

    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue

        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise ValueError(
                f"Invalid JSON at {path}:{line_number}: {error}"
            ) from error

    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()

    all_rows = {}
    failures = []

    for stem, (condition, defended) in FILES.items():
        path = args.directory / f"{stem}.jsonl"
        rows = load(path)
        all_rows[stem] = rows

        if len(rows) != 4:
            failures.append(
                f"{stem}: expected 4 rows, found {len(rows)}"
            )

        observed_topologies = {
            row.get("topology") for row in rows
        }

        if observed_topologies != TOPOLOGIES:
            failures.append(
                f"{stem}: topology mismatch {observed_topologies}"
            )

        for row in rows:
            prefix = f"{stem}/{row.get('topology')}"

            if row.get("attack_condition") != condition:
                failures.append(f"{prefix}: wrong attack_condition")

            if row.get("defense_prompt") is not defended:
                failures.append(f"{prefix}: wrong defense_prompt")

            events = row.get("events") or []
            native_events = [
                event
                for event in events
                if event.get("provenance") == "native_agentdojo"
            ]

            if condition == "clean":
                if row.get("attack_exposed") is not False:
                    failures.append(f"{prefix}: attack_exposed must be false")
                if row.get("exposed_agents") != []:
                    failures.append(f"{prefix}: clean exposed_agents not empty")
                if row.get("source_compromised_runtime") is not False:
                    failures.append(f"{prefix}: clean source compromised")
                if native_events:
                    failures.append(f"{prefix}: clean has native event")
                if float(row.get("exposure_ratio", -1)) != 0.0:
                    failures.append(f"{prefix}: clean exposure ratio is not zero")
            else:
                if row.get("attack_exposed") is not True:
                    failures.append(f"{prefix}: attack_exposed must be true")
                if row.get("exposed_agents") != ["a1"]:
                    failures.append(f"{prefix}: source must be a1")
                if row.get("source_compromised_runtime") is not True:
                    failures.append(f"{prefix}: source compromise missing")
                if len(native_events) != 1:
                    failures.append(
                        f"{prefix}: expected one native event, "
                        f"found {len(native_events)}"
                    )

    pairing = {}

    for stem, rows in all_rows.items():
        for row in rows:
            key = (
                row.get("seed_id"),
                row.get("topology"),
            )
            pairing.setdefault(key, {})[stem] = row.get("rng_seed")

    for key, values in pairing.items():
        if set(values) != set(FILES):
            failures.append(f"{key}: missing paired conditions")
            continue

        if len(set(values.values())) != 1:
            failures.append(f"{key}: rng_seed differs across conditions")

    print("=" * 72)
    print("CCFA FOUR-CONDITION SMOKE VALIDATION")
    print("=" * 72)
    print("files:", len(all_rows))
    print("rows:", sum(len(rows) for rows in all_rows.values()))
    print("paired task-topology keys:", len(pairing))

    if failures:
        print("status: FAILED")
        for failure in failures:
            print("-", failure)
        raise SystemExit(1)

    print("status: PASSED")
    print("Clean conditions contain no exposed agent or native compromise event.")
    print("Compromised conditions contain exactly one native source event at a1.")
    print("RNG seeds are paired across all four experimental conditions.")


if __name__ == "__main__":
    main()
