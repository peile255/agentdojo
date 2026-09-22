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
TOPOLOGIES = {"sequential", "peer_debate", "hierarchical", "safety_monitor"}


def load(path: Path) -> list[dict]:
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid JSON at {path}:{number}: {error}") from error
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    failures = []
    all_rows = {}

    for stem, (condition, defended) in FILES.items():
        path = args.directory / f"{stem}.jsonl"
        if not path.exists():
            failures.append(f"missing {path}")
            continue
        rows = load(path)
        all_rows[stem] = rows
        if len(rows) != 4:
            failures.append(f"{stem}: expected 4 rows, found {len(rows)}")
        if {row.get("topology") for row in rows} != TOPOLOGIES:
            failures.append(f"{stem}: topology set mismatch")

        for row in rows:
            label = f"{stem}/{row.get('topology')}"
            if row.get("attack_condition") != condition:
                failures.append(f"{label}: wrong condition")
            if row.get("defense_prompt") is not defended:
                failures.append(f"{label}: wrong defense flag")
            native = [
                event for event in row.get("events", [])
                if str(event.get("provenance", "")).startswith("native_agentdojo")
            ]
            if len(native) != 1 or native[0].get("agent_id") != "a1":
                failures.append(f"{label}: expected exactly one native a1 event")
                continue
            if condition == "clean":
                if native[0].get("provenance") != "native_agentdojo_clean":
                    failures.append(f"{label}: wrong clean provenance")
                if native[0].get("unsafe") is not False:
                    failures.append(f"{label}: clean source marked unsafe")
                if row.get("exposed_agents") != []:
                    failures.append(f"{label}: clean source exposed")
                if row.get("source_compromised_runtime") is not False:
                    failures.append(f"{label}: clean source compromised")
                if not row.get("source_sha256"):
                    failures.append(f"{label}: clean source hash missing")
            else:
                if native[0].get("provenance") != "native_agentdojo":
                    failures.append(f"{label}: wrong attack provenance")
                if native[0].get("unsafe") is not True:
                    failures.append(f"{label}: compromised source marked safe")
                if row.get("exposed_agents") != ["a1"]:
                    failures.append(f"{label}: compromised source is not a1")
                if row.get("source_compromised_runtime") is not True:
                    failures.append(f"{label}: compromise not observed")

    pairing = {}
    source_outputs = {}
    for stem, rows in all_rows.items():
        for row in rows:
            key = (row.get("seed_id"), row.get("topology"))
            pairing.setdefault(key, {})[stem] = row.get("rng_seed")
            native_events = [
                event for event in row.get("events", [])
                if str(event.get("provenance", "")).startswith("native_agentdojo")
            ]
            if len(native_events) != 1:
                continue
            native = native_events[0]
            source_key = (row.get("seed_id"), row.get("attack_condition"))
            source_outputs.setdefault(source_key, set()).add(native.get("output"))

    for key, values in pairing.items():
        if set(values) != set(FILES):
            failures.append(f"{key}: incomplete four-condition pair")
        elif len(set(values.values())) != 1:
            failures.append(f"{key}: RNG seed mismatch")
    for key, outputs in source_outputs.items():
        if len(outputs) != 1:
            failures.append(f"{key}: native source differs across topology/defense")

    print("=" * 72)
    print("NATIVE-PAIRED FOUR-CONDITION SMOKE VALIDATION")
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
    print("Both clean and compromised conditions use fixed native AgentDojo a1 traces.")
    print("Native a1 outputs are invariant across topology and defense within condition.")
    print("RNG seeds are paired across all four conditions.")


if __name__ == "__main__":
    main()
