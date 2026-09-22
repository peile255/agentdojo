from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


FILES = {
    "clean_standard": ("clean", False),
    "clean_defended": ("clean", True),
    "attack_standard": ("compromised", False),
    "attack_defended": ("compromised", True),
}
TOPOLOGIES = {"sequential", "peer_debate", "hierarchical", "safety_monitor"}
SHA256 = re.compile(r"^[0-9a-f]{64}$")


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    failures: list[str] = []
    all_rows: dict[str, list[dict]] = {}

    for stem, (condition, defended) in FILES.items():
        path = args.directory / f"{stem}.jsonl"
        if not path.exists():
            failures.append(f"missing {path}")
            continue
        rows = load(path)
        all_rows[stem] = rows
        if len(rows) != 44:
            failures.append(f"{stem}: expected 44 rows, found {len(rows)}")

        keys = [(row.get("seed_id"), row.get("topology")) for row in rows]
        if len(set(keys)) != 44:
            failures.append(f"{stem}: duplicate seed-topology keys")
        if len({row.get("seed_id") for row in rows}) != 11:
            failures.append(f"{stem}: expected 11 seeds")
        if {row.get("topology") for row in rows} != TOPOLOGIES:
            failures.append(f"{stem}: topology set mismatch")

        tiers = Counter(
            "primary" if row.get("primary_strict") is True else "semantic"
            for row in rows
        )
        if tiers != Counter({"primary": 36, "semantic": 8}):
            failures.append(f"{stem}: unexpected alignment strata {dict(tiers)}")

        for row in rows:
            label = f"{stem}/{row.get('seed_id')}/{row.get('topology')}"
            if row.get("attack_condition") != condition:
                failures.append(f"{label}: wrong condition")
            if row.get("defense_prompt") is not defended:
                failures.append(f"{label}: wrong defense flag")
            if not SHA256.match(str(row.get("canonical_evidence_sha256", ""))):
                failures.append(f"{label}: invalid evidence hash")
            if not row.get("alignment_tier"):
                failures.append(f"{label}: alignment tier missing")
            if not isinstance(row.get("primary_strict"), bool):
                failures.append(f"{label}: primary_strict is not Boolean")
            if not isinstance(row.get("utility"), bool):
                failures.append(f"{label}: utility is not Boolean")
            if not isinstance(row.get("final_unsafe"), bool):
                failures.append(f"{label}: final_unsafe is not Boolean")

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
                if "<INFORMATION" in str(native[0].get("output", "")).upper():
                    failures.append(f"{label}: injection marker in clean source")
            else:
                if native[0].get("provenance") != "native_agentdojo":
                    failures.append(f"{label}: wrong attack provenance")
                if native[0].get("unsafe") is not True:
                    failures.append(f"{label}: compromised source marked safe")
                if row.get("exposed_agents") != ["a1"]:
                    failures.append(f"{label}: compromised source is not a1")

    pairs: dict[tuple[str, str], dict[str, dict]] = defaultdict(dict)
    source_outputs: dict[tuple[str, str], set[str]] = defaultdict(set)
    for stem, rows in all_rows.items():
        for row in rows:
            key = (row["seed_id"], row["topology"])
            pairs[key][stem] = row
            native = [
                event for event in row.get("events", [])
                if str(event.get("provenance", "")).startswith("native_agentdojo")
            ]
            if len(native) == 1:
                source_outputs[(row["seed_id"], row["attack_condition"])].add(
                    native[0].get("output")
                )

    if len(pairs) != 44:
        failures.append(f"expected 44 paired keys, found {len(pairs)}")

    for key, rows in pairs.items():
        if set(rows) != set(FILES):
            failures.append(f"{key}: incomplete four-condition pair")
            continue
        for field in (
            "rng_seed",
            "canonical_evidence_sha256",
            "alignment_tier",
            "primary_strict",
            "clean_anchor_message_index",
            "attacked_anchor_message_index",
        ):
            if len({row.get(field) for row in rows.values()}) != 1:
                failures.append(f"{key}: {field} differs across conditions")

    for key, outputs in source_outputs.items():
        if len(outputs) != 1:
            failures.append(f"{key}: native source differs across topology/defense")

    print("=" * 80)
    print("EVIDENCE-ALIGNED FULL EXPERIMENT VALIDATION")
    print("=" * 80)
    print("files:", len(all_rows))
    print("rows:", sum(len(rows) for rows in all_rows.values()))
    print("paired seed-topology keys:", len(pairs))
    print("primary rows per condition: 36")
    print("semantic-sensitivity rows per condition: 8")
    if failures:
        print("status: FAILED")
        for failure in failures:
            print("-", failure)
        raise SystemExit(1)
    print("status: PASSED")
    print("All 176 rows form complete four-condition pairs.")
    print("Canonical clean evidence and RNG identifiers match within every pair.")
    print("Primary and semantic-alignment strata are preserved.")


if __name__ == "__main__":
    main()
