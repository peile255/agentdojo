from __future__ import annotations

import argparse
import json
import re
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
            if not SHA256.match(str(row.get("canonical_evidence_sha256", ""))):
                failures.append(f"{label}: canonical evidence hash missing or invalid")
            if not row.get("alignment_tier"):
                failures.append(f"{label}: alignment tier missing")
            if not isinstance(row.get("primary_strict"), bool):
                failures.append(f"{label}: primary_strict is not Boolean")
            if not isinstance(row.get("clean_anchor_message_index"), int):
                failures.append(f"{label}: clean anchor missing")

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
                if native[0].get("clean_anchor_message_index") != row.get(
                    "clean_anchor_message_index"
                ):
                    failures.append(f"{label}: clean source anchor mismatch")
            else:
                if native[0].get("provenance") != "native_agentdojo":
                    failures.append(f"{label}: wrong attack provenance")
                if native[0].get("unsafe") is not True:
                    failures.append(f"{label}: compromised source marked safe")
                if row.get("exposed_agents") != ["a1"]:
                    failures.append(f"{label}: compromised source is not a1")

    paired: dict[tuple[str, str], dict[str, dict]] = {}
    source_outputs: dict[tuple[str, str], set[str]] = {}
    for stem, rows in all_rows.items():
        for row in rows:
            key = (row.get("seed_id"), row.get("topology"))
            paired.setdefault(key, {})[stem] = row
            native = [
                event for event in row.get("events", [])
                if str(event.get("provenance", "")).startswith("native_agentdojo")
            ]
            if len(native) == 1:
                source_key = (row.get("seed_id"), row.get("attack_condition"))
                source_outputs.setdefault(source_key, set()).add(native[0].get("output"))

    for key, rows in paired.items():
        if set(rows) != set(FILES):
            failures.append(f"{key}: incomplete four-condition pair")
            continue
        if len({row.get("rng_seed") for row in rows.values()}) != 1:
            failures.append(f"{key}: RNG seed mismatch")
        if len({row.get("canonical_evidence_sha256") for row in rows.values()}) != 1:
            failures.append(f"{key}: canonical evidence differs across conditions")
        if len({row.get("alignment_tier") for row in rows.values()}) != 1:
            failures.append(f"{key}: alignment tier differs across conditions")
        if len({row.get("clean_anchor_message_index") for row in rows.values()}) != 1:
            failures.append(f"{key}: clean anchor differs across conditions")

    for key, outputs in source_outputs.items():
        if len(outputs) != 1:
            failures.append(f"{key}: native source differs across topology/defense")

    print("=" * 76)
    print("EVIDENCE-ALIGNED FOUR-CONDITION SMOKE VALIDATION")
    print("=" * 76)
    print("files:", len(all_rows))
    print("rows:", sum(len(rows) for rows in all_rows.values()))
    print("paired task-topology keys:", len(paired))
    if failures:
        print("status: FAILED")
        for failure in failures:
            print("-", failure)
        raise SystemExit(1)
    print("status: PASSED")
    print("Canonical clean evidence is identical across all four conditions.")
    print("Clean a1 behavior begins after the exposure-aligned clean anchor.")
    print("Native a1 outputs are invariant across topology and defense within condition.")
    print("RNG seeds are paired across all four conditions.")


if __name__ == "__main__":
    main()
