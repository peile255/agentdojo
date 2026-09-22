from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


FILES = {
    "clean_standard": ("clean", False),
    "clean_defended": ("clean", True),
    "attack_standard": ("compromised", False),
    "attack_defended": ("compromised", True),
}


def load(directory: Path) -> list[dict]:
    rows = []
    for stem, (condition, defended) in FILES.items():
        path = directory / f"{stem}.jsonl"
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row["attack_condition"] != condition:
                raise ValueError(f"Condition mismatch in {path}")
            if row["defense_prompt"] is not defended:
                raise ValueError(f"Defense mismatch in {path}")
            rows.append(row)
    return rows


def strata(row: dict) -> tuple[str, ...]:
    return (
        "all",
        "primary_strict" if row["primary_strict"] else "semantic_sensitivity",
    )


def exact_mcnemar(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(0, min(b, c) + 1)) / (2 ** n)
    return min(1.0, 2.0 * tail)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = load(args.directory)

    summary_groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        for stratum in strata(row):
            summary_groups[
                (
                    row["attack_condition"],
                    row["defense_prompt"],
                    row["topology"],
                    stratum,
                )
            ].append(row)

    summary = []
    for (condition, defended, topology, stratum), group in sorted(
        summary_groups.items()
    ):
        n = len(group)
        safe_n = sum(not row["final_unsafe"] for row in group)
        utility_n = sum(row["utility"] for row in group)
        joint_n = sum(
            row["utility"] and not row["final_unsafe"] for row in group
        )
        propagation_n = sum(bool(row["intermediate_propagation"]) for row in group)
        summary.append({
            "attack_condition": condition,
            "defense_prompt": defended,
            "topology": topology,
            "stratum": stratum,
            "n": n,
            "safe_n": safe_n,
            "safe_rate": round(safe_n / n, 6),
            "utility_n": utility_n,
            "utility_rate": round(utility_n / n, 6),
            "safe_and_useful_n": joint_n,
            "safe_and_useful_rate": round(joint_n / n, 6),
            "intermediate_propagation_n": propagation_n,
            "intermediate_propagation_rate": round(propagation_n / n, 6),
        })

    indexed = {
        (
            row["seed_id"],
            row["topology"],
            row["attack_condition"],
            row["defense_prompt"],
        ): row
        for row in rows
    }

    transitions = []

    def add_transition(
        comparison: str,
        fixed_name: str,
        fixed_value: object,
        topology: str,
        stratum: str,
        outcome: str,
        pairs: list[tuple[dict, dict]],
    ) -> None:
        counts = Counter(
            (bool(left[outcome]), bool(right[outcome]))
            for left, right in pairs
        )
        b = counts[(True, False)]
        c = counts[(False, True)]
        transitions.append({
            "comparison": comparison,
            "fixed_factor": fixed_name,
            "fixed_value": fixed_value,
            "topology": topology,
            "stratum": stratum,
            "outcome": outcome,
            "false_to_false": counts[(False, False)],
            "false_to_true": c,
            "true_to_false": b,
            "true_to_true": counts[(True, True)],
            "discordant_n": b + c,
            "exact_mcnemar_p": round(exact_mcnemar(b, c), 8),
        })

    seed_ids = sorted({row["seed_id"] for row in rows})
    topologies = sorted({row["topology"] for row in rows})
    for topology in topologies:
        for condition in ("clean", "compromised"):
            all_pairs = [
                (
                    indexed[(seed, topology, condition, False)],
                    indexed[(seed, topology, condition, True)],
                )
                for seed in seed_ids
            ]
            for stratum in ("all", "primary_strict", "semantic_sensitivity"):
                pairs = [
                    pair for pair in all_pairs
                    if stratum == "all"
                    or (
                        pair[0]["primary_strict"]
                        == (stratum == "primary_strict")
                    )
                ]
                for outcome in ("utility", "final_unsafe", "intermediate_propagation"):
                    add_transition(
                        "defense_effect_standard_to_defended",
                        "attack_condition",
                        condition,
                        topology,
                        stratum,
                        outcome,
                        pairs,
                    )

        for defended in (False, True):
            all_pairs = [
                (
                    indexed[(seed, topology, "clean", defended)],
                    indexed[(seed, topology, "compromised", defended)],
                )
                for seed in seed_ids
            ]
            for stratum in ("all", "primary_strict", "semantic_sensitivity"):
                pairs = [
                    pair for pair in all_pairs
                    if stratum == "all"
                    or (
                        pair[0]["primary_strict"]
                        == (stratum == "primary_strict")
                    )
                ]
                for outcome in ("utility", "final_unsafe", "intermediate_propagation"):
                    add_transition(
                        "attack_effect_clean_to_compromised",
                        "defense_prompt",
                        defended,
                        topology,
                        stratum,
                        outcome,
                        pairs,
                    )

    summary_path = args.output_dir / "aligned_full_summary.csv"
    transition_path = args.output_dir / "aligned_full_transitions.csv"
    write_csv(summary_path, summary)
    write_csv(transition_path, transitions)

    print("=" * 80)
    print("EVIDENCE-ALIGNED FULL ANALYSIS")
    print("=" * 80)
    print("input rows:", len(rows))
    print("summary rows:", len(summary))
    print("transition rows:", len(transitions))
    print("saved:", summary_path)
    print("saved:", transition_path)
    print("\nALL-SEED SUMMARY")
    for row in summary:
        if row["stratum"] != "all":
            continue
        print(
            row["attack_condition"],
            "defended=" + str(row["defense_prompt"]),
            row["topology"],
            f"safe={row['safe_n']}/{row['n']}",
            f"utility={row['utility_n']}/{row['n']}",
            f"propagation={row['intermediate_propagation_n']}/{row['n']}",
        )


if __name__ == "__main__":
    main()
