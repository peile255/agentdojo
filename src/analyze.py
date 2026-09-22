from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .metrics import (
    consensus_robustness,
    fsu,
    recovery,
    risk_propagation_rate_from_all_agents,
    unsafe_consensus_rate,
)


def get_core_agent_ids(row):
    """
    Return propagation-eligible agent IDs.

    New result files store core_agent_ids explicitly.
    For older result files, reconstruct them from events while
    excluding auxiliary AGGREGATOR and SAFETY_MONITOR roles.
    """
    stored = row.get("core_agent_ids")

    if isinstance(stored, list) and stored:
        return sorted(set(stored))

    ids = set()

    for event in row.get("events", []) or []:
        agent_id = event.get("agent_id")
        role = event.get("role")

        if not agent_id:
            continue

        if role in {
            "AGGREGATOR",
            "SAFETY_MONITOR",
        }:
            continue

        if str(agent_id).startswith("a"):
            ids.add(str(agent_id))

    if ids:
        return sorted(ids)

    return [
        f"a{i + 1}"
        for i in range(int(row["num_agents"]))
    ]


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--input",
        default="results/pilot_raw.jsonl"
    )

    ap.add_argument(
        "--output",
        default="results/pilot_summary.csv"
    )

    args = ap.parse_args()

    rows = [
        json.loads(x)
        for x in Path(args.input)
        .read_text(encoding="utf-8")
        .splitlines()
        if x.strip()
    ]

    if not rows:
        raise RuntimeError("No experiment results found.")

    df = pd.DataFrame(rows)

    df["Safety"] = 1.0 - df["final_unsafe"].astype(float)

    df["SSR"] = (
        df["source_compromised_runtime"]
        .astype(float)
    )

    df["RPR"] = df.apply(
        lambda r: risk_propagation_rate_from_all_agents(
            get_core_agent_ids(r),
            r["exposed_agents"],
            r["unsafe_agents"],
        ),
        axis=1,
    )

    df["RPR_cond"] = df["RPR"]

    df["CR"] = df["final_unsafe"].apply(
        consensus_robustness
    )

    df["UCR"] = df["final_unsafe"].apply(
        unsafe_consensus_rate
    )

    df["REC_case"] = df.apply(
        lambda r: recovery(
            r["intermediate_propagation"],
            r["final_unsafe"],
        ),
        axis=1,
    )

    df["F_SU"] = [
        fsu(s, float(u))
        for s, u in zip(
            df["Safety"],
            df["utility"]
        )
    ]

    keys = [
        "architecture",
        "defense_prompt",
        "num_agents",
        "num_rounds",
        "exposure_ratio",
        "model",
    ]

    summary = (
        df.groupby(keys, as_index=False)
        .agg(
            Utility=("utility", "mean"),
            Safety=("Safety", "mean"),
            SSR=("SSR", "mean"),
            RPR=("RPR", "mean"),
            RPR_cond=("RPR_cond", "mean"),
            CR=("CR", "mean"),
            UCR=("UCR", "mean"),
            REC=("REC_case", "mean"),
            F_SU=("F_SU", "mean"),
            N=("seed_id", "count"),
        )
    )

    Path(args.output).parent.mkdir(
        parents=True,
        exist_ok=True
    )

    summary.to_csv(
        args.output,
        index=False
    )

    print(summary.to_string(index=False))
    print("\nSaved", args.output)
    print(
        "\nNOTE: TSD is intentionally omitted until "
        "matched benign controls are available."
    )


if __name__ == "__main__":
    main()
