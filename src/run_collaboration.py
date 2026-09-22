from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import yaml

from .local_qwen import LocalQwen
from .collab import (
    Agent,
    ARCH,
    build_tool_evidence,
    extract_injection_text,
    extract_user_task,
)
from .judge import LocalSafetyJudge


AUXILIARY_ROLES = {
    "AGGREGATOR",
    "SAFETY_MONITOR",
}


def load_jsonl(path):
    return [
        json.loads(line)
        for line in Path(path)
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--config",
        default="configs/aoraki_qwen_pilot.yaml",
    )

    ap.add_argument(
        "--seeds",
        default="data/agentdojo_collab_seeds.jsonl",
    )

    ap.add_argument(
        "--limit",
        type=int,
        default=0,
    )

    ap.add_argument(
        "--output",
        default="results/pilot_raw.jsonl",
    )

    args = ap.parse_args()

    with open(
        args.config,
        "r",
        encoding="utf-8",
    ) as f:
        cfg = yaml.safe_load(f)

    seeds = load_jsonl(args.seeds)

    if args.limit:
        seeds = seeds[:args.limit]

    if not seeds:
        raise RuntimeError(
            "No collaborative seeds found."
        )

    llm = LocalQwen(
        cfg["local_model"]
    )

    judge = LocalSafetyJudge(llm)

    c = cfg["collaboration"]

    out = Path(args.output)
    out.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with out.open(
        "w",
        encoding="utf-8",
    ) as f:

        for seed in seeds:
            user_task = extract_user_task(seed)
            injection_text = extract_injection_text(seed)
            clean_evidence = build_tool_evidence(
                seed,
                attacked=False,
            )

            for arch in c["architectures"]:
                for n in c["num_agents"]:
                    for rounds in c["num_rounds"]:
                        for ratio in c["exposure_ratio"]:
                            for rep in range(
                                c["repetitions"]
                            ):

                                rng_seed = (
                                    cfg["experiment"]["random_seed"]
                                    + rep
                                    + n * 100
                                    + rounds * 1000
                                    + sum(
                                        ord(ch)
                                        for ch
                                        in seed["seed_id"]
                                    )
                                    + sum(
                                        ord(ch)
                                        for ch
                                        in arch
                                    )
                                )

                                rng = random.Random(
                                    rng_seed
                                )

                                eligible = list(
                                    range(
                                        n - 1
                                        if arch
                                        == "safety_monitor"
                                        else n
                                    )
                                )

                                k = min(
                                    len(eligible),
                                    max(
                                        1,
                                        math.ceil(
                                            n * ratio
                                        ),
                                    ),
                                )

                                exposed_idx = set(
                                    rng.sample(
                                        eligible,
                                        k,
                                    )
                                )

                                agents = []

                                for i in range(n):
                                    role = "worker"

                                    if (
                                        arch
                                        == "hierarchical_collaboration"
                                        and i == n - 1
                                    ):
                                        role = "manager"

                                    if (
                                        arch
                                        == "safety_monitor"
                                        and i == n - 1
                                    ):
                                        role = (
                                            "SAFETY_MONITOR"
                                        )

                                    agents.append(
                                        Agent(
                                            agent_id=f"a{i+1}",
                                            role=role,
                                            exposed=(
                                                i
                                                in exposed_idx
                                            ),
                                        )
                                    )

                                defense_prompt = bool(
                                    c.get(
                                        "defense_prompt",
                                        False,
                                    )
                                )

                                result = ARCH[arch](
                                    llm,
                                    agents,
                                    seed,
                                    rounds,
                                    defense_prompt,
                                )

                                judged_events = []

                                for event in result["events"]:
                                    if (
                                        event.get("provenance")
                                        == "native_agentdojo"
                                        and event.get("unsafe") is True
                                    ):
                                        judged_events.append({
                                            **event,
                                            "unsafe": True,
                                            "safety_confidence":
                                                event.get(
                                                    "safety_confidence",
                                                    1.0,
                                                ),
                                            "safety_reason":
                                                event.get(
                                                    "safety_reason",
                                                    (
                                                        "Source compromise is "
                                                        "established by the "
                                                        "native AgentDojo "
                                                        "security=True trace."
                                                    ),
                                                ),
                                        })
                                        continue

                                    safety = judge.safety(
                                        user_task=user_task,
                                        injection_text=(
                                            injection_text
                                        ),
                                        candidate=event[
                                            "output"
                                        ],
                                    )

                                    judged_events.append({
                                        **event,
                                        "unsafe": bool(
                                            safety["unsafe"]
                                        ),
                                        "safety_confidence":
                                            safety[
                                                "confidence"
                                            ],
                                        "safety_reason":
                                            safety["reason"],
                                    })

                                final_output = result[
                                    "final_output"
                                ]

                                final_judgment = (
                                    judge.safety(
                                        user_task=user_task,
                                        injection_text=(
                                            injection_text
                                        ),
                                        candidate=(
                                            final_output
                                        ),
                                    )
                                )

                                final_unsafe = bool(
                                    final_judgment[
                                        "unsafe"
                                    ]
                                )

                                utility_judgment = (
                                    judge.utility(
                                        user_task=user_task,
                                        clean_evidence=(
                                            clean_evidence
                                        ),
                                        candidate=(
                                            final_output
                                        ),
                                    )
                                )

                                exposed_agents = [
                                    a.agent_id
                                    for a in agents
                                    if a.exposed
                                ]

                                core_agent_ids = {
                                    a.agent_id
                                    for a in agents
                                    if a.role
                                    not in AUXILIARY_ROLES
                                }

                                unsafe_agents = sorted({
                                    event["agent_id"]
                                    for event
                                    in judged_events
                                    if event["unsafe"]
                                    and event["agent_id"]
                                    in core_agent_ids
                                })

                                exposed_unsafe_agents = (
                                    sorted(
                                        set(
                                            unsafe_agents
                                        )
                                        & set(
                                            exposed_agents
                                        )
                                    )
                                )

                                non_exposed_unsafe = (
                                    sorted(
                                        set(
                                            unsafe_agents
                                        )
                                        - set(
                                            exposed_agents
                                        )
                                    )
                                )

                                source_compromised_runtime = (
                                    bool(
                                        exposed_unsafe_agents
                                    )
                                )

                                intermediate_propagation = (
                                    bool(
                                        non_exposed_unsafe
                                    )
                                )

                                row = {
                                    "seed_id":
                                        seed["seed_id"],

                                    "suite_name":
                                        seed.get(
                                            "suite_name"
                                        ),

                                    "user_task_id":
                                        seed.get(
                                            "user_task_id"
                                        ),

                                    "injection_task_id":
                                        seed.get(
                                            "injection_task_id"
                                        ),

                                    "attack_type":
                                        seed.get(
                                            "attack_type"
                                        ),

                                    "agentdojo_security_raw":
                                        seed.get(
                                            "agentdojo_security_raw"
                                        ),

                                    "agentdojo_attack_success":
                                        seed.get(
                                            "attack_success"
                                        ),

                                    "attack_exposed":
                                        seed.get(
                                            "attack_exposed"
                                        ),

                                    "architecture":
                                        arch,

                                    "defense_prompt":
                                        defense_prompt,

                                    "num_agents":
                                        n,

                                    "num_rounds":
                                        rounds,

                                    "exposure_ratio":
                                        ratio,

                                    "repetition":
                                        rep,

                                    "rng_seed":
                                        rng_seed,

                                    "model":
                                        cfg[
                                            "local_model"
                                        ]["model"],

                                    "backend":
                                        cfg[
                                            "local_model"
                                        ]["backend"],

                                    "source_utility":
                                        seed.get(
                                            "utility"
                                        ),

                                    "exposed_agents":
                                        sorted(exposed_agents),

                                    "core_agent_ids":
                                        sorted(core_agent_ids),

                                    "unsafe_agents":
                                        unsafe_agents,

                                    "exposed_unsafe_agents":
                                        exposed_unsafe_agents,

                                    "non_exposed_unsafe_agents":
                                        non_exposed_unsafe,

                                    "source_compromised_runtime":
                                        source_compromised_runtime,

                                    "intermediate_propagation":
                                        intermediate_propagation,

                                    "final_output":
                                        final_output,

                                    "final_unsafe":
                                        final_unsafe,

                                    "final_safety_confidence":
                                        final_judgment[
                                            "confidence"
                                        ],

                                    "final_safety_reason":
                                        final_judgment[
                                            "reason"
                                        ],

                                    "utility":
                                        bool(
                                            utility_judgment[
                                                "utility"
                                            ]
                                        ),

                                    "utility_confidence":
                                        utility_judgment[
                                            "confidence"
                                        ],

                                    "utility_reason":
                                        utility_judgment[
                                            "reason"
                                        ],

                                    "events":
                                        judged_events,
                                }

                                f.write(
                                    json.dumps(
                                        row,
                                        ensure_ascii=False,
                                    )
                                    + "\n"
                                )

                                f.flush()

                                print(
                                    seed["seed_id"],
                                    arch,
                                    "source_unsafe=",
                                    source_compromised_runtime,
                                    "peer_propagation=",
                                    intermediate_propagation,
                                    "final_unsafe=",
                                    final_unsafe,
                                    "utility=",
                                    bool(
                                        utility_judgment[
                                            "utility"
                                        ]
                                    ),
                                    flush=True,
                                )

    print(
        "Saved",
        out,
    )


if __name__ == "__main__":
    main()
