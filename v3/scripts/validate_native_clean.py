from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                value = block.get("content")
                if value is not None:
                    parts.append(str(value))
            else:
                parts.append(str(block))
        return "\n".join(parts)
    return str(content)


def build_trace_text(messages: list[dict[str, Any]]) -> str:
    chunks = []
    for index, message in enumerate(messages):
        role = message.get("role", "unknown")
        content = content_to_text(message.get("content"))
        chunks.append(
            f"[MESSAGE {index:02d}] ROLE={str(role).upper()}\n{content}".strip()
        )
        for call in message.get("tool_calls") or []:
            chunks.append(
                "[TOOL_CALL]\n"
                + json.dumps(call, ensure_ascii=False, sort_keys=True)
            )
        if message.get("tool_call"):
            chunks.append(
                "[TOOL_RESULT_FOR]\n"
                + json.dumps(
                    message["tool_call"], ensure_ascii=False, sort_keys=True
                )
            )
        if message.get("error"):
            chunks.append(f"[MESSAGE_ERROR]\n{message['error']}")
    return "\n\n".join(chunks)


def extract_assistant_outputs(messages: list[dict[str, Any]]) -> list[dict]:
    return [
        {
            "message_index": index,
            "content": content_to_text(message.get("content")),
            "tool_calls": message.get("tool_calls") or [],
        }
        for index, message in enumerate(messages)
        if message.get("role") == "assistant"
    ]


def extract_tool_calls(messages: list[dict[str, Any]]) -> list[dict]:
    calls = []
    for index, message in enumerate(messages):
        for call in message.get("tool_calls") or []:
            calls.append(
                {
                    "message_index": index,
                    "function": call.get("function"),
                    "args": call.get("args"),
                    "id": call.get("id"),
                    "placeholder_args": call.get("placeholder_args"),
                }
            )
    return calls


def extract_tool_outputs(messages: list[dict[str, Any]]) -> list[dict]:
    outputs = []
    for index, message in enumerate(messages):
        if message.get("role") != "tool":
            continue
        call = message.get("tool_call") or {}
        outputs.append(
            {
                "message_index": index,
                "function": call.get("function"),
                "args": call.get("args"),
                "content": content_to_text(message.get("content")),
                "error": message.get("error"),
            }
        )
    return outputs


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
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


def read_manifest(path: Path) -> list[tuple[str, str]]:
    tasks = []
    seen = set()
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) != 2:
            raise ValueError(
                f"Expected two TSV fields at {path}:{line_number}"
            )
        key = (parts[0], parts[1])
        if key in seen:
            raise ValueError(f"Duplicate manifest task: {key}")
        seen.add(key)
        tasks.append(key)
    return tasks


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate and pair native clean AgentDojo source traces."
    )
    parser.add_argument(
        "--logdir",
        type=Path,
        default=Path("agentdojo_runs/ccfa_native_clean"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("ccfa_v3/native_clean/native_clean_manifest.tsv"),
    )
    parser.add_argument(
        "--seeds",
        type=Path,
        default=Path("data/agentdojo_compromised_all.jsonl"),
    )
    parser.add_argument(
        "--unique-output",
        type=Path,
        default=Path(
            "ccfa_v3/native_clean/agentdojo_native_clean_unique.jsonl"
        ),
    )
    parser.add_argument(
        "--paired-output",
        type=Path,
        default=Path(
            "ccfa_v3/native_clean/agentdojo_native_clean_paired.jsonl"
        ),
    )
    parser.add_argument(
        "--sha256-output",
        type=Path,
        default=Path("ccfa_v3/native_clean/native_clean_sha256.txt"),
    )
    args = parser.parse_args()

    tasks = read_manifest(args.manifest)
    seeds = read_jsonl(args.seeds)
    failures = []
    records: dict[tuple[str, str], dict] = {}
    expected_paths = set()

    required = {
        "suite_name",
        "pipeline_name",
        "user_task_id",
        "injection_task_id",
        "attack_type",
        "injections",
        "messages",
        "utility",
        "security",
        "error",
    }

    for suite, user_task in tasks:
        path = (
            args.logdir
            / "local"
            / suite
            / user_task
            / "none"
            / "none.json"
        )
        expected_paths.add(path.resolve())
        if not path.exists():
            failures.append(f"missing trace: {path}")
            continue

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as error:
            failures.append(f"unreadable trace {path}: {error}")
            continue

        missing = required - set(data)
        if missing:
            failures.append(f"{path}: missing fields {sorted(missing)}")
            continue

        checks = [
            (data.get("suite_name") == suite, "suite mismatch"),
            (data.get("user_task_id") == user_task, "user task mismatch"),
            (data.get("pipeline_name") == "local", "pipeline is not local"),
            (data.get("injection_task_id") is None, "injection task is not null"),
            (data.get("attack_type") is None, "attack type is not null"),
            (data.get("injections") in ({}, None), "injections are non-empty"),
            (data.get("security") is True, "clean security is not true"),
            (isinstance(data.get("utility"), bool), "utility is not boolean"),
            (data.get("error") in (None, ""), "trace contains an error"),
            (isinstance(data.get("messages"), list), "messages is not a list"),
            (bool(data.get("messages")), "messages is empty"),
        ]
        for passed, reason in checks:
            if not passed:
                failures.append(f"{path}: {reason}")

        messages = data.get("messages") or []
        if "<INFORMATION>" in build_trace_text(messages):
            failures.append(f"{path}: contains an injected INFORMATION marker")

        records[(suite, user_task)] = {
            "source_file": str(path),
            "source_sha256": sha256(path),
            "suite_name": suite,
            "pipeline_name": data.get("pipeline_name"),
            "user_task_id": user_task,
            "injection_task_id": None,
            "attack_type": None,
            "benchmark_version": data.get("benchmark_version"),
            "agentdojo_package_version": data.get("agentdojo_package_version"),
            "evaluation_timestamp": data.get("evaluation_timestamp"),
            "duration": data.get("duration"),
            "error": data.get("error"),
            "utility": data.get("utility"),
            "security": data.get("security"),
            "attack_exposed": False,
            "source_compromised": False,
            "injections": data.get("injections") or {},
            "messages": messages,
            "assistant_outputs": extract_assistant_outputs(messages),
            "tool_calls": extract_tool_calls(messages),
            "tool_outputs": extract_tool_outputs(messages),
            "source_trace": build_trace_text(messages),
        }

    actual_paths = {
        path.resolve() for path in args.logdir.rglob("none/none.json")
    }
    for path in sorted(actual_paths - expected_paths):
        failures.append(f"unexpected clean trace: {path}")

    seed_ids = set()
    paired_rows = []
    for index, seed in enumerate(seeds, start=1):
        seed_id = seed.get("seed_id")
        if not seed_id:
            failures.append(f"seed row {index}: missing seed_id")
            continue
        if seed_id in seed_ids:
            failures.append(f"duplicate seed_id: {seed_id}")
            continue
        seed_ids.add(seed_id)

        key = (seed.get("suite_name"), seed.get("user_task_id"))
        clean = records.get(key)
        if clean is None:
            failures.append(f"{seed_id}: no clean trace for {key}")
            continue

        paired_rows.append(
            {
                "seed_id": seed_id,
                "suite_name": seed.get("suite_name"),
                "user_task_id": seed.get("user_task_id"),
                "injection_task_id": seed.get("injection_task_id"),
                "compromised_source_file": seed.get("source_file"),
                "native_clean": clean,
            }
        )

    print("=" * 72)
    print("NATIVE CLEAN AGENTDOJO VALIDATION")
    print("=" * 72)
    print(f"manifest tasks: {len(tasks)}")
    print(f"unique clean traces: {len(records)}")
    print(f"paired compromised seeds: {len(paired_rows)}")
    print(
        "clean utility: "
        f"{sum(row['utility'] is True for row in records.values())}/"
        f"{len(records)}"
    )

    if failures:
        print("status: FAILED")
        for failure in failures:
            print(f"ERROR: {failure}")
        raise SystemExit(1)

    args.unique_output.parent.mkdir(parents=True, exist_ok=True)
    args.paired_output.parent.mkdir(parents=True, exist_ok=True)
    with args.unique_output.open("w", encoding="utf-8") as handle:
        for key in tasks:
            handle.write(json.dumps(records[key], ensure_ascii=False) + "\n")
    with args.paired_output.open("w", encoding="utf-8") as handle:
        for row in paired_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    hash_paths = sorted(expected_paths) + [
        args.manifest.resolve(),
        args.seeds.resolve(),
        args.unique_output.resolve(),
        args.paired_output.resolve(),
    ]
    args.sha256_output.parent.mkdir(parents=True, exist_ok=True)
    with args.sha256_output.open("w", encoding="utf-8") as handle:
        for path in hash_paths:
            handle.write(f"{sha256(path)}  {path}\n")

    print("status: PASSED")
    print(f"unique output: {args.unique_output}")
    print(f"paired output: {args.paired_output}")
    print(f"sha256 output: {args.sha256_output}")


if __name__ == "__main__":
    main()
