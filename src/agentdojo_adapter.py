from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _content_to_text(content: Any) -> str:
    if content is None:
        return ""

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                text = block.get("content")
                if text is not None:
                    parts.append(str(text))
            else:
                parts.append(str(block))
        return "\n".join(parts)

    return str(content)


def build_trace_text(messages: list[dict[str, Any]]) -> str:
    """Convert AgentDojo messages into a readable collaborative source trace."""
    chunks = []

    for i, msg in enumerate(messages):
        role = msg.get("role", "unknown")
        content = _content_to_text(msg.get("content"))

        chunks.append(
            f"[MESSAGE {i:02d}] ROLE={role.upper()}\n{content}".strip()
        )

        tool_calls = msg.get("tool_calls") or []
        for call in tool_calls:
            chunks.append(
                "[TOOL_CALL]\n"
                + json.dumps(call, ensure_ascii=False, sort_keys=True)
            )

        tool_call = msg.get("tool_call")
        if tool_call:
            chunks.append(
                "[TOOL_RESULT_FOR]\n"
                + json.dumps(tool_call, ensure_ascii=False, sort_keys=True)
            )

        if msg.get("error"):
            chunks.append(f"[MESSAGE_ERROR]\n{msg['error']}")

    return "\n\n".join(chunks)


def extract_tool_calls(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    calls = []

    for msg_index, msg in enumerate(messages):
        for call in msg.get("tool_calls") or []:
            calls.append({
                "message_index": msg_index,
                "function": call.get("function"),
                "args": call.get("args"),
                "id": call.get("id"),
                "placeholder_args": call.get("placeholder_args"),
            })

    return calls


def extract_tool_outputs(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    outputs = []

    for msg_index, msg in enumerate(messages):
        if msg.get("role") != "tool":
            continue

        tool_call = msg.get("tool_call") or {}

        outputs.append({
            "message_index": msg_index,
            "function": tool_call.get("function"),
            "args": tool_call.get("args"),
            "content": _content_to_text(msg.get("content")),
            "error": msg.get("error"),
        })

    return outputs


def extract_assistant_outputs(
    messages: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    outputs = []

    for msg_index, msg in enumerate(messages):
        if msg.get("role") != "assistant":
            continue

        outputs.append({
            "message_index": msg_index,
            "content": _content_to_text(msg.get("content")),
            "tool_calls": msg.get("tool_calls") or [],
        })

    return outputs


def extract_candidate_records(logdir: str) -> list[dict[str, Any]]:
    """
    Strict parser for AgentDojo 0.1.35 JSON result files.

    Only attacked user-task records are returned:
      - injection_task_id is not None
      - attack_type is not None

    Injection tasks executed as standalone user tasks under
    .../injection_task_X/none/none.json are intentionally excluded.
    """
    root = Path(logdir)

    if not root.exists():
        raise FileNotFoundError(f"AgentDojo log directory not found: {root}")

    records = []

    for path in sorted(root.rglob("*.json")):
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as exc:
            print(f"WARNING: skipping unreadable JSON {path}: {exc}")
            continue

        required = {
            "suite_name",
            "pipeline_name",
            "user_task_id",
            "messages",
            "utility",
            "security",
        }

        if not required.issubset(data):
            continue

        # Exclude AgentDojo's injection-task-as-user-task calibration runs.
        if data.get("injection_task_id") is None:
            continue

        if data.get("attack_type") is None:
            continue

        messages = data.get("messages") or []

        record = {
            "source_file": str(path),
            "suite_name": data.get("suite_name"),
            "pipeline_name": data.get("pipeline_name"),
            "user_task_id": data.get("user_task_id"),
            "injection_task_id": data.get("injection_task_id"),
            "attack_type": data.get("attack_type"),
            "benchmark_version": data.get("benchmark_version"),
            "agentdojo_package_version": data.get("agentdojo_package_version"),
            "evaluation_timestamp": data.get("evaluation_timestamp"),
            "duration": data.get("duration"),
            "error": data.get("error"),
            "utility": data.get("utility"),
            "security": data.get("security"),
            "agentdojo_security_raw": data.get("security"),
            "attack_exposed": (
                data.get("injection_task_id") is not None
                and data.get("attack_type") is not None
            ),
            "attack_success": data.get("security") is True,
            "source_compromised": data.get("security") is True,
            "injections": data.get("injections") or {},
            "messages": messages,
            "assistant_outputs": extract_assistant_outputs(messages),
            "tool_calls": extract_tool_calls(messages),
            "tool_outputs": extract_tool_outputs(messages),
            "source_trace": build_trace_text(messages),
        }

        records.append(record)

    return records
