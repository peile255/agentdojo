from __future__ import annotations

import hashlib
import json
from typing import Any

from . import collab


_ORIGINAL_EVIDENCE_FOR_AGENT = collab.evidence_for_agent
_PATCH_INSTALLED = False


def format_tool_outputs(items: list[dict[str, Any]]) -> str:
    chunks = []
    for item in items:
        function = str(item.get("function", "unknown"))
        args = item.get("args", {})
        content = str(item.get("content", "") or "")
        error = item.get("error")
        chunk = (
            f"TOOL: {function}\n"
            f"ARGS: {json.dumps(args, ensure_ascii=False)}\n"
            f"OUTPUT:\n{content}"
        )
        if error is not None:
            chunk += f"\nERROR: {error}"
        chunks.append(chunk)
    return "\n\n".join(chunks) if chunks else "<no tool evidence available>"


def prepare_aligned_seed(
    seed: dict[str, Any],
    alignment: dict[str, Any],
) -> dict[str, Any]:
    clean = seed.get("native_clean")
    if not isinstance(clean, dict):
        raise ValueError(f"Missing native_clean record for {seed.get('seed_id')}")

    attacked_exposures = [
        item for item in (seed.get("tool_outputs") or [])
        if "<INFORMATION" in str(item.get("content", "") or "").upper()
    ]
    if not attacked_exposures:
        raise ValueError(
            f"No attacked exposure output for {seed.get('seed_id')}"
        )
    attacked_anchor = int(attacked_exposures[0].get("message_index", -1))
    expected_attacked_anchor = int(alignment["attacked_anchor_message_index"])
    if attacked_anchor != expected_attacked_anchor:
        raise ValueError(
            f"Attacked anchor mismatch for {seed.get('seed_id')}: "
            f"manifest={expected_attacked_anchor}, trace={attacked_anchor}"
        )

    anchor = int(alignment["clean_anchor_message_index"])
    outputs = sorted(
        clean.get("tool_outputs") or [],
        key=lambda item: item.get("message_index", -1),
    )
    prefix = [
        item for item in outputs
        if int(item.get("message_index", -1)) <= anchor
    ]
    if not prefix or int(prefix[-1].get("message_index", -1)) != anchor:
        raise ValueError(
            f"Clean anchor {anchor} not found for {seed.get('seed_id')}"
        )

    evidence = format_tool_outputs(prefix)
    if "<INFORMATION" in evidence.upper():
        raise ValueError(
            f"Injection marker found in canonical evidence for {seed.get('seed_id')}"
        )

    seed["_evidence_alignment"] = alignment
    seed["_canonical_clean_evidence"] = evidence
    seed["_canonical_clean_evidence_sha256"] = hashlib.sha256(
        evidence.encode("utf-8")
    ).hexdigest()
    return seed


def aligned_evidence_for_agent(seed: dict[str, Any], agent: Any) -> str:
    if not agent.exposed:
        evidence = seed.get("_canonical_clean_evidence")
        if evidence:
            return str(evidence)
    return _ORIGINAL_EVIDENCE_FOR_AGENT(seed, agent)


def install_alignment_patch() -> None:
    global _PATCH_INSTALLED
    if _PATCH_INSTALLED:
        return
    collab.evidence_for_agent = aligned_evidence_for_agent
    _PATCH_INSTALLED = True
