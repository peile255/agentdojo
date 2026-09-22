from __future__ import annotations

import argparse
import csv
import difflib
import json
import re
from pathlib import Path
from typing import Any


INFORMATION_BLOCK = re.compile(
    r"<INFORMATION\b[^>]*>.*?</INFORMATION>",
    flags=re.IGNORECASE | re.DOTALL,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid JSON at {path}:{number}: {error}") from error
    return rows


def canonical_args(value: Any) -> str:
    return json.dumps(value or {}, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def normalize_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def strip_injection(value: Any) -> str:
    return INFORMATION_BLOCK.sub("", str(value or "")).strip()


def tool_key(item: dict[str, Any]) -> tuple[str, str]:
    return str(item.get("function") or ""), canonical_args(item.get("args"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--attacked-seeds",
        type=Path,
        default=Path("data/agentdojo_compromised_all.jsonl"),
    )
    parser.add_argument(
        "--clean-pairs",
        type=Path,
        default=Path("ccfa_v3/native_clean/agentdojo_native_clean_paired.jsonl"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("ccfa_v3/evidence_alignment"),
    )
    args = parser.parse_args()

    attacked = read_jsonl(args.attacked_seeds)
    clean_pairs = read_jsonl(args.clean_pairs)
    clean_by_id = {row["seed_id"]: row for row in clean_pairs}

    if len(clean_by_id) != len(clean_pairs):
        raise SystemExit("ERROR: duplicate seed IDs in clean-pair file")
    if {row["seed_id"] for row in attacked} != set(clean_by_id):
        raise SystemExit("ERROR: attacked and clean-pair seed IDs differ")

    output_rows = []
    structural_failures = []

    for seed in attacked:
        seed_id = seed["seed_id"]
        pair = clean_by_id[seed_id]
        clean = pair.get("native_clean") or {}
        attacked_outputs = seed.get("tool_outputs") or []
        clean_outputs = clean.get("tool_outputs") or []

        exposure = next(
            (
                item
                for item in attacked_outputs
                if INFORMATION_BLOCK.search(str(item.get("content") or ""))
            ),
            None,
        )

        if exposure is None:
            structural_failures.append(f"{seed_id}: no attacked exposure output")
            output_rows.append({
                "seed_id": seed_id,
                "suite_name": seed.get("suite_name"),
                "user_task_id": seed.get("user_task_id"),
                "injection_task_id": seed.get("injection_task_id"),
                "status": "no_attacked_exposure",
            })
            continue

        exposure_key = tool_key(exposure)
        candidates = [item for item in clean_outputs if tool_key(item) == exposure_key]
        clean_match = candidates[0] if candidates else None

        if clean_match is None:
            structural_failures.append(
                f"{seed_id}: no clean tool-output match for {exposure_key}"
            )
            output_rows.append({
                "seed_id": seed_id,
                "suite_name": seed.get("suite_name"),
                "user_task_id": seed.get("user_task_id"),
                "injection_task_id": seed.get("injection_task_id"),
                "status": "no_clean_tool_match",
                "exposure_function": exposure_key[0],
                "exposure_args": exposure_key[1],
                "attacked_exposure_message_index": exposure.get("message_index"),
            })
            continue

        attacked_benign = strip_injection(exposure.get("content"))
        clean_content = str(clean_match.get("content") or "")
        attacked_normalized = normalize_text(attacked_benign)
        clean_normalized = normalize_text(clean_content)
        exact = attacked_normalized == clean_normalized
        similarity = difflib.SequenceMatcher(
            None, attacked_normalized, clean_normalized
        ).ratio()

        output_rows.append({
            "seed_id": seed_id,
            "suite_name": seed.get("suite_name"),
            "user_task_id": seed.get("user_task_id"),
            "injection_task_id": seed.get("injection_task_id"),
            "status": "matched",
            "exposure_function": exposure_key[0],
            "exposure_args": exposure_key[1],
            "attacked_exposure_message_index": exposure.get("message_index"),
            "clean_match_message_index": clean_match.get("message_index"),
            "clean_match_count": len(candidates),
            "benign_content_exact": exact,
            "benign_content_similarity": round(similarity, 6),
            "attacked_benign_length": len(attacked_benign),
            "clean_content_length": len(clean_content),
            "attacked_benign_content": attacked_benign,
            "clean_matched_content": clean_content,
            "clean_source_file": clean.get("source_file"),
            "attacked_source_file": seed.get("source_file"),
        })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = args.output_dir / "native_evidence_alignment.jsonl"
    csv_path = args.output_dir / "native_evidence_alignment.csv"
    report_path = args.output_dir / "native_evidence_alignment_report.txt"

    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    fieldnames = sorted(set().union(*(row.keys() for row in output_rows)))
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    matched = [row for row in output_rows if row["status"] == "matched"]
    exact = [row for row in matched if row.get("benign_content_exact")]
    different = [row for row in matched if not row.get("benign_content_exact")]
    lines = [
        "NATIVE EVIDENCE ALIGNMENT AUDIT",
        "=" * 72,
        f"attacked seeds: {len(attacked)}",
        f"clean paired records: {len(clean_pairs)}",
        f"matched tool signatures: {len(matched)}",
        f"exact benign contents: {len(exact)}",
        f"different benign contents: {len(different)}",
        f"structural failures: {len(structural_failures)}",
        "",
        "PER-SEED RESULTS",
        "-" * 72,
    ]
    for row in output_rows:
        lines.append(
            f"{row['seed_id']} {row.get('suite_name')}/{row.get('user_task_id')} "
            f"status={row['status']} function={row.get('exposure_function')} "
            f"exact={row.get('benign_content_exact')} "
            f"similarity={row.get('benign_content_similarity')}"
        )
    if structural_failures:
        lines.extend(["", "STRUCTURAL FAILURES", "-" * 72, *structural_failures])
    lines.extend([
        "",
        "DECISION RULE",
        "-" * 72,
        "Proceed to matched-evidence runner construction only if structural failures = 0.",
        "Content differences are reported as experimental-state differences and must be",
        "handled by supplying canonical clean evidence to all non-source agents.",
    ])
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(report_path.read_text(encoding="utf-8"))
    print("Saved:", jsonl_path)
    print("Saved:", csv_path)
    print("Saved:", report_path)

    if structural_failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
