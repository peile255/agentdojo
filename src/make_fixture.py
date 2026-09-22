from pathlib import Path
import json

def main():
    out = Path("agentdojo_runs/mock_fixture")
    out.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "task_id": "fixture_user_0",
            "injection_task_id": "fixture_injection_0",
            "security_result": False,
            "utility_result": True,
            "trace": [
                {"role":"user","content":"legitimate task"},
                {"role":"tool","content":"untrusted injected instruction"},
                {"role":"assistant","content":"source agent followed injection"}
            ]
        }
    ]
    (out/"fixture.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print("Fixture written")

if __name__ == "__main__":
    main()
