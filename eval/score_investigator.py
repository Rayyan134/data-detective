"""Score the investigator against the fixed evaluation question set.

This runs the *planning* stage of the investigator against every question in
``eval/questions.json`` and compares the chosen tool and parameters with the
hand-checked expectations. It deliberately stops after planning (it does not ask
for the narrative interpretation) so the score measures tool selection, which is
the part the exam set was designed to test.

It calls the live Gemini API, so it needs ``GEMINI_API_KEY`` in the environment
or a local ``.env`` file. Run from the project root:

    .\\.venv\\Scripts\\python.exe eval/score_investigator.py

The same checks can be run offline with fake responses through the unit tests in
``tests/test_investigator.py``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.investigator import investigate  # noqa: E402
from src.llm_client import GeminiClient, LLMError, load_environment  # noqa: E402
from src.profiler import load_csv, profile_dataset  # noqa: E402
from src.tools import TOOL_REGISTRY  # noqa: E402

QUESTIONS_PATH = PROJECT_ROOT / "eval" / "questions.json"
DEFAULT_BASELINE_PATH = PROJECT_ROOT / "eval" / "baseline.md"
PAUSE_SECONDS = 12.0  # the free tier has a low requests-per-minute limit


def _params_match(expected_tool: str, expected: dict, chosen: dict) -> bool:
    """True when every expected parameter matches the chosen one.

    A parameter the plan omits is compared against the tool's declared default,
    since leaving an optional parameter out is a correct way to request it.
    """
    declared = TOOL_REGISTRY[expected_tool]["parameters"]
    for key, value in expected.items():
        if key in chosen:
            actual = chosen[key]
        else:
            spec = declared.get(key, {})
            if "default" not in spec:
                return False
            actual = spec["default"]
        if isinstance(value, list):
            if list(actual) != list(value):
                return False
        elif actual != value:
            return False
    return True


def _score_question(question: dict, df, profile: dict, client) -> dict:
    expected_tool = question["expected_tool"]
    expected_params = question.get("expected_params") or {}

    started = time.perf_counter()
    report = investigate(
        question["question"], df, profile=profile, client=client, interpret=False
    )
    elapsed = time.perf_counter() - started

    record = {
        "id": question["id"],
        "question": question["question"],
        "expected_tool": expected_tool,
        "chosen_tool": None,
        "chosen_params": {},
        "status": report.status,
        "verdict": "fail",
        "note": "",
        "seconds": round(elapsed, 1),
    }

    if report.error:
        record["note"] = report.error
        return record

    if expected_tool is None:
        # A refusal question: the plan must be explicitly unsupported.
        if report.status == "unsupported":
            record["verdict"] = "pass"
            record["chosen_tool"] = "(unsupported)"
        else:
            record["note"] = "expected an explicit refusal"
        return record

    if report.plan is None or not report.plan.is_supported:
        record["note"] = "no tool selected"
        return record

    chosen_step = report.plan.steps[0]
    record["chosen_tool"] = chosen_step.tool
    record["chosen_params"] = chosen_step.params

    if chosen_step.tool != expected_tool:
        record["note"] = f"expected {expected_tool}"
        return record
    if not _params_match(expected_tool, expected_params, chosen_step.params):
        record["note"] = f"parameters differ from {expected_params}"
        return record

    record["verdict"] = "pass"
    return record


def _write_baseline(records: list[dict], model: str, dataset: str, path: Path) -> None:
    passed = sum(1 for record in records if record["verdict"] == "pass")
    lines = [
        "# Investigator baseline",
        "",
        f"- Date: {time.strftime('%Y-%m-%d')}",
        f"- Model: `{model}`",
        f"- Dataset: `{dataset}`",
        "- Mode: planning only (tool selection and parameters scored)",
        f"- Score: **{passed} / {len(records)}**",
        "",
        "| ID | Expected tool | Chosen tool | Verdict | Seconds | Note |",
        "|---|---|---|---|---|---|",
    ]
    for record in records:
        note = record["note"].replace("|", "\\|")
        lines.append(
            f"| {record['id']} | {record['expected_tool'] or '(refuse)'} "
            f"| {record['chosen_tool'] or '(none)'} | {record['verdict']} "
            f"| {record['seconds']} | {note} |"
        )
    lines.append("")
    lines.append("## Chosen parameters")
    lines.append("")
    for record in records:
        lines.append(
            f"- **{record['id']}** `{record['chosen_tool']}` "
            f"{json.dumps(record['chosen_params'], sort_keys=True)}"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default=None,
        help="Override the model (defaults to GEMINI_MODEL or the client default).",
    )
    parser.add_argument(
        "--out",
        default=str(DEFAULT_BASELINE_PATH),
        help="Where to write the baseline report.",
    )
    args = parser.parse_args(argv)

    load_environment()
    exam = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    dataset_path = exam["dataset"]
    df = load_csv(PROJECT_ROOT / dataset_path)
    profile = profile_dataset(df)

    try:
        client = GeminiClient(model=args.model)
    except LLMError as exc:
        print(f"Cannot reach the model: {exc}")
        return 1

    print(f"Scoring {len(exam['questions'])} questions on {dataset_path}")
    print(f"Model: {client.model}")
    print()

    records: list[dict] = []
    for index, question in enumerate(exam["questions"]):
        if index:
            time.sleep(PAUSE_SECONDS)
        record = _score_question(question, df, profile, client)
        records.append(record)
        mark = "PASS" if record["verdict"] == "pass" else "FAIL"
        print(
            f"{record['id']}  {mark}  chosen={record['chosen_tool']}  "
            f"({record['seconds']}s)  {record['note']}"
        )

    passed = sum(1 for record in records if record["verdict"] == "pass")
    print()
    print(f"Score: {passed} / {len(records)}")

    out_path = Path(args.out)
    _write_baseline(records, client.model, dataset_path, out_path)
    print(f"Baseline written to {out_path}")
    return 0 if passed == len(records) else 2


if __name__ == "__main__":
    raise SystemExit(main())
