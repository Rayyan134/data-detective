"""Score the investigator against the fixed evaluation question set.

This is the Day 4 evaluation runner. It drives the *planning* stage of the
investigator for every question in ``eval/questions.json`` and scores four
independent things, all offline from the run's own computed findings:

* tool accuracy       -- did the plan choose the expected tool?
* parameter accuracy  -- did it pass the expected arguments?
* refusal accuracy    -- did it decline the questions no tool can answer?
* answer accuracy     -- does the computed evidence actually contain the answer?

It also records latency and token usage per question through
``src.evaluation.RecordingClient``.

By default it stops after planning (one model call per question), which is
cheap and still lets answer accuracy be measured, because the tools are
deterministic and still run. Pass ``--interpret N`` to additionally run the
narrative stage on the first N questions, which is how the faithfulness metric
(no unverified numbers in the answer) is produced.

It calls the live Gemini API, so it needs ``GEMINI_API_KEY`` in the environment
or a local ``.env`` file. Run from the project root:

    .\\.venv\\Scripts\\python.exe eval/score_investigator.py

The same checks can be run offline through ``tests/test_evaluation.py``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation import (  # noqa: E402
    QuestionScore,
    RecordingClient,
    aggregate,
    evaluate_question,
    load_questions,
)
from src.investigator import investigate  # noqa: E402
from src.llm_client import GeminiClient, LLMError, load_environment  # noqa: E402
from src.profiler import load_csv, profile_dataset  # noqa: E402

QUESTIONS_PATH = PROJECT_ROOT / "eval" / "questions.json"
DEFAULT_BASELINE_PATH = PROJECT_ROOT / "eval" / "baseline.md"
PAUSE_SECONDS = 12.0  # the free tier has a low requests-per-minute limit

PRICE_ENV_INPUT = "GEMINI_PRICE_INPUT_PER_MTOK"
PRICE_ENV_OUTPUT = "GEMINI_PRICE_OUTPUT_PER_MTOK"


def _resolve_price(name: str, explicit: float | None) -> float | None:
    """An explicit rate, else an environment rate, else None (no estimate)."""
    if explicit is not None:
        return float(explicit)
    raw = os.getenv(name, "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _note_for(score: QuestionScore) -> str:
    """A short reason the question did or did not pass."""
    if score.error:
        return score.error
    if score.expected_tool is None:
        return "" if score.refusal_ok else "expected an explicit refusal"
    if score.chosen_tool is None:
        return "no tool selected"
    if not score.tool_ok:
        return f"expected {score.expected_tool}"
    if not score.params_ok:
        return "parameters differ from the expected plan"
    if not score.answer.ok:
        return f"answer check failed: {score.answer.detail}"
    if score.repairs:
        return f"passed after normalising {len(score.repairs)} parameter(s)"
    return ""


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.0f}%"


def _is_rate_limit(message: str | None) -> bool:
    """True when a report error is a quota/rate limit rather than a real miss.

    The investigator reports an API failure as a normal ``error`` string, so the
    scorer has to recognise the rate-limit case to stop instead of recording the
    rest of the questions as failures.
    """
    if not message:
        return False
    lowered = message.lower()
    return "rate limit" in lowered or "429" in lowered or "quota" in lowered


def _run_question(question: dict, df, profile, client, *, interpret: bool) -> QuestionScore:
    """Plan (and optionally interpret) one question, measuring the call."""
    recording = RecordingClient(client)
    report = investigate(
        question["question"],
        df,
        profile=profile,
        client=recording,
        interpret=interpret,
    )
    return evaluate_question(
        question,
        report,
        latency_s=recording.total_latency_s,
        usage=recording.usage(),
    )


def _write_report(
    records: list[QuestionScore],
    metrics,
    *,
    model: str,
    dataset: str,
    interpret: int,
    faithful: list[QuestionScore],
    skipped: list[str],
    path: Path,
) -> None:
    lines = [
        "# Investigator evaluation",
        "",
        f"- Date: {time.strftime('%Y-%m-%d')}",
        f"- Model: `{model}`",
        f"- Dataset: `{dataset}`",
        "- Mode: planning only"
        + (f" (+ interpretation on the first {interpret} questions)" if interpret else ""),
    ]
    if skipped:
        lines.append(
            f"- **Partial run:** stopped early on a rate limit; "
            f"{len(records)} of {len(records) + len(skipped)} questions scored."
        )
    lines += [
        "",
        "## Metrics",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Questions | {metrics.n} |",
        f"| Plan accuracy (tool + parameters, or correct refusal) | {_pct(metrics.plan_accuracy)} |",
        f"| Tool accuracy | {_pct(metrics.tool_accuracy)} |",
        f"| Parameter accuracy | {_pct(metrics.param_accuracy)} |",
        f"| Refusal accuracy | {_pct(metrics.refusal_accuracy)} |",
        f"| Answer accuracy (evidence contains the expected value) | {_pct(metrics.answer_accuracy)} |",
        f"| Faithfulness (no unverified numbers) | {_pct(metrics.faithfulness)} |",
        f"| Normalised parameters (plans that needed a repair) | {metrics.repaired} |",
        "",
        "## Latency and tokens",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Latency mean | {metrics.latency_mean:.2f} s |",
        f"| Latency p50 | {metrics.latency_p50:.2f} s |",
        f"| Latency p95 | {metrics.latency_p95:.2f} s |",
        f"| Latency max | {metrics.latency_max:.2f} s |",
        f"| Prompt tokens | {metrics.prompt_tokens:,} |",
        f"| Output tokens | {metrics.output_tokens:,} |",
        f"| Thinking tokens | {metrics.thoughts_tokens:,} |",
        f"| Total tokens | {metrics.total_tokens:,} |",
    ]
    if metrics.cost_usd is None:
        lines.append(
            "| Cost | free tier: $0 marginal cost (set "
            f"`{PRICE_ENV_INPUT}` / `{PRICE_ENV_OUTPUT}` for a paid-tier estimate) |"
        )
    else:
        lines.append(f"| Estimated cost | ${metrics.cost_usd:.4f} |")

    lines += [
        "",
        "## Per question",
        "",
        "| ID | Expected tool | Chosen tool | Tool | Params | Answer | Plan | Seconds | Note |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for score in records:
        note = _note_for(score).replace("|", "\\|")
        lines.append(
            f"| {score.id} | {score.expected_tool or '(refuse)'} "
            f"| {score.chosen_tool or '(none)'} "
            f"| {'y' if score.tool_ok else ('n' if score.tool_ok is False else '-')} "
            f"| {'y' if score.params_ok else ('n' if score.params_ok is False else '-')} "
            f"| {'y' if score.answer.ok else 'n'} "
            f"| {'y' if score.plan_ok else 'n'} "
            f"| {score.latency_s:.1f} | {note} |"
        )

    lines += ["", "## Chosen parameters", ""]
    for score in records:
        if score.chosen_tool:
            params = json.dumps(score.chosen_params, sort_keys=True)
            lines.append(f"- **{score.id}** `{score.chosen_tool}` {params}")
        else:
            lines.append(f"- **{score.id}** (no step)")

    if skipped:
        lines += [
            "",
            "## Not attempted",
            "",
            "The run stopped early because the model returned HTTP 429 "
            "(free-tier daily quota). These questions were not scored:",
            "",
            " ".join(f"`{qid}`" for qid in skipped),
        ]

    if faithful:
        lines += [
            "",
            f"## Faithfulness pass (interpretation on {len(faithful)} questions)",
            "",
            "The narrative stage was also run on these questions. A question is "
            "faithful when the answer contains no number that could not be "
            "matched to a computed result.",
            "",
            "| ID | Faithful | Unverified numbers | Seconds |",
            "|---|---|---|---|",
        ]
        for score in faithful:
            numbers = ", ".join(f"`{n}`" for n in score.unverified) or "-"
            lines.append(
                f"| {score.id} | {'y' if not score.unverified else 'n'} "
                f"| {numbers} | {score.latency_s:.1f} |"
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
        help="Where to write the report.",
    )
    parser.add_argument(
        "--interpret",
        type=int,
        default=0,
        metavar="N",
        help="Also run the narrative stage on the first N questions (default 0).",
    )
    parser.add_argument(
        "--pause",
        type=float,
        default=PAUSE_SECONDS,
        help="Seconds to wait between questions (default 12).",
    )
    parser.add_argument("--price-in", type=float, default=None, help="USD per 1M input tokens.")
    parser.add_argument("--price-out", type=float, default=None, help="USD per 1M output tokens.")
    args = parser.parse_args(argv)

    load_environment()
    exam = load_questions(QUESTIONS_PATH)
    dataset_path = exam["dataset"]
    df = load_csv(PROJECT_ROOT / dataset_path)
    profile = profile_dataset(df)

    try:
        client = GeminiClient(model=args.model)
    except LLMError as exc:
        print(f"Cannot reach the model: {exc}")
        return 1

    questions = exam["questions"]
    interpret_n = max(0, min(args.interpret, len(questions)))
    print(f"Scoring {len(questions)} questions on {dataset_path}")
    print(f"Model: {client.model}")
    print(f"Interpretation pass: {interpret_n} question(s)")
    print()

    records: list[QuestionScore] = []
    skipped: list[str] = []
    rate_limited = False
    for index, question in enumerate(questions):
        if index:
            time.sleep(max(0.0, args.pause))
        score = _run_question(question, df, profile, client, interpret=False)
        if _is_rate_limit(score.error):
            rate_limited = True
            skipped = [q["id"] for q in questions[index:]]
            print(
                f"{score.id}  STOP  rate limit reached; "
                f"{len(skipped)} question(s) not attempted"
            )
            break
        records.append(score)
        print(
            f"{score.id}  {'PASS' if score.plan_ok else 'FAIL'}  "
            f"chosen={score.chosen_tool}  answer={'ok' if score.answer.ok else 'BAD'}  "
            f"({score.latency_s:.1f}s)  {_note_for(score)}"
        )

    price_in = _resolve_price(PRICE_ENV_INPUT, args.price_in)
    price_out = _resolve_price(PRICE_ENV_OUTPUT, args.price_out)
    metrics = aggregate(records, price_in=price_in, price_out=price_out)

    faithful: list[QuestionScore] = []
    if not rate_limited:
        for question in questions[:interpret_n]:
            if args.pause:
                time.sleep(max(0.0, args.pause))
            score = _run_question(question, df, profile, client, interpret=True)
            if _is_rate_limit(score.error):
                print(f"{score.id}  STOP  rate limit reached during the interpretation pass")
                break
            faithful.append(score)
            print(
                f"{score.id}  interpret  unverified={len(score.unverified)}  "
                f"({score.latency_s:.1f}s)"
            )

    print()
    print(f"Plan accuracy : {_pct(metrics.plan_accuracy)}  ({sum(1 for s in records if s.plan_ok)}/{metrics.n})")
    print(f"Answer accuracy: {_pct(metrics.answer_accuracy)}")
    print(f"Latency        : mean {metrics.latency_mean:.2f}s, p95 {metrics.latency_p95:.2f}s")
    print(f"Tokens         : {metrics.total_tokens:,} total")

    out_path = Path(args.out)
    _write_report(
        records,
        metrics,
        model=client.model,
        dataset=dataset_path,
        interpret=interpret_n,
        faithful=faithful,
        skipped=skipped,
        path=out_path,
    )
    print(f"Report written to {out_path}")
    if skipped:
        return 3
    return 0 if all(score.plan_ok for score in records) else 2


if __name__ == "__main__":
    raise SystemExit(main())
