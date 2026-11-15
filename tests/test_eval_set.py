"""Tests for the evaluation question set in eval/questions.json.

These guard against the question set drifting away from the code: every
expected tool must exist in the registry, and the referenced dataset must be
present in the repository.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.tools import TOOL_REGISTRY

PROJECT_ROOT = Path(__file__).resolve().parents[1]
QUESTIONS_PATH = PROJECT_ROOT / "eval" / "questions.json"


@pytest.fixture(scope="module")
def exam() -> dict:
    return json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))


def test_exam_set_has_between_10_and_20_questions(exam):
    assert 10 <= len(exam["questions"]) <= 20


def test_question_ids_are_unique(exam):
    ids = [item["id"] for item in exam["questions"]]

    assert len(ids) == len(set(ids))


def test_every_question_has_text(exam):
    for item in exam["questions"]:
        assert item["question"].strip()


def test_expected_tools_exist_in_the_registry(exam):
    for item in exam["questions"]:
        tool = item["expected_tool"]
        if tool is not None:
            assert tool in TOOL_REGISTRY, f"{item['id']} expects unknown tool '{tool}'"


def test_unsupported_questions_declare_expected_behaviour(exam):
    for item in exam["questions"]:
        if item["expected_tool"] is None:
            assert item["expected_behavior"] == "refuse_or_explain"


def test_available_tools_match_the_registry(exam):
    assert set(exam["available_tools"]) == set(TOOL_REGISTRY)


def _checks(question: dict) -> list[dict]:
    check = question.get("check")
    return check if isinstance(check, list) else [check]


def test_every_question_has_a_machine_checkable_answer(exam):
    for item in exam["questions"]:
        checks = _checks(item)
        assert checks, f"{item['id']} has no check"
        for check in checks:
            assert check["kind"] in exam["check_kinds"], f"{item['id']}: {check['kind']}"


def test_refusal_questions_use_the_refuse_check(exam):
    for item in exam["questions"]:
        if item["expected_tool"] is None:
            assert all(check["kind"] == "refuse" for check in _checks(item)), item["id"]


def test_supported_questions_do_not_use_the_refuse_check(exam):
    for item in exam["questions"]:
        if item["expected_tool"] is not None:
            assert all(check["kind"] != "refuse" for check in _checks(item)), item["id"]


def test_referenced_dataset_exists(exam):
    assert (PROJECT_ROOT / exam["dataset"]).is_file()
