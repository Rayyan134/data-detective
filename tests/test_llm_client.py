"""Tests for src.llm_client.

No network and no real API key: a fake session stands in for ``requests``, so
every error path can be exercised deterministically. Run with:

    .\\.venv\\Scripts\\python.exe -m pytest -q
"""

from __future__ import annotations

import json

import pytest
import requests

from src.llm_client import (
    DEFAULT_MODEL,
    DEFAULT_TIMEOUT,
    GeminiClient,
    LLMAuthError,
    LLMError,
    LLMRateLimitError,
    LLMResponseError,
    LLMTimeoutError,
    load_environment,
)


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------
class FakeResponse:
    """Stands in for a requests.Response."""

    def __init__(self, status_code=200, payload=None, headers=None, bad_json=False):
        self.status_code = status_code
        self.headers = headers or {}
        self._payload = payload
        self._bad_json = bad_json

    def json(self):
        if self._bad_json:
            raise ValueError("not json")
        return self._payload


class FakeSession:
    """Records calls and replays a canned response (or a sequence of them).

    Passing ``responses=[...]`` lets a test simulate a transient failure
    followed by a success, which is how the retry path is exercised.
    """

    def __init__(self, response=None, error=None, responses=None):
        self.error = error
        self.calls: list[tuple] = []
        if responses is not None:
            self._queue = list(responses)
        elif response is not None:
            self._queue = [response]
        else:
            self._queue = []

    def _handle(self, kind, url, headers, timeout, body=None):
        self.calls.append((kind, url, headers, timeout, body))
        if self.error is not None:
            raise self.error
        # Hold the last response so it repeats if more calls arrive.
        if len(self._queue) > 1:
            return self._queue.pop(0)
        return self._queue[0] if self._queue else None

    def post(self, url, headers=None, json=None, timeout=None):
        return self._handle("post", url, headers, timeout, json)

    def get(self, url, headers=None, timeout=None):
        return self._handle("get", url, headers, timeout)


def ok_body(text="connection ok"):
    return {
        "candidates": [
            {
                "content": {"role": "model", "parts": [{"text": text}]},
                "finishReason": "STOP",
            }
        ],
        "usageMetadata": {"promptTokenCount": 5, "totalTokenCount": 7},
        "modelVersion": "gemini-3.8-flash",
    }


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Keep the ambient environment from leaking into tests."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    monkeypatch.delenv("GEMINI_TIMEOUT", raising=False)


def make_client(**kwargs) -> tuple[GeminiClient, FakeSession]:
    session = kwargs.pop("session", None) or FakeSession(response=FakeResponse(payload=ok_body()))
    # Never actually sleep in tests; retry behaviour is asserted via call counts.
    kwargs.setdefault("sleep", lambda _seconds: None)
    client = GeminiClient(api_key="test-key", session=session, **kwargs)
    return client, session


# ---------------------------------------------------------------------------
# Configuration and key handling
# ---------------------------------------------------------------------------
def test_missing_key_raises_auth_error():
    with pytest.raises(LLMAuthError, match="No API key found"):
        GeminiClient()


def test_key_is_read_from_the_environment(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "from-env")

    client, _ = make_client()
    assert client._headers()["x-goog-api-key"] == "test-key"

    env_client = GeminiClient()
    assert env_client._headers()["x-goog-api-key"] == "from-env"


def test_model_can_be_overridden_by_environment(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-flash-latest")

    assert GeminiClient().model == "gemini-flash-latest"


def test_default_model_is_used_when_unset():
    client, _ = make_client()
    assert client.model == DEFAULT_MODEL


def test_blank_key_is_rejected(monkeypatch):
    with pytest.raises(LLMAuthError):
        GeminiClient(api_key="   ")


# ---------------------------------------------------------------------------
# Successful calls
# ---------------------------------------------------------------------------
def test_generate_returns_text_and_usage():
    client, _ = make_client()

    response = client.generate("hello")

    assert response.text == "connection ok"
    assert response.finish_reason == "STOP"
    assert response.usage["totalTokenCount"] == 7
    assert response.model == "gemini-3.8-flash"


def test_generate_targets_the_model_endpoint():
    client, session = make_client(model="gemini-3.8-flash")

    client.generate("hello")

    kind, url, headers, timeout, body = session.calls[0]
    assert kind == "post"
    assert url.endswith("/models/gemini-3.8-flash:generateContent")
    assert headers["x-goog-api-key"] == "test-key"
    assert body["contents"][0]["parts"][0]["text"] == "hello"
    assert timeout == client.timeout


def test_generate_joins_multiple_text_parts():
    body = {
        "candidates": [
            {"content": {"parts": [{"text": "part one "}, {"text": "part two"}]}}
        ]
    }
    client, _ = make_client(session=FakeSession(response=FakeResponse(payload=body)))

    assert client.generate("x").text == "part one part two"


def test_system_instruction_is_included_when_given():
    client, session = make_client()

    client.generate("hello", system_instruction="Be terse.")

    assert session.calls[0][4]["systemInstruction"]["parts"][0]["text"] == "Be terse."


def test_generation_config_is_built_from_arguments():
    client, session = make_client()

    client.generate("hello", temperature=0.2, max_output_tokens=64)

    config = session.calls[0][4]["generationConfig"]
    assert config == {"temperature": 0.2, "maxOutputTokens": 64}


# ---------------------------------------------------------------------------
# Thinking control
# ---------------------------------------------------------------------------
def test_thinking_level_is_sent_uppercased():
    client, session = make_client()

    client.generate("hello", thinking_level="low")

    config = session.calls[0][4]["generationConfig"]
    assert config["thinkingConfig"] == {"thinkingLevel": "LOW"}


def test_thinking_level_is_omitted_by_default():
    client, session = make_client()

    client.generate("hello")

    body = session.calls[0][4]
    assert "thinkingConfig" not in body.get("generationConfig", {})


def test_invalid_thinking_level_is_rejected_before_any_call():
    client, session = make_client()

    with pytest.raises(LLMError, match="thinking_level"):
        client.generate("hello", thinking_level="maximum")

    assert session.calls == []


def test_check_connection_uses_low_thinking():
    client, session = make_client()

    client.check_connection()

    config = session.calls[0][4]["generationConfig"]
    assert config["thinkingConfig"] == {"thinkingLevel": "LOW"}


# ---------------------------------------------------------------------------
# Timeout configuration
# ---------------------------------------------------------------------------
def test_timeout_defaults_to_the_module_default():
    client, _ = make_client()

    assert client.timeout == DEFAULT_TIMEOUT


def test_timeout_can_be_overridden_by_environment(monkeypatch):
    monkeypatch.setenv("GEMINI_TIMEOUT", "90")

    client, _ = make_client()

    assert client.timeout == 90.0


def test_timeout_argument_beats_environment(monkeypatch):
    monkeypatch.setenv("GEMINI_TIMEOUT", "90")

    client, _ = make_client(timeout=15)

    assert client.timeout == 15.0


def test_non_numeric_timeout_environment_falls_back(monkeypatch):
    monkeypatch.setenv("GEMINI_TIMEOUT", "soon")

    client, _ = make_client()

    assert client.timeout == DEFAULT_TIMEOUT


# ---------------------------------------------------------------------------
# Retries on transient failures
# ---------------------------------------------------------------------------
def test_transient_503_is_retried_then_succeeds():
    session = FakeSession(
        responses=[
            FakeResponse(status_code=503),
            FakeResponse(payload=ok_body("recovered")),
        ]
    )
    client, _ = make_client(session=session)

    response = client.generate("hello")

    assert response.text == "recovered"
    assert len(session.calls) == 2


def test_429_is_retried_then_succeeds():
    session = FakeSession(
        responses=[
            FakeResponse(status_code=429, headers={"Retry-After": "1"}),
            FakeResponse(payload=ok_body()),
        ]
    )
    client, _ = make_client(session=session)

    assert client.generate("hello").text == "connection ok"
    assert len(session.calls) == 2


def test_retries_are_exhausted_before_failing():
    session = FakeSession(response=FakeResponse(status_code=503))
    client, _ = make_client(session=session, max_retries=2)

    with pytest.raises(LLMError, match="server error"):
        client.generate("hello")

    assert len(session.calls) == 3  # first attempt + two retries


def test_max_retries_zero_fails_immediately():
    session = FakeSession(
        responses=[FakeResponse(status_code=503), FakeResponse(payload=ok_body())]
    )
    client, _ = make_client(session=session, max_retries=0)

    with pytest.raises(LLMError, match="server error"):
        client.generate("hello")

    assert len(session.calls) == 1


def test_auth_errors_are_not_retried():
    session = FakeSession(
        responses=[FakeResponse(status_code=401), FakeResponse(payload=ok_body())]
    )
    client, _ = make_client(session=session)

    with pytest.raises(LLMAuthError):
        client.generate("hello")

    assert len(session.calls) == 1


def test_retry_delay_honours_retry_after_header():
    slept: list[float] = []
    session = FakeSession(
        responses=[
            FakeResponse(status_code=429, headers={"Retry-After": "3"}),
            FakeResponse(payload=ok_body()),
        ]
    )
    client, _ = make_client(session=session, sleep=slept.append)

    client.generate("hello")

    assert slept == [3.0]


def test_retry_delay_backs_off_exponentially_without_a_header():
    slept: list[float] = []
    session = FakeSession(response=FakeResponse(status_code=503))
    client, _ = make_client(session=session, max_retries=2, backoff_factor=1.0, sleep=slept.append)

    with pytest.raises(LLMError):
        client.generate("hello")

    assert slept == [1.0, 2.0]


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("status", [401, 403])
def test_auth_statuses_raise_auth_error(status):
    client, _ = make_client(session=FakeSession(response=FakeResponse(status_code=status)))

    with pytest.raises(LLMAuthError):
        client.generate("hello")


def test_rate_limit_raises_with_retry_after():
    client, _ = make_client(
        session=FakeSession(
            response=FakeResponse(status_code=429, headers={"Retry-After": "12"})
        )
    )

    with pytest.raises(LLMRateLimitError) as excinfo:
        client.generate("hello")

    assert excinfo.value.retry_after == 12.0


def test_rate_limit_without_header_still_raises():
    client, _ = make_client(session=FakeSession(response=FakeResponse(status_code=429)))

    with pytest.raises(LLMRateLimitError) as excinfo:
        client.generate("hello")

    assert excinfo.value.retry_after is None


def test_server_error_raises_base_error():
    client, _ = make_client(session=FakeSession(response=FakeResponse(status_code=503)))

    with pytest.raises(LLMError, match="server error"):
        client.generate("hello")


def test_bad_request_includes_the_server_message():
    payload = {"error": {"message": "Invalid API key or model"}}
    client, _ = make_client(
        session=FakeSession(response=FakeResponse(status_code=400, payload=payload))
    )

    with pytest.raises(LLMError, match="Invalid API key or model"):
        client.generate("hello")


def test_timeout_raises_timeout_error():
    client, _ = make_client(session=FakeSession(error=requests.exceptions.Timeout()))

    with pytest.raises(LLMTimeoutError, match="timed out"):
        client.generate("hello")


def test_connection_error_raises_base_error():
    client, _ = make_client(
        session=FakeSession(error=requests.exceptions.ConnectionError("down"))
    )

    with pytest.raises(LLMError, match="Could not reach"):
        client.generate("hello")


def test_non_json_body_raises_response_error():
    client, _ = make_client(session=FakeSession(response=FakeResponse(bad_json=True)))

    with pytest.raises(LLMResponseError, match="not valid JSON"):
        client.generate("hello")


def test_blocked_prompt_raises_response_error():
    payload = {"promptFeedback": {"blockReason": "SAFETY"}, "candidates": []}
    client, _ = make_client(session=FakeSession(response=FakeResponse(payload=payload)))

    with pytest.raises(LLMResponseError, match="SAFETY"):
        client.generate("hello")


def test_no_candidates_raises_response_error():
    client, _ = make_client(session=FakeSession(response=FakeResponse(payload={})))

    with pytest.raises(LLMResponseError, match="no candidates"):
        client.generate("hello")


def test_candidate_without_text_raises_response_error():
    payload = {"candidates": [{"content": {"parts": [{"inlineData": {}}]}}]}
    client, _ = make_client(session=FakeSession(response=FakeResponse(payload=payload)))

    with pytest.raises(LLMResponseError, match="no text part"):
        client.generate("hello")


def test_empty_prompt_is_rejected_before_any_call():
    client, session = make_client()

    with pytest.raises(LLMError, match="non-empty"):
        client.generate("   ")

    assert session.calls == []


# ---------------------------------------------------------------------------
# JSON mode
# ---------------------------------------------------------------------------
def test_generate_json_requests_json_and_parses_it():
    body = ok_body(text=json.dumps({"steps": ["group_comparison"]}))
    client, session = make_client(session=FakeSession(response=FakeResponse(payload=body)))

    response = client.generate_json("plan this", schema={"type": "object"})

    assert response.json() == {"steps": ["group_comparison"]}
    config = session.calls[0][4]["generationConfig"]
    assert config["responseMimeType"] == "application/json"
    assert config["responseSchema"] == {"type": "object"}


def test_generate_json_rejects_non_json_text():
    client, _ = make_client(
        session=FakeSession(response=FakeResponse(payload=ok_body(text="not json at all")))
    )

    with pytest.raises(LLMResponseError, match="did not return valid JSON"):
        client.generate_json("plan this")


def test_llm_response_json_reports_the_offending_text():
    client, _ = make_client(
        session=FakeSession(response=FakeResponse(payload=ok_body(text="oops")))
    )
    response = client.generate("x")

    with pytest.raises(LLMResponseError) as excinfo:
        response.json()

    assert "oops" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Model listing and the connection test
# ---------------------------------------------------------------------------
def test_list_models_returns_the_model_list():
    payload = {"models": [{"name": "models/gemini-3.8-flash"}]}
    client, session = make_client(session=FakeSession(response=FakeResponse(payload=payload)))

    models = client.list_models()

    assert models[0]["name"] == "models/gemini-3.8-flash"
    assert session.calls[0][0] == "get"
    assert session.calls[0][1].endswith("/models")


def test_list_models_rejects_a_non_list_payload():
    client, _ = make_client(
        session=FakeSession(response=FakeResponse(payload={"models": "not-a-list"}))
    )

    with pytest.raises(LLMResponseError, match="list"):
        client.list_models()


def test_list_models_treats_missing_models_as_empty():
    client, _ = make_client(session=FakeSession(response=FakeResponse(payload={})))

    assert client.list_models() == []


def test_check_connection_returns_a_response():
    client, _ = make_client()

    response = client.check_connection()

    assert "connection ok" in response.text


def test_load_environment_returns_a_boolean():
    assert isinstance(load_environment(".definitely-not-a-real-env-file"), bool)
