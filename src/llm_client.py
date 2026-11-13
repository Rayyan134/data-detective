"""Minimal Gemini Flash client built on plain HTTP (no vendor SDK).

This module is the only place in the project that knows how to talk to the
language model. Everything else (profiling, tools, the investigator, the UI)
stays independent of it, which keeps the deterministic analysis testable on its
own and makes the provider swappable later.

Why REST instead of an SDK:
    ``requests`` is already a dependency, so this adds nothing to install. The
    API surface we need is one POST and one GET, which is small enough to own.

Design notes:
    * The API key is read from the ``GEMINI_API_KEY`` environment variable. It is
      never hard-coded, never written to disk, and never included in an error
      message.
    * Every failure mode the caller can act on has its own exception type:
      authentication, rate limiting, timeout, and malformed responses.
    * ``generate()`` accepts a plain prompt plus optional system instruction. It
      does not know about DataFrames, so there is no way for this module to
      leak a whole dataset into a request.

Public API:
    GeminiClient          -- the HTTP client
    LLMResponse           -- text plus metadata returned by one call
    check_connection()    -- one cheap round trip used to verify setup
    list_models()         -- models the current key can see
    load_environment()    -- load a local .env file if python-dotenv is present

Exceptions:
    LLMError              -- base class for every failure below
    LLMAuthError          -- missing, invalid or unauthorised key (401/403)
    LLMRateLimitError     -- quota or rate limit hit (429)
    LLMTimeoutError       -- the request exceeded the timeout
    LLMResponseError      -- HTTP 200 but the body was unusable
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any

import requests

__all__ = [
    "DEFAULT_MODEL",
    "DEFAULT_TIMEOUT",
    "API_BASE",
    "THINKING_LEVELS",
    "GeminiClient",
    "LLMResponse",
    "LLMError",
    "LLMAuthError",
    "LLMRateLimitError",
    "LLMTimeoutError",
    "LLMResponseError",
    "load_environment",
    "list_models",
    "check_connection",
]

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# A stable Flash model that the pricing page lists as free of charge on the free
# tier. Chosen over gemini-3.8-flash because the newest Flash returned HTTP 503
# (overloaded) on the free tier during setup, while 3.6 Flash served reliably.
# Override it with the GEMINI_MODEL environment variable, or pass ``model=``.
DEFAULT_MODEL = "gemini-3.6-flash"

API_BASE = "https://generativelanguage.googleapis.com/v1beta"

# Flash is a reasoning model: it "thinks" before answering, so even a one-line
# prompt can take a while, especially on the free tier. 60s is a comfortable
# default. Override with GEMINI_TIMEOUT (seconds) or the timeout argument.
DEFAULT_TIMEOUT = 60.0

API_KEY_ENV_VAR = "GEMINI_API_KEY"
MODEL_ENV_VAR = "GEMINI_MODEL"
TIMEOUT_ENV_VAR = "GEMINI_TIMEOUT"

# Thinking depth for Gemini 3+ models. Lower is faster and cheaper; "MINIMAL"
# is not supported on gemini-3.8-flash, so it is not offered as a default.
THINKING_LEVELS = ("MINIMAL", "LOW", "MEDIUM", "HIGH", "XHIGH")

# Transient failures worth retrying: the model is overloaded or a quota window
# is briefly full. Everything else (bad key, bad request) fails immediately.
RETRYABLE_STATUSES = frozenset({429, 500, 502, 503, 504})
MAX_RETRY_DELAY = 8.0


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------
class LLMError(RuntimeError):
    """Base class for every problem raised by this module.

    Subclassing RuntimeError (not ValueError) signals "this failed at call
    time", which is how the investigator should treat it: report it, do not
    crash the whole investigation.
    """


class LLMAuthError(LLMError):
    """The API key is missing, invalid, or not allowed to use this model."""


class LLMRateLimitError(LLMError):
    """The request was rejected because a quota or rate limit was reached."""

    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        # Seconds to wait, if the server told us. Handy for a caller that wants
        # to back off, and it keeps the value out of the message string.
        self.retry_after = retry_after


class LLMTimeoutError(LLMError):
    """The request did not complete within the configured timeout."""


class LLMResponseError(LLMError):
    """The server answered, but the response could not be used.

    Raised for invalid JSON, a missing ``candidates`` list, an empty reply, or
    a prompt the model refused to answer.
    """


# ---------------------------------------------------------------------------
# Response container
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LLMResponse:
    """One model response: the text, plus what we need for logging.

    Attributes:
        text: The concatenated text of the first candidate.
        model: The model string that produced the response.
        finish_reason: Why generation stopped, e.g. "STOP" or "MAX_TOKENS".
        usage: Token counts from ``usageMetadata`` (may be empty).
        raw: The full decoded JSON body, kept for debugging.
    """

    text: str
    model: str
    finish_reason: str | None = None
    usage: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)

    def json(self) -> Any:
        """Parse ``text`` as JSON, or raise ``LLMResponseError``.

        Used with ``generate_json``. The model is asked for JSON, but a model is
        not a schema validator, so a malformed body is a normal thing to defend
        against rather than an impossible one.
        """
        try:
            return json.loads(self.text)
        except (json.JSONDecodeError, TypeError) as exc:
            preview = (self.text or "")[:200]
            raise LLMResponseError(
                f"The model did not return valid JSON ({exc}). "
                f"Start of the response: {preview!r}"
            ) from exc


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
def load_environment(env_file: str = ".env") -> bool:
    """Load variables from a local ``.env`` file, if one exists.

    Uses ``python-dotenv`` (already a dependency). Returns True when a file was
    loaded. Safe to call more than once: existing environment variables win.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:  # pragma: no cover - dotenv is in requirements.txt
        return False
    return bool(load_dotenv(dotenv_path=env_file, override=False))


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------
class GeminiClient:
    """A thin wrapper around the Gemini ``generateContent`` endpoint.

    Args:
        api_key: The key to use. When omitted, ``GEMINI_API_KEY`` is read from
            the environment.
        model: Model identifier. Defaults to ``GEMINI_MODEL`` or
            ``DEFAULT_MODEL``.
        timeout: Seconds to wait for the server, per request. Defaults to
            ``GEMINI_TIMEOUT`` or ``DEFAULT_TIMEOUT``.
        session: An object with ``get``/``post`` methods. Defaults to a
            ``requests.Session``. Tests inject a fake here.

    Raises:
        LLMAuthError: When no key can be found. Raised immediately so the caller
            fails fast with a clear message instead of getting a confusing 400
            from the server.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
        session: Any | None = None,
        max_retries: int = 2,
        backoff_factor: float = 1.0,
        sleep: Any | None = None,
    ) -> None:
        resolved_key = api_key or os.getenv(API_KEY_ENV_VAR, "")
        resolved_key = resolved_key.strip()
        if not resolved_key:
            raise LLMAuthError(
                f"No API key found. Set the {API_KEY_ENV_VAR} environment "
                "variable (see .env.example) or pass api_key=... explicitly."
            )

        self._api_key = resolved_key
        self.model = model or os.getenv(MODEL_ENV_VAR) or DEFAULT_MODEL
        self.timeout = _resolve_timeout(timeout)
        self.max_retries = max(0, int(max_retries))
        self.backoff_factor = float(backoff_factor)
        self._sleep = sleep if sleep is not None else time.sleep
        self._session = session if session is not None else requests.Session()

    # -- internals ----------------------------------------------------------
    def _headers(self) -> dict[str, str]:
        return {
            "x-goog-api-key": self._api_key,
            "Content-Type": "application/json",
        }

    def _url(self, method: str) -> str:
        return f"{API_BASE}/models/{self.model}:{method}"

    def _request(
        self,
        url: str,
        *,
        json_body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Send a request, retrying transient overloads, then decode it.

        HTTP 429 and 5xx responses are usually temporary (the model is busy or
        the quota window is briefly full), so they are retried with a short
        backoff before being surfaced. Everything else fails immediately.
        """
        attempts = self.max_retries + 1
        for attempt in range(attempts):
            response = self._send(url, json_body)
            if response.status_code in RETRYABLE_STATUSES and attempt < attempts - 1:
                self._sleep(self._retry_delay(response, attempt))
                continue
            return self._decode(response)
        # The loop always returns or raises, so this is unreachable.
        raise LLMError("Request failed after retries.")  # pragma: no cover

    def _send(self, url: str, json_body: dict[str, Any] | None) -> Any:
        """Perform the HTTP call, mapping transport failures to typed errors."""
        try:
            if json_body is None:
                return self._session.get(
                    url, headers=self._headers(), timeout=self.timeout
                )
            return self._session.post(
                url,
                headers=self._headers(),
                json=json_body,
                timeout=self.timeout,
            )
        except requests.exceptions.Timeout as exc:
            raise LLMTimeoutError(
                f"The Gemini request timed out after {self.timeout} seconds. "
                "Check your connection, or raise the timeout."
            ) from exc
        except requests.exceptions.RequestException as exc:
            raise LLMError(f"Could not reach the Gemini API: {exc}") from exc

    def _retry_delay(self, response: Any, attempt: int) -> float:
        """Seconds to wait before the next attempt.

        Honours a server-provided ``Retry-After``; otherwise backs off
        exponentially and caps the wait so a retry cannot hang for long.
        """
        retry_after = _parse_retry_after(response)
        if retry_after is not None:
            return min(retry_after, MAX_RETRY_DELAY)
        return min(self.backoff_factor * (2**attempt), MAX_RETRY_DELAY)

    def _decode(self, response: Any) -> dict[str, Any]:
        """Turn a response into a payload, or raise the matching typed error."""
        status = response.status_code

        if status in (401, 403):
            raise LLMAuthError(
                f"Authentication failed (HTTP {status}). The API key is "
                "missing, invalid, or not allowed to use "
                f"'{self.model}'. Check {API_KEY_ENV_VAR} in your .env file."
            )
        if status == 429:
            raise LLMRateLimitError(
                "Rate limit or quota reached (HTTP 429). Wait and retry, or "
                "check your limits in Google AI Studio.",
                retry_after=_parse_retry_after(response),
            )
        if status >= 500:
            raise LLMError(
                f"The Gemini API returned a server error (HTTP {status}) after "
                f"{self.max_retries + 1} attempt(s). The model may be busy; "
                "try again shortly or switch GEMINI_MODEL."
            )
        if status >= 400:
            detail = _error_detail(response)
            raise LLMError(f"Gemini rejected the request (HTTP {status}): {detail}")

        try:
            payload = response.json()
        except ValueError as exc:
            raise LLMResponseError(
                "The Gemini API returned a body that is not valid JSON."
            ) from exc
        if not isinstance(payload, dict):
            raise LLMResponseError(
                f"Expected a JSON object from the API, got {type(payload).__name__}."
            )
        return payload

    # -- public API ---------------------------------------------------------
    def generate(
        self,
        prompt: str,
        *,
        system_instruction: str | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
        response_mime_type: str | None = None,
        response_schema: dict[str, Any] | None = None,
        thinking_level: str | None = None,
    ) -> LLMResponse:
        """Send a single-turn prompt and return the model's reply.

        Args:
            prompt: The user text. Keep it to schema, profile summaries and
                tool output; never pass raw dataset rows.
            system_instruction: Optional behaviour instruction for the model.
            temperature: Sampling temperature. Lower is more repeatable.
            max_output_tokens: Cap on generated tokens.
            response_mime_type: Set to ``"application/json"`` for JSON mode.
            response_schema: Optional OpenAPI-style schema for the JSON reply.
            thinking_level: One of ``THINKING_LEVELS``. Gemini 3+ models think
                by default; a lower level cuts latency and cost. Left unset, the
                model uses its own default.

        Raises:
            LLMAuthError, LLMRateLimitError, LLMTimeoutError, LLMResponseError,
            or LLMError.
        """
        if not isinstance(prompt, str) or not prompt.strip():
            raise LLMError("prompt must be a non-empty string.")

        generation_config: dict[str, Any] = {}
        if temperature is not None:
            generation_config["temperature"] = temperature
        if max_output_tokens is not None:
            generation_config["maxOutputTokens"] = max_output_tokens
        if response_mime_type is not None:
            generation_config["responseMimeType"] = response_mime_type
        if response_schema is not None:
            generation_config["responseSchema"] = response_schema
        if thinking_level is not None:
            generation_config["thinkingConfig"] = {
                "thinkingLevel": _normalise_thinking_level(thinking_level)
            }

        body: dict[str, Any] = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        }
        if system_instruction:
            body["systemInstruction"] = {"parts": [{"text": system_instruction}]}
        if generation_config:
            body["generationConfig"] = generation_config

        payload = self._request(self._url("generateContent"), json_body=body)

        # A blocked prompt comes back with no candidates and a reason. Surface
        # it as a clear error instead of an empty string.
        feedback = payload.get("promptFeedback") or {}
        block_reason = feedback.get("blockReason")
        if block_reason:
            raise LLMResponseError(
                f"The model refused the prompt (blockReason: {block_reason})."
            )

        candidates = payload.get("candidates") or []
        if not candidates:
            raise LLMResponseError(
                "The API returned no candidates, so there is no text to read."
            )

        candidate = candidates[0]
        text = _extract_text(candidate)
        if text is None:
            raise LLMResponseError(
                "The response candidate contained no text part."
            )

        return LLMResponse(
            text=text,
            model=payload.get("modelVersion") or self.model,
            finish_reason=candidate.get("finishReason"),
            usage=payload.get("usageMetadata") or {},
            raw=payload,
        )

    def generate_json(
        self,
        prompt: str,
        *,
        schema: dict[str, Any] | None = None,
        system_instruction: str | None = None,
        temperature: float = 0.0,
        max_output_tokens: int | None = None,
        thinking_level: str | None = None,
    ) -> LLMResponse:
        """Like ``generate``, but ask for and parse a JSON object.

        Returns the ``LLMResponse``; call ``.json()`` on it to get the parsed
        value, or ``.usage`` for token counts. Keeping the envelope means the
        caller still has the raw text when parsing fails.
        """
        response = self.generate(
            prompt,
            system_instruction=system_instruction,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            response_mime_type="application/json",
            response_schema=schema,
            thinking_level=thinking_level,
        )
        # Fail here rather than at the call site, so every malformed response
        # surfaces at the same place.
        response.json()
        return response

    def list_models(self) -> list[dict[str, Any]]:
        """Return the models the current key can see.

        Each entry has at least ``name`` (e.g. ``models/gemini-3.8-flash``) and
        ``supportedGenerationMethods``.
        """
        payload = self._request(f"{API_BASE}/models")
        models = payload.get("models") or []
        if not isinstance(models, list):
            raise LLMResponseError("Expected a list under 'models' in the response.")
        return models

    def check_connection(
        self,
        prompt: str = "Reply with exactly: connection ok",
        thinking_level: str = "LOW",
    ) -> LLMResponse:
        """Send one tiny prompt to prove the key, model and network work.

        Uses ``LOW`` thinking by default: the point is to verify the pipe, not
        to test reasoning, and the default ``MEDIUM`` level can make even a
        one-line reply slow on the free tier.
        """
        return self.generate(
            prompt,
            temperature=0.0,
            # Thinking tokens count towards maxOutputTokens, so leave headroom:
            # a tight cap can truncate the reply before any text is emitted.
            max_output_tokens=512,
            thinking_level=thinking_level,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _resolve_timeout(timeout: float | None) -> float:
    """Pick the timeout: explicit argument, then GEMINI_TIMEOUT, then default."""
    if timeout is not None:
        return float(timeout)
    raw = os.getenv(TIMEOUT_ENV_VAR, "").strip()
    if raw:
        try:
            return float(raw)
        except ValueError:
            pass
    return DEFAULT_TIMEOUT


def _normalise_thinking_level(level: str) -> str:
    """Validate and upper-case a thinking level, with a helpful message."""
    normalised = str(level).strip().upper()
    if normalised not in THINKING_LEVELS:
        raise LLMError(
            f"thinking_level must be one of {', '.join(THINKING_LEVELS)}; "
            f"got {level!r}. Note 'MINIMAL' is not supported on gemini-3.8-flash."
        )
    return normalised


def _extract_text(candidate: dict[str, Any]) -> str | None:
    """Join the text parts of one candidate, or return None if there are none."""
    content = candidate.get("content") or {}
    parts = content.get("parts") or []
    pieces = [
        part.get("text")
        for part in parts
        if isinstance(part, dict) and isinstance(part.get("text"), str)
    ]
    if not pieces:
        return None
    return "".join(pieces)


def _parse_retry_after(response: Any) -> float | None:
    """Read a Retry-After header in seconds, if the server sent one."""
    raw = None
    headers = getattr(response, "headers", None)
    if headers:
        raw = headers.get("Retry-After")
    if not raw:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _error_detail(response: Any) -> str:
    """Pull the human-readable message out of a Google error body."""
    try:
        payload = response.json()
    except ValueError:
        return "no details provided"
    error = payload.get("error") if isinstance(payload, dict) else None
    if isinstance(error, dict) and error.get("message"):
        return str(error["message"])
    return "no details provided"


# ---------------------------------------------------------------------------
# Module-level convenience wrappers
# ---------------------------------------------------------------------------
def list_models() -> list[dict[str, Any]]:
    """Convenience wrapper: list models using the environment configuration."""
    return GeminiClient().list_models()


def check_connection(
    prompt: str = "Reply with exactly: connection ok",
    thinking_level: str = "LOW",
) -> LLMResponse:
    """Convenience wrapper: one round trip using the environment configuration."""
    return GeminiClient().check_connection(prompt, thinking_level=thinking_level)


# ---------------------------------------------------------------------------
# Command-line connection test
# ---------------------------------------------------------------------------
def _main() -> int:
    """Minimal connection test: ``python -m src.llm_client``.

    Prints what it is doing, then exercises the two calls the investigator will
    depend on. Exits non-zero on failure so it can gate a build step.
    """
    load_environment()
    key = os.getenv(API_KEY_ENV_VAR, "").strip()
    model = os.getenv(MODEL_ENV_VAR) or DEFAULT_MODEL

    print("Gemini connection test")
    print("-" * 40)
    print(f"Endpoint : {API_BASE}")
    print(f"Model    : {model}")
    print(f"Timeout  : {os.getenv(TIMEOUT_ENV_VAR) or DEFAULT_TIMEOUT}s")
    print(f"Key      : {'found' if key else 'MISSING'} ({API_KEY_ENV_VAR})")

    if not key:
        print()
        print("No key found. To fix this:")
        print("  1. Copy-Item .env.example .env")
        print(f"  2. Put a real key in {API_KEY_ENV_VAR}=... inside .env")
        print("  3. Re-run: python -m src.llm_client")
        print("Get a free key at https://aistudio.google.com/api-keys")
        return 1

    try:
        client = GeminiClient(model=model)

        print()
        print("1) Listing available models ...")
        started = time.perf_counter()
        names = {entry.get("name", "").split("/")[-1] for entry in client.list_models()}
        print(f"   {len(names)} models visible to this key ({time.perf_counter() - started:.1f}s).")
        if model in names:
            print(f"   '{model}' is available.")
        else:
            print(f"   Note: '{model}' was not in the list; it may still work.")

        print()
        print("2) Sending a one-line prompt (thinking level: LOW) ...")
        started = time.perf_counter()
        reply = client.check_connection()
        elapsed = time.perf_counter() - started
        print(f"   Reply       : {reply.text.strip()!r}")
        print(f"   Finish      : {reply.finish_reason}")
        print(f"   Model used  : {reply.model}")
        print(f"   Latency     : {elapsed:.1f}s")
        tokens = reply.usage.get("totalTokenCount")
        if tokens is not None:
            print(f"   Total tokens: {tokens}")

        print()
        print("Connection OK.")
        return 0

    except LLMAuthError as exc:
        print(f"\nAUTH ERROR: {exc}")
    except LLMRateLimitError as exc:
        wait = f" (retry after {exc.retry_after}s)" if exc.retry_after else ""
        print(f"\nRATE LIMIT: {exc}{wait}")
    except LLMTimeoutError as exc:
        print(f"\nTIMEOUT: {exc}")
        print(f"Tip: set {TIMEOUT_ENV_VAR}=120 in .env for a longer wait.")
    except LLMResponseError as exc:
        print(f"\nBAD RESPONSE: {exc}")
    except LLMError as exc:
        print(f"\nERROR: {exc}")
    return 1


if __name__ == "__main__":  # pragma: no cover - exercised manually
    raise SystemExit(_main())
