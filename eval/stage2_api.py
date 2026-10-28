"""Configuration and client for the external second-stage judge API.

The evaluation protocol has two distinct stages: a local/evaluated model first
generates responses, and an external judge then scores those responses. This
module contains the opt-in configuration for the latter stage when the judge
is the explicitly selected model exposed by the configured OpenAI-compatible
endpoint.

The API key is deliberately read only from the process environment. It is
never accepted as a command-line argument, persisted, or included in log
messages. The default model is the endpoint's Luna deployment
(`gpt-5.6-luna`); callers may explicitly select another advertised model
through the stage-2 model environment variable. A failed request is surfaced
to the caller and is never retried with a different model or endpoint.
"""

from __future__ import annotations

import os
import requests
from dataclasses import dataclass
from typing import Any, Iterable


DEFAULT_STAGE2_API_BASE_URL = "https://sub.anzhiyu.com/v1"
DEFAULT_STAGE2_API_MODEL = "gpt-5.6-luna"
STAGE2_API_BASE_URL_ENV = "STAGE2_API_BASE_URL"
STAGE2_API_MODEL_ENV = "STAGE2_API_MODEL"
STAGE2_API_KEY_ENV = "STAGE2_API_KEY"


class Stage2APIError(RuntimeError):
    """A safe, actionable error from the stage-2 endpoint."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


def _redact_secret(text: str, secret: str | None) -> str:
    """Remove the configured bearer token from an arbitrary error string."""

    if not secret:
        return text
    return text.replace(secret, "<redacted-api-key>")


def _exception_status_code(error: BaseException) -> int | None:
    response = getattr(error, "response", None)
    status = getattr(response, "status_code", None)
    if isinstance(status, int):
        return status
    status = getattr(error, "status_code", None)
    return status if isinstance(status, int) else None


@dataclass(frozen=True)
class Stage2APIConfig:
    """Resolved stage-2 endpoint configuration.

    The api_key is held in memory only for constructing the SDK client. Callers
    should not serialize or print this object.
    """

    base_url: str
    model: str
    api_key: str
    timeout: float = 240.0

    @classmethod
    def from_env(
        cls,
        *,
        timeout: float = 240.0,
        environ: dict[str, str] | None = None,
    ) -> "Stage2APIConfig":
        env = os.environ if environ is None else environ
        base_url = (env.get(STAGE2_API_BASE_URL_ENV) or DEFAULT_STAGE2_API_BASE_URL).strip()
        if not base_url:
            raise Stage2APIError(f"{STAGE2_API_BASE_URL_ENV} must not be empty")
        base_url = base_url.rstrip("/")

        model = (env.get(STAGE2_API_MODEL_ENV) or DEFAULT_STAGE2_API_MODEL).strip()
        if not model:
            raise Stage2APIError(f"{STAGE2_API_MODEL_ENV} must not be empty")

        # STAGE2_API_KEY is the explicit profile variable. OPENAI_API_KEY is
        # accepted as a compatibility fallback because it is also an
        # environment-only secret in the existing evaluation launchers.
        api_key = (env.get(STAGE2_API_KEY_ENV) or env.get("OPENAI_API_KEY") or "").strip()
        if not api_key:
            raise Stage2APIError(
                f"{STAGE2_API_KEY_ENV} (or OPENAI_API_KEY) is required; "
                "the API key must be supplied through the environment"
            )
        if timeout <= 0:
            raise Stage2APIError("stage-2 API timeout must be positive")
        return cls(base_url=base_url, model=model, api_key=api_key, timeout=float(timeout))


def _iter_model_ids(payload: Any) -> Iterable[str]:
    data = getattr(payload, "data", None)
    if data is None and isinstance(payload, dict):
        data = payload.get("data")
    if not isinstance(data, (list, tuple)):
        return ()
    ids: list[str] = []
    for item in data:
        model_id = getattr(item, "id", None)
        if model_id is None and isinstance(item, dict):
            model_id = item.get("id")
        if model_id is not None:
            ids.append(str(model_id))
    return ids


class Stage2ExternalAPIClient:
    """Single-endpoint, explicitly selected-model stage-2 client.

    The client intentionally creates one SDK client and never rotates through
    a pool of URLs or model names. complete may be called repeatedly by a
    higher-level evaluator, but every request remains pinned to this exact
    endpoint and the configured model. Model availability is checked before
    the first completion; an unavailable requested model is reported instead
    of being replaced by another model.
    """

    def __init__(self, config: Stage2APIConfig) -> None:
        try:
            from openai import OpenAI
        except Exception as exc:  # pragma: no cover - dependency-specific
            raise Stage2APIError("the openai package is required for stage-2 API judging") from exc
        self.config = config
        try:
            self.client = OpenAI(
                api_key=config.api_key,
                base_url=config.base_url,
                timeout=config.timeout,
                # Retries are owned by the evaluator, if desired. The SDK
                # must not hide a model/endpoint error or select a fallback.
                max_retries=0,
            )
        except Exception as exc:  # pragma: no cover - dependency-specific
            raise Stage2APIError(
                _redact_secret(f"could not initialize stage-2 API client: {exc}", config.api_key),
                status_code=_exception_status_code(exc),
            ) from exc

    @classmethod
    def from_env(cls, *, timeout: float = 240.0) -> "Stage2ExternalAPIClient":
        return cls(Stage2APIConfig.from_env(timeout=timeout))

    def check_model(self) -> tuple[str, ...]:
        """Check reachability and require the configured model to be advertised."""

        try:
            models = self.client.models.list()
        except Exception as exc:
            # Leonardo can reset the TLS session used by the bundled httpx
            # transport while accepting the same authenticated request from
            # curl/requests. Retry through requests before surfacing failure.
            try:
                response = requests.get(
                    f"{self.config.base_url}/models",
                    headers={"Authorization": f"Bearer {self.config.api_key}"},
                    timeout=self.config.timeout,
                )
                response.raise_for_status()
                payload = response.json()
                model_ids = tuple(_iter_model_ids(payload))
            except Exception as fallback_exc:
                message = _redact_secret(
                    f"stage-2 endpoint {self.config.base_url}/models failed: {fallback_exc}",
                    self.config.api_key,
                )
                raise Stage2APIError(message, status_code=_exception_status_code(fallback_exc)) from exc
        else:
            model_ids = tuple(_iter_model_ids(models))
        if self.config.model not in model_ids:
            available = ", ".join(model_ids) if model_ids else "<none reported>"
            raise Stage2APIError(
                f"stage-2 endpoint {self.config.base_url} does not advertise model "
                f"{self.config.model!r}; available model ids: {available}"
            )
        return model_ids

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_tokens: int = 2048,
    ) -> str:
        """Call the configured stage-2 model once and return its content.

        No fallback model, fallback endpoint, or silent empty response is used.
        The original provider error is retained in Stage2APIError so a caller
        can report the concrete HTTP/API failure.
        """

        try:
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                timeout=self.config.timeout,
            )
            content = response.choices[0].message.content or ""
        except Exception as exc:
            try:
                response = requests.post(
                    f"{self.config.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.config.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.config.model,
                        "messages": messages,
                        "temperature": temperature,
                        "top_p": top_p,
                        "max_tokens": max_tokens,
                    },
                    timeout=self.config.timeout,
                )
                response.raise_for_status()
                content = response.json()["choices"][0]["message"].get("content") or ""
            except Exception as fallback_exc:
                message = _redact_secret(
                    f"stage-2 completion at {self.config.base_url} failed: {fallback_exc}",
                    self.config.api_key,
                )
                raise Stage2APIError(message, status_code=_exception_status_code(fallback_exc)) from exc
        if not isinstance(content, str) or not content.strip():
            raise Stage2APIError(
                f"stage-2 completion at {self.config.base_url} returned empty content"
            )
        return content
