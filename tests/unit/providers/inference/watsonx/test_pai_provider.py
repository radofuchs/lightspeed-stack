"""Unit tests for the WatsonX PAI provider."""

# pylint: disable=protected-access

from typing import Optional

import pytest
from pydantic_ai.exceptions import UserError

from providers.inference.watsonx.auth import WatsonxIAMAuth
from providers.inference.watsonx.pai_provider import (
    DEFAULT_WATSONX_BASE_URL,
    WATSONX_API_VERSION,
    WatsonxProvider,
)


def test_watsonx_provider_builds_client_from_config() -> None:
    """Builds AsyncOpenAI with WatsonX host, /ml/v1 once, version, and IAM auth."""
    provider = WatsonxProvider(
        api_key="test-key",
        base_url="https://us-south.ml.cloud.ibm.com/",
    )

    assert provider.name == "watsonx"
    assert provider.base_url == "https://us-south.ml.cloud.ibm.com"
    assert (
        str(provider.client.base_url).rstrip("/")
        == "https://us-south.ml.cloud.ibm.com/ml/v1"
    )
    assert provider.client.default_query == {"version": WATSONX_API_VERSION}
    assert isinstance(provider.client._client.auth, WatsonxIAMAuth)


def test_watsonx_provider_requires_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fails clearly when api_key cannot be resolved."""
    monkeypatch.delenv("WATSONX_API_KEY", raising=False)
    with pytest.raises(UserError, match="WATSONX_API_KEY"):
        WatsonxProvider(base_url="https://us-south.ml.cloud.ibm.com")


@pytest.mark.parametrize(
    ("explicit_base_url", "env_base_url", "expected_host"),
    [
        (
            "https://custom.example/watsonx/",
            None,
            "https://custom.example/watsonx",
        ),
        (
            None,
            "https://env.example/",
            "https://env.example",
        ),
        (
            None,
            None,
            DEFAULT_WATSONX_BASE_URL,
        ),
    ],
)
def test_watsonx_provider_resolves_host_and_appends_ml_v1_once(
    monkeypatch: pytest.MonkeyPatch,
    explicit_base_url: Optional[str],
    env_base_url: Optional[str],
    expected_host: str,
) -> None:
    """Resolves host from arg, env, or default; appends /ml/v1 once for the client."""
    monkeypatch.delenv("WATSONX_BASE_URL", raising=False)
    if env_base_url is not None:
        monkeypatch.setenv("WATSONX_BASE_URL", env_base_url)

    provider = WatsonxProvider(api_key="test-key", base_url=explicit_base_url)

    assert provider.base_url == expected_host
    assert str(provider.client.base_url).rstrip("/") == f"{expected_host}/ml/v1"


def test_watsonx_provider_uses_env_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Falls back to WATSONX_API_KEY when api_key is omitted."""
    monkeypatch.setenv("WATSONX_API_KEY", "env-key")
    monkeypatch.setenv("WATSONX_BASE_URL", "https://example.test")

    provider = WatsonxProvider()

    assert provider.base_url == "https://example.test"
    assert str(provider.client.base_url).rstrip("/") == "https://example.test/ml/v1"
    assert isinstance(provider.client._client.auth, WatsonxIAMAuth)


def test_model_profile_returns_openai_compatible_profile() -> None:
    """Returns a conservative OpenAI-compatible model profile."""
    profile = WatsonxProvider.model_profile("ibm/granite-3-8b-instruct")

    assert profile is not None
    assert profile.get("json_schema_transformer") is not None
