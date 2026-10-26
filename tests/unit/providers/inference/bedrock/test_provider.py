"""Unit tests for the Bedrock inference provider."""

# pylint: disable=redefined-outer-name,protected-access

from typing import Any

import pytest
from pydantic_ai.embeddings.bedrock import BedrockEmbeddingModel
from pydantic_ai.models.bedrock import BedrockConverseModel
from pydantic_ai.providers.bedrock import BedrockProvider

from models.config import BedrockInferenceProviderEntry
from providers.inference.bedrock.provider import (
    BedrockInferenceProvider,
    _BearerTokenSession,
)
from providers.inference.catalog import CatalogModel
from providers.inference.errors import ModelTypeNotSupportedError


def _entry(**config: Any) -> BedrockInferenceProviderEntry:
    """Build a Bedrock provider configuration."""
    return BedrockInferenceProviderEntry.model_validate(
        {
            "type": "bedrock",
            "id": "bedrock",
            "config": {
                "region": "us-east-1",
                "profile_name": "test-profile",
                **config,
            },
        }
    )


def _catalog_model(
    *,
    name: str,
    provider_model_id: str,
    model_type: str,
) -> CatalogModel:
    """Build a catalog model for create_* tests."""
    return CatalogModel(
        name=name,
        provider_model_id=provider_model_id,
        model_type=model_type,  # type: ignore[arg-type]
    )


@pytest.fixture
def provider() -> BedrockInferenceProvider:
    """Create a Bedrock provider with region and bearer credentials."""
    return BedrockInferenceProvider(_entry(api_key="test-bearer-token"))


def test_build_provider_passes_config(
    provider: BedrockInferenceProvider,
    mocker: Any,
) -> None:
    """Builds a Pydantic AI Bedrock provider from configured credentials."""
    built = mocker.MagicMock(spec=BedrockProvider)
    bedrock_provider_cls = mocker.patch(
        "providers.inference.bedrock.provider.BedrockProvider",
        return_value=built,
    )

    pai_provider = provider._build_provider()

    assert pai_provider is built
    bedrock_provider_cls.assert_called_once_with(
        region_name="us-east-1",
        profile_name="test-profile",
        api_key="test-bearer-token",
    )


def test_create_bedrock_client_uses_bearer_session(
    provider: BedrockInferenceProvider,
    mocker: Any,
) -> None:
    """Builds a control-plane client with bearer auth when a token is set."""
    client = mocker.MagicMock()
    session = mocker.MagicMock()
    session.client.return_value = client
    session_cls = mocker.patch(
        "providers.inference.bedrock.provider.boto3.Session",
        return_value=session,
    )

    assert provider._create_bedrock_client() is client

    session_kwargs = session_cls.call_args.kwargs
    assert session_kwargs["region_name"] == "us-east-1"
    assert session_kwargs["profile_name"] == "test-profile"
    assert isinstance(session_kwargs["botocore_session"], _BearerTokenSession)
    assert session_kwargs["botocore_session"].token == "test-bearer-token"
    assert session.client.call_args.args[0] == "bedrock"
    assert session.client.call_args.kwargs["config"].signature_version == "bearer"


def test_create_bedrock_client_uses_sigv4_without_token(
    mocker: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Builds a control-plane client with SigV4 when no bearer token is set."""
    monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)
    provider = BedrockInferenceProvider(_entry())
    client = mocker.MagicMock()
    session = mocker.MagicMock()
    session.client.return_value = client
    session_cls = mocker.patch(
        "providers.inference.bedrock.provider.boto3.Session",
        return_value=session,
    )

    assert provider._create_bedrock_client() is client

    session_cls.assert_called_once_with(
        profile_name="test-profile",
        region_name="us-east-1",
        botocore_session=None,
    )
    session.client.assert_called_once_with("bedrock", config=None)


@pytest.mark.asyncio
async def test_discover_models_types_and_skips_by_modality(
    provider: BedrockInferenceProvider,
    mocker: Any,
) -> None:
    """Keeps TEXT as llm and EMBEDDING as embedding; skips other modalities."""
    client = mocker.MagicMock()
    client.list_foundation_models.return_value = {
        "modelSummaries": [
            {
                "modelId": "anthropic.claude-3-haiku-20240307-v1:0",
                "outputModalities": ["TEXT"],
            },
            {
                "modelId": "amazon.titan-embed-text-v2:0",
                "outputModalities": ["EMBEDDING"],
            },
            {
                "modelId": "stability.stable-diffusion-xl-v1",
                "outputModalities": ["IMAGE"],
            },
            {"modelId": "custom-unknown"},
        ]
    }
    mocker.patch.object(provider, "_create_bedrock_client", return_value=client)

    discovered = await provider._discover_models()

    assert [(item.provider_model_id, item.model_type) for item in discovered] == [
        ("anthropic.claude-3-haiku-20240307-v1:0", "llm"),
        ("amazon.titan-embed-text-v2:0", "embedding"),
    ]
    client.list_foundation_models.assert_called_once_with()


def test_create_llm_uses_bedrock_converse_model(
    provider: BedrockInferenceProvider,
    mocker: Any,
) -> None:
    """Creates a Bedrock Converse model using the provider model ID."""
    pai_provider = mocker.MagicMock(spec=BedrockProvider)
    provider._provider = pai_provider

    model = provider._create_llm(
        _catalog_model(
            name="chat",
            provider_model_id="anthropic.claude-3-haiku-20240307-v1:0",
            model_type="llm",
        )
    )

    assert isinstance(model, BedrockConverseModel)
    assert model.model_name == "anthropic.claude-3-haiku-20240307-v1:0"
    assert model.provider is pai_provider


def test_create_embedding_uses_bedrock_embedding_model(
    provider: BedrockInferenceProvider,
    mocker: Any,
) -> None:
    """Creates a Bedrock embedding model using the provider model ID."""
    pai_provider = mocker.MagicMock(spec=BedrockProvider)
    provider._provider = pai_provider

    model = provider._create_embedding(
        _catalog_model(
            name="embed",
            provider_model_id="amazon.titan-embed-text-v2:0",
            model_type="embedding",
        )
    )

    assert isinstance(model, BedrockEmbeddingModel)
    assert model.model_name == "amazon.titan-embed-text-v2:0"
    assert model._provider is pai_provider


def test_create_rerank_is_unsupported(
    provider: BedrockInferenceProvider,
) -> None:
    """Rejects reranking because the Bedrock provider does not support it."""
    with pytest.raises(ModelTypeNotSupportedError) as exc_info:
        provider._create_rerank(
            _catalog_model(
                name="rerank",
                provider_model_id="rerank-model",
                model_type="rerank",
            )
        )

    assert exc_info.value.model_type == "rerank"
    assert exc_info.value.provider_id == "bedrock"
