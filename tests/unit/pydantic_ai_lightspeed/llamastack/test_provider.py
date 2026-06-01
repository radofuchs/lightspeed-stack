"""Unit tests for pydantic_ai_lightspeed.llamastack._provider module."""

# pylint: disable=protected-access

import httpx
import pytest
from openai import AsyncOpenAI
from pytest_mock import MockerFixture

from pydantic_ai_lightspeed.llamastack._provider import (
    DEFAULT_BASE_URL,
    LlamaStackProvider,
)


class TestLlamaStackProviderProperties:
    """Tests for LlamaStackProvider basic properties."""

    def test_name(self) -> None:
        """Test that the provider name is 'llama-stack'."""
        provider = LlamaStackProvider()
        assert provider.name == "llama-stack"

    def test_base_url_default(self) -> None:
        """Test that the default base URL matches the expected default."""
        provider = LlamaStackProvider()
        assert DEFAULT_BASE_URL in provider.base_url

    def test_client_returns_async_openai(self) -> None:
        """Test that the client property returns an AsyncOpenAI instance."""
        provider = LlamaStackProvider()
        assert isinstance(provider.client, AsyncOpenAI)

    def test_repr(self) -> None:
        """Test the string representation of the provider."""
        provider = LlamaStackProvider()
        result = repr(provider)
        assert "LlamaStackProvider" in result
        assert "llama-stack" in result

    def test_model_profile_known_model(self) -> None:
        """Test model_profile returns a profile for a known OpenAI model."""
        profile = LlamaStackProvider.model_profile("gpt-4o")
        assert profile is not None

    def test_model_profile_unknown_model(self) -> None:
        """Test model_profile returns a default profile for an unrecognized model."""
        profile = LlamaStackProvider.model_profile("totally-unknown-model-xyz")
        assert profile is not None


class TestLlamaStackProviderServerMode:
    """Tests for LlamaStackProvider server mode initialization."""

    def test_explicit_base_url(self) -> None:
        """Test that an explicit base_url is used."""
        provider = LlamaStackProvider(base_url="http://my-server:9999/v1")
        assert "my-server:9999" in provider.base_url

    def test_explicit_api_key(self) -> None:
        """Test that an explicit api_key is used."""
        provider = LlamaStackProvider(api_key="my-secret-key")
        assert provider.client.api_key == "my-secret-key"

    def test_default_api_key_is_not_needed(self) -> None:
        """Test that the default API key is 'not-needed'."""
        provider = LlamaStackProvider()
        assert provider.client.api_key == "not-needed"

    def test_custom_http_client(self) -> None:
        """Test that a provided http_client is used."""
        custom_client = httpx.AsyncClient()
        provider = LlamaStackProvider(http_client=custom_client)
        assert provider.client is not None


class TestLlamaStackProviderLibraryMode:
    """Tests for LlamaStackProvider library mode initialization."""

    def test_library_client_creates_transport(self, mocker: MockerFixture) -> None:
        """Test that providing a library_client sets up the transport-based client."""
        mock_lib_client = mocker.Mock()
        mock_lib_client.provider_data = None

        provider = LlamaStackProvider(library_client=mock_lib_client)

        assert provider._library_client is mock_lib_client
        assert "llama-stack-library" in provider.base_url

    def test_library_client_api_key_is_not_needed(self, mocker: MockerFixture) -> None:
        """Test that library mode sets the API key to 'not-needed'."""
        mock_lib_client = mocker.Mock()
        mock_lib_client.provider_data = None

        provider = LlamaStackProvider(library_client=mock_lib_client)

        assert provider.client.api_key == "not-needed"


class TestLlamaStackProviderMutualExclusion:
    """Tests for mutual exclusion between library_client and server mode options."""

    def test_library_client_and_base_url_raises(self, mocker: MockerFixture) -> None:
        """Test that providing both library_client and base_url raises AssertionError."""
        mock_lib_client = mocker.Mock()
        mock_lib_client.provider_data = None

        with pytest.raises(
            AssertionError,
            match="Cannot provide both `library_client` and `base_url`",
        ):
            LlamaStackProvider(
                library_client=mock_lib_client,
                base_url="http://localhost:8321/v1",
            )

    def test_library_client_and_api_key_raises(self, mocker: MockerFixture) -> None:
        """Test that providing both library_client and api_key raises AssertionError."""
        mock_lib_client = mocker.Mock()
        mock_lib_client.provider_data = None

        with pytest.raises(
            AssertionError,
            match="Cannot provide both `library_client` and `api_key`",
        ):
            LlamaStackProvider(
                library_client=mock_lib_client,
                api_key="my-key",
            )

    def test_library_client_and_http_client_raises(self, mocker: MockerFixture) -> None:
        """Test that providing both library_client and http_client raises AssertionError."""
        mock_lib_client = mocker.Mock()
        mock_lib_client.provider_data = None

        with pytest.raises(
            AssertionError,
            match="Cannot provide both `library_client` and `http_client`",
        ):
            LlamaStackProvider(
                library_client=mock_lib_client,
                http_client=httpx.AsyncClient(),
            )


class TestSetHttpClient:  # pylint: disable=too-few-public-methods
    """Tests for LlamaStackProvider._set_http_client."""

    def test_replaces_internal_http_client(self) -> None:
        """Test that _set_http_client replaces the underlying httpx client."""
        provider = LlamaStackProvider()
        new_client = httpx.AsyncClient()

        provider._set_http_client(new_client)

        assert provider._client._client is new_client
