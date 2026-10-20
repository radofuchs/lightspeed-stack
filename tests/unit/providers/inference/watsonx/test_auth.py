"""Unit tests for WatsonX IAM authentication."""

# pylint: disable=protected-access

import time
from typing import Any, Optional

import httpx
import pytest

from providers.inference.watsonx.auth import (
    IAM_GRANT_TYPE,
    IAM_TOKEN_URL,
    TOKEN_EXPIRATION_LEEWAY_SECONDS,
    WatsonxIAMAuth,
)


def _token_response(
    mocker: Any,
    *,
    access_token: str = "iam-token",
    expiration: Optional[float] = None,
) -> Any:
    """Build a mocked IAM token HTTP response."""
    response = mocker.MagicMock()
    response.raise_for_status = mocker.MagicMock()
    response.json.return_value = {
        "access_token": access_token,
        "expiration": expiration if expiration is not None else time.time() + 3600,
    }
    return response


def _mock_token_exchange(mocker: Any, *responses: Any) -> Any:
    """Patch httpx.AsyncClient and return the mocked post coroutine."""
    post = (
        mocker.AsyncMock(return_value=responses[0])
        if len(responses) == 1
        else mocker.AsyncMock(side_effect=responses)
    )
    mock_client = mocker.MagicMock()
    mock_client.post = post
    mock_client.__aenter__ = mocker.AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = mocker.AsyncMock(return_value=None)
    mocker.patch(
        "providers.inference.watsonx.auth.httpx.AsyncClient",
        return_value=mock_client,
    )
    return post


@pytest.mark.asyncio
async def test_get_token_exchanges_api_key_and_caches(mocker: Any) -> None:
    """Exchanges the API key once and reuses the cached token."""
    post = _mock_token_exchange(mocker, _token_response(mocker, access_token="token-1"))
    auth = WatsonxIAMAuth("api-key")

    first = await auth.get_token()
    second = await auth.get_token()

    assert first == "token-1"
    assert second == "token-1"
    assert post.await_count == 1
    assert post.await_args.args == (IAM_TOKEN_URL,)
    assert post.await_args.kwargs["data"] == {
        "grant_type": IAM_GRANT_TYPE,
        "apikey": "api-key",
    }


@pytest.mark.asyncio
async def test_get_token_refreshes_near_expiry(mocker: Any) -> None:
    """Refreshes the token when it is within the expiry leeway window."""
    near_expiry = time.time() + TOKEN_EXPIRATION_LEEWAY_SECONDS - 1
    post = _mock_token_exchange(
        mocker,
        _token_response(mocker, access_token="old-token", expiration=near_expiry),
        _token_response(mocker, access_token="new-token"),
    )
    auth = WatsonxIAMAuth("api-key")

    first = await auth.get_token()
    second = await auth.get_token()

    assert first == "old-token"
    assert second == "new-token"
    assert post.await_count == 2


@pytest.mark.asyncio
async def test_async_auth_flow_injects_bearer_header(mocker: Any) -> None:
    """Injects Authorization Bearer header from the IAM token."""
    auth = WatsonxIAMAuth("api-key")
    mocker.patch.object(auth, "get_token", mocker.AsyncMock(return_value="iam-token"))
    request = httpx.Request("GET", "https://example.test/ml/v1/models")

    flow = auth.async_auth_flow(request)
    authed_request = await flow.__anext__()

    assert authed_request.headers["Authorization"] == "Bearer iam-token"
    await flow.aclose()
