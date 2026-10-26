"""IBM IAM authentication for WatsonX HTTP clients."""

from __future__ import annotations

import time
from collections.abc import AsyncGenerator
from typing import Optional

import httpx
from pydantic import SecretStr

from log import get_logger

logger = get_logger(__name__)

IAM_TOKEN_URL = "https://iam.cloud.ibm.com/identity/token"
IAM_GRANT_TYPE = "urn:ibm:params:oauth:grant-type:apikey"
TOKEN_EXPIRATION_LEEWAY_SECONDS = 60


class WatsonxIAMAuth(httpx.Auth):
    """Exchange a WatsonX API key for short-lived IBM IAM bearer tokens.

    WatsonX rejects the raw API key on inference requests. This auth layer
    exchanges the key at IAM, caches the bearer token, and attaches it to
    each outbound HTTP request. The OpenAI-compatible client is never
    recreated when the token rotates.

    Concurrent refreshes are safe: IAM access tokens from the same API key
    remain independently valid until each expires.
    """

    requires_response_body = False

    def __init__(self, api_key: str) -> None:
        """Initialize IAM auth for a single API key.

        Args:
            api_key: WatsonX API key to exchange for IAM tokens.

        """
        self._api_key = api_key
        self._token: Optional[SecretStr] = None
        self._expires_at: float = 0.0

    async def async_auth_flow(
        self, request: httpx.Request
    ) -> AsyncGenerator[httpx.Request, httpx.Response]:
        """Attach a valid IAM bearer token to the outgoing request.

        Args:
            request: Outgoing HTTP request to authenticate.

        Yields:
            The request with an ``Authorization: Bearer`` header set.

        """
        token = await self.get_token()
        request.headers["Authorization"] = f"Bearer {token}"
        yield request

    async def get_token(self) -> str:
        """Return a cached IAM token, refreshing when near expiry.

        Returns:
            A valid IBM IAM access token.

        """
        if (cached := self._cached_token()) is not None:
            return cached
        return await self._exchange_token()

    def _cached_token(self) -> Optional[str]:
        """Return the cached token when present and not near expiry.

        Returns:
            The cached access token, or ``None`` when a refresh is needed.

        """
        if self._token is None:
            return None
        if time.time() >= self._expires_at - TOKEN_EXPIRATION_LEEWAY_SECONDS:
            return None
        return self._token.get_secret_value()

    async def _exchange_token(self) -> str:
        """Exchange the API key for a new IAM access token.

        Returns:
            A newly issued IBM IAM access token.

        """
        logger.info("Refreshing WatsonX IAM access token")
        async with httpx.AsyncClient() as client:
            response = await client.post(
                IAM_TOKEN_URL,
                data={
                    "grant_type": IAM_GRANT_TYPE,
                    "apikey": self._api_key,
                },
                timeout=httpx.Timeout(10.0, connect=5.0),
            )
            response.raise_for_status()
            data = response.json()

        token = data["access_token"]
        expires_at = float(data["expiration"])
        self._token = SecretStr(token)
        self._expires_at = expires_at
        logger.info(
            "WatsonX IAM access token refreshed, expires at %s",
            time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(expires_at)),
        )
        return token
