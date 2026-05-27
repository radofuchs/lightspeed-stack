"""httpx transport that routes OpenAI-compatible requests through a Llama Stack library client."""

from __future__ import annotations as _annotations

import json
from collections.abc import AsyncGenerator, AsyncIterator
from typing import Any

import httpx
from llama_stack.core.library_client import (
    AsyncLlamaStackAsLibraryClient,
    convert_pydantic_to_json_value,
)
from llama_stack.core.request_headers import (
    PROVIDER_DATA_VAR,
    request_provider_data_context,
)
from llama_stack.core.server.routes import find_matching_route
from llama_stack.core.utils.context import preserve_contexts_async_generator


class _AsyncByteStream(httpx.AsyncByteStream):
    """Wraps an async byte generator as an httpx AsyncByteStream."""

    def __init__(self, gen: AsyncGenerator[bytes, None]) -> None:
        self._gen = gen

    async def __aiter__(self) -> AsyncIterator[bytes]:
        async for chunk in self._gen:
            yield chunk


class LlamaStackLibraryTransport(httpx.AsyncBaseTransport):
    """Custom httpx transport that dispatches requests through a Llama Stack library client.

    Instead of making real HTTP calls, this transport routes requests directly
    to the Llama Stack's in-process route handlers via the library client's
    route matching and body conversion logic.
    """

    def __init__(self, client: AsyncLlamaStackAsLibraryClient) -> None:
        """Initialize the transport with a Llama Stack library client.

        Args:
            client: An initialized ``AsyncLlamaStackAsLibraryClient`` whose route
                handlers will receive dispatched requests.
        """
        self._client = client

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        """Dispatch an httpx request to the in-process Llama Stack route handlers.

        Args:
            request: The outgoing httpx request to route.

        Returns:
            An httpx response built from the matched route handler result.

        Raises:
            RuntimeError: If the library client has not been initialized.
        """
        if self._client.route_impls is None:
            raise RuntimeError(
                "Llama Stack library client not initialized. Call initialize() first."
            )

        method = request.method
        path = request.url.raw_path.decode("utf-8")

        body = json.loads(request.content) if request.content else {}

        headers: dict[str, str] = {
            k.decode("utf-8") if isinstance(k, bytes) else k: (
                v.decode("utf-8") if isinstance(v, bytes) else v
            )
            for k, v in request.headers.raw
        }

        if self._client.provider_data:
            keys = ["X-LlamaStack-Provider-Data", "x-llamastack-provider-data"]
            if all(key not in headers for key in keys):
                headers["X-LlamaStack-Provider-Data"] = json.dumps(
                    self._client.provider_data
                )

        with request_provider_data_context(headers):
            is_stream = body.get("stream", False)

            if is_stream:
                return await self._handle_streaming(request, method, path, body)
            return await self._handle_non_streaming(request, method, path, body)

    async def _handle_non_streaming(
        self,
        request: httpx.Request,
        method: str,
        path: str,
        body: dict[str, Any],
    ) -> httpx.Response:
        assert self._client.route_impls is not None

        matched_func, path_params, _, _ = find_matching_route(
            method, path, self._client.route_impls
        )
        body |= path_params
        body = self._client._convert_body(  # pylint: disable=protected-access
            matched_func, body
        )

        result = await matched_func(**body)

        json_content = json.dumps(convert_pydantic_to_json_value(result))
        status_code = httpx.codes.OK

        if method.upper() == "DELETE" and result is None:
            status_code = httpx.codes.NO_CONTENT
            json_content = ""

        return httpx.Response(
            status_code=status_code,
            content=json_content.encode("utf-8"),
            headers={"Content-Type": "application/json"},
            request=request,
        )

    async def _handle_streaming(
        self,
        request: httpx.Request,
        method: str,
        path: str,
        body: dict[str, Any],
    ) -> httpx.Response:
        assert self._client.route_impls is not None

        func, path_params, _, _ = find_matching_route(
            method, path, self._client.route_impls
        )
        body |= path_params
        body = self._client._convert_body(  # pylint: disable=protected-access
            func, body
        )

        result = await func(**body)

        async def gen() -> AsyncGenerator[bytes, None]:
            async for chunk in result:
                data = json.dumps(convert_pydantic_to_json_value(chunk))
                yield f"data: {data}\n\n".encode("utf-8")

        wrapped_gen = preserve_contexts_async_generator(gen(), [PROVIDER_DATA_VAR])

        return httpx.Response(
            status_code=httpx.codes.OK,
            stream=_AsyncByteStream(wrapped_gen),
            headers={"Content-Type": "text/event-stream"},
            request=request,
        )
