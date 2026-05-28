"""Behave steps for /v1/prompts endpoint end-to-end tests."""

from __future__ import annotations

import json
from typing import Any

import requests
from behave import (  # pyright: ignore[reportAttributeAccessIssue]
    step,
    then,
    when,
)
from behave.runner import Context

from tests.e2e.utils.utils import normalize_endpoint, request_with_transient_retry

DEFAULT_TIMEOUT = 10


def _prompts_url(context: Context, endpoint: str) -> str:
    """Build full URL for a prompts REST path under ``context.api_prefix``."""
    base = f"http://{context.hostname}:{context.port}"
    path = f"{context.api_prefix}/{normalize_endpoint(endpoint)}".replace("//", "/")
    return base + path


def _auth_headers(context: Context) -> dict[str, str]:
    """Return auth headers from context when present."""
    if hasattr(context, "auth_headers"):
        return context.auth_headers
    return {}


def _request_prompts_with_stored_id(context: Context, method: str) -> None:
    """Call prompts endpoint with prompt id stored from a previous response."""
    assert hasattr(
        context, "stored_prompt_id"
    ), "stored_prompt_id not set; run prompt creation first"
    endpoint = normalize_endpoint(f"prompts/{context.stored_prompt_id}")
    headers = _auth_headers(context)

    if method in ("GET", "DELETE"):
        context.response = request_with_transient_retry(
            method=method,
            url=_prompts_url(context, endpoint),
            headers=headers,
            timeout=DEFAULT_TIMEOUT,
        )
        return

    assert context.text is not None, "Payload needs to be specified"
    data = json.loads(context.text)
    context.response = request_with_transient_retry(
        method=method,
        url=_prompts_url(context, endpoint),
        json=data,
        headers=headers,
        timeout=DEFAULT_TIMEOUT,
    )

@step("I store the prompt_id from the last response")  # type: ignore[reportCallIssue]
def store_prompt_id(context: Context) -> None:
    """Store ``prompt_id`` from the latest JSON response."""
    assert context.response is not None, "Request needs to be performed first"
    body: dict[str, Any] = context.response.json()
    assert "prompt_id" in body, f"prompt_id not found in response body: {body}"
    context.stored_prompt_id = body["prompt_id"]
    if "version" in body:
        context.stored_prompt_version = body["version"]


@when(  # type: ignore[reportCallIssue]
    "I access REST API prompts endpoint with stored prompt id using HTTP GET method"
)
def get_prompt_by_stored_id(context: Context) -> None:
    """GET /v1/prompts/{stored_prompt_id}."""
    _request_prompts_with_stored_id(context, "GET")


@when(  # type: ignore[reportCallIssue]
    "I access REST API prompts endpoint with stored prompt id using HTTP PUT method"
)
def update_prompt_by_stored_id(context: Context) -> None:
    """PUT /v1/prompts/{stored_prompt_id}."""
    _request_prompts_with_stored_id(context, "PUT")


@when(  # type: ignore[reportCallIssue]
    "I access REST API prompts endpoint with stored prompt id using HTTP DELETE method"
)
def delete_prompt_by_stored_id(context: Context) -> None:
    """DELETE /v1/prompts/{stored_prompt_id}."""
    _request_prompts_with_stored_id(context, "DELETE")


@when(  # type: ignore[reportCallIssue]
    "I access REST API prompts endpoint with stored prompt id and version {version:d} using HTTP GET method"
)
def get_prompt_by_stored_id_and_version(context: Context, version: int) -> None:
    """GET /v1/prompts/{stored_prompt_id}?version={version}."""
    assert hasattr(
        context, "stored_prompt_id"
    ), "stored_prompt_id not set; run prompt creation first"
    endpoint = normalize_endpoint(f"prompts/{context.stored_prompt_id}")
    context.response = requests.get(
        _prompts_url(context, endpoint),
        params={"version": version},
        headers=_auth_headers(context),
        timeout=DEFAULT_TIMEOUT,
    )


@then("The prompt_id in the response matches the stored prompt id")  # type: ignore[reportCallIssue]
def response_prompt_id_matches_stored(context: Context) -> None:
    """Assert response ``prompt_id`` equals the stored prompt id."""
    assert context.response is not None, "Request needs to be performed first"
    assert hasattr(context, "stored_prompt_id"), "stored_prompt_id not set"
    body: dict[str, Any] = context.response.json()
    assert body["prompt_id"] == context.stored_prompt_id, (
        f"Expected prompt_id {context.stored_prompt_id!r}, "
        f"got {body.get('prompt_id')!r}"
    )


@then("The prompt version in the response is {expected_version:d}")  # type: ignore[reportCallIssue]
def prompt_version_matches(context: Context, expected_version: int) -> None:
    """Assert response ``version`` equals the expected value."""
    assert context.response is not None, "Request needs to be performed first"
    body: dict[str, Any] = context.response.json()
    assert body["version"] == expected_version, (
        f"Expected version {expected_version}, got {body.get('version')}"
    )


@then("The prompts list contains the stored prompt id")  # type: ignore[reportCallIssue]
def prompt_list_contains_stored_prompt(context: Context) -> None:
    """Assert one entry in ``data`` has the stored prompt id."""
    assert context.response is not None, "Request needs to be performed first"
    assert hasattr(context, "stored_prompt_id"), "stored_prompt_id not set"
    body: dict[str, Any] = context.response.json()
    prompts = body.get("data", [])
    assert isinstance(prompts, list), f"Expected data list, got: {type(prompts)}"
    assert any(p.get("prompt_id") == context.stored_prompt_id for p in prompts), (
        f"prompt_id {context.stored_prompt_id!r} not found in prompts list: {prompts}"
    )
