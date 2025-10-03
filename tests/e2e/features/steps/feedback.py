"""Implementation of common test steps for the feedback API."""

from behave import given, when, step  # pyright: ignore[reportAttributeAccessIssue]
from behave.runner import Context
import requests
import json
from tests.e2e.utils.utils import switch_config, restart_container
from tests.e2e.features.steps.common_http import access_rest_api_endpoint_get

# default timeout for HTTP operations
DEFAULT_TIMEOUT = 10


@step("The feedback is enabled")  # type: ignore
def enable_feedback(context: Context) -> None:
    """Enable the feedback endpoint and assert success."""
    assert context is not None
    payload = {"status": True}
    access_feedback_put_endpoint(context, payload)
    assert context.response.status_code == 200, "Enabling feedback was unsuccessful"


@step("The feedback is disabled")  # type: ignore
def disable_feedback(context: Context) -> None:
    """Disable the feedback endpoint and assert success."""
    assert context is not None
    payload = {"status": False}
    access_feedback_put_endpoint(context, payload)
    assert context.response.status_code == 200, "Disabling feedback was unsuccessful"


@when("I update feedback status with")  # type: ignore
def set_feedback(context: Context) -> None:
    """Enable or disable feedback via PUT request."""
    assert context.text is not None, "Payload needs to be specified"
    payload = json.loads(context.text or "{}")
    access_feedback_put_endpoint(context, payload)


def access_feedback_put_endpoint(context: Context, payload: dict) -> None:
    """Update feedback using a JSON payload."""
    assert context is not None
    endpoint = "feedback/status"
    base = f"http://{context.hostname}:{context.port}"
    path = f"{context.api_prefix}/{endpoint}".replace("//", "/")
    url = base + path
    headers = context.auth_headers if hasattr(context, "auth_headers") else {}
    response = requests.put(url, headers=headers, json=payload)
    context.response = response


@when("I submit the following feedback for the conversation created before")  # type: ignore
def submit_feedback_valid_conversation(context: Context) -> None:
    """Submit feedback for previousl created conversation."""
    assert (
        hasattr(context, "conversation_id") and context.conversation_id is not None
    ), "Conversation for feedback submission is not created"
    access_feedback_post_endpoint(context, context.conversation_id)


@when('I submit the following feedback for nonexisting conversation "{conversation_id}"')  # type: ignore
def submit_feedback_nonexisting_conversation(
    context: Context, conversation_id: str
) -> None:
    """Submit feedback for a non-existing conversation ID."""
    access_feedback_post_endpoint(context, conversation_id)


@when("I submit the following feedback without specifying conversation ID")  # type: ignore
def submit_feedback_without_conversation(context: Context) -> None:
    """Submit feedback with no conversation ID."""
    access_feedback_post_endpoint(context, None)


def access_feedback_post_endpoint(
    context: Context, conversation_id: str | None
) -> None:
    """Send POST HTTP request with JSON payload to tested service."""
    endpoint = "feedback"
    base = f"http://{context.hostname}:{context.port}"
    path = f"{context.api_prefix}/{endpoint}".replace("//", "/")
    url = base + path
    payload = json.loads(context.text or "{}")
    if conversation_id is not None:
        payload["conversation_id"] = conversation_id
    headers = context.auth_headers if hasattr(context, "auth_headers") else {}
    context.response = requests.post(url, headers=headers, json=payload)


@when("I retreive the current feedback status")  # type: ignore
def access_feedback_get_endpoint(context: Context) -> None:
    """Retrieve the current feedback status via GET request."""
    access_rest_api_endpoint_get(context, "feedback/status")


@given("A new conversation is initialized")  # type: ignore
def initialize_conversation(context: Context) -> None:
    """Create a conversation for submitting feedback."""
    endpoint = "query"
    base = f"http://{context.hostname}:{context.port}"
    path = f"{context.api_prefix}/{endpoint}".replace("//", "/")
    url = base + path
    headers = context.auth_headers if hasattr(context, "auth_headers") else {}
    payload = {"query": "Say Hello.", "system_prompt": "You are a helpful assistant"}

    response = requests.post(url, headers=headers, json=payload)
    assert (
        response.status_code == 200
    ), f"Failed to create conversation: {response.text}"

    body = response.json()
    context.conversation_id = body["conversation_id"]
    assert context.conversation_id, "Conversation was not created."
    context.feedback_conversations.append(context.conversation_id)
    context.response = response


@given("An invalid feedback storage path is configured")  # type: ignore
def configure_invalid_feedback_storage_path(context: Context) -> None:
    """Set an invalid feedback storage path and restart the container."""
    switch_config(context.scenario_config)
    restart_container("lightspeed-stack")
