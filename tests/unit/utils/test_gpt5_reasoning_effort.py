"""Tests for GPT-5 tool-call reasoning_effort workaround."""

from llama_stack_api.openai_responses import (
    OpenAIResponseInputToolFileSearch as InputToolFileSearch,
)
from llama_stack_api.openai_responses import (
    OpenAIResponseReasoning as Reasoning,
)

from models.common.responses.responses_api_params import ResponsesApiParams
from utils.responses import apply_gpt5_tool_call_reasoning_effort

_FILE_SEARCH_TOOL = InputToolFileSearch(type="file_search", vector_store_ids=["vs1"])


def _base_params(**overrides: object) -> ResponsesApiParams:
    defaults = {
        "input": "hello",
        "model": "openai/gpt-5.6-luna",
        "conversation": "conv-1",
        "stream": False,
        "store": True,
    }
    defaults.update(overrides)
    return ResponsesApiParams(**defaults)  # type: ignore[arg-type]


def test_sets_reasoning_none_when_gpt5_and_tools() -> None:
    """GPT-5 models with tools get reasoning.effort=none for LLS chat completions."""
    params = _base_params(tools=[_FILE_SEARCH_TOOL])
    updated = apply_gpt5_tool_call_reasoning_effort(params)
    assert updated.reasoning is not None
    assert updated.reasoning.effort == "none"


def test_no_change_without_tools() -> None:
    """No reasoning field is added when the request has no tools."""
    params = _base_params()
    updated = apply_gpt5_tool_call_reasoning_effort(params)
    assert updated.reasoning is None


def test_no_change_for_non_gpt5_model() -> None:
    """Non GPT-5 models are unchanged even when tools are present."""
    params = _base_params(model="openai/gpt-4o-mini", tools=[_FILE_SEARCH_TOOL])
    updated = apply_gpt5_tool_call_reasoning_effort(params)
    assert updated.reasoning is None


def test_preserves_explicit_non_none_reasoning() -> None:
    """Caller-requested reasoning effort other than none is not overwritten."""
    params = _base_params(
        tools=[_FILE_SEARCH_TOOL],
        reasoning=Reasoning(effort="high"),
    )
    updated = apply_gpt5_tool_call_reasoning_effort(params)
    assert updated.reasoning is not None
    assert updated.reasoning.effort == "high"
