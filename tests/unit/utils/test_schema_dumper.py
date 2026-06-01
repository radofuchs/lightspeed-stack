"""Unit tests for utils/schema_dumper module."""

from typing import Any

from utils.schema_dumper import recursive_update


def test_update_empty_input() -> None:
    """Test how recursive_update function transforms empty input."""
    original: dict[str, Any] = {}
    expected: dict[str, Any] = {}

    # perform the update
    result = recursive_update(original)

    # empty dict should be returned
    assert result == expected

    # ensure a new dict is returned, not the same object
    assert result is not original


def test_no_change_for_simple_schema() -> None:
    """Test how recursive_update function trasforms simple non-empty input."""
    original: dict[str, Any] = {
        "type": "string",
        "maxLength": 10,
    }

    # we need to distinguish between original and a copy
    expected = original.copy()

    # perform the update
    result = recursive_update(original)

    # non-empty dict with known content should be returned
    assert result == expected

    # ensure a new dict is returned, not the same object
    assert result is not original


def test_no_change_for_simple_object() -> None:
    """Test how recursive_update function trasforms simple non-empty input."""
    original: dict[str, Any] = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
        },
    }

    # we need to distinguish between original and a copy
    expected = original.copy()

    # perform the update
    result = recursive_update(original)

    # non-empty dict with known content should be returned
    assert result == expected

    # ensure a new dict is returned, not the same object
    assert result is not original


def test_recursive_recurse_into_subdicts() -> None:
    """Test the recursive_update on input containing sub-dictionaries."""
    original = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer", "exclusiveMinimum": 0},
        },
    }
    expected = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer", "minimum": 0},
        },
    }

    # perform the update
    result = recursive_update(original)

    # non-empty dict with known content should be returned
    assert result == expected

    # ensure a new dict is returned, not the same object
    assert result is not original


def test_handles_none_values() -> None:
    """None values should be preserved."""
    original = {"key": None}
    expected = original.copy()

    # perform the update
    result = recursive_update(original)

    # non-empty dict with known content should be returned
    assert result == expected
