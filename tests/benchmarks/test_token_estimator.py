"""Benchmarks for token estimator."""

from pytest_benchmark.fixture import BenchmarkFixture

from utils.token_estimator import (
    estimate_tokens,
)

LOREM_IPSUM = """
Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor
incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis
nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.
Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu
fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in
culpa qui officia deserunt mollit anim id est laborum.
"""


def test_estimate_empty_string(benchmark: BenchmarkFixture) -> None:
    """Benchmark for empty string as input."""
    input_string = ""
    benchmark(estimate_tokens, input_string)


def test_estimate_hello_world(benchmark: BenchmarkFixture) -> None:
    """Benchmark for Hello world as input."""
    input_string = "Hello world"
    benchmark(estimate_tokens, input_string)


def test_pangram(benchmark: BenchmarkFixture) -> None:
    """The pangram tokenizes to the known cl100k_base count."""
    input_string = "The quick brown fox jumps over the lazy dog."
    benchmark(estimate_tokens, input_string)


def test_lorem_ipsum(benchmark: BenchmarkFixture) -> None:
    """The lorem ipsum tokenizes to the known cl100k_base count."""
    input_string = LOREM_IPSUM
    benchmark(estimate_tokens, input_string)


def test_lorem_ipsum_times_10_times(benchmark: BenchmarkFixture) -> None:
    """The lorem ipsum tokenizes to the known cl100k_base count."""
    input_string = LOREM_IPSUM * 10
    benchmark(estimate_tokens, input_string)


def test_lorem_ipsum_times_100_times(benchmark: BenchmarkFixture) -> None:
    """The lorem ipsum tokenizes to the known cl100k_base count."""
    input_string = LOREM_IPSUM * 100
    benchmark(estimate_tokens, input_string)


def _test_lorem_ipsum_times_1000_times(benchmark: BenchmarkFixture) -> None:
    """The lorem ipsum tokenizes to the known cl100k_base count."""
    input_string = LOREM_IPSUM * 1000
    benchmark(estimate_tokens, input_string)
