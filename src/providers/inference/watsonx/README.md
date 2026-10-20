# List of source files stored in `src/providers/inference/watsonx` directory

## [__init__.py](__init__.py)

WatsonX inference provider package.

## [auth.py](auth.py)

IBM IAM authentication for WatsonX HTTP clients (`WatsonxIAMAuth`).

## [pai_provider.py](pai_provider.py)

Pydantic AI `Provider[AsyncOpenAI]` for WatsonX (`WatsonxProvider`).
Builds the IAM-authenticated AsyncOpenAI client from config/env.

## [provider.py](provider.py)

WatsonX runtime inference provider backed by Pydantic AI.
