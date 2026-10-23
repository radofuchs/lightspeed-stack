"""Wire models for WatsonX foundation_model_specs responses."""

from __future__ import annotations

from openai import BaseModel


class FoundationModelFunction(BaseModel):
    """A function capability advertised for a foundation model."""

    id: str


class FoundationModelResource(BaseModel):
    """Subset of a foundation_model_specs resource used for typing."""

    model_id: str
    functions: list[FoundationModelFunction]


class FoundationModelSpecsResponse(BaseModel):
    """Subset of GET /ml/v1/foundation_model_specs used for discovery."""

    resources: list[FoundationModelResource]
