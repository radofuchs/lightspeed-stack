"""Wire models for Bedrock ListFoundationModels responses."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class FoundationModelSummary(BaseModel):
    """Subset of a Bedrock modelSummary used for discovery."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    model_id: str = Field(alias="modelId")
    output_modalities: list[str] = Field(
        default_factory=list,
        alias="outputModalities",
    )
