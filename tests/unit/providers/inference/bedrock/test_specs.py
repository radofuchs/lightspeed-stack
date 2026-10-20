"""Unit tests for Bedrock foundation-model wire specs."""

from providers.inference.bedrock.specs import FoundationModelSummary


def test_foundation_model_summary_parses_wire_fields() -> None:
    """Maps camelCase Bedrock fields and ignores unknown extras."""
    model = FoundationModelSummary.model_validate(
        {
            "modelId": "amazon.titan-embed-text-v2:0",
            "outputModalities": ["EMBEDDING"],
            "providerName": "Amazon",
        }
    )

    assert model.model_id == "amazon.titan-embed-text-v2:0"
    assert model.output_modalities == ["EMBEDDING"]

    bare = FoundationModelSummary.model_validate(
        {"modelId": "stability.stable-diffusion-xl-v1"}
    )
    assert bare.output_modalities == []
