# List of source files stored in `src/providers/inference/bedrock` directory

## [__init__.py](__init__.py)

Bedrock inference provider package.

## [provider.py](provider.py)

AWS Bedrock runtime inference provider backed by Pydantic AI.
Discovers models via a separate boto3 control-plane client
(`ListFoundationModels`) and runs chat/embeddings through PAI's
`BedrockProvider`.
