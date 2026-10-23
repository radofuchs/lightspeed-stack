"""Context window resolution for conversation compaction.

The compaction trigger needs the context window, in tokens, of the model a
request goes to. :func:`resolve_context_window` returns one, taking the
first of:

1. ``metadata.context_window`` on the model's ``allowed_models`` entry under
   ``inference.providers`` (the provider catalog metadata).
2. The deprecated ``inference.context_windows`` map, keyed by the full model
   identifier.
3. Automatic resolution from the model data bundled with ``genai-prices``,
   when it knows the provider and the model.
4. :data:`constants.DEFAULT_CONTEXT_WINDOW`.

Both configured tiers come before automatic resolution, so a window an
operator sets is used even when the model data lists a larger one.

The catalog tier reads the configuration because the provider registry
(``providers.inference.registry``) is not initialised at startup yet.
:func:`get_catalog_context_window` is the function to point at the
registry's catalog once it is; after the move to pydantic-ai it also gives
the explicit window, with :data:`constants.DEFAULT_CONTEXT_WINDOW` as the
fallback, since the pydantic-ai compaction resolves windows automatically
itself.
"""

from functools import lru_cache
from typing import Optional

from genai_prices.data_snapshot import get_snapshot

from constants import DEFAULT_CONTEXT_WINDOW
from log import get_logger
from models.config import AllowedModel, InferenceConfiguration

logger = get_logger(__name__)


def _find_allowed_model(
    provider_id: str, model_name: str, inference_config: InferenceConfiguration
) -> tuple[Optional[str], Optional[AllowedModel]]:
    """Find a model in the configured provider catalog.

    OGX serves an aliased model under both its ``name`` and its
    ``provider_model_id``, so the model is matched by ``name`` first and by
    ``provider_model_id`` otherwise.

    Parameters:
        provider_id: ``id`` of the ``inference.providers`` entry.
        model_name: ``name`` or ``provider_model_id`` of the model in that
            entry's ``allowed_models``.
        inference_config: The application's inference configuration.

    Returns:
        The provider entry's ``type`` (``None`` when no entry has
        *provider_id*) and the matching allowed model (``None`` when the
        entry does not list *model_name*).

    """
    provider = next(
        (entry for entry in inference_config.providers if entry.id == provider_id),
        None,
    )
    if provider is None:
        return None, None
    model = next(
        (entry for entry in provider.allowed_models if entry.name == model_name),
        None,
    ) or next(
        (
            entry
            for entry in provider.allowed_models
            if entry.provider_model_id == model_name
        ),
        None,
    )
    return provider.type, model


def get_catalog_context_window(
    provider_id: str, model_name: str, inference_config: InferenceConfiguration
) -> Optional[int]:
    """Return the context window set in the model's catalog metadata.

    Reads ``metadata.context_window`` from the model's entry in
    ``inference.providers[].allowed_models``. Switch this to the provider
    registry's catalog once the registry is initialised at startup.

    Parameters:
        provider_id: ``id`` of the ``inference.providers`` entry.
        model_name: ``name`` or ``provider_model_id`` of the model in that
            entry's ``allowed_models``.
        inference_config: The application's inference configuration.

    Returns:
        The window in tokens, or ``None`` when the model is not listed or
        its metadata has no ``context_window``.

    """
    _, model = _find_allowed_model(provider_id, model_name, inference_config)
    if model is None:
        return None
    return model.metadata.get("context_window")


@lru_cache(maxsize=256)
def resolve_context_window_automatically(
    provider: str, model_ref: str
) -> Optional[int]:
    """Look up a model's context window in the ``genai-prices`` model data.

    The data is bundled with the package, so no network call is made. The
    first call loads it. Results are cached, misses included.

    Parameters:
        provider: Provider identifier understood by ``genai-prices`` (for
            example ``openai``, ``azure``, ``bedrock``). Never empty: without
            a provider the lookup could match another vendor's model of the
            same name.
        model_ref: The model identifier the provider uses.

    Returns:
        The window in tokens, or ``None`` when the provider or the model is
        unknown or the model has no positive window listed.

    """
    try:
        _, model_info = get_snapshot().find_provider_model(
            model_ref=model_ref,
            provider=None,
            provider_id=provider,
            provider_api_url=None,
        )
    except LookupError:
        return None
    window = model_info.context_window
    return window if window is not None and window > 0 else None


def resolve_context_window(
    model_id: str, inference_config: InferenceConfiguration
) -> int:
    """Return the context window for *model_id* in tokens.

    Takes the first of: catalog metadata, the deprecated
    ``inference.context_windows`` map, automatic resolution, and
    :data:`constants.DEFAULT_CONTEXT_WINDOW`.

    Parameters:
        model_id: Model identifier in the form ``"<provider_id>/<model>"``
            (for example ``"openai/gpt-5.6-luna"``).
        inference_config: The application's inference configuration.

    Returns:
        The context window size in tokens.

    """
    provider_id, _, model_name = model_id.partition("/")

    catalog_window = get_catalog_context_window(
        provider_id, model_name, inference_config
    )
    if catalog_window is not None:
        logger.debug("Context window of %s from catalog metadata", model_id)
        return catalog_window

    configured_window = inference_config.context_windows.get(model_id)
    if configured_window is not None:
        logger.debug("Context window of %s from inference.context_windows", model_id)
        return configured_window

    provider_type, model = _find_allowed_model(
        provider_id, model_name, inference_config
    )
    provider = provider_type or provider_id
    model_ref = model.provider_model_id if model is not None else model_name
    if provider and model_ref:
        automatic_window = resolve_context_window_automatically(provider, model_ref)
        if automatic_window is not None:
            logger.debug("Context window of %s resolved automatically", model_id)
            return automatic_window

    logger.debug(
        "Context window of %s defaults to %d", model_id, DEFAULT_CONTEXT_WINDOW
    )
    return DEFAULT_CONTEXT_WINDOW
