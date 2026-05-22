"""Step definitions for TLS configuration e2e tests.

These tests configure Llama Stack's run.yaml with NetworkConfig TLS settings
and verify the full pipeline works through the Lightspeed Stack.

Config switching uses the same pattern as other e2e tests: overwrite the
host-mounted run.yaml and restart Docker containers. Cleanup is handled
by a Background step that restores the backup before each scenario.
"""

import copy
import os
from typing import Any, Optional

from behave import given  # pyright: ignore[reportAttributeAccessIssue]
from behave.runner import Context

from tests.e2e.utils.llama_config_utils import (
    backup_llama_config,
    load_llama_config,
    write_llama_config,
)
from tests.e2e.utils.prow_utils import get_namespace, run_e2e_ops
from tests.e2e.utils.utils import is_prow_environment

_MOCK_TLS_PORT_TLS = 8443
_MOCK_TLS_PORT_MTLS = 8444
_MOCK_TLS_PORT_HOSTNAME_MISMATCH = 8445

_TLS_MODEL_RESOURCE: dict[str, str] = {
    "model_id": "mock-tls-model",
    "provider_id": "tls-openai",
    "provider_model_id": "mock-tls-model",
}

_mock_tls_cluster_deploy_state: dict[str, bool] = {"done": False}
_tls_prow_restart_state: dict[str, bool] = {"full_restart_done": False}


def reset_tls_prow_restart_optimization_state() -> None:
    """Reset per-feature Prow state (call from ``before_feature``)."""
    _mock_tls_cluster_deploy_state["done"] = False
    _tls_prow_restart_state["full_restart_done"] = False
    os.environ.pop("E2E_LLAMA_RELOAD_CONFIG_ONLY", None)
    os.environ.pop("E2E_COPY_MOCK_TLS_CERTS_TO_LLAMA", None)


def is_tls_configuration_feature(context: Context) -> bool:
    """Return True when the active Behave feature is ``tls.feature``."""
    feature = getattr(context, "feature", None)
    if feature is None:
        return False
    name = getattr(feature, "name", "") or ""
    return "TLS configuration" in name


def _prepare_tls_prow_llama_full_restart_env() -> None:
    """Env for a full llama pod recreate (first TLS scenario / recovery)."""
    os.environ["E2E_COPY_MOCK_TLS_CERTS_TO_LLAMA"] = "1"
    os.environ.pop("E2E_LLAMA_RELOAD_CONFIG_ONLY", None)


def _prepare_tls_prow_llama_reload_env() -> None:
    """Env for config-only reload (run.yaml already on pod with /certs mounted)."""
    os.environ["E2E_COPY_MOCK_TLS_CERTS_TO_LLAMA"] = "1"
    os.environ["E2E_LLAMA_RELOAD_CONFIG_ONLY"] = "1"


def restart_llama_for_tls_feature(context: Context) -> None:
    """Restart Llama for TLS tests: one full recreate per feature, then reload.

    Full pod delete+apply after every scenario (~16×) is flaky on Konflux
    (cert sync, mount races, port-forward). Later scenarios only change run.yaml
    in the ConfigMap; ``oc cp`` + container restart is enough.
    """
    from tests.e2e.utils.utils import restart_container

    if not is_prow_environment():
        restart_container("llama-stack")
        return

    if _tls_prow_restart_state["full_restart_done"]:
        _prepare_tls_prow_llama_reload_env()
    else:
        _prepare_tls_prow_llama_full_restart_env()

    try:
        restart_container("llama-stack")
    except Exception:
        _tls_prow_restart_state["full_restart_done"] = False
        raise

    _tls_prow_restart_state["full_restart_done"] = True


def _prepare_tls_prow_llama_restart_env() -> None:
    """Set env before writing run.yaml (used by ``_configure_tls``)."""
    os.environ["E2E_COPY_MOCK_TLS_CERTS_TO_LLAMA"] = "1"
    if _tls_prow_restart_state["full_restart_done"]:
        os.environ["E2E_LLAMA_RELOAD_CONFIG_ONLY"] = "1"
    else:
        os.environ.pop("E2E_LLAMA_RELOAD_CONFIG_ONLY", None)


def _cluster_mock_tls_inference_host() -> str:
    """DNS name of the in-cluster mock TLS inference server (Konflux / Prow)."""
    explicit = os.getenv("E2E_MOCK_TLS_INFERENCE_HOST", "").strip()
    if explicit:
        return explicit
    return f"e2e-mock-tls-inference.{get_namespace()}.svc.cluster.local"


def _mock_tls_base_url(port: int) -> str:
    """OpenAI-compatible base URL for the mock TLS inference server."""
    if is_prow_environment():
        host = _cluster_mock_tls_inference_host()
    else:
        host = "mock-tls-inference"
    return f"https://{host}:{port}/v1"


def _tls_provider_base() -> dict[str, Any]:
    """Default tls-openai provider dict with environment-appropriate base_url."""
    return {
        "provider_id": "tls-openai",
        "provider_type": "remote::openai",
        "config": {
            "api_key": "test-key",
            "base_url": _mock_tls_base_url(_MOCK_TLS_PORT_TLS),
            "allowed_models": ["mock-tls-model"],
        },
    }


def _deploy_cluster_mock_tls_inference() -> None:
    """Deploy the in-cluster mock TLS inference pod (Konflux / Prow)."""
    if _mock_tls_cluster_deploy_state["done"]:
        print("Using existing e2e-mock-tls-inference deployment")
        return

    result = run_e2e_ops("deploy-e2e-mock-tls-inference", timeout=300)
    print(result.stdout, end="")
    if result.returncode != 0:
        raise AssertionError(
            "Failed to deploy e2e-mock-tls-inference: "
            f"{result.stderr or result.stdout}"
        )
    _prepare_tls_prow_llama_restart_env()
    os.environ.setdefault(
        "E2E_MOCK_TLS_INFERENCE_HOST",
        _cluster_mock_tls_inference_host(),
    )
    _mock_tls_cluster_deploy_state["done"] = True


def _ensure_tls_provider(config: dict[str, Any]) -> dict[str, Any]:
    """Find or create the tls-openai inference provider in the config.

    If the provider does not exist, it is added along with the
    mock-tls-model registered resource.

    Parameters:
    ----------
        config: The Llama Stack configuration dictionary.

    Returns:
    -------
        The tls-openai provider configuration dictionary.
    """
    providers = config.setdefault("providers", {})
    inference = providers.setdefault("inference", [])

    for provider in inference:
        if provider.get("provider_id") == "tls-openai":
            return provider

    # Provider not found — add it
    provider = copy.deepcopy(_tls_provider_base())
    inference.append(provider)

    # Also register the model resource
    resources = config.setdefault("registered_resources", {})
    models = resources.setdefault("models", [])
    if not any(m.get("model_id") == "mock-tls-model" for m in models):
        models.append(copy.deepcopy(_TLS_MODEL_RESOURCE))

    return provider


def _configure_tls(tls_config: dict[str, Any], base_url: Optional[str] = None) -> None:
    """Configure TLS settings for the tls-openai provider.

    Parameters:
    ----------
        tls_config: The TLS configuration dictionary.
        base_url: Optional base URL override for the provider.
    """
    backup_llama_config()
    config = load_llama_config()
    provider = _ensure_tls_provider(config)
    provider.setdefault("config", {}).setdefault("network", {})
    if base_url is not None:
        provider["config"]["base_url"] = base_url
    else:
        provider["config"]["base_url"] = _mock_tls_base_url(_MOCK_TLS_PORT_TLS)
    provider["config"]["network"]["tls"] = tls_config
    write_llama_config(config)
    if is_prow_environment():
        _prepare_tls_prow_llama_restart_env()


# --- Background Steps ---
# ``The original Llama Stack config is restored if modified`` only restores
# run.yaml (see proxy.py). Restart steps are listed in tls.feature / proxy.feature.


@given("The mock TLS inference server is deployed")
def deploy_mock_tls_inference_server(context: Context) -> None:
    """Ensure mock TLS inference is reachable (Compose locally, pod in Prow)."""
    if is_prow_environment():
        _deploy_cluster_mock_tls_inference()
        return
    print("Using docker-compose mock-tls-inference service")


# --- TLS Configuration Steps ---


@given("Llama Stack is configured with TLS verification disabled")
def configure_tls_verify_false(context: Context) -> None:
    """Configure run.yaml with TLS verify: false."""
    _configure_tls({"verify": False})


@given("Llama Stack is configured with CA certificate verification")
def configure_tls_verify_ca(context: Context) -> None:
    """Configure run.yaml with TLS verify: /certs/ca.crt."""
    _configure_tls({"verify": "/certs/ca.crt", "min_version": "TLSv1.2"})


@given("Llama Stack is configured with TLS verification enabled")
def configure_tls_verify_true(context: Context) -> None:
    """Configure run.yaml with TLS verify: true (fails with self-signed certs)."""
    _configure_tls({"verify": True})


@given("Llama Stack is configured with mutual TLS authentication")
def configure_tls_mtls(context: Context) -> None:
    """Configure run.yaml with mutual TLS (client cert and key)."""
    _configure_tls(
        {
            "verify": "/certs/ca.crt",
            "client_cert": "/certs/client.crt",
            "client_key": "/certs/client.key",
        },
        base_url=_mock_tls_base_url(_MOCK_TLS_PORT_MTLS),
    )


@given('Llama Stack is configured with CA certificate path "{path}"')
def configure_tls_verify_ca_path(context: Context, path: str) -> None:
    """Configure run.yaml with TLS verify pointing to a specific CA cert path."""
    _configure_tls({"verify": path})


@given("Llama Stack is configured for mTLS without client certificate")
def configure_mtls_no_client_cert(context: Context) -> None:
    """Configure run.yaml for mTLS port without client cert (should fail)."""
    _configure_tls(
        {"verify": "/certs/ca.crt"},
        base_url=_mock_tls_base_url(_MOCK_TLS_PORT_MTLS),
    )


@given("Llama Stack is configured for mTLS with wrong client certificate")
def configure_mtls_wrong_client_cert(context: Context) -> None:
    """Configure run.yaml for mTLS with invalid client cert (CA cert as client cert)."""
    _configure_tls(
        {
            "verify": "/certs/ca.crt",
            "client_cert": "/certs/ca.crt",
            "client_key": "/certs/client.key",
        },
        base_url=_mock_tls_base_url(_MOCK_TLS_PORT_MTLS),
    )


@given("Llama Stack is configured for mTLS with untrusted client certificate")
def configure_mtls_untrusted_client_cert(context: Context) -> None:
    """Configure run.yaml for mTLS with client cert from untrusted CA."""
    _configure_tls(
        {
            "verify": "/certs/ca.crt",
            "client_cert": "/certs/untrusted-client.crt",
            "client_key": "/certs/untrusted-client.key",
        },
        base_url=_mock_tls_base_url(_MOCK_TLS_PORT_MTLS),
    )


@given("Llama Stack is configured for mTLS with expired client certificate")
def configure_mtls_expired_client_cert(context: Context) -> None:
    """Configure run.yaml for mTLS with an expired client certificate."""
    _configure_tls(
        {
            "verify": "/certs/ca.crt",
            "client_cert": "/certs/expired-client.crt",
            "client_key": "/certs/client.key",
        },
        base_url=_mock_tls_base_url(_MOCK_TLS_PORT_MTLS),
    )


@given("Llama Stack is configured with CA certificate and hostname mismatch server")
def configure_tls_hostname_mismatch(context: Context) -> None:
    """Configure run.yaml to connect to hostname-mismatch server (should fail)."""
    _configure_tls(
        {"verify": "/certs/ca.crt"},
        base_url=_mock_tls_base_url(_MOCK_TLS_PORT_HOSTNAME_MISMATCH),
    )


@given("Llama Stack is configured with mutual TLS and hostname mismatch server")
def configure_mtls_hostname_mismatch(context: Context) -> None:
    """Configure run.yaml for mTLS against hostname-mismatch server (should fail)."""
    _configure_tls(
        {
            "verify": "/certs/ca.crt",
            "client_cert": "/certs/client.crt",
            "client_key": "/certs/client.key",
        },
        base_url=_mock_tls_base_url(_MOCK_TLS_PORT_HOSTNAME_MISMATCH),
    )


@given(
    'Llama Stack is configured with TLS minimum version "{version}" and hostname mismatch server'
)
def configure_tls_min_version_hostname_mismatch(context: Context, version: str) -> None:
    """Configure run.yaml with TLS min version against hostname-mismatch server."""
    _configure_tls(
        {"verify": "/certs/ca.crt", "min_version": version},
        base_url=_mock_tls_base_url(_MOCK_TLS_PORT_HOSTNAME_MISMATCH),
    )


@given(
    'Llama Stack is configured with TLS minimum version "{version}" and CA certificate path "{path}"'
)
def configure_tls_min_version_with_ca_path(
    context: Context, version: str, path: str
) -> None:
    """Configure run.yaml with TLS minimum version and a specific CA cert path."""
    _configure_tls({"verify": path, "min_version": version})
