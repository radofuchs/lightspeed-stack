#!/usr/bin/env python3
"""Fail-fast checks for OGX-enriched FAISS/BYOK config (Konflux/GH e2e).

Validates that enrichment produced a usable FAISS setup for the e2e fixture:
namespaced persistence, resolvable SQLite db_path, and a vector_store_id that
expands to FAISS_VECTOR_STORE_ID (not a leftover ``${env....}`` literal).
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Any

import yaml

# OGX uses ``${env.VAR}`` or ``${env.VAR:=default}`` (note ``:=``, not bare ``=``).
_ENV_PATTERN = re.compile(
    r"\$\{env\.([A-Za-z_][A-Za-z0-9_]*)(?::=([^}]*))?\}"
)


def expand_env_refs(value: str) -> str:
    """Expand ``${env.VAR}`` / ``${env.VAR:=default}`` like OGX ``replace_env_vars``.

    Parameters:
        value: Raw config string that may contain env references.

    Returns:
        String with all env references replaced from the process environment.
    """

    def _replace(match: re.Match[str]) -> str:
        name = match.group(1)
        default = match.group(2)
        env_val = os.environ.get(name)
        if env_val is not None and env_val != "":
            return env_val
        if default is not None:
            return default
        return ""

    return _ENV_PATTERN.sub(_replace, value)


def _byok_backends(backends: dict[str, Any]) -> dict[str, Any]:
    """Return storage backends used for BYOK / RAG SQLite."""
    return {
        name: backend
        for name, backend in backends.items()
        if "byok" in name or name in {"kv_rag", "kv_default"}
    }


def _faiss_providers(vector_io: list[Any]) -> list[dict[str, Any]]:
    """Return inline FAISS vector_io provider dicts."""
    out: list[dict[str, Any]] = []
    for provider in vector_io:
        if not isinstance(provider, dict):
            continue
        ptype = str(provider.get("provider_type") or "")
        if ptype == "inline::faiss" or str(provider.get("provider_id", "")).startswith(
            "byok_"
        ):
            out.append(provider)
    return out


def verify_enriched_config(cfg: dict[str, Any]) -> list[str]:
    """Return human-readable errors for an enriched Llama/OGX run config.

    Parameters:
        cfg: Parsed enriched ``run.yaml`` mapping.

    Returns:
        List of error strings (empty means OK). Skips checks when no BYOK/FAISS
        providers are present.
    """
    errors: list[str] = []
    backends = (cfg.get("storage") or {}).get("backends") or {}
    vector_io = ((cfg.get("providers") or {}).get("vector_io")) or []
    if not isinstance(vector_io, list):
        vector_io = []
    stores = ((cfg.get("registered_resources") or {}).get("vector_stores")) or []
    if not isinstance(stores, list):
        stores = []

    faiss_providers = _faiss_providers(vector_io)
    byok_backends = {
        name: backend
        for name, backend in backends.items()
        if isinstance(name, str) and name.startswith("byok_")
    }

    if not faiss_providers and not byok_backends:
        print("[e2e-rag] enriched config: no BYOK/FAISS providers — skipping")
        return []

    expected_id = os.environ.get("FAISS_VECTOR_STORE_ID", "").strip()
    kv_rag_path = os.environ.get("KV_RAG_PATH", "").strip()

    print(f"[e2e-rag] KV_RAG_PATH={kv_rag_path!r}")
    print(f"[e2e-rag] KV_STORE_PATH={os.environ.get('KV_STORE_PATH')!r}")
    print(f"[e2e-rag] FAISS_VECTOR_STORE_ID={expected_id!r}")

    for name, backend in _byok_backends(backends).items():
        print(f"[e2e-rag] storage.backends[{name}]={backend}")

    for provider in faiss_providers:
        print(f"[e2e-rag] vector_io provider={provider}")
        persistence = (provider.get("config") or {}).get("persistence") or {}
        namespace = persistence.get("namespace")
        backend_name = persistence.get("backend")
        if namespace != "vector_io::faiss":
            errors.append(
                f"provider {provider.get('provider_id')!r} persistence.namespace "
                f"is {namespace!r}, expected 'vector_io::faiss'"
            )
        if str(provider.get("provider_id", "")).startswith("byok_"):
            if not backend_name or not str(backend_name).startswith("byok_"):
                errors.append(
                    f"BYOK provider {provider.get('provider_id')!r} "
                    f"persistence.backend is {backend_name!r}"
                )
            elif backend_name not in backends:
                errors.append(
                    f"BYOK persistence.backend {backend_name!r} missing from "
                    "storage.backends"
                )

    if not byok_backends and any(
        str(p.get("provider_id", "")).startswith("byok_") for p in faiss_providers
    ):
        errors.append("BYOK FAISS providers present but no byok_* storage backends")

    for name, backend in byok_backends.items():
        if not isinstance(backend, dict):
            errors.append(f"storage.backends[{name}] is not a mapping")
            continue
        raw_path = str(backend.get("db_path") or "")
        resolved = expand_env_refs(raw_path)
        print(f"[e2e-rag] resolved db_path[{name}]={resolved!r} (raw={raw_path!r})")
        if "${env." in resolved:
            errors.append(
                f"storage.backends[{name}].db_path still has unresolved env refs: "
                f"{resolved!r}"
            )
            continue
        if not resolved:
            errors.append(f"storage.backends[{name}].db_path resolved empty")
            continue
        # HOME-relative defaults from run-ci.yaml
        path = Path(resolved).expanduser()
        if not path.is_file():
            errors.append(f"storage.backends[{name}].db_path does not exist: {path}")
            continue
        size = path.stat().st_size
        print(f"[e2e-rag] db_path[{name}] size={size}")
        if size < 1_048_576:
            errors.append(
                f"storage.backends[{name}].db_path too small ({size} bytes): {path}"
            )
        if kv_rag_path and path.resolve() != Path(kv_rag_path).expanduser().resolve():
            # Warn in logs; only error when FAISS id is set (e2e fixture path).
            if expected_id:
                errors.append(
                    f"storage.backends[{name}].db_path {path} != KV_RAG_PATH "
                    f"{kv_rag_path}"
                )

    if not stores:
        errors.append("registered_resources.vector_stores is empty after enrichment")
    for store in stores:
        print(f"[e2e-rag] registered vector_store={store}")
        if not isinstance(store, dict):
            errors.append(f"vector_store entry is not a mapping: {store!r}")
            continue
        raw_id = str(store.get("vector_store_id") or "")
        resolved_id = expand_env_refs(raw_id)
        print(
            f"[e2e-rag] resolved vector_store_id={resolved_id!r} (raw={raw_id!r})"
        )
        if "${env." in resolved_id or not resolved_id:
            errors.append(
                f"vector_store_id did not expand (raw={raw_id!r}, "
                f"resolved={resolved_id!r}); is FAISS_VECTOR_STORE_ID set?"
            )
        elif expected_id and resolved_id != expected_id:
            errors.append(
                f"vector_store_id {resolved_id!r} != FAISS_VECTOR_STORE_ID "
                f"{expected_id!r}"
            )

    if expected_id and not expected_id.startswith("vs_"):
        errors.append(
            f"FAISS_VECTOR_STORE_ID looks invalid for OGX fixture: {expected_id!r}"
        )

    return errors


def main(argv: list[str] | None = None) -> int:
    """CLI entry: verify enriched run.yaml path (default ``/tmp/enriched-run.yaml``)."""
    args = list(sys.argv[1:] if argv is None else argv)
    path = args[0] if args else "/tmp/enriched-run.yaml"
    if not os.path.isfile(path):
        print(f"FATAL: enriched config missing: {path}", file=sys.stderr)
        return 1

    with open(path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    if not isinstance(cfg, dict):
        print(f"FATAL: enriched config is not a mapping: {path}", file=sys.stderr)
        return 1

    print(f"[e2e-rag] verifying enriched config={path}")
    errors = verify_enriched_config(cfg)
    if errors:
        for err in errors:
            print(f"FATAL: {err}", file=sys.stderr)
        return 1
    print("[e2e-rag] enriched FAISS/BYOK config OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
