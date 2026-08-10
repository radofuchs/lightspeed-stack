"""Unit tests for scripts/e2e_verify_enriched_rag_config.py."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[3]
    / "scripts"
    / "e2e_verify_enriched_rag_config.py"
)


def _load_module():
    """Load the verify script as a module without installing the package."""
    spec = importlib.util.spec_from_file_location(
        "e2e_verify_enriched_rag_config", SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(name="verify_mod")
def verify_mod_fixture():
    """Import the enriched-config verifier script."""
    return _load_module()


def test_expand_env_refs_uses_env_and_default(verify_mod, monkeypatch):
    """Env expansion matches OGX ``${env.VAR:=default}`` behavior."""
    monkeypatch.setenv("KV_RAG_PATH", "/tmp/fixture.db")
    monkeypatch.delenv("MISSING_VAR", raising=False)
    assert (
        verify_mod.expand_env_refs("${env.KV_RAG_PATH:=/unused}") == "/tmp/fixture.db"
    )
    assert verify_mod.expand_env_refs("${env.MISSING_VAR:=fallback}") == "fallback"


def test_verify_enriched_config_ok(verify_mod, tmp_path, monkeypatch):
    """Happy path: namespaced BYOK FAISS + existing fixture DB."""
    db = tmp_path / "kv_store.db"
    db.write_bytes(b"x" * 1_100_000)
    monkeypatch.setenv("FAISS_VECTOR_STORE_ID", "vs_abc")
    monkeypatch.setenv("KV_RAG_PATH", str(db))

    cfg = {
        "storage": {
            "backends": {
                "kv_rag": {
                    "type": "kv_sqlite",
                    "db_path": "${env.KV_RAG_PATH:=~/.llama/storage/rag/kv_store.db}",
                },
                "byok_e2e-test-docs_storage": {
                    "type": "kv_sqlite",
                    "db_path": "${env.KV_RAG_PATH:=~/.llama/storage/rag/kv_store.db}",
                },
            }
        },
        "providers": {
            "vector_io": [
                {
                    "provider_id": "faiss",
                    "provider_type": "inline::faiss",
                    "config": {
                        "persistence": {
                            "namespace": "vector_io::faiss",
                            "backend": "kv_rag",
                        }
                    },
                },
                {
                    "provider_id": "byok_e2e-test-docs",
                    "provider_type": "inline::faiss",
                    "config": {
                        "persistence": {
                            "namespace": "vector_io::faiss",
                            "backend": "byok_e2e-test-docs_storage",
                        }
                    },
                },
            ]
        },
        "registered_resources": {
            "vector_stores": [
                {
                    "vector_store_id": "${env.FAISS_VECTOR_STORE_ID}",
                    "provider_id": "byok_e2e-test-docs",
                    "embedding_model": "sentence-transformers/all-mpnet-base-v2",
                    "embedding_dimension": 768,
                }
            ]
        },
    }

    assert verify_mod.verify_enriched_config(cfg) == []


def test_verify_enriched_config_rejects_wrong_namespace(
    verify_mod, tmp_path, monkeypatch
):
    """Missing OGX persistence.namespace must fail."""
    db = tmp_path / "kv_store.db"
    db.write_bytes(b"x" * 1_100_000)
    monkeypatch.setenv("FAISS_VECTOR_STORE_ID", "vs_abc")
    monkeypatch.setenv("KV_RAG_PATH", str(db))

    cfg = {
        "storage": {
            "backends": {
                "byok_e2e-test-docs_storage": {
                    "type": "kv_sqlite",
                    "db_path": str(db),
                }
            }
        },
        "providers": {
            "vector_io": [
                {
                    "provider_id": "byok_e2e-test-docs",
                    "provider_type": "inline::faiss",
                    "config": {
                        "persistence": {
                            "namespace": "wrong",
                            "backend": "byok_e2e-test-docs_storage",
                        }
                    },
                }
            ]
        },
        "registered_resources": {
            "vector_stores": [{"vector_store_id": "vs_abc"}]
        },
    }

    errors = verify_mod.verify_enriched_config(cfg)
    assert any("persistence.namespace" in err for err in errors)


def test_verify_enriched_config_rejects_unexpanded_store_id(
    verify_mod, tmp_path, monkeypatch
):
    """Unset FAISS_VECTOR_STORE_ID leaves a literal env ref — must fail."""
    db = tmp_path / "kv_store.db"
    db.write_bytes(b"x" * 1_100_000)
    monkeypatch.delenv("FAISS_VECTOR_STORE_ID", raising=False)
    monkeypatch.setenv("KV_RAG_PATH", str(db))

    cfg = {
        "storage": {
            "backends": {
                "byok_e2e-test-docs_storage": {
                    "type": "kv_sqlite",
                    "db_path": str(db),
                }
            }
        },
        "providers": {
            "vector_io": [
                {
                    "provider_id": "byok_e2e-test-docs",
                    "provider_type": "inline::faiss",
                    "config": {
                        "persistence": {
                            "namespace": "vector_io::faiss",
                            "backend": "byok_e2e-test-docs_storage",
                        }
                    },
                }
            ]
        },
        "registered_resources": {
            "vector_stores": [
                {"vector_store_id": "${env.FAISS_VECTOR_STORE_ID}"}
            ]
        },
    }

    errors = verify_mod.verify_enriched_config(cfg)
    assert any("did not expand" in err for err in errors)


def test_main_ok_with_temp_yaml(verify_mod, tmp_path, monkeypatch):
    """CLI main returns 0 for a valid enriched YAML file."""
    db = tmp_path / "kv_store.db"
    db.write_bytes(b"x" * 1_100_000)
    monkeypatch.setenv("FAISS_VECTOR_STORE_ID", "vs_abc")
    monkeypatch.setenv("KV_RAG_PATH", str(db))

    cfg_path = tmp_path / "enriched-run.yaml"
    cfg_path.write_text(
        f"""
storage:
  backends:
    byok_e2e-test-docs_storage:
      type: kv_sqlite
      db_path: {db}
providers:
  vector_io:
    - provider_id: byok_e2e-test-docs
      provider_type: inline::faiss
      config:
        persistence:
          namespace: vector_io::faiss
          backend: byok_e2e-test-docs_storage
registered_resources:
  vector_stores:
    - vector_store_id: vs_abc
""",
        encoding="utf-8",
    )

    assert verify_mod.main([str(cfg_path)]) == 0
