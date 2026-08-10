#!/bin/bash
# Entrypoint for llama-stack container.
# Enriches config with lightspeed dynamic values, then starts llama-stack.

set -e

INPUT_CONFIG="${LLAMA_STACK_CONFIG:-/opt/app-root/run.yaml}"
ENRICHED_CONFIG="/tmp/enriched-run.yaml"
LIGHTSPEED_CONFIG="${LIGHTSPEED_CONFIG:-/opt/app-root/lightspeed-stack.yaml}"

# Enrich config if lightspeed config exists
if [ -f "$LIGHTSPEED_CONFIG" ]; then
    echo "Enriching llama-stack config..."
    ENRICHMENT_FAILED=0
    /opt/app-root/.venv/bin/python3 /opt/app-root/llama_stack_configuration.py \
        -c "$LIGHTSPEED_CONFIG" \
        -i "$INPUT_CONFIG" \
        -o "$ENRICHED_CONFIG" 2>&1 || ENRICHMENT_FAILED=1

    if [ -f "$ENRICHED_CONFIG" ] && [ "$ENRICHMENT_FAILED" -eq 0 ]; then
        echo "Using enriched config: $ENRICHED_CONFIG"
        # Evidence for Konflux RAG failures: which SQLite backends BYOK will open.
        /opt/app-root/.venv/bin/python3 - <<'PY' || true
import os, yaml
path = "/tmp/enriched-run.yaml"
with open(path, encoding="utf-8") as fh:
    cfg = yaml.safe_load(fh) or {}
backends = (cfg.get("storage") or {}).get("backends") or {}
print("[e2e-rag] KV_RAG_PATH=", os.environ.get("KV_RAG_PATH"))
print("[e2e-rag] KV_STORE_PATH=", os.environ.get("KV_STORE_PATH"))
print("[e2e-rag] FAISS_VECTOR_STORE_ID=", os.environ.get("FAISS_VECTOR_STORE_ID"))
for name, backend in backends.items():
    if "rag" in name or "byok" in name or name.startswith("kv_"):
        print(f"[e2e-rag] storage.backends[{name}]={backend}")
stores = ((cfg.get("registered_resources") or {}).get("vector_stores")) or []
for store in stores:
    print(f"[e2e-rag] registered vector_store={store}")
PY
        exec ogx stack run "$ENRICHED_CONFIG"
    fi
fi

echo "Using original config: $INPUT_CONFIG"
exec ogx stack run "$INPUT_CONFIG"
