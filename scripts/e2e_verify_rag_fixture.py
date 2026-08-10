#!/usr/bin/env python3
"""Verify the e2e FAISS RAG SQLite fixture before starting Llama Stack.

Prints [e2e-rag] lines for Konflux/Prow log dumps and exits non-zero when the
fixture is missing the FAISS index, does not match FAISS_VECTOR_STORE_ID, or
cannot be deserialized to a non-empty FAISS index.
"""

from __future__ import annotations

import base64
import io
import json
import os
import sqlite3
import sys


def _check_deserialized_ntotal(index_blob: str) -> int:
    """Deserialize the FAISS index payload and return ``ntotal``.

    Parameters:
        index_blob: JSON string stored under the faiss_index KV key.

    Returns:
        Number of vectors in the deserialized index.

    Raises:
        Exception: If faiss/numpy are unavailable or deserialization fails.
    """
    import faiss  # pylint: disable=import-outside-toplevel
    import numpy as np  # pylint: disable=import-outside-toplevel

    data = json.loads(index_blob)
    chunks = data.get("chunk_by_index") or {}
    buffer = io.BytesIO(base64.b64decode(data["faiss_index"]))
    index = faiss.deserialize_index(np.load(buffer, allow_pickle=False))
    print(f"[e2e-rag] chunk_by_index={len(chunks)} ntotal={index.ntotal} dim={index.d}")
    return int(index.ntotal)


def main() -> int:
    """Validate RAG fixture path from env and return a process exit code."""
    path = os.environ.get("RAG_WORK") or os.environ.get("KV_RAG_PATH")
    expected = os.environ.get("FAISS_VECTOR_STORE_ID", "")
    if not path:
        print("FATAL: RAG_WORK or KV_RAG_PATH must be set", file=sys.stderr)
        return 1
    if not os.path.isfile(path):
        print(f"FATAL: RAG fixture missing: {path}", file=sys.stderr)
        return 1

    size = os.path.getsize(path)
    conn = sqlite3.connect(path)
    rows = conn.execute(
        "SELECT key, length(value), value FROM kvstore WHERE key LIKE '%faiss_index%'"
    ).fetchall()
    vs_keys = conn.execute(
        "SELECT key FROM kvstore WHERE key LIKE '%vector_stores:v%::%' "
        "AND key NOT LIKE '%openai%' AND key NOT LIKE '%files%'"
    ).fetchall()
    conn.close()

    print(f"[e2e-rag] fixture={path} size={size}")
    print(f"[e2e-rag] FAISS_VECTOR_STORE_ID={expected!r}")
    print(f"[e2e-rag] vector_stores keys={vs_keys}")
    print(
        f"[e2e-rag] faiss_index keys="
        f"{[(key, val_len) for key, val_len, _ in rows]}"
    )

    if size < 1_048_576:
        print(f"FATAL: RAG fixture too small: {size}", file=sys.stderr)
        return 1
    if not rows:
        print("FATAL: no faiss_index key in RAG fixture", file=sys.stderr)
        return 1

    key, val_len, index_blob = rows[0]
    if expected and expected not in key:
        print(
            f"FATAL: FAISS_VECTOR_STORE_ID {expected!r} not in index key {key!r}",
            file=sys.stderr,
        )
        return 1
    if val_len < 100_000:
        print(f"FATAL: faiss_index value too small: {val_len}", file=sys.stderr)
        return 1

    try:
        ntotal = _check_deserialized_ntotal(index_blob)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        print(f"FATAL: failed to deserialize FAISS index: {exc}", file=sys.stderr)
        return 1
    if ntotal < 1:
        print("FATAL: deserialized FAISS index is empty (ntotal=0)", file=sys.stderr)
        return 1

    print("[e2e-rag] FAISS fixture OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
