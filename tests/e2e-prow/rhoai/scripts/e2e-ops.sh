#!/bin/bash
# Consolidated E2E operations script for OpenShift/Prow environment
# Usage: e2e-ops.sh <command> [args...]
#
# Architecture note (why logs look "fine" but CI flakes):
# - pipeline.sh starts oc port-forward to localhost:8080 for the whole test run.
# - Behave before_feature calls restart-lightspeed, which must kill that forward and start a new one.
# - In-cluster pod logs (Uvicorn up, Llama OK) do not reflect localhost bind races; "address already in use"
#   is the CI runner, not the application.
# - E2E_LSC_PORT_FORWARD_PID_FILE coordinates the handoff.
# - pipeline-konflux.sh (and hooks) forward llama-stack-service-svc to localhost:8321 for
#   Behave steps that call Llama Stack directly (MCP toolgroups, shields). When the llama
#   pod is recreated, that forward must be restarted or you get "PodSandbox ... not found" /
#   APIConnectionError on subsequent scenarios.
# - E2E_LLAMA_PORT_FORWARD_PID_FILE coordinates killing/restarting the 8321 forward.
#
# Commands:
#   restart-lightspeed              - Restart lightspeed-stack pod and port-forward
#   restart-llama-stack             - Restart/restore llama-stack pod and localhost:8321 forward
#   restart-port-forward            - Re-establish port-forward for lightspeed
#   restart-llama-port-forward      - Re-establish port-forward for Llama Stack (8321)
#   wait-for-pod <name> [attempts]  - Wait for a pod to be ready
#   update-configmap <name> <file>  - Update ConfigMap from file
#   get-configmap-content <name>    - Get ConfigMap content (outputs to stdout)
#   disrupt-llama-stack             - Delete llama-stack pod to disrupt connection
#   deploy-e2e-tunnel-proxy         - Deploy in-cluster tunnel proxy (proxy.feature step)
#   deploy-e2e-interception-proxy   - Deploy in-cluster interception proxy (proxy.feature step)
#   deploy-e2e-mock-tls-inference   - Deploy mock HTTPS inference server (tls.feature step)
#   sync-mock-tls-certs-secret      - Publish /certs PEMs to Secret for llama-stack mount

set -e

NAMESPACE="${NAMESPACE:-e2e-rhoai-dsc}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MANIFEST_DIR="$SCRIPT_DIR/../manifests/lightspeed"
# Written by pipeline.sh when it starts LCS port-forward; e2e-ops kills this PID before rebinding 8080.
E2E_LSC_PORT_FORWARD_PID_FILE="${E2E_LSC_PORT_FORWARD_PID_FILE:-/tmp/e2e-lightspeed-port-forward.pid}"
E2E_LLAMA_PORT_FORWARD_PID_FILE="${E2E_LLAMA_PORT_FORWARD_PID_FILE:-/tmp/e2e-llama-port-forward.pid}"
E2E_JWKS_PORT_FORWARD_PID_FILE="${E2E_JWKS_PORT_FORWARD_PID_FILE:-/tmp/e2e-jwks-port-forward.pid}"

# ============================================================================
# Helper functions
# ============================================================================

wait_for_pod() {
    local pod_name="$1"
    local max_attempts="${2:-24}"
    
    for ((attempt=1; attempt<=max_attempts; attempt++)); do
        local ready
        ready=$(oc get pod "$pod_name" -n "$NAMESPACE" -o jsonpath='{.status.containerStatuses[0].ready}' 2>/dev/null || echo "false")
        if [[ "$ready" == "true" ]]; then
            echo "✓ Pod $pod_name ready"
            return 0
        fi
        sleep 3
    done
    
    echo "Pod $pod_name not ready after $((max_attempts * 3))s"
    return 1
}

# Linux: find PIDs with a LISTEN socket on TCP port (decimal) via /proc when lsof/ss are missing.
kill_tcp_listen_pids_via_procfs() {
    local port="${1:?port required}"
    local port_hex inode pid fd link
    port_hex=$(printf '%04X' "$port")
    local -a inodes=()
    while read -r inode; do
        [[ -n "$inode" ]] && inodes+=("$inode")
    done < <(
        {
            awk -v p=":${port_hex}\$" '$1 ~ /^[0-9]+:$/ && $4 == "0A" && $2 ~ p { print $10 }' /proc/net/tcp 2>/dev/null
            awk -v p=":${port_hex}\$" '$1 ~ /^[0-9]+:$/ && $4 == "0A" && $2 ~ p { print $10 }' /proc/net/tcp6 2>/dev/null
        } | sort -u
    )
    for inode in "${inodes[@]}"; do
        for fd in /proc/[0-9]*/fd/*; do
            [[ -e "$fd" ]] || continue
            link=$(readlink "$fd" 2>/dev/null) || continue
            if [[ "$link" == "socket:[$inode]" ]]; then
                pid="${fd#/proc/}"
                pid="${pid%%/*}"
                if [[ "$pid" =~ ^[0-9]+$ && "$pid" != "1" ]]; then
                    kill -9 "$pid" 2>/dev/null || true
                fi
            fi
        done
    done
}

# Free a local TCP port. Stale oc port-forward often survives "lost connection" but still
# listens on 8080. Try every method: lsof, fuser, ss, then /proc (Konflux often has no lsof/ss).
free_local_tcp_port() {
    local port="${1:?port required}"
    local pid
    if command -v lsof >/dev/null >&2; then
        for pid in $(lsof -ti:"$port" -sTCP:LISTEN 2>/dev/null); do
            kill -9 "$pid" 2>/dev/null || true
        done
    fi
    if command -v fuser >/dev/null >&2; then
        fuser -k "${port}/tcp" 2>/dev/null || true
    fi
    if command -v ss >/dev/null >&2; then
        # LISTEN ... users:(("oc",pid=1234,fd=7))
        while read -r pid; do
            [[ -n "$pid" ]] && kill -9 "$pid" 2>/dev/null || true
        done < <(ss -lptnH "sport = :$port" 2>/dev/null | sed -n 's/.*pid=\([0-9][0-9]*\).*/\1/p' | sort -u)
    fi
    if [[ -r /proc/net/tcp ]]; then
        kill_tcp_listen_pids_via_procfs "$port"
    fi
    sleep 1
}

# Kill anything likely to hold the Lightspeed local forward; then free the port (twice for races).
kill_stale_lightspeed_forward() {
    local port="${1:-8080}"
    local saved_pf
    # Pipeline leaves oc port-forward on 8080; pkill patterns can miss (argv wrapping). PID file is authoritative.
    if [[ -f "$E2E_LSC_PORT_FORWARD_PID_FILE" ]]; then
        read -r saved_pf <"$E2E_LSC_PORT_FORWARD_PID_FILE" 2>/dev/null || true
        if [[ "$saved_pf" =~ ^[0-9]+$ ]]; then
            kill -9 "$saved_pf" 2>/dev/null || true
        fi
    fi
    pkill -9 -f "port-forward.*lightspeed-stack-service-svc" 2>/dev/null || true
    pkill -9 -f "kubectl port-forward.*lightspeed-stack-service-svc" 2>/dev/null || true
    pkill -9 -f "oc port-forward svc/lightspeed-stack-service-svc" 2>/dev/null || true
    pkill -9 -f "port-forward pod/lightspeed-stack-service" 2>/dev/null || true
    pkill -9 -f "port-forward.*${port}:${port}" 2>/dev/null || true
    free_local_tcp_port "$port"
    sleep 1
    free_local_tcp_port "$port"
}

# Kill anything likely to hold the Llama Stack local forward (localhost:8321).
kill_stale_llama_forward() {
    local port="${1:-8321}"
    local saved_pf
    if [[ -f "$E2E_LLAMA_PORT_FORWARD_PID_FILE" ]]; then
        read -r saved_pf <"$E2E_LLAMA_PORT_FORWARD_PID_FILE" 2>/dev/null || true
        if [[ "$saved_pf" =~ ^[0-9]+$ ]]; then
            kill -9 "$saved_pf" 2>/dev/null || true
        fi
    fi
    pkill -9 -f "port-forward.*llama-stack-service-svc.*${port}:${port}" 2>/dev/null || true
    pkill -9 -f "oc port-forward svc/llama-stack-service-svc ${port}:${port}" 2>/dev/null || true
    pkill -9 -f "port-forward pod/llama-stack-service.*${port}:${port}" 2>/dev/null || true
    free_local_tcp_port "$port"
    sleep 1
    free_local_tcp_port "$port"
}

# Kill anything likely to hold the mock-jwks local forward (localhost:8000).
kill_stale_jwks_forward() {
    local port="${1:-8000}"
    local saved_pf
    if [[ -f "$E2E_JWKS_PORT_FORWARD_PID_FILE" ]]; then
        read -r saved_pf <"$E2E_JWKS_PORT_FORWARD_PID_FILE" 2>/dev/null || true
        if [[ "$saved_pf" =~ ^[0-9]+$ ]]; then
            kill -9 "$saved_pf" 2>/dev/null || true
        fi
    fi
    pkill -9 -f "port-forward.*mock-jwks.*${port}:${port}" 2>/dev/null || true
    pkill -9 -f "oc port-forward svc/mock-jwks ${port}:${port}" 2>/dev/null || true
    free_local_tcp_port "$port"
    sleep 1
    free_local_tcp_port "$port"
}

# After oc port-forward dies in <2s, show recent oc stderr from the log file.
e2e_ops_emit_port_forward_immediate_failure_diag() {
    echo "[e2e-ops] /tmp/port-forward.log (tail 25):"
    if [[ -s /tmp/port-forward.log ]]; then
        tail -25 /tmp/port-forward.log 2>/dev/null | sed 's/^/[e2e-ops] /' || true
    else
        echo "[e2e-ops] (log empty or missing)"
    fi
}

e2e_ops_diagnose_forward_failure() {
    echo "[e2e-ops] Port-forward failed after all retries."
    if [[ -s /tmp/port-forward.log ]]; then
        echo "[e2e-ops] /tmp/port-forward.log (tail 30):"
        tail -30 /tmp/port-forward.log 2>/dev/null | sed 's/^/[e2e-ops] /' || true
    fi
    echo "[e2e-ops] oc get pods -n $NAMESPACE:"
    oc get pods -n "$NAMESPACE" -o wide 2>&1 | sed 's/^/[e2e-ops] /' || true
}

verify_connectivity() {
    local max_attempts="${1:-6}"
    local local_port="${LOCAL_PORT:-8080}"
    local http_code=""

    for ((attempt=1; attempt<=max_attempts; attempt++)); do
        # First check /readiness to see if port-forward is alive (accept 200, 401, or 503)
        http_code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 "http://localhost:$local_port/readiness" 2>/dev/null) || http_code="000"

        # LCS returns 503 when provider health fails (see health.py). Intentionally broken
        # Llama proxy e2e stays 503 forever while the tunnel is still fine. Only accept 503
        # on the last attempt so normal restarts keep retrying while providers warm up
        # (transient 503 then 200) and we do not short-circuit other suites on first 503.
        if [[ "$http_code" == "503" ]]; then
            if [[ "$attempt" -eq "$max_attempts" ]]; then
                echo "[e2e-ops] /readiness=503 after $max_attempts attempts — LCS reachable; providers still unhealthy (expected for some e2e)"
                return 0
            fi
            echo "[e2e-ops] /readiness=503 (attempt $attempt/$max_attempts); retrying in case providers recover..."
        fi

        if [[ "$http_code" == "200" || "$http_code" == "401" ]]; then
            # Port-forward works; now verify the app is fully initialized by hitting
            # a real endpoint. /v1/models requires the Llama Stack handshake to complete.
            # Accept 200 (no auth) or 401/403 (auth) — both prove the full app stack is up.
            #
            # Proxy/TLS e2e scenarios intentionally misconfigure Llama (e.g. unreachable
            # HTTP proxy). LCS still answers /v1/models with 5xx once the route exists;
            # treating those as success avoids false failures on restart-lightspeed while
            # still rejecting connection errors (000).
            local models_code
            models_code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 10 "http://localhost:$local_port/v1/models" 2>/dev/null) || models_code="000"
            if [[ "$models_code" == "200" || "$models_code" == "401" || "$models_code" == "403" ]]; then
                return 0
            fi
            if [[ "$models_code" =~ ^5[0-9][0-9]$ ]]; then
                echo "[e2e-ops] /v1/models=$models_code (LCS reachable; Llama/provider error expected in some e2e)"
                return 0
            fi
            echo "[e2e-ops] /readiness=$http_code but /v1/models=$models_code (app still initializing, attempt $attempt/$max_attempts)"
        fi

        if [[ $attempt -lt $max_attempts ]]; then
            sleep 5
        fi
    done

    echo "Connectivity check failed (readiness: ${http_code:-unknown})"
    return 1
}

# Single check: Llama serves /v1/health on loopback inside the main container.
_llama_stack_http_health_once() {
    local pod="llama-stack-service"
    local ctr="llama-stack-container"
    if oc exec -n "$NAMESPACE" "$pod" -c "$ctr" -- \
        curl -sf --max-time 10 "http://127.0.0.1:8321/v1/health" >/dev/null 2>&1; then
        return 0
    fi
    if oc exec -n "$NAMESPACE" "$pod" -c "$ctr" -- \
        /opt/app-root/.venv/bin/python -c \
        "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8321/v1/health', timeout=10).read()" \
        >/dev/null 2>&1; then
        return 0
    fi
    return 1
}

# After the pod is Ready, confirm the process is actually serving HTTP (not only kubelet probes).
wait_for_llama_stack_http_health() {
    local max_attempts="${1:-35}"
    local attempt

    echo "Verifying Llama Stack is fully up (GET /v1/health inside pod)..."
    for ((attempt=1; attempt<=max_attempts; attempt++)); do
        if _llama_stack_http_health_once; then
            echo "✓ Llama Stack /v1/health OK (attempt $attempt/$max_attempts)"
            return 0
        fi
        if [[ $attempt -lt $max_attempts ]]; then
            sleep 2
        fi
    done
    echo "ERROR: Llama Stack did not respond on http://127.0.0.1:8321/v1/health inside the pod"
    oc get pod llama-stack-service -n "$NAMESPACE" -o wide 2>&1 || true
    oc describe pod llama-stack-service -n "$NAMESPACE" 2>&1 | tail -40 || true
    oc logs llama-stack-service -n "$NAMESPACE" -c llama-stack-container --tail=120 2>&1 || true
    return 1
}

# ============================================================================
# Command implementations
# ============================================================================

cmd_restart_lightspeed() {
    echo "Restarting lightspeed-stack service..."

    # LCS hangs at startup if Llama Stack is unreachable (blocks Llama handshake,
    # never opens port 8080, readiness probe never passes).  Ensure Llama Stack
    # is healthy before recreating the LCS pod.
    if ! _llama_stack_http_health_once 2>/dev/null; then
        echo "⚠️  Llama Stack not healthy — restoring before LCS restart..."
        cmd_restart_llama_stack || echo "⚠️  Llama Stack restore failed; LCS may be slow to start"
    fi

    # Delete existing pod (short wait so hook stays within timeout; force if needed)
    timeout 20 oc delete pod lightspeed-stack-service -n "$NAMESPACE" --ignore-not-found=true --wait=true 2>/dev/null || {
        oc delete pod lightspeed-stack-service -n "$NAMESPACE" --ignore-not-found=true --force --grace-period=0 2>/dev/null || true
        sleep 2
    }
    
    # Apply manifest (expand LIGHTSPEED_STACK_IMAGE only; filter prevents blanking other $VAR refs)
    LIGHTSPEED_STACK_IMAGE="${LIGHTSPEED_STACK_IMAGE:-quay.io/lightspeed-core/lightspeed-stack:dev-latest}"
    export LIGHTSPEED_STACK_IMAGE
    _ls_manifest="$MANIFEST_DIR/lightspeed-stack.yaml"
    sed "s|\${LIGHTSPEED_STACK_IMAGE}|${LIGHTSPEED_STACK_IMAGE}|g" "$_ls_manifest" |
        oc apply -n "$NAMESPACE" -f -
    
    # Wait for pod to be ready (TCP probe passes when app listens on 8080).
    # Don't let a timeout here abort the function — still attempt port-forward
    # and diagnostics so later scenarios have a chance to recover.
    local pod_ready=true
    local lcs_pod_wait=40
    if [[ "${E2E_KONFLUX_E2E:-0}" == "1" ]]; then
        lcs_pod_wait=65
    fi
    if ! wait_for_pod "lightspeed-stack-service" "$lcs_pod_wait"; then
        pod_ready=false
        echo "⚠️  Pod not ready within 120s — dumping diagnostics:"
        oc describe pod lightspeed-stack-service -n "$NAMESPACE" 2>&1 | tail -30 || true
        oc logs lightspeed-stack-service -n "$NAMESPACE" --tail=40 2>&1 || true
    fi

    # Re-label pod for service discovery
    oc label pod lightspeed-stack-service pod=lightspeed-stack-service -n "$NAMESPACE" --overwrite

    # Re-establish port-forwards (may succeed even if readiness was slow)
    cmd_restart_port_forward
    cmd_restart_jwks_port_forward || echo "⚠️  Mock JWKS port-forward failed (RBAC tests may fail)"

    if [[ "$pod_ready" == "false" ]]; then
        echo "⚠️  Lightspeed restart completed but pod was slow to become ready"
        return 1
    fi
    echo "✓ Lightspeed restart complete"
}

cmd_restart_llama_stack() {
    echo "===== Restoring llama-stack service ====="
    # Pod.spec is largely immutable; delete so apply creates a pod with current volumes/env.
    echo "Deleting llama-stack pod (if any) before apply..."
    timeout 45 oc delete pod llama-stack-service -n "$NAMESPACE" --ignore-not-found=true --wait=true 2>/dev/null || {
        oc delete pod llama-stack-service -n "$NAMESPACE" --ignore-not-found=true --force --grace-period=0 2>/dev/null || true
        sleep 3
    }

    echo "Applying pod manifest..."
    if [[ "${E2E_KONFLUX_E2E:-0}" == "1" ]]; then
        # Interception-proxy e2e: refresh Secret before pod recreate so the volume mount is populated.
        if [[ "${E2E_COPY_INTERCEPTION_CA_TO_LLAMA:-0}" == "1" ]]; then
            echo "[e2e-ops] Syncing e2e-interception-proxy-ca secret before llama-stack apply..."
            if ! cmd_sync_interception_proxy_ca_secret; then
                echo "===== Llama-stack restore FAILED (interception CA secret sync) ====="
                exit 1
            fi
        fi
        if [[ "${E2E_COPY_MOCK_TLS_CERTS_TO_LLAMA:-0}" == "1" ]]; then
            echo "[e2e-ops] Syncing e2e-mock-tls-certs secret before llama-stack apply..."
            if ! cmd_sync_mock_tls_certs_secret; then
                echo "===== Llama-stack restore FAILED (mock TLS certs secret sync) ====="
                exit 1
            fi
        fi
        _LLAMA_SVC_FQDN="llama-stack-service-svc.${NAMESPACE}.svc.cluster.local"
        oc create secret generic llama-stack-ip-secret \
            --from-literal=key="$_LLAMA_SVC_FQDN" \
            -n "$NAMESPACE" \
            --dry-run=client -o yaml | oc apply -f -
        oc apply -n "$NAMESPACE" -f "$MANIFEST_DIR/llama-stack-openai.yaml"
        wait_for_pod "llama-stack-service" 90
        echo "Labeling pod for service..."
        oc label pod llama-stack-service pod=llama-stack-service -n "$NAMESPACE" --overwrite
        if [[ "${E2E_COPY_INTERCEPTION_CA_TO_LLAMA:-0}" == "1" ]]; then
            if ! _verify_interception_ca_mounted_in_llama; then
                echo "===== Llama-stack restore FAILED (interception CA not mounted) ====="
                exit 1
            fi
        fi
        if [[ "${E2E_COPY_MOCK_TLS_CERTS_TO_LLAMA:-0}" == "1" ]]; then
            if ! _verify_mock_tls_certs_mounted_in_llama; then
                echo "===== Llama-stack restore FAILED (mock TLS certs not mounted) ====="
                exit 1
            fi
        fi
        local llama_health_attempts=50
        if [[ "${E2E_COPY_MOCK_TLS_CERTS_TO_LLAMA:-0}" == "1" ]]; then
            llama_health_attempts=75
        fi
        if ! wait_for_llama_stack_http_health "$llama_health_attempts"; then
            echo "===== Llama-stack restore FAILED (HTTP not healthy) ====="
            exit 1
        fi
    else
        # Prow: vLLM Llama Stack image (matches pipeline.sh / pipeline-services.sh)
        # Use sed instead of envsubst to avoid blanking $VAR references in embedded bash scripts
        sed "s|\${LLAMA_STACK_IMAGE}|${LLAMA_STACK_IMAGE:-}|g" "$MANIFEST_DIR/llama-stack-prow.yaml" |
            oc apply -n "$NAMESPACE" -f -
        wait_for_pod "llama-stack-service" 24
        echo "Labeling pod for service..."
        oc label pod llama-stack-service pod=llama-stack-service -n "$NAMESPACE" --overwrite
    fi

    if ! cmd_restart_llama_port_forward; then
        echo "ERROR: Llama pod is up but localhost:${LOCAL_LLAMA_PORT:-8321} port-forward failed"
        exit 1
    fi

    echo "===== Llama-stack restore complete ====="
}

cmd_restart_port_forward() {
    local local_port="${LOCAL_PORT:-8080}"
    local remote_port="${REMOTE_PORT:-8080}"
    local max_attempts=6
    local pf_pid
    local pf_resource

    echo "Re-establishing port-forward on $local_port:$remote_port..."

    for ((attempt=1; attempt<=max_attempts; attempt++)); do
        kill_stale_lightspeed_forward "$local_port"
        # Let the kernel release LISTEN sockets after pkill (avoids immediate "address already in use")
        sleep 3

        # Service can lag endpoints after pod recreate; pod-direct forward is more reliable.
        if [[ $attempt -le 2 ]]; then
            pf_resource="svc/lightspeed-stack-service-svc"
        else
            pf_resource="pod/lightspeed-stack-service"
        fi
        echo "Port-forward attempt $attempt/$max_attempts -> $pf_resource"

        : > /tmp/port-forward.log
        # Redirect stdin from /dev/null so oc does not see a closed pipe when the parent is a short-lived subprocess.
        nohup oc port-forward "$pf_resource" "$local_port:$remote_port" -n "$NAMESPACE" \
            </dev/null >/tmp/port-forward.log 2>&1 &
        pf_pid=$!
        disown "$pf_pid" 2>/dev/null || true
        sleep 3

        # Bind error or API error: process exits quickly — surface /tmp/port-forward.log every time
        if ! kill -0 "$pf_pid" 2>/dev/null; then
            echo "Port-forward process exited immediately:"
            e2e_ops_emit_port_forward_immediate_failure_diag
            kill_stale_lightspeed_forward "$local_port"
            sleep 2
            continue
        fi
        sleep 6

        if verify_connectivity 15; then
            echo "$pf_pid" >"$E2E_LSC_PORT_FORWARD_PID_FILE"
            local readiness_code
            readiness_code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 "http://127.0.0.1:$local_port/readiness" 2>/dev/null) || readiness_code="000"
            echo "[e2e-ops] LCS through port-forward: GET http://127.0.0.1:$local_port/readiness -> HTTP $readiness_code (expect 200 or 401)"
            echo "✓ Port-forward established (PID: $pf_pid, $pf_resource)"
            return 0
        fi

        if grep -q "address already in use" /tmp/port-forward.log 2>/dev/null; then
            echo "Bind error in port-forward log; clearing listeners and retrying..."
            kill_stale_lightspeed_forward "$local_port"
        fi
        if [[ $attempt -lt $max_attempts ]]; then
            echo "Attempt $attempt failed, retrying..."
            kill -9 "$pf_pid" 2>/dev/null || true
            sleep 2
        fi
    done

    echo "Failed to establish port-forward"
    e2e_ops_diagnose_forward_failure
    return 1
}

verify_llama_local_forward() {
    local max_attempts="${1:-15}"
    local http_code=""
    local attempt

    for ((attempt=1; attempt<=max_attempts; attempt++)); do
        http_code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 "http://127.0.0.1:8321/v1/health" 2>/dev/null) || http_code="000"
        if [[ "$http_code" == "200" ]]; then
            return 0
        fi
        if [[ $attempt -lt $max_attempts ]]; then
            sleep 2
        fi
    done
    echo "Llama Stack localhost:8321 connectivity check failed (HTTP: ${http_code:-unknown})"
    return 1
}

cmd_restart_llama_port_forward() {
    local local_port="${LOCAL_LLAMA_PORT:-8321}"
    local remote_port="${REMOTE_LLAMA_PORT:-8321}"
    local max_attempts=6
    local pf_pid
    local pf_resource
    local llama_pf_log="/tmp/port-forward-llama.log"

    echo "Re-establishing Llama Stack port-forward on $local_port:$remote_port..."

    for ((attempt=1; attempt<=max_attempts; attempt++)); do
        kill_stale_llama_forward "$local_port"
        sleep 3

        if [[ "${E2E_KONFLUX_E2E:-0}" == "1" ]] || [[ $attempt -ge 3 ]]; then
            pf_resource="pod/llama-stack-service"
        else
            pf_resource="svc/llama-stack-service-svc"
        fi
        echo "Llama port-forward attempt $attempt/$max_attempts -> $pf_resource"

        : >"$llama_pf_log"
        nohup oc port-forward "$pf_resource" "$local_port:$remote_port" -n "$NAMESPACE" \
            </dev/null >"$llama_pf_log" 2>&1 &
        pf_pid=$!
        disown "$pf_pid" 2>/dev/null || true
        sleep 3

        if ! kill -0 "$pf_pid" 2>/dev/null; then
            echo "Llama port-forward process exited immediately:"
            if [[ -s "$llama_pf_log" ]]; then
                tail -25 "$llama_pf_log" 2>/dev/null | sed 's/^/[e2e-ops] /' || true
            fi
            kill_stale_llama_forward "$local_port"
            sleep 2
            continue
        fi
        sleep 4

        if verify_llama_local_forward 12; then
            echo "$pf_pid" >"$E2E_LLAMA_PORT_FORWARD_PID_FILE"
            echo "[e2e-ops] Llama through port-forward: GET http://127.0.0.1:$local_port/v1/health -> OK"
            echo "✓ Llama Stack port-forward established (PID: $pf_pid, $pf_resource)"
            return 0
        fi

        if [[ $attempt -lt $max_attempts ]]; then
            echo "Llama forward attempt $attempt failed, retrying..."
            kill -9 "$pf_pid" 2>/dev/null || true
            sleep 2
        fi
    done

    echo "Failed to establish Llama Stack port-forward on :$local_port"
    if [[ -s "$llama_pf_log" ]]; then
        tail -30 "$llama_pf_log" 2>/dev/null | sed 's/^/[e2e-ops] /' || true
    fi
    return 1
}

cmd_restart_jwks_port_forward() {
    local local_port="${LOCAL_JWKS_PORT:-8000}"
    local remote_port="${REMOTE_JWKS_PORT:-8000}"
    local max_attempts=4
    local pf_pid
    local jwks_pf_log="/tmp/port-forward-jwks.log"

    # Check if existing forward is still alive
    if [[ -f "$E2E_JWKS_PORT_FORWARD_PID_FILE" ]]; then
        local saved_pf
        read -r saved_pf <"$E2E_JWKS_PORT_FORWARD_PID_FILE" 2>/dev/null || true
        if [[ "$saved_pf" =~ ^[0-9]+$ ]] && kill -0 "$saved_pf" 2>/dev/null; then
            local http_code
            http_code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 "http://127.0.0.1:$local_port/tokens" 2>/dev/null) || http_code="000"
            if [[ "$http_code" != "000" ]]; then
                echo "✓ Mock JWKS port-forward already healthy (PID: $saved_pf)"
                return 0
            fi
        fi
    fi

    echo "Re-establishing mock-jwks port-forward on $local_port:$remote_port..."

    for ((attempt=1; attempt<=max_attempts; attempt++)); do
        kill_stale_jwks_forward "$local_port"
        sleep 2

        echo "JWKS port-forward attempt $attempt/$max_attempts"

        : >"$jwks_pf_log"
        nohup oc port-forward svc/mock-jwks "$local_port:$remote_port" -n "$NAMESPACE" \
            </dev/null >"$jwks_pf_log" 2>&1 &
        pf_pid=$!
        disown "$pf_pid" 2>/dev/null || true
        sleep 3

        if ! kill -0 "$pf_pid" 2>/dev/null; then
            echo "JWKS port-forward process exited immediately"
            continue
        fi

        local http_code
        http_code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 "http://127.0.0.1:$local_port/tokens" 2>/dev/null) || http_code="000"
        if [[ "$http_code" != "000" ]]; then
            echo "$pf_pid" >"$E2E_JWKS_PORT_FORWARD_PID_FILE"
            echo "✓ Mock JWKS port-forward established (PID: $pf_pid)"
            return 0
        fi

        if [[ $attempt -lt $max_attempts ]]; then
            echo "JWKS forward attempt $attempt failed, retrying..."
            kill -9 "$pf_pid" 2>/dev/null || true
            sleep 2
        fi
    done

    echo "Failed to establish mock-jwks port-forward on :$local_port"
    return 1
}

cmd_wait_for_pod() {
    local pod_name="${1:?Pod name required}"
    local max_attempts="${2:-24}"
    wait_for_pod "$pod_name" "$max_attempts"
}

cmd_update_configmap() {
    local configmap_name="${1:?ConfigMap name required}"
    local source_file="${2:?Source file required}"
    local configmap_key="${3:-lightspeed-stack.yaml}"

    echo "Updating ConfigMap $configmap_name from $source_file..."

    if [[ ! -f "$source_file" ]]; then
        echo "ERROR: source file does not exist: $source_file" >&2
        return 1
    fi

    # Use dry-run + apply to avoid the delete-then-create race.
    # If delete succeeds but create fails the ConfigMap is gone and every
    # subsequent attempt cascades into failure.
    if ! oc create configmap "$configmap_name" -n "$NAMESPACE" \
            --from-file="${configmap_key}=${source_file}" \
            --dry-run=client -o yaml | oc apply -n "$NAMESPACE" -f -; then
        echo "ERROR: oc apply for ConfigMap $configmap_name failed" >&2
        return 1
    fi

    echo "✓ ConfigMap $configmap_name updated successfully"
}

cmd_get_configmap_content() {
    local configmap_name="${1:?ConfigMap name required}"
    local configmap_key="${2:-lightspeed-stack.yaml}"
    oc get configmap "$configmap_name" -n "$NAMESPACE" \
        -o "go-template={{index .data \"$configmap_key\"}}"
}

cmd_tunnel_proxy_stats() {
    local pod_name
    pod_name=$(oc get pod -n "$NAMESPACE" -l app=e2e-tunnel-proxy \
        -o jsonpath='{.items[0].metadata.name}' 2>/dev/null) || pod_name=""

    if [[ -z "$pod_name" ]]; then
        echo "ERROR: no e2e-tunnel-proxy pod in namespace $NAMESPACE" >&2
        return 1
    fi

    oc exec -n "$NAMESPACE" "$pod_name" -- \
        python3 -c "import urllib.request; print(urllib.request.urlopen('http://127.0.0.1:8887/stats', timeout=5).read().decode())"
}

cmd_interception_proxy_stats() {
    local pod_name
    pod_name=$(oc get pod -n "$NAMESPACE" -l app=e2e-interception-proxy \
        -o jsonpath='{.items[0].metadata.name}' 2>/dev/null) || pod_name=""

    if [[ -z "$pod_name" ]]; then
        echo "ERROR: no e2e-interception-proxy pod in namespace $NAMESPACE" >&2
        return 1
    fi

    oc exec -n "$NAMESPACE" "$pod_name" -- \
        python3 -c "import urllib.request; print(urllib.request.urlopen('http://127.0.0.1:8886/stats', timeout=5).read().decode())"
}

cmd_sync_interception_proxy_ca_secret() {
    local proxy_pod_name tmp
    proxy_pod_name=$(oc get pod -n "$NAMESPACE" -l app=e2e-interception-proxy \
        -o jsonpath='{.items[0].metadata.name}' 2>/dev/null) || proxy_pod_name=""

    if [[ -z "$proxy_pod_name" ]]; then
        echo "ERROR: no e2e-interception-proxy pod in namespace $NAMESPACE" >&2
        return 1
    fi

    tmp=$(mktemp)
    if ! oc exec -n "$NAMESPACE" "$proxy_pod_name" -- \
        cat /tmp/interception-proxy-ca.pem >"$tmp"; then
        rm -f "$tmp"
        echo "ERROR: failed to read CA from e2e-interception-proxy pod" >&2
        return 1
    fi
    if [[ ! -s "$tmp" ]]; then
        rm -f "$tmp"
        echo "ERROR: interception-proxy CA PEM is empty" >&2
        return 1
    fi

    if ! oc create secret generic e2e-interception-proxy-ca \
        --from-file=ca.pem="$tmp" \
        -n "$NAMESPACE" \
        --dry-run=client -o yaml | oc apply -n "$NAMESPACE" -f -; then
        rm -f "$tmp"
        echo "ERROR: failed to apply e2e-interception-proxy-ca secret" >&2
        return 1
    fi
    rm -f "$tmp"
    echo "✓ Secret e2e-interception-proxy-ca updated (ca.pem)"
}

_verify_interception_ca_mounted_in_llama() {
    local llama_pod_name="llama-stack-service"
    if oc exec -n "$NAMESPACE" "$llama_pod_name" -c llama-stack-container -- \
        test -s /tmp/interception-proxy-ca.pem; then
        echo "✓ interception-proxy CA present at /tmp/interception-proxy-ca.pem in llama-stack"
        return 0
    fi
    echo "ERROR: /tmp/interception-proxy-ca.pem missing or empty in llama-stack pod" >&2
    oc exec -n "$NAMESPACE" "$llama_pod_name" -c llama-stack-container -- \
        ls -la /tmp/interception-proxy-ca.pem 2>&1 || true
    return 1
}

cmd_copy_interception_proxy_ca_to_llama() {
    # Legacy name: publish CA via Secret (mounted by llama-stack-openai.yaml).
    cmd_sync_interception_proxy_ca_secret
}

_MOCK_TLS_CERT_FILES=(
    ca.crt
    client.crt
    client.key
    untrusted-ca.crt
    expired-ca.crt
    untrusted-client.crt
    untrusted-client.key
    expired-client.crt
)

cmd_sync_mock_tls_certs_secret() {
    local mock_pod_name tmpdir f
    mock_pod_name=$(oc get pod -n "$NAMESPACE" -l app=e2e-mock-tls-inference \
        -o jsonpath='{.items[0].metadata.name}' 2>/dev/null) || mock_pod_name=""

    if [[ -z "$mock_pod_name" ]]; then
        echo "ERROR: no e2e-mock-tls-inference pod in namespace $NAMESPACE" >&2
        echo "  Run: e2e-ops.sh deploy-e2e-mock-tls-inference" >&2
        return 1
    fi

    tmpdir=$(mktemp -d)
    for f in "${_MOCK_TLS_CERT_FILES[@]}"; do
        if ! oc exec -n "$NAMESPACE" "$mock_pod_name" -c e2e-mock-tls-inference -- \
            cat "/certs/$f" >"$tmpdir/$f"; then
            echo "ERROR: failed to read /certs/$f from e2e-mock-tls-inference pod" >&2
            rm -rf "$tmpdir"
            return 1
        fi
        if [[ ! -s "$tmpdir/$f" ]]; then
            echo "ERROR: /certs/$f is empty in e2e-mock-tls-inference pod" >&2
            rm -rf "$tmpdir"
            return 1
        fi
    done

    if ! oc create secret generic e2e-mock-tls-certs \
        --from-file="$tmpdir" \
        -n "$NAMESPACE" \
        --dry-run=client -o yaml | oc apply -f -; then
        echo "ERROR: failed to apply e2e-mock-tls-certs secret" >&2
        rm -rf "$tmpdir"
        return 1
    fi
    rm -rf "$tmpdir"
    echo "✓ Secret e2e-mock-tls-certs updated (${#_MOCK_TLS_CERT_FILES[@]} files)"
}

_verify_mock_tls_certs_mounted_in_llama() {
    local llama_pod_name="llama-stack-service"
    if oc exec -n "$NAMESPACE" "$llama_pod_name" -c llama-stack-container -- \
        sh -c 'test -s /certs/ca.crt && test -s /certs/client.crt && test -s /certs/client.key'; then
        echo "✓ mock TLS certs present under /certs in llama-stack"
        return 0
    fi
    echo "ERROR: /certs missing or incomplete in llama-stack pod" >&2
    oc get secret e2e-mock-tls-certs -n "$NAMESPACE" 2>&1 || true
    oc exec -n "$NAMESPACE" "$llama_pod_name" -c llama-stack-container -- \
        ls -la /certs 2>&1 || true
    return 1
}

_e2e_repo_root() {
    cd "$SCRIPT_DIR/../../../.." && pwd
}

cmd_deploy_e2e_tunnel_proxy() {
    local repo_root
    repo_root="$(_e2e_repo_root)"
    echo "Deploying e2e-tunnel-proxy in namespace $NAMESPACE..."
    oc create configmap e2e-tunnel-proxy-script -n "$NAMESPACE" \
        --from-file=tunnel_proxy.py="$repo_root/tests/e2e/proxy/tunnel_proxy.py" \
        --dry-run=client -o yaml | oc apply -f -
    oc apply -n "$NAMESPACE" -f "$MANIFEST_DIR/e2e-tunnel-proxy.yaml"
    if ! oc wait pod/e2e-tunnel-proxy -n "$NAMESPACE" --for=condition=Ready --timeout=120s; then
        echo "ERROR: e2e-tunnel-proxy failed to become ready" >&2
        oc describe pod e2e-tunnel-proxy -n "$NAMESPACE" 2>/dev/null | tail -25 || true
        return 1
    fi
    echo "✓ e2e-tunnel-proxy ready at http://e2e-tunnel-proxy.${NAMESPACE}.svc.cluster.local:8888"
}

cmd_deploy_e2e_interception_proxy() {
    local repo_root
    repo_root="$(_e2e_repo_root)"
    echo "Deploying e2e-interception-proxy in namespace $NAMESPACE..."
    oc create configmap e2e-interception-proxy-script -n "$NAMESPACE" \
        --from-file=interception_proxy.py="$repo_root/tests/e2e/proxy/interception_proxy.py" \
        --dry-run=client -o yaml | oc apply -f -
    oc apply -n "$NAMESPACE" -f "$MANIFEST_DIR/e2e-interception-proxy.yaml"
    if ! oc wait pod/e2e-interception-proxy -n "$NAMESPACE" --for=condition=Ready --timeout=180s; then
        echo "ERROR: e2e-interception-proxy failed to become ready" >&2
        oc describe pod e2e-interception-proxy -n "$NAMESPACE" 2>/dev/null | tail -25 || true
        return 1
    fi
    echo "✓ e2e-interception-proxy ready at http://e2e-interception-proxy.${NAMESPACE}.svc.cluster.local:8889"
}

cmd_deploy_e2e_mock_tls_inference() {
    local repo_root
    repo_root="$(_e2e_repo_root)"
    echo "Deploying e2e-mock-tls-inference in namespace $NAMESPACE..."
    oc create configmap e2e-mock-tls-inference-script -n "$NAMESPACE" \
        --from-file=server.py="$repo_root/tests/e2e/mock_tls_inference_server/server.py" \
        --dry-run=client -o yaml | oc apply -f -
    oc delete pod e2e-mock-tls-inference -n "$NAMESPACE" --ignore-not-found=true --wait=true 2>/dev/null || true
    oc apply -n "$NAMESPACE" -f "$MANIFEST_DIR/e2e-mock-tls-inference.yaml"
    if ! oc wait pod/e2e-mock-tls-inference -n "$NAMESPACE" --for=condition=Ready --timeout=240s; then
        echo "ERROR: e2e-mock-tls-inference failed to become ready" >&2
        oc describe pod/e2e-mock-tls-inference -n "$NAMESPACE" 2>/dev/null | tail -30 || true
        oc logs e2e-mock-tls-inference -n "$NAMESPACE" --tail=40 2>&1 || true
        return 1
    fi
    echo "✓ e2e-mock-tls-inference ready at https://e2e-mock-tls-inference.${NAMESPACE}.svc.cluster.local:8443"
    if ! cmd_sync_mock_tls_certs_secret; then
        echo "WARNING: mock TLS server is up but e2e-mock-tls-certs secret sync failed" >&2
        return 1
    fi
}

cmd_disrupt_llama_stack() {
    local pod_name="llama-stack-service"

    local phase
    phase=$(oc get pod "$pod_name" -n "$NAMESPACE" -o jsonpath='{.status.phase}' 2>/dev/null || echo "NotFound")

    if [[ "$phase" == "Running" ]]; then
        oc delete pod "$pod_name" -n "$NAMESPACE" --wait=true
        sleep 2
        echo "Llama Stack connection disrupted successfully (pod deleted)"
        exit 0
    else
        echo "Llama Stack pod was not running (phase: $phase)"
        exit 2
    fi
}

# ============================================================================
# Main command dispatcher
# ============================================================================

COMMAND="${1:-}"
shift || true

case "$COMMAND" in
    restart-lightspeed)
        cmd_restart_lightspeed
        ;;
    restart-llama-stack)
        cmd_restart_llama_stack
        ;;
    restart-llama-port-forward)
        cmd_restart_llama_port_forward
        ;;
    restart-jwks-port-forward)
        cmd_restart_jwks_port_forward
        ;;
    restart-port-forward)
        cmd_restart_port_forward
        ;;
    wait-for-pod)
        cmd_wait_for_pod "$@"
        ;;
    update-configmap)
        cmd_update_configmap "$@"
        ;;
    get-configmap-content)
        cmd_get_configmap_content "$@"
        ;;
    disrupt-llama-stack)
        cmd_disrupt_llama_stack
        ;;
    tunnel-proxy-stats)
        cmd_tunnel_proxy_stats
        ;;
    interception-proxy-stats)
        cmd_interception_proxy_stats
        ;;
    copy-interception-proxy-ca-to-llama)
        cmd_copy_interception_proxy_ca_to_llama
        ;;
    sync-interception-proxy-ca-secret)
        cmd_sync_interception_proxy_ca_secret
        ;;
    deploy-e2e-tunnel-proxy)
        cmd_deploy_e2e_tunnel_proxy
        ;;
    deploy-e2e-interception-proxy)
        cmd_deploy_e2e_interception_proxy
        ;;
    deploy-e2e-mock-tls-inference)
        cmd_deploy_e2e_mock_tls_inference
        ;;
    sync-mock-tls-certs-secret)
        cmd_sync_mock_tls_certs_secret
        ;;
    *)
        echo "Usage: $0 <command> [args...]"
        echo ""
        echo "Commands:"
        echo "  restart-lightspeed              - Restart lightspeed-stack pod and port-forward"
        echo "  restart-llama-stack             - Restart/restore llama-stack pod"
        echo "  restart-llama-port-forward      - Re-establish port-forward for Llama (8321)"
        echo "  restart-port-forward            - Re-establish port-forward for lightspeed"
        echo "  wait-for-pod <name> [attempts]  - Wait for a pod to be ready"
        echo "  update-configmap <name> <file>  - Update ConfigMap from file"
        echo "  get-configmap-content <name>    - Get ConfigMap content (outputs to stdout)"
        echo "  disrupt-llama-stack             - Delete llama-stack pod to disrupt connection"
        echo "  tunnel-proxy-stats              - JSON stats from in-cluster e2e-tunnel-proxy"
        echo "  interception-proxy-stats        - JSON stats from in-cluster e2e-interception-proxy"
        echo "  copy-interception-proxy-ca-to-llama - Alias for sync-interception-proxy-ca-secret"
        echo "  sync-interception-proxy-ca-secret   - Publish trustme CA to Secret for llama mount"
        echo "  deploy-e2e-tunnel-proxy            - Deploy in-cluster tunnel proxy pod"
        echo "  deploy-e2e-interception-proxy      - Deploy in-cluster interception proxy pod"
        echo "  deploy-e2e-mock-tls-inference        - Deploy mock HTTPS inference server (tls.feature)"
        echo "  sync-mock-tls-certs-secret           - Publish mock TLS /certs to Secret for llama mount"
        exit 1
        ;;
esac
