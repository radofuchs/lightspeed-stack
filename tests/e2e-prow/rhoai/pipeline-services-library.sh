#!/bin/bash
# Deploy library-mode LCS pod (embedded OGX). Used by pipeline.sh when E2E_DEPLOYMENT_MODE=library.

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAMESPACE="${NAMESPACE:-e2e-rhoai-dsc}"
export NAMESPACE

LIGHTSPEED_STACK_IMAGE="${LIGHTSPEED_STACK_IMAGE:-quay.io/lightspeed-core/lightspeed-stack:dev-latest}"
export LIGHTSPEED_STACK_IMAGE

timeout 120 oc delete pod lightspeed-stack-service -n "$NAMESPACE" --ignore-not-found=true --wait=true 2>/dev/null || true

if command -v envsubst >/dev/null 2>&1; then
  envsubst '${LIGHTSPEED_STACK_IMAGE}' < "$BASE_DIR/manifests/lightspeed/lightspeed-stack-library.yaml" |
    oc apply -n "$NAMESPACE" -f -
else
  sed "s|\${LIGHTSPEED_STACK_IMAGE}|${LIGHTSPEED_STACK_IMAGE}|g" "$BASE_DIR/manifests/lightspeed/lightspeed-stack-library.yaml" |
    oc apply -n "$NAMESPACE" -f -
fi

oc label pod lightspeed-stack-service pod=lightspeed-stack-service -n "$NAMESPACE" --overwrite 2>/dev/null || true
oc expose pod lightspeed-stack-service \
  --name=lightspeed-stack-service-svc \
  --port=8080 \
  --type=ClusterIP \
  -n "$NAMESPACE" 2>/dev/null || true