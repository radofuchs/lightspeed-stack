#!/bin/bash
# Restart the lightspeed-stack-service pod
set -e

NAMESPACE="${NAMESPACE:-e2e-rhoai-dsc}"
POD_NAME="lightspeed-stack-service"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MANIFEST="$SCRIPT_DIR/../manifests/lightspeed/lightspeed-stack.yaml"

echo "===== Restarting lightspeed service ====="
echo "Namespace: $NAMESPACE"
echo "Manifest: $MANIFEST"

echo "Deleting pod $POD_NAME..."
oc delete pod "$POD_NAME" -n "$NAMESPACE" --ignore-not-found=true --wait=true

echo "Applying pod manifest..."
oc apply -f "$MANIFEST"

echo "Waiting for pod to be ready..."
oc wait --for=condition=Ready pod/"$POD_NAME" -n "$NAMESPACE" --timeout=120s

# Re-label pod so service can find it
echo "Labeling pod for service..."
oc label pod "$POD_NAME" pod="$POD_NAME" -n "$NAMESPACE" --overwrite

# Give port-forward time to reconnect
sleep 5

echo "===== Restart complete ====="
