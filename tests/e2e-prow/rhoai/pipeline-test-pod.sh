#!/bin/bash

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAMESPACE="${NAMESPACE:-e2e-rhoai-dsc}"

oc apply -n "$NAMESPACE" -f "$BASE_DIR/manifests/test-pod/spin-up.yaml"
