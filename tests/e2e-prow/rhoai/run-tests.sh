#!/bin/bash
set -e

# Go to repo root (run-tests.sh is in tests/e2e-prow/rhoai/)
cd "$(dirname "$0")/../../.."

echo "Running tests from: $(pwd)"
echo "E2E_LSC_HOSTNAME: $E2E_LSC_HOSTNAME"

curl -f http://$E2E_LSC_HOSTNAME:8080/v1/models || {
    echo "❌ Basic connectivity failed"
    exit 1
}
echo "✅ Service is responding"

echo "Installing test dependencies..."
pip install uv
uv sync

echo "Running e2e test suite..."
make test-e2e
