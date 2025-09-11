#!/usr/bin/env bash
set -euo pipefail

# Helper to run only the llama.cpp logprob tests either locally (if server reachable)
# or give the user the exact kubectl command for running inside the k8s pod.

TEST_PATH="test/pipelines/test_llamacpp_logprobs.py"

if [[ "${1:-}" == "--inside-pod" ]]; then
  # Assume environment already activated by v.sh wrapper
  echo "[INFO] Running tests inside pod environment" >&2
  pytest -q ${TEST_PATH}
  exit $?
fi

if command -v kubectl >/dev/null 2>&1; then
  POD_NAME=$(kubectl get pods -n ollama -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "")
  if [[ -n "${POD_NAME}" ]]; then
    echo "[INFO] To run inside the existing pod with proper environment activation execute:" >&2
    echo "kubectl exec -it -n ollama ${POD_NAME} -- /app/v.sh runner bash /app/run_llamacpp_logprob_tests.sh --inside-pod" >&2
  fi
fi

echo "[INFO] Attempting local run (expects llama.cpp server at ${LLAMA_CPP_SERVER_URL:-http://localhost:3000})" >&2
pytest -q ${TEST_PATH} || {
  echo "[WARN] Local run failed. If server runs only in k8s, use the kubectl exec command above." >&2
  exit 1
}
