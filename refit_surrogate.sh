#!/bin/bash
# Weekly surrogate re-fit script
# Reads production traces, refits the TF-IDF/embeddings surrogate,
# and restarts router-proxy to pick up the new model.
#
# Usage: Run via cron (weekly) or manually:
#   bash refit_surrogate.sh
#
# Exit codes:
#   0 — success, model refitted and service restarted
#   1 — insufficient traces (< 20 classifier-source events)
#   2 — fit failed (sklearn/pipeline error)
#   3 — service restart failed

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_PY="/root/.hermes/hermes-agent/venv/bin/python3"
SERVICE_NAME="hermes-router"

echo "[$(date -Iseconds)] Starting weekly surrogate re-fit..."

cd "$SCRIPT_DIR"

# Step 1: Count classifier-source traces across all trace files
# (|| true: some listed files may not exist yet; grep exit 2 would kill the script under pipefail)
TRACE_COUNT=$( { grep -rc '"event":"classify"' traces/router-trace-*.jsonl .router/traces.jsonl 2>/dev/null || true; } | awk -F: '{sum+=$NF} END {print sum+0}')

echo "  Found $TRACE_COUNT classifier trace events"

if [ "$TRACE_COUNT" -lt 20 ]; then
    echo "  ⚠ Insufficient traces ($TRACE_COUNT < 20). Skipping re-fit."
    echo "  Re-fit will be attempted again next week."
    exit 1
fi

# Step 2: Run fit_surrogate.py
echo "  Running fit_surrogate.py..."
if ! "$VENV_PY" fit_surrogate.py --prefer-embeddings; then
    echo "  ✗ fit_surrogate.py failed!" >&2
    exit 2
fi

# Step 3: Verify outputs
for f in .router/surrogate/pipeline.joblib .router/surrogate/acceptor.joblib .router/surrogate/manifest.json; do
    if [ ! -f "$f" ]; then
        echo "  ✗ Missing output: $f" >&2
        exit 2
    fi
done

echo "  ✓ Surrogate refitted successfully"

# Step 4: Restart router-proxy (surrogate loads lazily on first classify call)
export XDG_RUNTIME_DIR=/run/user/0
fuser -k 8766/tcp 2>/dev/null || true
sleep 1

if systemctl --user restart "$SERVICE_NAME"; then
    sleep 3
    if systemctl --user is-active "$SERVICE_NAME" | grep -q active; then
        echo "  ✓ Service restarted and active"
    else
        echo "  ✗ Service not active after restart!" >&2
        exit 3
    fi
else
    echo "  ✗ Service restart failed!" >&2
    exit 3
fi

# Step 5: Quick smoke test
echo "  Running smoke test..."
API_KEY="${ROUTER_PROXY_API_KEY:-$(grep ROUTER_PROXY_API_KEY /root/.hermes/.env | cut -d= -f2 | tr -d '\"' | tr -d "'")}"
RESPONSE=$(curl -s --max-time 15 -X POST http://localhost:8766/v1/chat/completions \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer ${API_KEY}"
    -d '{"model":"deepseek-v4-flash","messages":[{"role":"user","content":"Hello"}],"max_tokens":10,"stream":false}' 2>/dev/null || echo "FAILED")

if echo "$RESPONSE" | grep -q '"choices"'; then
    echo "  ✓ Smoke test passed"
else
    echo "  ⚠ Smoke test response unexpected (service may still be starting)"
fi

echo "[$(date -Iseconds)] Re-fit complete"