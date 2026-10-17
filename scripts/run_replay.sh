#!/usr/bin/env bash
# Replay a tick CSV onto the ZMQ bus: broker -> (optional consumer) -> ingester, then shut everything down.
#
# What it starts, in order:
#   1. scripts/run_zmq_broker.py, and waits until the broker itself reports that both sockets are bound
#      (its zmq_broker_started log line), then confirms both ports accept connections, before anything
#      connects. If the broker exits first (for example the port is already in use) the script stops and
#      never starts the ingester.
#   2. Optionally a consumer, from REPLAY_CONSUMER_CMD, given REPLAY_CONSUMER_WARMUP_SECONDS to subscribe.
#   3. scripts/historical_ingester.py on the CSV, in the foreground.
# When the ingester finishes, the consumer gets REPLAY_DRAIN_SECONDS to drain, then the consumer and the
# broker are stopped (also on Ctrl-C or on an error). The exit code is the ingester's.
#
# What it does not start: the Risk Gates, a broker simulator, or the z-score alpha. The gates live inside
# ExecutionAgent, and scripts/run_backend.py also opens live exchange feeds, which would mix with the
# replay. scripts/zscore_backtest.py reads files directly and does not use the bus.
#
# Usage:
#   bash scripts/run_replay.sh <ticks.csv> [historical_ingester.py options...]
#   bash scripts/run_replay.sh EURUSD.csv --timestamp-column Etc/UTC --bid-column Close --assumed-spread 0.0001 --speed 0
#
# The bus addresses come from ZMQ_BROKER_FRONTEND_ADDRESS / ZMQ_BROKER_BACKEND_ADDRESS (environment or .env).
# On Windows, `kill` on a native Python process is a hard stop, which is fine for a stateless broker.

set -uo pipefail

CSV="${1:-}"
if [[ -z "$CSV" || ! -f "$CSV" ]]; then
    echo "usage: bash scripts/run_replay.sh <ticks.csv> [historical_ingester.py options...]" >&2
    [[ -n "$CSV" ]] && echo "file not found: $CSV" >&2
    exit 2
fi
shift

cd "$(dirname "$0")/.." || exit 2
PYTHON="${PYTHON:-./.venv/Scripts/python.exe}"
[[ -x "$PYTHON" ]] || PYTHON="python"
READY_TIMEOUT="${REPLAY_READY_TIMEOUT_SECONDS:-30}"
WARMUP="${REPLAY_CONSUMER_WARMUP_SECONDS:-3}"
DRAIN="${REPLAY_DRAIN_SECONDS:-2}"
CONSUMER_CMD="${REPLAY_CONSUMER_CMD:-}"
BROKER_PID=""
CONSUMER_PID=""

ADDRESSES=$("$PYTHON" - <<'PY' 2>/dev/null
from urllib.parse import urlparse
from core.config import get_settings

zmq_settings = get_settings().zmq
frontend = zmq_settings.broker_bind_frontend_address or zmq_settings.broker_frontend_address
backend = zmq_settings.broker_bind_backend_address or zmq_settings.broker_backend_address
if not frontend or not backend:
    raise SystemExit(2)

def host_port(address):
    parsed = urlparse(address)
    return ("127.0.0.1" if parsed.hostname in (None, "", "0.0.0.0", "*") else parsed.hostname), parsed.port

print(*host_port(frontend), *host_port(backend))
PY
) || { echo "no broker addresses: set ZMQ_BROKER_FRONTEND_ADDRESS and ZMQ_BROKER_BACKEND_ADDRESS (environment or .env)" >&2; exit 2; }
read -r FE_HOST FE_PORT BE_HOST BE_PORT <<<"$ADDRESSES"

stop_process() {
    local pid="$1"
    [[ -z "$pid" ]] && return
    kill "$pid" 2>/dev/null || return
    for _ in 1 2 3 4 5 6 7 8 9 10; do
        kill -0 "$pid" 2>/dev/null || return
        sleep 0.5
    done
    kill -9 "$pid" 2>/dev/null
}

cleanup() {
    trap - EXIT INT TERM
    stop_process "$CONSUMER_PID"
    stop_process "$BROKER_PID"
}
trap cleanup EXIT INT TERM

port_is_open() {
    "$PYTHON" -c "import socket, sys; s = socket.socket(); s.settimeout(1); sys.exit(0 if s.connect_ex(('$1', $2)) == 0 else 1)"
}

wait_for_broker() {
    # The broker logs zmq_broker_started only after both binds succeeded. Checking the port alone would
    # accept another program already listening there while the broker itself failed to bind.
    local deadline=$((SECONDS + READY_TIMEOUT))
    until grep -q '"zmq_broker_started"' "$BROKER_LOG" 2>/dev/null; do
        if ! kill -0 "$BROKER_PID" 2>/dev/null; then
            echo "broker exited before binding its sockets (see $BROKER_LOG)" >&2
            return 1
        fi
        if ((SECONDS >= deadline)); then
            echo "timed out after ${READY_TIMEOUT}s waiting for the broker to bind (see $BROKER_LOG)" >&2
            return 1
        fi
        sleep 0.5
    done
    local host port
    for endpoint in "$FE_HOST $FE_PORT" "$BE_HOST $BE_PORT"; do
        read -r host port <<<"$endpoint"
        port_is_open "$host" "$port" || { echo "broker reported started but $host:$port does not accept connections" >&2; return 1; }
    done
}

BROKER_LOG="${REPLAY_BROKER_LOG:-$(mktemp)}"
echo "[replay] starting broker (log: $BROKER_LOG)"
"$PYTHON" scripts/run_zmq_broker.py >"$BROKER_LOG" 2>&1 &
BROKER_PID=$!

wait_for_broker || exit 1
echo "[replay] broker sockets bound: frontend $FE_HOST:$FE_PORT, backend $BE_HOST:$BE_PORT"

if [[ -n "$CONSUMER_CMD" ]]; then
    echo "[replay] starting consumer, ${WARMUP}s to subscribe"
    bash -c "$CONSUMER_CMD" &
    CONSUMER_PID=$!
    sleep "$WARMUP"
    kill -0 "$CONSUMER_PID" 2>/dev/null || { echo "consumer exited during warm-up" >&2; exit 1; }
fi

echo "[replay] ingesting $CSV"
"$PYTHON" scripts/historical_ingester.py "$CSV" "$@"
INGESTER_STATUS=$?

if [[ -n "$CONSUMER_PID" ]]; then
    echo "[replay] ingester finished (exit $INGESTER_STATUS); ${DRAIN}s for the consumer to drain"
    sleep "$DRAIN"
fi
exit "$INGESTER_STATUS"
