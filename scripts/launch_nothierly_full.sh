#!/usr/bin/env bash
# Séquence d'allumage de Nothierly : Redis -> pré-vol -> VPIN -> pont CURVE -> workers -> cerveau.
#
#   scripts/launch_nothierly_full.sh plan                      # affiche la séquence et les commandes, ne démarre RIEN
#   scripts/launch_nothierly_full.sh start                     # dry-run sur les réseaux de test (défaut)
#   scripts/launch_nothierly_full.sh start --live-venues "bybit" --network mainnet --i-understand-real-funds
#   scripts/launch_nothierly_full.sh status
#   scripts/launch_nothierly_full.sh stop
#
# Ordre STRICT, chaque étape attend sa condition de santé avant la suivante ; au premier échec tout ce que CE lancement a démarré est arrêté :
#   0. Redis            joignable (sinon démarré : redis-server, ou le faux Redis de test avec --allow-fake-redis)
#   1. Pré-vol          scripts/system_health_check.py : bloquant (jamais sauté en mode réel)
#   2. Démon VPIN       trades publics -> market:vpin:<SYMBOLE> dans Redis (aucune clé d'exchange)
#   3. Pont             publisher ZMQ signé et chiffré en CURVE, écoute sur $ENDPOINT (port à ouvrir pour le nœud Windows Darwinex)
#   4. Workers          Binance, Bybit, Hyperliquid (suivi des fills avec --live seulement : en dry-run aucun ordre ne part)
#   5. Cerveau          moteur complet (scripts/run_backend.py : agents de marché, PPO, XGBoost, ExecutionAgent) ; réutilisé s'il tourne déjà
#
# Garde-fous du mode réel (--live-venues ou --live) : --i-understand-real-funds, MOCK_MODE=false explicite (un moteur simulé ne doit
# jamais piloter de l'argent réel), pré-vol --live sans échec, homme mort actif (défaut des workers). Sans cela le script refuse.
#
# Le nœud Windows/Darwinex se lance à part (workers/darwinex) ; ce script n'en démarre aucun mais le pont écoute sur une adresse non
# locale : le chiffrement CURVE est donc exigé, et le port doit être ouvert au pare-feu.
#
# NON ÉPROUVÉ en démarrage complet : écrit et vérifié par `bash -n` et le mode `plan` ; les workers ne peuvent pas démarrer sans clés.

set -u
set -o pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1

COMMAND="${1:-help}"
[ $# -gt 0 ] && shift

NETWORK="testnet"
VENUES="hyperliquid bybit binance"
LIVE_VENUES=""
LIVE_ALL=0
ACK_REAL_FUNDS=0
ALLOW_FAKE_REDIS=0
SKIP_BRAIN=0
SKIP_HEALTHCHECK=0
ENDPOINT="tcp://0.0.0.0:25570"
VPIN_SOURCE="binance"
VPIN_SYMBOLS="BTCUSDT ETHUSDT"
ENGINE_ENV=".env"
PLATFORM_ENV="platform.env"
CURVE_DIR="secrets/curve"
RUN_DIR="run"
LOG_DIR="logs/launch"

while [ $# -gt 0 ]; do
  case "$1" in
    --network) NETWORK="$2"; shift 2 ;;
    --venues) VENUES="$2"; shift 2 ;;
    --live) LIVE_ALL=1; shift ;;
    --live-venues) LIVE_VENUES="$2"; shift 2 ;;
    --i-understand-real-funds) ACK_REAL_FUNDS=1; shift ;;
    --allow-fake-redis) ALLOW_FAKE_REDIS=1; shift ;;
    --skip-brain) SKIP_BRAIN=1; shift ;;
    --skip-healthcheck) SKIP_HEALTHCHECK=1; shift ;;
    --endpoint) ENDPOINT="$2"; shift 2 ;;
    --vpin-source) VPIN_SOURCE="$2"; shift 2 ;;
    --symbols) VPIN_SYMBOLS="$2"; shift 2 ;;
    --curve-dir) CURVE_DIR="$2"; shift 2 ;;
    -h|--help) COMMAND="help"; shift ;;
    *) echo "Option inconnue : $1" >&2; exit 2 ;;
  esac
done

[ "$LIVE_ALL" -eq 1 ] && [ -z "$LIVE_VENUES" ] && LIVE_VENUES="$VENUES"  # --live = toutes les plateformes, quel que soit l'ordre des options

if [ -x ".venv/Scripts/python.exe" ]; then PY=".venv/Scripts/python.exe"
elif [ -x ".venv/bin/python" ]; then PY=".venv/bin/python"
else PY="python"; fi

BRIDGE_PORT="${ENDPOINT##*:}"
STARTED=()   # composants démarrés par CE lancement, dans l'ordre

log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }
die() { log "ECHEC : $*" >&2; teardown; exit 1; }

# --- Utilitaires -----------------------------------------------------------------------------

env_get() {  # env_get FICHIER CLE : valeur d'une clé d'un fichier .env (sans l'exporter ailleurs)
  [ -f "$1" ] || return 0
  grep -E "^(export )?$2=" "$1" | tail -n 1 | sed -E "s/^(export )?$2=//; s/^['\"]//; s/['\"]$//"
}

redis_var() { local v; v="$(env_get "$ENGINE_ENV" "$1")"; echo "${v:-$2}"; }

pid_alive() { [ -f "$RUN_DIR/$1.pid" ] && kill -0 "$(cat "$RUN_DIR/$1.pid")" 2>/dev/null; }

redis_ping() {
  REDIS_HOST="$(redis_var REDIS_HOST 127.0.0.1)" REDIS_PORT="$(redis_var REDIS_PORT 6379)" REDIS_PASSWORD="$(redis_var REDIS_PASSWORD '')" \
    "$PY" - <<'PYEOF' >/dev/null 2>&1
import os, redis
r = redis.Redis(host=os.environ["REDIS_HOST"], port=int(os.environ["REDIS_PORT"]), password=os.environ.get("REDIS_PASSWORD") or None, socket_timeout=2, socket_connect_timeout=2)
raise SystemExit(0 if r.ping() else 1)
PYEOF
}

engine_alive() {
  REDIS_HOST="$(redis_var REDIS_HOST 127.0.0.1)" REDIS_PORT="$(redis_var REDIS_PORT 6379)" REDIS_PASSWORD="$(redis_var REDIS_PASSWORD '')" \
    "$PY" - <<'PYEOF' >/dev/null 2>&1
import os, redis
r = redis.Redis(host=os.environ["REDIS_HOST"], port=int(os.environ["REDIS_PORT"]), password=os.environ.get("REDIS_PASSWORD") or None, socket_timeout=2, socket_connect_timeout=2)
raise SystemExit(0 if r.exists("heartbeat:execution-agent") else 1)
PYEOF
}

port_listening() {  # port_listening PORT
  "$PY" -c "import socket,sys; s=socket.socket(); s.settimeout(1); sys.exit(0 if s.connect_ex(('127.0.0.1', int(sys.argv[1]))) == 0 else 1)" "$1" 2>/dev/null
}

wait_for() {  # wait_for "description" délai_s commande...
  local what="$1" limit="$2"; shift 2
  local waited=0
  while ! "$@"; do
    sleep 1; waited=$((waited + 1))
    [ "$waited" -ge "$limit" ] && return 1
  done
  log "  prêt : $what (${waited}s)"
}

is_live() { case " $LIVE_VENUES " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }

# start_component NOM "fichiers .env à charger (séparés par des espaces)" commande...
start_component() {
  local name="$1" files="$2"; shift 2
  mkdir -p "$RUN_DIR" "$LOG_DIR"
  if pid_alive "$name"; then die "$name tourne déjà (pid $(cat "$RUN_DIR/$name.pid")) : l'arrêter avec « stop » d'abord"; fi
  (
    set -a
    for f in $files; do [ -f "$f" ] && . "./$f"; done
    set +a
    exec "$@"
  ) >> "$LOG_DIR/$name.log" 2>&1 &
  echo $! > "$RUN_DIR/$name.pid"
  STARTED+=("$name")
  sleep 2
  pid_alive "$name" || die "$name s'est arrêté dès le démarrage : voir $LOG_DIR/$name.log"
  log "  démarré : $name (pid $(cat "$RUN_DIR/$name.pid"), journal $LOG_DIR/$name.log)"
}

stop_component() {
  local name="$1"
  [ -f "$RUN_DIR/$name.pid" ] || return 0
  local pid; pid="$(cat "$RUN_DIR/$name.pid")"
  if kill -0 "$pid" 2>/dev/null; then
    kill "$pid" 2>/dev/null
    for _ in 1 2 3 4 5 6 7 8 9 10; do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
    kill -0 "$pid" 2>/dev/null && { kill -9 "$pid" 2>/dev/null; log "  $name forcé (SIGKILL)"; }
    log "  arrêté : $name"
  fi
  rm -f "$RUN_DIR/$name.pid"
}

teardown() {  # arrête, en ordre inverse, ce que CE lancement a démarré
  local i
  for ((i=${#STARTED[@]}-1; i>=0; i--)); do stop_component "${STARTED[$i]}"; done
}

# --- Commandes -------------------------------------------------------------------------------

worker_args() {  # worker_args VENUE : arguments communs
  local venue="$1"
  local args=(--address "tcp://127.0.0.1:$BRIDGE_PORT" --network "$NETWORK" --db "$RUN_DIR/data/$venue.sqlite3")
  if is_live "$venue"; then
    args+=(--live)
    [ "$NETWORK" = "mainnet" ] && args+=(--i-understand-real-funds)
  fi
  printf '%s\n' "${args[@]}"
}

health_args() {
  local args=(--network "$NETWORK" --venues $VENUES --endpoint "$ENDPOINT" --curve-dir "$CURVE_DIR" --data-dir "$RUN_DIR/data" --vpin-source "$VPIN_SOURCE" --symbols $VPIN_SYMBOLS)
  [ -n "$LIVE_VENUES" ] && args+=(--live)
  printf '%s\n' "${args[@]}"
}

check_live_guards() {
  [ -z "$LIVE_VENUES" ] && return 0
  [ "$ACK_REAL_FUNDS" -eq 1 ] || die "mode réel demandé sans --i-understand-real-funds"
  [ "$SKIP_HEALTHCHECK" -eq 0 ] || die "le pré-vol ne peut pas être sauté en mode réel"
  local mock; mock="$(env_get "$ENGINE_ENV" MOCK_MODE)"; mock="${MOCK_MODE:-$mock}"
  case "$(echo "${mock:-true}" | tr '[:upper:]' '[:lower:]')" in
    false|0|no|off) ;;
    *) die "MOCK_MODE n'est pas explicitement à false : le moteur est simulé, des workers réels ne doivent pas suivre ses décisions" ;;
  esac
  for v in $LIVE_VENUES; do case " $VENUES " in *" $v "*) ;; *) die "--live-venues contient « $v » qui n'est pas dans --venues" ;; esac; done
}

cmd_plan() {
  echo "Séquence d'allumage (rien n'est démarré) — réseau : $NETWORK ; réel : ${LIVE_VENUES:-aucun (dry-run)}"
  echo "  0. Redis            : ping $(redis_var REDIS_HOST 127.0.0.1):$(redis_var REDIS_PORT 6379) ; sinon redis-server, ou --allow-fake-redis"
  echo "  1. Pré-vol          : $PY scripts/system_health_check.py $(health_args | tr '\n' ' ')"
  echo "  2. Démon VPIN       : $PY -m workers.market_data.vpin_tracker --source $VPIN_SOURCE --symbols $VPIN_SYMBOLS   (env : REDIS_* seulement)"
  echo "  3. Pont CURVE       : $PY scripts/run_platform_bridge.py --endpoint $ENDPOINT --venue ALL   (env : $ENGINE_ENV $PLATFORM_ENV $CURVE_DIR/curve_server.env)"
  local n=3
  for v in $VENUES; do
    n=$((n + 1))
    echo "  $n. Worker $v : $PY -m workers.$v.worker $(worker_args "$v" | tr '\n' ' ')  (env : $PLATFORM_ENV $CURVE_DIR/curve_$v.env)"
  done
  if [ "$SKIP_BRAIN" -eq 0 ]; then echo "  $((n + 1)). Cerveau          : $PY scripts/run_backend.py   (réutilisé si heartbeat:execution-agent existe déjà)"; fi
  echo "Journaux : $LOG_DIR/<composant>.log ; PID : $RUN_DIR/<composant>.pid"
}

cmd_start() {
  check_live_guards
  log "Allumage Nothierly — réseau $NETWORK, réel : ${LIVE_VENUES:-aucun (dry-run)}"
  mkdir -p "$RUN_DIR/data" "$LOG_DIR"

  log "0/5 Redis"
  if redis_ping; then log "  Redis déjà joignable : réutilisé"
  else
    local host port; host="$(redis_var REDIS_HOST 127.0.0.1)"; port="$(redis_var REDIS_PORT 6379)"
    case "$host" in 127.0.0.1|localhost) ;; *) die "Redis $host:$port injoignable et non local : le démarrer (docker compose) puis relancer" ;; esac
    if command -v redis-server >/dev/null 2>&1; then
      start_component redis "" redis-server --port "$port" --bind 127.0.0.1 --appendonly yes
    elif [ "$ALLOW_FAKE_REDIS" -eq 1 ]; then
      log "  ATTENTION : faux Redis en mémoire (état du kill switch perdu à l'arrêt) : test uniquement"
      start_component redis "" "$PY" scripts/run_fake_redis.py --host 127.0.0.1 --port "$port"
    else
      die "Redis $host:$port injoignable, pas de redis-server dans le PATH. Démarrer Redis, ou --allow-fake-redis pour un faux Redis de test"
    fi
    wait_for "Redis" 15 redis_ping || die "Redis ne répond pas après 15 s"
  fi

  log "1/5 Pré-vol (système de santé)"
  if [ "$SKIP_HEALTHCHECK" -eq 1 ]; then log "  pré-vol SAUTÉ (--skip-healthcheck) : déconseillé"
  else
    mapfile -t HARGS < <(health_args)
    "$PY" scripts/system_health_check.py "${HARGS[@]}" || die "le pré-vol a échoué : corriger ce qu'il liste, puis relancer (rien n'a été démarré après Redis)"
  fi

  log "2/5 Démon VPIN"
  REDIS_HOST="$(redis_var REDIS_HOST 127.0.0.1)" REDIS_PORT="$(redis_var REDIS_PORT 6379)" REDIS_PASSWORD="$(redis_var REDIS_PASSWORD '')" \
    start_component vpin "" "$PY" -m workers.market_data.vpin_tracker --source "$VPIN_SOURCE" --symbols $VPIN_SYMBOLS
  wait_for "flux de trades connecté" 60 grep -q "vpin_feed_connected" "$LOG_DIR/vpin.log" || die "le démon VPIN ne s'est pas connecté au flux public en 60 s : voir $LOG_DIR/vpin.log"

  log "3/5 Pont ZMQ (CURVE)"
  NOTHIERLY_ZMQ_ENCRYPTION=true start_component bridge "$ENGINE_ENV $PLATFORM_ENV $CURVE_DIR/curve_server.env" \
    "$PY" scripts/run_platform_bridge.py --endpoint "$ENDPOINT" --venue ALL
  wait_for "port $BRIDGE_PORT à l'écoute" 30 port_listening "$BRIDGE_PORT" || die "le pont n'écoute pas sur $BRIDGE_PORT : voir $LOG_DIR/bridge.log"

  log "4/5 Workers"
  for v in $VENUES; do
    mapfile -t WARGS < <(worker_args "$v")
    NOTHIERLY_ZMQ_ENCRYPTION=true start_component "worker-$v" "$PLATFORM_ENV $CURVE_DIR/curve_$v.env" "$PY" -m "workers.$v.worker" "${WARGS[@]}"
  done

  if [ "$SKIP_BRAIN" -eq 1 ]; then log "5/5 Cerveau : ignoré (--skip-brain)"
  elif engine_alive; then log "5/5 Cerveau : un moteur tourne déjà (heartbeat présent) : réutilisé, rien de lancé"
  else
    log "5/5 Cerveau (run_backend : agents de marché, PPO, XGBoost, ExecutionAgent)"
    start_component brain "$ENGINE_ENV" "$PY" scripts/run_backend.py
    wait_for "heartbeat de l'ExecutionAgent" 120 engine_alive || die "le moteur n'écrit pas son heartbeat en 120 s : voir $LOG_DIR/brain.log"
  fi

  log "Allumage terminé : ${#STARTED[@]} composant(s) démarré(s) par ce lancement. « status » pour l'état, « stop » pour tout arrêter."
}

cmd_status() {
  for name in redis vpin bridge $(for v in $VENUES; do echo "worker-$v"; done) brain; do
    if pid_alive "$name"; then echo "  UP    $name (pid $(cat "$RUN_DIR/$name.pid"))"; else echo "  --    $name (non démarré par ce script)"; fi
  done
  redis_ping && echo "  UP    redis (joignable)" || echo "  DOWN  redis (injoignable)"
  engine_alive && echo "  UP    moteur (heartbeat)" || echo "  DOWN  moteur (pas de heartbeat)"
  mapfile -t HARGS < <(health_args)
  "$PY" scripts/system_health_check.py "${HARGS[@]}" --running --no-network | tail -n 12
}

cmd_stop() {
  for name in brain $(for v in $(echo "$VENUES" | tr ' ' '\n' | tac); do echo "worker-$v"; done) bridge vpin redis; do stop_component "$name"; done
  log "Arrêt demandé pour tous les composants lancés par ce script."
}

case "$COMMAND" in
  plan) cmd_plan ;;
  start) cmd_start ;;
  status) cmd_status ;;
  stop) cmd_stop ;;
  *) sed -n '2,24p' "$0"; [ "$COMMAND" = "help" ] || exit 2 ;;
esac
