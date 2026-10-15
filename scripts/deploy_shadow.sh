#!/usr/bin/env bash
# Déploiement du Shadow Run sur un serveur Ubuntu 22.04/24.04 (à lancer SUR le serveur, depuis la racine du dépôt).
#
#   scripts/deploy_shadow.sh            # vérifie tout, ne démarre RIEN
#   scripts/deploy_shadow.sh --up       # idem, puis `docker compose up -d --build` si tout est vert
#   SERVICES="redis-broker timescaledb nothierly-agents nothierly-vpin-tracker nothierly-maintenance" scripts/deploy_shadow.sh --up
#                                       # ne démarre QUE ces services (ex. collecte de données sans workers : sans clés de testnet, un worker redémarre en boucle)
#
# Ce que ce script fait : installe Docker (dépôt officiel) s'il manque, vérifie les prérequis, crée les 4 volumes `external`,
# valide la configuration Compose, joue la maintenance à blanc, puis (avec --up) démarre la pile.
# Ce qu'il ne fait JAMAIS : créer, deviner ou afficher un secret. Les fichiers .env / platform.env / reporter.env / secrets/curve/ doivent
# exister (modèles : platform.env.example, reporter.env.example ; clés : scripts/generate_curve_keys.py). Il n'ouvre aucun port.
#
# NON exécuté : écrit sans accès à un serveur Ubuntu ni à Docker (`bash -n` seulement). Lire avant de lancer.

set -euo pipefail

UP=0
case "${1:-}" in
  "") ;;
  --up) UP=1 ;;
  *) echo "usage : $0 [--up]" >&2; exit 2 ;;
esac

cd "$(dirname "$0")/.."
log() { printf '[deploy] %s\n' "$*"; }
die() { printf '[deploy] ÉCHEC : %s\n' "$*" >&2; exit 1; }

# docker-compose.prod.yml est VOLONTAIREMENT absent : il met redis-broker, timescaledb et le moteur sur le seul réseau `internal_core`, alors que les services de la
# plateforme et du Shadow Run (pont, tracker VPIN, reporter, maintenance, exporteur) n'en déclarent aucun et resteraient sur le réseau par défaut, donc sans accès à Redis
# ni à TimescaleDB. L'empiler exige d'ajouter `networks: [internal_core]` à tous ces services (non fait, non testé sous Docker).
COMPOSE_FILES=(-f docker-compose.yml -f docker-compose.platform.yml -f docker-compose.shadow.yml)
VOLUMES=(nothierly_shadow_data nothierly_shadow_logs nothierly_timescaledb_data nothierly_redis_data)

# --- 1. Système --------------------------------------------------------------------------------
[ -r /etc/os-release ] || die "/etc/os-release illisible"
# shellcheck disable=SC1091
. /etc/os-release
[ "${ID:-}" = "ubuntu" ] || die "Ubuntu attendu (trouvé : ${ID:-inconnu}). Ce script installe Docker avec apt."
free_gb=$(df -BG --output=avail . | tail -1 | tr -dc '0-9')
mem_mb=$(free -m | awk '/^Mem:/ {print $2}')
[ "$free_gb" -ge 50 ] || die "moins de 50 Go libres ici (${free_gb} Go) : TimescaleDB et les archives en demandent davantage"
[ "$mem_mb" -ge 7000 ] || log "ATTENTION : ${mem_mb} Mo de RAM ; 16 Go recommandés (le moteur charge torch et FinBERT)"

# --- 2. Docker ---------------------------------------------------------------------------------
if ! command -v docker >/dev/null 2>&1; then
  log "installation de Docker (dépôt officiel Ubuntu)"
  sudo apt-get update -y
  sudo apt-get install -y ca-certificates curl
  sudo install -m 0755 -d /etc/apt/keyrings
  sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
  sudo chmod a+r /etc/apt/keyrings/docker.asc
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu ${UBUNTU_CODENAME:-$VERSION_CODENAME} stable" \
    | sudo tee /etc/apt/sources.list.d/docker.list >/dev/null
  sudo apt-get update -y
  sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
  sudo systemctl enable --now docker
fi
if docker info >/dev/null 2>&1; then DOCKER=(docker); else
  sudo docker info >/dev/null 2>&1 || die "le démon Docker ne répond pas"
  DOCKER=(sudo docker)
  log "docker via sudo (l'utilisateur n'est pas encore dans le groupe docker : sudo usermod -aG docker \$USER puis se reconnecter)"
fi
"${DOCKER[@]}" compose version >/dev/null 2>&1 || die "le plugin « docker compose » manque"

# --- 3. Prérequis (jamais créés ici) -----------------------------------------------------------
missing=()
for f in .env platform.env reporter.env; do [ -f "$f" ] || missing+=("$f"); done
ls secrets/curve/curve_server.env >/dev/null 2>&1 || missing+=("secrets/curve/ (python scripts/generate_curve_keys.py --clients hyperliquid bybit binance --out-dir secrets/curve)")
[ "${#missing[@]}" -eq 0 ] || die "fichiers manquants : ${missing[*]}"
for var in CLIENT_ID REDIS_PASSWORD DB_USER DB_PASSWORD; do
  grep -Eq "^${var}=.+" .env || die "${var} absent ou vide dans .env"
done
grep -Eq '^NOTHIERLY_ZMQ_SECRET=.{32,}' platform.env || die "NOTHIERLY_ZMQ_SECRET (>= 32 caractères) absent de platform.env"
grep -Eq '^NOTHIERLY_RECONCILE_VENUES=.+' .env || log "NOTE : NOTHIERLY_RECONCILE_VENUES absent de .env : le moteur repartira à plat après un redémarrage (voir docs, « Réconciliation »)"
for f in .env platform.env reporter.env; do
  [ "$(stat -c '%a' "$f")" -le 640 ] || die "$f est lisible par tous : chmod 600 $f"
done

# --- 4. Volumes `external` ---------------------------------------------------------------------
for v in "${VOLUMES[@]}"; do
  "${DOCKER[@]}" volume inspect "$v" >/dev/null 2>&1 || { log "création du volume $v"; "${DOCKER[@]}" volume create "$v" >/dev/null; }
done

# --- 5. Validation à blanc ---------------------------------------------------------------------
log "docker compose config"
"${DOCKER[@]}" compose "${COMPOSE_FILES[@]}" config -q || die "la configuration Compose est invalide (rien n'a été démarré)"
log "maintenance à blanc (ne modifie rien)"
"${DOCKER[@]}" compose "${COMPOSE_FILES[@]}" run --rm --build nothierly-maintenance python scripts/maintenance_cron.py --once --dry-run --skip-timescale \
  || die "la maintenance à blanc a échoué (rien n'a été démarré)"

if [ "$UP" -eq 0 ]; then
  log "tout est vert. Rien n'a été démarré : relancer avec --up pour démarrer la pile."
  exit 0
fi

# --- 6. Démarrage ------------------------------------------------------------------------------
log "démarrage (build puis up -d)"
# shellcheck disable=SC2086 -- SERVICES est une liste de noms séparés par des espaces, volontairement non citée
"${DOCKER[@]}" compose "${COMPOSE_FILES[@]}" up -d --build ${SERVICES:-}
sleep 20
"${DOCKER[@]}" compose "${COMPOSE_FILES[@]}" ps
log "à surveiller : docker compose ... logs -f nothierly-vpin-tracker nothierly-position-reporter nothierly-maintenance"
