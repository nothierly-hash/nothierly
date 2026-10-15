#!/usr/bin/env bash
# Crée .env / platform.env / reporter.env pour le Shadow Run (collecte publique) à partir des modèles, avec des secrets aléatoires.
# À lancer SUR le serveur, depuis n'importe où : `bash scripts/setup_shadow_env.sh` (puis `bash scripts/deploy_shadow.sh --up`).
#
#   bash scripts/setup_shadow_env.sh              crée les 3 fichiers (refuse s'ils existent déjà)
#   bash scripts/setup_shadow_env.sh --complete   fichiers existants : ne renseigne QUE les valeurs vides, ne change rien d'autre
#                                                 (sans danger après l'initialisation de TimescaleDB : aucun mot de passe renseigné n'est touché)
#   bash scripts/setup_shadow_env.sh --force      recrée tout (régénère les mots de passe : casse l'accès à une base DÉJÀ initialisée)
#
# Valeurs renseignées : CLIENT_ID, MOCK_MODE=true, REDIS_PASSWORD, DB_PASSWORD (+ TIMESCALE_PASSWORD identique), NOTHIERLY_ZMQ_SECRET,
# VAULT_MASTER_KEY (aléatoire), et des VALEURS BIDON non secrètes pour BROKER_API_KEY/SECRET, OANDA_API_TOKEN/ACCOUNT_ID : le moteur refuse de
# démarrer sans elles (core/config.py, « required, no default ») alors qu'avec MOCK_MODE=true il n'appelle aucun broker. Aucune vraie clé
# d'exchange n'est écrite ici. N'affiche JAMAIS un secret.

set -euo pipefail

MODE=create
case "${1:-}" in
  "") ;;
  --force) MODE=force ;;
  --complete) MODE=complete ;;
  *) echo "usage : $0 [--force|--complete]" >&2; exit 2 ;;
esac

cd "$(dirname "$0")/.."

PLACEHOLDER="shadow-run-placeholder"

for f in .env platform.env reporter.env; do
  [ -f "$f.example" ] || { echo "[setup] ÉCHEC : modèle $f.example introuvable" >&2; exit 1; }
  if [ "$MODE" = create ] && [ -e "$f" ]; then
    echo "[setup] $f existe déjà : rien n'est modifié (--complete pour renseigner les valeurs vides, --force pour tout recréer)." >&2
    exit 1
  fi
  if [ "$MODE" = complete ] && [ ! -f "$f" ]; then
    echo "[setup] ÉCHEC : $f absent : --complete suppose des fichiers existants (lancer sans option)." >&2
    exit 1
  fi
done

gen() {  # 48 caractères hex ; échoue bruyamment si Python manque (sinon un mot de passe VIDE serait écrit sans erreur)
  local v
  v="$(python3 -c "import secrets; print(secrets.token_hex(24))")" || return 1
  [ "${#v}" -eq 48 ] || { echo "[setup] ÉCHEC : génération de secret invalide" >&2; return 1; }
  printf '%s' "$v"
}

setk() {  # fichier clé valeur : écrit la valeur (remplace la ligne ou l'ajoute)
  if grep -q "^$2=" "$1"; then
    sed -i "s|^$2=.*|$2=$3|" "$1"
  else
    echo "$2=$3" >> "$1"
  fi
}

fillk() {  # fichier clé valeur : n'écrit que si la clé est absente ou VIDE (jamais d'écrasement)
  grep -Eq "^$2=.+" "$1" || setk "$1" "$2" "$3"
}

# Secrets générés AVANT toute écriture (un échec ne laisse donc aucun fichier à moitié rempli). Affectations simples :
# un échec de gen() arrête le script (dans un argument de fonction, set -e ne le verrait pas).
redis_password="$(gen)"
db_password="$(gen)"
zmq_secret="$(gen)"
vault_key="$(gen)"

if [ "$MODE" != complete ]; then
  rm -f .env platform.env reporter.env
  cp .env.example .env
  cp platform.env.example platform.env
  cp reporter.env.example reporter.env
fi
chmod 600 .env platform.env reporter.env

fillk .env CLIENT_ID nothierly
setk .env MOCK_MODE true
fillk .env REDIS_PASSWORD "$redis_password"
fillk .env DB_PASSWORD "$db_password"
fillk .env TIMESCALE_PASSWORD "$(grep -E '^DB_PASSWORD=' .env | cut -d= -f2-)"
fillk platform.env NOTHIERLY_ZMQ_SECRET "$zmq_secret"
fillk .env VAULT_MASTER_KEY "$vault_key"
fillk .env BROKER_API_KEY "$PLACEHOLDER"
fillk .env BROKER_API_SECRET "$PLACEHOLDER"
fillk .env OANDA_API_TOKEN "$PLACEHOLDER"
fillk .env OANDA_ACCOUNT_ID "$PLACEHOLDER"

echo "[setup] mode=$MODE, droits 600. Valeurs renseignées :"
grep -E "^(CLIENT_ID|MOCK_MODE|DB_USER|REDIS_PASSWORD|DB_PASSWORD|TIMESCALE_PASSWORD|VAULT_MASTER_KEY|BROKER_API_KEY|BROKER_API_SECRET|OANDA_API_TOKEN|OANDA_ACCOUNT_ID)=" .env | sed -E 's/=.+/=<rempli>/'
grep -E "^NOTHIERLY_ZMQ_SECRET=" platform.env | sed -E 's/=.+/=<rempli>/'
