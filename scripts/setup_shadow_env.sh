#!/usr/bin/env bash
# Crée .env / platform.env / reporter.env pour le Shadow Run (collecte publique) à partir des modèles, avec des secrets aléatoires.
# À lancer SUR le serveur, depuis n'importe où : `bash scripts/setup_shadow_env.sh` (puis `bash scripts/deploy_shadow.sh --up`).
#
# Remplit seulement ce que la collecte exige : CLIENT_ID, MOCK_MODE=true, REDIS_PASSWORD, DB_PASSWORD (+ TIMESCALE_PASSWORD identique),
# NOTHIERLY_ZMQ_SECRET. Aucune clé d'exchange. N'affiche JAMAIS un secret. Ne touche pas un fichier existant sauf avec --force
# (régénérer les mots de passe après l'initialisation de TimescaleDB casserait l'accès à la base : d'où le refus par défaut).

set -euo pipefail

FORCE=0
case "${1:-}" in
  "") ;;
  --force) FORCE=1 ;;
  *) echo "usage : $0 [--force]" >&2; exit 2 ;;
esac

cd "$(dirname "$0")/.."

for f in .env platform.env reporter.env; do
  [ -f "$f.example" ] || { echo "[setup] ÉCHEC : modèle $f.example introuvable" >&2; exit 1; }
  if [ -e "$f" ] && [ "$FORCE" -eq 0 ]; then
    echo "[setup] $f existe déjà : rien n'est modifié (relancer avec --force pour le recréer)." >&2
    exit 1
  fi
done

gen() {  # 48 caractères hex ; échoue bruyamment si Python manque (sinon un mot de passe VIDE serait écrit sans erreur)
  local v
  v="$(python3 -c "import secrets; print(secrets.token_hex(24))")" || return 1
  [ "${#v}" -eq 48 ] || { echo "[setup] ÉCHEC : génération de secret invalide" >&2; return 1; }
  printf '%s' "$v"
}

setk() {  # fichier clé valeur
  if grep -q "^$2=" "$1"; then
    sed -i "s|^$2=.*|$2=$3|" "$1"
  else
    echo "$2=$3" >> "$1"
  fi
}

# Secrets générés AVANT toute écriture (un échec ne laisse donc aucun fichier à moitié rempli). Affectations simples : un échec de gen() arrête le script (dans un argument de fonction, set -e ne le verrait pas).
redis_password="$(gen)"
db_password="$(gen)"
zmq_secret="$(gen)"

rm -f .env platform.env reporter.env
cp .env.example .env
cp platform.env.example platform.env
cp reporter.env.example reporter.env
chmod 600 .env platform.env reporter.env

setk .env CLIENT_ID nothierly
setk .env MOCK_MODE true
setk .env REDIS_PASSWORD "$redis_password"
setk .env DB_PASSWORD "$db_password"
setk .env TIMESCALE_PASSWORD "$db_password"
setk platform.env NOTHIERLY_ZMQ_SECRET "$zmq_secret"

echo "[setup] fichiers créés (droits 600). Valeurs renseignées :"
grep -E "^(CLIENT_ID|MOCK_MODE|DB_USER|REDIS_PASSWORD|DB_PASSWORD|TIMESCALE_PASSWORD)=" .env | sed -E 's/=.+/=<rempli>/'
grep -E "^NOTHIERLY_ZMQ_SECRET=" platform.env | sed -E 's/=.+/=<rempli>/'
