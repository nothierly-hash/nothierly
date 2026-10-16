#!/usr/bin/env bash
# Enregistre les clés d'API de TESTNET d'une plateforme dans platform.env, SANS jamais les afficher (saisie masquée, rien dans l'historique du shell,
# rien dans une commande ni dans un chat). À lancer SUR le serveur :
#
#   bash scripts/set_exchange_keys.sh hyperliquid   # adresse du compte + clé de l'« agent wallet » (HL_ACCOUNT_ADDRESS, HL_AGENT_SECRET)
#   bash scripts/set_exchange_keys.sh bybit         # BYBIT_API_KEY, BYBIT_API_SECRET  (clés du testnet Bybit)
#   bash scripts/set_exchange_keys.sh binance       # BINANCE_API_KEY, BINANCE_API_SECRET  (clés du testnet Binance Futures)
#
# Les workers démarrent en dry-run sur le réseau de TEST (docker-compose.platform.yml) : aucune clé de mainnet ne doit être saisie ici. Créer les
# clés avec le moins de droits possible (lecture + trading, JAMAIS de retrait). Le fichier est écrit en droits 600.

set -euo pipefail

venue="${1:-}"
case "$venue" in
  hyperliquid) fields=("HL_ACCOUNT_ADDRESS:visible:^0x[0-9a-fA-F]{40}$" "HL_AGENT_SECRET:secret:^(0x)?[0-9a-fA-F]{64}$") ;;
  bybit)       fields=("BYBIT_API_KEY:secret:^[A-Za-z0-9]{8,64}$" "BYBIT_API_SECRET:secret:^[A-Za-z0-9]{16,128}$") ;;
  binance)     fields=("BINANCE_API_KEY:secret:^[A-Za-z0-9]{16,128}$" "BINANCE_API_SECRET:secret:^[A-Za-z0-9]{16,128}$") ;;
  *) echo "usage : $0 hyperliquid|bybit|binance" >&2; exit 2 ;;
esac

cd "$(dirname "$0")/.."
[ -f platform.env ] || { echo "[keys] ÉCHEC : platform.env absent (lancer d'abord scripts/setup_shadow_env.sh)" >&2; exit 1; }

set_var() {  # fichier clé valeur : remplace la ligne (sans sed : une clé peut contenir n'importe quel caractère)
  local tmp
  tmp="$(mktemp)"
  { grep -v "^$2=" "$1" || true; } > "$tmp"
  printf '%s=%s\n' "$2" "$3" >> "$tmp"
  cat "$tmp" > "$1"
  rm -f "$tmp"
}

echo "[keys] plateforme : $venue (TESTNET uniquement). La saisie est masquée ; rien n'est affiché."
declare -A values=()
for spec in "${fields[@]}"; do
  name="${spec%%:*}"; rest="${spec#*:}"; mode="${rest%%:*}"; pattern="${rest#*:}"
  printf '%s : ' "$name" >&2
  if [ "$mode" = secret ]; then read -rs value; echo >&2; else read -r value; fi
  value="${value//[[:space:]]/}"  # un collage ajoute souvent un espace ou un retour de ligne
  if [ -z "$value" ]; then echo "[keys] ÉCHEC : $name vide (rien n'est modifié)" >&2; exit 1; fi
  if ! [[ "$value" =~ $pattern ]]; then echo "[keys] ÉCHEC : $name n'a pas le format attendu (rien n'est modifié)" >&2; exit 1; fi
  values["$name"]="$value"
done

chmod 600 platform.env
for spec in "${fields[@]}"; do
  name="${spec%%:*}"
  set_var platform.env "$name" "${values[$name]}"
done
echo "[keys] enregistré dans platform.env (droits 600) :"
for spec in "${fields[@]}"; do echo "  ${spec%%:*}=<enregistré>"; done
