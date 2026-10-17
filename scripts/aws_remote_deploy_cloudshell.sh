#!/usr/bin/env bash
# À COLLER DANS AWS CloudShell (console AWS, icône >_, région Tokyo). Déploie la collecte de données SUR l'instance, par SSM : aucun SSH, aucun terminal à ouvrir.
#
# Sur l'instance (utilisateur ubuntu) : clone le dépôt (clé de déploiement déjà posée par le script de lancement), crée .env / platform.env /
# reporter.env avec des secrets générés SUR PLACE (jamais affichés, jamais écrasés s'ils existent), génère les clés CURVE, puis lance
# scripts/deploy_shadow.sh --up pour les seuls services de collecte. Affiche : droits des fichiers, fin du script, état des conteneurs, logs du tracker VPIN.
# Relançable sans risque : rien d'existant n'est écrasé (un mot de passe de base changé après la première initialisation casserait TimescaleDB).
#
# NON exécuté : écrit sans accès à l'instance (bash -n seulement). Prérequis : la clé « nothierly-deploy » est dans les Deploy keys du dépôt GitHub.
set -euo pipefail
export AWS_DEFAULT_REGION=ap-northeast-1
ID="${INSTANCE_ID:-i-0f3749477797afa0d}"
REPO="${REPO_SSH:-git@github.com:nothierly-hash/nothierly.git}"
BRANCH="${BRANCH:-red-team-audit}"
SERVICES_LIST="${SERVICES:-redis-broker timescaledb nothierly-agents nothierly-vpin-tracker nothierly-maintenance}"

PING=$(aws ssm describe-instance-information --filters Key=InstanceIds,Values="$ID" --query 'InstanceInformationList[0].PingStatus' --output text 2>/dev/null || true)
[ "$PING" = "Online" ] || { echo "ÉCHEC : l'instance $ID n'est pas joignable par SSM (état : ${PING:-inconnu}). Vérifier qu'elle est « Running » et que le rôle nothierly-ssm est attaché."; exit 1; }

cat > /tmp/remote.sh <<'REMOTE'
set -euo pipefail
cat > /tmp/nothierly-run.sh <<'UBU'
set -euo pipefail
cd "$HOME"
if [ ! -d nothierly/.git ]; then
  git clone -b "$BRANCH" "$REPO" nothierly
else
  git -C nothierly fetch origin "$BRANCH" && git -C nothierly checkout "$BRANCH" && git -C nothierly merge --ff-only "origin/$BRANCH"
fi
cd nothierly
echo "== commit déployé : $(git log --oneline -1)"
umask 077
[ -f .env ] || printf 'CLIENT_ID=nothierly-prod\nDB_USER=nothierly\nDB_PASSWORD=%s\nREDIS_PASSWORD=%s\n' "$(python3 -c 'import secrets;print(secrets.token_hex(32))')" "$(python3 -c 'import secrets;print(secrets.token_hex(32))')" > .env
[ -f platform.env ] || printf 'NOTHIERLY_ZMQ_SECRET=%s\n' "$(python3 -c 'import secrets;print(secrets.token_hex(32))')" > platform.env
[ -f reporter.env ] || : > reporter.env
chmod 600 .env platform.env reporter.env
[ -f secrets/curve/curve_server.env ] || python3 scripts/generate_curve_keys.py --clients hyperliquid bybit binance --out-dir secrets/curve
echo "== droits des fichiers (aucun secret affiché)"
ls -la .env platform.env reporter.env secrets/curve
echo "== déploiement"
SERVICES="$SERVICES_LIST" bash scripts/deploy_shadow.sh --up 2>&1 | tee "$HOME/deploy.log" | tail -45
echo "== conteneurs"
F="-f docker-compose.yml -f docker-compose.platform.yml -f docker-compose.shadow.yml"
docker compose $F ps 2>&1 | tail -15 || sudo docker compose $F ps 2>&1 | tail -15
sleep 45
echo "== tracker VPIN (dernières lignes)"
docker compose $F logs --tail 25 nothierly-vpin-tracker 2>&1 | tail -25 || sudo docker compose $F logs --tail 25 nothierly-vpin-tracker 2>&1 | tail -25
UBU
chmod 755 /tmp/nothierly-run.sh
chown ubuntu:ubuntu /tmp/nothierly-run.sh
sudo -u ubuntu -H env BRANCH="__BRANCH__" REPO="__REPO__" SERVICES_LIST="__SERVICES__" bash /tmp/nothierly-run.sh
REMOTE
sed -i "s|__BRANCH__|$BRANCH|; s|__REPO__|$REPO|; s|__SERVICES__|$SERVICES_LIST|" /tmp/remote.sh
python3 -c 'import json;print(json.dumps({"commands":[open("/tmp/remote.sh").read()],"executionTimeout":["3600"]}))' > /tmp/params.json

CMD=$(aws ssm send-command --instance-ids "$ID" --document-name AWS-RunShellScript --timeout-seconds 3600 --parameters file:///tmp/params.json --query Command.CommandId --output text)
echo "Commande SSM envoyée ($CMD). Le premier build peut durer 10 à 25 minutes ; ne fermez pas CloudShell."
for i in $(seq 1 240); do
  STATUS=$(aws ssm get-command-invocation --command-id "$CMD" --instance-id "$ID" --query Status --output text 2>/dev/null || echo Pending)
  case "$STATUS" in Success|Failed|Cancelled|TimedOut) break ;; esac
  printf '.'; sleep 15
done
echo; echo "ÉTAT : $STATUS"
aws ssm get-command-invocation --command-id "$CMD" --instance-id "$ID" --query StandardOutputContent --output text
echo "---- erreurs (fin) ----"
aws ssm get-command-invocation --command-id "$CMD" --instance-id "$ID" --query StandardErrorContent --output text | tail -25
