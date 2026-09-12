#!/usr/bin/env bash
# stress_test.sh -- real anti-DDoS validation for p2p_api.py's rate limiter (SOC 2 mission, task 5).
#
# Fires 50 real curl requests against the public sandbox endpoint concurrently (backgrounded, not
# a sequential loop -- a sequential loop's own per-process curl spawn overhead risks spilling past
# the real 1s window this test needs to land inside) -- the sandbox tier's own configured limit is
# 2 requests/second per client IP (core/config.py's P2PApiSettings.sandbox_rate_limit_per_second),
# enforced via Redis INCR/EXPIRE on a real 1-second window (p2p_api.py's own _enforce_rate_limit).
# All 50 requests come from this one machine (one real source IP), so they all land in the same
# window -- expect the first 2 to succeed (200) and the remaining 48 to be rejected (429), never a
# crash/500 (see _enforce_rate_limit's own fail-closed docstring for the Redis-downtime case).
#
# Usage: ./stress_test.sh [base_url]   (defaults to http://localhost:8001)

set -uo pipefail  # deliberately not -e: one curl's own failure must not abort the whole test

BASE_URL="${1:-http://localhost:8001}"
URL="${BASE_URL}/api/v1/sandbox/premiums/latest"
REQUEST_COUNT=50
TMP_DIR=$(mktemp -d)
trap 'rm -rf "${TMP_DIR}"' EXIT

echo "Envoi de ${REQUEST_COUNT} requetes concurrentes vers ${URL} ..."
echo ""

for i in $(seq 1 "${REQUEST_COUNT}"); do
    curl -s -o /dev/null -w "%{http_code}" "${URL}" > "${TMP_DIR}/${i}" &
done
wait

success=0
throttled=0
other=0
for i in $(seq 1 "${REQUEST_COUNT}"); do
    status=$(cat "${TMP_DIR}/${i}" 2>/dev/null || echo "ERR")
    printf "Requete %2d -> HTTP %s\n" "${i}" "${status}"
    case "${status}" in
        200) success=$((success + 1)) ;;
        429) throttled=$((throttled + 1)) ;;
        *) other=$((other + 1)) ;;
    esac
done

echo ""
echo "Resultat : ${success} x 200 OK, ${throttled} x 429 Too Many Requests, ${other} autre(s)."
if [ "${success}" -le 2 ] && [ "${throttled}" -ge $((REQUEST_COUNT - 2)) ]; then
    echo "Rate limiter valide : le serveur a bien bloque le trafic excedentaire."
else
    echo "ATTENTION : la distribution des codes ne correspond pas au comportement attendu (${success} succes, ${throttled} bloques)."
fi
