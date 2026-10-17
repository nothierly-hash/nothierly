# Protocole dry-run : valider la chaîne contre le vrai moteur, sans rien publier

Objectif : voir ce que le pont **publierait** à partir de l'état réel de l'`ExecutionAgent`, avant d'utiliser Docker et
avant de connecter le moindre worker. En `--dry-run` le pont lit Redis et journalise chaque `PlatformSignal` ; il
n'ouvre **aucun socket ZMQ**, ne lit **aucune clé HMAC ni CURVE** et ne parle à aucune plateforme.

## 1. Prérequis

- Le moteur tourne et écrit dans Redis (`scripts/run_backend.py`, ou la stack Docker du moteur). Le pont lit les mêmes
  clés que lui : `NOTHIERLY_ACTIVE_POSITIONS`, `NOTHIERLY_EQUITY_CURVE`, `heartbeat:execution-agent` et le statut du kill switch.
- Le `.env` du projet (celui du moteur) est présent : le pont charge la configuration complète du moteur
  (`core.config`), pas seulement Redis. Il lit `REDIS_HOST` / `REDIS_PORT` / `REDIS_PASSWORD` de ce fichier.

Vérifier d'abord que le moteur écrit bien son état (lecture seule) :

```powershell
.venv\Scripts\python.exe -c "import redis; r = redis.Redis(host='127.0.0.1', port=16399, decode_responses=True); print(r.get('NOTHIERLY_ACTIVE_POSITIONS')); print(r.ttl('heartbeat:execution-agent'))"
```

`ttl` doit être positif (le heartbeat est une clé à durée de vie courte) ; `-2` veut dire que le moteur ne tourne pas, et le
pont publierait `FLAT` après 30 s. Adapter le port à votre `.env`.

## 2. La commande

Depuis la racine du dépôt :

```powershell
.venv\Scripts\python.exe scripts\run_platform_bridge.py --dry-run
```

Équivalent bash : `.venv/Scripts/python scripts/run_platform_bridge.py --dry-run`. Arrêt : `Ctrl+C`.

Options utiles (les valeurs ci-dessous sont les valeurs par défaut) :

| Option | Défaut | Effet |
|---|---|---|
| `--instruments` | `BTC/USDT ETH/USDT` | paires suivies (seules les paires crypto `BASE/USDT` sont traduites) |
| `--max-position-pct` | `20` | plafond de la taille publiée, en % de l'equity du moteur |
| `--min-hold-s` | `60` | durée minimale avant de publier la fermeture/inversion d'une position publiée ; `0` = aucune |
| `--size-bucket-pct` | `2.5` | pas de quantification de la taille ; `0` publie la taille exacte et exige `--allow-exact-size` (comparaisons seulement : c'est la conviction du moteur) |
| `--poll-s` / `--refresh-s` | `2` / `10` | lecture du moteur / republication périodique |
| `--engine-grace-s` | `30` | délai avant de publier `FLAT` si le moteur se tait |

## 3. Ce qui est attendu

Au démarrage, un avertissement `MOCK_MODE actif` si le moteur est en simulation (normal : ses positions sont alors
simulées), puis `redis_connected`. Ensuite, une ligne par instrument et par changement :

```
bridge_dry_run_signal {"signal_id": ..., "symbol":"CRYPTO_BTC_USD","direction":"FLAT", ... "confidence_score":null ...}
bridge_signal instrument=BTC/USDT FLAT 0% venue=ALL reason=engine_state
```

Points à contrôler, dans l'ordre :

1. **Un signal par instrument au démarrage**, `FLAT` si le moteur ne tient rien.
2. **`confidence_score` toujours `null`** : le contrat l'interdit pour toute plateforme crypto.
3. **Quand le moteur ouvre une position** : `LONG`/`SHORT` avec une taille qui est un multiple de `--size-bucket-pct`
   (ou `--max-position-pct` si la taille exacte le dépasse), jamais `0` pour une position ouverte.
4. **Quand il la ferme** : `FLAT` dès que la position publiée a `--min-hold-s` (60 s par défaut) ; avant, une ligne `bridge_hold` indique que la fermeture est retenue. Un kill switch, lui, publie `FLAT` sans attendre.
5. **Pas de republication inutile** : un état stable ne reparaît qu'au rythme de `--refresh-s`.
6. **Repli** : arrêter le moteur (ou laisser expirer son heartbeat) doit produire `FLAT` avec `reason=engine_down` après
   `--engine-grace-s` ; un kill switch actif produit `FLAT` avec `reason=kill_switch` immédiatement. Ces deux cas sont couverts par les
   tests unitaires ; les rejouer à la main est facultatif.

`bridge_engine_state_unreadable` en boucle signifie que Redis est injoignable ou répond mal : ne pas continuer.

## 4. Ce que le dry-run ne prouve pas

- **Il ne prouve rien sur le bus** : ni la signature HMAC, ni CURVE, ni la livraison à un worker. Pour cela, un worker en
  dry-run (par défaut) sur testnet, branché au pont **sans** `--dry-run`, avec les clés CURVE (voir `docs/MULTI_PLATFORM_WORKER_ARCHITECTURE.md` §1.5).
- **Il ne prouve rien sur les plateformes** : aucun appel réseau n'est fait, les adaptateurs Hyperliquid, Bybit et
  MetaTrader 5 n'ont jamais tourné contre un vrai service.
- **Des positions simulées restent simulées** : en `MOCK_MODE`, le moteur ne passe aucun ordre réel et son historique
  n'est pas un track record.

Observations du premier run contre le vrai moteur (2026-10-13, une minute) : les positions du moteur dépassaient le
plafond de 20 % (BTC à ~67 % de l'equity), donc la taille publiée vaut le plafond et la quantification n'a rien à
masquer ; et les positions ne durent que quelques secondes à une minute, ce qui ferait des allers-retours coûteux chez un
worker connecté. C'est ce que couvre `--min-hold-s` (60 s par défaut) : une position fermée par le moteur après 6 s reste publiée 60 s. Pour comparer, relancer avec `--min-hold-s 0`.

## 5. Docker, ensuite

Une fois le dry-run local satisfaisant : `docker-compose.platform.yml` (superposition à la stack du moteur, pas une
stack autonome : le pont doit lire **le** Redis du moteur). Le même contrôle en conteneur :

```bash
docker compose -f docker-compose.yml -f docker-compose.platform.yml run --rm --no-deps \
  nothierly-platform-bridge python scripts/run_platform_bridge.py --dry-run
```

Cette commande n'a jamais été exécutée (Docker n'est pas installé sur la machine de développement) ; elle exige que
`platform.env` et `secrets/curve/curve_server.env` existent, car compose lit les `env_file` même pour un `run`.

Pour la stack complète (pont chiffré + workers en dry-run sur testnet) :

```bash
python scripts/generate_curve_keys.py --clients hyperliquid bybit --out-dir secrets/curve
cp platform.env.example platform.env   # puis remplir NOTHIERLY_ZMQ_SECRET et les clés d'API testnet
docker compose -f docker-compose.yml -f docker-compose.platform.yml up -d --build \
  nothierly-platform-bridge nothierly-worker-bybit nothierly-worker-hyperliquid
```

Les workers démarrent en **dry-run sur testnet**. Passer en réel est une modification relue de leur `command`
(`--live`, et pour mainnet `--network mainnet --i-understand-real-funds`), pas une variable d'environnement.
