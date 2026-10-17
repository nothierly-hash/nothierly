# nothierly

Official Python SDK for Nothierly's institutional market-data API (P2P/forex arbitrage premiums, macro-sentiment risk scoring).

## Install

```bash
pip install -e .
```

## Usage

```python
from nothierly import NothierlyClient

with NothierlyClient(api_key="...", base_url="https://api.nothierly.com") as client:
    df = client.get_p2p_premium("ARS")
    print(df)
```

`api_key` falls back to the `NOTHIERLY_API_KEY` environment variable when omitted.

## Status

- `get_p2p_premium` -- live against `GET /api/v1/p2p-premiums`.
- `get_macro_sentiment` -- SDK-side complete (validation, retries, DataFrame conversion), but targets `GET /api/v1/macro-sentiment/{fiat}`, which is not deployed on the backend yet. `political_agent.py` already computes this data into Redis; nothing exposes it over REST yet.
- `LiquidityDepthResponse` -- model only, no backend endpoint exists for it yet.
