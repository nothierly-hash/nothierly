"""Local validation script for the Nothierly SDK -- exercises it against a real local backend.

Start the real backend first (from the repo root, in another terminal):

    uvicorn p2p_api:app --port 8000

Then, from this directory (``nothierly-python/``), with the SDK installed
(``pip install -e .``):

    python test_sdk.py
"""

from __future__ import annotations

from nothierly import NothierlyClient, NothierlyError

_TEST_FIAT = "ARS"


def main() -> None:
    with NothierlyClient(base_url="http://localhost:8000") as client:
        print(f"--- get_p2p_premium('{_TEST_FIAT}') ---")
        try:
            df = client.get_p2p_premium(_TEST_FIAT)
            print(df)
            if df.empty:
                print(f"(DataFrame vide, colonnes réelles conservées -- pas de liquidité P2P "
                      f"réelle pour {_TEST_FIAT} en ce moment, colonnes: {list(df.columns)})")
        except NothierlyError as exc:
            print(f"NothierlyError: {exc} (status_code={exc.status_code})")

        print(f"\n--- get_macro_sentiment('{_TEST_FIAT}') ---")
        print("(endpoint non encore déployé sur ce backend -- démontre la gestion d'erreur réelle du SDK)")
        try:
            sentiment_df = client.get_macro_sentiment(_TEST_FIAT)
            print(sentiment_df)
        except NothierlyError as exc:
            print(f"NothierlyError (attendu): {exc} (status_code={exc.status_code})")


if __name__ == "__main__":
    main()
