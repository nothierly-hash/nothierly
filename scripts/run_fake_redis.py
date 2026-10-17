"""Standalone throwaway Redis-protocol server for local testing, no Docker required.

Run this in its own terminal, then point ``REDIS_HOST``/``REDIS_PORT`` (or
just accept the defaults, which already match ``.env.example``) at it before
running anything that needs a real Redis connection --
e.g. ``tests/test_execution_agent_multi_asset.py``.

All data lives in this process's memory only: stopping it (Ctrl+C) discards
everything. Never point production configuration at this -- it is a test
double, not a persistent or replicated store.

Usage:
    python scripts/run_fake_redis.py [--port 6379] [--host 127.0.0.1]
"""
from __future__ import annotations

import argparse

from fakeredis import TcpFakeServer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1).")
    parser.add_argument("--port", type=int, default=6379, help="Bind port (default: 6379, matching .env.example).")
    args = parser.parse_args()

    server = TcpFakeServer((args.host, args.port), server_type="redis")
    print(f"fakeredis TCP server listening on {args.host}:{args.port} (Ctrl+C to stop) ...")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
