"""Real RSS headline collector -- prerequisite dataset-building step for fine-tuning Nothierly's
own proprietary crisis-scoring LLM (see ``scripts/finetune_political_llm.py``'s own module
docstring for the full, honestly-sequenced plan this is step 1 of).

**Real, zero-cost step, deliberately separated from labeling**: this script only *collects* real,
filtered headlines -- the exact same real RSS fetch + local keyword/currency filter
``political_agent.py`` already runs every cycle (:func:`political_agent._fetch_one_feed`, reused
here directly, not reimplemented) -- into a plain JSONL file. It does **not** call any LLM, paid or
free: labeling (assigning a real ``crisis_score``/``sentiment`` to each collected headline) is a
separate, later step that costs real money (a commercial LLM API) or real human time (manual
labeling), and is a deliberate decision for whoever owns that budget, not something this script
does silently on their behalf.

**Why collect before a label pipeline is even decided**: RSS feeds only carry a rolling recent
window (most publishers keep 24-72h of items) -- real headlines seen today are gone from the feed
itself in a few days. Running this script on a schedule (e.g. once daily, alongside
``political_agent.py``'s own real cycle) accumulates a real, growing raw corpus now, independent of
if/when a labeling budget is approved -- so that decision, whenever it happens, isn't blocked
waiting for headlines that already scrolled off every feed.

Usage, real and repeatable (append-only, deduplicated by entry_id across runs)::

    python scripts/collect_headline_dataset.py --output data/raw_headlines.jsonl
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Same real, verified-not-assumed bug scripts/analyze_shadow_run.py already fixes: Windows' default
# console codepage (cp1252) cannot encode this script's own accented French output.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

import argparse
import asyncio
import json
from dataclasses import asdict

from core.http_client import ResilientHttpClient
from core.logger import configure_logging, get_logger
from political_agent import _FEED_REQUEST_HEADERS, _RSS_FEEDS, _fetch_one_feed

_log = get_logger(__name__)


def _load_existing_entry_ids(output_path: Path) -> set[str]:
    """Real dedup across runs -- an entry already collected in a prior run is never re-appended."""
    if not output_path.exists():
        return set()
    entry_ids: set[str] = set()
    for line in output_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            entry_ids.add(json.loads(line)["entry_id"])
        except (json.JSONDecodeError, KeyError):
            continue
    return entry_ids


async def _collect() -> list[dict[str, object]]:
    """Real fetch + filter across every configured feed -- identical logic to political_agent.py's own cycle."""
    async with ResilientHttpClient(max_concurrent_requests=5, timeout_seconds=10.0) as http:
        results = await asyncio.gather(
            *(_fetch_one_feed(http, source, url) for source, url in _RSS_FEEDS), return_exceptions=True
        )

    all_filtered: list[dict[str, object]] = []
    for result in results:
        if isinstance(result, BaseException):
            continue
        all_filtered.extend(asdict(item) for item in result)
    return all_filtered


async def main() -> None:
    configure_logging(service_name="collect-headline-dataset")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--output", type=Path, default=Path("data/raw_headlines.jsonl"),
        help="Append-only JSONL file -- real, filtered headlines, one per line (default: data/raw_headlines.jsonl).",
    )
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    existing_ids = _load_existing_entry_ids(args.output)
    collected = await _collect()
    new_items = [item for item in collected if item["entry_id"] not in existing_ids]

    with args.output.open("a", encoding="utf-8") as f:
        for item in new_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    _log.info(
        "headline_collection_complete",
        feeds_polled=len(_RSS_FEEDS), total_filtered_this_run=len(collected),
        new_items_appended=len(new_items), output_path=str(args.output),
    )
    print(f"{len(new_items)} nouveau(x) titre(s) réel(s) ajouté(s) à {args.output} (sur {len(collected)} filtré(s) ce cycle).")
    print("Prochaine étape (coût réel, décision séparée) : voir scripts/finetune_political_llm.py.")


if __name__ == "__main__":
    asyncio.run(main())
