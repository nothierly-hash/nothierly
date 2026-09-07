"""Functional test for src/data_pipeline/data_lake_sync.py against a real
mocked S3 API (moto), not a hand-rolled fake -- verifies the actual
download/cache/load pipeline works end-to-end without real R2 credentials
or any real network call.

Requires the `moto[server]` extra (flask etc.) -- not a production
dependency of this project, so not in requirements.txt; install ad hoc
to run this file: `pip install "moto[server]"`.

Run directly: `python tests/test_data_lake_sync.py`
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import boto3
import polars as pl
from moto.server import ThreadedMotoServer
from pydantic import SecretStr

from core.config import R2Settings
from src.data_pipeline.data_lake_sync import (
    DataLakeConfigurationError,
    DataLakeDownloadError,
    _build_s3_client,
    _dataset_key,
    check_bucket_health,
    download_market_data,
    load_dataset_for_training,
)

_checks: list[tuple[str, bool]] = []


def check(label: str, condition: bool) -> None:
    _checks.append((label, condition))
    print(f"{'PASS' if condition else 'FAIL'} - {label}")


# --- _dataset_key: symbol sanitization ---
check("dataset key sanitizes '/' in symbol", _dataset_key("BTC/USDT", 2022) == "market_data/BTC_USDT/2022.parquet")
check("dataset key with underscore symbol unchanged", _dataset_key("EUR_USD", 2020) == "market_data/EUR_USD/2020.parquet")

# --- Missing configuration raises the right error ---
incomplete = R2Settings(account_id=None, bucket_name=None, access_key_id=None, secret_access_key=None)
try:
    _build_s3_client(incomplete)
    check("missing R2 config raises DataLakeConfigurationError", False)
except DataLakeConfigurationError as exc:
    check("missing R2 config raises DataLakeConfigurationError", True)
    check(
        "error message names the missing variables",
        "R2_ACCOUNT_ID" in str(exc) and "AWS_ACCESS_KEY_ID" in str(exc),
    )

partial = R2Settings(account_id="abc123", bucket_name="my-bucket", access_key_id=None, secret_access_key=None)
try:
    _build_s3_client(partial)
    check("partial R2 config (missing keys) also raises", False)
except DataLakeConfigurationError:
    check("partial R2 config (missing keys) also raises", True)

# --- endpoint_url construction ---
configured = R2Settings(
    account_id="myaccount123",
    bucket_name="my-bucket",
    access_key_id="fake_key",
    secret_access_key=SecretStr("fake_secret"),
)
check(
    "endpoint_url built correctly from account_id",
    configured.endpoint_url == "https://myaccount123.r2.cloudflarestorage.com",
)


# --- Real end-to-end pipeline against a real mocked S3 server (moto.server) ---
# moto's default mock_aws() decorator only intercepts requests to
# *recognized AWS* endpoint hostnames -- a custom endpoint_url (required
# for any S3-compatible-but-not-AWS service, including R2) bypasses it
# entirely and attempts a real network connection (confirmed while
# building this test: a genuine SSL handshake failure against the real,
# unreachable R2-shaped hostname on the first attempt).
# ThreadedMotoServer is moto's own documented answer for exactly this: a
# real local mock S3 server with its own real endpoint_url.
moto_server = ThreadedMotoServer(ip_address="127.0.0.1", port=0)
moto_server.start()
try:
    host, port = moto_server.get_host_and_port()
    mock_endpoint = f"http://{host}:{port}"

    settings = R2Settings(
        account_id="testaccount",
        bucket_name="test-market-data",
        access_key_id="test_key",
        secret_access_key=SecretStr("test_secret"),
    )
    # Client built directly against the local mock server's endpoint --
    # _build_s3_client() itself (which builds the real R2-shaped
    # endpoint_url) is already verified separately above ("endpoint_url
    # built correctly from account_id"); this exercises everything
    # download_market_data()/load_dataset_for_training() actually do
    # with a client, via the same s3_client injection point a real
    # caller could use.
    client = boto3.client(
        "s3",
        endpoint_url=mock_endpoint,
        aws_access_key_id="test_key",
        aws_secret_access_key="test_secret",
        region_name="auto",
    )
    # Bucket creation is test scaffolding only -- data_lake_sync.py never
    # creates a bucket in production (R2 buckets are provisioned via the
    # Cloudflare dashboard/API, out of this module's scope). Real S3's
    # CreateBucket rejects the "auto" region_name production code
    # otherwise uses, so this one setup call uses a real AWS region name
    # instead, purely to satisfy moto's request validation.
    setup_client = boto3.client(
        "s3",
        endpoint_url=mock_endpoint,
        aws_access_key_id="test_key",
        aws_secret_access_key="test_secret",
        region_name="us-east-1",
    )
    setup_client.create_bucket(Bucket="test-market-data")

    # Upload a real, small Parquet file to the mocked bucket at the exact
    # key download_market_data() will look for.
    real_df = pl.DataFrame({"timestamp": [1, 2, 3, 4, 5], "close": [100.0, 101.5, 99.8, 102.3, 103.1]})
    with tempfile.TemporaryDirectory() as tmp:
        upload_path = Path(tmp) / "upload.parquet"
        real_df.write_parquet(upload_path)
        client.upload_file(str(upload_path), "test-market-data", "market_data/BTC_USDT/2022.parquet")

        cache_path = Path(tmp) / "cache" / "btc_2022.parquet"
        check("cache file does not exist yet", not cache_path.exists())

        result_path = download_market_data("BTC/USDT", 2022, cache_path, settings=settings, s3_client=client)
        check("download_market_data returns the destination path", result_path == cache_path)
        check("file was actually downloaded to the cache path", cache_path.exists())

        loaded = load_dataset_for_training(cache_path)
        check("loaded a real polars.DataFrame", isinstance(loaded, pl.DataFrame))
        check("loaded correct row count", loaded.height == 5)
        check("loaded correct column count", loaded.width == 2)
        check(
            "loaded data matches what was uploaded",
            loaded["close"].to_list() == [100.0, 101.5, 99.8, 102.3, 103.1],
        )

        # --- check_bucket_health: real ping + real object enumeration against the same mocked bucket ---
        health = check_bucket_health(settings=settings, s3_client=client)
        check("bucket health reports reachable", health.reachable)
        check("bucket health reports the one real uploaded object", health.object_count == 1)
        check("bucket health reports a real, positive latency", health.latency_ms is not None and health.latency_ms >= 0)
        real_size = upload_path.stat().st_size
        check("bucket health reports the real uploaded object's real size", health.total_size_bytes == real_size)
        check("bucket health is not truncated (well under the real cap)", not health.truncated)

        missing_bucket_health = check_bucket_health(
            settings=R2Settings(
                account_id="testaccount", bucket_name="no-such-bucket",
                access_key_id="test_key", secret_access_key=SecretStr("test_secret"),
            ),
            s3_client=client,
        )
        check("a genuinely nonexistent bucket reports unreachable, not a fabricated empty summary", not missing_bucket_health.reachable)
        check("unreachable bucket carries a real error message", missing_bucket_health.error is not None)

        # Delete the object from the mocked bucket, then call again --
        # if this succeeds, the cache genuinely prevented a second
        # network call (a real re-download would now fail: the object
        # is gone).
        client.delete_object(Bucket="test-market-data", Key="market_data/BTC_USDT/2022.parquet")
        result_path_2 = download_market_data("BTC/USDT", 2022, cache_path, settings=settings, s3_client=client)
        check(
            "second call hits the local cache (object no longer even exists in the bucket)",
            result_path_2 == cache_path,
        )

        health_after_delete = check_bucket_health(settings=settings, s3_client=client)
        check("bucket health reflects the real post-deletion state (0 real objects)", health_after_delete.object_count == 0)
        check("bucket health still reachable after the object was deleted", health_after_delete.reachable)

        # A genuinely missing object (never uploaded) should raise a clear error.
        missing_cache_path = Path(tmp) / "cache" / "eth_2022.parquet"
        try:
            download_market_data("ETH/USDT", 2022, missing_cache_path, settings=settings, s3_client=client)
            check("downloading a nonexistent object raises DataLakeDownloadError", False)
        except DataLakeDownloadError:
            check("downloading a nonexistent object raises DataLakeDownloadError", True)
finally:
    moto_server.stop()


print("\n" + "=" * 60)
failed = [label for label, ok in _checks if not ok]
if failed:
    print(f"{len(failed)}/{len(_checks)} checks FAILED:")
    for label in failed:
        print(f"  - {label}")
    sys.exit(1)
print(f"All {len(_checks)} checks passed.")
