"""Tests du diagnostic global (``scripts/system_health_check.py``).

Preuves recherchées : chaque sous-système sait dire PASS, WARN ou FAIL pour de bonnes raisons (clés CURVE incohérentes, compte en
hedge, droit de retrait, état factice qui ne passe pas la quantification...), aucune valeur secrète n'apparaît dans le rapport, et le
code de sortie est non nul dès qu'il y a un échec. Aucun appel réseau vers une plateforme : clients simulés.
"""

from __future__ import annotations

import importlib.util
import socket
import sqlite3
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from contracts.curve_transport import generate_keypair
from risk.meta_labeling import FEATURES_V3

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "system_health_check.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("system_health_check_under_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["system_health_check_under_test"] = module
    spec.loader.exec_module(module)
    return module


hc = _load()
S = hc.Status
SECRET = "test-secret-0123456789abcdef-0123456789abcdef"


def _status(checks: Any, name_part: str) -> Any:
    items = [checks] if isinstance(checks, hc.Check) else list(checks)
    (match,) = [c for c in items if name_part in c.name]
    return match


# --- Environnement ---------------------------------------------------------------------------


def test_env_files_are_parsed_with_quotes_comments_and_export(tmp_path: Path) -> None:
    path = tmp_path / "x.env"
    path.write_text("# commentaire\nA=1\nexport B='deux mots'\nC=\"trois\"\n\nD=\nmal formé\nE=a=b\n", encoding="utf-8")
    assert hc.parse_env_file(path) == {"A": "1", "B": "deux mots", "C": "trois", "D": "", "E": "a=b"}
    assert hc.parse_env_file(tmp_path / "absent.env") == {}


def test_the_process_environment_wins_over_env_files_and_empty_values_do_not_mask(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("REDIS_HOST=fichier\nREDIS_PORT=1111\nVIDE=\n", encoding="utf-8")
    (tmp_path / "platform.env").write_text("REDIS_PORT=2222\nBYBIT_API_KEY=\n", encoding="utf-8")
    env = hc.build_env(tmp_path, environ={"REDIS_HOST": "processus"})
    assert env["REDIS_HOST"] == "processus" and env["REDIS_PORT"] == "2222" and "VIDE" not in env and "BYBIT_API_KEY" not in env


def test_secrets_are_masked_in_any_message() -> None:
    redact = hc.redactor({"BINANCE_API_SECRET": "super-secret-value-123", "WEBHOOK_URL": "https://discord.com/api/webhooks/1/TOKENTOKEN", "REDIS_HOST": "localhost"})
    text = redact("échec avec super-secret-value-123 sur https://discord.com/api/webhooks/1/TOKENTOKEN depuis localhost")
    assert "super-secret-value-123" not in text and "TOKENTOKEN" not in text and "localhost" in text  # un nom d'hôte n'est pas un secret


# --- SQLite ----------------------------------------------------------------------------------


def test_sqlite_check_proves_the_write_and_leaves_no_trace(tmp_path: Path) -> None:
    path = tmp_path / "fills.sqlite3"
    (check,) = hc.check_sqlite([path])
    assert check.status is S.PASS and "base créée" in check.detail
    with sqlite3.connect(path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM executions").fetchone()[0] == 0  # la ligne d'essai a été annulée
    assert "base créée" not in hc.check_sqlite([path])[0].detail


def test_sqlite_check_fails_on_a_file_that_is_not_a_database_and_on_missing_tables(tmp_path: Path) -> None:
    junk = tmp_path / "junk.sqlite3"
    junk.write_bytes(b"ceci n'est pas une base sqlite" * 20)
    assert hc.check_sqlite([junk])[0].status is S.FAIL
    old = tmp_path / "old.sqlite3"
    with sqlite3.connect(old) as conn:
        conn.execute("CREATE TABLE executions (id INTEGER)")  # schéma incomplet : FillsDatabase le complète, mais la colonne manque
    assert hc.check_sqlite([old])[0].status is S.FAIL
    assert hc.check_sqlite([])[0].status is S.WARN


# --- Port du pont ----------------------------------------------------------------------------


def test_bridge_port_check_detects_an_occupied_port_and_explains_remote_binds() -> None:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    free = hc.check_bridge_port(f"tcp://0.0.0.0:{port}", running_ok=False)
    assert free.status is S.PASS and "CURVE obligatoire" in free.detail and "pare-feu" in free.detail
    local = hc.check_bridge_port(f"tcp://127.0.0.1:{port}", running_ok=False)
    assert local.status is S.PASS and "CURVE" not in local.detail
    with socket.socket() as holder:
        holder.bind(("127.0.0.1", 0))
        holder.listen()
        busy_port = holder.getsockname()[1]
        assert hc.check_bridge_port(f"tcp://0.0.0.0:{busy_port}", running_ok=False).status is S.FAIL
        assert hc.check_bridge_port(f"tcp://0.0.0.0:{busy_port}", running_ok=True).status is S.WARN
    assert hc.check_bridge_port("pas-un-endpoint", running_ok=False).status is S.FAIL


# --- Sécurité --------------------------------------------------------------------------------


def _write_curve(directory: Path, venues: list[str], *, authorize: list[str] | None = None, pin_wrong_server: str | None = None) -> None:
    server, clients = generate_keypair(), {v: generate_keypair() for v in venues}
    allowed = venues if authorize is None else authorize
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "curve_server.env").write_text(
        f"NOTHIERLY_CURVE_SERVER_SECRET={server.secret}\nNOTHIERLY_CURVE_CLIENT_KEYS={','.join(clients[v].public for v in allowed)}\n", encoding="utf-8")
    for venue, pair in clients.items():
        server_public = generate_keypair().public if venue == pin_wrong_server else server.public
        (directory / f"curve_{venue}.env").write_text(
            f"NOTHIERLY_CURVE_SERVER_PUBLIC={server_public}\nNOTHIERLY_CURVE_CLIENT_PUBLIC={pair.public}\nNOTHIERLY_CURVE_CLIENT_SECRET={pair.secret}\n", encoding="utf-8")


def test_a_consistent_curve_setup_passes_and_never_prints_a_key(tmp_path: Path) -> None:
    _write_curve(tmp_path, ["bybit", "binance"])
    checks = hc.check_curve({"NOTHIERLY_ZMQ_ENCRYPTION": "true"}, tmp_path, ["bybit", "binance"])
    assert all(c.status is S.PASS for c in checks) and len(checks) == 4  # interrupteur, serveur, deux clients
    assert all(len(token) < 30 for c in checks for token in c.detail.split())  # aucune clé de 40 caractères dans le rapport


def test_a_client_that_pins_another_server_or_is_not_authorized_fails(tmp_path: Path) -> None:
    _write_curve(tmp_path / "pin", ["bybit", "binance"], pin_wrong_server="binance")
    checks = hc.check_curve({"NOTHIERLY_ZMQ_ENCRYPTION": "true"}, tmp_path / "pin", ["bybit", "binance"])
    assert _status(checks, "client bybit").status is S.PASS and _status(checks, "client binance").status is S.FAIL
    assert "pas celle du pont" in _status(checks, "client binance").detail
    _write_curve(tmp_path / "allow", ["bybit", "binance"], authorize=["bybit"])
    forgotten = hc.check_curve({"NOTHIERLY_ZMQ_ENCRYPTION": "true"}, tmp_path / "allow", ["bybit", "binance"])
    assert _status(forgotten, "client binance").status is S.FAIL and "liste blanche" in _status(forgotten, "client binance").detail


def test_missing_curve_keys_fail_with_the_command_to_generate_them_and_the_switch_is_warned_about(tmp_path: Path) -> None:
    checks = hc.check_curve({}, tmp_path / "absent", ["bybit"])
    assert _status(checks, "interrupteur").status is S.WARN and _status(checks, "serveur").status is S.FAIL
    assert "generate_curve_keys.py" in _status(checks, "serveur").detail
    off = hc.check_curve({"NOTHIERLY_ZMQ_ENCRYPTION": "false"}, tmp_path / "absent", ["bybit"])
    assert "en clair" in _status(off, "interrupteur").detail


def test_the_bus_secret_must_be_present_and_long_enough() -> None:
    assert hc.check_bus_secret({}).status is S.FAIL and hc.check_bus_secret({"NOTHIERLY_ZMQ_SECRET": "court"}).status is S.FAIL
    ok = hc.check_bus_secret({"NOTHIERLY_ZMQ_SECRET": SECRET})
    assert ok.status is S.PASS and SECRET not in ok.detail


def test_the_webhook_check_never_shows_the_token() -> None:
    assert hc.check_webhook({}, send_test=False).status is S.FAIL
    assert hc.check_webhook({"WEBHOOK_URL": "http://evil.example/TOKENTOKEN"}, send_test=False).status is S.FAIL
    ok = hc.check_webhook({"WEBHOOK_URL": "https://discord.com/api/webhooks/1/TOKENTOKEN"}, send_test=False)
    assert ok.status is S.PASS and "TOKENTOKEN" not in ok.detail and "discord.com" in ok.detail


# --- Cerveau ---------------------------------------------------------------------------------


def test_the_engine_mode_blocks_live_while_the_engine_is_simulated() -> None:
    assert hc.check_engine_mode({}, live=True).status is S.FAIL  # MOCK_MODE vaut vrai par défaut
    assert hc.check_engine_mode({"MOCK_MODE": "true"}, live=False).status is S.WARN
    assert hc.check_engine_mode({"MOCK_MODE": "false"}, live=True).status is S.PASS


def test_without_an_artifact_the_meta_label_check_reports_the_neutral_skeleton_and_still_exercises_kelly_and_the_grid(tmp_path: Path) -> None:
    checks = hc.check_meta_label(tmp_path / "metalabel_weights.json")
    assert _status(checks, "chargement").status is S.WARN and "squelette neutre" in _status(checks, "chargement").detail
    kelly = _status(checks, "Kelly")
    assert kelly.status is S.PASS and kelly.detail.count("1.000") == 3  # le squelette donne exactement 1,0 quel que soit le VPIN
    grid = _status(checks, "Quantification")
    assert grid.status is S.PASS and "plafond 20 %" in grid.detail


def test_a_trained_artifact_is_loaded_strictly_and_a_corrupt_one_is_a_failure_not_a_silent_fallback(tmp_path: Path) -> None:
    import numpy as np

    from tests.test_metalabel_pipeline import tm

    rng = np.random.default_rng(1)
    x = rng.uniform(0, 1, size=(400, len(FEATURES_V3)))
    y = (x[:, FEATURES_V3.index("vpin")] < 0.5).astype(np.int64)
    good = tmp_path / "metalabel_weights.json"
    tm.save_artifact(tm.fit_model(x, y), good, {"version": 3, "feature_names": list(FEATURES_V3), "payoff_ratio": 1.5})
    checks = hc.check_meta_label(good)
    assert _status(checks, "chargement").status is S.PASS and "vpin" in _status(checks, "chargement").detail
    kelly = _status(checks, "Kelly")
    assert kelly.status is S.PASS and "1.000, 1.000, 1.000" not in kelly.detail  # un vrai modèle réagit au VPIN factice
    assert _status(checks, "Quantification").status is S.PASS
    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text("pas un modèle", encoding="utf-8")
    bad = hc.check_meta_label(corrupt)
    assert len(bad) == 1 and bad[0].status is S.FAIL  # le moteur retomberait sur 0,5 en silence ; le diagnostic, lui, l'exige corrigé


# --- Plateformes (clients simulés) -----------------------------------------------------------


class _Snapshot:
    equity = 1000
    positions: dict[str, Any] = {}


class FakeBybit:
    def __init__(self, *, withdraw: bool = False, hedge: bool = False, read_only: bool = False, **_: Any) -> None:
        from workers.bybit.client import KeyInfo

        perms = {"ContractTrade": ["Order", "Position"], "Wallet": ["Withdraw"] if withdraw else []}
        self._info, self._hedge = KeyInfo(read_only=read_only, permissions=perms), hedge

    def key_info(self) -> Any:
        return self._info

    def snapshot(self) -> Any:
        if self._hedge:
            from workers.bybit.client import HedgeModeError

            raise HedgeModeError("positionIdx=1")
        return _Snapshot()

    def instruments(self) -> dict[str, Any]:
        return {"BTCUSDT": object()}


def _bybit_factory(**options: Any) -> Any:
    return lambda **kw: FakeBybit(**options, **kw)


_KEYS = {"BYBIT_API_KEY": "k" * 10, "BYBIT_API_SECRET": "s" * 10, "BINANCE_API_KEY": "k" * 10, "BINANCE_API_SECRET": "s" * 10, "HL_ACCOUNT_ADDRESS": "0x" + "1" * 40}


def test_bybit_is_rejected_for_withdrawal_rights_and_for_hedge_mode_and_accepted_when_clean() -> None:
    assert [c.status for c in hc.check_bybit(_KEYS, "testnet", live=False, client_factory=_bybit_factory())] == [S.PASS, S.PASS]
    withdraw = hc.check_bybit(_KEYS, "testnet", live=False, client_factory=_bybit_factory(withdraw=True))
    assert withdraw[0].status is S.FAIL and "retrait" in withdraw[0].detail
    hedge = hc.check_bybit(_KEYS, "testnet", live=False, client_factory=_bybit_factory(hedge=True))
    assert hedge[-1].status is S.FAIL and "hedge" in hedge[-1].detail
    assert hc.check_bybit(_KEYS, "testnet", live=True, client_factory=_bybit_factory(read_only=True))[0].status is S.FAIL
    missing = hc.check_bybit({}, "testnet", live=False)
    assert len(missing) == 1 and missing[0].status is S.FAIL and "BYBIT_API_KEY" in missing[0].detail


class FakeBinance:
    def __init__(self, *, withdraw: bool = False, hedge: bool = False, verified: bool = True, **_: Any) -> None:
        from workers.binance.client import KeyPermissions

        self._perms = KeyPermissions(verified, can_withdraw=withdraw if verified else None, can_read=True, can_trade_futures=True, ip_restricted=True, reason="" if verified else "BinanceApiError")
        self._hedge = hedge

    def key_permissions(self) -> Any:
        return self._perms

    def is_hedge_mode(self) -> bool:
        return self._hedge

    def snapshot(self) -> Any:
        return _Snapshot()

    def instruments(self) -> dict[str, Any]:
        return {"BTCUSDT": object()}


def _binance_factory(**options: Any) -> Any:
    return lambda **kw: FakeBinance(**options, **kw)


def test_binance_is_rejected_for_withdrawal_rights_hedge_mode_and_unreadable_mainnet_rights() -> None:
    assert [c.status for c in hc.check_binance(_KEYS, "mainnet", live=True, client_factory=_binance_factory())] == [S.PASS, S.PASS]
    assert hc.check_binance(_KEYS, "mainnet", live=False, client_factory=_binance_factory(withdraw=True))[0].status is S.FAIL
    assert hc.check_binance(_KEYS, "mainnet", live=False, client_factory=_binance_factory(hedge=True))[1].status is S.FAIL
    assert hc.check_binance(_KEYS, "mainnet", live=False, client_factory=_binance_factory(verified=False))[0].status is S.FAIL  # on ne peut pas prouver l'absence de retrait
    testnet = hc.check_binance(_KEYS, "testnet", live=False, client_factory=_binance_factory(verified=False))
    assert testnet[0].status is S.WARN and testnet[1].status is S.PASS  # non vérifiable sur le réseau de test : averti, pas bloquant


class FakeHyperliquid:
    def __init__(self, **_: Any) -> None:
        pass

    def snapshot(self) -> Any:
        return _Snapshot()

    def asset_decimals(self) -> dict[str, int]:
        return {"BTC": 5}


def test_hyperliquid_rejects_the_main_wallet_key_as_an_agent_and_needs_the_account_address() -> None:
    from eth_account import Account

    main_key = "0x" + "11" * 32
    main_address = Account.from_key(main_key).address
    env = {"HL_ACCOUNT_ADDRESS": main_address, "HL_AGENT_SECRET": main_key}
    bad = hc.check_hyperliquid(env, "testnet", live=True, client_factory=FakeHyperliquid)
    assert bad[0].status is S.FAIL and "COMPTE PRINCIPAL" in bad[0].detail and main_key not in bad[0].detail
    agent_env = {**env, "HL_AGENT_SECRET": "0x" + "22" * 32}
    assert [c.status for c in hc.check_hyperliquid(agent_env, "testnet", live=True, client_factory=FakeHyperliquid)] == [S.PASS, S.PASS]
    no_agent = hc.check_hyperliquid({"HL_ACCOUNT_ADDRESS": main_address}, "testnet", live=True, client_factory=FakeHyperliquid)
    assert no_agent[0].status is S.FAIL and hc.check_hyperliquid({"HL_ACCOUNT_ADDRESS": main_address}, "testnet", live=False, client_factory=FakeHyperliquid)[0].status is S.WARN
    assert hc.check_hyperliquid({}, "testnet", live=False)[0].status is S.FAIL


# --- Flux public -----------------------------------------------------------------------------


def test_the_public_stream_check_reads_one_real_trade_from_a_local_feed_and_fails_on_a_silent_one() -> None:
    pytest.importorskip("websockets")
    import asyncio
    import json

    from websockets.asyncio.server import serve

    from workers.market_data.vpin_tracker import BinanceAggTradeFeed

    trade = {"stream": "btcusdt@aggTrade", "data": {"e": "aggTrade", "s": "BTCUSDT", "a": 1, "p": "50000", "q": "0.01", "T": 1, "m": False}}

    async def scenario(messages: list[str]) -> Any:
        async def handler(ws: Any) -> None:
            for message in messages:
                await ws.send(message)
            await asyncio.sleep(2)

        async with serve(handler, "127.0.0.1", 0) as server:
            port = list(server.sockets)[0].getsockname()[1]
            feed = BinanceAggTradeFeed(["BTCUSDT"], f"ws://127.0.0.1:{port}")
            return await asyncio.get_running_loop().run_in_executor(None, lambda: hc.check_public_streams("binance", ["BTCUSDT"], timeout_s=1.0, feed=feed))

    ok = asyncio.run(scenario([json.dumps({"result": None, "id": 1}), json.dumps(trade)]))  # l'accusé est ignoré, le trade est lu
    assert ok.status is S.PASS and "BTCUSDT" in ok.detail
    silent = asyncio.run(scenario([json.dumps({"result": None, "id": 1})]))
    assert silent.status is S.FAIL and "aucun trade" in silent.detail


# --- Rapport et code de sortie ---------------------------------------------------------------


def test_the_report_groups_checks_and_lists_every_failure_with_its_detail() -> None:
    checks = [hc.passed("Infrastructure", "Redis", "ok"), hc.warned("Cerveau", "Méta-modèle", "squelette"), hc.failed("Plateformes", "Bybit", "droit de retrait")]
    report = hc.render(checks)
    assert "[Infrastructure]" in report and "[Cerveau]" in report and "PASS 1   WARN 1   FAIL 1   SKIP 0" in report
    assert "DETAIL DES ECHECS" in report and "- [Plateformes] Bybit : droit de retrait" in report
    assert "DETAIL DES ECHECS" not in hc.render([hc.passed("A", "b")])


def _args(**overrides: Any) -> SimpleNamespace:
    base: dict[str, Any] = dict(network="testnet", live=False, venues=["bybit"], curve_dir="", endpoint="tcp://127.0.0.1:1", data_dir=".", fills_db=[], meta_label_model="x", symbols=["BTCUSDT"],
                                vpin_source="binance", no_network=True, running=False, send_test_alert=False, strict=False, env_file=[])
    base.update(overrides)
    return SimpleNamespace(**base)


def test_the_exit_code_is_nonzero_on_a_failure_and_strict_counts_warnings(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(hc, "parse_args", lambda argv=None: _args())
    monkeypatch.setattr(hc, "run_all", lambda args, env: [hc.passed("A", "x"), hc.warned("A", "y", "w")])
    assert hc.main([]) == 0
    monkeypatch.setattr(hc, "parse_args", lambda argv=None: _args(strict=True))
    assert hc.main([]) == 1
    monkeypatch.setattr(hc, "parse_args", lambda argv=None: _args())
    monkeypatch.setattr(hc, "run_all", lambda args, env: [hc.failed("A", "z", "casse")])
    assert hc.main([]) == 1 and "casse" in capsys.readouterr().out


def test_a_subsystem_that_crashes_is_a_failure_with_its_message_redacted_never_a_stopped_diagnostic(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def explode(*a: Any, **k: Any) -> Any:
        raise RuntimeError("connexion refusée avec super-secret-value-123")

    monkeypatch.setattr(hc, "check_imports", explode)
    monkeypatch.setattr(hc, "check_redis", lambda env, engine_expected: [hc.passed("Infrastructure", "Redis", "ok")])
    monkeypatch.setattr(hc, "check_brain_models", lambda: [])
    monkeypatch.setattr(hc, "check_vpin_key", lambda env, symbols: hc.passed("Données", "VPIN", ""))
    args = _args(data_dir=str(tmp_path), meta_label_model=str(tmp_path / "absent.json"), venues=[])
    checks = hc.run_all(args, {"BINANCE_API_SECRET": "super-secret-value-123"})
    crashed = _status(checks, "Imports")
    assert crashed.status is S.FAIL and "super-secret-value-123" not in crashed.detail and "***" in crashed.detail
    assert any(c.name == "Redis" for c in checks)  # le reste du diagnostic a continué
