"""Tests du chiffrement CURVE du bus (``contracts/curve_transport.py``) et de son intégration (publisher, workers, scripts).

Preuves recherchées, sur de vrais sockets TCP locaux :

- le canal est **opaque** : un relais TCP qui enregistre tous les octets voit le contenu en clair sans CURVE (témoin) et
  rien de lisible avec CURVE ;
- le canal est **authentifié dans les deux sens** : un client hors liste blanche, un client sans CURVE, et un faux
  publisher (autre clé serveur) ne passent pas, et il n'y a aucun repli en clair.
"""

from __future__ import annotations

import asyncio
import importlib.util
import socket
import sys
import threading
import time
from pathlib import Path
from types import ModuleType
from typing import Any, ClassVar

import pytest
import zmq

from contracts.curve_transport import (
    ENV_CLIENT_KEYS,
    ENV_ENCRYPTION,
    ENV_CLIENT_PUBLIC,
    ENV_CLIENT_SECRET,
    ENV_SERVER_PUBLIC,
    ENV_SERVER_SECRET,
    CurveClientConfig,
    CurveConfigError,
    CurveServerConfig,
    apply_client,
    generate_keypair,
    is_local_endpoint,
    public_from_secret,
    transport_refusal,
)
from contracts.platform_signal import AssetClass, OrderType, PlatformSignal, SignalDirection, TargetVenue
from contracts.secure_signal_bus import DEFAULT_TOPIC, SECRET_ENV_VAR, SecureSignalPublisher, open_envelope
from workers.common.base import ExposureWorkerBase, ExposureWorkerConfig
from workers.numerai.client import DryRunClient
from workers.numerai.worker import NumeraiWorker, NumeraiWorkerConfig

SECRET = "test-secret-0123456789abcdef-0123456789abcdef"
MARKER = b"CRYPTO_BTC_USD"


def _signal(**overrides: Any) -> PlatformSignal:
    base: dict[str, Any] = dict(
        target_venue=TargetVenue.ALL, asset_class=AssetClass.CRYPTO, symbol="CRYPTO_BTC_USD",
        direction=SignalDirection.LONG, position_size_pct=5.0, order_type=OrderType.MARKET,
    )
    base.update(overrides)
    return PlatformSignal(**base)


class Keys:
    """Un serveur, un client autorisé, un intrus (non autorisé) et un faux serveur."""

    def __init__(self) -> None:
        self.server, self.client, self.intruder, self.rogue = (generate_keypair() for _ in range(4))
        self.server_cfg = CurveServerConfig(self.server.secret, frozenset({self.client.public}))
        self.client_cfg = CurveClientConfig(self.server.public, self.client.public, self.client.secret)
        self.intruder_cfg = CurveClientConfig(self.server.public, self.intruder.public, self.intruder.secret)
        self.rogue_server_cfg = CurveServerConfig(self.rogue.secret, frozenset({self.client.public}))
        self.pinned_to_real_server = self.client_cfg  # épingle la clé publique du VRAI serveur


@pytest.fixture
def keys() -> Keys:
    return Keys()


@pytest.fixture
def ctx() -> Any:
    context = zmq.Context()
    try:
        yield context
    finally:
        context.destroy(linger=0)


def _subscribe(ctx: zmq.Context[Any], endpoint: str, curve: CurveClientConfig | None) -> zmq.Socket[Any]:
    sub = ctx.socket(zmq.SUB)
    sub.setsockopt(zmq.LINGER, 0)
    sub.setsockopt(zmq.SUBSCRIBE, DEFAULT_TOPIC)
    if curve is not None:
        apply_client(sub, curve)
    sub.connect(endpoint)
    return sub


def _receive(pub: SecureSignalPublisher, sub: zmq.Socket[Any], timeout: float) -> list[bytes] | None:
    """Republie jusqu'à réception (slow joiner, poignée de main CURVE) ; ``None`` si rien n'arrive dans ``timeout``."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        pub.publish(_signal())
        if sub.poll(100):
            return sub.recv_multipart()
    return None


class TcpRelay:
    """Relais TCP transparent qui enregistre tous les octets échangés (un espion réseau)."""

    def __init__(self, upstream_port: int) -> None:
        self._upstream_port = upstream_port
        self._listener = socket.socket()
        self._listener.bind(("127.0.0.1", 0))
        self._listener.listen()
        self._listener.settimeout(0.2)
        self.port = int(self._listener.getsockname()[1])
        self.captured = bytearray()
        self._lock = threading.Lock()
        self._stop = False
        self._sockets: list[socket.socket] = []
        self._thread = threading.Thread(target=self._accept_loop, daemon=True)
        self._thread.start()

    def _accept_loop(self) -> None:
        while not self._stop:
            try:
                client, _ = self._listener.accept()
                upstream = socket.create_connection(("127.0.0.1", self._upstream_port))
            except OSError:
                continue
            self._sockets += [client, upstream]
            for src, dst in ((client, upstream), (upstream, client)):
                threading.Thread(target=self._pump, args=(src, dst), daemon=True).start()

    def _pump(self, src: socket.socket, dst: socket.socket) -> None:
        try:
            while data := src.recv(65536):
                with self._lock:
                    self.captured += data
                dst.sendall(data)
        except OSError:
            pass

    def close(self) -> None:
        self._stop = True
        for s in [self._listener, *self._sockets]:
            try:
                s.close()
            except OSError:
                pass
        self._thread.join(timeout=2)


# --- Clés et configuration -------------------------------------------------------------------


def test_generated_keys_are_valid_z85_and_the_public_key_derives_from_the_secret() -> None:
    pair = generate_keypair()
    assert len(pair.public) == len(pair.secret) == 40 and public_from_secret(pair.secret) == pair.public
    assert generate_keypair().public != pair.public
    assert pair.secret not in repr(pair)  # jamais dans un log


@pytest.mark.parametrize("bad", ["pas une cle " + "x" * 28, "x" * 39, "x" * 41, "x" * 39 + '"', "x" * 39 + "é", ""])
def test_malformed_keys_are_refused_without_echoing_them(bad: str) -> None:
    """Z85 n'a pas de somme de contrôle : seuls la longueur et l'alphabet sont vérifiables (une faute de frappe dans
    la clé publique d'un serveur épinglé empêche simplement le canal de s'établir, jamais un repli en clair)."""
    with pytest.raises(CurveConfigError) as exc:
        CurveServerConfig(bad, frozenset({generate_keypair().public}))
    assert bad == "" or bad not in str(exc.value)
    with pytest.raises(CurveConfigError):
        CurveServerConfig(generate_keypair().secret, frozenset({bad}))


def test_a_server_without_an_allow_list_is_refused() -> None:
    with pytest.raises(CurveConfigError, match="liste blanche"):
        CurveServerConfig(generate_keypair().secret, frozenset())


def test_a_client_whose_secret_does_not_match_its_public_key_is_refused() -> None:
    a, b, server = generate_keypair(), generate_keypair(), generate_keypair()
    with pytest.raises(CurveConfigError, match="ne correspond pas"):
        CurveClientConfig(server.public, a.public, b.secret)


def test_from_env_returns_none_when_unset_and_rejects_partial_configuration(keys: Keys) -> None:
    assert CurveServerConfig.from_env({}) is None and CurveClientConfig.from_env({}) is None
    with pytest.raises(CurveConfigError, match="partielle"):
        CurveServerConfig.from_env({ENV_SERVER_SECRET: keys.server.secret})
    with pytest.raises(CurveConfigError, match="partielle"):
        CurveClientConfig.from_env({ENV_SERVER_PUBLIC: keys.server.public})
    server = CurveServerConfig.from_env({ENV_SERVER_SECRET: keys.server.secret, ENV_CLIENT_KEYS: f" {keys.client.public} , {keys.intruder.public} "})
    assert server is not None and server.authorized_clients == {keys.client.public, keys.intruder.public}
    client = CurveClientConfig.from_env({ENV_SERVER_PUBLIC: keys.server.public, ENV_CLIENT_PUBLIC: keys.client.public, ENV_CLIENT_SECRET: keys.client.secret})
    assert client == keys.client_cfg


def test_the_encryption_switch_forces_disables_or_leaves_curve_to_the_keys(keys: Keys) -> None:
    server_env = {ENV_SERVER_SECRET: keys.server.secret, ENV_CLIENT_KEYS: keys.client.public}
    client_env = {ENV_SERVER_PUBLIC: keys.server.public, ENV_CLIENT_PUBLIC: keys.client.public, ENV_CLIENT_SECRET: keys.client.secret}
    for value in ("false", "0", "OFF"):  # désactivé : les clés présentes sont ignorées
        assert CurveServerConfig.from_env({**server_env, ENV_ENCRYPTION: value}) is None
        assert CurveClientConfig.from_env({**client_env, ENV_ENCRYPTION: value}) is None
    for value in ("", "auto", "AUTO"):  # auto : selon les clés
        assert CurveServerConfig.from_env({**server_env, ENV_ENCRYPTION: value}) is not None
        assert CurveClientConfig.from_env({ENV_ENCRYPTION: value}) is None
    for value in ("true", "1", "required"):  # obligatoire : des clés absentes sont une erreur, pas un repli en clair
        assert CurveClientConfig.from_env({**client_env, ENV_ENCRYPTION: value}) == keys.client_cfg
        with pytest.raises(CurveConfigError, match="ne sont pas définis"):
            CurveServerConfig.from_env({ENV_ENCRYPTION: value})
        with pytest.raises(CurveConfigError, match="ne sont pas définis"):
            CurveClientConfig.from_env({ENV_ENCRYPTION: value})
    with pytest.raises(CurveConfigError, match="true, false ou auto"):
        CurveServerConfig.from_env({**server_env, ENV_ENCRYPTION: "peut-etre"})  # une faute de frappe ne coupe pas le chiffrement
    # `false` n'ouvre pas le réseau : un endpoint non local reste refusé sans --allow-plaintext-remote
    assert transport_refusal("tcp://bridge:25570", curve_enabled=False, allow_plaintext_remote=False) is not None


def test_endpoint_policy() -> None:
    assert is_local_endpoint("tcp://127.0.0.1:25570") and is_local_endpoint("inproc://x") and not is_local_endpoint("tcp://10.0.0.5:25570")
    assert transport_refusal("tcp://127.0.0.1:1", curve_enabled=False, allow_plaintext_remote=False) is None
    assert "Refusé" in (transport_refusal("tcp://10.0.0.5:1", curve_enabled=False, allow_plaintext_remote=False) or "")
    assert transport_refusal("tcp://10.0.0.5:1", curve_enabled=True, allow_plaintext_remote=False) is None
    assert transport_refusal("tcp://10.0.0.5:1", curve_enabled=False, allow_plaintext_remote=True) is None


# --- Canal : chiffrement et authentification --------------------------------------------------


def test_an_authorized_client_receives_signals_over_the_encrypted_channel(keys: Keys, ctx: zmq.Context[Any]) -> None:
    with SecureSignalPublisher("tcp://127.0.0.1:*", SECRET, curve=keys.server_cfg) as pub:
        assert pub.encrypted
        sub = _subscribe(ctx, pub.endpoint, keys.client_cfg)
        frames = _receive(pub, sub, timeout=8.0)
    assert frames is not None and frames[0] == DEFAULT_TOPIC
    assert open_envelope(frames[1], SECRET, venue=TargetVenue.HYPERLIQUID).accepted  # le HMAC reste vérifié par-dessus


def test_the_wire_is_opaque_with_curve_and_readable_without(keys: Keys, ctx: zmq.Context[Any]) -> None:
    seen = {}
    for encrypted in (False, True):  # False = témoin : prouve que le relais voit bien le trafic
        pub = SecureSignalPublisher("tcp://127.0.0.1:*", SECRET, curve=keys.server_cfg if encrypted else None)
        relay = TcpRelay(int(pub.endpoint.rsplit(":", 1)[1]))
        try:
            sub = _subscribe(ctx, f"tcp://127.0.0.1:{relay.port}", keys.client_cfg if encrypted else None)
            assert _receive(pub, sub, timeout=8.0) is not None
            time.sleep(0.2)
            with relay._lock:
                seen[encrypted] = bytes(relay.captured)
        finally:
            relay.close()
            pub.close()
    assert MARKER in seen[False] and b"position_size_pct" in seen[False] and b"signature" in seen[False]  # témoin : en clair
    assert len(seen[True]) > 200  # du trafic a bien circulé...
    for fragment in (MARKER, b"position_size_pct", b"signature", b"payload", b"HYPERLIQUID", b"EXECUTION.SIGNAL"):
        assert fragment not in seen[True], fragment  # ...et rien n'y est lisible


def test_a_client_outside_the_allow_list_gets_nothing_and_the_refusal_is_logged(keys: Keys, ctx: zmq.Context[Any], caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level("WARNING", logger="contracts.curve_transport"):
        with SecureSignalPublisher("tcp://127.0.0.1:*", SECRET, curve=keys.server_cfg) as pub:
            intruder = _subscribe(ctx, pub.endpoint, keys.intruder_cfg)
            assert _receive(pub, intruder, timeout=2.0) is None
    assert "curve_client_refused" in caplog.text and keys.intruder.secret not in caplog.text


def test_a_client_without_curve_cannot_read_an_encrypted_publisher(keys: Keys, ctx: zmq.Context[Any]) -> None:
    with SecureSignalPublisher("tcp://127.0.0.1:*", SECRET, curve=keys.server_cfg) as pub:
        plain = _subscribe(ctx, pub.endpoint, None)
        assert _receive(pub, plain, timeout=2.0) is None  # pas de repli en clair


def test_a_curve_client_does_not_fall_back_to_a_plaintext_publisher(keys: Keys, ctx: zmq.Context[Any]) -> None:
    with SecureSignalPublisher("tcp://127.0.0.1:*", SECRET) as plain_pub:
        sub = _subscribe(ctx, plain_pub.endpoint, keys.client_cfg)
        assert _receive(plain_pub, sub, timeout=2.0) is None


def test_a_rogue_publisher_cannot_impersonate_the_real_one(keys: Keys, ctx: zmq.Context[Any]) -> None:
    """Le worker épingle la clé publique du vrai serveur : un faux publisher (autre clé), même chiffré, n'est pas écouté."""
    with SecureSignalPublisher("tcp://127.0.0.1:*", SECRET, curve=keys.rogue_server_cfg) as rogue:
        sub = _subscribe(ctx, rogue.endpoint, keys.pinned_to_real_server)
        assert _receive(rogue, sub, timeout=2.0) is None


def test_publisher_curve_configuration_must_match_its_role(keys: Keys, ctx: zmq.Context[Any]) -> None:
    with pytest.raises(ValueError, match="client CURVE"):
        SecureSignalPublisher("tcp://127.0.0.1:1", SECRET, bind=False, curve=keys.server_cfg)
    with pytest.raises(ValueError, match="serveur CURVE"):
        SecureSignalPublisher("tcp://127.0.0.1:*", SECRET, bind=True, curve=keys.client_cfg)
    with pytest.raises(ValueError, match="propre contexte"):
        SecureSignalPublisher("tcp://127.0.0.1:*", SECRET, context=ctx, curve=keys.server_cfg)
    assert not SecureSignalPublisher("tcp://127.0.0.1:*", SECRET).encrypted


def test_closing_an_encrypted_publisher_releases_everything(keys: Keys) -> None:
    pub = SecureSignalPublisher("tcp://127.0.0.1:*", SECRET, curve=keys.server_cfg)
    started = time.monotonic()
    pub.close()
    assert time.monotonic() - started < 2.0  # régression : l'ancien authentificateur bloquait ici pour toujours sous Windows
    pub.close()  # idempotent
    again = SecureSignalPublisher("tcp://127.0.0.1:*", SECRET, curve=keys.server_cfg)  # le gestionnaire ZAP précédent est bien parti
    again.close()


# --- Workers ---------------------------------------------------------------------------------


class _MiniWorker(ExposureWorkerBase):
    venue: ClassVar[TargetVenue] = TargetVenue.HYPERLIQUID

    def map_symbol(self, symbol: str) -> str | None:
        return "BTC" if symbol == "CRYPTO_BTC_USD" else None

    async def execute_signal(self, signal: PlatformSignal, asset: str, *, now_ms: int | None = None) -> str:
        return "ok"


def _worker_hears(tmp_path: Path, keys: Keys, worker_curve: CurveClientConfig | None, name: str, window: float) -> bool:
    async def scenario() -> bool:
        with SecureSignalPublisher("tcp://127.0.0.1:*", SECRET, curve=keys.server_cfg) as pub:
            cfg = ExposureWorkerConfig(address=pub.endpoint, secret_key=SECRET, db_path=tmp_path / f"{name}.sqlite3", poll_ms=20, curve=worker_curve)
            worker = _MiniWorker(cfg)
            stop = asyncio.Event()
            task = asyncio.create_task(worker.run(stop))
            deadline = time.monotonic() + window
            heard = False
            while time.monotonic() < deadline and not heard:
                pub.publish(_signal())
                await asyncio.sleep(0.05)
                heard = "BTC" in worker._pending or bool(worker.state.recent_signal_ids(0))
            stop.set()
            await asyncio.wait_for(task, timeout=5.0)
            return heard

    return asyncio.run(scenario(), loop_factory=asyncio.SelectorEventLoop)


def test_the_shared_worker_base_uses_the_curve_config_and_an_unconfigured_worker_hears_nothing(tmp_path: Path, keys: Keys) -> None:
    assert _worker_hears(tmp_path, keys, keys.client_cfg, "ok", window=8.0)
    assert not _worker_hears(tmp_path, keys, None, "plain", window=2.0)
    assert not _worker_hears(tmp_path, keys, keys.intruder_cfg, "intruder", window=2.0)


def test_the_numerai_worker_also_uses_curve(tmp_path: Path, keys: Keys) -> None:
    async def scenario() -> int:
        with SecureSignalPublisher("tcp://127.0.0.1:*", SECRET, curve=keys.server_cfg) as pub:
            cfg = NumeraiWorkerConfig(
                address=pub.endpoint, secret_key=SECRET, db_path=tmp_path / "n.sqlite3", output_dir=tmp_path / "out",
                poll_ms=20, schedule_check_interval_s=3600.0, curve=keys.client_cfg,
            )
            worker = NumeraiWorker(cfg, client=DryRunClient())
            stop = asyncio.Event()
            task = asyncio.create_task(worker.run(stop))
            signal = PlatformSignal(
                target_venue=TargetVenue.NUMERAI_SIGNALS, asset_class=AssetClass.EQUITY, symbol="EQUITY_US_AAPL",
                direction=SignalDirection.LONG, confidence_score=0.9, position_size_pct=0.0, order_type=OrderType.PREDICTION_ONLY,
            )
            deadline = time.monotonic() + 8.0
            while worker.store.count_scores() == 0 and time.monotonic() < deadline:
                pub.publish(signal)
                await asyncio.sleep(0.05)
            stop.set()
            await asyncio.wait_for(task, timeout=5.0)
            return worker.store.count_scores()

    assert asyncio.run(scenario(), loop_factory=asyncio.SelectorEventLoop) == 1


def test_worker_command_lines_refuse_a_remote_address_without_curve_and_pass_the_config_with_it(monkeypatch: pytest.MonkeyPatch, keys: Keys) -> None:
    from workers.bybit import worker as bybit_module
    from workers.hyperliquid import worker as hl_module
    from workers.numerai import worker as numerai_module

    for var in (ENV_SERVER_PUBLIC, ENV_CLIENT_PUBLIC, ENV_CLIENT_SECRET, "HL_ACCOUNT_ADDRESS", "BYBIT_API_KEY", "BYBIT_API_SECRET"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv(SECRET_ENV_VAR, SECRET)
    monkeypatch.setenv("HL_ACCOUNT_ADDRESS", "0x" + "11" * 20)
    monkeypatch.setenv("BYBIT_API_KEY", "k")
    monkeypatch.setenv("BYBIT_API_SECRET", "s")
    remote = "tcp://10.0.0.5:25570"
    for module in (hl_module, bybit_module, numerai_module):
        with pytest.raises(SystemExit):
            module.main(["--address", remote])  # distant, en clair : refusé

    seen: dict[str, Any] = {}

    class _Stop(Exception):
        pass

    def capture(name: str) -> Any:
        def factory(config: Any, *args: Any, **kwargs: Any) -> None:
            seen[name] = config
            raise _Stop
        return factory

    monkeypatch.setattr(hl_module, "SdkHyperliquidClient", lambda **_: object())
    monkeypatch.setattr(hl_module, "HyperliquidWorker", capture("hl"))
    for var, value in ((ENV_SERVER_PUBLIC, keys.server.public), (ENV_CLIENT_PUBLIC, keys.client.public), (ENV_CLIENT_SECRET, keys.client.secret)):
        monkeypatch.setenv(var, value)
    with pytest.raises(_Stop):
        hl_module.main(["--address", remote])  # distant mais chiffré : accepté
    assert seen["hl"].curve == keys.client_cfg
    monkeypatch.setattr(numerai_module, "NumeraiWorker", capture("numerai"))
    with pytest.raises(_Stop):
        numerai_module.main(["--address", remote])
    assert seen["numerai"].curve == keys.client_cfg
    for var in (ENV_SERVER_PUBLIC, ENV_CLIENT_PUBLIC, ENV_CLIENT_SECRET):
        monkeypatch.delenv(var)
    with pytest.raises(_Stop):
        hl_module.main(["--address", remote, "--allow-plaintext-remote"])  # choix explicite en clair
    assert seen["hl"].curve is None
    monkeypatch.setenv(ENV_SERVER_PUBLIC, keys.server.public)
    with pytest.raises(SystemExit):
        hl_module.main(["--address", "tcp://127.0.0.1:1"])  # configuration partielle : erreur, pas un repli en clair


# --- Scripts moteur --------------------------------------------------------------------------


def _load_script(name: str) -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def test_the_injector_publishes_over_curve_when_the_server_keys_are_in_the_environment(monkeypatch: pytest.MonkeyPatch, keys: Keys, ctx: zmq.Context[Any]) -> None:
    monkeypatch.setenv(SECRET_ENV_VAR, SECRET)
    monkeypatch.setenv(ENV_SERVER_SECRET, keys.server.secret)
    monkeypatch.setenv(ENV_CLIENT_KEYS, keys.client.public)
    endpoint = f"tcp://127.0.0.1:{_free_port()}"
    sub = _subscribe(ctx, endpoint, keys.client_cfg)
    assert _load_script("inject_test_signal").main(["--endpoint", endpoint, "--settle", "1.5", "--repeat", "3"]) == 0
    assert sub.poll(5000), "le signal chiffré n'est jamais arrivé"
    assert open_envelope(sub.recv_multipart()[1], SECRET, venue=TargetVenue.HYPERLIQUID).accepted


def test_the_scripts_refuse_a_remote_endpoint_without_curve_and_a_partial_curve_configuration(monkeypatch: pytest.MonkeyPatch, keys: Keys) -> None:
    for var in (ENV_SERVER_SECRET, ENV_CLIENT_KEYS):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv(SECRET_ENV_VAR, SECRET)
    injector, bridge = _load_script("inject_test_signal"), _load_script("run_platform_bridge")
    assert injector.main(["--endpoint", "tcp://10.0.0.5:25570", "--settle", "0"]) == 2
    assert bridge.main(["--endpoint", "tcp://10.0.0.5:25570"]) == 2
    monkeypatch.setenv(ENV_SERVER_SECRET, keys.server.secret)  # sans liste de clients autorisés : refusé
    assert injector.main(["--endpoint", f"tcp://127.0.0.1:{_free_port()}", "--settle", "0"]) == 2
    assert bridge.main(["--endpoint", f"tcp://127.0.0.1:{_free_port()}"]) == 2


# --- Génération des clés ---------------------------------------------------------------------


def _parse_env(path: Path) -> dict[str, str]:
    return dict(line.split("=", 1) for line in path.read_text(encoding="utf-8").splitlines() if line)


def test_key_generation_script_writes_working_env_files_and_never_prints_a_secret(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    script = _load_script("generate_curve_keys")
    assert script.main(["--clients", "hyperliquid", "bybit", "--out-dir", str(tmp_path)]) == 0
    printed = capsys.readouterr()
    server_env = _parse_env(tmp_path / "curve_server.env")
    hl_env, bybit_env = _parse_env(tmp_path / "curve_hyperliquid.env"), _parse_env(tmp_path / "curve_bybit.env")
    for secret in (server_env[ENV_SERVER_SECRET], hl_env[ENV_CLIENT_SECRET], bybit_env[ENV_CLIENT_SECRET]):
        assert secret not in printed.out + printed.err  # jamais affiché
    server = CurveServerConfig.from_env(server_env)
    hl, bybit = CurveClientConfig.from_env(hl_env), CurveClientConfig.from_env(bybit_env)
    assert server is not None and hl is not None and bybit is not None
    assert server.authorized_clients == {hl.public_key, bybit.public_key} and hl.server_public_key == bybit.server_public_key == server.public_key
    assert hl.public_key != bybit.public_key


def test_key_generation_never_overwrites_and_validates_names(tmp_path: Path) -> None:
    script = _load_script("generate_curve_keys")
    assert script.main(["--clients", "hyperliquid", "--out-dir", str(tmp_path)]) == 0
    before = (tmp_path / "curve_server.env").read_text(encoding="utf-8")
    assert script.main(["--clients", "hyperliquid", "--out-dir", str(tmp_path)]) == 2  # régénérer invaliderait les clés déployées
    assert (tmp_path / "curve_server.env").read_text(encoding="utf-8") == before
    assert script.main(["--clients", "Bad Name", "--out-dir", str(tmp_path / "x")]) == 2
    assert script.main(["--clients", "a", "a", "--out-dir", str(tmp_path / "y")]) == 2
