"""Real, executable tests for core/config.py's ZmqSettings and _validate_zmq_address.

Added alongside the ZMQ bind/connect address split (see ZmqSettings.
broker_bind_frontend_address's own docstring) that makes
docker-compose.yml's alpha-engine service reach nothierly-agents' real
ZMQ broker across containers. That required _validate_zmq_address to
accept a real DNS hostname (a Docker Compose service name like
"nothierly-agents"), not just an IPv4 literal -- its own docstring
already claimed to accept "a valid hostname" but the implementation, at
the time this file was added, did not; ``tcp://nothierly-agents:5559``
would have failed AppSettings validation outright at container startup.
"""

from __future__ import annotations

import pytest

from core.config import ZmqSettings, _validate_zmq_address


def test_accepts_ipv4_literal() -> None:
    assert _validate_zmq_address("tcp://127.0.0.1:5555") == "tcp://127.0.0.1:5555"


def test_accepts_wildcard_bind_address() -> None:
    assert _validate_zmq_address("tcp://0.0.0.0:5559") == "tcp://0.0.0.0:5559"


def test_accepts_localhost_and_star() -> None:
    assert _validate_zmq_address("tcp://localhost:5555") == "tcp://localhost:5555"
    assert _validate_zmq_address("tcp://*:5555") == "tcp://*:5555"


def test_accepts_a_real_docker_compose_service_name() -> None:
    """The exact real gap this file exists to lock in -- see module docstring."""
    assert _validate_zmq_address("tcp://nothierly-agents:5559") == "tcp://nothierly-agents:5559"


def test_accepts_a_dotted_hostname() -> None:
    assert _validate_zmq_address("tcp://redis-broker.internal:6379") == "tcp://redis-broker.internal:6379"


@pytest.mark.parametrize(
    "bad_address",
    [
        "tcp://-badhost:5555",  # leading hyphen -- not a valid hostname label
        "tcp://bad_host:5555",  # underscore -- not a valid hostname character
        "tcp://999.999.999.999:5555",  # dotted-quad shape, not a real IP, not a real hostname either
        "tcp://192.168.1.999:5555",  # same real gap, more realistic typo shape
        "tcp://:5555",  # empty host
        "udp://127.0.0.1:5555",  # unsupported scheme
    ],
)
def test_rejects_malformed_addresses(bad_address: str) -> None:
    with pytest.raises(ValueError):
        _validate_zmq_address(bad_address)


def test_broker_bind_addresses_default_to_none() -> None:
    """Unset (the original, still-default single-container topology) -- see the field's own docstring."""
    settings = ZmqSettings()
    assert settings.broker_bind_frontend_address is None
    assert settings.broker_bind_backend_address is None


def test_broker_bind_addresses_can_differ_from_connect_addresses() -> None:
    """The real bind/connect split a multi-container deployment needs -- see the field's own docstring."""
    settings = ZmqSettings(
        broker_frontend_address="tcp://nothierly-agents:5559",
        broker_backend_address="tcp://nothierly-agents:5560",
        broker_bind_frontend_address="tcp://0.0.0.0:5559",
        broker_bind_backend_address="tcp://0.0.0.0:5560",
    )
    assert settings.broker_frontend_address == "tcp://nothierly-agents:5559"
    assert settings.broker_bind_frontend_address == "tcp://0.0.0.0:5559"


def test_blank_env_var_means_unset_for_bind_addresses() -> None:
    """Same real pydantic-settings quirk (blank string, not None, for an unset env var) as the connect addresses already handle -- see _blank_broker_address_means_unset's own docstring."""
    settings = ZmqSettings(broker_bind_frontend_address="", broker_bind_backend_address="")
    assert settings.broker_bind_frontend_address is None
    assert settings.broker_bind_backend_address is None
