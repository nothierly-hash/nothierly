"""Real, executable tests for network/ebpf_ingestor.py.

Two real things are tested here, both genuinely verifiable on any OS
(see that module's own docstring on why the kernel-level XDP program
itself cannot be): (1) parse_ring_buffer_event's byte-level decoding,
built against synthetic buffers matching the exact wire layout the real
C struct would emit; (2) that EbpfIngestor honestly refuses to pretend
to work on this non-Linux machine, rather than silently no-opping.
"""

from __future__ import annotations

import struct

import pytest

from network.ebpf_ingestor import (
    EbpfIngestor,
    L2BookPacketEvent,
    LinuxOnlyFeatureError,
    _EVENT_HEADER_STRUCT_FORMAT,
    parse_ring_buffer_event,
    render_xdp_program_source,
)


def _build_raw_event(*, timestamp_ns: int, src_ip: int, src_port: int, payload: bytes, padded_to: int = 32) -> bytes:
    """Real synthetic ring-buffer bytes, matching the C struct's exact layout: header + payload + trailing padding."""
    header = struct.pack(_EVENT_HEADER_STRUCT_FORMAT, timestamp_ns, src_ip, src_port, len(payload))
    padding = b"\x00" * max(0, padded_to - len(payload))
    return header + payload + padding


def test_parses_a_well_formed_event() -> None:
    raw = _build_raw_event(timestamp_ns=1_700_000_000_000, src_ip=0x0100007F, src_port=5000, payload=b"L2-BOOK-DELTA-PAYLOAD")

    event = parse_ring_buffer_event(raw)

    assert isinstance(event, L2BookPacketEvent)
    assert event.timestamp_ns == 1_700_000_000_000
    assert event.src_port == 5000
    assert event.payload == b"L2-BOOK-DELTA-PAYLOAD"


def test_src_ip_str_decodes_dotted_quad_correctly() -> None:
    # 127.0.0.1 packed little-endian the way the kernel's ip->saddr (network byte order on the wire,
    # but read as a raw u32 here) would appear: 0x0100007F unpacks to "127.0.0.1" via the property.
    raw = _build_raw_event(timestamp_ns=1, src_ip=0x0100007F, src_port=1, payload=b"x")
    event = parse_ring_buffer_event(raw)
    assert event.src_ip_str == "127.0.0.1"


def test_payload_is_truncated_to_the_declared_length_not_the_full_padded_buffer() -> None:
    raw = _build_raw_event(timestamp_ns=1, src_ip=0, src_port=1, payload=b"short", padded_to=64)
    event = parse_ring_buffer_event(raw)
    assert event.payload == b"short"
    assert len(event.payload) == 5  # not 64 -- the trailing zero padding must never leak into the payload


def test_truncated_header_raises_value_error() -> None:
    with pytest.raises(ValueError, match="truncated"):
        parse_ring_buffer_event(b"\x00" * 4)  # far short of the 16-byte header


def test_payload_len_exceeding_available_bytes_raises_value_error() -> None:
    # A header claiming a 500-byte payload but only 3 bytes actually follow -- malformed/corrupt record.
    header = struct.pack(_EVENT_HEADER_STRUCT_FORMAT, 1, 0, 1, 500)
    with pytest.raises(ValueError, match="payload_len"):
        parse_ring_buffer_event(header + b"abc")


def test_empty_payload_is_valid() -> None:
    raw = _build_raw_event(timestamp_ns=1, src_ip=0, src_port=1, payload=b"", padded_to=0)
    event = parse_ring_buffer_event(raw)
    assert event.payload == b""


# =============================================================================
# render_xdp_program_source: pure string templating, no bcc/Linux dependency
# =============================================================================


def test_render_xdp_program_source_embeds_the_real_port() -> None:
    source = render_xdp_program_source(exchange_udp_port=9001)
    assert "#define EXCHANGE_UDP_PORT 9001" in source
    assert "BPF_RINGBUF_OUTPUT" in source
    assert "filter_l2_book_packets" in source


def test_render_xdp_program_source_rejects_invalid_port() -> None:
    with pytest.raises(ValueError):
        render_xdp_program_source(exchange_udp_port=70000)
    with pytest.raises(ValueError):
        render_xdp_program_source(exchange_udp_port=0)


# =============================================================================
# EbpfIngestor: the real, honest platform gate
# =============================================================================


def test_start_raises_linux_only_feature_error_on_this_non_linux_machine() -> None:
    """The actual, live-verifiable claim this module can make on this dev machine: it refuses, loudly, rather than faking success."""
    ingestor = EbpfIngestor(network_interface="eth0", exchange_udp_port=9001)
    with pytest.raises(LinuxOnlyFeatureError):
        ingestor.start()


def test_consume_forever_raises_if_start_never_succeeded() -> None:
    ingestor = EbpfIngestor(network_interface="eth0", exchange_udp_port=9001)
    with pytest.raises(RuntimeError, match="start"):
        ingestor.consume_forever(on_event=lambda _e: None)


def test_stop_before_start_is_a_safe_no_op() -> None:
    ingestor = EbpfIngestor(network_interface="eth0", exchange_udp_port=9001)
    ingestor.stop()  # must not raise
