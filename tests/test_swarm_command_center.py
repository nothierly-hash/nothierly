"""Real, executable tests for src/dashboard/ems_pages/swarm_command_center.py's own pure logic.

``_build_network_figure`` is tested directly (no Streamlit runtime, no real ZMQ socket) -- the
same "test the real logic, not the transport" split this project already applies elsewhere (e.g.
``tests/test_shadow_crossing_engine.py``'s own ``_handle_client_order_message`` tests). This is
also the exact real check that resolved a real live-testing question during development: whether
a stale-looking chart in the browser was a real rendering bug or just this session's own tool
round-trip latency exceeding the real 3s pulse window -- these tests prove the pure function
itself has always been correct (confirmed independently, live, against the real running page too).
"""

from __future__ import annotations

import time

import plotly.graph_objects as go

from src.dashboard.ems_pages.swarm_command_center import (
    _AGENT_NODES,
    _BUS_NODE_ID,
    _NODE_LABELS,
    _PULSE_WINDOW_SECONDS,
    _ReceivedMessage,
    _build_network_figure,
)


def _message(sender_id: str, *, seconds_ago: float = 0.0) -> _ReceivedMessage:
    return _ReceivedMessage(received_at=time.time() - seconds_ago, sender_id=sender_id, topic="TEST", data={})


def _label_for(node_id: str) -> str:
    return _NODE_LABELS.get(node_id, "Bus ZMQ")


def _node_color(fig: go.Figure, node_id: str) -> str:
    node_trace = fig.data[-1]
    index = list(node_trace.text).index(_label_for(node_id))
    return str(node_trace.marker.color[index])


def test_empty_history_leaves_every_node_inactive() -> None:
    fig = _build_network_figure([])
    node_trace = fig.data[-1]
    # The bus node is always its own distinct (non-pulsing) color -- only real agent nodes pulse.
    assert node_trace.marker.color[0] != "#00E676"
    assert all(color != "#00E676" for color in node_trace.marker.color[1:])


def test_a_real_recent_message_pulses_its_own_real_sender_node() -> None:
    fig = _build_network_figure([_message("swarm-scout")])
    assert _node_color(fig, "swarm-scout") == "#00E676"
    # Every other real agent node stays inactive -- a pulse is per-sender, never global.
    for other in _AGENT_NODES:
        if other != "swarm-scout":
            assert _node_color(fig, other) != "#00E676"


def test_a_message_older_than_the_real_pulse_window_does_not_pulse() -> None:
    fig = _build_network_figure([_message("swarm-analyst", seconds_ago=_PULSE_WINDOW_SECONDS + 1.0)])
    assert _node_color(fig, "swarm-analyst") != "#00E676"


def test_a_message_just_inside_the_real_pulse_window_still_pulses() -> None:
    fig = _build_network_figure([_message("swarm-risk-gate", seconds_ago=_PULSE_WINDOW_SECONDS - 0.5)])
    assert _node_color(fig, "swarm-risk-gate") == "#00E676"


def test_multiple_real_recent_senders_each_pulse_independently() -> None:
    fig = _build_network_figure([_message("swarm-scout"), _message("swarm-risk-gate")])
    assert _node_color(fig, "swarm-scout") == "#00E676"
    assert _node_color(fig, "swarm-risk-gate") == "#00E676"
    assert _node_color(fig, "swarm-analyst") != "#00E676"


def test_the_bus_node_is_always_present_regardless_of_history() -> None:
    fig = _build_network_figure([_message("swarm-scout")])
    node_trace = fig.data[-1]
    assert _label_for(_BUS_NODE_ID) in list(node_trace.text)
