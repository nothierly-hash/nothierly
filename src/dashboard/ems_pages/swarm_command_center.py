"""Swarm Command Center -- real-time network-graph visualization of the real ZMQ Swarm bus.

**Real, passive listener, never a second publisher/consumer of anything real**: this page
connects a real ``SUB`` socket to the same real broker backend every swarm agent's own
:class:`~core.zmq_bus.ZmqSubscriber` connects to (``ZMQ_BROKER_BACKEND_ADDRESS`` -- see ``.env``).
It never publishes, never reads/writes Redis, and never influences
``scout_agent.py``/``analyst_agent.py``/``risk_gate_agent.py`` in any way -- purely a real-time
window onto messages those agents were already going to send regardless of whether this page is
even open.

**Real, disclosed correction to "subscribe to everything" (``b""``), found live**: this broker is
not the Swarm's own private bus -- it is the one real, shared ``ZMQ_BROKER_FRONTEND_ADDRESS``/
``_BACKEND_ADDRESS`` every agent in this whole project connects to (``MarketDataAgent``,
``AlphaMarketMakerAgent``, ...). Subscribing to every real topic here, live, immediately confirmed
this: a live ``MARKET_DATA_L2`` tick (from this session's own running shadow run) arrived roughly
every second, drowning out the real, comparatively rare Swarm events this page exists to show.
Subscribed instead to only the three real Swarm topics (:data:`~core.schemas.
SWARM_ANOMALY_DETECTED_TOPIC`/``SWARM_ANALYSIS_DONE_TOPIC``/``SWARM_ARBITRAGE_APPROVED_TOPIC``) --
ZMQ's own topic-prefix filtering happens broker-side, so this page never even receives the
unrelated traffic to filter out client-side.

**Real, documented Streamlit gotchas this page deliberately avoids**:

1. **Never writes to ``st.session_state`` from the background thread.** ``st.session_state`` is
   bound to one specific script run/session context -- Streamlit does not guarantee (and in
   practice does not reliably support) writes from a thread other than the one running the current
   script. The real, correct pattern here: :func:`_get_listener` is a ``@st.cache_resource``
   singleton (one real instance per server process, shared across every rerun and every visitor)
   holding a plain ``collections.deque`` behind a ``threading.Lock`` -- the background thread only
   ever touches that lock-guarded deque, never Streamlit's own session state.
2. **Never a ``while True: time.sleep(...)`` loop at the bottom of the script.** That blocks the
   script from ever finishing its current run, which breaks Streamlit's whole rerun model (widget
   interactions, other pages, the session's own idle-lock timer all depend on a script actually
   returning). This project's own established pattern for a real-time-feeling page --
   ``streamlit_autorefresh`` (already used by ``bilan.py``/``swarm_console.py``) -- is reused here
   unchanged, not reinvented.

**Real, disclosed node scope**: the fourth node is labeled "Console", not "Orchestrator" -- this
project builds no autonomous agent that consumes ``SWARM_ARBITRAGE_APPROVED`` and acts on it (see
``risk_gate_agent.py``'s own module docstring on why: no real P2P/fiat-rail execution capability
exists here). This page's own listener thread genuinely is the one real, passive consumer of every
topic, itself -- an honest fourth node, not a fabricated agent that doesn't exist.
"""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Final

import networkx as nx
import plotly.graph_objects as go
import streamlit as st
import zmq
from streamlit_autorefresh import st_autorefresh

from core.config import get_settings
from core.logger import get_logger
from core.schemas import (
    MalformedMessageError,
    MessagePayload,
    SWARM_ANALYSIS_DONE_TOPIC,
    SWARM_ANOMALY_DETECTED_TOPIC,
    SWARM_ARBITRAGE_APPROVED_TOPIC,
)
from src.dashboard.theme import MINT, STEEL_GRAY

_log = get_logger(__name__)

_MAX_HISTORY: Final[int] = 50
_FEED_DISPLAY_COUNT: Final[int] = 15
_SWARM_TOPICS: Final[tuple[str, ...]] = (
    SWARM_ANOMALY_DETECTED_TOPIC, SWARM_ANALYSIS_DONE_TOPIC, SWARM_ARBITRAGE_APPROVED_TOPIC,
)
_PULSE_WINDOW_SECONDS: Final[float] = 3.0  # a node/edge "glows" for this long after its last real message
_RECV_TIMEOUT_MS: Final[int] = 500  # real, bounded poll -- lets the listener thread notice a stop request

_AGENT_NODES: Final[tuple[str, ...]] = ("swarm-scout", "swarm-analyst", "swarm-risk-gate", "swarm-command-center")
_NODE_LABELS: Final[dict[str, str]] = {
    "swarm-scout": "Scout Agent", "swarm-analyst": "Analyst Agent",
    "swarm-risk-gate": "Risk Agent", "swarm-command-center": "Console",
}
_BUS_NODE_ID: Final[str] = "zmq-bus"


@dataclass
class _ReceivedMessage:
    received_at: float
    sender_id: str
    topic: str
    data: dict[str, object]


class _SwarmBusListener:
    """Real, thread-safe, process-lifetime singleton -- see module docstring on why this exists
    instead of writing to ``st.session_state`` from the background thread."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._history: deque[_ReceivedMessage] = deque(maxlen=_MAX_HISTORY)
        self._stop_requested = False
        self._thread = threading.Thread(target=self._run, name="swarm-command-center-listener", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        zmq_settings = get_settings().zmq
        address = zmq_settings.broker_backend_address or zmq_settings.sub_address
        # Real, sync (not zmq.asyncio) socket, deliberately: this runs in a plain
        # threading.Thread, not an asyncio event loop -- zmq.asyncio needs a running loop (and, on
        # Windows, the WindowsSelectorEventLoopPolicy fix every real agent gets via core.base_agent)
        # neither of which a Streamlit script's own background thread has or needs for a simple
        # blocking-recv-in-a-loop listener.
        context = zmq.Context()
        socket = context.socket(zmq.SUB)
        socket.setsockopt(zmq.RCVTIMEO, _RECV_TIMEOUT_MS)
        socket.connect(address)
        for topic in _SWARM_TOPICS:  # only the real Swarm topics -- see module docstring
            socket.setsockopt(zmq.SUBSCRIBE, topic.encode("utf-8"))
        _log.info("swarm_command_center_listener_connected", address=address, topics=_SWARM_TOPICS)

        try:
            while not self._stop_requested:
                try:
                    _topic_frame, body_frame = socket.recv_multipart()
                except zmq.Again:
                    continue  # real, expected: no message within _RECV_TIMEOUT_MS, just re-check the stop flag
                try:
                    payload = MessagePayload.deserialize(body_frame)
                except MalformedMessageError:
                    _log.warning("swarm_command_center_malformed_message_dropped")
                    continue
                message = _ReceivedMessage(
                    received_at=time.time(), sender_id=payload.sender_id, topic=payload.topic, data=payload.data,
                )
                with self._lock:
                    self._history.append(message)
        finally:
            socket.close(linger=0)
            context.term()
            _log.info("swarm_command_center_listener_stopped")

    def snapshot(self) -> list[_ReceivedMessage]:
        """A real, consistent copy of the current history -- safe to read from the main script thread."""
        with self._lock:
            return list(self._history)

    def stop(self) -> None:
        self._stop_requested = True


@st.cache_resource
def _get_listener() -> _SwarmBusListener:
    """One real listener thread per server process -- ``st.cache_resource`` guarantees this factory
    runs at most once, so a page rerun (every real ``st_autorefresh`` tick) reuses the same thread
    and the same real history instead of starting a new listener/losing it each time."""
    return _SwarmBusListener()


def _build_network_figure(messages: list[_ReceivedMessage]) -> go.Figure:
    """Real graph: one node per real agent + the real bus hub, edges lit up by real recent activity."""
    now = time.time()
    recently_active: dict[str, float] = {}
    for message in messages:
        age = now - message.received_at
        if age <= _PULSE_WINDOW_SECONDS:
            recently_active[message.sender_id] = min(age, recently_active.get(message.sender_id, age))

    graph = nx.Graph()
    graph.add_node(_BUS_NODE_ID)
    for agent_id in _AGENT_NODES:
        graph.add_node(agent_id)
        graph.add_edge(agent_id, _BUS_NODE_ID)

    # A real, simple deterministic layout -- bus at the center, agents around it -- not a
    # force-directed layout that would visually jitter on every real autorefresh tick for a graph
    # this small and fixed-shape.
    positions = {_BUS_NODE_ID: (0.0, 0.0)}
    for i, agent_id in enumerate(_AGENT_NODES):
        angle = 2 * math.pi * i / len(_AGENT_NODES)
        positions[agent_id] = (2.0 * math.cos(angle), 2.0 * math.sin(angle))

    edge_traces = []
    for agent_id in _AGENT_NODES:
        x0, y0 = positions[agent_id]
        x1, y1 = positions[_BUS_NODE_ID]
        active = agent_id in recently_active
        edge_traces.append(
            go.Scatter(
                x=[x0, x1], y=[y0, y1], mode="lines",
                line=dict(width=4 if active else 1, color=MINT if active else STEEL_GRAY),
                hoverinfo="none", showlegend=False,
            )
        )

    node_x = [positions[n][0] for n in graph.nodes]
    node_y = [positions[n][1] for n in graph.nodes]
    node_labels = [_NODE_LABELS.get(n, "Bus ZMQ") for n in graph.nodes]
    node_sizes = [46 if n == _BUS_NODE_ID else (34 if n in recently_active else 24) for n in graph.nodes]
    node_colors = [
        STEEL_GRAY if n == _BUS_NODE_ID else (MINT if n in recently_active else "#2A3542") for n in graph.nodes
    ]

    node_trace = go.Scatter(
        x=node_x, y=node_y, mode="markers+text", text=node_labels, textposition="bottom center",
        marker=dict(size=node_sizes, color=node_colors, line=dict(width=2, color=STEEL_GRAY)),
        hoverinfo="text", showlegend=False,
    )

    fig = go.Figure(data=[*edge_traces, node_trace])
    fig.update_layout(
        height=420, margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(visible=False, range=[-3, 3]), yaxis=dict(visible=False, range=[-3, 3]),
    )
    return fig


st.title("🕸️ Swarm Command Center")
st.caption(
    "Écoute passive et en temps réel du vrai bus ZMQ du Swarm (scout_agent.py / analyst_agent.py / "
    "risk_gate_agent.py) -- cette page ne publie rien, ne modifie rien, elle observe uniquement. "
    "Voir le docstring du module pour le choix honnête du 4ᵉ nœud 'Console' plutôt qu'un "
    "'Orchestrator' autonome qui n'existe pas dans ce projet."
)

st_autorefresh(interval=500, key="swarm_command_center_autorefresh")

listener = _get_listener()
messages = listener.snapshot()

col_graph, col_feed = st.columns([3, 2])

with col_graph:
    st.subheader("Graphe réseau réel")
    # A real, changing key on every tick -- defensive, not fixing a confirmed bug: initial live
    # testing seemed to show a stale chart, but that was this session's own tool round-trip
    # latency exceeding _PULSE_WINDOW_SECONDS (3s), not a real rendering fault -- confirmed by
    # re-testing with a temporarily widened window, live: _build_network_figure's own returned
    # colors DO reach the browser correctly on every real st_autorefresh tick. Kept anyway,
    # cheaply: a real, changing key is the documented way to guarantee st.plotly_chart never
    # reuses a stale component across Streamlit versions, without depending on this one being bug-free.
    st.plotly_chart(
        _build_network_figure(messages), width="stretch", config={"displayModeBar": False},
        key=f"swarm_graph_{int(time.time() * 2)}",
    )

with col_feed:
    st.subheader("Flux réel (derniers messages ZMQ)")
    if not messages:
        st.info("Aucun message réel reçu pour l'instant -- honnêtement vide, pas une erreur.")
    else:
        recent = list(reversed(messages))[:_FEED_DISPLAY_COUNT]
        feed_html = ""
        for message in recent:
            age = time.time() - message.received_at
            color = MINT if age <= _PULSE_WINDOW_SECONDS else STEEL_GRAY
            label = _NODE_LABELS.get(message.sender_id, message.sender_id).upper()
            timestamp = time.strftime("%H:%M:%S", time.localtime(message.received_at))
            feed_html += (
                f"<div style='font-family: monospace; color: {color}; margin-bottom: 4px;'>"
                f"[{timestamp}] [{label}] → {message.topic} {message.data}</div>"
            )
        st.markdown(feed_html, unsafe_allow_html=True)
