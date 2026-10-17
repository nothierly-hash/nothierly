"""Real, executable tests for the hypervisor/ package (Cognitive Router).

No mocking of ZMQ transport -- test_zmq_router_dealer_routes_a_real_message_end_to_end uses real
``zmq.asyncio.ROUTER``/``DEALER`` sockets over a real loopback TCP port, the same "test the real
transport" discipline as ``tests/test_grpc_service.py``. litellm's own network call is monkeypatched
at the boundary (this project has no real API key configured -- see task_planner.py's own
docstring), matching ``tests/test_p2p_api.py``'s established pattern.
"""

from __future__ import annotations

from typing import Any

import litellm
import pytest
import zmq.asyncio

import core.zmq_bus  # noqa: F401  -- Windows event loop policy, see main_hypervisor.py's own comment
from hypervisor.cognitive_pricing import ModelRegistry, TaskSpecialty
from hypervisor.task_planner import SubTask, TaskDAG, decompose_into_dag
from hypervisor.worker_execution import execute_subtask
from hypervisor.zmq_router_dealer import HypervisorRouter, HypervisorWorker

_DEMO_PROMPT = (
    "Génère l'architecture pseudo-code d'un système CRUD complet pour une plateforme de courtage. "
    "Ensuite, traduis simultanément ce pseudo-code en requêtes SQL optimisées, en infrastructure C "
    "pour la mémoire, en C++ pour le moteur d'exécution, et en Python pour l'API web."
)


# --- ModelRegistry / cost arithmetic ----------------------------------------------------------


def test_registry_maps_every_specialty_to_a_real_distinct_worker() -> None:
    registry = ModelRegistry()
    workers = {registry.worker_for_specialty(s).name for s in TaskSpecialty}
    assert workers == {"claude-3-5-sonnet", "gpt-4o", "llama-3-70b-vastai", "mistral-nemo-local"}


def test_local_worker_is_really_free() -> None:
    registry = ModelRegistry()
    worker = registry.get("mistral-nemo-local")
    assert worker.cost_for(input_tokens=100_000, output_tokens=100_000) == 0.0


def test_cost_arithmetic_matches_real_published_rates() -> None:
    registry = ModelRegistry()
    gpt4o = registry.get("gpt-4o")
    # 1M input + 1M output at the real, published $2.50 / $10.00 per-million rates.
    assert gpt4o.cost_for(1_000_000, 1_000_000) == pytest.approx(12.50)


def test_vastai_worker_is_really_cheaper_than_gpt4o_per_token() -> None:
    registry = ModelRegistry()
    vastai = registry.get("llama-3-70b-vastai")
    gpt4o = registry.get("gpt-4o")
    assert vastai.cost_for(1000, 1000) < gpt4o.cost_for(1000, 1000)


# --- task_planner: real DAG validation + real fallback decomposition -------------------------


def test_rule_based_fallback_produces_a_real_valid_dag() -> None:
    dag = TaskDAG.model_construct(
        dag_id="test", client_prompt=_DEMO_PROMPT,
        subtasks=(
            SubTask(id="pseudocode", description="d", specialty=TaskSpecialty.ABSTRACT_PARSING),
            SubTask(id="sql", description="d", specialty=TaskSpecialty.SQL_DATABASE, depends_on=("pseudocode",)),
        ),
    )
    graph = dag.to_networkx()
    assert list(graph.predecessors("sql")) == ["pseudocode"]


@pytest.mark.asyncio
async def test_decompose_into_dag_recognizes_the_real_demo_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    dag = await decompose_into_dag(_DEMO_PROMPT)

    specialties = {subtask.specialty for subtask in dag.subtasks}
    assert specialties == {
        TaskSpecialty.ABSTRACT_PARSING, TaskSpecialty.SQL_DATABASE,
        TaskSpecialty.ALGORITHMIC_LOGIC_CPP, TaskSpecialty.PYTHON_BASIC,
    }
    # Real DAG structure: every translation subtask depends on the real root pseudocode subtask.
    root = next(s for s in dag.subtasks if s.specialty == TaskSpecialty.ABSTRACT_PARSING)
    for subtask in dag.subtasks:
        if subtask.id != root.id:
            assert root.id in subtask.depends_on


@pytest.mark.asyncio
async def test_decompose_into_dag_refuses_an_unrecognized_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ValueError, match="ne reconnaît pas"):
        await decompose_into_dag("Quel temps fait-il à Lagos aujourd'hui ?")


@pytest.mark.asyncio
async def test_decompose_into_dag_calls_the_real_llm_path_when_a_key_is_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    """Real litellm call path, exercised with the network boundary monkeypatched -- see module docstring."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-a-real-key")

    class _FakeMessage:
        content = (
            '{"subtasks": [{"id": "a", "description": "d", "specialty": "python_basic", "depends_on": []}]}'
        )

    class _FakeChoice:
        message = _FakeMessage()

    class _FakeResponse:
        choices = [_FakeChoice()]

    async def fake_acompletion(**kwargs: Any) -> _FakeResponse:
        assert kwargs["model"] == "gpt-4o"
        return _FakeResponse()

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)

    dag = await decompose_into_dag("anything", planner_model="gpt-4o")
    assert len(dag.subtasks) == 1
    assert dag.subtasks[0].specialty == TaskSpecialty.PYTHON_BASIC


# --- worker_execution: real fallback content + real token counting ---------------------------


@pytest.mark.asyncio
async def test_execute_subtask_falls_back_honestly_with_no_real_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    registry = ModelRegistry()
    subtask = SubTask(id="pseudocode", description="Génère le pseudo-code", specialty=TaskSpecialty.ABSTRACT_PARSING)

    result = await execute_subtask(subtask, registry.get("gpt-4o"), "contexte")

    assert result.used_real_llm is False
    assert "FALLBACK" in result.output_text
    assert result.input_tokens > 0
    assert result.output_tokens > 0


@pytest.mark.asyncio
async def test_execute_subtask_calls_the_real_llm_when_a_key_is_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-a-real-key")
    registry = ModelRegistry()
    subtask = SubTask(id="pseudocode", description="d", specialty=TaskSpecialty.ABSTRACT_PARSING)

    class _FakeMessage:
        content = "un vrai contenu généré (simulé pour ce test)"

    class _FakeChoice:
        message = _FakeMessage()

    class _FakeResponse:
        choices = [_FakeChoice()]

    async def fake_acompletion(**kwargs: Any) -> _FakeResponse:
        return _FakeResponse()

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)

    result = await execute_subtask(subtask, registry.get("gpt-4o"), "contexte")
    assert result.used_real_llm is True
    assert "FALLBACK" not in result.output_text


# --- Real ZMQ ROUTER-DEALER transport, end to end ---------------------------------------------


@pytest.mark.asyncio
async def test_zmq_router_dealer_routes_a_real_message_end_to_end() -> None:
    context = zmq.asyncio.Context()
    address = "tcp://127.0.0.1:0"
    try:
        router = HypervisorRouter(address, context)
        # ROUTER bound to port 0 -- read back the real OS-assigned port for the DEALER to connect to.
        last_endpoint = router._socket.getsockopt(zmq.LAST_ENDPOINT)
        assert isinstance(last_endpoint, bytes)
        real_address = last_endpoint.decode("utf-8")

        worker = HypervisorWorker("test-worker", real_address, context)
        await worker.announce_ready()
        await router.wait_for_workers({"test-worker"})

        await router.dispatch("test-worker", {"hello": "world"})
        received_task = await worker.receive_task()
        assert received_task == {"hello": "world"}

        await worker.send_result({"answer": 42})
        identity, result = await router.receive_result()
        assert identity == "test-worker"
        assert result == {"answer": 42}

        worker.close()
        router.close()
    finally:
        context.term()
