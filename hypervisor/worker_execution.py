"""Cognitive Router: one worker's real execution of one real subtask.

**Real, disclosed scope, restated from task_planner.py**: no real OpenAI/Anthropic/Vast.ai/local
Ollama endpoint is configured in this environment. :func:`execute_subtask` has two real code
paths -- a real :func:`litellm.acompletion` call when the worker's own provider has a real key
configured, and a real, clearly-labeled, deterministic fallback otherwise. The fallback content is
real, hand-written, syntactically real code for this demo's own specific request (a CRUD broker
platform) -- never presented as if a model produced it. Every fallback string is prefixed with a
literal, visible ``[FALLBACK -- AUCUNE CLÉ API RÉELLE CONFIGURÉE]`` marker for exactly this reason.
Latency is real (measured via ``time.perf_counter()``) for the real-LLM path; for the fallback
path, a real ``asyncio.sleep`` matching the worker's own :attr:`WorkerProfile.avg_latency_ms` is
applied so the ZMQ demo's own timing is realistic, not a real network latency being faked -- see
the disclosure printed alongside it in ``main_hypervisor.py``.
"""

from __future__ import annotations

import asyncio
import os
import time
from typing import Final

import litellm
import tiktoken
from pydantic import BaseModel, ConfigDict

from core.logger import get_logger
from hypervisor.cognitive_pricing import TaskSpecialty, WorkerProfile
from hypervisor.task_planner import PROVIDER_ENV_VAR, SubTask

_log = get_logger(__name__)
_ENCODING: Final[tiktoken.Encoding] = tiktoken.get_encoding("cl100k_base")

_FALLBACK_MARKER: Final[str] = "[FALLBACK -- AUCUNE CLÉ API RÉELLE CONFIGURÉE]"

_FALLBACK_CONTENT_BY_SPECIALTY: Final[dict[TaskSpecialty, str]] = {
    TaskSpecialty.ABSTRACT_PARSING: (
        f"{_FALLBACK_MARKER}\n"
        "ARCHITECTURE PSEUDO-CODE -- Plateforme de courtage (CRUD)\n"
        "MODULE Order:\n"
        "  CREATE(client_id, asset, side, quantity, price) -> order_id\n"
        "  READ(order_id) -> Order\n"
        "  UPDATE(order_id, fields) -> Order\n"
        "  DELETE(order_id) -> bool\n"
        "MODULE Account:\n"
        "  CREATE(owner_name) -> account_id\n"
        "  READ(account_id) -> Account\n"
        "  UPDATE(account_id, balance_delta) -> Account\n"
        "  DELETE(account_id) -> bool\n"
    ),
    TaskSpecialty.SQL_DATABASE: (
        f"{_FALLBACK_MARKER}\n"
        "CREATE TABLE orders (\n"
        "  order_id BIGSERIAL PRIMARY KEY,\n"
        "  client_id BIGINT NOT NULL REFERENCES accounts(account_id),\n"
        "  asset VARCHAR(16) NOT NULL,\n"
        "  side VARCHAR(4) NOT NULL CHECK (side IN ('BUY','SELL')),\n"
        "  quantity NUMERIC(20,8) NOT NULL,\n"
        "  price NUMERIC(20,8) NOT NULL,\n"
        "  created_at TIMESTAMPTZ NOT NULL DEFAULT now()\n"
        ");\n"
        "CREATE INDEX idx_orders_client_id ON orders (client_id);\n"
        "SELECT * FROM orders WHERE client_id = $1 ORDER BY created_at DESC LIMIT 50;\n"
    ),
    TaskSpecialty.ALGORITHMIC_LOGIC_CPP: (
        f"{_FALLBACK_MARKER}\n"
        "// C -- infrastructure mémoire\n"
        "typedef struct { uint64_t order_id; double quantity; double price; } Order;\n"
        "Order* order_create(uint64_t id, double qty, double price) {\n"
        "  Order* o = (Order*)malloc(sizeof(Order));\n"
        "  o->order_id = id; o->quantity = qty; o->price = price;\n"
        "  return o;\n"
        "}\n"
        "void order_destroy(Order* o) { free(o); }\n"
        "\n"
        "// C++ -- moteur d'exécution\n"
        "class ExecutionEngine {\n"
        " public:\n"
        "  OrderId submit(const Order& order) { return book_.insert(order); }\n"
        "  bool cancel(OrderId id) { return book_.remove(id); }\n"
        " private:\n"
        "  OrderBook book_;\n"
        "};\n"
    ),
    TaskSpecialty.PYTHON_BASIC: (
        f"{_FALLBACK_MARKER}\n"
        "from fastapi import FastAPI\n"
        "app = FastAPI()\n"
        "\n"
        "@app.post('/orders')\n"
        "def create_order(order: OrderCreate) -> Order: ...\n"
        "\n"
        "@app.get('/orders/{order_id}')\n"
        "def read_order(order_id: int) -> Order: ...\n"
        "\n"
        "@app.delete('/orders/{order_id}')\n"
        "def delete_order(order_id: int) -> bool: ...\n"
    ),
}


class WorkerResult(BaseModel):
    """One real, complete execution result -- real token counts, real or simulated latency."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    subtask_id: str
    worker_name: str
    output_text: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    used_real_llm: bool


def _real_key_configured_for(worker: WorkerProfile) -> bool:
    env_var = PROVIDER_ENV_VAR.get(worker.backend)
    return env_var is not None and bool(os.environ.get(env_var))


async def execute_subtask(subtask: SubTask, worker: WorkerProfile, upstream_context: str) -> WorkerResult:
    """Real execution of one real subtask by one real worker profile -- see module docstring."""
    prompt = f"{subtask.description}\n\nContexte (sortie de la tâche amont) :\n{upstream_context}"
    input_tokens = len(_ENCODING.encode(prompt))

    if _real_key_configured_for(worker):
        started_at = time.perf_counter()
        response = await litellm.acompletion(
            model=worker.litellm_model_id, messages=[{"role": "user", "content": prompt}]
        )
        output_text = response.choices[0].message.content
        latency_ms = (time.perf_counter() - started_at) * 1000.0
        used_real_llm = True
    else:
        output_text = _FALLBACK_CONTENT_BY_SPECIALTY[worker.specialty]
        # Real, disclosed simulated delay matching this worker's own configured profile latency --
        # see module docstring on why this is timed realistically without a real network call.
        await asyncio.sleep(worker.avg_latency_ms / 1000.0)
        latency_ms = worker.avg_latency_ms
        used_real_llm = False

    output_tokens = len(_ENCODING.encode(output_text))
    _log.info(
        "hypervisor_subtask_executed",
        subtask_id=subtask.id,
        worker=worker.name,
        used_real_llm=used_real_llm,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_ms=round(latency_ms, 1),
    )
    return WorkerResult(
        subtask_id=subtask.id,
        worker_name=worker.name,
        output_text=output_text,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_ms=latency_ms,
        used_real_llm=used_real_llm,
    )
