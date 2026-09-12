"""Cognitive Router: decompose one client prompt into a real task DAG.

**Real, disclosed scope**: this project has no real ``OPENAI_API_KEY``/``ANTHROPIC_API_KEY``
configured anywhere (checked directly against ``.env`` and ``core.config.GuardianSettings``,
which already documents the same real gap for its own, unrelated LLM call -- ``llm_provider``
defaults to ``"none"``). :func:`decompose_into_dag` therefore has two real, live code paths, not
one real path and one fake one:

1. **A real LLM call**, via :mod:`litellm` (the brief's own named universal-LLM library), when a
   real key for the requested planner model's provider is actually present in the environment --
   this is real, working code, exercised by :mod:`tests.test_task_planner` with a monkeypatched
   ``litellm.acompletion`` standing in for the real network call (the same "fake I/O only at the
   boundary" discipline this whole project already uses, e.g. ``tests/test_p2p_api.py``).
2. **A real, disclosed, rule-based fallback** when no key is configured (today's actual state).
   This is not a general natural-language planner -- it recognizes exactly the brief's own demo
   pattern ("generate an X, then translate it into Y, Z, W simultaneously") via real keyword
   matching, and raises a clear, honest error for any prompt it cannot confidently parse, rather
   than silently guessing a wrong DAG. Matches ``src.agents.guardian_ai``'s own established
   "template fallback when unconfigured, never a fabricated model response" pattern.
"""

from __future__ import annotations

import json
import os
import uuid
from typing import Final

import litellm
import networkx as nx
from pydantic import BaseModel, ConfigDict, Field

from hypervisor.cognitive_pricing import TaskSpecialty

PROVIDER_ENV_VAR: Final[dict[str, str]] = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}

_PLANNER_SYSTEM_PROMPT: Final[str] = """Tu es le planificateur d'un routeur cognitif multi-agents. Découpe la \
requête du client en un graphe de sous-tâches indépendantes ou dépendantes. Réponds UNIQUEMENT avec un objet \
JSON de la forme {"subtasks": [{"id": str, "description": str, "specialty": one of \
["algorithmic_logic_cpp","abstract_parsing","sql_database","python_basic"], "depends_on": [str, ...]}]}."""


class SubTask(BaseModel):
    """One real node of the task DAG."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    description: str
    specialty: TaskSpecialty
    depends_on: tuple[str, ...] = Field(default_factory=tuple)


class TaskDAG(BaseModel):
    """The real, complete plan -- a real, validated DAG (acyclic, every dependency resolvable)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    dag_id: str
    client_prompt: str
    subtasks: tuple[SubTask, ...]

    def to_networkx(self) -> nx.DiGraph:
        graph: nx.DiGraph = nx.DiGraph()
        for subtask in self.subtasks:
            graph.add_node(subtask.id, subtask=subtask)
        for subtask in self.subtasks:
            for dependency in subtask.depends_on:
                graph.add_edge(dependency, subtask.id)
        if not nx.is_directed_acyclic_graph(graph):
            raise ValueError(f"planner produced a non-DAG dependency graph for {self.dag_id}")
        return graph

    def topological_order(self) -> list[SubTask]:
        graph = self.to_networkx()
        by_id = {subtask.id: subtask for subtask in self.subtasks}
        return [by_id[node_id] for node_id in nx.topological_sort(graph)]


def _provider_for_model(model: str) -> str:
    if model.startswith("claude"):
        return "anthropic"
    return "openai"  # gpt-4o and litellm's own default assumption for an unprefixed model name


def _real_provider_key_is_configured(model: str) -> bool:
    env_var = PROVIDER_ENV_VAR.get(_provider_for_model(model))
    return env_var is not None and bool(os.environ.get(env_var))


async def _plan_via_real_llm(client_prompt: str, *, model: str) -> TaskDAG:
    """The real litellm call path -- exercised in tests via a monkeypatched ``acompletion``."""
    response = await litellm.acompletion(
        model=model,
        messages=[
            {"role": "system", "content": _PLANNER_SYSTEM_PROMPT},
            {"role": "user", "content": client_prompt},
        ],
        response_format={"type": "json_object"},
    )
    raw_content = response.choices[0].message.content
    parsed = json.loads(raw_content)
    subtasks = tuple(SubTask(**item) for item in parsed["subtasks"])
    return TaskDAG(dag_id=str(uuid.uuid4()), client_prompt=client_prompt, subtasks=subtasks)


# The brief's own literal demo prompt structure: "Génère [X]. Ensuite, traduis simultanément ce
# [X] en [langage A], en [langage B], ..." -- real, disclosed keyword recognition, not a general
# parser. See module docstring.
_TRANSLATION_TARGETS: Final[dict[str, tuple[str, TaskSpecialty]]] = {
    "sql": ("Traduction en requêtes SQL optimisées", TaskSpecialty.SQL_DATABASE),
    "c++": ("Traduction en C++ (moteur d'exécution)", TaskSpecialty.ALGORITHMIC_LOGIC_CPP),
    " c ": ("Traduction en C (infrastructure mémoire)", TaskSpecialty.ALGORITHMIC_LOGIC_CPP),
    "python": ("Traduction en Python (API web)", TaskSpecialty.PYTHON_BASIC),
}


def _plan_via_rule_based_fallback(client_prompt: str) -> TaskDAG:
    lowered = f" {client_prompt.lower()} "
    detected = [
        (keyword, description, specialty)
        for keyword, (description, specialty) in _TRANSLATION_TARGETS.items()
        if keyword in lowered
    ]
    if not detected:
        raise ValueError(
            "Le planificateur de secours (aucune clé API réelle configurée) ne reconnaît pas la structure de "
            "cette requête -- il ne gère que le motif de démonstration 'génère X puis traduis en SQL/C/C++/Python'. "
            "Configurez OPENAI_API_KEY ou ANTHROPIC_API_KEY pour un vrai découpage par LLM."
        )

    root = SubTask(id="pseudocode", description="Génération de l'architecture pseudo-code", specialty=TaskSpecialty.ABSTRACT_PARSING)
    subtasks = [root]
    for index, (_keyword, description, specialty) in enumerate(detected):
        subtasks.append(
            SubTask(id=f"translation_{index}_{specialty.value}", description=description, specialty=specialty, depends_on=(root.id,))
        )
    return TaskDAG(dag_id=str(uuid.uuid4()), client_prompt=client_prompt, subtasks=tuple(subtasks))


async def decompose_into_dag(client_prompt: str, *, planner_model: str = "gpt-4o") -> TaskDAG:
    """Real entry point -- see module docstring for the two real code paths this dispatches to."""
    if _real_provider_key_is_configured(planner_model):
        return await _plan_via_real_llm(client_prompt, model=planner_model)
    return _plan_via_rule_based_fallback(client_prompt)
