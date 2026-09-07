"""CPU-bound FinBERT sentiment inference, isolated from the event loop.

:class:`SentimentEngine` is deliberately a plain, synchronous, asyncio-free
class: model inference is CPU-bound work, and running it directly inside
an ``async def`` method would block the event loop for the whole
duration of every call (tens to hundreds of milliseconds per headline on
CPU) -- exactly the "ultra-faible latence" system this project must not
be. Keeping this class fully synchronous is what makes it safe to use:
the future async agent that wires this in is expected to call
:meth:`SentimentEngine.analyze_text` via ``loop.run_in_executor(...)``,
which moves the call onto a worker thread and lets the event loop keep
processing other agents' I/O in the meantime. That wiring is out of
scope for this chapter (see the module's docstring in the eventual
``src/agents/nlp/agent.py``).
"""

from __future__ import annotations

import threading
from typing import ClassVar, Final

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.tokenization_utils_base import BatchEncoding

from core.logger import get_logger

_log = get_logger(__name__)

_DEFAULT_MODEL_NAME: Final[str] = "ProsusAI/finbert"
_MAX_SEQUENCE_LENGTH: Final[int] = 512


class SentimentEngine:
    """Loads FinBERT once and runs synchronous sentiment inference on text.

    Model/tokenizer loading (downloading weights on first use, then
    deserializing several hundred MB into memory) is expensive -- on the
    order of seconds, versus milliseconds for a single inference call --
    so it must happen exactly once per process, not on every
    :meth:`analyze_text` call. Use :meth:`get_instance` (not the
    constructor directly) to get the process-wide singleton shared by
    every caller.
    """

    _instance: ClassVar[SentimentEngine | None] = None
    # A `threading.Lock`, not an `asyncio.Lock`: this class is intended to
    # be invoked from a `ThreadPoolExecutor` by its future async caller
    # (see the module docstring), so `get_instance` can genuinely be
    # entered by multiple concurrent OS threads -- something an
    # `asyncio.Lock` (which only serializes coroutines within a single
    # event loop thread) would not protect against.
    _instance_lock: ClassVar[threading.Lock] = threading.Lock()

    def __init__(self, model_name: str = _DEFAULT_MODEL_NAME) -> None:
        """Load the tokenizer and model. Expensive -- prefer :meth:`get_instance`.

        Args:
            model_name: HuggingFace Hub model identifier to load.
        """
        self._model_name = model_name
        self._log = get_logger(f"{__name__}.{model_name}")
        self._log.info("sentiment_engine_loading_model", model_name=model_name)

        # AutoTokenizer.from_pretrained is a dynamically-dispatched
        # classmethod that transformers' own type stubs don't fully
        # annotate (unlike AutoModelForSequenceClassification.from_pretrained,
        # just below, which mypy resolves fine).
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)  # type: ignore[no-untyped-call]
        self._model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self._model.eval()

        num_labels: int = self._model.config.num_labels
        self._labels: list[str] = [
            str(self._model.config.id2label[i]).upper() for i in range(num_labels)
        ]

        self._log.info(
            "sentiment_engine_model_loaded", model_name=model_name, labels=self._labels
        )

    @classmethod
    def get_instance(cls, model_name: str = _DEFAULT_MODEL_NAME) -> SentimentEngine:
        """Return the process-wide singleton, loading the model on first call.

        Thread-safe against concurrent first-call races (see the class
        docstring for why a ``threading.Lock`` is used here rather than
        an ``asyncio.Lock``).

        Args:
            model_name: HuggingFace Hub model identifier. Only used if the
                singleton has not yet been created; ignored afterwards.

        Returns:
            The shared :class:`SentimentEngine` instance.
        """
        if cls._instance is not None:
            return cls._instance
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls(model_name)
        return cls._instance

    def analyze_text(self, text: str) -> dict[str, str | float]:
        """Classify the sentiment of ``text``. Synchronous, CPU-bound.

        Args:
            text: The headline/text to analyze. Longer inputs are
                truncated to the model's maximum sequence length
                (:data:`_MAX_SEQUENCE_LENGTH` tokens).

        Returns:
            A dict with two keys: ``"polarity"`` (one of ``"POSITIVE"``,
            ``"NEGATIVE"``, ``"NEUTRAL"``) and ``"score"`` (the model's
            confidence in that polarity, as a ``float`` in ``[0, 1]``).

        Raises:
            ValueError: If ``text`` is empty or whitespace-only.
        """
        if not text.strip():
            raise ValueError("text must not be empty or whitespace-only.")

        inputs: BatchEncoding = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=_MAX_SEQUENCE_LENGTH,
        )

        with torch.no_grad():
            outputs: SequenceClassifierOutput = self._model(**inputs)
            # transformers types `.logits` as Optional since some output
            # subclasses share the base dataclass; a forward pass without
            # `labels=` always populates it in practice.
            assert outputs.logits is not None
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]

        best_index = int(torch.argmax(probabilities).item())
        polarity = self._labels[best_index]
        score = float(probabilities[best_index].item())

        return {"polarity": polarity, "score": score}
