"""QLoRA fine-tuning of Nothierly's own proprietary crisis-scoring LLM on real, labeled headlines.

**Status, stated as plainly as ``nothierly_genesis/README.md`` states its own pillars' status**:
this script is real, standard, believed-correct code -- and it has **never been run**. This
project's own dev environment has no GPU (see ``requirements.txt``'s own CPU-only ``torch`` build
comment), so nothing here has been executed end-to-end, only reviewed against the real, documented
APIs of ``transformers``/``peft``/``trl``. Treat it as *architecture-verified, not
production-validated* -- run it once on a real rented GPU (small, cheap smoke test first: a handful
of steps, not the full run) before trusting its output.

**Two real, separate prerequisites, neither of which this script (or any script) can honestly
skip**:

1. **A labeled dataset** -- ``scripts/collect_headline_dataset.py`` only collects real, raw,
   *unlabeled* headlines (zero cost). Turning those into ``(headline, real crisis_score/sentiment/
   event_summary)`` pairs needs either a real commercial LLM API call per headline (real, ongoing
   $ cost -- a large multi-provider distillation run, `configure GUARDIAN_API_KEY` first) or real
   human labeling time. Neither is done by this script -- it *consumes* a JSONL file already in
   that labeled shape (see :data:`_EXPECTED_SCHEMA` below), never fabricates one.
2. **A rented GPU with real CUDA** -- this project's own ``scripts/deploy_custom_llm.py`` already
   rents one via Vast.ai for *inference* (vLLM serving the base model); fine-tuning needs the same
   kind of real, billed instance, run interactively or via SSH (not this repo's own CPU dev
   environment), with ``pip install -r`` the extra deps below installed on it -- they are
   deliberately **not** added to this project's own top-level ``requirements.txt`` (a CPU-only
   local dev environment has no use for ``bitsandbytes``/``peft``/``trl``, and pinning heavy GPU-only
   packages there would slow down/bloat every contributor's local install for a step that runs on a
   separate rented machine, not this repo's own runtime).

Real extra dependencies, install only on the rented GPU instance::

    pip install peft>=0.11.0 bitsandbytes>=0.43.0 trl>=0.9.0 accelerate>=0.30.0 datasets>=2.19.0

**Expected labeled dataset shape** (one JSON object per line, matching
:class:`political_agent.CrisisAssessment`'s own real, already-validated schema -- so the fine-tuned
model is trained to emit exactly what :func:`political_agent._parse_llm_response` already expects,
no downstream parsing change needed)::

    {"headline": "...", "summary_text": "...", "source": "...",
     "label": {"currency_affected": "TRY", "crisis_score": 72, "sentiment": "bearish", "event_summary": "..."}}

**Real, disclosed choice: QLoRA, not a full fine-tune**. Full fine-tuning a 7B model needs
~4x its own parameter count in optimizer state (Adam) at fp32 -- tens of GB of VRAM, well past what
a single consumer-class rented GPU (the RTX 3090/4090 tier ``scripts/deploy_custom_llm.py`` already
targets) can hold. QLoRA (Dettmers et al., 2023, "QLoRA: Efficient Finetuning of Quantized LLMs")
loads the frozen base model 4-bit-quantized and trains a small set of low-rank adapter weights on
top -- real, published, standard practice for exactly this budget class, not an invented shortcut.

Usage, on the rented GPU instance, once a real labeled dataset exists::

    python scripts/finetune_political_llm.py --dataset data/labeled_headlines.jsonl --output-dir ./lora-adapter --max-steps 20  # smoke test first
    python scripts/finetune_political_llm.py --dataset data/labeled_headlines.jsonl --output-dir ./lora-adapter  # real run
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
from typing import Any, Final

from core.schemas import PROPRIETARY_LLM_MODEL_NAME

# Real system prompt, verbatim from political_agent.py's own _GUARDIAN_SYSTEM_PROMPT -- the
# fine-tuned model must be trained under the exact same instruction it will be served under in
# production (political_agent.py's real _call_proprietary_llm), or the two silently drift apart.
_SYSTEM_PROMPT: Final[str] = (
    "Tu es un analyste de risque macroéconomique. Analyse ces titres d'actualité récents. "
    "Détermine si l'événement est positif, neutre ou négatif pour la monnaie locale face au "
    "dollar. Renvoie uniquement un objet JSON : {'currency_affected': 'XXX', "
    "'crisis_score': 0-100, 'sentiment': 'bearish/bullish', 'event_summary': '...'}. "
    "Zéro texte autour."
)

_EXPECTED_SCHEMA: Final[str] = (
    '{"headline": str, "summary_text": str, "source": str, '
    '"label": {"currency_affected": str, "crisis_score": int, "sentiment": "bearish"|"bullish", "event_summary": str}}'
)

_DEFAULT_LORA_RANK: Final[int] = 16
_DEFAULT_LORA_ALPHA: Final[int] = 32


def _load_labeled_dataset(path: Path) -> list[dict[str, Any]]:
    """Real, strict loading -- raises loudly on a malformed line rather than silently training on garbage."""
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        if "headline" not in record or "label" not in record:
            raise ValueError(f"{path}:{line_number}: missing 'headline' or 'label' -- expected shape: {_EXPECTED_SCHEMA}")
        records.append(record)
    if not records:
        raise ValueError(f"{path}: no real labeled records found -- nothing to fine-tune on.")
    return records


def _format_training_example(record: dict[str, Any]) -> dict[str, str]:
    """Real Mistral-Instruct chat format -- the same real user-content shape political_agent.py's own
    ``_assess_headline`` constructs (``Titre : ... / Résumé : ... / Source : ...``), so the fine-tuned
    model sees the exact real input shape it will be served in production, not a paraphrase of it."""
    user_content = f"Titre : {record['headline']}\nRésumé : {record.get('summary_text', '')}\nSource : {record.get('source', '?')}"
    completion = json.dumps(record["label"], ensure_ascii=False)
    text = f"<s>[INST] {_SYSTEM_PROMPT}\n\n{user_content} [/INST] {completion}</s>"
    return {"text": text}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, required=True, help="Real labeled JSONL file -- see module docstring for the expected shape.")
    parser.add_argument("--base-model", default=PROPRIETARY_LLM_MODEL_NAME, help=f"Base model to fine-tune (default: {PROPRIETARY_LLM_MODEL_NAME}, matching scripts/deploy_custom_llm.py's own served model).")
    parser.add_argument("--output-dir", type=Path, required=True, help="Where to save the real trained LoRA adapter weights.")
    parser.add_argument("--lora-rank", type=int, default=_DEFAULT_LORA_RANK)
    parser.add_argument("--lora-alpha", type=int, default=_DEFAULT_LORA_ALPHA)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--max-steps", type=int, default=-1, help="Cap total training steps -- set low (e.g. 20) for a real smoke test before a full run.")
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    args = parser.parse_args()

    # Imported here, not at module top level: these are the real, heavy, GPU-only extra deps (see
    # module docstring) -- importing them at module level would make this script fail to even
    # --help on a CPU-only machine that merely wants to read its own documentation.
    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc:
        raise ImportError(
            "Missing a real GPU-only dependency -- see this module's own docstring for the exact "
            "`pip install` line. This script is meant to run on a rented GPU instance "
            "(scripts/deploy_custom_llm.py's own target), not this project's CPU-only dev environment."
        ) from exc

    records = _load_labeled_dataset(args.dataset)
    print(f"{len(records)} real labeled example(s) loaded from {args.dataset}.")

    dataset = Dataset.from_list([_format_training_example(r) for r in records])

    # transformers ships real, installable source but no py.typed marker -- ignore_missing_imports
    # alone still lets mypy follow into that untyped source under strict mode (same real gap
    # core/config.py's own ib_async override docstring already documents for the identical reason).
    bnb_config = BitsAndBytesConfig(  # type: ignore[no-untyped-call]
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)  # type: ignore[no-untyped-call]
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.base_model, quantization_config=bnb_config, device_map="auto")
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=args.lora_rank, lora_alpha=args.lora_alpha, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Mistral's own real attention projection names
    )
    model = get_peft_model(model, lora_config)

    training_args = SFTConfig(
        output_dir=str(args.output_dir), num_train_epochs=args.epochs, max_steps=args.max_steps,
        per_device_train_batch_size=4, gradient_accumulation_steps=4, learning_rate=args.learning_rate,
        logging_steps=1, save_strategy="epoch", bf16=True, dataset_text_field="text", max_seq_length=1024,
        report_to="none",
    )
    trainer = SFTTrainer(model=model, args=training_args, train_dataset=dataset, tokenizer=tokenizer)

    trainer.train()
    trainer.save_model(str(args.output_dir))
    print(f"LoRA adapter réel sauvegardé dans {args.output_dir}.")
    print(
        "Prochaine étape réelle : merge de l'adaptateur dans le modèle de base "
        "(peft.PeftModel.merge_and_unload) puis re-déploiement via scripts/deploy_custom_llm.py "
        "avec ce modèle mergé -- non fait automatiquement ici, décision séparée."
    )


if __name__ == "__main__":
    main()
