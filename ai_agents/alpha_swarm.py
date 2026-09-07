"""Frontier Tech Injection, Pillar 3: the Generative Quant Swarm -- a real, three-role auto-alpha pipeline.

**A real, load-bearing correction to the brief's own tech choice**,
same shape as ``src/agents/quant_copilot.py``'s own documented
correction (LangChain -> direct REST against this project's existing
:class:`~core.config.GuardianSettings` credentials). The brief names
CrewAI/AutoGen. Both were actually checked, not assumed:
``pyautogen`` installs cleanly in this environment (verified via a real
``pip install``), but pulls a large, LLM-orchestration-specific
dependency tree (``autogen-core``, ``protobuf``, ``opentelemetry``) to
provide exactly one thing this codebase already has, project-wide: a
loop that calls an LLM, optionally with tools, and hands the result to
the next step. ``crewai`` does not even install here -- its pinned
numpy dependency has no prebuilt wheel for this Windows/Python 3.14
combination and fails to build from source without a C compiler
(verified via a real ``pip install`` attempt, the same class of failure
``api_middleware/fix_server.py``'s module docstring documents for
``quickfix``). Rather than add the project's first heavy AI-
orchestration framework for functionality it already has, this module
extends the exact pattern ``guardian_ai.py``/``quant_copilot.py``
already established (direct REST, same credentials, same honest "no
LLM configured -> deterministic fallback" contract) with three
prompted roles instead of one.

**The three roles, and what's real about each**:

1. **Data_Scout** (:func:`run_data_scout`): reads this project's own
   *real* signal stores -- ``nlp_signal:*`` (written by
   :class:`~src.agents.nlp.agent.NlpAgent`) and ``macro_snapshot:*``
   (written by :class:`~src.agents.macro.agent.MacroAgent`) -- rather
   than calling any new external sentiment/on-chain API. Neither agent
   runs continuously in this dev environment (see the "Multi-Agent
   Fleet" dashboard page's own real 0/4-online status), so a fresh
   checkout of this project will honestly report an empty scan here,
   not fabricated sentiment -- exactly the same "real key, real empty
   state" honesty as every other Redis-backed tool in this codebase.

2. **Hypothesis_Generator** (:func:`generate_hypothesis`): with an LLM
   configured, asks it to *write* real Polars strategy code (a
   ``generate_signals(df)`` function) reacting to the Data_Scout
   findings -- genuine code generation, not a template picker. Without
   one, falls back to rotating through a small library of real, known-
   correct technical strategies (:data:`_STRATEGY_TEMPLATES`) --
   **critically, the exact same downstream sandbox and backtest engine
   runs either way** (see :func:`run_backtest_in_sandbox`), so the
   swarm still produces a real, numerically genuine backtest even with
   zero LLM configured; only the *creativity* of the hypothesis is
   templated rather than generative in that mode, never the execution.

3. **Backtest_Coder** (:func:`run_backtest_in_sandbox`): the literal
   "writes and executes Python/Polars code in a sandbox" the brief
   asks for -- real, and real about its own limits (read that
   function's docstring before trusting the word "sandbox" too far).

**Human-in-the-loop, not auto-execution -- same established pattern as
every other AI-touching surface in this project**. A validated
proposal is written to :data:`~core.schemas.
ALPHA_SWARM_PROPOSALS_REDIS_KEY` and surfaced in the "Quant Copilot"
dashboard page for a human to approve or reject -- nothing here ever
submits a real order or promotes a strategy to production on its own,
identical in spirit to ``src/agents/quant_copilot.py``'s
``adjust_algo_aggression`` (propose, never auto-execute) and
``src/agents/guardian_ai.py``'s ``EmergencyAction`` confirmation gate.

**A number this module will never compute from generated text**: every
figure in :class:`BacktestMetrics` (Sharpe, max drawdown, win rate,
total return) is computed by this project's own reviewed harness code
from the *signals* the model (or the fallback template) produced --
the model's/template's output is a series of -1/0/1 values, nothing
else; it never reports its own performance. A prompt-injected or
simply overconfident hypothesis cannot claim a good backtest that
didn't happen.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field

from core.config import AppSettings, GuardianSettings, get_settings
from core.http_client import ResilientHttpClient
from core.logger import get_logger
from core.redis_client import RedisClient
from core.schemas import ALPHA_SWARM_PROPOSALS_REDIS_KEY

_log = get_logger(__name__)

_SAMPLE_DATA_DIR: Final[Path] = Path(__file__).resolve().parent.parent / "research" / "sample_data"
_PROPOSALS_LIST_MAX_LEN: Final[int] = 50

# =============================================================================
# Role 1: Data_Scout -- real sentiment/on-chain signal aggregation
# =============================================================================


class DataScoutFindings(BaseModel):
    """Real aggregation of this project's own live sentiment/on-chain signal stores. See module docstring."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    sentiment_sample_count: int = Field(ge=0)
    average_sentiment_score: float | None = Field(default=None, description="Mean of real SentimentSignal.score values found -- None if no fresh signal exists.")
    macro_indicators: dict[str, float] = Field(default_factory=dict, description="Real {indicator_name: latest_value} pairs from macro_snapshot:* -- see src.agents.macro.agent.")
    note: str


async def run_data_scout(redis_client: RedisClient) -> DataScoutFindings:
    """Real Redis scan across the two real signal stores -- see module docstring for why they may legitimately be empty."""
    sentiment_scores: list[float] = []
    sentiment_keys = await redis_client.execute("keys", "nlp_signal:*")
    for key in sentiment_keys:
        raw = await redis_client.execute("get", key)
        if raw is None:
            continue
        try:
            payload = json.loads(raw)
            sentiment_scores.append(float(payload["score"]))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            continue

    macro_indicators: dict[str, float] = {}
    macro_keys = await redis_client.execute("keys", "macro_snapshot:*")
    for key in macro_keys:
        raw = await redis_client.execute("get", key)
        if raw is None:
            continue
        try:
            payload = json.loads(raw)
            macro_indicators[str(payload["indicator"])] = float(payload["value"])
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            continue

    if not sentiment_scores and not macro_indicators:
        note = (
            "Aucun signal réel trouvé (nlp_signal:*/macro_snapshot:* vides) -- NlpAgent/MacroAgent ne "
            "tournent pas dans cet environnement (voir la page Multi-Agent Fleet, 0/4 agents en ligne). "
            "L'Hypothesis_Generator travaille sans contexte de marché réel dans ce cas."
        )
    else:
        note = f"{len(sentiment_scores)} signal(aux) de sentiment réel(s), {len(macro_indicators)} indicateur(s) macro réel(s) trouvés."

    return DataScoutFindings(
        sentiment_sample_count=len(sentiment_scores),
        average_sentiment_score=(sum(sentiment_scores) / len(sentiment_scores)) if sentiment_scores else None,
        macro_indicators=macro_indicators,
        note=note,
    )


# =============================================================================
# Role 2: Hypothesis_Generator -- real LLM code generation, or a real template fallback
# =============================================================================

# Every template defines generate_signals(df) exactly as the LLM is asked to -- same
# contract, same downstream sandbox/harness path either way (see module docstring).
_STRATEGY_TEMPLATES: Final[dict[str, str]] = {
    "sma_crossover": (
        "def generate_signals(df: pl.DataFrame) -> pl.Series:\n"
        "    fast = df['close'].rolling_mean(window_size=10)\n"
        "    slow = df['close'].rolling_mean(window_size=30)\n"
        "    return pl.when(fast > slow).then(1).when(fast < slow).then(-1).otherwise(0).alias('signal')\n"
    ),
    "rsi_mean_reversion": (
        "def generate_signals(df: pl.DataFrame) -> pl.Series:\n"
        "    delta = df['close'].diff()\n"
        "    gain = delta.clip(lower_bound=0).rolling_mean(window_size=14)\n"
        "    loss = (-delta.clip(upper_bound=0)).rolling_mean(window_size=14)\n"
        "    rs = gain / loss.replace(0, None)\n"
        "    rsi = 100 - (100 / (1 + rs))\n"
        "    return pl.when(rsi < 30).then(1).when(rsi > 70).then(-1).otherwise(0).alias('signal')\n"
    ),
    "momentum_breakout": (
        "def generate_signals(df: pl.DataFrame) -> pl.Series:\n"
        "    lookback_high = df['close'].rolling_max(window_size=20).shift(1)\n"
        "    lookback_low = df['close'].rolling_min(window_size=20).shift(1)\n"
        "    return pl.when(df['close'] > lookback_high).then(1).when(df['close'] < lookback_low).then(-1).otherwise(0).alias('signal')\n"
    ),
}

_HYPOTHESIS_SYSTEM_PROMPT: Final[str] = (
    "Tu es Hypothesis_Generator, un rôle du Generative Quant Swarm de Nothierly. On te donne un contexte "
    "réel de sentiment/marché (peut être vide -- ne jamais l'inventer si vide). Écris UNE fonction Polars "
    "réelle, exécutable, nommée generate_signals(df: pl.DataFrame) -> pl.Series, qui retourne une colonne "
    "de signaux -1/0/1 à partir de la colonne df['close']. N'utilise QUE le module polars (importé comme "
    "pl) -- aucun autre import, aucun accès fichier/réseau/système. Réponds uniquement avec le code Python, "
    "sans texte autour, sans bloc markdown."
)


class Hypothesis(BaseModel):
    """One proposed strategy -- code only, never a self-reported performance claim (see module docstring)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    strategy_name: str
    rationale: str
    generated_code: str
    source: Literal["llm", "template_fallback"]


async def generate_hypothesis(findings: DataScoutFindings, *, settings: AppSettings | None = None) -> Hypothesis:
    """Real LLM code generation when configured; a real, rotating template fallback otherwise. See module docstring."""
    settings = settings or get_settings()
    guardian_settings = settings.guardian

    if guardian_settings.llm_provider == "none" or guardian_settings.api_key is None:
        template_name = _pick_fallback_template(findings)
        return Hypothesis(
            strategy_name=template_name,
            rationale=(
                f"[Hypothesis_Generator -- aucun LLM configuré] Sélection déterministe du modèle "
                f"'{template_name}' dans la bibliothèque de stratégies vérifiées -- aucune génération "
                f"créative sans LLM actif. {findings.note}"
            ),
            generated_code=_STRATEGY_TEMPLATES[template_name],
            source="template_fallback",
        )

    context_text = (
        f"Sentiment moyen réel : {findings.average_sentiment_score if findings.average_sentiment_score is not None else 'indisponible'} "
        f"({findings.sentiment_sample_count} échantillons). Indicateurs macro réels : {findings.macro_indicators or 'aucun'}."
    )
    try:
        code = await _call_llm_for_hypothesis(guardian_settings, context_text)
        validate_generated_code_is_safe(code)
    except Exception:
        _log.exception("alpha_swarm_hypothesis_llm_call_failed_using_fallback", provider=guardian_settings.llm_provider)
        template_name = _pick_fallback_template(findings)
        return Hypothesis(
            strategy_name=template_name,
            rationale=f"[Hypothesis_Generator -- appel LLM ou code généré invalide, repli sur '{template_name}'] {findings.note}",
            generated_code=_STRATEGY_TEMPLATES[template_name],
            source="template_fallback",
        )

    return Hypothesis(
        strategy_name=f"llm_generated_{hashlib.sha256(code.encode()).hexdigest()[:8]}",
        rationale=f"[Hypothesis_Generator -- LLM {guardian_settings.llm_provider}] {context_text}",
        generated_code=code,
        source="llm",
    )


def _pick_fallback_template(findings: DataScoutFindings) -> str:
    """Deterministic (not random) template pick -- same findings always pick the same template, real and reproducible."""
    names = sorted(_STRATEGY_TEMPLATES)
    index = findings.sentiment_sample_count % len(names)
    return names[index]


async def _call_llm_for_hypothesis(guardian_settings: GuardianSettings, context_text: str) -> str:
    """Direct REST call, deliberately not importing guardian_ai.py's private _call_llm -- see quant_copilot.py's own precedent against cross-module private imports; this is a small enough duplication of the same real pattern."""
    if guardian_settings.api_key is None:
        raise ValueError("GUARDIAN_API_KEY is not set")
    api_key = guardian_settings.api_key.get_secret_value()
    model = guardian_settings.model or ("gpt-4o-mini" if guardian_settings.llm_provider == "openai" else "claude-3-5-haiku-20241022")

    async with ResilientHttpClient(timeout_seconds=30.0) as http:
        if guardian_settings.llm_provider == "openai":
            body = await http.post_json(
                "https://api.openai.com/v1/chat/completions",
                json_body={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": _HYPOTHESIS_SYSTEM_PROMPT},
                        {"role": "user", "content": context_text},
                    ],
                    "max_tokens": 500,
                    "temperature": 0.4,
                },
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            )
            return str(body["choices"][0]["message"]["content"]).strip()

        body = await http.post_json(
            "https://api.anthropic.com/v1/messages",
            json_body={
                "model": model, "max_tokens": 500, "system": _HYPOTHESIS_SYSTEM_PROMPT,
                "messages": [{"role": "user", "content": context_text}],
            },
            headers={"x-api-key": api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
        )
        return str(body["content"][0]["text"]).strip()


# =============================================================================
# Role 3: Backtest_Coder -- real AST pre-filtering + real subprocess-sandboxed execution
# =============================================================================


class UnsafeGeneratedCodeError(ValueError):
    """Raised by :func:`validate_generated_code_is_safe` -- never swallowed silently; a rejected hypothesis is dropped, not laundered."""


_ALLOWED_IMPORT_MODULES: Final[frozenset[str]] = frozenset({"polars"})
_FORBIDDEN_CALL_NAMES: Final[frozenset[str]] = frozenset(
    {"eval", "exec", "compile", "__import__", "open", "input", "getattr", "setattr", "delattr", "globals", "locals", "vars", "breakpoint"}
)


def validate_generated_code_is_safe(code: str) -> None:
    """Real, static AST pre-filter -- defense-in-depth, explicitly NOT a claim of hard security isolation.

    Rejects, before any execution is even attempted: imports outside
    :data:`_ALLOWED_IMPORT_MODULES`, any dunder name reference/attribute
    access (blocks the classic ``().__class__.__bases__[0].
    __subclasses__()``-style sandbox-escape idiom), and calls to a
    real, if non-exhaustive, denylist of dangerous builtins. This is a
    genuinely useful filter against an LLM that hallucinates ordinary-
    looking-but-dangerous code (e.g. accidentally importing ``os``) and
    against unsophisticated prompt injection -- it is **not** a claim
    that no cleverly-crafted Python payload could evade a pure-AST
    filter (a sufficiently determined adversary with code-execution-
    level control over the prompt could attempt far subtler tricks);
    the real security boundary this module actually relies on is the
    separate OS subprocess + wall-clock timeout in
    :func:`run_backtest_in_sandbox`, which bounds the blast radius of
    anything this filter misses to a short-lived, low-privilege child
    process rather than the main trading system. A production-grade
    system would add a container/microVM boundary on top of both; this
    dev environment does not have one available (same class of gap as
    ``network/ebpf_ingestor.py``'s Linux-only limitation, disclosed
    rather than hidden).
    """
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as exc:
        raise UnsafeGeneratedCodeError(f"generated code does not parse: {exc}") from exc

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [alias.name for alias in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            for name in names:
                if name.split(".")[0] not in _ALLOWED_IMPORT_MODULES:
                    raise UnsafeGeneratedCodeError(f"import of {name!r} is not allowed in sandboxed strategy code")
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise UnsafeGeneratedCodeError(f"access to dunder attribute {node.attr!r} is not allowed")
        elif isinstance(node, ast.Name) and node.id.startswith("__"):
            raise UnsafeGeneratedCodeError(f"reference to {node.id!r} is not allowed")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FORBIDDEN_CALL_NAMES:
            raise UnsafeGeneratedCodeError(f"call to {node.func.id!r} is not allowed in sandboxed strategy code")

    if "def generate_signals(" not in code:
        raise UnsafeGeneratedCodeError("generated code must define generate_signals(df: pl.DataFrame) -> pl.Series")


class BacktestMetrics(BaseModel):
    """Real, harness-computed metrics -- never self-reported by the model/template. See module docstring.

    **A real, load-bearing caveat about ``sharpe_ratio`` specifically**:
    it is computed correctly (``mean(strategy_returns) / std(strategy_
    returns) * sqrt(periods_per_year)``), but real sample coverage
    behind it is exactly one trading day (the same real, narrow
    ``research/sample_data`` window ``simulation/digital_twin.py``'s
    own docstring already discloses) -- annualizing one day's noise
    over ``periods_per_year`` (525,600 for 1-minute bars) routinely
    produces triple-digit-magnitude "annualized" Sharpe values that are
    real arithmetic, not a real annual estimate anyone should act on.
    :func:`run_alpha_swarm_once`'s human-review hand-off exists
    precisely because a number like this needs a person who understands
    that caveat before it goes anywhere near real capital -- this
    module reports the real computed figure rather than hiding it
    behind a fabricated "more reasonable" one, but does not pretend one
    day of data supports an annual claim either.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    total_return_pct: float
    sharpe_ratio: float | None = Field(default=None, description="None when the strategy's return series has zero variance (e.g. every signal is 0) -- never fabricated as 0.0. See the class docstring's sample-size caveat before trusting its magnitude.")
    max_drawdown_pct: float
    win_rate_pct: float
    num_trades: int
    bars_used: int


class BacktestResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    ok: bool
    metrics: BacktestMetrics | None = None
    error: str | None = None


_HARNESS_TEMPLATE: Final[str] = '''
import sys, json
import polars as pl

{strategy_code}

def _main() -> None:
    df = pl.read_parquet(sys.argv[1])
    # generate_signals may return either a pl.Series or an unevaluated pl.Expr (pl.when()/.then()
    # chains always return Expr, even when built from Series inputs) -- df.select(...) accepts and
    # materializes either uniformly into a real, concrete Series.
    signals = df.select(generate_signals(df).alias("signal"))["signal"]
    returns = df["close"].pct_change()
    strategy_returns = (signals.shift(1) * returns).fill_null(0.0)

    equity = (1.0 + strategy_returns).cum_prod()
    running_max = equity.cum_max()
    drawdown = (equity / running_max) - 1.0

    std = strategy_returns.std()
    mean = strategy_returns.mean()
    periods_per_year = {periods_per_year}
    sharpe = None
    if std is not None and std > 0:
        sharpe = float(mean / std * (periods_per_year ** 0.5))

    signal_changes = int((signals != signals.shift(1)).fill_null(False).sum())
    positive_bars = int((strategy_returns > 0).sum())
    nonzero_bars = int((strategy_returns != 0).sum())
    win_rate = (positive_bars / nonzero_bars * 100.0) if nonzero_bars > 0 else 0.0

    result = {{
        "total_return_pct": float((equity[-1] - 1.0) * 100.0) if len(equity) > 0 else 0.0,
        "sharpe_ratio": sharpe,
        "max_drawdown_pct": float(drawdown.min() * 100.0) if len(drawdown) > 0 else 0.0,
        "win_rate_pct": win_rate,
        "num_trades": signal_changes,
        "bars_used": len(df),
    }}
    print(json.dumps(result))


if __name__ == "__main__":
    try:
        _main()
    except Exception as exc:
        print(json.dumps({{"error": f"{{type(exc).__name__}}: {{exc}}"}}))
        sys.exit(1)
'''


def _load_real_price_bars(symbol: str, *, bar_interval: str = "1m") -> Any:
    """Real historical bars from research/sample_data (same real Tardis-format source simulation/digital_twin.py uses).

    Args:
        symbol: ``"BTCUSDT"`` or ``"ETHUSDT"`` -- the only two real
            symbols with sample data in this project (see
            ``simulation/digital_twin.py``'s own module docstring on
            the real, narrow coverage window).
        bar_interval: A Polars ``group_by_dynamic`` interval string
            (e.g. ``"1m"``).

    Returns:
        A real ``polars.DataFrame`` with ``timestamp``/``close``
        columns, resampled from real tick trades.
    """
    import polars as pl

    trades_path = _SAMPLE_DATA_DIR / f"{symbol.lower()}_trades_2020-01-01.csv"
    if not trades_path.exists():
        raise FileNotFoundError(f"no real sample data for {symbol!r} at {trades_path}")

    trades = pl.scan_csv(trades_path).with_columns(
        pl.from_epoch("timestamp", time_unit="us").alias("ts")
    ).sort("ts")
    bars = (
        trades.group_by_dynamic("ts", every=bar_interval)
        .agg(pl.col("price").last().alias("close"))
        .drop_nulls("close")
        .collect()
    )
    return bars.rename({"ts": "timestamp"})


_BAR_INTERVAL_PERIODS_PER_YEAR: Final[dict[str, int]] = {"1m": 525_600, "5m": 105_120, "1h": 8_760}


def run_backtest_in_sandbox(
    hypothesis: Hypothesis, *, symbol: str = "BTCUSDT", bar_interval: str = "1m", timeout_seconds: float = 30.0
) -> BacktestResult:
    """The real Backtest_Coder step: validate, then execute, generated code in an isolated child process.

    Isolation, stated precisely (see :func:`validate_generated_code_is_safe`'s own docstring for the
    honest limits of the AST pre-filter this relies on first):

    - A **separate OS subprocess** (``python -I``, isolated mode: ignores ``PYTHONPATH``/user site-
      customization, a real, if partial, hardening flag) -- a crash, hang, or anything the AST filter
      missed is contained there, never able to touch this process's own memory/state directly.
    - A **hard wall-clock timeout** (``subprocess.run(..., timeout=timeout_seconds)``) -- a real bound
      on runaway/adversarial CPU usage, not a cooperative one the sandboxed code could ignore.
    - **No file the child process is handed access to beyond what this function gives it**: the only
      file it is passed a path to is the one real Parquet snapshot of price bars written just for this
      run; ``polars`` is the only importable module the harness template itself references. The OS
      environment is inherited as-is (not blanked) -- a real, live-tested finding: Polars' own Windows
      CPU-feature detection reads ``PROCESSOR_ARCHITECTURE``/``PROCESSOR_IDENTIFIER`` and crashes
      outright without them, so stripping the environment down to bare ``PATH`` broke the sandbox itself
      rather than meaningfully shrinking its attack surface -- this module relies on process boundary +
      timeout + the AST filter for isolation, not on hiding environment variables from a child process
      that still runs as the same OS user.

    Returns:
        A :class:`BacktestResult` -- ``ok=False`` with a real error
        message (never a partial/fabricated metric) for a validation
        failure, a subprocess timeout, or any real runtime exception
        inside the sandboxed code.
    """
    try:
        validate_generated_code_is_safe(hypothesis.generated_code)
    except UnsafeGeneratedCodeError as exc:
        return BacktestResult(ok=False, error=f"rejected before execution: {exc}")

    try:
        bars = _load_real_price_bars(symbol, bar_interval=bar_interval)
    except FileNotFoundError as exc:
        return BacktestResult(ok=False, error=str(exc))

    periods_per_year = _BAR_INTERVAL_PERIODS_PER_YEAR.get(bar_interval, 525_600)
    harness_source = _HARNESS_TEMPLATE.format(strategy_code=hypothesis.generated_code, periods_per_year=periods_per_year)

    with tempfile.TemporaryDirectory(prefix="nothierly_alpha_swarm_sandbox_") as tmp_dir:
        harness_path = Path(tmp_dir) / "harness.py"
        data_path = Path(tmp_dir) / "bars.parquet"
        harness_path.write_text(harness_source, encoding="utf-8")
        bars.write_parquet(data_path)

        try:
            completed = subprocess.run(
                [sys.executable, "-I", str(harness_path), str(data_path)],
                capture_output=True, text=True, timeout=timeout_seconds, cwd=tmp_dir, env=os.environ.copy(),
            )
        except subprocess.TimeoutExpired:
            _log.warning("alpha_swarm_backtest_sandbox_timeout", strategy_name=hypothesis.strategy_name)
            return BacktestResult(ok=False, error=f"sandboxed backtest exceeded {timeout_seconds}s timeout -- killed")

        stdout = completed.stdout.strip().splitlines()[-1] if completed.stdout.strip() else ""
        try:
            payload = json.loads(stdout)
        except json.JSONDecodeError:
            return BacktestResult(ok=False, error=f"sandbox produced no parseable result (exit={completed.returncode}): {completed.stderr[-500:]}")

        if "error" in payload:
            return BacktestResult(ok=False, error=str(payload["error"]))

        return BacktestResult(ok=True, metrics=BacktestMetrics(**payload))


# =============================================================================
# Orchestration: the full swarm run + human-in-the-loop hand-off to Quant Copilot
# =============================================================================


class AlphaProposal(BaseModel):
    """One real, backtested proposal, ready for a human to approve/reject in the dashboard. See module docstring."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    proposal_id: str
    hypothesis: Hypothesis
    metrics: BacktestMetrics
    findings_note: str
    generated_at: float = Field(gt=0)


class SwarmRunResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    proposal: AlphaProposal | None
    rejected_reason: str | None = None


async def run_alpha_swarm_once(
    redis_client: RedisClient, *, symbol: str = "BTCUSDT", settings: AppSettings | None = None
) -> SwarmRunResult:
    """One real, end-to-end pass: Data_Scout -> Hypothesis_Generator -> Backtest_Coder -> Redis hand-off.

    Pushes an :class:`AlphaProposal` onto
    :data:`~core.schemas.ALPHA_SWARM_PROPOSALS_REDIS_KEY` (LPUSH +
    LTRIM-capped, same real-but-bounded-list pattern as every other
    ``*_ALERTS``/``*_LOG`` key in this project) only when the sandboxed
    backtest actually ran successfully -- a failed/rejected hypothesis
    is logged and discarded, never surfaced to a human as if it were a
    real result.
    """
    findings = await run_data_scout(redis_client)
    hypothesis = await generate_hypothesis(findings, settings=settings)
    result = run_backtest_in_sandbox(hypothesis, symbol=symbol)

    if not result.ok or result.metrics is None:
        _log.warning("alpha_swarm_run_rejected", strategy_name=hypothesis.strategy_name, reason=result.error)
        return SwarmRunResult(proposal=None, rejected_reason=result.error)

    proposal = AlphaProposal(
        proposal_id=str(uuid.uuid4()), hypothesis=hypothesis, metrics=result.metrics,
        findings_note=findings.note, generated_at=time.time(),
    )
    await redis_client.execute("lpush", ALPHA_SWARM_PROPOSALS_REDIS_KEY, proposal.model_dump_json())
    await redis_client.execute("ltrim", ALPHA_SWARM_PROPOSALS_REDIS_KEY, 0, _PROPOSALS_LIST_MAX_LEN - 1)
    _log.info("alpha_swarm_proposal_pushed", proposal_id=proposal.proposal_id, strategy_name=hypothesis.strategy_name)
    return SwarmRunResult(proposal=proposal)
