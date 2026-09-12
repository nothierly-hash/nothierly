"""Global emergency Kill Switch and a timeout/circuit-breaker guard for
money-committing API calls.

This complements, it does not replace, :class:`~src.agents.supervisor.
agent.SupervisorAgent`'s own local Kill Switch (Redis-latency/spread-
anomaly/consecutive-error conditions, scoped to that one agent's decision
loop). This module is a *system-wide* panic flag any agent, anywhere, can
trip -- a compromised credential, a wildly anomalous price, an operator's
manual intervention -- and every money-*creating* call (placing an order,
renting a GPU) checks before acting. Cleanup/cancellation calls (cancelling
orders, destroying a rented instance) deliberately never check it: blocking
those during an emergency would strand exposure instead of tearing it down,
which is the opposite of what a kill switch is for. The same principle now
has a second, real instance -- closing a position that already exists, not
just cancelling a resting order -- see :meth:`~src.agents.market_data.
exchange_client.AsyncExchangeClient.create_market_order_bypassing_kill_switch`,
used by exactly one real, disclosed, human-triggered caller
(:class:`~src.agents.alpha_engine.agent.AlphaMarketMakerAgent`'s Emergency
Liquidate), not a general-purpose escape hatch.

Reuses the project's single :class:`~core.redis_client.RedisClient`
connection pool (never a second, independent Redis connection): Redis
settings (host/port/password/TLS) stay centralized in
:class:`~core.config.RedisSettings`, and the pool's own reconnect backoff
(see that module's docstring on why redis-py's internal retry is disabled)
is not duplicated here.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from functools import wraps
from typing import Any, Final, ParamSpec, TypeVar

from core.logger import get_logger
from core.message_integrity import sign, verify
from core.redis_client import RedisClient, RedisUnavailableError

_log = get_logger(__name__)

SYSTEM_STATUS_KEY: Final[str] = "SYSTEM_STATUS"
# Companion key holding an HMAC-SHA256 of f"{SYSTEM_STATUS_KEY value}:{SYSTEM_STATUS_EPOCH_KEY value}"
# (see core/message_integrity.py). Real gap this closes, found by tests/red_team_hacker.py's Attack 2:
# SYSTEM_STATUS is a plain Redis key with no source authentication -- by default (no REDIS_PASSWORD
# locally) or even with a shared password, nothing distinguishes a dashboard operator's real
# reset_kill_switch() call from anyone else's raw `redis-cli SET SYSTEM_STATUS SAFE`. is_system_safe()
# now fails closed (blocks trading) whenever the value present does not carry a matching signature,
# rather than trusting the value alone.
SYSTEM_STATUS_SIG_KEY: Final[str] = "SYSTEM_STATUS:sig"
# Monotonic counter (Redis INCR), bumped on every real trip_kill_switch/reset_kill_switch call, bound
# into the signed payload above. Real gap this closes, found by tests/red_team_hacker.py's Attack 2b
# (a real, confirmed limitation of the signature scheme above, discovered by red-teaming this session's
# own fix): sign()/verify() alone authenticate WHO could have written a value, never WHEN -- an attacker
# who merely OBSERVES one real (SYSTEM_STATUS, SYSTEM_STATUS_SIG_KEY) pair once (e.g. read-only Redis
# access at any point a genuine reset happened) could replay it verbatim, indefinitely, including to
# silently un-trip a LATER, unrelated, real panic. Binding the epoch means that pair only verifies for
# the ONE state transition it was actually signed for -- any subsequent real trip_kill_switch/
# reset_kill_switch call (which always bumps the epoch) makes every earlier signed pair permanently
# stale, closed or not.
#
# **Honest residual limit, disclosed, not hidden**: this alone does not make replay
# cryptographically impossible -- an attacker with the same unrestricted Redis WRITE access this
# whole class of attack already assumes (see Attack 1/2's own threat model) could roll this counter
# back down with a raw SET alongside the old signature, since Redis has no ACL in this codebase
# distinguishing "may INCR" from "may SET". _MIN_ACCEPTED_EPOCH below (a per-process, in-memory,
# monotonically-increasing high-water mark) closes that specific residual: it independently refuses
# any epoch this process has already seen a HIGHER value for, regardless of what Redis currently
# reports -- Redis can be rolled back, this process's own memory cannot be, from the outside, without
# also compromising the process itself (a strictly bigger compromise than "can write to Redis"). The
# one remaining honest gap: a brand-new process (freshly started, never having observed the real
# epoch yet) has no high-water mark to check against on its very first call, and trusts whatever
# Redis reports at that moment -- the same fundamental limit every replay defense built on a single,
# fully-attacker-writable store eventually reaches. Closing that too needs an external, non-Redis
# source of truth (e.g. the WORM audit ledger, already tamper-evident) -- a real, larger architecture
# change, not a signature detail, and out of scope for this patch.
SYSTEM_STATUS_EPOCH_KEY: Final[str] = "SYSTEM_STATUS:epoch"
STATUS_SAFE: Final[str] = "SAFE"
STATUS_PANIC: Final[str] = "PANIC"

# Per-process high-water mark -- see SYSTEM_STATUS_EPOCH_KEY's own docstring above. Module-level by
# design: every caller within one process shares the same defense, the same way RedisClient._instance
# is one process-wide singleton. Resets only on process restart, never manually -- an intentional
# reset would itself reopen exactly the replay window this exists to close.
_min_accepted_epoch: int = 0

# Upper bound on how long a Kill Switch *check* itself may take, regardless
# of whichever agent happened to initialize the shared RedisClient
# singleton first (its own bounded/unbounded reconnect-attempt policy is
# fixed at first construction -- see RedisClient.get_instance's docstring
# -- so this module cannot assume it is bounded). A trading decision must
# never stall indefinitely on the safety check meant to protect it.
_CHECK_TIMEOUT_SECONDS: Final[float] = 3.0
_TRIP_TIMEOUT_SECONDS: Final[float] = 3.0

_P = ParamSpec("_P")
_R = TypeVar("_R")


class KillSwitchActiveError(PermissionError):
    """Raised when a money-committing call is refused because SYSTEM_STATUS is PANIC."""


class FinancialCallAbortedError(RuntimeError):
    """Raised when a protected financial call exhausts its attempt budget.

    Raising this also trips the global Kill Switch (see
    :func:`safe_financial_call`): an API this unreliable is treated as a
    systemic failure serious enough to halt new trading, not a one-off to
    shrug off and continue past.
    """


async def is_system_safe() -> bool:
    """Check Redis's ``SYSTEM_STATUS`` flag. Fails closed on any doubt.

    Returns:
        ``False`` (block) if Redis is unreachable, the check times out,
        ``SYSTEM_STATUS`` is ``"PANIC"``, a value is present but its
        companion :data:`SYSTEM_STATUS_SIG_KEY` is missing/does not match
        the value+epoch it claims to authenticate, or the epoch itself is
        older than one this process has already observed (a replay -- see
        :data:`SYSTEM_STATUS_EPOCH_KEY`'s own docstring for what this does
        and does not defend against). ``True`` (safe) only if the key is
        absent, or present, signed, fresh, and equal to ``"SAFE"``.
    """
    global _min_accepted_epoch
    try:
        client = await asyncio.wait_for(RedisClient.get_instance(), timeout=_CHECK_TIMEOUT_SECONDS)
        status, signature, raw_epoch = await asyncio.wait_for(
            asyncio.gather(
                client.execute("get", SYSTEM_STATUS_KEY),
                client.execute("get", SYSTEM_STATUS_SIG_KEY),
                client.execute("get", SYSTEM_STATUS_EPOCH_KEY),
            ),
            timeout=_CHECK_TIMEOUT_SECONDS,
        )
    except (RedisUnavailableError, asyncio.TimeoutError) as exc:
        _log.error(
            "kill_switch_check_failed_failing_closed",
            error_type=type(exc).__name__,
            error=str(exc),
        )
        return False

    if status is None:
        return True

    epoch = int(raw_epoch) if raw_epoch is not None else 0
    if epoch < _min_accepted_epoch:
        _log.critical(
            "kill_switch_epoch_rollback_detected_failing_closed",
            status=status, observed_epoch=epoch, min_accepted_epoch=_min_accepted_epoch,
        )
        return False

    if not verify(f"{status}:{epoch}", signature):
        _log.critical(
            "kill_switch_signature_invalid_failing_closed",
            status=status, epoch=epoch,
        )
        return False

    _min_accepted_epoch = max(_min_accepted_epoch, epoch)
    return bool(status == STATUS_SAFE)


async def _set_system_status(status: str) -> None:
    """Shared real write path for :func:`trip_kill_switch`/:func:`reset_kill_switch`.

    Every real state transition bumps :data:`SYSTEM_STATUS_EPOCH_KEY` via
    an atomic Redis ``INCR`` *first*, then signs ``f"{status}:{new_epoch}"``
    -- binding every signature to the one specific transition it was
    issued for, never reusable for another.
    """
    client = await RedisClient.get_instance()
    new_epoch = await client.execute("incr", SYSTEM_STATUS_EPOCH_KEY)
    await asyncio.gather(
        client.execute("set", SYSTEM_STATUS_KEY, status),
        client.execute("set", SYSTEM_STATUS_SIG_KEY, sign(f"{status}:{new_epoch}")),
    )


async def trip_kill_switch(reason: str) -> None:
    """Set ``SYSTEM_STATUS=PANIC`` (signed, epoch-bound) in Redis. Any agent may call this at any time.

    Best-effort: if Redis itself is unreachable, the flag can't be
    persisted -- but :func:`is_system_safe` already fails closed in that
    same situation, so trading is blocked either way.

    Args:
        reason: Human-readable cause, logged at CRITICAL and useful for
            whoever investigates the panic afterward.
    """
    _log.critical("GLOBAL_KILL_SWITCH_TRIPPED", reason=reason)
    try:
        await asyncio.wait_for(_set_system_status(STATUS_PANIC), timeout=_TRIP_TIMEOUT_SECONDS)
    except (RedisUnavailableError, asyncio.TimeoutError) as exc:
        _log.error(
            "kill_switch_trip_could_not_persist_to_redis",
            error_type=type(exc).__name__,
            error=str(exc),
        )


async def reset_kill_switch() -> None:
    """Set ``SYSTEM_STATUS=SAFE`` (signed, epoch-bound) in Redis. An explicit, deliberate operator action.

    No code path in this system calls this automatically -- recovering
    from a global panic is a decision for a human to make on purpose,
    never something an agent decides for itself.
    """
    _log.warning("GLOBAL_KILL_SWITCH_RESET")
    await _set_system_status(STATUS_SAFE)


def requires_system_active(func: Callable[_P, Awaitable[_R]]) -> Callable[_P, Awaitable[_R]]:
    """Decorator: block an async call while the global Kill Switch is tripped.

    Apply this **only** to calls that create new financial exposure
    (placing an order, renting a GPU instance) -- never to a
    cancellation/cleanup call (cancelling orders, destroying a rented
    instance). Blocking those during a panic would strand live exposure
    (an open order, a still-billing GPU instance) instead of tearing it
    down, defeating the point of the switch.

    Raises:
        KillSwitchActiveError: If ``SYSTEM_STATUS`` is ``PANIC`` (or
            unreachable -- see :func:`is_system_safe`) at call time.
    """

    @wraps(func)
    async def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        if not await is_system_safe():
            _log.critical(
                "kill_switch_blocked_call",
                function=f"{func.__module__}.{func.__qualname__}",
            )
            raise KillSwitchActiveError(
                f"Global Kill Switch is active: refusing to call {func.__qualname__}."
            )
        return await func(*args, **kwargs)

    return wrapper


async def safe_financial_call(
    api_function: Callable[..., Awaitable[_R]],
    *args: Any,
    max_retries: int = 3,
    timeout_seconds: float = 10.0,
    retry_on_failure: bool = True,
    **kwargs: Any,
) -> _R:
    """Timeout-bound, optionally retrying wrapper for a financial API call.

    Retry-safety note -- read before setting ``retry_on_failure=True``:
    retrying is only safe for calls that have no side effect when
    repeated (checking a balance, a price, an instance's status). A
    request that times out client-side may still have *succeeded*
    server-side; retrying an order-placement or instance-rental call
    risks a second, duplicate charge on top of one that already went
    through. This project's own
    :meth:`~src.agents.supervisor.broker_client.OandaClient.
    create_market_order` and
    :meth:`~src.mlops.vast_client.VastAIClient.rent_instance` already
    make exactly one attempt for exactly this reason -- pass
    ``retry_on_failure=False`` (the single-attempt, timeout-guarded mode)
    when wrapping anything that spends money on each call; reserve
    ``True`` for read-only lookups.

    On exhausting every attempt, this trips the global Kill Switch (see
    :func:`trip_kill_switch`) and raises :class:`FinancialCallAbortedError`.

    Args:
        api_function: An async callable to invoke.
        *args: Positional arguments forwarded to ``api_function``.
        max_retries: Attempt budget when ``retry_on_failure`` is ``True``;
            ignored (treated as ``1``) otherwise.
        timeout_seconds: Per-attempt timeout.
        retry_on_failure: Whether a failed/timed-out attempt may be
            retried at all. See the retry-safety note above.
        **kwargs: Keyword arguments forwarded to ``api_function``.

    Returns:
        Whatever ``api_function`` returns, on success.

    Raises:
        FinancialCallAbortedError: If every attempt fails. The global
            Kill Switch is tripped before this is raised.
    """
    function_name = getattr(api_function, "__name__", repr(api_function))
    attempts = max_retries if retry_on_failure else 1
    last_error: BaseException | None = None

    for attempt in range(1, attempts + 1):
        try:
            async with asyncio.timeout(timeout_seconds):
                return await api_function(*args, **kwargs)
        except KillSwitchActiveError:
            # api_function was itself @requires_system_active-guarded and
            # refused because the switch is already tripped -- not a
            # failure to retry or re-trip (it's already exactly the
            # state trip_kill_switch would put it in), and not something
            # a future attempt could ever succeed past. Propagate the
            # specific, meaningful exception as-is instead of letting it
            # fall into the generic "failed, maybe retry, then trip"
            # path below, which would mask it behind
            # FinancialCallAbortedError.
            raise
        except asyncio.TimeoutError as exc:
            last_error = exc
            _log.warning(
                "financial_call_timeout",
                function=function_name,
                attempt=attempt,
                max_attempts=attempts,
            )
        except Exception as exc:
            last_error = exc
            _log.warning(
                "financial_call_failed",
                function=function_name,
                attempt=attempt,
                max_attempts=attempts,
                error_type=type(exc).__name__,
                error=str(exc),
            )
            if not retry_on_failure:
                break

        if attempt < attempts:
            await asyncio.sleep(2 ** (attempt - 1))  # 1s, 2s, 4s, ...

    await trip_kill_switch(
        f"{function_name} failed after {attempt} attempt(s): {last_error}"
    )
    raise FinancialCallAbortedError(
        f"{function_name} failed after {attempt} attempt(s); see logs for details."
    ) from last_error
