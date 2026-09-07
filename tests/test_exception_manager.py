"""Real, executable tests for core/exception_manager.py.

``os._exit()`` is real, load-bearing production behavior here (see that
module's own docstring on why plain ``sys.exit()`` would be a lie) --
every test below patches ``os._exit`` to raise :class:`_ExitCalled`
instead of actually terminating the pytest process, then asserts on the
raised marker's exit code. This is not a mock of the module under test:
``graceful_shutdown`` runs its real ``asyncio.wait_for``/``try``/``except``/
``finally`` control flow unchanged, only the one process-terminating
syscall at the very end is intercepted.
"""

from __future__ import annotations

import asyncio
from typing import Any, Generator

import pytest

from core.exception_manager import graceful_shutdown, install_global_exception_handler
from core.notifier import TelegramNotifier
from core.config import TelegramSettings
from core.schemas import SYSTEM_FATAL_EXCEPTION_TOPIC
from core.zmq_bus import ZmqPublisher, ZmqSubscriber


class _ExitCalled(Exception):
    """Raised by the patched os._exit stand-in -- lets a real os._exit() call be observed without killing the test process."""

    def __init__(self, code: int) -> None:
        super().__init__(f"os._exit({code})")
        self.code = code


class _RecordingZmqPublisher:
    """Same real-fake shape as tests/test_alert_agent.py's own -- no mocking library exists for ZMQ in this project."""

    def __init__(self) -> None:
        self.published: list[Any] = []

    async def publish(self, payload: Any) -> None:
        self.published.append(payload)

    def close(self, linger_ms: int = 0) -> None:
        pass


@pytest.fixture
def exit_calls(monkeypatch: pytest.MonkeyPatch) -> Generator[list[int], None, None]:
    calls: list[int] = []

    def fake_exit(code: int) -> None:
        calls.append(code)
        raise _ExitCalled(code)

    monkeypatch.setattr("os._exit", fake_exit)
    yield calls


# =============================================================================
# graceful_shutdown -- always ends in a real os._exit(), on every path
# =============================================================================


@pytest.mark.asyncio
async def test_graceful_shutdown_exits_after_a_successful_callback(exit_calls: list[int]) -> None:
    ran: list[bool] = []

    async def callback() -> None:
        ran.append(True)

    with pytest.raises(_ExitCalled) as exc_info:
        await graceful_shutdown(callback, reason="test success", timeout_seconds=1.0)

    assert ran == [True]
    assert exc_info.value.code == 1


@pytest.mark.asyncio
async def test_graceful_shutdown_exits_on_timeout_even_if_callback_never_finishes(exit_calls: list[int]) -> None:
    async def callback() -> None:
        await asyncio.sleep(10)  # deliberately longer than the timeout below

    with pytest.raises(_ExitCalled) as exc_info:
        await graceful_shutdown(callback, reason="test timeout", timeout_seconds=0.05)

    assert exc_info.value.code == 1


@pytest.mark.asyncio
async def test_graceful_shutdown_exits_even_if_callback_raises(exit_calls: list[int]) -> None:
    async def callback() -> None:
        raise ValueError("cancel_all_orders failed")

    with pytest.raises(_ExitCalled):
        await graceful_shutdown(callback, reason="test callback error", timeout_seconds=1.0)

    assert exit_calls == [1]


@pytest.mark.asyncio
async def test_graceful_shutdown_respects_a_custom_exit_code(exit_calls: list[int]) -> None:
    async def callback() -> None:
        pass

    with pytest.raises(_ExitCalled) as exc_info:
        await graceful_shutdown(callback, reason="custom code", timeout_seconds=1.0, exit_code=17)

    assert exc_info.value.code == 17


# =============================================================================
# install_global_exception_handler -- real asyncio.AbstractEventLoop.set_exception_handler wiring
# =============================================================================


async def _run_and_collect_new_tasks(
    loop: asyncio.AbstractEventLoop, context: dict[str, Any], monkeypatch: pytest.MonkeyPatch,
) -> list[asyncio.Task[Any]]:
    """Real technique, no production code changed: call the real public asyncio entrypoint the loop
    itself uses internally, capturing every task it schedules via a recording wrapper around
    ``loop.create_task``.

    **Not** ``asyncio.all_tasks(loop) - tasks_before``, which was tried first and is subtly wrong:
    ``all_tasks()`` only returns *not-yet-done* tasks -- a task that both starts and finishes within
    this helper's own ``asyncio.sleep(0.05)`` (exactly what a fast, successful ``graceful_shutdown``
    does) is already excluded by the time the "after" snapshot is taken, so its exception is never
    retrieved by the caller and asyncio's own "Task exception was never retrieved" handler fires
    instead -- routing right back into the handler under test with a *different* context than the one
    this helper meant to observe.
    """
    created: list[asyncio.Task[Any]] = []
    original_create_task = loop.create_task

    def recording_create_task(coro: Any, **kwargs: Any) -> asyncio.Task[Any]:
        task = original_create_task(coro, **kwargs)
        created.append(task)
        return task

    monkeypatch.setattr(loop, "create_task", recording_create_task)
    loop.call_exception_handler(context)
    await asyncio.sleep(0.05)
    return created


@pytest.mark.asyncio
async def test_unhandled_exception_triggers_graceful_shutdown(
    exit_calls: list[int], monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop = asyncio.get_running_loop()
    shutdown_ran: list[bool] = []

    async def shutdown_callback() -> None:
        shutdown_ran.append(True)

    install_global_exception_handler(loop, shutdown_callback, sender_id="test-agent")

    new_tasks = await _run_and_collect_new_tasks(
        loop, {"exception": ValueError("boom"), "message": "boom happened"}, monkeypatch,
    )
    results = await asyncio.gather(*new_tasks, return_exceptions=True)

    assert shutdown_ran == [True]
    assert any(isinstance(r, _ExitCalled) and r.code == 1 for r in results)


@pytest.mark.asyncio
async def test_handler_tolerates_a_context_with_no_exception_object(
    exit_calls: list[int], monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Some real asyncio loop errors carry only a message, no exception -- must not crash the handler itself."""
    loop = asyncio.get_running_loop()
    shutdown_ran: list[bool] = []

    async def shutdown_callback() -> None:
        shutdown_ran.append(True)

    install_global_exception_handler(loop, shutdown_callback, sender_id="test-agent")

    new_tasks = await _run_and_collect_new_tasks(loop, {"message": "Fatal error on socket transport"}, monkeypatch)
    await asyncio.gather(*new_tasks, return_exceptions=True)

    assert shutdown_ran == [True]


@pytest.mark.asyncio
async def test_second_exception_during_shutdown_does_not_trigger_a_second_shutdown(
    exit_calls: list[int], monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop = asyncio.get_running_loop()
    call_count: list[bool] = []

    async def shutdown_callback() -> None:
        call_count.append(True)
        await asyncio.sleep(10)  # kept "in flight" for the whole test -- never actually completes

    install_global_exception_handler(loop, shutdown_callback, sender_id="test-agent", shutdown_timeout_seconds=10)

    created: list[asyncio.Task[Any]] = []
    original_create_task = loop.create_task

    def recording_create_task(coro: Any, **kwargs: Any) -> asyncio.Task[Any]:
        task = original_create_task(coro, **kwargs)
        created.append(task)
        return task

    monkeypatch.setattr(loop, "create_task", recording_create_task)
    loop.call_exception_handler({"exception": ValueError("first"), "message": "first"})
    await asyncio.sleep(0.05)
    loop.call_exception_handler({"exception": ValueError("second"), "message": "second"})
    await asyncio.sleep(0.05)

    assert call_count == [True]  # exactly one shutdown_callback invocation, not two

    for task in created:
        task.cancel()
    await asyncio.gather(*created, return_exceptions=True)


@pytest.mark.asyncio
async def test_broken_notifier_does_not_prevent_the_real_shutdown(
    exit_calls: list[int], monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unconfigured TelegramNotifier (no bot_token/chat_id) must never block graceful_shutdown."""
    loop = asyncio.get_running_loop()
    shutdown_ran: list[bool] = []

    async def shutdown_callback() -> None:
        shutdown_ran.append(True)

    notifier = TelegramNotifier(TelegramSettings())
    assert not notifier.is_configured
    install_global_exception_handler(loop, shutdown_callback, sender_id="test-agent", notifier=notifier)

    new_tasks = await _run_and_collect_new_tasks(loop, {"exception": ValueError("boom"), "message": "boom"}, monkeypatch)
    results = await asyncio.gather(*new_tasks, return_exceptions=True)

    assert shutdown_ran == [True]
    assert any(isinstance(r, _ExitCalled) for r in results)


@pytest.mark.asyncio
async def test_fatal_exception_is_recorded_by_a_fake_publisher(
    exit_calls: list[int], monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop = asyncio.get_running_loop()
    publisher = _RecordingZmqPublisher()

    async def shutdown_callback() -> None:
        pass

    install_global_exception_handler(
        loop, shutdown_callback, sender_id="test-agent", publisher=publisher,  # type: ignore[arg-type]
    )

    new_tasks = await _run_and_collect_new_tasks(
        loop, {"exception": ValueError("boom"), "message": "boom happened"}, monkeypatch,
    )
    await asyncio.gather(*new_tasks, return_exceptions=True)

    assert len(publisher.published) == 1
    assert publisher.published[0].topic == SYSTEM_FATAL_EXCEPTION_TOPIC
    assert publisher.published[0].sender_id == "test-agent"
    assert publisher.published[0].data["message"] == "boom happened"


@pytest.mark.asyncio
async def test_fatal_exception_is_broadcast_over_a_real_zmq_socket(
    exit_calls: list[int], monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Real end-to-end: a real ZmqPublisher/ZmqSubscriber pair over a real TCP loopback socket, not a fake."""
    address = "tcp://127.0.0.1:57896"
    publisher = ZmqPublisher(address, bind=True)
    subscriber = ZmqSubscriber(address)
    subscriber.subscribe(SYSTEM_FATAL_EXCEPTION_TOPIC)
    try:
        await asyncio.sleep(0.2)  # real ZMQ slow-joiner -- let the SUB socket's subscription propagate
        loop = asyncio.get_running_loop()

        async def shutdown_callback() -> None:
            pass

        install_global_exception_handler(loop, shutdown_callback, sender_id="test-agent", publisher=publisher)

        new_tasks = await _run_and_collect_new_tasks(
            loop, {"exception": ValueError("real zmq boom"), "message": "real zmq boom"}, monkeypatch,
        )
        received = await asyncio.wait_for(subscriber.receive(), timeout=3.0)

        assert received.topic == SYSTEM_FATAL_EXCEPTION_TOPIC
        assert received.data["message"] == "real zmq boom"
        assert "ValueError" in received.data["traceback"]

        await asyncio.gather(*new_tasks, return_exceptions=True)
    finally:
        publisher.close()
        subscriber.close()
