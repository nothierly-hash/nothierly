"""Celery app + the one real background task: sending the welcome email.

**Celery + Redis, not RabbitMQ**: the brief offered either broker. This project's established
async-messaging technology, everywhere it already exists, is Redis (pub/sub for the dashboard,
``INCR``/``EXPIRE`` for rate limiting) and ZMQ (the trading bus) -- RabbitMQ appears nowhere in
this codebase today. Celery's broker is pluggable and supports Redis natively, so picking Redis
here introduces zero new message-broker *technology*, only a new (fully separate, air-gapped --
see ``docker-compose.retail.yml``) Redis container. This is the real reason, not a shortcut: the
brief's actual requirement -- signup returns instantly, the email send happens out-of-band, in a
separate worker process -- is satisfied identically either way.

Run the worker with::

    celery -A retail_gateway.tasks worker --loglevel=info
"""

from __future__ import annotations

from celery import Celery

from retail_gateway.config import get_settings
from retail_gateway.email_provider import build_email_provider
from retail_gateway.logging_config import configure_logging, get_logger
from retail_gateway.sms_provider import build_sms_provider

_settings = get_settings()

celery_app = Celery(
    "retail_gateway",
    broker=_settings.redis.url(db=_settings.redis.celery_broker_db),
    backend=_settings.redis.url(db=_settings.redis.celery_result_backend_db),
)
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    task_track_started=True,
)

configure_logging(service_name="retail-email-worker")
_log = get_logger(__name__)


@celery_app.task(
    name="retail_gateway.send_welcome_email",
    autoretry_for=(RuntimeError,),
    retry_backoff=True,
    retry_backoff_max=300,
    retry_kwargs={"max_retries": 5},
)
def send_welcome_email(to_address: str) -> None:
    """Queued by ``POST /api/v1/signup`` immediately after account creation -- see
    ``main.py``'s own docstring on why the HTTP response never waits on this.
    """
    provider = build_email_provider(_settings.email)
    provider.send_welcome_email(to_address=to_address)
    _log.info("welcome_email_task_completed", to=to_address)


@celery_app.task(
    name="retail_gateway.send_otp_email",
    autoretry_for=(RuntimeError,),
    retry_backoff=True,
    retry_backoff_max=60,
    retry_kwargs={"max_retries": 3},
)
def send_otp_email(to_address: str, otp_code: str) -> None:
    """Queued by ``POST /api/v1/otp/request`` (channel=email). Shorter retry ceiling than the
    welcome email (60s vs 300s, 3 vs 5 attempts): the code's own 5-minute TTL makes a slow
    eventual delivery worthless past a point, unlike a welcome email that is still useful hours
    late.
    """
    provider = build_email_provider(_settings.email)
    provider.send_otp_email(to_address=to_address, otp_code=otp_code)
    _log.info("otp_email_task_completed", to=to_address)


@celery_app.task(
    name="retail_gateway.send_otp_sms",
    autoretry_for=(RuntimeError,),
    retry_backoff=True,
    retry_backoff_max=60,
    retry_kwargs={"max_retries": 3},
)
def send_otp_sms(to_number: str, otp_code: str) -> None:
    """Queued by ``POST /api/v1/otp/request`` (channel=sms). Same short retry ceiling as
    :func:`send_otp_email`, for the same TTL-driven reason.
    """
    provider = build_sms_provider(_settings.sms)
    provider.send_otp_sms(to_number=to_number, otp_code=otp_code)
    _log.info("otp_sms_task_completed", to=to_number)
