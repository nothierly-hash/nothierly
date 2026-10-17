from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from pydantic import SecretStr

from retail_gateway.config import EmailSettings
from retail_gateway.email_provider import (
    ConsoleEmailProvider,
    ResendEmailProvider,
    build_email_provider,
)


def test_console_provider_never_raises() -> None:
    ConsoleEmailProvider().send_welcome_email(to_address="user@example.com")


def test_resend_provider_sends_a_real_post_request() -> None:
    provider = ResendEmailProvider(api_key="re_test_key", from_address="welcome@nothierly.example")
    fake_response = MagicMock()
    fake_response.status = 200
    fake_response.__enter__ = MagicMock(return_value=fake_response)
    fake_response.__exit__ = MagicMock(return_value=False)

    with patch("retail_gateway.email_provider.urllib.request.urlopen", return_value=fake_response) as mock_urlopen:
        provider.send_welcome_email(to_address="user@example.com")

    request = mock_urlopen.call_args[0][0]
    assert request.full_url == "https://api.resend.com/emails"
    assert request.get_header("Authorization") == "Bearer re_test_key"
    body: dict[str, Any] = json.loads(request.data)
    assert body["to"] == ["user@example.com"]
    assert body["from"] == "welcome@nothierly.example"


def test_resend_provider_raises_on_error_status_so_celery_can_retry() -> None:
    provider = ResendEmailProvider(api_key="re_test_key", from_address="welcome@nothierly.example")
    fake_response = MagicMock()
    fake_response.status = 500
    fake_response.__enter__ = MagicMock(return_value=fake_response)
    fake_response.__exit__ = MagicMock(return_value=False)

    with patch("retail_gateway.email_provider.urllib.request.urlopen", return_value=fake_response):
        with pytest.raises(RuntimeError):
            provider.send_welcome_email(to_address="user@example.com")


def test_build_email_provider_defaults_to_console() -> None:
    settings = EmailSettings(provider="console")
    assert isinstance(build_email_provider(settings), ConsoleEmailProvider)


def test_build_email_provider_falls_back_to_console_without_an_api_key() -> None:
    settings = EmailSettings(provider="resend", resend_api_key=None)
    assert isinstance(build_email_provider(settings), ConsoleEmailProvider)


def test_build_email_provider_builds_resend_when_configured() -> None:
    settings = EmailSettings(provider="resend", resend_api_key=SecretStr("re_test_key"))
    assert isinstance(build_email_provider(settings), ResendEmailProvider)


def test_build_email_provider_ses_is_disclosed_not_implemented() -> None:
    settings = EmailSettings(provider="ses")
    assert isinstance(build_email_provider(settings), ConsoleEmailProvider)
