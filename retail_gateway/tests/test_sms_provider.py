from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from pydantic import SecretStr

from retail_gateway.config import SmsSettings
from retail_gateway.sms_provider import ConsoleSmsProvider, TwilioSmsProvider, build_sms_provider


def test_console_provider_never_raises() -> None:
    ConsoleSmsProvider().send_otp_sms(to_number="+15551234567", otp_code="123456")


def test_twilio_provider_sends_a_real_post_request() -> None:
    provider = TwilioSmsProvider(account_sid="ACxxx", auth_token="secret_token", from_number="+15550001111")
    fake_response = MagicMock()
    fake_response.status = 200
    fake_response.__enter__ = MagicMock(return_value=fake_response)
    fake_response.__exit__ = MagicMock(return_value=False)

    with patch("retail_gateway.sms_provider.urllib.request.urlopen", return_value=fake_response) as mock_urlopen:
        provider.send_otp_sms(to_number="+15551234567", otp_code="123456")

    request = mock_urlopen.call_args[0][0]
    assert request.full_url == "https://api.twilio.com/2010-04-01/Accounts/ACxxx/Messages.json"
    assert request.get_header("Authorization").startswith("Basic ")
    body = request.data.decode("utf-8")
    assert "To=%2B15551234567" in body
    assert "123456" in body


def test_twilio_provider_raises_on_error_status_so_celery_can_retry() -> None:
    provider = TwilioSmsProvider(account_sid="ACxxx", auth_token="secret_token", from_number="+15550001111")
    fake_response = MagicMock()
    fake_response.status = 500
    fake_response.__enter__ = MagicMock(return_value=fake_response)
    fake_response.__exit__ = MagicMock(return_value=False)

    with patch("retail_gateway.sms_provider.urllib.request.urlopen", return_value=fake_response):
        with pytest.raises(RuntimeError):
            provider.send_otp_sms(to_number="+15551234567", otp_code="123456")


def test_build_sms_provider_defaults_to_console() -> None:
    settings = SmsSettings(provider="console")
    assert isinstance(build_sms_provider(settings), ConsoleSmsProvider)


def test_build_sms_provider_falls_back_to_console_without_full_credentials() -> None:
    settings = SmsSettings(provider="twilio", twilio_account_sid=None)
    assert isinstance(build_sms_provider(settings), ConsoleSmsProvider)


def test_build_sms_provider_builds_twilio_when_fully_configured() -> None:
    settings = SmsSettings(
        provider="twilio",
        twilio_account_sid="ACxxx",
        twilio_auth_token=SecretStr("secret_token"),
        twilio_from_number="+15550001111",
    )
    assert isinstance(build_sms_provider(settings), TwilioSmsProvider)
