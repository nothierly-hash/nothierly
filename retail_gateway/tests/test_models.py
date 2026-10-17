from __future__ import annotations

import pytest
from pydantic import ValidationError

from retail_gateway.models import OtpRequest, OtpVerify, SignupRequest


def test_signup_request_accepts_a_valid_strong_password() -> None:
    request = SignupRequest(email="user@example.com", password="Correct-Horse-9")
    assert request.email == "user@example.com"


def test_signup_request_rejects_a_password_below_the_length_floor() -> None:
    with pytest.raises(ValidationError):
        SignupRequest(email="user@example.com", password="Short1!")


def test_signup_request_rejects_a_password_with_no_character_diversity() -> None:
    with pytest.raises(ValidationError):
        SignupRequest(email="user@example.com", password="aaaaaaaaaaaaaaaa")


def test_signup_request_rejects_an_invalid_email() -> None:
    with pytest.raises(ValidationError):
        SignupRequest(email="not-an-email", password="Correct-Horse-9")


def test_otp_request_accepts_a_valid_email_identifier() -> None:
    request = OtpRequest(channel="email", identifier="user@example.com")
    assert request.identifier == "user@example.com"


def test_otp_request_accepts_a_valid_e164_phone_identifier() -> None:
    request = OtpRequest(channel="sms", identifier="+15551234567")
    assert request.identifier == "+15551234567"


def test_otp_request_rejects_a_non_e164_phone_for_sms_channel() -> None:
    with pytest.raises(ValidationError):
        OtpRequest(channel="sms", identifier="0612345678")


def test_otp_request_rejects_an_invalid_email_for_email_channel() -> None:
    with pytest.raises(ValidationError):
        OtpRequest(channel="email", identifier="not-an-email")


def test_otp_request_rejects_an_email_shaped_identifier_for_sms_channel() -> None:
    with pytest.raises(ValidationError):
        OtpRequest(channel="sms", identifier="user@example.com")


def test_otp_verify_rejects_a_non_numeric_code() -> None:
    with pytest.raises(ValidationError):
        OtpVerify(channel="email", identifier="user@example.com", code="abcdef")


def test_otp_verify_accepts_a_six_digit_code() -> None:
    verify = OtpVerify(channel="email", identifier="user@example.com", code="123456")
    assert verify.code == "123456"
