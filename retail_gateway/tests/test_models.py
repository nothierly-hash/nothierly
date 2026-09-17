from __future__ import annotations

import pytest
from pydantic import ValidationError

from retail_gateway.models import SignupRequest


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
