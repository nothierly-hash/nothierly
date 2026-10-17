"""Request/response schemas for the Retail Gateway's HTTP API."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Final, Literal
from uuid import UUID

from pydantic import BaseModel, EmailStr, Field, TypeAdapter, ValidationError, field_validator, model_validator

_email_adapter: Final = TypeAdapter(EmailStr)

_MIN_PASSWORD_LENGTH = 12  # matches this project's own strength-scoring convention (core/iam.py)

# E.164: a leading '+', 8-15 digits total, no leading zero after the '+' -- the same format
# Twilio's own API requires for both `To` and `From` (see sms_provider.py). A regex, not the
# `phonenumbers` library, for the same "don't add a dependency the stdlib already covers"
# discipline email_provider.py's own docstring already applies to Resend.
_E164_PHONE_PATTERN = re.compile(r"^\+[1-9]\d{7,14}$")
_OTP_CODE_PATTERN = re.compile(r"^\d{4,10}$")


class OtpRequest(BaseModel):
    """Request one OTP over ``channel`` -- the identifier's shape is validated against that channel."""

    channel: Literal["email", "sms"]
    identifier: str = Field(min_length=3, max_length=254, description="An email address for channel=email, an E.164 phone number for channel=sms.")

    @model_validator(mode="after")
    def _validate_identifier_matches_channel(self) -> "OtpRequest":
        if self.channel == "sms" and not _E164_PHONE_PATTERN.match(self.identifier):
            raise ValueError("identifier must be an E.164 phone number (e.g. +15551234567) when channel=sms.")
        if self.channel == "email":
            # Reuses EmailStr's own validation via TypeAdapter, rather than a second hand-rolled
            # email regex living beside SignupRequest's real one.
            try:
                _email_adapter.validate_python(self.identifier)
            except ValidationError as exc:
                raise ValueError("identifier must be a valid email address when channel=email.") from exc
        return self


class OtpVerify(BaseModel):
    channel: Literal["email", "sms"]
    identifier: str = Field(min_length=3, max_length=254)
    code: str = Field(pattern=_OTP_CODE_PATTERN.pattern)


class SignupRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=_MIN_PASSWORD_LENGTH, max_length=256)

    @field_validator("password")
    @classmethod
    def _require_character_diversity(cls, value: str) -> str:
        """Same real heuristic class as ``core.iam.password_strength_score`` -- length +
        character-class diversity, not a full zxcvbn-grade check (this project has no such
        library installed). Requires at least 2 of the 4 classes on top of the length floor,
        loose enough not to reject real, reasonable passphrases outright.
        """
        classes_present = sum(
            [
                any(c.islower() for c in value),
                any(c.isupper() for c in value),
                any(c.isdigit() for c in value),
                any(not c.isalnum() for c in value),
            ]
        )
        if classes_present < 2:
            raise ValueError("Password must mix at least two of: lowercase, uppercase, digit, symbol.")
        return value


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_at: datetime


class UserPublic(BaseModel):
    id: UUID
    email: str | None = None
    phone_number: str | None = None
    created_at: datetime


class PerformanceMetric(BaseModel):
    metric_name: str
    metric_value: float
    as_of: datetime
