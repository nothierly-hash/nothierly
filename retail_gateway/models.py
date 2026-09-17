"""Request/response schemas for the Retail Gateway's HTTP API."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, EmailStr, Field, field_validator

_MIN_PASSWORD_LENGTH = 12  # matches this project's own strength-scoring convention (core/iam.py)


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
    email: str
    created_at: datetime


class PerformanceMetric(BaseModel):
    metric_name: str
    metric_value: float
    as_of: datetime
