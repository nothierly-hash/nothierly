"""Real, executable tests for core/iam.py -- against a real in-memory fakeredis server, not mocks."""

from __future__ import annotations

import pyotp
import pytest
from fakeredis import FakeRedis, FakeServer

from core.iam import (
    IamStore,
    InvalidCredentialsError,
    UserLockedError,
    UserNotFoundError,
    password_strength_score,
)
from src.execution.security_vault import SecurityVault


def _decode(value: object) -> object:
    """This fakeredis version doesn't fully honor decode_responses=True for every command -- see tests/test_alert_agent.py."""
    return value.decode() if isinstance(value, bytes) else value


@pytest.fixture
def store() -> IamStore:
    r = FakeRedis(server=FakeServer(), decode_responses=True)
    vault = SecurityVault("test-master-key-not-real")
    return IamStore(r, vault)


# -- bootstrap / user CRUD --------------------------------


def test_ensure_bootstrap_admin_creates_once(store: IamStore) -> None:
    store.ensure_bootstrap_admin(username="ops", password="Sup3rSecret!")
    user = store.get_user("ops")
    assert user is not None
    assert user.role == "admin"
    store.ensure_bootstrap_admin(username="ops", password="DifferentPassword!")  # must not overwrite
    still = store.get_user("ops")
    assert still is not None
    assert still.password_hash == user.password_hash


def test_create_user_rejects_duplicate(store: IamStore) -> None:
    store.create_user(username="trader1", password="Passw0rd!", role="trader")
    with pytest.raises(Exception):
        store.create_user(username="trader1", password="Other!", role="trader")


def test_list_users_sorted_by_username(store: IamStore) -> None:
    store.create_user(username="zoe", password="Passw0rd!", role="trader")
    store.create_user(username="alice", password="Passw0rd!", role="trader")
    usernames = [u.username for u in store.list_users()]
    assert usernames == ["alice", "zoe"]


# -- authentication --------------------------------


def test_authenticate_succeeds_with_real_password(store: IamStore) -> None:
    store.create_user(username="trader1", password="CorrectHorse123!", role="trader")
    user = store.authenticate(username="trader1", password="CorrectHorse123!", ip="1.2.3.4", user_agent="pytest")
    assert user.username == "trader1"


def test_authenticate_rejects_wrong_password(store: IamStore) -> None:
    store.create_user(username="trader1", password="CorrectHorse123!", role="trader")
    with pytest.raises(InvalidCredentialsError):
        store.authenticate(username="trader1", password="wrong", ip="1.2.3.4", user_agent="pytest")


def test_authenticate_rejects_unknown_user(store: IamStore) -> None:
    with pytest.raises(InvalidCredentialsError):
        store.authenticate(username="ghost", password="whatever", ip="1.2.3.4", user_agent="pytest")


def test_authenticate_rejects_locked_user(store: IamStore) -> None:
    store.create_user(username="trader1", password="CorrectHorse123!", role="trader")
    store.lock_user("trader1")
    with pytest.raises(UserLockedError):
        store.authenticate(username="trader1", password="CorrectHorse123!", ip="1.2.3.4", user_agent="pytest")


def test_authenticate_records_both_success_and_failure_events(store: IamStore) -> None:
    store.create_user(username="trader1", password="CorrectHorse123!", role="trader")
    with pytest.raises(InvalidCredentialsError):
        store.authenticate(username="trader1", password="wrong", ip="9.9.9.9", user_agent="pytest-fail")
    store.authenticate(username="trader1", password="CorrectHorse123!", ip="1.2.3.4", user_agent="pytest-ok")
    sessions = store.list_sessions("trader1")
    assert len(sessions) == 2
    assert sessions[0]["event_type"] == "login_succeeded"
    assert sessions[1]["event_type"] == "login_failed_bad_password"


# -- verify_current_password --------------------------------


def test_verify_current_password_true_for_real_password(store: IamStore) -> None:
    store.create_user(username="trader1", password="Passw0rd!", role="trader")
    assert store.verify_current_password(username="trader1", password="Passw0rd!") is True


def test_verify_current_password_false_for_wrong_password(store: IamStore) -> None:
    store.create_user(username="trader1", password="Passw0rd!", role="trader")
    assert store.verify_current_password(username="trader1", password="wrong") is False


def test_verify_current_password_does_not_revoke_sessions(store: IamStore) -> None:
    """Real regression guard: unlike change_password, this must have zero side effects on live sessions."""
    store.create_user(username="trader1", password="Passw0rd!", role="trader")
    token = store.register_session("trader1")
    store.verify_current_password(username="trader1", password="Passw0rd!")
    assert store.is_session_still_valid(username="trader1", session_token=token) is True


# -- change_password (Session Confirmation rule) --------------------------------


def test_change_password_requires_real_current_password(store: IamStore) -> None:
    store.create_user(username="trader1", password="OldPassw0rd!", role="trader")
    with pytest.raises(InvalidCredentialsError):
        store.change_password(username="trader1", current_password="wrong", new_password="NewPassw0rd!")


def test_change_password_updates_and_new_password_authenticates(store: IamStore) -> None:
    store.create_user(username="trader1", password="OldPassw0rd!", role="trader")
    store.change_password(username="trader1", current_password="OldPassw0rd!", new_password="NewPassw0rd!")
    user = store.authenticate(username="trader1", password="NewPassw0rd!", ip="1.2.3.4", user_agent="pytest")
    assert user.username == "trader1"


def test_change_password_revokes_all_sessions(store: IamStore) -> None:
    store.create_user(username="trader1", password="OldPassw0rd!", role="trader")
    token = store.register_session("trader1")
    assert store.is_session_still_valid(username="trader1", session_token=token) is True
    store.change_password(username="trader1", current_password="OldPassw0rd!", new_password="NewPassw0rd!")
    assert store.is_session_still_valid(username="trader1", session_token=token) is False


# -- TOTP / 2FA --------------------------------


def test_totp_enrollment_round_trip(store: IamStore) -> None:
    store.create_user(username="trader1", password="Passw0rd!", role="trader")
    secret, uri, qr_png = store.begin_totp_enrollment("trader1")
    assert uri.startswith("otpauth://totp/")
    assert len(qr_png) > 0
    real_code = pyotp.TOTP(secret).now()
    store.confirm_totp_enrollment(username="trader1", pending_secret=secret, code=real_code)
    user = store.get_user("trader1")
    assert user is not None
    assert user.totp_enabled is True
    assert user.totp_secret_encrypted is not None
    assert secret not in user.totp_secret_encrypted  # never stored in plaintext


def test_totp_enrollment_rejects_wrong_code(store: IamStore) -> None:
    store.create_user(username="trader1", password="Passw0rd!", role="trader")
    secret, _, _ = store.begin_totp_enrollment("trader1")
    with pytest.raises(InvalidCredentialsError):
        store.confirm_totp_enrollment(username="trader1", pending_secret=secret, code="000000")
    user = store.get_user("trader1")
    assert user is not None
    assert user.totp_enabled is False


def test_verify_totp_after_enrollment(store: IamStore) -> None:
    store.create_user(username="trader1", password="Passw0rd!", role="trader")
    secret, _, _ = store.begin_totp_enrollment("trader1")
    store.confirm_totp_enrollment(username="trader1", pending_secret=secret, code=pyotp.TOTP(secret).now())
    assert store.verify_totp(username="trader1", code=pyotp.TOTP(secret).now()) is True
    assert store.verify_totp(username="trader1", code="000000") is False


def test_verify_totp_false_when_not_enabled(store: IamStore) -> None:
    store.create_user(username="trader1", password="Passw0rd!", role="trader")
    assert store.verify_totp(username="trader1", code="123456") is False


def test_disable_totp_requires_password_and_clears_backup_codes(store: IamStore) -> None:
    store.create_user(username="trader1", password="Passw0rd!", role="trader")
    secret, _, _ = store.begin_totp_enrollment("trader1")
    store.confirm_totp_enrollment(username="trader1", pending_secret=secret, code=pyotp.TOTP(secret).now())
    store.generate_backup_codes("trader1")
    assert store.remaining_backup_code_count("trader1") == 10

    with pytest.raises(InvalidCredentialsError):
        store.disable_totp(username="trader1", current_password="wrong")

    store.disable_totp(username="trader1", current_password="Passw0rd!")
    user = store.get_user("trader1")
    assert user is not None
    assert user.totp_enabled is False
    assert store.remaining_backup_code_count("trader1") == 0


# -- backup codes --------------------------------


def test_generate_backup_codes_returns_ten_unique_plaintext_codes(store: IamStore) -> None:
    store.create_user(username="trader1", password="Passw0rd!", role="trader")
    codes = store.generate_backup_codes("trader1")
    assert len(codes) == 10
    assert len(set(codes)) == 10


def test_consume_backup_code_works_once_only(store: IamStore) -> None:
    store.create_user(username="trader1", password="Passw0rd!", role="trader")
    codes = store.generate_backup_codes("trader1")
    first = codes[0]
    assert store.consume_backup_code(username="trader1", code=first) is True
    assert store.consume_backup_code(username="trader1", code=first) is False  # burned, not reusable
    assert store.remaining_backup_code_count("trader1") == 9


def test_regenerating_backup_codes_invalidates_old_set(store: IamStore) -> None:
    store.create_user(username="trader1", password="Passw0rd!", role="trader")
    old_codes = store.generate_backup_codes("trader1")
    store.generate_backup_codes("trader1")
    assert store.consume_backup_code(username="trader1", code=old_codes[0]) is False


# -- session revocation --------------------------------


def test_register_session_returns_unique_tokens(store: IamStore) -> None:
    store.create_user(username="trader1", password="Passw0rd!", role="trader")
    token_a = store.register_session("trader1")
    token_b = store.register_session("trader1")
    assert token_a != token_b
    assert store.is_session_still_valid(username="trader1", session_token=token_a) is True
    assert store.is_session_still_valid(username="trader1", session_token=token_b) is True


def test_unregistered_token_is_invalid(store: IamStore) -> None:
    store.create_user(username="trader1", password="Passw0rd!", role="trader")
    assert store.is_session_still_valid(username="trader1", session_token="never-issued") is False


def test_terminate_session_revokes_only_that_one(store: IamStore) -> None:
    store.create_user(username="trader1", password="Passw0rd!", role="trader")
    token_a = store.register_session("trader1")
    token_b = store.register_session("trader1")
    store.terminate_session("trader1", token_a)
    assert store.is_session_still_valid(username="trader1", session_token=token_a) is False
    assert store.is_session_still_valid(username="trader1", session_token=token_b) is True


def test_terminate_all_sessions_revokes_everything(store: IamStore) -> None:
    store.create_user(username="trader1", password="Passw0rd!", role="trader")
    token = store.register_session("trader1")
    store.terminate_all_sessions("trader1")
    assert store.is_session_still_valid(username="trader1", session_token=token) is False


def test_terminate_all_other_sessions_spares_the_current_one(store: IamStore) -> None:
    store.create_user(username="trader1", password="Passw0rd!", role="trader")
    current = store.register_session("trader1")
    other_a = store.register_session("trader1")
    other_b = store.register_session("trader1")

    revoked_count = store.terminate_all_other_sessions("trader1", keep_session_token=current)

    assert revoked_count == 2
    assert store.is_session_still_valid(username="trader1", session_token=current) is True
    assert store.is_session_still_valid(username="trader1", session_token=other_a) is False
    assert store.is_session_still_valid(username="trader1", session_token=other_b) is False


# -- IP whitelisting --------------------------------


def test_ip_whitelist_disabled_by_default_allows_everything(store: IamStore) -> None:
    store.create_user(username="trader1", password="Passw0rd!", role="trader")
    assert store.is_ip_allowed("trader1", "1.2.3.4") is True


def test_ip_whitelist_enabled_blocks_unknown_ip(store: IamStore) -> None:
    store.create_user(username="trader1", password="Passw0rd!", role="trader")
    store.set_ip_whitelist_enabled("trader1", True)
    store.add_whitelisted_ip("trader1", "1.2.3.4")
    assert store.is_ip_allowed("trader1", "1.2.3.4") is True
    assert store.is_ip_allowed("trader1", "9.9.9.9") is False


def test_remove_whitelisted_ip(store: IamStore) -> None:
    store.create_user(username="trader1", password="Passw0rd!", role="trader")
    store.set_ip_whitelist_enabled("trader1", True)
    store.add_whitelisted_ip("trader1", "1.2.3.4")
    store.remove_whitelisted_ip("trader1", "1.2.3.4")
    assert store.is_ip_allowed("trader1", "1.2.3.4") is False


# -- admin: roles / locking / global timeout --------------------------------


def test_set_role(store: IamStore) -> None:
    store.create_user(username="trader1", password="Passw0rd!", role="trader")
    store.set_role("trader1", "admin")
    user = store.get_user("trader1")
    assert user is not None
    assert user.role == "admin"


def test_lock_and_unlock_user(store: IamStore) -> None:
    store.create_user(username="trader1", password="Passw0rd!", role="trader")
    store.lock_user("trader1")
    assert store.get_user("trader1").locked is True  # type: ignore[union-attr]
    store.unlock_user("trader1")
    assert store.get_user("trader1").locked is False  # type: ignore[union-attr]


def test_force_password_reset_flags_and_revokes_sessions(store: IamStore) -> None:
    store.create_user(username="trader1", password="Passw0rd!", role="trader")
    token = store.register_session("trader1")
    store.force_password_reset("trader1")
    user = store.get_user("trader1")
    assert user is not None
    assert user.must_change_password is True
    assert store.is_session_still_valid(username="trader1", session_token=token) is False


def test_global_session_timeout_defaults_and_is_settable(store: IamStore) -> None:
    assert store.get_global_session_timeout_minutes() == 15
    store.set_global_session_timeout_minutes(30)
    assert store.get_global_session_timeout_minutes() == 30


def test_global_session_timeout_rejects_non_positive(store: IamStore) -> None:
    with pytest.raises(ValueError):
        store.set_global_session_timeout_minutes(0)


def test_operations_on_unknown_user_raise_user_not_found(store: IamStore) -> None:
    with pytest.raises(UserNotFoundError):
        store.set_role("ghost", "admin")
    with pytest.raises(UserNotFoundError):
        store.lock_user("ghost")


# -- password_strength_score --------------------------------


def test_password_strength_empty_is_zero() -> None:
    score, label = password_strength_score("")
    assert score == 0
    assert label == "Vide"


def test_password_strength_increases_with_length_and_diversity() -> None:
    weak_score, _ = password_strength_score("abc")
    medium_score, _ = password_strength_score("abcdefgh")
    strong_score, _ = password_strength_score("Abcdefgh123!@#")
    assert weak_score < medium_score < strong_score
    assert strong_score == 4
