"""Chapter 13, Étape 5.9: real session gate + idle screen lock for the Nothierly EMS.

Replaces ``src/dashboard/app.py``'s retired ``check_password()`` (a
single shared ``DASHBOARD_USER``/``DASHBOARD_PASSWORD`` pair, no 2FA,
no session tracking, no per-user anything) with
:class:`core.iam.IamStore`-backed authentication: real per-user
accounts, an optional real TOTP second factor, and revocation
(session-epoch bump, account lock, IP whitelist) that is re-checked on
**every** rerun, not only at the moment of login -- an admin locking a
compromised account or hitting "terminate all sessions" takes effect
on that user's very next click, not their next fresh login.

**Real, urgent gap this closes**: before this module, ``src/dashboard/
nothierly_ems.py`` (the live router) called no login gate at all --
only the now-archived single-page ``app.py`` ever had one. This module
is wired into the router exactly once (see that file) so every page
under ``ems_pages/`` is unreachable without a real, current session.

**Idle lock -- what "preserving state" actually means here**: Streamlit
reruns the whole script on every interaction, and ``st.Page`` pages are
only ever loaded through ``st.navigation.run()`` -- there is no way for
this module to paint a literal CSS blur over another page's
already-rendered DOM without a custom JS component, which this project
does not have. What *is* real: the lock check runs before
``navigation.run()`` is ever called, so a locked session never executes
the underlying page module at all -- ``st.session_state`` is never
touched or cleared while locked, so every widget's state beneath it
(filters, chart selections, in-progress form input) is exactly as the
user left it the instant they unlock. The overlay is a real
frosted-glass-styled full-page card in the Nothierly dark palette, not
a blur filter layered over hidden content.

**Activity signal**: "idle" here means "no Streamlit rerun triggered by
this browser tab for N minutes" -- every widget interaction is a real
rerun, so this is a genuine, if coarse, server-side activity clock. It
is not raw mouse-movement/keystroke tracking (that needs a custom
component this project does not have) -- a page left open and
untouched, with no widget clicked, is exactly the case this catches.
"""

from __future__ import annotations

import time

import streamlit as st

from core.iam import IamStore, InvalidCredentialsError, UserLockedError
from src.dashboard.theme import render_brand_header


def client_ip() -> str:
    """Real client IP via Streamlit's own request context (``st.context.ip_address``, 1.37+).

    Behind a reverse proxy this reflects whatever address the proxy
    hands Streamlit -- if a deployment terminates TLS at nginx/an ALB
    without forwarding the real client IP, this degrades to the
    proxy's own address. Not fabricated either way: it is exactly what
    Streamlit itself reports.
    """
    ip = getattr(st.context, "ip_address", None)
    return ip if ip else "unknown"


def client_user_agent() -> str:
    headers = getattr(st.context, "headers", None)
    if headers is None:
        return "unknown"
    ua = headers.get("User-Agent")
    return ua if ua else "unknown"


def _clear_session() -> None:
    for key in ("iam_authenticated", "iam_username", "iam_session_token", "login_at", "last_activity_at", "iam_locked", "iam_pending_totp_username"):
        st.session_state.pop(key, None)


def get_current_username() -> str | None:
    """The logged-in username for this browser session, or ``None`` if not authenticated.

    Only the username is cached in session state -- role, lock state,
    and 2FA status are deliberately re-read from ``IamStore`` on demand
    (``iam_store.get_user(username)``) rather than cached here: role
    changes and lock actions from an admin must be visible immediately,
    not only after a fresh login.
    """
    if not st.session_state.get("iam_authenticated", False):
        return None
    return str(st.session_state["iam_username"])


def logout(iam_store: IamStore) -> None:
    """Explicit, voluntary sign-out: revokes this one session's token and clears local session state.

    Distinct from idle-lock (which never revokes -- see
    :func:`render_lock_overlay_if_needed`) and from
    ``terminate_all_sessions``/``terminate_all_other_sessions`` (which
    act on every/other sessions) -- this ends exactly the session that
    called it.
    """
    session_token = st.session_state.get("iam_session_token")
    username = st.session_state.get("iam_username")
    if session_token is not None and username is not None:
        iam_store.terminate_session(str(username), str(session_token))
    _clear_session()


def require_login(iam_store: IamStore) -> str:
    """Gate the rest of the app behind real authentication. Returns the authenticated username.

    Never returns without one: a not-yet-authenticated visitor sees the
    login form (and, for 2FA-enabled accounts, the second-factor form)
    and the function calls ``st.stop()``, halting this rerun entirely --
    same control-flow contract the retired ``check_password()`` had.

    Revocation is live, not just checked at login: an already-"logged
    in" browser session is re-validated against the account's current
    live-session-token set, lock state, and IP whitelist on every
    single call -- an admin's "terminate all sessions" or "lock this
    user" action is enforced on this visitor's very next rerun.
    """
    if st.session_state.get("iam_authenticated", False):
        username = str(st.session_state["iam_username"])
        session_token = str(st.session_state["iam_session_token"])
        user = iam_store.get_user(username)
        if user is None or user.locked:
            _clear_session()
            st.warning("Session terminée -- ce compte a été verrouillé ou supprimé. Merci de vous reconnecter.")
            st.rerun()
        if not iam_store.is_session_still_valid(username=username, session_token=session_token):
            _clear_session()
            st.warning("Session terminée à distance (changement de mot de passe ou déconnexion forcée). Merci de vous reconnecter.")
            st.rerun()
        if not iam_store.is_ip_allowed(username, client_ip()):
            _clear_session()
            st.error("Accès refusé : votre adresse IP n'est pas autorisée pour ce compte.")
            st.stop()
        return username

    _render_login_form(iam_store)
    st.stop()
    raise AssertionError("unreachable -- st.stop() halts the script")


def _render_login_form(iam_store: IamStore) -> None:
    pending_totp_username = st.session_state.get("iam_pending_totp_username")
    if pending_totp_username is not None:
        _render_second_factor_form(iam_store, username=str(pending_totp_username))
        return

    render_brand_header(subtitle="MASTER CONTROL TERMINAL")
    st.subheader("🔒 Authentification requise")

    with st.form("iam_login_form"):
        username = st.text_input("Utilisateur")
        password = st.text_input("Mot de passe", type="password")
        submitted = st.form_submit_button("Déverrouiller")

    if not submitted:
        return

    try:
        user = iam_store.authenticate(username=username, password=password, ip=client_ip(), user_agent=client_user_agent())
    except (InvalidCredentialsError, UserLockedError) as exc:
        st.error(str(exc))
        return

    if user.totp_enabled:
        st.session_state["iam_pending_totp_username"] = user.username
        st.rerun()

    _finish_login(iam_store, user.username)
    st.rerun()


def _render_second_factor_form(iam_store: IamStore, *, username: str) -> None:
    render_brand_header(subtitle="MASTER CONTROL TERMINAL")
    st.subheader("🔐 Vérification en deux étapes")
    st.caption(f"Compte : **{username}**")

    with st.form("iam_totp_form"):
        code = st.text_input("Code de l'application d'authentification (ou code de secours)", max_chars=32)
        submitted = st.form_submit_button("Valider")
        cancel = st.form_submit_button("Annuler")

    if cancel:
        st.session_state.pop("iam_pending_totp_username", None)
        st.rerun()

    if not submitted:
        return

    if iam_store.verify_totp(username=username, code=code.strip()) or iam_store.consume_backup_code(username=username, code=code.strip()):
        st.session_state.pop("iam_pending_totp_username", None)
        _finish_login(iam_store, username)
        st.rerun()
    else:
        st.error("Code invalide.")


def _finish_login(iam_store: IamStore, username: str) -> None:
    st.session_state["iam_authenticated"] = True
    st.session_state["iam_username"] = username
    st.session_state["iam_session_token"] = iam_store.register_session(username)
    st.session_state["last_activity_at"] = time.time()


def touch_activity() -> None:
    """Record this rerun as real activity. Call once per authenticated page render, before the idle check."""
    st.session_state["last_activity_at"] = time.time()


def enforce_idle_lock(iam_store: IamStore) -> None:
    """Compare time since last activity to the admin-configured global timeout; lock if exceeded."""
    if st.session_state.get("iam_locked", False):
        return
    last_activity = st.session_state.get("last_activity_at")
    if last_activity is None:
        return
    timeout_seconds = iam_store.get_global_session_timeout_minutes() * 60
    if time.time() - float(last_activity) > timeout_seconds:
        st.session_state["iam_locked"] = True


def render_lock_overlay_if_needed(iam_store: IamStore) -> None:
    """If idle-locked, render the frosted re-authentication card and halt the script. No-op otherwise."""
    if not st.session_state.get("iam_locked", False):
        return

    username = str(st.session_state.get("iam_username", ""))
    st.markdown(
        """
        <style>
        .nothierly-lock-card {
            max-width: 420px; margin: 12vh auto 0 auto; padding: 2.5rem;
            border-radius: 16px; background: rgba(15, 20, 25, 0.85);
            border: 1px solid rgba(0, 229, 255, 0.25);
            box-shadow: 0 20px 60px rgba(0,0,0,0.5);
            backdrop-filter: blur(14px); text-align: center;
        }
        .nothierly-lock-card h2 { color: #E8EAED; margin-bottom: 0.25rem; }
        .nothierly-lock-card p { color: rgba(232,234,237,0.65); }
        </style>
        <div class="nothierly-lock-card">
            <h2>🔒 Session verrouillée</h2>
            <p>Inactivité détectée. Ressaisissez votre mot de passe pour continuer.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.form("iam_unlock_form"):
        st.text_input("Utilisateur", value=username, disabled=True)
        password = st.text_input("Mot de passe", type="password")
        submitted = st.form_submit_button("Déverrouiller")

    if submitted:
        try:
            iam_store.authenticate(username=username, password=password, ip=client_ip(), user_agent=client_user_agent())
        except (InvalidCredentialsError, UserLockedError) as exc:
            st.error(str(exc))
        else:
            st.session_state["iam_locked"] = False
            st.session_state["last_activity_at"] = time.time()
            st.rerun()

    st.stop()
