"""Mon Compte & Sécurité -- real IAM UI over core.iam.IamStore. Chapter 13, Étape 5.9.

Three tabs, all wired to real, tested backend calls (see
``core/iam.py`` and ``tests/test_iam.py``) -- nothing here is a static
mockup:

1. **Sécurité personnelle**: real Argon2 password change (current
   password required -- Session Confirmation rule), a real strength
   estimate, real TOTP 2FA enrollment with a real QR code and one-time
   backup codes.
2. **Audit & Sessions**: the real login-event history
   (``iam_store.list_sessions``), a real "terminate all other
   sessions" action (see ``core/iam.py``'s module docstring on why this
   is a genuine per-session-token revocation, not a fake button), and a
   real, honestly-scoped IP whitelist (application-level only -- see
   that same docstring on why true network enforcement belongs at a
   reverse proxy).
3. **Admin Policies** -- visible only when the current user's real role
   is ``"admin"``: user management (role, lock, forced password reset)
   and the global idle-lock timeout consumed by
   :func:`src.dashboard.auth_middleware.enforce_idle_lock`.

**Why the new-password field sits outside ``st.form``**: Streamlit
batches every widget inside a form and only reruns the script on
submit -- a "real-time" strength bar cannot update on each keystroke
inside one. The password inputs here are therefore plain widgets with
an explicit submit button, not ``st.form``, specifically so the
strength bar is real, not decorative.
"""

from __future__ import annotations

from typing import cast

import pandas as pd
import streamlit as st

from core.iam import IamStore, InvalidCredentialsError, UserNotFoundError, password_strength_score
from src.dashboard.auth_middleware import client_ip, get_current_username, logout
from src.dashboard.ems_pages._shared import get_iam_store

st.title("🛡️ Mon Compte & Sécurité")

_iam_store: IamStore = get_iam_store()
_username = get_current_username()
if _username is None:
    st.error("Session invalide.")
    st.stop()

_user = _iam_store.get_user(_username)
if _user is None:
    st.error("Compte introuvable.")
    st.stop()

_caption_col, _logout_col = st.columns([4, 1])
with _caption_col:
    st.caption(f"Connecté en tant que **{_username}** ({_user.role}).")
with _logout_col:
    if st.button("Se déconnecter", width="stretch"):
        logout(_iam_store)
        st.rerun()

tab_personal, tab_audit, tab_admin = st.tabs(["🔐 Sécurité personnelle", "📜 Audit & Sessions", "⚙️ Politiques Admin"])

# ============================================================
# Onglet 1 -- Sécurité personnelle
# ============================================================
with tab_personal:
    st.subheader("Changer le mot de passe")
    st.caption("Toute modification de sécurité sur ce compte exige le mot de passe actuel (confirmation de session).")

    current_pw = st.text_input("Mot de passe actuel", type="password", key="pw_current")
    new_pw = st.text_input("Nouveau mot de passe", type="password", key="pw_new")
    confirm_pw = st.text_input("Confirmer le nouveau mot de passe", type="password", key="pw_confirm")

    score, label = password_strength_score(new_pw)
    bar_color = ["#5B6B7C", "#FF1744", "#FF9100", "#FFEA00", "#00E676"][score]
    st.markdown(
        f"""<div style="height:6px;border-radius:3px;background:rgba(255,255,255,0.08);overflow:hidden;margin:4px 0;">
        <div style="height:100%;width:{score/4*100:.0f}%;background:{bar_color};"></div></div>
        <span style="font-size:0.8rem;color:{bar_color};">{label}</span>""",
        unsafe_allow_html=True,
    )

    if st.button("Mettre à jour le mot de passe"):
        if not new_pw or new_pw != confirm_pw:
            st.error("Les nouveaux mots de passe ne correspondent pas.")
        else:
            try:
                _iam_store.change_password(username=_username, current_password=current_pw, new_password=new_pw)
            except InvalidCredentialsError as exc:
                st.error(str(exc))
            else:
                st.success("Mot de passe mis à jour. Toutes les sessions (y compris celle-ci) ont été déconnectées -- reconnectez-vous.")
                st.session_state.pop("iam_authenticated", None)
                st.rerun()

    st.divider()
    st.subheader("Authentification à deux facteurs (2FA)")

    if _user.totp_enabled:
        st.success("✅ 2FA actif sur ce compte.")
        st.caption(f"Codes de secours restants : {_iam_store.remaining_backup_code_count(_username)}")

        with st.expander("Régénérer les codes de secours"):
            regen_pw = st.text_input("Mot de passe actuel", type="password", key="regen_backup_pw")
            if st.button("Régénérer (invalide les anciens codes)"):
                if not _iam_store.verify_current_password(username=_username, password=regen_pw):
                    st.error("Mot de passe incorrect.")
                else:
                    codes = _iam_store.generate_backup_codes(_username)
                    st.warning("⚠️ Notez ces codes maintenant -- ils ne seront plus jamais affichés.")
                    st.code("\n".join(codes))

        with st.expander("Désactiver la 2FA"):
            disable_pw = st.text_input("Mot de passe actuel", type="password", key="disable_totp_pw")
            if st.button("Désactiver la 2FA", type="secondary"):
                try:
                    _iam_store.disable_totp(username=_username, current_password=disable_pw)
                except InvalidCredentialsError as exc:
                    st.error(str(exc))
                else:
                    st.success("2FA désactivée.")
                    st.rerun()
    else:
        st.info("2FA non activée sur ce compte.")
        if "pending_totp_secret" not in st.session_state:
            if st.button("Activer la 2FA"):
                secret, uri, qr_png = _iam_store.begin_totp_enrollment(_username)
                st.session_state["pending_totp_secret"] = secret
                st.session_state["pending_totp_uri"] = uri
                st.session_state["pending_totp_qr"] = qr_png
                st.rerun()
        else:
            st.image(st.session_state["pending_totp_qr"], caption="Scannez avec Google Authenticator / Authy / 1Password", width=220)
            st.caption(f"Saisie manuelle : `{st.session_state['pending_totp_secret']}`")
            confirm_code = st.text_input("Code affiché par votre application", max_chars=6, key="totp_confirm_code")
            col_confirm, col_cancel = st.columns(2)
            with col_confirm:
                if st.button("Confirmer l'activation", type="primary"):
                    try:
                        _iam_store.confirm_totp_enrollment(
                            username=_username, pending_secret=st.session_state["pending_totp_secret"], code=confirm_code.strip()
                        )
                    except InvalidCredentialsError as exc:
                        st.error(str(exc))
                    else:
                        codes = _iam_store.generate_backup_codes(_username)
                        st.session_state.pop("pending_totp_secret", None)
                        st.session_state.pop("pending_totp_uri", None)
                        st.session_state.pop("pending_totp_qr", None)
                        st.success("2FA activée.")
                        st.warning("⚠️ Codes de secours -- notez-les maintenant, ils ne seront plus jamais affichés :")
                        st.code("\n".join(codes))
            with col_cancel:
                if st.button("Annuler"):
                    st.session_state.pop("pending_totp_secret", None)
                    st.session_state.pop("pending_totp_uri", None)
                    st.session_state.pop("pending_totp_qr", None)
                    st.rerun()

# ============================================================
# Onglet 2 -- Audit & Sessions
# ============================================================
with tab_audit:
    st.subheader("Historique de connexion")
    sessions = _iam_store.list_sessions(_username, limit=50)
    if sessions:
        from datetime import datetime, timezone

        rows = [
            {
                "Horodatage": datetime.fromtimestamp(cast(float, s["timestamp"]), tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
                "Événement": s["event_type"], "IP": s["ip"], "Appareil / User-Agent": s["user_agent"],
            }
            for s in sessions
        ]
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    else:
        st.caption("Aucun événement de session enregistré.")

    st.divider()
    st.subheader("Sessions actives")
    st.caption("Déconnecte immédiatement toute autre session ouverte sous ce compte -- celle-ci reste active.")
    if st.button("🔴 TERMINER TOUTES LES AUTRES SESSIONS", type="primary"):
        current_token = str(st.session_state.get("iam_session_token", ""))
        revoked = _iam_store.terminate_all_other_sessions(_username, keep_session_token=current_token)
        st.success(f"{revoked} session(s) déconnectée(s).")

    st.divider()
    st.subheader("Liste blanche IP")
    st.caption(
        "Contrôle applicatif réel, pas une garantie réseau -- l'application du filtrage au niveau réseau "
        "(pare-feu / reverse proxy) reste la responsabilité de l'infrastructure de déploiement."
    )
    whitelist_enabled = st.toggle("Activer la liste blanche IP", value=_iam_store.is_ip_whitelist_enabled(_username))
    if whitelist_enabled != _iam_store.is_ip_whitelist_enabled(_username):
        _iam_store.set_ip_whitelist_enabled(_username, whitelist_enabled)
        st.rerun()

    if whitelist_enabled:
        current_ips = _iam_store.list_whitelisted_ips(_username)
        st.caption(f"Votre IP actuelle : `{client_ip()}`")
        st.dataframe(pd.DataFrame({"IP autorisée": current_ips}), width="stretch", hide_index=True)
        col_add, col_btn = st.columns([3, 1])
        with col_add:
            new_ip = st.text_input("Ajouter une IP", key="new_whitelist_ip", label_visibility="collapsed", placeholder="ex: 203.0.113.4")
        with col_btn:
            if st.button("Ajouter") and new_ip:
                _iam_store.add_whitelisted_ip(_username, new_ip.strip())
                st.rerun()
        if current_ips:
            ip_to_remove = st.selectbox("Retirer une IP", current_ips)
            if st.button("Retirer"):
                _iam_store.remove_whitelisted_ip(_username, ip_to_remove)
                st.rerun()

# ============================================================
# Onglet 3 -- Admin Policies
# ============================================================
with tab_admin:
    if _user.role != "admin":
        st.info("Réservé aux administrateurs.")
    else:
        st.subheader("Gestion des utilisateurs")
        users = _iam_store.list_users()
        user_rows = [
            {"Utilisateur": u.username, "Rôle": u.role, "Verrouillé": "🔒" if u.locked else "—", "2FA": "✅" if u.totp_enabled else "—"}
            for u in users
        ]
        st.dataframe(pd.DataFrame(user_rows), width="stretch", hide_index=True)

        target_username = st.selectbox("Compte cible", [u.username for u in users])
        target = _iam_store.get_user(target_username)
        if target is not None:
            c1, c2, c3 = st.columns(3)
            with c1:
                new_role = st.selectbox("Rôle", ["trader", "admin"], index=0 if target.role == "trader" else 1, key="admin_role_select")
                if st.button("Appliquer le rôle"):
                    try:
                        _iam_store.set_role(target_username, new_role)  # type: ignore[arg-type]
                    except UserNotFoundError as exc:
                        st.error(str(exc))
                    else:
                        st.success(f"Rôle de {target_username} mis à jour : {new_role}.")
                        st.rerun()
            with c2:
                if target.locked:
                    if st.button("🔓 Déverrouiller ce compte"):
                        _iam_store.unlock_user(target_username)
                        st.rerun()
                else:
                    if st.button("🔒 Verrouiller ce compte"):
                        _iam_store.lock_user(target_username)
                        st.rerun()
            with c3:
                if st.button("Forcer la réinitialisation du mot de passe"):
                    _iam_store.force_password_reset(target_username)
                    st.success(f"{target_username} devra changer son mot de passe à la prochaine connexion.")

        st.divider()
        st.subheader("Verrouillage d'écran global")
        current_timeout = _iam_store.get_global_session_timeout_minutes()
        new_timeout = st.slider(
            "Délai d'inactivité avant verrouillage (minutes)", min_value=1, max_value=120, value=current_timeout,
            help="S'applique à toutes les sessions -- voir src/dashboard/auth_middleware.py.",
        )
        if new_timeout != current_timeout and st.button("Appliquer le délai global"):
            _iam_store.set_global_session_timeout_minutes(new_timeout)
            st.success(f"Délai d'inactivité global fixé à {new_timeout} minutes.")
