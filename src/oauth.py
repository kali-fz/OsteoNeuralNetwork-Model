"""Google Sign-In for the hosted app, via Streamlit's native OIDC support.

WHY FEDERATED LOGIN AT ALL
--------------------------
The password path works, but it makes this project the custodian of other
people's credentials: hashes to store, resets to implement, breaches to be
responsible for. For a research prototype that will be handed round to a
handful of testers, that is a liability with no matching benefit. Delegating to
Google means ONNM never receives a password, so there is none to leak.

WHAT IS STORED
--------------
Only what identity requires: the email address and Google's ``sub`` claim. No
token, no refresh token, no profile picture, nothing that would let this app act
on the user's Google account. The ID token is verified and consumed by
Streamlit; nothing here ever sees the user's Google password, and Google's own
credentials never reach Cloudflare.

CONFIGURATION
-------------
``.streamlit/secrets.toml`` (or the Streamlit Cloud secrets box)::

    [auth]
    redirect_uri  = "https://<your-app>.streamlit.app/oauth2callback"
    cookie_secret = "<64 random hex characters>"

    [auth.google]
    client_id           = "<id>.apps.googleusercontent.com"
    client_secret       = "<secret>"
    server_metadata_url = "https://accounts.google.com/.well-known/openid-configuration"

Absent that block the module reports itself unconfigured and the app falls back
to password login, which is what a local run should do -- OAuth against
localhost would otherwise need its own Google client for no gain.
"""

from __future__ import annotations

import logging
from typing import Any

import streamlit as st

from backend import get_or_create_oauth_user

logger = logging.getLogger(__name__)

PROVIDER = "google"


def oidc_configured() -> bool:
    """True when a Google OAuth client is configured for this deployment.

    Deliberately tolerant: reading ``st.secrets`` raises rather than returning
    empty when no secrets file exists at all, which is the normal state of a
    fresh local checkout and must not be an error.
    """
    if not hasattr(st, "login"):
        return False  # Streamlit older than 1.42
    try:
        auth = st.secrets.get("auth", {})
    except Exception:  # noqa: BLE001 - no secrets file is a normal local state
        return False
    try:
        google = auth.get(PROVIDER, {})
        return bool(auth.get("redirect_uri") and google.get("client_id"))
    except AttributeError:
        return False


def _claims() -> Any:
    """The signed-in identity, or None. Never raises."""
    user = getattr(st, "user", None)
    if user is None:
        return None
    try:
        return user if user.get("is_logged_in") else None
    except Exception:  # noqa: BLE001
        return None


def is_signed_in() -> bool:
    return _claims() is not None


def render_sign_in() -> None:
    """The whole login screen when Google Sign-In is configured."""
    st.subheader("Sign in")
    st.caption(
        "ONNM uses Google Sign-In, so this app never receives or stores your "
        "password. Only your email address and Google's account identifier are "
        "kept, to attach your scans to you."
    )
    if st.button("Continue with Google", type="primary", use_container_width=True):
        st.login(PROVIDER)
    st.caption(
        "By continuing you accept the Terms of Service and Privacy Policy below, "
        "and acknowledge that this is an unvalidated research prototype and not "
        "a medical device."
    )


def sign_out() -> None:
    st.logout()


def resolve_account():
    """Map the signed-in Google identity to an ONNM account row.

    Returns a ``User``, or None when nobody is signed in or the identity cannot
    be trusted.

    An unverified email is refused. Google will assert an address it has not
    confirmed for some account types, and since accounts here are keyed on
    identity, accepting one would let someone claim an address they do not
    control -- and, through the email fallback in ``get_or_create_oauth_user``,
    potentially reach an existing account.
    """
    claims = _claims()
    if claims is None:
        return None

    email = (claims.get("email") or "").strip().lower()
    subject = claims.get("sub") or ""
    if not email or not subject:
        st.error("Google did not return an email address for this account.")
        return None
    if claims.get("email_verified") is False:
        st.error(
            "That Google account's email address is not verified, so it cannot "
            "be used to sign in here."
        )
        return None

    try:
        return get_or_create_oauth_user(email, str(subject), auth_provider=PROVIDER)
    except Exception as exc:  # noqa: BLE001 - surface, do not crash the app
        logger.warning("could not resolve OAuth account: %s", exc)
        st.error(f"Could not open your account: {exc}")
        return None
