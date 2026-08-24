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

Absent that block the module reports itself unconfigured. A local-only run may
still use the legacy password path, but a hosted/D1 deployment must fail closed
with a configuration message instead of quietly presenting password signup.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any
from urllib.parse import urlparse

import streamlit as st

from backend import get_or_create_oauth_user

logger = logging.getLogger(__name__)

PROVIDER = "google"
_SHARED_SETTINGS = ("redirect_uri", "cookie_secret")
_PROVIDER_SETTINGS = ("client_id", "client_secret", "server_metadata_url")


def _oidc_target() -> tuple[bool, str | None]:
    """Return whether OIDC is usable and the provider name for ``st.login``.

    Streamlit supports both the named ``[auth.google]`` layout used by ONNM's
    deployment guide and a single-provider layout with the Google settings
    directly under ``[auth]``. Supporting both prevents a valid Google setup
    from being mistaken for permission to expose the password-registration
    fallback.
    """
    if not all(hasattr(st, command) for command in ("login", "logout", "user")):
        return False, None
    try:
        auth = st.secrets.get("auth", {})
    except Exception:  # noqa: BLE001 - no secrets file is a normal local state
        return False, None
    if not isinstance(auth, Mapping):
        return False, None

    named = auth.get(PROVIDER)
    if isinstance(named, Mapping):
        provider_settings = named
        provider_name: str | None = PROVIDER
    else:
        provider_settings = auth
        provider_name = None

    shared_ready = all(bool(auth.get(key)) for key in _SHARED_SETTINGS)
    provider_ready = all(bool(provider_settings.get(key)) for key in _PROVIDER_SETTINGS)
    return shared_ready and provider_ready, provider_name


def oidc_configured() -> bool:
    """True when a Google OAuth client is configured for this deployment.

    Deliberately tolerant: reading ``st.secrets`` raises rather than returning
    empty when no secrets file exists at all, which is the normal state of a
    fresh local checkout and must not be an error.
    """
    configured, _ = _oidc_target()
    return configured


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


def identity_profile() -> dict[str, str]:
    """Return display-only Google claims, with the picture URL allow-listed."""
    claims = _claims()
    if claims is None:
        return {"name": "", "picture": "", "subject": ""}
    name = str(claims.get("name") or "").strip()[:80]
    subject = str(claims.get("sub") or "")
    picture = str(claims.get("picture") or "").strip()
    try:
        parsed = urlparse(picture)
        host = (parsed.hostname or "").lower()
        if parsed.scheme != "https" or not (
            host == "googleusercontent.com" or host.endswith(".googleusercontent.com")
        ):
            picture = ""
    except ValueError:
        picture = ""
    return {"name": name, "picture": picture[:2048], "subject": subject}


def render_sign_in() -> None:
    """Render Google Sign-In, failing closed when hosted secrets are incomplete."""
    st.subheader("Sign in with Google")
    st.caption(
        "ONNM uses Google Sign-In and never receives your Google password. Your email "
        "address and Google account identifier link saved scans to your account. Your "
        "name and photo are not shown publicly unless you choose to appear as a contributor."
    )

    configured, provider_name = _oidc_target()
    if not configured:
        logger.error("Google OIDC is required here but is not fully configured")
        st.error(
            "Google Sign-In is temporarily unavailable because this deployment's "
            "authentication settings are incomplete. Please try again later."
        )
        return

    if st.button("Continue with Google", type="primary", use_container_width=True):
        if provider_name is None:
            st.login()
        else:
            st.login(provider_name)
    st.caption(
        "By continuing, you accept the Terms of Service and Privacy Policy below. "
        "You also acknowledge that this is an unvalidated research prototype, not "
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
