"""Streamlit widgets for the community feedback loop.

Kept out of ``app.py`` because that file is already long, and because these
three functions are the whole loop and are easier to reason about together:

    render_share_consent  -> opt-in, default OFF
    render_feedback       -> "this looks wrong" (a signal, not a label)
    render_admin_review   -> a human assigns the true label (the gate)

The ordering matters. Nothing a user does in the first two functions can put a
label into training. Only ``render_admin_review`` can, and it cannot approve
anything without stating a class -- the database refuses.
"""

from __future__ import annotations

from datetime import timedelta

import streamlit as st

from community import VALID_LABELS, decode_shared_image, encode_image_for_sharing, get_client

SHARE_HELP = (
    "Stores the 256-pixel processed image (not your original file) so it can be "
    "reviewed and, if useful, added to future training. Off by default. "
    "Never tick this for an identifiable patient radiograph."
)


@st.cache_data(ttl=timedelta(days=1), show_spinner=False)
def community_status() -> dict | None:
    """Health of the community API, or None when disabled/unreachable.

    Cached for a day. Streamlit re-executes the whole script on every widget
    interaction, so without this an HTTP round-trip to Cloudflare would sit in
    front of every slider drag and checkbox tick -- the API would be blamed for
    making the app feel slow when it is only being asked far too often.

    A day is right because the values are decorative: submission counts and the
    storage gauge inform, they do not gate anything. The one number that must be
    current -- whether a write succeeded -- comes back from the write itself.

    Note the distinction from the *timeout*: this is how often to ask
    (a day), not how long to wait for an answer (HEALTH_TIMEOUT, 3 seconds).
    A long timeout would mean an unreachable API hangs the page.

    Call ``community_status.clear()`` to force a refresh.
    """
    client = get_client()
    return client.health() if client.enabled else None


def render_share_consent(key: str) -> bool:
    """The opt-in checkbox. Returns True only on an explicit tick.

    Default off, and deliberately not remembered across uploads: consent is
    given per image, because the user's willingness to share a teaching example
    says nothing about the next file they open.
    """
    if not get_client().enabled:
        return False
    return st.checkbox(
        "Share this image to help improve the model",
        value=False,
        key=f"share_{key}",
        help=SHARE_HELP,
    )


def record_submission(user_id, result, *, shared, preprocessed, ood_flagged=False,
                      ood_score=None, checkpoint=None) -> str | None:
    """Send one prediction to Cloudflare. Returns a submission id, or None.

    Fails soft: if the API is down the app carries on, because inference is
    local and a dead logging endpoint must not stop someone reading a film.
    """
    client = get_client()
    if not client.enabled:
        return None

    image_b64 = digest = None
    if shared:
        try:
            image_b64, digest, _ = encode_image_for_sharing(preprocessed)
        except Exception as exc:  # noqa: BLE001 - never break a prediction over logging
            st.caption(f"Could not prepare the image for sharing: {exc}")
            return None

    return client.create_submission(
        user_id, result,
        shared=shared, image_b64=image_b64, image_sha256=digest,
        ood_flagged=ood_flagged, ood_score=ood_score, checkpoint=checkpoint,
    )


def render_feedback(submission_id: str, user_id: str, key: str) -> None:
    """Let the user dispute a result.

    This writes only to untrusted columns. The wording avoids inviting a
    diagnosis -- "what do you think this actually is" from an anonymous user is
    not evidence, and presenting it as though it were would encourage treating
    it as one during review.
    """
    if not submission_id or not get_client().enabled:
        return

    state_key = f"feedback_done_{key}"
    if st.session_state.get(state_key):
        st.success("Thanks — flagged for review.")
        return

    with st.expander("Does this result look wrong?"):
        st.caption(
            "Your report is a flag for a human reviewer, not a correction applied "
            "to the model. Nothing you enter here changes future predictions until "
            "someone has reviewed the image."
        )
        suggestion = st.selectbox(
            "If you know what it actually is, say so (optional)",
            ["I'd rather not say", *VALID_LABELS],
            key=f"sug_{key}",
        )
        comment = st.text_area(
            "Anything else worth knowing?", key=f"cmt_{key}", max_chars=2000,
            placeholder="e.g. this is not a radiograph at all",
        )
        if st.button("Report this result", key=f"btn_{key}"):
            ok = get_client().submit_feedback(
                submission_id, user_id,
                says_wrong=True,
                suggested_label=None if suggestion == "I'd rather not say" else suggestion,
                comment=comment or None,
            )
            if ok:
                st.session_state[state_key] = True
                st.rerun()
            else:
                st.error("Could not send that just now — your result is unaffected.")


def render_admin_review() -> None:
    """The review queue. Visible only when ONNM_ADMIN_KEY is set.

    Approving requires choosing a label. There is no "approve as-is" button,
    because that is exactly the affordance that would let a tired reviewer wave
    through the model's own guess and train on it -- which teaches the model
    nothing except to be more confident about what it already believed.
    """
    client = get_client()
    if not client.admin_enabled:
        st.info(
            "Set ONNM_ADMIN_KEY to review submissions. It is deliberately not "
            "configured in the hosted app, so a leak of the app's key cannot "
            "approve its own training data."
        )
        return

    health = client.health()
    if health:
        columns = st.columns(4)
        columns[0].metric("Submissions", health.get("submissions", 0))
        columns[1].metric("Awaiting review", health.get("pending_review", 0))
        columns[2].metric("Approved", health.get("approved", 0))
        used = 100 * float(health.get("capacity_used", 0.0))
        columns[3].metric("Storage used", f"{used:.1f}%")
        if used > 80:
            st.warning("Community storage is over 80% of its cap; new shares will be refused soon.")

    pending = client.pending_review(limit=25, with_images=True)
    if not pending:
        st.success("Nothing awaiting review.")
        return

    st.caption(
        f"{len(pending)} awaiting review — disputed results first, since those "
        "carry information the model does not already have."
    )

    for item in pending:
        sid = item["submission_id"]
        disputed = " · **user says this is wrong**" if item.get("user_says_wrong") else ""
        ood = " · flagged out-of-distribution" if item.get("ood_flagged") else ""
        with st.expander(f"{sid[:8]} — model said **{item.get('model_label')}**{disputed}{ood}"):
            left, right = st.columns([1, 1])
            with left:
                if item.get("image_b64"):
                    try:
                        st.image(decode_shared_image(item["image_b64"]),
                                 caption="as the model saw it (256px)", width="stretch")
                    except Exception as exc:  # noqa: BLE001
                        st.error(f"could not decode image: {exc}")
            with right:
                st.write(f"P(lesion) = **{float(item.get('lesion_probability', 0)):.3f}**")
                if item.get("user_suggested_label"):
                    st.write(f"User suggested: *{item['user_suggested_label']}*")
                if item.get("user_comment"):
                    st.info(item["user_comment"])
                st.caption(
                    "The user's suggestion is context, not evidence. Assign the label "
                    "you can defend from the image."
                )

                label = st.selectbox(
                    "Ground truth", ["— choose —", *VALID_LABELS], key=f"lbl_{sid}"
                )
                note = st.text_input("Reviewer note (optional)", key=f"note_{sid}")

                approve, reject = st.columns(2)
                if approve.button("Approve for training", key=f"ok_{sid}", type="primary"):
                    if label == "— choose —":
                        st.error("Choose the ground-truth label before approving.")
                    else:
                        ok, message = client.review_submission(
                            sid, decision="approved", admin_label=label, note=note or None
                        )
                        st.success("Approved.") if ok else st.error(message)
                        if ok:
                            st.rerun()
                if reject.button("Reject", key=f"no_{sid}"):
                    ok, message = client.review_submission(
                        sid, decision="rejected", note=note or None
                    )
                    st.success("Rejected.") if ok else st.error(message)
                    if ok:
                        st.rerun()
