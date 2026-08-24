"""Streamlit widgets for the community feedback loop.

Kept out of ``app.py`` because that file is already long, and because these
functions are the whole loop and are easier to reason about together:

    render_share_consent      -> opt-in, default OFF
    record_submission         -> a prediction, with the image only on consent
    record_rejection          -> an upload the OOD gate refused (the misc bucket)
    render_feedback           -> "this looks wrong" (a signal, not a label)
    render_rejection_dispute  -> "this really is a radiograph" (also a signal)
    render_admin_review       -> a human assigns bucket and label (the gate)

The ordering matters. Nothing a user does in the first five functions can put a
label into training. Only ``render_admin_review`` can, it is reachable by
exactly one account, and it cannot approve anything without stating both a
bucket and a class -- the Worker refuses, and behind it the database refuses.

THE THREE BUCKETS
-----------------
Every shared submission is triaged into one of three queues, and the review
decision is a different question in each:

    valid_bone     "which class is this lesion?"      -> retrains the classifier
    misc           "is this really not a radiograph?" -> retrains the OOD gate
    contradiction  "which half of the system was wrong?"

They are separate tabs rather than one list precisely because the questions
differ. A single scrolling queue is how a reviewer ends up assigning a clinical
class to a photograph of a car park out of sheer momentum.
"""

from __future__ import annotations

from datetime import timedelta

import streamlit as st

from community import (
    ADMIN_EMAIL,
    BUCKET_CONTRADICTION,
    BUCKET_MISC,
    BUCKET_TITLES,
    BUCKET_VALID_BONE,
    BUCKETS,
    MISC_LABEL,
    VALID_LABELS,
    decode_shared_image,
    encode_image_for_sharing,
    encode_payload_for_sharing,
    get_client,
    is_admin,
)

SHARE_HELP = (
    "Stores a 256-pixel processed copy, not your original file, so a human can "
    "review it for possible use in future training. This is off by default. "
    "Never tick this for an identifiable patient radiograph."
)

#: What a reviewer may pick, per bucket. A misc row has no diagnosis, so the
#: only honest label for it is 'misc'; offering "benign" beside a photograph of
#: a hotdog is the exact affordance this whole design exists to remove.
BUCKET_LABEL_CHOICES = {
    BUCKET_VALID_BONE: list(VALID_LABELS),
    BUCKET_MISC: [MISC_LABEL],
    # A contradiction row may turn out to be either: the bucket records that the
    # gate got it wrong, the label records what the image actually was.
    BUCKET_CONTRADICTION: [*VALID_LABELS, MISC_LABEL],
}

BUCKET_PROMPTS = {
    BUCKET_VALID_BONE: (
        "The gate accepted these as radiographs. Assign the class you can defend "
        "from the image; these retrain the lesion classifier."
    ),
    BUCKET_MISC: (
        "The gate rejected these as non-radiographs. Confirming one keeps it as a "
        "negative example for the OOD detector, which currently learns from no "
        "data and relies on hand-written thresholds."
    ),
    BUCKET_CONTRADICTION: (
        "The system disagreed with itself here: the gate turned away something the "
        "user says is a radiograph, or accepted something that is not one. Each of "
        "these is a demonstrated gate failure with the image still attached, which "
        "makes them the most valuable rows in the queue."
    ),
}


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
        "Share a processed copy to help improve the model",
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


def record_rejection(user_id: str, payload: bytes, *, shared: bool,
                     filename: str = "") -> str | None:
    """Record an upload the OOD gate refused. Returns a submission id, or None.

    Rejections used to leave no trace, which meant the ``misc`` bucket could
    only ever be empty and the OOD detector could only ever be retuned by hand.
    Every hotdog someone uploads is a labelled negative example that the gate
    needs and does not otherwise have.

    Consent still governs the image: without the share tick a row is recorded
    with no pixels at all, which is worth something (it counts the misuse) and
    trains nothing. DICOM rejections are also recorded without an image --
    :func:`community.encode_payload_for_sharing` refuses them, because their
    identifiers live in headers that Pillow cannot be shown to have stripped.
    """
    client = get_client()
    if not client.enabled:
        return None

    image_b64 = digest = None
    if shared and not filename.lower().endswith((".dcm", ".dicom", ".ima")):
        try:
            image_b64, digest, _ = encode_payload_for_sharing(payload)
        except Exception:  # noqa: BLE001 - a rejected file may be anything at all
            image_b64 = digest = None

    return client.create_rejected_submission(
        user_id, shared=shared, image_b64=image_b64, image_sha256=digest,
    )


def render_feedback(submission_id: str, user_id: str, key: str) -> None:
    """Let the user dispute a result.

    This writes only to untrusted columns. The wording avoids inviting a
    diagnosis -- "what do you think this actually is" from an anonymous user is
    not evidence, and presenting it as though it were would encourage treating
    it as one during review.

    One option here does more than flag: "this is not a radiograph at all" moves
    the row into the contradiction queue, because it means the gate accepted
    something it should have turned away. Still not a label -- it is a statement
    about the gate, not about the image, and a human confirms it either way.
    """
    if not submission_id or not get_client().enabled:
        return

    state_key = f"feedback_done_{key}"
    if st.session_state.get(state_key):
        st.success("Thank you. The result has been flagged for review.")
        return

    not_a_radiograph = "This is not a radiograph at all"
    with st.expander("Does this result look wrong?"):
        st.caption(
            "This report alerts a human reviewer. It does not change the model or "
            "future predictions until the image has been reviewed."
        )
        suggestion = st.selectbox(
            "If you know what it actually is, say so (optional)",
            ["I'd rather not say", *VALID_LABELS, not_a_radiograph],
            key=f"sug_{key}",
        )
        comment = st.text_area(
            "Anything else worth knowing?", key=f"cmt_{key}", max_chars=2000,
            placeholder="e.g. this is not a radiograph at all",
        )
        if st.button("Report this result", key=f"btn_{key}"):
            if suggestion == not_a_radiograph:
                suggested = MISC_LABEL
            elif suggestion == "I'd rather not say":
                suggested = None
            else:
                suggested = suggestion
            ok = get_client().submit_feedback(
                submission_id, user_id,
                says_wrong=True,
                suggested_label=suggested,
                comment=comment or None,
            )
            if ok:
                st.session_state[state_key] = True
                st.rerun()
            else:
                st.error("We could not send the report. Your scan result is unaffected.")


def render_rejection_dispute(submission_id: str, user_id: str, key: str) -> None:
    """Let a user say the gate was wrong to reject their image.

    This is the only witness to a false rejection. Inference never ran, so there
    is no prediction to disagree with and nothing in the stored row suggests the
    gate erred; without someone saying so, a genuine radiograph turned away by a
    heuristic looks exactly like a hotdog turned away by the same heuristic.

    Pressing it moves the row from ``misc`` to ``contradiction``. It is a signal,
    not a label: the reviewer decides whether the gate or the user was right.
    """
    if not submission_id or not get_client().enabled:
        return

    state_key = f"dispute_done_{key}"
    if st.session_state.get(state_key):
        st.caption("Thank you. The image has been flagged for review.")
        return

    if st.button("This really is a radiograph", key=f"dispute_{key}",
                 help="Sends the image to a human reviewer. It does not run the model."):
        ok = get_client().submit_feedback(
            submission_id, user_id, says_wrong=True, suggested_label=None,
            comment="user disputes the out-of-distribution rejection",
        )
        if ok:
            st.session_state[state_key] = True
            st.rerun()
        else:
            st.error("Could not send that just now.")


# ---------------------------------------------------------------------------
# Admin review
# ---------------------------------------------------------------------------
def admin_can_review(user_id: str | None, email: str | None) -> bool:
    """Whether the signed-in session may see the review queue.

    Two conditions, and both are necessary. ``is_admin`` asks whether this is
    the owning account -- the same id the Worker and the schema pin. The client
    check asks whether this deployment even holds an admin key. The hosted app
    deliberately does not, so a leak of its app key cannot approve its own
    training data; the queue is opened from a local checkout instead.
    """
    return is_admin(user_id, email) and get_client().admin_enabled


def _render_queue_metrics(health: dict) -> None:
    columns = st.columns(4)
    columns[0].metric("Submissions", health.get("submissions", 0))
    columns[1].metric("Awaiting review", health.get("pending_review", 0))
    columns[2].metric("Approved", health.get("approved", 0))
    used = 100 * float(health.get("capacity_used", 0.0))
    columns[3].metric("Storage used", f"{used:.1f}%")
    if used > 80:
        st.warning("Community storage is over 80% of its cap; new shares will be refused soon.")


def _render_review_card(item: dict, bucket: str) -> None:
    """One submission: the decoded image on the left, the decision on the right."""
    sid = item["submission_id"]
    left, right = st.columns([1, 1])

    with left:
        encoded = item.get("image_b64")
        if encoded:
            # Decoded here rather than shown as base64: the whole point of the
            # queue is that a human looks at the picture, and asking anyone to
            # eyeball a 30 KB string is asking them to approve blind.
            try:
                st.image(decode_shared_image(encoded),
                         caption="as stored (256px, greyscale)", width="stretch")
            except Exception as exc:  # noqa: BLE001
                st.error(f"could not decode image: {exc}")
        else:
            st.info(
                "No image: the user did not consent to sharing, or the file was a "
                "DICOM that could not be de-identified for storage. Nothing here "
                "can be approved for training."
            )

    with right:
        st.caption(f"`{sid}`")
        if item.get("model_label") == "rejected":
            st.write("**The model did not run.** The image check rejected this upload.")
        else:
            st.write(f"Model said **{item.get('model_label')}** · "
                     f"P(lesion) = **{float(item.get('lesion_probability', 0)):.3f}**")
        if item.get("ood_score") is not None:
            st.write(f"OOD score: {float(item['ood_score']):.3f}")
        if item.get("triage_reason"):
            st.caption(f"Triaged here because {item['triage_reason']}.")
        if item.get("user_suggested_label"):
            st.write(f"User suggested: *{item['user_suggested_label']}*")
        if item.get("user_comment"):
            st.info(item["user_comment"])
        st.caption(
            "Treat the user's suggestion as context, not evidence. Assign the bucket and "
            "label you can defend from the image."
        )

        # No preselection, on either control. A radio already sitting on the
        # automatic guess is an "approve as-is" button wearing a disguise: it
        # would feed the gate its own output and teach it nothing but confidence
        # in what it already believed.
        chosen_bucket = st.radio(
            "Bucket",
            BUCKETS,
            index=None,
            format_func=lambda b: BUCKET_TITLES[b],
            key=f"bucket_{sid}",
            help="Confirm the triage, or move the row. The automatic bucket is the "
                 "guess of the system being retrained, so it is never preselected.",
            horizontal=False,
        )
        choices = BUCKET_LABEL_CHOICES.get(chosen_bucket or bucket, list(VALID_LABELS))
        label = st.selectbox(
            "Ground truth", ["Select a label", *choices], key=f"lbl_{sid}",
            help="'misc' means the image is not a bone radiograph at all.",
        )
        note = st.text_input("Reviewer note (optional)", key=f"note_{sid}")

        approve, reject = st.columns(2)
        client = get_client()
        if approve.button("Approve for training", key=f"ok_{sid}", type="primary"):
            if chosen_bucket is None:
                st.error("Confirm which bucket this belongs in before approving.")
            elif label == "Select a label":
                st.error("Choose the ground-truth label before approving.")
            elif not item.get("image_b64"):
                st.error("There is no image on this row, so there is nothing to train on.")
            else:
                try:
                    ok, message = client.review_submission(
                        sid, decision="approved", admin_label=label,
                        admin_bucket=chosen_bucket, note=note or None,
                    )
                except ValueError as exc:
                    ok, message = False, str(exc)
                if ok:
                    st.success("Approved.")
                    community_status.clear()
                    st.rerun()
                else:
                    st.error(message)
        if reject.button("Reject", key=f"no_{sid}",
                         help="Discards the row. Use for anything unusable, or for "
                              "an image that should not have been shared at all."):
            ok, message = client.review_submission(
                sid, decision="rejected", note=note or None
            )
            if ok:
                st.success("Rejected.")
                community_status.clear()
                st.rerun()
            else:
                st.error(message)


def _render_bucket_tab(bucket: str, pending_count: int) -> None:
    st.caption(BUCKET_PROMPTS[bucket])
    pending = get_client().pending_review(limit=25, with_images=True, bucket=bucket)
    if not pending:
        st.success(f"Nothing awaiting review in {BUCKET_TITLES[bucket].lower()}.")
        return
    st.caption(
        f"{len(pending)} of {pending_count} shown. Disputed results appear first because "
        "those carry information the model does not already have."
    )
    for item in pending:
        disputed = " · **user says this is wrong**" if item.get("user_says_wrong") else ""
        model_label = item.get("model_label")
        verdict = (
            "gate rejected it" if model_label == "rejected" else f"model said **{model_label}**"
        )
        with st.expander(f"{item['submission_id'][:8]}: {verdict}{disputed}"):
            _render_review_card(item, bucket)


def render_admin_review(user_id: str | None = None, email: str | None = None) -> None:
    """The review queue: three buckets, one tab each. One account, hardcoded.

    Approving requires choosing both a bucket and a label. There is no
    "approve as-is" button, because that is exactly the affordance that would
    let a tired reviewer wave through the model's own guess and train on it --
    which teaches the model nothing except to be more confident about what it
    already believed.

    ``user_id``/``email`` identify the signed-in session. The check here is the
    UI's; the Worker performs its own on every admin request and the D1 schema
    pins the flag to the same account, so hiding the widget is a convenience
    rather than the security boundary.
    """
    client = get_client()

    if not is_admin(user_id, email):
        # Deliberately terse. Naming the admin address would be pointless (it is
        # in the source) but enumerating what the queue holds would not be.
        st.info("Submission review is restricted to the project owner's account.")
        return

    if not client.admin_enabled:
        st.info(
            f"Signed in as {ADMIN_EMAIL}, but ONNM_ADMIN_KEY is not set in this "
            "deployment. It is deliberately absent from the hosted app, so a leak "
            "of the app's key cannot approve its own training data. Run the app "
            "from a local checkout with the admin key to review."
        )
        return

    health = client.health()
    if health:
        _render_queue_metrics(health)

    by_bucket = (health or {}).get("pending_by_bucket", {}) or {}
    tabs = st.tabs([
        f"{BUCKET_TITLES[bucket]} ({by_bucket.get(bucket, 0)})" for bucket in BUCKETS
    ])
    for tab, bucket in zip(tabs, BUCKETS, strict=True):
        with tab:
            _render_bucket_tab(bucket, by_bucket.get(bucket, 0))

    st.divider()
    st.caption(
        "Approved rows leave here through `scripts/export_batch.py`, which writes "
        "one manifest for the lesion classifier and one for the OOD detector. "
        "Nothing else reads this table for training."
    )
