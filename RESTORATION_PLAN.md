# ONNM Cloudflare Restoration Backbone

**Status:** implementation in progress

**Working branch:** `recovery/cloudflare-parity`

**Technical base:** `origin/cloudflare-migration`
**Visual and behavioural baseline:** the final Streamlit application preserved in
`app.py`, `src/theme.py`, `src/community_ui.py`, and `review_app.py`

This document is the source of truth for restoring the Cloudflare-hosted ONNM
application. Update the checkboxes and evidence notes as work lands. Do not narrow a
phase silently: move unfinished work into the next phase with a reason.

## 1. Outcome

Publish the faster Cloudflare Worker + Vite + inference-container application without
losing the product that already worked. The rebuilt application must accurately preserve
the final Streamlit experience, medical-safety path, privacy rules, account behaviour,
community review loop, and cost controls.

Vite is only the bundler. The frontend remains a small, same-origin JavaScript
application; this plan does not introduce React or another framework.

## 2. Canonical sources

When screenshots, documents, and implementations disagree, use this order:

1. Safety, privacy, identity, training-label, and cost invariants in `overview.md`.
2. Working behaviour in the final Streamlit implementation.
3. The supplied old-version screenshots and the restored Streamlit deployment.
4. `REDESIGN_BRIEF.md` for the intended warm-ivory/clinical visual direction.
5. `MIGRATION_STATUS.md` for Cloudflare architecture and operational constraints.

The old landing-page baseline is `app.py:353-576`. The photographed review flow is
implemented in `src/community_ui.py` and the full-width local console in `review_app.py`.

## 3. Non-negotiable invariants

- Signed-out visitors cannot reach inference, profile data, submissions, or review data.
- Google identity keys on the verified `sub`, not email.
- Federated accounts have no password hash; password accounts have no provider subject.
- Consent is per image and off by default. Uploading is not consent to share.
- OOD validation runs before inference and rejected inputs never receive a diagnosis.
- User feedback and suggested labels never become training labels.
- Only a human-set admin bucket and label can reach export/training.
- DICOM identifiers and standard-image metadata are removed before storage.
- Country capture remains country-only and never stores IP address or precise location.
- API, admin, inference, and session secrets never enter frontend JavaScript or HTML.
- Container cost guards remain: one instance, 90-second sleep, no keep-warm cron, and a
  fail-closed monthly runtime breaker.
- A scan that was not saved must never be presented as saved or reviewable.

## 4. Confirmed audit findings

### Release blockers

- [x] First-time Google registration supplies the required generated `user_id` to the
  existing `/users` storage contract.
- [ ] Existing-account Google sign-in is retested end to end after `5593a8b`.
- [ ] A new Google client secret is installed and the disclosed old secret is revoked.
- [x] Globe CSS is component-scoped; it must not set global `html`, `body`, `canvas`, or
  universal-selector layout rules.
- [x] Vite structural classes do not collide with generated Streamlit component classes.
- [ ] The public page can scroll to every section and footer at all target widths.
- [x] The new frontend/Worker contracts have automated tests and run in CI.

### Product-parity gaps

- [ ] Final mission-left/globe-right homepage composition and original copy.
- [ ] Model-version CTA and signed-in account name/avatar/sign-out treatment.
- [ ] Statistics, three-part scan benefit band, contributor cards, and legal links.
- [ ] Multi-image scan flow and optional model/heatmap controls.
- [ ] Feedback and rejection-dispute paths.
- [ ] Rejected-upload recording with consent-aware image retention.
- [ ] Contributor visibility toggle and profile synchronisation.
- [ ] ROC view, report/download actions, full legal content, and richer private history.
- [ ] Visible storage failure state when inference succeeds but D1 persistence fails.

### Review operations

- [ ] The pinned `kzfhero@gmail.com` administrator can review pending images.
- [ ] Bucket and ground-truth controls start unselected.
- [ ] Approve/reject/export works through the local console before public testing.
- [ ] Concurrent conflicting reviews cannot both succeed.
- [ ] `reviewed_by` is derived server-side rather than accepted from caller input.

The first public release keeps `review_app.py` as the operational review console. A
hosted `/admin` page is a later, separately security-reviewed decision; browser-side
email checks are never access control.

## 5. Implementation phases

### Phase 0 - safe branch and P0 repairs

- [x] Create `recovery/cloudflare-parity` from `origin/cloudflare-migration`.
- [x] Avoid merging polluted `main`, which contains tracked `node_modules` but no Vite
  source tree.
- [x] Repair first-time Google registration and strict verified-email handling.
- [x] Isolate generated globe CSS and rename colliding Vite layout classes.
- [x] Add focused regression tests for both fixes.
- [x] Confirm `npm run build` and Worker dry-run.

### Phase 1 - landing-page fidelity

- [x] Port the original headline, supporting copy, chips, CTA, and disclaimer.
- [x] Recreate the desktop mission/globe composition and narrow-screen stacking order.
- [x] Restore registered-user, reviewed-scan, and country totals.
- [x] Restore the three scan-benefit items and contributor roll.
- [x] Restore signed-in name, avatar, profile link, and sign-out hierarchy.
- [x] Restore complete footer links and research/medical disclaimer.
- [ ] Capture approved screenshots at 1440, 1024, 768, and 390 pixels.

GitHub counters are not required for parity with the final old deployment. Add them only
as a separately approved enhancement.

### Phase 2 - scanner, profile, and community loop

- [ ] Preserve single-scan safety behaviour while restoring multi-file support.
- [ ] Restore threshold, Grad-CAM class, opacity, attention-floor, and colour-map controls.
- [ ] Restore accepted-result feedback and OOD rejection dispute.
- [ ] Record consent-aware OOD rejection rows correctly.
- [ ] Show whether a result was saved/shared and explain persistence failures.
- [ ] Restore contributor visibility settings and accurate profile synchronisation.
- [ ] Restore report, overlay, JSON, legal, and detailed-history views.

### Phase 3 - review hardening

- [ ] Exercise local pending queues for `valid_bone`, `misc`, and `contradiction`.
- [ ] Make review updates atomic and return 409 to a losing concurrent reviewer.
- [ ] Derive review actor from the pinned authenticated context.
- [ ] Verify approved lesion rows and OOD negatives export to separate manifests.
- [ ] Decide whether hosted admin access is actually required after the local flow passes.

### Phase 4 - automated release gate

- [ ] Unit/contract tests cover every new Worker-to-storage-Worker request and response.
- [ ] OAuth tests cover existing, new, duplicate-email, same-sub/new-email, cancelled,
  expired, tampered, and missing-secret cases.
- [ ] Scan tests cover consent on/off, OOD, persistence failure, history ownership, and
  rejection dispute.
- [ ] Browser tests cover sign-in, scan, history, review, approval, and updated status.
- [x] CI runs `npm ci`, frontend/Worker tests, `npm run build`, `npm run check`, Ruff, and
  the appropriate Python suite.
- [ ] `scripts/check_inference_parity.py` passes, including Grad-CAM peak location.
- [ ] Accessibility checks report no serious/critical findings and keyboard/NVDA manual
  checks pass.

## 6. Visual acceptance criteria

At 1440px and 1024px:

- Mission content and globe share one hero, with the globe approximately 380-460px.
- The CTA is obvious and the full hero is understandable above the fold.
- Statistics follow without an artificial 480px void.
- Moss/organic atmosphere never reduces clinical-copy contrast.

At 768px and 390px:

- Order is header, mission, CTA, globe, statistics, benefits, contributors, footer.
- Hero height is content-driven; no fixed-height clipping or scroll lock.
- No horizontal overflow, clipped canvas, detached labels, or overlapping controls.
- Interactive targets are at least 44x44px.

At every width:

- Loading, empty, disabled, validation, network-failure, result, and OOD-refusal states
  use the same visual system.
- Page title and focus update after SPA navigation.
- Motion respects `prefers-reduced-motion`.
- Globe country/count information has an adjacent accessible text representation.
- Scan history works at 200% zoom and on narrow screens.

## 7. Controlled user and stress-test sequence

1. Run browser and load tests locally against mocked D1 and a local inference container.
2. Verify one existing Google account, one genuinely new Google account, and the pinned
   administrator on the deployed staging URL.
3. Run consent-off, consent-on, OOD-rejected, disputed, and persistence-failure journeys.
4. Review and export a known pending row through the local console.
5. Invite a small supervised cohort before broad testing.
6. Exercise short production concurrency bursts only after local load tests pass; never
   add a keep-warm request.
7. Watch container runtime, breaker balance, D1 errors, authentication failures, scan
   latency, save rate, OOD rejection rate, and review backlog.

Stop the test if authentication, consent, ownership, storage truthfulness, review labels,
or spend controls diverge from the invariants above.

## 8. Required evidence before cutover

- Existing-user and first-time Google login evidence.
- D1 rows for consent off/on and rejection paths, with expected pixel retention.
- Admin approve/reject/export evidence with actor and final status.
- Hosted-versus-local inference parity report.
- Four approved responsive screenshots and a no-horizontal-overflow measurement.
- Keyboard, reduced-motion, 200%-zoom, and screen-reader notes.
- Frontend/Worker/Python test output and production build/dry-run output.
- Container cost/breaker measurements under the controlled concurrency test.
- Confirmation that the old Streamlit deployment remains available until all gates pass.

## 9. Update protocol

For every implementation batch:

1. Mark only completed checkboxes.
2. Add the validation command and outcome to the commit or handoff.
3. Record any deliberately deferred item and its reason.
4. Never claim deployment, visual parity, authentication, persistence, or stress-test
   success without direct evidence.

## 10. Implementation evidence

### 2026-08-28 - P0 repair and first landing-page restoration

- `npm run test:auth`: 10/10 account-contract tests passed, including first-time
  creation, legacy email fallback, non-404 failures, and concurrent-create recovery.
- `npm run build`: Vite production build passed (147 modules transformed).
- `npm run check -- --containers-rollout=none`: Worker bundle and bindings passed a
  non-deploying Wrangler dry-run. Docker image construction remains a later release-gate
  check on a machine with Docker running.
- Focused Python regressions: 55/55 passed across OAuth accounts, contributor profiles,
  UI regressions, and globe fallback/resize behaviour.
- Local desktop visual review confirmed the old mission/globe composition, original
  wording, CTA, current public totals, benefit band, contributor roll, and footer.
- A post-implementation review then added an honest globe API-failure state, responsive
  resize/cleanup, skip-to-main, route-specific titles/focus, and 44px interactive targets.
- GitHub Actions now installs the locked Node dependencies, runs authentication contracts,
  builds the frontend, and performs a non-deploying Worker bundle check alongside the
  existing Ruff and Python jobs.
- The footer is deliberately labelled as a legal/privacy summary. Full Cloudflare-specific
  policy text remains a Phase 2 publication gate because the old `src/legal.py` wording
  describes Streamlit/local-storage behaviour that is no longer accurate.
- Responsive screenshots, end-to-end Google redirects, admin review, full inference
  parity, and deployment remain deliberately unclaimed and open above.
