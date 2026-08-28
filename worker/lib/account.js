/**
 * Resolve a Google identity to the application's legacy user contract.
 *
 * The stable Google subject remains the lookup key. `user_id` is an
 * application-owned UUID used only as the primary key expected by the D1
 * Worker; it is never derived from, or substituted for, Google's `sub`.
 */
export async function resolveGoogleAccount(
  profile,
  { lookupBySubject, lookupByEmail, createUser, makeUserId = () => crypto.randomUUID() },
) {
  const account = await lookupBySubject(profile.subject);
  if (account.ok && account.payload?.user_id) {
    return { ok: true, userId: String(account.payload.user_id), created: false };
  }

  // A failed lookup is not proof that the subject is new. Only the backend's
  // explicit 404 is allowed to enter the creation path; otherwise a transient
  // D1 failure could be misread as a registration attempt.
  if (account.status !== 404) {
    return { ok: false, status: account.status, payload: account.payload };
  }

  // Preserve the legacy resolver contract for accounts that predate Google
  // subjects. Returning the existing row keeps its provider identity intact;
  // persisting a provider conversion or subject link requires an explicit
  // migration rather than an incidental sign-in side effect.
  const emailAccount = await lookupByEmail(profile.email);
  if (emailAccount.ok && emailAccount.payload?.user_id) {
    return { ok: true, userId: String(emailAccount.payload.user_id), created: false };
  }
  if (emailAccount.status !== 404) {
    return { ok: false, status: emailAccount.status, payload: emailAccount.payload };
  }

  const userId = makeUserId();
  const created = await createUser({
    user_id: userId,
    email: profile.email,
    auth_provider: "google",
    provider_subject: profile.subject,
    display_name: profile.name,
    profile_picture_url: profile.picture,
  });
  if (!created.ok) {
    // Two callbacks for one first sign-in can both observe the subject as
    // absent. If the other callback won the insert race, accept only the row
    // now keyed by this exact Google subject. A duplicate email belonging to a
    // different subject must remain a failure.
    if (created.status === 409) {
      const racedAccount = await lookupBySubject(profile.subject);
      if (racedAccount.ok && racedAccount.payload?.user_id) {
        return {
          ok: true,
          userId: String(racedAccount.payload.user_id),
          created: false,
        };
      }
    }
    return { ok: false, status: created.status, payload: created.payload };
  }

  const returnedUserId = created.payload?.user_id || created.payload?.user?.user_id;
  if (!returnedUserId || returnedUserId !== userId) {
    return {
      ok: false,
      status: 502,
      payload: { error: "account service returned an unexpected user_id" },
    };
  }

  return { ok: true, userId, created: true };
}
