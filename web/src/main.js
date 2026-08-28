/**
 * Application shell and router.
 *
 * WHAT REPLACED WHAT
 * ------------------
 * Streamlit re-executed the whole script on every interaction and rebuilt the
 * page from the top; `app.py` routed with an if/elif chain over
 * `st.session_state["current_page"]`. That model is the source of the lag this
 * migration exists to remove, so it is not reproduced. Pages render once, and
 * only the part that changed is re-rendered.
 *
 * The four pages are the same four the Streamlit app had -- landing, auth,
 * scanner, profile -- and the URL is the source of truth for which is showing,
 * so a deep link works and the back button behaves. `not_found_handling` is set
 * to single-page-application in wrangler.jsonc, which is what makes a cold load
 * of /scanner return the shell rather than a 404.
 */

import "./styles/theme.css";
import "./styles/components.css";
import "./globe/globe.css";

import { getSession, signOut } from "./api.js";
import { renderLanding } from "./pages/landing.js";
import { renderScanner } from "./pages/scanner.js";
import { renderProfile } from "./pages/profile.js";

const ROUTES = {
  "/": renderLanding,
  "/scanner": renderScanner,
  "/profile": renderProfile,
};

/** Shared application state. Deliberately small and explicitly passed. */
const state = {
  session: null,
  teardown: null,
};

/** Navigate without a page load. */
export function navigate(path) {
  if (window.location.pathname === path) return;
  window.history.pushState({}, "", path);
  render();
}

function signInHref() {
  return "/api/auth/google/start";
}

/**
 * Translate an ?auth_error= code into something a person can act on.
 *
 * The OAuth callback redirects here rather than rendering an error page,
 * because it is reached by a top-level browser navigation back from Google and
 * whatever it returns is what the visitor sees.
 */
const AUTH_ERRORS = {
  declined: "Sign-in was cancelled.",
  missing_code: "Google did not return an authorisation code. Please try again.",
  expired: "That sign-in attempt timed out. Please try again.",
  state_mismatch: "That sign-in could not be verified. Please try again.",
  exchange_failed: "Google sign-in failed. Please try again in a moment.",
  email_unverified:
    "That Google account's email address is not verified. Verify it with Google, then sign in again.",
  registration_closed:
    "This research deployment has reached its account limit, so new sign-ups are closed.",
  account_failed: "Your account could not be opened. Please try again.",
};

function renderHeader(session) {
  const signedIn = Boolean(session?.signed_in);
  const user = session?.user;

  const nav = signedIn
    ? `
      <a class="onnm-navlink" href="/scanner" data-link>Scanner</a>
      <a class="onnm-navlink" href="/profile" data-link>Profile</a>
      <button class="onnm-navlink onnm-linkbutton" type="button" data-signout>Sign out</button>
      ${
        user?.picture
          ? `<img class="onnm-avatar" src="${user.picture}" alt="" width="28" height="28" referrerpolicy="no-referrer" />`
          : ""
      }`
    : `<a class="onnm-btn onnm-btn-primary" href="${signInHref()}">Sign in with Google</a>`;

  return `
    <header class="onnm-header">
      <a class="onnm-wordmark" href="/" data-link>
        <span class="onnm-wordmark-main">ONNM</span>
        <span class="onnm-wordmark-sub">Osteosarcoma Neural Network Model</span>
      </a>
      <nav class="onnm-nav">${nav}</nav>
    </header>`;
}

function renderFooter() {
  return `
    <footer class="onnm-footer">
      <p class="onnm-footer-warning">
        <strong>Research demonstration only.</strong> ONNM is not a medical
        device and has not been clinically validated. It must not be used to
        make, confirm or delay any clinical decision. If you have a health
        concern, speak to a qualified clinician.
      </p>
      <p class="onnm-footer-meta">
        Model weights are licensed CC BY-NC 4.0. Uploaded images are never shared
        unless you explicitly consent per image. Location is recorded only at
        country level, and no IP address is stored.
      </p>
    </footer>`;
}

function showBanner(container) {
  const params = new URLSearchParams(window.location.search);
  const code = params.get("auth_error");
  if (!code) return;

  const message = AUTH_ERRORS[code] || "Sign-in did not complete. Please try again.";
  const banner = document.createElement("div");
  banner.className = "onnm-banner onnm-banner-error";
  banner.setAttribute("role", "alert");
  banner.textContent = message;
  container.prepend(banner);

  // Clear the query string so a refresh does not re-show a stale error.
  window.history.replaceState({}, "", window.location.pathname);
}

async function render() {
  const app = document.getElementById("app");

  // Let the previous page release its animation frames and observers. Without
  // this the globe keeps drawing after you route away from the landing page.
  if (state.teardown) {
    try {
      state.teardown();
    } catch {
      /* a broken teardown must not block navigation */
    }
    state.teardown = null;
  }

  const path = window.location.pathname;
  const route = ROUTES[path] || ROUTES["/"];

  app.innerHTML = `${renderHeader(state.session)}<main id="onnm-main" class="onnm-main"></main>${renderFooter()}`;
  const main = app.querySelector("#onnm-main");
  showBanner(main);

  // A signed-out visitor cannot reach the scanner or the profile. This mirrors
  // the page guards in app.py; it is a usability measure, not the security
  // boundary -- every route that touches data re-derives the account from the
  // session cookie server-side.
  if ((path === "/scanner" || path === "/profile") && !state.session?.signed_in) {
    main.insertAdjacentHTML(
      "beforeend",
      `<section class="onnm-panel">
         <h1>Sign in to continue</h1>
         <p class="onnm-muted">
           Scans are saved to your account, so this page needs you signed in.
         </p>
         <p><a class="onnm-btn onnm-btn-primary" href="${signInHref()}">Sign in with Google</a></p>
       </section>`,
    );
    return;
  }

  state.teardown = await route(main, state);
}

function wireGlobalHandlers() {
  // One delegated listener rather than one per link, so pages re-render freely.
  document.addEventListener("click", (event) => {
    const link = event.target.closest("a[data-link]");
    if (link && link.origin === window.location.origin) {
      event.preventDefault();
      navigate(link.pathname);
      return;
    }
    if (event.target.closest("[data-signout]")) {
      event.preventDefault();
      signOut()
        .catch(() => {})
        .finally(async () => {
          state.session = await getSession().catch(() => null);
          navigate("/");
          render();
        });
    }
  });

  window.addEventListener("popstate", render);
}

async function start() {
  wireGlobalHandlers();
  // Resolved before the first paint so the header never flickers from
  // signed-out to signed-in.
  state.session = await getSession().catch(() => null);
  await render();
}

start();
