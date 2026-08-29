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
import { renderAdmin } from "./pages/admin.js";

const ROUTES = {
  "/": renderLanding,
  "/scanner": renderScanner,
  "/profile": renderProfile,
  "/admin": renderAdmin,
};

const SITE_TITLE = "OsteoNeuralNetwork";

/**
 * Tab titles. The site name is spelled out in full rather than abbreviated to
 * ONNM, so a tab, a bookmark and the domain all read as the same thing.
 *
 * The landing page is the bare name; every other route puts the page in front
 * of it, which is what keeps four open tabs tellable apart.
 */
const ROUTE_TITLES = {
  "/": SITE_TITLE,
  "/scanner": `Scanner · ${SITE_TITLE}`,
  "/profile": `My profile · ${SITE_TITLE}`,
  "/admin": `Review queue · ${SITE_TITLE}`,
};

/** Pages a signed-out visitor cannot use. */
const SIGNED_IN_ROUTES = new Set(["/scanner", "/profile", "/admin"]);

/** Shared application state. Deliberately small and explicitly passed. */
const state = {
  session: null,
  teardown: null,
  focusMainOnRender: false,
};

/**
 * Navigate without a page load.
 *
 * `path` may carry a hash. The pathname decides which page renders and the hash
 * is applied afterwards by followHash(), once the target element exists -- the
 * browser cannot do it itself here, because on a cold load nothing is in the
 * document until the session has resolved and the route has run.
 */
export function navigate(path) {
  const [pathname, hash] = String(path).split("#");
  if (window.location.pathname === pathname && !hash) return;
  window.history.pushState({}, "", path);
  state.focusMainOnRender = true;
  render();
}

/**
 * Scroll to whatever the URL's hash names, if anything.
 *
 * Wrapped, and silent on failure: a hash that matches no element, or one that
 * is not a valid selector, must never stop a page rendering.
 */
function followHash() {
  const hash = window.location.hash;
  if (!hash || hash.length < 2) return;
  try {
    const target = document.getElementById(decodeURIComponent(hash.slice(1)));
    if (target) requestAnimationFrame(() => target.scrollIntoView({ block: "start" }));
  } catch {
    /* an unusable hash is not an error worth showing anyone */
  }
}

function signInHref() {
  return "/api/auth/google/start";
}

function escapeHtml(value) {
  return String(value ?? "").replace(
    /[&<>"']/g,
    (character) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[
        character
      ],
  );
}

function safeImageUrl(value) {
  try {
    const url = new URL(String(value || ""));
    return url.protocol === "https:" ? escapeHtml(url.href) : "";
  } catch {
    return "";
  }
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

  const picture = safeImageUrl(user?.picture);
  const accountLabel = String(user?.name || user?.email || "My profile");
  const accountName = escapeHtml(accountLabel);
  const accountInitial = escapeHtml(accountLabel.trim().slice(0, 1).toUpperCase() || "A");

  // Drawn from the flag the server computes per request, never from anything
  // stored in the browser. Hiding it is a convenience for everyone else, not a
  // control: /api/admin/* refuses any account but the owner's whatever the page
  // happens to be showing.
  const adminLink = session?.is_admin
    ? `<a class="onnm-site-navlink" href="/admin" data-link>Admin</a>`
    : "";

  const nav = signedIn
    ? `
      <a class="onnm-site-navlink" href="/scanner" data-link>Scanner</a>
      ${adminLink}
      <a class="onnm-account-chip" href="/profile" data-link>
        ${
          picture
            ? `<img class="onnm-account-chip-avatar" src="${picture}" alt="" width="32" height="32" referrerpolicy="no-referrer" />`
            : `<span class="onnm-account-chip-avatar onnm-account-chip-initials" aria-hidden="true">${accountInitial}</span>`
        }
        <span>${accountName}</span>
      </a>
      <button class="onnm-site-navlink onnm-site-linkbutton" type="button" data-signout>Sign out</button>`
    : `<a class="onnm-btn onnm-btn-primary" href="${signInHref()}">Sign in with Google</a>`;

  return `
    <header class="onnm-header">
      <a class="onnm-wordmark" href="/" data-link>
        <span class="onnm-wordmark-main">OsteoNeuralNetwork Model</span>
        <span class="onnm-wordmark-sub">Open research prototype · ONNM</span>
      </a>
      <nav class="onnm-site-nav" aria-label="Primary navigation">${nav}</nav>
    </header>`;
}

function renderFooter() {
  return `
    <footer class="onnm-site-footer" id="legal">
      <p class="onnm-site-footer-warning">
        <strong>Research demonstration only.</strong> ONNM is not a medical
        device and has not been clinically validated. It must not be used to
        make, confirm or delay any clinical decision. If you have a health
        concern, speak to a qualified clinician.
      </p>
      <p class="onnm-site-footer-meta">
        Model weights are licensed CC BY-NC 4.0. Uploaded images are never shared
        unless you explicitly consent per image. Location is recorded only at
        country level, and no IP address is stored.
      </p>
      <p class="onnm-legal-status">
        <strong>Legal and privacy overview.</strong> These are concise research-use
        notices, not the complete deployment-specific Terms or Privacy Policy.
        Publication still requires review of the full notices for this Cloudflare deployment.
      </p>
      <nav class="onnm-legal-links" aria-label="Legal and privacy information">
        <a href="#legal-terms">Research-use terms summary</a>
        <a href="#legal-privacy">Privacy summary</a>
        <a href="#legal-medical">Medical notice</a>
        <a href="#legal-cookies">Session notice</a>
      </nav>
      <div class="onnm-legal-summaries">
        <details id="legal-terms">
          <summary>Research-use terms summary</summary>
          <p>Use ONNM only for research and education. Do not use its output to diagnose, treat, or delay care, and only upload images you are authorised to use.</p>
        </details>
        <details id="legal-privacy">
          <summary>Privacy summary</summary>
          <p>Your account identifies your own saved scans. Images are retained for research review only when you explicitly consent for that image; public map data is aggregated at country level.</p>
        </details>
        <details id="legal-medical">
          <summary>Medical notice</summary>
          <p>This unvalidated prototype has no FDA, CE, or MHRA clearance. Every radiograph requires review by a qualified clinician.</p>
        </details>
        <details id="legal-cookies">
          <summary>Session notice</summary>
          <p>ONNM uses a secure, HTTP-only session cookie to keep you signed in. It is not used for advertising or cross-site tracking.</p>
        </details>
      </div>
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
  document.title = ROUTE_TITLES[path] || ROUTE_TITLES["/"];

  app.innerHTML = `<a class="onnm-skip-link" href="#onnm-main">Skip to main content</a>${renderHeader(state.session)}<main id="onnm-main" class="onnm-main" tabindex="-1"></main>${renderFooter()}`;
  const main = app.querySelector("#onnm-main");
  showBanner(main);

  const focusRouteMain = () => {
    if (!state.focusMainOnRender) return;
    state.focusMainOnRender = false;
    requestAnimationFrame(() => main.focus({ preventScroll: true }));
  };

  // A signed-out visitor cannot reach the scanner or the profile. This mirrors
  // the page guards in app.py; it is a usability measure, not the security
  // boundary -- every route that touches data re-derives the account from the
  // session cookie server-side.
  if (SIGNED_IN_ROUTES.has(path) && !state.session?.signed_in) {
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
    focusRouteMain();
    return;
  }

  // A signed-in stranger who types /admin gets the ordinary "no such page"
  // answer rather than a locked door, which is also what the API returns them.
  if (path === "/admin" && !state.session?.is_admin) {
    main.insertAdjacentHTML(
      "beforeend",
      `<section class="onnm-panel">
         <h1>Page not found</h1>
         <p class="onnm-muted">There is nothing at this address.</p>
         <p><a class="onnm-btn" href="/" data-link>Back to the home page</a></p>
       </section>`,
    );
    focusRouteMain();
    return;
  }

  const routeResult = route(main, state);
  focusRouteMain();
  state.teardown = await routeResult;
  followHash();
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

  window.addEventListener("popstate", () => {
    state.focusMainOnRender = true;
    render();
  });
}

async function start() {
  wireGlobalHandlers();
  // Resolved before the first paint so the header never flickers from
  // signed-out to signed-in.
  state.session = await getSession().catch(() => null);
  await render();
}

start();
