/**
 * Turn a `GET /globe` response into markers the globe component can draw.
 *
 * A faithful port of `build_markers` in `src/geo.py`. The Python remains the
 * reference implementation and is covered by `tests/test_geolocation.py`; this
 * exists because the Streamlit server that used to call it is gone, and the
 * conversion must now happen at the edge.
 *
 * WHY THE CONVERSION IS DONE HERE AND NOT IN THE BROWSER
 * ------------------------------------------------------
 * It could be done in the browser -- country centroids are public data. Keeping
 * it server-side means there is still exactly one place where a location becomes
 * a coordinate, which is the property that makes the privacy claim auditable.
 * A reviewer can read one function and see the whole of what the map can reveal.
 *
 * WHAT THE MAP CAN AND CANNOT SHOW
 * --------------------------------
 * A centroid, and nothing finer. The database stores a two-letter country code
 * and never anything more precise: `cloudflare/src/worker.js:countryOf` takes
 * the country Cloudflare has already resolved at the edge, so no IP address is
 * ever seen or written, and the browser Geolocation API is never called. Codes
 * with no centroid -- 'T1' for Tor, 'XX' for undetermined, or a country not yet
 * in the table -- are counted as `unplaced` rather than dropped, so the totals
 * on the page always add up even when the map cannot show everybody.
 */

import { COUNTRY_CENTROIDS, JITTER_DEGREES } from "./centroids.js";

export const SIGNUP_LAYER = "signup";
export const CONTRIBUTOR_LAYER = "contributor";

const ENCODER = new TextEncoder();

/**
 * A fixed, reproducible offset for one country on one layer.
 *
 * Derived from a hash so it never changes between renders. A marker that moved
 * on every page load would imply live tracking, and nothing here tracks anybody.
 *
 * Byte-for-byte the same derivation as `src/geo.py:_jitter`: SHA-256 over
 * "CODE:layer", then bytes 0 and 1 mapped from [0, 255] onto [-1, 1] and scaled.
 */
async function jitter(code, layer) {
  const digest = new Uint8Array(
    await crypto.subtle.digest("SHA-256", ENCODER.encode(`${code}:${layer}`)),
  );
  const latUnit = (digest[0] / 255) * 2 - 1;
  const lngUnit = (digest[1] / 255) * 2 - 1;
  return [latUnit * JITTER_DEGREES, lngUnit * JITTER_DEGREES];
}

/** Coerce Worker count values without letting malformed data break a page. */
function nonNegativeInt(value) {
  if (typeof value === "boolean") return 0;
  const parsed = Number.parseInt(value, 10);
  return Number.isFinite(parsed) ? Math.max(0, parsed) : 0;
}

/** Round half-away-from-zero to 3 dp, matching Python's round() closely enough for display. */
function round3(value) {
  return Math.round(value * 1000) / 1000;
}

async function markersForLayer(rows, layer) {
  const markers = [];
  let unplaced = 0;

  for (const row of Array.isArray(rows) ? rows : []) {
    if (!row || typeof row !== "object") continue;
    const code = String(row.country || "").toUpperCase();
    const count = nonNegativeInt(row.count);
    if (count <= 0) continue;

    const entry = COUNTRY_CENTROIDS[code];
    if (!entry) {
      // 'T1', 'XX', or a country not yet in the table. Counted, not drawn.
      unplaced += count;
      continue;
    }

    const [name, lat, lng] = entry;
    const [dLat, dLng] = await jitter(code, layer);
    markers.push({
      lat: round3(lat + dLat),
      lng: round3(lng + dLng),
      label: name,
      country: code,
      count,
      layer,
    });
  }

  // Busiest first, then alphabetically, so the draw order is stable between
  // renders and the largest marker is painted last.
  markers.sort((a, b) => b.count - a.count || a.country.localeCompare(b.country));
  return { markers, unplaced };
}

/**
 * Build the payload the globe component consumes.
 *
 * Returns the markers plus the counts deliberately NOT on the map: people in
 * suppressed countries, and people whose country has no centroid. Those numbers
 * are surfaced rather than hidden, so the map never silently disagrees with the
 * headline totals beside it.
 */
export async function buildMarkers(payload) {
  const layers = payload?.layers || {};
  const signups = await markersForLayer(layers?.signups?.plotted, SIGNUP_LAYER);
  const contributors = await markersForLayer(
    layers?.contributors?.plotted,
    CONTRIBUTOR_LAYER,
  );

  return {
    ok: true,
    markers: [...signups.markers, ...contributors.markers],
    totals: payload?.totals || {},
    unplaced: {
      signups: signups.unplaced,
      contributors: contributors.unplaced,
    },
    suppressed: {
      signups: nonNegativeInt(layers?.signups?.suppressed_countries),
      contributors: nonNegativeInt(layers?.contributors?.suppressed_countries),
    },
    elsewhere: {
      signups: nonNegativeInt(layers?.signups?.elsewhere),
      contributors: nonNegativeInt(layers?.contributors?.elsewhere),
    },
    generated_at: payload?.generated_at || null,
  };
}
