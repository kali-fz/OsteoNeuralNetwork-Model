"""Country codes to globe markers.

This module is the only place in the project that turns a location into a
coordinate, and it does so from a fixed lookup table rather than from anything
a user supplied. That is deliberate and it is the whole design:

    the Worker stores a country code           (cloudflare/migrations/0004)
    the Worker returns aggregated counts       (GET /globe)
    this module attaches a coordinate          (here)

No latitude or longitude is ever recorded, transmitted, or derived from a
person. A dot on the landing-page globe is the centroid of a country that
contains at least ``k_anonymity_min`` users, nudged by a fixed offset derived
from the country code itself. It is a picture of a country, not of anybody.

WHY THE JITTER EXISTS
---------------------
Two reasons, and neither is privacy theatre -- the privacy work already
happened upstream, in the schema that cannot hold a coordinate and the endpoint
that suppresses small counts.

1. The two layers (signups, contributors) would otherwise draw one dot exactly
   on top of another for every country that appears in both, and the lower
   layer would simply be invisible.
2. A dot sitting precisely on a country's mathematical centroid reads as a
   claim about a place. Nudging it makes the mark obviously approximate, which
   is what it is.

The offset is derived by hashing the country code, so it is stable between
renders. A dot that wandered on every page load would look like live tracking,
which is the opposite of what is happening.

COVERAGE
--------
The table below is not every ISO 3166-1 code. Anything missing -- and the
Cloudflare placeholders 'T1' (Tor) and 'XX' (undetermined), which have no
location by definition -- is not silently dropped: :func:`build_markers` adds
it to the ``unplaced`` count it returns, so the caller can say "and N more"
rather than quietly under-reporting the map. Adding a country here is a
one-line change and needs no migration.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from typing import Any

# ---------------------------------------------------------------------------
# ISO 3166-1 alpha-2 -> (display name, latitude, longitude).
#
# Approximate land centroids, rounded to one decimal. They place a marker on
# the right country at globe scale and are not claimed to be more than that;
# nothing in this project depends on their precision.
# ---------------------------------------------------------------------------
COUNTRY_CENTROIDS: dict[str, tuple[str, float, float]] = {
    "AE": ("United Arab Emirates", 23.4, 53.8),
    "AF": ("Afghanistan", 33.9, 67.7),
    "AL": ("Albania", 41.2, 20.2),
    "AM": ("Armenia", 40.1, 45.0),
    "AO": ("Angola", -11.2, 17.9),
    "AR": ("Argentina", -38.4, -63.6),
    "AT": ("Austria", 47.5, 14.6),
    "AU": ("Australia", -25.3, 133.8),
    "AZ": ("Azerbaijan", 40.1, 47.6),
    "BA": ("Bosnia and Herzegovina", 43.9, 17.7),
    "BD": ("Bangladesh", 23.7, 90.4),
    "BE": ("Belgium", 50.5, 4.5),
    "BF": ("Burkina Faso", 12.2, -1.6),
    "BG": ("Bulgaria", 42.7, 25.5),
    "BH": ("Bahrain", 26.1, 50.6),
    "BI": ("Burundi", -3.4, 29.9),
    "BJ": ("Benin", 9.3, 2.3),
    "BO": ("Bolivia", -16.3, -63.6),
    "BR": ("Brazil", -14.2, -51.9),
    "BW": ("Botswana", -22.3, 24.7),
    "BY": ("Belarus", 53.7, 27.9),
    "CA": ("Canada", 56.1, -106.3),
    "CD": ("DR Congo", -4.0, 21.8),
    "CF": ("Central African Republic", 6.6, 20.9),
    "CG": ("Congo", -0.2, 15.8),
    "CH": ("Switzerland", 46.8, 8.2),
    "CI": ("Cote d'Ivoire", 7.5, -5.5),
    "CL": ("Chile", -35.7, -71.5),
    "CM": ("Cameroon", 7.4, 12.4),
    "CN": ("China", 35.9, 104.2),
    "CO": ("Colombia", 4.6, -74.3),
    "CR": ("Costa Rica", 9.7, -83.8),
    "CU": ("Cuba", 21.5, -77.8),
    "CY": ("Cyprus", 35.1, 33.4),
    "CZ": ("Czechia", 49.8, 15.5),
    "DE": ("Germany", 51.2, 10.5),
    "DK": ("Denmark", 56.3, 9.5),
    "DO": ("Dominican Republic", 18.7, -70.2),
    "DZ": ("Algeria", 28.0, 1.7),
    "EC": ("Ecuador", -1.8, -78.2),
    "EE": ("Estonia", 58.6, 25.0),
    "EG": ("Egypt", 26.8, 30.8),
    "ES": ("Spain", 40.5, -3.7),
    "ET": ("Ethiopia", 9.1, 40.5),
    "FI": ("Finland", 61.9, 25.7),
    "FR": ("France", 46.2, 2.2),
    "GA": ("Gabon", -0.8, 11.6),
    "GB": ("United Kingdom", 55.4, -3.4),
    "GE": ("Georgia", 42.3, 43.4),
    "GH": ("Ghana", 7.9, -1.0),
    "GN": ("Guinea", 9.9, -9.7),
    "GR": ("Greece", 39.1, 21.8),
    "GT": ("Guatemala", 15.8, -90.2),
    "HK": ("Hong Kong", 22.3, 114.2),
    "HN": ("Honduras", 15.2, -86.2),
    "HR": ("Croatia", 45.1, 15.2),
    "HU": ("Hungary", 47.2, 19.5),
    "ID": ("Indonesia", -0.8, 113.9),
    "IE": ("Ireland", 53.4, -8.2),
    "IL": ("Israel", 31.0, 34.9),
    "IN": ("India", 20.6, 79.0),
    "IQ": ("Iraq", 33.2, 43.7),
    "IR": ("Iran", 32.4, 53.7),
    "IS": ("Iceland", 65.0, -19.0),
    "IT": ("Italy", 41.9, 12.6),
    "JM": ("Jamaica", 18.1, -77.3),
    "JO": ("Jordan", 30.6, 36.2),
    "JP": ("Japan", 36.2, 138.3),
    "KE": ("Kenya", -0.0, 37.9),
    "KG": ("Kyrgyzstan", 41.2, 74.8),
    "KH": ("Cambodia", 12.6, 104.9),
    "KR": ("South Korea", 35.9, 127.8),
    "KW": ("Kuwait", 29.3, 47.5),
    "KZ": ("Kazakhstan", 48.0, 66.9),
    "LA": ("Laos", 19.9, 102.5),
    "LB": ("Lebanon", 33.9, 35.9),
    "LK": ("Sri Lanka", 7.9, 80.8),
    "LT": ("Lithuania", 55.2, 23.9),
    "LU": ("Luxembourg", 49.8, 6.1),
    "LV": ("Latvia", 56.9, 24.6),
    "LY": ("Libya", 26.3, 17.2),
    "MA": ("Morocco", 31.8, -7.1),
    "MD": ("Moldova", 47.4, 28.4),
    "ME": ("Montenegro", 42.7, 19.4),
    "MG": ("Madagascar", -18.8, 47.0),
    "MK": ("North Macedonia", 41.6, 21.7),
    "ML": ("Mali", 17.6, -4.0),
    "MM": ("Myanmar", 21.9, 95.9),
    "MN": ("Mongolia", 46.9, 103.8),
    "MT": ("Malta", 35.9, 14.4),
    "MU": ("Mauritius", -20.3, 57.6),
    "MV": ("Maldives", 3.2, 73.2),
    "MW": ("Malawi", -13.3, 34.3),
    "MX": ("Mexico", 23.6, -102.6),
    "MY": ("Malaysia", 4.2, 101.98),
    "MZ": ("Mozambique", -18.7, 35.5),
    "NA": ("Namibia", -22.96, 18.5),
    "NE": ("Niger", 17.6, 8.1),
    "NG": ("Nigeria", 9.1, 8.7),
    "NI": ("Nicaragua", 12.9, -85.2),
    "NL": ("Netherlands", 52.1, 5.3),
    "NO": ("Norway", 60.5, 8.5),
    "NP": ("Nepal", 28.4, 84.1),
    "NZ": ("New Zealand", -40.9, 174.9),
    "OM": ("Oman", 21.5, 55.9),
    "PA": ("Panama", 8.5, -80.8),
    "PE": ("Peru", -9.2, -75.0),
    "PH": ("Philippines", 12.9, 121.8),
    "PK": ("Pakistan", 30.4, 69.3),
    "PL": ("Poland", 51.9, 19.1),
    "PS": ("Palestine", 31.9, 35.2),
    "PT": ("Portugal", 39.4, -8.2),
    "PY": ("Paraguay", -23.4, -58.4),
    "QA": ("Qatar", 25.4, 51.2),
    "RO": ("Romania", 45.9, 25.0),
    "RS": ("Serbia", 44.0, 21.0),
    "RU": ("Russia", 61.5, 105.3),
    "RW": ("Rwanda", -1.9, 29.9),
    "SA": ("Saudi Arabia", 23.9, 45.1),
    "SD": ("Sudan", 12.9, 30.2),
    "SE": ("Sweden", 60.1, 18.6),
    "SG": ("Singapore", 1.35, 103.8),
    "SI": ("Slovenia", 46.2, 15.0),
    "SK": ("Slovakia", 48.7, 19.7),
    "SN": ("Senegal", 14.5, -14.5),
    "SO": ("Somalia", 5.2, 46.2),
    "SV": ("El Salvador", 13.8, -88.9),
    "SY": ("Syria", 34.8, 39.0),
    "TG": ("Togo", 8.6, 0.8),
    "TH": ("Thailand", 15.9, 101.0),
    "TN": ("Tunisia", 33.9, 9.6),
    "TR": ("Turkiye", 39.0, 35.2),
    "TT": ("Trinidad and Tobago", 10.7, -61.2),
    "TW": ("Taiwan", 23.7, 121.0),
    "TZ": ("Tanzania", -6.4, 34.9),
    "UA": ("Ukraine", 48.4, 31.2),
    "UG": ("Uganda", 1.4, 32.3),
    "US": ("United States", 39.8, -98.6),
    "UY": ("Uruguay", -32.5, -55.8),
    "UZ": ("Uzbekistan", 41.4, 64.6),
    "VE": ("Venezuela", 6.4, -66.6),
    "VN": ("Vietnam", 14.1, 108.3),
    "YE": ("Yemen", 15.6, 48.5),
    "ZA": ("South Africa", -30.6, 22.9),
    "ZM": ("Zambia", -13.1, 27.8),
    "ZW": ("Zimbabwe", -19.0, 29.2),
}

#: Layers the globe understands. Mirrored in the Worker's ``/globe`` payload.
SIGNUP_LAYER = "signup"
CONTRIBUTOR_LAYER = "contributor"

#: Maximum jitter in degrees. Small enough that a marker stays over its own
#: country at globe scale, large enough to separate the two layers visually.
JITTER_DEGREES = 1.8


def _jitter(code: str, layer: str) -> tuple[float, float]:
    """A fixed, reproducible offset for one country on one layer.

    Derived from a hash so it never changes between renders. A marker that
    moved on every page load would imply live tracking, and nothing here is
    tracking anybody.
    """
    digest = hashlib.sha256(f"{code}:{layer}".encode()).digest()
    # Two independent bytes -> two offsets in [-1, 1], scaled to degrees.
    lat_unit = (digest[0] / 255.0) * 2.0 - 1.0
    lng_unit = (digest[1] / 255.0) * 2.0 - 1.0
    return lat_unit * JITTER_DEGREES, lng_unit * JITTER_DEGREES


def _markers_for_layer(rows: Iterable[dict[str, Any]], layer: str) -> tuple[list[dict], int]:
    markers: list[dict] = []
    unplaced = 0
    for row in rows or []:
        code = str(row.get("country") or "").upper()
        count = int(row.get("count") or 0)
        if count <= 0:
            continue
        entry = COUNTRY_CENTROIDS.get(code)
        if entry is None:
            # 'T1', 'XX', or a country not yet in the table. Counted, not drawn.
            unplaced += count
            continue
        name, lat, lng = entry
        d_lat, d_lng = _jitter(code, layer)
        markers.append(
            {
                "lat": round(lat + d_lat, 3),
                "lng": round(lng + d_lng, 3),
                "label": name,
                "country": code,
                "count": count,
                "layer": layer,
            }
        )
    markers.sort(key=lambda m: (-m["count"], m["country"]))
    return markers, unplaced


def build_markers(payload: dict[str, Any] | None) -> dict[str, Any]:
    """Turn a ``GET /globe`` response into markers the globe component can draw.

    Returns a dict with ``markers`` (the contract the globe component consumes:
    ``lat``, ``lng``, ``label``, ``count``, ``layer``), the headline ``totals``,
    and the counts that are deliberately *not* on the map -- people in
    suppressed countries, and people whose country has no centroid here. Those
    are returned rather than dropped so the page can say "and N elsewhere"
    instead of quietly showing a smaller world than exists.

    Accepts ``None`` -- if the backend is unreachable the caller gets an empty,
    well-formed result rather than an exception. The globe is decoration on a
    landing page; it must never be the reason the page fails to render.
    """
    if not payload or not payload.get("ok"):
        return {
            "markers": [],
            "totals": {"users": 0, "contributors": 0, "approved_submissions": 0,
                       "countries_represented": 0},
            "elsewhere": {"signup": 0, "contributor": 0},
            "unplaced": {"signup": 0, "contributor": 0},
            "k_anonymity_min": None,
            "available": False,
        }

    layers = payload.get("layers") or {}
    signups = layers.get("signups") or {}
    contributors = layers.get("contributors") or {}

    signup_markers, signup_unplaced = _markers_for_layer(
        signups.get("plotted"), SIGNUP_LAYER
    )
    contributor_markers, contributor_unplaced = _markers_for_layer(
        contributors.get("plotted"), CONTRIBUTOR_LAYER
    )

    return {
        "markers": signup_markers + contributor_markers,
        "totals": payload.get("totals") or {},
        # People in countries too small to plot without identifying them.
        "elsewhere": {
            SIGNUP_LAYER: int(signups.get("elsewhere") or 0),
            CONTRIBUTOR_LAYER: int(contributors.get("elsewhere") or 0),
        },
        # People whose country code has no centroid in this table.
        "unplaced": {SIGNUP_LAYER: signup_unplaced, CONTRIBUTOR_LAYER: contributor_unplaced},
        "k_anonymity_min": payload.get("k_anonymity_min"),
        "available": True,
    }
