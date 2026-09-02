"""The globe fills the shape of the country a marker names, and no other.

**The regression.** Belgium's marker filled the whole of France, including
French Guiana on the far side of the Atlantic, while the tooltip said Belgium.
Two independent faults combined.

The first was in the data. Markers were displaced from their country's centroid
by a hash-derived offset of up to ``JITTER_DEGREES`` (1.8), whose stated purpose
was to stop the signup and contributor dots overlapping. The globe merges those
two layers into one dot per country before drawing, so there was nothing left to
separate -- and 1.8 degrees is wider than a small country. Belgium's centroid is
50.5N 4.5E; the offset put its marker at 49.6N 2.8E, in Picardy.

The second was in the renderer, and is the one that made it visible. The globe
chose which polygon to fill by asking which one contained the marker's
coordinate (``d3.geoContains``) -- geometry answering a question about identity.
The displaced point fell inside France, so France was filled.

Removing the offset alone would have hidden the second fault rather than fixed
it, because point-in-polygon is wrong even for perfect centroids: Singapore's
centroid lies inside Malaysia's polygon at this resolution, and Bahrain, Malta,
Mauritius, the Maldives, New Zealand and the Philippines have centroids in open
water between islands. Every marker already carries an ISO 3166-1 alpha-2 code,
so the shape is now looked up by that code.

These tests pin the lookup table for all 146 countries rather than for the one
that broke, because the failure is invisible until somebody from that particular
country signs up.
"""

from __future__ import annotations

import json
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]
MAP = ROOT / "web" / "src" / "globe" / "countries-110m.json"
IDS = ROOT / "web" / "src" / "globe" / "country-ids.js"
CENTROIDS = ROOT / "worker" / "lib" / "centroids.js"

# Natural Earth's 110m map omits these entirely, so no shape can be filled for
# them. Their dot and tooltip still appear. Listed explicitly so that a country
# silently falling out of the map is a failure rather than a shrug.
NO_POLYGON_IN_MAP = {"BH", "HK", "MT", "MU", "MV", "SG"}

# Same country, different spelling on the map.
MAP_NAME_ALIASES = {
    "Bosnia and Herzegovina": "Bosnia and Herz.",
    "DR Congo": "Dem. Rep. Congo",
    "Central African Republic": "Central African Rep.",
    "Cote d'Ivoire": "Côte d'Ivoire",
    "Dominican Republic": "Dominican Rep.",
    "North Macedonia": "Macedonia",
    "Turkiye": "Turkey",
    "United States": "United States of America",
    "South Sudan": "S. Sudan",
    "Equatorial Guinea": "Eq. Guinea",
    "Eswatini": "eSwatini",
    "Solomon Islands": "Solomon Is.",
}


def centroid_names() -> dict[str, str]:
    """code -> display name, read from the table the Worker actually serves."""
    source = CENTROIDS.read_text(encoding="utf-8")
    table = source[source.index("COUNTRY_CENTROIDS = {") :]
    return {
        code: name
        for code, name in re.findall(r'"([A-Z]{2})":\s*\["([^"]+)"', table)
    }


def mapped_ids() -> dict[str, str]:
    """code -> map feature id, read from the generated lookup."""
    source = IDS.read_text(encoding="utf-8")
    table = source[source.index("COUNTRY_MAP_IDS = {") :]
    return dict(re.findall(r'"([A-Z]{2})":\s*"([^"]+)"', table))


def map_features() -> dict[str, str]:
    """feature id -> country name, from the map the browser downloads."""
    topo = json.loads(MAP.read_text(encoding="utf-8"))
    return {
        str(geom["id"]): geom["properties"]["name"]
        for geom in topo["objects"]["countries"]["geometries"]
        if geom.get("id") is not None
    }


def test_every_country_we_can_plot_has_a_shape_to_fill() -> None:
    """A country with a centroid but no id would draw a dot over nothing."""
    missing = set(centroid_names()) - set(mapped_ids()) - NO_POLYGON_IN_MAP
    assert not missing, f"no map id for: {sorted(missing)}"


def test_no_country_is_wrongly_listed_as_absent_from_the_map() -> None:
    """The exception list must shrink if the map gains a country, not linger."""
    wrongly_excluded = NO_POLYGON_IN_MAP & set(mapped_ids())
    assert not wrongly_excluded, (
        f"these now have a polygon and should not be excluded: {sorted(wrongly_excluded)}"
    )


def test_every_mapped_id_exists_in_the_map_file() -> None:
    """An id that matches nothing fills nothing, silently."""
    features = map_features()
    dangling = {code: fid for code, fid in mapped_ids().items() if fid not in features}
    assert not dangling, f"ids not present in the map: {dangling}"


def test_each_country_points_at_its_own_shape() -> None:
    """The check that would have caught Belgium filling France."""
    features = map_features()
    names = centroid_names()
    wrong = {}
    for code, feature_id in mapped_ids().items():
        expected = names[code]
        actual = features[feature_id]
        if actual not in (expected, MAP_NAME_ALIASES.get(expected)):
            wrong[code] = f"{expected} -> fills {actual}"
    assert not wrong, f"countries pointing at the wrong shape: {wrong}"


def test_no_two_countries_share_a_shape() -> None:
    """Singapore filling Malaysia was exactly this, via point-in-polygon."""
    seen: dict[str, str] = {}
    clashes = []
    for code, feature_id in sorted(mapped_ids().items()):
        if feature_id in seen:
            clashes.append(f"{code} and {seen[feature_id]} both fill {feature_id}")
        seen[feature_id] = code
    assert not clashes, clashes


def test_the_renderer_selects_shapes_by_code_not_by_geometry() -> None:
    """Point-in-polygon must not creep back in as the identity test."""
    globe = (ROOT / "web" / "src" / "globe" / "globe.js").read_text(encoding="utf-8")
    start = globe.index("const activeCountries")
    block = globe[start : globe.index(".filter(Boolean)", start)]

    assert "COUNTRY_MAP_IDS[marker.country] === String(feature.id)" in block
    assert "geoContains" not in block


def test_markers_carry_the_country_code_the_lookup_needs() -> None:
    """The renderer's lookup is only possible because the payload includes it."""
    geo = (ROOT / "worker" / "lib" / "geo.js").read_text(encoding="utf-8")
    block = geo[geo.index("markers.push({") : geo.index("});", geo.index("markers.push({"))]

    assert "country: code" in block
    # And the coordinate is the centroid itself, with nothing added to it.
    assert "lat: round3(lat)" in block
    assert "lng: round3(lng)" in block
