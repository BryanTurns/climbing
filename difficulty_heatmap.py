"""Render a Folium heatmap of route difficulty per area, toggleable by climb type.

For every JSON file in ``data/`` we, for each supported climb type
(``Sport``, ``Trad``, ``Boulder``):
  1. keep only routes whose ``type`` includes the climb type's token,
  2. translate each grade to a numeric value (YDS for ropes, V-scale for
     boulders; in both cases a full-grade jump is 4 points and the
     within-grade modifiers are 1 point apart),
  3. average the numeric difficulty per route area,
  4. render each area as a colour-coded blob on a Folium map.
     The hue is picked from a cool-to-hot colormap by the area's mean
     difficulty; the alpha alone fades with distance from the GPS point,
     so an area's colour stays constant inside its blob.  Each blob's
     radius scales with the number of routes in the area (area ∝ route
     count), so dense crags read as visibly larger blobs than minor ones.

A layer-control radio in the corner switches between climb types, and the
legend at the bottom of the map updates to match the active scale.  Only
climb types that actually have data in the loaded JSON files appear in the
toggle, so a sport-only crag won't show empty Trad/Boulder buttons.

Run::

    python difficulty_heatmap.py

The output is written to ``route_difficulty_heatmap.html``.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import branca.colormap as cm
import folium
from branca.element import MacroElement
from jinja2 import Template


# ---------------------------------------------------------------------------
# YDS -> numeric translation (Sport & Trad)
# ---------------------------------------------------------------------------

# Each full grade is 4 points; letters a, b, c, d are 1 point apart.
LETTER_OFFSET = {"a": 0.0, "b": 1.0, "c": 2.0, "d": 3.0}

# 5.0 .. 5.15d  =>  0 .. 63
YDS_MIN_NUMERIC = 0
YDS_MAX_NUMERIC = 15 * 4 + LETTER_OFFSET["d"]  # 63

# Recognised forms:
#   5.0 .. 5.9                 (single digit, no letter)
#   5.10 .. 5.15               (no letter)
#   5.10a .. 5.15d             (letter)
#   5.10a/b, 5.10b/c, ...      (slashed letters)
#   5.10/11                    (slashed full grades)
#   5.10+, 5.10-               (plus / minus modifier)
YDS_GRADE_RE = re.compile(
    r"""
    ^5\.
    (?P<major>\d{1,2})
    (?:
        (?P<letter>[a-d])(?:/(?P<letter2>[a-d]))?
      | /(?P<slash_major>\d{1,2})
      | (?P<sign>[+-])
    )?
    $
    """,
    re.VERBOSE,
)


def yds_to_numeric(grade: str | None) -> float | None:
    """Translate a YDS grade string to a numeric difficulty.

    Returns ``None`` for missing, malformed, or out-of-range grades (anything
    below 5.0 or above 5.15d, or non-YDS entries like ``"V3"`` or ``"3rd"``).
    """
    if not isinstance(grade, str):
        return None

    match = YDS_GRADE_RE.match(grade.strip().lower())
    if match is None:
        return None

    major = int(match.group("major"))
    letter = match.group("letter")
    letter2 = match.group("letter2")
    slash_major = match.group("slash_major")
    sign = match.group("sign")

    base = major * 4

    if letter and letter2:
        # e.g. "5.10a/b" -> midpoint of a and b
        score = base + (LETTER_OFFSET[letter] + LETTER_OFFSET[letter2]) / 2
    elif letter:
        # e.g. "5.10c"
        score = base + LETTER_OFFSET[letter]
    elif slash_major:
        # e.g. "5.10/11" -> midpoint of 5.10 and 5.11
        score = (base + int(slash_major) * 4) / 2
    elif sign == "+":
        # 5.10+ sits in the upper half (between c and d)
        score = base + 2.5
    elif sign == "-":
        # 5.10- sits in the lower half (between a and b)
        score = base + 0.5
    else:
        # No modifier. For 5.0..5.9 there is no letter system; for 5.10+
        # without a letter, treat the grade as the middle of its band.
        score = float(base) if major < 10 else base + 1.5

    if score < YDS_MIN_NUMERIC or score > YDS_MAX_NUMERIC:
        return None
    return float(score)


def numeric_to_yds(score: float | None) -> str:
    """Translate a numeric difficulty back into a human-readable YDS grade.

    The numeric scale is an internal aggregation tool only; nothing the user
    sees should display it.  Averaged scores rarely land on an exact grade
    boundary, so we round to the nearest integer slot before converting.
    """
    if score is None:
        return "?"

    # ``YDS_MAX_NUMERIC`` is a float (because LETTER_OFFSET is), so coerce
    # back to int after clamping so we can use the result as a list index.
    rounded = int(max(YDS_MIN_NUMERIC, min(YDS_MAX_NUMERIC, int(round(score)))))

    major = rounded // 4
    letter_idx = rounded % 4

    # 5.0 .. 5.9 don't use letter suffixes.
    if major < 10:
        return f"5.{major}"
    return f"5.{major}{['a', 'b', 'c', 'd'][letter_idx]}"


# ---------------------------------------------------------------------------
# V-scale -> numeric translation (Bouldering)
# ---------------------------------------------------------------------------
#
# We mirror the YDS spacing (4 points per full grade, 1 point per modifier)
# so the bucketing/colormap machinery downstream is unchanged — only the
# scale endpoints differ between climb types.  V-grades don't use letters,
# but they DO use ``+``, ``-``, ``/`` and ``-`` (range) modifiers.

# Recognised forms:
#   VB                          (V-Beginner: easier than V0)
#   V0 .. V17                   (no modifier)
#   V0+, V0-                    (plus / minus modifier)
#   V0/1                        (slashed full grades)
#   V0-1                        (dashed range, equivalent to slashed)
V_GRADE_RE = re.compile(
    r"""
    ^v
    (?:
        b
      | (?P<major>\d{1,2})
        (?:
            /(?P<slash_major>\d{1,2})
          | -(?P<dash_major>\d{1,2})
          | (?P<sign>[+-])
        )?
    )
    $
    """,
    re.VERBOSE,
)

V_MIN_NUMERIC = -4          # VB sits one full grade below V0
# V17 is the hardest established boulder grade.  We pad the upper bound
# by the same 3-slot offset YDS uses for its trailing "d" letter so that
# bare ``V17`` (= 17*4 + 1.5, the middle of the V17 band) and ``V17+``
# (= 17*4 + 2.5) both stay inside the valid range — without the pad
# they'd land above the cap and ``v_grade_to_numeric`` would reject them.
V_MAX_NUMERIC = 17 * 4 + 3  # 71


def v_grade_to_numeric(grade: str | None) -> float | None:
    """Translate a V-scale boulder grade string to a numeric difficulty.

    Returns ``None`` for missing, malformed, or out-of-range grades, or for
    rope grades like ``"5.10a"`` that occasionally show up on Mountain
    Project routes flagged ``Boulder`` alongside another climb type.
    """
    if not isinstance(grade, str):
        return None

    cleaned = grade.strip().lower()
    if cleaned == "vb":
        return float(V_MIN_NUMERIC)

    match = V_GRADE_RE.match(cleaned)
    if match is None or match.group("major") is None:
        return None

    major = int(match.group("major"))
    slash_major = match.group("slash_major")
    dash_major = match.group("dash_major")
    sign = match.group("sign")

    base = major * 4

    if slash_major is not None:
        # e.g. "V5/6" -> midpoint of V5 and V6
        score = (base + int(slash_major) * 4) / 2
    elif dash_major is not None:
        # e.g. "V5-6" -> midpoint of V5 and V6
        score = (base + int(dash_major) * 4) / 2
    elif sign == "+":
        # V5+ sits in the upper half of the V5 band
        score = base + 2.5
    elif sign == "-":
        # V5- sits in the lower half of the V5 band
        score = base + 0.5
    else:
        # Bare "V5" -> middle of the V5 band, matching how unmodified
        # 5.10..5.15 are treated above.
        score = float(base) + 1.5

    if score < V_MIN_NUMERIC or score > V_MAX_NUMERIC:
        return None
    return float(score)


def numeric_to_v_grade(score: float | None) -> str:
    """Translate a numeric difficulty back into a human-readable V-grade."""
    if score is None:
        return "?"

    rounded = int(max(V_MIN_NUMERIC, min(V_MAX_NUMERIC, int(round(score)))))
    if rounded < 0:
        return "VB"
    return f"V{rounded // 4}"


# ---------------------------------------------------------------------------
# Climb-type configuration
# ---------------------------------------------------------------------------
#
# Each supported climb type bundles the Mountain Project ``type`` token we
# filter on, the grade <-> numeric pair, and the focused scale endpoints
# the colormap and legend use.  Adding a new climb type is a matter of
# appending another entry to ``CLIMB_TYPES`` (and providing grade
# conversion functions if it doesn't share an existing scale).

@dataclass
class ClimbTypeConfig:
    label: str                                            # display name
    type_token: str                                       # MP route ``type`` token
    # Which grade field on a route record this climb type reads. Sport
    # and Trad pull from ``yds_grade``; Boulder pulls from
    # ``boulder_grade``. Routes that carry both (e.g. a hard sport climb
    # also rated as a boulder problem) correctly contribute to each
    # climb type's aggregation through its own field.
    grade_field: str
    grade_to_numeric: Callable[[str | None], float | None]
    numeric_to_grade: Callable[[float | None], str]
    scale_min_grade: str                                  # colormap vmin
    scale_max_grade: str                                  # colormap vmax
    tick_grades: list[str]                                # legend tick labels
    legend_subtitle: str                                  # legend header text


SPORT = ClimbTypeConfig(
    label="Sport",
    type_token="Sport",
    grade_field="yds_grade",
    grade_to_numeric=yds_to_numeric,
    numeric_to_grade=numeric_to_yds,
    scale_min_grade="5.7",
    scale_max_grade="5.14d",
    tick_grades=[
        "5.7", "5.8", "5.9", "5.10a", "5.11a",
        "5.12a", "5.13a", "5.14a", "5.14d",
    ],
    legend_subtitle="Mean sport-route difficulty per area (5.7–5.14d)",
)

TRAD = ClimbTypeConfig(
    label="Trad",
    type_token="Trad",
    grade_field="yds_grade",
    grade_to_numeric=yds_to_numeric,
    numeric_to_grade=numeric_to_yds,
    # Trad climbers tend to operate at lower grades than sport, so the
    # focused band starts a couple grades lower and tops out earlier.
    # Areas outside this band still render — they just saturate to the
    # gradient endpoints.
    scale_min_grade="5.5",
    scale_max_grade="5.13a",
    tick_grades=[
        "5.5", "5.7", "5.8", "5.9", "5.10a",
        "5.11a", "5.12a", "5.13a",
    ],
    legend_subtitle="Mean trad-route difficulty per area (5.5–5.13a)",
)

BOULDER = ClimbTypeConfig(
    label="Boulder",
    type_token="Boulder",
    grade_field="boulder_grade",
    grade_to_numeric=v_grade_to_numeric,
    numeric_to_grade=numeric_to_v_grade,
    scale_min_grade="V0",
    scale_max_grade="V12",
    tick_grades=["V0", "V2", "V4", "V6", "V8", "V10", "V12"],
    legend_subtitle="Mean boulder difficulty per area (V0–V12)",
)

CLIMB_TYPES: list[ClimbTypeConfig] = [SPORT, TRAD, BOULDER]


def _scale_bounds(climb_type: ClimbTypeConfig) -> tuple[float, float]:
    """Return ``(min, max)`` numeric scores for the climb type's focused band."""
    lo = climb_type.grade_to_numeric(climb_type.scale_min_grade)
    hi = climb_type.grade_to_numeric(climb_type.scale_max_grade)
    if lo is None or hi is None:  # pragma: no cover -- config error
        raise ValueError(
            f"Scale endpoints for {climb_type.label} did not parse: "
            f"{climb_type.scale_min_grade!r}..{climb_type.scale_max_grade!r}"
        )
    return float(lo), float(hi)


def _quantize_to_bucket(score: float, scale_min: float, scale_max: float) -> int:
    """Round a numeric score to its colour bucket within the focused band.

    Areas in the same bucket get the same colour and can share a single
    HeatMap layer.  Bucketing on integer numeric slots gives at most ~32
    buckets across the [scale_min, scale_max] range — far fewer than
    potentially thousands of areas, which is what keeps the rendered HTML
    small.  Inputs outside the focused range clamp to the endpoints,
    matching the colormap's own saturation behaviour.
    """
    clamped = max(scale_min, min(scale_max, score))
    return int(round(clamped))


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_by_area(
    payload: dict, climb_type: ClimbTypeConfig
) -> list[dict]:
    """Average a climb type's route difficulty per area, attaching GPS coords.

    Aggregation keys on ``route_area_id`` -- the numeric ID embedded in
    the Mountain Project URL -- rather than the area name. Names are not
    unique (e.g. multiple crags called "The Slabs"), so keying by name
    silently merges routes from different areas into one bucket and
    drops all but one of the matching ``route_areas`` records when
    building the lookup. The ID is unique by construction.

    A route with multiple type tokens (e.g. ``["Sport", "TR"]``) counts
    toward every climb type it advertises, but only if its grade parses
    on that climb type's scale: a Boulder/Sport route graded ``5.12a``
    contributes to Sport but not to Boulder, and vice versa for V-grades.
    """
    areas_by_id = {a["id"]: a for a in payload.get("route_areas", []) if a.get("id")}

    scores_by_id: dict[str, list[float]] = {}
    for route in payload.get("routes", []):
        if climb_type.type_token not in (route.get("type") or []):
            continue
        score = climb_type.grade_to_numeric(route.get(climb_type.grade_field))
        if score is None:
            continue
        area_id = route.get("route_area_id")
        if area_id is None:
            # Older scrapes (pre route_area_id) won't aggregate; rather
            # than silently bucketing them under None, drop them.
            continue
        scores_by_id.setdefault(area_id, []).append(score)

    rows: list[dict] = []
    for area_id, scores in scores_by_id.items():
        area = areas_by_id.get(area_id)
        if not area or not area.get("gps"):
            continue
        rows.append(
            {
                "id": area_id,
                "name": area["name"],
                "lat": area["gps"]["lat"],
                "lon": area["gps"]["lon"],
                "mean_difficulty": sum(scores) / len(scores),
                "max_difficulty": max(scores),
                "min_difficulty": min(scores),
                "n_routes": len(scores),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Map rendering
# ---------------------------------------------------------------------------

# Cool-to-hot palette for the legend AND the per-area colour pick.  All
# climb types share the palette; only the scale endpoints differ, so a
# trad 5.10a and a sport 5.10a may render in different colours (each
# scale is calibrated to its own discipline's distribution).
DIFFICULTY_COLOURS = ["#2c7bb6", "#abd9e9", "#ffffbf", "#fdae61", "#d7191c"]


def _hex_to_rgb(hex_str: str) -> tuple[int, int, int]:
    h = hex_str.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


# Per-blob radius is scaled from the area's route count. We use sqrt scaling
# so the *area* of the blob (which is what the eye perceives as "size") is
# proportional to the route count rather than its radius — otherwise a
# 100-route crag would render with 100x the visual mass of a 1-route one and
# swamp the map. Bounds keep tiny areas legible and stop a single mega-area
# from eating the whole viewport at low zoom levels.
MIN_BLOB_RADIUS = 30      # pixels; floor so 1-route areas are still visible
MAX_BLOB_RADIUS = 140     # pixels; cap so a 500-route crag doesn't dominate
BLOB_RADIUS_SCALE = 22    # multiplier on sqrt(n_routes)


def _radius_for_n_routes(n_routes: int) -> float:
    """Pixel radius for an area's blob, scaled so blob area ∝ route count."""
    n = max(1, int(n_routes))
    raw = BLOB_RADIUS_SCALE * math.sqrt(n)
    return max(MIN_BLOB_RADIUS, min(MAX_BLOB_RADIUS, raw))


# ---------------------------------------------------------------------------
# Custom Leaflet layer: paint every area's blob into a single shared canvas.
# ---------------------------------------------------------------------------
#
# The previous implementation created one folium ``HeatMap`` (i.e. one
# Leaflet.heat instance, one canvas, one set of event handlers) per route
# area.  At ~600 areas that is ~600 canvases all redrawing on every pan/zoom,
# which freezes the browser.
#
# This layer takes the same per-area data — (lat, lon, rgb) — and draws all
# blobs into ONE canvas in a single pass per redraw.  Visually the result is
# identical: each blob has a constant hue with alpha fading from centre to
# edge, just like before.  Off-screen blobs are skipped (viewport culling) and
# the canvas is sized for high-DPI displays so gradients stay crisp.
class _BlobsCanvasLayer(MacroElement):
    """A single canvas overlay that paints N radial-gradient blobs at once.

    Each entry in ``blobs`` is ``[lat, lon, r, g, b, radius]`` — both the
    colour and the radius vary per blob, so areas with different mean
    difficulties get different hues and areas with different route counts
    get different sizes.
    """

    _name = "BlobsCanvasLayer"

    _template = Template(
        """
        {% macro script(this, kwargs) %}
        (function () {
            var blobs   = {{ this.blobs | tojson }};
            var map     = {{ this._parent.get_name() }};

            var BlobsLayer = L.Layer.extend({
                onAdd: function (map) {
                    this._map = map;
                    var canvas = this._canvas = L.DomUtil.create(
                        'canvas', 'leaflet-zoom-hide'
                    );
                    canvas.style.pointerEvents = 'none';
                    map.getPanes().overlayPane.appendChild(canvas);
                    map.on('moveend resize zoomend', this._reset, this);
                    this._reset();
                },
                onRemove: function (map) {
                    L.DomUtil.remove(this._canvas);
                    map.off('moveend resize zoomend', this._reset, this);
                },
                _reset: function () {
                    var map   = this._map;
                    var size  = map.getSize();
                    var dpr   = window.devicePixelRatio || 1;
                    var c     = this._canvas;
                    // Resize backing store for crisp rendering on retina.
                    if (c.width  !== size.x * dpr) c.width  = size.x * dpr;
                    if (c.height !== size.y * dpr) c.height = size.y * dpr;
                    c.style.width  = size.x + 'px';
                    c.style.height = size.y + 'px';
                    // Pin the canvas to the map's current top-left so that
                    // Leaflet's overlayPane translation moves it with pans
                    // until we redraw.
                    var topLeft = map.containerPointToLayerPoint([0, 0]);
                    L.DomUtil.setPosition(c, topLeft);
                    this._draw(dpr);
                },
                _draw: function (dpr) {
                    var ctx = this._canvas.getContext('2d');
                    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
                    ctx.clearRect(
                        0, 0, this._canvas.width, this._canvas.height
                    );
                    var map = this._map;
                    // Pad the visible bounds so blobs whose centre is just
                    // off-screen still paint their visible halo.  Padding
                    // is generous because the largest blobs can extend a
                    // long way past their centre at high zoom levels.
                    var bounds = map.getBounds().pad(0.5);
                    for (var i = 0; i < blobs.length; i++) {
                        var b = blobs[i];
                        var lat = b[0], lon = b[1];
                        var radius = b[5];
                        if (lat < bounds.getSouth() || lat > bounds.getNorth()
                         || lon < bounds.getWest()  || lon > bounds.getEast()) {
                            continue;
                        }
                        var p = map.latLngToContainerPoint([lat, lon]);
                        var rgb = b[2] + ',' + b[3] + ',' + b[4];
                        var grad = ctx.createRadialGradient(
                            p.x, p.y, 0, p.x, p.y, radius
                        );
                        // Stops mirror the previous HeatMap gradient: a soft
                        // falloff from ~0.9 alpha at centre to fully
                        // transparent at the edge.
                        grad.addColorStop(0.0, 'rgba(' + rgb + ',0.90)');
                        grad.addColorStop(0.3, 'rgba(' + rgb + ',0.65)');
                        grad.addColorStop(0.6, 'rgba(' + rgb + ',0.35)');
                        grad.addColorStop(1.0, 'rgba(' + rgb + ',0.00)');
                        ctx.fillStyle = grad;
                        ctx.fillRect(
                            p.x - radius, p.y - radius,
                            radius * 2,   radius * 2
                        );
                    }
                }
            });

            (new BlobsLayer()).addTo(map);
        })();
        {% endmacro %}
        """
    )

    def __init__(self, blobs: list[list]) -> None:
        super().__init__()
        self.blobs = blobs


def _populate_climb_type_layer(
    feature_group: folium.FeatureGroup,
    rows: list[dict],
    climb_type: ClimbTypeConfig,
) -> None:
    """Attach this climb type's difficulty blobs and click-target dots to ``feature_group``.

    Each FeatureGroup is added to the map as a base layer (``overlay=False``),
    so when the user picks a different climb type from the layer-control radio
    the previous group's BlobsCanvasLayer and CircleMarkers are removed in one
    go and the new group's are added.
    """
    if not rows:
        return

    scale_min, scale_max = _scale_bounds(climb_type)

    # Map each area's mean difficulty to a single hue using a colormap whose
    # domain is focused on this climb type's typical band.  Easier areas
    # saturate to the coolest colour; harder areas saturate to the hottest —
    # branca's LinearColormap.__call__ clamps inputs outside [vmin, vmax] to
    # the endpoint colours, so no special handling is needed for the tails.
    #
    # We DON'T add this colormap to the map directly: branca renders it with
    # numeric tick labels, and the internal numeric scale is meant to stay
    # invisible to the user.  See ``_build_legend_html`` below for the
    # custom YDS/V-labelled legend we render instead.
    colormap = cm.LinearColormap(
        DIFFICULTY_COLOURS, vmin=scale_min, vmax=scale_max,
    )

    # Build one blob per area: colour comes from the difficulty colormap
    # (quantised into discrete buckets so similar-grade crags share a hue),
    # radius comes from the area's route count (sqrt-scaled so blob *area*
    # is what's proportional to the count -- see _radius_for_n_routes).
    # All blobs paint into a single shared canvas via _BlobsCanvasLayer,
    # which keeps rendering fast at hundreds of areas.  Adding the layer
    # to the FeatureGroup (rather than directly to the map) means it gets
    # added/removed automatically when the climb-type radio toggles which
    # FeatureGroup is the active base layer.
    blobs: list[list[float]] = []
    for row in rows:
        bucket = _quantize_to_bucket(
            row["mean_difficulty"], scale_min, scale_max,
        )
        r, g, b = _hex_to_rgb(colormap(bucket))
        radius = _radius_for_n_routes(row["n_routes"])
        blobs.append([row["lat"], row["lon"], r, g, b, radius])

    _BlobsCanvasLayer(blobs).add_to(feature_group)

    # A small black dot at each area's GPS point makes the source obvious
    # and gives a click target for the popup.  Pack every area into a
    # single GeoJson layer instead of calling ``folium.CircleMarker(...)``
    # once per area: a per-area call generates ~15 lines of JS in the
    # rendered HTML, which for thousands of areas is megabytes of script
    # for the browser to parse.  GeoJson packs the whole set into one
    # FeatureCollection plus one ``pointToLayer`` callback.  Popups must
    # show climber-facing grades, never the internal numeric score.
    features = []
    for row in rows:
        mean_grade = climb_type.numeric_to_grade(row["mean_difficulty"])
        min_grade = climb_type.numeric_to_grade(row["min_difficulty"])
        max_grade = climb_type.numeric_to_grade(row["max_difficulty"])
        grade_range = (
            min_grade if min_grade == max_grade
            else f"{min_grade} – {max_grade}"
        )
        features.append(
            {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [row["lon"], row["lat"]],
                },
                "properties": {
                    "name": row["name"],
                    "n_routes": row["n_routes"],
                    "mean_grade": mean_grade,
                    "range": grade_range,
                },
            }
        )

    folium.GeoJson(
        {"type": "FeatureCollection", "features": features},
        marker=folium.CircleMarker(
            radius=3,
            color="#222",
            weight=1,
            fill=True,
            fill_color="#222",
            fill_opacity=0.85,
        ),
        popup=folium.GeoJsonPopup(
            fields=["name", "n_routes", "mean_grade", "range"],
            aliases=["Area", f"{climb_type.label} routes", "Mean grade", "Range"],
            max_width=320,
            localize=False,
        ),
    ).add_to(feature_group)


def build_heatmap(
    rows_by_type: dict[str, list[dict]],
    output_path: str | Path,
    climb_types: list[ClimbTypeConfig],
) -> Path:
    """Render the multi-climb-type heatmap to a single HTML file."""
    if not any(rows_by_type.get(ct.label) for ct in climb_types):
        raise ValueError(
            "No areas with valid grades were found in any climb type."
        )

    # Skip climb types that have zero rows so the radio toggle doesn't show
    # an empty option.  Order is preserved (Sport → Trad → Boulder), so the
    # first non-empty type becomes the default visible layer.
    active_types = [ct for ct in climb_types if rows_by_type.get(ct.label)]
    default_type = active_types[0]

    # Centre on the union of all areas across climb types so the initial
    # view fits the whole dataset, not just the default-active climb type.
    all_rows = [r for ct in active_types for r in rows_by_type[ct.label]]
    center_lat = sum(r["lat"] for r in all_rows) / len(all_rows)
    center_lon = sum(r["lon"] for r in all_rows) / len(all_rows)

    fmap = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=14,
        # Don't let folium auto-attach its default OSM tile layer: that
        # variant doesn't set a referrerPolicy, so depending on how the
        # HTML is loaded the browser sends no Referer header and OSM's
        # tile servers respond 403.  We attach an explicit TileLayer
        # below that DOES set the policy.
        tiles=None,
        # Render CircleMarkers (and any other vector layers) onto a single
        # shared canvas instead of one SVG element each.  Important when we
        # have hundreds of click-target dots.
        prefer_canvas=True,
    )

    # OpenStreetMap's tile usage policy expects a valid HTTP Referer on
    # every tile request; without one, the CDN now serves 403s.  Setting
    # ``referrer_policy`` here flows through to Leaflet's TileLayer
    # ``referrerPolicy`` option, which in turn sets ``referrerpolicy`` on
    # each generated <img> tile element so the browser emits a Referer
    # header.  ``strict-origin-when-cross-origin`` matches modern browser
    # defaults: it sends the page's origin to OSM (which is what their
    # policy wants) without leaking the full URL.
    folium.TileLayer(
        tiles="https://tile.openstreetmap.org/{z}/{x}/{y}.png",
        attr=(
            '&copy; <a href="https://www.openstreetmap.org/copyright">'
            "OpenStreetMap</a> contributors"
        ),
        name="OpenStreetMap",
        max_zoom=19,
        referrer_policy="strict-origin-when-cross-origin",
        # Keep the tile layer OUT of LayerControl: the climb-type
        # FeatureGroups below register as base layers (radio buttons), and
        # if the tile layer also showed up there Leaflet would treat OSM
        # and the climb types as mutually exclusive — selecting "Sport"
        # would deselect OSM and the map's base imagery would disappear.
        control=False,
    ).add_to(fmap)

    # One FeatureGroup per active climb type, registered as a *base layer*
    # (overlay=False) so folium's LayerControl renders them as a
    # mutually-exclusive radio group rather than independent checkboxes.
    # Only the default starts with show=True, so only its children are on
    # the map at page load; the others are created but unattached, ready
    # for L.control.layers to swap them in on user toggle.
    for ct in active_types:
        feature_group = folium.FeatureGroup(
            name=ct.label,
            overlay=False,
            show=(ct is default_type),
        )
        _populate_climb_type_layer(
            feature_group, rows_by_type[ct.label], ct,
        )
        feature_group.add_to(fmap)

    folium.LayerControl(
        position="topright",
        collapsed=False,
        # Don't autosort: we want the radio order to follow the priority
        # we put in CLIMB_TYPES (Sport, Trad, Boulder), not alphabetical.
        autoZIndex=True,
    ).add_to(fmap)

    # Render every climb type's legend in the same fixed position; only
    # the default starts visible.  The ``LegendSwapper`` macro below
    # listens for ``baselayerchange`` and toggles ``display`` accordingly.
    legend_blocks = "".join(
        _build_legend_html(ct, hidden=(ct is not default_type))
        for ct in active_types
    )
    fmap.get_root().html.add_child(folium.Element(legend_blocks))
    fmap.add_child(_LegendSwapper(default_type.label))

    output_path = Path(output_path)
    fmap.save(str(output_path))
    return output_path


class _LegendSwapper(MacroElement):
    """Swap which climb-type legend is visible when the base layer changes.

    Folium's LayerControl emits Leaflet's ``baselayerchange`` event with the
    layer's display name on ``e.name`` — that matches the ``label`` field on
    each ``ClimbTypeConfig``, which is also written into the legend block's
    ``data-climb-type`` attribute.  We just toggle ``display`` on whichever
    legend block matches.
    """

    _name = "LegendSwapper"

    _template = Template(
        """
        {% macro script(this, kwargs) %}
        (function () {
            var map = {{ this._parent.get_name() }};
            function showLegendFor(name) {
                var legends = document.querySelectorAll(
                    '.difficulty-legend[data-climb-type]'
                );
                legends.forEach(function (el) {
                    el.style.display =
                        (el.dataset.climbType === name) ? 'block' : 'none';
                });
            }
            map.on('baselayerchange', function (e) {
                showLegendFor(e.name);
            });
            // Run once on load so the default layer's legend is visible
            // even if the user never toggles.
            showLegendFor({{ this.default_label | tojson }});
        })();
        {% endmacro %}
        """
    )

    def __init__(self, default_label: str) -> None:
        super().__init__()
        self.default_label = default_label


def _build_legend_html(
    climb_type: ClimbTypeConfig, *, hidden: bool,
) -> str:
    """Return an HTML overlay that mirrors the colormap with grade tick labels.

    We can't reuse branca's built-in legend: it renders ticks from ``vmin``
    to ``vmax`` as raw numbers, which would expose the internal numeric
    scale.  Instead, we paint a CSS gradient that matches
    :data:`DIFFICULTY_COLOURS` and pin a handful of recognisable grades
    (YDS or V-scale, depending on the climb type) at the same proportional
    positions they occupy on the (focused) numeric scale.

    All climb types render legends at the same fixed position; the
    :class:`_LegendSwapper` macro toggles ``display`` so only the active
    type's legend is visible.
    """
    scale_min, scale_max = _scale_bounds(climb_type)
    span = scale_max - scale_min

    ticks: list[tuple[str, float]] = []
    for label in climb_type.tick_grades:
        score = climb_type.grade_to_numeric(label)
        if score is None:
            continue
        if score < scale_min or score > scale_max:
            continue
        pct = (score - scale_min) / span * 100
        ticks.append((label, pct))

    gradient_css = f"linear-gradient(to right, {', '.join(DIFFICULTY_COLOURS)})"

    label_spans = "".join(
        (
            f'<span style="position:absolute;left:{pct:.2f}%;'
            f'transform:translateX(-50%);font-size:11px;color:#222;">{label}</span>'
        )
        for label, pct in ticks
    )

    tick_marks = "".join(
        (
            f'<span style="position:absolute;left:{pct:.2f}%;top:0;'
            f'width:1px;height:6px;background:#444;'
            f'transform:translateX(-50%);"></span>'
        )
        for _, pct in ticks
    )

    display = "none" if hidden else "block"

    return f"""
    <div class="difficulty-legend" data-climb-type="{climb_type.label}" style="
        display: {display};
        position: fixed;
        bottom: 24px;
        left: 50%;
        transform: translateX(-50%);
        background: rgba(255,255,255,0.95);
        padding: 10px 16px 8px;
        border: 1px solid #888;
        border-radius: 4px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.2);
        z-index: 1000;
        font-family: sans-serif;
    ">
        <div style="text-align:center;font-size:12px;color:#222;margin-bottom:6px;">
            {climb_type.legend_subtitle}
        </div>
        <div style="position:relative;width:440px;">
            <div style="
                width:100%;
                height:14px;
                background:{gradient_css};
                border:1px solid #ccc;
            "></div>
            <div style="position:relative;height:6px;margin-top:0;">
                {tick_marks}
            </div>
            <div style="position:relative;height:16px;margin-top:1px;">
                {label_spans}
            </div>
        </div>
    </div>
    """


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def load_all(data_dir: Path) -> Iterable[dict]:
    for path in sorted(data_dir.glob("*.json")):
        with path.open() as fp:
            yield json.load(fp)


def main() -> None:
    data_dir = Path("data")
    if not data_dir.is_dir():
        raise SystemExit("Expected a 'data/' directory next to this script.")

    rows_by_type: dict[str, list[dict]] = {ct.label: [] for ct in CLIMB_TYPES}
    for payload in load_all(data_dir):
        for ct in CLIMB_TYPES:
            rows_by_type[ct.label].extend(aggregate_by_area(payload, ct))

    if not any(rows_by_type.values()):
        raise SystemExit(
            "No climbing areas with valid grades were found in any climb type."
        )

    output = build_heatmap(
        rows_by_type, "route_difficulty_heatmap.html", CLIMB_TYPES,
    )
    summary = ", ".join(
        f"{ct.label}: {len(rows_by_type[ct.label])}" for ct in CLIMB_TYPES
    )
    print(f"Wrote heatmap to {output} (areas per type — {summary})")


if __name__ == "__main__":
    main()
