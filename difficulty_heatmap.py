"""Render a Folium heatmap of sport-climbing route difficulty.

For every JSON file in ``data/`` we:
  1. keep only routes whose ``type`` includes ``"Sport"``,
  2. translate each YDS grade to a numeric value (each full-grade jump = 4
     points, letters a/b/c/d are 1 point apart inside a grade),
  3. average the numeric difficulty per route area,
  4. render each area as its own single-hue heat source on a Folium map.
     The hue is picked from a cool-to-hot colormap by the area's mean
     difficulty; the alpha alone fades with distance from the GPS point,
     so an area's colour stays constant inside its blob.

Run::

    python difficulty_heatmap.py

The output is written to ``route_difficulty_heatmap.html``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

import branca.colormap as cm
import folium
from branca.element import MacroElement
from jinja2 import Template


# ---------------------------------------------------------------------------
# YDS -> numeric translation
# ---------------------------------------------------------------------------

# Each full grade is 4 points; letters a, b, c, d are 1 point apart.
LETTER_OFFSET = {"a": 0.0, "b": 1.0, "c": 2.0, "d": 3.0}

# 5.0 .. 5.15d  =>  0 .. 63
MIN_NUMERIC = 0
MAX_NUMERIC = 15 * 4 + LETTER_OFFSET["d"]  # 63

# Recognised forms:
#   5.0 .. 5.9                 (single digit, no letter)
#   5.10 .. 5.15               (no letter)
#   5.10a .. 5.15d             (letter)
#   5.10a/b, 5.10b/c, ...      (slashed letters)
#   5.10/11                    (slashed full grades)
#   5.10+, 5.10-               (plus / minus modifier)
GRADE_RE = re.compile(
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

    match = GRADE_RE.match(grade.strip().lower())
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

    if score < MIN_NUMERIC or score > MAX_NUMERIC:
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

    # ``MAX_NUMERIC`` is a float (because LETTER_OFFSET is), so coerce back
    # to int after clamping so we can use the result as a list index.
    rounded = int(max(MIN_NUMERIC, min(MAX_NUMERIC, int(round(score)))))

    major = rounded // 4
    letter_idx = rounded % 4

    # 5.0 .. 5.9 don't use letter suffixes.
    if major < 10:
        return f"5.{major}"
    return f"5.{major}{['a', 'b', 'c', 'd'][letter_idx]}"


# The colormap's *visible* domain.  The full YDS range still defines what's
# a valid grade (MIN/MAX_NUMERIC), but the color gradient is focused on the
# band that contains the vast majority of climbers.  Areas outside this band
# saturate to the gradient endpoints — branca's LinearColormap clamps inputs
# below vmin / above vmax to the endpoint colors automatically, so no extra
# code is needed to handle the tails.  Hard-coded (not derived from data)
# so that two crags with the same mean grade always render in the same
# colour, regardless of which JSON files happen to be loaded.
SCALE_MIN_GRADE = "5.7"
SCALE_MAX_GRADE = "5.14d"
SCALE_MIN_NUMERIC = yds_to_numeric(SCALE_MIN_GRADE)   # 28.0
SCALE_MAX_NUMERIC = yds_to_numeric(SCALE_MAX_GRADE)   # 59.0


def _quantize_to_bucket(score: float) -> int:
    """Round a numeric score to its colour bucket.

    Areas in the same bucket get the same colour and can share a single
    HeatMap layer.  Bucketing on integer numeric slots gives at most ~32
    buckets across the focused [SCALE_MIN_NUMERIC, SCALE_MAX_NUMERIC]
    range — far fewer than potentially thousands of areas, which is what
    keeps the rendered HTML small.  Inputs outside the focused range
    clamp to the endpoints, matching the colormap's own saturation
    behaviour.
    """
    clamped = max(SCALE_MIN_NUMERIC, min(SCALE_MAX_NUMERIC, score))
    return int(round(clamped))


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def is_sport(route: dict) -> bool:
    """Mountain Project routes have a list ``type`` (e.g. ``["Sport", "TR"]``)."""
    return "Sport" in (route.get("type") or [])


def aggregate_by_area(payload: dict) -> list[dict]:
    """Average sport-route difficulty per area, attaching GPS coordinates."""
    areas_by_name = {a["name"]: a for a in payload.get("route_areas", [])}

    scores_by_area: dict[str, list[float]] = {}
    for route in payload.get("routes", []):
        if not is_sport(route):
            continue
        score = yds_to_numeric(route.get("grade"))
        if score is None:
            continue
        scores_by_area.setdefault(route["route_area"], []).append(score)

    rows: list[dict] = []
    for name, scores in scores_by_area.items():
        area = areas_by_name.get(name)
        if not area or "gps" not in area:
            continue
        rows.append(
            {
                "name": name,
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

# Cool-to-hot palette for the legend AND the per-area colour pick.
DIFFICULTY_COLOURS = ["#2c7bb6", "#abd9e9", "#ffffbf", "#fdae61", "#d7191c"]


def _hex_to_rgb(hex_str: str) -> tuple[int, int, int]:
    h = hex_str.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


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
    """A single canvas overlay that paints N radial-gradient blobs at once."""

    _name = "BlobsCanvasLayer"

    _template = Template(
        """
        {% macro script(this, kwargs) %}
        (function () {
            var blobs   = {{ this.blobs | tojson }};
            var radius  = {{ this.radius }};
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
                    // Pad the visible bounds slightly so blobs whose centre
                    // is just off-screen still paint their visible halo.
                    var bounds = map.getBounds().pad(0.25);
                    for (var i = 0; i < blobs.length; i++) {
                        var b = blobs[i];
                        var lat = b[0], lon = b[1];
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

    def __init__(self, blobs: list[list], radius: int = 80) -> None:
        super().__init__()
        self.blobs = blobs
        self.radius = radius


def build_heatmap(rows: list[dict], output_path: str | Path) -> Path:
    if not rows:
        raise ValueError("No sport route areas with valid YDS grades were found.")

    center_lat = sum(r["lat"] for r in rows) / len(rows)
    center_lon = sum(r["lon"] for r in rows) / len(rows)

    fmap = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=14,
        tiles="OpenStreetMap",
        # Render CircleMarkers (and any other vector layers) onto a single
        # shared canvas instead of one SVG element each.  Important when we
        # have hundreds of click-target dots.
        prefer_canvas=True,
    )

    # Map each area's mean difficulty to a single hue using a colormap whose
    # domain is focused on the band where the vast majority of climbers
    # operate (SCALE_MIN_GRADE .. SCALE_MAX_GRADE).  Easier areas saturate
    # to the coolest colour; harder areas saturate to the hottest — branca's
    # LinearColormap.__call__ clamps inputs outside [vmin, vmax] to the
    # endpoint colours, so no special handling is needed for the tails.
    # The endpoints are hard-coded YDS grades (not derived from data), so
    # two crags with the same mean grade always render in the same colour
    # regardless of which JSON files happen to be loaded.
    #
    # We DON'T add this colormap to the map directly: branca renders it with
    # numeric tick labels, and the internal numeric scale is meant to stay
    # invisible to the user.  See ``_build_legend_html`` below for the
    # custom YDS-labelled legend we render instead.
    colormap = cm.LinearColormap(
        DIFFICULTY_COLOURS,
        vmin=SCALE_MIN_NUMERIC,      # 5.7
        vmax=SCALE_MAX_NUMERIC,      # 5.14d
    )

    # Group areas by their colour bucket and emit ONE HeatMap layer per
    # bucket (capped at ~32) instead of one per area (potentially
    # thousands).  Each bucket's layer uses a one-colour gradient where
    # only the alpha changes with intensity: centre = full-alpha-colour,
    # edge = same colour but transparent.  Because every stop in the
    # gradient shares the same RGB, every blob in the layer keeps the
    # bucket's colour and only the transparency fades with distance from
    # each GPS point.
    #
    # Side-effect of folding multiple points into one layer: clusters of
    # similar-grade crags accumulate alpha and read as a brighter blob in
    # that hue, which is a useful density cue.  leaflet-heat caps the
    # accumulated intensity at ``max`` (default 1.0), so the blob stays
    # at the gradient's maximum opacity rather than washing out.
    buckets: dict[int, list[list[float]]] = {}
    for row in rows:
        bucket = _quantize_to_bucket(row["mean_difficulty"])
        buckets.setdefault(bucket, []).append([row["lat"], row["lon"], 1.0])

    for bucket, points in buckets.items():
        r, g, b = _hex_to_rgb(colormap(bucket))
        single_hue_gradient = {
            0.05: f"rgba({r},{g},{b},0)",     # far edge: transparent
            0.40: f"rgba({r},{g},{b},0.35)",
            0.70: f"rgba({r},{g},{b},0.65)",
            1.00: f"rgba({r},{g},{b},0.90)",  # centre: opaque
        }

        HeatMap(
            points,
            radius=55,   # how far the heat reaches, in pixels
            blur=40,     # softness of the alpha falloff
            min_opacity=0,
            max_zoom=1,  # treat the intensity as already normalised
            gradient=single_hue_gradient,
        ).add_to(fmap)


    # A small black dot at each area's GPS point makes the source obvious
    # and gives a click target for the popup.  Pack every area into a
    # single GeoJson layer instead of calling ``folium.CircleMarker(...)``
    # once per area: a per-area call generates ~15 lines of JS in the
    # rendered HTML, which for thousands of areas is megabytes of script
    # for the browser to parse.  GeoJson packs the whole set into one
    # FeatureCollection plus one ``pointToLayer`` callback.  Popups must
    # show YDS grades, never the internal numeric score.
    features = []
    for row in rows:
        mean_yds = numeric_to_yds(row["mean_difficulty"])
        min_yds = numeric_to_yds(row["min_difficulty"])
        max_yds = numeric_to_yds(row["max_difficulty"])
        grade_range = (
            min_yds if min_yds == max_yds else f"{min_yds} – {max_yds}"
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
                    "mean_grade": mean_yds,
                    "range": grade_range,
                },
            }
        )

    folium.GeoJson(
        {"type": "FeatureCollection", "features": features},
        name="Areas",
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
            aliases=["Area", "Sport routes", "Mean grade", "Range"],
            max_width=320,
            localize=False,
        ),
    ).add_to(fmap)

    fmap.get_root().html.add_child(folium.Element(_build_legend_html()))

    output_path = Path(output_path)
    fmap.save(str(output_path))
    return output_path


def _build_legend_html() -> str:
    """Return an HTML overlay that mirrors the colormap with YDS tick labels.

    We can't reuse branca's built-in legend: it renders ticks from ``vmin``
    to ``vmax`` as raw numbers, which would expose the internal numeric
    scale.  Instead, we paint a CSS gradient that matches
    :data:`DIFFICULTY_COLOURS` and pin a handful of recognisable YDS grades
    at the same proportional positions they occupy on the (focused)
    numeric scale.
    """
    # Reference grades to label.  Endpoints anchor the bar; the rest cover
    # the range climbers actually care about.  All ticks must fall inside
    # [SCALE_MIN_NUMERIC, SCALE_MAX_NUMERIC] — anything outside that band
    # would map to a position past the gradient and be confusing.
    tick_grades = [
        "5.7", "5.8", "5.9", "5.10a", "5.11a", "5.12a", "5.13a", "5.14a", "5.14d",
    ]

    span = SCALE_MAX_NUMERIC - SCALE_MIN_NUMERIC
    ticks = []
    for label in tick_grades:
        score = yds_to_numeric(label)
        if score is None:
            continue
        if score < SCALE_MIN_NUMERIC or score > SCALE_MAX_NUMERIC:
            continue
        pct = (score - SCALE_MIN_NUMERIC) / span * 100
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

    return f"""
    <div style="
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
            Mean route difficulty per area (scale focused on 5.7–5.14)
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

    rows: list[dict] = []
    for payload in load_all(data_dir):
        rows.extend(aggregate_by_area(payload))

    if not rows:
        raise SystemExit("No sport climbing areas with valid YDS grades found.")

    output = build_heatmap(rows, "route_difficulty_heatmap.html")
    print(f"Wrote heatmap covering {len(rows)} sport areas to {output}")


if __name__ == "__main__":
    main()
