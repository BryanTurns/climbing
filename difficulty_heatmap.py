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

    # Map each area's mean difficulty to a single hue using a colormap that
    # spans the full YDS range (5.0 .. 5.15d  =  0 .. 63), NOT the range of
    # the current data.  Pinning the scale this way means two crags with the
    # same mean grade always render in the same colour, regardless of which
    # JSON files happen to be loaded.  The same colormap is added as a
    # legend so users can read difficulty off colour.
    colormap = cm.LinearColormap(
        DIFFICULTY_COLOURS,
        vmin=MIN_NUMERIC,            # 5.0
        vmax=MAX_NUMERIC,            # 5.15d
        caption=(
            "Mean YDS score per area  "
            "(0 = 5.0 · 16 = 5.4 · 36 = 5.9 · 48 = 5.12a · 56 = 5.14a · 63 = 5.15d)"
        ),
    )

    # Build one (lat, lon, r, g, b) tuple per area and hand the whole batch
    # off to a single canvas layer.  This replaces the previous loop that
    # spawned ~600 Leaflet.heat instances — the killer of browser perf.
    blobs = [
        [row["lat"], row["lon"], *_hex_to_rgb(colormap(row["mean_difficulty"]))]
        for row in rows
    ]
    _BlobsCanvasLayer(blobs, radius=80).add_to(fmap)

    # A small black dot at each area's GPS point makes the source obvious
    # and gives a click target for the popup.
    for row in rows:
        popup = folium.Popup(
            (
                f"<b>{row['name']}</b><br>"
                f"Sport routes: {row['n_routes']}<br>"
                f"Mean YDS score: {row['mean_difficulty']:.1f}<br>"
                f"Range: {row['min_difficulty']:.1f} – {row['max_difficulty']:.1f}"
            ),
            max_width=320,
        )
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=3,
            color="#222",
            weight=1,
            fill=True,
            fill_color="#222",
            fill_opacity=0.85,
            popup=popup,
        ).add_to(fmap)

    colormap.add_to(fmap)

    output_path = Path(output_path)
    fmap.save(str(output_path))
    return output_path


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
