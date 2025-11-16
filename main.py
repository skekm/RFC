#!/usr/bin/env python3
import math
import os
import json
from pathlib import Path

import requests
from PIL import Image, ImageDraw

# ============================================================
# CONFIG
# ============================================================

# Either load from a local file:
GEOJSON_FILE = "cip_rfc_europe.geojson"   # put your downloaded FeatureCollection here

# Or fetch directly from CIP (comment out if you use local file)
CIP_GEOJSON_URL = (
    "https://cip.rne.eu/api/feature-collection/sections"
    "?source=cache"
    "&layer=131&layer=132&layer=109&layer=108&layer=107&layer=106&layer=105"
    "&layer=104&layer=103&layer=102&layer=101"
    "&showPolyline=true"
    "&propertyType=layers"
    "&left=-13.205890927977881"
    "&bottom=33.78523007002315"
    "&right=32.53507495564878"
    "&top=64.71857967286385"
)

# Approx Europe bbox matching your request (lon/lat)
BBOX_LEFT   = -13.205890927977881
BBOX_BOTTOM = 33.78523007002315
BBOX_RIGHT  = 32.53507495564878
BBOX_TOP    = 64.71857967286385

# Zoom range to generate
MIN_ZOOM = 4
MAX_ZOOM = 10

# Output directory for tiles
OUTPUT_ROOT = Path("tiles")

# RFC colors (change to your exact palette)
RFC_COLORS = {
    "101": (255,   0,   0),   # Rhine-Alpine – red
    "102": (  0, 255, 255),   # North Sea-Mediterranean – cyan
    "103": (139,  69,  19),   # Scandinavian-Mediterranean – brown
    "104": (  0,   0, 255),   # Atlantic – blue
    "105": (  0, 128,   0),   # Baltic-Adriatic – green
    "106": (255, 165,   0),   # Mediterranean – orange
    "107": (148,   0, 211),   # Orient/East-Med – purple
    "108": (  0,   0,   0),   # North Sea-Baltic – black
    "109": (255, 192, 203),   # Rhine-Danube – pink
    "131": ( 70, 130, 180),   # Amber – steelblue
    "132": (210, 105,  30),   # Alpine-Western Balkan – chocolate
}

DEFAULT_COLOR = (200, 200, 200)  # for anything not in RFC_COLORS

TILE_SIZE = 256


# ============================================================
# COORDINATE MATH: WGS84 <-> XYZ tiles
# ============================================================

def lonlat_to_tile_xy_float(lon, lat, z):
    """
    Return fractional XYZ tile coordinates for a given lon/lat at zoom z.
    """
    lat_rad = math.radians(lat)
    n = 2.0 ** z
    xtile = (lon + 180.0) / 360.0 * n
    ytile = (1.0 - math.log(math.tan(lat_rad) + 1.0 / math.cos(lat_rad)) / math.pi) / 2.0 * n
    return xtile, ytile


def latlon_to_world_px(lat, lon, z):
    """
    WGS84 (lat, lon) → world pixel coordinates in Web Mercator for zoom z.
    """
    n = 2.0 ** z
    lat_rad = math.radians(lat)
    x = (lon + 180.0) / 360.0 * TILE_SIZE * n
    y = (1.0 - math.log(math.tan(lat_rad) + 1.0 / math.cos(lat_rad)) / math.pi) / 2.0 * TILE_SIZE * n
    return x, y


def latlon_to_tile_px(lat, lon, z, x_tile, y_tile):
    """
    WGS84 → pixel inside a specific tile (0–255, 0–255).
    """
    world_x, world_y = latlon_to_world_px(lat, lon, z)
    px = world_x - x_tile * TILE_SIZE
    py = world_y - y_tile * TILE_SIZE
    return px, py


def bbox_to_tile_range(left, bottom, right, top, z):
    """
    Given a lon/lat bbox and zoom, return integer ranges of x and y tiles.
    """
    # left/bottom
    x_min_f, y_max_f = lonlat_to_tile_xy_float(left, bottom, z)
    # right/top
    x_max_f, y_min_f = lonlat_to_tile_xy_float(right, top, z)

    x_min = int(math.floor(x_min_f))
    x_max = int(math.floor(x_max_f))
    y_min = int(math.floor(y_min_f))
    y_max = int(math.floor(y_max_f))

    # Clamp to valid 0..(2^z-1)
    n = 2 ** z
    x_min = max(0, min(n - 1, x_min))
    x_max = max(0, min(n - 1, x_max))
    y_min = max(0, min(n - 1, y_min))
    y_max = max(0, min(n - 1, y_max))

    return x_min, x_max, y_min, y_max


def tile_to_bbox(z, x, y):
    """
    Tile indices -> lon/lat bbox of the tile (left, bottom, right, top).
    """
    n = 2.0 ** z

    lon_left = x / n * 360.0 - 180.0
    lon_right = (x + 1) / n * 360.0 - 180.0

    def mercator_to_lat(tile_y):
        lat_rad = math.atan(math.sinh(math.pi - 2.0 * math.pi * tile_y / n))
        return math.degrees(lat_rad)

    lat_top = mercator_to_lat(y)
    lat_bottom = mercator_to_lat(y + 1)

    return lon_left, lat_bottom, lon_right, lat_top


# ============================================================
# FEATURE / STYLE HELPERS
# ============================================================

def get_corridor_color(feature):
    """
    Choose a color for a feature based on properties.propertyDisplayValue.
    Uses the first ID that is in RFC_COLORS.
    """
    props = feature.get("properties", {})
    ids = props.get("propertyDisplayValue") or []

    for id_str in ids:
        if id_str in RFC_COLORS:
            return RFC_COLORS[id_str]

    return DEFAULT_COLOR


def feature_bbox(feature):
    """
    Rough lon/lat bbox for a (multi)linestring to cheaply cull tiles.
    """
    geom = feature.get("geometry", {})
    gtype = geom.get("type")
    coords = geom.get("coordinates", [])

    if gtype == "LineString":
        all_coords = coords
    elif gtype == "MultiLineString":
        all_coords = [c for line in coords for c in line]
    else:
        return None

    if not all_coords:
        return None

    lons = [c[0] for c in all_coords]
    lats = [c[1] for c in all_coords]

    return min(lons), min(lats), max(lons), max(lats)


def bbox_intersects(a, b):
    """
    Check if two lon/lat bboxes intersect: a=(l,b,r,t), b=(l,b,r,t)
    """
    a_left, a_bottom, a_right, a_top = a
    b_left, b_bottom, b_right, b_top = b

    return not (a_right < b_left or
                a_left > b_right or
                a_top < b_bottom or
                a_bottom > b_top)


# ============================================================
# RENDERING
# ============================================================

def draw_feature_on_tile(draw, feature, z, x_tile, y_tile, color, width):
    geom = feature.get("geometry", {})
    gtype = geom.get("type")
    coords = geom.get("coordinates", [])

    def draw_linestring(line_coords):
        pixels = []
        for lon, lat in line_coords:
            px, py = latlon_to_tile_px(lat, lon, z, x_tile, y_tile)
            pixels.append((px, py))

        if len(pixels) >= 2:
            draw.line(pixels, fill=color + (255,), width=width, joint="curve")

    if gtype == "LineString":
        draw_linestring(coords)
    elif gtype == "MultiLineString":
        for line in coords:
            draw_linestring(line)
    else:
        # ignore other geometry types
        return


def line_width_for_zoom(z):
    """
    Simple heuristic for line thickness vs zoom.
    Tweak this to taste.
    """
    if z <= 4:
        return 1
    elif z <= 6:
        return 2
    elif z <= 8:
        return 3
    else:
        return 4


# ============================================================
# MAIN
# ============================================================

def load_geojson():
    """
    Load FeatureCollection from local file if present, otherwise from CIP URL.
    """
    if Path(GEOJSON_FILE).exists():
        print(f"Loading GeoJSON from {GEOJSON_FILE}")
        with open(GEOJSON_FILE, "r", encoding="utf-8") as f:
            return json.load(f)

    print(f"Fetching GeoJSON from CIP: {CIP_GEOJSON_URL}")
    resp = requests.get(CIP_GEOJSON_URL)
    resp.raise_for_status()
    data = resp.json()

    # Optionally save for later
    with open(GEOJSON_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f)

    return data


def main():
    data = load_geojson()
    features = data.get("features", [])
    print(f"Loaded {len(features)} features")

    # Precompute feature bboxes to speed up culling
    feature_infos = []
    for f in features:
        fbbox = feature_bbox(f)
        feature_infos.append((f, fbbox))

    for z in range(MIN_ZOOM, MAX_ZOOM + 1):
        print(f"\n=== Generating zoom {z} ===")

        x_min, x_max, y_min, y_max = bbox_to_tile_range(
            BBOX_LEFT, BBOX_BOTTOM, BBOX_RIGHT, BBOX_TOP, z
        )
        print(f"Tile range z={z}: x={x_min}..{x_max}, y={y_min}..{y_max}")

        for x in range(x_min, x_max + 1):
            for y in range(y_min, y_max + 1):
                tile_bbox = tile_to_bbox(z, x, y)

                # Create transparent image
                img = Image.new("RGBA", (TILE_SIZE, TILE_SIZE), (0, 0, 0, 0))
                draw = ImageDraw.Draw(img, "RGBA")
                width = line_width_for_zoom(z)

                # Draw any feature that intersects this tile
                for f, fbbox in feature_infos:
                    if fbbox is None:
                        continue
                    if not bbox_intersects(tile_bbox, fbbox):
                        continue

                    color = get_corridor_color(f)
                    draw_feature_on_tile(draw, f, z, x, y, color, width)

                # Optionally, skip tiles that are empty (fully transparent)
                # by checking the bounding box of non-transparent pixels:
                if img.getbbox() is None:
                    # no content, don't write file
                    continue

                out_dir = OUTPUT_ROOT / str(z) / str(x)
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / f"{y}.png"
                img.save(out_path, format="PNG")
                # print(f"Saved {out_path}")

    print("\nDone. Tiles are in:", OUTPUT_ROOT.resolve())


if __name__ == "__main__":
    main()
