"""
Layer 1 - Data Ingestion for UHI/LST Prediction (Mumbai)  -  v2
================================================================
Two parts:
  1. HISTORICAL FEATURES (Google Earth Engine, batch/one-time):
     One row per (30 m grid pixel, year) for 2013-2024 with columns:
       PointID, Year, Lat, Long, H, LST, NDVI, NDWI, NDBI, NDBaI,
       Albedo, LULC, LULC_Year, Humidity
  2. LIVE DATA (Open-Meteo, runs every time you execute the script) -
     unchanged from v1.

Setup:
  pip install earthengine-api requests
  earthengine authenticate         # one-time browser auth (GEE part only)
Run:
  python Layer1_dataingestion_v2.py

WHAT CHANGED vs v1 (and why)
  1. COMMON 30 m GRID. Every layer is brought onto the Landsat 8/9 30 m pixel
     grid (UTM 43N) before sampling:
       - Landsat (LST + indices, Albedo): native 30 m (USGS resamples the
         100 m thermal band to 30 m in the Level-2 product).
       - Dynamic World LULC: 10 m -> 30 m by taking the MODE of the 10 m
         labels inside each 30 m pixel (a proper aggregation, not a single
         10 m pixel picked at random).
       - SRTM elevation: ~30 m, bilinear-resampled onto the Landsat grid.
       - ERA5-Land humidity: ~11 km, bilinearly interpolated to 30 m.
         NOTE: this makes the layer the same *pixel size*, it does NOT add
         information. Humidity remains a smooth regional signal.
  2. REGULAR GRID instead of random points. Points sit on a regular lattice
     (GRID_STEP_M apart, in metres). No two points can share a 30 m pixel, so
     the v1 duplicate rows disappear, and each location has a stable PointID
     for grouped / spatial train-test splits.
  3. NO WATER. Pixels are removed if they are water in the JRC Global Surface
     Water layer (occurrence >= WATER_OCCURRENCE_MIN %) OR Dynamic World class
     0, and the water mask is grown by WATER_BUFFER_M so coastal mixed pixels
     (land + sea inside one thermal pixel) are dropped too. Consequently LULC
     never equals 0 in the output.
  4. NDMI REMOVED (it was exactly -NDBI). NDBaI is kept by default, but see
     INCLUDE_NDBAI: it contains the thermal band (it is ~ -1 + 2*SWIR1/T).
  5. SAME SEASON EVERY YEAR. The composite uses only SEASON_MONTHS (default
     Oct-May, i.e. excluding the June-Sept monsoon when clouds mask most of
     the scenes), so year-to-year LST differences are not driven by which
     cloud-free dates happened to survive. Humidity is averaged over the SAME
     months as the Landsat composite.
  6. Extra quality masks: unphysical reflectance/temperature values removed.
  7. LULC_Year column records which Dynamic World year was used
     (2013-2015 reuse 2016).

REMAINING ASSUMPTIONS / LIMITATIONS
  - 2013: Landsat 8 starts in April, so 2013 has no Jan-Mar scenes.
    Landsat 9 starts in late 2021.
  - LULC for 2013-2015 is the 2016 map (Dynamic World starts mid-2015).
  - One median composite per year-season = a seasonal snapshot, not the
    hottest day.
  - The study area is still a bounding box; water is masked, but land outside
    Greater Mumbai's administrative limits (Thane / Navi Mumbai side) is kept.
    Swap in a ward boundary FeatureCollection for an exact clip.
"""

import ee
import requests

# ===========================================================================
# PART 1 - HISTORICAL FEATURES (Google Earth Engine)
# ===========================================================================

GEE_PROJECT = "model-fastness-509911-f6"  # <-- replace with your GEE project

START_YEAR = 2013
END_YEAR = 2024  # inclusive

# Dynamic World starts 2015-06-27, so 2016 is the first full year.
DW_FIRST_FULL_YEAR = 2016

# ---- Common grid ----------------------------------------------------------
GRID_SCALE_M = 30          # pixel size of every layer (Landsat native)
GRID_CRS = "EPSG:32643"    # UTM 43N - used only to lay out the sampling lattice
# Spacing between sampled pixels, in metres (multiple of 30 recommended).
#   300 m -> roughly 6-8k land points per year (default, fast)
#   90 m  -> ~10x more points; 30 m = every pixel (hundreds of thousands of
#            points per year - only with a much smaller study area).
GRID_STEP_M = 300
EXPORT_PER_YEAR = False    # True = one export task per year (more robust)

# ---- Seasonal window (same months every year) ------------------------------
# Dry/clear-sky season; excludes the June-Sept monsoon. Use list(range(1, 13))
# for the full calendar year like v1.
SEASON_MONTHS = [10, 11, 12, 1, 2, 3, 4, 5]
MIN_SCENES_WARN = 8        # warn if a year has fewer usable scenes than this

# ---- Water exclusion ----------------------------------------------------------
WATER_OCCURRENCE_MIN = 50  # JRC Global Surface Water occurrence (%), 0-100
WATER_BUFFER_M = 30        # grow the water mask by this much (coastal mixed pixels)

# ---- Features -----------------------------------------------------------------
# NDBaI = (SWIR1 - TIR)/(SWIR1 + TIR) uses the thermal band that defines LST
# (in practice it is ~ a rescaled SWIR1). Set False to drop it (leakage-safe).
INCLUDE_NDBAI = True

# Approximate Greater Mumbai bounding box (lon_min, lat_min, lon_max, lat_max)
MUMBAI_BBOX_COORDS = [72.77, 18.89, 72.98, 19.30]

# Final CSV columns (order preserved)
FEATURE_COLUMNS = (
    ["PointID", "Year", "Lat", "Long", "H", "LST", "NDVI", "NDWI", "NDBI"]
    + (["NDBaI"] if INCLUDE_NDBAI else [])
    + ["Albedo", "LULC", "LULC_Year", "Humidity"]
)
# Properties that must be non-null for a row to be kept
REQUIRED_PROPERTIES = [c for c in FEATURE_COLUMNS if c not in ("PointID", "Year")]


# ---------------------------------------------------------------------------
# 1. Reference grid + seasonal filter
# ---------------------------------------------------------------------------
def get_reference_projection(bbox):
    """
    Projection (UTM 43N, 30 m) of one Landsat 8 scene over Mumbai. Every layer
    is reprojected onto this, so all features share exactly the same pixels.
    """
    scene = (
        ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
        .filterBounds(bbox)
        .filterDate("2020-01-01", "2021-01-01")
        .sort("system:time_start")
        .first()
    )
    return ee.Image(scene).select("SR_B5").projection()


def season_filter():
    """Keep only images acquired in SEASON_MONTHS (any year)."""
    return ee.Filter.Or(*[ee.Filter.calendarRange(m, m, "month") for m in SEASON_MONTHS])


# ---------------------------------------------------------------------------
# 2. Landsat 8/9 Collection 2 Level-2: masks, scaling, indices
#    Surface Reflectance scale/offset : 0.0000275 / -0.2
#    Surface Temperature scale/offset : 0.00341802 / 149.0  (-> Kelvin)
#    SR_B2 Blue | SR_B3 Green | SR_B4 Red | SR_B5 NIR | SR_B6 SWIR1 |
#    SR_B7 SWIR2 | ST_B10 Thermal
# ---------------------------------------------------------------------------
def mask_l89_clouds(image):
    qa = image.select("QA_PIXEL")
    dilated, cirrus, cloud, shadow = 1 << 1, 1 << 2, 1 << 3, 1 << 4
    bits = dilated | cirrus | cloud | shadow
    return image.updateMask(qa.bitwiseAnd(bits).eq(0))


def scale_l89(image):
    optical = image.select("SR_B.").multiply(0.0000275).add(-0.2)
    thermal = image.select("ST_B10").multiply(0.00341802).add(149.0)  # Kelvin
    return image.addBands(optical, None, True).addBands(thermal, None, True)


def mask_unphysical(image):
    """Drop pixels with impossible reflectance (<=0 or >=1) or temperature."""
    sr = image.select(["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"])
    tir = image.select("ST_B10")
    ok = (
        sr.reduce(ee.Reducer.min()).gt(0)
        .And(sr.reduce(ee.Reducer.max()).lt(1))
        .And(tir.gt(250))
        .And(tir.lt(340))
    )
    return image.updateMask(ok)


def get_landsat_indices(bbox, year):
    """
    Seasonal median composite for `year` from Landsat 8 + 9 (Tier 1), native 30 m.
    Returns LST (deg C), NDVI, NDWI, NDBI, NDBaI, Albedo - or None if no scenes.
    """
    start, end = f"{year}-01-01", f"{year + 1}-01-01"  # end date is exclusive

    def _filtered(collection_id):
        return (
            ee.ImageCollection(collection_id)
            .filterBounds(bbox)
            .filterDate(start, end)
            .filter(season_filter())
        )

    coll = (
        _filtered("LANDSAT/LC08/C02/T1_L2")
        .merge(_filtered("LANDSAT/LC09/C02/T1_L2"))
        .map(mask_l89_clouds)
        .map(scale_l89)
        .map(mask_unphysical)
    )

    count = coll.size().getInfo()
    flag = "  <-- LOW, treat this year with caution" if count < MIN_SCENES_WARN else ""
    print(f"  [{year}] Landsat 8/9 scenes in season: {count}{flag}")
    if count == 0:
        return None

    img = coll.median()

    blue, red = img.select("SR_B2"), img.select("SR_B4")
    nir, swir1, swir2 = img.select("SR_B5"), img.select("SR_B6"), img.select("SR_B7")
    tir_k = img.select("ST_B10")  # Kelvin

    lst = tir_k.subtract(273.15).rename("LST")
    ndvi = img.normalizedDifference(["SR_B5", "SR_B4"]).rename("NDVI")  # (NIR-Red)
    ndwi = img.normalizedDifference(["SR_B3", "SR_B5"]).rename("NDWI")  # (Green-NIR)
    ndbi = img.normalizedDifference(["SR_B6", "SR_B5"]).rename("NDBI")  # (SWIR1-NIR)
    # NDMI intentionally NOT computed: NDMI == -NDBI exactly.

    # Broadband albedo: Liang (2001) narrow-to-broadband conversion
    albedo = (
        blue.multiply(0.356)
        .add(red.multiply(0.130))
        .add(nir.multiply(0.373))
        .add(swir1.multiply(0.085))
        .add(swir2.multiply(0.072))
        .subtract(0.0018)
        .rename("Albedo")
    )

    out = lst.addBands(ndvi).addBands(ndwi).addBands(ndbi)
    if INCLUDE_NDBAI:
        ndbai = swir1.subtract(tir_k).divide(swir1.add(tir_k)).rename("NDBaI")
        out = out.addBands(ndbai)
    return out.addBands(albedo)


# ---------------------------------------------------------------------------
# 3. LULC (Dynamic World, 10 m -> 30 m) and Humidity (ERA5-Land -> 30 m)
# ---------------------------------------------------------------------------
def get_lulc_30m(bbox, year, ref_proj):
    """
    Annual mode of the Dynamic World 'label' band (10 m), then aggregated to the
    30 m Landsat grid by the mode of the 10 m labels inside each 30 m pixel.
    Classes: 0 water, 1 trees, 2 grass, 3 flooded vegetation, 4 crops,
             5 shrub & scrub, 6 built, 7 bare, 8 snow & ice.
    Returns (LULC image, LULC_Year image). Years before DW_FIRST_FULL_YEAR
    reuse that year's map.
    """
    lulc_year = max(year, DW_FIRST_FULL_YEAR)
    if lulc_year != year:
        print(f"  [{year}] NOTE: Dynamic World unavailable; using {lulc_year} LULC.")

    dw = (
        ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")
        .filterBounds(bbox)
        .filterDate(f"{lulc_year}-01-01", f"{lulc_year + 1}-01-01")
        .select("label")
    )
    dw_proj = dw.first().projection()  # native 10 m projection

    lulc = (
        dw.mode()
        .setDefaultProjection(dw_proj)
        .reduceResolution(reducer=ee.Reducer.mode(), maxPixels=256)
        .reproject(ref_proj)
        .rename("LULC")
    )
    lulc_year_img = ee.Image.constant(lulc_year).toInt().rename("LULC_Year")
    return lulc, lulc_year_img


def _era5_relative_humidity(image):
    """Relative humidity (%) from 2 m temperature and dew point (Magnus)."""
    a, b = 17.625, 243.04
    t = image.select("temperature_2m").subtract(273.15)
    td = image.select("dewpoint_temperature_2m").subtract(273.15)
    rh = (
        td.multiply(a).divide(td.add(b))
        .subtract(t.multiply(a).divide(t.add(b)))
        .exp()
        .multiply(100)
        .clamp(0, 100)
    )
    return rh.rename("Humidity")


def get_humidity_30m(bbox, year):
    """
    Mean daily relative humidity (%) over SEASON_MONTHS of `year` - the same
    months used for the Landsat composite.
    ERA5-Land is ~11 km; bilinear interpolation gives a smooth 30 m surface but
    no new detail. Where bilinear leaves gaps (coast next to ERA5's ocean mask)
    the nearest-neighbour value fills in.
    """
    era = (
        ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR")
        .filterBounds(bbox)
        .filterDate(f"{year}-01-01", f"{year + 1}-01-01")
        .filter(season_filter())
        .select(["temperature_2m", "dewpoint_temperature_2m"])
    )
    smooth = era.map(lambda im: _era5_relative_humidity(im.resample("bilinear"))).mean()
    blocky = era.map(_era5_relative_humidity).mean()
    return smooth.unmask(blocky).rename("Humidity")


# ---------------------------------------------------------------------------
# 4. Land mask (removes sea, creeks, lakes + a coastal buffer)
# ---------------------------------------------------------------------------
def get_land_mask(lulc, ref_proj):
    """
    1 on land, 0 on water. Water = JRC permanent-water occurrence >=
    WATER_OCCURRENCE_MIN % OR Dynamic World class 0. The water mask is then
    grown by WATER_BUFFER_M to drop coastal pixels that mix land and water.
    """
    occurrence = (
        ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence").unmask(0)
    )
    water = (
        occurrence.gte(WATER_OCCURRENCE_MIN)
        .Or(lulc.eq(0))
        .reproject(ref_proj)
        .focalMax(radius=WATER_BUFFER_M, kernelType="square", units="meters")
    )
    return water.Not()


# ---------------------------------------------------------------------------
# 5. Regular sampling lattice
# ---------------------------------------------------------------------------
def build_grid_points(bbox):
    """
    Regular lattice (GRID_STEP_M apart, in UTM metres) covering the bbox.
    Each point gets a stable PointID = "<easting>_<northing>".
    """
    corners = bbox.transform(GRID_CRS, 1).bounds().getInfo()["coordinates"][0]
    xs = [c[0] for c in corners]
    ys = [c[1] for c in corners]
    x_min, x_max, y_min, y_max = min(xs), max(xs), min(ys), max(ys)

    x_list = ee.List.sequence(x_min, x_max, GRID_STEP_M)
    y_list = ee.List.sequence(y_min, y_max, GRID_STEP_M)

    def per_x(x):
        def per_y(y):
            pid = ee.Number(x).format("%.0f").cat("_").cat(ee.Number(y).format("%.0f"))
            geom = ee.Geometry.Point([x, y], GRID_CRS)
            return ee.Feature(geom, {"PointID": pid})
        return y_list.map(per_y)

    points = ee.FeatureCollection(ee.List(x_list.map(per_x)).flatten())
    return points.filterBounds(bbox)


# ---------------------------------------------------------------------------
# 6. Build the feature table and export
# ---------------------------------------------------------------------------
def build_and_export_historical_features():
    bbox = ee.Geometry.Rectangle(MUMBAI_BBOX_COORDS, geodesic=False)
    ref_proj = get_reference_projection(bbox)

    elevation = ee.Image("USGS/SRTMGL1_003").select("elevation").resample("bilinear").rename("H")
    lat_lon = ee.Image.pixelLonLat().rename(["Long", "Lat"])  # pixel centres

    grid_points = build_grid_points(bbox)
    print(f"Sampling lattice: {GRID_STEP_M} m spacing, features on a "
          f"{GRID_SCALE_M} m grid.")
    print(f"Building features for {START_YEAR}-{END_YEAR}, "
          f"months {SEASON_MONTHS}...")

    yearly_tables = []
    for year in range(START_YEAR, END_YEAR + 1):
        landsat = get_landsat_indices(bbox, year)
        if landsat is None:
            print(f"  >>> WARNING: no Landsat scenes for {year}; year skipped.")
            continue

        lulc, lulc_year = get_lulc_30m(bbox, year, ref_proj)
        land = get_land_mask(lulc, ref_proj)

        stack = (
            lat_lon.addBands(elevation)
            .addBands(landsat)
            .addBands(lulc)
            .addBands(lulc_year)
            .addBands(get_humidity_30m(bbox, year))
            .reproject(ref_proj)          # everything on the same 30 m pixels
            .updateMask(land)             # sea / water / coast -> masked
        )

        def add_year(f, y=year):
            return f.set("Year", y)

        sampled = (
            stack.reduceRegions(
                collection=grid_points,
                reducer=ee.Reducer.first(),
                scale=GRID_SCALE_M,
                tileScale=4,
            )
            .filter(ee.Filter.notNull(REQUIRED_PROPERTIES))  # drops masked pixels
            .map(add_year)
        )
        yearly_tables.append((year, sampled))

    if not yearly_tables:
        raise RuntimeError("No Landsat scenes found for any year in the range.")

    prefix = f"uhi_layer1_v2_mumbai_{START_YEAR}_{END_YEAR}"
    if EXPORT_PER_YEAR:
        for year, table in yearly_tables:
            ee.batch.Export.table.toDrive(
                collection=table,
                description=f"{prefix}_{year}",
                fileFormat="CSV",
                selectors=FEATURE_COLUMNS,
            ).start()
        print(f"{len(yearly_tables)} export tasks started - see the GEE Tasks tab.")
    else:
        all_records = ee.FeatureCollection([t for _, t in yearly_tables]).flatten()
        ee.batch.Export.table.toDrive(
            collection=all_records,
            description=prefix,
            fileFormat="CSV",
            selectors=FEATURE_COLUMNS,
        ).start()
        print("Historical feature export task started - check the GEE Tasks tab "
              "or your Google Drive.")


# ===========================================================================
# PART 2 - LIVE DATA (Open-Meteo, no API key required)  -  unchanged from v1
# ===========================================================================

MUMBAI_LIVE_POINTS = [
    ("Colaba (South Mumbai)", 18.9067, 72.8147),
    ("Bandra", 19.0596, 72.8295),
    ("Andheri", 19.1136, 72.8697),
    ("Borivali (North)", 19.2288, 72.8567),
]

LIVE_VARIABLES = (
    "temperature_2m,relative_humidity_2m,apparent_temperature,"
    "wind_speed_10m,cloud_cover,precipitation"
)


def fetch_live_weather(lat, lon):
    """
    Pulls current weather conditions for one point via the free Open-Meteo
    API. Swap this function's internals for an IMD station API later; the
    rest of the script doesn't need to change.
    """
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": LIVE_VARIABLES,
        "timezone": "auto",
    }
    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()
    data = response.json()
    return data["current"], data["current_units"], data.get("timezone", "local")


def print_live_data(points=MUMBAI_LIVE_POINTS):
    """Fetches and prints live weather for each point in `points`."""
    print("\n" + "=" * 60)
    print("LIVE DATA SNAPSHOT")
    print("=" * 60)
    for name, lat, lon in points:
        try:
            current, units, tz = fetch_live_weather(lat, lon)
        except requests.RequestException as exc:
            print(f"\n--- {name} ---\n  [error fetching live data: {exc}]")
            continue

        print(f"\n--- {name} ({lat}, {lon}) ---")
        print(f"  Observed at : {current['time']} ({tz})")
        for key, value in current.items():
            if key == "time":
                continue
            unit = units.get(key, "")
            print(f"  {key:22s}: {value} {unit}")


# ===========================================================================
# MAIN
# ===========================================================================

if __name__ == "__main__":
    ee.Initialize(project=GEE_PROJECT)
    build_and_export_historical_features()
    print_live_data()