"""
Hotspot spatial clustering module.

Provides distance-aware DBSCAN clustering in projected metric coordinate systems,
noise point segregation, cluster centroid calculation, convex hull generation,
and hotspot summary metric aggregation.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely.geometry
from shapely.geometry import MultiPoint, Point, Polygon
from shapely.ops import transform as shapely_transform
from sklearn.cluster import DBSCAN

logger = logging.getLogger("waste_hotspot.clustering")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def reproject_to_metric_crs(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Reprojects a GeoDataFrame from geographic coordinates (degrees) to a metric projected CRS (meters).

    If the CRS is already projected, returns a copy as-is.
    If geographic, determines the appropriate local UTM zone based on spatial centroid,
    falling back to Web Mercator (EPSG:3857) if UTM cannot be determined.

    Args:
        gdf: Input GeoDataFrame.

    Returns:
        GeoDataFrame reprojected to a metric CRS.
    """
    if len(gdf) == 0:
        if gdf.crs is not None and gdf.crs.is_projected:
            return gdf.copy()
        return gdf.to_crs("EPSG:3857") if gdf.crs is not None else gdf.set_crs("EPSG:3857")

    if gdf.crs is not None and gdf.crs.is_projected:
        return gdf.copy()

    # Determine UTM zone from spatial centroid
    try:
        union_geom = gdf.geometry.union_all() if hasattr(gdf.geometry, "union_all") else gdf.geometry.unary_union
        centroid = union_geom.centroid
        lon = float(centroid.x)
        lat = float(centroid.y)

        utm_zone = int((lon + 180) // 6) + 1
        epsg_code = 32600 + utm_zone if lat >= 0 else 32700 + utm_zone
        target_crs = f"EPSG:{epsg_code}"
        return gdf.to_crs(target_crs)
    except Exception as exc:
        logger.debug("Falling back to EPSG:3857 due to UTM calculation error: %s", exc)
        return gdf.to_crs("EPSG:3857")


def _create_empty_summary_gdf(crs: Any) -> gpd.GeoDataFrame:
    """Helper to instantiate an empty summary GeoDataFrame with valid geometry schema and CRS."""
    cols = [
        "cluster_id",
        "detection_count",
        "mean_confidence",
        "max_confidence",
        "centroid",
        "status",
        "disclaimer",
        "geometry",
    ]
    empty_df = pd.DataFrame(columns=cols)
    return gpd.GeoDataFrame(empty_df, geometry="geometry", crs=crs)


def cluster_hotspots(
    detections_gdf: gpd.GeoDataFrame,
    eps_meters: float = 150.0,
    min_samples: int = 2,
    confidence_col: Optional[str] = None,
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Clusters candidate detections into spatial hotspots using DBSCAN in a metric projected CRS.

    Guarantees:
      - Clustering distances are computed strictly in meters (not degrees).
      - Isolated noise points receive cluster_id = -1.
      - Convex hulls for collinear or single-location clusters are buffered to valid Polygons with area > 0.
      - Summaries include detection count, mean/max confidence, spatial centroid, and disclaimer.

    Args:
        detections_gdf: GeoDataFrame containing detection geometries (Points or Polygons).
        eps_meters: Maximum neighborhood radius in meters for DBSCAN.
        min_samples: Minimum detections required to form a core cluster.
        confidence_col: Column name containing detection confidence scores.

    Returns:
        Tuple of:
          - clustered_gdf: Original detections with added 'cluster_id' column.
          - summary_gdf: One row per valid cluster (cluster_id >= 0) with Polygon geometry.
    """
    base_crs = detections_gdf.crs if detections_gdf.crs is not None else "EPSG:4326"

    # Handle empty input
    if len(detections_gdf) == 0:
        clustered_empty = detections_gdf.copy()
        clustered_empty["cluster_id"] = pd.Series(dtype=int)
        return clustered_empty, _create_empty_summary_gdf(base_crs)

    # Resolve confidence score column
    conf_col = confidence_col
    if conf_col is None or conf_col not in detections_gdf.columns:
        candidates = ["prob_waste", "confidence", "probability", "score"]
        conf_col = next((c for c in candidates if c in detections_gdf.columns), None)

    clustered_gdf = detections_gdf.copy()

    # Handle single point input directly
    if len(detections_gdf) < min_samples:
        clustered_gdf["cluster_id"] = -1
        return clustered_gdf, _create_empty_summary_gdf(base_crs)

    # Project to metric CRS for distance calculation
    metric_gdf = reproject_to_metric_crs(detections_gdf)

    # Extract centroids in metric space
    coords = np.array([[geom.centroid.x, geom.centroid.y] for geom in metric_gdf.geometry])

    # Run DBSCAN
    db = DBSCAN(eps=float(eps_meters), min_samples=int(min_samples), metric="euclidean")
    labels = db.fit_predict(coords)
    clustered_gdf["cluster_id"] = labels

    unique_clusters = sorted([c for c in set(labels) if c >= 0])
    summary_rows: List[Dict[str, Any]] = []

    for cid in unique_clusters:
        member_mask = (labels == cid)
        member_metric_geoms = [geom.centroid for geom in metric_gdf.loc[member_mask, "geometry"]]
        member_orig_geoms = [geom.centroid for geom in detections_gdf.loc[member_mask, "geometry"]]

        det_count = int(np.sum(member_mask))

        # Confidence statistics
        if conf_col is not None and conf_col in detections_gdf.columns:
            scores = detections_gdf.loc[member_mask, conf_col].astype(float).values
            mean_c = float(np.mean(scores))
            max_c = float(np.max(scores))
        else:
            mean_c = 1.0
            max_c = 1.0

        # Centroid in original CRS
        orig_xs = [g.x for g in member_orig_geoms]
        orig_ys = [g.y for g in member_orig_geoms]
        centroid_orig = Point(float(np.mean(orig_xs)), float(np.mean(orig_ys)))

        # Determine if base_crs is projected (metric) or geographic (degrees)
        is_proj = False
        try:
            if detections_gdf.crs is not None and hasattr(detections_gdf.crs, "is_projected"):
                is_proj = bool(detections_gdf.crs.is_projected)
        except Exception:
            pass

        buffer_dist = max(10.0, float(eps_meters) * 0.15) if is_proj else max(0.0001, (float(eps_meters) * 0.15) / 111320.0)

        # Create convex hull directly in original CRS
        multi_orig = MultiPoint(member_orig_geoms)
        hull_orig = multi_orig.convex_hull

        # Guarantee Polygon with non-zero area (handle collinear lines, single points, or duplicate coords)
        if not isinstance(hull_orig, Polygon) or hull_orig.area < 1e-12:
            hull_orig = hull_orig.buffer(buffer_dist)

        # Enforce geometric validity
        if not hull_orig.is_valid:
            try:
                valid_geom = shapely.make_valid(hull_orig)
                if hasattr(valid_geom, "geoms"):
                    polys = [p for p in valid_geom.geoms if isinstance(p, Polygon)]
                    if polys:
                        hull_orig = max(polys, key=lambda p: p.area)
                elif isinstance(valid_geom, Polygon):
                    hull_orig = valid_geom
            except Exception:
                hull_orig = hull_orig.buffer(0)

        summary_rows.append({
            "cluster_id": int(cid),
            "detection_count": int(det_count),
            "mean_confidence": float(round(mean_c, 4)),
            "max_confidence": float(round(max_c, 4)),
            "centroid": centroid_orig,
            "status": "candidate_hotspot",
            "disclaimer": "Candidate screening signal only; requires ground verification.",
            "geometry": hull_orig,
        })

    if summary_rows:
        summary_gdf = gpd.GeoDataFrame(summary_rows, geometry="geometry", crs=base_crs)
    else:
        summary_gdf = _create_empty_summary_gdf(base_crs)
    return clustered_gdf, summary_gdf


def build_parser() -> argparse.ArgumentParser:
    """CLI argument parser for clustering module."""
    parser = argparse.ArgumentParser(
        prog="python -m waste_hotspot.clustering",
        description="Cluster candidate waste detections into spatial hotspots.",
    )
    parser.add_argument("--input", "-i", required=True, type=str, help="Input detections CSV or GeoJSON.")
    parser.add_argument("--output", "-o", default="outputs/hotspots.geojson", type=str, help="Output GeoJSON path.")
    parser.add_argument("--eps", "-e", default=150.0, type=float, help="DBSCAN epsilon radius in meters (default: 150.0).")
    parser.add_argument("--min-samples", "-m", default=2, type=int, help="Minimum detections to form a cluster (default: 2).")
    return parser


def main(args: Optional[Sequence[str]] = None) -> int:
    """CLI entrypoint."""
    parser = build_parser()
    parsed = parser.parse_args(args)

    input_p = Path(parsed.input)
    if not input_p.exists():
        logger.error("Input file not found: %s", input_p)
        print(f"Error: Input file not found: {input_p}", file=sys.stderr)
        return 1

    if input_p.suffix.lower() == ".geojson":
        gdf = gpd.read_file(input_p)
    elif input_p.suffix.lower() == ".csv":
        df = pd.read_csv(input_p)
        if "longitude" in df.columns and "latitude" in df.columns and df["longitude"].notna().any():
            valid_geo = df.dropna(subset=["longitude", "latitude"])
            gdf = gpd.GeoDataFrame(
                valid_geo,
                geometry=gpd.points_from_xy(valid_geo["longitude"], valid_geo["latitude"]),
                crs="EPSG:4326",
            )
        else:
            print(f"Error: CSV contains no valid georeferenced coordinates.", file=sys.stderr)
            return 1
    else:
        print(f"Error: Unsupported format: {input_p.suffix}", file=sys.stderr)
        return 1

    clustered, summary = cluster_hotspots(gdf, eps_meters=parsed.eps, min_samples=parsed.min_samples)
    from waste_hotspot.geospatial import export_geojson
    export_geojson(summary, parsed.output)
    print(f"\nHotspot Clustering Complete:")
    print(f"  Total Candidates : {len(gdf)}")
    print(f"  Hotspot Clusters : {len(summary)}")
    print(f"  Hotspots Saved   : {parsed.output}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
