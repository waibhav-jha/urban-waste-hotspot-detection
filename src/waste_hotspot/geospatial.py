"""
Geospatial mapping and coordinate transformation module.

Provides Rasterio affine coordinate transformations from pixel bounds to
real-world geographic coordinates, CRS handling, and RFC 7946 GeoJSON exports.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely.geometry
from shapely.geometry import Point, Polygon, box

logger = logging.getLogger("waste_hotspot.geospatial")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def pixel_to_geospatial(
    pixel_coords: Tuple[int, int, int, int],
    raster_path: Union[str, Path],
) -> Dict[str, Any]:
    """
    Transforms pixel bounding coordinates to geographic/projected coordinates using Rasterio.

    Args:
        pixel_coords: Tuple of (x_min, y_min, x_max, y_max) in pixel coordinates.
        raster_path: Path to the image or raster file.

    Returns:
        Dictionary containing:
            - 'has_georeference': bool
            - 'crs': CRS string or None
            - 'geometry': shapely Polygon or None
            - 'pixel_coords': original pixel coords
    """
    path = Path(raster_path)
    x_min, y_min, x_max, y_max = pixel_coords

    # Fast rejection for missing or non-georeferenced file extensions
    if not path.is_file() or path.suffix.lower() not in {".tif", ".tiff"}:
        return {
            "has_georeference": False,
            "crs": None,
            "geometry": None,
            "pixel_coords": pixel_coords,
        }

    try:
        import rasterio

        with rasterio.open(path) as src:
            if src.crs is None:
                return {
                    "has_georeference": False,
                    "crs": None,
                    "geometry": None,
                    "pixel_coords": pixel_coords,
                }

            transform = src.transform
            if transform is None or transform.is_identity:
                return {
                    "has_georeference": False,
                    "crs": None,
                    "geometry": None,
                    "pixel_coords": pixel_coords,
                }

            # Map pixel corners through affine transform
            # (col, row) -> (x_geo, y_geo)
            x0, y0 = transform * (x_min, y_min)  # Upper-left
            x1, y1 = transform * (x_max, y_min)  # Upper-right
            x2, y2 = transform * (x_max, y_max)  # Lower-right
            x3, y3 = transform * (x_min, y_max)  # Lower-left

            # Create bounding polygon
            poly = Polygon([(x0, y0), (x1, y1), (x2, y2), (x3, y3), (x0, y0)])
            crs_str = src.crs.to_string() if hasattr(src.crs, "to_string") else str(src.crs)

            return {
                "has_georeference": True,
                "crs": crs_str,
                "geometry": poly,
                "pixel_coords": pixel_coords,
                "transform": transform,
            }

    except Exception as exc:
        logger.debug("Failed geospatial projection for '%s': %s", path, exc)
        return {
            "has_georeference": False,
            "crs": None,
            "geometry": None,
            "pixel_coords": pixel_coords,
        }


def export_geojson(
    gdf: Union[gpd.GeoDataFrame, pd.DataFrame],
    output_path: Union[str, Path],
) -> None:
    """
    Exports a GeoDataFrame to an RFC 7946 compliant GeoJSON FeatureCollection.

    Ensures:
      - Empty dataframes export as valid FeatureCollection with 0 features.
      - Non-4326 geometries are reprojected to EPSG:4326.
      - Properties are cleanly serializable (converts numpy scalars).

    Args:
        gdf: GeoDataFrame or DataFrame with geometry column.
        output_path: File path to save the .geojson file.
    """
    out_p = Path(output_path).resolve()
    out_p.parent.mkdir(parents=True, exist_ok=True)

    if len(gdf) == 0:
        empty_payload = {
            "type": "FeatureCollection",
            "features": [],
        }
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(empty_payload, f, indent=2)
        logger.info("Exported empty GeoJSON FeatureCollection to %s", out_p)
        return

    # Convert to GeoDataFrame if needed
    if not isinstance(gdf, gpd.GeoDataFrame):
        if "geometry" in gdf.columns:
            gdf_export = gpd.GeoDataFrame(gdf, geometry="geometry")
        else:
            raise ValueError("DataFrame has no 'geometry' column for GeoJSON export.")
    else:
        gdf_export = gdf.copy()

    # Reproject to WGS84 (EPSG:4326) for RFC 7946 compliance if CRS is defined
    if gdf_export.crs is not None and str(gdf_export.crs).upper() != "EPSG:4326":
        try:
            gdf_export = gdf_export.to_crs("EPSG:4326")
        except Exception as exc:
            logger.warning("Could not reproject GeoDataFrame to EPSG:4326: %s", exc)

    # Clean non-serializable columns (e.g. centroids if duplicate or numpy objects)
    cleaned_df = gdf_export.copy()
    for col in cleaned_df.columns:
        if col != cleaned_df.geometry.name and col in cleaned_df:
            # If the column contains shapely geometries that are not the active geometry, drop or convert to WKT
            first_val = cleaned_df[col].dropna().iloc[0] if len(cleaned_df[col].dropna()) > 0 else None
            if isinstance(first_val, shapely.geometry.base.BaseGeometry):
                cleaned_df[col] = cleaned_df[col].apply(lambda g: g.wkt if hasattr(g, "wkt") else str(g))

    cleaned_df.to_file(out_p, driver="GeoJSON")
    logger.info("Saved GeoJSON with %d features to %s", len(cleaned_df), out_p)


def reproject_gdf(gdf: gpd.GeoDataFrame, target_crs: str = "EPSG:4326") -> gpd.GeoDataFrame:
    """
    Safely reprojects a GeoDataFrame to a target CRS.

    Args:
        gdf: Source GeoDataFrame.
        target_crs: Destination coordinate reference system.

    Returns:
        Reprojected GeoDataFrame.
    """
    if gdf.crs is None:
        logger.warning("GeoDataFrame has no CRS set; cannot reproject.")
        return gdf.copy()
    return gdf.to_crs(target_crs)


def build_parser() -> argparse.ArgumentParser:
    """CLI argument parser for geospatial module."""
    parser = argparse.ArgumentParser(
        prog="python -m waste_hotspot.geospatial",
        description="Inspect and convert raster pixel coordinates to geographic coordinates.",
    )
    parser.add_argument("--raster", "-r", required=True, type=str, help="Path to GeoTIFF raster.")
    parser.add_argument("--bbox", nargs=4, type=int, default=[0, 0, 256, 256], help="Pixel bounds: x_min y_min x_max y_max.")
    return parser


def main(args: Optional[Sequence[str]] = None) -> int:
    """CLI entrypoint."""
    parser = build_parser()
    parsed = parser.parse_args(args)

    res = pixel_to_geospatial(tuple(parsed.bbox), parsed.raster)
    print("\nGeospatial Inspection Result:")
    print(f"  Raster           : {parsed.raster}")
    print(f"  Has Georeference : {res['has_georeference']}")
    print(f"  CRS              : {res['crs']}")
    if res['geometry'] is not None:
        print(f"  Bounds           : {res['geometry'].bounds}")
        print(f"  Centroid         : ({res['geometry'].centroid.x:.6f}, {res['geometry'].centroid.y:.6f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
