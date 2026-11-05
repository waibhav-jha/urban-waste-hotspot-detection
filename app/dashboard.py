"""
Streamlit GIS Dashboard for Urban Waste Hotspot Detection.

Interactive screening interface for analyzing aerial and satellite imagery,
inspecting candidate waste detections, visualizing georeferenced hotspots on
interactive GIS maps, and exporting structured predictions (CSV and GeoJSON).
"""

from __future__ import annotations

import io
import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

import folium
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import shapely.geometry
from shapely.geometry import Point, Polygon
import streamlit as st
import streamlit.components.v1 as components

# Add project root and src/ to sys.path
APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parent
SRC_DIR = REPO_ROOT / "src"
for p in [str(REPO_ROOT), str(SRC_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from waste_hotspot.clustering import cluster_hotspots, reproject_to_metric_crs
from waste_hotspot.geospatial import export_geojson, pixel_to_geospatial
from waste_hotspot.inference import (
    PREDICTION_COLUMNS,
    check_georeference,
    load_model_for_inference,
    predict_image,
    predict_image_tiled,
)
from waste_hotspot.tiling import generate_tiles, load_image_as_array

# Streamlit Page Configuration
st.set_page_config(
    page_title="Urban Waste Hotspot Detection | GIS Dashboard",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .disclaimer-box {
        background-color: #FEF3C7;
        border-left: 5px solid #F59E0B;
        padding: 12px 18px;
        border-radius: 4px;
        margin-bottom: 20px;
        color: #92400E;
        font-size: 0.95rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }
    .badge-waste {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.9rem;
    }
    .badge-clean {
        background-color: #DCFCE7;
        color: #166534;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def render_folium_map(
    detections_df: pd.DataFrame,
    summary_gdf: Optional[gpd.GeoDataFrame] = None,
    zoom_start: int = 14,
) -> folium.Map:
    """Creates an interactive Folium leaflet map with detections and cluster polygons."""
    valid_pts = detections_df.dropna(subset=["longitude", "latitude"])
    if len(valid_pts) == 0:
        center_lat, center_lon = 45.4642, 9.1900
    else:
        center_lat = float(valid_pts["latitude"].mean())
        center_lon = float(valid_pts["longitude"].mean())

    fmap = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=zoom_start,
        tiles="CartoDB positron",
    )

    # 1. Plot individual detection markers
    for _, row in valid_pts.iterrows():
        is_candidate = str(row.get("status", "")).lower() == "candidate_detection"
        prob = float(row.get("prob_waste", 0.0))
        color = "#DC2626" if is_candidate else "#16A34A"
        icon_name = "exclamation-triangle" if is_candidate else "check-circle"

        popup_html = f"""
        <div style="font-family: sans-serif; font-size: 12px; min-width: 140px;">
            <b>ID:</b> {row.get('image_id', 'Tile')}<br/>
            <b>Status:</b> {'Candidate Waste' if is_candidate else 'Non-Waste'}<br/>
            <b>Confidence:</b> {prob * 100:.1f}%<br/>
            <b>Source:</b> {row.get('source_image', 'N/A')}<br/>
        </div>
        """

        folium.CircleMarker(
            location=[float(row["latitude"]), float(row["longitude"])],
            radius=6 if is_candidate else 4,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.75,
            popup=folium.Popup(popup_html, max_width=250),
        ).add_to(fmap)

    # 2. Plot hotspot cluster convex hulls
    if summary_gdf is not None and len(summary_gdf) > 0:
        for _, row in summary_gdf.iterrows():
            geom = row["geometry"]
            cid = row["cluster_id"]
            count = row["detection_count"]
            mean_c = row["mean_confidence"]

            if isinstance(geom, Polygon):
                # Folium expects [[lat, lon], ...]
                coords = [[y, x] for x, y in geom.exterior.coords]
                popup_text = f"Hotspot Cluster #{cid}<br/>Count: {count} detections<br/>Avg Conf: {mean_c*100:.1f}%"

                folium.Polygon(
                    locations=coords,
                    color="#DC2626",
                    weight=2,
                    fill=True,
                    fill_color="#EF4444",
                    fill_opacity=0.35,
                    popup=folium.Popup(popup_text, max_width=200),
                ).add_to(fmap)

                if "centroid" in row and isinstance(row["centroid"], Point):
                    c_pt = row["centroid"]
                    folium.Marker(
                        location=[c_pt.y, c_pt.x],
                        icon=folium.Icon(color="red", icon="bullseye", prefix="fa"),
                        popup=f"Centroid of Hotspot #{cid}",
                    ).add_to(fmap)

    return fmap


def main():
    st.markdown('<div class="main-header">🛰️ Urban Waste Hotspot Detection</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">AI-Powered Aerial & Satellite Imagery Screening Pipeline with Geospatial Hotspot Clustering</div>',
        unsafe_allow_html=True,
    )

    # Mandatory Ethical & Operational Disclaimer
    st.markdown(
        """
        <div class="disclaimer-box">
            <b>⚠️ OPERATIONAL SCREENING NOTICE:</b>
            Model outputs are statistical screening signals intended for preliminary prioritization.
            Predictions represent candidate areas for human inspection and do <b>not</b> constitute legal proof of illegal dumping.
            Ground truthing and professional review are required prior to any operational or enforcement action.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- SIDEBAR CONFIGURATION ---
    st.sidebar.header("⚙️ Configuration")

    # Checkpoint selection
    default_ckpt = REPO_ROOT / "artifacts" / "best_model.pth"
    ckpt_path_str = st.sidebar.text_input(
        "Model Checkpoint (.pth)",
        value=str(default_ckpt) if default_ckpt.exists() else "artifacts/best_model.pth",
    )

    # Threshold slider
    threshold = st.sidebar.slider(
        "Decision Threshold (T*)",
        min_value=0.05,
        max_value=0.95,
        value=0.50,
        step=0.01,
        help="Adjust probability cutoff for flagging candidate waste detections. Tuned to optimize F1 score.",
    )

    # Tiling options
    st.sidebar.subheader("🪟 Image Tiling")
    use_tiling = st.sidebar.checkbox("Enable Sliding-Window Tiling", value=True, help="Partitions large images into overlapping tiles.")
    tile_size = st.sidebar.selectbox("Tile Size (px)", [128, 224, 256, 512], index=2)
    stride = st.sidebar.slider("Tile Stride (px)", min_value=64, max_value=512, value=200, step=32)

    # Clustering options
    st.sidebar.subheader("📍 Hotspot Clustering (DBSCAN)")
    eps_meters = st.sidebar.slider(
        "Neighborhood Radius (meters)",
        min_value=50.0,
        max_value=2500.0,
        value=300.0,
        step=50.0,
        help="Maximum distance in meters to group adjacent detections into one hotspot.",
    )
    min_samples = st.sidebar.slider(
        "Minimum Cluster Size",
        min_value=2,
        max_value=10,
        value=2,
        help="Minimum number of positive candidate tiles required to form a verified hotspot cluster.",
    )

    # Main Tabs
    tab_inspect, tab_geo, tab_metrics, tab_guide = st.tabs([
        "🔍 Detection & Analysis",
        "🗺️ Geospatial Map & Hotspots",
        "📊 Model Evaluation Metrics",
        "📖 Responsible Use & Workflow",
    ])

    with tab_inspect:
        col_input, col_preview = st.columns([1, 1])

        with col_input:
            st.subheader("1. Select Imagery Source")
            input_mode = st.radio("Input Mode:", ["Upload Image File", "Select Sample Dataset Image"], horizontal=True)

            target_file_path: Optional[Path] = None
            uploaded_img_arr: Optional[np.ndarray] = None
            is_geotiff = False

            if input_mode == "Upload Image File":
                uploaded_file = st.file_uploader(
                    "Upload Satellite / Drone Image (.tif, .png, .jpg)",
                    type=["png", "jpg", "jpeg", "tif", "tiff"],
                )
                if uploaded_file is not None:
                    # Save to temporary scratch file
                    temp_dir = REPO_ROOT / "outputs" / "temp_uploads"
                    temp_dir.mkdir(parents=True, exist_ok=True)
                    target_file_path = temp_dir / uploaded_file.name
                    target_file_path.write_bytes(uploaded_file.getbuffer())
                    st.success(f"Uploaded: {uploaded_file.name} ({len(uploaded_file.getvalue()) / 1024:.1f} KB)")
            else:
                # Browse data/raw/images or synthetic samples
                images_dir = REPO_ROOT / "data" / "raw" / "images"
                sample_files: List[Path] = []
                if images_dir.exists():
                    sample_files = list(images_dir.glob("*.png")) + list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.tif"))
                
                if sample_files:
                    selected_sample = st.selectbox("Choose sample image:", [p.name for p in sample_files[:20]])
                    target_file_path = next((p for p in sample_files if p.name == selected_sample), None)
                else:
                    st.info("No files found in data/raw/images. Please upload an image above.")

            run_btn = st.button("🚀 Run Waste Detection Pipeline", type="primary", use_container_width=True)

        with col_preview:
            st.subheader("2. Image Preview")
            if target_file_path is not None and target_file_path.exists():
                try:
                    img_arr, _ = load_image_as_array(target_file_path)
                    st.image(img_arr, caption=f"{target_file_path.name} ({img_arr.shape[1]}x{img_arr.shape[0]} px)", use_container_width=True)
                except Exception as exc:
                    st.warning(f"Could not load image preview: {exc}")
            else:
                st.info("Select or upload an image to preview.")

        # --- EXECUTE INFERENCE ---
        if run_btn and target_file_path is not None:
            with st.spinner("Executing detection and spatial analysis..."):
                try:
                    out_csv = REPO_ROOT / "outputs" / "dashboard_predictions.csv"
                    out_csv.parent.mkdir(parents=True, exist_ok=True)

                    if use_tiling:
                        pred_df = predict_image_tiled(
                            image_path=target_file_path,
                            checkpoint_path=ckpt_path_str if Path(ckpt_path_str).exists() else None,
                            output_csv=out_csv,
                            tile_size=tile_size,
                            stride=stride,
                            threshold=threshold,
                            device="cpu",
                        )
                    else:
                        single_res = predict_image(
                            image_path=target_file_path,
                            checkpoint_path=ckpt_path_str if Path(ckpt_path_str).exists() else None,
                            threshold=threshold,
                            device="cpu",
                        )
                        pred_df = pd.DataFrame([{col: single_res.get(col, np.nan) for col in PREDICTION_COLUMNS}])
                        pred_df.to_csv(out_csv, index=False)

                    st.session_state["predictions_df"] = pred_df
                    st.session_state["target_file"] = str(target_file_path)
                    st.success(f"Processing complete: {len(pred_df)} detections/tiles evaluated.")
                except Exception as exc:
                    st.error(f"Inference execution failed: {exc}")

        # --- RESULTS SUMMARY ---
        if "predictions_df" in st.session_state:
            df = st.session_state["predictions_df"]
            num_total = len(df)
            num_candidates = int((df["status"] == "candidate_detection").sum())
            has_geo = bool(df["has_georeference"].iloc[0]) if "has_georeference" in df else False

            st.markdown("---")
            st.subheader("3. Screening Results Summary")

            m_col1, m_col2, m_col3, m_col4 = st.columns(4)
            with m_col1:
                st.metric("Total Tiles/Frames", f"{num_total}")
            with m_col2:
                st.metric("Candidate Detections", f"{num_candidates}", delta=f"{(num_candidates/num_total*100):.1f}% of total" if num_total > 0 else "0%")
            with m_col3:
                max_conf = float(df["prob_waste"].max()) if len(df) > 0 else 0.0
                st.metric("Peak Confidence", f"{max_conf*100:.1f}%")
            with m_col4:
                st.metric("Georeferenced", "Yes (WGS84)" if has_geo else "No (Image-Space Only)")

            # Table of Predictions
            st.dataframe(
                df[[
                    "image_id", "source_image", "is_tile", "x_min", "y_min", "x_max", "y_max",
                    "prob_waste", "status", "has_georeference", "longitude", "latitude"
                ]].style.format({"prob_waste": "{:.3f}"}),
                use_container_width=True,
                height=250,
            )

    with tab_geo:
        st.subheader("Geospatial Hotspot Map & Spatial Clusters")

        if "predictions_df" in st.session_state:
            df = st.session_state["predictions_df"]
            has_geo = bool(df["has_georeference"].iloc[0]) if "has_georeference" in df else False

            if has_geo and df["longitude"].notna().any():
                valid_geo = df[df["longitude"].notna() & df["latitude"].notna()].copy()
                candidate_pts = valid_geo[valid_geo["status"] == "candidate_detection"].copy()

                detections_gdf = gpd.GeoDataFrame(
                    candidate_pts,
                    geometry=gpd.points_from_xy(candidate_pts["longitude"], candidate_pts["latitude"]),
                    crs="EPSG:4326",
                )

                # Cluster candidate detections
                clustered_gdf, summary_gdf = cluster_hotspots(
                    detections_gdf=detections_gdf,
                    eps_meters=eps_meters,
                    min_samples=min_samples,
                )

                # Render Map
                st.markdown(f"**Map Visualization** ({len(summary_gdf)} Hotspot Clusters Identified, eps={eps_meters}m, min={min_samples}):")
                folium_map = render_folium_map(valid_geo, summary_gdf)
                components.html(folium_map._repr_html_(), height=520)

                # Hotspot table
                if len(summary_gdf) > 0:
                    st.subheader(f"Identified Hotspots ({len(summary_gdf)})")
                    display_summary = summary_gdf.copy()
                    display_summary["centroid_coords"] = display_summary["centroid"].apply(lambda p: f"({p.y:.5f}, {p.x:.5f})")
                    st.dataframe(
                        display_summary[[
                            "cluster_id", "detection_count", "mean_confidence", "max_confidence",
                            "centroid_coords", "status", "disclaimer"
                        ]].style.format({"mean_confidence": "{:.1%}", "max_confidence": "{:.1%}"}),
                        use_container_width=True,
                    )

                    # Export Hotspots GeoJSON
                    geojson_path = REPO_ROOT / "outputs" / "dashboard_hotspots.geojson"
                    export_geojson(summary_gdf, geojson_path)
                    with open(geojson_path, "r", encoding="utf-8") as gf:
                        st.download_button(
                            "📥 Download Hotspots GeoJSON (RFC 7946)",
                            data=gf.read(),
                            file_name="hotspots.geojson",
                            mime="application/geo+json",
                        )
                else:
                    st.info("No multi-point clusters met the minimum sample threshold. Isolated points are classified as noise.")

            else:
                st.warning(
                    """
                    **Geospatial Mapping Unavailable:**
                    The selected image is a standard image (e.g., PNG/JPEG) lacking GeoTIFF spatial metadata (CRS and affine geotransform).
                    Coordinates are strictly preserved in pixel space `(x_min, y_min, x_max, y_max)`.
                    The system explicitly forbids manufacturing or guessing geographic coordinates.
                    """
                )
        else:
            st.info("Please run detection in the 'Detection & Analysis' tab first.")

    with tab_metrics:
        st.subheader("Model Architecture & Baseline Evaluation Report")

        eval_report_path = REPO_ROOT / "outputs" / "evaluation_report.json"
        if eval_report_path.exists():
            with open(eval_report_path, "r", encoding="utf-8") as f:
                rep = json.load(f)

            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            with col_m1:
                st.metric("F1-Score", f"{rep.get('f1', 0.0):.3f}")
            with col_m2:
                st.metric("Precision", f"{rep.get('precision', 0.0):.3f}")
            with col_m3:
                st.metric("Recall", f"{rep.get('recall', 0.0):.3f}")
            with col_m4:
                st.metric("PR-AUC", f"{rep.get('pr_auc', 0.0):.3f}")

            # Confusion Matrix
            if "confusion_matrix" in rep:
                cm = np.array(rep["confusion_matrix"])
                st.markdown("#### Confusion Matrix")
                cm_df = pd.DataFrame(
                    cm,
                    index=["Actual Non-Waste (0)", "Actual Waste (1)"],
                    columns=["Pred Non-Waste (0)", "Pred Waste (1)"],
                )
                st.table(cm_df)

            st.json(rep)
        else:
            st.info(
                "Evaluation report not found at `outputs/evaluation_report.json`. "
                "Run `python -m waste_hotspot.evaluate --config configs/default.yaml` to generate held-out test metrics."
            )

    with tab_guide:
        st.subheader("Operational Guidance & Ethical Guidelines")
        st.markdown(
            """
            ### 1. Methodology & Data Lineage
            - **Dataset**: AerialWaste benchmark (Zenodo: records/7991872).
            - **Model**: Transfer learning with ResNet-50 backbone pretrained on ImageNet, custom binary classification head.
            - **Data Splitting**: Strict scene-level deduplication and grouping to prevent geospatial data leakage.
            - **Hotspot Clustering**: Metric CRS reprojection (UTM/EPSG:3857) and DBSCAN distance grouping in meters.

            ### 2. Known Limitations & False Positives
            - **Environmental Noise**: Dense shadow patterns, dry vegetation, and agricultural mulch can mimic synthetic plastic signatures.
            - **Construction Sites**: Excavated rubble and building materials may register as waste-affected without representing illegal dumping.
            - **Domain Shift**: Imagery taken under different sun angles, sensor calibrations, or seasons require local threshold recalibration.

            ### 3. Verification Protocol
            1. Screen candidate clusters with high density (`detection_count >= 3`).
            2. Cross-reference spatial coordinates with municipal zoning records.
            3. Deploy ground inspection teams prior to issuing public notices or regulatory fines.
            """
        )

    # Global Export Button in Sidebar
    if "predictions_df" in st.session_state:
        st.sidebar.markdown("---")
        st.sidebar.subheader("📥 Export Data")
        csv_buffer = io.StringIO()
        st.session_state["predictions_df"].to_csv(csv_buffer, index=False)
        st.sidebar.download_button(
            "Download Predictions CSV",
            data=csv_buffer.getvalue(),
            file_name="waste_predictions.csv",
            mime="text/csv",
            use_container_width=True,
        )


if __name__ == "__main__":
    main()
