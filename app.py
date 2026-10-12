"""
app.py — Hospital Patient Readmission Prediction (Streamlit Application).

Run with:  streamlit run app.py
"""

import io
import os
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from train_model import (
    detect_columns,
    detect_target_labels,
    normalize_name,
    predict_single,
    prepare_data,
    save_model,
    load_model,
    train_and_evaluate,
)

# ═══════════════════════════════════════════════════════════════════════════════
# Page config
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Hospital Readmission Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════════════════
# Custom CSS — professional healthcare theme
# ═══════════════════════════════════════════════════════════════════════════════

CUSTOM_CSS = """
<style>
/* ---- Global ---- */
html, body, [class*="css"] { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
.main { background-color: #f5f7fa; }
.block-container { padding-top: 2rem; }

/* ---- Sidebar ---- */
section[data-testid="stSidebar"] { background-color: #1a2332; }
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] label { color: #e8ecf1 !important; }
section[data-testid="stSidebar"] .stMarkdown p { color: #c5cdd8; }

/* ---- Metric cards ---- */
div[data-testid="stMetric"] {
    background: #ffffff;
    border-radius: 10px;
    padding: 18px 16px;
    box-shadow: 0 2px 6px rgba(0,0,0,0.07);
    border-left: 5px solid #0d6efd;
}

/* ---- Headers ---- */
h1 { color: #1a2332 !important; }
h2 { color: #1a2332 !important; }
h3 { color: #1b4f72 !important; }

/* ---- Status boxes ---- */
.status-card {
    padding: 1rem 1.25rem;
    border-radius: 8px;
    margin: 0.75rem 0;
}
.status-ok   { background: #d4edda; border-left: 5px solid #28a745; }
.status-info { background: #d1ecf1; border-left: 5px solid #17a2b8; }
.status-warn { background: #fff3cd; border-left: 5px solid #ffc107; }
.status-err  { background: #f8d7da; border-left: 5px solid #dc3545; }

/* ---- Upload zone ---- */
.upload-zone {
    background: linear-gradient(135deg, #e8f4fd 0%, #dce8f5 100%);
    border: 2px dashed #5b9bd5;
    border-radius: 12px;
    padding: 2rem;
    text-align: center;
    margin: 1.5rem 0;
}

/* ---- Prediction result cards ---- */
.pred-card {
    border-radius: 10px;
    padding: 1.5rem;
    text-align: center;
    color: #fff;
    font-size: 1.1rem;
}
.pred-readmit  { background: linear-gradient(135deg, #e74c3c, #c0392b); }
.pred-safe     { background: linear-gradient(135deg, #27ae60, #1e8449); }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Session-state defaults
# ═══════════════════════════════════════════════════════════════════════════════

_DEFAULTS = {
    "df": None,
    "filename": None,
    "column_config": None,
    "target_info": None,
    "target_confirmed": False,
    "model_results": None,
    "trained_on_file": None,
    "page": "🏠 Home",
}
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ═══════════════════════════════════════════════════════════════════════════════
# File-loading helpers
# ═══════════════════════════════════════════════════════════════════════════════

SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls"}
MAX_FILE_MB = 200  # upload size guard


def _read_file(file_bytes: bytes, filename: str, sheet: str | None = None) -> pd.DataFrame:
    """Read a CSV or Excel file from raw bytes."""
    ext = Path(filename).suffix.lower()
    buf = io.BytesIO(file_bytes)
    if ext == ".csv":
        return pd.read_csv(buf)
    if ext in {".xlsx", ".xls"}:
        return pd.read_excel(buf, sheet_name=sheet, engine="openpyxl")
    raise ValueError(f"Unsupported extension: {ext}")


def _list_zip_datasets(zip_bytes: bytes) -> list[str]:
    """Return names of CSV / Excel files inside a ZIP (skip directories & hidden files)."""
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        safe: list[str] = []
        for info in zf.infolist():
            if info.is_dir():
                continue
            name = info.filename
            # Security: reject absolute or traversal paths
            if name.startswith("/") or ".." in name:
                continue
            if Path(name).suffix.lower() in SUPPORTED_EXTENSIONS:
                safe.append(name)
        return safe


def _read_from_zip(zip_bytes: bytes, inner_name: str) -> pd.DataFrame:
    """Extract and read one dataset from a ZIP archive."""
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        data = zf.read(inner_name)
    return _read_file(data, inner_name)


# ═══════════════════════════════════════════════════════════════════════════════
# Sidebar
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 🏥 Readmission Predictor")
    st.caption("Hospital Patient Readmission\nPrediction Using ML")
    st.divider()

    page = st.radio(
        "Navigate",
        [
            "🏠 Home",
            "📊 Dataset Analysis",
            "🔍 Patient Lookup",
            "🆕 New Patient Prediction",
            "📈 Model Performance",
            "ℹ️ About",
        ],
        label_visibility="collapsed",
    )
    st.session_state["page"] = page

    st.divider()

    # Status badges
    if st.session_state.df is not None:
        st.success(f"📁 **{st.session_state.filename}**")
        rows, cols = st.session_state.df.shape
        st.info(f"Rows: {rows:,}  ·  Columns: {cols}")
    else:
        st.warning("No dataset loaded")

    if st.session_state.model_results is not None:
        best = st.session_state.model_results.get("best_model_name", "—")
        st.success(f"✅ Model trained\n\n**{best}**")
    else:
        st.caption("Model not trained yet")


# ═══════════════════════════════════════════════════════════════════════════════
# Helper: invalidate model when dataset changes
# ═══════════════════════════════════════════════════════════════════════════════

def _invalidate_model():
    st.session_state.model_results = None
    st.session_state.trained_on_file = None
    st.session_state.target_confirmed = False
    st.session_state.target_info = None
    st.session_state.column_config = None


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE: Home
# ═══════════════════════════════════════════════════════════════════════════════

def page_home():
    st.title("🏥 Hospital Patient Readmission Prediction")
    st.markdown(
        "Predict whether a patient will be **readmitted within 30 days** "
        "using machine learning trained on *your* hospital dataset."
    )

    # ── Upload section ───────────────────────────────────────────────────
    st.markdown("---")
    st.header("📤 Upload Your Hospital Dataset")
    st.markdown(
        '<div class="upload-zone">'
        "<h4>Drop your file here or click <em>Browse files</em></h4>"
        "<p>Supported formats: <strong>CSV</strong>, <strong>Excel (.xlsx / .xls)</strong>, "
        "<strong>ZIP</strong> containing a CSV or Excel file</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    uploaded = st.file_uploader(
        "Upload your hospital patient dataset",
        type=["csv", "xlsx", "xls", "zip"],
        label_visibility="collapsed",
    )

    if uploaded is None:
        if st.session_state.df is not None:
            st.info(
                f"Currently loaded: **{st.session_state.filename}** "
                f"({st.session_state.df.shape[0]:,} rows, "
                f"{st.session_state.df.shape[1]} columns)"
            )
        else:
            st.info("👆 Upload a dataset to get started.")
        return

    # Guard: file size
    raw = uploaded.getvalue()
    size_mb = len(raw) / (1024 * 1024)
    if size_mb > MAX_FILE_MB:
        st.error(f"File is {size_mb:.1f} MB — maximum supported size is {MAX_FILE_MB} MB.")
        return

    fname = uploaded.name

    # ── ZIP handling ─────────────────────────────────────────────────────
    if Path(fname).suffix.lower() == ".zip":
        try:
            candidates = _list_zip_datasets(raw)
        except zipfile.BadZipFile:
            st.error("The uploaded file is not a valid ZIP archive.")
            return

        if not candidates:
            st.error("No CSV or Excel files found inside the ZIP archive.")
            return

        if len(candidates) == 1:
            inner = candidates[0]
            st.info(f"Found **{inner}** inside the archive — loading automatically.")
        else:
            inner = st.selectbox(
                "Multiple datasets found — select one:",
                candidates,
            )

        try:
            df = _read_from_zip(raw, inner)
            fname = inner  # use the inner filename for display
        except Exception as exc:
            st.error(f"Failed to read **{inner}** from ZIP: {exc}")
            return

    # ── Excel sheet selection ────────────────────────────────────────────
    elif Path(fname).suffix.lower() in {".xlsx", ".xls"}:
        try:
            xls = pd.ExcelFile(io.BytesIO(raw), engine="openpyxl")
            sheets = xls.sheet_names
        except Exception as exc:
            st.error(f"Cannot read Excel file: {exc}")
            return

        if len(sheets) > 1:
            sheet = st.selectbox("Select worksheet:", sheets)
        else:
            sheet = sheets[0]

        try:
            df = pd.read_excel(io.BytesIO(raw), sheet_name=sheet, engine="openpyxl")
        except Exception as exc:
            st.error(f"Failed to read sheet **{sheet}**: {exc}")
            return

    # ── CSV ──────────────────────────────────────────────────────────────
    else:
        try:
            df = pd.read_csv(io.BytesIO(raw))
        except Exception as exc:
            st.error(f"Failed to parse CSV: {exc}")
            return

    if df.empty:
        st.error("The uploaded file contains no data rows.")
        return

    # ── Persist and invalidate old model if file changed ─────────────────
    if st.session_state.filename != fname:
        _invalidate_model()

    st.session_state.df = df
    st.session_state.filename = fname

    # ── Quick preview ────────────────────────────────────────────────────
    st.success(f"✅ Loaded **{fname}** — {df.shape[0]:,} rows × {df.shape[1]} columns")
    st.dataframe(df.head(10), use_container_width=True)

    # ── Column detection ─────────────────────────────────────────────────
    config = detect_columns(df)
    st.session_state.column_config = config

    st.subheader("🔎 Automatic Column Detection")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**Target column:** `{config['target_col'] or '⚠️ not detected'}`")
        st.markdown(f"**Patient ID column:** `{config['id_col'] or '⚠️ not detected'}`")
    with c2:
        st.markdown(f"**Numeric features:** {len(config['numeric_cols'])}")
        st.markdown(f"**Categorical features:** {len(config['categorical_cols'])}")

    if config["confidence"] != "high" or config["target_col"] is None:
        st.warning("Auto-detection is uncertain — please verify below.")

    # ── Manual column mapping ────────────────────────────────────────────
    with st.expander("⚙️ Column Mapping — click to review or adjust", expanded=(config["target_col"] is None)):
        all_cols = ["(none)"] + list(df.columns)

        sel_target = st.selectbox(
            "Readmission target column",
            all_cols,
            index=all_cols.index(config["target_col"]) if config["target_col"] in all_cols else 0,
        )
        sel_id = st.selectbox(
            "Patient ID column",
            all_cols,
            index=all_cols.index(config["id_col"]) if config["id_col"] in all_cols else 0,
        )

        exclude = {sel_target, sel_id, "(none)"}
        default_features = [c for c in df.columns if c not in exclude]
        sel_features = st.multiselect(
            "Feature columns (used for training)",
            [c for c in df.columns if c not in {sel_target, sel_id}],
            default=default_features,
        )

        if st.button("💾 Apply Mapping"):
            if sel_target == "(none)":
                st.error("You must select a target column for supervised training.")
            elif not sel_features:
                st.error("Select at least one feature column.")
            else:
                new_cfg = detect_columns(df)
                new_cfg["target_col"] = sel_target
                new_cfg["id_col"] = sel_id if sel_id != "(none)" else None
                new_cfg["feature_cols"] = sel_features

                # Re-classify selected features
                new_cfg["numeric_cols"] = [c for c in sel_features if pd.api.types.is_numeric_dtype(df[c]) or _try_numeric(df[c])]
                new_cfg["categorical_cols"] = [c for c in sel_features if c not in new_cfg["numeric_cols"]]

                # Detect BP column
                for c in sel_features:
                    if normalize_name(c) in ["blood_pressure", "bp", "bloodpressure"]:
                        new_cfg["bp_col"] = c
                        if c in new_cfg["numeric_cols"]:
                            new_cfg["numeric_cols"].remove(c)
                        if c not in new_cfg["categorical_cols"]:
                            new_cfg["categorical_cols"].append(c)
                        break

                st.session_state.column_config = new_cfg
                _invalidate_model()
                st.success("✅ Column mapping updated.")
                st.rerun()

    # ── Target label confirmation ────────────────────────────────────────
    cfg = st.session_state.column_config
    if cfg and cfg["target_col"]:
        target_series = df[cfg["target_col"]]
        tinfo = detect_target_labels(target_series)
        st.session_state.target_info = tinfo

        st.subheader("🏷️ Target Label Detection")

        if "error" in tinfo:
            st.error(tinfo["error"])
            st.info(
                "Supervised training requires exactly **two** outcome classes "
                "(e.g. Yes/No, 1/0). Please correct the target column."
            )
        else:
            tc1, tc2 = st.columns(2)
            with tc1:
                st.metric("Positive (Readmitted)", str(tinfo["positive_label"]))
            with tc2:
                st.metric("Negative (Not Readmitted)", str(tinfo["negative_label"]))

            # Value counts
            for label, count in tinfo["value_counts"].items():
                pct = count / len(target_series) * 100
                st.write(f"  `{label}`: **{count:,}** ({pct:.1f}%)")

            if tinfo["needs_confirmation"]:
                st.warning(
                    "Could not auto-detect which label means *Readmitted*. "
                    "Please confirm below."
                )
                swap = st.checkbox("Swap labels (mark the other value as Readmitted)")
                if swap:
                    tinfo["positive_label"], tinfo["negative_label"] = (
                        tinfo["negative_label"],
                        tinfo["positive_label"],
                    )
                    tinfo["mapping"] = {
                        tinfo["positive_label"]: 1,
                        tinfo["negative_label"]: 0,
                    }
                    st.session_state.target_info = tinfo

            if st.button("✅ Confirm Labels & Proceed"):
                st.session_state.target_confirmed = True
                st.success("Labels confirmed.")

            if not tinfo["needs_confirmation"]:
                st.session_state.target_confirmed = True

    # ── Train button ─────────────────────────────────────────────────────
    if st.session_state.target_confirmed and st.session_state.column_config:
        st.markdown("---")
        st.subheader("🚀 Train Model")

        if st.session_state.model_results is not None and st.session_state.trained_on_file == fname:
            st.info(
                f"A model is already trained on this dataset "
                f"(**{st.session_state.model_results['best_model_name']}**). "
                f"Click below to retrain."
            )

        if st.button("🚀 Train Model", type="primary"):
            cfg = st.session_state.column_config
            tinfo = st.session_state.target_info
            with st.spinner("Training models… this may take a moment."):
                res = train_and_evaluate(df, cfg, tinfo["mapping"])

            if "error" in res:
                st.error(f"Training failed: {res['error']}")
            else:
                st.session_state.model_results = res
                st.session_state.trained_on_file = fname
                st.success(
                    f"✅ Training complete!  Best model: **{res['best_model_name']}** "
                    f"(Test F1: {res['test_metrics']['f1']:.3f})"
                )
                st.balloons()


def _try_numeric(series: pd.Series) -> bool:
    try:
        pd.to_numeric(series.dropna(), errors="raise")
        return True
    except (ValueError, TypeError):
        return False


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE: Dataset Analysis
# ═══════════════════════════════════════════════════════════════════════════════

def page_analysis():
    st.title("📊 Dataset Analysis")

    if st.session_state.df is None:
        st.warning("No dataset loaded. Go to **Home** to upload one.")
        return

    df = st.session_state.df
    cfg = st.session_state.column_config or {}

    # ── Dimensions ───────────────────────────────────────────────────────
    st.subheader("Dataset Overview")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Rows", f"{df.shape[0]:,}")
    m2.metric("Columns", df.shape[1])
    m3.metric("Numeric", len(cfg.get("numeric_cols", [])))
    m4.metric("Categorical", len(cfg.get("categorical_cols", [])))

    # ── Preview ──────────────────────────────────────────────────────────
    st.subheader("Data Preview")
    st.dataframe(df.head(10), use_container_width=True)

    # ── Data types ───────────────────────────────────────────────────────
    with st.expander("Column Data Types"):
        dtypes_df = pd.DataFrame({
            "Column": df.columns,
            "Type": [str(dt) for dt in df.dtypes],
            "Non-Null": [df[c].count() for c in df.columns],
            "Null": [df[c].isna().sum() for c in df.columns],
            "Null %": [f"{df[c].isna().mean()*100:.1f}%" for c in df.columns],
            "Unique": [df[c].nunique() for c in df.columns],
        })
        st.dataframe(dtypes_df, use_container_width=True, hide_index=True)

    # ── Missing values ───────────────────────────────────────────────────
    st.subheader("Missing Values")
    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if missing.empty:
        st.success("No missing values detected.")
    else:
        fig = px.bar(
            x=missing.index, y=missing.values,
            labels={"x": "Column", "y": "Missing Count"},
            title="Missing Values by Column",
            color_discrete_sequence=["#e74c3c"],
        )
        fig.update_layout(template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

    # ── Duplicates ───────────────────────────────────────────────────────
    dup_count = df.duplicated().sum()
    st.metric("Duplicate Rows", f"{dup_count:,}")

    # ── Class distribution ───────────────────────────────────────────────
    target_col = cfg.get("target_col")
    if target_col and target_col in df.columns:
        st.subheader("Target Class Distribution")
        vc = df[target_col].value_counts()
        c1, c2 = st.columns(2)
        with c1:
            fig = px.pie(
                names=vc.index.astype(str),
                values=vc.values,
                title=f"Distribution of '{target_col}'",
                color_discrete_sequence=["#27ae60", "#e74c3c"],
                hole=0.4,
            )
            fig.update_layout(template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            for label, count in vc.items():
                pct = count / len(df) * 100
                st.metric(str(label), f"{count:,} ({pct:.1f}%)")
            ratio = vc.max() / vc.min() if vc.min() > 0 else float("inf")
            if ratio > 3:
                st.warning(f"Class imbalance detected (ratio ≈ {ratio:.1f}:1).")
            else:
                st.success("Classes are reasonably balanced.")

    # ── Numeric distributions ────────────────────────────────────────────
    num_cols = cfg.get("numeric_cols", [])
    if num_cols:
        st.subheader("Numeric Feature Distributions")
        sel = st.selectbox("Select feature", num_cols)
        if sel:
            fig = px.histogram(
                df, x=sel, nbins=40,
                color_discrete_sequence=["#0d6efd"],
                title=f"Distribution of {sel}",
            )
            fig.update_layout(template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

    # ── Categorical ──────────────────────────────────────────────────────
    cat_cols = cfg.get("categorical_cols", [])
    if cat_cols:
        st.subheader("Categorical Feature Counts")
        sel_cat = st.selectbox("Select categorical feature", cat_cols, key="cat_sel")
        if sel_cat:
            vc = df[sel_cat].value_counts().head(20)
            fig = px.bar(
                x=vc.index.astype(str), y=vc.values,
                labels={"x": sel_cat, "y": "Count"},
                color_discrete_sequence=["#17a2b8"],
                title=f"Top categories — {sel_cat}",
            )
            fig.update_layout(template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE: Patient Lookup
# ═══════════════════════════════════════════════════════════════════════════════

def page_lookup():
    st.title("🔍 Search Patient by ID")

    if st.session_state.df is None:
        st.warning("No dataset loaded. Go to **Home** to upload one.")
        return

    cfg = st.session_state.column_config or {}
    id_col = cfg.get("id_col")
    if not id_col:
        st.error("No Patient ID column was detected or mapped. Please set it on the Home page.")
        return

    df = st.session_state.df

    search_id = st.text_input("Enter Patient ID", placeholder=f"e.g. {df[id_col].iloc[0]}")

    if st.button("🔍 Search", type="primary") and search_id:
        # Match (convert both to str for safety)
        mask = df[id_col].astype(str).str.strip() == str(search_id).strip()
        matches = df[mask]

        if matches.empty:
            st.error(f"No patient found with ID **{search_id}**.")
            st.info(f"The dataset contains {df[id_col].nunique():,} unique IDs.")
            return

        if len(matches) > 1:
            st.warning(
                f"**{len(matches)}** records found for ID **{search_id}**. "
                "This patient has duplicate entries — select a row."
            )
            idx = st.selectbox("Select row index", matches.index.tolist())
            patient = matches.loc[idx]
        else:
            patient = matches.iloc[0]

        # ── Display patient details ──────────────────────────────────────
        st.subheader("📋 Patient Details")
        detail_df = pd.DataFrame({
            "Field": patient.index,
            "Value": patient.values,
        })
        st.dataframe(detail_df, use_container_width=True, hide_index=True)

        # ── Historical outcome ───────────────────────────────────────────
        target_col = cfg.get("target_col")
        if target_col and target_col in patient.index:
            st.markdown("**Historical Outcome (from dataset):**")
            hist_val = patient[target_col]
            st.info(f"`{target_col}` = **{hist_val}**")

        # ── Model prediction ────────────────────────────────────────────
        model_res = st.session_state.model_results
        if model_res and "pipeline" in model_res:
            st.subheader("🤖 Model Prediction")

            feature_cols = model_res["feature_cols"]
            # Build feature dict — handle BP splitting
            patient_series = patient.copy()
            bp_col = cfg.get("bp_col")
            if bp_col and bp_col in patient_series.index:
                from train_model import split_blood_pressure
                tmp = pd.DataFrame([patient_series])
                tmp, _ = split_blood_pressure(tmp, bp_col)
                patient_series = tmp.iloc[0]

            feat_dict = {c: patient_series.get(c) for c in feature_cols}
            try:
                pred = predict_single(model_res["pipeline"], feat_dict, feature_cols)
            except Exception as exc:
                st.error(f"Prediction error: {exc}")
                return

            if pred["prediction"] == 1:
                st.markdown(
                    '<div class="pred-card pred-readmit">'
                    "⚠️ <strong>Predicted: READMITTED</strong></div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div class="pred-card pred-safe">'
                    "✅ <strong>Predicted: NOT READMITTED</strong></div>",
                    unsafe_allow_html=True,
                )

            if "probability_readmitted" in pred:
                p1, p2 = st.columns(2)
                p1.metric("P(Readmitted)", f"{pred['probability_readmitted']:.1%}")
                p2.metric("P(Not Readmitted)", f"{pred['probability_not_readmitted']:.1%}")

            if target_col:
                st.caption(
                    "⚖️ The **historical outcome** is what actually happened. "
                    "The **model prediction** is what the model estimates from input features alone."
                )
        else:
            st.info("Train a model on the **Home** page to see predictions here.")


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE: New Patient Prediction
# ═══════════════════════════════════════════════════════════════════════════════

def page_predict():
    st.title("🆕 New Patient Prediction")

    model_res = st.session_state.model_results
    if not model_res or "pipeline" not in model_res:
        st.warning("No trained model available. Train a model on the **Home** page first.")
        return

    cfg = st.session_state.column_config or {}
    df = st.session_state.df

    feature_cols = model_res["feature_cols"]
    numeric_cols = model_res["numeric_cols"]
    categorical_cols = model_res["categorical_cols"]

    st.markdown(
        "Enter the patient's details below. "
        "The form adapts to the features used during training."
    )

    with st.form("new_patient_form"):
        values: dict = {}

        # Create two-column layout for the form
        col_a, col_b = st.columns(2)
        for i, col in enumerate(feature_cols):
            target_col = col_a if i % 2 == 0 else col_b
            with target_col:
                if col in numeric_cols:
                    # Provide sensible defaults from training data
                    if df is not None and col in df.columns:
                        med = float(df[col].median()) if pd.api.types.is_numeric_dtype(df[col]) else 0.0
                        mn = float(df[col].min()) if pd.api.types.is_numeric_dtype(df[col]) else 0.0
                        mx = float(df[col].max()) if pd.api.types.is_numeric_dtype(df[col]) else 1000.0
                    else:
                        med, mn, mx = 0.0, 0.0, 1000.0
                    values[col] = st.number_input(
                        col, value=med, min_value=mn, max_value=mx * 2, key=f"np_{col}",
                    )
                elif col in categorical_cols:
                    if df is not None and col in df.columns:
                        options = sorted(df[col].dropna().unique().astype(str))
                    else:
                        options = ["Unknown"]
                    values[col] = st.selectbox(col, options, key=f"np_{col}")
                else:
                    values[col] = st.text_input(col, key=f"np_{col}")

        submitted = st.form_submit_button("🔮 Predict", type="primary")

    if submitted:
        try:
            pred = predict_single(model_res["pipeline"], values, feature_cols)
        except Exception as exc:
            st.error(f"Prediction error: {exc}")
            return

        st.markdown("---")
        st.subheader("Prediction Result")

        if pred["prediction"] == 1:
            st.markdown(
                '<div class="pred-card pred-readmit">'
                "⚠️ <strong>Predicted: READMITTED within 30 days</strong></div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="pred-card pred-safe">'
                "✅ <strong>Predicted: NOT READMITTED within 30 days</strong></div>",
                unsafe_allow_html=True,
            )

        if "probability_readmitted" in pred:
            st.markdown("")
            pc1, pc2 = st.columns(2)
            pc1.metric("P(Readmitted)", f"{pred['probability_readmitted']:.1%}")
            pc2.metric("P(Not Readmitted)", f"{pred['probability_not_readmitted']:.1%}")

            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=pred["probability_readmitted"] * 100,
                title={"text": "Readmission Risk (%)"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#e74c3c" if pred["probability_readmitted"] > 0.5 else "#27ae60"},
                    "steps": [
                        {"range": [0, 30], "color": "#d4edda"},
                        {"range": [30, 60], "color": "#fff3cd"},
                        {"range": [60, 100], "color": "#f8d7da"},
                    ],
                },
            ))
            fig.update_layout(height=300, template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

        st.caption(
            "⚠️ This prediction is an estimate from a machine-learning model. "
            "It is **not** a clinical diagnosis."
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE: Model Performance
# ═══════════════════════════════════════════════════════════════════════════════

def page_performance():
    st.title("📈 Model Performance")

    res = st.session_state.model_results
    if not res or "test_metrics" not in res:
        st.warning("No model has been trained yet. Go to **Home** to upload data and train a model.")
        return

    tm = res["test_metrics"]
    best = res["best_model_name"]

    st.success(f"**Best model:** {best}")

    # ── Key metrics ──────────────────────────────────────────────────────
    st.subheader("Test-Set Metrics (Readmission Class)")
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Accuracy", f"{tm['accuracy']:.3f}")
    k2.metric("Precision", f"{tm['precision']:.3f}")
    k3.metric("Recall", f"{tm['recall']:.3f}")
    k4.metric("F1-Score", f"{tm['f1']:.3f}")
    k5.metric("ROC-AUC", f"{tm['roc_auc']:.3f}" if tm["roc_auc"] is not None else "N/A")

    # ── Confusion matrix ─────────────────────────────────────────────────
    st.subheader("Confusion Matrix")
    cm = tm["confusion_matrix"]
    labels = ["Not Readmitted", "Readmitted"]
    fig_cm = px.imshow(
        cm,
        text_auto=True,
        x=labels, y=labels,
        labels={"x": "Predicted", "y": "Actual", "color": "Count"},
        color_continuous_scale="Blues",
        title="Test-Set Confusion Matrix",
    )
    fig_cm.update_layout(template="plotly_white", height=400)
    st.plotly_chart(fig_cm, use_container_width=True)

    # ── Actual vs predicted distribution ─────────────────────────────────
    st.subheader("Actual vs. Predicted Distribution")
    y_test = tm["y_test"]
    y_pred = tm["y_test_pred"]
    dist_df = pd.DataFrame({
        "Class": labels * 2,
        "Count": [
            int((y_test == 0).sum()), int((y_test == 1).sum()),
            int((y_pred == 0).sum()), int((y_pred == 1).sum()),
        ],
        "Source": ["Actual"] * 2 + ["Predicted"] * 2,
    })
    fig_dist = px.bar(
        dist_df, x="Class", y="Count", color="Source",
        barmode="group", color_discrete_sequence=["#0d6efd", "#e74c3c"],
        title="Actual vs Predicted Class Counts (Test Set)",
    )
    fig_dist.update_layout(template="plotly_white")
    st.plotly_chart(fig_dist, use_container_width=True)

    # ── ROC Curve ────────────────────────────────────────────────────────
    if tm["y_test_proba"] is not None and tm["roc_auc"] is not None:
        st.subheader("ROC Curve")
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(y_test, tm["y_test_proba"])
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{best} (AUC={tm['roc_auc']:.3f})"))
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="gray"), name="Random"))
        fig_roc.update_layout(
            title="ROC Curve", xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate", template="plotly_white", height=400,
        )
        st.plotly_chart(fig_roc, use_container_width=True)

    # ── Classification report ────────────────────────────────────────────
    st.subheader("Full Classification Report")
    st.code(tm["classification_report_text"])

    # ── Model comparison ─────────────────────────────────────────────────
    st.subheader("Model Comparison (Validation Set)")
    comp = res["model_comparison"]
    comp_rows = []
    for name, info in comp.items():
        if info["status"] == "Trained":
            comp_rows.append({
                "Model": name,
                "Val Accuracy": f"{info['val_accuracy']:.3f}",
                "Val Precision": f"{info['val_precision']:.3f}",
                "Val Recall": f"{info['val_recall']:.3f}",
                "Val F1": f"{info['val_f1']:.3f}",
                "Selected": "✅" if info["selected"] else "",
            })
        else:
            comp_rows.append({"Model": name, "Val Accuracy": "Failed", "Val Precision": "", "Val Recall": "", "Val F1": "", "Selected": ""})

    st.dataframe(pd.DataFrame(comp_rows), use_container_width=True, hide_index=True)

    # ── Data split info ──────────────────────────────────────────────────
    st.subheader("Data Splits")
    splits = res["split_sizes"]
    s1, s2, s3 = st.columns(3)
    s1.metric("Train", f"{splits['train']:,}")
    s2.metric("Validation", f"{splits['val']:,}")
    s3.metric("Test", f"{splits['test']:,}")

    if not res.get("stratified", True):
        st.warning(
            "⚠️ Stratified splitting was not possible due to very low class counts. "
            "A random split was used instead — metrics may be less reliable."
        )

    # ── Class distribution per split ─────────────────────────────────────
    with st.expander("Class distribution per split"):
        for split_name, counts in res["class_distribution"].items():
            st.write(f"**{split_name.title()}**: {counts}")


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE: About
# ═══════════════════════════════════════════════════════════════════════════════

def page_about():
    st.title("ℹ️ About This Application")

    st.markdown("""
### Project Description

This application uses **machine learning** to predict whether a hospital
patient will be readmitted within 30 days of discharge.  It is designed as a
complete, self-contained tool that lets you:

1. **Upload** your hospital dataset (CSV, Excel, or ZIP).
2. **Explore** the data with automatic detection and interactive charts.
3. **Train** multiple classifiers and pick the best one.
4. **Look up** any patient by ID and compare the prediction with the historical outcome.
5. **Predict** readmission risk for new patients not in the dataset.

---

### Machine-Learning Workflow

```
Upload Dataset
     │
     ▼
Auto-Detect Columns & Target Labels
     │
     ▼
User Confirms / Adjusts Mapping
     │
     ▼
Preprocess (impute, scale, encode, split BP)
     │
     ▼
Stratified Train / Validation / Test Split (60 / 20 / 20)
     │
     ▼
Train  ─── Logistic Regression
       ├── Random Forest  (class_weight='balanced')
       └── Gradient Boosting
     │
     ▼
Select Best Model by Validation F1 (readmission class)
     │
     ▼
Evaluate on Held-Out Test Set
     │
     ▼
Report Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix
```

---

### Handling Class Imbalance

- **Logistic Regression** and **Random Forest** use `class_weight='balanced'`
  to give the minority (readmitted) class higher weight.
- The app reports actual vs. predicted class distributions so you can verify
  the model does **not** predict a single class for every patient.
- If the model performs poorly, results are reported honestly.

---

### Known Limitations

| Limitation | Explanation |
|---|---|
| No external validation | Model is evaluated on a single random test split — not on a separate institution's data. |
| No time-series handling | The model does not account for temporal patterns across multiple admissions. |
| Feature set | The model uses only the features present in the uploaded dataset. Adding lab results or comorbidity indices could improve performance. |
| Threshold | The default decision threshold is 0.5. Threshold tuning is possible but not currently exposed in the UI. |
| Session-only | Models live in session state and are lost when the browser tab is closed unless saved to disk with joblib. |

---

### Medical Disclaimer

> ⚠️ **This application is a demonstration and educational tool.**
>
> It is **not** a certified medical device and must **not** be used for
> clinical decision-making, patient diagnosis, or treatment planning.
>
> Predictions are statistical estimates with inherent uncertainty.
> Always consult qualified healthcare professionals for medical decisions.

---

### Technology Stack

| Component | Technology |
|---|---|
| Web Framework | Streamlit |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Visualization | Plotly |
| Excel Support | OpenPyXL |
| Model Persistence | Joblib |
""")


# ═══════════════════════════════════════════════════════════════════════════════
# Routing
# ═══════════════════════════════════════════════════════════════════════════════

_PAGES = {
    "🏠 Home": page_home,
    "📊 Dataset Analysis": page_analysis,
    "🔍 Patient Lookup": page_lookup,
    "🆕 New Patient Prediction": page_predict,
    "📈 Model Performance": page_performance,
    "ℹ️ About": page_about,
}

_PAGES[page]()
