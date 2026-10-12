"""
train_model.py — Machine-learning module for Hospital Readmission Prediction.

Provides:
  • Automatic column-role detection (ID, target, features)
  • Target-label detection and mapping (Yes/No, 1/0, etc.)
  • Blood-pressure splitting (systolic / diastolic)
  • Scikit-learn Pipeline + ColumnTransformer construction
  • Multi-model training (Logistic Regression, Random Forest, Gradient Boosting)
  • Stratified train / validation / test split
  • Best-model selection on validation F1
  • Hold-out test evaluation with full classification report
  • Single-patient prediction with probability
"""

import re
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ═══════════════════════════════════════════════════════════════════════════════
# Column-name utilities
# ═══════════════════════════════════════════════════════════════════════════════

def normalize_name(name: str) -> str:
    """Lowercase, strip, and collapse whitespace / underscores."""
    return re.sub(r"[\s_]+", "_", str(name).strip().lower())


# Aliases for automatic column mapping
TARGET_ALIASES: List[str] = [
    "readmitted_30_days", "readmitted_30days", "readmitted",
    "readmission", "readmit", "readmission_30_days",
    "readmitted_30_day", "re_admitted",
]

ID_ALIASES: List[str] = [
    "patient_id", "patientid", "patient_no", "patient_number", "pid",
    "mrn", "medical_record_number",
]

EXPECTED_FEATURES: Dict[str, List[str]] = {
    "age":                   ["age", "patient_age", "age_years", "age_at_admission"],
    "gender":                ["gender", "sex", "patient_gender", "patient_sex"],
    "blood_pressure":        ["blood_pressure", "bp", "bloodpressure", "blood_pressure_reading"],
    "cholesterol":           ["cholesterol", "chol", "cholesterol_level", "total_cholesterol"],
    "bmi":                   ["bmi", "body_mass_index", "bodymassindex"],
    "diabetes":              ["diabetes", "diabetic", "diabetes_status", "has_diabetes"],
    "hypertension":          ["hypertension", "htn", "hypertension_status", "has_hypertension"],
    "medication_count":      ["medication_count", "med_count", "medications",
                              "num_medications", "medication_number", "number_of_medications"],
    "length_of_stay":        ["length_of_stay", "los", "stay_duration",
                              "hospital_stay", "days_in_hospital"],
    "discharge_destination": ["discharge_destination", "discharge_dest",
                              "discharged_to", "discharge_location", "discharge_disposition"],
}


# ═══════════════════════════════════════════════════════════════════════════════
# Column detection
# ═══════════════════════════════════════════════════════════════════════════════

def detect_columns(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Examine *df* and return a dict describing auto-detected column roles.

    Keys:
        target_col, id_col, feature_cols, numeric_cols, categorical_cols,
        detected_mappings (standard_name → actual column), bp_col,
        all_columns, confidence ('high' | 'medium' | 'low').
    """
    norm_map: Dict[str, str] = {col: normalize_name(col) for col in df.columns}

    result: Dict[str, Any] = {
        "target_col": None,
        "id_col": None,
        "feature_cols": [],
        "numeric_cols": [],
        "categorical_cols": [],
        "detected_mappings": {},
        "bp_col": None,
        "all_columns": list(df.columns),
        "confidence": "high",
    }

    # --- Target ---
    for col, norm in norm_map.items():
        if norm in TARGET_ALIASES:
            result["target_col"] = col
            break
    if result["target_col"] is None:
        for col, norm in norm_map.items():
            if "readmi" in norm:
                result["target_col"] = col
                result["confidence"] = "medium"
                break

    # --- Patient ID ---
    for col, norm in norm_map.items():
        if norm in ID_ALIASES:
            result["id_col"] = col
            break
    if result["id_col"] is None:
        for col, norm in norm_map.items():
            if norm == "id" or (norm.endswith("_id") and col != result["target_col"]):
                result["id_col"] = col
                break

    # --- Expected features ---
    for std_name, aliases in EXPECTED_FEATURES.items():
        for col, norm in norm_map.items():
            if norm in aliases:
                result["detected_mappings"][std_name] = col
                if std_name == "blood_pressure":
                    result["bp_col"] = col
                break

    # --- Feature / type classification ---
    exclude = {result["target_col"], result["id_col"]}
    for col in df.columns:
        if col in exclude:
            continue
        result["feature_cols"].append(col)
        if _is_numeric_column(df[col]):
            result["numeric_cols"].append(col)
        else:
            result["categorical_cols"].append(col)

    return result


def _is_numeric_column(series: pd.Series) -> bool:
    """Return True when a column can be treated as numeric."""
    if pd.api.types.is_numeric_dtype(series):
        return True
    try:
        pd.to_numeric(series.dropna(), errors="raise")
        return True
    except (ValueError, TypeError):
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# Target-label detection
# ═══════════════════════════════════════════════════════════════════════════════

POSITIVE_PATTERNS = {"yes", "1", "true", "readmitted", "y", "1.0"}
NEGATIVE_PATTERNS = {"no", "0", "false", "not readmitted", "not_readmitted", "n", "0.0"}


def detect_target_labels(series: pd.Series) -> Dict[str, Any]:
    """
    Inspect a target Series and return label-mapping information.

    Returns dict with keys: unique_values, value_counts, positive_label,
    negative_label, mapping, is_binary, needs_confirmation, error (if any).
    """
    original_unique = series.dropna().unique()
    clean_unique = [str(v).strip().lower() for v in original_unique]

    result: Dict[str, Any] = {
        "unique_values": list(original_unique),
        "value_counts": series.value_counts().to_dict(),
        "positive_label": None,
        "negative_label": None,
        "mapping": {},
        "is_binary": len(set(clean_unique)) == 2,
        "needs_confirmation": True,
    }

    if not result["is_binary"]:
        result["error"] = (
            f"Expected exactly 2 classes but found {len(original_unique)}: "
            f"{list(original_unique)}"
        )
        return result

    pos_found = neg_found = None
    for val in original_unique:
        low = str(val).strip().lower()
        if low in POSITIVE_PATTERNS:
            pos_found = val
        elif low in NEGATIVE_PATTERNS:
            neg_found = val

    if pos_found is not None and neg_found is not None:
        result.update(positive_label=pos_found, negative_label=neg_found,
                      mapping={pos_found: 1, neg_found: 0},
                      needs_confirmation=False)
    elif pos_found is not None:
        neg_found = [v for v in original_unique if v != pos_found][0]
        result.update(positive_label=pos_found, negative_label=neg_found,
                      mapping={pos_found: 1, neg_found: 0})
    elif neg_found is not None:
        pos_found = [v for v in original_unique if v != neg_found][0]
        result.update(positive_label=pos_found, negative_label=neg_found,
                      mapping={pos_found: 1, neg_found: 0})
    else:
        vals = list(original_unique)
        result.update(positive_label=vals[0], negative_label=vals[1],
                      mapping={vals[0]: 1, vals[1]: 0},
                      needs_confirmation=True)

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Blood-pressure splitting
# ═══════════════════════════════════════════════════════════════════════════════

_BP_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*[/\\]\s*(\d+(?:\.\d+)?)\s*$")


def split_blood_pressure(df: pd.DataFrame, bp_col: str) -> Tuple[pd.DataFrame, List[str]]:
    """
    Split a column like ``'130/72'`` into *systolic_bp* and *diastolic_bp*.
    Returns (modified_df, [new_column_names]).
    """
    df = df.copy()

    def _parse(val):
        if pd.isna(val):
            return np.nan, np.nan
        m = _BP_RE.match(str(val))
        if m:
            return float(m.group(1)), float(m.group(2))
        try:
            return float(val), np.nan
        except (ValueError, TypeError):
            return np.nan, np.nan

    parsed = df[bp_col].apply(_parse)
    df["systolic_bp"] = parsed.apply(lambda t: t[0])
    df["diastolic_bp"] = parsed.apply(lambda t: t[1])
    return df, ["systolic_bp", "diastolic_bp"]


# ═══════════════════════════════════════════════════════════════════════════════
# Data preparation
# ═══════════════════════════════════════════════════════════════════════════════

def prepare_data(
    df: pd.DataFrame, config: Dict[str, Any]
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Prepare a DataFrame for training:
    • Splits blood-pressure if present.
    • Re-classifies columns into numeric / categorical.

    Returns (prepared_df, numeric_feature_cols, categorical_feature_cols).
    """
    df = df.copy()
    feature_cols = list(config["feature_cols"])
    numeric_cols = list(config["numeric_cols"])
    categorical_cols = list(config["categorical_cols"])

    bp_col = config.get("bp_col")
    if bp_col and bp_col in df.columns and bp_col in feature_cols:
        df, new_bp = split_blood_pressure(df, bp_col)
        feature_cols.remove(bp_col)
        for lst in (numeric_cols, categorical_cols):
            if bp_col in lst:
                lst.remove(bp_col)
        feature_cols.extend(new_bp)
        numeric_cols.extend(new_bp)

    # Re-verify types
    final_num, final_cat = [], []
    for col in feature_cols:
        if col in numeric_cols:
            final_num.append(col)
        elif col in categorical_cols:
            final_cat.append(col)
        elif _is_numeric_column(df[col]):
            final_num.append(col)
        else:
            final_cat.append(col)

    return df, final_num, final_cat


# ═══════════════════════════════════════════════════════════════════════════════
# Pipeline construction
# ═══════════════════════════════════════════════════════════════════════════════

def build_preprocessor(
    numeric_features: List[str], categorical_features: List[str]
) -> ColumnTransformer:
    """Build a ColumnTransformer with imputation + scaling / encoding."""
    transformers = []
    if numeric_features:
        transformers.append((
            "num",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]),
            numeric_features,
        ))
    if categorical_features:
        transformers.append((
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]),
            categorical_features,
        ))
    return ColumnTransformer(transformers=transformers, remainder="drop")


# ═══════════════════════════════════════════════════════════════════════════════
# Training & evaluation
# ═══════════════════════════════════════════════════════════════════════════════

def train_and_evaluate(
    df: pd.DataFrame,
    config: Dict[str, Any],
    target_mapping: Dict,
) -> Dict[str, Any]:
    """
    Full training pipeline.

    1.  Prepare features (split BP, classify columns).
    2.  Map target → 0/1.
    3.  Stratified train / val / test split (60/20/20).
    4.  Train Logistic Regression, Random Forest, Gradient Boosting.
    5.  Select the best model by validation F1 for the positive class.
    6.  Evaluate the best model on the held-out test set.

    Returns a results dict (``'success': True`` when training succeeded).
    """
    df = df.copy()
    target_col = config["target_col"]

    # Prepare features
    df, numeric_cols, categorical_cols = prepare_data(df, config)
    feature_cols = numeric_cols + categorical_cols

    if not feature_cols:
        return {"error": "No feature columns available for training."}

    # Map target
    df["_target"] = df[target_col].map(target_mapping)
    df = df.dropna(subset=["_target"])
    df["_target"] = df["_target"].astype(int)

    if df["_target"].nunique() < 2:
        return {"error": "Target column has fewer than 2 classes after mapping."}

    X = df[feature_cols]
    y = df["_target"]

    class_counts = y.value_counts()
    min_class = class_counts.min()

    # Splitting
    use_stratify = min_class >= 4
    try:
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42,
            stratify=y if use_stratify else None,
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.25, random_state=42,
            stratify=y_temp if use_stratify else None,
        )
    except ValueError as exc:
        return {"error": f"Cannot split data: {exc}"}

    preprocessor = build_preprocessor(numeric_cols, categorical_cols)

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=42, solver="lbfgs",
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=150, learning_rate=0.1, max_depth=4, random_state=42,
        ),
    }

    results: Dict[str, Any] = {}
    best_name: Optional[str] = None
    best_f1 = -1.0

    for name, clf in models.items():
        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", clf),
        ])
        try:
            pipe.fit(X_train, y_train)
        except Exception as exc:  # noqa: BLE001
            results[name] = {"error": str(exc)}
            continue

        y_vp = pipe.predict(X_val)
        vf1 = f1_score(y_val, y_vp, pos_label=1, zero_division=0)
        vrc = recall_score(y_val, y_vp, pos_label=1, zero_division=0)
        vpc = precision_score(y_val, y_vp, pos_label=1, zero_division=0)
        vacc = accuracy_score(y_val, y_vp)

        results[name] = {
            "pipeline": pipe,
            "val_f1": vf1,
            "val_recall": vrc,
            "val_precision": vpc,
            "val_accuracy": vacc,
        }

        if vf1 > best_f1:
            best_f1 = vf1
            best_name = name

    if best_name is None:
        return {"error": "All models failed to train."}

    # ── Test evaluation ──────────────────────────────────────────────────
    best_pipe = results[best_name]["pipeline"]
    y_tp = best_pipe.predict(X_test)

    has_proba = hasattr(best_pipe, "predict_proba")
    y_tprob = best_pipe.predict_proba(X_test)[:, 1] if has_proba else None
    try:
        roc = roc_auc_score(y_test, y_tprob) if y_tprob is not None else None
    except ValueError:
        roc = None

    cm = confusion_matrix(y_test, y_tp)
    report_dict = classification_report(
        y_test, y_tp,
        target_names=["Not Readmitted", "Readmitted"],
        output_dict=True, zero_division=0,
    )
    report_text = classification_report(
        y_test, y_tp,
        target_names=["Not Readmitted", "Readmitted"],
        zero_division=0,
    )

    test_metrics = {
        "accuracy": accuracy_score(y_test, y_tp),
        "precision": precision_score(y_test, y_tp, pos_label=1, zero_division=0),
        "recall": recall_score(y_test, y_tp, pos_label=1, zero_division=0),
        "f1": f1_score(y_test, y_tp, pos_label=1, zero_division=0),
        "roc_auc": roc,
        "confusion_matrix": cm,
        "classification_report": report_dict,
        "classification_report_text": report_text,
        "y_test": y_test.values,
        "y_test_pred": y_tp,
        "y_test_proba": y_tprob,
    }

    comparison: Dict[str, Any] = {}
    for name, res in results.items():
        if "error" in res:
            comparison[name] = {"status": "Failed", "error": res["error"]}
        else:
            comparison[name] = {
                "status": "Trained",
                "val_f1": res["val_f1"],
                "val_recall": res["val_recall"],
                "val_precision": res["val_precision"],
                "val_accuracy": res["val_accuracy"],
                "selected": name == best_name,
            }

    return {
        "success": True,
        "best_model_name": best_name,
        "pipeline": best_pipe,
        "test_metrics": test_metrics,
        "model_comparison": comparison,
        "feature_cols": feature_cols,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "class_distribution": {
            "train": y_train.value_counts().to_dict(),
            "val": y_val.value_counts().to_dict(),
            "test": y_test.value_counts().to_dict(),
        },
        "split_sizes": {
            "train": len(X_train),
            "val": len(X_val),
            "test": len(X_test),
        },
        "stratified": use_stratify,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Single-patient prediction
# ═══════════════════════════════════════════════════════════════════════════════

def predict_single(
    pipeline, feature_values: Dict[str, Any], feature_cols: List[str]
) -> Dict[str, Any]:
    """Run the trained pipeline on a single patient's feature dict."""
    row = pd.DataFrame([{c: feature_values.get(c) for c in feature_cols}])

    prediction = int(pipeline.predict(row)[0])
    result: Dict[str, Any] = {
        "prediction": prediction,
        "label": "Readmitted" if prediction == 1 else "Not Readmitted",
    }

    if hasattr(pipeline, "predict_proba"):
        proba = pipeline.predict_proba(row)[0]
        result["probability_readmitted"] = float(proba[1])
        result["probability_not_readmitted"] = float(proba[0])

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Model persistence helpers
# ═══════════════════════════════════════════════════════════════════════════════

def save_model(pipeline, path: str, metadata: Optional[Dict] = None) -> None:
    """Persist a trained pipeline (and optional metadata) with joblib."""
    joblib.dump({"pipeline": pipeline, "metadata": metadata or {}}, path)


def load_model(path: str) -> Dict[str, Any]:
    """Load a previously saved pipeline + metadata."""
    return joblib.load(path)
