# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "numpy>=1.24",
#   "pandas>=2.0",
#   "scikit-learn>=1.3",
# ]
# ///
"""
Codebook-FIGS Tabular Benchmark Suite builder.
Loads 15 tabular datasets (10 classification + 5 regression) from sklearn/OpenML,
preprocesses them, generates 5-fold CV splits, creates mini versions,
and writes the full_data_out.json in exp_sel_data_out schema format.

Each data ROW is a separate example. Output is grouped by dataset.
"""
import json
import os
import sys
import traceback
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.datasets import (
    fetch_california_housing,
    fetch_openml,
    load_breast_cancer,
    load_diabetes,
)
from sklearn.model_selection import KFold, StratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
WORKSPACE = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(WORKSPACE, "temp", "datasets")
os.makedirs(OUTPUT_DIR, exist_ok=True)
RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def assign_fold_labels(n_samples: int, folds: list[dict]) -> np.ndarray:
    """Map each sample to the fold index where it appears as a TEST sample."""
    labels = np.full(n_samples, -1, dtype=int)
    for fold_idx, fold in enumerate(folds):
        for idx in fold["test"]:
            labels[idx] = fold_idx
    assert np.all(labels >= 0), "Some samples not assigned to any fold"
    return labels


def generate_cv_folds(X, y, task_type: str, n_splits: int = 5):
    if task_type == "classification":
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
        return [{"train": tr.tolist(), "test": te.tolist()} for tr, te in splitter.split(X, y)]
    else:
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
        return [{"train": tr.tolist(), "test": te.tolist()} for tr, te in splitter.split(X)]


def create_mini_indices(X, y, task_type: str, max_samples: int = 200):
    n = len(y)
    if n <= max_samples:
        return np.arange(n)
    if task_type == "classification":
        sss = StratifiedShuffleSplit(n_splits=1, train_size=max_samples, random_state=RANDOM_STATE)
        indices, _ = next(sss.split(X, y))
        return np.sort(indices)
    else:
        rng = np.random.RandomState(RANDOM_STATE)
        indices = rng.choice(n, size=max_samples, replace=False)
        return np.sort(indices)


def one_hot_encode(df: pd.DataFrame, cat_cols: list[str]):
    """One-hot encode categorical columns → (numpy array, feature_name list)."""
    result_cols, result_names = [], []
    for col in df.columns:
        if col in cat_cols:
            categories = sorted(df[col].dropna().unique())
            for cat in categories:
                result_cols.append((df[col] == cat).astype(float).values)
                clean = str(cat).replace(" ", "_").replace("'", "")
                result_names.append(f"{col}_{clean}")
        else:
            result_cols.append(pd.to_numeric(df[col], errors="coerce").values.astype(float))
            result_names.append(col)
    X = np.column_stack(result_cols) if result_cols else np.empty((len(df), 0))
    return X, result_names


def build_dataset_group(
    dataset_id: str,
    X: np.ndarray,
    y: np.ndarray,
    task_type: str,
    n_classes: int | None,
    domain: str,
    domain_description: str,
    source: str,
    feature_names: list[str],
    original_feature_names: list[str],
    feature_descriptions: dict | None,
    is_mini: bool = False,
    parent_indices: np.ndarray | None = None,
):
    """Build one dataset group entry (one element of the top-level 'datasets' array)."""
    n_samples, n_features = X.shape
    folds = generate_cv_folds(X, y, task_type)
    fold_labels = assign_fold_labels(n_samples, folds)

    did = dataset_id + ("_mini" if is_mini else "")

    examples = []
    for i in range(n_samples):
        input_vec = X[i].tolist()
        output_val = str(int(y[i])) if task_type == "classification" else str(float(y[i]))
        row_index = int(parent_indices[i]) if parent_indices is not None else i

        ex = {
            "input": json.dumps(input_vec),
            "output": output_val,
            "metadata_fold": int(fold_labels[i]),
            "metadata_row_index": row_index,
            "metadata_task_type": task_type,
        }
        examples.append(ex)

    # Attach rich metadata to the first example only (to avoid bloating the file)
    if examples:
        examples[0]["metadata_n_classes"] = n_classes if n_classes else 0
        examples[0]["metadata_n_samples"] = n_samples
        examples[0]["metadata_n_features"] = n_features
        examples[0]["metadata_domain"] = domain
        examples[0]["metadata_domain_description"] = domain_description
        examples[0]["metadata_source"] = source
        examples[0]["metadata_feature_names"] = feature_names          # proper list
        examples[0]["metadata_original_feature_names"] = original_feature_names  # proper list
        if feature_descriptions:
            examples[0]["metadata_feature_descriptions"] = json.dumps(feature_descriptions)

    return {"dataset": did, "examples": examples}


def validate_group(group: dict) -> list[str]:
    """Validate one dataset group. Returns list of error strings."""
    did = group["dataset"]
    examples = group["examples"]
    errors = []
    if not examples:
        return [f"{did}: No examples"]

    first_inp = json.loads(examples[0]["input"])
    n_features = len(first_inp)
    folds_seen = set()

    for i, ex in enumerate(examples):
        inp = json.loads(ex["input"])
        if len(inp) != n_features:
            errors.append(f"{did}[{i}]: input len {len(inp)} != {n_features}")
            break
        arr = np.array(inp)
        if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
            errors.append(f"{did}[{i}]: NaN/inf in input")
            break
        try:
            oval = float(ex["output"])
            if not np.isfinite(oval):
                errors.append(f"{did}[{i}]: non-finite output")
        except (ValueError, TypeError):
            errors.append(f"{did}[{i}]: output not numeric: {ex['output']!r}")
        folds_seen.add(ex["metadata_fold"])

    if folds_seen != {0, 1, 2, 3, 4}:
        errors.append(f"{did}: folds = {folds_seen}, expected {{0..4}}")

    fn = examples[0].get("metadata_feature_names")
    if fn is not None and len(fn) != n_features:
        errors.append(f"{did}: feature_names len {len(fn)} != {n_features}")

    return errors


# ---------------------------------------------------------------------------
# Dataset loaders – each returns (X, y, feature_names, orig_names, feat_descs, meta_dict)
# ---------------------------------------------------------------------------

def load_heart_disease():
    data = fetch_openml(data_id=49, as_frame=True, parser="auto")
    df = data.frame.dropna().reset_index(drop=True)
    target_col = "num"
    y_raw = df[target_col]
    feat_df = df.drop(columns=[target_col])
    cat_cols = feat_df.select_dtypes(include=["category", "object"]).columns.tolist()
    orig_names = list(feat_df.columns)
    X, fnames = one_hot_encode(feat_df, cat_cols)
    y = np.array([0 if v == "<50" else 1 for v in y_raw])
    descs = {
        "age": "Age in years", "sex": "Sex (male/female)",
        "cp": "Chest pain type (typical angina, atypical angina, non-anginal, asymptomatic)",
        "trestbps": "Resting blood pressure (mm Hg)", "chol": "Serum cholesterol (mg/dl)",
        "fbs": "Fasting blood sugar > 120 mg/dl", "restecg": "Resting ECG results",
        "thalach": "Maximum heart rate achieved", "exang": "Exercise induced angina",
        "oldpeak": "ST depression induced by exercise relative to rest",
        "slope": "Slope of peak exercise ST segment",
        "ca": "Number of major vessels colored by fluoroscopy",
        "thal": "Thalassemia (normal, fixed defect, reversible defect)",
    }
    meta = dict(task_type="classification", n_classes=2, domain="cardiology",
                domain_description="Predict presence of heart disease from clinical measurements",
                source="OpenML ID 49 (Cleveland Heart Disease)")
    return X, y, fnames, orig_names, descs, meta


def load_diabetes_pima():
    data = fetch_openml(data_id=37, as_frame=True, parser="auto")
    df = data.frame
    y_raw = df["class"]
    feat_df = df.drop(columns=["class"])
    orig_names = list(feat_df.columns)
    X = feat_df.values.astype(float)
    fnames = orig_names[:]
    y = np.array([1 if v == "tested_positive" else 0 for v in y_raw])
    descs = {
        "preg": "Number of pregnancies",
        "plas": "Plasma glucose concentration (2h oral glucose tolerance test)",
        "pres": "Diastolic blood pressure (mm Hg)", "skin": "Triceps skinfold thickness (mm)",
        "insu": "2-hour serum insulin (mu U/ml)", "mass": "Body mass index (weight/height^2)",
        "pedi": "Diabetes pedigree function", "age": "Age in years",
    }
    meta = dict(task_type="classification", n_classes=2, domain="diabetes",
                domain_description="Predict onset of diabetes from medical measurements in Pima Indian women",
                source="OpenML ID 37 (Pima Indians Diabetes)")
    return X, y, fnames, orig_names, descs, meta


def load_breast_cancer_wdbc():
    bc = load_breast_cancer()
    X = bc.data.astype(float)
    y = bc.target.astype(int)
    fnames = [str(fn) for fn in bc.feature_names]
    orig_names = fnames[:]
    descs = {}
    for fn in fnames:
        if fn.startswith("mean "):
            descs[fn] = f"Mean of {fn.replace('mean ', '')} measurements"
        elif "error" in fn:
            descs[fn] = f"Standard error of {fn.replace(' error', '')}"
        elif fn.startswith("worst "):
            descs[fn] = f"Worst (largest) value of {fn.replace('worst ', '')}"
    meta = dict(task_type="classification", n_classes=2, domain="oncology",
                domain_description="Predict malignant vs benign breast tumors from cell nucleus measurements",
                source="sklearn load_breast_cancer() / UCI WDBC")
    return X, y, fnames, orig_names, descs, meta


def load_credit_german():
    data = fetch_openml(data_id=31, as_frame=True, parser="auto")
    df = data.frame
    y_raw = df["class"]
    feat_df = df.drop(columns=["class"])
    cat_cols = feat_df.select_dtypes(include=["category", "object"]).columns.tolist()
    orig_names = list(feat_df.columns)
    X, fnames = one_hot_encode(feat_df, cat_cols)
    y = np.array([0 if v == "good" else 1 for v in y_raw])
    descs = {
        "checking_status": "Status of existing checking account",
        "duration": "Duration of credit in months", "credit_history": "Credit history",
        "purpose": "Purpose of credit", "credit_amount": "Credit amount",
        "savings_status": "Savings account/bonds status", "employment": "Present employment since",
        "installment_commitment": "Installment rate (% of disposable income)",
        "personal_status": "Personal status and sex", "other_parties": "Other debtors/guarantors",
        "residence_since": "Present residence since (years)", "property_magnitude": "Property type",
        "age": "Age in years", "other_payment_plans": "Other installment plans",
        "housing": "Housing (rent, own, free)", "existing_credits": "Number of existing credits",
        "job": "Job type", "num_dependents": "Number of dependents",
        "own_telephone": "Telephone registered", "foreign_worker": "Foreign worker",
    }
    meta = dict(task_type="classification", n_classes=2, domain="finance",
                domain_description="Predict credit risk (good/bad) from financial and personal attributes",
                source="OpenML ID 31 (German Credit)")
    return X, y, fnames, orig_names, descs, meta


def load_ionosphere():
    data = fetch_openml(data_id=59, as_frame=True, parser="auto")
    df = data.frame
    y_raw = df["class"]
    feat_df = df.drop(columns=["class"])
    if "a02" in feat_df.columns:
        feat_df = feat_df.drop(columns=["a02"])
    orig_names = list(feat_df.columns)
    X = feat_df.values.astype(float)
    fnames = orig_names[:]
    y = np.array([1 if v == "g" else 0 for v in y_raw])
    meta = dict(task_type="classification", n_classes=2, domain="radar",
                domain_description="Classify radar returns from the ionosphere as good or bad",
                source="OpenML ID 59 (Ionosphere)")
    return X, y, fnames, orig_names, None, meta


def load_sonar():
    data = fetch_openml(data_id=40, as_frame=True, parser="auto")
    df = data.frame
    y_raw = df["Class"]
    feat_df = df.drop(columns=["Class"])
    orig_names = list(feat_df.columns)
    X = feat_df.values.astype(float)
    fnames = orig_names[:]
    y = np.array([1 if v == "Mine" else 0 for v in y_raw])
    meta = dict(task_type="classification", n_classes=2, domain="signal_processing",
                domain_description="Classify sonar returns as mine or rock",
                source="OpenML ID 40 (Sonar / Mines vs Rocks)")
    return X, y, fnames, orig_names, None, meta


def load_vehicle():
    data = fetch_openml(data_id=54, as_frame=True, parser="auto")
    df = data.frame
    y_raw = df["Class"]
    feat_df = df.drop(columns=["Class"])
    orig_names = list(feat_df.columns)
    X = feat_df.values.astype(float)
    fnames = orig_names[:]
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    nc = len(le.classes_)
    meta = dict(task_type="classification", n_classes=nc, domain="computer_vision",
                domain_description=f"Classify vehicle silhouettes into {nc} types: {', '.join(str(c) for c in le.classes_)}",
                source="OpenML ID 54 (Vehicle Silhouettes)")
    return X, y, fnames, orig_names, None, meta


def load_segment():
    data = fetch_openml(data_id=36, as_frame=True, parser="auto")
    df = data.frame
    y_raw = df["class"]
    feat_df = df.drop(columns=["class"])
    drop_cols = ["region-centroid-col", "region-centroid-row", "region-pixel-count"]
    feat_df = feat_df.drop(columns=[c for c in drop_cols if c in feat_df.columns])
    orig_names = list(feat_df.columns)
    X = feat_df.values.astype(float)
    fnames = orig_names[:]
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    nc = len(le.classes_)
    meta = dict(task_type="classification", n_classes=nc, domain="computer_vision",
                domain_description=f"Classify image segments into {nc} types: {', '.join(str(c) for c in le.classes_)}",
                source="OpenML ID 36 (Image Segmentation)")
    return X, y, fnames, orig_names, None, meta


def load_climate_crashes():
    data = fetch_openml(data_id=40994, as_frame=True, parser="auto")
    df = data.frame
    y_raw = df["outcome"]
    feat_df = df.drop(columns=["outcome"])
    for col in ["Study", "Run"]:
        if col in feat_df.columns:
            feat_df = feat_df.drop(columns=[col])
    orig_names = list(feat_df.columns)
    X = feat_df.values.astype(float)
    fnames = orig_names[:]
    y = np.array([int(float(v)) for v in y_raw])
    meta = dict(task_type="classification", n_classes=2, domain="climate_science",
                domain_description="Predict climate model simulation crashes from input parameters",
                source="OpenML ID 40994 (Climate Model Simulation Crashes)")
    return X, y, fnames, orig_names, None, meta


def load_spambase():
    data = fetch_openml(data_id=44, as_frame=True, parser="auto")
    df = data.frame
    y_raw = df["class"]
    feat_df = df.drop(columns=["class"])
    orig_names = list(feat_df.columns)
    X = feat_df.values.astype(float)
    fnames = orig_names[:]
    y = np.array([int(float(v)) for v in y_raw])
    meta = dict(task_type="classification", n_classes=2, domain="email",
                domain_description="Classify emails as spam or not spam based on word/character frequencies",
                source="OpenML ID 44 (Spambase)")
    return X, y, fnames, orig_names, None, meta


def load_diabetes_regression():
    diab = load_diabetes()
    X = diab.data.astype(float)
    y = diab.target.astype(float)
    fnames = [str(fn) for fn in diab.feature_names]
    orig_names = fnames[:]
    descs = {
        "age": "Age", "sex": "Sex", "bmi": "Body mass index",
        "bp": "Average blood pressure", "s1": "tc, total serum cholesterol",
        "s2": "ldl, low-density lipoproteins", "s3": "hdl, high-density lipoproteins",
        "s4": "tch, total cholesterol / HDL", "s5": "ltg, log of serum triglycerides",
        "s6": "glu, blood sugar level",
    }
    meta = dict(task_type="regression", n_classes=None, domain="medical",
                domain_description="Predict diabetes disease progression one year after baseline",
                source="sklearn load_diabetes()")
    return X, y, fnames, orig_names, descs, meta


def load_california_housing():
    cal = fetch_california_housing()
    X = cal.data.astype(float)
    y = cal.target.astype(float)
    fnames = [str(fn) for fn in cal.feature_names]
    orig_names = fnames[:]
    descs = {
        "MedInc": "Median income in block group (tens of thousands USD)",
        "HouseAge": "Median house age in block group (years)",
        "AveRooms": "Average number of rooms per household",
        "AveBedrms": "Average number of bedrooms per household",
        "Population": "Block group population",
        "AveOccup": "Average number of household members",
        "Latitude": "Block group latitude", "Longitude": "Block group longitude",
    }
    meta = dict(task_type="regression", n_classes=None, domain="housing",
                domain_description="Predict median house values in California census block groups",
                source="sklearn fetch_california_housing()")
    return X, y, fnames, orig_names, descs, meta


def load_abalone():
    data = fetch_openml(data_id=183, as_frame=True, parser="auto")
    df = data.frame
    y_raw = df["Class_number_of_rings"]
    feat_df = df.drop(columns=["Class_number_of_rings"])
    cat_cols = ["Sex"]
    orig_names = list(feat_df.columns)
    X, fnames = one_hot_encode(feat_df, cat_cols)
    y = np.array([float(v) for v in y_raw])
    meta = dict(task_type="regression", n_classes=None, domain="marine_biology",
                domain_description="Predict age of abalone (number of rings) from physical measurements",
                source="OpenML ID 183 (Abalone)")
    return X, y, fnames, orig_names, None, meta


def load_auto_mpg():
    data = fetch_openml(data_id=196, as_frame=True, parser="auto")
    df = data.frame
    y_raw = df["class"]
    feat_df = df.drop(columns=["class"])
    for col in ["car_name", "car name"]:
        if col in feat_df.columns:
            feat_df = feat_df.drop(columns=[col])
    mask = feat_df["horsepower"].notna()
    feat_df = feat_df[mask].reset_index(drop=True)
    y_raw = y_raw[mask].reset_index(drop=True)
    cat_cols = feat_df.select_dtypes(include=["category", "object"]).columns.tolist()
    orig_names = list(feat_df.columns)
    X, fnames = one_hot_encode(feat_df, cat_cols)
    y = np.array([float(v) for v in y_raw])
    descs = {
        "cylinders": "Number of cylinders", "displacement": "Engine displacement (cu in)",
        "horsepower": "Engine horsepower", "weight": "Vehicle weight (lbs)",
        "acceleration": "Time to accelerate 0-60 mph (sec)", "model": "Model year",
        "origin": "Origin (1=American, 2=European, 3=Japanese)",
    }
    meta = dict(task_type="regression", n_classes=None, domain="automotive",
                domain_description="Predict fuel efficiency (miles per gallon) from vehicle attributes",
                source="OpenML ID 196 (Auto MPG)")
    return X, y, fnames, orig_names, descs, meta


def load_wine_quality_red():
    data = fetch_openml(data_id=40691, as_frame=True, parser="auto")
    df = data.frame
    y_raw = df["class"]
    feat_df = df.drop(columns=["class"])
    orig_names = list(feat_df.columns)
    X = feat_df.values.astype(float)
    fnames = orig_names[:]
    y = np.array([float(v) for v in y_raw])
    descs = {
        "fixed_acidity": "Fixed acidity (tartaric acid, g/dm^3)",
        "volatile_acidity": "Volatile acidity (acetic acid, g/dm^3)",
        "citric_acid": "Citric acid (g/dm^3)", "residual_sugar": "Residual sugar (g/dm^3)",
        "chlorides": "Chlorides (sodium chloride, g/dm^3)",
        "free_sulfur_dioxide": "Free sulfur dioxide (mg/dm^3)",
        "total_sulfur_dioxide": "Total sulfur dioxide (mg/dm^3)",
        "density": "Density (g/cm^3)", "pH": "pH level",
        "sulphates": "Sulphates (potassium sulphate, g/dm^3)",
        "alcohol": "Alcohol content (% by volume)",
    }
    meta = dict(task_type="regression", n_classes=None, domain="chemistry",
                domain_description="Predict sensory quality score of red wine from physicochemical properties",
                source="OpenML ID 40691 (Wine Quality Red)")
    return X, y, fnames, orig_names, descs, meta


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
# Best 10 datasets selected for Codebook-FIGS evaluation:
#  - 8 domain-semantic datasets (heart, diabetes, breast_cancer, credit, diabetes_regr,
#    california, wine, auto_mpg) for codebook interpretability validation
#  - 6 classification + 4 regression covering features 8–61, samples 296–20,640
#  - Maximum overlap with FIGS/RO-FIGS/SPORF benchmarks
#
# Dropped: sonar (too small, anonymised features), vehicle (multiclass, no domain semantics),
#   segment (multiclass, no domain semantics), climate_crashes (imbalanced, no semantics),
#   abalone (weak semantics, redundant with other regression datasets)
ALL_LOADERS = [
    ("heart_disease",        load_heart_disease),
    ("diabetes_pima",        load_diabetes_pima),
    ("breast_cancer_wdbc",   load_breast_cancer_wdbc),
    ("credit_german",        load_credit_german),
    ("ionosphere",           load_ionosphere),
    ("spambase",             load_spambase),
    ("diabetes_regression",  load_diabetes_regression),
    ("california_housing",   load_california_housing),
    ("auto_mpg",             load_auto_mpg),
    ("wine_quality_red",     load_wine_quality_red),
]


def main():
    all_groups: list[dict] = []
    total_errors: list[str] = []

    for ds_id, loader_fn in ALL_LOADERS:
        print(f"Loading {ds_id} ... ", end="", flush=True)
        try:
            X, y, fnames, orig_names, descs, meta = loader_fn()
        except Exception as exc:
            print(f"LOAD ERROR: {exc}")
            traceback.print_exc()
            continue

        # --- full version ---
        group = build_dataset_group(
            ds_id, X, y, meta["task_type"], meta["n_classes"],
            meta["domain"], meta["domain_description"], meta["source"],
            fnames, orig_names, descs,
        )
        errs = validate_group(group)
        if errs:
            total_errors.extend(errs)
            print(f"VALIDATION ERRORS: {errs}")
        else:
            all_groups.append(group)
            print(f"OK  {X.shape[0]:>6d} rows × {X.shape[1]:>3d} feats", end="")

        # --- mini version ---
        idx_mini = create_mini_indices(X, y, meta["task_type"])
        X_mini, y_mini = X[idx_mini], y[idx_mini]
        group_mini = build_dataset_group(
            ds_id, X_mini, y_mini, meta["task_type"], meta["n_classes"],
            meta["domain"], meta["domain_description"], meta["source"],
            fnames, orig_names, descs,
            is_mini=True, parent_indices=idx_mini,
        )
        errs_mini = validate_group(group_mini)
        if errs_mini:
            total_errors.extend(errs_mini)
            print(f"  | mini ERRORS: {errs_mini}")
        else:
            all_groups.append(group_mini)
            print(f"  | mini {len(idx_mini):>4d} rows")

    # --- Final validation summary ---
    print("\n" + "=" * 70)
    if total_errors:
        print(f"TOTAL VALIDATION ERRORS: {len(total_errors)}")
        for e in total_errors:
            print(f"  ✗ {e}")
        sys.exit(1)
    else:
        print(f"ALL {len(all_groups)} dataset groups validated successfully.")

    # --- Write output ---
    output = {"datasets": all_groups}
    out_path = os.path.join(WORKSPACE, "full_data_out.json")
    with open(out_path, "w") as f:
        json.dump(output, f)

    fsize = os.path.getsize(out_path)
    print(f"\nWritten: {out_path}")
    print(f"File size: {fsize / (1024 * 1024):.2f} MB")
    print(f"Datasets: {len(all_groups)} ({len(all_groups) // 2} full + {len(all_groups) // 2} mini)")

    # Summary table
    print(f"\n{'Dataset':40s} | {'Rows':>6s} | {'Feats':>5s} | {'Task':15s} | {'Domain':20s}")
    print("-" * 100)
    for g in all_groups:
        did = g["dataset"]
        n = len(g["examples"])
        inp0 = json.loads(g["examples"][0]["input"])
        nf = len(inp0)
        tt = g["examples"][0].get("metadata_task_type", "?")
        dom = g["examples"][0].get("metadata_domain", "")
        print(f"{did:40s} | {n:6d} | {nf:5d} | {tt:15s} | {dom:20s}")


if __name__ == "__main__":
    main()
