from __future__ import annotations

import json
import secrets
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "raw" / "Raw_Breast_Cancer_Data.csv"
MODEL_CONFIG = {
    "Logistic Regression": {
        "model_path": ROOT / "models" / "Logistic_Regression" / "logistic_regression_pipeline.joblib",
        "metadata_path": ROOT / "reports" / "figures" / "Logistic_Regression" / "model_metadata.json",
        "accent": "#176B87",
    },
    "KNN": {
        "model_path": ROOT / "models" / "KNN" / "knn_pipeline.joblib",
        "metadata_path": ROOT / "reports" / "figures" / "KNN" / "model_metadata.json",
        "accent": "#B4552D",
    },
}
SEED = 42
LABELS = {0: "Benign", 1: "Malignant"}

st.set_page_config(
    page_title="Breast Cancer Model Comparison",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .block-container {max-width: 1180px; padding-top: 2.3rem; padding-bottom: 3rem;}
    .hero {padding: 1.4rem 1.6rem; border-radius: 18px; background: linear-gradient(120deg,#102A43,#174C5E); color: white; margin-bottom: 1rem;}
    .hero h1 {color: white; margin: 0 0 .35rem 0; font-size: 2.15rem;}
    .hero p {color: #D9EAF0; margin: 0; font-size: 1rem;}
    .notice {padding: .9rem 1rem; border-left: 4px solid #D18B24; background: #FFF8E8; border-radius: 8px; color: #473512; margin: .8rem 0 1.3rem 0;}
    .result-label {font-size: .82rem; color: #52616B; text-transform: uppercase; letter-spacing: .06em;}
    .result-class {font-size: 1.7rem; font-weight: 700; margin: .25rem 0 .6rem 0;}
    .small-muted {font-size: .9rem; color: #52616B;}
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")
    frame = pd.read_csv(DATA_PATH)
    blank_columns = frame.columns[frame.isna().all()].tolist()
    return frame.drop(columns=blank_columns).copy()


@st.cache_resource(show_spinner="Loading saved models…")
def load_models() -> tuple[dict[str, object], list[str], dict[str, dict]]:
    models: dict[str, object] = {}
    metadata: dict[str, dict] = {}
    shared_features: list[str] | None = None
    for name, config in MODEL_CONFIG.items():
        if not config["model_path"].exists():
            raise FileNotFoundError(f"Saved {name} pipeline is missing: {config['model_path']}")
        if not config["metadata_path"].exists():
            raise FileNotFoundError(f"Saved {name} metadata is missing: {config['metadata_path']}")
        details = json.loads(config["metadata_path"].read_text(encoding="utf-8"))
        estimator = joblib.load(config["model_path"])
        features = details.get("features")
        if not features or list(estimator.feature_names_in_) != features:
            raise ValueError(f"Feature metadata does not match the saved {name} pipeline.")
        if shared_features is not None and features != shared_features:
            raise ValueError("The saved models do not share the same ordered feature list.")
        shared_features = features
        models[name] = estimator
        metadata[name] = details
    return models, shared_features or [], metadata


def show_model_prediction(name: str, model: object, row: pd.DataFrame,
                          actual: int | None = None) -> dict:
    prediction = int(model.predict(row)[0])
    classes = list(model.classes_)
    malignant_probability = float(model.predict_proba(row)[0, classes.index(1)])
    class_color = "#A33D32" if prediction == 1 else "#176B55"
    with st.container(border=True):
        st.markdown(f"<div class='result-label'>{name}</div>", unsafe_allow_html=True)
        st.markdown(
            f"<div class='result-class' style='color:{class_color}'>{LABELS[prediction]}</div>",
            unsafe_allow_html=True,
        )
        st.progress(malignant_probability, text=f"Model score for malignant: {malignant_probability:.1%}")
        if actual is not None:
            correct = prediction == actual
            st.success("Correct for this labeled example") if correct else st.error("Incorrect for this labeled example")
    return {"model": name, "prediction": prediction, "malignant_probability": malignant_probability}


def compute_test_summary(models: dict[str, object], features: list[str], frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    labels = frame["diagnosis"].map({"B": 0, "M": 1})
    indices = np.arange(len(frame))
    _, test_indices = train_test_split(
        indices, test_size=0.20, random_state=SEED, stratify=labels
    )
    test_frame = frame.iloc[test_indices].copy()
    x_test = test_frame[features]
    y_test = labels.iloc[test_indices]
    rows = []
    for name, estimator in models.items():
        y_pred = estimator.predict(x_test)
        malignant_score = estimator.predict_proba(x_test)[:, list(estimator.classes_).index(1)]
        tn = int(((y_test == 0) & (y_pred == 0)).sum())
        fp = int(((y_test == 0) & (y_pred == 1)).sum())
        fn = int(((y_test == 1) & (y_pred == 0)).sum())
        rows.append({
            "Model": name,
            "Malignant recall": recall_score(y_test, y_pred, pos_label=1, zero_division=0),
            "Precision": precision_score(y_test, y_pred, pos_label=1, zero_division=0),
            "F1": f1_score(y_test, y_pred, pos_label=1, zero_division=0),
            "Accuracy": accuracy_score(y_test, y_pred),
            "ROC-AUC": roc_auc_score(y_test, malignant_score),
            "False negatives": fn,
            "False positives": fp,
            "Total errors": fn + fp,
        })
    return pd.DataFrame(rows).set_index("Model"), test_frame


st.markdown("""
<div class="hero">
  <h1>Breast Cancer Model Comparison</h1>
  <p>Explore predictions from Logistic Regression and K-Nearest Neighbors using the same 30 cell-measurement features.</p>
</div>
<div class="notice"><strong>Research demonstration only.</strong> These models are not medical devices and must not be used to diagnose or guide care. A model score is not a calibrated clinical risk.</div>
""", unsafe_allow_html=True)

try:
    data = load_data()
    models, feature_names, model_metadata = load_models()
except Exception as exc:
    st.error(f"Could not load the application data or models: {exc}")
    st.stop()

if "diagnosis" not in data.columns or "id" not in data.columns:
    st.error("The CSV does not have the expected diagnosis and id columns.")
    st.stop()
if data["diagnosis"].isna().any() or not set(data["diagnosis"].unique()).issubset({"B", "M"}):
    st.error("The CSV has missing or unexpected diagnosis labels.")
    st.stop()

feature_frame = data[feature_names]
y_all = data["diagnosis"].map({"B": 0, "M": 1})
all_indices = np.arange(len(data))
train_dev_indices, test_indices = train_test_split(
    all_indices, test_size=0.20, random_state=SEED, stratify=y_all
)
train_indices, _ = train_test_split(
    train_dev_indices, test_size=0.25, random_state=SEED,
    stratify=y_all.iloc[train_dev_indices]
)
training_medians = feature_frame.iloc[train_indices].median()

try:
    test_summary, test_frame = compute_test_summary(models, feature_names, data)
except Exception as exc:
    st.error(f"Could not calculate the saved-model comparison: {exc}")
    st.stop()

st.subheader("Compare the saved models")
st.caption("The table uses the same 114-record stratified test split reported in the model evaluation notebooks. Models are loaded from disk; this app does not retrain them.")
st.dataframe(
    test_summary.style.format({c: "{:.1%}" for c in ["Malignant recall", "Precision", "F1", "Accuracy", "ROC-AUC"]}),
    width="stretch",
)
st.bar_chart(test_summary[["False negatives", "False positives"]], height=190)

st.divider()
st.subheader("Try an example")
mode = st.radio(
    "Choose how to provide measurements",
    ["Enter measurements", "Pick a random labeled test example"],
    horizontal=True,
)
st.caption("There are 30 numeric inputs. The record ID and diagnosis are not model features.")

if mode == "Enter measurements":
    st.info("For manual input, the diagnosis is unknown. The app can compare predictions and model scores, but cannot calculate which model is wrong without a known diagnosis.")
    feature_groups = {
        "Mean measurements": [name for name in feature_names if name.endswith("_mean")],
        "Standard error measurements": [name for name in feature_names if name.endswith("_se")],
        "Worst measurements": [name for name in feature_names if name.endswith("_worst")],
    }
    submitted_values: dict[str, float] = {}
    with st.form("manual_measurements"):
        columns = st.columns(3)
        for (group_name, group_features), column in zip(feature_groups.items(), columns):
            with column:
                st.markdown(f"**{group_name}**")
                for feature in group_features:
                    step = max(float(feature_frame[feature].std()) / 100, 0.0001)
                    display_format = "%.5f" if step < 0.1 else "%.2f"
                    submitted_values[feature] = st.number_input(
                        feature.replace("_", " ").title(),
                        min_value=0.0,
                        value=max(0.0, float(training_medians[feature])),
                        step=step,
                        format=display_format,
                        key=f"input_{feature}",
                    )
        submitted = st.form_submit_button("Compare both models", type="primary", width="stretch")
    if submitted:
        manual_row = pd.DataFrame([[submitted_values[name] for name in feature_names]], columns=feature_names)
        st.subheader("Prediction comparison")
        left, right = st.columns(2)
        outputs = []
        for column, (name, model) in zip([left, right], models.items()):
            with column:
                outputs.append(show_model_prediction(name, model, manual_row))
        same = outputs[0]["prediction"] == outputs[1]["prediction"]
        st.success("Both models agree on this input.") if same else st.warning("The models disagree on this input. No known diagnosis was supplied, so correctness cannot be determined.")

else:
    st.info("Each random example comes from the reserved test split and has a known diagnosis. These records were already included in the comparison table above.")
    if st.button("Draw a random labeled test example", type="primary"):
        st.session_state["random_test_index"] = secrets.choice(test_frame.index.tolist())
    if "random_test_index" in st.session_state:
        sample_index = st.session_state["random_test_index"]
        sample = test_frame.loc[[sample_index]]
        actual_label = 1 if sample["diagnosis"].iloc[0] == "M" else 0
        st.markdown(f"**Reference record:** `{sample['id'].iloc[0]}` &nbsp; · &nbsp; **Known diagnosis:** {LABELS[actual_label]}")
        with st.expander("Show the 30 measurements"):
            st.dataframe(sample[feature_names].T.rename(columns={sample.index[0]: "Value"}), width="stretch")
        st.markdown("**Per-model result**")
        prediction_columns = st.columns(2)
        for column, (name, model) in zip(prediction_columns, models.items()):
            with column:
                show_model_prediction(name, model, sample[feature_names], actual=actual_label)
    else:
        st.caption("Select the button to draw a test example and compare both predictions with its known diagnosis.")

st.divider()
st.caption("Model artifacts are loaded from the local project image. This demo does not retrain, update, or persist user-entered data.")

