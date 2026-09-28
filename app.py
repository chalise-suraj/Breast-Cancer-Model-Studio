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
    },
    "KNN": {
        "model_path": ROOT / "models" / "KNN" / "knn_pipeline.joblib",
        "metadata_path": ROOT / "reports" / "figures" / "KNN" / "model_metadata.json",
    },
}
SEED = 42
CLASS_LABELS = {
    0: ("Benign", "Non-cancerous pattern predicted · B"),
    1: ("Malignant", "Cancerous pattern predicted · M"),
}

st.set_page_config(
    page_title="Model Lab | Breast Cancer Prediction",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed",
)
st.markdown("""
<style>
    :root { --ink:#17324D; --muted:#5C6B78; --teal:#147D78; --orange:#C76B37; --paper:#F4F7FA; }
    .stApp { background:var(--paper); }
    .block-container { max-width:1220px; padding-top:1.6rem; padding-bottom:3rem; }
    .hero { background:linear-gradient(120deg,#102A43 0%,#155E68 100%); padding:1.65rem 1.8rem; border-radius:20px; color:white; margin-bottom:1.1rem; box-shadow:0 10px 26px rgba(16,42,67,.13); }
    .hero h1 { color:white; font-size:2.1rem; margin:0 0 .35rem 0; letter-spacing:-.02em; }
    .hero p { color:#DCEAF0; margin:0; font-size:1rem; }
    .safety { background:#FFF8E9; border:1px solid #F1D9A7; border-left:5px solid #C78A26; color:#59441D; padding:.8rem 1rem; border-radius:10px; margin:.9rem 0 1.3rem 0; }
    .result-card { border-radius:16px; }
    div[data-testid="stMetric"] { background:white; border:1px solid #E2E8EE; padding:1rem 1.1rem; border-radius:14px; }
    .muted { color:var(--muted); }
    @media (max-width: 760px) { .block-container {padding-left:1rem;padding-right:1rem;} .hero h1 {font-size:1.65rem;} }
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")
    frame = pd.read_csv(DATA_PATH)
    blank_columns = frame.columns[frame.isna().all()].tolist()
    return frame.drop(columns=blank_columns).copy()


@st.cache_resource(show_spinner="Loading the saved model pipelines…")
def load_models() -> tuple[dict[str, object], list[str]]:
    models: dict[str, object] = {}
    shared_features: list[str] | None = None
    for name, config in MODEL_CONFIG.items():
        if not config["model_path"].exists() or not config["metadata_path"].exists():
            raise FileNotFoundError(f"The saved {name} model or its metadata is missing.")
        metadata = json.loads(config["metadata_path"].read_text(encoding="utf-8"))
        estimator = joblib.load(config["model_path"])
        features = metadata.get("features")
        if not features or list(estimator.feature_names_in_) != features:
            raise ValueError(f"Feature metadata does not match the saved {name} pipeline.")
        if shared_features is not None and features != shared_features:
            raise ValueError("The two models do not use the same ordered features.")
        shared_features = features
        models[name] = estimator
    return models, shared_features or []


def test_set_comparison(models: dict[str, object], features: list[str], frame: pd.DataFrame) -> pd.DataFrame:
    """Calculate aggregate model errors on the same untouched test split used in Notebook 4."""
    actual = frame["diagnosis"].map({"B": 0, "M": 1})
    _, test_indices = train_test_split(
        np.arange(len(frame)), test_size=0.20, random_state=SEED, stratify=actual
    )
    test_frame = frame.iloc[test_indices]
    x_test = test_frame[features]
    y_test = actual.iloc[test_indices].to_numpy()
    rows = []
    for name, estimator in models.items():
        prediction = estimator.predict(x_test)
        malignant_score = estimator.predict_proba(x_test)[:, list(estimator.classes_).index(1)]
        tn = int(((y_test == 0) & (prediction == 0)).sum())
        fp = int(((y_test == 0) & (prediction == 1)).sum())
        fn = int(((y_test == 1) & (prediction == 0)).sum())
        rows.append({
            "Model": name,
            "Wrong predictions": fn + fp,
            "Error rate": (fn + fp) / len(y_test),
            "Missed cancerous cases (FN)": fn,
            "False cancer alarms (FP)": fp,
            "Malignant recall": recall_score(y_test, prediction, pos_label=1, zero_division=0),
            "Precision": precision_score(y_test, prediction, pos_label=1, zero_division=0),
            "F1 score": f1_score(y_test, prediction, pos_label=1, zero_division=0),
            "Accuracy": accuracy_score(y_test, prediction),
            "ROC-AUC": roc_auc_score(y_test, malignant_score),
        })
    return pd.DataFrame(rows).set_index("Model")


def predict_both(models: dict[str, object], feature_names: list[str], values: dict[str, float]) -> list[dict]:
    row = pd.DataFrame([[values[name] for name in feature_names]], columns=feature_names)
    predictions = []
    for name, estimator in models.items():
        label = int(estimator.predict(row)[0])
        classes = list(estimator.classes_)
        score = float(estimator.predict_proba(row)[0, classes.index(1)])
        predictions.append({"name": name, "label": label, "score": score})
    return predictions


def display_predictions(predictions: list[dict]) -> None:
    columns = st.columns(2)
    for column, result in zip(columns, predictions):
        label, explanation = CLASS_LABELS[result["label"]]
        color = "#A33D32" if result["label"] == 1 else "#147D78"
        with column:
            with st.container(border=True):
                st.markdown(f"**{result['name']}**")
                st.markdown(
                    f"<div style='font-size:1.45rem;font-weight:750;color:{color};margin:.45rem 0'>{label}</div>",
                    unsafe_allow_html=True,
                )
                st.caption(explanation)
                st.progress(result["score"], text=f"Model score for malignant class: {result['score']:.1%}")
                st.caption("This score is not a calibrated medical risk.")
    if predictions[0]["label"] == predictions[1]["label"]:
        st.info("The models predict the same class. Agreement does not establish that the prediction is correct.")
    else:
        st.warning("The models predict different classes. A known diagnosis is needed to tell which prediction is wrong.")


st.markdown("""
<div class="hero">
  <h1>Breast Cancer Model Lab</h1>
  <p>Enter cell measurements or generate a synthetic example, then compute predictions from both saved models.</p>
</div>
<div class="safety"><strong>Research demonstration only.</strong> These models are not medical devices and must not be used to diagnose or guide care. A “benign” prediction does not rule out cancer.</div>
""", unsafe_allow_html=True)

try:
    data = load_data()
    models, feature_names = load_models()
except Exception as exc:
    st.error(f"Could not load the dataset or saved models: {exc}")
    st.stop()

if not {"diagnosis", "id"}.issubset(data.columns):
    st.error("The dataset is missing the expected diagnosis or record ID column.")
    st.stop()
if data["diagnosis"].isna().any() or not set(data["diagnosis"].unique()).issubset({"B", "M"}):
    st.error("The dataset contains missing or unexpected diagnosis labels.")
    st.stop()
if feature_names != [c for c in data.columns if c not in ["diagnosis", "id"]]:
    st.error("The dataset feature columns do not match the saved model inputs.")
    st.stop()

feature_frame = data[feature_names]
labels = data["diagnosis"].map({"B": 0, "M": 1})
indices = np.arange(len(data))
train_dev_indices, _ = train_test_split(
    indices, test_size=0.20, random_state=SEED, stratify=labels
)
train_indices, _ = train_test_split(
    train_dev_indices, test_size=0.25, random_state=SEED,
    stratify=labels.iloc[train_dev_indices]
)
training_features = feature_frame.iloc[train_indices]
training_medians = training_features.median()
training_stds = training_features.std().fillna(0)

predict_tab, errors_tab = st.tabs(["Make a prediction", "Compare model errors"])

with predict_tab:
    st.subheader("Choose your input method")
    input_mode = st.radio(
        "How would you like to provide the measurements?",
        ["Enter measurements", "Generate random features"],
        horizontal=True,
        label_visibility="collapsed",
        key="input_mode",
    )
    st.caption("The app uses 30 numeric measurements. Record ID and diagnosis are not model inputs.")

    if input_mode == "Enter measurements":
        st.markdown("#### Cell measurements")
        st.caption("Values start at the training-set medians. Adjust the measurements, then compute both model outputs.")
        groups = [
            ("Mean values", [name for name in feature_names if name.endswith("_mean")]),
            ("Standard error", [name for name in feature_names if name.endswith("_se")]),
            ("Worst values", [name for name in feature_names if name.endswith("_worst")]),
        ]
        input_values: dict[str, float] = {}
        with st.form("manual_measurements"):
            group_tabs = st.tabs([group[0] for group in groups])
            for group_tab, (_, group_features) in zip(group_tabs, groups):
                with group_tab:
                    left, right = st.columns(2)
                    for position, feature in enumerate(group_features):
                        step = max(float(training_stds[feature]) / 100, 0.0001)
                        number_format = "%.5f" if step < 0.1 else "%.2f"
                        target_column = left if position % 2 == 0 else right
                        with target_column:
                            input_values[feature] = st.number_input(
                                feature.replace("_", " ").title(),
                                min_value=0.0,
                                value=max(0.0, float(training_medians[feature])),
                                step=step,
                                format=number_format,
                                key=f"manual_{feature}",
                            )
            manual_submitted = st.form_submit_button(
                "Compute Output", type="primary", width="stretch"
            )
        if manual_submitted:
            st.session_state["manual_predictions"] = predict_both(models, feature_names, input_values)
        if st.session_state.get("manual_predictions"):
            st.markdown("#### Model outputs")
            display_predictions(st.session_state["manual_predictions"])

    else:
        st.markdown("#### Create a synthetic feature set")
        st.markdown(
            "Random values are generated around a randomly selected training-feature profile, with small noise and nonnegative bounds. "
            "This keeps values near the observed feature ranges while avoiding reuse of a recorded test row. "
            "A synthetic example has no known diagnosis, so its prediction error cannot be measured."
        )
        if st.button("Generate random values", type="secondary", width="stretch"):
            generator = np.random.default_rng(secrets.randbits(32))
            profile_index = int(generator.integers(0, len(training_features)))
            profile = training_features.iloc[profile_index].to_numpy(dtype=float)
            noise = generator.normal(0.0, training_stds.to_numpy(dtype=float) * 0.025)
            generated = np.maximum(profile + noise, 0.0)
            st.session_state["synthetic_values"] = {
                feature: float(value) for feature, value in zip(feature_names, generated)
            }
            st.session_state.pop("synthetic_predictions", None)
        if st.session_state.get("synthetic_values"):
            generated_values = st.session_state["synthetic_values"]
            with st.expander("Review generated feature values", expanded=False):
                st.dataframe(
                    pd.DataFrame({"Measurement": feature_names,
                                  "Generated value": [generated_values[name] for name in feature_names]}),
                    hide_index=True,
                    width="stretch",
                )
            if st.button("Compute Output", type="primary", width="stretch", key="compute_synthetic"):
                st.session_state["synthetic_predictions"] = predict_both(models, feature_names, generated_values)
            if st.session_state.get("synthetic_predictions"):
                st.markdown("#### Model outputs")
                display_predictions(st.session_state["synthetic_predictions"])
        else:
            st.info("Generate a feature set first. You can review its values before computing the model outputs.")

with errors_tab:
    st.subheader("Which model made fewer wrong predictions?")
    st.markdown(
        "This comparison uses the fixed 114-record test set with known diagnoses—not your manual or synthetic input. "
        "Each model's errors are counted against the known diagnosis."
    )
    try:
        errors = test_set_comparison(models, feature_names, data)
    except Exception as exc:
        st.error(f"Could not calculate the held-out error comparison: {exc}")
    else:
        left, right = st.columns(2)
        for column, model_name in zip([left, right], errors.index):
            row = errors.loc[model_name]
            with column:
                with st.container(border=True):
                    st.markdown(f"### {model_name}")
                    st.metric("Wrong predictions", f"{int(row['Wrong predictions'])} / 114",
                              delta=f"{row['Error rate']:.1%} error rate", delta_color="inverse")
                    st.write(f"Missed malignant cases: **{int(row['Missed cancerous cases (FN)'])}**")
                    st.write(f"False malignant alarms: **{int(row['False cancer alarms (FP)'])}**")
                    st.caption(f"Malignant recall on this test set: {row['Malignant recall']:.1%}")
        st.markdown("#### Side-by-side metrics")
        formatted = errors.copy()
        for column in ["Error rate", "Malignant recall", "Precision", "F1 score", "Accuracy", "ROC-AUC"]:
            formatted[column] = formatted[column].map(lambda value: f"{value:.1%}")
        st.dataframe(formatted, width="stretch")
        st.caption(
            "The test set has only 42 malignant cases. These measured errors describe this split; they are not a guarantee of future performance. "
            "Manual and generated inputs have no ground-truth labels, so no per-input error can be calculated for them."
        )

st.divider()
st.caption("Neither model retrains in the app. Inputs are processed in memory and are not saved.")
