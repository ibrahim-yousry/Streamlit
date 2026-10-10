"""Run with the light-theme command provided with this project."""

from pathlib import Path
import re
import warnings

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.base import is_classifier
from sklearn.datasets import load_iris
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted


MODEL_PATH = Path(__file__).resolve().parent / "iris_model.pkl"
FEATURES = [
    "sepal length (cm)", "sepal width (cm)",
    "petal length (cm)", "petal width (cm)",
]
LABELS = ["Sepal Length (cm)", "Sepal Width (cm)",
          "Petal Length (cm)", "Petal Width (cm)"]
SPECIES = ["Iris Setosa", "Iris Versicolor", "Iris Virginica"]
ACCENT = "#527A78"
COLORS = [ACCENT, "#8B96A5", "#B7A68E"]


@st.cache_data
def get_dataset():
    iris = load_iris()
    frame = pd.DataFrame(iris.data, columns=FEATURES)
    frame["Species"] = [SPECIES[int(value)] for value in iris.target]
    return frame


@st.cache_resource
def load_model(path, modified_ns, file_size):
    # File metadata invalidates the cache when the artifact is replaced.
    with warnings.catch_warnings():
        warnings.simplefilter("error", InconsistentVersionWarning)
        model = joblib.load(path)  # Also reads ordinary pickle-saved estimators.
    if not is_classifier(model) or not callable(getattr(model, "predict", None)):
        raise ValueError(
            "The file must contain a fitted classifier or classification Pipeline. "
            "A dictionary, regressor, or separate transformer is not supported."
        )
    check_is_fitted(model)
    return model


def get_local_model(show_error=True):
    try:
        metadata = MODEL_PATH.stat()
        return load_model(str(MODEL_PATH), metadata.st_mtime_ns, metadata.st_size)
    except FileNotFoundError:
        message = "Place your existing iris_model.pkl in the same folder as app.py."
    except InconsistentVersionWarning as exc:
        message = (
            f"This artifact was saved with scikit-learn {exc.original_sklearn_version}. "
            "Install that version and the original training dependencies, then restart."
        )
    except (ModuleNotFoundError, ImportError) as exc:
        message = (
            f"A training dependency or custom component is missing: {exc}. "
            "Install the original dependency or make the custom component importable."
        )
    except Exception as exc:
        message = f"Could not load the model: {exc}"
    if show_error:
        st.error(message)
    return None


def feature_key(name):
    # Recognize common spellings while retaining the original model column names.
    key = re.sub(r"[^a-z]", "", str(name).lower())
    return key.removesuffix("cm")


def inspect_schema(model):
    count = getattr(model, "n_features_in_", None)
    if count is None or int(count) != 4:
        raise ValueError(
            f"Expected four input features; the artifact reports {count!r}. "
            "Use the fitted raw-input Pipeline and its documented input schema."
        )
    names = getattr(model, "feature_names_in_", None)
    if names is None:
        return None, list(FEATURES)
    names = list(names)
    lookup = {feature_key(feature): feature for feature in FEATURES}
    keys = [feature_key(name) for name in names]
    if len(keys) != 4 or set(keys) != set(lookup):
        raise ValueError(
            f"The stored input columns are {names}. They cannot be mapped uniquely "
            "to the four Iris measurements. Provide the training column mapping "
            "or save the complete raw-input Pipeline; prediction is unavailable."
        )
    return names, [lookup[key] for key in keys]


def inspect_classes(model):
    classes = np.asarray(getattr(model, "classes_", []))
    if classes.ndim != 1 or len(classes) != 3:
        raise ValueError("The artifact must expose three single-output target classes.")
    if classes.dtype.kind in "iuf":
        if set(classes.tolist()) != {0, 1, 2}:
            raise ValueError(
                "Numeric labels are not the standard Iris labels 0, 1, 2. "
                "The original target-label mapping is required."
            )
        return classes, [SPECIES[int(label)] for label in classes], True
    lookup = {re.sub(r"[^a-z]", "", name.lower()): name for name in SPECIES}
    lookup.update({name.split()[-1].lower(): name for name in SPECIES})
    names = [lookup.get(re.sub(r"[^a-z]", "", str(label).lower())) for label in classes]
    if None in names or set(names) != set(SPECIES):
        raise ValueError(
            f"Unknown target labels: {classes.tolist()}. "
            "The original label encoder or target-label mapping is required."
        )
    return classes, names, False


def make_sample(values, names, order):
    row = [[values[feature] for feature in order]]
    if names is not None:
        return pd.DataFrame(row, columns=names)
    return np.asarray(row, dtype=float)


def style_chart(figure, height=320, legend=False):
    figure.update_layout(
        template="plotly_white", height=height,
        font=dict(family="Arial, sans-serif", size=12, color="#25313C"),
        paper_bgcolor="white", plot_bgcolor="white",
        margin=dict(l=20, r=20, t=30, b=30),
        showlegend=legend,
        legend=dict(title_text="", orientation="h", y=1.12, x=0),
        bargap=0.3,
    )
    figure.update_xaxes(gridcolor="#EDF0F2", zeroline=False)
    figure.update_yaxes(gridcolor="#EDF0F2", zeroline=False)
    return figure


def show_chart(figure, key):
    st.plotly_chart(
        figure, width="stretch", theme=None, key=key,
        config={"displayModeBar": False, "scrollZoom": False},
    )


def render_overview():
    st.title("Iris Flower Classification")
    st.write(
        "Iris species can be distinguished using measurements of their sepals "
        "and petals. This project uses a trained classification model to predict "
        "a flower's species from those measurements."
    )
    st.subheader("Project objective")
    st.write("Predict the Iris species using four flower measurements in centimeters.")
    with st.container(border=True):
        st.dataframe(pd.DataFrame({
            "Project information": ["Dataset", "Problem type", "Input features", "Target classes"],
            "Value": ["Standard Iris dataset", "Multiclass classification", "4", "3"],
        }), hide_index=True, width="stretch")
    left, right = st.columns(2, gap="large")
    with left:
        st.subheader("Input features")
        st.write("\n".join(f"- {label}" for label in LABELS))
    with right:
        st.subheader("Target classes")
        st.write("\n".join(f"- {name}" for name in SPECIES))
    st.subheader("Classification workflow")
    st.write(
        "1. Explore the reference dataset and the relationships between measurements.\n"
        "2. Load the existing trained model and check its input schema.\n"
        "3. Submit measurements to predict the species and, when supported, view class probabilities."
    )
    st.caption(
        "Dataset exploration uses scikit-learn's standard Iris data. "
        "It is a reference dataset, not a verified copy of the model's training split."
    )
    model = get_local_model(show_error=False)
    if model is not None:
        with st.expander("Saved model details"):
            st.write(f"**Artifact type:** {type(model).__name__}")
            if isinstance(model, Pipeline):
                st.dataframe(pd.DataFrame([
                    {"Step": name, "Component": type(component).__name__}
                    for name, component in model.steps
                ]), hide_index=True, width="stretch")
            st.write(f"**Input feature count:** {getattr(model, 'n_features_in_', 'Not stored')}")
            st.caption("No evaluation results are stored or calculated by this app.")


def render_dataset_summary():
    st.title("Dataset Summary")
    st.caption("Reference data: the standard Iris dataset from sklearn.datasets.load_iris().")
    frame = get_dataset()
    for column, label, value in zip(st.columns(3),
                                    ["Observations", "Features", "Classes"],
                                    [len(frame), len(FEATURES), frame["Species"].nunique()]):
        column.metric(label, value)
    st.subheader("Feature descriptions")
    st.dataframe(pd.DataFrame({
        "Feature": LABELS,
        "Description": ["Length of the outer flower segment (sepal).",
                        "Width of the outer flower segment (sepal).",
                        "Length of the inner flower segment (petal).",
                        "Width of the inner flower segment (petal)."],
        "Unit": ["cm"] * 4,
    }), hide_index=True, width="stretch")
    st.subheader("Class distribution")
    counts = frame["Species"].value_counts().reindex(SPECIES).rename_axis("Species").reset_index(name="Observations")
    figure = px.bar(counts, x="Species", y="Observations", text="Observations",
                    color_discrete_sequence=[ACCENT])
    figure.update_traces(textposition="outside", cliponaxis=False)
    figure.update_yaxes(range=[0, counts["Observations"].max() * 1.2], dtick=10)
    show_chart(style_chart(figure, height=280), "class_distribution")
    st.subheader("Feature distributions")
    st.caption("Each histogram shows the number of observations in a measurement interval.")
    for start in (0, 2):
        for column, feature, label in zip(st.columns(2), FEATURES[start:start + 2], LABELS[start:start + 2]):
            with column:
                st.write(f"**{label}**")
                figure = px.histogram(frame, x=feature, nbins=16,
                                      labels={feature: label}, color_discrete_sequence=[ACCENT])
                figure.update_yaxes(title="Observations")
                show_chart(style_chart(figure, height=260), f"histogram_{start}_{feature}")
    left, right = st.columns(2, gap="large")
    with left:
        st.subheader("Feature correlations")
        correlation = frame[FEATURES].corr()
        short_labels = [label.replace(" (cm)", "") for label in LABELS]
        figure = px.imshow(
            correlation.to_numpy(), x=short_labels, y=short_labels,
            zmin=-1, zmax=1, text_auto=".2f", aspect="auto",
            color_continuous_scale=["#B8C1CD", "#F5F6F7", ACCENT],
            labels={"color": "Pearson r"},
        )
        figure.update_xaxes(side="bottom", tickangle=-25)
        show_chart(style_chart(figure, height=360), "correlation")
    with right:
        st.subheader("Petal measurements by species")
        figure = px.scatter(
            frame, x=FEATURES[2], y=FEATURES[3], color="Species",
            color_discrete_map=dict(zip(SPECIES, COLORS)),
            labels={FEATURES[2]: LABELS[2], FEATURES[3]: LABELS[3]},
            category_orders={"Species": SPECIES},
        )
        figure.update_traces(marker=dict(size=7, opacity=0.8))
        show_chart(style_chart(figure, height=360, legend=True), "petal_scatter")
    st.subheader("Data preview")
    st.dataframe(frame.head(10), hide_index=True, width="stretch")
    st.subheader("Statistical summary")
    st.dataframe(frame[FEATURES].describe().round(2), width="stretch")


def render_predict():
    st.title("Predict New Samples")
    st.write("Enter four flower measurements in centimeters to predict the Iris species.")
    model = get_local_model()
    if model is None:
        return
    try:
        names, order = inspect_schema(model)
        classes, class_names, numeric_labels = inspect_classes(model)
    except Exception as exc:
        st.error(f"The saved artifact is incompatible: {exc}")
        return
    with st.expander("Input schema"):
        st.write("**Input format:** " + ("DataFrame with stored column names" if names else "Array; training order requires confirmation"))
        st.dataframe(pd.DataFrame({
            "Position": range(1, 5), "Model column": names if names else order,
            "Measurement": order,
        }), hide_index=True, width="stretch")
    if names is None:
        st.info(
            "Feature names were not saved. Confirm the training order is: "
            "sepal length, sepal width, petal length, petal width. "
            "If it differs, the training schema is needed before prediction."
        )
    if not isinstance(model, Pipeline):
        st.info(
            "This artifact is a standalone estimator. If training used a separate "
            "scaler or other preprocessing, save and supply the fitted Pipeline. "
            "The app cannot recover external preprocessing from this file."
        )
    frame = get_dataset()
    with st.form("prediction_form"):
        values = {}
        columns = st.columns(2, gap="large")
        for index, (feature, label) in enumerate(zip(FEATURES, LABELS)):
            observed_min = float(frame[feature].min())
            observed_max = float(frame[feature].max())
            with columns[index % 2]:
                values[feature] = st.number_input(
                    label, min_value=0.01,
                    max_value=round(observed_max * 1.5, 2),
                    value=round(float(frame[feature].median()), 2),
                    step=0.1, format="%.2f", key=feature,
                    help=f"Reference dataset range: {observed_min:g}–{observed_max:g} cm.",
                )
        order_confirmed = True
        if names is None:
            order_confirmed = st.checkbox("My model uses the four measurements in the order shown above.")
        preprocessing_confirmed = st.checkbox(
            "My saved artifact accepts raw measurements in cm; any required preprocessing is included."
        )
        labels_confirmed = True
        if numeric_labels:
            labels_confirmed = st.checkbox(
                "My target encoding is 0 = Setosa, 1 = Versicolor, 2 = Virginica."
            )
        submitted = st.form_submit_button("Predict Species", type="primary")
    if not submitted:
        return
    if not (order_confirmed and preprocessing_confirmed and labels_confirmed):
        st.warning("Confirm the training schema, raw-input compatibility, and target mapping before predicting.")
        return
    outside = [LABELS[i] for i, feature in enumerate(FEATURES)
               if not float(frame[feature].min()) <= values[feature] <= float(frame[feature].max())]
    if outside:
        st.warning("Outside the reference dataset range: " + ", ".join(outside) + ". The result may be less reliable.")
    sample = make_sample(values, names, order)
    try:
        prediction = np.asarray(model.predict(sample))
        if prediction.shape != (1,):
            raise ValueError("Expected a single classification label.")
        matching = np.flatnonzero(classes == prediction[0])
        if len(matching) != 1:
            raise ValueError("The prediction does not match the stored target classes.")
        predicted_name = class_names[int(matching[0])]
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        st.caption("Check the original training schema and supply any missing fitted preprocessing components.")
        return
    st.divider()
    st.subheader("Prediction")
    st.success(f"Predicted species: {predicted_name}")
    st.dataframe(pd.DataFrame({"Measurement": LABELS,
                               "Submitted value (cm)": [values[feature] for feature in FEATURES]}),
                 hide_index=True, width="stretch")
    if callable(getattr(model, "predict_proba", None)):
        try:
            probabilities = np.asarray(model.predict_proba(sample), dtype=float)
            if (probabilities.shape != (1, len(classes))
                    or not np.isfinite(probabilities).all()
                    or (probabilities < 0).any() or (probabilities > 1).any()
                    or not np.isclose(probabilities[0].sum(), 1.0)):
                raise ValueError("The artifact returned invalid class probabilities.")
            probability_frame = pd.DataFrame({"Species": class_names, "Probability": probabilities[0]})
            figure = px.bar(probability_frame, x="Probability", y="Species", orientation="h",
                            text="Probability", color_discrete_sequence=[ACCENT])
            figure.update_traces(texttemplate="%{x:.1%}", textposition="outside", cliponaxis=False)
            figure.update_xaxes(range=[0, 1.12], tickformat=".0%", title="Class probability")
            figure.update_yaxes(title=None, autorange="reversed")
            st.subheader("Class probabilities")
            show_chart(style_chart(figure, height=240), "prediction_probabilities")
            st.caption("Probabilities are model estimates; they do not measure overall model accuracy.")
        except Exception as exc:
            st.warning(f"Species prediction succeeded, but probabilities could not be displayed: {exc}")
    else:
        st.caption("This saved model does not expose class probabilities.")


def main():
    st.set_page_config(page_title="Iris Flower Classification", layout="wide", initial_sidebar_state="expanded")
    st.markdown("""
        <style>
        .block-container { max-width: 1120px; padding-top: 2rem; padding-bottom: 3rem; }
        h1 { font-size: 2rem !important; }
        h2 { font-size: 1.35rem !important; }
        h3 { font-size: 1.15rem !important; }
        </style>
        """, unsafe_allow_html=True)
    with st.sidebar:
        st.caption("IRIS CLASSIFICATION")
        st.write("A measurement-based classification project.")
    page = st.navigation([
        st.Page(render_overview, title="Overview", default=True),
        st.Page(render_dataset_summary, title="Dataset Summary", url_path="dataset"),
        st.Page(render_predict, title="Predict New Samples", url_path="predict"),
    ], position="sidebar")
    page.run()


if __name__ == "__main__":
    main()
