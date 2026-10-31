"""Streamlit app: recommend a crop from soil N, P, K and pH."""
import json
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

from src.features import add_features
from src.train import DATA, METRICS_PATH, MODEL_PATH, main as train_model

st.set_page_config(page_title="Crop Recommender", page_icon="🌾", layout="wide")


@st.cache_resource(show_spinner="Loading model...")
def load_model():
    try:
        return joblib.load(MODEL_PATH)
    except Exception:  # missing file, or saved with a different scikit-learn version
        with st.spinner("Saved model unavailable - retraining (takes a few minutes)..."):
            train_model()
        return joblib.load(MODEL_PATH)


@st.cache_data
def load_data():
    return pd.read_csv(DATA)


@st.cache_data
def load_metrics():
    return json.loads(Path(METRICS_PATH).read_text()) if Path(METRICS_PATH).exists() else None


model, data, metrics = load_model(), load_data(), load_metrics()

st.title("🌾 Crop Recommendation System")
st.caption("Enter your soil's nutrient levels and pH to get the most suitable crops.")

tab_predict, tab_data, tab_model = st.tabs(["🔮 Predict", "📊 Data explorer", "🧠 Model performance"])

# ---------------------------------------------------------------- Predict
with tab_predict:
    left, right = st.columns([1, 2], gap="large")
    with left:
        st.subheader("Soil measurements")
        n = st.slider("Nitrogen (N)", 0, 140, 90)
        p = st.slider("Phosphorus (P)", 5, 145, 40)
        k = st.slider("Potassium (K)", 5, 205, 40)
        ph = st.slider("Soil pH", 3.5, 10.0, 6.5, step=0.1)
        run = st.button("Recommend crop", type="primary", width="stretch")

    with right:
        st.subheader("Recommendation")
        if run or "last" in st.session_state:
            if run:
                st.session_state["last"] = (n, p, k, ph)
            n_, p_, k_, ph_ = st.session_state["last"]
            X = add_features(pd.DataFrame({"N": [n_], "P": [p_], "K": [k_], "ph": [ph_]}))

            if hasattr(model, "predict_proba"):
                proba = pd.Series(model.predict_proba(X)[0], index=model.classes_).sort_values(ascending=False)
                top = proba.head(3)
                st.success(f"Best match: **{top.index[0].title()}** ({top.iloc[0]:.0%} confidence)")
                cols = st.columns(3)
                for col, (crop, pr) in zip(cols, top.items()):
                    col.metric(crop.title(), f"{pr:.0%}")
                fig = px.bar(top[::-1], orientation="h", labels={"value": "Probability", "index": "Crop"})
                fig.update_layout(showlegend=False, height=260, margin=dict(l=0, r=0, t=10, b=0),
                                  xaxis_tickformat=".0%")
                st.plotly_chart(fig, width="stretch")
                if top.iloc[0] < 0.5:
                    st.info("Confidence is low - these soil values sit between several crops' typical ranges.")
            else:
                st.success(f"Best match: **{model.predict(X)[0].title()}**")
        else:
            st.info("Set your soil values and press **Recommend crop**.")

# ---------------------------------------------------------------- Data explorer
with tab_data:
    st.subheader("Dataset overview")
    c1, c2, c3 = st.columns(3)
    c1.metric("Samples", f"{len(data):,}")
    c2.metric("Crops", data["crop"].nunique())
    c3.metric("Features", "N, P, K, pH")

    crop = st.selectbox("Inspect a crop", ["All"] + sorted(data["crop"].unique()))
    view = data if crop == "All" else data[data["crop"] == crop]
    st.dataframe(view.describe().T.round(2), width="stretch")

    feat = st.selectbox("Distribution of", ["N", "P", "K", "ph"])
    fig = px.histogram(view, x=feat, nbins=30, color=None if crop != "All" else None)
    st.plotly_chart(fig, width="stretch")

    st.markdown("**Average soil profile per crop**")
    st.dataframe(data.groupby("crop").mean().round(1), width="stretch")

# ---------------------------------------------------------------- Model
with tab_model:
    if metrics:
        st.subheader(f"Selected model: {metrics['best_model']}")
        st.metric("Held-out test accuracy", f"{metrics['test_accuracy']:.1%}")
        comp = pd.DataFrame(metrics["comparison"]).drop(columns=["best_params"])
        comp.columns = ["Algorithm", "CV accuracy", "Train accuracy", "Validation accuracy"]
        st.dataframe(comp.style.format({c: "{:.1%}" for c in comp.columns[1:]}), width="stretch", hide_index=True)
        st.caption("The model is picked on validation accuracy; the test set is used once for the final score.")
    else:
        st.info("Run `python -m src.train` to generate model metrics.")

    if hasattr(model, "feature_importances_"):
        imp = pd.Series(model.feature_importances_, index=model.feature_names_in_).sort_values()
        st.plotly_chart(px.bar(imp, orientation="h", labels={"value": "Importance", "index": "Feature"},
                               title="Feature importance"), width="stretch")

    st.warning("Only four soil features are used (no rainfall, temperature or humidity), so some crops with "
               "similar soil profiles - e.g. lentil, pigeonpeas, apple - are hard to separate.")
