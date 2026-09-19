
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from pathlib import Path

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="GeneScope | Gene Expression AI",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #08111f, #101c32);
    color: #f1f5f9;
}
[data-testid="stSidebar"] {
    background: #0b1526;
    border-right: 1px solid #263650;
}
.hero {
    padding: 30px;
    border-radius: 20px;
    background: linear-gradient(120deg, #123e59, #39418a);
    margin-bottom: 25px;
    border: 1px solid #4e648d;
}
.hero h1 {
    color: white;
    font-size: 38px;
    margin-bottom: 8px;
}
.hero p {
    color: #dbeafe;
    font-size: 17px;
}
.metric-card {
    padding: 20px;
    border-radius: 15px;
    background: #14243a;
    border: 1px solid #2d4564;
}
.metric-label {
    color: #a8bdd6;
    font-size: 14px;
}
.metric-value {
    color: #ffffff;
    font-size: 27px;
    font-weight: 700;
}
.section-title {
    color: #93c5fd;
    font-size: 23px;
    font-weight: 700;
    margin-top: 10px;
}
div.stButton > button {
    border-radius: 10px;
    border: 0;
    background: linear-gradient(90deg, #2563eb, #7c3aed);
    color: white;
    font-weight: 600;
    min-height: 45px;
}
div.stButton > button:hover {
    border: 1px solid #93c5fd;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
MODEL_PATH = Path("cancer_model.pkl")

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

try:
    model = load_model()
except Exception as e:
    st.error(f"Could not load cancer_model.pkl: {e}")
    st.stop()

GENES = [f"Gene{i}" for i in range(1, 21)]

# ---------------- HEADER ----------------
st.markdown("""
<div class="hero">
    <h1>🧬 GeneScope AI</h1>
    <p>Gene Expression Analytics & Machine Learning Research Dashboard</p>
    <p>Explore your dataset, visualize gene patterns, and run model predictions.</p>
</div>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
st.sidebar.title("🧬 GeneScope")
st.sidebar.caption("Cancer research demonstration")

page = st.sidebar.radio(
    "Navigate",
    [
        "Overview",
        "Dataset Explorer",
        "Single Prediction",
        "Batch Prediction",
        "About"
    ]
)

st.sidebar.divider()
st.sidebar.info(
    "Research/educational demonstration only. "
    "Predictions are not medical diagnoses."
)

# ---------------- DATA UPLOAD ----------------
st.sidebar.subheader("📂 Dataset")
uploaded_file = st.sidebar.file_uploader(
    "Upload gene expression CSV",
    type=["csv"]
)

@st.cache_data
def read_csv(file):
    return pd.read_csv(file)

df = None
if uploaded_file is not None:
    try:
        df = read_csv(uploaded_file)
    except Exception as e:
        st.sidebar.error(f"Could not read CSV: {e}")

if df is None and Path("gene_expression.csv").exists():
    try:
        df = pd.read_csv("gene_expression.csv")
    except Exception:
        df = None

if df is not None:
    missing = [g for g in GENES if g not in df.columns]
    if missing:
        st.warning(
            "The uploaded dataset is missing expected gene columns: "
            + ", ".join(missing)
        )

# ---------------- HELPERS ----------------
def make_gene_input(values):
    return pd.DataFrame([values], columns=GENES)

def predict_values(values_df):
    return model.predict(values_df)

def metric_card(label, value):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

def safe_numeric_genes(data):
    return data[GENES].apply(pd.to_numeric, errors="coerce")

# ---------------- OVERVIEW ----------------
if page == "Overview":
    st.markdown('<div class="section-title">📊 Project Overview</div>',
                unsafe_allow_html=True)

    if df is None:
        st.warning("Upload gene_expression.csv using the sidebar to view analytics.")
    else:
        total_samples = len(df)
        gene_count = sum(g in df.columns for g in GENES)
        diagnosis_count = (
            df["Diagnosis"].nunique()
            if "Diagnosis" in df.columns else "—"
        )

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            metric_card("Total Samples", f"{total_samples:,}")
        with c2:
            metric_card("Gene Features", gene_count)
        with c3:
            metric_card("Diagnosis Classes", diagnosis_count)
        with c4:
            metric_card("Missing Values", int(df.isna().sum().sum()))

        st.write("")
        left, right = st.columns([1.2, 1])

        with left:
            st.subheader("Dataset Preview")
            st.dataframe(df.head(10), use_container_width=True)

        with right:
            st.subheader("Diagnosis Distribution")
            if "Diagnosis" in df.columns:
                counts = df["Diagnosis"].value_counts().reset_index()
                counts.columns = ["Diagnosis", "Samples"]
                fig = px.pie(
                    counts,
                    names="Diagnosis",
                    values="Samples",
                    hole=0.55,
                    title="Samples by diagnosis"
                )
                fig.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No Diagnosis column found.")

# ---------------- DATASET EXPLORER ----------------
elif page == "Dataset Explorer":
    st.markdown('<div class="section-title">🔎 Dataset Explorer</div>',
                unsafe_allow_html=True)

    if df is None:
        st.warning("Upload a CSV dataset first.")
    else:
        st.subheader("Data preview")
        st.dataframe(df, use_container_width=True)

        st.subheader("Summary statistics")
        st.dataframe(df.describe(include="all"), use_container_width=True)

        st.subheader("Gene distribution")
        available_genes = [g for g in GENES if g in df.columns]

        if available_genes:
            selected_gene = st.selectbox("Select a gene", available_genes)
            gene_values = pd.to_numeric(
                df[selected_gene], errors="coerce"
            ).dropna()

            if not gene_values.empty:
                fig = px.histogram(
                    gene_values,
                    x=gene_values,
                    nbins=30,
                    title=f"{selected_gene} expression distribution"
                )
                fig.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)"
                )
                st.plotly_chart(fig, use_container_width=True)

        st.subheader("Gene correlation heatmap")
        numeric = safe_numeric_genes(df)
        if numeric.shape[1] > 1:
            corr = numeric.corr()
            fig = px.imshow(
                corr,
                text_auto=".2f",
                aspect="auto",
                title="Correlation between gene features"
            )
            fig.update_layout(template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)

        st.download_button(
            "⬇️ Download dataset",
            data=df.to_csv(index=False),
            file_name="gene_expression_dataset.csv",
            mime="text/csv"
        )

# ---------------- SINGLE PREDICTION ----------------
elif page == "Single Prediction":
    st.markdown('<div class="section-title">🧪 Single Sample Prediction</div>',
                unsafe_allow_html=True)

    st.write(
        "Enter expression values for all 20 genes, then run the model."
    )

    with st.form("single_prediction_form"):
        values = {}
        cols = st.columns(4)

        for i, gene in enumerate(GENES):
            with cols[i % 4]:
                values[gene] = st.number_input(
                    gene,
                    value=0.0,
                    format="%.6f",
                    key=f"single_{gene}"
                )

        submitted = st.form_submit_button(
            "🔍 Predict Diagnosis",
            use_container_width=True
        )

    if submitted:
        try:
            input_df = make_gene_input(values)
            prediction = model.predict(input_df)[0]

            st.success(f"Model prediction: **{prediction}**")

            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(input_df)[0]
                classes = model.classes_

                prob_df = pd.DataFrame({
                    "Class": classes,
                    "Probability": probabilities
                }).sort_values("Probability", ascending=False)

                st.subheader("Model probability estimates")
                fig = px.bar(
                    prob_df,
                    x="Class",
                    y="Probability",
                    color="Class",
                    text=prob_df["Probability"].map(
                        lambda x: f"{x:.1%}"
                    ),
                    title="Predicted class probabilities"
                )
                fig.update_layout(template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)

                st.caption(
                    "Probabilities are model outputs, not clinical risk estimates."
                )

        except Exception as e:
            st.error(f"Prediction failed: {e}")

# ---------------- BATCH PREDICTION ----------------
elif page == "Batch Prediction":
    st.markdown('<div class="section-title">📑 Batch Prediction</div>',
                unsafe_allow_html=True)

    st.write(
        "Upload a CSV containing Gene1 through Gene20. "
        "The app will add a Predicted_Diagnosis column."
    )

    batch_file = st.file_uploader(
        "Upload samples CSV",
        type=["csv"],
        key="batch_upload"
    )

    if batch_file is not None:
        try:
            batch_df = pd.read_csv(batch_file)
            missing = [g for g in GENES if g not in batch_df.columns]

            if missing:
                st.error("Missing gene columns: " + ", ".join(missing))
            else:
                batch_features = safe_numeric_genes(batch_df)

                if batch_features.isna().any().any():
                    st.error(
                        "Some gene values are missing or non-numeric. "
                        "Please clean the CSV and upload it again."
                    )
                else:
                    if st.button("🚀 Run Batch Prediction"):
                        predictions = model.predict(batch_features)
                        result_df = batch_df.copy()
                        result_df["Predicted_Diagnosis"] = predictions

                        st.success(
                            f"Completed predictions for {len(result_df)} samples."
                        )
                        st.dataframe(result_df, use_container_width=True)

                        st.download_button(
                            "⬇️ Download predictions",
                            data=result_df.to_csv(index=False),
                            file_name="gene_expression_predictions.csv",
                            mime="text/csv"
                        )

        except Exception as e:
            st.error(f"Could not process the CSV: {e}")

# ---------------- ABOUT ----------------
elif page == "About":
    st.markdown('<div class="section-title">ℹ️ About GeneScope AI</div>',
                unsafe_allow_html=True)

    st.markdown("""
    ### Gene Expression Analysis for Cancer Diagnosis

    This project demonstrates how machine learning can be applied to
    gene-expression data for a binary classification task.

    **Features**
    - Dataset exploration and descriptive statistics
    - Interactive gene-expression visualizations
    - Single-sample model prediction
    - Batch CSV prediction and result export

    **Technology**
    - Python
    - Streamlit
    - Pandas and NumPy
    - Scikit-learn / Joblib
    - Plotly

    **Important limitation**

    This application is an educational machine-learning demonstration.
    It is not validated for clinical use and must not be used to diagnose
    cancer or guide medical decisions.
    """)

st.divider()
st.caption("GeneScope AI • Gene Expression Research Dashboard • Educational use only")
