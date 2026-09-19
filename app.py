
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Cancer Diagnosis",
    page_icon="🧬"
)

st.title("🧬 Gene Expression Analysis")
st.subheader("Cancer Diagnosis Using Machine Learning")

# Load trained model
model = joblib.load("cancer_model.pkl")

# Upload dataset
uploaded_file = st.file_uploader(
    "Upload Gene Expression CSV",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Diagnosis Distribution")
    st.bar_chart(df["Diagnosis"].value_counts())

    gene_cols = [
        col for col in df.columns
        if col.startswith("Gene")
    ]

    st.subheader("Predict Diagnosis")

    with st.form("prediction_form"):

        values = {}

        for gene in gene_cols:
            values[gene] = st.number_input(
                gene,
                value=float(df[gene].median())
            )

        submit = st.form_submit_button("Predict")

    if submit:

        sample = pd.DataFrame([values])

        prediction = model.predict(sample)[0]

        st.success(
            f"Predicted Diagnosis: {prediction}"
        )

else:
    st.info("Upload your gene expression CSV.")
