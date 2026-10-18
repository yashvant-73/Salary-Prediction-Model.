from pathlib import Path
import json

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "salary_model.joblib"
DATA_PATH = ROOT / "data" / "Salary Data.csv"
METRICS_PATH = ROOT / "models" / "metrics.json"
FEATURES = ["Age", "Gender", "Education Level", "Job Title", "Years of Experience"]

st.set_page_config(page_title="Salary Prediction", page_icon="💼", layout="wide")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_data
def load_metrics():
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


st.title("💼 Salary Prediction")
st.write("Explore an educational salary estimate based on the provided employee details.")
st.info("This is a dataset-based estimate, not a salary guarantee or a compensation recommendation.")

try:
    model, df, metrics = load_model(), load_data(), load_metrics()
except Exception as exc:
    st.error(f"The model, metrics, or dataset could not be loaded: {exc}")
    st.stop()

education_options = sorted(df["Education Level"].dropna().astype(str).unique())
job_counts = metrics.get("job_title_counts", {})
job_options = sorted(df["Job Title"].dropna().astype(str).unique())
gender_options = sorted(df["Gender"].dropna().astype(str).unique())

st.subheader("Enter details")
with st.form("prediction_form"):
    left, right = st.columns(2)
    age = left.number_input("Age", min_value=18, max_value=100, value=25, step=1)
    experience = right.number_input("Years of experience", min_value=0.0, max_value=80.0, value=2.0, step=0.5)
    education = left.selectbox("Education level", education_options)
    gender = right.selectbox("Gender", gender_options)
    job_title = st.selectbox("Job title", job_options, format_func=lambda title: f"{title} ({job_counts.get(title, 0)} records)")
    submitted = st.form_submit_button("Predict Salary", type="primary")

if submitted:
    if age < experience + 14:
        st.warning("The entered experience is unusually high for this age. Please check the values.")
    if age < df["Age"].min() or age > df["Age"].max() or experience < df["Years of Experience"].min() or experience > df["Years of Experience"].max():
        st.warning("This age or experience is outside the observed dataset range; the estimate may be less reliable.")
    row = pd.DataFrame([{"Age": age, "Gender": gender, "Education Level": education, "Job Title": job_title, "Years of Experience": experience}], columns=FEATURES)
    estimate = float(model.predict(row)[0])
    st.subheader("Estimated salary")
    if estimate < 0:
        st.warning(f"The model returned a negative estimate ({estimate:,.0f}). This input is outside a useful prediction range.")
    else:
        st.success(f"{estimate:,.0f} (salary units used in the dataset)")
    st.caption("Individual outcomes vary. The training data may not reflect current market conditions or all relevant factors.")

st.divider()
st.subheader("Model performance")
st.caption(f"Selected model: {metrics['selected_model']} · held-out test set: {metrics['test_rows']} rows")
comparison = pd.DataFrame(metrics["comparison"]).set_index("model")
st.dataframe(comparison.style.format({"mae": "{:,.0f}", "rmse": "{:,.0f}", "r2": "{:.3f}", "cv_mae_mean": "{:,.0f}", "cv_r2_mean": "{:.3f}"}), use_container_width=True)
st.caption("Models are compared on the same split. MAE and RMSE use dataset salary units; R² is not an accuracy percentage.")

st.subheader("Actual versus predicted")
test = df.drop_duplicates().dropna(subset=["Salary"])
# Reproduce the fixed held-out split for a transparent visual; this is diagnostic only.
from sklearn.model_selection import train_test_split
_, test_indices = train_test_split(test.index, test_size=0.2, random_state=42)
actual = test.loc[test_indices, FEATURES + ["Salary"]]
predicted = model.predict(actual[FEATURES])
plot_data = pd.DataFrame({"Actual salary": actual["Salary"].values, "Predicted salary": predicted})
st.scatter_chart(plot_data, x="Actual salary", y="Predicted salary")

st.subheader("About and limitations")
st.markdown("""
This app uses a scikit-learn preprocessing and regression pipeline. Gender is retained as a feature to match the supplied project data; historical data may encode unfair differences. Removing a sensitive field does not remove proxy bias. The data includes many job titles with only one observation and a very low salary value that should be verified against the original source. Results should not be used as the sole basis for hiring, compensation, or financial decisions.
""")
