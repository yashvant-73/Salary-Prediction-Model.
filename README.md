# Salary Prediction Using Machine Learning

An educational Streamlit application that estimates salary from age, gender, education, job title, and years of experience. It compares four regression models with consistent preprocessing and displays held-out test metrics.

## Project structure

```text
data/Salary Data.csv              Source dataset
notebooks/salary_prediction_model.ipynb  Original notebook (preserved)
src/train_model.py                Training, comparison, and evaluation
models/salary_model.joblib        Selected fitted pipeline (created by training)
models/metrics.json               Model comparison and app metadata
app.py                            Streamlit interface
requirements.txt                  Pinned runtime dependencies
```

## Dataset and data quality

The supplied CSV has 375 rows and columns `Age`, `Gender`, `Education Level`, `Job Title`, `Years of Experience`, and `Salary`. Initial inspection found 2 missing values in every column, 50 exact duplicate rows, 174 distinct job titles, and one salary of 350 while the next-lowest is 30,000. The source file is preserved. Training removes exact duplicate rows and rows with missing/non-numeric target salary; feature missing values are imputed inside the train-fitted pipeline. The 350 salary should be checked against its source before drawing conclusions.

## Run locally

Use Python 3.10 or later. In the project root:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python src/train_model.py
streamlit run app.py
```

Open the local URL printed by Streamlit, usually `http://localhost:8501`. Training writes the selected pipeline and metrics to `models/`.

## Evaluation

The training script compares Linear Regression, Decision Tree, Random Forest, and Gradient Boosting using the same 80/20 split after dropping exact duplicates. It selects by mean five-fold cross-validation MAE on training data. Results from the supplied dataset and this fixed split:

| Model | Test MAE | Test RMSE | Test R² | Mean CV MAE |
|---|---:|---:|---:|---:|
| Linear Regression | 12,305 | 15,852 | 0.867 | 14,668 |
| Decision Tree | 11,532 | 16,900 | 0.849 | 13,778 |
| Random Forest | 10,924 | 15,732 | 0.869 | 12,433 |
| Gradient Boosting (selected) | 10,636 | 14,729 | 0.885 | 11,257 |

There were 259 training rows and 65 held-out test rows. These are dataset salary units. Do not interpret R² as a percentage of correct individual predictions; the unusually low salary and sparse job titles may affect results. The supplied notebook's previous score is not directly comparable because its preprocessing and split differ.

## Responsible use and limitations

This is an educational estimate and not a salary guarantee. The dataset has many sparse job titles and possible data quality issues; current salary also depends on location, industry, company, and market conditions. Gender is included to match the source features, but historical data may encode unfair differences. Removing gender alone does not resolve proxy bias. Do not use this model as the sole basis for hiring, compensation, or financial decisions.

## Deployment

For Streamlit Community Cloud, push this project to a GitHub repository, then create an app using the repository's `main` branch and `app.py` as the entry point. Ensure the dataset and generated `models/salary_model.joblib` plus `models/metrics.json` are committed only if you have permission to redistribute the dataset and model. Community Cloud builds the app from `requirements.txt`; after deployment, add the resulting URL here.

No live application URL is configured yet.
