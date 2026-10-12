# 🏥 Hospital Patient Readmission Prediction

A professional web application that predicts whether a hospital patient will be **readmitted within 30 days** using machine learning.

Upload your own hospital dataset through the app's interface — no pre-existing data file required.

---

## Features

| Feature | Description |
|---|---|
| **Dataset Upload** | Upload CSV, Excel (`.xlsx` / `.xls`), or ZIP files directly in the browser |
| **Auto-Detection** | Automatically identifies columns, target labels, and data types |
| **Column Mapping** | Manually correct column roles when auto-detection is uncertain |
| **ML Training** | Trains Logistic Regression, Random Forest, and Gradient Boosting models |
| **Best Model Selection** | Selects the model with the highest validation F1-score for readmission |
| **Patient Lookup** | Search by Patient ID to see historical data and model prediction side-by-side |
| **New Patient Prediction** | Enter details for a patient not in the dataset and get a prediction |
| **Performance Dashboard** | Accuracy, Precision, Recall, F1, ROC-AUC, confusion matrix, and model comparison |
| **Class-Imbalance Handling** | Uses class-weighted training and reports actual vs. predicted distributions |

---

## Quick Start (Windows)

```bash
# 1. Clone or download the project
cd hospital-readmission-app

# 2. Create a virtual environment
python -m venv .venv

# 3. Activate it
.venv\Scripts\activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Launch the app
streamlit run app.py
```

The app opens at **http://localhost:8501** in your default browser.

---

## How to Use

1. **Launch** the app — it starts without any dataset.
2. On the **Home** page, click **Browse files** to upload your hospital dataset.
3. The app auto-detects columns and the readmission target. Confirm or adjust the mapping.
4. Click **🚀 Train Model** to train and evaluate multiple classifiers.
5. Navigate to **Patient Lookup** to search for a specific patient.
6. Use **New Patient Prediction** for patients not in the dataset.
7. Review results on the **Model Performance** dashboard.

### Supported File Formats

| Format | Extension |
|---|---|
| CSV | `.csv` |
| Excel | `.xlsx`, `.xls` |
| ZIP archive | `.zip` (must contain a CSV or Excel file) |

### Expected Columns

The app works best with these columns (names are matched case-insensitively):

| Column | Purpose |
|---|---|
| `patient_id` | Patient identifier |
| `age` | Patient age |
| `gender` | Gender |
| `blood_pressure` | e.g. `130/72` — auto-split into systolic/diastolic |
| `cholesterol` | Cholesterol measurement |
| `bmi` | Body mass index |
| `diabetes` | Diabetes status |
| `hypertension` | Hypertension status |
| `medication_count` | Number of medications |
| `length_of_stay` | Days in hospital |
| `discharge_destination` | Where patient was discharged to |
| `readmitted_30_days` | **Target** — Yes/No, 1/0, True/False |

Other column names are supported; the app will ask you to confirm mappings.

---

## Project Structure

```
hospital-readmission-app/
├── app.py              # Streamlit web application (all pages)
├── train_model.py      # ML training, preprocessing, evaluation
├── requirements.txt    # Python dependencies
├── README.md           # This file
└── .gitignore          # Git ignore rules
```

---

## Technology Stack

- **Python 3.9+**
- **Streamlit** — interactive web UI
- **Pandas / NumPy** — data processing
- **Scikit-learn** — machine learning pipelines
- **Plotly** — interactive charts
- **OpenPyXL** — Excel file support
- **Joblib** — model persistence

---

## Medical Disclaimer

> This application is a **demonstration and educational tool**.
> It is **not** a certified medical device and must **not** be used for
> clinical decision-making, patient diagnosis, or treatment planning.
> Always consult qualified healthcare professionals.

---

## License

This project is provided for educational purposes.
