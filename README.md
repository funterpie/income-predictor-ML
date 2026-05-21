# 💰 Income Predictor — Adult Census Dataset
### SMIT Batch 10 | Machine Learning Classification Assignment

[![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-FastAPI%20Live-blue)](https://funterpie-income-predictor-ml.hf.space)
[![Streamlit](https://img.shields.io/badge/Streamlit-Frontend%20Live-red)](https://income-predictor-ml-funterpie.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.10-green)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## 📌 Project Overview

This project predicts whether a person's annual income exceeds **$50,000** based on census data. Built as part of the SMIT Batch 10 AI & Data Science program, it demonstrates a complete ML pipeline from data preprocessing to production deployment.

**Dataset:** [UCI Adult Census Income Dataset](https://archive.ics.uci.edu/ml/datasets/adult)
**Task:** Binary Classification (`<=50K` / `>50K`)
**Records:** 48,842 rows | 15 features

---

## 🚀 Live Demo

| Service | URL | Status |
|---------|-----|--------|
| 🤗 FastAPI Backend | [HuggingFace Space](https://funterpie-income-predictor-ml.hf.space) | ✅ Live |
| 🎯 Streamlit Frontend | `[https://income-predictor-ml-funterpie.streamlit.app/]` |✅ Live |
| 📖 API Docs (Swagger) | [/docs](https://funterpie-income-predictor-ml.hf.space/docs) | ✅ Live |

---

## 📊 Model Results

All 4 models were trained using **sklearn Pipeline** with:
- `StandardScaler` + `PolynomialFeatures(degree=2)` → Numeric features
- `OneHotEncoder` → Categorical features
- `SMOTE` → Class imbalance handling (3.18:1 ratio)

| Model | Accuracy | ROC-AUC | Status |
|-------|----------|---------|--------|
| **SVC (RBF Kernel)** | **79.62%** | **0.9026** | 🏆 Best Model |
| Logistic Regression | 81.82% | 0.8902 | ✅ |
| Decision Tree | 81.82% | 0.8902 | ✅ |
| KNN (k=5) | 78.56% | 0.8521 | ✅ |

### 🏆 Why SVC is the Best Model?

Although Logistic Regression and Decision Tree show higher accuracy (81.82%), **SVC wins** with the highest **ROC-AUC of 0.9026** — the more reliable metric for imbalanced datasets.

> Accuracy can be misleading when classes are imbalanced (76% vs 24%). ROC-AUC measures the model's ability to distinguish between both classes equally — SVC handles the minority class (`>50K`) significantly better than other models.

---

## 🏗️ Project Architecture

```
User (Streamlit UI)
        ↓
  Select dropdowns
        ↓
  POST /predict
        ↓
  FastAPI Backend (HuggingFace)
        ↓
  ImbPipeline:
    ├── ColumnTransformer
    │     ├── Numeric  → StandardScaler → PolynomialFeatures
    │     └── Categorical → OneHotEncoder
    └── SVC (RBF, probability=True)
        ↓
  { income, confidence, probabilities }
        ↓
  Result displayed to User
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| ML Framework | scikit-learn 1.6.1 |
| Imbalance Handling | imbalanced-learn (SMOTE) |
| Backend API | FastAPI + Uvicorn |
| Frontend | Streamlit |
| Containerization | Docker |
| Backend Deploy | HuggingFace Spaces |
| Frontend Deploy | Streamlit Cloud |
| Notebook | Jupyter |

---

## 📁 Repository Structure

```
income-predictor-ML/
│
├── Notebook.ipynb        # Complete ML pipeline notebook
├── model.pkl             # Saved best model (SVC pipeline)
├── col_info.json         # Column metadata for API
│
├── main.py               # FastAPI backend
├── app.py                # Streamlit frontend
│
├── Dockerfile            # HuggingFace deployment
├── requirements.txt      # Dependencies
└── README.md             # You are here
```

---

## ⚙️ Pipeline Design

```python
# Numeric Pipeline
numeric_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('poly',   PolynomialFeatures(degree=2))
])

# Categorical Pipeline
categorical_pipeline = Pipeline([
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combined Preprocessor
preprocessor = ColumnTransformer([
    ('num', numeric_pipeline,    num_cols),
    ('cat', categorical_pipeline, cat_cols)
])

# Full Pipeline with SMOTE
final_pipeline = ImbPipeline([
    ('preprocessor', preprocessor),
    ('smote',        SMOTE(random_state=42)),
    ('classifier',   SVC(kernel='rbf', probability=True))
])
```

---

## 🔌 API Reference

**Base URL:** `https://funterpie-income-predictor-ml.hf.space`

### `GET /`
Health check
```json
{ "status": "Income Predictor API live ✓" }
```

### `POST /predict`
**Request:**
```json
{
  "age": 35,
  "workclass": "Private",
  "educational_num": 13,
  "marital_status": "Married-civ-spouse",
  "occupation": "Exec-managerial",
  "relationship": "Husband",
  "race": "White",
  "gender": "Male",
  "capital_gain": 5000,
  "capital_loss": 0,
  "hours_per_week": 45,
  "native_country": "United-States"
}
```
**Response:**
```json
{
  "prediction": 1,
  "income": ">50K",
  "confidence": "84.2%",
  "prob_below": "15.8%",
  "prob_above": "84.2%"
}
```

---

## 🏃 Run Locally

```bash
# Clone repo
git clone https://github.com/funterpie/income-predictor-ML
cd income-predictor-ML

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run FastAPI
uvicorn main:app --reload

# Run Streamlit (new terminal)
streamlit run app.py
```

---

## 👨‍💻 Author

**Muhammad Taha Sattar Arain (Funter Pie)**
- 🌐 Portfolio: [tahatradz.online](https://tahatradz.online)
- 🏢 Agency: [alphaorbit.site](https://alphaorbit.site)
- 💼 LinkedIn: [taha-arain](https://linkedin.com/in/taha-arain)
- 🐙 GitHub: [funterpie](https://github.com/funterpie)

---

## 📚 Program

**SMIT (Saylani Mass IT Training)**
Batch 10 — AI & Data Science
Karachi, Pakistan

---

*Built with ❤️ by Funter Pie | Alpha Orbit*
