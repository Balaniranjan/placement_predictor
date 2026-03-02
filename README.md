# Placement Probability Predictor

An end-to-end Machine Learning system to predict student placement chances based on academic, skill, and activity metrics.

## Features
- **Synthetic Dataset Generator**: Configurable generation of student records with realistic correlations.
- **Robust Preprocessing Pipeline**: Automated imputation, encoding, and scaling using `scikit-learn`.
- **Feature Engineering**: Derivation of Composite scores (`Skill_Score`, `Activity_Score`) and automated feature selection using Random Forest importance.
- **Multi-Model Evaluation**: Automated training, evaluation, cross-validation, and hyperparameter tuning of 5 advanced ML Models (XGBoost, RandomForest, GradientBoosting, etc).
- **FastAPI Backend**: A production-ready API for realtime model inference.
- **Streamlit Dashboard**: A modern UI for interactive predictions with Data Analytics.
- **ML Explainability**: Integrated SHAP plots to explain why the model made a specific prediction.

## Project Structure
```bash
placement_predictor/
├── data/                       # Contains datasets
├── models/                     # Saved models and preprocessors
├── src/
│   ├── data_generator.py       # Script to generate synthetic data
│   ├── preprocessing.py        # Pipeline for cleaning data
│   ├── feature_engineering.py  # Script for creating derived features
│   ├── train.py                # Model training and tuning
│   └── evaluate.py             # Cross-validation and edge case evaluation
├── api/
│   └── main.py                 # FastAPI application
├── dashboard/
│   └── app.py                  # Streamlit frontend dashboard
├── requirements.txt            # Python dependencies
└── README.md
```

## Setup & Running

1. **Install Requirements**
```bash
pip install -r requirements.txt
```

2. **Generate Data & Train Model**
Run the training script which automatically triggers data generation and preprocessing if needed.
```bash
cd placement_predictor
python src/train.py
```

3. **Evaluate Model (Cross-validation & Edge cases)**
```bash
python src/evaluate.py
```

4. **Start the API (Backend)**
```bash
# Keep this running in a separate terminal
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

5. **Start the Dashboard (Frontend)**
```bash
streamlit run dashboard/app.py
```
