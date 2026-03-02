import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import KFold, cross_val_score
from preprocessing import load_and_preprocess_data
from train import evaluate_model

def cross_validation_evaluation():
    """
    Performs 5-fold cross-validation on the best model.
    """
    print("Loading data for cross-validation...")
    df = pd.read_csv('data/placement_data_engineered.csv')
    X = df.drop('Placed', axis=1)
    y = df['Placed']
    
    print("Loading preprocessor and best model...")
    try:
        preprocessor = joblib.load('models/preprocessor.pkl')
        model = joblib.load('models/placement_model.pkl')
    except FileNotFoundError:
        print("Error: Models not found. Run train.py first.")
        return
        
    # Preprocess all data
    X_processed = preprocessor.transform(X)
    
    print("\nRunning 5-Fold Cross Validation...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores_acc = cross_val_score(model, X_processed, y, cv=kf, scoring='accuracy', n_jobs=-1)
    cv_scores_roc = cross_val_score(model, X_processed, y, cv=kf, scoring='roc_auc', n_jobs=-1)
    
    print(f"Cross-Validation Accuracy Scores: {cv_scores_acc}")
    print(f"Mean CV Accuracy: {cv_scores_acc.mean():.4f} (+/- {cv_scores_acc.std() * 2:.4f})")
    
    print(f"\nCross-Validation ROC-AUC Scores: {cv_scores_roc}")
    print(f"Mean CV ROC-AUC: {cv_scores_roc.mean():.4f} (+/- {cv_scores_roc.std() * 2:.4f})")

def custom_test_cases():
    """
    Testing specific edge cases.
    """
    print("\n--- Edge Case Testing ---")
    
    try:
        preprocessor = joblib.load('models/preprocessor.pkl')
        model = joblib.load('models/placement_model.pkl')
    except FileNotFoundError:
         print("Error: Models not found.")
         return
         
    # Expected Features layout based on generator
    # Load feature names from the raw dataset
    df_raw = pd.read_csv('data/placement_data.csv')
    df_raw = df_raw.drop('Placed', axis=1)
    
    # We need to recreate the derived features for edge cases properly
    from feature_engineering import create_derived_features
    
    # Base configuration template
    def build_student(**kwargs):
        # Default average student
        student = {
             'CGPA': 7.5,
             'Backlogs': 0,
             'Branch': 'CSE',
             '10th_Percentage': 80.0,
             '12th_Percentage': 80.0,
             'Programming_Languages_Known': 3,
             'DSA_Score': 70.0,
             'Projects_Count': 2,
             'Certifications': 1,
             'Internships': 1,
             'Hackathons_Participated': 1,
             'Coding_Contest_Rating': 1200,
             'Clubs': 1,
             'Leadership_Roles': 0
        }
        student.update(kwargs)
        df_student = pd.DataFrame([student])
        df_student = create_derived_features(df_student)
        return df_student
        
    # Case 1: Low CGPA + High Skills
    student_1 = build_student(
        CGPA=5.5,
        Backlogs=2,
        DSA_Score=95.0,
        Projects_Count=5,
        Internships=3,
        Coding_Contest_Rating=1800,
        Hackathons_Participated=4
    )
    
    # Case 2: High CGPA + Low Skills
    student_2 = build_student(
        CGPA=9.5,
        Backlogs=0,
        DSA_Score=20.0,
        Projects_Count=0,
        Internships=0,
        Coding_Contest_Rating=800,
        Hackathons_Participated=0
    )
    
    students_df = pd.concat([student_1, student_2], ignore_index=True)
    
    # Needs to match exactly the column structure after feature engineering
    # Drop features we know were removed by importance analyzer
    # The preprocessor expects exactly what it was fitted on
    # We will get column names from preprocessor if possible, or fitted data
    # Safe fallback: load training data columns
    df_engineered = pd.read_csv('data/placement_data_engineered.csv')
    expected_cols = df_engineered.drop('Placed', axis=1).columns
    
    students_final = students_df[expected_cols]
    
    X_processed = preprocessor.transform(students_final)
    
    probs = model.predict_proba(X_processed)[:, 1]
    
    print("\nCase 1: Low CGPA (5.5) + High Skills (DSA 95, 5 Projects, 3 Internships)")
    print(f"Predicted Placement Probability: {probs[0]:.2%}")
    
    print("\nCase 2: High CGPA (9.5) + Low Skills (DSA 20, 0 Projects, 0 Internships)")
    print(f"Predicted Placement Probability: {probs[1]:.2%}")

if __name__ == "__main__":
    cross_validation_evaluation()
    custom_test_cases()
