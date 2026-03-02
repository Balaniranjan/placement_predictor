import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV
import os

from preprocessing import load_and_preprocess_data

def evaluate_model(model, X_test, y_test):
    """
    Evaluates a model and returns metrics.
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred
    
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1_Score': f1_score(y_test, y_pred),
        'ROC_AUC': roc_auc_score(y_test, y_prob)
    }
    return metrics

def train_and_select_best_model():
    """
    Trains multiple models, evaluates them, tunes the best one, and saves it.
    """
    print("Loading and preprocessing data...")
    X_train, X_test, y_train, y_test, preprocessor = load_and_preprocess_data()
    
    # Process features for training
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # Initialize models
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'XGBoost': XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
    }
    
    print("\nTraining models and evaluating...")
    results = {}
    
    for name, model in models.items():
        model.fit(X_train_processed, y_train)
        metrics = evaluate_model(model, X_test_processed, y_test)
        results[name] = metrics
        print(f"{name} trained. ROC-AUC: {metrics['ROC_AUC']:.4f}")
        
    # Convert results to DataFrame for easy comparison
    results_df = pd.DataFrame(results).T
    print("\nModel Benchmark Results:")
    print(results_df)
    
    # Select Best Model based on ROC-AUC
    best_model_name = results_df['ROC_AUC'].idxmax()
    print(f"\nBest model selected: {best_model_name} with ROC-AUC {results_df.loc[best_model_name, 'ROC_AUC']:.4f}")
    
    best_model = models[best_model_name]
    
    # Hyperparameter Tuning using RandomizedSearchCV
    print(f"\nTuning {best_model_name}...")
    
    if best_model_name == 'Random Forest':
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    elif best_model_name == 'XGBoost':
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 1.0]
        }
    elif best_model_name == 'Gradient Boosting':
        param_grid = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7]
        }
    else:
        # Default fallback tuning options or skip
         param_grid = {}
         print(f"Skipping hyperparameter tuning for {best_model_name}")
         
    if param_grid:
        random_search = RandomizedSearchCV(
            estimator=best_model, 
            param_distributions=param_grid, 
            n_iter=10, 
            scoring='roc_auc', 
            cv=3, 
            verbose=1, 
            random_state=42, 
            n_jobs=-1
        )
        random_search.fit(X_train_processed, y_train)
        best_model = random_search.best_estimator_
        print(f"Best parameters: {random_search.best_params_}")
        
        # Evaluate tuned model
        tuned_metrics = evaluate_model(best_model, X_test_processed, y_test)
        print(f"Tuned Model ROC-AUC: {tuned_metrics['ROC_AUC']:.4f}")

    # Save Best Model
    os.makedirs('models', exist_ok=True)
    joblib.dump(best_model, 'models/placement_model.pkl')
    print("\nBest model saved to models/placement_model.pkl")

if __name__ == "__main__":
    train_and_select_best_model()
