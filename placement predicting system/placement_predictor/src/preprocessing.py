import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import joblib
import os

def load_and_preprocess_data(data_path='data/placement_data_engineered.csv'):
    """
    Loads data, splits into train/test, and creates a preprocessing pipeline.
    """
    df = pd.read_csv(data_path)
    
    # Handle missing values if any
    # (Synthetic data shouldn't have any, but good for robust pipeline)
    X = df.drop('Placed', axis=1)
    y = df['Placed']
    
    # Identify numerical and categorical features
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns
    
    # Create preprocessing steps
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first'))
    ])
    
    # Combine steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # Split data (80-20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Fit the preprocessor on the training data BEFORE saving
    preprocessor.fit(X_train)
    
    # Save preprocessing pipeline for API
    os.makedirs('models', exist_ok=True)
    joblib.dump(preprocessor, 'models/preprocessor.pkl')
    print("Pre-processing pipeline saved to models/preprocessor.pkl")
    
    return X_train, X_test, y_train, y_test, preprocessor

if __name__ == "__main__":
    from feature_engineering import create_derived_features, feature_importance_analysis, correlation_analysis
    
    print("Running Preprocessing pipeline test...")
    if not os.path.exists('data/placement_data_engineered.csv'):
        print("Engineered dataset not found. Running feature engineering first...")
        df = pd.read_csv('data/placement_data.csv')
        df = create_derived_features(df)
        df_engineered = feature_importance_analysis(df)
        df_engineered.to_csv('data/placement_data_engineered.csv', index=False)
        print("Feature engineering completed.")
    
    X_train, X_test, y_train, y_test, preprocessor = load_and_preprocess_data()
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
