import pandas as pd
import numpy as np
import json

def create_derived_features(df):
    """
    Creates derived features for the placement model.
    """
    df = df.copy()
    
    # Process dynamic skills mapping
    def process_skills_dynamic(skills_str):
        try:
            if pd.isna(skills_str):
                return pd.Series({'total_skills': 0, 'average_skill_level': 1, 'weighted_skill_score': 0, 'technical_skill_ratio': 0})
            
            # Use json.loads for string, handle direct list if passed from API
            skills_list = json.loads(skills_str) if isinstance(skills_str, str) else skills_str
            
            if not skills_list:
                return pd.Series({'total_skills': 0, 'average_skill_level': 1, 'weighted_skill_score': 0, 'technical_skill_ratio': 0})
                
            total_skills = len(skills_list)
            
            core_tech_skills = ['python', 'java', 'c++', 'dsa', 'machine learning', 'sql', 'cloud', 'react', 'node', 'devops', 'cybersecurity', 'ai', 'data science']
            supporting_tech_skills = ['html', 'css', 'git', 'docker', 'excel', 'testing', 'figma']
            soft_skills = ['communication', 'leadership', 'teamwork', 'presentation', 'management']
            
            level_map = {'Beginner': 1, 'Intermediate': 2, 'Advanced': 3, 'Expert': 4}
            
            total_level = 0
            weighted_score = 0
            tech_count = 0
            
            for skill_item in skills_list:
                s_name = str(skill_item.get('skill', '')).lower().strip()
                level_str = skill_item.get('level', 'Beginner')
                level_num = level_map.get(level_str, 1)
                
                total_level += level_num
                
                if s_name in core_tech_skills:
                    weighted_score += level_num * 1.5
                    tech_count += 1
                elif s_name in supporting_tech_skills:
                    weighted_score += level_num * 1.2
                    tech_count += 1
                elif s_name in soft_skills:
                    weighted_score += level_num * 0.8
                else:
                    weighted_score += level_num * 1.0 # Unknown / General Tech
                    tech_count += 1 # Assume unknown is technical unless shown otherwise
                    
            avg_level = total_level / total_skills
            tech_ratio = tech_count / total_skills
            
            return pd.Series({
                'total_skills': total_skills,
                'average_skill_level': avg_level,
                'weighted_skill_score': weighted_score,
                'technical_skill_ratio': tech_ratio
            })
            
        except Exception as e:
            # Fallback for errors
            return pd.Series({'total_skills': 0, 'average_skill_level': 1, 'weighted_skill_score': 0, 'technical_skill_ratio': 0})
            
    # Apply processing
    if 'skills' in df.columns:
        skill_features = df['skills'].apply(process_skills_dynamic)
        df = pd.concat([df, skill_features], axis=1)
        # Drop raw skills column for modeling
        df = df.drop(columns=['skills'])
    else:
        # Failsafe if not generated 
        df['total_skills'] = 0
        df['average_skill_level'] = 1.0
        df['weighted_skill_score'] = 0.0
        df['technical_skill_ratio'] = 0.0
    
    # Updated Skill Score incorporating the weighted score and existing features
    # normalize weighted score conceptually max ~45
    df['Skill_Score'] = (
        (df['DSA_Score'] / 100) * 0.30 +
        (df['weighted_skill_score'] / 45) * 0.35 +
        (df['Projects_Count'] / 6) * 0.15 +
        (df['Certifications'] / 5) * 0.10 +
        (df['Internships'] / 3) * 0.10
    ) * 100
    
    # Activity Score = Hackathons (30%), Coding Rating (40%), Clubs (20%), Leadership (10%)
    df['Activity_Score'] = (
        (df['Hackathons_Participated'] / 5) * 0.3 +
        ((df['Coding_Contest_Rating'] - 800) / 1200) * 0.4 +
        (df['Clubs'] / 3) * 0.2 +
        (df['Leadership_Roles'] / 2) * 0.1
    ) * 100
    
    # Academic Strength = CGPA - (Backlogs * 0.3)
    df['Academic_Strength'] = df['CGPA'] - (df['Backlogs'] * 0.3)
    
    # Clip just in case
    df['Skill_Score'] = np.clip(df['Skill_Score'], 0, 100)
    df['Activity_Score'] = np.clip(df['Activity_Score'], 0, 100)
    
    # Drop intermediate collinear features to prevent negative weights in linear models
    collinear_cols = ['total_skills', 'average_skill_level', 'weighted_skill_score', 'technical_skill_ratio']
    df = df.drop(columns=[col for col in collinear_cols if col in df.columns], errors='ignore')
    
    return df

def feature_importance_analysis(df, target_col='Placed'):
    """
    Analyzes feature importance using a quick Random Forest model and returns the dataframe without irrelevant features.
    """
    from sklearn.ensemble import RandomForestClassifier
    import matplotlib.pyplot as plt
    import os

    # Quick encoding for 'Branch' just for feature importance analysis
    df_encoded = pd.get_dummies(df, columns=['Branch'], drop_first=True)
    
    X = df_encoded.drop(columns=[target_col])
    y = df_encoded[target_col]
    
    rf = RandomForestClassifier(random_state=42, n_estimators=50)
    rf.fit(X, y)
    
    importances = pd.Series(rf.feature_importances_, index=X.columns)
    importances = importances.sort_values(ascending=False)
    
    print("\nFeature Importances:")
    print(importances)
    
    # Keep the original string 'Branch' column for the main pipeline, but drop features with very low importance
    # Threshold < 0.01
    low_importance = importances[importances < 0.01].index.tolist()
    
    # Carefully translate encoding back to original column drop if necessary
    # In this logic, we might just drop '10th_Percentage' or '12th_Percentage' if they are too low
    cols_to_drop = []
    
    if 'Branch' not in df.columns:
         # If already encoded just drop encoded cols
         cols_to_drop = low_importance
    else:
        # Check original columns
        for col in df.columns:
            if col in low_importance and col != 'Branch' and col != target_col:
                cols_to_drop.append(col)
                
    if cols_to_drop:
        print(f"\nDropping low importance features: {cols_to_drop}")
        df = df.drop(columns=cols_to_drop)

    return df

def correlation_analysis(df):
    """
    Prints high correlations above 0.8 to monitor multicollinearity.
    """
    # Select numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr().abs()
    
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.85)]
    
    print(f"\nHighly correlated features (>0.85): {to_drop}")
    if to_drop:
        print("Note: These features might cause multicollinearity issues.")

if __name__ == "__main__":
    print("Testing feature engineering logic...")
    try:
        df = pd.read_csv('data/placement_data.csv')
        df = create_derived_features(df)
        correlation_analysis(df)
        df = feature_importance_analysis(df)
        df.to_csv('data/placement_data_engineered.csv', index=False)
        print("\nFeature engineering completed successfully!")
    except FileNotFoundError:
        print("Error: data/placement_data.csv not found. Run generator first.")
