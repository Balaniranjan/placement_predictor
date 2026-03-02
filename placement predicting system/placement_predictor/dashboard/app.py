import streamlit as st
import pandas as pd
import numpy as np
import requests
import joblib
import shap
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import os
import sys
import json

# Configure page
st.set_page_config(page_title="Placement Probability Predictor", layout="wide", page_icon="🎓")

# Custom CSS for Modern UI
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: bold;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #45a049;
        box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
    }
    h1 {
        color: #2c3e50;
        text-align: center;
        margin-bottom: 30px;
    }
    .metric-card {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# API Endpoint
API_URL = "http://localhost:8000/predict"

# Load Model Details for Dashboard Charts
@st.cache_resource
def load_resources():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    model_path = os.path.join(base_dir, 'models', 'placement_model.pkl')
    preprocessor_path = os.path.join(base_dir, 'models', 'preprocessor.pkl')
    test_data_path = os.path.join(base_dir, 'data', 'placement_data_engineered.csv')
    
    try:
        model = joblib.load(model_path)
        preprocessor = joblib.load(preprocessor_path)
        df_test = pd.read_csv(test_data_path)
        return model, preprocessor, df_test
    except Exception as e:
        st.error(f"Failed to load required files. Make sure data, preprocessor and model exist. Details: {e}")
        return None, None, None

model, preprocessor, df_test = load_resources()

# Insert src into path so we can import feature_engineering locally for the dashboard
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from feature_engineering import create_derived_features

st.title("🎓 Placement Probability Predictor")
st.markdown("Enter your academic and extracurricular details to predict your chances of getting placed.")

# Create main layout
col1, col2 = st.columns([1, 2])

with col1:
    st.header("Student Profile")
    
    with st.form("prediction_form"):
        st.subheader("Academic Details")
        cgpa = st.slider("CGPA", 5.0, 10.0, 7.5, 0.1)
        backlogs = st.number_input("Active Backlogs", 0, 10, 0)
        branch = st.selectbox("Branch", ["CSE", "IT", "ECE", "MECH", "CIVIL", "CYBER", "AIDS", "AIML"])
        tenth = st.slider("10th Percentage", 60.0, 100.0, 80.0, 0.1)
        twelfth = st.slider("12th Percentage", 60.0, 100.0, 80.0, 0.1)
        
        st.subheader("Skill Details")
        st.markdown("Add your skills dynamically. You can add new rows to specify your proficiencies.")
        
        # Define default skills state to help users
        default_skills = pd.DataFrame([
            {"Skill Name": "Python", "Level": "Intermediate"},
            {"Skill Name": "Communication", "Level": "Advanced"},
            {"Skill Name": "", "Level": "Beginner"}
        ])
        
        skills_df = st.data_editor(
            default_skills, 
            num_rows="dynamic",
            column_config={
                "Skill Name": st.column_config.TextColumn("Skill Name"),
                "Level": st.column_config.SelectboxColumn("Level", options=["Beginner", "Intermediate", "Advanced", "Expert"], required=True)
            },
            hide_index=True
        )
        
        dsa = st.slider("DSA Score (Out of 100)", 0.0, 100.0, 60.0, 1.0)
        projects = st.number_input("Number of Projects", 0, 10, 2)
        certs = st.number_input("Certifications", 0, 10, 1)
        internships = st.number_input("Internships", 0, 5, 0)
        
        st.subheader("Activity Details")
        hackathons = st.number_input("Hackathons Participated", 0, 15, 0)
        coding_rating = st.slider("Coding Contest Rating", 800, 2500, 1200, 10)
        clubs = st.number_input("Clubs Participated", 0, 10, 0)
        leadership = st.number_input("Leadership Roles", 0, 5, 0)
        
        submit_button = st.form_submit_button(label="Predict Placement Probability")

with col2:
    if submit_button:
        # Prepare skills payload
        skills_payload = []
        for index, row in skills_df.iterrows():
            if pd.notna(row['Skill Name']) and str(row['Skill Name']).strip() != "":
                skills_payload.append({
                    "skill": str(row['Skill Name']).strip(),
                    "level": str(row['Level'])
                })
        
        # Prepare Payload
        student_data = {
            "CGPA": cgpa,
            "Backlogs": backlogs,
            "Branch": branch,
            "10th_Percentage": tenth,
            "12th_Percentage": twelfth,
            "skills": skills_payload,
            "DSA_Score": dsa,
            "Projects_Count": projects,
            "Certifications": certs,
            "Internships": internships,
            "Hackathons_Participated": hackathons,
            "Coding_Contest_Rating": coding_rating,
            "Clubs": clubs,
            "Leadership_Roles": leadership
        }
        
        with st.spinner('Analyzing profile...'):
            try:
                # Local Prediction logic if API is not running, 
                # helps when doing full end-to-end local testing 
                # where we may not want to start the API yet
                
                df_input = pd.DataFrame([student_data])
                
                # Transform list of dicts to JSON string to simulate feature engineering processing logic natively
                df_input['skills'] = df_input['skills'].apply(lambda x: json.dumps(x))
                
                df_input_engineered = create_derived_features(df_input)
                
                # Align columns
                expected_cols = df_test.drop('Placed', axis=1).columns
                df_input_final = df_input_engineered[expected_cols]
                
                # Transform and predict
                X_processed = preprocessor.transform(df_input_final)
                probability = model.predict_proba(X_processed)[0][1] * 100
                
                if probability > 75:
                    risk = "High Chance"
                    color = "#4CAF50" # Green
                elif probability >= 50:
                    risk = "Moderate Chance"
                    color = "#FFC107" # Yellow
                else:
                    risk = "Low Chance"
                    color = "#F44336" # Red
                
                st.header("Prediction Results")
                
                # 1. Gauge Chart
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = probability,
                    title = {'text': "Placement Probability (%)"},
                    gauge = {
                        'axis': {'range': [None, 100]},
                        'bar': {'color': color},
                        'steps': [
                            {'range': [0, 50], 'color': "rgba(244, 67, 54, 0.2)"},
                            {'range': [50, 75], 'color': "rgba(255, 193, 7, 0.2)"},
                            {'range': [75, 100], 'color': "rgba(76, 175, 80, 0.2)"}
                        ],
                        'threshold': {
                            'line': {'color': "black", 'width': 4},
                            'thickness': 0.75,
                            'value': probability
                        }
                    }
                ))
                fig.update_layout(height=400, margin=dict(l=20, r=20, t=50, b=20))
                st.plotly_chart(fig, use_container_width=True)
                
                st.success(f"Risk Assessment: **{risk}**")

                st.markdown("---")
                st.header("Model Explainability (SHAP)")
                
                st.markdown("""
                This section explains **why** the model made this specific prediction. 
                Features pushing the prediction higher are shown in **red/pink**, while features pushing it lower are in **blue**.
                """)
                
                # Generate SHAP explanation for the single instance
                # We need the tree explainer for tree-based models
                try:
                    # Get feature names from preprocessor
                    # Cat columns will be tricky without one-hot encoding feature names
                    # We can use the processed column names if possible.
                    if hasattr(preprocessor, 'get_feature_names_out'):
                         feature_names = preprocessor.get_feature_names_out()
                    else:
                         # fallback
                         feature_names = [f"Feature {i}" for i in range(X_processed.shape[1])]
                    
                    # Convert to dataframe with feature names for SHAP
                    X_processed_df = pd.DataFrame(X_processed, columns=feature_names)
                    
                    # Determine the right explainer based on model type
                    if type(model).__name__ == 'LogisticRegression':
                        # Use LinearExplainer for Logistic Regression
                        explainer = shap.LinearExplainer(model, X_processed_df)
                    else:
                        # Use TreeExplainer for Tree based models
                        explainer = shap.TreeExplainer(model)
                        
                    shap_values = explainer.shap_values(X_processed_df)
                    
                    # For classification, shap returns list of arrays (one for each class) for TreeExplainer
                    # But LinearExplainer returns a single array
                    if isinstance(shap_values, list) and len(shap_values) > 1:
                        target_shap = shap_values[1]
                        base_value = explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value
                    else:
                        target_shap = shap_values
                        base_value = explainer.expected_value
                        
                    # Fix formatting for numpy target shape
                    if len(target_shap.shape) > 1:
                         target_shap_instance = target_shap[0]
                    else:
                         target_shap_instance = target_shap
                    
                    # Plot waterfall
                    fig_shap, ax = plt.subplots(figsize=(10, 6))
                    shap.waterfall_plot(shap.Explanation(
                         values=target_shap_instance, 
                         base_values=base_value, 
                         data=X_processed_df.iloc[0], 
                         feature_names=feature_names
                    ), show=False)
                    plt.tight_layout()
                    st.pyplot(fig_shap)
                    
                    # Highlight top 3 factors
                    # Get absolute values to find most impactful
                    shap_abs = np.abs(target_shap_instance)
                    top_indices = np.argsort(shap_abs)[-3:][::-1]
                    
                    st.subheader("Top 3 Contributing Factors:")
                    for i, idx in enumerate(top_indices):
                        feat_name = feature_names[idx]
                        feat_val = target_shap_instance[idx]
                        direction = "Positive" if feat_val > 0 else "Negative"
                        color = "green" if feat_val > 0 else "red"
                        st.markdown(f"{i+1}. **{feat_name}**: <span style='color:{color}'>{direction} impact</span> (Value: {X_processed_df.iloc[0][feat_name]:.2f})", unsafe_allow_html=True)
                        
                except Exception as e:
                     st.warning(f"SHAP explanation limited or unavailable for this model type. Error: {e}")
                
            except Exception as e:
                st.error(f"Error making prediction: {e}")
    else:
        st.info("👈 Fill out the form and click 'Predict Placement Probability' to see your results.")

# Secondary Layout for Exploratory Analytics
st.markdown("---")
st.header("Dataset Analytics & Global Explanations")

if df_test is not None:
    tab1, tab2, tab3 = st.tabs(["Feature Importance", "CGPA vs Placement", "Skill Score vs Placement"])
    
    with tab1:
        st.subheader("Global Feature Importance")
        st.markdown("Shows the overall importance of features across all students.")
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            if hasattr(preprocessor, 'get_feature_names_out'):
                 feature_names = preprocessor.get_feature_names_out()
            else:
                 feature_names = [f"Feature {i}" for i in range(len(importances))]
                 
            imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
            imp_df = imp_df.sort_values(by='Importance', ascending=True).tail(15)
            
            fig = px.bar(imp_df, x='Importance', y='Feature', orientation='h', 
                         title="Top 15 Most Important Features")
            st.plotly_chart(fig, use_container_width=True)
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_[0])
            if hasattr(preprocessor, 'get_feature_names_out'):
                 feature_names = preprocessor.get_feature_names_out()
            else:
                 feature_names = [f"Feature {i}" for i in range(len(importances))]
                 
            imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
            imp_df = imp_df.sort_values(by='Importance', ascending=True).tail(15)
            
            fig = px.bar(imp_df, x='Importance', y='Feature', orientation='h', 
                         title="Top 15 Most Important Features (Absolute Coefficients)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Feature importance not supported by this model.")
            
    with tab2:
        st.subheader("CGPA vs Placement Distribution")
        # Visualizing the density plot of CGPA grouped by Placement status
        fig = px.histogram(df_test, x="CGPA", color="Placed", marginal="box",
                           title="CGPA Distribution by Placement Status",
                           color_discrete_map={0: "#f44336", 1: "#4caf50"},
                           labels={'Placed': 'Placement Status'})
        st.plotly_chart(fig, use_container_width=True)
        
    with tab3:
         st.subheader("Skill Score vs Placement Distribution")
         if 'Skill_Score' in df_test.columns:
             fig = px.histogram(df_test, x="Skill_Score", color="Placed", marginal="box",
                               title="Skill Score Distribution by Placement Status",
                               color_discrete_map={0: "#f44336", 1: "#4caf50"},
                               labels={'Placed': 'Placement Status'})
             st.plotly_chart(fig, use_container_width=True)
         else:
             st.info("Skill_Score feature not available in the dataset.")
