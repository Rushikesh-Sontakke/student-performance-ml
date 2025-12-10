"""
Interactive Streamlit Dashboard for Student Performance Prediction
Combines all models with visualizations and explainability
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from ensemble_model import StudentPerformanceEnsemble
from utils import load_data, calculate_metrics
import os

# Page configuration
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_ensemble_model():
    """Load the trained ensemble model (cached)"""
    try:
        ensemble = StudentPerformanceEnsemble()
        ensemble.load_models()
        return ensemble
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.info("Please run ensemble_model.py first to train and save the models.")
        return None

@st.cache_data
def load_dataset():
    """Load the dataset (cached)"""
    try:
        X_train, y_train, X_test, y_test = load_data()
        return X_train, y_train, X_test, y_test
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None, None, None

def home_page():
    """Home page with project overview"""
    st.markdown('<h1 class="main-header">🎓 Student Performance Prediction System</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(label="🤖 Models", value="4 + Ensemble", delta="ML Algorithms")
    with col2:
        st.metric(label="📊 Data Points", value="6,607", delta="Students")
    with col3:
        st.metric(label="🎯 Features", value="19", delta="Predictors")
    
    st.markdown("---")
    
    st.markdown("""
    ## 🌟 Project Overview
    
    This advanced machine learning system predicts student exam scores based on **19 academic and socio-economic factors**.
    Our approach combines **four state-of-the-art ML algorithms** into an intelligent ensemble for superior accuracy.
    
    ### 🔬 Models Implemented
    
    1. **Linear Regression** - Baseline statistical model
    2. **Random Forest** - Ensemble of decision trees
    3. **XGBoost** - Gradient boosting champion
    4. **Neural Network** - Deep learning approach
    5. **Weighted Ensemble** - Intelligent combination of all models
    
    ### ✨ Key Features
    
    - 🎯 **Superior Accuracy**: Ensemble approach combines strengths of all models
    - 🔍 **Explainability**: SHAP values show which factors matter most
    - 📈 **Interactive Predictions**: Try different scenarios in real-time
    - 📊 **Comprehensive Analysis**: Deep dive into model performance
    
    ### 📚 How to Use This Dashboard
    
    Use the **sidebar** to navigate between different sections:
    - **Home**: Project overview (you are here)
    - **Model Comparison**: See how different models perform
    - **Make Predictions**: Try the model with your own inputs
    - **Feature Importance**: Understand what drives predictions
    - **About**: Learn about the methodology
    """)
    
    st.markdown("---")
    st.info("💡 **Tip**: Start with 'Model Comparison' to see performance metrics, " +
            "then try 'Make Predictions' to test the models!")

def model_comparison_page():
    """Model comparison page"""
    st.markdown('<h1 class="main-header">📊 Model Performance Comparison</h1>', 
                unsafe_allow_html=True)
    
    # Load results if available
    if os.path.exists('results/ensemble_comparison.csv'):
        results_df = pd.read_csv('results/ensemble_comparison.csv')
        
        st.markdown("### 📈 Performance Metrics")
        
        # Highlight the best model
        styled_df = results_df.style.highlight_min(
            subset=['MSE', 'RMSE', 'MAE', 'MAPE'],
            color='lightgreen'
        ).highlight_max(
            subset=['R²'],
            color='lightgreen'
        ).format({
            'MSE': '{:.4f}',
            'RMSE': '{:.4f}',
            'MAE': '{:.4f}',
            'R²': '{:.4f}',
            'MAPE': '{:.2f}%'
        })
        
        st.dataframe(styled_df, use_container_width=True)
        
        # Visualizations
        st.markdown("### 📊 Visual Comparison")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # RMSE comparison
            fig_rmse = px.bar(
                results_df.sort_values('RMSE'),
                x='RMSE',
                y='Model',
                orientation='h',
                title='RMSE Comparison (Lower is Better)',
                color='Model',
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig_rmse.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig_rmse, use_container_width=True)
        
        with col2:
            # R² comparison
            fig_r2 = px.bar(
                results_df.sort_values('R²', ascending=False),
                x='R²',
                y='Model',
                orientation='h',
                title='R² Score Comparison (Higher is Better)',
                color='Model',
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            fig_r2.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig_r2, use_container_width=True)
        
        # MAE and MAPE comparison
        col3, col4 = st.columns(2)
        
        with col3:
            fig_mae = px.bar(
                results_df.sort_values('MAE'),
                x='MAE',
                y='Model',
                orientation='h',
                title='MAE Comparison (Lower is Better)',
                color='Model',
                color_discrete_sequence=px.colors.qualitative.Bold
            )
            fig_mae.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig_mae, use_container_width=True)
        
        with col4:
            # Metrics radar chart
            best_model = results_df.iloc[0]['Model']
            st.markdown(f"### 🏆 Best Model: **{best_model}**")
            st.success(f"RMSE: {results_df.iloc[0]['RMSE']:.4f} | " +
                      f"R²: {results_df.iloc[0]['R²']:.4f}")
    else:
        st.warning("⚠️ Model comparison results not found. Please run `ensemble_model.py` first.")
        st.code("python ensemble_model.py", language="bash")

def prediction_page():
    """Interactive prediction page"""
    st.markdown('<h1 class="main-header">🎯 Make Predictions</h1>', 
                unsafe_allow_html=True)
    
    ensemble = load_ensemble_model()
    
    if ensemble is None:
        st.error("Please train the model first by running ensemble_model.py")
        return
    
    # Import preprocessor
    try:
        from preprocessor import RAW_INPUT_CONFIG, RawInputPreprocessor
        processor = RawInputPreprocessor()
    except Exception as e:
        st.error(f"Error loading preprocessor: {str(e)}")
        return
    
    st.markdown("### 📝 Enter Student Characteristics")
    
    # Create input form with raw values
    with st.form("prediction_form"):
        st.markdown("#### 📚 Academic Information")
        col1, col2, col3 = st.columns(3)
        
        input_values = {}
        
        # Row 1: Academic basics
        with col1:
            input_values['Hours_Studied'] = st.number_input(
                "Hours Studied per Week",
                min_value=0, max_value=44, value=20,
                help="How many hours does the student study per week?"
            )
        with col2:
            input_values['Attendance'] = st.number_input(
                "Attendance (%)",
                min_value=60, max_value=100, value=85,
                help="Class attendance percentage"
            )
        with col3:
            input_values['Previous_Scores'] = st.number_input(
                "Previous Exam Scores",
                min_value=50, max_value=100, value=75,
                help="Average of previous exam scores"
            )
        
        # Row 2: Support & Resources
        st.markdown("#### 🎓 Support & Resources")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            input_values['Parental_Involvement'] = st.selectbox(
                "Parental Involvement",
                options=['None', 'Low', 'Medium', 'High'],
                index=2
            )
        with col2:
            input_values['Access_to_Resources'] = st.selectbox(
                "Access to Resources",
                options=['Low', 'Medium', 'High'],
                index=1
            )
        with col3:
            input_values['Tutoring_Sessions'] = st.number_input(
                "Tutoring Sessions per Month",
                min_value=0, max_value=8, value=2
            )
        
        # Row 3: School Environment
        st.markdown("#### 🏫 School Environment")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            input_values['Teacher_Quality'] = st.selectbox(
                "Teacher Quality",
                options=['Low', 'Medium', 'High'],
                index=2
            )
        with col2:
            input_values['School_Type'] = st.selectbox(
                "School Type",
                options=['Public', 'Private'],
                index=0
            )
        with col3:
            input_values['Peer_Influence'] = st.selectbox(
                "Peer Influence",
                options=['Positive', 'Neutral', 'Negative'],
                index=0
            )
        
        # Row 4: Personal Factors
        st.markdown("#### 👤 Personal Factors")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            input_values['Sleep_Hours'] = st.number_input(
                "Sleep Hours per Night",
                min_value=4, max_value=10, value=7
            )
        with col2:
            input_values['Physical_Activity'] = st.number_input(
                "Physical Activity (hours/week)",
                min_value=0, max_value=6, value=3
            )
        with col3:
            input_values['Motivation_Level'] = st.selectbox(
                "Motivation Level",
                options=['Low', 'Medium', 'High'],
                index=1
            )
        
        # Row 5: Demographics & Other
        st.markdown("#### 📊 Demographics & Other Information")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            input_values['Gender'] = st.selectbox(
                "Gender",
                options=['Male', 'Female'],
                index=0
            )
        with col2:
            input_values['Family_Income'] = st.selectbox(
                "Family Income",
                options=['Low', 'Medium', 'High'],
                index=1
            )
        with col3:
            input_values['Parental_Education_Level'] = st.selectbox(
                "Parental Education",
                options=['High School', 'College', 'Postgraduate'],
                index=1
            )
        
        # Row 6: Additional Factors
        col1, col2, col3 = st.columns(3)
        
        with col1:
            input_values['Distance_from_Home'] = st.selectbox(
                "Distance from Home",
                options=['Near', 'Moderate', 'Far'],
                index=0
            )
        with col2:
            input_values['Extracurricular_Activities'] = st.selectbox(
                "Extracurricular Activities",
                options=['Yes', 'No'],
                index=0
            )
        with col3:
            input_values['Internet_Access'] = st.selectbox(
                "Internet Access",
                options=['Yes', 'No'],
                index=0
            )
        
        col1, col2, col3 = st.columns(3)
        with col1:
            input_values['Learning_Disabilities'] = st.selectbox(
                "Learning Disabilities",
                options=['No', 'Yes'],
                index=0
            )
        
        submitted = st.form_submit_button("🔮 Predict Exam Score", use_container_width=True)
    
    if submitted:
        # Validate input
        is_valid, message = processor.validate_input(input_values)
        
        if not is_valid:
            st.error(f"❌ Invalid input: {message}")
            return
        
        # Show raw inputs
        with st.expander("📋 View Raw Input Values"):
            st.json(input_values)
        
        # Preprocess raw input
        try:
            X_processed = processor.transform_raw_input(input_values)
            
            # Make predictions with all models
            predictions = ensemble.predict_all(X_processed)
            
            st.markdown("---")
            st.markdown("### 🎯 Prediction Results")
            
            # Display predictions in metric cards
            cols = st.columns(5)
            for idx, (model_name, pred) in enumerate(predictions.items()):
                with cols[idx]:
                    # Highlight ensemble
                    delta_val = None if model_name == 'Ensemble' else f"{pred[0] - predictions['Ensemble'][0]:+.1f}"
                    
                    st.metric(
                        label=model_name,
                        value=f"{pred[0]:.1f}",
                        delta=delta_val,
                        delta_color="off"
                    )
            
            # Visualization
            st.markdown("### 📊 Prediction Comparison")
            
            pred_df = pd.DataFrame({
                'Model': list(predictions.keys()),
                'Predicted Score': [p[0] for p in predictions.values()]
            })
            
            fig = px.bar(
                pred_df,
                x='Model',
                y='Predicted Score',
                title='Predicted Exam Scores by Model',
                color='Model',
                color_discrete_sequence=px.colors.qualitative.Set2,
                text='Predicted Score'
            )
            fig.update_traces(texttemplate='%{text:.1f}', textposition='outside')
            fig.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Ensemble explanation
            st.info(f"**🎓 Ensemble Prediction: {predictions['Ensemble'][0]:.1f}** - " +
                   "This is the weighted average of all models, typically the most reliable prediction.")
            
            # Student profile summary
            st.markdown("### 📖 Student Profile Summary")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Strong Points:**")
                strong_points = []
                if input_values['Hours_Studied'] >= 25:
                    strong_points.append("✅ Excellent study hours")
                if input_values['Attendance'] >= 90:
                    strong_points.append("✅ High attendance")
                if input_values['Parental_Involvement'] in ['High', 'Medium']:
                    strong_points.append("✅ Good parental support")
                if input_values['Motivation_Level'] == 'High':
                    strong_points.append("✅ Highly motivated")
                
                if strong_points:
                    for point in strong_points:
                        st.write(point)
                else:
                    st.write("Consider improving study habits")
            
            with col2:
                st.markdown("**Areas for Improvement:**")
                improvements = []
                if input_values['Hours_Studied'] < 15:
                    improvements.append("⚠️ Increase study hours")
                if input_values['Attendance'] < 80:
                    improvements.append("⚠️ Improve attendance")
                if input_values['Sleep_Hours'] < 6:
                    improvements.append("⚠️ Get more sleep")
                if input_values['Tutoring_Sessions'] == 0:
                    improvements.append("💡 Consider tutoring")
                
                if improvements:
                    for point in improvements:
                        st.write(point)
                else:
                    st.write("✅ All factors look good!")
            
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            st.code(str(e))

def feature_importance_page():
    """Feature importance visualization page"""
    st.markdown('<h1 class="main-header">🔍 Feature Importance Analysis</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    Understanding which factors most strongly influence student performance helps educators 
    make informed decisions about resource allocation and intervention strategies.
    """)
    
    # Check if SHAP results exist
    importance_files = [
        f for f in os.listdir('results') if f.startswith('feature_importance_')
    ] if os.path.exists('results') else []
    
    if importance_files:
        st.markdown("### 📊 Top 10 Most Important Features by Model")
        
        # Model selector
        model_options = [f.replace('feature_importance_', '').replace('.csv', '').replace('_', ' ').title() 
                        for f in importance_files]
        selected_model = st.selectbox("Select Model", model_options)
        
        # Load corresponding file
        filename = f"results/feature_importance_{selected_model.lower().replace(' ', '_')}.csv"
        if os.path.exists(filename):
            importance_df = pd.read_csv(filename).head(10)
            
            # Plot
            fig = px.bar(
                importance_df,
                y='Feature',
                x='Importance',
                orientation='h',
                title=f'{selected_model}: Feature Importance (SHAP)',
                color='Importance',
                color_continuous_scale='Viridis',
                text='Importance'
            )
            fig.update_traces(texttemplate='%{text:.4f}', textposition='outside')
            fig.update_layout(height=500, showlegend=False)
            fig.update_yaxes(categoryorder='total ascending')
            st.plotly_chart(fig, use_container_width=True)
            
            # Display dataframe
            st.dataframe(importance_df, use_container_width=True)
        
        # Show SHAP visualizations if available
        st.markdown("### 📈 SHAP Visualizations")
        
        viz_files = [f for f in os.listdir('results') if f.endswith('.png')] if os.path.exists('results') else []
        
        if viz_files:
            selected_viz = st.selectbox("Select Visualization", viz_files)
            st.image(f"results/{selected_viz}", caption=selected_viz, use_column_width=True)
    else:
        st.warning("⚠️ Feature importance analysis not found. Please run `model_explainer.py` first.")
        st.code("python model_explainer.py", language="bash")

def about_page():
    """About page with methodology"""
    st.markdown('<h1 class="main-header">ℹ️ About This Project</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    ## 🎯 Project Objective
    
    Develop a robust machine learning system to predict student exam scores based on various 
    academic and socio-economic factors, enabling early intervention for at-risk students.
    
    ## 🔬 Methodology
    
    ### 1. Data Collection & Preprocessing
    - **Dataset**: 6,607 student records with 19 features
    - **Features**: Study hours, attendance, parental involvement, access to resources, etc.
    - **Preprocessing**: Normalization, encoding categorical variables, train-test split
    
    ### 2. Model Development
    
    #### Linear Regression
    - **Type**: Statistical baseline model
    - **Strengths**: Interpretable, fast training
    - **Use Case**: Understanding linear relationships
    
    #### Random Forest
    - **Type**: Ensemble of decision trees
    - **Hyperparameters**: 300 trees, max depth 30
    - **Strengths**: Handles non-linear relationships, robust to outliers
    
    #### XGBoost
    - **Type**: Gradient boosting algorithm
    - **Hyperparameters**: 300 estimators, learning rate 0.1
    - **Strengths**: State-of-the-art performance, feature importance
    
    #### Neural Network
    - **Architecture**: 4 hidden layers (256→128→64→32 neurons)
    - **Activation**: ReLU with batch normalization
    - **Regularization**: L2 regularization, dropout, early stopping
    
    ### 3. Ensemble Strategy
    
    Our final model uses **weighted averaging** where each model's contribution is proportional 
    to its validation performance:
    
    ```
    Ensemble Prediction = Σ (weight_i × prediction_i)
    ```
    
    Weights are calculated as:
    ```
    weight_i = (1 / RMSE_i) / Σ(1 / RMSE_j)
    ```
    
    ### 4. Explainability (SHAP)
    
    We use **SHAP (SHapley Additive exPlanations)** to interpret model predictions:
    - Identify most important features
    - Understand feature interactions
    - Explain individual predictions
    
    ## 📊 Evaluation Metrics
    
    - **RMSE** (Root Mean Squared Error): Penalizes large errors
    - **MAE** (Mean Absolute Error): Average prediction error
    - **R²** (R-squared): Proportion of variance explained
    - **MAPE** (Mean Absolute Percentage Error): Percentage error
    
    ## 🚀 Key Differentiators
    
    1. **Multi-Model Ensemble**: Combines 4 different algorithms
    2. **Explainable AI**: SHAP values for transparency
    3. **Interactive Dashboard**: User-friendly interface
    4. **Comprehensive Analysis**: Deep performance evaluation
    
    ## 👥 Technologies Used
    
    - **ML Frameworks**: scikit-learn, XGBoost, TensorFlow
    - **Explainability**: SHAP
    - **Visualization**: Matplotlib, Seaborn, Plotly
    - **Dashboard**: Streamlit
    - **Data Processing**: Pandas, NumPy
    
    ## 📧 Contact
    
    For questions or collaborations, please reach out through Email: sontakkerushikesh547@gmail.com.
    """)


def main():
    """Main app function"""
    
    # Sidebar navigation
    st.sidebar.title("🎓 Navigation")
    
    page = st.sidebar.radio(
        "Go to",
        ["🏠 Home", "📊 Model Comparison", "🎯 Make Predictions", 
         "🔍 Feature Importance", "ℹ️ About"],
        label_visibility="collapsed"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    ### 📚 Quick Actions
    - Run all models: `python ensemble_model.py`
    - Generate SHAP: `python model_explainer.py`
    - Compare models: `python model_comparison.py`
    """)
    
    st.sidebar.markdown("---")
    st.sidebar.info("💡 This dashboard showcases an advanced ML ensemble for student performance prediction.")
    
    # Route to appropriate page
    if page == "🏠 Home":
        home_page()
    elif page == "📊 Model Comparison":
        model_comparison_page()
    elif page == "🎯 Make Predictions":
        prediction_page()
    elif page == "🔍 Feature Importance":
        feature_importance_page()
    elif page == "ℹ️ About":
        about_page()

if __name__ == "__main__":
    main()
