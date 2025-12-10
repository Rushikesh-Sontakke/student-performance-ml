# 🎓 Student Performance Prediction - ML Ensemble Project

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12-orange.svg)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An advanced machine learning system that predicts student exam scores using an **intelligent ensemble of 4 ML algorithms** with **SHAP explainability** and an **interactive Streamlit dashboard**.

## 🌟 Key Differentiators

This project stands out through:

- **🤖 Multi-Model Ensemble**: Combines Linear Regression, Random Forest, XGBoost, and Neural Network with **mathematically optimized** weights
- **🔬 Optimization-Based Weighting**: Uses `scipy.optimize` to find exact weights that minimize validation error (not heuristics!)
- **🔍 Explainable AI**: SHAP (SHapley Additive exPlanations) for model interpretability
- **📊 Interactive Dashboard**: Beautiful Streamlit web app with **human-readable inputs** for live predictions
- **🎯 Raw Input Processing**: Users enter values like "High" and "Male" instead of encoded numbers
- **📈 Comprehensive Analysis**: Deep performance evaluation with multiple metrics
- **💡 Student Profile Analysis**: Automated identification of strong points and improvement areas
- **🎯 Production-Ready**: Modular code structure with model persistence and configuration

---

## 📊 Model Performance

| Model | RMSE ↓ | MAE ↓ | R² ↑ | MAPE ↓ |
|-------|--------|-------|------|--------|
| **Weighted Ensemble** | TBD | TBD | TBD | TBD |
| XGBoost | TBD | TBD | TBD | TBD |
| Random Forest | TBD | TBD | TBD | TBD |
| Neural Network | TBD | TBD | TBD | TBD |
| Linear Regression | TBD | TBD | TBD | TBD |

*Run `python ensemble_model.py` to populate these metrics*

---

## 🚀 Quick Start

### 1. Installation

**Option A: Using Conda (Recommended)**
```bash
# Create environment from file
conda env create -f environment.yml

# Activate environment
conda activate student_perf
```

**Option B: Using pip**
```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Mac/Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Train Models

```bash
# Train all 4 models and create optimized ensemble
python ensemble_model.py
```

This will:
- Train Linear Regression, Random Forest, XGBoost, and Neural Network
- Use **scipy.optimize** to find mathematically optimal ensemble weights
- Guarantee ensemble performs at least as well as the best individual model
- Save all models to `models/` directory
- Generate performance comparison visualizations in `results/`

### 3. Generate Explainability Analysis

```bash
# Create SHAP feature importance visualizations
python model_explainer.py
```

This generates:
- Feature importance rankings for each model
- SHAP summary plots showing feature impacts
- Waterfall plots for individual predictions

### 4. Launch Interactive Dashboard

```bash
# Start Streamlit web application
streamlit run streamlit_app.py
```

Then open your browser to `http://localhost:8501`

---

## 📁 Project Structure

```
student-performance-ml/
│
├── Data/
│   ├── Raw/                          # Original dataset
│   └── Processed/                    # Train/test splits
│       ├── X_train.csv
│       ├── X_test.csv
│       ├── y_train.csv
│       └── y_test.csv
│
├── models/                           # Saved trained models
│   ├── linear_regression.joblib
│   ├── random_forest.joblib
│   ├── xgboost.joblib
│   ├── neural_network.keras
│   └── ensemble_config.json          # Ensemble weights
│
├── results/                          # Outputs and visualizations
│   ├── ensemble_performance.png
│   ├── shap_feature_importance_comparison.png
│   └── model_comparison_results.png
│
├── docs/                             # Additional documentation
│   ├── QUICKSTART.md                 # Step-by-step setup guide
│   ├── OPTIMIZATION_GUIDE.md         # Ensemble optimization explained
│   └── RAW_INPUT_GUIDE.md            # Input preprocessing details
│
├── Notebooks/                        # Jupyter notebooks for exploration
│   ├── preprocessing.ipynb.ipynb
│   └── decision_tree_carrin.ipynb
│
├── src/                              # Source code modules
│   ├── data_prep.py
│   ├── neuralnetworkmethod.py
│   ├── Random_Forest/
│   └── xgboost/
│
├── ensemble_model.py                 # Main ensemble training script (with optimization)
├── model_explainer.py                # SHAP explainability module
├── model_comparison.py               # Comprehensive model comparison
├── streamlit_app.py                  # Interactive web dashboard
├── preprocessor.py                   # Raw input preprocessing for dashboard
├── utils.py                          # Shared utility functions
│
├── requirements.txt                  # Python dependencies
├── environment.yml                   # Conda environment
└── README.md                         # This file
```

---

## 📖 Additional Documentation

For more detailed information, check out these guides:

- **[Quick Start Guide](docs/QUICKSTART.md)** - Step-by-step setup and troubleshooting
- **[Optimization Guide](docs/OPTIMIZATION_GUIDE.md)** - Deep dive into ensemble weighting mathematics
- **[Raw Input Guide](docs/RAW_INPUT_GUIDE.md)** - How preprocessing transforms user inputs

---

## 🎯 Features

### Machine Learning Models

1. **Linear Regression**
   - Fast baseline model
   - Interpretable coefficients
   - Shows linear relationships

2. **Random Forest**
   - 300 decision trees
   - Handles non-linearity well
   - Built-in feature importance

3. **XGBoost**
   - Gradient boosting algorithm
   - State-of-the-art tabular data performance
   - Optimized hyperparameters

4. **Neural Network**
   - 4 hidden layers (256→128→64→32)
   - Batch normalization & dropout
   - Early stopping for optimal training

5. **Optimized Ensemble**
   - Combines all 4 models intelligently
   - **Mathematical optimization** using scipy.optimize.minimize
   - Finds exact weights that minimize validation error
   - **Guaranteed** to perform at least as well as best single model
   - Uses SLSQP algorithm (Sequential Least Squares Programming)

### Explainability (SHAP)

- **Feature Importance**: Identifies which factors most influence predictions
- **Summary Plots**: Visualizes feature impacts across all predictions
- **Waterfall Plots**: Explains individual student predictions
- **Model Comparison**: Shows how different models prioritize features differently

### Interactive Dashboard

- **🏠 Home**: Project overview and quick stats
- **📊 Model Comparison**: Performance metrics and visualizations
- **🎯 Make Predictions**: Interactive form with **human-readable inputs** (dropdowns for categories, numbers for metrics)
  - Automatic preprocessing from raw values ("High", "Male") to model format
  - Student profile analysis showing strong points and improvement areas
  - Live predictions from all 5 models simultaneously
- **🔍 Feature Importance**: SHAP visualizations and analysis
- **ℹ️ About**: Detailed methodology and documentation

---

## 📚 Dataset

- **Size**: 6,607 student records
- **Features**: 19 predictors including:
  - Academic: Hours studied, attendance, previous scores
  - Socio-economic: Family income, parental education, distance from home
  - Support: Tutoring sessions, teacher quality, internet access
  - Personal: Sleep hours, physical activity, learning disabilities
- **Target**: Exam score (0-100)

---

## 🔬 Methodology

### Ensemble Strategy - Optimization-Based Weighting

Instead of using heuristic formulas, we use **mathematical optimization** to find the best ensemble weights:

```
minimize:  RMSE(Ensemble Predictions)
subject to: Σ weights = 1
           0 ≤ weight_i ≤ 1
```

The optimization process:
1. Train all 4 models on training data
2. Generate predictions on validation set
3. Use `scipy.optimize.minimize` with SLSQP algorithm
4. Find exact weights that minimize validation RMSE
5. **Result**: Ensemble is guaranteed ≥ best single model

**Example Output:**
```
Optimized Ensemble Weights:
  Linear Regression    | Weight: 0.85
  Random Forest        | Weight: 0.08
  XGBoost              | Weight: 0.05
  Neural Network       | Weight: 0.02

Ensemble RMSE: 3.21 (vs Best Single: 3.25)
```

**Why This Works:**
- If one model is significantly better, it gets most of the weight
- If models are complementary, weights are balanced
- Mathematically optimal - not a guess!

### Why Ensemble?

- **Mathematically Guaranteed**: Optimization ensures ensemble ≥ best model
- **Reduced Overfitting**: Different models make different errors
- **Improved Accuracy**: Captures complementary model strengths
- **Robustness**: Less sensitive to outliers and anomalies
- **Best Practice**: Used in winning solutions across ML competitions

---

## 📊 Usage Examples

### Making Predictions Programmatically

```python
from ensemble_model import StudentPerformanceEnsemble
import pandas as pd

# Load trained ensemble
ensemble = StudentPerformanceEnsemble()
ensemble.load_models()

# Prepare input (example values)
student_data = pd.DataFrame({
    'Hours_Studied': [25],
    'Attendance': [90],
    'Parental_Involvement': [1],  # Encoded value
    # ... other features
})

# Get predictions from all models
predictions = ensemble.predict_all(student_data)

print(f"Ensemble Prediction: {predictions['Ensemble'][0]:.2f}")
print(f"XGBoost Prediction: {predictions['XGBoost'][0]:.2f}")
```

### Generating SHAP Explanations

```python
from model_explainer import ModelExplainer

# Create explainer
explainer = ModelExplainer(ensemble, X_train, X_test, feature_names)
explainer.create_explainers()
explainer.calculate_shap_values()

# Get feature importance
importance = explainer.get_feature_importance('XGBoost')
print(importance.head(10))

# Explain single prediction
explanation = explainer.explain_single_prediction('XGBoost', instance_idx=0)
```

---

## 🎓 Educational Value

This project demonstrates:

- **Ensemble Learning**: Combining multiple models for better performance
- **Hyperparameter Tuning**: Optimizing model configurations
- **Model Explainability**: Using SHAP for interpretability
- **Deep Learning**: Building and training neural networks
- **Web Development**: Creating interactive ML dashboards
- **Software Engineering**: Modular, maintainable code structure
- **Data Science Workflow**: End-to-end ML pipeline

---

## 🔧 Advanced Configuration

### Custom Ensemble Weights

Edit `ensemble_model.py` to manually set weights:

```python
self.weights = {
    'Linear Regression': 0.10,
    'Random Forest': 0.30,
    'XGBoost': 0.40,
    'Neural Network': 0.20
}
```

### Hyperparameter Tuning

Modify model configurations in `ensemble_model.py`:

```python
def build_xgboost(self):
    return xgb.XGBRegressor(
        n_estimators=500,      # Increase trees
        max_depth=8,           # Deeper trees
        learning_rate=0.05,    # Slower learning
        # ... other parameters
    )
```

---

## 📈 Results & Insights

### Top Influential Features

Based on SHAP analysis, the most important factors are typically:

1. **Hours Studied** - Direct correlation with performance
2. **Attendance** - Consistent class participation matters
3. **Previous Scores** - Historical performance is predictive
4. **Tutoring Sessions** - Additional support helps
5. **Sleep Hours** - Rest impacts cognitive function

### Model Insights

- **XGBoost** typically performs best on this tabular data
- **Neural Network** captures complex non-linear patterns
- **Random Forest** provides robust predictions
- **Ensemble** combines their strengths for optimal results

---

## 🤝 Contributing

This is an academic project, but suggestions are welcome:

1. Fork the repository
2. Create a feature branch
3. Make your improvements
4. Submit a pull request

---

## 📄 License

This project is created for educational purposes as part of an ML course final project.

---

## 🙏 Acknowledgments

- Dataset: Student Performance Factors
- SHAP Library: For explainability
- Streamlit: For rapid dashboard development
- scikit-learn, XGBoost, TensorFlow: ML frameworks

---

## 📧 Contact

For questions about this project, please contact through Email: sontakkerushikesh547@gmail.com

---

**⭐ If you found this project helpful, please star it!**
