# Quick Start Guide

Get your Student Performance ML Ensemble up and running in minutes.

## Prerequisites

- Python 3.10+
- Conda (recommended) or pip
- 2GB free disk space

---

## Step 1: Clone and Setup Environment

### Option A: Using Conda (Recommended)

```bash
# Navigate to project directory
cd student-performance-ml-main

# Create conda environment
conda env create -f environment.yml

# Activate environment
conda activate student_perf
```

### Option B: Using pip

```bash
# Navigate to project directory
cd student-performance-ml-main

# Create virtual environment
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on Mac/Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Step 2: Train Models (5-10 minutes)

```bash
python ensemble_model.py
```

**What this does:**
- ✅ Trains Linear Regression, Random Forest, XGBoost, Neural Network
- ✅ Uses scipy.optimize to find optimal ensemble weights
- ✅ Saves all models to `models/` directory
- ✅ Generates performance comparison charts in `results/`
- ✅ Outputs performance metrics to console

**Expected Output:**
```
==================================================================
TRAINING ENSEMBLE MODEL
==================================================================
[1/4] Training Linear Regression...
✓ Linear Regression trained
...
🔬 Using scipy.optimize to find mathematically optimal weights...
✓ Ensemble model training complete!
```

---

## Step 3: Generate Explainability Analysis (2-3 minutes)

```bash
python model_explainer.py
```

**What this does:**
- ✅ Creates SHAP explainers for all models
- ✅ Calculates feature importance
- ✅ Generates visualizations (summary plots, waterfall plots)
- ✅ Saves results to `results/` directory

**Expected Output:**
```
==================================================================
CREATING SHAP EXPLAINERS
==================================================================
[1/4] Creating explainer for Linear Regression...
...
✓ SHAP ANALYSIS COMPLETE
```

---

## Step 4: Launch Interactive Dashboard

```bash
streamlit run streamlit_app.py
```

**What this does:**
- 🌐 Starts web server on `http://localhost:8501`
- 📊 Opens interactive dashboard in your browser
- 🎯 Enables live predictions with human-readable inputs

**Dashboard Features:**
1. **Home** - Project overview
2. **Model Comparison** - Performance metrics and charts
3. **Make Predictions** - Interactive form to test models
4. **Feature Importance** - SHAP visualizations
5. **About** - Methodology documentation

---

## Step 5: (Optional) Run Full Comparison

```bash
python model_comparison.py
```

Generates additional comparison metrics and visualizations.

---

## Generated Outputs

After running the above steps, you'll have:

```
models/
├── linear_regression.joblib       # Trained Linear Regression
├── random_forest.joblib           # Trained Random Forest  
├── xgboost.joblib                 # Trained XGBoost
├── neural_network.keras           # Trained Neural Network
└── ensemble_config.json           # Ensemble weights

results/
├── ensemble_performance.png                    # Performance comparison
├── ensemble_comparison.csv                     # Metrics table
├── shap_feature_importance_comparison.png      # Feature importance
├── shap_summary_*.png                         # SHAP summary plots
└── feature_importance_*.csv                   # Feature rankings
```

---

## Expected Performance

Typical metrics on the student performance dataset:

- **RMSE**: 2-5 (lower is better)
- **MAE**: 1-3 (lower is better)
- **R²**: 0.80-0.95 (higher is better)
- **MAPE**: 5-15% (lower is better)

Exact values depend on data preprocessing and random seed.

---

## Troubleshooting

### Issue: Python not found
**Solution:** Ensure Python 3.10+ is installed and added to PATH

### Issue: Module not found errors
**Solution:** 
```bash
# Reinstall dependencies
pip install -r requirements.txt
```

### Issue: CUDA/GPU errors in TensorFlow
**Solution:** TensorFlow will automatically use CPU. This is fine for this dataset.

### Issue: Streamlit won't start
**Solution:**
```bash
# Make sure streamlit is installed
pip install streamlit

# Try running on different port
streamlit run streamlit_app.py --server.port 8502
```

### Issue: Model files not found in dashboard
**Solution:** Run `python ensemble_model.py` first to train and save models

---

## Next Steps

After setup:

1. ✅ Test the dashboard thoroughly
2. ✅ Experiment with different student profiles
3. ✅ Review SHAP visualizations to understand feature importance
4. ✅ Explore the code to understand the implementation

---

## Additional Resources

- [Optimization Guide](OPTIMIZATION_GUIDE.md) - Deep dive into ensemble weighting
- [Raw Input Guide](RAW_INPUT_GUIDE.md) - How preprocessing works
- [README](../README.md) - Project overview

---

**🎓 You're all set! Your ML ensemble is ready to predict student performance.**
