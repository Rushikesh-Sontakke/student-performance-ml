# Raw Input Preprocessing Guide

How the Streamlit dashboard handles human-readable inputs and transforms them for model prediction.

## Overview

The dashboard accepts **human-readable raw inputs** (like "High", "Male", "Yes") and automatically preprocesses them into model-ready format. Users don't need to know about encoding schemes or standardization.

---

## Architecture

### Preprocessing Pipeline

```
Raw Input (User enters: "High", "Male", 25)
    ↓
Ordinal Encoding (High → 3, Male stays as-is)
    ↓
One-Hot Encoding (Male → [1, 0] or [0, 1])
    ↓
Standardization (Hours: 25 → (25 - μ) / σ)
    ↓
Model-Ready Array (numpy array)
```

### Components

1. **`preprocessor.py`** - Handles transformation logic
2. **`streamlit_app.py`** - User interface with input forms
3. **`Data/Processed/preprocessor.joblib`** - Saved sklearn transformer

---

## Feature Types

### Ordinal Features (Ordered Categories)

Features with natural ordering are encoded numerically:

| Feature | Mapping |
|---------|---------|
| Parental_Involvement | None=0, Low=1, Medium=2, High=3 |
| Access_to_Resources | Low=0, Medium=1, High=2 |
| Motivation_Level | Low=0, Medium=1, High=2 |
| Teacher_Quality | Low=0, Medium=1, High=2 |
| Parental_Education_Level | High School=0, College=1, Postgraduate=2 |
| Family_Income | Low=0, Medium=1, High=2 |
| Distance_from_Home | Near=0, Moderate=1, Far=2 |

### Categorical Features (Nominal, One-Hot Encoded)

Features without ordering are one-hot encoded:

- **Gender**: Male, Female
- **School_Type**: Public, Private
- **Extracurricular_Activities**: Yes, No
- **Internet_Access**: Yes, No
- **Peer_Influence**: Positive, Neutral, Negative
- **Learning_Disabilities**: Yes, No

### Numeric Features (Standardized)

Continuous variables are standardized (zero mean, unit variance):

| Feature | Range | Standardization |
|---------|-------|-----------------|
| Hours_Studied | 0-44 hours/week | (x - μ) / σ |
| Attendance | 60-100% | (x - μ) / σ |
| Sleep_Hours | 4-10 hours/night | (x - μ) / σ |
| Previous_Scores | 50-100 | (x - μ) / σ |
| Tutoring_Sessions | 0-8 per month | (x - μ) / σ |
| Physical_Activity | 0-6 hours/week | (x - μ) / σ |

---

## Implementation

### RawInputPreprocessor Class

```python
class RawInputPreprocessor:
    """Handles conversion from raw inputs to preprocessed format"""
    
    def __init__(self, preprocessor_path='Data/Processed/preprocessor.joblib'):
        self.preprocessor = joblib.load(preprocessor_path)
    
    def transform_raw_input(self, raw_input_dict):
        """
        Transform raw input dictionary to model format
        
        Args:
            raw_input_dict: Dict with feature names and raw values
            
        Returns:
            numpy array ready for prediction
        """
        # Create DataFrame
        df = pd.DataFrame([raw_input_dict])
        
        # Apply ordinal encoding
        df = self.apply_ordinal_encoding(df)
        
        # Apply sklearn preprocessing (standardization + one-hot)
        X_processed = self.preprocessor.transform(df)
        
        return X_processed
```

### Input Validation

```python
def validate_input(self, raw_input_dict):
    """Validate raw input dictionary"""
    
    # Check all features present
    for feature in REQUIRED_FEATURES:
        if feature not in raw_input_dict:
            return False, f"Missing: {feature}"
    
    # Validate numeric ranges
    if raw_input_dict['Hours_Studied'] > 44:
        return False, "Hours_Studied max is 44"
    
    # Validate categorical options
    if raw_input_dict['Gender'] not in ['Male', 'Female']:
        return False, "Gender must be Male or Female"
    
    return True, "Valid"
```

---

## Dashboard Integration

### Input Form Structure

The Streamlit dashboard organizes inputs into logical sections:

1. **📚 Academic Information**
   - Hours_Studied, Attendance, Previous_Scores

2. **🎓 Support & Resources**
   - Parental_Involvement, Access_to_Resources, Tutoring_Sessions

3. **🏫 School Environment**
   - Teacher_Quality, School_Type, Peer_Influence

4. **👤 Personal Factors**
   - Sleep_Hours, Physical_Activity, Motivation_Level

5. **📊 Demographics**
   - Gender, Family_Income, Parental_Education_Level

6. **Additional Factors**
   - Distance_from_Home, Extracurricular_Activities, Internet_Access, Learning_Disabilities

### Code Flow

```python
# In streamlit_app.py
from preprocessor import RawInputPreprocessor

# User fills form
input_values = {
    'Hours_Studied': 25,
    'Gender': 'Male',
    'Parental_Involvement': 'High',
    # ... other features
}

# Validate
processor = RawInputPreprocessor()
is_valid, message = processor.validate_input(input_values)

if is_valid:
    # Transform
    X_processed = processor.transform_raw_input(input_values)
    
    # Predict
    predictions = ensemble.predict_all(X_processed)
```

---

## Example Usage

### Programmatic Usage

```python
from preprocessor import RawInputPreprocessor, create_sample_raw_input

# Create processor
processor = RawInputPreprocessor()

# Raw input (human-readable!)
raw_input = {
    'Hours_Studied': 25,
    'Attendance': 90,
    'Parental_Involvement': 'High',    # Not encoded!
    'Gender': 'Male',
    'School_Type': 'Public',
    # ... other features
}

# Validate
is_valid, message = processor.validate_input(raw_input)

if is_valid:
    # Transform
    X_processed = processor.transform_raw_input(raw_input)
    
    # X_processed is now ready for model.predict()
    predictions = ensemble.predict_all(X_processed)
```

### Dashboard Usage

1. User navigates to "Make Predictions" page
2. Fills form with dropdowns and number inputs
3. Clicks "Predict Exam Score"
4. System validates → preprocesses → predicts
5. Results displayed with analysis

---

## Benefits

| Benefit | Description |
|---------|-------------|
| **User-Friendly** | No encoding knowledge needed |
| **Type-Safe** | Dropdowns prevent invalid inputs |
| **Validated** | Automatic range/option checking |
| **Transparent** | Can view raw inputs before processing |
| **Professional** | Production-quality UX |

---

## Preprocessing Consistency

### Training vs Prediction

**Critical:** The same `preprocessor.joblib` must be used for both:
- Fitting during training (in `src/data_prep.py`)
- Transforming during prediction (in `preprocessor.py`)

This ensures:
- Same feature order
- Same encoding mappings
- Same standardization parameters (μ, σ)

### Regenerating Preprocessor

If you retrain models on new data:

```bash
# Regenerate preprocessor
python src/data_prep.py

# Retrain models
python ensemble_model.py
```

---

## Troubleshooting

### Issue: "Error loading preprocessor"
**Solution:** Ensure `Data/Processed/preprocessor.joblib` exists
```bash
python src/data_prep.py  # Regenerate if missing
```

### Issue: "Invalid input" error
**Solution:** Check all required fields are filled and within valid ranges

### Issue: Predictions seem incorrect
**Solution:** Verify preprocessor matches training data
- Check preprocessing order
- Ensure same feature names
- Verify encoding mappings

---

## Configuration

All input configurations are defined in `preprocessor.py`:

```python
RAW_INPUT_CONFIG = {
    'Hours_Studied': {
        'type': 'numeric',
        'min': 0,
        'max': 44,
        'default': 20
    },
    'Parental_Involvement': {
        'type': 'ordinal',
        'options': ['None', 'Low', 'Medium', 'High']
    },
    # ... etc
}
```

Modify this dict to change validation rules or defaults.

---

## Further Reading

- [scikit-learn Preprocessing](https://scikit-learn.org/stable/modules/preprocessing.html)
- [Streamlit Input Widgets](https://docs.streamlit.io/library/api-reference/widgets)
- [Feature Engineering Best Practices](https://developers.google.com/machine-learning/data-prep/transform/introduction)

---

**This preprocessing system provides a professional, user-friendly interface while maintaining technical rigor.**
