"""
Preprocessing Wrapper for Raw Input Transformation
Converts human-readable raw inputs to model-ready preprocessed format
"""

import pandas as pd
import numpy as np
import joblib
import os

# Ordinal mappings (from data_prep.py)
ORDINAL_MAPPINGS = {
    'Parental_Involvement': {'None': 0, 'Low': 1, 'Medium': 2, 'High': 3},
    'Access_to_Resources': {'Low': 0, 'Medium': 1, 'High': 2},
    'Motivation_Level': {'Low': 0, 'Medium': 1, 'High': 2},
    'Teacher_Quality': {'Low': 0, 'Medium': 1, 'High': 2},
    'Parental_Education_Level': {'High School': 0, 'College': 1, 'Postgraduate': 2},
    'Family_Income': {'Low': 0, 'Medium': 1, 'High': 2},
    'Distance_from_Home': {'Near': 0, 'Moderate': 1, 'Far': 2}
}

# Feature configuration
RAW_INPUT_CONFIG = {
    # Numeric features (direct input)
    'Hours_Studied': {'type': 'numeric', 'min': 0, 'max': 44, 'default': 20},
    'Attendance': {'type': 'numeric', 'min': 60, 'max': 100, 'default': 85},
    'Sleep_Hours': {'type': 'numeric', 'min': 4, 'max': 10, 'default': 7},
    'Previous_Scores': {'type': 'numeric', 'min': 50, 'max': 100, 'default': 75},
    'Tutoring_Sessions': {'type': 'numeric', 'min': 0, 'max': 8, 'default': 2},
    'Physical_Activity': {'type': 'numeric', 'min': 0, 'max': 6, 'default': 3},
    
    # Ordinal features (dropdown with ordering)
    'Parental_Involvement': {'type': 'ordinal', 'options': ['None', 'Low', 'Medium', 'High']},
    'Access_to_Resources': {'type': 'ordinal', 'options': ['Low', 'Medium', 'High']},
    'Motivation_Level': {'type': 'ordinal', 'options': ['Low', 'Medium', 'High']},
    'Teacher_Quality': {'type': 'ordinal', 'options': ['Low', 'Medium', 'High']},
    'Parental_Education_Level': {'type': 'ordinal', 'options': ['High School', 'College', 'Postgraduate']},
    'Family_Income': {'type': 'ordinal', 'options': ['Low', 'Medium', 'High']},
    'Distance_from_Home': {'type': 'ordinal', 'options': ['Near', 'Moderate', 'Far']},
    
    # Categorical features (one-hot encoded)
    'Gender': {'type': 'categorical', 'options': ['Male', 'Female']},
    'School_Type': {'type': 'categorical', 'options': ['Public', 'Private']},
    'Extracurricular_Activities': {'type': 'categorical', 'options': ['Yes', 'No']},
    'Internet_Access': {'type': 'categorical', 'options': ['Yes', 'No']},
    'Peer_Influence': {'type': 'categorical', 'options': ['Positive', 'Neutral', 'Negative']},
    'Learning_Disabilities': {'type': 'categorical', 'options': ['Yes', 'No']}
}

class RawInputPreprocessor:
    """
    Handles conversion from raw user inputs to preprocessed model inputs
    """
    
    def __init__(self, preprocessor_path='Data/Processed/preprocessor.joblib'):
        """
        Args:
            preprocessor_path: Path to saved sklearn ColumnTransformer
        """
        self.preprocessor_path = preprocessor_path
        self.preprocessor = None
        
        if os.path.exists(preprocessor_path):
            self.preprocessor = joblib.load(preprocessor_path)
            print(f"✓ Loaded preprocessor from {preprocessor_path}")
        else:
            print(f"⚠ Preprocessor not found at {preprocessor_path}")
            print("  Will use manual transformation")
    
    def create_raw_dataframe(self, raw_input_dict):
        """
        Convert raw input dictionary to pandas DataFrame
        
        Args:
            raw_input_dict: Dictionary with raw feature values
            
        Returns:
            DataFrame with single row of raw inputs
        """
        return pd.DataFrame([raw_input_dict])
    
    def apply_ordinal_encoding(self, df):
        """Apply ordinal mappings to categorical features"""
        df_copy = df.copy()
        for col, mapping in ORDINAL_MAPPINGS.items():
            if col in df_copy.columns:
                df_copy[col] = df_copy[col].map(mapping).astype(float)
        return df_copy
    
    def transform_raw_input(self, raw_input_dict):
        """
        Transform raw input dictionary to preprocessed format
        
        Args:
            raw_input_dict: Dictionary with feature names as keys and raw values
            
        Returns:
            numpy array ready for model prediction
        """
        # Create DataFrame from raw input
        df_raw = self.create_raw_dataframe(raw_input_dict)
        
        # Apply ordinal encoding
        df_encoded = self.apply_ordinal_encoding(df_raw)
        
        # Apply sklearn preprocessor (scaling + one-hot encoding)
        if self.preprocessor is not None:
            X_processed = self.preprocessor.transform(df_encoded)
        else:
            # Fallback: manual transformation
            X_processed = self._manual_transform(df_encoded)
        
        return X_processed
    
    def _manual_transform(self, df):
        """
        Manual transformation if preprocessor is not available
        This is a backup method - not recommended for production
        """
        # Just return the values as-is (not ideal but functional)
        return df.values
    
    def get_default_values(self):
        """
        Get default values for all features
        
        Returns:
            Dictionary with default values
        """
        defaults = {}
        for feature, config in RAW_INPUT_CONFIG.items():
            if config['type'] == 'numeric':
                defaults[feature] = config['default']
            elif config['type'] in ['ordinal', 'categorical']:
                defaults[feature] = config['options'][0]
        
        return defaults
    
    def validate_input(self, raw_input_dict):
        """
        Validate raw input dictionary
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        for feature in RAW_INPUT_CONFIG.keys():
            if feature not in raw_input_dict:
                return False, f"Missing required feature: {feature}"
        
        # Validate numeric ranges
        for feature, value in raw_input_dict.items():
            config = RAW_INPUT_CONFIG.get(feature)
            if not config:
                continue
                
            if config['type'] == 'numeric':
                if not isinstance(value, (int, float)):
                    return False, f"{feature} must be numeric"
                if value < config['min'] or value > config['max']:
                    return False, f"{feature} must be between {config['min']} and {config['max']}"
                    
            elif config['type'] in ['ordinal', 'categorical']:
                if value not in config['options']:
                    return False, f"{feature} must be one of {config['options']}"
        
        return True, "Valid"


def create_sample_raw_input():
    """Create a sample raw input for testing"""
    return {
        'Hours_Studied': 25,
        'Attendance': 90,
        'Parental_Involvement': 'High',
        'Access_to_Resources': 'Medium',
        'Extracurricular_Activities': 'Yes',
        'Sleep_Hours': 7,
        'Previous_Scores': 80,
        'Motivation_Level': 'High',
        'Internet_Access': 'Yes',
        'Tutoring_Sessions': 3,
        'Family_Income': 'Medium',
        'Teacher_Quality': 'High',
        'School_Type': 'Public',
        'Peer_Influence': 'Positive',
        'Physical_Activity': 4,
        'Learning_Disabilities': 'No',
        'Parental_Education_Level': 'College',
        'Distance_from_Home': 'Near',
        'Gender': 'Male'
    }


# Convenience function for easy import
def preprocess_raw_input(raw_input_dict, preprocessor_path='Data/Processed/preprocessor.joblib'):
    """
    Convenience function to transform raw input
    
    Args:
        raw_input_dict: Dictionary with raw feature values
        preprocessor_path: Path to saved preprocessor
        
    Returns:
        Preprocessed numpy array ready for prediction
    """
    processor = RawInputPreprocessor(preprocessor_path)
    return processor.transform_raw_input(raw_input_dict)


if __name__ == "__main__":
    # Test the preprocessor
    print("Testing Raw Input Preprocessor\n")
    
    # Create sample input
    sample_input = create_sample_raw_input()
    print("Sample Raw Input:")
    for key, value in sample_input.items():
        print(f"  {key}: {value}")
    
    # Transform
    processor = RawInputPreprocessor()
    
    # Validate
    is_valid, message = processor.validate_input(sample_input)
    print(f"\nValidation: {message}")
    
    if is_valid:
        # Transform
        X_processed = processor.transform_raw_input(sample_input)
        print(f"\nProcessed shape: {X_processed.shape}")
        print(f"Processed values (first 10): {X_processed[0][:10]}")
