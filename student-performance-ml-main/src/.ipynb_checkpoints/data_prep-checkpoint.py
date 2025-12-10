import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

RANDOM_STATE = 42

# Explicit mappings for ordinal features (consistent with analysis)
ORDINAL_MAPPINGS = {
    'Parental_Involvement': {'None':0, 'Low':1, 'Medium':2, 'High':3},
    'Access_to_Resources': {'Low':0, 'Medium':1, 'High':2},
    'Motivation_Level': {'Low':0, 'Medium':1, 'High':2},
    'Teacher_Quality': {'Low':0, 'Medium':1, 'High':2},
    'Parental_Education_Level': {'High School':0, 'College':1, 'Postgraduate':2},
    'Family_Income': {'Low':0, 'Medium':1, 'High':2},
    'Distance_from_Home': {'Near':0, 'Moderate':1, 'Far':2}
}

ONEHOT_COLS = [
    'Gender', 'School_Type', 'Extracurricular_Activities',
    'Internet_Access', 'Peer_Influence', 'Learning_Disabilities'
]

NUMERIC_COLS = [
    'Hours_Studied', 'Attendance', 'Sleep_Hours',
    'Previous_Scores', 'Tutoring_Sessions', 'Physical_Activity'
]

TARGET = 'Exam_Score'

def load_data(path="data/raw/StudentPerformanceFactors.csv"):
    return pd.read_csv(path)

def basic_cleaning(df):
    # Drop exact duplicates
    df = df.drop_duplicates().reset_index(drop=True)
    # Impute categorical missing with mode for the specific columns
    for col in ['Teacher_Quality', 'Parental_Education_Level', 'Distance_from_Home']:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0])
    return df

def apply_ordinals(df):
    for col, mapping in ORDINAL_MAPPINGS.items():
        if col in df.columns:
            df[col] = df[col].map(mapping).astype(float)
    return df

def build_preprocessor():
    # ColumnTransformer: numeric scaler, onehot for nominal, passthrough addresses ordinals already mapped
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    onehot = OneHotEncoder(handle_unknown='ignore', sparse=False, drop='first')
    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, NUMERIC_COLS),
        ('oh', onehot, ONEHOT_COLS)
    ], remainder='passthrough')  # passthrough keeps ordinal columns and any other columns (e.g., ordinals)
    return preprocessor

def preprocess_and_split(df, test_size=0.2, random_state=RANDOM_STATE):
    df = basic_cleaning(df)
    df = apply_ordinals(df)

    # Ensure target exists
    if TARGET not in df.columns:
        raise ValueError(f"Target column {TARGET} not found in dataframe")

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    preprocessor = build_preprocessor()
    # Fit scaler/onehot on training data only
    preprocessor.fit(X_train)

    # Transform and convert back to DataFrame if desired
    X_train_trans = preprocessor.transform(X_train)
    X_test_trans = preprocessor.transform(X_test)

    # Save preprocessor for reuse
    joblib.dump(preprocessor, "data/processed/preprocessor.joblib")

    # save numpy arrays and DataFrames
    np.save("data/processed/X_train.npy", X_train_trans)
    np.save("data/processed/X_test.npy", X_test_trans)
    y_train.to_csv("data/processed/y_train.csv", index=False)
    y_test.to_csv("data/processed/y_test.csv", index=False)

    return X_train_trans, X_test_trans, y_train, y_test
