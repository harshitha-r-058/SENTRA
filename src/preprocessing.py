import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

def clean_dataset(df):
    """
    Clean dataset and rename columns for consistency.
    """
    df = df.copy()
    
    # Rename columns to match requested feature names
    rename_mapping = {
        'flow intensity': 'flow_intensity',
        'Cervical mucus code': 'Cervical_mucus_code'
    }
    df = df.rename(columns=rename_mapping)
    
    # Remove exact duplicate rows
    df = df.drop_duplicates()
    
    # Clean phase target values
    if 'phase' in df.columns:
        df['phase'] = df['phase'].astype(str).str.strip().str.capitalize()
        
    # Validations to replace impossible values with NaN
    if 'age' in df.columns:
        df.loc[df['age'] < 0, 'age'] = np.nan
        
    if 'bleeding_days' in df.columns:
        df.loc[df['bleeding_days'] < 0, 'bleeding_days'] = np.nan
        
    if 'BMI' in df.columns:
        df.loc[df['BMI'] < 0, 'BMI'] = np.nan
        
    if 'inter_cycle_length' in df.columns:
        df.loc[df['inter_cycle_length'] < 0, 'inter_cycle_length'] = np.nan

    return df

def get_candidate_features():
    """
    Return the list of allowed candidate features.
    """
    return [
        'age', 'bleeding_days', 'lh', 'estrogen', 'flow_intensity', 'flow_color', 
        'appetite', 'exerciselevel', 'headaches', 'cramps', 'sorebreasts', 'fatigue', 
        'sleepissue', 'moodswing', 'stress', 'foodcravings', 'indigestion', 'bloating', 
        'nightly_temperature', 'BMI', 'inter_cycle_length', 'cycle_variability', 'BBT', 
        'Cervical_mucus_code', 'LH_result', 'pain_nrs', 'mood', 'sleep_hours', 
        'activity_level', 'day_from_last_menses', 'lag1_BBT', 'lag2_BBT', 
        'BBT_mean_7d', 'BBT_std_7d'
    ]

def get_leakage_features():
    """
    Return the list of features that MUST be excluded.
    """
    return [
        'phase', 'phase.1', 'advice_text', 'advice_type', 'advice_source', 
        'advice_confidence', 'validator_flag', 'synthetic_augmented_flag', 'id'
    ]

def select_features(df):
    """
    Select features that exist in the dataframe and are in candidate features,
    excluding leakage features.
    """
    candidate_features = get_candidate_features()
    leakage_features = get_leakage_features()
    
    selected_features = []
    for col in df.columns:
        if col in candidate_features and col not in leakage_features:
            selected_features.append(col)
            
    return selected_features

def build_preprocessing_pipeline(df, selected_features):
    """
    Build scikit-learn preprocessing pipeline.
    """
    numerical_features = df[selected_features].select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = df[selected_features].select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median'))
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'
    )
    
    return preprocessor, numerical_features, categorical_features
