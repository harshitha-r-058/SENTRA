import pandas as pd
import numpy as np
from src.utils import load_model, load_json

model = None
label_mapping = None
inverse_label_mapping = None
selected_features = None

def load_resources():
    global model, label_mapping, inverse_label_mapping, selected_features
    if model is None:
        model = load_model('model/menstrual_phase_model.pkl')
        label_mapping = load_json('model/label_mapping.json')
        inverse_label_mapping = {v: k for k, v in label_mapping.items()}
        selected_features = load_json('model/feature_columns.json')

def predict_cycle_phase(user_data: dict) -> dict:
    load_resources()
    
    # Convert input to DataFrame
    df = pd.DataFrame([user_data])
    
    # Ensure all selected features exist in the df, even if missing
    for col in selected_features:
        if col not in df.columns:
            df[col] = np.nan
            
    # Filter only selected features
    X = df[selected_features]
    
    # Predict probabilities
    probas = model.predict_proba(X)[0]
    
    # Find max probability class
    pred_idx = np.argmax(probas)
    pred_class = inverse_label_mapping[pred_idx]
    confidence = float(probas[pred_idx])
    
    probabilities_dict = {
        inverse_label_mapping[i]: float(p) for i, p in enumerate(probas)
    }
    
    return {
        "predicted_phase": pred_class,
        "confidence": confidence,
        "probabilities": probabilities_dict
    }
