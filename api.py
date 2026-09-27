from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, create_model
from typing import Optional, Dict, Any
import uvicorn
from src.predict import predict_cycle_phase, load_resources
from src.utils import load_json
import os

app = FastAPI(title="Menstrual Cycle Phase Prediction API")

# Define the Pydantic model for input data based on candidate features
class CycleData(BaseModel):
    age: Optional[float] = None
    bleeding_days: Optional[float] = None
    lh: Optional[float] = None
    estrogen: Optional[float] = None
    flow_intensity: Optional[str] = None
    flow_color: Optional[str] = None
    appetite: Optional[str] = None
    exerciselevel: Optional[str] = None
    headaches: Optional[str] = None
    cramps: Optional[str] = None
    sorebreasts: Optional[str] = None
    fatigue: Optional[str] = None
    sleepissue: Optional[str] = None
    moodswing: Optional[str] = None
    stress: Optional[str] = None
    foodcravings: Optional[str] = None
    indigestion: Optional[str] = None
    bloating: Optional[str] = None
    nightly_temperature: Optional[float] = None
    BMI: Optional[float] = None
    inter_cycle_length: Optional[float] = None
    cycle_variability: Optional[float] = None
    BBT: Optional[float] = None
    Cervical_mucus_code: Optional[str] = None
    LH_result: Optional[str] = None
    pain_nrs: Optional[float] = None
    mood: Optional[str] = None
    sleep_hours: Optional[float] = None
    activity_level: Optional[str] = None
    day_from_last_menses: Optional[float] = None
    lag1_BBT: Optional[float] = None
    lag2_BBT: Optional[float] = None
    BBT_mean_7d: Optional[float] = None
    BBT_std_7d: Optional[float] = None

@app.on_event("startup")
def startup_event():
    try:
        load_resources()
    except Exception as e:
        print(f"Error loading model: {e}")

@app.get("/health")
def health_check():
    model_loaded = os.path.exists("model/menstrual_phase_model.pkl")
    return {
        "status": "healthy",
        "model_loaded": model_loaded
    }

@app.get("/model-info")
def model_info():
    try:
        metadata = load_json("model/model_metadata.json")
        return {
            "model_name": metadata.get("model_name"),
            "model_version": metadata.get("model_version"),
            "target": metadata.get("target_column"),
            "supported_phases": metadata.get("target_classes"),
            "feature_count": len(metadata.get("selected_features", []))
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail="Model metadata not found")

@app.post("/predict-cycle-phase")
def predict(data: CycleData):
    try:
        # Convert Pydantic model to dictionary, excluding None values or keeping them as NaN?
        # Actually predict_cycle_phase will handle missing selected features by filling with NaN
        user_data = data.dict(exclude_unset=True)
        result = predict_cycle_phase(user_data)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
