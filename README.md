# Menstrual Cycle Phase Prediction System

This project is the Machine Learning module for the Digital Health Twin, responsible for estimating a user's current menstrual cycle phase from tracked physiological and lifestyle data. 

## PROJECT
Menstrual Cycle Phase Prediction System

## OBJECTIVE
Build a machine-learning system that estimates the user's current menstrual-cycle phase. The system classifies the user's phase into one of four phases:
1. Menstrual
2. Follicular
3. Fertility
4. Luteal

## DATASET
The main dataset is `cleaned_dataset.csv`. It contains records from various users, with physiological markers, self-reported symptoms, and cycle tracking information.

## TARGET
The target column is `phase`.

## FEATURES
The system selects relevant physiological and tracking features available prior to prediction, such as:
`age`, `bleeding_days`, `lh`, `estrogen`, `flow_intensity`, `flow_color`, `appetite`, `exerciselevel`, `headaches`, `cramps`, `sorebreasts`, `fatigue`, `sleepissue`, `moodswing`, `stress`, `foodcravings`, `indigestion`, `bloating`, `nightly_temperature`, `BMI`, `inter_cycle_length`, `cycle_variability`, `BBT`, `Cervical_mucus_code`, `LH_result`, `pain_nrs`, `mood`, `sleep_hours`, `activity_level`, `day_from_last_menses`, `lag1_BBT`, `lag2_BBT`, `BBT_mean_7d`, `BBT_std_7d`.

## EXCLUDED FEATURES
Features that introduce target leakage or are unavailable at prediction time were excluded:
`phase`, `phase.1`, `advice_text`, `advice_type`, `advice_source`, `advice_confidence`, `validator_flag`, `synthetic_augmented_flag`, and `id`. 

## DATA PREPROCESSING
Data is cleaned to handle missing values and impossible values. Categorical features use most-frequent imputation and one-hot encoding. Numerical features use median imputation. Preprocessing is implemented using scikit-learn `Pipeline` and is strictly fitted only on the training data.

## DATA LEAKAGE PREVENTION
Multiple steps ensure no data leakage occurs:
- Exclusion of post-prediction features and duplicated targets.
- The `id` column is excluded from the input features to avoid memorization.
- Group-aware splitting prevents records of the same user from appearing in both train and test sets.

## GROUP-AWARE TRAIN/TEST SPLIT
The dataset is split into an 80% development set and a 20% completely unseen test set using `StratifiedGroupKFold`. This ensures all records for a specific user ID fall exclusively into either the train or test set.

## RANDOM FOREST
The primary model is a `RandomForestClassifier` configured with balanced class weights.

## XGBOOST
An optional `XGBoost Classifier` is also trained and compared against Random Forest using the same cross-validation folds.

## CROSS VALIDATION
A 5-fold group-aware cross-validation (`StratifiedGroupKFold`) on the development set is used to compare models robustly without touching the final test set.

## EVALUATION
Models are compared based on Accuracy and Macro F1 score. The best model is chosen and evaluated on the completely unseen test groups to generate the final classification report and confusion matrix.

## MODEL SAVING
The complete pipeline (preprocessing + model) is saved as `menstrual_phase_model.pkl` using joblib. Feature lists, label encodings, and model metadata are also saved.

## PREDICTION
The prediction script uses the saved pipeline to accept raw JSON/dict user data, apply the appropriate transformations, and output the predicted phase with probabilities using `predict_proba()`.

## FASTAPI
A FastAPI application serves the trained model via HTTP endpoints.

## API ENDPOINTS
- `GET /health` : Returns system health and model loading status.
- `GET /model-info` : Returns model version, feature count, and target classes.
- `POST /predict-cycle-phase` : Accepts tracking data and returns the estimated phase along with confidence probabilities.

## FRONTEND INTEGRATION
JavaScript Example:
```javascript
const userData = {
    "age": 22,
    "bleeding_days": 5,
    "sleep_hours": 7,
    "activity_level": "Moderate",
    "day_from_last_menses": 8,
    "BBT": 36.45
};

const response = await fetch(
    "http://localhost:8000/predict-cycle-phase",
    {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify(userData)
    }
);

const result = await response.json();
console.log(result);
```

## INSTALLATION
Create environment:
```cmd
python -m venv venv
```

Activate:
```cmd
venv\Scripts\activate
```

Install:
```cmd
pip install -r requirements.txt
```

## RUNNING THE PROJECT
Train:
```cmd
python src/train.py
```

Evaluate:
```cmd
python src/evaluate.py
```

Run API:
```cmd
uvicorn api:app --reload
```
Swagger documentation will be available at `http://localhost:8000/docs`

## LIMITATIONS
This model relies on self-reported tracking data. Accuracy heavily depends on the consistency and quality of user inputs. Temporary physiological variations may affect predictions.

## MEDICAL DISCLAIMER
This system is a student/research prototype that estimates menstrual-cycle phase from available tracking data. It is not a medical diagnostic system and should not be used as a substitute for professional medical advice.
