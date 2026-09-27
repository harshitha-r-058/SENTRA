import pandas as pd
import numpy as np
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.pipeline import Pipeline
import joblib

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

from preprocessing import clean_dataset, select_features, build_preprocessing_pipeline
from utils import save_json, save_model

def get_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    return acc, prec, rec, f1_macro, f1_weighted

def main():
    print("===============================")
    print("MENSTRUAL CYCLE ML MODEL")
    print("========================")
    
    # 1. Load dataset
    data_path = 'data/cleaned_dataset.csv'
    print(f"\nDataset:\n{os.path.basename(data_path)}")
    df = pd.read_csv(data_path)
    
    # 2. Inspect dataset (Initial print)
    print(f"\nInitial Records:\n{len(df)}")
    
    # 3. Clean data
    df = clean_dataset(df)
    
    # Filter target to known classes
    target_classes = ["Menstrual", "Follicular", "Fertility", "Luteal"]
    df = df[df['phase'].isin(target_classes)]
    
    # 4. Identify features & 5. Exclude leakage
    selected_features = select_features(df)
    excluded_features = [col for col in df.columns if col not in selected_features and col != 'phase']
    
    print(f"\nFeatures used:\n{len(selected_features)}")
    print(f"\nFeatures excluded:\n{len(excluded_features)}")
    
    unique_ids = df['id'].nunique()
    print(f"\nUnique IDs:\n{unique_ids}")
    
    print("\nTarget:\nphase")
    print("\nClasses:")
    for cls in target_classes:
        print(cls)
        
    # Drop rows where target or ID is missing
    df = df.dropna(subset=['phase', 'id'])
    
    # 6. Split using groups
    X = df[selected_features + ['id']]
    y = df['phase']
    groups = df['id']
    
    # Create 80/20 train/test split using StratifiedGroupKFold
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    train_idx, test_idx = next(sgkf.split(X, y, groups))
    
    X_dev = X.iloc[train_idx].drop(columns=['id'])
    y_dev = y.iloc[train_idx]
    groups_dev = groups.iloc[train_idx]
    
    X_test = X.iloc[test_idx].drop(columns=['id'])
    y_test = y.iloc[test_idx]
    groups_test = groups.iloc[test_idx]
    
    print(f"\nTraining records:\n{len(X_dev)}")
    print(f"\nTesting records:\n{len(X_test)}")
    print(f"\nTraining IDs:\n{groups_dev.nunique()}")
    print(f"\nTesting IDs:\n{groups_test.nunique()}")
    
    # Map target labels to integers for models that require it (like XGBoost)
    label_mapping = {label: idx for idx, label in enumerate(target_classes)}
    inverse_label_mapping = {idx: label for label, idx in label_mapping.items()}
    y_dev_encoded = y_dev.map(label_mapping)
    y_test_encoded = y_test.map(label_mapping)
    
    save_json(label_mapping, 'model/label_mapping.json')
    save_json(selected_features, 'model/feature_columns.json')
    save_json(test_idx.tolist(), 'results/test_group_ids.json') # Just row indices for reproducibility
    save_json({'train_ids': groups_dev.unique().tolist(), 'test_ids': groups_test.unique().tolist()}, 'results/split_ids.json')
    
    # 7. Build preprocessing pipeline
    preprocessor, num_cols, cat_cols = build_preprocessing_pipeline(df, selected_features)
    
    # Models to evaluate
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=300, random_state=42, class_weight='balanced', n_jobs=-1)
    }
    if HAS_XGBOOST:
        models['XGBoost'] = XGBClassifier(n_estimators=300, random_state=42, use_label_encoder=False, eval_metric='mlogloss', n_jobs=-1)
        
    model_results = {}
    print("\n---")
    print("\n## MODEL COMPARISON")
    
    # Cross validation on development set
    cv_splits = 5
    sgkf_cv = StratifiedGroupKFold(n_splits=cv_splits, shuffle=True, random_state=42)
    
    for name, clf in models.items():
        pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', clf)])
        
        acc_scores, f1_macro_scores = [], []
        
        for cv_train_idx, cv_val_idx in sgkf_cv.split(X_dev, y_dev_encoded, groups_dev):
            X_cv_train, y_cv_train = X_dev.iloc[cv_train_idx], y_dev_encoded.iloc[cv_train_idx]
            X_cv_val, y_cv_val = X_dev.iloc[cv_val_idx], y_dev_encoded.iloc[cv_val_idx]
            
            pipeline.fit(X_cv_train, y_cv_train)
            preds = pipeline.predict(X_cv_val)
            
            acc = accuracy_score(y_cv_val, preds)
            f1_mac = f1_score(y_cv_val, preds, average='macro', zero_division=0)
            
            acc_scores.append(acc)
            f1_macro_scores.append(f1_mac)
            
        mean_acc = np.mean(acc_scores)
        mean_f1_mac = np.mean(f1_macro_scores)
        model_results[name] = {'mean_acc': mean_acc, 'mean_f1_mac': mean_f1_mac, 'pipeline': pipeline}
        
        print(f"\n{name}:")
        print(f"Accuracy: {mean_acc:.4f}")
        print(f"Macro F1: {mean_f1_mac:.4f}")
        
    # 12. Select final model
    best_model_name = max(model_results, key=lambda x: model_results[x]['mean_f1_mac'])
    final_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', models[best_model_name])])
    
    # 13. Train final model on development data
    final_pipeline.fit(X_dev, y_dev_encoded)
    
    # 14. Evaluate on unseen test groups
    test_preds = final_pipeline.predict(X_test)
    acc, prec, rec, f1_mac, f1_weighted = get_metrics(y_test_encoded, test_preds)
    
    print("\n---")
    print("\n## FINAL MODEL")
    print(f"\nModel:\n{best_model_name}")
    print(f"\nTest Accuracy:\n{acc * 100:.2f}%")
    print(f"\nTest Macro F1:\n{f1_mac * 100:.2f}%")
    print(f"\nTest Weighted F1:\n{f1_weighted * 100:.2f}%")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test_encoded, test_preds)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_classes, yticklabels=target_classes)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig('results/confusion_matrix.png')
    plt.close()
    
    cm_df = pd.DataFrame(cm, index=target_classes, columns=target_classes)
    cm_df.to_csv('results/confusion_matrix.csv')
    
    # 21. Feature Importance
    if hasattr(final_pipeline.named_steps['classifier'], 'feature_importances_'):
        importances = final_pipeline.named_steps['classifier'].feature_importances_
        # Get feature names out
        cat_encoder = final_pipeline.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot']
        if cat_cols:
            cat_feature_names = cat_encoder.get_feature_names_out(cat_cols)
        else:
            cat_feature_names = []
        all_feature_names = num_cols + list(cat_feature_names)
        
        fi_df = pd.DataFrame({'feature': all_feature_names, 'importance': importances})
        fi_df = fi_df.sort_values(by='importance', ascending=False)
        fi_df.to_csv('results/feature_importance.csv', index=False)
        
        plt.figure(figsize=(10,6))
        sns.barplot(x='importance', y='feature', data=fi_df.head(15))
        plt.title('Top Feature Importances')
        plt.tight_layout()
        plt.savefig('results/feature_importance.png')
        plt.close()
        
        print("\n---")
        print("\n## TOP FEATURES")
        for i, (idx, row) in enumerate(fi_df.head(10).iterrows(), 1):
            print(f"{i}. {row['feature']}")
            
    # 15. Save model
    model_path = 'model/menstrual_phase_model.pkl'
    save_model(final_pipeline, model_path)
    
    # 16. Save metadata
    metadata = {
        'model_name': best_model_name,
        'model_version': '1.0',
        'dataset_name': 'cleaned_dataset.csv',
        'target_column': 'phase',
        'target_classes': target_classes,
        'selected_features': selected_features,
        'excluded_features': excluded_features,
        'training_records': len(X_dev),
        'testing_records': len(X_test),
        'training_unique_ids': int(groups_dev.nunique()),
        'testing_unique_ids': int(groups_test.nunique()),
        'final_test_metrics': {
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1_macro': f1_mac,
            'f1_weighted': f1_weighted
        },
        'random_seed': 42,
        'preprocessing_description': 'Median imputation for numericals, most frequent + one hot for categoricals'
    }
    save_json(metadata, 'model/model_metadata.json')
    
    # 18. Save evaluation results
    eval_results = {
        'metrics': metadata['final_test_metrics'],
        'classification_report': classification_report(y_test_encoded, test_preds, target_names=target_classes, output_dict=True)
    }
    save_json(eval_results, 'results/evaluation_results.json')
    
    print("\n---")
    print(f"\n## MODEL FILE\n\n{model_path}")
    print("\n---")
    print("\n## API\n")
    print("POST /predict-cycle-phase\n")
    print("GET /health\n")
    print("GET /model-info\n")
    print("===============================\n")

if __name__ == '__main__':
    main()
