import pandas as pd
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from preprocessing import clean_dataset, select_features
from utils import load_model, load_json

def main():
    print("Loading test data and model for evaluation...")
    
    # Load dataset
    df = pd.read_csv('data/cleaned_dataset.csv')
    df = clean_dataset(df)
    
    target_classes = ["Menstrual", "Follicular", "Fertility", "Luteal"]
    df = df[df['phase'].isin(target_classes)]
    df = df.dropna(subset=['phase', 'id'])
    
    # Load metadata and splits
    metadata = load_json('model/model_metadata.json')
    splits = load_json('results/split_ids.json')
    test_ids = splits['test_ids']
    
    # Filter test set
    test_df = df[df['id'].isin(test_ids)]
    X_test = test_df[metadata['selected_features']]
    y_test = test_df['phase']
    
    label_mapping = load_json('model/label_mapping.json')
    y_test_encoded = y_test.map(label_mapping)
    
    # Load model
    model = load_model('model/menstrual_phase_model.pkl')
    
    # Evaluate
    preds = model.predict(X_test)
    
    acc = accuracy_score(y_test_encoded, preds)
    prec = precision_score(y_test_encoded, preds, average='macro', zero_division=0)
    rec = recall_score(y_test_encoded, preds, average='macro', zero_division=0)
    f1_mac = f1_score(y_test_encoded, preds, average='macro', zero_division=0)
    f1_weighted = f1_score(y_test_encoded, preds, average='weighted', zero_division=0)
    
    print("\n--- EVALUATION RESULTS ---")
    print(f"Accuracy: {acc * 100:.2f}%")
    print(f"Macro F1: {f1_mac * 100:.2f}%")
    print(f"Weighted F1: {f1_weighted * 100:.2f}%")
    print("\nClassification Report:")
    target_names = [k for k, v in sorted(label_mapping.items(), key=lambda item: item[1])]
    print(classification_report(y_test_encoded, preds, target_names=target_names))
    
    eval_results = {
        'accuracy': acc,
        'precision_macro': prec,
        'recall_macro': rec,
        'f1_macro': f1_mac,
        'f1_weighted': f1_weighted,
        'classification_report': classification_report(y_test_encoded, preds, target_names=target_names, output_dict=True)
    }
    save_path = 'results/evaluation_results_standalone.json'
    with open(save_path, 'w') as f:
        json.dump(eval_results, f, indent=4)
        
    print(f"\nResults saved to {save_path}")

if __name__ == '__main__':
    main()
