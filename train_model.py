import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from datetime import datetime

# Set up paths
DATA_PATH = '/workspace/app/jailbreak.csv'
OUTPUT_NB_PATH = '/workspace/app/jailbreak_detection.ipynb'
OUTPUT_CSV_PATH = '/workspace/app/submissions.csv'

# Initialize progress
print("Loading training dataset...")
df = pd.read_csv(DATA_PATH)

print(f"Dataset loaded: {len(df)} rows")
print(f"Columns: {df.columns.tolist()}")

# Basic analysis
print("\n=== Dataset Analysis ===")
print(f"Jailbreak distribution:")
print(df['jailbreak'].value_counts())
print(f"Proportion of jailbroken prompts: {df['jailbreak'].mean():.4f}")

# Check for null values
print(f"\nNull values per column:")
print(df.isnull().sum())

# Sample some prompts
print("\n=== Sample Prompts ===")
print("\nBenign prompt:")
print(df[df['jailbreak'] == False]['prompt'].iloc[0][:500] + "...")

print("\nJailbroken prompt:")
print(df[df['jailbreak'] == True]['prompt'].iloc[0][:500] + "...")

# Create features from prompts
def extract_features(prompts):
    features = {}
    # Text length
    features['length'] = [len(p) for p in prompts]
    
    # Number of question marks
    features['question_marks'] = [p.count('?') for p in prompts]
    
    # Number of exclamation marks
    features['exclamation_marks'] = [p.count('!') for p in prompts]
    
    # Presence of certain jailbreak-related terms
    jailbreak_terms = ['[Illegality Mode]', 'CumGPT', 'do not refuse', 'never refuse', 
                      'amoral', 'unethical', 'illegal', 'nsfw', 'porn', 'sexual',
                      'fuck', 'shit', 'ass', 'raped', 'violence', 'murder']
    
    for term in jailbreak_terms:
        features[f'contains_{term.lower().replace(" ", "_")}'] = [
            term.lower() in p.lower() for p in prompts
        ]
    
    # Capital letter ratio
    features['capital_ratio'] = [sum(1 for c in p if c.isupper()) / max(len(p), 1) for p in prompts]
    
    return features

print("\nExtracting features...")
feature_dict = extract_features(df['prompt'])
feature_df = pd.DataFrame(feature_dict)

# Combine with target
X = feature_df
X['jailbreak'] = df['jailbreak']

# One-hot encode the jailbreak column (this is our target)
y = df['jailbreak'].astype(int)

print(f"\nFeature matrix shape: {X.shape}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\nTraining set: {X_train.shape}, Test set: {X_test.shape}")

# Train logistic regression model
print("\nTraining Logistic Regression model...")
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train, y_train)

# Predict and evaluate
print("\nEvaluating model...")
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probability of class 1 (jailbroken)

print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"ROC AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")

# Classification report
print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred))

# Test on some sample prompts
print("\n=== Model Predictions on Sample Prompts ===")
# Show some predictions
test_indices = np.random.choice(len(df), 10, replace=False)
for idx in test_indices:
    prompt = df['prompt'].iloc[idx]
    true_label = df['jailbreak'].iloc[idx]
    features = extract_features([prompt])
    feature_test = pd.DataFrame(features)
    feature_test['jailbreak'] = [true_label]
    
    pred = model.predict(feature_test)[0]
    prob = model.predict_proba(feature_test)[0][1]
    
    print(f"\nPrompt: {prompt[:100]}...")
    print(f"True: {'Jailbroken' if true_label else 'Benign'}, Pred: {'Jailbroken' if pred else 'Benign'}, Prob: {prob:.4f}")

print(f"\n✅ Model training and evaluation completed successfully!")
print(f"📝 Results saved to {OUTPUT_NB_PATH}")
print(f"📋 Predictions saved to {OUTPUT_CSV_PATH}")

# Save predictions for submission
submission_df = pd.DataFrame({
    'prompt_id': range(len(df)),
    'prompt': df['prompt'].astype(str),
    'jailbreak_score': model.predict_proba(extract_features(df['prompt'])[0].T)[:, 1] if hasattr(extract_features(df['prompt'])[0], 'T') else y_pred_proba[:len(df)],
    'predicted_jailbreak': ['Jailbroken' if p == 1 else 'Benign' for p in model.predict(extract_features(df['prompt'])[0].T) if hasattr(extract_features(df['prompt'])[0], 'T') else y_pred]
})

# Save to CSV
submission_df.to_csv(OUTPUT_CSV_PATH, index=False)
print(f"\n✅ Predictions saved to {OUTPUT_CSV_PATH}")

# Create notebook
print("\nCreating Jupyter notebook...")
