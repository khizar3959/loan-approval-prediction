import os
import json
import pandas as pd
import numpy as np

def train_model():
    # Load dataset
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_dir, '..', 'loan prediction train.csv')
    
    print(f"Loading training data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Define exact columns and order based on original notebook
    feature_cols = [
        'Gender',
        'Married',
        'Dependents',
        'Education',
        'Self_Employed',
        'ApplicantIncome',
        'CoapplicantIncome',
        'LoanAmount',
        'Loan_Amount_Term',
        'Credit_History',
        'Property_Area'
    ]
    
    cat_features = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']
    num_features = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History']
    
    # 1. Calculate Imputers
    print("Calculating imputation parameters...")
    medians = {}
    for col in num_features:
        median_val = float(df[col].median())
        medians[col] = median_val
        df[col] = df[col].fillna(median_val)
        print(f"  Median for {col}: {median_val}")
        
    modes = {}
    for col in cat_features:
        mode_val = str(df[col].mode()[0])
        modes[col] = mode_val
        df[col] = df[col].fillna(mode_val)
        print(f"  Mode for {col}: {mode_val}")
        
    # 2. Encode Categorical Features
    print("Encoding categorical features...")
    category_mappings = {}
    for col in cat_features:
        # Sort classes alphabetically to match scikit-learn LabelEncoder behavior
        classes = sorted(list(df[col].unique()))
        mapping = {cls: idx for idx, cls in enumerate(classes)}
        category_mappings[col] = mapping
        df[col] = df[col].map(mapping)
        print(f"  Encoding mapping for {col}: {mapping}")
        
    # Encode target variable: Loan_Status (N -> 0, Y -> 1)
    target_classes = sorted(list(df['Loan_Status'].unique())) # Should be ['N', 'Y']
    target_mapping = {cls: idx for idx, cls in enumerate(target_classes)}
    y_train = df['Loan_Status'].map(target_mapping).tolist()
    print(f"  Encoding target Loan_Status: {target_mapping}")

    # X training matrix in the exact column order
    X_train_df = df[feature_cols]

    # 3. Calculate Scaling Parameters (StandardScaler)
    print("Calculating scaling parameters...")
    means = []
    stds = []
    
    X_train_scaled = X_train_df.copy()
    for col in feature_cols:
        mean_val = float(X_train_df[col].mean())
        # Scikit-learn StandardScaler uses population std (ddof=0)
        std_val = float(X_train_df[col].std(ddof=0))
        
        # Guard against division by zero (if std is zero, set to 1.0)
        if std_val == 0:
            std_val = 1.0
            
        means.append(mean_val)
        stds.append(std_val)
        
        X_train_scaled[col] = (X_train_df[col] - mean_val) / std_val
        print(f"  Scale for {col}: mean = {mean_val:.4f}, std = {std_val:.4f}")

    # Convert scaled X to list of lists for JSON serialization
    X_train_list = X_train_scaled.values.tolist()

    # 4. Package all artifacts into a JSON bundle
    model_bundle = {
        "feature_cols": feature_cols,
        "cat_features": cat_features,
        "num_features": num_features,
        "medians": medians,
        "modes": modes,
        "category_mappings": category_mappings,
        "means": means,
        "stds": stds,
        "X_train": X_train_list,
        "y_train": y_train
    }

    # Save to file
    model_dir = os.path.join(current_dir, 'model')
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'model.json')
    
    with open(model_path, 'w') as f:
        json.dump(model_bundle, f, indent=2)
        
    print(f"\nModel training data successfully exported to {model_path}")
    print(f"Total training samples: {len(X_train_list)}")

if __name__ == '__main__':
    train_model()
