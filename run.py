import os
import argparse
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Utility functions for training and testing
def load_data(task_type):
    """Load dataset based on the task type."""
    base_dir = "lifespan_merged_datasets"
    
    # Map task type to the correct CSV file
    dataset_files = {
        "classification-drug1": "mergedworms_Drug1.csv",
        "classification-drug2": "mergedworms_Drug2.csv",
        "classification-multiclass": "mergedworms_combined.csv"
    }
    
    if task_type not in dataset_files:
        raise ValueError(f"Invalid task type specified: {task_type}")
    
    dataset_path = os.path.join(base_dir, dataset_files[task_type])
    
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")
    
    print(f"Loading dataset from {dataset_path}...")
    df = pd.read_csv(dataset_path)

    # Drop unnecessary columns and prepare features/targets
    X = df.drop(columns=['id', 'worm_id', 'average_distance_per_frame', 
                         'maximal_distance_traveled', 'average_acceleration', 'drugged'])
    
    if task_type == "classification-drug1":
        y = (df['drugged'] == 1).astype(int)  # Binary classification for Drug1
    elif task_type == "classification-drug2":
        y = (df['drugged'] == 2).astype(int)  # Binary classification for Drug2 (drug presence)
    elif task_type == "classification-multiclass":
        y = df['drugged']  # Multi-class classification
    else:
        raise ValueError("Invalid task type specified")
    
    return X, y, df['worm_id']

def train_model(X, y, groups, model_path):
    """Train the model using GroupKFold and save it."""
    model = SVC(random_state=42, class_weight='balanced')
    scaler = StandardScaler()
    imputer = SimpleImputer(strategy='median')

    gkf = GroupKFold(n_splits=5)
    accuracies = []

    # Cross-validation loop
    for train_idx, test_idx in gkf.split(X, y, groups):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Handle missing values
        X_train = imputer.fit_transform(X_train)
        X_test = imputer.transform(X_test)

        # Standardize features
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Train and evaluate
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        accuracies.append(acc)
    
    print(f"Mean accuracy: {np.mean(accuracies):.4f}, Std: {np.std(accuracies):.4f}")
    
    # Save the model
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    print(f"Model saved at: {model_path}")

    return model

def test_model(model, X, y):
    """Test the loaded model."""
    print("Testing the model...")
    X = SimpleImputer(strategy='median').fit_transform(X)
    X = StandardScaler().fit_transform(X)
    
    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)
    print(f"Test accuracy: {acc:.4f}")

# Main function with argument parsing
def main():
    parser = argparse.ArgumentParser(description="Run ML models for C. elegans project.")
    parser.add_argument('--type', type=str, required=True,
                        choices=['classification-drug1', 
                                 'classification-drug2', 
                                 'classification-multiclass'],
                        help="Type of classification to perform.")
    
    args = parser.parse_args()
    task_type = args.type

    # Define paths
    model_files = {
        "classification-drug1": "best_model_Drug1.pkl",
        "classification-drug2": "best_model_Drug2.pkl",
        "classification-multiclass": "best_model_multiclass.pkl"
    }
    model_path = os.path.join("models", model_files[task_type])

    # Load data
    X, y, groups = load_data(task_type)

    # Check if model exists
    if os.path.exists(model_path):
        print("Model found. Loading...")
        model = joblib.load(model_path)
    else:
        print("No model found. Training a new model...")
        model = train_model(X, y, groups, model_path)

    # Test the model
    test_model(model, X, y)

if __name__ == "__main__":
    main()
