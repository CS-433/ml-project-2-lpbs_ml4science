import os
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

def load_classification_data():
    """Load dataset and prepare data for classification."""
    dataset_path = "lifespan_merged_datasets/mergedworms_combined.csv"
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")
    
    print(f"Loading classification dataset from {dataset_path}...")
    df = pd.read_csv(dataset_path)

    X = df.drop(columns=['id', 'worm_id', 'average_distance_per_frame', 
                         'maximal_distance_traveled', 'average_acceleration', 'drugged'])
    # y = (df['drugged'] == 1).astype(int)  # Binary classification for Drug1
    y = df['drugged']
    return X, y

def load_lifespan_data():
    """Load dataset and prepare data for lifespan prediction."""
    dataset_path = "lifespan_merged_datasets/mergedworms_combined2.csv"
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")
    
    print(f"Loading lifespan prediction dataset from {dataset_path}...")
    df = pd.read_csv(dataset_path)
    df["lifespan"] = df["lifespan"] / 24  # Convert lifespan to days

    X = df.drop(columns=["lifespan", "worm_id", "id", "group"])
    y = df["lifespan"]
    return X, y

def test_classification_model():
    """Test the best classification model."""
    model_path = "models/best_model_multiclass.pkl"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Classification model not found at {model_path}")
    
    print(f"Loading classification model from {model_path}...")
    model = joblib.load(model_path)

    X, y = load_classification_data()
    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()

    X = imputer.fit_transform(X)
    X = scaler.fit_transform(X)

    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    print(f"Classification Accuracy: {accuracy:.4f}")

def test_lifespan_model():
    """Test the best lifespan prediction model."""
    model_path = "models/lifespan_prediction_all.pkl"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Lifespan prediction model not found at {model_path}")
    
    print(f"Loading lifespan prediction model from {model_path}...")
    model = joblib.load(model_path)

    X, y = load_lifespan_data()
    imputer = SimpleImputer(strategy='median')

    X = imputer.fit_transform(X)

    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    print(f"Lifespan Prediction MSE: {mse:.4f}")
    print(f"Lifespan Prediction R^2: {r2:.4f}")

if __name__ == "__main__":
    print("Testing Best Models...")
    test_classification_model()
    test_lifespan_model()
