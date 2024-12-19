import os
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from keras.models import load_model
from tensorflow.keras.utils import to_categorical


time_steps = 10  # Use 20 past timesteps for prediction

def load_classification_data():
    """Load dataset and prepare data for classification."""
    dataset_path = "lifespan_merged_datasets/mergedworms_Drug1.csv"
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")
    
    print(f"Loading classification dataset from {dataset_path}...")
    df = pd.read_csv(dataset_path)
    groups = df['worm_id']
    X = df.drop(columns=['id', 'worm_id', 'average_distance_per_frame', 
                         'maximal_distance_traveled', 'average_acceleration', 'drugged'])
    # y = (df['drugged'] == 1).astype(int)  # Binary classification for Drug1
    y = df['drugged']
    # groups = df['worm_id']
    return X, y, groups

def load_lifespan_data():
    """Load dataset and prepare data for lifespan prediction."""
    dataset_path = "lifespan_merged_datasets/mergedworms_Drug1.csv"
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")
    
    print(f"Loading lifespan prediction dataset from {dataset_path}...")
    df = pd.read_csv(dataset_path)
    df["lifespan"] = df["lifespan"] / 24  # Convert lifespan to days

    X = df.drop(columns=["lifespan", "worm_id", "id", "group"])
    y = df["lifespan"]
    groups = df['worm_id']
    return X, y, groups

def test_classification_model():
    """Test the best classification model using GroupKFold."""
    print("\n\nClassification of Worm (Drugged vs Control)")
    # model_path = "models/best_model_Drug1.pkl"  # Update path as needed
    # if not os.path.exists(model_path):
    #     raise FileNotFoundError(f"Classification model not found at {model_path}")
    
    # print(f"Loading classification model from {model_path}...")
    # model = joblib.load(model_path)
    model = SVC(random_state=42, class_weight='balanced')

    # Load data
    X, y, groups = load_classification_data()

    # Initialize GroupKFold
    gkf = GroupKFold(n_splits=5)  # Use the same number of splits as training

    print(f"Training Model...")
    accuracies = []
    for train_idx, test_idx in gkf.split(X, y, groups):
        # Split into train and test based on indices
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Preprocess the data
        imputer = SimpleImputer(strategy='median')
        scaler = StandardScaler()

        X_train = imputer.fit_transform(X_train)
        X_test = imputer.transform(X_test)
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Train the model  
        model.fit(X_train, y_train)

        # Test the pre-trained model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)
        # print(f"Fold Accuracy: {accuracy:.4f}")

    # Calculate overall accuracy
    print(f"Average Classification Accuracy: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}\n\n")

def test_lifespan_model():
    """Test the best lifespan prediction model using GroupKFold."""
    # model_path = "models/lifespan_prediction_all.pkl"  # Update path as needed
    # if not os.path.exists(model_path):
    #     raise FileNotFoundError(f"Lifespan prediction model not found at {model_path}")
    
    # print(f"Loading lifespan prediction model from {model_path}...")
    # model = joblib.load(model_path)
    model = LinearRegression()
    # Load data
    X, y, groups = load_lifespan_data()

    # Initialize GroupKFold
    gkf = GroupKFold(n_splits=5)  # Use the same number of splits as training

    mse_scores = []
    r2_scores = []

    print(f"Training Model...")
    for train_idx, test_idx in gkf.split(X, y, groups):
        # Split into train and test based on indices
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Preprocess the data
        imputer = SimpleImputer(strategy='median')
        X_train = imputer.fit_transform(X_train)
        X_test = imputer.transform(X_test)

        #Fit the model
        model.fit(X_train, y_train)

        # Test the pre-trained model
        y_pred = model.predict(X_test)

        # Evaluate the model
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        mse_scores.append(mse)
        r2_scores.append(r2)

    # Calculate overall metrics
    print(f"Average MSE: {np.mean(mse_scores):.4f} ± {np.std(mse_scores):.4f}")
    print(f"Average R^2: {np.mean(r2_scores):.4f} ± {np.std(r2_scores):.4f}\n\n")


# Function to create sequences of the desired timestep length
def create_sequences(data, time_steps):
    sequences = []
    for i in range(len(data) - time_steps + 1):  # Ensure enough timesteps are available
        sequences.append(data[i:i+time_steps])
    return np.array(sequences)

def test_optogenetics_model():
    """Test the best optogenetics model using GroupKFold."""
    print("\n\nOptogenetics Model")
    model_path = "models/LSTM_model.keras"  # Update path as needed
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Optogenetics model not found at {model_path}")
    
    print(f"Loading optogenetics model from {model_path}...")
    model = load_model(model_path)

    combined_data = pd.read_csv("optogen_data/merged_worm_data.csv")  # Replace with your combined dataset file

    columns_to_standardize = ['X', 'Y', 'euclidean_speed', 'roaming_frac', 'std_speed', 'mean_speed', 'speed_variability', 'speed_diff', 'Angular Change', 'Total Distance', 'Instantaneous Distance', 'Changed Pixels', 'Angle', 'speed_change']
    combined_data[columns_to_standardize] = (combined_data[columns_to_standardize] - combined_data[columns_to_standardize].mean()) / combined_data[columns_to_standardize].std()

    features = ['Frame','X', 'Y', 'Light_Pulse', 'euclidean_speed', 'roaming_frac', 'Angular Change', 'Instantaneous Distance', 'worm_id', 'time_hours', 'Changed Pixels', 'Total Distance', 'Angle', 'mean_speed', 'speed_variability', 'speed_change', 'std_speed']
    combined_data = combined_data[features]

    combined_data = combined_data.sort_values(by=['worm_id', 'Frame'])
    combined_data = combined_data.fillna(0)

    train_data = combined_data.groupby('worm_id').apply(lambda x: x.iloc[:len(x)//2]).reset_index(drop=True)
    test_data = combined_data.groupby('worm_id').apply(lambda x: x.iloc[len(x)//2:]).reset_index(drop=True)

    # Separate features and target for training
    X_train = train_data.drop(columns=['worm_id', 'Frame'])

    y_train = train_data['worm_id']

    # Prepare test features (exclude speed for testing)
    X_test = test_data.drop(columns=['worm_id', 'Frame'])
    y_test = test_data['worm_id']


    y_train_encoded = to_categorical(y_train)
    y_test_encoded = to_categorical(y_test)


    # Create sequences for training and testing
    X_train_LSTM = create_sequences(X_train.to_numpy(), time_steps)
    X_test_LSTM = create_sequences(X_test.to_numpy(), time_steps)

    y_train_LSTM = y_train_encoded[time_steps - 1:]  # Align targets with the sequence start
    y_test_LSTM = y_test_encoded[time_steps - 1:]

    test_loss, test_accuracy = model.evaluate(X_test_LSTM, y_test_LSTM)
    print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")
    

if __name__ == "__main__":
    print("Testing Best Models...")
    test_classification_model()
    test_lifespan_model()
    test_optogenetics_model()
