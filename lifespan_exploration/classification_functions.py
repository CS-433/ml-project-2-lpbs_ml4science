import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import StackingClassifier
import matplotlib.pyplot as plt
import seaborn as sns

#code to return the model along with accuracies and stdev for a specific model
def return_model(df, model_name, n_splits=5, random_state=42):
    """
    Returns a trained model, mean accuracy, and standard deviation after GroupKFold validation.

    Args:
        df: DataFrame containing the dataset.
        model_name: Name of the model ('svm', 'logistic', etc.).
        n_splits: Number of folds for cross-validation.
        random_state: Random state.

    Returns:
        model: The trained model on the full dataset.
        mean_accuracy: Mean accuracy from cross-validation.
        std_accuracy: Standard deviation of accuracy from cross-validation.
    """
    # Extract features (X), target (y), and groups (worm_id)
    X = df.drop(columns=['id', 'worm_id', 'drugged', 'average_distance_per_frame', 
                         'maximal_distance_traveled', 'average_acceleration'])
    y = df['drugged']
    groups = df['worm_id']

    # Select model based on model_name
    if model_name == "svm":
        model = SVC(random_state=random_state, class_weight='balanced')
    elif model_name == "logistic":
         model = LogisticRegression(random_state=random_state, penalty='l2', max_iter=1000, multi_class='multinomial', solver='saga', class_weight={0:0.5,1:0.25,2:0.25})
    elif model_name == "random_forest":
        model = RandomForestClassifier(random_state=random_state)
    elif model_name == "stacking_classifier":
        estimators = [('rf', RandomForestClassifier()), ('svm', SVC(probability=True))] #for stacking
        model = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())
    else:
        raise ValueError("Unsupported model name. Choose from 'svm', 'logistic', or 'random_forest'.")

    # Initialize GroupKFold
    gkf = GroupKFold(n_splits=n_splits)
    accuracies = []

    # Cross-validation loop
    for train_idx, test_idx in gkf.split(X, y, groups):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Handle missing values
        if np.isnan(X_train).any().any() or np.isnan(X_test).any().any():
            imputer = SimpleImputer(strategy='median')
            X_train = imputer.fit_transform(X_train)
            X_test = imputer.transform(X_test)

        # Standardize features if needed
        if model_name in ["logistic", "svm"]:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        # Train and evaluate the model
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        accuracies.append(acc)

    # Retrain the model on the full dataset after cross-validation
    if model_name in ["logistic", "svm"]:
        scaler = StandardScaler()
        imputer = SimpleImputer(strategy='median')
        X = imputer.fit_transform(X)
        X_scaled = scaler.fit_transform(X)
    else:
        imputer = SimpleImputer(strategy='median')
        X_scaled = imputer.fit_transform(X)

    model.fit(X_scaled, y)

    # Compute mean and std deviation
    mean_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)

    return model, mean_accuracy, std_accuracy



#code to run kfold validation for all models and return a dictionary of the results
def group_kfold_validation_all_models(df, n_splits=5, random_state=42):
    """
    Perform GroupKFold cross-validation for multiple models on the dataset based on worm_id.

    Args:
        df: DataFrame containing the dataset.
        n_splits: Number of folds for cross-validation.
        random_state

    Returns:
        A dictionary with model names, mean accuracy, and standard deviation.
    """
    # Extract features (X), target (y), and groups (worm_id)
    X = df.drop(columns=['id', 'worm_id', 'drugged', 'average_distance_per_frame', 'maximal_distance_traveled', 'average_acceleration'])  # Drop 'worm_id' and target
    y = df['drugged']  # Target variable
    groups = df['worm_id']  # Group variable for GroupKFold
    estimators = [('rf', RandomForestClassifier()), ('svm', SVC(probability=True))] #for stacking


    # Define models to evaluate
    num_classes = len(pd.unique(y))
    if num_classes == 2:
      models_dict = {
          "logistic": LogisticRegression(random_state=random_state, max_iter=500, solver='lbfgs'),
          "random_forest": RandomForestClassifier(random_state=random_state),
          "decision_tree": DecisionTreeClassifier(random_state=random_state),
          "svm": SVC(random_state=random_state, class_weight='balanced'),
          "xgboost": XGBClassifier(random_state=random_state, use_label_encoder=False, eval_metric='mlogloss'),
          "knn": KNeighborsClassifier(n_neighbors=5),
          "naive_bayes": GaussianNB(),
          "mlp_classifier": MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42),
          "stacking_classifier": StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())
      }
    elif num_classes == 3:
      models_dict = {
          "logistic": LogisticRegression(random_state=random_state, penalty='l2', max_iter=1000, multi_class='multinomial', solver='saga', class_weight={0:0.5,1:0.25,2:0.25}),
          "random_forest": RandomForestClassifier(random_state=random_state),
          "decision_tree": DecisionTreeClassifier(random_state=random_state),
          "svm": SVC(random_state=random_state, class_weight='balanced'),
          "xgboost": XGBClassifier(random_state=random_state, use_label_encoder=False, eval_metric='mlogloss', n_estimators=100, learning_rate=0.05, gamma=0.001),
          "knn": KNeighborsClassifier(n_neighbors=5),
          "naive_bayes": GaussianNB(),
          "mlp_classifier": MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42),
          "stacking_classifier": StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())
      }
    else:
      model_dict = {}
      print("Error: Unexpected number of unique values in y: ", num_classes)

    # Initialize GroupKFold
    gkf = GroupKFold(n_splits=n_splits)
    results = []

    for model_name, model in models_dict.items():
        print(f"Evaluating model: {model_name}")
        accuracies = []
        for train_idx, test_idx in gkf.split(X, y, groups):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            # Handle missing values
            if np.isnan(X_train).any().any() or np.isnan(X_test).any().any():
                imputer = SimpleImputer(strategy='median')
                X_train = imputer.fit_transform(X_train)
                X_test = imputer.transform(X_test)

            # Standardize features if needed
            if model_name in ["logistic", "svm"]:
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)

            # Train the model
            model.fit(X_train, y_train)

            # Evaluate the model
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            accuracies.append(acc)

        # Store the results
        results.append({
            "model": model_name,
            "mean_accuracy": np.mean(accuracies),
            "std_deviation": np.std(accuracies)
        })


    return results

def plot_all_df(all_results):
  # Set up the plot
  plt.figure(figsize=(16, 8))

  # Barplot for mean accuracy
  sns.barplot(
      data=all_results,
      x="model",
      y="mean_accuracy",
      hue="task",
      ci=None  # Disable seaborn's internal confidence intervals
  )

  # Add error bars for individual bars
  for i, bar in enumerate(plt.gca().patches):
      # Retrieve mean_accuracy and std_deviation from DataFrame
      model = all_results.iloc[i % len(all_results)]  # Handles bar group looping
      mean = model["mean_accuracy"]
      std = model["std_deviation"]
      
      # Add error bar centered at the top of the bar
      plt.errorbar(
          x=bar.get_x() + bar.get_width() / 2,  # Center of the bar
          y=mean,  # Mean value
          yerr=std,  # Standard deviation as error
          fmt='none',  # No marker
          c='black',
          capsize=5  # Small caps on error bars
      )
  # Formatting the plot
  plt.title("Model Performance Across Tasks", fontsize=16)
  plt.ylabel("Mean Accuracy", fontsize=14)
  plt.xlabel("Model", fontsize=14)
  plt.xticks(rotation=45)
  plt.legend(title="Task")
  plt.tight_layout()

  # Show plot
  plt.show()


def plot_best_five_individual(plots_df):
  plt.rcParams.update({'errorbar.capsize': 20})
  count = 0
  for df in plots_df:
      # print(df)
  # Create the bar plot
      plt.figure(figsize=(10, 6))
      sns.barplot(x='model', y='mean_accuracy', data=df, yerr=df['std_deviation'], ecolor='black')
      if count == 0:
      # Add labels and title
          plt.title('Binary Classification (Drug1) Results: Model Accuracy')
      elif count == 1:
          plt.title('Binary Classification (Drug2) Results: Model Accuracy')
      else:
          plt.title('Multi-class Classification Results: Model Accuracy')
      plt.xlabel('Model')
      plt.ylabel('Mean Accuracy')

      # Show the plot
      plt.xticks(rotation=45)
      plt.tight_layout()
      plt.show()
      
      count += 1