#Import libraries
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from xgboost import XGBRegressor
from sklearn.linear_model import ElasticNet


def plot_results(results_df):
    print(results_df)

    #Plot R^2 and MSE
    plt.figure(figsize=(8, 6))
    plt.bar(results_df["model"], results_df["r2_mean"], yerr=results_df["r2_std"], capsize=5, alpha=0.7)
    plt.ylabel("Mean R^2 Score")
    plt.title("Model Comparison: R^2 with Standard Deviation")
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.bar(results_df["model"], results_df["mse_mean"], yerr=results_df["mse_std"], capsize=5, alpha=0.7)
    plt.ylabel("Mean MSE")
    plt.title("Model Comparison: MSE with Standard Deviation")
    plt.show()

def evaluate_models(df, models):
    results = []

    for model in models:
        result = evaluate_model_kfold(df, model_select=model)
        results.append(result)

    return results


def evaluate_model_kfold(data, model_select="linear", n_splits=5, random_state=42):
    df = data.copy()
    df["lifespan"] = df["lifespan"] / 24

    X = df.drop(columns=["lifespan", "worm_id", "id", "group"])
    # X = df.drop(columns=["lifespan", "worm_id", "group_first", "time_elapsed_(hours)"])

    y = df["lifespan"]
    groups = df["worm_id"]

    imputer = SimpleImputer(strategy="median")
    X = imputer.fit_transform(X)

    if model_select == "linear":
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        model = LinearRegression()
    elif model_select == "random_forest":
        model = RandomForestRegressor(random_state=random_state)
    elif model_select == "xgboost":
        model = XGBRegressor(random_state=random_state, n_estimators=100, learning_rate=0.05)
    elif model_select == "elasticnet":
        model = ElasticNet(random_state=random_state, alpha=0.1, l1_ratio=0.5)
    else:
        raise ValueError("Invalid model selection.")

    # Perform GroupKFold CV and collect metrics
    gkf = GroupKFold(n_splits=n_splits)
    r2_scores = []
    mse_scores = []

    for train_idx, test_idx in gkf.split(X, y, groups):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mse_scores.append(mean_squared_error(y_test, y_pred))
        r2_scores.append(r2_score(y_test, y_pred))

    # Calculate mean and std of metrics
    mse_mean, mse_std = np.mean(mse_scores), np.std(mse_scores)
    r2_mean, r2_std = np.mean(r2_scores), np.std(r2_scores)

    # Feature importance for linear models
    if model_select in ["linear", "elasticnet"]:
        model.fit(X, y)  # Fit the model
        feature_importances = sorted(zip(df.drop(columns=["lifespan", "worm_id"]).columns, model.coef_), key=lambda x: -abs(x[1]))
        print(model_select, "\nFeature Importances (Linear Regression Coefficients):")
        for feature, coef in feature_importances:
            if coef >= 0.05:
                print(f"{feature}: {coef:.4f}")
        print("\n")

    # Feature importance for tree-based models
    if model_select in ["random_forest", "xgboost"]:
        model.fit(X, y)
        importances = model.feature_importances_
        feature_importances = sorted(zip(df.drop(columns=["lifespan", "worm_id"]).columns, importances), key=lambda x: -x[1])
        print(model_select, "\nFeature Importances:")
        for feature, importance in feature_importances:
            if importance >= 0.05:
                print(f"{feature}: {importance:.4f}")
        print("\n")

    plt.scatter(y_test, y_pred, alpha=0.7)
    plt.xlabel("Actual Lifespan (days)")
    plt.ylabel("Predicted Lifespan (days)")
    plt.title(f"Predicted vs. Actual Lifespan: {model_select}")
    plt.show()

    return {"model": model_select, "mse_mean": mse_mean, "mse_std": mse_std, 
            "r2_mean": r2_mean, "r2_std": r2_std}

def return_model(data, model_select, n_splits=5, random_state=42):
    df = data.copy()
    df["lifespan"] = df["lifespan"] / 24

    X = df.drop(columns=["lifespan", "worm_id", "id", "group"])
    # X = df.drop(columns=["lifespan", "worm_id", "group_first", "time_elapsed_(hours)"])

    y = df["lifespan"]
    groups = df["worm_id"]

    imputer = SimpleImputer(strategy="median")
    X = imputer.fit_transform(X)

    if model_select == "linear":
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        model = LinearRegression()
    if model_select == "elasticnet":
        model = ElasticNet(random_state=random_state, alpha=0.1, l1_ratio=0.5)
    
    # Perform GroupKFold CV and collect metrics
    gkf = GroupKFold(n_splits=n_splits)
    r2_scores = []
    mse_scores = []

    for train_idx, test_idx in gkf.split(X, y, groups):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mse_scores.append(mean_squared_error(y_test, y_pred))
        r2_scores.append(r2_score(y_test, y_pred))

    # Calculate mean and std of metrics
    mse_mean, mse_std = np.mean(mse_scores), np.std(mse_scores)
    r2_mean, r2_std = np.mean(r2_scores), np.std(r2_scores)

    model.fit(X, y)


    return model, [mse_mean, mse_std, r2_mean, r2_std]
