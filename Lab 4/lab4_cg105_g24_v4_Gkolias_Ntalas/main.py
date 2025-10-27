import random
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

import numpy as np
from sklearn.model_selection import cross_val_score, train_test_split

from datasets import load_dataset


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


if __name__ == "__main__":
    # Set seed for reproducibility
    seed = 0
    set_seed(seed)

    # TODO Load and preprocess dataset
    X, y = load_dataset("titanic")

    # Drop redundant features or features with too many missing values
    X = X.drop(columns=["deck", "embark_town", "alive", "who", "class", "adult_male", "alone"])

    # Fill missing values
    X["age"] = X["age"].fillna(X["age"].median())
    X["embarked"] = X["embarked"].fillna("S")

    # Encode categorical features
    X = pd.get_dummies(X, columns=["sex", "embarked"], drop_first=True)

    # Normalize numerical features
    scaler = StandardScaler()
    X[["age", "fare", "parch", "sibsp"]] = scaler.fit_transform(X[["age", "fare", "parch", "sibsp"]])


    # Split data into train and test partitions with 80% train and 20% test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed
    )

    # Logistic Regression tuning
    logistic_results = []
    C_values = [0.01, 0.1, 1, 10]
    for c in C_values:
        model = LogisticRegression(C=c, max_iter=200)
        scores = cross_val_score(model, X_train, y_train, cv=4, scoring='accuracy')
        logistic_results.append((c, scores.mean()))

    logistic_df = pd.DataFrame(logistic_results, columns=["C", "CV Accuracy"])
    print("\nLogistic Regression Results:")
    print(logistic_df)
    best_logistic_score = max(logistic_df["CV Accuracy"])
    best_logistic_params = logistic_df.loc[logistic_df["CV Accuracy"].idxmax(), "C"]
    print(f"\nBest Logistic Regression Params: {best_logistic_params}")
    print(f"Best CV Accuracy: {best_logistic_score:.4f}")

    # Random Forest tuning
    rf_results = []
    n_estimators = [50, 100, 200]
    max_depths = [None, 5, 10]
    min_samples_split = [2, 5]

    best_rf_score = 0
    best_rf_params = None

    for n in n_estimators:
        for d in max_depths:
            for s in min_samples_split:
                model = RandomForestClassifier(n_estimators=n, max_depth=d, min_samples_split=s, random_state=seed)
                scores = cross_val_score(model, X_train, y_train, cv=4, scoring='accuracy')
                mean_score = scores.mean()
                rf_results.append((n, d, s, mean_score))
                if mean_score > best_rf_score:
                    best_rf_score = mean_score
                    best_rf_params = (n, d, s)

    rf_df = pd.DataFrame(rf_results, columns=["n_estimators", "max_depth", "min_samples_split", "CV Accuracy"])
    print("\nRandom Forest Results:")
    print(rf_df)

    print(f"\nBest Random Forest Params: {best_rf_params}")
    print(f"Best CV Accuracy: {best_rf_score:.4f}")




    # TODO Define the models
    model1 = LogisticRegression(C=0.1, max_iter=200)
    model2 = RandomForestClassifier(n_estimators=100, max_depth=10,min_samples_split=5, random_state=seed)

    # TODO evaluate model using cross-validation
    scores1 = cross_val_score(model1, X_train, y_train, cv=4, scoring="accuracy")
    scores2 = cross_val_score(model2, X_train, y_train, cv=4, scoring="accuracy")

    print(f"Logistic Regression CV Accuracy: {scores1.mean():.4f} ± {scores1.std():.4f}")
    print(f"Random Forest CV Accuracy: {scores2.mean():.4f} ± {scores2.std():.4f}")

    # Fit the best model on the entire training set and get the predictions
    final_model1 = model1.fit(X_train, y_train)
    final_model2 = model2.fit(X_train, y_train)


    # TODO Evaluate the final predictions with the metric of your choice
    train_pred1 = final_model1.predict(X_train)
    test_pred1 = final_model1.predict(X_test)

    print("\n--- Logistic Regression Results ---")
    print("Training Accuracy:", accuracy_score(y_train, train_pred1))
    print("Test Accuracy:", accuracy_score(y_test, test_pred1))

    # Evaluate Random Forest
    train_pred2 = final_model2.predict(X_train)
    test_pred2 = final_model2.predict(X_test)

    print("\n--- Random Forest Results ---")
    print("Training Accuracy:", accuracy_score(y_train, train_pred2))
    print("Test Accuracy:", accuracy_score(y_test, test_pred2))
