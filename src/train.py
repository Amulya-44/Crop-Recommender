"""Train, compare and save the best crop-recommendation model.

Usage:  python -m src.train
Mirrors the notebook: 70/10/20 stratified split, GridSearchCV per algorithm,
model chosen on validation accuracy, test set used once at the end.
"""
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.features import add_features

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "soil_measures.csv"
MODEL_PATH = ROOT / "models" / "crop_model.joblib"
METRICS_PATH = ROOT / "models" / "metrics.json"


def candidates():
    return {
        "Logistic Regression": (
            Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=2000, random_state=42))]),
            {"model__C": [0.1, 1, 10]},
        ),
        "KNN": (
            Pipeline([("scaler", StandardScaler()), ("model", KNeighborsClassifier())]),
            {"model__n_neighbors": [3, 5, 7, 9], "model__weights": ["uniform", "distance"]},
        ),
        "SVM": (
            Pipeline([("scaler", StandardScaler()), ("model", SVC())]),
            {"model__C": [0.1, 1, 10], "model__kernel": ["rbf", "linear"]},
        ),
        "Random Forest": (
            RandomForestClassifier(random_state=42),
            {"n_estimators": [100, 200], "max_depth": [10, 20, None],
             "min_samples_leaf": [1, 2, 4], "max_features": ["sqrt", "log2"]},
        ),
        "Extra Trees": (
            ExtraTreesClassifier(random_state=42),
            {"n_estimators": [100, 200], "max_depth": [10, 20, None],
             "min_samples_leaf": [1, 2, 4], "max_features": ["sqrt", "log2"]},
        ),
        "Gradient Boosting": (
            GradientBoostingClassifier(random_state=42),
            {"n_estimators": [100, 150], "learning_rate": [0.05, 0.1], "max_depth": [2, 3, 4]},
        ),
    }


def main():
    df = add_features(pd.read_csv(DATA))
    X, y = df.drop(columns=["crop"]), df["crop"]

    X_tmp, X_test, y_tmp, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_tmp, y_tmp, test_size=0.125, random_state=42, stratify=y_tmp)

    results, fitted = [], {}
    for name, (estimator, grid) in candidates().items():
        print(f"Tuning {name}...")
        search = GridSearchCV(estimator, grid, cv=3, scoring="accuracy", n_jobs=-1).fit(X_train, y_train)
        fitted[name] = search.best_estimator_
        results.append({
            "algorithm": name,
            "cv_accuracy": round(search.best_score_, 4),
            "train_accuracy": round(accuracy_score(y_train, search.best_estimator_.predict(X_train)), 4),
            "validation_accuracy": round(accuracy_score(y_val, search.best_estimator_.predict(X_val)), 4),
            "best_params": search.best_params_,
        })

    results.sort(key=lambda r: r["validation_accuracy"], reverse=True)
    best_name = results[0]["algorithm"]
    best = fitted[best_name]

    test_pred = best.predict(X_test)
    test_acc = accuracy_score(y_test, test_pred)
    print(f"\nBest: {best_name} | test accuracy: {test_acc:.4f}")
    print(classification_report(y_test, test_pred))

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(best, MODEL_PATH, compress=3)
    METRICS_PATH.write_text(json.dumps(
        {"best_model": best_name, "test_accuracy": round(test_acc, 4), "comparison": results}, indent=2))
    print(f"Saved {MODEL_PATH.name} and {METRICS_PATH.name}")


if __name__ == "__main__":
    main()
