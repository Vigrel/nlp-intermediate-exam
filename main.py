import itertools
import time

import mlflow
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from nltk.stem import WordNetLemmatizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.naive_bayes import BernoulliNB
from sklearn.pipeline import Pipeline
from sklearn.utils import parallel_backend


def load_process_data() -> pd.DataFrame:
    lemmatizer = WordNetLemmatizer()

    train_df = pd.read_csv("data/fake-news/train.csv", index_col="id")
    test_df = pd.read_csv("data/fake-news/test.csv", index_col="id")
    test_df_label = pd.read_csv("data/fake-news/submit.csv", index_col="id")

    test_df = test_df.join(test_df_label, on="id")
    data = pd.concat([train_df, test_df])
    data["text"] = data.text.values.astype("U")
    data["label"] = data.label.values.astype("U")

    def preprocess_text(text):
        return " ".join([lemmatizer.lemmatize(word) for word in text.split()])

    data["text"] = data.text.apply(preprocess_text)

    return data[:, ["text", "label"]]

# Evaluate a single pipeline using cross-validation
def evaluate_pipeline(pipeline, X, y, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    accuracies, precisions, recalls, f1_scores = [], [], [], []
    start_time = time.time()
    
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        
        # Compute and store metrics
        accuracies.append(accuracy_score(y_test, y_pred))
        precisions.append(precision_score(y_test, y_pred, average="macro"))
        recalls.append(recall_score(y_test, y_pred, average="macro"))
        f1_scores.append(f1_score(y_test, y_pred, average="macro"))
    
    training_time = time.time() - start_time
    return {
        "mean_accuracy": np.mean(accuracies),
        "std_accuracy": np.std(accuracies),
        "mean_precision": np.mean(precisions),
        "mean_recall": np.mean(recalls),
        "mean_f1": np.mean(f1_scores),
        "training_time": training_time
    }

# Classifiers and vectorizers setup
classifiers = {
    "RandomForestClassifier": RandomForestClassifier(),
    "LogisticRegression": LogisticRegression(),
    "BernoulliNB": BernoulliNB(),
}

vectorizers = {
    "CVBinaryIgnoreCase": CountVectorizer(lowercase=False, stop_words="english", binary=True),
    "CVBinary": CountVectorizer(stop_words="english", binary=True),
    "CV": CountVectorizer(stop_words="english"),
    "TfidfVIgnoreCase": TfidfVectorizer(lowercase=False, stop_words="english", binary=True),
    "TfidfVBinary": TfidfVectorizer(stop_words="english", binary=True),
    "TfidfV": TfidfVectorizer(stop_words="english"),
    "TfidfVNoNorm": TfidfVectorizer(stop_words="english", norm=None),
    "TfidfVL1": TfidfVectorizer(stop_words="english", norm="l1"),
}

# Parameter grids for hyperparameter tuning
param_grids = {
    "RandomForestClassifier": {"n_estimators": [100, 200], "max_depth": [10, 20]},
    "LogisticRegression": {"C": [0.1, 1.0, 10.0]},
    "BernoulliNB": {"alpha": [0.5, 1.0]},
}

# Train and log results with MLFlow
def run_experiments(data, n_splits=5, n_jobs=-1):
    X, y = data["text"], data["label"]
    results = []

    with parallel_backend('threading', n_jobs=n_jobs):
        for clf_name, clf in classifiers.items():
            for vec_name, vec in vectorizers.items():
                param_grid = param_grids[clf_name]
                keys, values = zip(*param_grid.items())

                for v in [dict(zip(keys, combination)) for combination in itertools.product(*values)]:
                    with mlflow.start_run(run_name=f"{clf_name} + {vec_name}"):
                        pipe = Pipeline([(vec_name, vec), (clf_name, clf.set_params(**v))])
                        
                        # Evaluate model
                        metrics = evaluate_pipeline(pipe, X, y, n_splits=n_splits)

                        # Log parameters and metrics in MLFlow
                        mlflow.log_param("classifier", clf_name)
                        mlflow.log_param("vectorizer", vec_name)
                        mlflow.log_params(v)
                        mlflow.log_metric("mean_accuracy", metrics["mean_accuracy"])
                        mlflow.log_metric("std_accuracy", metrics["std_accuracy"])
                        mlflow.log_metric("mean_precision", metrics["mean_precision"])
                        mlflow.log_metric("mean_recall", metrics["mean_recall"])
                        mlflow.log_metric("mean_f1", metrics["mean_f1"])
                        mlflow.log_metric("training_time", metrics["training_time"])

                        # Log the model itself
                        mlflow.sklearn.log_model(pipe, artifact_path="model")
                        
                        # Store results for selecting the best model
                        results.append((pipe, metrics["mean_accuracy"], clf_name, vec_name, v))

    # Find and log the best model
    best_pipeline, best_accuracy, best_clf, best_vec, best_params = max(results, key=lambda x: x[1])

    with mlflow.start_run(run_name="Best Model"):
        mlflow.log_param("best_classifier", best_clf)
        mlflow.log_param("best_vectorizer", best_vec)
        mlflow.log_params(best_params)
        mlflow.log_metric("best_accuracy", best_accuracy)
        
        mlflow.sklearn.log_model(best_pipeline, artifact_path="best_model")

if __name__ == "__main__":
    # data = load_process_data()
    data = pd.read_csv("exp.csv")
    data["text"] = data['text'].astype("U") 
    # Run experiments
    run_experiments(data, n_splits=5, n_jobs=-1)
