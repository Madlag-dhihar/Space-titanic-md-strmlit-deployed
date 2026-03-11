import mlflow
import optuna
import numpy as np
import pickle

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from mlflow.models.signature import infer_signature

RANDOM_STATE = 42

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Spaceship_Titanic")

def split_data(X, y):
    """Split training and validation data"""

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print("Training set:", X_train.shape)
    print("Validation set:", X_test.shape)

    return X_train, X_test, y_train, y_test


def train_baseline(X_train, y_train):

    with mlflow.start_run(run_name="LGR Baseline") as run:

        model = LogisticRegression(random_state=RANDOM_STATE)

        cv_scores = cross_val_score(
            model,
            X_train,
            y_train,
            cv=5,
            scoring="accuracy"
        )

        mean_acc = np.mean(cv_scores)
        std_acc = np.std(cv_scores)

        mlflow.log_metric("cv_mean_accuracy", mean_acc)
        mlflow.log_metric("cv_std_accuracy", std_acc)

        model.fit(X_train, y_train)

        signature = infer_signature(X_train, model.predict(X_train))

        mlflow.sklearn.log_model(
            model,
            "baseline_model",
            signature=signature
        )

        run_id = run.info.run_id

        print("Baseline CV Accuracy:", mean_acc)

    return model, run_id


def objective_LRG(trial, X_train, y_train):

    solver = trial.suggest_categorical("solver", ["lbfgs", "liblinear"])

    if solver == "lbfgs":
        penalty = "l2"
    else:
        penalty = trial.suggest_categorical("penalty", ["l1", "l2"])

    C = trial.suggest_float("C", 0.01, 10.0)
    max_iter = trial.suggest_int("max_iter", 100, 1000)

    model = LogisticRegression(
        penalty=penalty,
        C=C,
        solver=solver,
        max_iter=max_iter,
        random_state=RANDOM_STATE
    )

    scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=5,
        scoring="accuracy"
    )

    return scores.mean()


def tune_model(X_train, y_train):

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE)
    )

    study.optimize(
        lambda trial: objective_LRG(trial, X_train, y_train),
        n_trials=50
    )

    print("Best parameters:", study.best_params)

    best_model = LogisticRegression(
        **study.best_params,
        random_state=RANDOM_STATE
    )

    best_model.fit(X_train, y_train)

    return best_model


def save_model(best_model, feature_columns):
    """Save model and features for Streamlit"""

    with open("model.pkl", "wb") as f:
        pickle.dump(best_model, f)

    with open("features.pkl", "wb") as f:
        pickle.dump(feature_columns, f)

    print("Model saved as model.pkl")