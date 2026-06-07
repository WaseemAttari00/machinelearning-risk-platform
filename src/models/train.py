"""
Training entry point for both risk prediction domains.

Orchestrates: data ingestion → validation → feature engineering →
train/test split → baseline model → Optuna-tuned XGBoost →
MLflow logging → model serialization → SHAP analysis.

Usage:
    python -m src.models.train --domain credit_risk
    python -m src.models.train --domain network_intrusion
    python -m src.models.train --domain all
"""

import argparse
from pathlib import Path

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
import optuna
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

from src.data.ingestion import (
    load_credit_risk,
    load_network_intrusion,
    save_processed,
)
from src.data.preprocessing import (
    make_train_test_split,
    save_pipeline,
    split_features_target,
)
from src.data.validation import validate_dataframe
from src.features.credit_features import build_credit_pipeline, get_feature_names
from src.features.network_features import (
    binarize_labels,
    build_network_pipeline,
    get_feature_names_after_encoding,
)
from src.models.evaluate import evaluate_model, save_evaluation_report
from src.explainability.shap_analysis import run_shap_analysis
from src.utils.config import get_project_root, load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

PROJECT_ROOT = get_project_root()


def train_credit_risk() -> None:
    """Full training pipeline for the credit risk domain."""
    cfg = load_config("credit_risk")

    # ── Load and validate ─────────────────────────────────────────────────────
    df = load_credit_risk(cfg)

    all_features = cfg["features"]["numeric_features"]
    report = validate_dataframe(
        df=df,
        expected_columns=all_features + [cfg["data"]["target_column"]],
        target_column=cfg["data"]["target_column"],
        domain="credit_risk",
    )
    if not report["passed"]:
        raise RuntimeError(f"Data validation failed: {report['errors']}")

    # ── Feature / target split ────────────────────────────────────────────────
    X, y = split_features_target(
        df,
        target_column=cfg["data"]["target_column"],
        drop_columns=cfg["features"].get("drop_columns", []),
    )

    X_train, X_test, y_train, y_test = make_train_test_split(
        X, y,
        test_size=cfg["data"]["test_size"],
        random_state=cfg["data"]["random_state"],
        stratify=True,
    )

    # ── Feature engineering (fit on train only) ───────────────────────────────
    pipeline = build_credit_pipeline()
    X_train_proc = pipeline.fit_transform(X_train)
    X_test_proc = pipeline.transform(X_test)

    feature_names = get_feature_names(cfg)

    train_df = pd.DataFrame(X_train_proc, columns=feature_names)
    train_df[cfg["data"]["target_column"]] = y_train.values
    test_df = pd.DataFrame(X_test_proc, columns=feature_names)
    test_df[cfg["data"]["target_column"]] = y_test.values
    save_processed(train_df, cfg["data"]["processed_train_path"])
    save_processed(test_df, cfg["data"]["processed_test_path"])

    pipeline_path = str(PROJECT_ROOT / "models" / "credit_risk" / "preprocessing_pipeline.joblib")
    save_pipeline(pipeline, pipeline_path)

    # ── MLflow setup ──────────────────────────────────────────────────────────
    mlruns_path = PROJECT_ROOT / cfg["mlflow"]["tracking_uri"]
    mlruns_path.mkdir(exist_ok=True)
    mlflow.set_tracking_uri(mlruns_path.as_uri())
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

    # ── Baseline: Logistic Regression ─────────────────────────────────────────
    logger.info("Training baseline Logistic Regression model...")
    with mlflow.start_run(run_name="baseline_logistic_regression"):
        baseline_params = cfg["model"]["baseline"]["params"]
        baseline_model = LogisticRegression(**baseline_params)
        baseline_model.fit(X_train_proc, y_train)

        baseline_metrics = evaluate_model(
            model=baseline_model,
            X_test=X_test_proc,
            y_test=y_test,
            feature_names=feature_names,
            domain="credit_risk",
            model_name="baseline_lr",
        )

        mlflow.log_params(baseline_params)
        mlflow.log_metrics(baseline_metrics["scalar_metrics"])
        mlflow.sklearn.log_model(baseline_model, "model")

        logger.info(
            "Baseline ROC-AUC: {auc:.4f}",
            auc=baseline_metrics["scalar_metrics"]["roc_auc"],
        )

    # ── Optuna hyperparameter search ──────────────────────────────────────────
    logger.info("Starting Optuna hyperparameter search for XGBoost...")

    tuning_cfg = cfg["tuning"]
    ss = tuning_cfg["search_space"]

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", ss["n_estimators"][0], ss["n_estimators"][1]),
            "max_depth": trial.suggest_int("max_depth", ss["max_depth"][0], ss["max_depth"][1]),
            "learning_rate": trial.suggest_float("learning_rate", ss["learning_rate"][0], ss["learning_rate"][1], log=True),
            "subsample": trial.suggest_float("subsample", ss["subsample"][0], ss["subsample"][1]),
            "colsample_bytree": trial.suggest_float("colsample_bytree", ss["colsample_bytree"][0], ss["colsample_bytree"][1]),
            "min_child_weight": trial.suggest_int("min_child_weight", ss["min_child_weight"][0], ss["min_child_weight"][1]),
            "scale_pos_weight": (y_train == 0).sum() / (y_train == 1).sum(),
            "random_state": cfg["data"]["random_state"],
            "eval_metric": "auc",
            "use_label_encoder": False,
        }

        model = XGBClassifier(**params, verbosity=0)
        model.fit(
            X_train_proc, y_train,
            eval_set=[(X_test_proc, y_test)],
            verbose=False,
        )

        y_prob = model.predict_proba(X_test_proc)[:, 1]
        return roc_auc_score(y_test, y_prob)

    study = optuna.create_study(
        direction=tuning_cfg["direction"],
        sampler=optuna.samplers.TPESampler(seed=cfg["data"]["random_state"]),
    )
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=tuning_cfg["n_trials"])

    logger.info("Best trial: ROC-AUC = {auc:.4f}", auc=study.best_value)
    logger.info("Best params: {params}", params=study.best_params)

    # ── Final XGBoost with best params ────────────────────────────────────────
    best_params = study.best_params
    best_params["scale_pos_weight"] = (y_train == 0).sum() / (y_train == 1).sum()
    best_params["random_state"] = cfg["data"]["random_state"]
    best_params["eval_metric"] = "auc"

    with mlflow.start_run(run_name="xgboost_tuned"):
        xgb_model = XGBClassifier(**best_params, verbosity=0)
        xgb_model.fit(
            X_train_proc, y_train,
            eval_set=[(X_test_proc, y_test)],
            verbose=False,
        )

        metrics = evaluate_model(
            model=xgb_model,
            X_test=X_test_proc,
            y_test=y_test,
            feature_names=feature_names,
            domain="credit_risk",
            model_name="xgboost_tuned",
        )

        mlflow.log_params(best_params)
        mlflow.log_metrics(metrics["scalar_metrics"])
        mlflow.log_param("optuna_n_trials", tuning_cfg["n_trials"])
        mlflow.log_metric("optuna_best_auc", study.best_value)
        mlflow.xgboost.log_model(xgb_model, "model")

        model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
        mlflow.register_model(model_uri, cfg["mlflow"]["model_name"])

        logger.info("XGBoost tuned — ROC-AUC: {auc:.4f}", auc=metrics["scalar_metrics"]["roc_auc"])

        import joblib
        model_path = str(PROJECT_ROOT / "models" / "credit_risk" / "xgboost_model.joblib")
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(xgb_model, model_path)
        logger.info("Model saved to {path}", path=model_path)

        save_evaluation_report(metrics, "credit_risk")

    logger.info("Running SHAP analysis for credit risk...")
    run_shap_analysis(
        model=xgb_model,
        X_test=X_test_proc,
        feature_names=feature_names,
        domain="credit_risk",
        sample_size=cfg["shap"]["sample_size"],
    )

    logger.info("Credit risk training complete.")


def train_network_intrusion() -> None:
    """
    Full training pipeline for the network intrusion detection domain.

    Mirrors train_credit_risk() but handles NSL-KDD specifics:
    pre-split train/test files, string label binarization, and
    mixed categorical + numeric features.
    """
    cfg = load_config("network_intrusion")

    # ── Load and preprocess ───────────────────────────────────────────────────
    train_df, test_df = load_network_intrusion(cfg)

    for col in cfg["features"].get("drop_columns", []):
        for df in [train_df, test_df]:
            if col in df.columns:
                df.drop(columns=[col], inplace=True)

    target_col = cfg["data"]["target_column"]
    label_map = cfg["features"]["binary_label_map"]
    y_train = binarize_labels(train_df[target_col], label_map)
    y_test = binarize_labels(test_df[target_col], label_map)

    numeric_features = cfg["features"]["numeric_features"]
    categorical_features = cfg["features"]["categorical_features"]
    all_features = numeric_features + categorical_features

    validate_dataframe(
        df=train_df,
        expected_columns=all_features,
        target_column=target_col,
        domain="network_intrusion",
    )

    X_train = train_df[all_features]
    X_test = test_df[all_features]

    # ── Feature engineering (fit on train only) ───────────────────────────────
    pipeline = build_network_pipeline(numeric_features, categorical_features)
    X_train_proc = pipeline.fit_transform(X_train)
    X_test_proc = pipeline.transform(X_test)

    feature_names = get_feature_names_after_encoding(
        pipeline, numeric_features, categorical_features
    )

    pipeline_path = str(PROJECT_ROOT / "models" / "network_intrusion" / "preprocessing_pipeline.joblib")
    save_pipeline(pipeline, pipeline_path)

    # ── MLflow setup ──────────────────────────────────────────────────────────
    mlruns_path = PROJECT_ROOT / cfg["mlflow"]["tracking_uri"]
    mlruns_path.mkdir(exist_ok=True)
    mlflow.set_tracking_uri(mlruns_path.as_uri())
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

    # ── Baseline: Logistic Regression ─────────────────────────────────────────
    logger.info("Training baseline Logistic Regression model...")
    with mlflow.start_run(run_name="baseline_logistic_regression"):
        baseline_params = cfg["model"]["baseline"]["params"]
        baseline_model = LogisticRegression(**baseline_params)
        baseline_model.fit(X_train_proc, y_train)

        baseline_metrics = evaluate_model(
            model=baseline_model,
            X_test=X_test_proc,
            y_test=y_test,
            feature_names=feature_names,
            domain="network_intrusion",
            model_name="baseline_lr",
        )
        mlflow.log_params(baseline_params)
        mlflow.log_metrics(baseline_metrics["scalar_metrics"])
        mlflow.sklearn.log_model(baseline_model, "model")

    # ── Optuna hyperparameter search ──────────────────────────────────────────
    tuning_cfg = cfg["tuning"]
    ss = tuning_cfg["search_space"]

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", ss["n_estimators"][0], ss["n_estimators"][1]),
            "max_depth": trial.suggest_int("max_depth", ss["max_depth"][0], ss["max_depth"][1]),
            "learning_rate": trial.suggest_float("learning_rate", ss["learning_rate"][0], ss["learning_rate"][1], log=True),
            "subsample": trial.suggest_float("subsample", ss["subsample"][0], ss["subsample"][1]),
            "colsample_bytree": trial.suggest_float("colsample_bytree", ss["colsample_bytree"][0], ss["colsample_bytree"][1]),
            "min_child_weight": trial.suggest_int("min_child_weight", ss["min_child_weight"][0], ss["min_child_weight"][1]),
            "random_state": cfg["data"]["random_state"],
        }
        model = XGBClassifier(**params, verbosity=0)
        model.fit(X_train_proc, y_train, eval_set=[(X_test_proc, y_test)], verbose=False)
        y_prob = model.predict_proba(X_test_proc)[:, 1]
        return roc_auc_score(y_test, y_prob)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction=tuning_cfg["direction"],
        sampler=optuna.samplers.TPESampler(seed=cfg["data"]["random_state"]),
    )
    study.optimize(objective, n_trials=tuning_cfg["n_trials"])
    logger.info("Best trial: ROC-AUC = {auc:.4f}", auc=study.best_value)

    # ── Final XGBoost with best params ────────────────────────────────────────
    best_params = {**study.best_params, "random_state": cfg["data"]["random_state"]}

    with mlflow.start_run(run_name="xgboost_tuned"):
        xgb_model = XGBClassifier(**best_params, verbosity=0)
        xgb_model.fit(X_train_proc, y_train, eval_set=[(X_test_proc, y_test)], verbose=False)

        metrics = evaluate_model(
            model=xgb_model,
            X_test=X_test_proc,
            y_test=y_test,
            feature_names=feature_names,
            domain="network_intrusion",
            model_name="xgboost_tuned",
        )

        mlflow.log_params(best_params)
        mlflow.log_metrics(metrics["scalar_metrics"])
        mlflow.xgboost.log_model(xgb_model, "model")
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
        mlflow.register_model(model_uri, cfg["mlflow"]["model_name"])

        import joblib
        model_path = str(PROJECT_ROOT / "models" / "network_intrusion" / "xgboost_model.joblib")
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(xgb_model, model_path)

        save_evaluation_report(metrics, "network_intrusion")

    logger.info("Running SHAP analysis for network intrusion...")
    run_shap_analysis(
        model=xgb_model,
        X_test=X_test_proc,
        feature_names=feature_names,
        domain="network_intrusion",
        sample_size=cfg["shap"]["sample_size"],
    )

    logger.info("Network intrusion training complete.")


def main():
    parser = argparse.ArgumentParser(description="Train risk prediction models.")
    parser.add_argument(
        "--domain",
        choices=["credit_risk", "network_intrusion", "all"],
        required=True,
        help="Which domain to train. Use 'all' to train both sequentially.",
    )
    args = parser.parse_args()

    if args.domain == "credit_risk":
        train_credit_risk()
    elif args.domain == "network_intrusion":
        train_network_intrusion()
    elif args.domain == "all":
        train_credit_risk()
        train_network_intrusion()


if __name__ == "__main__":
    main()
