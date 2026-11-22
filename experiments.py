"""Воспроизводимый скрипт сравнения моделей для лабораторной работы № 4.

Запуск: python experiments.py
Скрипт использует german.csv рядом с собой и сохраняет experiment_results.json.
Подбор выполняется только на train-части. Test используется один раз для
итоговой оценки модели, выбранной по средней ROC-AUC на кросс-валидации.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline

SEED = 123
TEST_SIZE = 0.20
N_SPLITS = 5
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "german.csv"
OUTPUT_PATH = BASE_DIR / "experiment_results.json"


def make_pipeline(classifier) -> Pipeline:
    """Поместить импутацию внутрь CV-конвейера, исключив утечку данных."""
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("model", classifier),
        ]
    )


def build_experiments(seed: int) -> list[tuple[str, Pipeline, dict]]:
    """Вернуть заранее определённые конфигурации, сравниваемые на CV."""
    candidates = [
        (
            "Random Forest: 300 деревьев, leaf=2",
            RandomForestClassifier(
                n_estimators=300, max_features="sqrt", min_samples_leaf=2,
                class_weight="balanced_subsample", random_state=seed, n_jobs=1,
            ),
            {"n_estimators": 300, "max_features": "sqrt", "min_samples_leaf": 2,
                "class_weight": "balanced_subsample"},
        ),
        (
            "Random Forest: 500 деревьев, leaf=4",
            RandomForestClassifier(
                n_estimators=500, max_features="sqrt", min_samples_leaf=4,
                class_weight="balanced_subsample", random_state=seed, n_jobs=1,
            ),
            {"n_estimators": 500, "max_features": "sqrt", "min_samples_leaf": 4,
                "class_weight": "balanced_subsample"},
        ),
        (
            "Extra Trees: sqrt, leaf=2",
            ExtraTreesClassifier(
                n_estimators=500, max_features="sqrt", min_samples_leaf=2,
                class_weight="balanced", random_state=seed, n_jobs=1,
            ),
            {"n_estimators": 500, "max_features": "sqrt", "min_samples_leaf": 2,
                "class_weight": "balanced"},
        ),
        (
            "Extra Trees: max_features=0.7, leaf=2",
            ExtraTreesClassifier(
                n_estimators=500, max_features=0.7, min_samples_leaf=2,
                random_state=seed, n_jobs=1,
            ),
            {"n_estimators": 500, "max_features": 0.7, "min_samples_leaf": 2},
        ),
        (
            "Extra Trees: max_features=0.7, leaf=4",
            ExtraTreesClassifier(
                n_estimators=500, max_features=0.7, min_samples_leaf=4,
                random_state=seed, n_jobs=1,
            ),
            {"n_estimators": 500, "max_features": 0.7, "min_samples_leaf": 4},
        ),
        (
            "Gradient Boosting: 150 итераций, lr=0.05, depth=2",
            GradientBoostingClassifier(
                n_estimators=150, learning_rate=0.05, max_depth=2,
                min_samples_leaf=5, random_state=seed,
            ),
            {"n_estimators": 150, "learning_rate": 0.05, "max_depth": 2,
                "min_samples_leaf": 5},
        ),
        (
            "Gradient Boosting: 200 итераций, lr=0.03, depth=2",
            GradientBoostingClassifier(
                n_estimators=200, learning_rate=0.03, max_depth=2,
                min_samples_leaf=5, random_state=seed,
            ),
            {"n_estimators": 200, "learning_rate": 0.03, "max_depth": 2,
                "min_samples_leaf": 5},
        ),
        (
            "Gradient Boosting: 150 итераций, lr=0.05, depth=1",
            GradientBoostingClassifier(
                n_estimators=150, learning_rate=0.05, max_depth=1,
                min_samples_leaf=5, random_state=seed,
            ),
            {"n_estimators": 150, "learning_rate": 0.05, "max_depth": 1,
                "min_samples_leaf": 5},
        ),
        (
            "HistGradientBoosting: 150 итераций, lr=0.05, 15 листьев",
            HistGradientBoostingClassifier(
                max_iter=150, learning_rate=0.05, max_leaf_nodes=15,
                l2_regularization=1.0, random_state=seed,
            ),
            {"max_iter": 150, "learning_rate": 0.05, "max_leaf_nodes": 15,
                "l2_regularization": 1.0},
        ),
    ]

    experiments = [
        (name, make_pipeline(classifier), params)
        for name, classifier, params in candidates
    ]

    # Мягкое голосование: вероятность класса усредняется с равными весами.
    rf = RandomForestClassifier(
        n_estimators=300, max_features="sqrt", min_samples_leaf=4,
        class_weight="balanced_subsample", random_state=seed, n_jobs=1,
    )
    et = ExtraTreesClassifier(
        n_estimators=300, max_features="sqrt", min_samples_leaf=2,
        class_weight="balanced", random_state=seed, n_jobs=1,
    )
    shared_estimators = [("random_forest", rf), ("extra_trees", et)]
    shared_parameters = {
        "random_forest": {
            "n_estimators": 300, "max_features": "sqrt",
            "min_samples_leaf": 4, "class_weight": "balanced_subsample",
        },
        "extra_trees": {
            "n_estimators": 300, "max_features": "sqrt",
            "min_samples_leaf": 2, "class_weight": "balanced",
        },
    }
    for weights in ([1, 1], [2, 1], [1, 2]):
        voting = VotingClassifier(
            estimators=shared_estimators,
            voting="soft", weights=list(weights), n_jobs=1,
        )
        experiments.append(
            (
                f"Soft Voting: Random Forest + Extra Trees (weights {weights[0]}:{weights[1]})",
                make_pipeline(voting),
                {"voting": "soft", "weights": list(weights), **shared_parameters},
            )
        )
    return experiments


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Не найден файл датасета: {DATA_PATH}")

    frame = pd.read_csv(DATA_PATH, sep=";")
    target_column = "Creditability"
    if target_column not in frame.columns:
        raise ValueError(f"В датасете отсутствует целевой столбец {target_column!r}.")

    features = frame.drop(columns=[target_column])
    target = frame[target_column].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=target,
    )
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    results = []
    fitted_candidates = []
    for name, pipeline, params in build_experiments(SEED):
        fold_results = cross_validate(
            pipeline,
            X_train,
            y_train,
            scoring={
                "roc_auc": "roc_auc",
                "accuracy": "accuracy",
                "precision": "precision",
                "recall": "recall",
                "f1": "f1",
            },
            cv=cv,
            n_jobs=1,
            return_train_score=False,
        )
        row = {
            "model_name": name,
            "parameters": params,
            "cv_roc_auc_mean": float(fold_results["test_roc_auc"].mean()),
            "cv_roc_auc_std": float(fold_results["test_roc_auc"].std()),
            "cv_accuracy_mean": float(fold_results["test_accuracy"].mean()),
            "cv_precision_mean": float(fold_results["test_precision"].mean()),
            "cv_recall_mean": float(fold_results["test_recall"].mean()),
            "cv_f1_mean": float(fold_results["test_f1"].mean()),
        }
        results.append(row)
        fitted_candidates.append((float(fold_results["test_roc_auc"].mean()), name, pipeline, params))
        print(
            f"{name}: CV ROC-AUC={fold_results['test_roc_auc'].mean():.4f} "
            f"(+/- {fold_results['test_roc_auc'].std():.4f})",
            flush=True,
        )

    # Выбор строго по среднему ROC-AUC кросс-валидации на train.
    _, chosen_name, chosen_pipeline, chosen_params = max(
        fitted_candidates, key=lambda item: item[0]
    )
    chosen_pipeline.fit(X_train, y_train)
    test_probabilities = chosen_pipeline.predict_proba(X_test)[:, 1]

    test_predictions = (test_probabilities >= 0.5).astype(int)
    test_metrics = {
        "roc_auc": float(roc_auc_score(y_test, test_probabilities)),
        "accuracy": float(accuracy_score(y_test, test_predictions)),
        "precision": float(precision_score(y_test, test_predictions, zero_division=0)),
        "recall": float(recall_score(y_test, test_predictions, zero_division=0)),
        "f1": float(f1_score(y_test, test_predictions, zero_division=0)),
    }

    output = {
        "dataset": {
            "file": "german.csv",
            "rows": int(len(frame)),
            "features": int(features.shape[1]),
            "target_column": target_column,
            "positive_class": 1,
            "class_counts": {
                str(label): int(count)
                for label, count in target.value_counts().sort_index().items()
            },
            "missing_cells": int(frame.isna().sum().sum()),
        },
        "split": {
            "method": "train_test_split",
            "test_size": TEST_SIZE,
            "random_state": SEED,
            "stratified": True,
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
        },
        "cross_validation": {
            "method": "StratifiedKFold",
            "n_splits": N_SPLITS,
            "shuffle": True,
            "random_state": SEED,
            "selection_metric": "mean ROC-AUC",
            "test_target_used_for_selection": False,
        },
        "experiments": sorted(
            results, key=lambda row: row["cv_roc_auc_mean"], reverse=True
        ),
        "selected_model": {
            "model_name": chosen_name,
            "selection_reason": "Максимальный средний ROC-AUC на 5-fold CV по train.",
            "parameters": chosen_params,
        },
        "final_test_metrics": test_metrics,
        "final_test_confusion_matrix": confusion_matrix(
            y_test, test_predictions
        ).tolist(),
        "final_test_prediction_counts": {
            str(label): int(count)
            for label, count in pd.Series(test_predictions).value_counts().sort_index().items()
        },
    }

    OUTPUT_PATH.write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nВыбрана модель: {chosen_name}")
    print("Итоговые метрики на тестовой выборке:")
    for metric_name, metric_value in test_metrics.items():
        print(f"  {metric_name}: {metric_value:.4f}")
    print(f"Результаты сохранены в {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
