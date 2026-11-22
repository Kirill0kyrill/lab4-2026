"""Реализация эксперимента с ансамблем моделей для лабораторной работы № 4."""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from grader_contracts.ensembles import EnsembleInput, EnsembleResult


def run_ensemble_experiment(data: EnsembleInput) -> EnsembleResult:
    """Обучить мягкий ансамбль Random Forest + Extra Trees и оценить его на test.

    Все обучаемые преобразования и обе базовые модели fit-ятся только на
    train-части. test_target используется исключительно для расчёта метрик.
    """
    train_features = data.train_features
    test_features = data.test_features
    train_target = np.asarray(data.train_target).reshape(-1)
    test_target = np.asarray(data.test_target).reshape(-1)
    seed = int(data.random_seed)

    if train_target.size != train_features.shape[0]:
        raise ValueError("Число объектов train_features и train_target не совпадает.")
    if test_target.size != test_features.shape[0]:
        raise ValueError("Число объектов test_features и test_target не совпадает.")
    if np.unique(train_target).size != 2:
        raise ValueError("Для обучения требуется бинарная целевая переменная с двумя классами.")
    if not np.any(train_target == 1):
        raise ValueError("Положительный класс должен иметь метку 1.")

    random_forest = RandomForestClassifier(
        n_estimators=300,
        max_features="sqrt",
        min_samples_leaf=4,
        class_weight="balanced_subsample",
        random_state=seed,
        n_jobs=1,
    )
    extra_trees = ExtraTreesClassifier(
        n_estimators=300,
        max_features="sqrt",
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=seed,
        n_jobs=1,
    )

    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "ensemble",
                VotingClassifier(
                    estimators=[
                        ("random_forest", random_forest),
                        ("extra_trees", extra_trees),
                    ],
                    voting="soft",
                    weights=[1, 1],
                    n_jobs=1,
                ),
            ),
        ]
    )

    # Важно: fit вызывается только с train_features и train_target.
    model.fit(train_features, train_target)

    fitted_ensemble = model.named_steps["ensemble"]
    classes = np.asarray(fitted_ensemble.classes_)
    positive_class_indices = np.flatnonzero(classes == 1)
    if positive_class_indices.size != 1:
        raise ValueError("После обучения не удалось однозначно определить положительный класс 1.")

    probabilities = model.predict_proba(test_features)[:, int(positive_class_indices[0])]
    predictions = (probabilities >= 0.5).astype(int)

    return EnsembleResult(
        model_name="Soft Voting: Random Forest + Extra Trees (weights 1:1)",
        probabilities=probabilities,
        predictions=predictions,
        roc_auc=float(roc_auc_score(test_target, probabilities)),
        accuracy=float(accuracy_score(test_target, predictions)),
        precision=float(precision_score(test_target, predictions, zero_division=0)),
        recall=float(recall_score(test_target, predictions, zero_division=0)),
        f1=float(f1_score(test_target, predictions, zero_division=0)),
    )
