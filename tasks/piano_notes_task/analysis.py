"""Grouped development analysis for the fixed C4/E4 readout."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def make_classifier():
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000, random_state=0),
    )


def grouped_folds(labels: Sequence[str], recording_ids: Sequence[str], condition_groups: Sequence[str], n_splits: int = 5):
    labels = np.asarray(labels)
    recording_ids = np.asarray(recording_ids)
    condition_groups = np.asarray(condition_groups)
    if not (len(labels) == len(recording_ids) == len(condition_groups)):
        raise ValueError("labels, recording_ids, and condition_groups must have equal lengths")
    if len(np.unique(condition_groups)) < n_splits:
        raise ValueError("not enough paired condition groups for requested folds")
    return list(StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=0).split(np.zeros((len(labels), 1)), labels, condition_groups))


@dataclass(frozen=True)
class ClassificationMetrics:
    accuracy: float
    balanced_accuracy: float
    roc_auc: float
    confusion: np.ndarray
    class_labels: np.ndarray
    true_labels: np.ndarray
    predicted_labels: np.ndarray
    probabilities: np.ndarray
    recall_by_class: dict[str, float]


def cross_validated_predictions(X: np.ndarray, labels: Sequence[str], folds):
    labels = np.asarray(labels)
    predictions = np.empty(len(labels), dtype=labels.dtype)
    probabilities = np.zeros((len(labels), len(np.unique(labels))), dtype=float)
    class_labels = np.unique(labels)
    for train, test in folds:
        model = make_classifier()
        model.fit(X[train], labels[train])
        predictions[test] = model.predict(X[test])
        local = model.predict_proba(X[test])
        for column, label in enumerate(model.classes_):
            probabilities[test, int(np.flatnonzero(class_labels == label)[0])] = local[:, column]
    return predictions, probabilities, class_labels


def evaluate_features(X: np.ndarray, labels: Sequence[str], folds) -> ClassificationMetrics:
    labels = np.asarray(labels)
    predictions, probabilities, class_labels = cross_validated_predictions(X, labels, folds)
    try:
        auc = float(roc_auc_score((labels == "E4").astype(int), probabilities[:, list(class_labels).index("E4")]))
    except ValueError:
        auc = float("nan")
    return ClassificationMetrics(
        accuracy=float(accuracy_score(labels, predictions)),
        balanced_accuracy=float(balanced_accuracy_score(labels, predictions)),
        roc_auc=auc, confusion=confusion_matrix(labels, predictions, labels=class_labels),
        class_labels=class_labels, true_labels=labels, predicted_labels=predictions,
        probabilities=probabilities,
        recall_by_class={str(label): float(np.mean(predictions[labels == label] == label)) for label in class_labels},
    )


def paired_label_permutation(labels: Sequence[str], condition_groups: Sequence[str], rng: np.random.Generator) -> np.ndarray:
    labels = np.asarray(labels, dtype=str).copy()
    groups = np.asarray(condition_groups, dtype=str)
    output = labels.copy()
    for group in np.unique(groups):
        indices = np.flatnonzero(groups == group)
        values, counts = np.unique(labels[indices], return_counts=True)
        if set(values) != {"C4", "E4"} or len(set(counts.tolist())) != 1:
            raise ValueError(f"condition group {group!r} must contain balanced C4/E4 repeats")
        if bool(rng.integers(0, 2)):
            output[indices] = np.where(labels[indices] == "C4", "E4", "C4")
    return output


def permutation_p_value(X: np.ndarray, labels: Sequence[str], condition_groups: Sequence[str], folds, observed: float, permutations: int = 100, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(permutations):
        shuffled = paired_label_permutation(labels, condition_groups, rng)
        shuffled_folds = grouped_folds(shuffled, np.arange(len(shuffled)), condition_groups, n_splits=len(folds))
        scores.append(evaluate_features(X, shuffled, shuffled_folds).balanced_accuracy)
    return float((1 + np.sum(np.asarray(scores) >= observed)) / (permutations + 1))


def grouped_bootstrap_ci(X: np.ndarray, labels: Sequence[str], condition_groups: Sequence[str], folds, repetitions: int = 2000, seed: int = 0) -> tuple[float, float]:
    labels = np.asarray(labels)
    groups = np.asarray(condition_groups)
    rng = np.random.default_rng(seed)
    unique = np.unique(groups)
    scores = []
    for _ in range(repetitions):
        chosen = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([np.flatnonzero(groups == group) for group in chosen])
        if len(np.unique(labels[indices])) < 2:
            continue
        # Bootstrap is an uncertainty estimate around the fixed pipeline; fit/evaluate on sampled groups.
        try:
            sampled_folds = grouped_folds(labels[indices], np.arange(len(indices)), groups[indices], n_splits=min(5, len(np.unique(groups[indices]))))
            scores.append(evaluate_features(X[indices], labels[indices], sampled_folds).balanced_accuracy)
        except ValueError:
            continue
    if not scores:
        return float("nan"), float("nan")
    return tuple(np.percentile(scores, [2.5, 97.5]).astype(float))


def holm_bonferroni(p_values: Sequence[float]) -> np.ndarray:
    p_values = np.asarray(p_values, dtype=float)
    order = np.argsort(p_values)
    adjusted = np.empty_like(p_values)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(p_values) - rank) * p_values[index]))
        adjusted[index] = running
    return adjusted


def recording_level_probabilities(probabilities: np.ndarray, recording_ids: Sequence[str], labels: Sequence[str] | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    probabilities = np.asarray(probabilities, dtype=float)
    recording_ids = np.asarray(recording_ids, dtype=str)
    if len(recording_ids) != len(probabilities):
        raise ValueError("probabilities and recording_ids must have equal lengths")
    unique = np.unique(recording_ids)
    means = np.stack([probabilities[recording_ids == recording_id].mean(axis=0) for recording_id in unique])
    if labels is None:
        return unique, means, None
    labels = np.asarray(labels, dtype=str)
    recording_labels = np.asarray([np.unique(labels[recording_ids == recording_id])[0] for recording_id in unique])
    return unique, means, recording_labels


def external_recording_metrics(model, feature_dataset) -> dict:
    """Score frozen external recordings, averaging probabilities across repeats."""
    probabilities = model.predict_proba(feature_dataset.X)
    ids, means, labels = recording_level_probabilities(probabilities, feature_dataset.recording_ids, feature_dataset.labels)
    predictions = model.classes_[np.argmax(means, axis=1)]
    return {
        "recording_ids": ids.tolist(), "n_recordings": int(len(ids)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "class_labels": model.classes_.tolist(), "predicted_labels": predictions.tolist(),
        "note": "frozen development model; no retraining or tuning",
    }


def development_memory_horizon(windows_ms: Sequence[tuple[float, float]], balanced_accuracies: Sequence[float], confidence_intervals: Sequence[tuple[float, float]], corrected_p_values: Sequence[float]) -> float | None:
    """Return the end of the last contiguous significant >50% window."""
    horizon = None
    expected_start = 5.0
    for window, score, interval, p_value in zip(windows_ms, balanced_accuracies, confidence_intervals, corrected_p_values):
        start, end = map(float, window)
        if not np.isclose(start, expected_start) or score <= 0.5 or interval[0] <= 0.5 or p_value >= 0.05:
            break
        horizon = end
        expected_start = end
    return horizon
