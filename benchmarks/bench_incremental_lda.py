"""Benchmark incremental vs batch Linear Discriminant Analysis on large data."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from time import perf_counter
from typing import Iterable, List, Optional

import numpy as np

from sklearn.datasets import make_classification
from sklearn.discriminant_analysis import (
    IncrementalLinearDiscriminantAnalysis,
    LinearDiscriminantAnalysis,
)
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split


@dataclass
class LDAResult:
    estimator: str
    solver: str
    fit_time: float
    predict_time: float
    accuracy: float
    balanced_accuracy: float
    n_batches: int = 1
    prediction_agreement: Optional[float] = None
    accuracy_gap: Optional[float] = None
    max_decision_difference: Optional[float] = None
    _predictions: Optional[np.ndarray] = field(default=None, repr=False)
    _decision_values: Optional[np.ndarray] = field(default=None, repr=False)


def _parse_shrinkage(value: str) -> Optional[float | str]:
    if value in {None, "", "none", "None"}:
        return None
    if value == "auto":
        return value
    try:
        return float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "shrinkage must be 'auto', 'None', or a float"
        ) from exc


def _generate_dataset(
    *,
    n_samples: int,
    n_features: int,
    n_classes: int,
    test_size: float,
    random_state: int,
):
    n_informative = min(n_features, max(10, n_classes * 5))
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=0,
        n_repeated=0,
        n_classes=n_classes,
        n_clusters_per_class=1,
        class_sep=2.5,
        flip_y=0.0,
        random_state=random_state,
    )
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


def _format_value(value, *, width: int, precision: int = 4) -> str:
    if value is None:
        return f"{'-':>{width}}"
    if isinstance(value, float):
        return f"{value:>{width}.{precision}f}"
    return f"{value:>{width}}"


def _print_results(results: Iterable[LDAResult]) -> None:
    results = list(results)
    if not results:
        return

    include_agreement = any(r.prediction_agreement is not None for r in results)
    include_gap = any(r.accuracy_gap is not None for r in results)
    include_decision = any(r.max_decision_difference is not None for r in results)

    columns = [
        ("Estimator", "estimator", 22, False),
        ("Solver", "solver", 6, False),
        ("Fit time (s)", "fit_time", 14, True),
        ("Predict time (s)", "predict_time", 17, True),
        ("Accuracy", "accuracy", 10, True),
        ("Balanced acc.", "balanced_accuracy", 15, True),
        ("Batches", "n_batches", 8, False),
    ]
    if include_agreement:
        columns.append(("Agreement", "prediction_agreement", 12, True))
    if include_gap:
        columns.append(("Acc. gap", "accuracy_gap", 10, True))
    if include_decision:
        columns.append(("Max |Δ decision|", "max_decision_difference", 18, True))

    header = " | ".join(name.ljust(width) for name, _, width, _ in columns)
    print(header)
    print("-" * len(header))

    for res in results:
        row = []
        for _, attr, width, is_float in columns:
            value = getattr(res, attr)
            precision = 6 if attr in {"fit_time", "predict_time"} else 4
            row.append(_format_value(value, width=width, precision=precision))
        print(" | ".join(row))


def _make_solver_kwargs(
    solver: str, *, shrinkage: Optional[float | str], store_covariance: bool
):
    kwargs = {"solver": solver}
    if solver == "svd":
        if shrinkage not in {None, "auto"}:
            raise ValueError("shrinkage is not supported with the 'svd' solver")
        kwargs["store_covariance"] = store_covariance
    elif shrinkage is not None:
        kwargs["shrinkage"] = shrinkage
    return kwargs


def _time_partial_fit(
    estimator: IncrementalLinearDiscriminantAnalysis,
    X: np.ndarray,
    y: np.ndarray,
    *,
    batch_size: int,
    classes: np.ndarray,
):
    start = 0
    n_samples = X.shape[0]
    batch_count = 0
    tic = perf_counter()
    while start < n_samples:
        end = min(start + batch_size, n_samples)
        X_batch = X[start:end]
        y_batch = y[start:end]
        if start == 0:
            estimator.partial_fit(X_batch, y_batch, classes=classes)
        else:
            estimator.partial_fit(X_batch, y_batch)
        start = end
        batch_count += 1
    return perf_counter() - tic, batch_count


def benchmark_batch(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    solver: str,
    shrinkage: Optional[float | str],
    store_covariance: bool,
) -> LDAResult:
    lda = LinearDiscriminantAnalysis(
        **_make_solver_kwargs(
            solver, shrinkage=shrinkage, store_covariance=store_covariance
        )
    )
    tic = perf_counter()
    lda.fit(X_train, y_train)
    fit_time = perf_counter() - tic

    tic = perf_counter()
    predictions = lda.predict(X_test)
    predict_time = perf_counter() - tic

    accuracy = accuracy_score(y_test, predictions)
    balanced = balanced_accuracy_score(y_test, predictions)
    decision_values = lda.decision_function(X_test)

    return LDAResult(
        estimator="LinearDiscriminantAnalysis",
        solver=solver,
        fit_time=fit_time,
        predict_time=predict_time,
        accuracy=accuracy,
        balanced_accuracy=balanced,
        _predictions=predictions,
        _decision_values=decision_values,
    )


def benchmark_incremental(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    solver: str,
    shrinkage: Optional[float | str],
    store_covariance: bool,
    batch_size: int,
    baseline: Optional[LDAResult] = None,
) -> LDAResult:
    classes = np.unique(y_train)
    estimator = IncrementalLinearDiscriminantAnalysis(
        **_make_solver_kwargs(
            solver, shrinkage=shrinkage, store_covariance=store_covariance
        )
    )
    fit_time, n_batches = _time_partial_fit(
        estimator, X_train, y_train, batch_size=batch_size, classes=classes
    )

    tic = perf_counter()
    predictions = estimator.predict(X_test)
    predict_time = perf_counter() - tic

    accuracy = accuracy_score(y_test, predictions)
    balanced = balanced_accuracy_score(y_test, predictions)

    agreement = None
    accuracy_gap = None
    max_decision_difference = None

    if baseline is not None and baseline._predictions is not None:
        agreement = float(np.mean(predictions == baseline._predictions))
        incremental_decision = estimator.decision_function(X_test)
        if baseline._decision_values is not None:
            max_decision_difference = float(
                np.max(np.abs(incremental_decision - baseline._decision_values))
            )
        accuracy_gap = balanced - baseline.balanced_accuracy

    return LDAResult(
        estimator="IncrementalLinearDiscriminantAnalysis",
        solver=solver,
        fit_time=fit_time,
        predict_time=predict_time,
        accuracy=accuracy,
        balanced_accuracy=balanced,
        n_batches=n_batches,
        prediction_agreement=agreement,
        accuracy_gap=accuracy_gap,
        max_decision_difference=max_decision_difference,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--n-samples", type=int, default=500_000)
    parser.add_argument("--n-features", type=int, default=100)
    parser.add_argument("--n-classes", type=int, default=5)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument(
        "--solvers",
        nargs="+",
        default=["svd", "lsqr", "eigen"],
        choices=["svd", "lsqr", "eigen"],
        help="Solvers to benchmark.",
    )
    parser.add_argument(
        "--shrinkage",
        type=_parse_shrinkage,
        default=None,
        help="Shrinkage parameter for 'lsqr' and 'eigen' solvers.",
    )
    parser.add_argument(
        "--store-covariance",
        action="store_true",
        help="Store the pooled covariance when benchmarking the 'svd' solver.",
    )
    parser.add_argument("--random-state", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    X_train, X_test, y_train, y_test = _generate_dataset(
        n_samples=args.n_samples,
        n_features=args.n_features,
        n_classes=args.n_classes,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    print(
        "Dataset:",
        f"{X_train.shape[0] + X_test.shape[0]:,} samples",
        f"({X_train.shape[0]:,} train / {X_test.shape[0]:,} test),",
        f"{X_train.shape[1]} features, {len(np.unique(y_train))} classes",
    )
    print(f"Batch size: {args.batch_size:,}")

    results: List[LDAResult] = []
    for solver in args.solvers:
        shrinkage = args.shrinkage if solver != "svd" else None
        baseline = benchmark_batch(
            X_train,
            y_train,
            X_test,
            y_test,
            solver=solver,
            shrinkage=shrinkage,
            store_covariance=args.store_covariance,
        )
        results.append(baseline)
        incremental = benchmark_incremental(
            X_train,
            y_train,
            X_test,
            y_test,
            solver=solver,
            shrinkage=shrinkage,
            store_covariance=args.store_covariance,
            batch_size=args.batch_size,
            baseline=baseline,
        )
        results.append(incremental)

    print()
    _print_results(results)


if __name__ == "__main__":
    main()

