"""Benchmarks for discriminant analysis estimators."""
from __future__ import annotations

import numpy as np
from sklearn.discriminant_analysis import (
    IncrementalLinearDiscriminantAnalysis,
    LinearDiscriminantAnalysis,
)
from sklearn.metrics import balanced_accuracy_score

from .common import Benchmark, Estimator, Predictor
from .datasets import _incremental_lda_dataset
from .utils import make_gen_classif_scorers


class LinearDiscriminantAnalysisBenchmark(Predictor, Estimator, Benchmark):
    """Benchmarks for the batch LinearDiscriminantAnalysis estimator."""

    param_names = ["solver"]
    params = (["svd", "lsqr", "eigen"],)

    def make_data(self, params):
        (solver,) = params
        if Benchmark.data_size == "large":
            n_samples = 200_000
            n_features = 100
        else:
            n_samples = 60_000
            n_features = 60
        data = _incremental_lda_dataset(
            n_samples=n_samples,
            n_features=n_features,
            n_classes=5,
        )
        return data

    def make_estimator(self, params):
        (solver,) = params
        kwargs = {"solver": solver}
        if solver == "svd":
            kwargs["store_covariance"] = False
        return LinearDiscriminantAnalysis(**kwargs)

    def make_scorers(self):
        make_gen_classif_scorers(self)


class IncrementalLinearDiscriminantAnalysisBenchmark(Benchmark):
    """Benchmarks incremental fitting and accuracy of incremental LDA."""

    param_names = ["solver", "batch_size"]
    params = (
        ["svd", "lsqr", "eigen"],
        [512, 2048] if Benchmark.data_size == "small" else [2048, 8192],
    )

    def setup(self, solver, batch_size):
        if Benchmark.data_size == "large":
            n_samples = 200_000
            n_features = 100
        else:
            n_samples = 60_000
            n_features = 60

        self.X, self.X_val, self.y, self.y_val = _incremental_lda_dataset(
            n_samples=n_samples,
            n_features=n_features,
            n_classes=5,
        )
        self.classes_ = np.unique(self.y)
        self.batch_size_ = batch_size
        self.solver_ = solver

        self._incremental_kwargs = {"solver": solver}
        if solver == "svd":
            self._incremental_kwargs["store_covariance"] = False

        self._batch_kwargs = {"solver": solver}
        if solver == "svd":
            self._batch_kwargs["store_covariance"] = False

        self._reference_estimator = LinearDiscriminantAnalysis(**self._batch_kwargs)
        self._reference_estimator.fit(self.X, self.y)
        self._reference_predictions = self._reference_estimator.predict(self.X_val)
        self._reference_accuracy = balanced_accuracy_score(
            self.y_val, self._reference_predictions
        )

        self._prefit_incremental = self._fit_incremental_estimator(batch_size)
        self._prefit_predictions = self._prefit_incremental.predict(self.X_val)
        self._prefit_accuracy = balanced_accuracy_score(
            self.y_val, self._prefit_predictions
        )

    def _fit_incremental_estimator(self, batch_size):
        estimator = IncrementalLinearDiscriminantAnalysis(**self._incremental_kwargs)
        offset = 0
        n_samples = self.X.shape[0]
        while offset < n_samples:
            end = min(offset + batch_size, n_samples)
            batch_X = self.X[offset:end]
            batch_y = self.y[offset:end]
            if offset == 0:
                estimator.partial_fit(batch_X, batch_y, classes=self.classes_)
            else:
                estimator.partial_fit(batch_X, batch_y)
            offset = end
        return estimator

    def _iterate_batches(self, batch_size):
        n_samples = self.X.shape[0]
        offset = 0
        while offset < n_samples:
            end = min(offset + batch_size, n_samples)
            yield offset, end
            offset = end

    def time_partial_fit(self, solver, batch_size):
        estimator = IncrementalLinearDiscriminantAnalysis(**self._incremental_kwargs)
        for start, end in self._iterate_batches(batch_size):
            batch_X = self.X[start:end]
            batch_y = self.y[start:end]
            if start == 0:
                estimator.partial_fit(batch_X, batch_y, classes=self.classes_)
            else:
                estimator.partial_fit(batch_X, batch_y)

    def time_predict(self, solver, batch_size):
        self._prefit_incremental.predict(self.X_val)

    def track_incremental_accuracy(self, solver, batch_size):
        return float(self._prefit_accuracy)

    def track_accuracy_gap(self, solver, batch_size):
        return float(self._prefit_accuracy - self._reference_accuracy)

    def track_prediction_agreement(self, solver, batch_size):
        return float(np.mean(self._prefit_predictions == self._reference_predictions))

