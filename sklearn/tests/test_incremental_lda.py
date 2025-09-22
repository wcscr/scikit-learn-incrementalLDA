import numpy as np
import pytest

from sklearn.covariance import LedoitWolf
from sklearn.datasets import load_iris, make_classification
from sklearn.discriminant_analysis import (
    IncrementalLinearDiscriminantAnalysis,
    LinearDiscriminantAnalysis,
)


def _iterate_batches(X, y, sample_weight=None, n_splits=3):
    X_batches = np.array_split(X, n_splits)
    y_batches = np.array_split(y, n_splits)
    if sample_weight is None:
        sw_batches = [None] * n_splits
    else:
        sw_batches = np.array_split(sample_weight, n_splits)
    for X_batch, y_batch, sw_batch in zip(X_batches, y_batches, sw_batches):
        yield X_batch, y_batch, sw_batch


@pytest.mark.parametrize(
    "solver, shrinkage",
    [
        ("svd", None),
        ("lsqr", None),
        ("lsqr", 0.2),
        ("lsqr", "auto"),
        ("eigen", None),
        ("eigen", 0.2),
        ("eigen", "auto"),
    ],
)
def test_incremental_matches_batch_solver(solver, shrinkage):
    data = load_iris()
    X, y = data.data, data.target

    lda_kwargs = {"solver": solver}
    if solver != "svd" and shrinkage is not None:
        lda_kwargs["shrinkage"] = shrinkage

    lda = LinearDiscriminantAnalysis(**lda_kwargs)
    lda.fit(X, y)

    ilda_kwargs = {"solver": solver}
    if solver != "svd" and shrinkage is not None:
        ilda_kwargs["shrinkage"] = shrinkage
    if solver == "svd":
        ilda_kwargs["store_covariance"] = True

    ilda = IncrementalLinearDiscriminantAnalysis(**ilda_kwargs)
    classes = np.unique(y)
    for X_batch, y_batch, _ in _iterate_batches(X, y):
        ilda.partial_fit(X_batch, y_batch, classes=classes)

    np.testing.assert_array_equal(ilda.predict(X), lda.predict(X))
    np.testing.assert_allclose(
        ilda.decision_function(X), lda.decision_function(X), rtol=1e-5, atol=1e-8
    )
    np.testing.assert_allclose(
        ilda.predict_proba(X), lda.predict_proba(X), rtol=1e-5, atol=1e-8
    )

    if solver in {"svd", "eigen"}:
        np.testing.assert_allclose(
            ilda.transform(X), lda.transform(X), rtol=1e-5, atol=1e-8
        )


def test_incremental_sample_weight_matches_repetition():
    X, y = make_classification(
        n_samples=120, n_features=6, n_informative=4, n_classes=3, random_state=42
    )
    rng = np.random.default_rng(0)
    sample_weight = rng.integers(1, 4, size=X.shape[0]).astype(float)

    repeated_X = np.repeat(X, sample_weight.astype(int), axis=0)
    repeated_y = np.repeat(y, sample_weight.astype(int), axis=0)

    lda = LinearDiscriminantAnalysis()
    lda.fit(repeated_X, repeated_y)

    ilda = IncrementalLinearDiscriminantAnalysis()
    ilda.fit(X, y, sample_weight=sample_weight)

    np.testing.assert_array_equal(ilda.predict(X), lda.predict(X))
    np.testing.assert_allclose(
        ilda.decision_function(X), lda.decision_function(X), rtol=1e-5, atol=1e-8
    )


@pytest.mark.parametrize("solver", ["lsqr", "eigen"])
def test_incremental_auto_shrinkage_with_sample_weight_matches_batch(solver):
    X, y = make_classification(
        n_samples=160, n_features=8, n_informative=5, n_classes=3, random_state=7
    )
    rng = np.random.default_rng(4)
    sample_weight = rng.integers(1, 4, size=X.shape[0]).astype(float)

    repeated_X = np.repeat(X, sample_weight.astype(int), axis=0)
    repeated_y = np.repeat(y, sample_weight.astype(int), axis=0)

    lda = LinearDiscriminantAnalysis(solver=solver, shrinkage="auto")
    lda.fit(repeated_X, repeated_y)

    ilda = IncrementalLinearDiscriminantAnalysis(solver=solver, shrinkage="auto")
    classes = np.unique(y)
    for X_batch, y_batch, sw_batch in _iterate_batches(
        X, y, sample_weight=sample_weight, n_splits=4
    ):
        ilda.partial_fit(X_batch, y_batch, classes=classes, sample_weight=sw_batch)

    np.testing.assert_array_equal(ilda.predict(X), lda.predict(X))
    np.testing.assert_allclose(
        ilda.decision_function(X), lda.decision_function(X), rtol=2e-4, atol=1e-7
    )


def test_partial_fit_requires_classes():
    X, y = load_iris(return_X_y=True)
    ilda = IncrementalLinearDiscriminantAnalysis()
    with pytest.raises(ValueError, match="classes must be provided"):
        ilda.partial_fit(X[:10], y[:10])


def test_partial_fit_rejects_unseen_class():
    X, y = load_iris(return_X_y=True)
    classes = np.unique(y)
    ilda = IncrementalLinearDiscriminantAnalysis()
    ilda.partial_fit(X[:30], y[:30], classes=classes)
    with pytest.raises(ValueError, match="unseen labels"):
        y_bad = y[30:60].copy()
        y_bad[0] = y.max() + 1
        ilda.partial_fit(X[30:60], y_bad)


def test_partial_fit_sample_weight_matches_repetition():
    X, y = make_classification(
        n_samples=90, n_features=5, n_informative=3, n_classes=3, random_state=0
    )
    rng = np.random.default_rng(1)
    sample_weight = rng.integers(1, 4, size=X.shape[0]).astype(float)

    repeated_X = np.repeat(X, sample_weight.astype(int), axis=0)
    repeated_y = np.repeat(y, sample_weight.astype(int), axis=0)

    lda = LinearDiscriminantAnalysis().fit(repeated_X, repeated_y)

    ilda = IncrementalLinearDiscriminantAnalysis()
    classes = np.unique(y)
    for X_batch, y_batch, sw_batch in _iterate_batches(
        X, y, sample_weight=sample_weight
    ):
        ilda.partial_fit(X_batch, y_batch, classes=classes, sample_weight=sw_batch)

    np.testing.assert_array_equal(ilda.predict(X), lda.predict(X))
    np.testing.assert_allclose(
        ilda.decision_function(X), lda.decision_function(X), rtol=1e-5, atol=1e-8
    )


def test_svd_shrinkage_not_supported():
    X, y = load_iris(return_X_y=True)
    classes = np.unique(y)
    ilda = IncrementalLinearDiscriminantAnalysis(solver="svd", shrinkage=0.1)
    with pytest.raises(NotImplementedError, match="shrinkage not supported"):
        ilda.partial_fit(X, y, classes=classes)


def test_partial_fit_rejects_covariance_estimator():
    X, y = load_iris(return_X_y=True)
    classes = np.unique(y)
    ilda = IncrementalLinearDiscriminantAnalysis(covariance_estimator=LedoitWolf())
    with pytest.raises(
        NotImplementedError, match="covariance_estimator is not supported"
    ):
        ilda.partial_fit(X, y, classes=classes)


def test_transform_respects_n_components():
    X, y = load_iris(return_X_y=True)
    classes = np.unique(y)
    ilda = IncrementalLinearDiscriminantAnalysis(n_components=1, solver="svd")
    for X_batch, y_batch, _ in _iterate_batches(X, y):
        ilda.partial_fit(X_batch, y_batch, classes=classes)

    transformed = ilda.transform(X)
    assert transformed.shape == (X.shape[0], 1)
