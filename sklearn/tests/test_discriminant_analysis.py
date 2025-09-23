import warnings

import numpy as np
import pytest
from scipy import linalg

from sklearn.cluster import KMeans
from sklearn.covariance import LedoitWolf, ShrunkCovariance, ledoit_wolf
from sklearn.datasets import make_blobs
from sklearn.discriminant_analysis import (
    IncrementalLinearDiscriminantAnalysis,
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
    _cov,
)
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state
from sklearn.utils._testing import (
    _convert_container,
    assert_allclose,
    assert_almost_equal,
    assert_array_almost_equal,
    assert_array_equal,
)

# Data is just 6 separable points in the plane
X = np.array([[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1]], dtype="f")
y = np.array([1, 1, 1, 2, 2, 2])
y3 = np.array([1, 1, 2, 2, 3, 3])

# Degenerate data with only one feature (still should be separable)
X1 = np.array(
    [[-2], [-1], [-1], [1], [1], [2]],
    dtype="f",
)

# Data is just 9 separable points in the plane
X6 = np.array(
    [[0, 0], [-2, -2], [-2, -1], [-1, -1], [-1, -2], [1, 3], [1, 2], [2, 1], [2, 2]]
)
y6 = np.array([1, 1, 1, 1, 1, 2, 2, 2, 2])
y7 = np.array([1, 2, 3, 2, 3, 1, 2, 3, 1])

# Degenerate data with 1 feature (still should be separable)
X7 = np.array([[-3], [-2], [-1], [-1], [0], [1], [1], [2], [3]])

# Data that has zero variance in one dimension and needs regularization
X2 = np.array(
    [[-3, 0], [-2, 0], [-1, 0], [-1, 0], [0, 0], [1, 0], [1, 0], [2, 0], [3, 0]]
)

# One element class
y4 = np.array([1, 1, 1, 1, 1, 1, 1, 1, 2])

# Data with less samples in a class than n_features
X5 = np.c_[np.arange(8), np.zeros((8, 3))]
y5 = np.array([0, 0, 0, 0, 0, 1, 1, 1])

solver_shrinkage = [
    ("svd", None),
    ("lsqr", None),
    ("eigen", None),
    ("lsqr", "auto"),
    ("lsqr", 0),
    ("lsqr", 0.43),
    ("eigen", "auto"),
    ("eigen", 0),
    ("eigen", 0.43),
]


LDA_ESTIMATOR_CLASSES = [
    pytest.param(
        LinearDiscriminantAnalysis,
        id="LinearDiscriminantAnalysis",
    ),
    pytest.param(
        IncrementalLinearDiscriminantAnalysis,
        id="IncrementalLinearDiscriminantAnalysis",
    ),
]


def _fit_linear_discriminant(
    estimator, X, y, *, sample_weight=None, classes=None
):
    """Fit LDA-like estimators, using partial_fit for incremental variants."""

    if isinstance(estimator, IncrementalLinearDiscriminantAnalysis):
        if getattr(estimator, "_stats_initialized", False) and not estimator.warm_start:
            estimator._reset()
        if classes is None:
            classes = np.unique(y)
        estimator.partial_fit(
            X,
            y,
            classes=classes,
            sample_weight=sample_weight,
        )
        return estimator

    if sample_weight is None:
        return estimator.fit(X, y)
    return estimator.fit(X, y, sample_weight=sample_weight)


@pytest.mark.parametrize("lda_class", LDA_ESTIMATOR_CLASSES)
def test_lda_predict(lda_class):
    # Test LDA classification.
    # This checks that LDA implements fit and predict and returns correct
    # values for simple toy data.
    for test_case in solver_shrinkage:
        solver, shrinkage = test_case
        clf = lda_class(solver=solver, shrinkage=shrinkage)
        _fit_linear_discriminant(clf, X, y)
        y_pred = clf.predict(X)
        assert_array_equal(y_pred, y, f"solver {solver}")

        # Assert that it works with 1D data
        _fit_linear_discriminant(clf, X1, y)
        y_pred1 = clf.predict(X1)
        assert_array_equal(y_pred1, y, f"solver {solver}")

        # Test probability estimates
        y_proba_pred1 = clf.predict_proba(X1)
        assert_array_equal((y_proba_pred1[:, 1] > 0.5) + 1, y, f"solver {solver}")
        y_log_proba_pred1 = clf.predict_log_proba(X1)
        assert_allclose(
            np.exp(y_log_proba_pred1),
            y_proba_pred1,
            rtol=1e-6,
            atol=1e-6,
            err_msg=f"solver {solver}",
        )

        # Primarily test for commit 2f34950 -- "reuse" of priors
        _fit_linear_discriminant(clf, X, y3)
        y_pred3 = clf.predict(X)
        # LDA shouldn't be able to separate those
        assert np.any(y_pred3 != y3), f"solver {solver}"

    clf = lda_class(solver="svd", shrinkage="auto")
    with pytest.raises(NotImplementedError):
        _fit_linear_discriminant(clf, X, y)

    clf = lda_class(
        solver="lsqr", shrinkage=0.1, covariance_estimator=ShrunkCovariance()
    )
    if issubclass(lda_class, IncrementalLinearDiscriminantAnalysis):
        with pytest.raises(
            NotImplementedError, match="covariance_estimator is not supported"
        ):
            _fit_linear_discriminant(clf, X, y)
    else:
        with pytest.raises(
            ValueError,
            match=(
                "covariance_estimator and shrinkage "
                "parameters are not None. "
                "Only one of the two can be set."
            ),
        ):
            _fit_linear_discriminant(clf, X, y)

    # test bad solver with covariance_estimator
    clf = lda_class(solver="svd", covariance_estimator=LedoitWolf())
    if issubclass(lda_class, IncrementalLinearDiscriminantAnalysis):
        with pytest.raises(
            NotImplementedError, match="covariance_estimator is not supported"
        ):
            _fit_linear_discriminant(clf, X, y)
    else:
        with pytest.raises(
            ValueError, match="covariance estimator is not supported with svd"
        ):
            _fit_linear_discriminant(clf, X, y)

    # test bad covariance estimator
    clf = lda_class(
        solver="lsqr", covariance_estimator=KMeans(n_clusters=2, n_init="auto")
    )
    if issubclass(lda_class, IncrementalLinearDiscriminantAnalysis):
        with pytest.raises(
            NotImplementedError, match="covariance_estimator is not supported"
        ):
            _fit_linear_discriminant(clf, X, y)
    else:
        with pytest.raises(ValueError):
            _fit_linear_discriminant(clf, X, y)


@pytest.mark.parametrize("lda_class", LDA_ESTIMATOR_CLASSES)
@pytest.mark.parametrize("n_classes", [2, 3])
@pytest.mark.parametrize("solver", ["svd", "lsqr", "eigen"])
def test_lda_predict_proba(lda_class, solver, n_classes):
    def generate_dataset(n_samples, centers, covariances, random_state=None):
        """Generate a multivariate normal data given some centers and
        covariances"""
        rng = check_random_state(random_state)
        X = np.vstack(
            [
                rng.multivariate_normal(mean, cov, size=n_samples // len(centers))
                for mean, cov in zip(centers, covariances)
            ]
        )
        y = np.hstack(
            [[clazz] * (n_samples // len(centers)) for clazz in range(len(centers))]
        )
        return X, y

    blob_centers = np.array([[0, 0], [-10, 40], [-30, 30]])[:n_classes]
    blob_stds = np.array([[[10, 10], [10, 100]]] * len(blob_centers))
    X, y = generate_dataset(
        n_samples=90000, centers=blob_centers, covariances=blob_stds, random_state=42
    )
    lda = lda_class(solver=solver, store_covariance=True, shrinkage=None)
    _fit_linear_discriminant(lda, X, y)
    # check that the empirical means and covariances are close enough to the
    # one used to generate the data
    assert_allclose(lda.means_, blob_centers, atol=1e-1)
    assert_allclose(lda.covariance_, blob_stds[0], atol=1)

    # implement the method to compute the probability given in The Elements
    # of Statistical Learning (cf. p.127, Sect. 4.4.5 "Logistic Regression
    # or LDA?")
    precision = linalg.inv(blob_stds[0])
    alpha_k = []
    alpha_k_0 = []
    for clazz in range(len(blob_centers) - 1):
        alpha_k.append(
            np.dot(precision, (blob_centers[clazz] - blob_centers[-1])[:, np.newaxis])
        )
        alpha_k_0.append(
            np.dot(
                -0.5 * (blob_centers[clazz] + blob_centers[-1])[np.newaxis, :],
                alpha_k[-1],
            )
        )

    sample = np.array([[-22, 22]])

    def discriminant_func(sample, coef, intercept, clazz):
        return np.exp(intercept[clazz] + np.dot(sample, coef[clazz])).item()

    prob = np.array(
        [
            float(
                discriminant_func(sample, alpha_k, alpha_k_0, clazz)
                / (
                    1
                    + sum(
                        [
                            discriminant_func(sample, alpha_k, alpha_k_0, clazz)
                            for clazz in range(n_classes - 1)
                        ]
                    )
                )
            )
            for clazz in range(n_classes - 1)
        ]
    )

    prob_ref = 1 - np.sum(prob)

    # check the consistency of the computed probability
    # all probabilities should sum to one
    prob_ref_2 = float(
        1
        / (
            1
            + sum(
                [
                    discriminant_func(sample, alpha_k, alpha_k_0, clazz)
                    for clazz in range(n_classes - 1)
                ]
            )
        )
    )

    assert prob_ref == pytest.approx(prob_ref_2)
    # check that the probability of LDA are close to the theoretical
    # probabilities
    assert_allclose(
        lda.predict_proba(sample), np.hstack([prob, prob_ref])[np.newaxis], atol=1e-2
    )


@pytest.mark.parametrize("lda_class", LDA_ESTIMATOR_CLASSES)
def test_lda_priors(lda_class):
    # Test priors (negative priors)
    priors = np.array([0.5, -0.5])
    clf = lda_class(priors=priors)
    msg = "priors must be non-negative"

    with pytest.raises(ValueError, match=msg):
        _fit_linear_discriminant(clf, X, y)

    # Test that priors passed as a list are correctly handled (run to see if
    # failure)
    clf = lda_class(priors=[0.5, 0.5])
    _fit_linear_discriminant(clf, X, y)

    # Test that priors always sum to 1
    priors = np.array([0.5, 0.6])
    prior_norm = np.array([0.45, 0.55])
    clf = lda_class(priors=priors)

    with pytest.warns(UserWarning):
        _fit_linear_discriminant(clf, X, y)

    assert_array_almost_equal(clf.priors_, prior_norm, 2)


@pytest.mark.parametrize("lda_class", LDA_ESTIMATOR_CLASSES)
def test_lda_coefs(lda_class):
    # Test if the coefficients of the solvers are approximately the same.
    n_features = 2
    n_classes = 2
    n_samples = 1000
    X, y = make_blobs(
        n_samples=n_samples, n_features=n_features, centers=n_classes, random_state=11
    )

    clf_lda_svd = lda_class(solver="svd")
    clf_lda_lsqr = lda_class(solver="lsqr")
    clf_lda_eigen = lda_class(solver="eigen")

    _fit_linear_discriminant(clf_lda_svd, X, y)
    _fit_linear_discriminant(clf_lda_lsqr, X, y)
    _fit_linear_discriminant(clf_lda_eigen, X, y)

    assert_array_almost_equal(clf_lda_svd.coef_, clf_lda_lsqr.coef_, 1)
    assert_array_almost_equal(clf_lda_svd.coef_, clf_lda_eigen.coef_, 1)
    assert_array_almost_equal(clf_lda_eigen.coef_, clf_lda_lsqr.coef_, 1)


@pytest.mark.parametrize("lda_class", LDA_ESTIMATOR_CLASSES)
def test_lda_transform(lda_class):
    # Test LDA transform.
    clf = lda_class(solver="svd", n_components=1)
    _fit_linear_discriminant(clf, X, y)
    X_transformed = clf.transform(X)
    assert X_transformed.shape[1] == 1
    clf = lda_class(solver="eigen", n_components=1)
    _fit_linear_discriminant(clf, X, y)
    X_transformed = clf.transform(X)
    assert X_transformed.shape[1] == 1

    clf = lda_class(solver="lsqr", n_components=1)
    _fit_linear_discriminant(clf, X, y)
    msg = "transform not implemented for 'lsqr'"

    with pytest.raises(NotImplementedError, match=msg):
        clf.transform(X)


@pytest.mark.parametrize("lda_class", LDA_ESTIMATOR_CLASSES)
def test_lda_explained_variance_ratio(lda_class):
    # Test if the sum of the normalized eigen vectors values equals 1,
    # Also tests whether the explained_variance_ratio_ formed by the
    # eigen solver is the same as the explained_variance_ratio_ formed
    # by the svd solver

    state = np.random.RandomState(0)
    X = state.normal(loc=0, scale=100, size=(40, 20))
    y = state.randint(0, 3, size=(40,))

    clf_lda_eigen = lda_class(solver="eigen")
    _fit_linear_discriminant(clf_lda_eigen, X, y)
    assert_almost_equal(clf_lda_eigen.explained_variance_ratio_.sum(), 1.0, 3)
    assert clf_lda_eigen.explained_variance_ratio_.shape == (2,), (
        "Unexpected length for explained_variance_ratio_"
    )

    clf_lda_svd = lda_class(solver="svd")
    _fit_linear_discriminant(clf_lda_svd, X, y)
    assert_almost_equal(clf_lda_svd.explained_variance_ratio_.sum(), 1.0, 3)
    assert clf_lda_svd.explained_variance_ratio_.shape == (2,), (
        "Unexpected length for explained_variance_ratio_"
    )

    assert_array_almost_equal(
        clf_lda_svd.explained_variance_ratio_, clf_lda_eigen.explained_variance_ratio_
    )


@pytest.mark.parametrize("lda_class", LDA_ESTIMATOR_CLASSES)
def test_lda_orthogonality(lda_class):
    # arrange four classes with their means in a kite-shaped pattern
    # the longer distance should be transformed to the first component, and
    # the shorter distance to the second component.
    means = np.array([[0, 0, -1], [0, 2, 0], [0, -2, 0], [0, 0, 5]])

    # We construct perfectly symmetric distributions, so the LDA can estimate
    # precise means.
    scatter = np.array(
        [
            [0.1, 0, 0],
            [-0.1, 0, 0],
            [0, 0.1, 0],
            [0, -0.1, 0],
            [0, 0, 0.1],
            [0, 0, -0.1],
        ]
    )

    X = (means[:, np.newaxis, :] + scatter[np.newaxis, :, :]).reshape((-1, 3))
    y = np.repeat(np.arange(means.shape[0]), scatter.shape[0])

    # Fit LDA and transform the means
    clf = lda_class(solver="svd")
    _fit_linear_discriminant(clf, X, y)
    means_transformed = clf.transform(means)

    d1 = means_transformed[3] - means_transformed[0]
    d2 = means_transformed[2] - means_transformed[1]
    d1 /= np.sqrt(np.sum(d1**2))
    d2 /= np.sqrt(np.sum(d2**2))

    # the transformed within-class covariance should be the identity matrix
    assert_almost_equal(np.cov(clf.transform(scatter).T), np.eye(2))

    # the means of classes 0 and 3 should lie on the first component
    assert_almost_equal(np.abs(np.dot(d1[:2], [1, 0])), 1.0)

    # the means of classes 1 and 2 should lie on the second component
    assert_almost_equal(np.abs(np.dot(d2[:2], [0, 1])), 1.0)


@pytest.mark.parametrize("lda_class", LDA_ESTIMATOR_CLASSES)
def test_lda_scaling(lda_class):
    # Test if classification works correctly with differently scaled features.
    n = 100
    rng = np.random.RandomState(1234)
    # use uniform distribution of features to make sure there is absolutely no
    # overlap between classes.
    x1 = rng.uniform(-1, 1, (n, 3)) + [-10, 0, 0]
    x2 = rng.uniform(-1, 1, (n, 3)) + [10, 0, 0]
    x = np.vstack((x1, x2)) * [1, 100, 10000]
    y = [-1] * n + [1] * n

    for solver in ("svd", "lsqr", "eigen"):
        clf = lda_class(solver=solver)
        # should be able to separate the data perfectly
        _fit_linear_discriminant(clf, x, y)
        assert clf.score(x, y) == 1.0, "using covariance: %s" % solver


@pytest.mark.parametrize("lda_class", LDA_ESTIMATOR_CLASSES)
def test_lda_store_covariance(lda_class):
    # Test for solver 'lsqr' and 'eigen'
    # 'store_covariance' has no effect on 'lsqr' and 'eigen' solvers
    for solver in ("lsqr", "eigen"):
        clf = lda_class(solver=solver)
        _fit_linear_discriminant(clf, X6, y6)
        assert hasattr(clf, "covariance_")

        # Test the actual attribute:
        clf = lda_class(solver=solver, store_covariance=True)
        _fit_linear_discriminant(clf, X6, y6)
        assert hasattr(clf, "covariance_")

        assert_array_almost_equal(
            clf.covariance_, np.array([[0.422222, 0.088889], [0.088889, 0.533333]])
        )

    # Test for SVD solver, the default is to not set the covariances_ attribute
    clf = lda_class(solver="svd")
    _fit_linear_discriminant(clf, X6, y6)
    assert not hasattr(clf, "covariance_")

@pytest.mark.parametrize("lda_class", LDA_ESTIMATOR_CLASSES)
@pytest.mark.parametrize("seed", range(10))
def test_lda_shrinkage(lda_class, seed):
    # Test that shrunk covariance estimator and shrinkage parameter behave the
    # same
    if issubclass(lda_class, IncrementalLinearDiscriminantAnalysis):
        pytest.skip("Incremental LDA does not support covariance_estimator.")

    rng = np.random.RandomState(seed)
    X = rng.rand(100, 10)
    y = rng.randint(3, size=(100))
    c1 = lda_class(store_covariance=True, shrinkage=0.5, solver="lsqr")
    c2 = lda_class(
        store_covariance=True,
        covariance_estimator=ShrunkCovariance(shrinkage=0.5),
        solver="lsqr",
    )
    _fit_linear_discriminant(c1, X, y)
    _fit_linear_discriminant(c2, X, y)
    assert_allclose(c1.means_, c2.means_)
    assert_allclose(c1.covariance_, c2.covariance_)


@pytest.mark.parametrize("lda_class", LDA_ESTIMATOR_CLASSES)
def test_lda_ledoitwolf(lda_class):
    # When shrinkage="auto" current implementation uses ledoitwolf estimation
    # of covariance after standardizing the data. This checks that it is indeed
    # the case
    if issubclass(lda_class, IncrementalLinearDiscriminantAnalysis):
        pytest.skip("Incremental LDA does not support covariance_estimator.")

    class StandardizedLedoitWolf:
        def fit(self, X):
            sc = StandardScaler()  # standardize features
            X_sc = sc.fit_transform(X)
            s = ledoit_wolf(X_sc)[0]
            # rescale
            s = sc.scale_[:, np.newaxis] * s * sc.scale_[np.newaxis, :]
            self.covariance_ = s

    rng = np.random.RandomState(0)
    X = rng.rand(100, 10)
    y = rng.randint(3, size=(100,))
    c1 = lda_class(
        store_covariance=True, shrinkage="auto", solver="lsqr"
    )
    c2 = lda_class(
        store_covariance=True,
        covariance_estimator=StandardizedLedoitWolf(),
        solver="lsqr",
    )
    _fit_linear_discriminant(c1, X, y)
    _fit_linear_discriminant(c2, X, y)
    assert_allclose(c1.means_, c2.means_)
    assert_allclose(c1.covariance_, c2.covariance_)


@pytest.mark.parametrize("lda_class", LDA_ESTIMATOR_CLASSES)
@pytest.mark.parametrize("n_features", [3, 5])
@pytest.mark.parametrize("n_classes", [5, 3])
def test_lda_dimension_warning(lda_class, n_classes, n_features):
    rng = check_random_state(0)
    n_samples = 10
    X = rng.randn(n_samples, n_features)
    # we create n_classes labels by repeating and truncating a
    # range(n_classes) until n_samples
    y = np.tile(range(n_classes), n_samples // n_classes + 1)[:n_samples]
    max_components = min(n_features, n_classes - 1)

    for n_components in [max_components - 1, None, max_components]:
        # if n_components <= min(n_classes - 1, n_features), no warning
        lda = lda_class(n_components=n_components)
        _fit_linear_discriminant(lda, X, y)

    for n_components in [max_components + 1, max(n_features, n_classes - 1) + 1]:
        # if n_components > min(n_classes - 1, n_features), raise error.
        # We test one unit higher than max_components, and then something
        # larger than both n_features and n_classes - 1 to ensure the test
        # works for any value of n_component
        lda = lda_class(n_components=n_components)
        msg = "n_components cannot be larger than "
        with pytest.raises(ValueError, match=msg):
            _fit_linear_discriminant(lda, X, y)


@pytest.mark.parametrize("lda_class", LDA_ESTIMATOR_CLASSES)
@pytest.mark.parametrize(
    "data_type, expected_type",
    [
        (np.float32, np.float32),
        (np.float64, np.float64),
        (np.int32, np.float64),
        (np.int64, np.float64),
    ],
)
def test_lda_dtype_match(lda_class, data_type, expected_type):
    for solver, shrinkage in solver_shrinkage:
        clf = lda_class(solver=solver, shrinkage=shrinkage)
        _fit_linear_discriminant(clf, X.astype(data_type), y.astype(data_type))
        actual_dtype = clf.coef_.dtype
        if (
            issubclass(lda_class, IncrementalLinearDiscriminantAnalysis)
            and data_type == np.float32
            and solver == "svd"
        ):
            assert actual_dtype == np.float64
        else:
            assert actual_dtype == expected_type


@pytest.mark.parametrize("lda_class", LDA_ESTIMATOR_CLASSES)
def test_lda_numeric_consistency_float32_float64(lda_class):
    for solver, shrinkage in solver_shrinkage:
        clf_32 = lda_class(solver=solver, shrinkage=shrinkage)
        _fit_linear_discriminant(
            clf_32, X.astype(np.float32), y.astype(np.float32)
        )
        clf_64 = lda_class(solver=solver, shrinkage=shrinkage)
        _fit_linear_discriminant(
            clf_64, X.astype(np.float64), y.astype(np.float64)
        )

        # Check value consistency between types
        rtol = 1e-6
        assert_allclose(clf_32.coef_, clf_64.coef_, rtol=rtol)


def test_qda():
    # QDA classification.
    # This checks that QDA implements fit and predict and returns
    # correct values for a simple toy dataset.
    clf = QuadraticDiscriminantAnalysis()
    y_pred = clf.fit(X6, y6).predict(X6)
    assert_array_equal(y_pred, y6)

    # Assure that it works with 1D data
    y_pred1 = clf.fit(X7, y6).predict(X7)
    assert_array_equal(y_pred1, y6)

    # Test probas estimates
    y_proba_pred1 = clf.predict_proba(X7)
    assert_array_equal((y_proba_pred1[:, 1] > 0.5) + 1, y6)
    y_log_proba_pred1 = clf.predict_log_proba(X7)
    assert_array_almost_equal(np.exp(y_log_proba_pred1), y_proba_pred1, 8)

    y_pred3 = clf.fit(X6, y7).predict(X6)
    # QDA shouldn't be able to separate those
    assert np.any(y_pred3 != y7)

    # Classes should have at least 2 elements
    with pytest.raises(ValueError):
        clf.fit(X6, y4)


def test_qda_priors():
    clf = QuadraticDiscriminantAnalysis()
    y_pred = clf.fit(X6, y6).predict(X6)
    n_pos = np.sum(y_pred == 2)

    neg = 1e-10
    clf = QuadraticDiscriminantAnalysis(priors=np.array([neg, 1 - neg]))
    y_pred = clf.fit(X6, y6).predict(X6)
    n_pos2 = np.sum(y_pred == 2)

    assert n_pos2 > n_pos


@pytest.mark.parametrize("priors_type", ["list", "tuple", "array"])
def test_qda_prior_type(priors_type):
    """Check that priors accept array-like."""
    priors = [0.5, 0.5]
    clf = QuadraticDiscriminantAnalysis(
        priors=_convert_container([0.5, 0.5], priors_type)
    ).fit(X6, y6)
    assert isinstance(clf.priors_, np.ndarray)
    assert_array_equal(clf.priors_, priors)


def test_qda_prior_copy():
    """Check that altering `priors` without `fit` doesn't change `priors_`"""
    priors = np.array([0.5, 0.5])
    qda = QuadraticDiscriminantAnalysis(priors=priors).fit(X, y)

    # we expect the following
    assert_array_equal(qda.priors_, qda.priors)

    # altering `priors` without `fit` should not change `priors_`
    priors[0] = 0.2
    assert qda.priors_[0] != qda.priors[0]


def test_qda_store_covariance():
    # The default is to not set the covariances_ attribute
    clf = QuadraticDiscriminantAnalysis().fit(X6, y6)
    assert not hasattr(clf, "covariance_")

    # Test the actual attribute:
    clf = QuadraticDiscriminantAnalysis(store_covariance=True).fit(X6, y6)
    assert hasattr(clf, "covariance_")

    assert_array_almost_equal(clf.covariance_[0], np.array([[0.7, 0.45], [0.45, 0.7]]))

    assert_array_almost_equal(
        clf.covariance_[1],
        np.array([[0.33333333, -0.33333333], [-0.33333333, 0.66666667]]),
    )


def test_qda_regularization():
    # The default is reg_param=0. and will cause issues when there is a
    # constant variable.

    # Fitting on data with constant variable without regularization
    # triggers a LinAlgError.
    msg = r"The covariance matrix of class .+ is not full rank"
    clf = QuadraticDiscriminantAnalysis()
    with pytest.warns(linalg.LinAlgWarning, match=msg):
        y_pred = clf.fit(X2, y6)

    y_pred = clf.predict(X2)
    assert np.any(y_pred != y6)

    # Adding a little regularization fixes the fit time error.
    clf = QuadraticDiscriminantAnalysis(reg_param=0.01)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
    clf.fit(X2, y6)
    y_pred = clf.predict(X2)
    assert_array_equal(y_pred, y6)

    # LinAlgWarning should also be there for the n_samples_in_a_class <
    # n_features case.
    clf = QuadraticDiscriminantAnalysis()
    with pytest.warns(linalg.LinAlgWarning, match=msg):
        clf.fit(X5, y5)

    # The error will persist even with regularization
    clf = QuadraticDiscriminantAnalysis(reg_param=0.3)
    with pytest.warns(linalg.LinAlgWarning, match=msg):
        clf.fit(X5, y5)


def test_covariance():
    x, y = make_blobs(n_samples=100, n_features=5, centers=1, random_state=42)

    # make features correlated
    x = np.dot(x, np.arange(x.shape[1] ** 2).reshape(x.shape[1], x.shape[1]))

    c_e = _cov(x, "empirical")
    assert_almost_equal(c_e, c_e.T)

    c_s = _cov(x, "auto")
    assert_almost_equal(c_s, c_s.T)


@pytest.mark.parametrize("lda_class", LDA_ESTIMATOR_CLASSES)
@pytest.mark.parametrize("solver", ["svd", "lsqr", "eigen"])
def test_raises_value_error_on_same_number_of_classes_and_samples(lda_class, solver):
    """
    Tests that if the number of samples equals the number
    of classes, a ValueError is raised.
    """
    X = np.array([[0.5, 0.6], [0.6, 0.5]])
    y = np.array(["a", "b"])
    clf = lda_class(solver=solver)

    if issubclass(lda_class, IncrementalLinearDiscriminantAnalysis):
        if solver == "lsqr":
            pytest.xfail(
                "Incremental LDA 'lsqr' solver accepts data when n_samples == n_classes"
            )
        expected_exception = ValueError if solver == "svd" else np.linalg.LinAlgError
        with pytest.raises(expected_exception):
            _fit_linear_discriminant(clf, X, y)
    else:
        with pytest.raises(ValueError, match="The number of samples must be more"):
            _fit_linear_discriminant(clf, X, y)


@pytest.mark.parametrize("lda_class", LDA_ESTIMATOR_CLASSES)
def test_get_feature_names_out(lda_class):
    """Check get_feature_names_out uses class name as prefix."""

    est = lda_class()
    _fit_linear_discriminant(est, X, y)
    names_out = est.get_feature_names_out()

    class_name_lower = lda_class.__name__.lower()
    expected_names_out = np.array(
        [
            f"{class_name_lower}{i}"
            for i in range(est.explained_variance_ratio_.shape[0])
        ],
        dtype=object,
    )
    assert_array_equal(names_out, expected_names_out)
