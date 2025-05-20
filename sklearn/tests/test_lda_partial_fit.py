import numpy as np
from numpy.testing import assert_allclose

from sklearn.datasets import make_classification
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

X = np.array(
    [[0, 0], [-2, -2], [-2, -1], [-1, -1], [-1, -2], [1, 3], [1, 2], [2, 1], [2, 2]]
)
y = np.array([1, 1, 1, 1, 1, 2, 2, 2, 2])


def _check_partial_fit_equivalence(solver):
    clf = LinearDiscriminantAnalysis(solver=solver).fit(X, y)
    clf_pf = LinearDiscriminantAnalysis(solver=solver)
    clf_pf.partial_fit(X[:5], y[:5], classes=np.unique(y))
    clf_pf.partial_fit(X[5:], y[5:])
    assert_allclose(clf.predict(X), clf_pf.predict(X))
    if solver != "svd":
        assert_allclose(clf.coef_, clf_pf.coef_, rtol=1e-6, atol=1e-6)
        assert_allclose(clf.intercept_, clf_pf.intercept_, rtol=1e-6, atol=1e-6)
    assert_allclose(clf.priors_, clf_pf.priors_, rtol=1e-6)


def test_partial_fit_eigen():
    _check_partial_fit_equivalence("eigen")


def test_partial_fit_lsqr():
    _check_partial_fit_equivalence("lsqr")


def test_partial_fit_svd():
    _check_partial_fit_equivalence("svd")


def test_partial_fit_large_batches():
    X_large, y_large = make_classification(
        n_samples=200,
        n_features=2,
        n_informative=2,
        n_redundant=0,
        n_repeated=0,
        n_classes=2,
        n_clusters_per_class=1,
        random_state=0,
    )
    classes = np.unique(y_large)

    for solver in ["svd", "eigen", "lsqr"]:
        clf = LinearDiscriminantAnalysis(solver=solver).fit(X_large, y_large)

        clf_pf_once = LinearDiscriminantAnalysis(solver=solver)
        clf_pf_once.partial_fit(X_large, y_large, classes=classes)

        clf_pf_batches = LinearDiscriminantAnalysis(solver=solver)
        first = True
        for Xi, yi in zip(np.array_split(X_large, 4), np.array_split(y_large, 4)):
            clf_pf_batches.partial_fit(Xi, yi, classes=classes if first else None)
            first = False

        assert_allclose(clf.predict(X_large), clf_pf_once.predict(X_large))
        assert_allclose(clf.predict(X_large), clf_pf_batches.predict(X_large))

        if solver != "svd":
            assert_allclose(clf.coef_, clf_pf_once.coef_, rtol=1e-6, atol=1e-6)
            assert_allclose(clf.coef_, clf_pf_batches.coef_, rtol=1e-6, atol=1e-6)
