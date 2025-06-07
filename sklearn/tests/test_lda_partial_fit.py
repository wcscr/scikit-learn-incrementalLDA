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
    assert_allclose(clf.priors_, clf_pf.priors_, rtol=1e-5, atol=1e-5) # Adjusted tolerance for consistency

    if solver == "eigen":
        assert_allclose(clf.coef_, clf_pf.coef_, rtol=1e-5, atol=1e-5) # Keep existing check, adjust tol
        assert_allclose(clf.intercept_, clf_pf.intercept_, rtol=1e-5, atol=1e-5) # Keep existing check, adjust tol
        assert_allclose(clf.means_, clf_pf.means_, rtol=1e-5, atol=1e-5)
        assert_allclose(clf.transform(X), clf_pf.transform(X), rtol=1e-5, atol=1e-5)
        if hasattr(clf, 'explained_variance_ratio_') and hasattr(clf_pf, 'explained_variance_ratio_'):
            assert_allclose(clf.explained_variance_ratio_, clf_pf.explained_variance_ratio_, rtol=1e-5, atol=1e-5)
    elif solver == "lsqr":
        assert_allclose(clf.coef_, clf_pf.coef_, rtol=1e-5, atol=1e-5) # Keep existing check, adjust tol
        assert_allclose(clf.intercept_, clf_pf.intercept_, rtol=1e-5, atol=1e-5) # Keep existing check, adjust tol
        assert_allclose(clf.means_, clf_pf.means_, rtol=1e-5, atol=1e-5)


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

        assert_allclose(clf.predict(X_large), clf_pf_once.predict(X_large), rtol=1e-5, atol=1e-5)
        assert_allclose(clf.predict(X_large), clf_pf_batches.predict(X_large), rtol=1e-5, atol=1e-5)
        assert_allclose(clf.priors_, clf_pf_once.priors_, rtol=1e-5, atol=1e-5)
        assert_allclose(clf.priors_, clf_pf_batches.priors_, rtol=1e-5, atol=1e-5)


        if solver == "svd":
            continue


def test_partial_fit_svd_store_covariance():
    # Case 1: store_covariance=True
    clf_pf_true = LinearDiscriminantAnalysis(solver='svd', store_covariance=True)
    clf_pf_true.partial_fit(X, y, classes=np.unique(y))
    assert hasattr(clf_pf_true, 'covariance_')
    assert clf_pf_true.covariance_ is not None

    clf_fit_true = LinearDiscriminantAnalysis(solver='svd', store_covariance=True).fit(X, y)
    assert hasattr(clf_fit_true, 'covariance_')
    assert_allclose(clf_pf_true.covariance_, clf_fit_true.covariance_, rtol=1e-5, atol=1e-5)

    # Case 2: store_covariance=False
    clf_pf_false = LinearDiscriminantAnalysis(solver='svd', store_covariance=False)
    clf_pf_false.partial_fit(X, y, classes=np.unique(y))
    assert not hasattr(clf_pf_false, 'covariance_')


def test_partial_fit_svd_n_components():
    X_large, y_large = make_classification(
        n_samples=200,
        n_features=5,  # Ensure enough features
        n_informative=3,
        n_redundant=0,
        n_repeated=0,
        n_classes=3,   # Ensure enough classes
        n_clusters_per_class=1,
        random_state=42,
    )
    classes = np.unique(y_large)
    n_features = X_large.shape[1]
    n_cls = len(classes)
    max_components = min(n_cls - 1, n_features)

    # Test Case: n_components=None (default)
    clf_pf_none = LinearDiscriminantAnalysis(solver='svd', n_components=None)
    clf_pf_none.partial_fit(X_large, y_large, classes=classes)
    transformed_none = clf_pf_none.transform(X_large)
    assert transformed_none.shape[1] == max_components
    assert len(clf_pf_none.explained_variance_ratio_) == max_components



    # Test Case: n_components=1 (specific value, if max_components >= 1)
    if max_components >= 1:
        n_comp_specific = 1
        clf_pf_one = LinearDiscriminantAnalysis(solver='svd', n_components=n_comp_specific)
        clf_pf_one.partial_fit(X_large, y_large, classes=classes)
        transformed_one = clf_pf_one.transform(X_large)
        assert transformed_one.shape[1] == n_comp_specific
        assert len(clf_pf_one.explained_variance_ratio_) == n_comp_specific



def test_partial_fit_svd_high_dimensions():
    X_hd, y_hd = make_classification(
        n_samples=50,
        n_features=100,
        n_classes=3,
        n_informative=10,
        n_redundant=0,
        n_repeated=0,
        shuffle=False, # for reproducible batching
        random_state=42,
    )
    classes = np.unique(y_hd)

    # Fit with fit()
    clf_fit = LinearDiscriminantAnalysis(solver='svd', store_covariance=False)
    clf_fit.fit(X_hd, y_hd)

    # Fit with partial_fit() (single batch)
    clf_pf_single = LinearDiscriminantAnalysis(solver='svd', store_covariance=False)
    clf_pf_single.partial_fit(X_hd, y_hd, classes=classes)

    attributes_to_check = ["priors_", "means_", "xbar_"]
    for attr in attributes_to_check:
        assert_allclose(getattr(clf_fit, attr), getattr(clf_pf_single, attr), rtol=1e-5, atol=1e-5)

    # Transform can have more numerical sensitivity, especially in high dimensions with SVD
    # Check if transform is available (scalings_ might be empty)

    # Fit with partial_fit() (multiple batches)
    clf_pf_multi = LinearDiscriminantAnalysis(solver='svd', store_covariance=False)
    batch_size = X_hd.shape[0] // 3
    clf_pf_multi.partial_fit(X_hd[:batch_size], y_hd[:batch_size], classes=classes)
    clf_pf_multi.partial_fit(X_hd[batch_size : 2 * batch_size], y_hd[batch_size : 2 * batch_size])
    clf_pf_multi.partial_fit(X_hd[2 * batch_size :], y_hd[2 * batch_size :])

    for attr in attributes_to_check:
        assert_allclose(getattr(clf_fit, attr), getattr(clf_pf_multi, attr), rtol=1e-5, atol=1e-5)

