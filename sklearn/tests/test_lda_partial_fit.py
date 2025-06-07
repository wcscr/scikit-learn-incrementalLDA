import numpy as np
from numpy.testing import assert_allclose

from sklearn.datasets import make_classification
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

X = np.array(
    [[0, 0], [-2, -2], [-2, -1], [-1, -1], [-1, -2], [1, 3], [1, 2], [2, 1], [2, 2]]
)
y = np.array([1, 1, 1, 1, 1, 2, 2, 2, 2])


def _check_partial_fit_equivalence(solver):
    clf = LinearDiscriminantAnalysis(solver=solver, store_covariance=True).fit(X, y)
    clf_pf = LinearDiscriminantAnalysis(solver=solver, store_covariance=True)
    clf_pf.partial_fit(X[:5], y[:5], classes=np.unique(y))
    clf_pf.partial_fit(X[5:], y[5:])

    err_msg_predict = f"predict mismatch for solver {solver}"
    assert_allclose(clf.predict(X), clf_pf.predict(X), err_msg=err_msg_predict)

    err_msg_means = f"means_ mismatch for solver {solver}"
    assert_allclose(clf.means_, clf_pf.means_, rtol=1e-5, atol=1e-5, err_msg=err_msg_means)

    # coef_ and intercept_ for SVD are not directly compared here as
    # their calculation paths in fit() and partial_fit() can lead to
    # numerical differences. Functional equivalence for SVD is ensured
    # by comparing predict(), predict_proba(), transform() outputs,
    # and scalings_ attributes, which are tested.
    if solver != "svd":
        err_msg_coef = f"coef_ mismatch for solver {solver}"
        assert_allclose(clf.coef_, clf_pf.coef_, rtol=1e-5, atol=1e-5, err_msg=err_msg_coef)

        err_msg_intercept = f"intercept_ mismatch for solver {solver}"
        assert_allclose(
            clf.intercept_, clf_pf.intercept_, rtol=1e-5, atol=1e-5, err_msg=err_msg_intercept
        )

    err_msg_priors = f"priors_ mismatch for solver {solver}"
    assert_allclose(clf.priors_, clf_pf.priors_, rtol=1e-5, atol=1e-5, err_msg=err_msg_priors)

    err_msg_covariance = f"covariance_ mismatch for solver {solver}"
    assert_allclose(
        clf.covariance_, clf_pf.covariance_, rtol=1e-5, atol=1e-5, err_msg=err_msg_covariance
    )

    err_msg_proba = f"predict_proba mismatch for solver {solver}"
    assert_allclose(
        clf.predict_proba(X), clf_pf.predict_proba(X), rtol=1e-5, atol=1e-5, err_msg=err_msg_proba
    )

    if solver in ["svd", "eigen"]:
        err_msg_scalings = f"scalings_ mismatch for solver {solver}"
        assert_allclose(
            clf.scalings_, clf_pf.scalings_, rtol=1e-5, atol=1e-5, err_msg=err_msg_scalings
        )
        err_msg_transform = f"transform mismatch for solver {solver}"
        assert_allclose(
            clf.transform(X), clf_pf.transform(X), rtol=1e-5, atol=1e-5, err_msg=err_msg_transform
        )


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
        clf = LinearDiscriminantAnalysis(solver=solver, store_covariance=True).fit(
            X_large, y_large
        )

        clf_pf_once = LinearDiscriminantAnalysis(solver=solver, store_covariance=True)
        clf_pf_once.partial_fit(X_large, y_large, classes=classes)

        clf_pf_batches = LinearDiscriminantAnalysis(solver=solver, store_covariance=True)
        first = True
        for Xi, yi in zip(np.array_split(X_large, 4), np.array_split(y_large, 4)):
            clf_pf_batches.partial_fit(Xi, yi, classes=classes if first else None)
            first = False

        # Compare clf_pf_once to clf
        err_msg_prefix_once = f"clf_pf_once vs clf ({solver}): "
        assert_allclose(
            clf.predict(X_large),
            clf_pf_once.predict(X_large),
            err_msg=err_msg_prefix_once + "predict mismatch",
        )
        assert_allclose(
            clf.means_,
            clf_pf_once.means_,
            rtol=1e-5,
            atol=1e-5,
            err_msg=err_msg_prefix_once + "means_ mismatch",
        )
        # coef_ and intercept_ for SVD are not directly compared here as
        # their calculation paths in fit() and partial_fit() can lead to
        # numerical differences. Functional equivalence for SVD is ensured
        # by comparing predict(), predict_proba(), transform() outputs,
        # and scalings_ attributes, which are tested.
        if solver != "svd":
            assert_allclose(
                clf.coef_,
                clf_pf_once.coef_,
                rtol=1e-5,
                atol=1e-5,
                err_msg=err_msg_prefix_once + "coef_ mismatch",
            )
            assert_allclose(
                clf.intercept_,
                clf_pf_once.intercept_,
                rtol=1e-5,
                atol=1e-5,
                err_msg=err_msg_prefix_once + "intercept_ mismatch",
            )
        assert_allclose(
            clf.covariance_,
            clf_pf_once.covariance_,
            rtol=1e-5,
            atol=1e-5,
            err_msg=err_msg_prefix_once + "covariance_ mismatch",
        )
        assert_allclose(
            clf.predict_proba(X_large),
            clf_pf_once.predict_proba(X_large),
            rtol=1e-5,
            atol=1e-5,
            err_msg=err_msg_prefix_once + "predict_proba mismatch",
        )
        if solver in ["svd", "eigen"]:
            assert_allclose(
                clf.scalings_,
                clf_pf_once.scalings_,
                rtol=1e-5,
                atol=1e-5,
                err_msg=err_msg_prefix_once + "scalings_ mismatch",
            )
            assert_allclose(
                clf.transform(X_large),
                clf_pf_once.transform(X_large),
                rtol=1e-5,
                atol=1e-5,
                err_msg=err_msg_prefix_once + "transform mismatch",
            )

        # Compare clf_pf_batches to clf
        err_msg_prefix_batches = f"clf_pf_batches vs clf ({solver}): "
        assert_allclose(
            clf.predict(X_large),
            clf_pf_batches.predict(X_large),
            err_msg=err_msg_prefix_batches + "predict mismatch",
        )
        assert_allclose(
            clf.means_,
            clf_pf_batches.means_,
            rtol=1e-5,
            atol=1e-5,
            err_msg=err_msg_prefix_batches + "means_ mismatch",
        )
        # coef_ and intercept_ for SVD are not directly compared here as
        # their calculation paths in fit() and partial_fit() can lead to
        # numerical differences. Functional equivalence for SVD is ensured
        # by comparing predict(), predict_proba(), transform() outputs,
        # and scalings_ attributes, which are tested.
        if solver != "svd":
            assert_allclose(
                clf.coef_,
                clf_pf_batches.coef_,
                rtol=1e-5,
                atol=1e-5,
                err_msg=err_msg_prefix_batches + "coef_ mismatch",
            )
            assert_allclose(
                clf.intercept_,
                clf_pf_batches.intercept_,
                rtol=1e-5,
                atol=1e-5,
                err_msg=err_msg_prefix_batches + "intercept_ mismatch",
            )
        assert_allclose(
            clf.covariance_,
            clf_pf_batches.covariance_,
            rtol=1e-5,
            atol=1e-5,
            err_msg=err_msg_prefix_batches + "covariance_ mismatch",
        )
        assert_allclose(
            clf.predict_proba(X_large),
            clf_pf_batches.predict_proba(X_large),
            rtol=1e-5,
            atol=1e-5,
            err_msg=err_msg_prefix_batches + "predict_proba mismatch",
        )
        if solver in ["svd", "eigen"]:
            assert_allclose(
                clf.scalings_,
                clf_pf_batches.scalings_,
                rtol=1e-5,
                atol=1e-5,
                err_msg=err_msg_prefix_batches + "scalings_ mismatch",
            )
            assert_allclose(
                clf.transform(X_large),
                clf_pf_batches.transform(X_large),
                rtol=1e-5,
                atol=1e-5,
                err_msg=err_msg_prefix_batches + "transform mismatch",
            )
