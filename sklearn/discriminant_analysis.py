"""Linear and quadratic discriminant analysis."""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import warnings
from collections import deque
from numbers import Integral, Real

import numpy as np
import scipy.linalg
from scipy import linalg

from sklearn.base import (
    BaseEstimator,
    ClassifierMixin,
    ClassNamePrefixFeaturesOutMixin,
    TransformerMixin,
    _fit_context,
)
from sklearn.covariance import (
    empirical_covariance,
    ledoit_wolf,
    ledoit_wolf_shrinkage,
    shrunk_covariance,
)
from sklearn.linear_model._base import LinearClassifierMixin
from sklearn.preprocessing import StandardScaler
from sklearn.utils._array_api import _expit, device, get_namespace, size
from sklearn.utils._param_validation import HasMethods, Interval, StrOptions
from sklearn.utils.extmath import softmax
from sklearn.utils.multiclass import check_classification_targets, unique_labels
from sklearn.utils.validation import (
    _check_sample_weight,
    check_is_fitted,
    validate_data,
)

__all__ = [
    "IncrementalLinearDiscriminantAnalysis",
    "LinearDiscriminantAnalysis",
    "QuadraticDiscriminantAnalysis",
]


def _cov(X, shrinkage=None, covariance_estimator=None):
    """Estimate covariance matrix (using optional covariance_estimator).
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Input data.

    shrinkage : {'empirical', 'auto'} or float, default=None
        Shrinkage parameter, possible values:
          - None or 'empirical': no shrinkage (default).
          - 'auto': automatic shrinkage using the Ledoit-Wolf lemma.
          - float between 0 and 1: fixed shrinkage parameter.

        Shrinkage parameter is ignored if  `covariance_estimator`
        is not None.

    covariance_estimator : estimator, default=None
        If not None, `covariance_estimator` is used to estimate
        the covariance matrices instead of relying on the empirical
        covariance estimator (with potential shrinkage).
        The object should have a fit method and a ``covariance_`` attribute
        like the estimators in :mod:`sklearn.covariance``.
        If None the shrinkage parameter drives the estimate.

        .. versionadded:: 0.24

    Returns
    -------
    s : ndarray of shape (n_features, n_features)
        Estimated covariance matrix.
    """
    if covariance_estimator is None:
        shrinkage = "empirical" if shrinkage is None else shrinkage
        if isinstance(shrinkage, str):
            if shrinkage == "auto":
                sc = StandardScaler()  # standardize features
                X = sc.fit_transform(X)
                s = ledoit_wolf(X)[0]
                # rescale
                s = sc.scale_[:, np.newaxis] * s * sc.scale_[np.newaxis, :]
            elif shrinkage == "empirical":
                s = empirical_covariance(X)
        elif isinstance(shrinkage, Real):
            s = shrunk_covariance(empirical_covariance(X), shrinkage)
    else:
        if shrinkage is not None and shrinkage != 0:
            raise ValueError(
                "covariance_estimator and shrinkage parameters "
                "are not None. Only one of the two can be set."
            )
        covariance_estimator.fit(X)
        if not hasattr(covariance_estimator, "covariance_"):
            raise ValueError(
                "%s does not have a covariance_ attribute"
                % covariance_estimator.__class__.__name__
            )
        s = covariance_estimator.covariance_
    return s


def _class_means(X, y):
    """Compute class means.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Input data.

    y : array-like of shape (n_samples,) or (n_samples, n_targets)
        Target values.

    Returns
    -------
    means : array-like of shape (n_classes, n_features)
        Class means.
    """
    xp, is_array_api_compliant = get_namespace(X)
    classes, y = xp.unique_inverse(y)
    means = xp.zeros((classes.shape[0], X.shape[1]), device=device(X), dtype=X.dtype)

    if is_array_api_compliant:
        for i in range(classes.shape[0]):
            means[i, :] = xp.mean(X[y == i], axis=0)
    else:
        # TODO: Explore the choice of using bincount + add.at as it seems sub optimal
        # from a performance-wise
        cnt = np.bincount(y)
        np.add.at(means, y, X)
        means /= cnt[:, None]
    return means


def _class_cov(X, y, priors, shrinkage=None, covariance_estimator=None):
    """Compute weighted within-class covariance matrix.

    The per-class covariance are weighted by the class priors.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Input data.

    y : array-like of shape (n_samples,) or (n_samples, n_targets)
        Target values.

    priors : array-like of shape (n_classes,)
        Class priors.

    shrinkage : 'auto' or float, default=None
        Shrinkage parameter, possible values:
          - None: no shrinkage (default).
          - 'auto': automatic shrinkage using the Ledoit-Wolf lemma.
          - float between 0 and 1: fixed shrinkage parameter.

        Shrinkage parameter is ignored if `covariance_estimator` is not None.

    covariance_estimator : estimator, default=None
        If not None, `covariance_estimator` is used to estimate
        the covariance matrices instead of relying the empirical
        covariance estimator (with potential shrinkage).
        The object should have a fit method and a ``covariance_`` attribute
        like the estimators in sklearn.covariance.
        If None, the shrinkage parameter drives the estimate.

        .. versionadded:: 0.24

    Returns
    -------
    cov : array-like of shape (n_features, n_features)
        Weighted within-class covariance matrix
    """
    classes = np.unique(y)
    cov = np.zeros(shape=(X.shape[1], X.shape[1]))
    for idx, group in enumerate(classes):
        Xg = X[y == group, :]
        cov += priors[idx] * np.atleast_2d(_cov(Xg, shrinkage, covariance_estimator))
    return cov


def _merge_second_order_statistics(
    weight_a, mean_a, m2_a, weight_b, mean_b, m2_b
):
    """Merge weighted means and second order moments.

    Parameters
    ----------
    weight_a : float
        Sum of sample weights for the first set.

    mean_a : ndarray of shape (n_features,)
        Mean vector of the first set.

    m2_a : ndarray of shape (n_features, n_features)
        Second central moment matrix of the first set.

    weight_b : float
        Sum of sample weights for the second set.

    mean_b : ndarray of shape (n_features,)
        Mean vector of the second set.

    m2_b : ndarray of shape (n_features, n_features)
        Second central moment matrix of the second set.

    Returns
    -------
    weight : float
        Total sum of sample weights.

    mean : ndarray of shape (n_features,)
        Updated mean vector.

    m2 : ndarray of shape (n_features, n_features)
        Updated second central moment matrix.
    """

    if weight_b == 0:
        return weight_a, mean_a, m2_a
    if weight_a == 0:
        return weight_b, mean_b, m2_b

    total_weight = weight_a + weight_b
    delta = mean_b - mean_a
    mean = mean_a + delta * (weight_b / total_weight)
    m2 = m2_a + m2_b + np.outer(delta, delta) * (weight_a * weight_b / total_weight)
    return total_weight, mean, m2


class DiscriminantAnalysisPredictionMixin:
    """Mixin class for QuadraticDiscriminantAnalysis and NearestCentroid."""

    def decision_function(self, X):
        """Apply decision function to an array of samples.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Array of samples (test vectors).

        Returns
        -------
        y_scores : ndarray of shape (n_samples,) or (n_samples, n_classes)
            Decision function values related to each class, per sample.
            In the two-class case, the shape is `(n_samples,)`, giving the
            log likelihood ratio of the positive class.
        """
        y_scores = self._decision_function(X)
        if len(self.classes_) == 2:
            return y_scores[:, 1] - y_scores[:, 0]
        return y_scores

    def predict(self, X):
        """Perform classification on an array of vectors `X`.

        Returns the class label for each sample.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Input vectors, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Class label for each sample.
        """
        scores = self._decision_function(X)
        return self.classes_.take(scores.argmax(axis=1))

    def predict_proba(self, X):
        """Estimate class probabilities.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        y_proba : ndarray of shape (n_samples, n_classes)
            Probability estimate of the sample for each class in the
            model, where classes are ordered as they are in `self.classes_`.
        """
        return np.exp(self.predict_log_proba(X))

    def predict_log_proba(self, X):
        """Estimate log class probabilities.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        y_log_proba : ndarray of shape (n_samples, n_classes)
            Estimated log probabilities.
        """
        scores = self._decision_function(X)
        log_likelihood = scores - scores.max(axis=1)[:, np.newaxis]
        return log_likelihood - np.log(
            np.exp(log_likelihood).sum(axis=1)[:, np.newaxis]
        )


class LinearDiscriminantAnalysis(
    ClassNamePrefixFeaturesOutMixin,
    LinearClassifierMixin,
    TransformerMixin,
    BaseEstimator,
):
    """Linear Discriminant Analysis.

    A classifier with a linear decision boundary, generated by fitting class
    conditional densities to the data and using Bayes' rule.

    The model fits a Gaussian density to each class, assuming that all classes
    share the same covariance matrix.

    The fitted model can also be used to reduce the dimensionality of the input
    by projecting it to the most discriminative directions, using the
    `transform` method.

    .. versionadded:: 0.17

    For a comparison between
    :class:`~sklearn.discriminant_analysis.LinearDiscriminantAnalysis`
    and :class:`~sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis`, see
    :ref:`sphx_glr_auto_examples_classification_plot_lda_qda.py`.

    Read more in the :ref:`User Guide <lda_qda>`.

    Parameters
    ----------
    solver : {'svd', 'lsqr', 'eigen'}, default='svd'
        Solver to use, possible values:
          - 'svd': Singular value decomposition (default).
            Does not compute the covariance matrix, therefore this solver is
            recommended for data with a large number of features.
          - 'lsqr': Least squares solution.
            Can be combined with shrinkage or custom covariance estimator.
          - 'eigen': Eigenvalue decomposition.
            Can be combined with shrinkage or custom covariance estimator.

        .. versionchanged:: 1.2
            `solver="svd"` now has experimental Array API support. See the
            :ref:`Array API User Guide <array_api>` for more details.

    shrinkage : 'auto' or float, default=None
        Shrinkage parameter, possible values:
          - None: no shrinkage (default).
          - 'auto': automatic shrinkage using the Ledoit-Wolf lemma.
          - float between 0 and 1: fixed shrinkage parameter.

        This should be left to None if `covariance_estimator` is used.
        Note that shrinkage works only with 'lsqr' and 'eigen' solvers.

        For a usage example, see
        :ref:`sphx_glr_auto_examples_classification_plot_lda.py`.

    priors : array-like of shape (n_classes,), default=None
        The class prior probabilities. By default, the class proportions are
        inferred from the training data.

    n_components : int, default=None
        Number of components (<= min(n_classes - 1, n_features)) for
        dimensionality reduction. If None, will be set to
        min(n_classes - 1, n_features). This parameter only affects the
        `transform` method.

        For a usage example, see
        :ref:`sphx_glr_auto_examples_decomposition_plot_pca_vs_lda.py`.

    store_covariance : bool, default=False
        If True, explicitly compute the weighted within-class covariance
        matrix when solver is 'svd'. The matrix is always computed
        and stored for the other solvers.

        .. versionadded:: 0.17

    tol : float, default=1.0e-4
        Absolute threshold for a singular value of X to be considered
        significant, used to estimate the rank of X. Dimensions whose
        singular values are non-significant are discarded. Only used if
        solver is 'svd'.

        .. versionadded:: 0.17

    covariance_estimator : covariance estimator, default=None
        If not None, `covariance_estimator` is used to estimate
        the covariance matrices instead of relying on the empirical
        covariance estimator (with potential shrinkage).
        The object should have a fit method and a ``covariance_`` attribute
        like the estimators in :mod:`sklearn.covariance`.
        if None the shrinkage parameter drives the estimate.

        This should be left to None if `shrinkage` is used.
        Note that `covariance_estimator` works only with 'lsqr' and 'eigen'
        solvers.

        .. versionadded:: 0.24

    Attributes
    ----------
    coef_ : ndarray of shape (n_features,) or (n_classes, n_features)
        Weight vector(s).

    intercept_ : ndarray of shape (n_classes,)
        Intercept term.

    covariance_ : array-like of shape (n_features, n_features)
        Weighted within-class covariance matrix. It corresponds to
        `sum_k prior_k * C_k` where `C_k` is the covariance matrix of the
        samples in class `k`. The `C_k` are estimated using the (potentially
        shrunk) biased estimator of covariance. If solver is 'svd', only
        exists when `store_covariance` is True.

    explained_variance_ratio_ : ndarray of shape (n_components,)
        Percentage of variance explained by each of the selected components.
        If ``n_components`` is not set then all components are stored and the
        sum of explained variances is equal to 1.0. Only available when eigen
        or svd solver is used.

    means_ : array-like of shape (n_classes, n_features)
        Class-wise means.

    priors_ : array-like of shape (n_classes,)
        Class priors (sum to 1).

    scalings_ : array-like of shape (rank, n_classes - 1)
        Scaling of the features in the space spanned by the class centroids.
        Only available for 'svd' and 'eigen' solvers.

    xbar_ : array-like of shape (n_features,)
        Overall mean. Only present if solver is 'svd'.

    classes_ : array-like of shape (n_classes,)
        Unique class labels.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    QuadraticDiscriminantAnalysis : Quadratic Discriminant Analysis.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    >>> X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    >>> y = np.array([1, 1, 1, 2, 2, 2])
    >>> clf = LinearDiscriminantAnalysis()
    >>> clf.fit(X, y)
    LinearDiscriminantAnalysis()
    >>> print(clf.predict([[-0.8, -1]]))
    [1]
    """

    _parameter_constraints: dict = {
        "solver": [StrOptions({"svd", "lsqr", "eigen"})],
        "shrinkage": [StrOptions({"auto"}), Interval(Real, 0, 1, closed="both"), None],
        "n_components": [Interval(Integral, 1, None, closed="left"), None],
        "priors": ["array-like", None],
        "store_covariance": ["boolean"],
        "tol": [Interval(Real, 0, None, closed="left")],
        "covariance_estimator": [HasMethods("fit"), None],
    }

    def __init__(
        self,
        solver="svd",
        shrinkage=None,
        priors=None,
        n_components=None,
        store_covariance=False,
        tol=1e-4,
        covariance_estimator=None,
    ):
        self.solver = solver
        self.shrinkage = shrinkage
        self.priors = priors
        self.n_components = n_components
        self.store_covariance = store_covariance  # used only in svd solver
        self.tol = tol  # used only in svd solver
        self.covariance_estimator = covariance_estimator

    def _solve_lstsq(self, X, y, shrinkage, covariance_estimator):
        """Least squares solver.

        The least squares solver computes a straightforward solution of the
        optimal decision rule based directly on the discriminant functions. It
        can only be used for classification (with any covariance estimator),
        because
        estimation of eigenvectors is not performed. Therefore, dimensionality
        reduction with the transform is not supported.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        y : array-like of shape (n_samples,) or (n_samples, n_classes)
            Target values.

        shrinkage : 'auto', float or None
            Shrinkage parameter, possible values:
              - None: no shrinkage.
              - 'auto': automatic shrinkage using the Ledoit-Wolf lemma.
              - float between 0 and 1: fixed shrinkage parameter.

            Shrinkage parameter is ignored if  `covariance_estimator` is
            not None

        covariance_estimator : estimator, default=None
            If not None, `covariance_estimator` is used to estimate
            the covariance matrices instead of relying the empirical
            covariance estimator (with potential shrinkage).
            The object should have a fit method and a ``covariance_`` attribute
            like the estimators in sklearn.covariance.
            if None the shrinkage parameter drives the estimate.

            .. versionadded:: 0.24

        Notes
        -----
        This solver is based on [1]_, section 2.6.2, pp. 39-41.

        References
        ----------
        .. [1] R. O. Duda, P. E. Hart, D. G. Stork. Pattern Classification
           (Second Edition). John Wiley & Sons, Inc., New York, 2001. ISBN
           0-471-05669-3.
        """
        self.means_ = _class_means(X, y)
        self.covariance_ = _class_cov(
            X, y, self.priors_, shrinkage, covariance_estimator
        )
        self.coef_ = linalg.lstsq(self.covariance_, self.means_.T)[0].T
        self.intercept_ = -0.5 * np.diag(np.dot(self.means_, self.coef_.T)) + np.log(
            self.priors_
        )

    def _solve_eigen(self, X, y, shrinkage, covariance_estimator):
        """Eigenvalue solver.

        The eigenvalue solver computes the optimal solution of the Rayleigh
        coefficient (basically the ratio of between class scatter to within
        class scatter). This solver supports both classification and
        dimensionality reduction (with any covariance estimator).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Target values.

        shrinkage : 'auto', float or None
            Shrinkage parameter, possible values:
              - None: no shrinkage.
              - 'auto': automatic shrinkage using the Ledoit-Wolf lemma.
              - float between 0 and 1: fixed shrinkage constant.

            Shrinkage parameter is ignored if  `covariance_estimator` i
            not None

        covariance_estimator : estimator, default=None
            If not None, `covariance_estimator` is used to estimate
            the covariance matrices instead of relying the empirical
            covariance estimator (with potential shrinkage).
            The object should have a fit method and a ``covariance_`` attribute
            like the estimators in sklearn.covariance.
            if None the shrinkage parameter drives the estimate.

            .. versionadded:: 0.24

        Notes
        -----
        This solver is based on [1]_, section 3.8.3, pp. 121-124.

        References
        ----------
        .. [1] R. O. Duda, P. E. Hart, D. G. Stork. Pattern Classification
           (Second Edition). John Wiley & Sons, Inc., New York, 2001. ISBN
           0-471-05669-3.
        """
        self.means_ = _class_means(X, y)
        self.covariance_ = _class_cov(
            X, y, self.priors_, shrinkage, covariance_estimator
        )

        Sw = self.covariance_  # within scatter
        St = _cov(X, shrinkage, covariance_estimator)  # total scatter
        Sb = St - Sw  # between scatter

        evals, evecs = linalg.eigh(Sb, Sw)
        self.explained_variance_ratio_ = np.sort(evals / np.sum(evals))[::-1][
            : self._max_components
        ]
        evecs = evecs[:, np.argsort(evals)[::-1]]  # sort eigenvectors

        self.scalings_ = evecs
        self.coef_ = np.dot(self.means_, evecs).dot(evecs.T)
        self.intercept_ = -0.5 * np.diag(np.dot(self.means_, self.coef_.T)) + np.log(
            self.priors_
        )

    def _solve_svd(self, X, y):
        """SVD solver.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Target values.
        """
        xp, is_array_api_compliant = get_namespace(X)

        if is_array_api_compliant:
            svd = xp.linalg.svd
        else:
            svd = scipy.linalg.svd

        n_samples, _ = X.shape
        n_classes = self.classes_.shape[0]

        self.means_ = _class_means(X, y)
        if self.store_covariance:
            self.covariance_ = _class_cov(X, y, self.priors_)

        Xc = []
        for idx, group in enumerate(self.classes_):
            Xg = X[y == group]
            Xc.append(Xg - self.means_[idx, :])

        self.xbar_ = self.priors_ @ self.means_

        Xc = xp.concat(Xc, axis=0)

        # 1) within (univariate) scaling by with classes std-dev
        std = xp.std(Xc, axis=0)
        # avoid division by zero in normalization
        std[std == 0] = 1.0
        fac = xp.asarray(1.0 / (n_samples - n_classes), dtype=X.dtype, device=device(X))

        # 2) Within variance scaling
        X = xp.sqrt(fac) * (Xc / std)
        # SVD of centered (within)scaled data
        _, S, Vt = svd(X, full_matrices=False)

        rank = xp.sum(xp.astype(S > self.tol, xp.int32))
        # Scaling of within covariance is: V' 1/S
        scalings = (Vt[:rank, :] / std).T / S[:rank]
        fac = 1.0 if n_classes == 1 else 1.0 / (n_classes - 1)

        # 3) Between variance scaling
        # Scale weighted centers
        X = (
            (xp.sqrt((n_samples * self.priors_) * fac)) * (self.means_ - self.xbar_).T
        ).T @ scalings
        # Centers are living in a space with n_classes-1 dim (maximum)
        # Use SVD to find projection in the space spanned by the
        # (n_classes) centers
        _, S, Vt = svd(X, full_matrices=False)

        if self._max_components == 0:
            self.explained_variance_ratio_ = xp.empty((0,), dtype=S.dtype)
        else:
            self.explained_variance_ratio_ = (S**2 / xp.sum(S**2))[
                : self._max_components
            ]

        rank = xp.sum(xp.astype(S > self.tol * S[0], xp.int32))
        self.scalings_ = scalings @ Vt.T[:, :rank]
        coef = (self.means_ - self.xbar_) @ self.scalings_
        self.intercept_ = -0.5 * xp.sum(coef**2, axis=1) + xp.log(self.priors_)
        self.coef_ = coef @ self.scalings_.T
        self.intercept_ -= self.xbar_ @ self.coef_.T

    @_fit_context(
        # LinearDiscriminantAnalysis.covariance_estimator is not validated yet
        prefer_skip_nested_validation=False
    )
    def fit(self, X, y):
        """Fit the Linear Discriminant Analysis model.

        .. versionchanged:: 0.19
            `store_covariance` and `tol` has been moved to main constructor.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        xp, _ = get_namespace(X)

        X, y = validate_data(
            self, X, y, ensure_min_samples=2, dtype=[xp.float64, xp.float32]
        )
        self.classes_ = unique_labels(y)
        n_samples, n_features = X.shape
        n_classes = self.classes_.shape[0]

        if n_samples == n_classes:
            raise ValueError(
                "The number of samples must be more than the number of classes."
            )

        if self.priors is None:  # estimate priors from sample
            _, cnts = xp.unique_counts(y)  # non-negative ints
            self.priors_ = xp.astype(cnts, X.dtype) / float(y.shape[0])
        else:
            self.priors_ = xp.asarray(self.priors, dtype=X.dtype)

        if xp.any(self.priors_ < 0):
            raise ValueError("priors must be non-negative")

        if xp.abs(xp.sum(self.priors_) - 1.0) > 1e-5:
            warnings.warn("The priors do not sum to 1. Renormalizing", UserWarning)
            self.priors_ = self.priors_ / self.priors_.sum()

        # Maximum number of components no matter what n_components is
        # specified:
        max_components = min(n_classes - 1, n_features)

        if self.n_components is None:
            self._max_components = max_components
        else:
            if self.n_components > max_components:
                raise ValueError(
                    "n_components cannot be larger than min(n_features, n_classes - 1)."
                )
            self._max_components = self.n_components

        if self.solver == "svd":
            if self.shrinkage is not None:
                raise NotImplementedError("shrinkage not supported with 'svd' solver.")
            if self.covariance_estimator is not None:
                raise ValueError(
                    "covariance estimator "
                    "is not supported "
                    "with svd solver. Try another solver"
                )
            self._solve_svd(X, y)
        elif self.solver == "lsqr":
            self._solve_lstsq(
                X,
                y,
                shrinkage=self.shrinkage,
                covariance_estimator=self.covariance_estimator,
            )
        elif self.solver == "eigen":
            self._solve_eigen(
                X,
                y,
                shrinkage=self.shrinkage,
                covariance_estimator=self.covariance_estimator,
            )
        if size(self.classes_) == 2:  # treat binary case as a special case
            coef_ = xp.asarray(self.coef_[1, :] - self.coef_[0, :], dtype=X.dtype)
            self.coef_ = xp.reshape(coef_, (1, -1))
            intercept_ = xp.asarray(
                self.intercept_[1] - self.intercept_[0], dtype=X.dtype
            )
            self.intercept_ = xp.reshape(intercept_, (1,))
        self._n_features_out = self._max_components
        return self

    def transform(self, X):
        """Project data to maximize class separation.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_components) or \
            (n_samples, min(rank, n_components))
            Transformed data. In the case of the 'svd' solver, the shape
            is (n_samples, min(rank, n_components)).
        """
        if self.solver == "lsqr":
            raise NotImplementedError(
                "transform not implemented for 'lsqr' solver (use 'svd' or 'eigen')."
            )
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)

        if self.solver == "svd":
            X_new = (X - self.xbar_) @ self.scalings_
        elif self.solver == "eigen":
            X_new = X @ self.scalings_

        return X_new[:, : self._max_components]

    def predict_proba(self, X):
        """Estimate probability.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        C : ndarray of shape (n_samples, n_classes)
            Estimated probabilities.
        """
        check_is_fitted(self)
        xp, _ = get_namespace(X)
        decision = self.decision_function(X)
        if size(self.classes_) == 2:
            proba = _expit(decision, xp)
            return xp.stack([1 - proba, proba], axis=1)
        else:
            return softmax(decision)

    def predict_log_proba(self, X):
        """Estimate log probability.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        C : ndarray of shape (n_samples, n_classes)
            Estimated log probabilities.
        """
        xp, _ = get_namespace(X)
        prediction = self.predict_proba(X)

        info = xp.finfo(prediction.dtype)
        if hasattr(info, "smallest_normal"):
            smallest_normal = info.smallest_normal
        else:
            # smallest_normal was introduced in NumPy 1.22
            smallest_normal = info.tiny

        prediction[prediction == 0.0] += smallest_normal
        return xp.log(prediction)

    def decision_function(self, X):
        """Apply decision function to an array of samples.

        The decision function is equal (up to a constant factor) to the
        log-posterior of the model, i.e. `log p(y = k | x)`. In a binary
        classification setting this instead corresponds to the difference
        `log p(y = 1 | x) - log p(y = 0 | x)`. See :ref:`lda_qda_math`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Array of samples (test vectors).

        Returns
        -------
        y_scores : ndarray of shape (n_samples,) or (n_samples, n_classes)
            Decision function values related to each class, per sample.
            In the two-class case, the shape is `(n_samples,)`, giving the
            log likelihood ratio of the positive class.
        """
        # Only override for the doc
        return super().decision_function(X)

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.array_api_support = True
        return tags


class IncrementalLinearDiscriminantAnalysis(
    ClassNamePrefixFeaturesOutMixin,
    LinearClassifierMixin,
    TransformerMixin,
    BaseEstimator,
):
    """Incremental Linear Discriminant Analysis.

    This estimator implements an online variant of Linear Discriminant Analysis
    (LDA). The model parameters are updated from mini-batches using
    :meth:`partial_fit`, without requiring access to previously seen samples.

    Parameters are consistent with :class:`LinearDiscriminantAnalysis`. The
    estimator maintains class-wise sufficient statistics (counts, means and
    within-class scatter) together with global totals. These statistics are
    updated in a numerically stable fashion using Pébay's streaming formulas.
    After each update the linear model is recomputed for the requested solver.

    Parameters
    ----------
    solver : {'svd', 'lsqr', 'eigen'}, default='svd'
        Solver to use. See :class:`LinearDiscriminantAnalysis` for details.

    shrinkage : 'auto' or float, default=None
        Shrinkage parameter for the ``'lsqr'`` and ``'eigen'`` solvers. The
        ``'svd'`` solver does not support shrinkage. A float provides a fixed
        shrinkage coefficient while ``'auto'`` estimates a Ledoit-Wolf
        shrinkage value from a rolling buffer of recent within-class
        residuals.

    priors : array-like of shape (n_classes,), default=None
        Class priors. If None, priors are inferred from the data seen so far.

    n_components : int, default=None
        Number of components for dimensionality reduction. Only used by the
        ``'svd'`` and ``'eigen'`` solvers. If None, ``min(n_classes - 1,
        n_features)`` is used.

    store_covariance : bool, default=False
        If True and ``solver='svd'``, store the pooled covariance matrix.

    tol : float, default=1e-4
        Threshold used by the ``'svd'`` solver when determining effective
        rank.

    warm_start : bool, default=False
        If True, successive calls to :meth:`fit` will reuse the state learned
        by previous calls instead of restarting from scratch.

    svd_method : {'auto', 'cov', 'randomized'}, default='auto'
        Strategy used by the ``'svd'`` solver. The ``'cov'`` strategy rebuilds
        the solver from the accumulated covariance statistics. Other options
        are reserved for future extensions.

    random_state : int, RandomState instance or None, default=None
        Controls the randomness of the optional randomized SVD path. Currently
        unused. Pass an int for reproducible results across calls.

    covariance_estimator : estimator, default=None
        Placeholder for compatibility with :class:`LinearDiscriminantAnalysis`.
        ``partial_fit`` raises ``NotImplementedError`` when a custom covariance
        estimator that does not implement ``partial_fit`` is provided.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        Class labels known to the classifier.

    coef_ : ndarray of shape (n_features,) or (n_classes, n_features)
        Weight vector(s).

    intercept_ : ndarray of shape (n_classes,)
        Intercept term(s).

    covariance_ : ndarray of shape (n_features, n_features)
        Pooled within-class covariance matrix. Only computed when requested by
        the solver.

    explained_variance_ratio_ : ndarray of shape (n_components,)
        Percentage of variance explained by each selected component. Only
        available for the ``'svd'`` and ``'eigen'`` solvers.

    means_ : ndarray of shape (n_classes, n_features)
        Class-wise means.

    priors_ : ndarray of shape (n_classes,)
        Class prior probabilities.

    scalings_ : ndarray of shape (n_features, n_features)
        Transformation matrix used by :meth:`transform` when supported by the
        solver.

    xbar_ : ndarray of shape (n_features,)
        Overall weighted mean. Only computed when ``solver='svd'``.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X` has
        feature names that are all strings.

    See Also
    --------
    LinearDiscriminantAnalysis : Linear Discriminant Analysis trained in batch
        mode.

    Notes
    -----
    The implementation stores only aggregate statistics, enabling streaming
    updates without retaining past samples. Support for shrinkage
    ``shrinkage='auto'`` relies on a rolling buffer of within-class residuals
    (4096 vectors by default) to estimate a Ledoit-Wolf shrinkage coefficient.
    Randomized SVD will be added in future work.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.discriminant_analysis import IncrementalLinearDiscriminantAnalysis
    >>> X, y = make_classification(random_state=0)
    >>> ild = IncrementalLinearDiscriminantAnalysis()
    >>> ild.partial_fit(X[:50], y[:50], classes=np.unique(y))
    IncrementalLinearDiscriminantAnalysis()
    >>> ild.partial_fit(X[50:], y[50:])
    IncrementalLinearDiscriminantAnalysis()
    >>> ild.predict(X[:2])
    array([0, 0])
    """

    __metadata_request__partial_fit = {"classes": True, "sample_weight": True}
    _auto_shrinkage_max_samples = 4096

    _parameter_constraints: dict = {
        "solver": [StrOptions({"svd", "lsqr", "eigen"})],
        "shrinkage": [StrOptions({"auto"}), Interval(Real, 0, 1, closed="both"), None],
        "priors": ["array-like", None],
        "n_components": [Interval(Integral, 1, None, closed="left"), None],
        "store_covariance": ["boolean"],
        "tol": [Interval(Real, 0, None, closed="left")],
        "warm_start": ["boolean"],
        "svd_method": [StrOptions({"auto", "cov", "randomized"})],
        "random_state": ["random_state"],
        "covariance_estimator": [HasMethods("fit"), None],
    }

    def __init__(
        self,
        *,
        solver="svd",
        shrinkage=None,
        priors=None,
        n_components=None,
        store_covariance=False,
        tol=1e-4,
        warm_start=False,
        svd_method="auto",
        random_state=None,
        covariance_estimator=None,
    ):
        self.solver = solver
        self.shrinkage = shrinkage
        self.priors = priors
        self.n_components = n_components
        self.store_covariance = store_covariance
        self.tol = tol
        self.warm_start = warm_start
        self.svd_method = svd_method
        self.random_state = random_state
        self.covariance_estimator = covariance_estimator

    def _reset(self):
        attributes = [
            "classes_",
            "coef_",
            "intercept_",
            "covariance_",
            "explained_variance_ratio_",
            "means_",
            "priors_",
            "scalings_",
            "xbar_",
            "_class_weight_sum_",
            "_class_means_",
            "_class_m2_",
            "_total_weight_",
            "_positive_weight_count_",
            "_mean_total_",
            "_m2_total_",
            "_n_features_out",
            "_auto_shrinkage_class_buffers_",
            "_auto_shrinkage_total_buffer_",
        ]
        for attr in attributes:
            if hasattr(self, attr):
                delattr(self, attr)
        self._stats_initialized = False
        self._model_ready = False

    def _init_stats(self, n_classes, n_features, dtype):
        self._class_weight_sum_ = np.zeros(n_classes, dtype=np.float64)
        self._class_means_ = np.zeros((n_classes, n_features), dtype=dtype)
        self._class_m2_ = np.zeros((n_classes, n_features, n_features), dtype=dtype)
        self._total_weight_ = 0.0
        self._positive_weight_count_ = 0
        self._mean_total_ = np.zeros(n_features, dtype=dtype)
        self._m2_total_ = np.zeros((n_features, n_features), dtype=dtype)
        if self._using_auto_shrinkage():
            self._auto_shrinkage_class_buffers_ = [
                deque(maxlen=self._auto_shrinkage_max_samples)
                for _ in range(n_classes)
            ]
            self._auto_shrinkage_total_buffer_ = deque(
                maxlen=self._auto_shrinkage_max_samples
            )
        else:
            self._auto_shrinkage_class_buffers_ = None
            self._auto_shrinkage_total_buffer_ = None

    def _check_priors(self, dtype):
        if self.priors is None:
            priors = self._class_weight_sum_ / self._total_weight_
        else:
            priors = np.asarray(self.priors, dtype=dtype)
            if priors.shape[0] != self.classes_.shape[0]:
                raise ValueError("Number of priors must match number of classes")
            if np.any(priors < 0):
                raise ValueError("priors must be non-negative")
            if abs(priors.sum() - 1.0) > 1e-5:
                warnings.warn("The priors do not sum to 1. Renormalizing", UserWarning)
                priors = priors / priors.sum()
        self.priors_ = priors.astype(dtype, copy=False)

    def _update_global_stats(self, X, sample_weight):
        weight_batch = float(np.sum(sample_weight))
        if weight_batch == 0:
            return
        mean_batch = np.average(X, axis=0, weights=sample_weight)
        diff = X - mean_batch
        weighted_diff = diff * sample_weight[:, None]
        m2_batch = diff.T @ weighted_diff
        (
            self._total_weight_,
            self._mean_total_,
            self._m2_total_,
        ) = _merge_second_order_statistics(
            self._total_weight_,
            self._mean_total_,
            self._m2_total_,
            weight_batch,
            mean_batch,
            m2_batch,
        )
        if self._using_auto_shrinkage():
            sqrt_w = np.sqrt(sample_weight.astype(np.float64, copy=False))
            self._append_samples_to_buffer(
                getattr(self, "_auto_shrinkage_total_buffer_", None), X, sqrt_w
            )

    def _using_auto_shrinkage(self):
        return self.shrinkage == "auto" and self.solver in {"lsqr", "eigen"}

    def _append_samples_to_buffer(self, buffer, samples, sqrt_weights):
        if buffer is None or samples.size == 0:
            return
        if samples.ndim == 1:
            samples = samples.reshape(1, -1)
        sqrt_weights = np.asarray(sqrt_weights, dtype=np.float64)
        if sqrt_weights.ndim == 0:
            sqrt_weights = np.repeat(sqrt_weights, samples.shape[0])
        for sample, sw in zip(samples, sqrt_weights):
            sample_arr = np.asarray(sample, dtype=np.float64)
            weight = float(sw) ** 2
            integer = int(np.floor(weight))
            fractional = weight - integer
            for _ in range(integer):
                buffer.append((sample_arr, 1.0))
            if fractional > 1e-12:
                buffer.append((sample_arr, np.sqrt(fractional)))

    def _compute_shrinkage_from_buffer(self, buffer, mean, scale):
        if buffer is None or len(buffer) <= 1:
            return 0.0
        samples = np.array([entry[0] for entry in buffer], dtype=np.float64)
        sqrt_weights = np.array([entry[1] for entry in buffer], dtype=np.float64)
        if samples.shape[0] <= 1:
            return 0.0
        residuals = (samples - mean) * sqrt_weights[:, None]
        data_scaled = residuals / scale
        return float(ledoit_wolf_shrinkage(data_scaled, assume_centered=False))

    def _auto_shrunk_covariance(self, covariance, buffer, mean):
        if buffer is None or len(buffer) <= 1:
            return covariance
        dtype = covariance.dtype
        covariance64 = covariance.astype(np.float64, copy=False)
        scale = np.sqrt(np.clip(np.diag(covariance64), 0, None))
        scale[scale == 0] = 1.0
        shrinkage = self._compute_shrinkage_from_buffer(buffer, mean, scale)
        if shrinkage == 0:
            return covariance
        scale_matrix = scale[:, None] * scale[None, :]
        cov_scaled = covariance64 / scale_matrix
        cov_shrunk_scaled = shrunk_covariance(cov_scaled, shrinkage)
        cov_shrunk = scale_matrix * cov_shrunk_scaled
        return cov_shrunk.astype(dtype, copy=False)

    def _update_class_stats(self, X, y_encoded, sample_weight):
        for class_idx in np.unique(y_encoded):
            mask = y_encoded == class_idx
            w = sample_weight[mask]
            weight_batch = float(np.sum(w))
            if weight_batch == 0:
                continue
            Xk = X[mask]
            mean_batch = np.average(Xk, axis=0, weights=w)
            diff = Xk - mean_batch
            weighted_diff = diff * w[:, None]
            m2_batch = diff.T @ weighted_diff
            (
                self._class_weight_sum_[class_idx],
                self._class_means_[class_idx],
                self._class_m2_[class_idx],
            ) = _merge_second_order_statistics(
                float(self._class_weight_sum_[class_idx]),
                self._class_means_[class_idx],
                self._class_m2_[class_idx],
                weight_batch,
                mean_batch,
                m2_batch,
            )
            if self._using_auto_shrinkage():
                class_buffers = getattr(self, "_auto_shrinkage_class_buffers_", None)
                if class_buffers is not None:
                    buffer = class_buffers[class_idx]
                    sqrt_w = np.sqrt(w.astype(np.float64, copy=False))
                    self._append_samples_to_buffer(buffer, Xk, sqrt_w)

    def _pooled_covariance(self, dtype):
        n_features = self._class_means_.shape[1]
        pooled = np.zeros((n_features, n_features), dtype=dtype)
        class_covariances = []
        for idx in range(self.classes_.shape[0]):
            weight = self._class_weight_sum_[idx]
            if weight <= 0:
                cov_k = np.zeros((n_features, n_features), dtype=dtype)
            else:
                cov_k = self._class_m2_[idx] / weight
            if isinstance(self.shrinkage, Real):
                cov_k = shrunk_covariance(cov_k, float(self.shrinkage))
            elif self._using_auto_shrinkage():
                class_buffers = getattr(self, "_auto_shrinkage_class_buffers_", None)
                buffer = None if class_buffers is None else class_buffers[idx]
                cov_k = self._auto_shrunk_covariance(
                    cov_k, buffer, self._class_means_[idx]
                )
            class_covariances.append(cov_k)
            pooled += self.priors_[idx] * cov_k
        self._class_covariances_ = class_covariances
        return pooled

    def _solve_lsqr(self, pooled_covariance):
        self.covariance_ = pooled_covariance
        solution = linalg.lstsq(pooled_covariance, self.means_.T)[0]
        self.coef_ = solution.T
        self.intercept_ = -0.5 * np.sum(self.means_ * self.coef_, axis=1)
        self.intercept_ += np.log(self.priors_)

    def _solve_eigen(self, pooled_covariance, total_covariance):
        self.covariance_ = pooled_covariance
        Sb = total_covariance - pooled_covariance
        evals, evecs = linalg.eigh(Sb, pooled_covariance)
        order = np.argsort(evals)[::-1]
        evals = evals[order]
        evecs = evecs[:, order]
        total = np.sum(evals)
        if total > 0:
            self.explained_variance_ratio_ = (evals / total)[: self._max_components]
        else:
            self.explained_variance_ratio_ = np.zeros(
                (0,), dtype=pooled_covariance.dtype
            )
        self.scalings_ = evecs
        self.coef_ = self.means_ @ evecs @ evecs.T
        self.intercept_ = -0.5 * np.sum(self.means_ * self.coef_, axis=1)
        self.intercept_ += np.log(self.priors_)

    def _solve_svd(self, pooled_covariance, dtype):
        if self.shrinkage is not None:
            raise NotImplementedError("shrinkage not supported with 'svd' solver.")
        if self.svd_method not in {"auto", "cov"}:
            raise NotImplementedError(
                "svd_method='randomized' is not yet implemented for incremental LDA"
            )
        Sw = np.sum(self._class_m2_, axis=0)
        total_weight = self._total_weight_
        n_classes = self.classes_.shape[0]
        n_features = self._class_means_.shape[1]
        std = np.sqrt(np.clip(np.diag(Sw), 0, None) / max(total_weight, 1.0))
        std[std == 0] = 1.0
        fac = 1.0 / max(total_weight - n_classes, 1.0)
        scaling = Sw.copy()
        inv_std = 1.0 / std
        scaling *= fac
        scaling = (inv_std[:, None] * scaling) * inv_std[None, :]
        evals, evecs = linalg.eigh(scaling)
        order = np.argsort(evals)[::-1]
        evals = evals[order]
        evecs = evecs[:, order]
        S = np.sqrt(np.clip(evals, 0.0, None))
        rank = np.sum(S > self.tol)
        if rank == 0:
            rank = 1
        scalings = (evecs[:, :rank] * inv_std[:, None]) / S[:rank]

        fac_between = 1.0 if n_classes == 1 else 1.0 / (n_classes - 1)
        centered = (self.means_ - self.xbar_) * np.sqrt(self.priors_)[:, None]
        X = np.sqrt(total_weight * fac_between) * centered @ scalings
        _, S2, Vt = linalg.svd(X, full_matrices=False)
        if S2.size == 0 or S2[0] == 0:
            rank2 = 0
        else:
            rank2 = np.sum(S2 > self.tol * S2[0])
        if rank2 == 0:
            self.explained_variance_ratio_ = np.zeros((0,), dtype=dtype)
        else:
            self.explained_variance_ratio_ = (S2**2 / np.sum(S2**2))[
                : self._max_components
            ]
        self.scalings_ = scalings @ Vt.T[:, :rank2]
        coef = (self.means_ - self.xbar_) @ self.scalings_
        self.intercept_ = -0.5 * np.sum(coef**2, axis=1) + np.log(self.priors_)
        self.coef_ = coef @ self.scalings_.T
        self.intercept_ -= self.xbar_ @ self.coef_.T
        if self.store_covariance:
            self.covariance_ = pooled_covariance

    def _recompute_model(self, dtype):
        if self._total_weight_ <= 0:
            return
        active_classes = np.sum(self._class_weight_sum_ > 0)
        if active_classes < 2:
            return
        self.means_ = self._class_means_.astype(dtype, copy=True)
        self.xbar_ = self._mean_total_.astype(dtype, copy=True)
        self._check_priors(dtype)

        n_features = self._class_means_.shape[1]
        n_classes = self.classes_.shape[0]
        if n_classes < 2:
            raise ValueError(
                "The number of classes has to be greater than one; got %d class"
                % (n_classes)
            )
        max_components = min(n_classes - 1, n_features)
        if self.n_components is None:
            self._max_components = max_components
        else:
            if self.n_components > max_components:
                raise ValueError(
                    "n_components cannot be larger than min(n_features, n_classes - 1)."
                )
            self._max_components = self.n_components

        if self.covariance_estimator is not None:
            raise NotImplementedError(
                "covariance_estimator is not supported in "
                "IncrementalLinearDiscriminantAnalysis"
            )

        pooled_covariance = self._pooled_covariance(dtype)
        total_covariance = self._m2_total_ / max(self._total_weight_, 1.0)
        if self.solver == "eigen":
            if isinstance(self.shrinkage, Real):
                total_covariance = shrunk_covariance(
                    total_covariance, float(self.shrinkage)
                )
            elif self._using_auto_shrinkage():
                total_covariance = self._auto_shrunk_covariance(
                    total_covariance,
                    getattr(self, "_auto_shrinkage_total_buffer_", None),
                    self._mean_total_,
                )

        if self.solver == "svd":
            self._solve_svd(pooled_covariance, dtype)
        elif self.solver == "lsqr":
            self._solve_lsqr(pooled_covariance)
        elif self.solver == "eigen":
            self._solve_eigen(pooled_covariance, total_covariance)

        if size(self.classes_) == 2:
            coef = self.coef_[1, :] - self.coef_[0, :]
            self.coef_ = coef.reshape(1, -1)
            intercept = self.intercept_[1] - self.intercept_[0]
            self.intercept_ = np.array([intercept], dtype=self.coef_.dtype)

        if hasattr(self, "_max_components"):
            self._n_features_out = self._max_components
        self._model_ready = True

    @_fit_context(prefer_skip_nested_validation=False)
    def fit(self, X, y, sample_weight=None):
        """Fit the model according to the given training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vectors, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        y : array-like of shape (n_samples,)
            Target values.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. Samples with a weight of zero do not contribute to
            the fitted model.

        Returns
        -------
        self : object
            Fitted estimator.

        Notes
        -----
        This method is functionally equivalent to calling :meth:`partial_fit`
        once with all samples and ``classes=np.unique(y)``.
        """
        self._validate_params()
        if not self.warm_start or not getattr(self, "_stats_initialized", False):
            self._reset()
        y = validate_data(self, y=y)

        classes = np.unique(y)
        if classes.size < 2:
            raise ValueError(
                "fit requires at least two classes; "
                f"got {classes.size} class{'es' if classes.size != 1 else ''}"
            )

        if sample_weight is not None:
            checked_sample_weight = _check_sample_weight(
                sample_weight,
                y,
            )
            positive_mask = checked_sample_weight > 0
            if not np.any(positive_mask):
                raise ValueError(
                    "fit requires at least two classes with positive sample_weight; "
                    "got 0 classes"
                )
            positive_classes = np.unique(y[positive_mask])
            if positive_classes.size < 2:
                raise ValueError(
                    "fit requires at least two classes with positive sample_weight; "
                    f"got {positive_classes.size} class"
                    f"{'es' if positive_classes.size != 1 else ''}"
                )

        return self.partial_fit(X, y, classes=classes, sample_weight=sample_weight)

    @_fit_context(prefer_skip_nested_validation=False)
    def partial_fit(self, X, y, classes=None, sample_weight=None):
        """Incrementally fit the model with a mini-batch of samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vectors, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        y : array-like of shape (n_samples,)
            Target values.

        classes : array-like of shape (n_classes,), default=None
            Array of class labels. This parameter is required on the first
            call to :meth:`partial_fit` and must contain all classes that will
            be observed. On subsequent calls it must be omitted or identical to
            the class labels provided initially.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights applied to individual samples.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        self._validate_params()
        first_call = not getattr(self, "_stats_initialized", False)
        if first_call:
            if classes is None:
                raise ValueError(
                    "classes must be provided on the first call to partial_fit"
                )
            classes = np.unique(classes)
            if classes.shape[0] < 2:
                raise ValueError(
                    "partial_fit requires at least two classes; "
                    f"got {classes.shape[0]} class"
                    f"{'es' if classes.shape[0] != 1 else ''}"
                )
            self.classes_ = classes
        else:
            if classes is not None:
                classes = np.unique(classes)
                if not np.array_equal(classes, self.classes_):
                    raise ValueError(
                        "`classes` parameter must contain the same values as during "
                        "initialization"
                    )

        X, y = validate_data(
            self,
            X,
            y,
            ensure_min_samples=1,
            dtype=[np.float64, np.float32],
            reset=first_call,
        )
        check_classification_targets(y)
        sample_weight = _check_sample_weight(sample_weight, X, dtype=X.dtype)

        if first_call:
            self._init_stats(self.classes_.shape[0], X.shape[1], X.dtype)
            self._stats_initialized = True

        encoded = np.searchsorted(self.classes_, y)
        if np.any(encoded >= self.classes_.shape[0]) or np.any(
            self.classes_[encoded] != y
        ):
            raise ValueError("y contains previously unseen labels")

        positive_mask = sample_weight > 0
        positive_batch = int(np.count_nonzero(positive_mask))
        current_active_mask = self._class_weight_sum_ > 0
        active_classes_before = int(np.count_nonzero(current_active_mask))
        if positive_batch > 0:
            classes_in_batch = np.unique(encoded[positive_mask])
            newly_active = int(np.count_nonzero(~current_active_mask[classes_in_batch]))
            active_classes_after = active_classes_before + newly_active
        else:
            active_classes_after = active_classes_before

        positive_count_after = self._positive_weight_count_ + positive_batch
        if (
            self.solver == "lsqr"
            and active_classes_after >= 2
            and active_classes_after == self.classes_.shape[0]
            and positive_count_after <= active_classes_after
        ):
            raise np.linalg.LinAlgError(
                "The number of samples must be more than the number of classes."
            )

        self._positive_weight_count_ = positive_count_after

        self._update_global_stats(X, sample_weight)
        self._update_class_stats(X, encoded, sample_weight)
        self._recompute_model(X.dtype)
        return self

    def transform(self, X):
        """Project the data onto the most discriminative components.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_components)
            Transformed data.

        Raises
        ------
        NotImplementedError
            If ``solver='lsqr'``.
        """
        if self.solver == "lsqr":
            raise NotImplementedError(
                "transform not implemented for 'lsqr' solver (use 'svd' or 'eigen')."
            )
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)
        if self.solver == "svd":
            X_new = (X - self.xbar_) @ self.scalings_
        else:  # eigen solver
            X_new = X @ self.scalings_
        return X_new[:, : self._max_components]

    def predict_proba(self, X):
        """Estimate class probabilities for the input samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
            Estimated class probabilities for each sample.
        """
        check_is_fitted(self)
        decision = self.decision_function(X)
        if size(self.classes_) == 2:
            proba = _expit(decision)
            return np.vstack([1 - proba, proba]).T
        return softmax(decision)

    def predict_log_proba(self, X):
        """Estimate the logarithm of class probabilities for input samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        log_proba : ndarray of shape (n_samples, n_classes)
            Logarithm of the estimated class probabilities.
        """
        prediction = self.predict_proba(X)
        info = np.finfo(prediction.dtype)
        smallest = getattr(info, "smallest_normal", info.tiny)
        prediction[prediction == 0.0] += smallest
        return np.log(prediction)

    def __sklearn_is_fitted__(self):
        return getattr(self, "_model_ready", False)

class QuadraticDiscriminantAnalysis(
    DiscriminantAnalysisPredictionMixin, ClassifierMixin, BaseEstimator
):
    """Quadratic Discriminant Analysis.

    A classifier with a quadratic decision boundary, generated
    by fitting class conditional densities to the data
    and using Bayes' rule.

    The model fits a Gaussian density to each class.

    .. versionadded:: 0.17

    For a comparison between
    :class:`~sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis`
    and :class:`~sklearn.discriminant_analysis.LinearDiscriminantAnalysis`, see
    :ref:`sphx_glr_auto_examples_classification_plot_lda_qda.py`.

    Read more in the :ref:`User Guide <lda_qda>`.

    Parameters
    ----------
    priors : array-like of shape (n_classes,), default=None
        Class priors. By default, the class proportions are inferred from the
        training data.

    reg_param : float, default=0.0
        Regularizes the per-class covariance estimates by transforming S2 as
        ``S2 = (1 - reg_param) * S2 + reg_param * np.eye(n_features)``,
        where S2 corresponds to the `scaling_` attribute of a given class.

    store_covariance : bool, default=False
        If True, the class covariance matrices are explicitly computed and
        stored in the `self.covariance_` attribute.

        .. versionadded:: 0.17

    tol : float, default=1.0e-4
        Absolute threshold for the covariance matrix to be considered rank
        deficient after applying some regularization (see `reg_param`) to each
        `Sk` where `Sk` represents covariance matrix for k-th class. This
        parameter does not affect the predictions. It controls when a warning
        is raised if the covariance matrix is not full rank.

        .. versionadded:: 0.17

    Attributes
    ----------
    covariance_ : list of len n_classes of ndarray \
            of shape (n_features, n_features)
        For each class, gives the covariance matrix estimated using the
        samples of that class. The estimations are unbiased. Only present if
        `store_covariance` is True.

    means_ : array-like of shape (n_classes, n_features)
        Class-wise means.

    priors_ : array-like of shape (n_classes,)
        Class priors (sum to 1).

    rotations_ : list of len n_classes of ndarray of shape (n_features, n_k)
        For each class k an array of shape (n_features, n_k), where
        ``n_k = min(n_features, number of elements in class k)``
        It is the rotation of the Gaussian distribution, i.e. its
        principal axis. It corresponds to `V`, the matrix of eigenvectors
        coming from the SVD of `Xk = U S Vt` where `Xk` is the centered
        matrix of samples from class k.

    scalings_ : list of len n_classes of ndarray of shape (n_k,)
        For each class, contains the scaling of
        the Gaussian distributions along its principal axes, i.e. the
        variance in the rotated coordinate system. It corresponds to `S^2 /
        (n_samples - 1)`, where `S` is the diagonal matrix of singular values
        from the SVD of `Xk`, where `Xk` is the centered matrix of samples
        from class k.

    classes_ : ndarray of shape (n_classes,)
        Unique class labels.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    LinearDiscriminantAnalysis : Linear Discriminant Analysis.

    Examples
    --------
    >>> from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
    >>> import numpy as np
    >>> X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    >>> y = np.array([1, 1, 1, 2, 2, 2])
    >>> clf = QuadraticDiscriminantAnalysis()
    >>> clf.fit(X, y)
    QuadraticDiscriminantAnalysis()
    >>> print(clf.predict([[-0.8, -1]]))
    [1]
    """

    _parameter_constraints: dict = {
        "priors": ["array-like", None],
        "reg_param": [Interval(Real, 0, 1, closed="both")],
        "store_covariance": ["boolean"],
        "tol": [Interval(Real, 0, None, closed="left")],
    }

    def __init__(
        self, *, priors=None, reg_param=0.0, store_covariance=False, tol=1.0e-4
    ):
        self.priors = priors
        self.reg_param = reg_param
        self.store_covariance = store_covariance
        self.tol = tol

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y):
        """Fit the model according to the given training data and parameters.

        .. versionchanged:: 0.19
            ``store_covariances`` has been moved to main constructor as
            ``store_covariance``.

        .. versionchanged:: 0.19
            ``tol`` has been moved to main constructor.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vector, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        y : array-like of shape (n_samples,)
            Target values (integers).

        Returns
        -------
        self : object
            Fitted estimator.
        """
        X, y = validate_data(self, X, y)
        check_classification_targets(y)
        self.classes_, y = np.unique(y, return_inverse=True)
        n_samples, n_features = X.shape
        n_classes = len(self.classes_)
        if n_classes < 2:
            raise ValueError(
                "The number of classes has to be greater than one; got %d class"
                % (n_classes)
            )
        if self.priors is None:
            self.priors_ = np.bincount(y) / float(n_samples)
        else:
            self.priors_ = np.array(self.priors)

        cov = None
        store_covariance = self.store_covariance
        if store_covariance:
            cov = []
        means = []
        scalings = []
        rotations = []
        for ind in range(n_classes):
            Xg = X[y == ind, :]
            meang = Xg.mean(0)
            means.append(meang)
            if len(Xg) == 1:
                raise ValueError(
                    "y has only 1 sample in class %s, covariance is ill defined."
                    % str(self.classes_[ind])
                )
            Xgc = Xg - meang
            # Xgc = U * S * V.T
            _, S, Vt = np.linalg.svd(Xgc, full_matrices=False)
            S2 = (S**2) / (len(Xg) - 1)
            S2 = ((1 - self.reg_param) * S2) + self.reg_param
            rank = np.sum(S2 > self.tol)
            if rank < n_features:
                warnings.warn(
                    f"The covariance matrix of class {ind} is not full rank. "
                    "Increasing the value of parameter `reg_param` might help"
                    " reducing the collinearity.",
                    linalg.LinAlgWarning,
                )
            if self.store_covariance or store_covariance:
                # cov = V * (S^2 / (n-1)) * V.T
                cov.append(np.dot(S2 * Vt.T, Vt))
            scalings.append(S2)
            rotations.append(Vt.T)
        if self.store_covariance or store_covariance:
            self.covariance_ = cov
        self.means_ = np.asarray(means)
        self.scalings_ = scalings
        self.rotations_ = rotations
        return self

    def _decision_function(self, X):
        # return log posterior, see eq (4.12) p. 110 of the ESL.
        check_is_fitted(self)

        X = validate_data(self, X, reset=False)
        norm2 = []
        for i in range(len(self.classes_)):
            R = self.rotations_[i]
            S = self.scalings_[i]
            Xm = X - self.means_[i]
            X2 = np.dot(Xm, R * (S ** (-0.5)))
            norm2.append(np.sum(X2**2, axis=1))
        norm2 = np.array(norm2).T  # shape = [len(X), n_classes]
        u = np.asarray([np.sum(np.log(s)) for s in self.scalings_])
        return -0.5 * (norm2 + u) + np.log(self.priors_)

    def decision_function(self, X):
        """Apply decision function to an array of samples.

        The decision function is equal (up to a constant factor) to the
        log-posterior of the model, i.e. `log p(y = k | x)`. In a binary
        classification setting this instead corresponds to the difference
        `log p(y = 1 | x) - log p(y = 0 | x)`. See :ref:`lda_qda_math`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Array of samples (test vectors).

        Returns
        -------
        C : ndarray of shape (n_samples,) or (n_samples, n_classes)
            Decision function values related to each class, per sample.
            In the two-class case, the shape is `(n_samples,)`, giving the
            log likelihood ratio of the positive class.
        """
        return super().decision_function(X)

    def predict(self, X):
        """Perform classification on an array of test vectors X.

        The predicted class C for each sample in X is returned.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Vector to be scored, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        Returns
        -------
        C : ndarray of shape (n_samples,)
            Estimated probabilities.
        """
        return super().predict(X)

    def predict_proba(self, X):
        """Return posterior probabilities of classification.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Array of samples/test vectors.

        Returns
        -------
        C : ndarray of shape (n_samples, n_classes)
            Posterior probabilities of classification per class.
        """
        # compute the likelihood of the underlying gaussian models
        # up to a multiplicative constant.
        return super().predict_proba(X)

    def predict_log_proba(self, X):
        """Return log of posterior probabilities of classification.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Array of samples/test vectors.

        Returns
        -------
        C : ndarray of shape (n_samples, n_classes)
            Posterior log-probabilities of classification per class.
        """
        # XXX : can do better to avoid precision overflows
        return super().predict_log_proba(X)
