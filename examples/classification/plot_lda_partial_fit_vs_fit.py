"""
==========================================
Compare fit and partial_fit LDA on Iris
==========================================

This example demonstrates the classification accuracy of
:class:`~sklearn.discriminant_analysis.LinearDiscriminantAnalysis`
when trained in batch with :meth:`fit` versus incrementally with
:meth:`partial_fit`.
"""

from sklearn.datasets import load_iris
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

solvers = ["svd", "lsqr", "eigen"]

print("Accuracy using fit")
for solver in solvers:
    clf = LinearDiscriminantAnalysis(solver=solver).fit(X_train, y_train)
    acc = accuracy_score(y_test, clf.predict(X_test))
    print(f"{solver}: {acc:.3f}")

print("\nAccuracy using partial_fit")
for solver in solvers:
    clf_pf = LinearDiscriminantAnalysis(solver=solver)
    classes = np.unique(y_train)
    half = X_train.shape[0] // 2
    clf_pf.partial_fit(X_train[:half], y_train[:half], classes=classes)
    clf_pf.partial_fit(X_train[half:], y_train[half:])
    acc = accuracy_score(y_test, clf_pf.predict(X_test))
    print(f"{solver}: {acc:.3f}")
