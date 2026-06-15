import numpy as np
import pytest
from sklearn.datasets import load_iris

from once_trees.once_trees import ReadOnceDecisionTreeClassifier


def test_feature_importances_sum_to_one_and_shape():
    X, y = load_iris(return_X_y=True)
    clf = ReadOnceDecisionTreeClassifier(random_state=0).fit(X, y)
    fi = clf.feature_importances_
    assert fi.shape == (X.shape[1],)
    assert np.all(fi >= 0)
    assert fi.sum() == pytest.approx(1.0)


def test_feature_importances_zero_for_unused_features():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 4))
    y = (X[:, 0] > 0).astype(int)
    X[:, 2:] = 0.0  # constant -> never split on
    clf = ReadOnceDecisionTreeClassifier(random_state=0).fit(X, y)
    fi = clf.feature_importances_
    assert fi[2] == 0.0
    assert fi[3] == 0.0
    assert fi[0] > 0


def test_feature_importances_leaf_only_tree_is_all_zero():
    X = np.zeros((10, 3))
    y = np.zeros(10, dtype=int)
    clf = ReadOnceDecisionTreeClassifier().fit(X, y)
    fi = clf.feature_importances_
    assert fi.shape == (3,)
    assert np.all(fi == 0.0)


def test_feature_importances_not_fitted_raises():
    from sklearn.exceptions import NotFittedError
    clf = ReadOnceDecisionTreeClassifier()
    with pytest.raises(NotFittedError):
        clf.feature_importances_
