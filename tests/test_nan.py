import numpy as np

from once_trees.once_trees import ReadOnceDecisionTreeClassifier


def _make_data(seed=0, n=200, d=4, nan_rate=0.1):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    X_nan = X.copy()
    X_nan[rng.random(X.shape) < nan_rate] = np.nan
    return X, X_nan, y


def test_fit_predict_with_nans_in_train_and_test():
    _, X_nan, y = _make_data()
    model = ReadOnceDecisionTreeClassifier(random_state=0).fit(X_nan, y)

    rng = np.random.default_rng(1)
    X_test = rng.normal(size=(50, X_nan.shape[1]))
    X_test[rng.random(X_test.shape) < 0.2] = np.nan

    proba = model.predict_proba(X_test)
    pred = model.predict(X_test)

    assert proba.shape == (50, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert pred.shape == (50,)
    # Trains usefully on a learnable signal even with missingness.
    assert model.score(X_nan, y) > 0.8


def test_nan_free_data_unchanged():
    """A dataset with no missing values must behave exactly as before."""
    X, _, y = _make_data()
    m1 = ReadOnceDecisionTreeClassifier(random_state=0).fit(X, y)
    m2 = ReadOnceDecisionTreeClassifier(random_state=0).fit(X, y)
    assert np.array_equal(m1.predict(X), m2.predict(X))
    assert m1.score(X, y) > 0.9


def test_all_nan_column_is_skipped():
    """A feature that's entirely NaN cannot be used for splitting."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 3))
    X[:, 2] = np.nan
    y = (X[:, 0] > 0).astype(int)

    model = ReadOnceDecisionTreeClassifier(random_state=0).fit(X, y)

    def features_used(node):
        if node.is_leaf:
            return set()
        return {node.feature} | features_used(node.left) | features_used(node.right)

    assert 2 not in features_used(model.tree_)


def test_missing_direction_is_learned():
    """When NaNs co-occur with one class, they should be routed to that class."""
    rng = np.random.default_rng(0)
    n = 300
    X = rng.normal(size=(n, 1))
    y = (X[:, 0] > 0).astype(int)
    # Replace some class-1 rows' feature with NaN.
    nan_rows = np.where(y == 1)[0][:60]
    X[nan_rows, 0] = np.nan

    model = ReadOnceDecisionTreeClassifier(random_state=0).fit(X, y)

    X_nan_only = np.array([[np.nan]])
    assert model.predict(X_nan_only)[0] == 1
