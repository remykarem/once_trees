from __future__ import annotations

import numpy as np

from once_trees.node import _Node
from once_trees.split import _best_split


def _build_tree(X, y_enc, *, depth, available, n_classes, criterion,
                max_depth, min_samples_leaf, min_impurity_decrease,
                feature_order) -> _Node:
    n_samples = X.shape[0]
    counts = np.bincount(y_enc, minlength=n_classes)

    # --- stopping conditions -------------------------------------------------
    pure = np.count_nonzero(counts) <= 1
    if (pure or depth >= max_depth or not available
        or n_samples < 2 * min_samples_leaf):
        return _leaf(y_enc, n_classes)

    split = _best_split(X, y_enc, n_classes, available, criterion,
                        min_samples_leaf, feature_order)
    if split is None or split[0] <= min_impurity_decrease:
        return _leaf(y_enc, n_classes)

    _, feature, threshold, missing_go_left = split
    col = X[:, feature]
    mask = col <= threshold  # NaN comparisons yield False
    if missing_go_left:
        mask = mask | np.isnan(col)

    # The read-once constraint, in one line: the chosen feature is removed
    # from the set handed to BOTH subtrees, so it can never recur on this path.
    child_available = available - {feature}

    left = _build_tree(
        X[mask], y_enc[mask], depth=depth + 1, available=child_available,
        n_classes=n_classes, criterion=criterion, max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        min_impurity_decrease=min_impurity_decrease, feature_order=feature_order,
    )
    right = _build_tree(
        X[~mask], y_enc[~mask], depth=depth + 1, available=child_available,
        n_classes=n_classes, criterion=criterion, max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        min_impurity_decrease=min_impurity_decrease, feature_order=feature_order,
    )
    return _Node(feature=feature, threshold=threshold, left=left, right=right,
                 n_samples=n_samples, missing_go_left=missing_go_left)


def _leaf(y_enc, n_classes) -> _Node:
    counts = np.bincount(y_enc, minlength=n_classes).astype(float)
    return _Node(proba=counts / counts.sum(), n_samples=len(y_enc))
