"""Read-once decision tree classifier.

A read-once tree tests each feature **at most once on any root-to-leaf path**
(the standard "read-once" property from the boolean-function literature). A
feature may still appear in different branches of the tree, just never twice
along a single path. For continuous features this means each feature
contributes a single threshold cut per path, which limits expressiveness in
exchange for interpretability.

Only numeric features are supported (binary threshold splits).
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_random_state
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

__all__ = ["ReadOnceDecisionTreeClassifier"]


# --------------------------------------------------------------------------- #
# Tree node
# --------------------------------------------------------------------------- #
class _Node:
    """A single node: an internal split or a leaf.

    Leaves carry ``proba`` (class distribution aligned to ``classes_``).
    Internal nodes carry ``feature``, ``threshold`` and the two children;
    a sample goes ``left`` iff ``x[feature] <= threshold``.
    """

    __slots__ = ("feature", "threshold", "left", "right", "proba", "n_samples",
                 "missing_go_left")

    def __init__(self, *, proba=None, n_samples=0, feature=None,
                 threshold=None, left=None, right=None, missing_go_left=True):
        self.proba = proba
        self.n_samples = n_samples
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.missing_go_left = missing_go_left

    @property
    def is_leaf(self) -> bool:
        return self.feature is None


# --------------------------------------------------------------------------- #
# Impurity
# --------------------------------------------------------------------------- #
def _impurity_from_counts(counts: np.ndarray, criterion: str) -> float:
    """Impurity of a node given its per-class sample counts."""
    total = counts.sum()
    if total == 0:
        return 0.0
    p = counts / total
    if criterion == "gini":
        return 1.0 - np.dot(p, p)
    # entropy
    nz = p > 0
    return -np.sum(p[nz] * np.log2(p[nz]))


# --------------------------------------------------------------------------- #
# Split finding (operates in ORIGINAL feature-index space)
# --------------------------------------------------------------------------- #
def _best_split(X, y_enc, n_classes, available, criterion, min_samples_leaf,
                feature_order):
    """Find the best (feature, threshold) over ``available`` features.

    Returns ``(decrease, feature, threshold, missing_go_left)`` for the
    impurity-reducing split, or ``None`` if no valid split exists.
    ``feature_order`` decides tie-breaks. Missing values (NaN) do not
    generate candidate thresholds, but at each candidate they are routed
    to whichever child yields the larger impurity decrease (sklearn's
    ``DecisionTreeClassifier`` behavior).
    """
    n_samples = X.shape[0]
    total_counts = np.bincount(y_enc, minlength=n_classes).astype(float)
    parent_impurity = _impurity_from_counts(total_counts, criterion)

    best = None  # (decrease, feature, threshold, missing_go_left)
    for f in feature_order:
        if f not in available:
            continue
        col = X[:, f]
        nan_mask = np.isnan(col)
        n_nan = int(nan_mask.sum())
        n_finite = n_samples - n_nan
        if n_finite < 2:
            continue

        finite_idx = np.where(~nan_mask)[0]
        order = finite_idx[np.argsort(col[finite_idx], kind="mergesort")]
        x_sorted = col[order]
        y_sorted = y_enc[order]

        nan_counts = (np.bincount(y_enc[nan_mask], minlength=n_classes).astype(float)
                      if n_nan else np.zeros(n_classes))
        finite_counts = total_counts - nan_counts

        left_counts = np.zeros(n_classes)
        right_counts = finite_counts.copy()

        for i in range(1, n_finite):
            c = y_sorted[i - 1]
            left_counts[c] += 1.0
            right_counts[c] -= 1.0
            # Never split between two identical feature values.
            if x_sorted[i] == x_sorted[i - 1]:
                continue

            # Try sending missing rows to whichever side helps more. When
            # there are no missing rows the two branches are identical and
            # we keep ``missing_go_left=True`` as the convention default.
            directions = (True,) if n_nan == 0 else (True, False)
            for missing_left in directions:
                if missing_left:
                    lc = left_counts + nan_counts
                    rc = right_counts
                    n_left = i + n_nan
                    n_right = n_finite - i
                else:
                    lc = left_counts
                    rc = right_counts + nan_counts
                    n_left = i
                    n_right = (n_finite - i) + n_nan

                if n_left < min_samples_leaf or n_right < min_samples_leaf:
                    continue

                imp_left = _impurity_from_counts(lc, criterion)
                imp_right = _impurity_from_counts(rc, criterion)
                weighted = (n_left * imp_left + n_right * imp_right) / n_samples
                decrease = parent_impurity - weighted

                if best is None or decrease > best[0]:
                    threshold = (x_sorted[i] + x_sorted[i - 1]) / 2.0
                    best = (decrease, f, threshold, missing_left)

    return best


# --------------------------------------------------------------------------- #
# Recursive tree construction
# --------------------------------------------------------------------------- #
def _leaf(y_enc, n_classes) -> _Node:
    counts = np.bincount(y_enc, minlength=n_classes).astype(float)
    return _Node(proba=counts / counts.sum(), n_samples=len(y_enc))


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


# --------------------------------------------------------------------------- #
# Estimator
# --------------------------------------------------------------------------- #
class ReadOnceDecisionTreeClassifier(ClassifierMixin, BaseEstimator):
    """A decision tree that uses each feature at most once per root-to-leaf path.

    Parameters
    ----------
    criterion : {"gini", "entropy"}, default="gini"
        Impurity measure for evaluating splits.
    max_depth : int or None, default=None
        Maximum tree depth. ``None`` means unlimited (still naturally bounded
        by ``n_features_in_`` under the read-once constraint).
    min_samples_leaf : int, default=1
        Minimum samples required in each child of a split.
    min_impurity_decrease : float, default=0.0
        A split is only accepted if it reduces impurity by strictly more than
        this value.
    random_state : int, RandomState instance or None, default=None
        Controls the feature-iteration order used to break ties between
        equally good splits. Split selection is otherwise deterministic.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
    n_classes_ : int
    n_features_in_ : int
    tree_ : _Node
        Root of the fitted tree.
    """

    def __init__(self, criterion="gini", max_depth=None, min_samples_leaf=1,
                 min_impurity_decrease=0.0, random_state=None):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.random_state = random_state

    # -- fitting -------------------------------------------------------------
    def fit(self, X, y):
        X, y = check_X_y(X, y, ensure_all_finite="allow-nan")
        if self.criterion not in ("gini", "entropy"):
            raise ValueError("criterion must be 'gini' or 'entropy'")

        self.classes_ = unique_labels(y)
        self.n_classes_ = self.classes_.shape[0]
        self.n_features_in_ = X.shape[1]

        # Encode labels to 0..K-1 aligned with classes_ (sorted by unique_labels).
        y_enc = np.searchsorted(self.classes_, y).astype(np.intp)

        rng = check_random_state(self.random_state)
        feature_order = rng.permutation(self.n_features_in_)
        max_depth = np.inf if self.max_depth is None else self.max_depth

        self.tree_ = _build_tree(
            X, y_enc, depth=0, available=set(range(self.n_features_in_)),
            n_classes=self.n_classes_, criterion=self.criterion,
            max_depth=max_depth, min_samples_leaf=self.min_samples_leaf,
            min_impurity_decrease=self.min_impurity_decrease,
            feature_order=feature_order,
        )
        return self

    # -- prediction ----------------------------------------------------------
    def _route(self, row, node: _Node):
        while not node.is_leaf:
            val = row[node.feature]
            if np.isnan(val):
                node = node.left if node.missing_go_left else node.right
            else:
                node = node.left if val <= node.threshold else node.right
        return node.proba

    def predict_proba(self, X):
        check_is_fitted(self)
        X = check_array(X, ensure_all_finite="allow-nan")
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, expected {self.n_features_in_}")
        return np.vstack([self._route(row, self.tree_) for row in X])

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    def _more_tags(self):
        return {"allow_nan": True}

    # -- introspection -------------------------------------------------------
    def get_depth(self) -> int:
        """Depth of the fitted tree (a leaf-only tree has depth 0)."""
        check_is_fitted(self)

        def _d(node):
            return 0 if node.is_leaf else 1 + max(_d(node.left), _d(node.right))

        return _d(self.tree_)
