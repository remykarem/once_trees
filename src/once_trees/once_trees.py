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
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import check_random_state
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from once_trees.node import _Node

__all__ = ["ReadOnceDecisionTreeClassifier"]

from once_trees.split import _impurity_from_counts

from once_trees.tree import _build_tree
from once_trees.tree_adapter import _SklearnTreeAdapter

# Sentinels matching sklearn.tree._tree
TREE_LEAF = -1
TREE_UNDEFINED = -2


# --------------------------------------------------------------------------- #
# Estimator
# --------------------------------------------------------------------------- #
class ReadOnceDecisionTreeClassifier(DecisionTreeClassifier):
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
    tree_ : object
        sklearn-compatible flat tree (arrays of ``children_left``,
        ``children_right``, ``feature``, ``threshold``, ``value``, ``impurity``,
        ``n_node_samples``, ``weighted_n_node_samples``, ``node_count``,
        ``max_depth``) — works with ``sklearn.tree.plot_tree`` /
        ``export_text``.
    _root_ : _Node
        Root of the fitted tree as a linked-node structure.
    """

    def __init__(self, criterion="gini", max_depth=None, min_samples_leaf=1,
                 min_impurity_decrease=0.0, random_state=None):
        # Subclasses DecisionTreeClassifier purely so sklearn.tree.plot_tree /
        # export_text accept us through their isinstance check. We override
        # fit/predict entirely; the parent's tree-building code never runs.
        super().__init__(
            criterion=criterion,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            min_impurity_decrease=min_impurity_decrease,
            random_state=random_state,
        )

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

        self._root_ = _build_tree(
            X, y_enc, depth=0, available=set(range(self.n_features_in_)),
            n_classes=self.n_classes_, criterion=self.criterion,
            max_depth=max_depth, min_samples_leaf=self.min_samples_leaf,
            min_impurity_decrease=self.min_impurity_decrease,
            feature_order=feature_order,
        )
        self.n_outputs_ = 1
        self.tree_ = _SklearnTreeAdapter(
            self._root_, self.n_features_in_, self.n_classes_, self.criterion)
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
        return np.vstack([self._route(row, self._root_) for row in X])

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    def _more_tags(self):
        return {"allow_nan": True}

    # -- introspection -------------------------------------------------------
    @property
    def feature_importances_(self) -> np.ndarray:
        """Mean decrease in impurity per feature, normalized to sum to 1.

        Computed sklearn-style: each internal node contributes
        ``(n_node / n_total) * (impurity - weighted child impurity)`` to its
        split feature; the per-feature totals are normalized to sum to 1
        (an all-zero vector is returned unchanged).
        """
        check_is_fitted(self)
        importances = np.zeros(self.n_features_in_, dtype=float)
        n_total = self._root_.n_samples

        def _counts(node):
            if node.is_leaf:
                return node.proba * node.n_samples
            cl = _counts(node.left)
            cr = _counts(node.right)
            c = cl + cr
            imp = _impurity_from_counts(c, self.criterion)
            imp_l = _impurity_from_counts(cl, self.criterion)
            imp_r = _impurity_from_counts(cr, self.criterion)
            weighted = (node.left.n_samples * imp_l
                        + node.right.n_samples * imp_r) / node.n_samples
            importances[node.feature] += (
                                             node.n_samples / n_total) * (imp - weighted)
            return c

        _counts(self._root_)
        total = importances.sum()
        if total > 0:
            importances /= total
        return importances

    def get_depth(self) -> int:
        """Depth of the fitted tree (a leaf-only tree has depth 0)."""
        check_is_fitted(self)

        def _d(node):
            return 0 if node.is_leaf else 1 + max(_d(node.left), _d(node.right))

        return _d(self._root_)
