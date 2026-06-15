from __future__ import annotations

import numpy as np

from once_trees.node import _Node
from once_trees.once_trees import TREE_LEAF, TREE_UNDEFINED
from once_trees.split import _impurity_from_counts


class _SklearnTreeAdapter:
    """Flat-array view of a ``_Node`` tree, mimicking ``sklearn.tree._tree.Tree``.

    sklearn-compatible flat Tree (duck-typed for plot_tree / export_text)

    Exposes the array attributes that ``sklearn.tree.plot_tree``,
    ``export_text`` and ``export_graphviz`` read. Single-output only.
    """

    def __init__(self, root: _Node, n_features: int, n_classes: int,
                 criterion: str):
        nodes = []  # list of _Node in pre-order

        def walk(node):
            idx = len(nodes)
            nodes.append(node)
            if node.is_leaf:
                return idx
            walk(node.left)
            walk(node.right)
            return idx

        walk(root)
        n = len(nodes)

        self.node_count = n
        self.n_features = n_features
        self.n_classes = np.array([n_classes], dtype=np.intp)
        self.n_outputs = 1
        self.children_left = np.full(n, TREE_LEAF, dtype=np.intp)
        self.children_right = np.full(n, TREE_LEAF, dtype=np.intp)
        self.feature = np.full(n, TREE_UNDEFINED, dtype=np.intp)
        self.threshold = np.full(n, float(TREE_UNDEFINED), dtype=np.float64)
        self.impurity = np.zeros(n, dtype=np.float64)
        self.n_node_samples = np.zeros(n, dtype=np.intp)
        self.weighted_n_node_samples = np.zeros(n, dtype=np.float64)
        self.value = np.zeros((n, 1, n_classes), dtype=np.float64)

        # Second pass: fill arrays. We need child indices, so re-walk while
        # tracking the index we assigned above.
        id_of = {id(node): i for i, node in enumerate(nodes)}

        def fill(node):
            i = id_of[id(node)]
            self.n_node_samples[i] = node.n_samples
            self.weighted_n_node_samples[i] = node.n_samples
            if node.is_leaf:
                counts = node.proba * node.n_samples
                self.value[i, 0, :] = counts / counts.sum() if counts.sum() else counts
                self.impurity[i] = _impurity_from_counts(counts, criterion)
                return counts
            cl = fill(node.left)
            cr = fill(node.right)
            counts = cl + cr
            self.value[i, 0, :] = counts / counts.sum() if counts.sum() else counts
            self.impurity[i] = _impurity_from_counts(counts, criterion)
            self.feature[i] = node.feature
            self.threshold[i] = node.threshold
            self.children_left[i] = id_of[id(node.left)]
            self.children_right[i] = id_of[id(node.right)]
            return counts

        fill(root)

        def depth(node):
            return 0 if node.is_leaf else 1 + max(depth(node.left),
                                                  depth(node.right))

        self.max_depth = depth(root)
