import matplotlib
matplotlib.use("Agg")

import numpy as np
from sklearn.datasets import load_iris
from sklearn.tree import export_text, plot_tree

from once_trees.once_trees import ReadOnceDecisionTreeClassifier


def test_export_text_runs():
    X, y = load_iris(return_X_y=True)
    clf = ReadOnceDecisionTreeClassifier(random_state=0).fit(X, y)
    txt = export_text(clf, feature_names=[f"f{i}" for i in range(X.shape[1])])
    assert "f0" in txt or "f1" in txt or "f2" in txt or "f3" in txt


def test_plot_tree_runs():
    X, y = load_iris(return_X_y=True)
    clf = ReadOnceDecisionTreeClassifier(random_state=0).fit(X, y)
    annotations = plot_tree(clf)
    # plot_tree returns one annotation per node plus arrow labels (True/False).
    assert len(annotations) >= clf.tree_.node_count


def test_tree_adapter_shape_and_invariants():
    X, y = load_iris(return_X_y=True)
    clf = ReadOnceDecisionTreeClassifier(random_state=0).fit(X, y)
    t = clf.tree_
    n = t.node_count
    assert t.value.shape == (n, 1, clf.n_classes_)
    leaves = t.children_left == -1
    assert np.all(t.children_right[leaves] == -1)
    assert np.all(t.feature[leaves] == -2)
    # Internal-node child indices are in range and point forward.
    internal = ~leaves
    assert np.all(t.children_left[internal] > np.arange(n)[internal])
    assert np.all(t.children_right[internal] > np.arange(n)[internal])
