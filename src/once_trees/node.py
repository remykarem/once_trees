from __future__ import annotations


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
