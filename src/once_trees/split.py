from __future__ import annotations

import numpy as np


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
