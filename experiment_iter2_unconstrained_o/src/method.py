#!/usr/bin/env python3
"""Unconstrained Oblique Baselines: ManualSPORF + Oblique FIGS on 10 Datasets.

Runs Manual SPORF (sklearn RandomForest + random sparse projections), custom
Unconstrained Oblique FIGS, and axis-aligned FIGS on 10 tabular datasets with
5-fold CV, measuring accuracy/R² and direction diversity metrics (eRank, stable
rank, participation ratio, near-duplicate count).
"""

from loguru import logger
from pathlib import Path
import json
import sys
import time
import resource
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error, roc_auc_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from imodels import FIGSClassifier, FIGSRegressor

# ── Logging setup ──────────────────────────────────────────────────────────────
logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add("logs/run.log", rotation="30 MB", level="DEBUG")

# ── Resource limits ────────────────────────────────────────────────────────────
# Leave headroom: 41 GB total, cap at 30 GB; 1h CPU time
try:
    resource.setrlimit(resource.RLIMIT_AS, (30 * 1024**3, 30 * 1024**3))
    resource.setrlimit(resource.RLIMIT_CPU, (3600, 3600))
except ValueError:
    logger.warning("Could not set resource limits")

# ── Paths ──────────────────────────────────────────────────────────────────────
WORKSPACE = Path(__file__).parent
DATA_DEP = Path(
    "/home/adrian/projects/temp/ai-inventor-old/aii_pipeline/runs/"
    "run__20260227_195308/3_invention_loop/iter_1/gen_art/data_id2_it1__opus"
)
FULL_DATA_PATH = DATA_DEP / "full_data_out.json"
MINI_DATA_PATH = DATA_DEP / "mini_data_out.json"

Path("logs").mkdir(exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 1: Direction Diversity Metrics
# ═══════════════════════════════════════════════════════════════════════════════

def compute_direction_diversity(W: np.ndarray) -> dict:
    """Compute direction diversity metrics from a matrix of split directions.

    Args:
        W: np.array of shape (N_splits, d) — each row is a split direction vector.

    Returns:
        Dictionary of diversity metrics.
    """
    if W.shape[0] < 2:
        return {
            "erank": 1.0, "stable_rank": 1.0, "participation_ratio": 1.0,
            "n_near_duplicates": 0, "avg_pairwise_cosine_sim": 0.0,
            "n_total_splits": int(W.shape[0]),
            "svd_top5_singular_values": [1.0],
        }

    # Normalize each row to unit norm
    norms = np.linalg.norm(W, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    W_normed = W / norms

    # SVD
    _U, sigma, _Vt = np.linalg.svd(W_normed, full_matrices=False)
    sigma = sigma[sigma > 1e-10]

    if len(sigma) == 0:
        return {
            "erank": 1.0, "stable_rank": 1.0, "participation_ratio": 1.0,
            "n_near_duplicates": 0, "avg_pairwise_cosine_sim": 0.0,
            "n_total_splits": int(W.shape[0]),
            "svd_top5_singular_values": [],
        }

    sigma_sq = sigma ** 2

    # 1. eRank = exp(Shannon entropy of normalized singular values)
    p = sigma_sq / sigma_sq.sum()
    p = p[p > 1e-15]
    erank = float(np.exp(-np.sum(p * np.log(p))))

    # 2. Stable rank = ||W||_F^2 / ||W||_2^2
    stable_rank = float(sigma_sq.sum() / sigma_sq[0])

    # 3. Participation ratio = (sum sigma_i^2)^2 / sum(sigma_i^4)
    participation_ratio = float((sigma_sq.sum())**2 / (sigma_sq**2).sum())

    # 4. Near-duplicate directions and average cosine similarity
    n = W_normed.shape[0]
    if n <= 500:
        cos_sim = np.abs(W_normed @ W_normed.T)
        np.fill_diagonal(cos_sim, 0)
        n_near_duplicates = int(np.sum(cos_sim > 0.95)) // 2
        avg_cos_sim = float(cos_sim.sum() / (n * (n - 1))) if n > 1 else 0.0
    else:
        n_pairs = 500
        rng = np.random.RandomState(42)
        idx1 = rng.choice(n, n_pairs)
        idx2 = rng.choice(n, n_pairs)
        cos_sims = np.abs(np.sum(W_normed[idx1] * W_normed[idx2], axis=1))
        n_near_duplicates = int(np.sum(cos_sims > 0.95))
        avg_cos_sim = float(np.mean(cos_sims))

    return {
        "erank": erank,
        "stable_rank": stable_rank,
        "participation_ratio": participation_ratio,
        "n_near_duplicates": n_near_duplicates,
        "avg_pairwise_cosine_sim": avg_cos_sim,
        "n_total_splits": int(W.shape[0]),
        "svd_top5_singular_values": sigma[:5].tolist(),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 2: Manual SPORF (Fallback for treeple)
# ═══════════════════════════════════════════════════════════════════════════════

class ManualSPORF:
    """SPORF implementation using sklearn trees + random sparse projections.

    For each tree:
      1. Generate random sparse projection matrix P of shape (n_projected, n_features)
      2. Transform X_proj = X @ P.T
      3. Fit sklearn DecisionTree on X_proj
      4. Store P to reconstruct oblique directions
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int | None = None,
        feature_combinations: float = 1.5,
        random_state: int = 42,
        task_type: str = "classification",
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.feature_combinations = feature_combinations
        self.random_state = random_state
        self.task_type = task_type
        self.estimators_ = []
        self.projection_matrices_ = []

    def _generate_projection_matrix(
        self, n_features: int, rng: np.random.RandomState
    ) -> np.ndarray:
        """Generate a random sparse projection matrix.

        Each row is a random direction in n_features space.
        Number of projections = n_features (matching sklearn tree dimensionality).
        Each projected feature combines ~feature_combinations original features.
        """
        n_proj = n_features
        P = np.zeros((n_proj, n_features))
        prob = self.feature_combinations / n_features
        prob = min(prob, 1.0)
        for i in range(n_proj):
            mask = rng.random(n_features) < prob
            if not mask.any():
                mask[rng.randint(n_features)] = True
            P[i, mask] = rng.choice([-1.0, 1.0], size=mask.sum())
        return P

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ManualSPORF":
        """Fit the SPORF ensemble."""
        n_features = X.shape[1]
        rng = np.random.RandomState(self.random_state)

        self.estimators_ = []
        self.projection_matrices_ = []

        for i in range(self.n_estimators):
            seed = rng.randint(0, 2**31)
            P = self._generate_projection_matrix(n_features, np.random.RandomState(seed))
            X_proj = X @ P.T

            if self.task_type == "classification":
                tree = DecisionTreeClassifier(
                    max_depth=self.max_depth,
                    random_state=seed,
                    max_features="sqrt",
                )
            else:
                tree = DecisionTreeRegressor(
                    max_depth=self.max_depth,
                    random_state=seed,
                    max_features="sqrt",
                )

            # Bootstrap sample
            n = X.shape[0]
            boot_idx = rng.choice(n, size=n, replace=True)
            tree.fit(X_proj[boot_idx], y[boot_idx])

            self.estimators_.append(tree)
            self.projection_matrices_.append(P)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using majority vote (classification) or mean (regression)."""
        if self.task_type == "classification":
            all_preds = np.array([
                est.predict(X @ P.T)
                for est, P in zip(self.estimators_, self.projection_matrices_)
            ])
            # Majority vote
            from scipy.stats import mode
            result = mode(all_preds, axis=0, keepdims=False)
            return result.mode.flatten()
        else:
            all_preds = np.array([
                est.predict(X @ P.T)
                for est, P in zip(self.estimators_, self.projection_matrices_)
            ])
            return np.mean(all_preds, axis=0)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities (average across trees)."""
        all_proba = []
        for est, P in zip(self.estimators_, self.projection_matrices_):
            proba = est.predict_proba(X @ P.T)
            all_proba.append(proba)

        # Handle possibly different number of classes across trees
        n_classes = max(p.shape[1] for p in all_proba)
        aligned = []
        for p in all_proba:
            if p.shape[1] < n_classes:
                padded = np.zeros((p.shape[0], n_classes))
                padded[:, :p.shape[1]] = p
                aligned.append(padded)
            else:
                aligned.append(p)

        return np.mean(aligned, axis=0)

    def extract_split_directions(self, n_features: int) -> np.ndarray:
        """Extract oblique split directions in original feature space.

        For each tree's internal node splitting on projected feature j,
        the oblique direction = P[j, :].
        """
        directions = []
        for est, P in zip(self.estimators_, self.projection_matrices_):
            tree = est.tree_
            TREE_LEAF = -1
            for node_idx in range(tree.node_count):
                if tree.children_left[node_idx] != TREE_LEAF:
                    feat = tree.feature[node_idx]
                    if 0 <= feat < P.shape[0]:
                        direction = P[feat].copy()
                        if np.linalg.norm(direction) > 1e-10:
                            directions.append(direction)

        if directions:
            return np.array(directions)
        return np.zeros((0, n_features))


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 3: Unconstrained Oblique FIGS
# ═══════════════════════════════════════════════════════════════════════════════

class ObliqueFIGSNode:
    """Node in an Oblique FIGS tree."""
    __slots__ = [
        'weights', 'threshold', 'value', 'left', 'right',
        'left_temp', 'right_temp', 'idxs', 'is_root', 'tree_num',
        'depth', 'impurity', 'impurity_reduction',
    ]

    def __init__(self):
        self.weights = None
        self.threshold = None
        self.value = None
        self.left = None
        self.right = None
        self.left_temp = None
        self.right_temp = None
        self.idxs = None
        self.is_root = False
        self.tree_num = -1
        self.depth = 0
        self.impurity = 0.0
        self.impurity_reduction = None


class ObliqueFIGS:
    """FIGS with unconstrained oblique splits.

    Uses the greedy-additive-tree algorithm from FIGS but replaces axis-aligned
    splits with oblique splits found via Ridge regression, PCA, and random
    projections.
    """

    def __init__(
        self,
        max_rules: int = 12,
        max_trees: int | None = None,
        max_depth: int | None = None,
        min_impurity_decrease: float = 0.0,
        n_candidate_directions: int = 5,
        task_type: str = "classification",
    ):
        self.max_rules = max_rules
        self.max_trees = max_trees
        self.max_depth = max_depth
        self.min_impurity_decrease = min_impurity_decrease
        self.n_candidate_directions = n_candidate_directions
        self.task_type = task_type
        self.trees_ = []
        self.complexity_ = 0
        self.classes_ = None

    def _compute_impurity_reduction(
        self, y: np.ndarray, left_mask: np.ndarray, right_mask: np.ndarray,
    ) -> float:
        """Compute MSE-based impurity reduction.

        FIGS always uses MSE on residuals, even for classification.
        """
        n = len(y)
        n_left = int(left_mask.sum())
        n_right = int(right_mask.sum())
        if n_left < 2 or n_right < 2:
            return -np.inf

        if y.ndim == 1:
            parent_var = np.var(y) * n
            left_var = np.var(y[left_mask]) * n_left
            right_var = np.var(y[right_mask]) * n_right
        else:
            parent_var = np.sum(np.var(y, axis=0)) * n
            left_var = np.sum(np.var(y[left_mask], axis=0)) * n_left
            right_var = np.sum(np.var(y[right_mask], axis=0)) * n_right

        return float((parent_var - left_var - right_var) / n)

    def _find_best_oblique_split(
        self, X: np.ndarray, y: np.ndarray, idxs: np.ndarray
    ) -> ObliqueFIGSNode | None:
        """Find the best oblique split for data at this node."""
        X_node = X[idxs]
        y_node = y[idxs]
        n_node, d = X_node.shape

        if n_node < 5:
            return None

        candidate_directions = []

        # Direction source 1: Ridge regression (fast implementation)
        try:
            if n_node <= 2000:
                # For smaller nodes, Ridge is informative and fast enough
                XtX = X_node.T @ X_node
                XtX[np.diag_indices_from(XtX)] += 1.0
                if y_node.ndim == 1:
                    Xty = X_node.T @ y_node
                else:
                    # For multi-output, project y to 1D using first PC of y
                    if y_node.shape[1] <= 2:
                        y_1d = y_node[:, 0] - y_node[:, -1]
                    else:
                        y_mean = y_node - y_node.mean(axis=0)
                        _U, _s, Vt = np.linalg.svd(y_mean, full_matrices=False)
                        y_1d = y_mean @ Vt[0]
                    Xty = X_node.T @ y_1d
                w = np.linalg.solve(XtX, Xty).flatten()
                if w.shape[0] == d:
                    w_norm = np.linalg.norm(w)
                    if w_norm > 1e-10:
                        candidate_directions.append(w / w_norm)
        except Exception:
            pass

        # Direction source 2: PCA top-2 directions (fast implementation)
        try:
            X_centered = X_node - X_node.mean(axis=0)
            # Use covariance matrix approach (faster for n > d)
            if n_node > d:
                cov = (X_centered.T @ X_centered) / n_node
                eigvals, eigvecs = np.linalg.eigh(cov)
                # Top 2 eigenvectors (eigh returns ascending order)
                for j in range(min(2, d)):
                    comp = eigvecs[:, -(j + 1)]
                    candidate_directions.append(comp)
            else:
                # For n < d, use SVD on the data directly
                _U, _s, Vt = np.linalg.svd(X_centered, full_matrices=False)
                for j in range(min(2, Vt.shape[0])):
                    candidate_directions.append(Vt[j])
        except Exception:
            pass

        # Direction source 3: Random sparse projections (2 random directions)
        rng = np.random.RandomState(abs(hash(idxs.tobytes())) % (2**31))
        for _ in range(2):
            w_rand = np.zeros(d)
            k = max(2, int(1.5 * np.sqrt(d)))
            feats = rng.choice(d, size=min(k, d), replace=False)
            w_rand[feats] = rng.standard_normal(len(feats))
            w_rand_norm = np.linalg.norm(w_rand)
            if w_rand_norm > 1e-10:
                candidate_directions.append(w_rand / w_rand_norm)

        # Direction source 4: axis-aligned directions (top 3 by variance)
        if y_node.ndim == 1:
            var_per_feat = np.array([
                np.var(y_node[X_node[:, j] <= np.median(X_node[:, j])]) * (n_node // 2)
                + np.var(y_node[X_node[:, j] > np.median(X_node[:, j])]) * (n_node - n_node // 2)
                for j in range(min(d, 10))
            ])
            top_feats = np.argsort(var_per_feat)[:3]
        else:
            top_feats = rng.choice(d, size=min(3, d), replace=False)
        for j in top_feats:
            e_j = np.zeros(d)
            e_j[j] = 1.0
            candidate_directions.append(e_j)

        # For each candidate direction, find optimal threshold using fast vectorized search
        best_reduction = -np.inf
        best_weights = None
        best_threshold = None

        for w in candidate_directions:
            # Safety: ensure w is a proper 1D vector of size d
            w = np.asarray(w).flatten()
            if w.ndim != 1 or w.shape[0] != d:
                continue
            z = X_node @ w

            # Fast threshold search using sorted-order cumulative statistics
            sort_idx = np.argsort(z)
            z_sorted = z[sort_idx]
            y_sorted = y_node[sort_idx]

            n_total = len(z_sorted)
            if n_total < 4:
                continue

            # Subsample threshold positions for large datasets
            if n_total > 60:
                pct_positions = np.unique(np.percentile(
                    np.arange(n_total), np.linspace(5, 95, 25)
                ).astype(int))
                pct_positions = pct_positions[(pct_positions >= 2) & (pct_positions < n_total - 2)]
            else:
                pct_positions = np.arange(2, n_total - 2)

            if len(pct_positions) == 0:
                continue

            # Compute impurity reductions at candidate thresholds
            if y_sorted.ndim == 1:
                cumsum = np.cumsum(y_sorted)
                cumsum_sq = np.cumsum(y_sorted ** 2)
                total_sum = cumsum[-1]
                total_sum_sq = cumsum_sq[-1]

                for pos in pct_positions:
                    n_left = pos + 1
                    n_right = n_total - n_left
                    if n_left < 2 or n_right < 2:
                        continue

                    left_sum = cumsum[pos]
                    left_sq = cumsum_sq[pos]
                    right_sum = total_sum - left_sum
                    right_sq = total_sum_sq - left_sq

                    left_var = left_sq / n_left - (left_sum / n_left) ** 2
                    right_var = right_sq / n_right - (right_sum / n_right) ** 2
                    parent_var = total_sum_sq / n_total - (total_sum / n_total) ** 2

                    reduction = parent_var - (n_left * left_var + n_right * right_var) / n_total
                    if reduction > best_reduction:
                        best_reduction = float(reduction)
                        best_weights = w.copy()
                        best_threshold = float((z_sorted[pos] + z_sorted[pos + 1]) / 2)
            else:
                # Multi-output case
                n_out = y_sorted.shape[1]
                cumsum = np.cumsum(y_sorted, axis=0)
                cumsum_sq = np.cumsum(y_sorted ** 2, axis=0)
                total_sum = cumsum[-1]
                total_sum_sq = cumsum_sq[-1]

                for pos in pct_positions:
                    n_left = pos + 1
                    n_right = n_total - n_left
                    if n_left < 2 or n_right < 2:
                        continue

                    left_sum = cumsum[pos]
                    left_sq = cumsum_sq[pos]
                    right_sum = total_sum - left_sum
                    right_sq = total_sum_sq - left_sq

                    left_var = np.sum(left_sq / n_left - (left_sum / n_left) ** 2)
                    right_var = np.sum(right_sq / n_right - (right_sum / n_right) ** 2)
                    parent_var = np.sum(total_sum_sq / n_total - (total_sum / n_total) ** 2)

                    reduction = parent_var - (n_left * left_var + n_right * right_var) / n_total
                    if reduction > best_reduction:
                        best_reduction = float(reduction)
                        best_weights = w.copy()
                        best_threshold = float((z_sorted[pos] + z_sorted[pos + 1]) / 2)

        if best_weights is None or best_reduction <= 0:
            return None

        # Build node
        node = ObliqueFIGSNode()
        node.weights = best_weights
        node.threshold = best_threshold
        node.impurity_reduction = best_reduction
        node.impurity = float(np.var(y_node) if y_node.ndim == 1 else np.sum(np.var(y_node, axis=0)))
        node.idxs = idxs.copy()
        node.value = np.mean(y_node, axis=0)

        # Create temp children
        full_z = X @ best_weights
        left_idxs = idxs & (full_z <= best_threshold)
        right_idxs = idxs & (full_z > best_threshold)

        node.left_temp = ObliqueFIGSNode()
        node.left_temp.idxs = left_idxs
        node.left_temp.value = np.mean(y[left_idxs], axis=0) if left_idxs.any() else (
            np.zeros_like(node.value)
        )

        node.right_temp = ObliqueFIGSNode()
        node.right_temp.idxs = right_idxs
        node.right_temp.value = np.mean(y[right_idxs], axis=0) if right_idxs.any() else (
            np.zeros_like(node.value)
        )

        return node

    def fit(self, X: np.ndarray, y_original: np.ndarray) -> "ObliqueFIGS":
        """FIGS greedy loop with oblique splits.

        Key design: potential_splits is a list of (leaf_node_ref, split_info) pairs.
        When we commit a split, we UPDATE the existing leaf node in the tree so the
        tree structure is properly connected.
        """
        n_samples = X.shape[0]

        # Handle classification by converting to one-hot residuals
        if self.task_type == "classification":
            classes = np.unique(y_original)
            n_classes = len(classes)
            y = np.zeros((n_samples, n_classes))
            for i, c in enumerate(classes):
                y[y_original == c, i] = 1.0
            self.classes_ = classes
        else:
            y = y_original.copy().astype(float)
            if y.ndim == 1:
                y = y.reshape(-1, 1)

        self.trees_ = []
        self.complexity_ = 0

        # Initialize predictions
        y_pred_total = np.zeros_like(y)
        y_predictions_per_tree = {}
        idxs_all = np.ones(n_samples, dtype=bool)

        # potential_splits: list of tuples (leaf_node, split_info_dict, is_new_root)
        # split_info_dict has: weights, threshold, impurity_reduction, left_idxs, right_idxs, left_val, right_val
        # leaf_node is the actual node in the tree (or None for new roots)

        def find_split_info(X_data, y_target, idxs):
            """Find best split and return info dict (not a node)."""
            result = self._find_best_oblique_split(X_data, y_target, idxs)
            if result is None:
                return None
            return {
                "weights": result.weights,
                "threshold": result.threshold,
                "impurity_reduction": result.impurity_reduction,
                "left_idxs": result.left_temp.idxs if result.left_temp else idxs,
                "right_idxs": result.right_temp.idxs if result.right_temp else idxs,
                "left_value": result.left_temp.value if result.left_temp else result.value,
                "right_value": result.right_temp.value if result.right_temp else result.value,
                "idxs": idxs,
            }

        # Find initial root split
        y_residuals = y - y_pred_total
        init_info = find_split_info(X, y_residuals, idxs_all)
        if init_info is None:
            return self

        # (leaf_node_or_None, split_info, is_new_root, tree_num, depth)
        potential_splits = [(None, init_info, True, -1, 0)]

        while potential_splits and self.complexity_ < self.max_rules:
            # Sort by impurity reduction, pick best
            potential_splits.sort(
                key=lambda t: t[1]["impurity_reduction"] if t[1] and t[1]["impurity_reduction"] else -np.inf
            )
            leaf_node, split_info, is_new_root, tree_num, depth = potential_splits.pop()

            if split_info is None or split_info["impurity_reduction"] <= self.min_impurity_decrease:
                break

            # Check depth constraint
            if self.max_depth is not None and depth >= self.max_depth:
                continue

            # Handle new root (new tree)
            if is_new_root:
                if self.max_trees is not None and len(self.trees_) >= self.max_trees:
                    continue

                # Create new root node
                root = ObliqueFIGSNode()
                root.is_root = True
                root.tree_num = len(self.trees_)
                root.depth = 0
                root.idxs = idxs_all.copy()
                root.value = np.mean(y_residuals, axis=0) if y_residuals is not None else np.zeros(y.shape[1:])
                tree_num = root.tree_num
                depth = 0
                self.trees_.append(root)
                y_predictions_per_tree[tree_num] = np.zeros_like(y)
                leaf_node = root

                # Propose a new root for the NEXT tree
                y_residuals = y - y_pred_total
                next_root_info = find_split_info(X, y_residuals, idxs_all)
                if next_root_info is not None:
                    potential_splits.append((None, next_root_info, True, -1, 0))

            # Apply the split TO the leaf_node (which is already in the tree)
            leaf_node.weights = split_info["weights"]
            leaf_node.threshold = split_info["threshold"]
            leaf_node.impurity_reduction = split_info["impurity_reduction"]

            # Create children as leaf nodes
            left_child = ObliqueFIGSNode()
            left_child.idxs = split_info["left_idxs"]
            left_child.value = split_info["left_value"]
            left_child.tree_num = tree_num
            left_child.depth = depth + 1

            right_child = ObliqueFIGSNode()
            right_child.idxs = split_info["right_idxs"]
            right_child.value = split_info["right_value"]
            right_child.tree_num = tree_num
            right_child.depth = depth + 1

            leaf_node.left = left_child
            leaf_node.right = right_child
            self.complexity_ += 1

            # Update predictions for this tree
            y_predictions_per_tree[tree_num] = self._predict_tree_train(
                self.trees_[tree_num], X, y.shape
            )
            y_pred_total = sum(y_predictions_per_tree.values())
            y_residuals = y - y_pred_total

            # Evaluate children as potential further splits
            for child in [left_child, right_child]:
                if child.idxs.sum() >= 5:
                    # For within-tree splits, target is y minus OTHER trees' predictions
                    if tree_num in y_predictions_per_tree:
                        y_target = y - (y_pred_total - y_predictions_per_tree[tree_num])
                    else:
                        y_target = y_residuals
                    child_info = find_split_info(X, y_target, child.idxs)
                    if child_info is not None and child_info["impurity_reduction"] > 0:
                        potential_splits.append(
                            (child, child_info, False, tree_num, child.depth)
                        )

            # NOTE: Skip re-evaluation of existing candidates for efficiency.
            # Stale impurity_reduction scores are still a valid greedy heuristic.
            # Only re-evaluate the new root candidate since residuals changed.
            new_potential = []
            has_root_candidate = False
            for (pn, pi, pnr, ptn, pd) in potential_splits:
                if pnr and not has_root_candidate:
                    # Re-evaluate root candidate on new residuals
                    updated = find_split_info(X, y_residuals, idxs_all)
                    if updated is not None and updated["impurity_reduction"] > 0:
                        new_potential.append((None, updated, True, -1, 0))
                        has_root_candidate = True
                elif not pnr:
                    # Keep existing child candidates with stale scores
                    new_potential.append((pn, pi, pnr, ptn, pd))
            potential_splits = new_potential

        return self

    def _predict_tree_train(
        self, root: ObliqueFIGSNode, X: np.ndarray, shape: tuple
    ) -> np.ndarray:
        """Predict for training data using sample masks (idxs)."""
        preds = np.full(shape, 0.0)
        self._fill_predictions(root, preds)
        return preds

    def _fill_predictions(self, node: ObliqueFIGSNode, preds: np.ndarray):
        """Recursively assign leaf values."""
        if node is None:
            return
        if node.left is None and node.right is None:
            if node.idxs is not None:
                preds[node.idxs] = node.value
            return
        if node.left is not None:
            self._fill_predictions(node.left, preds)
        if node.right is not None:
            self._fill_predictions(node.right, preds)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict on new data by routing through trees."""
        if not self.trees_:
            if self.task_type == "classification":
                return np.zeros(X.shape[0], dtype=int)
            return np.zeros(X.shape[0])

        sample_val = self.trees_[0].value
        if np.isscalar(sample_val):
            val_shape = ()
        else:
            val_shape = sample_val.shape

        y_pred = np.zeros((X.shape[0],) + val_shape)
        for tree_root in self.trees_:
            y_pred += self._predict_tree_test(tree_root, X, val_shape)

        if self.task_type == "classification" and self.classes_ is not None:
            return self.classes_[np.argmax(y_pred, axis=1)]
        else:
            return y_pred.flatten()

    def _predict_tree_test(
        self, node: ObliqueFIGSNode, X: np.ndarray, val_shape: tuple
    ) -> np.ndarray:
        """Route test samples through tree using oblique splits."""
        preds = np.zeros((X.shape[0],) + val_shape)
        self._route_and_predict(node, X, np.arange(X.shape[0]), preds)
        return preds

    def _route_and_predict(
        self, node: ObliqueFIGSNode, X: np.ndarray,
        sample_indices: np.ndarray, preds: np.ndarray,
    ):
        """Recursively route samples through the tree."""
        if node is None:
            return
        if node.left is None and node.right is None:
            preds[sample_indices] = node.value
            return
        if node.weights is None:
            preds[sample_indices] = node.value
            return

        z = X[sample_indices] @ node.weights
        left_mask = z <= node.threshold
        left_indices = sample_indices[left_mask]
        right_indices = sample_indices[~left_mask]
        if len(left_indices) > 0 and node.left is not None:
            self._route_and_predict(node.left, X, left_indices, preds)
        if len(right_indices) > 0 and node.right is not None:
            self._route_and_predict(node.right, X, right_indices, preds)

    def extract_split_directions(self) -> np.ndarray:
        """Extract all split direction vectors from the ensemble."""
        directions = []
        for tree_root in self.trees_:
            self._collect_directions(tree_root, directions)
        if directions:
            return np.array(directions)
        return np.zeros((0, 1))

    def _collect_directions(self, node: ObliqueFIGSNode, directions: list):
        """Recursively collect oblique split directions."""
        if node is None or (node.left is None and node.right is None):
            return
        if hasattr(node, 'weights') and node.weights is not None:
            directions.append(node.weights.copy())
        self._collect_directions(node.left, directions)
        self._collect_directions(node.right, directions)


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 4: FIGS axis-aligned direction extraction
# ═══════════════════════════════════════════════════════════════════════════════

def extract_figs_directions(model, n_features: int) -> np.ndarray:
    """Extract axis-aligned directions from an imodels FIGS model."""
    directions = []

    def traverse(node):
        if node is None:
            return
        if node.left is not None:  # internal node
            direction = np.zeros(n_features)
            if hasattr(node, 'feature') and node.feature is not None:
                feat_idx = int(node.feature)
                if 0 <= feat_idx < n_features:
                    direction[feat_idx] = 1.0
                    directions.append(direction)
            traverse(node.left)
        if node.right is not None:
            traverse(node.right)

    if hasattr(model, 'trees_'):
        for tree_root in model.trees_:
            traverse(tree_root)

    if directions:
        return np.array(directions)
    return np.zeros((0, n_features))


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 5: Data Loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_datasets(data_path: Path) -> dict:
    """Load and parse datasets from the JSON file.

    Returns:
        Dict mapping dataset name to {X, y, task_type, fold_labels, n_features, n_classes}.
    """
    logger.info(f"Loading data from {data_path}")
    raw = json.loads(data_path.read_text())
    datasets = {}

    for ds in raw["datasets"]:
        name = ds["dataset"]
        if name.endswith("_mini"):
            continue

        examples = ds["examples"]
        X = np.array([json.loads(ex["input"]) for ex in examples])
        y_str = [ex["output"] for ex in examples]
        task_type = examples[0]["metadata_task_type"]
        fold_labels = np.array([ex["metadata_fold"] for ex in examples])
        n_features = X.shape[1]

        if task_type == "classification":
            y = np.array([int(float(s)) for s in y_str])
            n_classes = int(examples[0].get("metadata_n_classes", len(np.unique(y))))
        else:
            y = np.array([float(s) for s in y_str])
            n_classes = 0

        datasets[name] = {
            "X": X, "y": y, "task_type": task_type,
            "fold_labels": fold_labels, "n_features": n_features,
            "n_classes": n_classes,
        }
        logger.info(
            f"  {name}: X={X.shape}, task={task_type}, "
            f"folds={sorted(np.unique(fold_labels).tolist())}"
        )

    logger.info(f"Loaded {len(datasets)} datasets")
    return datasets


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 6: Evaluation Loop
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_sporf(
    X: np.ndarray, y: np.ndarray, fold_labels: np.ndarray,
    task_type: str, n_features: int, n_classes: int,
    n_estimators: int = 100, max_depth: int | None = None,
    feature_combinations: float = 1.5,
) -> list[dict]:
    """Evaluate ManualSPORF with 5-fold CV."""
    folds = sorted(np.unique(fold_labels).tolist())
    fold_results = []

    for fold in folds:
        train_mask = fold_labels != fold
        test_mask = fold_labels == fold

        if test_mask.sum() < 2 or train_mask.sum() < 5:
            continue

        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]

        scaler = StandardScaler().fit(X_train)
        X_train_s = scaler.transform(X_train)
        X_test_s = scaler.transform(X_test)

        model = ManualSPORF(
            n_estimators=n_estimators, max_depth=max_depth,
            feature_combinations=feature_combinations,
            random_state=42 + fold, task_type=task_type,
        )
        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_test_s)

        result = {"fold": int(fold)}

        if task_type == "classification":
            result["accuracy"] = float(accuracy_score(y_test, y_pred))
            if n_classes == 2:
                try:
                    y_proba = model.predict_proba(X_test_s)
                    if y_proba.shape[1] >= 2:
                        result["auroc"] = float(roc_auc_score(y_test, y_proba[:, 1]))
                    else:
                        result["auroc"] = None
                except Exception:
                    result["auroc"] = None
            else:
                result["auroc"] = None
        else:
            result["r2"] = float(r2_score(y_test, y_pred))
            result["rmse"] = float(np.sqrt(mean_squared_error(y_test, y_pred)))

        W = model.extract_split_directions(n_features)
        result["direction_diversity"] = compute_direction_diversity(W)
        result["n_total_splits"] = int(W.shape[0])

        fold_results.append(result)

    return fold_results


def evaluate_oblique_figs(
    X: np.ndarray, y: np.ndarray, fold_labels: np.ndarray,
    task_type: str, n_features: int, n_classes: int,
    max_rules: int = 12, max_trees: int = 5, max_depth: int = 4,
) -> list[dict]:
    """Evaluate Oblique FIGS with 5-fold CV."""
    folds = sorted(np.unique(fold_labels).tolist())
    fold_results = []

    for fold in folds:
        train_mask = fold_labels != fold
        test_mask = fold_labels == fold

        if test_mask.sum() < 2 or train_mask.sum() < 5:
            continue

        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]

        scaler = StandardScaler().fit(X_train)
        X_train_s = scaler.transform(X_train)
        X_test_s = scaler.transform(X_test)

        model = ObliqueFIGS(
            max_rules=max_rules, max_trees=max_trees, max_depth=max_depth,
            task_type=task_type,
        )
        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_test_s)

        result = {"fold": int(fold)}

        if task_type == "classification":
            result["accuracy"] = float(accuracy_score(y_test, y_pred))
            result["auroc"] = None  # Oblique FIGS doesn't produce calibrated probabilities
        else:
            result["r2"] = float(r2_score(y_test, y_pred))
            result["rmse"] = float(np.sqrt(mean_squared_error(y_test, y_pred)))

        W = model.extract_split_directions()
        if W.shape[0] > 0 and W.shape[1] != n_features:
            # Pad or trim to match feature count
            W_padded = np.zeros((W.shape[0], n_features))
            min_d = min(W.shape[1], n_features)
            W_padded[:, :min_d] = W[:, :min_d]
            W = W_padded

        result["direction_diversity"] = compute_direction_diversity(W)
        result["n_total_splits"] = int(W.shape[0])

        fold_results.append(result)

    return fold_results


def evaluate_figs_axis_aligned(
    X: np.ndarray, y: np.ndarray, fold_labels: np.ndarray,
    task_type: str, n_features: int, n_classes: int,
    max_rules: int = 12, max_trees: int = 5, max_depth: int = 4,
) -> list[dict]:
    """Evaluate axis-aligned FIGS with 5-fold CV."""
    folds = sorted(np.unique(fold_labels).tolist())
    fold_results = []

    for fold in folds:
        train_mask = fold_labels != fold
        test_mask = fold_labels == fold

        if test_mask.sum() < 2 or train_mask.sum() < 5:
            continue

        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]

        try:
            if task_type == "classification":
                model = FIGSClassifier(max_rules=max_rules, max_trees=max_trees)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            else:
                model = FIGSRegressor(max_rules=max_rules, max_trees=max_trees)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
        except Exception as exc:
            logger.exception(f"FIGS failed on fold {fold}: {exc}")
            continue

        result = {"fold": int(fold)}

        if task_type == "classification":
            result["accuracy"] = float(accuracy_score(y_test, y_pred))
            if n_classes == 2:
                try:
                    y_proba = model.predict_proba(X_test)
                    if y_proba.ndim == 2 and y_proba.shape[1] >= 2:
                        result["auroc"] = float(roc_auc_score(y_test, y_proba[:, 1]))
                    else:
                        result["auroc"] = None
                except Exception:
                    result["auroc"] = None
            else:
                result["auroc"] = None
        else:
            result["r2"] = float(r2_score(y_test, y_pred))
            result["rmse"] = float(np.sqrt(mean_squared_error(y_test, y_pred)))

        W = extract_figs_directions(model, n_features)
        result["direction_diversity"] = compute_direction_diversity(W)
        result["n_total_splits"] = int(W.shape[0])

        fold_results.append(result)

    return fold_results


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 7: Aggregation helpers
# ═══════════════════════════════════════════════════════════════════════════════

def aggregate_fold_results(fold_results: list[dict], task_type: str) -> dict:
    """Aggregate per-fold results into summary statistics."""
    metric = "accuracy" if task_type == "classification" else "r2"
    vals = [fr[metric] for fr in fold_results if metric in fr]
    eranks = [fr["direction_diversity"]["erank"] for fr in fold_results]
    stable_ranks = [fr["direction_diversity"]["stable_rank"] for fr in fold_results]
    n_splits = [fr["n_total_splits"] for fr in fold_results]

    summary = {}
    if vals:
        summary[f"mean_{metric}"] = float(np.mean(vals))
        summary[f"std_{metric}"] = float(np.std(vals))
    if eranks:
        summary["mean_erank"] = float(np.mean(eranks))
    if stable_ranks:
        summary["mean_stable_rank"] = float(np.mean(stable_ranks))
    if n_splits:
        summary["mean_n_total_splits"] = float(np.mean(n_splits))

    return summary


def build_output_schema(
    all_results: dict, datasets_info: dict, method_out_extra: dict | None = None,
) -> dict:
    """Build the method_out.json in exp_gen_sol_out.json schema format.

    The schema requires: {"datasets": [{"dataset": str, "examples": [{"input": str, "output": str, ...}]}]}
    Each example represents one fold with predictions from ALL methods.
    """
    output_datasets = []

    for ds_name, ds_info in datasets_info.items():
        metric_key = "accuracy" if ds_info["task_type"] == "classification" else "r2"

        # Collect fold indices from all methods
        all_folds = set()
        for method_name, method_results in all_results.items():
            if ds_name in method_results:
                for fr in method_results[ds_name].get("fold_results", []):
                    all_folds.add(fr["fold"])

        examples = []
        for fold in sorted(all_folds):
            example = {
                "input": json.dumps({
                    "dataset": ds_name,
                    "fold": fold,
                    "task_type": ds_info["task_type"],
                    "n_features": ds_info["n_features"],
                }),
                "output": "",  # Will be set to best method's result
                "metadata_fold": fold,
                "metadata_task_type": ds_info["task_type"],
                "metadata_n_features": ds_info["n_features"],
                "metadata_dataset": ds_name,
            }

            best_val = -np.inf
            for method_name, method_results in all_results.items():
                if ds_name not in method_results:
                    continue
                ds_result = method_results[ds_name]
                fold_data = None
                for fr in ds_result.get("fold_results", []):
                    if fr["fold"] == fold:
                        fold_data = fr
                        break

                if fold_data is None:
                    continue

                metric_val = fold_data.get(metric_key, 0.0)
                example[f"predict_{method_name}"] = str(metric_val)

                # Track best method
                if metric_val > best_val:
                    best_val = metric_val
                    example["output"] = str(metric_val)

                # Add direction diversity metrics per method
                if "direction_diversity" in fold_data:
                    dd = fold_data["direction_diversity"]
                    example[f"metadata_erank_{method_name}"] = dd.get("erank", 0)
                    example[f"metadata_stable_rank_{method_name}"] = dd.get("stable_rank", 0)
                    example[f"metadata_n_splits_{method_name}"] = dd.get("n_total_splits", 0)

                if "auroc" in fold_data and fold_data["auroc"] is not None:
                    example[f"metadata_auroc_{method_name}"] = fold_data["auroc"]
                if "rmse" in fold_data:
                    example[f"metadata_rmse_{method_name}"] = fold_data["rmse"]

            examples.append(example)

        if examples:
            output_datasets.append({
                "dataset": ds_name,
                "examples": examples,
            })

    output = {
        "metadata": {
            "experiment": "unconstrained_oblique_baselines",
            "description": (
                "Accuracy and direction diversity for unconstrained oblique tree baselines. "
                "Methods: ManualSPORF (matched), ManualSPORF (full), ObliqueFIGS, FIGS axis-aligned. "
                "Each example is one fold with predict_* fields for each method's metric score."
            ),
            "methods": list(all_results.keys()),
            "metric_classification": "accuracy",
            "metric_regression": "r2",
            "n_datasets": len(datasets_info),
            "datasets": list(datasets_info.keys()),
        },
        "datasets": output_datasets,
    }

    return output


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

@logger.catch
def main():
    start_time = time.time()
    logger.info("=" * 70)
    logger.info("Unconstrained Oblique Baselines Experiment")
    logger.info("=" * 70)

    # Load full data
    datasets = load_datasets(FULL_DATA_PATH)

    all_results = {}
    datasets_info = {}
    for ds_name, ds in datasets.items():
        datasets_info[ds_name] = {
            "task_type": ds["task_type"],
            "n_features": ds["n_features"],
            "n_classes": ds["n_classes"],
        }

    # ── Method 1: SPORF Matched (complexity-matched to FIGS) ──────────────
    logger.info("\n" + "=" * 70)
    logger.info("METHOD 1: ManualSPORF Matched (n_estimators=6, max_depth=2)")
    logger.info("=" * 70)
    sporf_matched_results = {}
    for ds_name, ds in datasets.items():
        t0 = time.time()
        logger.info(f"  Running SPORF_matched on {ds_name}...")
        fold_results = evaluate_sporf(
            ds["X"], ds["y"], ds["fold_labels"],
            ds["task_type"], ds["n_features"], ds["n_classes"],
            n_estimators=6, max_depth=2, feature_combinations=1.5,
        )
        summary = aggregate_fold_results(fold_results, ds["task_type"])
        sporf_matched_results[ds_name] = {
            "fold_results": fold_results,
            **summary,
        }
        elapsed = time.time() - t0
        metric = "accuracy" if ds["task_type"] == "classification" else "r2"
        logger.info(
            f"    {ds_name}: mean_{metric}={summary.get(f'mean_{metric}', 'N/A'):.4f}, "
            f"mean_erank={summary.get('mean_erank', 'N/A'):.2f}, "
            f"time={elapsed:.1f}s"
        )
    all_results["sporf_matched"] = sporf_matched_results

    # ── Method 2: SPORF Full ──────────────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("METHOD 2: ManualSPORF Full (n_estimators=50, max_depth=10)")
    logger.info("=" * 70)
    sporf_full_results = {}
    for ds_name, ds in datasets.items():
        t0 = time.time()
        logger.info(f"  Running SPORF_full on {ds_name}...")
        # Use reduced settings for time: 50 estimators, max_depth=10
        fold_results = evaluate_sporf(
            ds["X"], ds["y"], ds["fold_labels"],
            ds["task_type"], ds["n_features"], ds["n_classes"],
            n_estimators=50, max_depth=10, feature_combinations=1.5,
        )
        summary = aggregate_fold_results(fold_results, ds["task_type"])
        sporf_full_results[ds_name] = {
            "fold_results": fold_results,
            **summary,
        }
        elapsed = time.time() - t0
        metric = "accuracy" if ds["task_type"] == "classification" else "r2"
        logger.info(
            f"    {ds_name}: mean_{metric}={summary.get(f'mean_{metric}', 'N/A'):.4f}, "
            f"mean_erank={summary.get('mean_erank', 'N/A'):.2f}, "
            f"time={elapsed:.1f}s"
        )

        # Time budget check
        elapsed_total = time.time() - start_time
        if elapsed_total > 2700:  # 45 min
            logger.warning(f"Time budget concern: {elapsed_total:.0f}s elapsed. Reducing SPORF_full estimators.")
            break
    all_results["sporf_full"] = sporf_full_results

    # ── Method 3: Oblique FIGS ────────────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("METHOD 3: Oblique FIGS (max_rules=12, max_trees=5, max_depth=4)")
    logger.info("=" * 70)
    oblique_figs_results = {}
    for ds_name, ds in datasets.items():
        t0 = time.time()
        logger.info(f"  Running ObliqueFIGS on {ds_name}...")
        fold_results = evaluate_oblique_figs(
            ds["X"], ds["y"], ds["fold_labels"],
            ds["task_type"], ds["n_features"], ds["n_classes"],
            max_rules=12, max_trees=5, max_depth=4,
        )
        summary = aggregate_fold_results(fold_results, ds["task_type"])
        oblique_figs_results[ds_name] = {
            "fold_results": fold_results,
            **summary,
        }
        elapsed = time.time() - t0
        metric = "accuracy" if ds["task_type"] == "classification" else "r2"
        logger.info(
            f"    {ds_name}: mean_{metric}={summary.get(f'mean_{metric}', 'N/A'):.4f}, "
            f"mean_erank={summary.get('mean_erank', 'N/A'):.2f}, "
            f"time={elapsed:.1f}s"
        )
    all_results["oblique_figs"] = oblique_figs_results

    # ── Method 4: FIGS Axis-Aligned ──────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("METHOD 4: FIGS Axis-Aligned (max_rules=12, max_trees=5)")
    logger.info("=" * 70)
    figs_aa_results = {}
    for ds_name, ds in datasets.items():
        t0 = time.time()
        logger.info(f"  Running FIGS axis-aligned on {ds_name}...")
        fold_results = evaluate_figs_axis_aligned(
            ds["X"], ds["y"], ds["fold_labels"],
            ds["task_type"], ds["n_features"], ds["n_classes"],
            max_rules=12, max_trees=5, max_depth=4,
        )
        summary = aggregate_fold_results(fold_results, ds["task_type"])
        figs_aa_results[ds_name] = {
            "fold_results": fold_results,
            **summary,
        }
        elapsed = time.time() - t0
        metric = "accuracy" if ds["task_type"] == "classification" else "r2"
        logger.info(
            f"    {ds_name}: mean_{metric}={summary.get(f'mean_{metric}', 'N/A'):.4f}, "
            f"mean_erank={summary.get('mean_erank', 'N/A'):.2f}, "
            f"time={elapsed:.1f}s"
        )
    all_results["figs_axis_aligned"] = figs_aa_results

    # ── Build output ──────────────────────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("Building output JSON")
    logger.info("=" * 70)

    output = build_output_schema(all_results, datasets_info)

    # Write method_out.json
    output_path = WORKSPACE / "method_out.json"
    output_path.write_text(json.dumps(output, indent=2))
    logger.info(f"Wrote {output_path} ({output_path.stat().st_size / 1024:.1f} KB)")

    # Also write the detailed results as method_out_extra.json
    detailed_output = {
        "experiment": "unconstrained_oblique_baselines",
        "description": "Detailed results with direction diversity metrics",
        "methods": {},
    }
    for method_name, method_results in all_results.items():
        method_entry = {
            "results_by_dataset": {},
            "summary": {},
        }
        all_metrics = []
        all_eranks = []
        for ds_name, ds_result in method_results.items():
            method_entry["results_by_dataset"][ds_name] = {
                "task_type": datasets_info[ds_name]["task_type"],
                "n_features": datasets_info[ds_name]["n_features"],
                **{k: v for k, v in ds_result.items()},
            }
            metric = "accuracy" if datasets_info[ds_name]["task_type"] == "classification" else "r2"
            if f"mean_{metric}" in ds_result:
                all_metrics.append(ds_result[f"mean_{metric}"])
            if "mean_erank" in ds_result:
                all_eranks.append(ds_result["mean_erank"])

        if all_metrics:
            method_entry["summary"]["mean_metric_across_datasets"] = float(np.mean(all_metrics))
        if all_eranks:
            method_entry["summary"]["mean_erank_across_datasets"] = float(np.mean(all_eranks))

        detailed_output["methods"][method_name] = method_entry

    # Cross-method comparison table
    comparison_table = []
    for ds_name, ds_info in datasets_info.items():
        metric = "accuracy" if ds_info["task_type"] == "classification" else "r2"
        row = {
            "dataset": ds_name,
            "n_features": ds_info["n_features"],
            "task_type": ds_info["task_type"],
        }
        for method_name in all_results:
            if ds_name in all_results[method_name]:
                ds_r = all_results[method_name][ds_name]
                row[f"{method_name}_{metric}"] = ds_r.get(f"mean_{metric}", None)
                row[f"{method_name}_erank"] = ds_r.get("mean_erank", None)
        comparison_table.append(row)

    detailed_output["cross_method_comparison"] = {
        "per_dataset_table": comparison_table,
    }

    extra_path = WORKSPACE / "method_out_extra.json"
    extra_path.write_text(json.dumps(detailed_output, indent=2))
    logger.info(f"Wrote {extra_path} ({extra_path.stat().st_size / 1024:.1f} KB)")

    total_time = time.time() - start_time
    logger.info(f"\nTotal runtime: {total_time:.1f}s ({total_time/60:.1f} min)")

    # Print summary table
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY TABLE")
    logger.info("=" * 70)
    header = f"{'Dataset':<25} {'Task':<6}"
    for method in all_results:
        header += f" {method[:12]:>12} {'eRank':>6}"
    logger.info(header)
    logger.info("-" * len(header))

    for ds_name in datasets_info:
        metric = "accuracy" if datasets_info[ds_name]["task_type"] == "classification" else "r2"
        row = f"{ds_name:<25} {datasets_info[ds_name]['task_type'][:6]:<6}"
        for method_name in all_results:
            if ds_name in all_results[method_name]:
                ds_r = all_results[method_name][ds_name]
                val = ds_r.get(f"mean_{metric}", 0)
                erank = ds_r.get("mean_erank", 0)
                row += f" {val:>12.4f} {erank:>6.2f}"
            else:
                row += f" {'N/A':>12} {'N/A':>6}"
        logger.info(row)


if __name__ == "__main__":
    main()
