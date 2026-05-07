#!/usr/bin/env python3
"""Codebook-FIGS: Benchmarking codebook-constrained oblique FIGS against baselines.

Implements Codebook-FIGS from scratch with PCA-initialized shared codebook,
vectorized sorted-scan split search, and K-SVD-inspired alternating optimization.
Benchmarks against FIGS (imodels), XGBoost, and LightGBM across 10 tabular datasets
with K in {3,5,8,12,20} and 5-fold CV.
"""

import json
import os
import sys
import time
import resource
import warnings
from pathlib import Path
from typing import Any, Optional

import numpy as np
from loguru import logger
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)

warnings.filterwarnings("ignore")

# ── Logging ──────────────────────────────────────────────────────────────────
logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
WORKSPACE = Path(__file__).resolve().parent
LOG_DIR = WORKSPACE / "logs"
LOG_DIR.mkdir(exist_ok=True)
logger.add(str(LOG_DIR / "run.log"), rotation="30 MB", level="DEBUG")

# ── Resource limits ──────────────────────────────────────────────────────────
try:
    resource.setrlimit(resource.RLIMIT_AS, (14 * 1024**3, 14 * 1024**3))  # 14GB RAM
    resource.setrlimit(resource.RLIMIT_CPU, (3500, 3500))  # ~58 min CPU
except Exception:
    pass

# ── Config ───────────────────────────────────────────────────────────────────
K_VALUES = [3, 5, 8, 12, 20]
N_FOLDS = 5
MAX_RULES = 12
MAX_DEPTH = 4
N_ALTERNATION_ROUNDS = 3
TIMEOUT_PER_COMBO_SEC = 300  # 5 min per (dataset, method, K, fold)


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 1: DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_datasets(json_path: str) -> dict[str, dict[str, Any]]:
    """Load datasets from full_data_out.json or mini_data_out.json.

    Returns dict: dataset_name -> {X, y, folds, task_type, n_classes, ...}
    """
    logger.info(f"Loading datasets from {json_path}")
    raw = json.loads(Path(json_path).read_text())
    datasets: dict[str, dict[str, Any]] = {}

    for group in raw["datasets"]:
        name = group["dataset"]
        examples = group["examples"]
        if len(examples) == 0:
            logger.warning(f"Skipping empty dataset: {name}")
            continue

        meta = examples[0]
        task_type = meta["metadata_task_type"]
        n_classes = meta.get("metadata_n_classes", 0)
        feature_names = meta.get("metadata_feature_names", [])
        n_features_meta = meta.get("metadata_n_features")
        domain = meta.get("metadata_domain", "")

        X_list, y_list, fold_list = [], [], []
        for ex in examples:
            X_list.append(json.loads(ex["input"]))
            y_list.append(float(ex["output"]))
            fold_list.append(ex["metadata_fold"])

        X = np.array(X_list, dtype=np.float64)
        y = np.array(y_list, dtype=np.float64)
        folds = np.array(fold_list, dtype=int)

        datasets[name] = {
            "X": X,
            "y": y,
            "folds": folds,
            "task_type": task_type,
            "n_classes": int(n_classes),
            "feature_names": feature_names,
            "domain": domain,
            "n_features": X.shape[1],
            "n_samples": X.shape[0],
        }
        logger.debug(
            f"  {name}: n={X.shape[0]}, d={X.shape[1]}, task={task_type}, "
            f"classes={n_classes}, folds={len(np.unique(folds))}"
        )

    logger.info(f"Loaded {len(datasets)} datasets")
    return datasets


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 2: CODEBOOK-FIGS IMPLEMENTATION
# ═══════════════════════════════════════════════════════════════════════════════


class CBNode:
    """A node in a Codebook-FIGS tree, supporting oblique splits."""

    __slots__ = [
        "codebook_idx",
        "threshold",
        "value",
        "idxs",
        "impurity",
        "impurity_reduction",
        "left",
        "right",
        "is_root",
        "tree_num",
        "depth",
    ]

    def __init__(self) -> None:
        self.codebook_idx: Optional[int] = None
        self.threshold: Optional[float] = None
        self.value: float = 0.0
        self.idxs: Optional[np.ndarray] = None
        self.impurity: float = 0.0
        self.impurity_reduction: float = 0.0
        self.left: Optional["CBNode"] = None
        self.right: Optional["CBNode"] = None
        self.is_root: bool = False
        self.tree_num: int = -1
        self.depth: int = 0


def _make_leaf(
    y_subset: np.ndarray,
    idxs: np.ndarray,
    tree_num: int,
    depth: int,
) -> CBNode:
    """Create a leaf node."""
    leaf = CBNode()
    leaf.codebook_idx = None
    leaf.threshold = None
    leaf.value = float(np.mean(y_subset)) if len(y_subset) > 0 else 0.0
    leaf.idxs = idxs
    leaf.impurity = float(np.var(y_subset)) if len(y_subset) > 0 else 0.0
    leaf.impurity_reduction = 0.0
    leaf.tree_num = tree_num
    leaf.depth = depth
    leaf.left = None
    leaf.right = None
    leaf.is_root = False
    return leaf


class CodebookFIGS:
    """Codebook-constrained Fast Interpretable Greedy-Tree Sums.

    All oblique splits select from a shared codebook of K direction vectors.
    Uses simplified sequential tree boosting (grow trees one at a time on residuals).
    """

    def __init__(
        self,
        K: int = 10,
        max_rules: int = 12,
        max_trees: Optional[int] = None,
        max_depth: Optional[int] = None,
        min_impurity_decrease: float = 0.0,
        n_alternation_rounds: int = 3,
        random_state: int = 42,
    ) -> None:
        self.K = K
        self.max_rules = max_rules
        self.max_trees = max_trees if max_trees is not None else 10
        self.max_depth = max_depth if max_depth is not None else 4
        self.min_impurity_decrease = min_impurity_decrease
        self.n_alternation_rounds = n_alternation_rounds
        self.random_state = random_state

        self.codebook_: Optional[np.ndarray] = None
        self.trees_: list[CBNode] = []
        self.complexity_: int = 0
        self.is_classifier_: Optional[bool] = None
        self.n_classes_: Optional[int] = None
        self.codebook_usage_: dict[int, int] = {}

    # ── Codebook initialization ──────────────────────────────────────────

    def _init_codebook(self, X: np.ndarray, K: int) -> np.ndarray:
        """Initialize codebook via PCA on raw training data."""
        n_samples, d = X.shape
        K_actual = min(K, d, n_samples)

        pca = PCA(n_components=K_actual, random_state=self.random_state)
        pca.fit(X)
        codebook = pca.components_.copy()  # (K_actual, d)

        if K_actual < K:
            rng = np.random.RandomState(self.random_state)
            extra = rng.randn(K - K_actual, d)
            norms_extra = np.linalg.norm(extra, axis=1, keepdims=True)
            norms_extra[norms_extra == 0] = 1.0
            extra = extra / norms_extra
            codebook = np.vstack([codebook, extra])

        # Normalize all to unit norm
        norms = np.linalg.norm(codebook, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        codebook = codebook / norms

        self.codebook_ = codebook
        return codebook

    # ── Vectorized split search ──────────────────────────────────────────

    def _find_best_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        idxs: np.ndarray,
    ) -> tuple[Optional[int], Optional[float], float]:
        """Find best (codebook_entry, threshold) split for samples at idxs.

        Uses fully vectorized sorted-scan for speed.
        Returns (best_k, best_threshold, best_reduction).
        """
        X_node = X[idxs]
        y_node = y[idxs]
        n_node = X_node.shape[0]

        if n_node < 4:
            return None, None, -np.inf

        parent_mean = np.mean(y_node)
        parent_impurity = np.mean((y_node - parent_mean) ** 2)

        if parent_impurity < 1e-15:
            return None, None, -np.inf

        best_reduction = -np.inf
        best_k: Optional[int] = None
        best_threshold: Optional[float] = None

        K = self.codebook_.shape[0]

        for k in range(K):
            c_k = self.codebook_[k]
            projections = X_node @ c_k

            sort_idx = np.argsort(projections)
            sorted_proj = projections[sort_idx]
            sorted_y = y_node[sort_idx]

            # Vectorized threshold scan
            cum_sum = np.cumsum(sorted_y)
            cum_sq_sum = np.cumsum(sorted_y ** 2)
            total_sum = cum_sum[-1]
            total_sq_sum = cum_sq_sum[-1]

            n_left = np.arange(1, n_node, dtype=np.float64)
            n_right = float(n_node) - n_left

            left_mean = cum_sum[:-1] / n_left
            left_mse = cum_sq_sum[:-1] / n_left - left_mean ** 2
            # Clamp negative MSE from floating point
            left_mse = np.maximum(left_mse, 0.0)

            right_sum = total_sum - cum_sum[:-1]
            right_sq_sum = total_sq_sum - cum_sq_sum[:-1]
            right_mean = right_sum / n_right
            right_mse = right_sq_sum / n_right - right_mean ** 2
            right_mse = np.maximum(right_mse, 0.0)

            weighted_impurity = (left_mse * n_left + right_mse * n_right) / n_node
            reductions = (parent_impurity - weighted_impurity) * n_node

            # Mask invalid splits (consecutive equal projection values)
            valid = sorted_proj[1:] != sorted_proj[:-1]
            # Also require min 2 samples per side
            valid &= n_left >= 2
            valid &= n_right >= 2
            reductions[~valid] = -np.inf

            if np.all(~valid):
                continue

            best_i = np.argmax(reductions)
            if reductions[best_i] > best_reduction:
                best_reduction = float(reductions[best_i])
                best_k = k
                best_threshold = float(
                    (sorted_proj[best_i] + sorted_proj[best_i + 1]) / 2.0
                )

        return best_k, best_threshold, best_reduction

    # ── Recursive tree building ──────────────────────────────────────────

    def _build_node(
        self,
        X: np.ndarray,
        y: np.ndarray,
        idxs: np.ndarray,
        depth: int,
        max_depth: int,
        splits_remaining: Optional[list] = None,
    ) -> CBNode:
        """Recursively build tree by finding best codebook split.

        splits_remaining: mutable list [int] used as a counter to enforce max_rules.
        """
        n_node = int(np.sum(idxs))
        y_node = y[idxs]

        if n_node < 4 or depth >= max_depth:
            return _make_leaf(y_node, idxs, -1, depth)

        # Check global split budget
        if splits_remaining is not None and splits_remaining[0] <= 0:
            return _make_leaf(y_node, idxs, -1, depth)

        best_k, best_threshold, best_reduction = self._find_best_split(X, y, idxs)

        if best_k is None or best_reduction <= self.min_impurity_decrease:
            return _make_leaf(y_node, idxs, -1, depth)

        # Create split
        projections = X @ self.codebook_[best_k]
        split_mask = projections <= best_threshold
        left_idxs = idxs & split_mask
        right_idxs = idxs & ~split_mask

        n_left = int(np.sum(left_idxs))
        n_right = int(np.sum(right_idxs))

        if n_left == 0 or n_right == 0:
            return _make_leaf(y_node, idxs, -1, depth)

        # Consume one split from budget
        if splits_remaining is not None:
            splits_remaining[0] -= 1

        node = CBNode()
        node.codebook_idx = best_k
        node.threshold = best_threshold
        node.value = float(np.mean(y_node))
        node.idxs = idxs
        node.impurity_reduction = best_reduction
        node.depth = depth
        node.is_root = False

        node.left = self._build_node(X, y, left_idxs, depth + 1, max_depth, splits_remaining)
        node.right = self._build_node(X, y, right_idxs, depth + 1, max_depth, splits_remaining)

        return node

    # ── Tree utilities ───────────────────────────────────────────────────

    @staticmethod
    def _count_splits(node: CBNode) -> int:
        """Count number of internal (split) nodes."""
        if node is None or (node.left is None and node.right is None):
            return 0
        return 1 + CodebookFIGS._count_splits(node.left) + CodebookFIGS._count_splits(node.right)

    @staticmethod
    def _get_internal_nodes(node: CBNode) -> list[CBNode]:
        """Get all internal (non-leaf) nodes."""
        if node is None or (node.left is None and node.right is None):
            return []
        result = [node]
        result.extend(CodebookFIGS._get_internal_nodes(node.left))
        result.extend(CodebookFIGS._get_internal_nodes(node.right))
        return result

    @staticmethod
    def _get_leaves(node: CBNode) -> list[CBNode]:
        """Get all leaf nodes."""
        if node is None:
            return []
        if node.left is None and node.right is None:
            return [node]
        result = []
        result.extend(CodebookFIGS._get_leaves(node.left))
        result.extend(CodebookFIGS._get_leaves(node.right))
        return result

    # ── Simplified FIGS fit ──────────────────────────────────────────────

    def _fit_single_round(self, X: np.ndarray, y: np.ndarray) -> None:
        """Simplified FIGS: grow trees sequentially on residuals."""
        self.trees_ = []
        self.complexity_ = 0
        self.codebook_usage_ = {}

        residuals = y.copy()
        # Use mutable list as counter so _build_node can enforce global limit
        splits_remaining = [self.max_rules]

        for tree_idx in range(self.max_trees):
            if splits_remaining[0] <= 0:
                break

            all_idxs = np.ones(X.shape[0], dtype=bool)
            root = self._build_node(
                X, residuals, all_idxs,
                depth=0, max_depth=self.max_depth,
                splits_remaining=splits_remaining,
            )

            if root is None or (root.left is None and root.right is None):
                break

            tree_splits = self._count_splits(root)
            if tree_splits == 0:
                break

            root.is_root = True
            root.tree_num = tree_idx
            self.trees_.append(root)
            self.complexity_ += tree_splits

            # Update residuals
            tree_pred = self._predict_tree(root, X)
            residuals = residuals - tree_pred

        # Update codebook usage counts
        for root in self.trees_:
            for node in self._get_internal_nodes(root):
                k = node.codebook_idx
                self.codebook_usage_[k] = self.codebook_usage_.get(k, 0) + 1

    # ── Codebook refinement ──────────────────────────────────────────────

    def _refine_codebook(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fix tree structures, optimize each codebook entry via PCA."""
        K = self.codebook_.shape[0]

        # Collect which nodes use which codebook entry
        node_groups: dict[int, list[tuple[CBNode, np.ndarray, np.ndarray]]] = {
            k: [] for k in range(K)
        }
        for root in self.trees_:
            for node in self._get_internal_nodes(root):
                if node.codebook_idx is not None and node.idxs is not None:
                    X_node = X[node.idxs]
                    y_node = y[node.idxs]
                    node_groups[node.codebook_idx].append((node, X_node, y_node))

        for k in range(K):
            nodes_k = node_groups[k]
            if len(nodes_k) == 0:
                # Unused entry: re-initialize with random direction
                rng = np.random.RandomState(self.random_state + k + 100)
                self.codebook_[k] = rng.randn(X.shape[1])
                norm = np.linalg.norm(self.codebook_[k])
                if norm > 0:
                    self.codebook_[k] /= norm
                continue

            # Pool data from all nodes using this entry
            X_pooled = np.vstack([X_n for _, X_n, _ in nodes_k])
            if X_pooled.shape[0] < 2:
                continue

            try:
                pca1 = PCA(n_components=1, random_state=self.random_state)
                pca1.fit(X_pooled)
                new_direction = pca1.components_[0].copy()
                norm = np.linalg.norm(new_direction)
                if norm > 1e-12:
                    new_direction /= norm
                    self.codebook_[k] = new_direction
            except Exception:
                pass  # keep old direction

        # Normalize all codebook entries
        norms = np.linalg.norm(self.codebook_, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.codebook_ = self.codebook_ / norms

    # ── Main fit ─────────────────────────────────────────────────────────

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        task_type: str = "classification",
    ) -> "CodebookFIGS":
        """Full fit with alternating optimization."""
        self.is_classifier_ = task_type == "classification"
        if self.is_classifier_:
            self.n_classes_ = len(np.unique(y))
        y_work = y.copy()

        # Initialize codebook
        K_actual = min(self.K, X.shape[1], X.shape[0])
        self._init_codebook(X, K_actual)

        # Alternating optimization
        for round_idx in range(self.n_alternation_rounds):
            self._fit_single_round(X, y_work)
            if round_idx < self.n_alternation_rounds - 1 and len(self.trees_) > 0:
                self._refine_codebook(X, y_work)

        return self

    # ── Prediction ───────────────────────────────────────────────────────

    def _predict_tree(self, node: CBNode, X: np.ndarray) -> np.ndarray:
        """Recursive tree prediction with oblique splits."""
        if node is None:
            return np.zeros(X.shape[0])
        if node.left is None and node.right is None:
            return np.full(X.shape[0], node.value)

        projections = X @ self.codebook_[node.codebook_idx]
        go_left = projections <= node.threshold

        result = np.zeros(X.shape[0])
        n_left = int(np.sum(go_left))
        n_right = X.shape[0] - n_left

        if n_left > 0:
            result[go_left] = self._predict_tree(node.left, X[go_left])
        if n_right > 0:
            result[~go_left] = self._predict_tree(node.right, X[~go_left])
        return result

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Sum predictions across all trees."""
        pred = np.zeros(X.shape[0])
        for root in self.trees_:
            pred += self._predict_tree(root, X)
        return pred

    def predict_classification(self, X: np.ndarray) -> np.ndarray:
        """For classification: return class predictions."""
        raw = self.predict(X)
        if self.n_classes_ == 2:
            return (raw >= 0.5).astype(int)
        else:
            return np.round(raw).astype(int).clip(0, self.n_classes_ - 1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """For binary classification: return probability of each class."""
        raw = self.predict(X)
        proba = np.clip(raw, 0.0, 1.0)
        return np.column_stack([1 - proba, proba])


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 3: METRICS
# ═══════════════════════════════════════════════════════════════════════════════


def compute_erank(codebook: np.ndarray, usage_counts: np.ndarray) -> float:
    """Compute effective rank (direction diversity) of used codebook directions."""
    used_mask = usage_counts > 0
    n_used = int(np.sum(used_mask))
    if n_used <= 1:
        return float(n_used)

    used_directions = codebook[used_mask]
    svd_vals = np.linalg.svd(used_directions, compute_uv=False)
    svd_vals = svd_vals[svd_vals > 1e-10]

    if len(svd_vals) == 0:
        return 0.0

    p = svd_vals / svd_vals.sum()
    H = -np.sum(p * np.log(p + 1e-15))
    return float(np.exp(H))


def evaluate_fold(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray],
    task_type: str,
) -> dict[str, Any]:
    """Compute metrics for one fold."""
    metrics: dict[str, Any] = {}
    if task_type == "classification":
        metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
        if y_proba is not None:
            try:
                proba_1 = np.clip(y_proba[:, 1], 1e-7, 1 - 1e-7)
                metrics["auroc"] = float(roc_auc_score(y_true, proba_1))
            except ValueError:
                metrics["auroc"] = None
        else:
            metrics["auroc"] = None
        metrics["rmse"] = None
        metrics["r2"] = None
    else:
        metrics["rmse"] = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        metrics["r2"] = float(r2_score(y_true, y_pred))
        metrics["accuracy"] = None
        metrics["auroc"] = None
    return metrics


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 4: BASELINES
# ═══════════════════════════════════════════════════════════════════════════════


def run_figs_baseline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    task_type: str,
) -> tuple[np.ndarray, Optional[np.ndarray], int]:
    """Run FIGS baseline from imodels."""
    from imodels import FIGSClassifier, FIGSRegressor

    if task_type == "classification":
        model = FIGSClassifier(max_rules=MAX_RULES)
    else:
        model = FIGSRegressor(max_rules=MAX_RULES)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = None
    if task_type == "classification":
        try:
            y_proba = model.predict_proba(X_test)
        except Exception:
            pass
    n_splits = getattr(model, "complexity_", 0)
    return y_pred, y_proba, n_splits


def run_xgboost_baseline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    task_type: str,
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """Run XGBoost baseline."""
    import xgboost as xgb

    if task_type == "classification":
        model = xgb.XGBClassifier(
            max_depth=4,
            n_estimators=100,
            learning_rate=0.1,
            random_state=42,
            eval_metric="logloss",
            verbosity=0,
            use_label_encoder=False,
        )
    else:
        model = xgb.XGBRegressor(
            max_depth=4,
            n_estimators=100,
            learning_rate=0.1,
            random_state=42,
            verbosity=0,
        )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = None
    if task_type == "classification":
        try:
            y_proba = model.predict_proba(X_test)
        except Exception:
            pass
    return y_pred, y_proba


def run_lgbm_baseline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    task_type: str,
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """Run LightGBM baseline."""
    import lightgbm as lgb

    if task_type == "classification":
        model = lgb.LGBMClassifier(
            max_depth=4,
            n_estimators=100,
            learning_rate=0.1,
            random_state=42,
            verbose=-1,
        )
    else:
        model = lgb.LGBMRegressor(
            max_depth=4,
            n_estimators=100,
            learning_rate=0.1,
            random_state=42,
            verbose=-1,
        )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = None
    if task_type == "classification":
        try:
            y_proba = model.predict_proba(X_test)
        except Exception:
            pass
    return y_pred, y_proba


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 5: MAIN EXPERIMENT LOOP
# ═══════════════════════════════════════════════════════════════════════════════


def compute_summary(all_results: list[dict]) -> list[dict]:
    """Aggregate per-fold results into mean±std per (dataset, method, K)."""
    from collections import defaultdict

    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in all_results:
        key = (r["dataset"], r["method"], r.get("K"))
        groups[key].append(r)

    summary_rows = []
    agg_cols = [
        "accuracy", "auroc", "rmse", "r2", "erank",
        "n_splits", "train_time_sec", "n_codebook_used",
    ]
    for (ds, method, K), rows in groups.items():
        row: dict[str, Any] = {"dataset": ds, "method": method, "K": K}
        for col in agg_cols:
            vals = [r.get(col) for r in rows if r.get(col) is not None]
            if vals:
                arr = np.array(vals, dtype=float)
                row[f"{col}_mean"] = float(np.nanmean(arr))
                row[f"{col}_std"] = float(np.nanstd(arr))
        summary_rows.append(row)
    return summary_rows


def format_output_for_schema(
    all_results: list[dict],
    summary: list[dict],
    full_datasets: dict,
    config: dict,
) -> dict:
    """Format output to match exp_gen_sol_out.json schema.

    Schema requires: {datasets: [{dataset: str, examples: [{input: str, output: str, ...}]}]}
    We map each (dataset) to a group with per-example predictions.
    """
    datasets_out = []

    # Group results by dataset
    from collections import defaultdict
    by_dataset: dict[str, list[dict]] = defaultdict(list)
    for r in all_results:
        by_dataset[r["dataset"]].append(r)

    for ds_name, ds_info in full_datasets.items():
        # Skip mini datasets
        if ds_name.endswith("_mini"):
            continue

        ds_results = by_dataset.get(ds_name, [])
        if not ds_results:
            continue

        # Create one example per fold with all method predictions as metadata
        examples = []
        for fold_idx in range(N_FOLDS):
            fold_results = [r for r in ds_results if r.get("fold") == fold_idx]
            if not fold_results:
                continue

            # Build example: input=fold description, output=best method result
            fold_mask = ds_info["folds"] == fold_idx
            n_test = int(np.sum(fold_mask))
            n_train = int(np.sum(~fold_mask))

            example: dict[str, Any] = {
                "input": json.dumps({
                    "dataset": ds_name,
                    "fold": fold_idx,
                    "n_train": n_train,
                    "n_test": n_test,
                    "task_type": ds_info["task_type"],
                }),
                "output": json.dumps({
                    "fold_results": fold_results,
                }),
                "metadata_fold": fold_idx,
                "metadata_dataset": ds_name,
                "metadata_task_type": ds_info["task_type"],
            }

            # Add per-method predictions
            for r in fold_results:
                method_key = r["method"]
                K_val = r.get("K")
                if K_val is not None:
                    pred_key = f"predict_{method_key}_K{K_val}"
                else:
                    pred_key = f"predict_{method_key}"

                # Serialize key metrics
                pred_val = {}
                for mk in ["accuracy", "auroc", "rmse", "r2", "erank", "n_splits", "train_time_sec"]:
                    if r.get(mk) is not None:
                        pred_val[mk] = r[mk]
                example[pred_key] = json.dumps(pred_val)

            examples.append(example)

        if examples:
            datasets_out.append({
                "dataset": ds_name,
                "examples": examples,
            })

    output = {
        "metadata": {
            "experiment": "codebook_figs_benchmark",
            "description": (
                "Codebook-FIGS vs baselines (FIGS, XGBoost, LightGBM) across "
                "10 tabular datasets with K in {3,5,8,12,20}"
            ),
            "config": config,
            "summary": summary,
        },
        "datasets": datasets_out,
    }
    return output


@logger.catch
def main() -> None:
    """Main experiment entry point."""
    start_time = time.time()
    logger.info("=" * 70)
    logger.info("Codebook-FIGS Benchmark Starting")
    logger.info("=" * 70)

    # Resolve dependency paths — data is in iter_1's gen_art directory
    data_dir = WORKSPACE.parent / "data_id2_it1__opus"
    if not data_dir.exists():
        # Try iter_1 path (data was generated in iteration 1)
        # WORKSPACE = .../3_invention_loop/iter_2/gen_art/exp_id1_it2__opus
        # parents[2] = .../3_invention_loop
        data_dir = (
            WORKSPACE.parents[2]  # up to 3_invention_loop
            / "iter_1" / "gen_art" / "data_id2_it1__opus"
        )
    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    logger.info(f"Data directory: {data_dir}")

    # ── Load data ────────────────────────────────────────────────────────
    mini_path = data_dir / "mini_data_out.json"
    full_path = data_dir / "full_data_out.json"

    mini_datasets = load_datasets(str(mini_path))
    full_datasets = load_datasets(str(full_path))

    # Separate mini vs full
    full_only = {k: v for k, v in full_datasets.items() if not k.endswith("_mini")}
    mini_only = {k: v for k, v in mini_datasets.items() if k.endswith("_mini")}

    logger.info(f"Full datasets: {list(full_only.keys())}")
    logger.info(f"Mini datasets: {list(mini_only.keys())}")

    all_results: list[dict[str, Any]] = []

    # ── PHASE 5A: Smoke test on mini dataset ─────────────────────────────
    # Use the _mini datasets from full_data_out.json (200 samples each)
    logger.info("\n=== SMOKE TEST on mini dataset ===")
    smoke_ds_name = "heart_disease_mini"
    smoke_ds = full_datasets.get(smoke_ds_name)
    if smoke_ds is not None and smoke_ds["n_samples"] >= 10:
        ds = smoke_ds
        X, y, folds = ds["X"], ds["y"], ds["folds"]
        fold0_mask = folds == 0
        if np.sum(fold0_mask) > 0 and np.sum(~fold0_mask) > 0:
            X_tr, X_te = X[~fold0_mask], X[fold0_mask]
            y_tr, y_te = y[~fold0_mask], y[fold0_mask]

            try:
                model = CodebookFIGS(
                    K=5, max_rules=4, max_depth=3,
                    n_alternation_rounds=1, random_state=42,
                )
                model.fit(X_tr, y_tr, task_type="classification")
                preds = model.predict_classification(X_te)
                proba = model.predict_proba(X_te)
                acc = accuracy_score(y_te, preds)
                logger.info(
                    f"Smoke test PASSED: acc={acc:.3f}, "
                    f"complexity={model.complexity_}, "
                    f"codebook shape={model.codebook_.shape}, "
                    f"n_trees={len(model.trees_)}"
                )
                assert np.all(np.isfinite(preds)), "Non-finite predictions"
                K_expected = min(5, X.shape[1], X_tr.shape[0])
                assert model.codebook_.shape[0] == K_expected, (
                    f"Bad codebook shape: {model.codebook_.shape}, expected K={K_expected}"
                )
                # Check unit norms
                norms = np.linalg.norm(model.codebook_, axis=1)
                assert np.allclose(norms, 1.0, atol=1e-6), f"Codebook not unit norm: {norms}"
            except Exception:
                logger.exception("Smoke test FAILED")
                raise
        else:
            logger.warning("Fold 0 split not valid for smoke test, skipping")
    else:
        logger.warning(f"{smoke_ds_name} not found or too small for smoke test")

    # ── PHASE 5B: Full benchmark loop ────────────────────────────────────
    logger.info("\n=== FULL BENCHMARK ===")

    for ds_name, ds_info in full_only.items():
        X, y = ds_info["X"], ds_info["y"]
        folds = ds_info["folds"]
        task_type = ds_info["task_type"]
        n_features = ds_info["n_features"]

        logger.info(
            f"\n=== Dataset: {ds_name} ({task_type}, n={X.shape[0]}, d={X.shape[1]}) ==="
        )

        for fold_idx in range(N_FOLDS):
            test_mask = folds == fold_idx
            train_mask = ~test_mask
            if np.sum(test_mask) == 0 or np.sum(train_mask) == 0:
                logger.warning(f"  Fold {fold_idx}: empty split, skipping")
                continue

            X_train, X_test = X[train_mask], X[test_mask]
            y_train, y_test = y[train_mask], y[test_mask]

            # ── Baselines ────────────────────────────────────────────
            for method_name in ["figs", "xgboost", "lightgbm"]:
                t0 = time.time()
                try:
                    if method_name == "figs":
                        y_pred, y_proba, n_splits = run_figs_baseline(
                            X_train, y_train, X_test, task_type
                        )
                    elif method_name == "xgboost":
                        y_pred, y_proba = run_xgboost_baseline(
                            X_train, y_train, X_test, task_type
                        )
                        n_splits = None
                    elif method_name == "lightgbm":
                        y_pred, y_proba = run_lgbm_baseline(
                            X_train, y_train, X_test, task_type
                        )
                        n_splits = None
                    else:
                        continue
                    elapsed = time.time() - t0

                    metrics = evaluate_fold(y_test, y_pred, y_proba, task_type)
                    result: dict[str, Any] = {
                        "dataset": ds_name,
                        "method": method_name,
                        "K": None,
                        "fold": fold_idx,
                        "train_time_sec": round(elapsed, 4),
                        "n_splits": n_splits,
                        "erank": None,
                        "n_codebook_used": None,
                        **metrics,
                    }
                    all_results.append(result)
                    logger.info(f"  {method_name} fold={fold_idx}: {metrics} ({elapsed:.2f}s)")

                except Exception:
                    logger.exception(f"  {method_name} fold={fold_idx} FAILED")
                    result = {
                        "dataset": ds_name,
                        "method": method_name,
                        "K": None,
                        "fold": fold_idx,
                        "train_time_sec": time.time() - t0,
                        "n_splits": None,
                        "erank": None,
                        "n_codebook_used": None,
                        "accuracy": None,
                        "auroc": None,
                        "rmse": None,
                        "r2": None,
                    }
                    all_results.append(result)

            # ── Codebook-FIGS for each K ─────────────────────────────
            for K in K_VALUES:
                K_actual = min(K, n_features)
                t0 = time.time()

                try:
                    model = CodebookFIGS(
                        K=K_actual,
                        max_rules=MAX_RULES,
                        max_trees=None,
                        max_depth=MAX_DEPTH,
                        min_impurity_decrease=0.0,
                        n_alternation_rounds=N_ALTERNATION_ROUNDS,
                        random_state=42,
                    )
                    model.fit(X_train, y_train, task_type=task_type)
                    elapsed = time.time() - t0

                    if task_type == "classification":
                        y_pred = model.predict_classification(X_test)
                        y_proba = model.predict_proba(X_test)
                    else:
                        y_pred = model.predict(X_test)
                        y_proba = None

                    metrics = evaluate_fold(y_test, y_pred, y_proba, task_type)

                    # Compute eRank and codebook usage
                    usage = model.codebook_usage_
                    usage_array = np.array(
                        [usage.get(k, 0) for k in range(K_actual)], dtype=float
                    )
                    erank = compute_erank(model.codebook_, usage_array)
                    n_used = int(np.sum(usage_array > 0))

                    result = {
                        "dataset": ds_name,
                        "method": "codebook_figs",
                        "K": K,
                        "fold": fold_idx,
                        "train_time_sec": round(elapsed, 4),
                        "n_splits": model.complexity_,
                        "erank": round(erank, 4),
                        "n_codebook_used": n_used,
                        **metrics,
                    }
                    all_results.append(result)
                    logger.info(
                        f"  CB-FIGS K={K} fold={fold_idx}: {metrics}, "
                        f"erank={erank:.2f}, used={n_used}/{K_actual} ({elapsed:.2f}s)"
                    )

                except Exception:
                    logger.exception(f"  CB-FIGS K={K} fold={fold_idx} FAILED")
                    result = {
                        "dataset": ds_name,
                        "method": "codebook_figs",
                        "K": K,
                        "fold": fold_idx,
                        "train_time_sec": time.time() - t0,
                        "n_splits": None,
                        "erank": None,
                        "n_codebook_used": None,
                        "accuracy": None,
                        "auroc": None,
                        "rmse": None,
                        "r2": None,
                    }
                    all_results.append(result)

                # Check total runtime budget
                if time.time() - start_time > 3000:  # 50 min guard
                    logger.warning("Approaching time limit, stopping early")
                    break
            else:
                continue
            break  # break outer fold loop if inner K loop broke

        # Check total runtime budget
        if time.time() - start_time > 3000:
            logger.warning("Approaching time limit, stopping early")
            break

    # ── PHASE 5C: Aggregate and summarize ────────────────────────────────
    logger.info("\n=== AGGREGATING RESULTS ===")
    summary = compute_summary(all_results)

    config = {
        "K_values": K_VALUES,
        "n_folds": N_FOLDS,
        "max_rules": MAX_RULES,
        "max_depth": MAX_DEPTH,
        "n_alternation_rounds": N_ALTERNATION_ROUNDS,
        "methods": ["codebook_figs", "figs", "xgboost", "lightgbm"],
        "datasets": list(full_only.keys()),
    }

    # ── PHASE 5D: Write output ───────────────────────────────────────────
    logger.info("\n=== WRITING OUTPUT ===")

    # Write schema-compliant output
    output = format_output_for_schema(all_results, summary, full_datasets, config)
    out_path = WORKSPACE / "method_out.json"
    out_path.write_text(json.dumps(output, indent=2, default=str))
    logger.info(f"Schema-compliant output written to {out_path}")

    # Also write detailed results (for analysis)
    detailed = {
        "experiment": "codebook_figs_benchmark",
        "description": (
            "Codebook-FIGS vs baselines across 10 tabular datasets "
            "with K in {3,5,8,12,20}"
        ),
        "results": all_results,
        "summary": summary,
        "config": config,
    }
    detail_path = WORKSPACE / "detailed_results.json"
    detail_path.write_text(json.dumps(detailed, indent=2, default=str))
    logger.info(f"Detailed results written to {detail_path}")

    elapsed_total = time.time() - start_time
    logger.info(f"\nTotal runtime: {elapsed_total:.1f}s ({elapsed_total/60:.1f}min)")
    logger.info(f"Total result rows: {len(all_results)}")
    logger.info("Done!")


if __name__ == "__main__":
    main()
