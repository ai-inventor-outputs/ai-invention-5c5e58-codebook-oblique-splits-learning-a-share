#!/usr/bin/env python3
"""Codebook-FIGS Ablation: Initialization, Refinement & Stability.

Implements Codebook-FIGS from scratch, ablates 3 initialization strategies
(PCA, Random, LDA-inspired) × 2 refinement strategies (closed-form WLS, L-BFGS)
= 6 configurations on ALL datasets with K=8 and 5-fold CV.

Measures codebook stability via Hungarian-aligned cosine similarity across folds,
tracks convergence (accuracy trajectory, eRank per round), and identifies the
best configuration.  Also runs standard axis-aligned FIGS as a baseline.
"""

from loguru import logger
from pathlib import Path
import json
import sys
import time
import resource
import numpy as np
from typing import List, Dict, Tuple, Optional, Any

# ── logging ──────────────────────────────────────────────────────────────────
WORKSPACE = Path(__file__).resolve().parent
LOG_DIR = WORKSPACE / "logs"
LOG_DIR.mkdir(exist_ok=True)

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add(str(LOG_DIR / "run.log"), rotation="30 MB", level="DEBUG")

# ── resource limits ──────────────────────────────────────────────────────────
# 20 GB RAM cap, 1h CPU
try:
    resource.setrlimit(resource.RLIMIT_AS, (20 * 1024**3, 20 * 1024**3))
    resource.setrlimit(resource.RLIMIT_CPU, (3600, 3600))
except Exception:
    logger.warning("Could not set resource limits")

# ── paths ────────────────────────────────────────────────────────────────────
DATA_DIR = WORKSPACE.parent.parent.parent / "iter_1" / "gen_art" / "data_id2_it1__opus"
FULL_DATA_PATH = DATA_DIR / "full_data_out.json"
MINI_DATA_PATH = DATA_DIR / "mini_data_out.json"


# ============================================================
# PHASE 0: DATA LOADING
# ============================================================

def load_all_datasets(data_path: Path) -> Dict[str, Dict]:
    """Load all datasets from a data_out JSON file.

    Returns dict keyed by dataset name with keys:
      X, y, folds, task_type, feature_names, n_features
    """
    logger.info(f"Loading datasets from {data_path}")
    raw = json.loads(data_path.read_text())
    datasets: Dict[str, Dict] = {}

    for group in raw["datasets"]:
        name = group["dataset"]
        # Skip mini versions – we load those separately when needed
        if name.endswith("_mini"):
            continue
        examples = group["examples"]
        first = examples[0]
        task_type: str = first["metadata_task_type"]
        n_features: int = first.get("metadata_n_features", None)
        feature_names: List[str] = first.get("metadata_feature_names", [])

        X_list, y_list, fold_list = [], [], []
        for ex in examples:
            X_list.append(json.loads(ex["input"]))
            y_list.append(float(ex["output"]))
            fold_list.append(int(ex["metadata_fold"]))

        X = np.array(X_list, dtype=np.float64)
        y = np.array(y_list, dtype=np.float64)
        folds = np.array(fold_list, dtype=np.int32)

        if n_features is None:
            n_features = X.shape[1]

        datasets[name] = {
            "X": X,
            "y": y,
            "folds": folds,
            "task_type": task_type,
            "feature_names": feature_names,
            "n_features": n_features,
        }
        logger.info(f"  {name}: {X.shape[0]} samples, {X.shape[1]} features, {task_type}")

    return datasets


def load_mini_datasets(data_path: Path) -> Dict[str, Dict]:
    """Load mini datasets (for testing) from mini_data_out.json."""
    logger.info(f"Loading mini datasets from {data_path}")
    raw = json.loads(data_path.read_text())
    datasets: Dict[str, Dict] = {}

    for group in raw["datasets"]:
        name = group["dataset"]
        if not name.endswith("_mini"):
            continue
        examples = group["examples"]
        first = examples[0]
        task_type: str = first["metadata_task_type"]
        n_features: int = first.get("metadata_n_features", None)
        feature_names: List[str] = first.get("metadata_feature_names", [])

        X_list, y_list, fold_list = [], [], []
        for ex in examples:
            X_list.append(json.loads(ex["input"]))
            y_list.append(float(ex["output"]))
            fold_list.append(int(ex["metadata_fold"]))

        X = np.array(X_list, dtype=np.float64)
        y = np.array(y_list, dtype=np.float64)
        folds = np.array(fold_list, dtype=np.int32)

        if n_features is None:
            n_features = X.shape[1]

        # Store without _mini suffix for easy matching
        base_name = name.replace("_mini", "")
        datasets[base_name] = {
            "X": X,
            "y": y,
            "folds": folds,
            "task_type": task_type,
            "feature_names": feature_names,
            "n_features": n_features,
        }
        logger.info(f"  {base_name} (mini): {X.shape[0]} samples, {X.shape[1]} features, {task_type}")

    return datasets


# ============================================================
# PHASE 1: CODEBOOK-FIGS CORE IMPLEMENTATION
# ============================================================

class Node:
    """Tree node for Codebook-FIGS."""
    __slots__ = [
        "codebook_idx", "weights", "threshold", "value",
        "idxs", "left", "right", "left_temp", "right_temp",
        "impurity_reduction", "is_root", "tree_num", "depth", "n_samples",
    ]

    def __init__(self) -> None:
        self.codebook_idx: Optional[int] = None
        self.weights: Optional[np.ndarray] = None
        self.threshold: Optional[float] = None
        self.value: float = 0.0
        self.idxs: Optional[np.ndarray] = None
        self.left: Optional["Node"] = None
        self.right: Optional["Node"] = None
        self.left_temp: Optional["Node"] = None
        self.right_temp: Optional["Node"] = None
        self.impurity_reduction: Optional[float] = None
        self.is_root: bool = False
        self.tree_num: int = -1
        self.depth: int = 0
        self.n_samples: int = 0


# ── Codebook initialization strategies ──────────────────────────────────────

def init_codebook_pca(X_train: np.ndarray, K: int) -> np.ndarray:
    """PCA initialization: top-K principal components of X_train."""
    from sklearn.decomposition import PCA
    X_centered = X_train - X_train.mean(axis=0)
    K_actual = min(K, min(X_train.shape[0], X_train.shape[1]))
    pca = PCA(n_components=K_actual)
    pca.fit(X_centered)
    C = pca.components_.copy()  # [K_actual, n_features]
    if K_actual < K:
        rng = np.random.RandomState(42)
        extra = rng.randn(K - K_actual, X_train.shape[1])
        extra = extra / (np.linalg.norm(extra, axis=1, keepdims=True) + 1e-12)
        C = np.vstack([C, extra])
    # Normalize rows
    norms = np.linalg.norm(C, axis=1, keepdims=True)
    C = C / np.maximum(norms, 1e-12)
    return C


def init_codebook_random(X_train: np.ndarray, K: int, random_state: int = 42) -> np.ndarray:
    """Random initialization: K random unit vectors from standard normal."""
    rng = np.random.RandomState(random_state)
    C = rng.randn(K, X_train.shape[1])
    C = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-12)
    return C


def init_codebook_lda(
    X_train: np.ndarray,
    y_train: np.ndarray,
    K: int,
    task_type: str,
) -> np.ndarray:
    """LDA-inspired initialization via between-class scatter SVD."""
    try:
        if task_type == "regression":
            n_bins = min(10, max(2, len(np.unique(y_train)) // 2))
            from sklearn.preprocessing import KBinsDiscretizer
            kbd = KBinsDiscretizer(
                n_bins=n_bins, encode="ordinal", strategy="quantile",
            )
            y_binned = kbd.fit_transform(y_train.reshape(-1, 1)).ravel().astype(int)
        else:
            y_binned = y_train.astype(int)

        classes = np.unique(y_binned)
        n_features = X_train.shape[1]
        mu = X_train.mean(axis=0)

        S_B = np.zeros((n_features, n_features))
        for c in classes:
            mask = y_binned == c
            n_c = mask.sum()
            if n_c == 0:
                continue
            mu_c = X_train[mask].mean(axis=0)
            diff = mu_c - mu
            S_B += n_c * np.outer(diff, diff)

        _U, _S_vals, Vt = np.linalg.svd(S_B)
        n_lda = min(len(classes) - 1, n_features, K)
        n_lda = max(n_lda, 1)

        C_lda = Vt[:n_lda].copy()
        norms = np.linalg.norm(C_lda, axis=1, keepdims=True)
        C_lda = C_lda / np.maximum(norms, 1e-12)

        if n_lda < K:
            C_pca = init_codebook_pca(X_train, K)
            C = np.vstack([C_lda, C_pca[n_lda:K]])
        else:
            C = C_lda[:K]

        norms = np.linalg.norm(C, axis=1, keepdims=True)
        C = C / np.maximum(norms, 1e-12)
        return C
    except Exception:
        logger.warning("LDA init failed – falling back to PCA")
        return init_codebook_pca(X_train, K)


# ── Split search ─────────────────────────────────────────────────────────────

def find_best_codebook_split(
    X: np.ndarray,
    y_residuals: np.ndarray,
    idxs: np.ndarray,
    codebook: np.ndarray,
    min_samples_leaf: int = 5,
    quantile_threshold: int = 5000,
) -> Tuple[Optional[int], Optional[float], float, Optional[np.ndarray], Optional[np.ndarray]]:
    """Find the best (codebook direction, threshold) split."""
    node_indices = np.where(idxs)[0]
    n_node = len(node_indices)

    if n_node < 2 * min_samples_leaf:
        return None, None, -np.inf, None, None

    X_node = X[node_indices]
    y_node = y_residuals[node_indices]
    parent_var = np.var(y_node) * n_node

    best_k: Optional[int] = None
    best_threshold: Optional[float] = None
    best_reduction = -np.inf
    K = codebook.shape[0]

    # For large nodes, use quantile-based threshold search
    use_quantiles = n_node > quantile_threshold
    n_quantiles = 100

    for k in range(K):
        proj = X_node @ codebook[k]
        order = np.argsort(proj)
        sorted_proj = proj[order]
        sorted_y = y_node[order]

        cum_sum = np.cumsum(sorted_y)
        cum_sq_sum = np.cumsum(sorted_y ** 2)
        total_sum = cum_sum[-1]
        total_sq_sum = cum_sq_sum[-1]

        if use_quantiles:
            # Evaluate only at quantile positions
            positions = np.linspace(
                min_samples_leaf - 1,
                n_node - min_samples_leaf - 1,
                min(n_quantiles, n_node - 2 * min_samples_leaf),
            ).astype(int)
            positions = np.unique(positions)
        else:
            positions = np.arange(min_samples_leaf - 1, n_node - min_samples_leaf)

        for i in positions:
            if i + 1 < n_node and sorted_proj[i] == sorted_proj[i + 1]:
                continue

            n_left = i + 1
            n_right = n_node - n_left

            left_sum = cum_sum[i]
            right_sum = total_sum - left_sum

            left_mse = cum_sq_sum[i] - (left_sum ** 2) / n_left
            right_mse = (total_sq_sum - cum_sq_sum[i]) - (right_sum ** 2) / n_right

            reduction = parent_var - left_mse - right_mse

            if reduction > best_reduction:
                best_reduction = reduction
                best_k = k
                if i + 1 < n_node:
                    best_threshold = (sorted_proj[i] + sorted_proj[i + 1]) / 2.0
                else:
                    best_threshold = sorted_proj[i]

    if best_k is None:
        return None, None, -np.inf, None, None

    # Compute child masks on FULL data
    full_proj = X @ codebook[best_k]
    idxs_left = idxs & (full_proj <= best_threshold)
    idxs_right = idxs & (full_proj > best_threshold)

    return best_k, best_threshold, best_reduction, idxs_left, idxs_right


# ── Effective rank ───────────────────────────────────────────────────────────

def effective_rank(C: np.ndarray, eps: float = 1e-12) -> float:
    """Compute effective rank (eRank) of codebook matrix."""
    s = np.linalg.svd(C, compute_uv=False)
    s = s[s > eps]
    if len(s) == 0:
        return 0.0
    p = s / np.sum(s)
    entropy = -np.sum(p * np.log(p + 1e-30))
    return float(np.exp(entropy))


# ── Codebook-FIGS model ─────────────────────────────────────────────────────

class CodebookFIGS:
    """Codebook-FIGS: FIGS with oblique splits constrained to a shared codebook."""

    def __init__(
        self,
        K: int = 8,
        max_rules: int = 12,
        max_trees: Optional[int] = None,
        min_impurity_decrease: float = 0.0,
        max_depth: Optional[int] = None,
        n_alternation_rounds: int = 5,
        init_strategy: str = "pca",
        refine_strategy: str = "wls",
        random_state: int = 42,
        min_samples_leaf: int = 5,
    ) -> None:
        self.K = K
        self.max_rules = max_rules
        self.max_trees = max_trees
        self.min_impurity_decrease = min_impurity_decrease
        self.max_depth = max_depth
        self.n_alternation_rounds = n_alternation_rounds
        self.init_strategy = init_strategy
        self.refine_strategy = refine_strategy
        self.random_state = random_state
        self.min_samples_leaf = min_samples_leaf

        # Populated during fit
        self.trees_: List[Node] = []
        self.codebook_: Optional[np.ndarray] = None
        self.scaler_: Any = None
        self.task_type_: str = "classification"
        self.complexity_: int = 0
        self.history_: Dict[str, list] = {}
        self.converged_at_round_: int = 0

    def fit(self, X: np.ndarray, y: np.ndarray, task_type: str = "classification") -> "CodebookFIGS":
        from sklearn.preprocessing import StandardScaler

        n_samples, n_features = X.shape
        self.task_type_ = task_type

        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X)

        # Initialize codebook
        if self.init_strategy == "pca":
            self.codebook_ = init_codebook_pca(X_scaled, self.K)
        elif self.init_strategy == "random":
            self.codebook_ = init_codebook_random(X_scaled, self.K, self.random_state)
        elif self.init_strategy == "lda":
            self.codebook_ = init_codebook_lda(X_scaled, y, self.K, task_type)
        else:
            raise ValueError(f"Unknown init_strategy: {self.init_strategy}")

        self.history_ = {
            "codebooks": [],
            "train_losses": [],
            "eranks": [],
            "n_splits": [],
            "accuracies": [],
        }

        initial_loss: Optional[float] = None

        for round_idx in range(self.n_alternation_rounds):
            # STEP A: Grow trees with current codebook
            self._grow_trees(X_scaled, y)

            # Compute training loss
            preds = self._predict_raw(X_scaled)
            loss = float(np.mean((y - preds) ** 2))

            if task_type == "classification":
                acc = float(np.mean((preds >= 0.5).astype(int) == y.astype(int)))
                self.history_["accuracies"].append(acc)
            else:
                ss_res = np.sum((y - preds) ** 2)
                ss_tot = np.sum((y - y.mean()) ** 2)
                r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
                self.history_["accuracies"].append(float(r2))

            self.history_["train_losses"].append(loss)
            self.history_["codebooks"].append(self.codebook_.copy())
            self.history_["eranks"].append(effective_rank(self.codebook_))
            self.history_["n_splits"].append(self.complexity_)

            if initial_loss is None:
                initial_loss = loss

            # STEP B: Refine codebook (skip on last round)
            if round_idx < self.n_alternation_rounds - 1:
                if self.refine_strategy == "wls":
                    self._refine_codebook_wls(X_scaled, y)
                elif self.refine_strategy == "lbfgs":
                    self._refine_codebook_lbfgs(X_scaled, y)

                # Check for codebook collapse — reinitialize duplicates
                self._check_codebook_collapse()

                # Convergence check
                if len(self.history_["train_losses"]) >= 2 and initial_loss is not None:
                    loss_change = abs(
                        self.history_["train_losses"][-1] - self.history_["train_losses"][-2]
                    )
                    if initial_loss > 0 and loss_change < 0.01 * initial_loss:
                        self.converged_at_round_ = round_idx + 1
                        break

        if not hasattr(self, "converged_at_round_") or self.converged_at_round_ == 0:
            self.converged_at_round_ = len(self.history_["train_losses"])

        return self

    # ── Tree growing ─────────────────────────────────────────────────────────

    def _grow_trees(self, X: np.ndarray, y: np.ndarray) -> None:
        n_samples = X.shape[0]
        self.trees_ = []
        self.complexity_ = 0
        y_pred_per_tree: Dict[int, np.ndarray] = {}

        all_idxs = np.ones(n_samples, dtype=bool)
        initial_residuals = y.copy()

        best_k, best_t, best_red, idxs_l, idxs_r = find_best_codebook_split(
            X, initial_residuals, all_idxs, self.codebook_, self.min_samples_leaf,
        )

        if best_k is None:
            # No valid split — create trivial model
            root = Node()
            root.is_root = True
            root.tree_num = 0
            root.idxs = all_idxs
            root.value = float(np.mean(y))
            root.n_samples = n_samples
            self.trees_.append(root)
            y_pred_per_tree[0] = np.full(n_samples, root.value)
            return

        root = Node()
        root.is_root = True
        root.tree_num = -1
        root.idxs = all_idxs
        root.value = float(np.mean(y))
        root.impurity_reduction = best_red
        root.codebook_idx = best_k
        root.weights = self.codebook_[best_k].copy()
        root.threshold = best_t
        root.depth = 0
        root.n_samples = n_samples

        root.left_temp = Node()
        root.left_temp.idxs = idxs_l
        root.left_temp.value = float(np.mean(initial_residuals[idxs_l])) if idxs_l.sum() > 0 else 0.0
        root.left_temp.n_samples = int(idxs_l.sum())
        root.left_temp.depth = 1

        root.right_temp = Node()
        root.right_temp.idxs = idxs_r
        root.right_temp.value = float(np.mean(initial_residuals[idxs_r])) if idxs_r.sum() > 0 else 0.0
        root.right_temp.n_samples = int(idxs_r.sum())
        root.right_temp.depth = 1

        potential_splits: List[Node] = [root]
        max_new_trees_per_round = 10  # safety cap

        while potential_splits and self.complexity_ < self.max_rules:
            potential_splits.sort(
                key=lambda n: n.impurity_reduction if n.impurity_reduction is not None else -np.inf,
                reverse=True,
            )
            best_node = potential_splits.pop(0)

            if best_node.impurity_reduction is None or best_node.impurity_reduction < self.min_impurity_decrease:
                break

            if self.max_depth is not None and best_node.depth >= self.max_depth:
                continue

            # If root, register new tree
            if best_node.is_root:
                if self.max_trees is not None and len(self.trees_) >= self.max_trees:
                    continue
                if len(self.trees_) >= max_new_trees_per_round:
                    continue
                best_node.tree_num = len(self.trees_)
                self.trees_.append(best_node)
                y_pred_per_tree[best_node.tree_num] = np.full(n_samples, best_node.value)

            # Commit split
            self.complexity_ += 1
            best_node.left = best_node.left_temp
            best_node.right = best_node.right_temp
            best_node.left.tree_num = best_node.tree_num
            best_node.right.tree_num = best_node.tree_num
            best_node.left_temp = None
            best_node.right_temp = None

            # Update predictions for this tree
            tree_root = self.trees_[best_node.tree_num]
            y_pred_per_tree[best_node.tree_num] = self._predict_tree(tree_root, X)

            # Add children as potential splits
            for child in [best_node.left, best_node.right]:
                if child.n_samples < 2 * self.min_samples_leaf:
                    continue

                # Compute residuals for the tree containing this child
                t = child.tree_num
                y_residual = y.copy()
                for other_t, pred in y_pred_per_tree.items():
                    if other_t != t:
                        y_residual -= pred

                bk, bt, br, il, ir = find_best_codebook_split(
                    X, y_residual, child.idxs, self.codebook_, self.min_samples_leaf,
                )

                if bk is not None and br > self.min_impurity_decrease:
                    child.codebook_idx = bk
                    child.weights = self.codebook_[bk].copy()
                    child.threshold = bt
                    child.impurity_reduction = br

                    child.left_temp = Node()
                    child.left_temp.idxs = il
                    child.left_temp.value = float(np.mean(y_residual[il])) if il.sum() > 0 else 0.0
                    child.left_temp.n_samples = int(il.sum())
                    child.left_temp.depth = child.depth + 1

                    child.right_temp = Node()
                    child.right_temp.idxs = ir
                    child.right_temp.value = float(np.mean(y_residual[ir])) if ir.sum() > 0 else 0.0
                    child.right_temp.n_samples = int(ir.sum())
                    child.right_temp.depth = child.depth + 1

                    potential_splits.append(child)

            # Try adding a new root for the next tree
            if len(self.trees_) < max_new_trees_per_round:
                y_total_pred = np.zeros(n_samples)
                for pred in y_pred_per_tree.values():
                    y_total_pred += pred
                y_res_new = y - y_total_pred

                bk, bt, br, il, ir = find_best_codebook_split(
                    X, y_res_new, np.ones(n_samples, dtype=bool), self.codebook_, self.min_samples_leaf,
                )

                if bk is not None and br > self.min_impurity_decrease:
                    new_root = Node()
                    new_root.is_root = True
                    new_root.tree_num = -1
                    new_root.idxs = np.ones(n_samples, dtype=bool)
                    new_root.value = float(np.mean(y_res_new))
                    new_root.impurity_reduction = br
                    new_root.codebook_idx = bk
                    new_root.weights = self.codebook_[bk].copy()
                    new_root.threshold = bt
                    new_root.depth = 0
                    new_root.n_samples = n_samples

                    new_root.left_temp = Node()
                    new_root.left_temp.idxs = il
                    new_root.left_temp.value = float(np.mean(y_res_new[il])) if il.sum() > 0 else 0.0
                    new_root.left_temp.n_samples = int(il.sum())
                    new_root.left_temp.depth = 1

                    new_root.right_temp = Node()
                    new_root.right_temp.idxs = ir
                    new_root.right_temp.value = float(np.mean(y_res_new[ir])) if ir.sum() > 0 else 0.0
                    new_root.right_temp.n_samples = int(ir.sum())
                    new_root.right_temp.depth = 1

                    potential_splits.append(new_root)

    # ── Tree prediction ──────────────────────────────────────────────────────

    def _predict_tree(self, root: Node, X: np.ndarray) -> np.ndarray:
        preds = np.full(X.shape[0], root.value)
        if root.left is not None and root.right is not None:
            proj = X @ root.weights
            left_mask = proj <= root.threshold
            right_mask = ~left_mask
            left_preds = self._predict_tree(root.left, X)
            right_preds = self._predict_tree(root.right, X)
            preds[left_mask] = left_preds[left_mask]
            preds[right_mask] = right_preds[right_mask]
        return preds

    def _predict_raw(self, X: np.ndarray) -> np.ndarray:
        if not self.trees_:
            return np.zeros(X.shape[0])
        preds = np.zeros(X.shape[0])
        for tree in self.trees_:
            preds += self._predict_tree(tree, X)
        return preds

    def predict(self, X_raw: np.ndarray) -> np.ndarray:
        X = self.scaler_.transform(X_raw)
        raw = self._predict_raw(X)
        if self.task_type_ == "classification":
            return (raw >= 0.5).astype(int)
        return raw

    def predict_proba(self, X_raw: np.ndarray) -> np.ndarray:
        X = self.scaler_.transform(X_raw)
        raw = self._predict_raw(X)
        probs = np.clip(raw, 0.0, 1.0)
        return np.column_stack([1 - probs, probs])

    # ── Codebook refinement ──────────────────────────────────────────────────

    def _collect_nodes_with_codebook_idx(
        self, node: Optional[Node], k: int, result_list: List[Node],
    ) -> None:
        if node is None:
            return
        if node.left is not None and node.right is not None:
            if node.codebook_idx == k:
                result_list.append(node)
            self._collect_nodes_with_codebook_idx(node.left, k, result_list)
            self._collect_nodes_with_codebook_idx(node.right, k, result_list)

    def _refine_codebook_wls(self, X: np.ndarray, y: np.ndarray) -> None:
        K = self.codebook_.shape[0]
        n_features = self.codebook_.shape[1]

        for k in range(K):
            nodes_k: List[Node] = []
            for tree in self.trees_:
                self._collect_nodes_with_codebook_idx(tree, k, nodes_k)

            if len(nodes_k) == 0:
                rng = np.random.RandomState(self.random_state + k)
                self.codebook_[k] = rng.randn(n_features)
                self.codebook_[k] /= np.linalg.norm(self.codebook_[k]) + 1e-12
                continue

            all_X_pooled: List[np.ndarray] = []
            all_y_residual_pooled: List[np.ndarray] = []

            for node in nodes_k:
                y_res = y.copy()
                for t_idx, tree in enumerate(self.trees_):
                    if t_idx != node.tree_num:
                        y_res -= self._predict_tree(tree, X)
                node_mask = node.idxs
                all_X_pooled.append(X[node_mask])
                all_y_residual_pooled.append(y_res[node_mask])

            X_pool = np.vstack(all_X_pooled)
            y_pool = np.concatenate(all_y_residual_pooled)

            direction = X_pool.T @ y_pool
            norm = np.linalg.norm(direction)
            if norm > 1e-12:
                self.codebook_[k] = direction / norm

    def _refine_codebook_lbfgs(self, X: np.ndarray, y: np.ndarray) -> None:
        from scipy.optimize import minimize

        K = self.codebook_.shape[0]
        n_features = self.codebook_.shape[1]
        msl = self.min_samples_leaf

        for k in range(K):
            nodes_k: List[Node] = []
            for tree in self.trees_:
                self._collect_nodes_with_codebook_idx(tree, k, nodes_k)

            if len(nodes_k) == 0:
                rng = np.random.RandomState(self.random_state + k)
                self.codebook_[k] = rng.randn(n_features)
                self.codebook_[k] /= np.linalg.norm(self.codebook_[k]) + 1e-12
                continue

            # Precompute residuals for each node
            node_data: List[Tuple[np.ndarray, np.ndarray]] = []
            for node in nodes_k:
                y_res = y.copy()
                for t_idx, tree in enumerate(self.trees_):
                    if t_idx != node.tree_num:
                        y_res -= self._predict_tree(tree, X)
                node_data.append((X[node.idxs], y_res[node.idxs]))

            def neg_impurity_reduction(c_flat: np.ndarray) -> float:
                c = c_flat / (np.linalg.norm(c_flat) + 1e-12)
                total_reduction = 0.0
                for X_node, y_node in node_data:
                    n = len(y_node)
                    if n < 2 * msl:
                        continue
                    proj = X_node @ c
                    parent_var = np.var(y_node) * n
                    order = np.argsort(proj)
                    sy = y_node[order]
                    cs_y = np.cumsum(sy)
                    css_y = np.cumsum(sy ** 2)
                    best_red = -np.inf
                    # Use quantile positions for speed
                    positions = np.linspace(msl - 1, n - msl - 1, min(50, n - 2 * msl)).astype(int)
                    positions = np.unique(positions)
                    for i in positions:
                        nl = i + 1
                        nr = n - nl
                        l_mse = css_y[i] - cs_y[i] ** 2 / nl
                        r_mse = (css_y[-1] - css_y[i]) - (cs_y[-1] - cs_y[i]) ** 2 / nr
                        red = parent_var - l_mse - r_mse
                        if red > best_red:
                            best_red = red
                    total_reduction += best_red if best_red > -np.inf else 0.0
                return -total_reduction

            c0 = self.codebook_[k].copy()
            try:
                result = minimize(
                    neg_impurity_reduction, c0, method="L-BFGS-B",
                    jac="2-point",
                    options={"maxiter": 30, "ftol": 1e-6, "eps": 1e-5},
                )
                c_new = result.x
                norm = np.linalg.norm(c_new)
                if norm > 1e-12:
                    self.codebook_[k] = c_new / norm
            except Exception:
                logger.debug(f"L-BFGS failed for codebook entry {k}, keeping existing direction")

    def _check_codebook_collapse(self) -> int:
        """Check for collapsed codebook entries (cosine sim > 0.99) and reinitialize."""
        from sklearn.metrics.pairwise import cosine_similarity
        K = self.codebook_.shape[0]
        n_features = self.codebook_.shape[1]
        cos_sim = np.abs(cosine_similarity(self.codebook_))
        np.fill_diagonal(cos_sim, 0.0)
        n_collapses = 0
        for i in range(K):
            for j in range(i + 1, K):
                if cos_sim[i, j] > 0.99:
                    rng = np.random.RandomState(self.random_state + 100 + j)
                    self.codebook_[j] = rng.randn(n_features)
                    self.codebook_[j] /= np.linalg.norm(self.codebook_[j]) + 1e-12
                    n_collapses += 1
        return n_collapses


# ============================================================
# PHASE 2: METRICS & EVALUATION
# ============================================================

def align_codebooks_hungarian(
    C_ref: np.ndarray, C_target: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Align C_target to C_ref using Hungarian algorithm on |cosine similarity|."""
    from scipy.optimize import linear_sum_assignment
    from sklearn.metrics.pairwise import cosine_similarity

    cos_sim = cosine_similarity(C_ref, C_target)
    abs_cos_sim = np.abs(cos_sim)

    row_ind, col_ind = linear_sum_assignment(abs_cos_sim, maximize=True)

    C_aligned = C_target[col_ind].copy()
    matched_sims = []
    for i in range(len(row_ind)):
        sim = cos_sim[row_ind[i], col_ind[i]]
        matched_sims.append(abs(sim))
        if sim < 0:
            C_aligned[i] *= -1

    return C_aligned, col_ind, np.array(matched_sims)


def compute_codebook_stability(codebooks: List[np.ndarray]) -> Dict[str, Any]:
    """Compute codebook stability across CV folds."""
    if len(codebooks) < 2:
        return {
            "mean_cosine_sim": 1.0,
            "min_cosine_sim": 1.0,
            "per_entry_mean_sim": [1.0] * codebooks[0].shape[0],
            "per_entry_std_sim": [0.0] * codebooks[0].shape[0],
        }

    C_ref = codebooks[0]
    all_sims = []
    for fold_idx in range(1, len(codebooks)):
        _, _, sims = align_codebooks_hungarian(C_ref, codebooks[fold_idx])
        all_sims.append(sims)

    all_sims_arr = np.array(all_sims)
    return {
        "mean_cosine_sim": float(np.mean(all_sims_arr)),
        "min_cosine_sim": float(np.min(all_sims_arr)),
        "per_entry_mean_sim": all_sims_arr.mean(axis=0).tolist(),
        "per_entry_std_sim": all_sims_arr.std(axis=0).tolist(),
    }


def evaluate_fold_codebook(
    model: CodebookFIGS,
    X_test: np.ndarray,
    y_test: np.ndarray,
    task_type: str,
) -> Dict[str, Any]:
    """Evaluate CodebookFIGS on one fold's test set."""
    if task_type == "classification":
        y_pred = model.predict(X_test)
        accuracy = float(np.mean(y_pred == y_test.astype(int)))
        try:
            from sklearn.metrics import roc_auc_score
            y_prob = model.predict_proba(X_test)[:, 1]
            auc = float(roc_auc_score(y_test.astype(int), y_prob))
        except Exception:
            auc = None
        return {"accuracy": accuracy, "auc": auc}
    else:
        y_pred = model.predict(X_test)
        mse = float(np.mean((y_test - y_pred) ** 2))
        ss_res = float(np.sum((y_test - y_pred) ** 2))
        ss_tot = float(np.sum((y_test - y_test.mean()) ** 2))
        r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
        return {"mse": mse, "r2": float(r2)}


def evaluate_fold_sklearn(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    task_type: str,
) -> Dict[str, Any]:
    """Evaluate a sklearn-compatible model on one fold's test set."""
    if task_type == "classification":
        y_pred = model.predict(X_test)
        accuracy = float(np.mean(y_pred == y_test.astype(int)))
        try:
            from sklearn.metrics import roc_auc_score
            y_prob = model.predict_proba(X_test)[:, 1]
            auc = float(roc_auc_score(y_test.astype(int), y_prob))
        except Exception:
            auc = None
        return {"accuracy": accuracy, "auc": auc}
    else:
        y_pred = model.predict(X_test)
        mse = float(np.mean((y_test - y_pred) ** 2))
        ss_res = float(np.sum((y_test - y_pred) ** 2))
        ss_tot = float(np.sum((y_test - y_test.mean()) ** 2))
        r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
        return {"mse": mse, "r2": float(r2)}


def aggregate_fold_metrics(
    fold_metrics: List[Dict[str, Any]], task_type: str,
) -> Dict[str, Any]:
    """Aggregate metrics across folds."""
    if task_type == "classification":
        accs = [m["accuracy"] for m in fold_metrics]
        aucs = [m["auc"] for m in fold_metrics if m.get("auc") is not None]
        result = {
            "mean_accuracy": float(np.mean(accs)),
            "std_accuracy": float(np.std(accs)),
        }
        if aucs:
            result["mean_auc"] = float(np.mean(aucs))
            result["std_auc"] = float(np.std(aucs))
        else:
            result["mean_auc"] = None
            result["std_auc"] = None
        return result
    else:
        mses = [m["mse"] for m in fold_metrics]
        r2s = [m["r2"] for m in fold_metrics]
        return {
            "mean_mse": float(np.mean(mses)),
            "std_mse": float(np.std(mses)),
            "mean_r2": float(np.mean(r2s)),
            "std_r2": float(np.std(r2s)),
        }


# ============================================================
# PHASE 3: EXPERIMENT RUNNER
# ============================================================

INIT_STRATEGIES = ["pca", "random", "lda"]
REFINE_STRATEGIES = ["wls", "lbfgs"]
K = 8
N_ALTERNATION_ROUNDS = 5
MAX_RULES = 12
N_FOLDS = 5


def run_codebook_figs_ablation(
    datasets: Dict[str, Dict],
    dataset_names: List[str],
    K: int = 8,
    max_rules: int = 12,
    n_alternation_rounds: int = 5,
    n_folds: int = 5,
) -> Tuple[Dict[str, Dict], Dict[str, Dict[str, np.ndarray]]]:
    """Run full Codebook-FIGS ablation across datasets and configs.

    Returns:
        results: {dataset: {config: metrics_dict}}
        predictions: {dataset: {config: predictions_array[n_samples]}}
    """
    results: Dict[str, Dict] = {}
    all_predictions: Dict[str, Dict[str, np.ndarray]] = {}

    for ds_name in dataset_names:
        logger.info(f"=== Dataset: {ds_name} ===")
        ds = datasets[ds_name]
        X, y, folds, task_type = ds["X"], ds["y"], ds["folds"], ds["task_type"]
        n_samples = X.shape[0]

        # Build fold splits
        unique_folds = sorted(np.unique(folds))
        actual_n_folds = min(n_folds, len(unique_folds))
        fold_splits = []
        for fold_id in unique_folds[:actual_n_folds]:
            test_mask = folds == fold_id
            train_mask = ~test_mask
            fold_splits.append((train_mask, test_mask))

        dataset_results: Dict[str, Dict] = {}
        ds_predictions: Dict[str, np.ndarray] = {}

        for init_strat in INIT_STRATEGIES:
            for refine_strat in REFINE_STRATEGIES:
                config_name = f"{init_strat}_{refine_strat}"
                logger.info(f"  Config: {config_name}")
                t0 = time.time()

                fold_metrics: List[Dict] = []
                fold_codebooks: List[np.ndarray] = []
                fold_histories: List[Dict] = []
                fold_converged_rounds: List[int] = []
                # Per-example out-of-fold predictions
                oof_preds = np.full(n_samples, np.nan)

                for fold_id in range(actual_n_folds):
                    train_mask, test_mask = fold_splits[fold_id]
                    X_train, y_train = X[train_mask], y[train_mask]
                    X_test, y_test = X[test_mask], y[test_mask]

                    model = CodebookFIGS(
                        K=K,
                        max_rules=max_rules,
                        n_alternation_rounds=n_alternation_rounds,
                        init_strategy=init_strat,
                        refine_strategy=refine_strat,
                        random_state=42 + fold_id,
                        min_samples_leaf=5,
                    )

                    try:
                        model.fit(X_train, y_train, task_type=task_type)
                        metrics = evaluate_fold_codebook(model, X_test, y_test, task_type)
                        # Store OOF predictions
                        y_pred_test = model.predict(X_test)
                        oof_preds[test_mask] = y_pred_test
                    except Exception:
                        logger.exception(f"Failed on {ds_name}/{config_name}/fold{fold_id}")
                        if task_type == "classification":
                            metrics = {"accuracy": 0.0, "auc": None}
                        else:
                            metrics = {"mse": 1e6, "r2": -1.0}
                        model = CodebookFIGS(K=K)
                        model.codebook_ = np.eye(K, X.shape[1])[:K]
                        model.history_ = {"train_losses": [0], "eranks": [1], "n_splits": [0], "accuracies": [0], "codebooks": []}
                        model.converged_at_round_ = 0

                    fold_metrics.append(metrics)
                    if model.codebook_ is not None:
                        fold_codebooks.append(model.codebook_.copy())
                    fold_histories.append(model.history_)
                    fold_converged_rounds.append(model.converged_at_round_)

                # Replace remaining NaNs with mean prediction
                nan_mask = np.isnan(oof_preds)
                if nan_mask.any():
                    oof_preds[nan_mask] = np.nanmean(oof_preds) if not np.all(nan_mask) else 0.0
                ds_predictions[config_name] = oof_preds

                # Aggregate
                agg_metrics = aggregate_fold_metrics(fold_metrics, task_type)

                # Stability
                if len(fold_codebooks) >= 2:
                    stability = compute_codebook_stability(fold_codebooks)
                else:
                    stability = {
                        "mean_cosine_sim": None, "min_cosine_sim": None,
                        "per_entry_mean_sim": [], "per_entry_std_sim": [],
                    }

                # Convergence
                convergence = {
                    "mean_converged_round": float(np.mean(fold_converged_rounds)),
                    "per_fold_converged_round": [int(r) for r in fold_converged_rounds],
                    "per_fold_loss_trajectories": [h.get("train_losses", []) for h in fold_histories],
                    "per_fold_erank_trajectories": [h.get("eranks", []) for h in fold_histories],
                    "per_fold_accuracy_trajectories": [h.get("accuracies", []) for h in fold_histories],
                }

                elapsed = time.time() - t0
                logger.info(f"    {config_name} done in {elapsed:.1f}s | {agg_metrics}")

                dataset_results[config_name] = {
                    "per_fold_metrics": fold_metrics,
                    "aggregate_metrics": agg_metrics,
                    "codebook_stability": stability,
                    "convergence": convergence,
                    "final_erank_per_fold": [h["eranks"][-1] if h.get("eranks") else 0 for h in fold_histories],
                    "n_splits_per_fold": [h["n_splits"][-1] if h.get("n_splits") else 0 for h in fold_histories],
                    "time_seconds": round(elapsed, 2),
                }

        results[ds_name] = dataset_results
        all_predictions[ds_name] = ds_predictions

    return results, all_predictions


def run_figs_baseline(
    datasets: Dict[str, Dict],
    dataset_names: List[str],
    max_rules: int = 12,
    n_folds: int = 5,
) -> Tuple[Dict[str, Dict], Dict[str, np.ndarray]]:
    """Run standard axis-aligned FIGS baseline using imodels.

    Returns:
        figs_results: {dataset: {metrics}}
        figs_predictions: {dataset: predictions_array[n_samples]}
    """
    figs_predictions: Dict[str, np.ndarray] = {}

    try:
        from imodels import FIGSClassifier, FIGSRegressor
        logger.info("imodels loaded successfully")
    except ImportError:
        logger.warning("imodels not available – skipping FIGS baseline")
        return {"status": "imodels_not_available"}, {}

    figs_results: Dict[str, Dict] = {}

    for ds_name in dataset_names:
        logger.info(f"  FIGS baseline: {ds_name}")
        ds = datasets[ds_name]
        X, y, folds, task_type = ds["X"], ds["y"], ds["folds"], ds["task_type"]
        n_samples = X.shape[0]

        unique_folds = sorted(np.unique(folds))
        actual_n_folds = min(n_folds, len(unique_folds))
        fold_metrics: List[Dict] = []
        oof_preds = np.full(n_samples, np.nan)

        for fold_id in unique_folds[:actual_n_folds]:
            test_mask = folds == fold_id
            train_mask = ~test_mask
            X_train, y_train = X[train_mask], y[train_mask]
            X_test, y_test = X[test_mask], y[test_mask]

            try:
                if task_type == "classification":
                    figs_model = FIGSClassifier(max_rules=max_rules)
                else:
                    figs_model = FIGSRegressor(max_rules=max_rules)

                figs_model.fit(X_train, y_train)
                metrics = evaluate_fold_sklearn(figs_model, X_test, y_test, task_type)
                y_pred_test = figs_model.predict(X_test)
                oof_preds[test_mask] = y_pred_test
            except Exception:
                logger.exception(f"FIGS baseline failed on {ds_name}/fold{fold_id}")
                if task_type == "classification":
                    metrics = {"accuracy": 0.0, "auc": None}
                else:
                    metrics = {"mse": 1e6, "r2": -1.0}

            fold_metrics.append(metrics)

        # Fill NaNs
        nan_mask = np.isnan(oof_preds)
        if nan_mask.any():
            oof_preds[nan_mask] = np.nanmean(oof_preds) if not np.all(nan_mask) else 0.0
        figs_predictions[ds_name] = oof_preds

        agg = aggregate_fold_metrics(fold_metrics, task_type)
        figs_results[ds_name] = {
            "per_fold_metrics": fold_metrics,
            "aggregate_metrics": agg,
        }
        logger.info(f"    FIGS baseline {ds_name}: {agg}")

    return figs_results, figs_predictions


# ============================================================
# PHASE 4: SUMMARY & RANKING
# ============================================================

def compute_summary(
    codebook_results: Dict[str, Dict],
    figs_results: Dict[str, Dict],
    dataset_info: Dict[str, str],  # ds_name -> task_type
) -> Dict[str, Any]:
    """Compute summary statistics: best config, stability, convergence."""

    best_config_per_dataset: Dict[str, Dict] = {}
    config_ranks: Dict[str, List[float]] = {}

    for ds_name, ds_results in codebook_results.items():
        task_type = dataset_info[ds_name]

        # Rank configs by primary metric
        config_scores: List[Tuple[str, float]] = []
        for config_name, config_data in ds_results.items():
            agg = config_data["aggregate_metrics"]
            if task_type == "classification":
                score = agg.get("mean_accuracy", 0.0)
            else:
                score = agg.get("mean_r2", -1e6)
            config_scores.append((config_name, score))

        config_scores.sort(key=lambda x: x[1], reverse=True)

        best_name, best_score = config_scores[0]
        best_config_per_dataset[ds_name] = {
            "best_config": best_name,
            "score": best_score,
            "metric": "mean_accuracy" if task_type == "classification" else "mean_r2",
        }

        # Compute ranks (1 = best)
        for rank, (cname, _) in enumerate(config_scores, 1):
            config_ranks.setdefault(cname, []).append(rank)

    # Overall best = lowest mean rank
    overall_scores = {
        cname: float(np.mean(ranks)) for cname, ranks in config_ranks.items()
    }
    overall_best = min(overall_scores, key=overall_scores.get)

    # Stability summary
    stability_summary: Dict[str, Dict] = {}
    for ds_name, ds_results in codebook_results.items():
        best_cfg = best_config_per_dataset[ds_name]["best_config"]
        stab = ds_results[best_cfg]["codebook_stability"]
        stability_summary[ds_name] = {
            "config": best_cfg,
            "mean_cosine_sim": stab.get("mean_cosine_sim"),
            "min_cosine_sim": stab.get("min_cosine_sim"),
            "stable": (stab.get("mean_cosine_sim") or 0) > 0.8,
        }

    # Convergence summary
    convergence_summary: Dict[str, Dict] = {}
    for config_name in config_ranks:
        rounds_list = []
        for ds_name, ds_results in codebook_results.items():
            if config_name in ds_results:
                conv = ds_results[config_name]["convergence"]
                rounds_list.append(conv["mean_converged_round"])
        convergence_summary[config_name] = {
            "mean_converged_round_across_datasets": float(np.mean(rounds_list)) if rounds_list else None,
        }

    return {
        "best_config_per_dataset": best_config_per_dataset,
        "overall_best_config": overall_best,
        "overall_mean_ranks": overall_scores,
        "stability_summary": stability_summary,
        "convergence_summary": convergence_summary,
    }


# ============================================================
# PHASE 5: OUTPUT FORMATTING (exp_gen_sol_out schema)
# ============================================================

def format_output(
    codebook_results: Dict[str, Dict],
    figs_results: Dict[str, Dict],
    summary: Dict[str, Any],
    datasets: Dict[str, Dict],
    settings: Dict[str, Any],
    codebook_predictions: Dict[str, Dict[str, np.ndarray]],
    figs_predictions: Dict[str, np.ndarray],
) -> Dict[str, Any]:
    """Format output according to exp_gen_sol_out.json schema.

    Schema requires: {"datasets": [{"dataset": str, "examples": [{"input": str, "output": str, ...}]}]}
    Each example can have predict_* and metadata_* fields.
    """
    output_datasets: List[Dict[str, Any]] = []

    # Determine best config per dataset for the predict_codebook_figs_best field
    best_configs = summary.get("best_config_per_dataset", {})

    for ds_name in sorted(datasets.keys()):
        ds_data = datasets[ds_name]
        X = ds_data["X"]
        y = ds_data["y"]
        folds = ds_data["folds"]
        task_type = ds_data["task_type"]
        n_samples = X.shape[0]

        # Get per-example predictions
        ds_cb_preds = codebook_predictions.get(ds_name, {})
        ds_figs_preds = figs_predictions.get(ds_name, np.zeros(n_samples))
        best_cfg = best_configs.get(ds_name, {}).get("best_config", "pca_wls")
        best_preds = ds_cb_preds.get(best_cfg, np.zeros(n_samples))

        examples: List[Dict[str, Any]] = []

        for i in range(n_samples):
            ex: Dict[str, Any] = {
                "input": json.dumps(X[i].tolist()),
                "output": str(y[i]),
                "predict_codebook_figs_best": str(best_preds[i]),
                "predict_figs_baseline": str(ds_figs_preds[i]),
                "metadata_fold": int(folds[i]),
                "metadata_task_type": task_type,
            }

            # Add first-example metadata
            if i == 0:
                ex["metadata_n_samples"] = n_samples
                ex["metadata_n_features"] = int(X.shape[1])
                ex["metadata_feature_names"] = ds_data.get("feature_names", [])
                ex["metadata_best_config"] = best_cfg

                # Add per-config aggregate scores
                ds_codebook = codebook_results.get(ds_name, {})
                for config_name, config_data in ds_codebook.items():
                    agg = config_data["aggregate_metrics"]
                    if task_type == "classification":
                        score = agg.get("mean_accuracy", 0.0)
                    else:
                        score = agg.get("mean_r2", 0.0)
                    ex[f"metadata_codebook_figs_{config_name}_score"] = score

            examples.append(ex)

        output_datasets.append({
            "dataset": ds_name,
            "examples": examples,
        })

    # Metadata
    metadata = {
        "experiment": "codebook_figs_ablation_init_refine_stability",
        "settings": settings,
        "codebook_figs_results": _sanitize_for_json(codebook_results),
        "figs_baseline_results": _sanitize_for_json(figs_results),
        "summary": _sanitize_for_json(summary),
    }

    return {
        "metadata": metadata,
        "datasets": output_datasets,
    }


def _sanitize_for_json(obj: Any) -> Any:
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return obj
    return obj


# ============================================================
# MAIN
# ============================================================

@logger.catch
def main() -> None:
    t_start = time.time()

    # ── Determine mode ───────────────────────────────────────────────────────
    # Check for --mini flag for testing
    mini_mode = "--mini" in sys.argv
    quick_mode = "--quick" in sys.argv

    if mini_mode:
        logger.info("=== MINI MODE (testing) ===")
        datasets = load_mini_datasets(MINI_DATA_PATH)
        n_alt = 2
        k_val = 3
        mr_val = 4
    elif quick_mode:
        logger.info("=== QUICK MODE (reduced settings) ===")
        datasets = load_all_datasets(FULL_DATA_PATH)
        n_alt = 3
        k_val = 8
        mr_val = 12
    else:
        logger.info("=== FULL MODE ===")
        datasets = load_all_datasets(FULL_DATA_PATH)
        n_alt = N_ALTERNATION_ROUNDS
        k_val = K
        mr_val = MAX_RULES

    dataset_names = sorted(datasets.keys())
    logger.info(f"Datasets ({len(dataset_names)}): {dataset_names}")

    dataset_info = {name: ds["task_type"] for name, ds in datasets.items()}

    # ── Run Codebook-FIGS ablation ───────────────────────────────────────────
    logger.info("Starting Codebook-FIGS ablation...")
    codebook_results, codebook_predictions = run_codebook_figs_ablation(
        datasets, dataset_names,
        K=k_val, max_rules=mr_val,
        n_alternation_rounds=n_alt,
        n_folds=N_FOLDS,
    )
    t_codebook = time.time()
    logger.info(f"Codebook-FIGS ablation done in {t_codebook - t_start:.1f}s")

    # ── Run FIGS baseline ────────────────────────────────────────────────────
    logger.info("Starting FIGS baseline...")
    figs_results, figs_predictions = run_figs_baseline(
        datasets, dataset_names,
        max_rules=mr_val, n_folds=N_FOLDS,
    )
    t_figs = time.time()
    logger.info(f"FIGS baseline done in {t_figs - t_codebook:.1f}s")

    # ── Compute summary ──────────────────────────────────────────────────────
    summary = compute_summary(codebook_results, figs_results, dataset_info)

    settings = {
        "K": k_val,
        "max_rules": mr_val,
        "n_alternation_rounds": n_alt,
        "n_folds": N_FOLDS,
        "datasets": dataset_names,
        "init_strategies": INIT_STRATEGIES,
        "refine_strategies": REFINE_STRATEGIES,
        "min_samples_leaf": 5,
        "mini_mode": mini_mode,
        "quick_mode": quick_mode,
    }

    # ── Format and save output ───────────────────────────────────────────────
    output = format_output(
        codebook_results, figs_results, summary, datasets, settings,
        codebook_predictions, figs_predictions,
    )

    out_path = WORKSPACE / "method_out.json"
    out_path.write_text(json.dumps(output, indent=2, default=str))

    # Check file size
    file_size_mb = out_path.stat().st_size / (1024 * 1024)
    logger.info(f"Output file size: {file_size_mb:.1f} MB")

    t_total = time.time() - t_start
    logger.info(f"Total time: {t_total:.1f}s")
    logger.info(f"Summary: best overall config = {summary.get('overall_best_config', 'N/A')}")
    logger.info(f"Rankings: {summary.get('overall_mean_ranks', {})}")

    # Log key results
    for ds_name in dataset_names:
        best_info = summary["best_config_per_dataset"].get(ds_name, {})
        stab_info = summary["stability_summary"].get(ds_name, {})
        logger.info(
            f"  {ds_name}: best={best_info.get('best_config', '?')} "
            f"score={best_info.get('score', '?'):.4f} "
            f"stability={stab_info.get('mean_cosine_sim', '?')}"
        )


if __name__ == "__main__":
    main()
