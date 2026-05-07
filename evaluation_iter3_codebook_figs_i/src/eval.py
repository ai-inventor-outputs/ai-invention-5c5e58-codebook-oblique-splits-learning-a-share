#!/usr/bin/env python3
"""Codebook-FIGS Interpretability Evaluation.

Re-trains Codebook-FIGS models on 5 domain-semantic datasets to extract K=8
codebook directions, then computes 20 metrics across 7 groups:
  1. Direction Sparsity (L1 norm, Gini, top-3 concentration, n_active_features)
  2. Domain Alignment (hit rate against known domain risk factors)
  3. Usage Distribution (entropy, active codebook size)
  4. Complexity Comparison (vs FIGS baseline)
  5. Direction Diversity (pairwise cosine)
  6. Cross-Fold Semantic Stability (top-3 feature overlap, sign consistency)
  7. Human-Readable Summaries and Verdict
"""

from loguru import logger
from pathlib import Path
import json
import sys
import time
import resource
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from itertools import combinations

# ── logging ──────────────────────────────────────────────────────────────────
WORKSPACE = Path(__file__).resolve().parent
LOG_DIR = WORKSPACE / "logs"
LOG_DIR.mkdir(exist_ok=True)

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add(str(LOG_DIR / "run.log"), rotation="30 MB", level="DEBUG")

# ── resource limits ──────────────────────────────────────────────────────────
try:
    resource.setrlimit(resource.RLIMIT_AS, (20 * 1024**3, 20 * 1024**3))
    resource.setrlimit(resource.RLIMIT_CPU, (3600, 3600))
except Exception:
    logger.warning("Could not set resource limits")

# ── paths ────────────────────────────────────────────────────────────────────
DATA_DIR = WORKSPACE.parent.parent.parent / "iter_1" / "gen_art" / "data_id2_it1__opus"
EXP_DIR = WORKSPACE.parent.parent.parent / "iter_2" / "gen_art" / "exp_id3_it2__opus"
FULL_DATA_PATH = DATA_DIR / "full_data_out.json"
MINI_DATA_PATH = DATA_DIR / "mini_data_out.json"
FULL_METHOD_PATH = EXP_DIR / "full_method_out.json"

# ── target datasets (5 domain-semantic) ──────────────────────────────────────
TARGET_DATASETS = [
    "breast_cancer_wdbc",
    "heart_disease",
    "diabetes_pima",
    "auto_mpg",
    "california_housing",
]

# ── domain risk factors per dataset ──────────────────────────────────────────
DOMAIN_FACTORS: Dict[str, List[str]] = {
    "breast_cancer_wdbc": [
        "worst radius", "worst perimeter", "worst area",
        "worst concave points", "worst concavity",
        "mean radius", "mean perimeter", "mean area",
        "mean concave points", "mean concavity",
        "mean compactness", "worst compactness",
    ],
    "heart_disease": [
        "age", "trestbps", "chol", "thalach", "oldpeak",
        "ca", "cp_asympt", "exang_yes", "sex_male",
        "thal_reversable_defect", "thal_fixed_defect",
        "slope_flat", "slope_down",
    ],
    "diabetes_pima": [
        "plas", "mass", "age", "pedi", "insu", "preg", "pres",
    ],
    "auto_mpg": [
        "displacement", "horsepower", "weight", "acceleration",
        "cylinders_8", "cylinders_6", "cylinders_4",
        "origin_1", "origin_2", "origin_3",
    ],
    "california_housing": [
        "MedInc", "HouseAge", "AveRooms", "AveBedrms",
        "Population", "AveOccup", "Latitude", "Longitude",
    ],
}

K = 8
MAX_RULES = 12
N_FOLDS = 5
N_ALTERNATION_ROUNDS = 5
MIN_SAMPLES_LEAF = 5


# ============================================================
# DATA LOADING
# ============================================================

def load_datasets(data_path: Path, target_names: Optional[List[str]] = None) -> Dict[str, Dict]:
    """Load datasets from a data_out JSON file."""
    logger.info(f"Loading datasets from {data_path}")
    raw = json.loads(data_path.read_text())
    datasets: Dict[str, Dict] = {}

    for group in raw["datasets"]:
        name = group["dataset"]
        if name.endswith("_mini"):
            continue
        if target_names and name not in target_names:
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


def load_mini_datasets(data_path: Path, target_names: Optional[List[str]] = None) -> Dict[str, Dict]:
    """Load mini datasets from mini_data_out.json."""
    logger.info(f"Loading mini datasets from {data_path}")
    raw = json.loads(data_path.read_text())
    datasets: Dict[str, Dict] = {}

    for group in raw["datasets"]:
        name = group["dataset"]
        if not name.endswith("_mini"):
            continue
        base_name = name.replace("_mini", "")
        if target_names and base_name not in target_names:
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
# CODEBOOK-FIGS CORE (copied from method.py for re-training)
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


def init_codebook_pca(X_train: np.ndarray, K: int) -> np.ndarray:
    from sklearn.decomposition import PCA
    X_centered = X_train - X_train.mean(axis=0)
    K_actual = min(K, min(X_train.shape[0], X_train.shape[1]))
    pca = PCA(n_components=K_actual)
    pca.fit(X_centered)
    C = pca.components_.copy()
    if K_actual < K:
        rng = np.random.RandomState(42)
        extra = rng.randn(K - K_actual, X_train.shape[1])
        extra = extra / (np.linalg.norm(extra, axis=1, keepdims=True) + 1e-12)
        C = np.vstack([C, extra])
    norms = np.linalg.norm(C, axis=1, keepdims=True)
    C = C / np.maximum(norms, 1e-12)
    return C


def init_codebook_random(X_train: np.ndarray, K: int, random_state: int = 42) -> np.ndarray:
    rng = np.random.RandomState(random_state)
    C = rng.randn(K, X_train.shape[1])
    C = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-12)
    return C


def init_codebook_lda(X_train: np.ndarray, y_train: np.ndarray, K: int, task_type: str) -> np.ndarray:
    try:
        if task_type == "regression":
            n_bins = min(10, max(2, len(np.unique(y_train)) // 2))
            from sklearn.preprocessing import KBinsDiscretizer
            kbd = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="quantile")
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


def find_best_codebook_split(
    X: np.ndarray,
    y_residuals: np.ndarray,
    idxs: np.ndarray,
    codebook: np.ndarray,
    min_samples_leaf: int = 5,
    quantile_threshold: int = 5000,
) -> Tuple[Optional[int], Optional[float], float, Optional[np.ndarray], Optional[np.ndarray]]:
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
    K_val = codebook.shape[0]

    use_quantiles = n_node > quantile_threshold
    n_quantiles = 100

    for k in range(K_val):
        proj = X_node @ codebook[k]
        order = np.argsort(proj)
        sorted_proj = proj[order]
        sorted_y = y_node[order]

        cum_sum = np.cumsum(sorted_y)
        cum_sq_sum = np.cumsum(sorted_y ** 2)
        total_sum = cum_sum[-1]
        total_sq_sum = cum_sq_sum[-1]

        if use_quantiles:
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

    full_proj = X @ codebook[best_k]
    idxs_left = idxs & (full_proj <= best_threshold)
    idxs_right = idxs & (full_proj > best_threshold)

    return best_k, best_threshold, best_reduction, idxs_left, idxs_right


def effective_rank(C: np.ndarray, eps: float = 1e-12) -> float:
    s = np.linalg.svd(C, compute_uv=False)
    s = s[s > eps]
    if len(s) == 0:
        return 0.0
    p = s / np.sum(s)
    entropy = -np.sum(p * np.log(p + 1e-30))
    return float(np.exp(entropy))


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
            self._grow_trees(X_scaled, y)

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

            if round_idx < self.n_alternation_rounds - 1:
                if self.refine_strategy == "wls":
                    self._refine_codebook_wls(X_scaled, y)
                elif self.refine_strategy == "lbfgs":
                    self._refine_codebook_lbfgs(X_scaled, y)

                self._check_codebook_collapse()

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
        max_new_trees_per_round = 10

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

            if best_node.is_root:
                if self.max_trees is not None and len(self.trees_) >= self.max_trees:
                    continue
                if len(self.trees_) >= max_new_trees_per_round:
                    continue
                best_node.tree_num = len(self.trees_)
                self.trees_.append(best_node)
                y_pred_per_tree[best_node.tree_num] = np.full(n_samples, best_node.value)

            self.complexity_ += 1
            best_node.left = best_node.left_temp
            best_node.right = best_node.right_temp
            best_node.left.tree_num = best_node.tree_num
            best_node.right.tree_num = best_node.tree_num
            best_node.left_temp = None
            best_node.right_temp = None

            tree_root = self.trees_[best_node.tree_num]
            y_pred_per_tree[best_node.tree_num] = self._predict_tree(tree_root, X)

            for child in [best_node.left, best_node.right]:
                if child.n_samples < 2 * self.min_samples_leaf:
                    continue

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
        K_val = self.codebook_.shape[0]
        n_features = self.codebook_.shape[1]

        for k in range(K_val):
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

        K_val = self.codebook_.shape[0]
        n_features = self.codebook_.shape[1]
        msl = self.min_samples_leaf

        for k in range(K_val):
            nodes_k: List[Node] = []
            for tree in self.trees_:
                self._collect_nodes_with_codebook_idx(tree, k, nodes_k)

            if len(nodes_k) == 0:
                rng = np.random.RandomState(self.random_state + k)
                self.codebook_[k] = rng.randn(n_features)
                self.codebook_[k] /= np.linalg.norm(self.codebook_[k]) + 1e-12
                continue

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
        from sklearn.metrics.pairwise import cosine_similarity
        K_val = self.codebook_.shape[0]
        n_features = self.codebook_.shape[1]
        cos_sim = np.abs(cosine_similarity(self.codebook_))
        np.fill_diagonal(cos_sim, 0.0)
        n_collapses = 0
        for i in range(K_val):
            for j in range(i + 1, K_val):
                if cos_sim[i, j] > 0.99:
                    rng = np.random.RandomState(self.random_state + 100 + j)
                    self.codebook_[j] = rng.randn(n_features)
                    self.codebook_[j] /= np.linalg.norm(self.codebook_[j]) + 1e-12
                    n_collapses += 1
        return n_collapses

    def get_node_codebook_usage(self) -> Dict[int, int]:
        """Count how many split nodes use each codebook entry."""
        usage: Dict[int, int] = {}
        for tree in self.trees_:
            self._count_usage(tree, usage)
        return usage

    def _count_usage(self, node: Optional[Node], usage: Dict[int, int]) -> None:
        if node is None:
            return
        if node.left is not None and node.right is not None:
            idx = node.codebook_idx
            if idx is not None:
                usage[idx] = usage.get(idx, 0) + 1
            self._count_usage(node.left, usage)
            self._count_usage(node.right, usage)


# ============================================================
# HUNGARIAN ALIGNMENT (for cross-fold comparison)
# ============================================================

def align_codebooks_hungarian(
    C_ref: np.ndarray, C_target: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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


# ============================================================
# METRIC COMPUTATION FUNCTIONS
# ============================================================

def compute_sparsity_metrics(codebook: np.ndarray) -> Dict[str, Any]:
    """Group 1: Direction sparsity metrics for a codebook [K, d]."""
    K_val, d = codebook.shape
    results = {
        "per_direction_l1_norm": [],
        "per_direction_n_active": [],
        "per_direction_top3_concentration": [],
        "per_direction_gini": [],
    }

    for k in range(K_val):
        w = codebook[k]
        abs_w = np.abs(w)

        # L1 norm of unit-norm vector
        l1 = float(np.sum(abs_w))
        results["per_direction_l1_norm"].append(l1)

        # n_active_features (|weight| > 0.1)
        n_active = int(np.sum(abs_w > 0.1))
        results["per_direction_n_active"].append(n_active)

        # Top-3 concentration
        sorted_abs = np.sort(abs_w)[::-1]
        total_abs = np.sum(abs_w) + 1e-12
        top3_conc = float(np.sum(sorted_abs[:3]) / total_abs)
        results["per_direction_top3_concentration"].append(top3_conc)

        # Gini coefficient
        gini = _gini_coefficient(abs_w)
        results["per_direction_gini"].append(gini)

    # Averages
    results["mean_l1_norm"] = float(np.mean(results["per_direction_l1_norm"]))
    results["mean_n_active"] = float(np.mean(results["per_direction_n_active"]))
    results["mean_top3_concentration"] = float(np.mean(results["per_direction_top3_concentration"]))
    results["mean_gini"] = float(np.mean(results["per_direction_gini"]))

    return results


def _gini_coefficient(values: np.ndarray) -> float:
    """Compute Gini coefficient of a distribution."""
    v = np.sort(np.abs(values))
    n = len(v)
    if n == 0 or np.sum(v) < 1e-12:
        return 0.0
    index = np.arange(1, n + 1)
    return float((2.0 * np.sum(index * v) / (n * np.sum(v))) - (n + 1.0) / n)


def compute_domain_alignment(
    codebook: np.ndarray,
    feature_names: List[str],
    domain_factors: List[str],
) -> Dict[str, Any]:
    """Group 2: Domain alignment metrics."""
    K_val, d = codebook.shape
    # Map feature names to indices of domain factors
    domain_indices = set()
    for i, fname in enumerate(feature_names):
        if fname in domain_factors:
            domain_indices.add(i)

    results = {
        "per_direction_top3_hit_rate": [],
        "per_direction_top5_hit_rate": [],
        "per_direction_weighted_alignment": [],
    }

    for k in range(K_val):
        abs_w = np.abs(codebook[k])
        sorted_idx = np.argsort(abs_w)[::-1]

        # Top-3 hit rate
        top3_idx = sorted_idx[:3]
        hits_3 = sum(1 for i in top3_idx if i in domain_indices)
        results["per_direction_top3_hit_rate"].append(hits_3 / 3.0)

        # Top-5 hit rate
        top5_idx = sorted_idx[:5]
        hits_5 = sum(1 for i in top5_idx if i in domain_indices)
        results["per_direction_top5_hit_rate"].append(hits_5 / 5.0)

        # Weighted domain alignment
        total_weight = np.sum(abs_w) + 1e-12
        domain_weight = sum(abs_w[i] for i in range(d) if i in domain_indices)
        results["per_direction_weighted_alignment"].append(float(domain_weight / total_weight))

    results["mean_top3_hit_rate"] = float(np.mean(results["per_direction_top3_hit_rate"]))
    results["mean_top5_hit_rate"] = float(np.mean(results["per_direction_top5_hit_rate"]))
    results["mean_weighted_alignment"] = float(np.mean(results["per_direction_weighted_alignment"]))

    return results


def compute_usage_distribution(
    model: CodebookFIGS,
    K_val: int,
) -> Dict[str, Any]:
    """Group 3: Direction usage distribution."""
    usage = model.get_node_codebook_usage()

    # Fill in counts for all K entries
    counts = np.array([usage.get(k, 0) for k in range(K_val)], dtype=np.float64)
    total_splits = int(np.sum(counts))
    active_size = int(np.sum(counts > 0))

    # Usage entropy / uniformity
    if total_splits > 0 and active_size > 0:
        p = counts / np.sum(counts)
        p = p[p > 0]
        entropy = -np.sum(p * np.log2(p + 1e-30))
        max_entropy = np.log2(K_val)
        uniformity = float(entropy / max_entropy) if max_entropy > 0 else 0.0
    else:
        entropy = 0.0
        uniformity = 0.0

    return {
        "usage_counts": counts.tolist(),
        "total_splits": total_splits,
        "active_codebook_size": active_size,
        "usage_entropy": float(entropy),
        "usage_uniformity": uniformity,
    }


def compute_complexity_comparison(
    codebook_active_size: int,
    total_splits: int,
    erank: float,
    figs_unique_features: int,
) -> Dict[str, Any]:
    """Group 4: Interpretability complexity comparison."""
    compression_ratio = total_splits / max(codebook_active_size, 1)

    return {
        "figs_baseline_unique_features": figs_unique_features,
        "active_codebook_size": codebook_active_size,
        "compression_ratio": float(compression_ratio),
        "effective_rank": float(erank),
    }


def compute_direction_diversity(codebook: np.ndarray) -> Dict[str, Any]:
    """Group 5: Within-dataset direction diversity."""
    from sklearn.metrics.pairwise import cosine_similarity

    K_val = codebook.shape[0]
    if K_val < 2:
        return {
            "mean_pairwise_abs_cos": 0.0,
            "max_pairwise_abs_cos": 0.0,
            "diversity_score": 1.0,
        }

    cos_sim = cosine_similarity(codebook)
    abs_cos = np.abs(cos_sim)
    np.fill_diagonal(abs_cos, 0.0)

    # Extract upper triangle
    upper_idx = np.triu_indices(K_val, k=1)
    pairwise_abs = abs_cos[upper_idx]

    mean_abs = float(np.mean(pairwise_abs))
    max_abs = float(np.max(pairwise_abs))
    diversity = 1.0 - mean_abs

    # Flag near-redundant pairs
    redundant_pairs = []
    for i, j in zip(upper_idx[0], upper_idx[1]):
        if abs_cos[i, j] > 0.9:
            redundant_pairs.append((int(i), int(j), float(abs_cos[i, j])))

    return {
        "mean_pairwise_abs_cos": mean_abs,
        "max_pairwise_abs_cos": max_abs,
        "diversity_score": float(diversity),
        "n_redundant_pairs": len(redundant_pairs),
        "redundant_pairs": redundant_pairs,
    }


def compute_cross_fold_stability(
    codebooks: List[np.ndarray],
    feature_names: List[str],
) -> Dict[str, Any]:
    """Group 6: Cross-fold semantic stability."""
    if len(codebooks) < 2:
        K_val = codebooks[0].shape[0]
        return {
            "mean_top3_feature_overlap": 1.0,
            "mean_sign_consistency": 1.0,
            "mean_cosine_stability": 1.0,
            "per_direction_top3_overlap": [1.0] * K_val,
            "per_direction_sign_consistency": [1.0] * K_val,
            "per_direction_cosine_stability": [1.0] * K_val,
        }

    C_ref = codebooks[0]
    K_val = C_ref.shape[0]

    # Align all folds to fold 0
    aligned_codebooks = [C_ref]
    all_sims = []
    for fold_idx in range(1, len(codebooks)):
        C_aligned, _, sims = align_codebooks_hungarian(C_ref, codebooks[fold_idx])
        aligned_codebooks.append(C_aligned)
        all_sims.append(sims)

    # Per-direction metrics across all fold pairs
    per_dir_top3_overlaps = [[] for _ in range(K_val)]
    per_dir_sign_consistencies = [[] for _ in range(K_val)]
    per_dir_cosine_sims = [[] for _ in range(K_val)]

    n_folds = len(aligned_codebooks)
    for i_fold in range(n_folds):
        for j_fold in range(i_fold + 1, n_folds):
            for k in range(K_val):
                w_i = aligned_codebooks[i_fold][k]
                w_j = aligned_codebooks[j_fold][k]

                # Top-3 feature overlap
                top3_i = set(np.argsort(np.abs(w_i))[-3:])
                top3_j = set(np.argsort(np.abs(w_j))[-3:])
                overlap = len(top3_i & top3_j) / 3.0
                per_dir_top3_overlaps[k].append(overlap)

                # Sign consistency
                n_features = len(w_i)
                sign_match = np.sum(np.sign(w_i) == np.sign(w_j)) / n_features
                per_dir_sign_consistencies[k].append(float(sign_match))

                # Cosine similarity
                cos_sim = np.dot(w_i, w_j) / (np.linalg.norm(w_i) * np.linalg.norm(w_j) + 1e-12)
                per_dir_cosine_sims[k].append(float(np.abs(cos_sim)))

    # Average over fold pairs
    per_dir_overlap_mean = [float(np.mean(x)) if x else 0.0 for x in per_dir_top3_overlaps]
    per_dir_sign_mean = [float(np.mean(x)) if x else 0.0 for x in per_dir_sign_consistencies]
    per_dir_cosine_mean = [float(np.mean(x)) if x else 0.0 for x in per_dir_cosine_sims]

    return {
        "mean_top3_feature_overlap": float(np.mean(per_dir_overlap_mean)),
        "mean_sign_consistency": float(np.mean(per_dir_sign_mean)),
        "mean_cosine_stability": float(np.mean(per_dir_cosine_mean)),
        "per_direction_top3_overlap": per_dir_overlap_mean,
        "per_direction_sign_consistency": per_dir_sign_mean,
        "per_direction_cosine_stability": per_dir_cosine_mean,
    }


def generate_semantic_labels(
    codebook: np.ndarray,
    feature_names: List[str],
    top_n: int = 5,
) -> List[Dict[str, Any]]:
    """Group 7: Generate human-readable labels for each codebook direction."""
    K_val = codebook.shape[0]
    labels = []

    for k in range(K_val):
        w = codebook[k]
        abs_w = np.abs(w)
        sorted_idx = np.argsort(abs_w)[::-1][:top_n]

        top_features = []
        for idx in sorted_idx:
            fname = feature_names[idx] if idx < len(feature_names) else f"feature_{idx}"
            weight = float(w[idx])
            top_features.append({"feature": fname, "weight": round(weight, 4)})

        # Generate a label from the top 2-3 features
        label_parts = []
        for feat in top_features[:3]:
            sign = "+" if feat["weight"] > 0 else "-"
            label_parts.append(f"{sign}{abs(feat['weight']):.2f}*{feat['feature']}")
        label = " ".join(label_parts)

        # Attempt a semantic name
        semantic_name = _propose_semantic_name(top_features, feature_names)

        labels.append({
            "direction_idx": k,
            "semantic_name": semantic_name,
            "formula": label,
            "top_features": top_features,
        })

    return labels


def _propose_semantic_name(
    top_features: List[Dict[str, Any]],
    all_feature_names: List[str],
) -> str:
    """Propose a short semantic name for a codebook direction based on its top features."""
    feat_names = [f["feature"].lower() for f in top_features[:3]]
    joined = " ".join(feat_names)

    # Breast cancer patterns
    if any("radius" in f or "perimeter" in f or "area" in f for f in feat_names):
        return "Tumor Size"
    if any("concav" in f for f in feat_names):
        return "Tumor Shape/Concavity"
    if any("texture" in f for f in feat_names) and any("smooth" in f for f in feat_names):
        return "Tumor Texture"
    if any("compact" in f or "symmetry" in f for f in feat_names):
        return "Tumor Morphology"

    # Heart disease patterns
    if any("thalach" in f or "exang" in f for f in feat_names):
        return "Exercise Cardiac Response"
    if any("trestbps" in f or "chol" in f for f in feat_names):
        return "Cardiovascular Risk"
    if any("oldpeak" in f or "slope" in f for f in feat_names):
        return "ST Segment Pattern"
    if any("cp_" in f for f in feat_names):
        return "Chest Pain Type"
    if any("thal_" in f for f in feat_names):
        return "Thalassemia Type"

    # Diabetes patterns
    if any("plas" in f or "glucose" in f for f in feat_names):
        return "Glucose/Metabolic"
    if any("mass" in f or "bmi" in f for f in feat_names):
        return "Body Composition"
    if any("pedi" in f for f in feat_names):
        return "Genetic Predisposition"
    if any("insu" in f for f in feat_names):
        return "Insulin Response"

    # Auto MPG patterns
    if any("displacement" in f or "horsepower" in f or "weight" in f for f in feat_names):
        return "Engine Power/Weight"
    if any("cylinder" in f for f in feat_names):
        return "Engine Configuration"
    if any("origin" in f for f in feat_names):
        return "Vehicle Origin"
    if any("model" in f for f in feat_names):
        return "Model Year"
    if any("acceleration" in f for f in feat_names):
        return "Vehicle Performance"

    # California housing patterns
    if any("medinc" in f for f in feat_names):
        return "Income Level"
    if any("latitude" in f or "longitude" in f for f in feat_names):
        return "Geographic Location"
    if any("averooms" in f or "avebedrms" in f for f in feat_names):
        return "Housing Size"
    if any("population" in f or "aveoccup" in f for f in feat_names):
        return "Population Density"
    if any("houseage" in f for f in feat_names):
        return "Housing Age"

    # Fallback: use top feature name
    return f"Direction ({top_features[0]['feature']})"


def compute_interpretability_verdict(
    sparsity: Dict, domain_align: Dict, usage: Dict, stability: Dict,
) -> Dict[str, str]:
    """Synthesize metrics into an overall interpretability verdict."""
    # Sparsity assessment
    if sparsity["mean_top3_concentration"] > 0.7 and sparsity["mean_gini"] > 0.6:
        sparsity_level = "high"
    elif sparsity["mean_top3_concentration"] > 0.5 and sparsity["mean_gini"] > 0.4:
        sparsity_level = "medium"
    else:
        sparsity_level = "low"

    # Domain alignment assessment
    if domain_align["mean_top3_hit_rate"] > 0.6:
        alignment_level = "high"
    elif domain_align["mean_top3_hit_rate"] > 0.3:
        alignment_level = "medium"
    else:
        alignment_level = "low"

    # Utilization assessment
    if usage["usage_uniformity"] > 0.7 and usage["active_codebook_size"] >= 6:
        utilization_level = "high"
    elif usage["usage_uniformity"] > 0.4 and usage["active_codebook_size"] >= 4:
        utilization_level = "medium"
    else:
        utilization_level = "low"

    # Cross-fold reliability assessment
    if stability["mean_top3_feature_overlap"] > 0.6 and stability["mean_cosine_stability"] > 0.7:
        reliability_level = "high"
    elif stability["mean_top3_feature_overlap"] > 0.3 and stability["mean_cosine_stability"] > 0.5:
        reliability_level = "medium"
    else:
        reliability_level = "low"

    return {
        "sparsity": sparsity_level,
        "domain_alignment": alignment_level,
        "utilization": utilization_level,
        "cross_fold_reliability": reliability_level,
    }


# ============================================================
# FIGS BASELINE FEATURE COUNT
# ============================================================

def count_figs_unique_features(
    X_train: np.ndarray,
    y_train: np.ndarray,
    task_type: str,
    max_rules: int = 12,
) -> int:
    """Train axis-aligned FIGS and count unique features used across all splits."""
    try:
        # Need enough samples for FIGS to work
        if X_train.shape[0] < 10:
            logger.debug(f"Too few samples ({X_train.shape[0]}) for FIGS feature count")
            return -1

        from imodels import FIGSClassifier, FIGSRegressor

        if task_type == "classification":
            figs_model = FIGSClassifier(max_rules=max_rules)
        else:
            figs_model = FIGSRegressor(max_rules=max_rules)

        figs_model.fit(X_train, y_train)

        # Collect split features from the internal tree structure
        unique_features = set()
        if hasattr(figs_model, 'trees_'):
            for tree in figs_model.trees_:
                _collect_figs_features(tree, unique_features)

        return max(len(unique_features), 1)  # At least 1 if FIGS trained
    except Exception:
        logger.warning("FIGS feature count failed (expected with small datasets)")
        return -1


def _collect_figs_features(node: Any, features: set) -> None:
    """Recursively collect feature indices from a FIGS tree."""
    if node is None:
        return
    if hasattr(node, 'feature'):
        feat = node.feature
        if feat is not None and feat >= 0:
            features.add(int(feat))
    # imodels uses left and right attributes
    if hasattr(node, 'left'):
        _collect_figs_features(node.left, features)
    if hasattr(node, 'right'):
        _collect_figs_features(node.right, features)
    # Also check child nodes stored as list
    if hasattr(node, 'children_left') and hasattr(node, 'children_right'):
        pass
    if hasattr(node, 'left_child'):
        _collect_figs_features(node.left_child, features)
    if hasattr(node, 'right_child'):
        _collect_figs_features(node.right_child, features)


def count_figs_unique_features_from_model(figs_model: Any) -> int:
    """Count unique features from a fitted FIGS model."""
    unique_features = set()
    if hasattr(figs_model, 'trees_'):
        for tree in figs_model.trees_:
            _collect_figs_tree_features(tree, unique_features)
    return len(unique_features)


def _collect_figs_tree_features(node: Any, features: set) -> None:
    """Walk the FIGS tree (sklearn-style) and collect feature indices."""
    if node is None:
        return
    # imodels FIGSRegressor/Classifier stores trees as sklearn DecisionTreeRegressor/Classifier
    # that have tree_ attribute with feature array
    if hasattr(node, 'tree_'):
        tree_struct = node.tree_
        for feat_idx in tree_struct.feature:
            if feat_idx >= 0:
                features.add(int(feat_idx))
    # Also try walking recursively if it's a custom node
    if hasattr(node, 'feature'):
        feat = getattr(node, 'feature', None)
        if feat is not None and isinstance(feat, (int, np.integer)) and feat >= 0:
            features.add(int(feat))
    if hasattr(node, 'left'):
        _collect_figs_tree_features(getattr(node, 'left', None), features)
    if hasattr(node, 'right'):
        _collect_figs_tree_features(getattr(node, 'right', None), features)
    if hasattr(node, 'left_child'):
        _collect_figs_tree_features(getattr(node, 'left_child', None), features)
    if hasattr(node, 'right_child'):
        _collect_figs_tree_features(getattr(node, 'right_child', None), features)


# ============================================================
# MAIN EVALUATION PIPELINE
# ============================================================

def train_and_evaluate_dataset(
    ds_name: str,
    ds_data: Dict,
    init_strategy: str = "random",
    refine_strategy: str = "wls",
    experiment_stability: Optional[float] = None,
) -> Dict[str, Any]:
    """Train Codebook-FIGS on a single dataset and compute all interpretability metrics."""
    logger.info(f"=== Evaluating {ds_name} ===")
    t0 = time.time()

    X = ds_data["X"]
    y = ds_data["y"]
    folds = ds_data["folds"]
    task_type = ds_data["task_type"]
    feature_names = ds_data["feature_names"]

    unique_folds = sorted(np.unique(folds))
    actual_n_folds = min(N_FOLDS, len(unique_folds))

    fold_codebooks: List[np.ndarray] = []
    fold_models: List[CodebookFIGS] = []
    fold_usage_dicts: List[Dict[str, Any]] = []
    fold_eranks: List[float] = []
    fold_figs_unique_features: List[int] = []

    for fold_id in range(actual_n_folds):
        test_mask = folds == unique_folds[fold_id]
        train_mask = ~test_mask
        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]

        logger.info(f"  Fold {fold_id}: train={X_train.shape[0]}, test={X_test.shape[0]}")

        try:
            model = CodebookFIGS(
                K=K,
                max_rules=MAX_RULES,
                n_alternation_rounds=N_ALTERNATION_ROUNDS,
                init_strategy=init_strategy,
                refine_strategy=refine_strategy,
                random_state=42 + fold_id,
                min_samples_leaf=MIN_SAMPLES_LEAF,
            )
            model.fit(X_train, y_train, task_type=task_type)

            fold_codebooks.append(model.codebook_.copy())
            fold_models.append(model)

            # Usage distribution for this fold
            usage_info = compute_usage_distribution(model, K)
            fold_usage_dicts.append(usage_info)

            # Effective rank
            erank = effective_rank(model.codebook_)
            fold_eranks.append(erank)

            # FIGS baseline unique features for this fold
            figs_uf = count_figs_unique_features(X_train, y_train, task_type, MAX_RULES)
            fold_figs_unique_features.append(figs_uf)

        except Exception:
            logger.exception(f"Failed on fold {fold_id}")
            continue

    if not fold_codebooks:
        logger.error(f"All folds failed for {ds_name}")
        return {"error": "all folds failed"}

    # Use fold 0 codebook for primary analysis (as specified in artifact plan)
    primary_codebook = fold_codebooks[0]
    primary_model = fold_models[0]

    # ── Group 1: Direction Sparsity ──────────────────────────────────────
    sparsity = compute_sparsity_metrics(primary_codebook)
    logger.info(f"  Sparsity: mean_top3_conc={sparsity['mean_top3_concentration']:.3f}, "
                f"mean_gini={sparsity['mean_gini']:.3f}")

    # ── Group 2: Domain Alignment ────────────────────────────────────────
    domain_factors = DOMAIN_FACTORS.get(ds_name, [])
    if domain_factors and feature_names:
        domain_align = compute_domain_alignment(primary_codebook, feature_names, domain_factors)
    else:
        domain_align = {
            "per_direction_top3_hit_rate": [],
            "per_direction_top5_hit_rate": [],
            "per_direction_weighted_alignment": [],
            "mean_top3_hit_rate": 0.0,
            "mean_top5_hit_rate": 0.0,
            "mean_weighted_alignment": 0.0,
        }
    logger.info(f"  Domain align: mean_top3_hit={domain_align['mean_top3_hit_rate']:.3f}")

    # ── Group 3: Usage Distribution (average across folds) ───────────────
    if fold_usage_dicts:
        avg_active_size = float(np.mean([u["active_codebook_size"] for u in fold_usage_dicts]))
        avg_uniformity = float(np.mean([u["usage_uniformity"] for u in fold_usage_dicts]))
        avg_entropy = float(np.mean([u["usage_entropy"] for u in fold_usage_dicts]))
        avg_total_splits = float(np.mean([u["total_splits"] for u in fold_usage_dicts]))
        # Use fold 0 usage counts for display
        usage_info = fold_usage_dicts[0]
        usage_info["avg_active_codebook_size"] = avg_active_size
        usage_info["avg_usage_uniformity"] = avg_uniformity
        usage_info["avg_usage_entropy"] = avg_entropy
        usage_info["avg_total_splits"] = avg_total_splits
    else:
        usage_info = {
            "usage_counts": [0] * K,
            "total_splits": 0,
            "active_codebook_size": 0,
            "usage_entropy": 0.0,
            "usage_uniformity": 0.0,
            "avg_active_codebook_size": 0.0,
            "avg_usage_uniformity": 0.0,
            "avg_usage_entropy": 0.0,
            "avg_total_splits": 0.0,
        }
    logger.info(f"  Usage: active_size={usage_info['active_codebook_size']}, "
                f"uniformity={usage_info.get('avg_usage_uniformity', 0):.3f}")

    # ── Group 4: Complexity Comparison ───────────────────────────────────
    valid_figs_uf = [f for f in fold_figs_unique_features if f > 0]
    figs_uf_mean = int(np.mean(valid_figs_uf)) if valid_figs_uf else 0
    avg_erank = float(np.mean(fold_eranks)) if fold_eranks else 0.0
    complexity = compute_complexity_comparison(
        codebook_active_size=usage_info["active_codebook_size"],
        total_splits=usage_info["total_splits"],
        erank=avg_erank,
        figs_unique_features=figs_uf_mean,
    )
    logger.info(f"  Complexity: FIGS features={figs_uf_mean}, "
                f"codebook active={usage_info['active_codebook_size']}, "
                f"erank={avg_erank:.2f}")

    # ── Group 5: Direction Diversity ─────────────────────────────────────
    diversity = compute_direction_diversity(primary_codebook)
    logger.info(f"  Diversity: mean_abs_cos={diversity['mean_pairwise_abs_cos']:.3f}, "
                f"max_abs_cos={diversity['max_pairwise_abs_cos']:.3f}")

    # ── Group 6: Cross-Fold Stability ────────────────────────────────────
    stability = compute_cross_fold_stability(fold_codebooks, feature_names)
    logger.info(f"  Stability: top3_overlap={stability['mean_top3_feature_overlap']:.3f}, "
                f"cosine_stab={stability['mean_cosine_stability']:.3f}")

    # ── Group 7: Semantic Labels & Verdict ───────────────────────────────
    labels = generate_semantic_labels(primary_codebook, feature_names)
    verdict = compute_interpretability_verdict(sparsity, domain_align, usage_info, stability)
    logger.info(f"  Verdict: {verdict}")

    elapsed = time.time() - t0
    logger.info(f"  Completed {ds_name} in {elapsed:.1f}s")

    return {
        "dataset": ds_name,
        "task_type": task_type,
        "n_features": int(X.shape[1]),
        "n_samples": int(X.shape[0]),
        "n_folds_trained": len(fold_codebooks),
        "config": f"{init_strategy}_{refine_strategy}",
        "sparsity": sparsity,
        "domain_alignment": domain_align,
        "usage_distribution": usage_info,
        "complexity_comparison": complexity,
        "direction_diversity": diversity,
        "cross_fold_stability": stability,
        "semantic_labels": labels,
        "verdict": verdict,
        "experiment_cosine_stability": experiment_stability,
        "time_seconds": round(elapsed, 2),
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


def load_experiment_results() -> Dict[str, Any]:
    """Load pre-computed experiment results for stability/erank data."""
    try:
        logger.info(f"Loading experiment results from {FULL_METHOD_PATH}")
        raw = json.loads(FULL_METHOD_PATH.read_text())
        return raw.get("metadata", {})
    except Exception:
        logger.exception("Failed to load experiment results")
        return {}


def format_eval_output(
    all_results: Dict[str, Dict],
    datasets: Dict[str, Dict],
    experiment_meta: Dict[str, Any],
) -> Dict[str, Any]:
    """Format evaluation output according to exp_eval_sol_out.json schema."""

    # Compute aggregate metrics across all datasets
    metrics_agg: Dict[str, float] = {}

    # Average sparsity
    sparsity_top3 = [r["sparsity"]["mean_top3_concentration"] for r in all_results.values() if "sparsity" in r]
    sparsity_gini = [r["sparsity"]["mean_gini"] for r in all_results.values() if "sparsity" in r]
    sparsity_l1 = [r["sparsity"]["mean_l1_norm"] for r in all_results.values() if "sparsity" in r]
    sparsity_nactive = [r["sparsity"]["mean_n_active"] for r in all_results.values() if "sparsity" in r]

    if sparsity_top3:
        metrics_agg["mean_top3_concentration"] = float(np.mean(sparsity_top3))
    if sparsity_gini:
        metrics_agg["mean_gini_coefficient"] = float(np.mean(sparsity_gini))
    if sparsity_l1:
        metrics_agg["mean_l1_norm"] = float(np.mean(sparsity_l1))
    if sparsity_nactive:
        metrics_agg["mean_n_active_features"] = float(np.mean(sparsity_nactive))

    # Average domain alignment
    align_top3 = [r["domain_alignment"]["mean_top3_hit_rate"] for r in all_results.values() if "domain_alignment" in r]
    align_top5 = [r["domain_alignment"]["mean_top5_hit_rate"] for r in all_results.values() if "domain_alignment" in r]
    align_weighted = [r["domain_alignment"]["mean_weighted_alignment"] for r in all_results.values() if "domain_alignment" in r]

    if align_top3:
        metrics_agg["mean_domain_top3_hit_rate"] = float(np.mean(align_top3))
    if align_top5:
        metrics_agg["mean_domain_top5_hit_rate"] = float(np.mean(align_top5))
    if align_weighted:
        metrics_agg["mean_domain_weighted_alignment"] = float(np.mean(align_weighted))

    # Average usage
    active_sizes = [r["usage_distribution"].get("avg_active_codebook_size", r["usage_distribution"]["active_codebook_size"]) for r in all_results.values() if "usage_distribution" in r]
    uniformities = [r["usage_distribution"].get("avg_usage_uniformity", r["usage_distribution"]["usage_uniformity"]) for r in all_results.values() if "usage_distribution" in r]

    if active_sizes:
        metrics_agg["mean_active_codebook_size"] = float(np.mean(active_sizes))
    if uniformities:
        metrics_agg["mean_usage_uniformity"] = float(np.mean(uniformities))

    # Average complexity
    compression_ratios = [r["complexity_comparison"]["compression_ratio"] for r in all_results.values() if "complexity_comparison" in r]
    eranks = [r["complexity_comparison"]["effective_rank"] for r in all_results.values() if "complexity_comparison" in r]
    figs_feats = [r["complexity_comparison"]["figs_baseline_unique_features"] for r in all_results.values() if "complexity_comparison" in r and r["complexity_comparison"]["figs_baseline_unique_features"] > 0]

    if compression_ratios:
        metrics_agg["mean_compression_ratio"] = float(np.mean(compression_ratios))
    if eranks:
        metrics_agg["mean_effective_rank"] = float(np.mean(eranks))
    if figs_feats:
        metrics_agg["mean_figs_unique_features"] = float(np.mean(figs_feats))

    # Average diversity
    div_mean = [r["direction_diversity"]["mean_pairwise_abs_cos"] for r in all_results.values() if "direction_diversity" in r]
    div_score = [r["direction_diversity"]["diversity_score"] for r in all_results.values() if "direction_diversity" in r]

    if div_mean:
        metrics_agg["mean_pairwise_abs_cosine"] = float(np.mean(div_mean))
    if div_score:
        metrics_agg["mean_diversity_score"] = float(np.mean(div_score))

    # Average stability
    stab_overlap = [r["cross_fold_stability"]["mean_top3_feature_overlap"] for r in all_results.values() if "cross_fold_stability" in r]
    stab_sign = [r["cross_fold_stability"]["mean_sign_consistency"] for r in all_results.values() if "cross_fold_stability" in r]
    stab_cosine = [r["cross_fold_stability"]["mean_cosine_stability"] for r in all_results.values() if "cross_fold_stability" in r]

    if stab_overlap:
        metrics_agg["mean_cross_fold_top3_overlap"] = float(np.mean(stab_overlap))
    if stab_sign:
        metrics_agg["mean_cross_fold_sign_consistency"] = float(np.mean(stab_sign))
    if stab_cosine:
        metrics_agg["mean_cross_fold_cosine_stability"] = float(np.mean(stab_cosine))

    # Verdict counts
    verdicts = [r.get("verdict", {}) for r in all_results.values() if "verdict" in r]
    for axis in ["sparsity", "domain_alignment", "utilization", "cross_fold_reliability"]:
        counts = {"high": 0, "medium": 0, "low": 0}
        for v in verdicts:
            level = v.get(axis, "low")
            counts[level] = counts.get(level, 0) + 1
        metrics_agg[f"verdict_{axis}_high_count"] = float(counts["high"])
        metrics_agg[f"verdict_{axis}_medium_count"] = float(counts["medium"])
        metrics_agg[f"verdict_{axis}_low_count"] = float(counts["low"])

    # Number of datasets evaluated
    metrics_agg["n_datasets_evaluated"] = float(len(all_results))

    # Format per-dataset output (examples)
    output_datasets: List[Dict[str, Any]] = []

    for ds_name in sorted(all_results.keys()):
        ds_result = all_results[ds_name]
        ds_data = datasets.get(ds_name)
        if ds_data is None:
            continue

        X = ds_data["X"]
        y = ds_data["y"]
        folds = ds_data["folds"]
        task_type = ds_data["task_type"]
        feature_names = ds_data["feature_names"]
        n_samples = X.shape[0]

        examples: List[Dict[str, Any]] = []

        for i in range(n_samples):
            ex: Dict[str, Any] = {
                "input": json.dumps(X[i].tolist()),
                "output": str(y[i]),
                "metadata_fold": int(folds[i]),
                "metadata_task_type": task_type,
            }

            # Per-example evaluation scores (all same for this dataset-level eval)
            ex["eval_sparsity_top3_concentration"] = ds_result["sparsity"]["mean_top3_concentration"]
            ex["eval_domain_top3_hit_rate"] = ds_result["domain_alignment"]["mean_top3_hit_rate"]
            ex["eval_usage_uniformity"] = ds_result["usage_distribution"].get("avg_usage_uniformity",
                                                                               ds_result["usage_distribution"]["usage_uniformity"])
            ex["eval_diversity_score"] = ds_result["direction_diversity"]["diversity_score"]
            ex["eval_cross_fold_top3_overlap"] = ds_result["cross_fold_stability"]["mean_top3_feature_overlap"]
            ex["eval_cross_fold_cosine_stability"] = ds_result["cross_fold_stability"]["mean_cosine_stability"]

            # Add first-example metadata with full results
            if i == 0:
                ex["metadata_n_samples"] = n_samples
                ex["metadata_n_features"] = int(X.shape[1])
                ex["metadata_feature_names"] = feature_names

            examples.append(ex)

        output_datasets.append({
            "dataset": ds_name,
            "examples": examples,
        })

    # Build metadata
    metadata = {
        "evaluation": "codebook_figs_interpretability",
        "description": "Evaluates Codebook-FIGS interpretability via 20 metrics across 7 groups",
        "target_datasets": sorted(all_results.keys()),
        "K": K,
        "max_rules": MAX_RULES,
        "n_folds": N_FOLDS,
        "best_config_used": "random_wls",
        "per_dataset_results": _sanitize_for_json(all_results),
        "experiment_summary": _sanitize_for_json(experiment_meta.get("summary", {})),
    }

    return {
        "metadata": metadata,
        "metrics_agg": metrics_agg,
        "datasets": output_datasets,
    }


# ============================================================
# MAIN
# ============================================================

@logger.catch
def main() -> None:
    t_start = time.time()

    mini_mode = "--mini" in sys.argv
    max_examples = None
    for arg in sys.argv:
        if arg.startswith("--max-examples="):
            max_examples = int(arg.split("=")[1])

    # Load experiment metadata (for pre-computed stability etc.)
    experiment_meta = load_experiment_results()
    exp_stability = experiment_meta.get("codebook_figs_results", {})
    exp_summary = experiment_meta.get("summary", {})
    best_config = exp_summary.get("overall_best_config", "random_wls")
    init_strat, refine_strat = best_config.split("_", 1)
    logger.info(f"Using best config from experiment: {best_config} ({init_strat}, {refine_strat})")

    # Load data
    if mini_mode:
        logger.info("=== MINI MODE ===")
        datasets = load_mini_datasets(MINI_DATA_PATH, TARGET_DATASETS)
    else:
        logger.info("=== FULL MODE ===")
        datasets = load_datasets(FULL_DATA_PATH, TARGET_DATASETS)

    if max_examples:
        # Limit examples for testing
        for ds_name, ds_data in datasets.items():
            n = min(max_examples, ds_data["X"].shape[0])
            ds_data["X"] = ds_data["X"][:n]
            ds_data["y"] = ds_data["y"][:n]
            ds_data["folds"] = ds_data["folds"][:n]
            logger.info(f"  Limited {ds_name} to {n} examples")

    dataset_names = sorted(datasets.keys())
    logger.info(f"Target datasets ({len(dataset_names)}): {dataset_names}")

    # Get per-dataset experiment stability
    exp_stab_per_dataset: Dict[str, Optional[float]] = {}
    for ds_name in dataset_names:
        stab_summary = exp_summary.get("stability_summary", {}).get(ds_name, {})
        exp_stab_per_dataset[ds_name] = stab_summary.get("mean_cosine_sim")

    # Evaluate each dataset
    all_results: Dict[str, Dict] = {}
    for ds_name in dataset_names:
        try:
            result = train_and_evaluate_dataset(
                ds_name=ds_name,
                ds_data=datasets[ds_name],
                init_strategy=init_strat,
                refine_strategy=refine_strat,
                experiment_stability=exp_stab_per_dataset.get(ds_name),
            )
            all_results[ds_name] = result
        except Exception:
            logger.exception(f"Failed to evaluate {ds_name}")
            continue

    # Format output
    output = format_eval_output(all_results, datasets, experiment_meta)

    # Save output
    out_path = WORKSPACE / "eval_out.json"
    out_path.write_text(json.dumps(_sanitize_for_json(output), indent=2, default=str))

    file_size_mb = out_path.stat().st_size / (1024 * 1024)
    logger.info(f"Output file size: {file_size_mb:.1f} MB")

    t_total = time.time() - t_start
    logger.info(f"Total evaluation time: {t_total:.1f}s")

    # Log summary
    logger.info("=== EVALUATION SUMMARY ===")
    for ds_name, result in all_results.items():
        verdict = result.get("verdict", {})
        logger.info(f"  {ds_name}: sparsity={verdict.get('sparsity','?')}, "
                     f"domain_align={verdict.get('domain_alignment','?')}, "
                     f"utilization={verdict.get('utilization','?')}, "
                     f"reliability={verdict.get('cross_fold_reliability','?')}")
    logger.info(f"Aggregate metrics: {json.dumps({k: round(v, 4) for k, v in output['metrics_agg'].items()}, indent=2)}")


if __name__ == "__main__":
    main()
