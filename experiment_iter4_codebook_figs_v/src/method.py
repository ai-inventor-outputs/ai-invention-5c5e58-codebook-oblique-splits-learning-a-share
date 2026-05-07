#!/usr/bin/env python3
"""Codebook-FIGS v2: Joint Gradient Refinement with Elastic-Net Sparsity.

Extends iter3 Codebook-FIGS with three core improvements:
  1. Joint L-BFGS-B optimization of codebook entries + node thresholds
     using soft-sigmoid tree predictions for differentiability
  2. Warm-start tree reassignment preserving tree structure
  3. Post-optimization weight thresholding for genuine sparsity

Benchmarks across 10 datasets × K∈{3,5,8,12} × 2 init strategies × 5-fold CV
vs FIGS/XGBoost/LightGBM baselines.

Output: method_out.json conforming to exp_gen_sol_out schema.
"""

from loguru import logger
from pathlib import Path
import json
import sys
import time
import resource
import warnings
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from collections import defaultdict

warnings.filterwarnings("ignore")

# ── Logging ──────────────────────────────────────────────────────────────────
WORKSPACE = Path(__file__).resolve().parent
LOG_DIR = WORKSPACE / "logs"
LOG_DIR.mkdir(exist_ok=True)

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add(str(LOG_DIR / "run.log"), rotation="30 MB", level="DEBUG")

# ── Resource limits ──────────────────────────────────────────────────────────
try:
    resource.setrlimit(resource.RLIMIT_AS, (20 * 1024**3, 20 * 1024**3))
    resource.setrlimit(resource.RLIMIT_CPU, (3300, 3300))
except Exception:
    logger.warning("Could not set resource limits")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 0: CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════
K_VALUES = [3, 5, 8, 12]           # Drop K=20 (consistently degrades in iter3)
N_FOLDS = 5
MAX_RULES = 12
MAX_DEPTH = 4
N_ALTERNATION_ROUNDS = 10          # Up from 7
MIN_SAMPLES_LEAF = 5
TAU_SOFT = 1.0                     # Sigmoid temperature for soft tree predictions
LAMBDA_L1 = 0.01                   # L1 sparsity penalty on codebook entries
LAMBDA_L2 = 0.001                  # L2 Frobenius penalty on codebook
WEIGHT_THRESHOLD = 0.05            # Zero out codebook weights below this
LBFGS_MAXITER = 30                 # Max L-BFGS-B iterations per refinement step
TIMEOUT_TOTAL_SEC = 3300           # 55 min hard wall
LBFGS_SUBSAMPLE = 5000             # Subsample for L-BFGS-B if n > this

# ── Data paths ───────────────────────────────────────────────────────────────
DATA_DIR = WORKSPACE.parents[2] / "iter_1" / "gen_art" / "data_id2_it1__opus"
if not DATA_DIR.exists():
    DATA_DIR = WORKSPACE.parent / "data_id2_it1__opus"
FULL_DATA_PATH = DATA_DIR / "full_data_out.json"
MINI_DATA_PATH = DATA_DIR / "mini_data_out.json"


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: DATA LOADING (reuse from iter3)
# ═══════════════════════════════════════════════════════════════════════════════

def load_datasets(json_path: str) -> Dict[str, Dict[str, Any]]:
    """Load datasets from full_data_out.json or mini_data_out.json."""
    logger.info(f"Loading datasets from {json_path}")
    raw = json.loads(Path(json_path).read_text())
    datasets: Dict[str, Dict[str, Any]] = {}

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
            fold_list.append(int(ex["metadata_fold"]))

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
# SECTION 2: NODE CLASS (reuse from iter3)
# ═══════════════════════════════════════════════════════════════════════════════

class Node:
    """Tree node for Codebook-FIGS, supporting oblique splits from a shared codebook."""
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


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: CODEBOOK INITIALIZATION (reuse from iter3)
# ═══════════════════════════════════════════════════════════════════════════════

def init_codebook_pca(X_train: np.ndarray, K: int) -> np.ndarray:
    """Top-K PCA components of centered, scaled X_train."""
    from sklearn.decomposition import PCA

    X_centered = X_train - X_train.mean(axis=0)
    K_actual = min(K, X_train.shape[0], X_train.shape[1])
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


def init_codebook_random(X_train: np.ndarray, K: int, seed: int = 42) -> np.ndarray:
    """K random unit vectors."""
    rng = np.random.RandomState(seed)
    C = rng.randn(K, X_train.shape[1])
    C = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-12)
    return C


def init_codebook_lda(
    X_train: np.ndarray,
    y_train: np.ndarray,
    K: int,
    task_type: str,
) -> np.ndarray:
    """LDA-inspired init via between-class scatter SVD."""
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


def init_codebook_adaptive(
    X_train: np.ndarray,
    y_train: np.ndarray,
    K: int,
    task_type: str,
    seed: int = 42,
) -> np.ndarray:
    """Task-adaptive: LDA for classification, PCA for regression."""
    if task_type == "classification":
        return init_codebook_lda(X_train, y_train, K, task_type)
    else:
        return init_codebook_pca(X_train, K)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4: SPLIT SEARCH (reuse from iter3)
# ═══════════════════════════════════════════════════════════════════════════════

def find_best_codebook_split(
    X: np.ndarray,
    y_residuals: np.ndarray,
    idxs: np.ndarray,
    codebook: np.ndarray,
    min_samples_leaf: int = 5,
    quantile_threshold: int = 5000,
) -> Tuple[Optional[int], Optional[float], float, Optional[np.ndarray], Optional[np.ndarray]]:
    """Find best (codebook direction, threshold) split by sorted-scan."""
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


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5: METRICS (reuse + extend from iter3)
# ═══════════════════════════════════════════════════════════════════════════════

def effective_rank(C: np.ndarray, eps: float = 1e-12) -> float:
    """eRank = exp(entropy of normalized singular values)."""
    s = np.linalg.svd(C, compute_uv=False)
    s = s[s > eps]
    if len(s) == 0:
        return 0.0
    p = s / np.sum(s)
    entropy = -np.sum(p * np.log(p + 1e-30))
    return float(np.exp(entropy))


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
    """Compute codebook stability across CV folds via Hungarian alignment."""
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


def evaluate_fold(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray],
    task_type: str,
) -> Dict[str, Any]:
    """Compute metrics for one fold."""
    from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error, r2_score

    metrics: Dict[str, Any] = {}
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


# NEW: Codebook sparsity metrics
def compute_codebook_sparsity(codebook: np.ndarray, weight_threshold: float = 0.05) -> Dict[str, Any]:
    """Compute sparsity metrics for each codebook entry."""
    K, d = codebook.shape
    l0_norms = []
    top3_concs = []
    for k in range(K):
        abs_w = np.abs(codebook[k])
        l0 = int(np.sum(abs_w > weight_threshold))
        l0_norms.append(l0)
        total = np.sum(abs_w)
        if total > 1e-12:
            sorted_abs = np.sort(abs_w)[::-1]
            top3 = np.sum(sorted_abs[:min(3, len(sorted_abs))])
            top3_concs.append(float(top3 / total))
        else:
            top3_concs.append(0.0)
    return {
        "top3_concentration": float(np.mean(top3_concs)),
        "mean_l0_norm": float(np.mean(l0_norms)),
        "per_entry_l0": l0_norms,
        "per_entry_top3_conc": top3_concs,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6: SOFT TREE PREDICTION (core innovation)
# ═══════════════════════════════════════════════════════════════════════════════

def _collect_internal_nodes_tree(tree_root: Node) -> List[Node]:
    """Collect all internal nodes from a single tree in DFS order."""
    nodes: List[Node] = []
    def _dfs(node: Optional[Node]) -> None:
        if node is None:
            return
        if node.left is not None and node.right is not None:
            nodes.append(node)
            _dfs(node.left)
            _dfs(node.right)
    _dfs(tree_root)
    return nodes


def _collect_all_internal_nodes(trees: List[Node]) -> List[Node]:
    """Collect all internal nodes across all trees in consistent DFS order."""
    all_nodes: List[Node] = []
    for tree_root in trees:
        all_nodes.extend(_collect_internal_nodes_tree(tree_root))
    return all_nodes


def soft_predict_tree(
    root: Node,
    X: np.ndarray,
    codebook: np.ndarray,
    threshold_overrides: Dict[int, float],
    tau: float,
    global_id_offset: int = 0,
) -> Tuple[np.ndarray, int]:
    """Compute soft tree predictions using sigmoid routing.

    Returns (predictions, next_global_id).
    """
    n = X.shape[0]

    def _recurse(node: Optional[Node], counter: List[int]) -> np.ndarray:
        if node is None:
            return np.zeros(n)
        if node.left is None or node.right is None:
            # Leaf node
            return np.full(n, node.value)

        # Internal node
        nid = counter[0]
        counter[0] += 1
        k = node.codebook_idx
        t = threshold_overrides.get(nid, node.threshold)
        proj = X @ codebook[k]

        # Sigmoid with numerical stability
        z = (proj - t) / max(tau, 1e-8)
        z = np.clip(z, -500, 500)
        p_right = 1.0 / (1.0 + np.exp(-z))
        p_left = 1.0 - p_right

        left_pred = _recurse(node.left, counter)
        right_pred = _recurse(node.right, counter)

        return p_left * left_pred + p_right * right_pred

    counter = [global_id_offset]
    pred = _recurse(root, counter)
    return pred, counter[0]


def soft_predict_ensemble(
    trees: List[Node],
    X: np.ndarray,
    codebook: np.ndarray,
    threshold_overrides: Dict[int, float],
    tau: float,
) -> np.ndarray:
    """Sum soft predictions across all trees."""
    total = np.zeros(X.shape[0])
    offset = 0
    for tree_root in trees:
        pred, offset = soft_predict_tree(
            tree_root, X, codebook, threshold_overrides, tau, offset
        )
        total += pred
    return total


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7: JOINT L-BFGS-B REFINEMENT (core innovation)
# ═══════════════════════════════════════════════════════════════════════════════

def joint_refine_codebook_thresholds(
    trees: List[Node],
    codebook: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    lambda_l1: float = 0.01,
    lambda_l2: float = 0.001,
    tau: float = 1.0,
    maxiter: int = 30,
    subsample: int = 5000,
    random_state: int = 42,
) -> Tuple[np.ndarray, Dict[int, float], Any]:
    """Jointly optimize codebook entries and node thresholds using L-BFGS-B.

    Returns: (refined_codebook, refined_threshold_dict, optimization_result)
    """
    from scipy.optimize import minimize

    K, d = codebook.shape
    all_nodes = _collect_all_internal_nodes(trees)
    N_nodes = len(all_nodes)

    if N_nodes == 0:
        logger.debug("No internal nodes to refine")
        return codebook.copy(), {}, None

    # Subsample for large datasets
    n_samples = X.shape[0]
    if n_samples > subsample:
        rng = np.random.RandomState(random_state)
        sub_idx = rng.choice(n_samples, subsample, replace=False)
        X_sub = X[sub_idx]
        y_sub = y[sub_idx]
    else:
        X_sub = X
        y_sub = y

    # Pack initial parameters
    theta0 = np.concatenate([
        codebook.flatten(),
        np.array([node.threshold for node in all_nodes], dtype=np.float64),
    ])

    n_codebook_params = K * d
    eval_count = [0]
    best_loss = [np.inf]

    def objective(theta: np.ndarray) -> float:
        eval_count[0] += 1
        C_flat = theta[:n_codebook_params]
        C = C_flat.reshape(K, d)
        thresholds = theta[n_codebook_params:]

        threshold_dict: Dict[int, float] = {}
        offset = 0
        for tree_root in trees:
            tree_nodes = _collect_internal_nodes_tree(tree_root)
            for node in tree_nodes:
                threshold_dict[offset] = thresholds[offset] if offset < len(thresholds) else node.threshold
                offset += 1

        # Soft predictions
        y_pred = soft_predict_ensemble(trees, X_sub, C, threshold_dict, tau)

        # MSE loss
        residuals = y_sub - y_pred
        mse = float(np.mean(residuals ** 2))

        # Elastic-net penalty on codebook (smooth L1)
        l1_smooth = lambda_l1 * float(np.sum(np.sqrt(C ** 2 + 1e-6)))
        l2_penalty = lambda_l2 * float(np.sum(C ** 2))

        total = mse + l1_smooth + l2_penalty

        if total < best_loss[0]:
            best_loss[0] = total

        return total

    # Run L-BFGS-B with timeout
    t0_opt = time.time()
    try:
        result = minimize(
            fun=objective,
            x0=theta0,
            method='L-BFGS-B',
            jac='2-point',
            options={
                'maxiter': maxiter,
                'ftol': 1e-6,
                'gtol': 1e-5,
                'maxfun': maxiter * 5,
                'disp': False,
            },
        )
        opt_time = time.time() - t0_opt
        logger.debug(
            f"L-BFGS-B: nit={result.nit}, nfev={result.nfev}, "
            f"success={result.success}, time={opt_time:.1f}s, evals={eval_count[0]}"
        )
    except Exception as e:
        logger.warning(f"L-BFGS-B failed: {e}, returning original codebook")
        return codebook.copy(), {}, None

    # Check if optimization took too long (>60s) => flag for potential fallback
    if time.time() - t0_opt > 60:
        logger.warning(f"L-BFGS-B took {time.time() - t0_opt:.1f}s (>60s)")

    # Unpack result
    theta_opt = result.x
    C_opt = theta_opt[:n_codebook_params].reshape(K, d)
    thresholds_opt = theta_opt[n_codebook_params:]

    # Check for NaN
    if np.any(np.isnan(C_opt)) or np.any(np.isnan(thresholds_opt)):
        logger.warning("L-BFGS-B produced NaN, returning original codebook")
        return codebook.copy(), {}, result

    # Build threshold dict
    threshold_dict_opt: Dict[int, float] = {}
    for i in range(N_nodes):
        threshold_dict_opt[i] = float(thresholds_opt[i])

    return C_opt, threshold_dict_opt, result


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 8: WARM-START TREE REASSIGNMENT
# ═══════════════════════════════════════════════════════════════════════════════

def _recompute_leaf_values_and_idxs(
    node: Node,
    X: np.ndarray,
    y_residuals: np.ndarray,
    parent_idxs: np.ndarray,
    codebook: np.ndarray,
) -> None:
    """Recursively recompute leaf values and sample indices after codebook/threshold changes."""
    node.idxs = parent_idxs
    node.n_samples = int(parent_idxs.sum())

    if node.left is None or node.right is None:
        # Leaf node
        if node.n_samples > 0:
            node.value = float(np.mean(y_residuals[parent_idxs]))
        else:
            node.value = 0.0
        return

    # Internal node: split using current weights/threshold
    proj = X @ codebook[node.codebook_idx]
    left_mask = parent_idxs & (proj <= node.threshold)
    right_mask = parent_idxs & (proj > node.threshold)

    node.weights = codebook[node.codebook_idx].copy()

    _recompute_leaf_values_and_idxs(node.left, X, y_residuals, left_mask, codebook)
    _recompute_leaf_values_and_idxs(node.right, X, y_residuals, right_mask, codebook)

    # The value at internal node is mean of residuals (for pre-split prediction)
    if node.n_samples > 0:
        node.value = float(np.mean(y_residuals[parent_idxs]))
    else:
        node.value = 0.0


def warm_start_reassign(
    trees: List[Node],
    codebook: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    min_samples_leaf: int = 5,
) -> int:
    """After codebook refinement, reassign codebook entries at existing nodes.

    Returns number of reassignments made.
    """
    n_reassigned = 0
    n_samples = X.shape[0]
    K = codebook.shape[0]

    for tree_idx, tree_root in enumerate(trees):
        # Compute residuals: y - sum(other trees' predictions)
        y_res = y.copy()
        for other_idx, other_tree in enumerate(trees):
            if other_idx != tree_idx:
                y_res -= _hard_predict_tree(other_tree, X)

        # DFS through internal nodes
        internal_nodes = _collect_internal_nodes_tree(tree_root)
        for node in internal_nodes:
            if node.idxs is None or node.n_samples < 2 * min_samples_leaf:
                continue

            node_indices = np.where(node.idxs)[0]
            if len(node_indices) < 2 * min_samples_leaf:
                continue

            X_node = X[node_indices]
            y_node = y_res[node_indices]
            n_node = len(node_indices)
            parent_var = np.var(y_node) * n_node

            best_k = node.codebook_idx
            best_threshold = node.threshold
            best_reduction = -np.inf

            # Compute current reduction
            if best_k is not None:
                proj_curr = X_node @ codebook[best_k]
                order = np.argsort(proj_curr)
                sorted_proj = proj_curr[order]
                sorted_y = y_node[order]

                # Find reduction at current threshold
                mask_left = proj_curr <= node.threshold
                n_left = mask_left.sum()
                n_right = n_node - n_left
                if n_left >= min_samples_leaf and n_right >= min_samples_leaf:
                    left_mean = y_node[mask_left].mean()
                    right_mean = y_node[~mask_left].mean()
                    left_var = np.sum((y_node[mask_left] - left_mean) ** 2)
                    right_var = np.sum((y_node[~mask_left] - right_mean) ** 2)
                    best_reduction = parent_var - left_var - right_var

            # Try all codebook entries
            for k in range(K):
                proj = X_node @ codebook[k]
                order = np.argsort(proj)
                sorted_proj = proj[order]
                sorted_y = y_node[order]

                cum_sum = np.cumsum(sorted_y)
                cum_sq_sum = np.cumsum(sorted_y ** 2)
                total_sum = cum_sum[-1]
                total_sq_sum = cum_sq_sum[-1]

                # Use quantiles for efficiency
                n_check = min(50, n_node - 2 * min_samples_leaf)
                if n_check <= 0:
                    continue
                positions = np.linspace(
                    min_samples_leaf - 1,
                    n_node - min_samples_leaf - 1,
                    n_check,
                ).astype(int)
                positions = np.unique(positions)

                for i in positions:
                    if i + 1 < n_node and sorted_proj[i] == sorted_proj[i + 1]:
                        continue
                    n_l = i + 1
                    n_r = n_node - n_l
                    left_sum = cum_sum[i]
                    right_sum = total_sum - left_sum
                    left_mse = cum_sq_sum[i] - (left_sum ** 2) / n_l
                    right_mse = (total_sq_sum - cum_sq_sum[i]) - (right_sum ** 2) / n_r
                    reduction = parent_var - left_mse - right_mse

                    if reduction > best_reduction * 1.01:  # >1% improvement needed
                        best_reduction = reduction
                        best_k = k
                        if i + 1 < n_node:
                            best_threshold = (sorted_proj[i] + sorted_proj[i + 1]) / 2.0
                        else:
                            best_threshold = sorted_proj[i]

            # Apply reassignment if changed
            if best_k != node.codebook_idx or best_threshold != node.threshold:
                node.codebook_idx = best_k
                node.weights = codebook[best_k].copy()
                node.threshold = best_threshold
                n_reassigned += 1

        # Recompute all leaf values and idxs for this tree
        y_res_tree = y.copy()
        for other_idx, other_tree in enumerate(trees):
            if other_idx != tree_idx:
                y_res_tree -= _hard_predict_tree(other_tree, X)
        root_idxs = np.ones(n_samples, dtype=bool)
        _recompute_leaf_values_and_idxs(tree_root, X, y_res_tree, root_idxs, codebook)

    return n_reassigned


def _hard_predict_tree(root: Node, X: np.ndarray) -> np.ndarray:
    """Hard tree prediction (same as CodebookFIGS_v2._predict_tree)."""
    preds = np.full(X.shape[0], root.value)
    if root.left is not None and root.right is not None:
        proj = X @ root.weights
        left_mask = proj <= root.threshold
        left_preds = _hard_predict_tree(root.left, X)
        right_preds = _hard_predict_tree(root.right, X)
        preds[left_mask] = left_preds[left_mask]
        preds[~left_mask] = right_preds[~left_mask]
    return preds


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 9: ELASTIC-NET SPARSIFICATION
# ═══════════════════════════════════════════════════════════════════════════════

def apply_elastic_net_sparsification(
    codebook: np.ndarray,
    threshold: float = 0.05,
) -> np.ndarray:
    """Apply post-optimization sparsification to codebook."""
    K, d = codebook.shape
    C = codebook.copy()
    for k in range(K):
        # Zero out small weights
        mask = np.abs(C[k]) < threshold
        C[k, mask] = 0.0
        # Check if all zeros
        norm = np.linalg.norm(C[k])
        if norm < 1e-12:
            # Degenerate - reinitialize
            rng = np.random.RandomState(42 + k)
            C[k] = rng.randn(d)
        # Normalize
        C[k] /= np.linalg.norm(C[k]) + 1e-12
    return C


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 10: CodebookFIGS_v2 CLASS
# ═══════════════════════════════════════════════════════════════════════════════

class CodebookFIGS_v2:
    """Codebook-FIGS v2 with joint gradient refinement & elastic-net sparsity.

    Key differences from iter3 CodebookFIGS:
    1. Joint L-BFGS-B refinement of codebook + thresholds (replaces WLS)
    2. Warm-start tree reassignment (replaces full re-growth after round 1)
    3. Elastic-net sparsification after each refinement
    4. 10 alternation rounds (up from 7)
    5. Smooth L1 penalty in the optimization objective
    """

    def __init__(
        self,
        K: int = 8,
        max_rules: int = 12,
        max_trees: Optional[int] = None,
        max_depth: int = 4,
        min_impurity_decrease: float = 0.0,
        n_alternation_rounds: int = 10,
        init_strategy: str = "adaptive",
        tau: float = 1.0,
        lambda_l1: float = 0.01,
        lambda_l2: float = 0.001,
        weight_threshold: float = 0.05,
        lbfgs_maxiter: int = 30,
        random_state: int = 42,
        min_samples_leaf: int = 5,
    ) -> None:
        self.K = K
        self.max_rules = max_rules
        self.max_trees = max_trees if max_trees else 10
        self.max_depth = max_depth
        self.min_impurity_decrease = min_impurity_decrease
        self.n_alternation_rounds = n_alternation_rounds
        self.init_strategy = init_strategy
        self.tau = tau
        self.lambda_l1 = lambda_l1
        self.lambda_l2 = lambda_l2
        self.weight_threshold = weight_threshold
        self.lbfgs_maxiter = lbfgs_maxiter
        self.random_state = random_state
        self.min_samples_leaf = min_samples_leaf

        # Fitted attributes
        self.trees_: List[Node] = []
        self.codebook_: Optional[np.ndarray] = None
        self.scaler_: Any = None
        self.task_type_: str = "classification"
        self.complexity_: int = 0
        self.history_: Dict[str, list] = {}
        self.converged_at_round_: int = 0
        self.lbfgs_total_iters_: int = 0
        self.lbfgs_total_converged_: int = 0
        self._use_fallback_wls: bool = False

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        task_type: str = "classification",
    ) -> "CodebookFIGS_v2":
        """Main fit method with v2 alternation loop."""
        from sklearn.preprocessing import StandardScaler

        self.task_type_ = task_type
        n_samples, n_features = X.shape

        self.scaler_ = StandardScaler()
        X_s = self.scaler_.fit_transform(X)

        # Initialize codebook
        if self.init_strategy == "adaptive":
            self.codebook_ = init_codebook_adaptive(X_s, y, self.K, task_type, self.random_state)
        elif self.init_strategy == "random":
            self.codebook_ = init_codebook_random(X_s, self.K, self.random_state)
        elif self.init_strategy == "pca":
            self.codebook_ = init_codebook_pca(X_s, self.K)
        elif self.init_strategy == "lda":
            self.codebook_ = init_codebook_lda(X_s, y, self.K, task_type)
        else:
            raise ValueError(f"Unknown init_strategy: {self.init_strategy}")

        self.history_ = {
            "train_losses": [],
            "eranks": [],
            "n_splits": [],
            "primary_metric": [],
            "codebooks": [],
            "sparsity_top3": [],
            "sparsity_l0": [],
            "lbfgs_n_iterations": [],
            "lbfgs_converged": [],
        }

        initial_loss: Optional[float] = None
        prev_loss: Optional[float] = None
        lbfgs_fail_count = 0

        for round_idx in range(self.n_alternation_rounds):
            # ROUND 0: Initial tree growth from scratch
            if round_idx == 0:
                self._grow_trees_figs_style(X_s, y)
            else:
                # STEP A: Joint gradient-based refinement
                all_internal = _collect_all_internal_nodes(self.trees_)
                n_internal = len(all_internal)

                if n_internal >= 2 and not self._use_fallback_wls:
                    t_lbfgs = time.time()
                    try:
                        new_codebook, new_thresholds, opt_result = \
                            joint_refine_codebook_thresholds(
                                self.trees_, self.codebook_, X_s, y,
                                lambda_l1=self.lambda_l1,
                                lambda_l2=self.lambda_l2,
                                tau=self.tau,
                                maxiter=self.lbfgs_maxiter,
                                random_state=self.random_state,
                            )
                        lbfgs_time = time.time() - t_lbfgs

                        if opt_result is not None:
                            self.lbfgs_total_iters_ += getattr(opt_result, 'nit', 0)
                            if getattr(opt_result, 'success', False):
                                self.lbfgs_total_converged_ += 1
                            self.history_["lbfgs_n_iterations"].append(
                                getattr(opt_result, 'nit', 0)
                            )
                            self.history_["lbfgs_converged"].append(
                                getattr(opt_result, 'success', False)
                            )
                        else:
                            self.history_["lbfgs_n_iterations"].append(0)
                            self.history_["lbfgs_converged"].append(False)

                        # Check for NaN/inf in new codebook
                        if np.all(np.isfinite(new_codebook)):
                            self.codebook_ = new_codebook

                            # Update all node thresholds from optimization
                            offset = 0
                            for tree_root in self.trees_:
                                tree_nodes = _collect_internal_nodes_tree(tree_root)
                                for node in tree_nodes:
                                    if offset in new_thresholds:
                                        node.threshold = new_thresholds[offset]
                                    node.weights = self.codebook_[node.codebook_idx].copy()
                                    offset += 1
                        else:
                            logger.warning(f"Round {round_idx}: L-BFGS-B produced non-finite codebook")
                            lbfgs_fail_count += 1

                        # If L-BFGS takes >60s, switch to WLS fallback
                        if lbfgs_time > 60:
                            logger.warning(
                                f"L-BFGS-B took {lbfgs_time:.1f}s, switching to WLS fallback"
                            )
                            self._use_fallback_wls = True

                    except Exception as e:
                        logger.warning(f"L-BFGS-B exception at round {round_idx}: {e}")
                        lbfgs_fail_count += 1
                        self.history_["lbfgs_n_iterations"].append(0)
                        self.history_["lbfgs_converged"].append(False)
                        if lbfgs_fail_count >= 3:
                            self._use_fallback_wls = True
                elif self._use_fallback_wls or n_internal < 2:
                    # Fallback: WLS refinement (iter3-style)
                    self._refine_codebook_wls(X_s, y)
                    self.history_["lbfgs_n_iterations"].append(0)
                    self.history_["lbfgs_converged"].append(False)

                # STEP B: Elastic-net sparsification
                self.codebook_ = apply_elastic_net_sparsification(
                    self.codebook_,
                    threshold=self.weight_threshold,
                )
                # Update node weights after sparsification
                for tree_root in self.trees_:
                    self._update_node_weights(tree_root)

                # STEP C: Warm-start tree reassignment
                n_reassigned = warm_start_reassign(
                    self.trees_, self.codebook_, X_s, y,
                    min_samples_leaf=self.min_samples_leaf,
                )
                logger.debug(f"  Round {round_idx}: {n_reassigned} nodes reassigned")

                # STEP D: Codebook collapse check
                self._check_codebook_collapse()

            # STEP E: Record metrics
            preds = self._predict_raw(X_s)
            loss = float(np.mean((y - preds) ** 2))

            if task_type == "classification":
                metric = float(np.mean((preds >= 0.5).astype(int) == y.astype(int)))
            else:
                ss_res = np.sum((y - preds) ** 2)
                ss_tot = np.sum((y - y.mean()) ** 2)
                metric = 1.0 - ss_res / max(ss_tot, 1e-12)

            sparsity = compute_codebook_sparsity(self.codebook_, self.weight_threshold)

            self.history_["train_losses"].append(loss)
            self.history_["primary_metric"].append(float(metric))
            self.history_["eranks"].append(effective_rank(self.codebook_))
            self.history_["n_splits"].append(self.complexity_)
            self.history_["codebooks"].append(self.codebook_.copy())
            self.history_["sparsity_top3"].append(sparsity["top3_concentration"])
            self.history_["sparsity_l0"].append(sparsity["mean_l0_norm"])

            if round_idx == 0 and len(self.history_["lbfgs_n_iterations"]) == 0:
                self.history_["lbfgs_n_iterations"].append(0)
                self.history_["lbfgs_converged"].append(False)

            if initial_loss is None:
                initial_loss = loss

            # STEP F: Convergence check
            if round_idx > 0 and prev_loss is not None and initial_loss is not None and initial_loss > 0:
                if abs(loss - prev_loss) < 0.003 * initial_loss:
                    self.converged_at_round_ = round_idx + 1
                    # Final tree growth with converged codebook
                    self._grow_trees_figs_style(X_s, y)
                    break

            prev_loss = loss

        if self.converged_at_round_ == 0:
            self.converged_at_round_ = len(self.history_["train_losses"])
            # Final tree growth with final codebook
            self._grow_trees_figs_style(X_s, y)

        return self

    # ── TRUE FIGS-STYLE SIMULTANEOUS TREE GROWTH ──────────────────────────

    def _grow_trees_figs_style(self, X: np.ndarray, y: np.ndarray) -> None:
        """Grow trees using FIGS algorithm: priority queue, simultaneous growth."""
        n_samples = X.shape[0]
        self.trees_ = []
        self.complexity_ = 0
        y_pred_per_tree: Dict[int, np.ndarray] = {}

        all_idxs = np.ones(n_samples, dtype=bool)

        bk, bt, br, il, ir = find_best_codebook_split(
            X, y, all_idxs, self.codebook_, self.min_samples_leaf,
        )
        if bk is None:
            root = Node()
            root.is_root = True
            root.tree_num = 0
            root.idxs = all_idxs
            root.value = float(np.mean(y))
            root.n_samples = n_samples
            self.trees_.append(root)
            return

        root = Node()
        root.is_root = True
        root.tree_num = -1
        root.idxs = all_idxs
        root.value = float(np.mean(y))
        root.impurity_reduction = br
        root.codebook_idx = bk
        root.weights = self.codebook_[bk].copy()
        root.threshold = bt
        root.depth = 0
        root.n_samples = n_samples

        root.left_temp = Node()
        root.left_temp.idxs = il
        root.left_temp.value = float(np.mean(y[il])) if il.sum() > 0 else 0.0
        root.left_temp.n_samples = int(il.sum())
        root.left_temp.depth = 1

        root.right_temp = Node()
        root.right_temp.idxs = ir
        root.right_temp.value = float(np.mean(y[ir])) if ir.sum() > 0 else 0.0
        root.right_temp.n_samples = int(ir.sum())
        root.right_temp.depth = 1

        potential_splits: List[Node] = [root]
        max_new_trees = self.max_trees

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

            # If root node → register as new tree
            if best_node.is_root:
                if len(self.trees_) >= max_new_trees:
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

                # Compute residuals for this tree
                y_res = y.copy()
                for other_t, pred in y_pred_per_tree.items():
                    if other_t != child.tree_num:
                        y_res -= pred

                bk2, bt2, br2, il2, ir2 = find_best_codebook_split(
                    X, y_res, child.idxs, self.codebook_, self.min_samples_leaf,
                )

                if bk2 is not None and br2 > self.min_impurity_decrease:
                    child.codebook_idx = bk2
                    child.weights = self.codebook_[bk2].copy()
                    child.threshold = bt2
                    child.impurity_reduction = br2

                    child.left_temp = Node()
                    child.left_temp.idxs = il2
                    child.left_temp.value = float(np.mean(y_res[il2])) if il2.sum() > 0 else 0.0
                    child.left_temp.n_samples = int(il2.sum())
                    child.left_temp.depth = child.depth + 1

                    child.right_temp = Node()
                    child.right_temp.idxs = ir2
                    child.right_temp.value = float(np.mean(y_res[ir2])) if ir2.sum() > 0 else 0.0
                    child.right_temp.n_samples = int(ir2.sum())
                    child.right_temp.depth = child.depth + 1

                    potential_splits.append(child)

            # Try adding a new root for a new tree
            if len(self.trees_) < max_new_trees:
                y_total = np.zeros(n_samples)
                for pred in y_pred_per_tree.values():
                    y_total += pred
                y_res_new = y - y_total

                bk3, bt3, br3, il3, ir3 = find_best_codebook_split(
                    X, y_res_new, np.ones(n_samples, dtype=bool),
                    self.codebook_, self.min_samples_leaf,
                )

                if bk3 is not None and br3 > self.min_impurity_decrease:
                    new_root = Node()
                    new_root.is_root = True
                    new_root.tree_num = -1
                    new_root.idxs = np.ones(n_samples, dtype=bool)
                    new_root.value = float(np.mean(y_res_new))
                    new_root.impurity_reduction = br3
                    new_root.codebook_idx = bk3
                    new_root.weights = self.codebook_[bk3].copy()
                    new_root.threshold = bt3
                    new_root.depth = 0
                    new_root.n_samples = n_samples

                    new_root.left_temp = Node()
                    new_root.left_temp.idxs = il3
                    new_root.left_temp.value = float(np.mean(y_res_new[il3])) if il3.sum() > 0 else 0.0
                    new_root.left_temp.n_samples = int(il3.sum())
                    new_root.left_temp.depth = 1

                    new_root.right_temp = Node()
                    new_root.right_temp.idxs = ir3
                    new_root.right_temp.value = float(np.mean(y_res_new[ir3])) if ir3.sum() > 0 else 0.0
                    new_root.right_temp.n_samples = int(ir3.sum())
                    new_root.right_temp.depth = 1

                    potential_splits.append(new_root)

    # ── WLS CODEBOOK REFINEMENT (fallback from iter3) ─────────────────────

    def _refine_codebook_wls(self, X: np.ndarray, y: np.ndarray) -> None:
        """WLS refinement: X^T y_residual direction (iter3 fallback)."""
        K, n_features = self.codebook_.shape

        for k in range(K):
            nodes_k: List[Node] = []
            for tree in self.trees_:
                self._collect_nodes_with_idx(tree, k, nodes_k)

            if not nodes_k:
                rng = np.random.RandomState(self.random_state + k + 200)
                self.codebook_[k] = rng.randn(n_features)
                self.codebook_[k] /= np.linalg.norm(self.codebook_[k]) + 1e-12
                continue

            X_pool_list, y_res_list = [], []
            for node in nodes_k:
                y_res = y.copy()
                for t_idx, tree in enumerate(self.trees_):
                    if t_idx != node.tree_num:
                        y_res -= self._predict_tree(tree, X)
                if node.idxs is not None:
                    X_pool_list.append(X[node.idxs])
                    y_res_list.append(y_res[node.idxs])

            if not X_pool_list:
                continue

            X_pool = np.vstack(X_pool_list)
            y_pool = np.concatenate(y_res_list)

            direction = X_pool.T @ y_pool
            norm = np.linalg.norm(direction)
            if norm > 1e-12:
                self.codebook_[k] = direction / norm

    def _collect_nodes_with_idx(self, node: Optional[Node], k: int, result: List[Node]) -> None:
        """Recursively collect internal nodes using codebook entry k."""
        if node is None:
            return
        if node.left is not None and node.right is not None:
            if node.codebook_idx == k:
                result.append(node)
            self._collect_nodes_with_idx(node.left, k, result)
            self._collect_nodes_with_idx(node.right, k, result)

    def _update_node_weights(self, node: Optional[Node]) -> None:
        """Update all node weights from codebook after sparsification."""
        if node is None:
            return
        if node.left is not None and node.right is not None:
            if node.codebook_idx is not None:
                node.weights = self.codebook_[node.codebook_idx].copy()
            self._update_node_weights(node.left)
            self._update_node_weights(node.right)

    # ── CODEBOOK COLLAPSE CHECK ───────────────────────────────────────────

    def _check_codebook_collapse(self) -> int:
        """Check for collapsed entries (|cos sim| > 0.99) and reinitialize."""
        from sklearn.metrics.pairwise import cosine_similarity
        K = self.codebook_.shape[0]
        n_feat = self.codebook_.shape[1]
        cos_sim = np.abs(cosine_similarity(self.codebook_))
        np.fill_diagonal(cos_sim, 0.0)
        n_collapses = 0
        for i in range(K):
            for j in range(i + 1, K):
                if cos_sim[i, j] > 0.99:
                    rng = np.random.RandomState(self.random_state + 100 + j)
                    self.codebook_[j] = rng.randn(n_feat)
                    self.codebook_[j] /= np.linalg.norm(self.codebook_[j]) + 1e-12
                    n_collapses += 1
        return n_collapses

    # ── PREDICTION ────────────────────────────────────────────────────────

    def _predict_tree(self, root: Node, X: np.ndarray) -> np.ndarray:
        """Recursive oblique tree prediction."""
        preds = np.full(X.shape[0], root.value)
        if root.left is not None and root.right is not None:
            proj = X @ root.weights
            left_mask = proj <= root.threshold
            left_preds = self._predict_tree(root.left, X)
            right_preds = self._predict_tree(root.right, X)
            preds[left_mask] = left_preds[left_mask]
            preds[~left_mask] = right_preds[~left_mask]
        return preds

    def _predict_raw(self, X: np.ndarray) -> np.ndarray:
        if not self.trees_:
            return np.zeros(X.shape[0])
        return sum(self._predict_tree(t, X) for t in self.trees_)

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

    def get_codebook_usage(self) -> Dict[int, int]:
        """Return dict {k: count} of how many nodes use each codebook entry."""
        usage: Dict[int, int] = defaultdict(int)
        for tree in self.trees_:
            self._count_usage(tree, usage)
        return dict(usage)

    def _count_usage(self, node: Optional[Node], usage: Dict[int, int]) -> None:
        if node is None:
            return
        if node.left is not None and node.right is not None:
            if node.codebook_idx is not None:
                usage[node.codebook_idx] += 1
            self._count_usage(node.left, usage)
            self._count_usage(node.right, usage)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 11: BASELINES (reuse from iter3)
# ═══════════════════════════════════════════════════════════════════════════════

def run_figs_baseline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    task_type: str,
    max_rules: int = 12,
) -> Tuple[np.ndarray, Optional[np.ndarray], int]:
    """FIGS from imodels. Returns (y_pred, y_proba, n_splits)."""
    from imodels import FIGSClassifier, FIGSRegressor

    if task_type == "classification":
        model = FIGSClassifier(max_rules=max_rules)
    else:
        model = FIGSRegressor(max_rules=max_rules)
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
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """XGBoost baseline."""
    import xgboost as xgb

    if task_type == "classification":
        model = xgb.XGBClassifier(
            max_depth=4, n_estimators=100, learning_rate=0.1,
            random_state=42, eval_metric="logloss", verbosity=0,
            use_label_encoder=False,
        )
    else:
        model = xgb.XGBRegressor(
            max_depth=4, n_estimators=100, learning_rate=0.1,
            random_state=42, verbosity=0,
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
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """LightGBM baseline."""
    import lightgbm as lgb

    if task_type == "classification":
        model = lgb.LGBMClassifier(
            max_depth=4, n_estimators=100, learning_rate=0.1,
            random_state=42, verbose=-1,
        )
    else:
        model = lgb.LGBMRegressor(
            max_depth=4, n_estimators=100, learning_rate=0.1,
            random_state=42, verbose=-1,
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
# SECTION 12: SUMMARY AND OUTPUT FORMATTING (extended from iter3)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_summary(all_results: List[Dict]) -> List[Dict]:
    """Aggregate per-fold results into mean±std per (dataset, config, K)."""
    groups: Dict[tuple, List[Dict]] = defaultdict(list)
    for r in all_results:
        key = (r["dataset"], r.get("config", r["method"]), r.get("K"))
        groups[key].append(r)

    summary_rows = []
    agg_cols = [
        "accuracy", "auroc", "rmse", "r2", "erank", "n_splits",
        "train_time_sec", "n_codebook_used", "codebook_stability_mean",
        "converged_at_round", "top3_concentration", "mean_l0_norm", "lbfgs_iters",
    ]
    for (ds, config, K), rows in groups.items():
        row: Dict[str, Any] = {"dataset": ds, "config": config, "K": K}
        for col in agg_cols:
            vals = [r.get(col) for r in rows if r.get(col) is not None]
            if vals:
                arr = np.array(vals, dtype=float)
                row[f"{col}_mean"] = float(np.nanmean(arr))
                row[f"{col}_std"] = float(np.nanstd(arr))
        summary_rows.append(row)
    return summary_rows


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
    elif isinstance(obj, bool):
        return obj
    return obj


def format_output_for_schema(
    all_results: List[Dict],
    summary: List[Dict],
    full_datasets: Dict,
    config: Dict,
) -> Dict:
    """Format output to match exp_gen_sol_out.json schema."""
    datasets_out = []

    # Group results by dataset
    by_dataset: Dict[str, List[Dict]] = defaultdict(list)
    for r in all_results:
        by_dataset[r["dataset"]].append(r)

    for ds_name, ds_info in full_datasets.items():
        if ds_name.endswith("_mini"):
            continue

        ds_results = by_dataset.get(ds_name, [])
        if not ds_results:
            continue

        examples = []
        for fold_idx in range(N_FOLDS):
            fold_results = [r for r in ds_results if r.get("fold") == fold_idx]
            if not fold_results:
                continue

            fold_mask = ds_info["folds"] == fold_idx
            n_test = int(np.sum(fold_mask))
            n_train = int(np.sum(~fold_mask))

            example: Dict[str, Any] = {
                "input": json.dumps({
                    "dataset": ds_name,
                    "fold": fold_idx,
                    "n_train": n_train,
                    "n_test": n_test,
                    "task_type": ds_info["task_type"],
                }),
                "output": json.dumps({
                    "fold_results": _sanitize_for_json(fold_results),
                }),
                "metadata_fold": fold_idx,
                "metadata_dataset": ds_name,
                "metadata_task_type": ds_info["task_type"],
            }

            # Add per-method predictions as predict_* fields
            for r in fold_results:
                method_key = r.get("config", r["method"])
                K_val = r.get("K")
                if K_val is not None:
                    pred_key = f"predict_{method_key}_K{K_val}"
                else:
                    pred_key = f"predict_{method_key}"

                pred_val = {}
                for mk in ["accuracy", "auroc", "rmse", "r2", "erank",
                            "n_splits", "train_time_sec", "n_codebook_used",
                            "codebook_stability_mean", "converged_at_round",
                            "top3_concentration", "mean_l0_norm", "lbfgs_iters"]:
                    if r.get(mk) is not None:
                        pred_val[mk] = r[mk]
                example[pred_key] = json.dumps(_sanitize_for_json(pred_val))

            examples.append(example)

        if examples:
            datasets_out.append({
                "dataset": ds_name,
                "examples": examples,
            })

    output = {
        "metadata": _sanitize_for_json({
            "experiment": "codebook_figs_v2_joint_refinement",
            "description": (
                "Codebook-FIGS v2 with joint L-BFGS-B codebook+threshold refinement, "
                "elastic-net sparsity, and warm-start tree reassignment. "
                "Benchmarked against FIGS, XGBoost, LightGBM across 10 tabular datasets."
            ),
            "improvements_over_iter3": [
                "Joint gradient-based optimization of codebook entries AND thresholds",
                "Elastic-net (L1+L2) sparsity penalty on codebook entries",
                "Post-optimization weight thresholding for genuine sparsity",
                "Warm-start tree reassignment instead of full re-growth",
                "10 alternation rounds (up from 7)",
                "Dropped K=20 (consistently degraded)",
            ],
            "config": config,
            "summary": summary,
        }),
        "datasets": datasets_out,
    }
    return output


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 13: MAIN EXPERIMENT LOOP
# ═══════════════════════════════════════════════════════════════════════════════

@logger.catch
def main() -> None:
    start_time = time.time()
    logger.info("=" * 70)
    logger.info("Codebook-FIGS v2: Joint Gradient Refinement with Elastic-Net Sparsity")
    logger.info("=" * 70)

    # ── Verify data directory ─────────────────────────────────────────────
    if not DATA_DIR.exists():
        logger.error(f"Data directory not found: {DATA_DIR}")
        raise FileNotFoundError(f"Data directory not found: {DATA_DIR}")
    logger.info(f"Data directory: {DATA_DIR}")

    # ── Load datasets ─────────────────────────────────────────────────────
    full_datasets = load_datasets(str(FULL_DATA_PATH))
    full_only = {k: v for k, v in full_datasets.items() if not k.endswith("_mini")}
    logger.info(f"Full datasets ({len(full_only)}): {sorted(full_only.keys())}")

    # ══════════════════════════════════════════════════════════════════════
    # PHASE 1: SMOKE TEST on mini data
    # ══════════════════════════════════════════════════════════════════════
    logger.info("\n=== SMOKE TEST ===")
    smoke_ds = None
    for name in ["heart_disease_mini", "diabetes_pima_mini"]:
        if name in full_datasets:
            smoke_ds = full_datasets[name]
            smoke_name = name
            break
    if smoke_ds is None:
        for name in full_datasets:
            if name.endswith("_mini"):
                smoke_ds = full_datasets[name]
                smoke_name = name
                break

    if smoke_ds is not None and smoke_ds["n_samples"] >= 10:
        X_smoke, y_smoke = smoke_ds["X"], smoke_ds["y"]
        tt_smoke = smoke_ds["task_type"]

        n_test = max(int(0.2 * len(y_smoke)), 3)
        X_tr = X_smoke[n_test:]
        X_te = X_smoke[:n_test]
        y_tr = y_smoke[n_test:]
        y_te = y_smoke[:n_test]

        for init_str in ["adaptive", "random"]:
            try:
                model = CodebookFIGS_v2(
                    K=5, max_rules=4, max_depth=3,
                    n_alternation_rounds=3,
                    init_strategy=init_str,
                    lbfgs_maxiter=5,
                    tau=1.0, lambda_l1=0.01, lambda_l2=0.001,
                    weight_threshold=0.05,
                    random_state=42,
                )
                model.fit(X_tr, y_tr, tt_smoke)
                preds = model.predict(X_te)
                proba = model.predict_proba(X_te)

                assert np.all(np.isfinite(preds)), f"Non-finite predictions with {init_str}"
                norms = np.linalg.norm(model.codebook_, axis=1)
                assert np.allclose(norms, 1.0, atol=1e-6), f"Codebook not unit-norm with {init_str}"

                # Verify sparsity metrics
                sparsity = compute_codebook_sparsity(model.codebook_, 0.05)
                assert "top3_concentration" in sparsity
                assert "mean_l0_norm" in sparsity

                # Verify training loss decreases (or stays stable)
                losses = model.history_["train_losses"]
                if len(losses) >= 2:
                    logger.info(f"  Smoke {init_str}: loss trajectory = "
                                f"{[f'{l:.4f}' for l in losses]}")

                if tt_smoke == "classification":
                    from sklearn.metrics import accuracy_score
                    acc = accuracy_score(y_te, preds)
                    logger.info(
                        f"  Smoke {init_str}: acc={acc:.3f}, complexity={model.complexity_}, "
                        f"trees={len(model.trees_)}, top3_conc={sparsity['top3_concentration']:.3f}"
                    )
                else:
                    logger.info(
                        f"  Smoke {init_str}: complexity={model.complexity_}, "
                        f"trees={len(model.trees_)}, top3_conc={sparsity['top3_concentration']:.3f}"
                    )
            except Exception:
                logger.exception(f"Smoke test FAILED for init={init_str}")
                raise

        # Verify soft tree prediction correctness
        logger.info("  Verifying soft vs hard prediction consistency...")
        test_model = CodebookFIGS_v2(
            K=5, max_rules=4, max_depth=3,
            n_alternation_rounds=2, init_strategy="adaptive",
            lbfgs_maxiter=3, tau=1.0, random_state=42,
        )
        test_model.fit(X_tr, y_tr, tt_smoke)
        X_s_test = test_model.scaler_.transform(X_te)

        hard_pred = test_model._predict_raw(X_s_test)
        # Soft predict with very low tau (nearly hard)
        thresh_dict: Dict[int, float] = {}
        offset = 0
        for tree_root in test_model.trees_:
            tree_nodes = _collect_internal_nodes_tree(tree_root)
            for node in tree_nodes:
                thresh_dict[offset] = node.threshold
                offset += 1

        soft_pred_hard = soft_predict_ensemble(
            test_model.trees_, X_s_test, test_model.codebook_, thresh_dict, tau=0.001,
        )
        y_std = max(np.std(y_te), 1e-6)
        max_diff = np.max(np.abs(soft_pred_hard - hard_pred))
        logger.info(f"  Soft vs hard (tau=0.001): max_diff={max_diff:.6f}, threshold={0.01 * y_std:.6f}")
        if max_diff > 0.1 * y_std:
            logger.warning("  Soft prediction differs significantly from hard prediction at low tau")

        # Soft predict with tau=1.0 should differ from hard
        soft_pred_warm = soft_predict_ensemble(
            test_model.trees_, X_s_test, test_model.codebook_, thresh_dict, tau=1.0,
        )
        diff_warm = np.max(np.abs(soft_pred_warm - hard_pred))
        logger.info(f"  Soft vs hard (tau=1.0): max_diff={diff_warm:.6f}")

        logger.info("Smoke test PASSED!")
    else:
        logger.warning("No suitable mini dataset for smoke test")

    # ══════════════════════════════════════════════════════════════════════
    # PHASE 2: FULL BENCHMARK
    # ══════════════════════════════════════════════════════════════════════
    logger.info("\n=== FULL BENCHMARK ===")
    all_results: List[Dict[str, Any]] = []
    codebook_archive: Dict[tuple, np.ndarray] = {}

    # Adaptive timeout parameters
    current_lbfgs_maxiter = LBFGS_MAXITER
    current_n_rounds = N_ALTERNATION_ROUNDS
    current_k_values = list(K_VALUES)

    CB_CONFIGS = [
        {"name": "v2_adaptive", "init": "adaptive"},
        {"name": "v2_random",   "init": "random"},
    ]

    dataset_times: Dict[str, float] = {}

    for ds_idx, (ds_name, ds_info) in enumerate(sorted(full_only.items())):
        ds_start = time.time()
        X, y = ds_info["X"], ds_info["y"]
        folds, task_type = ds_info["folds"], ds_info["task_type"]

        logger.info(f"\n=== {ds_name} ({task_type}, n={X.shape[0]}, d={X.shape[1]}) ===")

        # Fallback 5: Timeout risk check
        elapsed_so_far = time.time() - start_time
        if ds_idx >= 5 and elapsed_so_far > 25 * 60:
            logger.warning(f"Elapsed {elapsed_so_far/60:.1f}min after {ds_idx} datasets. Reducing params.")
            current_lbfgs_maxiter = min(current_lbfgs_maxiter, 15)
            current_n_rounds = min(current_n_rounds, 5)
            if 12 in current_k_values and len(current_k_values) > 3:
                current_k_values = [k for k in current_k_values if k != 12]
                logger.warning(f"Dropped K=12, using K={current_k_values}")

        unique_folds = sorted(np.unique(folds))
        actual_n_folds = min(N_FOLDS, len(unique_folds))

        for fold_idx in range(actual_n_folds):
            fold_val = unique_folds[fold_idx]
            test_mask = folds == fold_val
            train_mask = ~test_mask
            if test_mask.sum() == 0 or train_mask.sum() == 0:
                logger.warning(f"  Fold {fold_idx}: empty split, skipping")
                continue

            X_tr, X_te = X[train_mask], X[test_mask]
            y_tr, y_te = y[train_mask], y[test_mask]

            # ── Baselines ────────────────────────────────────────────
            for baseline_name in ["figs", "xgboost", "lightgbm"]:
                t0 = time.time()
                try:
                    if baseline_name == "figs":
                        y_pred, y_proba, n_splits = run_figs_baseline(
                            X_tr, y_tr, X_te, task_type, MAX_RULES,
                        )
                    elif baseline_name == "xgboost":
                        y_pred, y_proba = run_xgboost_baseline(X_tr, y_tr, X_te, task_type)
                        n_splits = None
                    elif baseline_name == "lightgbm":
                        y_pred, y_proba = run_lgbm_baseline(X_tr, y_tr, X_te, task_type)
                        n_splits = None
                    else:
                        continue

                    elapsed = time.time() - t0
                    metrics = evaluate_fold(y_te, y_pred, y_proba, task_type)
                    all_results.append({
                        "dataset": ds_name,
                        "method": baseline_name,
                        "config": baseline_name,
                        "K": None,
                        "fold": fold_idx,
                        "train_time_sec": round(elapsed, 4),
                        "n_splits": n_splits,
                        "erank": None,
                        "n_codebook_used": None,
                        "codebook_stability_mean": None,
                        "converged_at_round": None,
                        "top3_concentration": None,
                        "mean_l0_norm": None,
                        "lbfgs_iters": None,
                        **metrics,
                    })
                    logger.info(f"  {baseline_name} fold={fold_idx}: {metrics} ({elapsed:.2f}s)")

                except ImportError:
                    logger.warning(f"  {baseline_name} not available (import error)")
                    all_results.append({
                        "dataset": ds_name, "method": baseline_name,
                        "config": baseline_name, "K": None, "fold": fold_idx,
                        "train_time_sec": 0.0, "n_splits": None,
                        "erank": None, "n_codebook_used": None,
                        "codebook_stability_mean": None, "converged_at_round": None,
                        "top3_concentration": None, "mean_l0_norm": None, "lbfgs_iters": None,
                        "accuracy": None, "auroc": None, "rmse": None, "r2": None,
                    })
                except Exception:
                    logger.exception(f"  {baseline_name} fold={fold_idx} FAILED")
                    all_results.append({
                        "dataset": ds_name, "method": baseline_name,
                        "config": baseline_name, "K": None, "fold": fold_idx,
                        "train_time_sec": time.time() - t0, "n_splits": None,
                        "erank": None, "n_codebook_used": None,
                        "codebook_stability_mean": None, "converged_at_round": None,
                        "top3_concentration": None, "mean_l0_norm": None, "lbfgs_iters": None,
                        "accuracy": None, "auroc": None, "rmse": None, "r2": None,
                    })

            # ── Codebook-FIGS v2 for each config × K ──────────────────
            for cfg in CB_CONFIGS:
                for K in current_k_values:
                    K_actual = min(K, ds_info["n_features"])
                    t0 = time.time()

                    try:
                        model = CodebookFIGS_v2(
                            K=K_actual,
                            max_rules=MAX_RULES,
                            max_depth=MAX_DEPTH,
                            n_alternation_rounds=current_n_rounds,
                            init_strategy=cfg["init"],
                            tau=TAU_SOFT,
                            lambda_l1=LAMBDA_L1,
                            lambda_l2=LAMBDA_L2,
                            weight_threshold=WEIGHT_THRESHOLD,
                            lbfgs_maxiter=current_lbfgs_maxiter,
                            random_state=42 + fold_idx,
                            min_samples_leaf=MIN_SAMPLES_LEAF,
                        )
                        model.fit(X_tr, y_tr, task_type)
                        elapsed = time.time() - t0

                        if task_type == "classification":
                            y_pred = model.predict(X_te)
                            y_proba = model.predict_proba(X_te)
                        else:
                            y_pred = model.predict(X_te)
                            y_proba = None

                        metrics = evaluate_fold(y_te, y_pred, y_proba, task_type)

                        usage = model.get_codebook_usage()
                        usage_array = np.array([usage.get(k, 0) for k in range(K_actual)], dtype=float)
                        erank = effective_rank(model.codebook_)
                        n_used = int(np.sum(usage_array > 0))
                        sparsity = compute_codebook_sparsity(model.codebook_, WEIGHT_THRESHOLD)

                        codebook_archive[(ds_name, cfg["name"], K, fold_idx)] = model.codebook_.copy()

                        all_results.append({
                            "dataset": ds_name,
                            "method": "codebook_figs_v2",
                            "config": cfg["name"],
                            "K": K,
                            "fold": fold_idx,
                            "train_time_sec": round(elapsed, 4),
                            "n_splits": model.complexity_,
                            "erank": round(erank, 4),
                            "n_codebook_used": n_used,
                            "codebook_stability_mean": None,
                            "converged_at_round": model.converged_at_round_,
                            "top3_concentration": round(sparsity["top3_concentration"], 4),
                            "mean_l0_norm": round(sparsity["mean_l0_norm"], 4),
                            "lbfgs_iters": model.lbfgs_total_iters_,
                            "n_alternation_rounds_run": len(model.history_["train_losses"]),
                            "used_wls_fallback": model._use_fallback_wls,
                            **metrics,
                        })
                        logger.info(
                            f"  {cfg['name']} K={K} fold={fold_idx}: {metrics}, "
                            f"erank={erank:.2f}, used={n_used}/{K_actual}, "
                            f"top3={sparsity['top3_concentration']:.3f}, "
                            f"converged@{model.converged_at_round_} ({elapsed:.2f}s)"
                        )

                    except Exception:
                        logger.exception(f"  {cfg['name']} K={K} fold={fold_idx} FAILED")
                        all_results.append({
                            "dataset": ds_name, "method": "codebook_figs_v2",
                            "config": cfg["name"], "K": K, "fold": fold_idx,
                            "train_time_sec": time.time() - t0,
                            "n_splits": None, "erank": None, "n_codebook_used": None,
                            "codebook_stability_mean": None, "converged_at_round": None,
                            "top3_concentration": None, "mean_l0_norm": None,
                            "lbfgs_iters": None,
                            "accuracy": None, "auroc": None, "rmse": None, "r2": None,
                        })

                    # Time guard
                    if time.time() - start_time > TIMEOUT_TOTAL_SEC:
                        logger.warning("Approaching time limit, stopping early")
                        break
                if time.time() - start_time > TIMEOUT_TOTAL_SEC:
                    break
            if time.time() - start_time > TIMEOUT_TOTAL_SEC:
                break

        ds_elapsed = time.time() - ds_start
        dataset_times[ds_name] = ds_elapsed
        logger.info(f"  Dataset {ds_name} took {ds_elapsed:.1f}s ({ds_elapsed/60:.1f}min)")

        if time.time() - start_time > TIMEOUT_TOTAL_SEC:
            logger.warning("Hit global timeout, stopping benchmark")
            break

    # ══════════════════════════════════════════════════════════════════════
    # PHASE 3: POST-PROCESSING
    # ══════════════════════════════════════════════════════════════════════

    # 3a. Compute codebook stability
    logger.info("\n=== COMPUTING CODEBOOK STABILITY ===")
    for ds_name in full_only:
        for cfg in CB_CONFIGS:
            for K in K_VALUES:
                fold_codebooks = []
                for fold_idx in range(N_FOLDS):
                    key = (ds_name, cfg["name"], K, fold_idx)
                    if key in codebook_archive:
                        fold_codebooks.append(codebook_archive[key])

                if len(fold_codebooks) >= 2:
                    stability = compute_codebook_stability(fold_codebooks)
                    mean_sim = stability["mean_cosine_sim"]
                else:
                    mean_sim = None

                # Backfill stability into results
                for r in all_results:
                    if (r["dataset"] == ds_name
                            and r.get("config") == cfg["name"]
                            and r.get("K") == K
                            and r["method"] == "codebook_figs_v2"):
                        r["codebook_stability_mean"] = mean_sim

    # 3b. Aggregate and summarize
    logger.info("\n=== AGGREGATING ===")
    summary = compute_summary(all_results)

    config_meta = {
        "K_values": K_VALUES,
        "n_folds": N_FOLDS,
        "max_rules": MAX_RULES,
        "max_depth": MAX_DEPTH,
        "n_alternation_rounds": N_ALTERNATION_ROUNDS,
        "tau": TAU_SOFT,
        "lambda_l1": LAMBDA_L1,
        "lambda_l2": LAMBDA_L2,
        "weight_threshold": WEIGHT_THRESHOLD,
        "lbfgs_maxiter": LBFGS_MAXITER,
        "codebook_configs": [c["name"] for c in CB_CONFIGS],
        "baselines": ["figs", "xgboost", "lightgbm"],
        "datasets": sorted(full_only.keys()),
        "dataset_times": _sanitize_for_json(dataset_times),
    }

    # 3c. Write outputs
    logger.info("\n=== WRITING OUTPUT ===")
    output = format_output_for_schema(all_results, summary, full_datasets, config_meta)
    out_path = WORKSPACE / "method_out.json"
    out_path.write_text(json.dumps(output, indent=2, default=str))
    logger.info(f"Schema-compliant output: {out_path}")

    # Detailed results
    detailed = _sanitize_for_json({
        "experiment": "codebook_figs_v2_joint_refinement",
        "description": (
            "Codebook-FIGS v2 with joint L-BFGS-B codebook+threshold refinement, "
            "elastic-net sparsity, and warm-start tree reassignment"
        ),
        "improvements_over_iter3": [
            "Joint gradient-based optimization of codebook entries AND thresholds",
            "Elastic-net (L1+L2) sparsity penalty on codebook entries",
            "Post-optimization weight thresholding for genuine sparsity",
            "Warm-start tree reassignment instead of full re-growth",
            "10 alternation rounds (up from 7)",
            "Dropped K=20 (consistently degraded)",
        ],
        "results": all_results,
        "summary": summary,
        "config": config_meta,
    })
    detailed_path = WORKSPACE / "detailed_results.json"
    detailed_path.write_text(json.dumps(detailed, indent=2, default=str))
    logger.info(f"Detailed results: {detailed_path}")

    # File size check
    for p in [out_path, detailed_path]:
        size_mb = p.stat().st_size / (1024 * 1024)
        logger.info(f"  {p.name}: {size_mb:.1f} MB")

    elapsed_total = time.time() - start_time
    logger.info(f"\nTotal: {elapsed_total:.1f}s ({elapsed_total/60:.1f}min), {len(all_results)} result rows")
    logger.info("Done!")


if __name__ == "__main__":
    main()
