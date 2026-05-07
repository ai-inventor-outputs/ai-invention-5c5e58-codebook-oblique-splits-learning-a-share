#!/usr/bin/env python3
"""Comprehensive Statistical Evaluation of Codebook-FIGS Experiments.

Evaluates 3 experiments:
  exp_id1: Codebook-FIGS benchmark (10 datasets × 5 folds × K∈{3,5,8,12,20})
  exp_id2: Unconstrained oblique baselines (SPORF, ObliqueFIGS, FIGS axis-aligned)
  exp_id3: Ablation (3 init × 2 refine = 6 configs, stability, convergence)

Metrics computed:
  1. Paired Wilcoxon signed-rank tests (per dataset × per baseline)
  2. Friedman test + post-hoc Nemenyi
  3. Accuracy-vs-K curves with elbow detection
  4. Direction diversity compression ratios
  5. Ablation factor analysis
  6. Unified 7-method ranking with bootstrap CIs
  7. Gap analysis with failure characterization
  8. Hypothesis verdict
"""

from loguru import logger
from pathlib import Path
import json
import sys
import resource
import math
import itertools

import numpy as np
from scipy import stats

# ── Logging ──────────────────────────────────────────────────────────────
logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add("logs/run.log", rotation="30 MB", level="DEBUG")

# ── Resource Limits ──────────────────────────────────────────────────────
resource.setrlimit(resource.RLIMIT_AS, (14 * 1024**3, 14 * 1024**3))
resource.setrlimit(resource.RLIMIT_CPU, (3600, 3600))

# ── Paths ────────────────────────────────────────────────────────────────
WORKSPACE = Path(__file__).resolve().parent
ITER2_DIR = WORKSPACE.parents[2] / "iter_2" / "gen_art"
EXP1_DIR = ITER2_DIR / "exp_id1_it2__opus"
EXP2_DIR = ITER2_DIR / "exp_id2_it2__opus"
EXP3_DIR = ITER2_DIR / "exp_id3_it2__opus"

K_VALUES = [3, 5, 8, 12, 20]
DATASETS_ALL = [
    "heart_disease", "diabetes_pima", "breast_cancer_wdbc",
    "credit_german", "ionosphere", "spambase",
    "diabetes_regression", "california_housing", "auto_mpg", "wine_quality_red",
]
CLASSIFICATION_DATASETS = {
    "heart_disease", "diabetes_pima", "breast_cancer_wdbc",
    "credit_german", "ionosphere", "spambase",
}
REGRESSION_DATASETS = {
    "diabetes_regression", "california_housing", "auto_mpg", "wine_quality_red",
}
INIT_STRATEGIES = ["pca", "random", "lda"]
REFINE_STRATEGIES = ["wls", "lbfgs"]
CONFIGS_6 = [f"{i}_{r}" for i in INIT_STRATEGIES for r in REFINE_STRATEGIES]

# ── Data Loading ─────────────────────────────────────────────────────────

def load_json(path: Path) -> dict:
    logger.info(f"Loading {path.name} ({path.stat().st_size / 1e6:.1f} MB)")
    return json.loads(path.read_text())


def get_score_key(task_type: str) -> str:
    """Return the key used in predict JSON for this task type."""
    return "accuracy" if task_type == "classification" else "r2"


def extract_exp1_scores(exp1_data: dict) -> dict:
    """Extract per-dataset, per-fold scores for each method from exp_id1.

    Returns: {dataset: {method_key: [fold_scores]}}
      method_key in: figs, xgboost, lightgbm, codebook_figs_K3..K20
    """
    result = {}
    for ds_entry in exp1_data["datasets"]:
        ds_name = ds_entry["dataset"]
        task_type = ds_entry["examples"][0]["metadata_task_type"]
        score_key = get_score_key(task_type)
        methods_scores: dict[str, list[float]] = {}
        for ex in ds_entry["examples"]:
            for pred_key in ex:
                if not pred_key.startswith("predict_"):
                    continue
                method_name = pred_key.replace("predict_", "")
                pred_val = json.loads(ex[pred_key])
                score = pred_val.get(score_key)
                if score is None:
                    continue
                methods_scores.setdefault(method_name, []).append(score)
        result[ds_name] = methods_scores
    return result


def extract_exp1_eranks(exp1_data: dict) -> dict:
    """Extract per-dataset, per-fold eRank for codebook_figs at each K.

    Returns: {dataset: {K: [fold_eranks]}}
    """
    result = {}
    for ds_entry in exp1_data["datasets"]:
        ds_name = ds_entry["dataset"]
        k_eranks: dict[int, list[float]] = {}
        for ex in ds_entry["examples"]:
            for k in K_VALUES:
                pred_key = f"predict_codebook_figs_K{k}"
                if pred_key in ex:
                    pred_val = json.loads(ex[pred_key])
                    erank = pred_val.get("erank")
                    if erank is not None:
                        k_eranks.setdefault(k, []).append(erank)
        result[ds_name] = k_eranks
    return result


def extract_exp2_scores(exp2_data: dict) -> dict:
    """Extract per-dataset, per-fold scores and eRanks from exp_id2.

    Returns: {dataset: {method: {scores: [...], eranks: [...]}}}
    """
    methods = ["sporf_matched", "sporf_full", "oblique_figs", "figs_axis_aligned"]
    result = {}
    for ds_entry in exp2_data["datasets"]:
        ds_name = ds_entry["dataset"]
        ds_result: dict[str, dict] = {}
        for m in methods:
            ds_result[m] = {"scores": [], "eranks": []}
        for ex in ds_entry["examples"]:
            for m in methods:
                pred_key = f"predict_{m}"
                if pred_key in ex:
                    score = float(ex[pred_key])
                    ds_result[m]["scores"].append(score)
                erank_key = f"metadata_erank_{m}"
                if erank_key in ex:
                    ds_result[m]["eranks"].append(float(ex[erank_key]))
        result[ds_name] = ds_result
    return result


def extract_exp2_dataset_info(exp2_data: dict) -> dict:
    """Extract per-dataset metadata (task_type, n_features, n_samples) from exp_id2.

    Returns: {dataset: {task_type, n_features}}
    """
    result = {}
    for ds_entry in exp2_data["datasets"]:
        ds_name = ds_entry["dataset"]
        ex = ds_entry["examples"][0]
        inp = json.loads(ex["input"])
        result[ds_name] = {
            "task_type": inp.get("task_type", "unknown"),
            "n_features": inp.get("n_features", 0),
        }
    return result


# ── 1. Paired Wilcoxon Signed-Rank Tests ─────────────────────────────────

def compute_wilcoxon_tests(exp1_scores: dict) -> dict:
    """Paired Wilcoxon signed-rank tests: Codebook-FIGS (oracle-best K) vs each baseline."""
    logger.info("Computing Wilcoxon signed-rank tests")
    baselines = ["figs", "xgboost", "lightgbm"]
    results = {}

    for ds_name in DATASETS_ALL:
        if ds_name not in exp1_scores:
            logger.warning(f"Dataset {ds_name} not found in exp_id1 scores")
            continue
        ds_scores = exp1_scores[ds_name]

        # Find oracle-best K (highest mean score across folds)
        best_k = None
        best_mean = -float("inf")
        for k in K_VALUES:
            k_key = f"codebook_figs_K{k}"
            if k_key in ds_scores and len(ds_scores[k_key]) > 0:
                mean_score = np.mean(ds_scores[k_key])
                if mean_score > best_mean:
                    best_mean = mean_score
                    best_k = k

        if best_k is None:
            logger.warning(f"No codebook_figs scores for {ds_name}")
            continue

        cb_key = f"codebook_figs_K{best_k}"
        cb_scores = np.array(ds_scores[cb_key])
        ds_result = {"oracle_best_K": best_k, "codebook_figs_mean": float(best_mean)}
        tests = {}

        for baseline in baselines:
            if baseline not in ds_scores or len(ds_scores[baseline]) == 0:
                continue
            bl_scores = np.array(ds_scores[baseline])
            n = min(len(cb_scores), len(bl_scores))
            cb_arr = cb_scores[:n]
            bl_arr = bl_scores[:n]
            diff = cb_arr - bl_arr
            mean_diff = float(np.mean(diff))

            # Wilcoxon requires non-zero differences
            nonzero_diffs = diff[diff != 0]
            if len(nonzero_diffs) < 2:
                tests[baseline] = {
                    "statistic": None,
                    "p_value": 1.0,
                    "mean_diff": mean_diff,
                    "direction": "tie" if mean_diff == 0 else ("codebook_better" if mean_diff > 0 else "baseline_better"),
                    "n_nonzero": int(len(nonzero_diffs)),
                    "note": "Too few non-zero differences for Wilcoxon test",
                }
                continue

            try:
                stat, p_val = stats.wilcoxon(cb_arr, bl_arr, alternative="two-sided")
                tests[baseline] = {
                    "statistic": float(stat),
                    "p_value": float(p_val),
                    "mean_diff": mean_diff,
                    "direction": "codebook_better" if mean_diff > 0 else "baseline_better",
                    "n_nonzero": int(len(nonzero_diffs)),
                }
            except Exception as e:
                logger.exception(f"Wilcoxon failed for {ds_name} vs {baseline}")
                tests[baseline] = {
                    "statistic": None,
                    "p_value": 1.0,
                    "mean_diff": mean_diff,
                    "direction": "error",
                    "error": str(e),
                }

        ds_result["tests"] = tests
        results[ds_name] = ds_result

    return results


# ── 2. Friedman Test + Post-Hoc Nemenyi ──────────────────────────────────

def compute_friedman_test(exp1_scores: dict) -> dict:
    """Friedman test across 10 datasets for 4 methods with post-hoc Nemenyi."""
    logger.info("Computing Friedman test + Nemenyi post-hoc")

    methods = ["codebook_figs_best", "figs", "xgboost", "lightgbm"]
    # Build score matrix: (n_datasets, n_methods)
    score_matrix = []
    dataset_names_used = []

    for ds_name in DATASETS_ALL:
        if ds_name not in exp1_scores:
            continue
        ds_scores = exp1_scores[ds_name]

        # codebook_figs: oracle-best K
        best_mean = -float("inf")
        for k in K_VALUES:
            k_key = f"codebook_figs_K{k}"
            if k_key in ds_scores and len(ds_scores[k_key]) > 0:
                m = np.mean(ds_scores[k_key])
                if m > best_mean:
                    best_mean = m

        row = [best_mean]
        skip = False
        for m_name in ["figs", "xgboost", "lightgbm"]:
            if m_name in ds_scores and len(ds_scores[m_name]) > 0:
                row.append(float(np.mean(ds_scores[m_name])))
            else:
                skip = True
                break
        if skip or best_mean == -float("inf"):
            continue
        score_matrix.append(row)
        dataset_names_used.append(ds_name)

    score_matrix = np.array(score_matrix)  # (n_datasets, 4)
    n_datasets, n_methods = score_matrix.shape

    # Compute ranks per dataset (higher score = rank 1)
    rank_matrix = np.zeros_like(score_matrix)
    for i in range(n_datasets):
        rank_matrix[i] = stats.rankdata(-score_matrix[i])  # negative for descending

    mean_ranks = rank_matrix.mean(axis=0)

    # Friedman test
    try:
        chi2, p_value = stats.friedmanchisquare(*[rank_matrix[:, j] for j in range(n_methods)])
    except Exception as e:
        logger.exception("Friedman test failed")
        chi2, p_value = float("nan"), 1.0

    # Nemenyi critical difference
    # CD = q_alpha * sqrt(n_methods * (n_methods + 1) / (6 * n_datasets))
    # For alpha=0.05, n_methods=4: q_alpha ≈ 2.569 (from Nemenyi tables)
    q_alpha_table = {3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949}
    q_alpha = q_alpha_table.get(n_methods, 2.569)
    cd = q_alpha * math.sqrt(n_methods * (n_methods + 1) / (6 * n_datasets))

    # Pairwise comparisons
    pairwise = {}
    for i in range(n_methods):
        for j in range(i + 1, n_methods):
            diff = abs(mean_ranks[i] - mean_ranks[j])
            pair_key = f"{methods[i]}_vs_{methods[j]}"
            pairwise[pair_key] = {
                "rank_diff": float(diff),
                "critical_diff": float(cd),
                "significant": bool(diff > cd),
            }

    return {
        "chi2": float(chi2),
        "p_value": float(p_value),
        "n_datasets": int(n_datasets),
        "n_methods": int(n_methods),
        "mean_ranks": {methods[i]: float(mean_ranks[i]) for i in range(n_methods)},
        "critical_difference": float(cd),
        "pairwise_nemenyi": pairwise,
        "datasets_used": dataset_names_used,
    }


# ── 3. Accuracy-vs-K Curves with Elbow Detection ────────────────────────

def detect_elbow(k_vals: list[int], scores: list[float]) -> dict:
    """Detect elbow in accuracy-vs-K curve using piecewise linear fit."""
    k_arr = np.array(k_vals, dtype=float)
    s_arr = np.array(scores, dtype=float)

    if len(k_arr) < 3:
        return {"elbow_K": None, "rss_improvement_pct": 0.0, "has_clear_elbow": False}

    # Single linear fit RSS
    slope_full, intercept_full, _, _, _ = stats.linregress(k_arr, s_arr)
    pred_full = slope_full * k_arr + intercept_full
    rss_full = float(np.sum((s_arr - pred_full) ** 2))

    # Try piecewise breakpoints at each interior K
    best_rss_pw = rss_full
    best_break = None

    for bp_idx in range(1, len(k_arr) - 1):
        # Left segment: k_arr[:bp_idx+1], s_arr[:bp_idx+1]
        # Right segment: k_arr[bp_idx:], s_arr[bp_idx:]
        left_k = k_arr[: bp_idx + 1]
        left_s = s_arr[: bp_idx + 1]
        right_k = k_arr[bp_idx:]
        right_s = s_arr[bp_idx:]

        rss_left = 0.0
        rss_right = 0.0

        if len(left_k) >= 2:
            sl, il, _, _, _ = stats.linregress(left_k, left_s)
            rss_left = float(np.sum((left_s - (sl * left_k + il)) ** 2))
        if len(right_k) >= 2:
            sr, ir, _, _, _ = stats.linregress(right_k, right_s)
            rss_right = float(np.sum((right_s - (sr * right_k + ir)) ** 2))

        total_rss = rss_left + rss_right
        if total_rss < best_rss_pw:
            best_rss_pw = total_rss
            best_break = int(k_vals[bp_idx])

    rss_improvement = 0.0
    if rss_full > 1e-12:
        rss_improvement = (rss_full - best_rss_pw) / rss_full * 100.0

    # Saturation detection: if last 3 K values have same score (within 1e-6)
    saturated = False
    if len(scores) >= 3:
        last3 = scores[-3:]
        if max(last3) - min(last3) < 1e-6:
            saturated = True

    # Optimal K: the K with highest score
    optimal_k = int(k_vals[int(np.argmax(s_arr))])

    return {
        "elbow_K": best_break,
        "rss_improvement_pct": float(rss_improvement),
        "has_clear_elbow": rss_improvement > 50.0,
        "optimal_K": optimal_k,
        "saturated": saturated,
        "rss_single_line": float(rss_full),
        "rss_piecewise": float(best_rss_pw),
    }


def compute_accuracy_vs_k(exp1_scores: dict) -> dict:
    """Accuracy-vs-K analysis with elbow detection per dataset."""
    logger.info("Computing accuracy-vs-K curves with elbow detection")
    results = {}

    for ds_name in DATASETS_ALL:
        if ds_name not in exp1_scores:
            continue
        ds_scores = exp1_scores[ds_name]

        # Mean scores at each K
        k_means = []
        k_stds = []
        valid_ks = []
        for k in K_VALUES:
            k_key = f"codebook_figs_K{k}"
            if k_key in ds_scores and len(ds_scores[k_key]) > 0:
                valid_ks.append(k)
                k_means.append(float(np.mean(ds_scores[k_key])))
                k_stds.append(float(np.std(ds_scores[k_key])))

        if len(valid_ks) < 3:
            continue

        elbow = detect_elbow(valid_ks, k_means)
        results[ds_name] = {
            "K_values": valid_ks,
            "mean_scores": k_means,
            "std_scores": k_stds,
            **elbow,
        }

    return results


# ── 4. Direction Diversity Compression Ratios ────────────────────────────

def compute_compression_ratios(
    exp1_scores: dict,
    exp1_eranks: dict,
    exp2_scores: dict,
) -> dict:
    """Compute eRank compression ratio: eRank(unconstrained) / eRank(Codebook-FIGS best K)."""
    logger.info("Computing direction diversity compression ratios")
    baselines = ["sporf_matched", "sporf_full", "oblique_figs", "figs_axis_aligned"]
    results = {}

    for ds_name in DATASETS_ALL:
        if ds_name not in exp1_scores or ds_name not in exp2_scores:
            continue

        # Find best K for codebook_figs
        best_k = None
        best_mean = -float("inf")
        ds_scores = exp1_scores[ds_name]
        for k in K_VALUES:
            k_key = f"codebook_figs_K{k}"
            if k_key in ds_scores and len(ds_scores[k_key]) > 0:
                m = np.mean(ds_scores[k_key])
                if m > best_mean:
                    best_mean = m
                    best_k = k

        if best_k is None or ds_name not in exp1_eranks:
            continue

        # Mean eRank of Codebook-FIGS at best K
        cb_eranks = exp1_eranks[ds_name].get(best_k, [])
        if not cb_eranks:
            continue
        cb_mean_erank = float(np.mean(cb_eranks))

        ds_result = {"best_K": best_k, "codebook_figs_mean_erank": cb_mean_erank}
        ratios_per_baseline = {}

        for bl_name in baselines:
            if bl_name not in exp2_scores[ds_name]:
                continue
            bl_eranks = exp2_scores[ds_name][bl_name]["eranks"]
            if not bl_eranks:
                continue
            bl_mean_erank = float(np.mean(bl_eranks))
            ratio = bl_mean_erank / cb_mean_erank if cb_mean_erank > 1e-10 else float("nan")
            ratios_per_baseline[bl_name] = {
                "baseline_mean_erank": bl_mean_erank,
                "compression_ratio": float(ratio),
            }

        ds_result["baselines"] = ratios_per_baseline
        results[ds_name] = ds_result

    # Summary stats
    for bl_name in baselines:
        all_ratios = []
        for ds_name, ds_result in results.items():
            if bl_name in ds_result.get("baselines", {}):
                all_ratios.append(ds_result["baselines"][bl_name]["compression_ratio"])
        if all_ratios:
            results[f"summary_{bl_name}"] = {
                "mean_compression": float(np.mean(all_ratios)),
                "median_compression": float(np.median(all_ratios)),
                "min_compression": float(np.min(all_ratios)),
                "max_compression": float(np.max(all_ratios)),
                "n_datasets": len(all_ratios),
            }

    return results


# ── 5. Ablation Factor Analysis ─────────────────────────────────────────

def compute_ablation_analysis(exp3_data: dict) -> dict:
    """Analyze ablation: factor decomposition, stability, convergence."""
    logger.info("Computing ablation factor analysis")
    meta = exp3_data["metadata"]
    cb_results = meta["codebook_figs_results"]
    figs_baseline = meta.get("figs_baseline_results", {})

    # Collect per-dataset per-config scores
    # For classification: accuracy or accuracy-like; for regression: r2
    config_scores: dict[str, dict[str, float]] = {c: {} for c in CONFIGS_6}
    figs_scores: dict[str, float] = {}

    for ds_name in DATASETS_ALL:
        if ds_name not in cb_results:
            continue

        for config_name in CONFIGS_6:
            if config_name not in cb_results[ds_name]:
                continue
            agg = cb_results[ds_name][config_name]["aggregate_metrics"]
            # Use r2 for regression, accuracy for classification
            if ds_name in REGRESSION_DATASETS:
                score = agg.get("mean_r2", agg.get("mean_accuracy", 0))
            else:
                score = agg.get("mean_accuracy", agg.get("mean_r2", 0))
            config_scores[config_name][ds_name] = score

        if ds_name in figs_baseline:
            agg = figs_baseline[ds_name]["aggregate_metrics"]
            if ds_name in REGRESSION_DATASETS:
                figs_scores[ds_name] = agg.get("mean_r2", agg.get("mean_accuracy", 0))
            else:
                figs_scores[ds_name] = agg.get("mean_accuracy", agg.get("mean_r2", 0))

    # Mean rank of each config across datasets
    # For each dataset, rank 6 configs (rank 1 = best)
    rank_matrix = []
    datasets_for_ranking = []
    for ds_name in DATASETS_ALL:
        row_scores = []
        valid = True
        for config_name in CONFIGS_6:
            if ds_name not in config_scores[config_name]:
                valid = False
                break
            row_scores.append(config_scores[config_name][ds_name])
        if not valid:
            continue
        datasets_for_ranking.append(ds_name)
        # Rank descending (higher = better)
        rank_matrix.append(stats.rankdata(-np.array(row_scores)))

    rank_matrix = np.array(rank_matrix)
    mean_ranks = rank_matrix.mean(axis=0) if len(rank_matrix) > 0 else np.zeros(len(CONFIGS_6))

    config_mean_ranks = {CONFIGS_6[i]: float(mean_ranks[i]) for i in range(len(CONFIGS_6))}

    # Factor decomposition: init effect vs refine effect
    # Init effect: mean rank of pca_* vs random_* vs lda_*
    init_effect = {}
    for init_s in INIT_STRATEGIES:
        idxs = [i for i, c in enumerate(CONFIGS_6) if c.startswith(init_s)]
        if len(rank_matrix) > 0:
            init_effect[init_s] = float(np.mean(mean_ranks[idxs]))

    refine_effect = {}
    for ref_s in REFINE_STRATEGIES:
        idxs = [i for i, c in enumerate(CONFIGS_6) if c.endswith(ref_s)]
        if len(rank_matrix) > 0:
            refine_effect[ref_s] = float(np.mean(mean_ranks[idxs]))

    # Init effect size: range of init means
    init_range = max(init_effect.values()) - min(init_effect.values()) if init_effect else 0.0
    refine_range = max(refine_effect.values()) - min(refine_effect.values()) if refine_effect else 0.0

    # Pairwise Wilcoxon: init strategies
    init_wilcoxon = {}
    for i1, i2 in itertools.combinations(INIT_STRATEGIES, 2):
        scores_1 = []
        scores_2 = []
        for ds_name in datasets_for_ranking:
            # Average across refine strategies for this init
            vals_1 = [config_scores[f"{i1}_{r}"][ds_name] for r in REFINE_STRATEGIES if ds_name in config_scores[f"{i1}_{r}"]]
            vals_2 = [config_scores[f"{i2}_{r}"][ds_name] for r in REFINE_STRATEGIES if ds_name in config_scores[f"{i2}_{r}"]]
            if vals_1 and vals_2:
                scores_1.append(np.mean(vals_1))
                scores_2.append(np.mean(vals_2))
        if len(scores_1) >= 3:
            nonzero = [a - b for a, b in zip(scores_1, scores_2) if abs(a - b) > 1e-12]
            if len(nonzero) >= 2:
                try:
                    stat, pval = stats.wilcoxon(scores_1, scores_2)
                    init_wilcoxon[f"{i1}_vs_{i2}"] = {"statistic": float(stat), "p_value": float(pval)}
                except Exception:
                    init_wilcoxon[f"{i1}_vs_{i2}"] = {"statistic": None, "p_value": 1.0}
            else:
                init_wilcoxon[f"{i1}_vs_{i2}"] = {"statistic": None, "p_value": 1.0, "note": "too few nonzero diffs"}

    # Pairwise Wilcoxon: refine strategies
    refine_wilcoxon = {}
    scores_wls = []
    scores_lbfgs = []
    for ds_name in datasets_for_ranking:
        vals_wls = [config_scores[f"{i}_wls"][ds_name] for i in INIT_STRATEGIES if ds_name in config_scores[f"{i}_wls"]]
        vals_lbfgs = [config_scores[f"{i}_lbfgs"][ds_name] for i in INIT_STRATEGIES if ds_name in config_scores[f"{i}_lbfgs"]]
        if vals_wls and vals_lbfgs:
            scores_wls.append(np.mean(vals_wls))
            scores_lbfgs.append(np.mean(vals_lbfgs))
    if len(scores_wls) >= 3:
        nonzero = [a - b for a, b in zip(scores_wls, scores_lbfgs) if abs(a - b) > 1e-12]
        if len(nonzero) >= 2:
            try:
                stat, pval = stats.wilcoxon(scores_wls, scores_lbfgs)
                refine_wilcoxon["wls_vs_lbfgs"] = {"statistic": float(stat), "p_value": float(pval)}
            except Exception:
                refine_wilcoxon["wls_vs_lbfgs"] = {"statistic": None, "p_value": 1.0}
        else:
            refine_wilcoxon["wls_vs_lbfgs"] = {"statistic": None, "p_value": 1.0, "note": "too few nonzero diffs"}

    # Codebook stability analysis
    stability_results = {}
    n_above_threshold = 0
    all_mean_cosines = []

    for ds_name in DATASETS_ALL:
        if ds_name not in cb_results:
            continue
        ds_stab = {}
        for config_name in CONFIGS_6:
            if config_name not in cb_results[ds_name]:
                continue
            stab = cb_results[ds_name][config_name].get("codebook_stability", {})
            mean_cos = stab.get("mean_cosine_sim", 0)
            ds_stab[config_name] = float(mean_cos)
            all_mean_cosines.append(mean_cos)

        # Best config stability for this dataset
        if ds_stab:
            best_stab = max(ds_stab.values())
            stability_results[ds_name] = {
                "per_config": ds_stab,
                "best_stability": float(best_stab),
                "above_threshold_08": best_stab > 0.8,
            }
            if best_stab > 0.8:
                n_above_threshold += 1

    # Convergence analysis
    convergence_results = {}
    all_conv_rounds = []
    for ds_name in DATASETS_ALL:
        if ds_name not in cb_results:
            continue
        ds_conv = {}
        for config_name in CONFIGS_6:
            if config_name not in cb_results[ds_name]:
                continue
            conv = cb_results[ds_name][config_name].get("convergence", {})
            mean_round = conv.get("mean_converged_round", float("nan"))
            ds_conv[config_name] = float(mean_round)
            if not math.isnan(mean_round):
                all_conv_rounds.append(mean_round)
        convergence_results[ds_name] = ds_conv

    return {
        "config_mean_ranks": config_mean_ranks,
        "best_config": min(config_mean_ranks, key=config_mean_ranks.get) if config_mean_ranks else None,
        "init_effect": init_effect,
        "refine_effect": refine_effect,
        "init_effect_size": float(init_range),
        "refine_effect_size": float(refine_range),
        "init_wilcoxon": init_wilcoxon,
        "refine_wilcoxon": refine_wilcoxon,
        "stability": {
            "per_dataset": stability_results,
            "mean_cosine_all": float(np.mean(all_mean_cosines)) if all_mean_cosines else 0.0,
            "n_datasets_above_08": n_above_threshold,
            "n_datasets_total": len(stability_results),
        },
        "convergence": {
            "per_dataset": convergence_results,
            "mean_rounds_all": float(np.mean(all_conv_rounds)) if all_conv_rounds else float("nan"),
            "median_rounds_all": float(np.median(all_conv_rounds)) if all_conv_rounds else float("nan"),
        },
        "datasets_used": datasets_for_ranking,
        "codebook_figs_outperforms_figs": _count_outperforms(config_scores, figs_scores),
    }


def _count_outperforms(config_scores: dict, figs_scores: dict) -> dict:
    """Count datasets where best codebook_figs config beats FIGS baseline."""
    wins = 0
    total = 0
    details = {}
    for ds_name in DATASETS_ALL:
        best_score = -float("inf")
        best_cfg = None
        for config_name in CONFIGS_6:
            if ds_name in config_scores[config_name]:
                s = config_scores[config_name][ds_name]
                if s > best_score:
                    best_score = s
                    best_cfg = config_name
        if ds_name in figs_scores and best_cfg is not None:
            total += 1
            figs_s = figs_scores[ds_name]
            if best_score > figs_s:
                wins += 1
            details[ds_name] = {
                "best_config": best_cfg,
                "codebook_figs_score": float(best_score),
                "figs_score": float(figs_s),
                "codebook_wins": best_score > figs_s,
            }
    return {"wins": wins, "total": total, "details": details}


# ── 6. Unified Method Ranking with Bootstrap CIs ────────────────────────

def compute_unified_ranking(exp1_scores: dict, exp2_scores: dict) -> dict:
    """Rank 7 methods across 10 datasets with bootstrap CIs."""
    logger.info("Computing unified 7-method ranking with bootstrap CIs")
    methods_7 = [
        "codebook_figs", "figs", "xgboost", "lightgbm",
        "sporf_matched", "sporf_full", "oblique_figs",
    ]

    # Build score table: {method: {dataset: mean_score}}
    score_table: dict[str, dict[str, float]] = {m: {} for m in methods_7}

    for ds_name in DATASETS_ALL:
        # Codebook-FIGS: oracle-best K
        if ds_name in exp1_scores:
            best_mean = -float("inf")
            for k in K_VALUES:
                k_key = f"codebook_figs_K{k}"
                if k_key in exp1_scores[ds_name] and len(exp1_scores[ds_name][k_key]) > 0:
                    m = np.mean(exp1_scores[ds_name][k_key])
                    if m > best_mean:
                        best_mean = m
            if best_mean > -float("inf"):
                score_table["codebook_figs"][ds_name] = float(best_mean)

            for m_name in ["figs", "xgboost", "lightgbm"]:
                if m_name in exp1_scores[ds_name] and len(exp1_scores[ds_name][m_name]) > 0:
                    score_table[m_name][ds_name] = float(np.mean(exp1_scores[ds_name][m_name]))

        if ds_name in exp2_scores:
            for m_name in ["sporf_matched", "sporf_full", "oblique_figs"]:
                if m_name in exp2_scores[ds_name] and exp2_scores[ds_name][m_name]["scores"]:
                    score_table[m_name][ds_name] = float(np.mean(exp2_scores[ds_name][m_name]["scores"]))

    # Find datasets with all 7 methods
    complete_datasets = [
        ds for ds in DATASETS_ALL
        if all(ds in score_table[m] for m in methods_7)
    ]
    logger.info(f"Unified ranking: {len(complete_datasets)} datasets with all 7 methods")

    if len(complete_datasets) < 2:
        return {"error": "Not enough datasets with all 7 methods", "complete_datasets": complete_datasets}

    # Build matrix
    n_ds = len(complete_datasets)
    n_m = len(methods_7)
    score_mat = np.zeros((n_ds, n_m))
    for i, ds in enumerate(complete_datasets):
        for j, m in enumerate(methods_7):
            score_mat[i, j] = score_table[m][ds]

    # Rank per dataset (rank 1 = best)
    rank_mat = np.zeros_like(score_mat)
    for i in range(n_ds):
        rank_mat[i] = stats.rankdata(-score_mat[i])

    mean_ranks = rank_mat.mean(axis=0)

    # Bootstrap CIs (1000 resamples)
    rng = np.random.default_rng(42)
    n_bootstrap = 1000
    bootstrap_ranks = np.zeros((n_bootstrap, n_m))
    for b in range(n_bootstrap):
        idx = rng.choice(n_ds, size=n_ds, replace=True)
        bootstrap_ranks[b] = rank_mat[idx].mean(axis=0)

    ci_lower = np.percentile(bootstrap_ranks, 2.5, axis=0)
    ci_upper = np.percentile(bootstrap_ranks, 97.5, axis=0)

    ranking_results = {}
    for j, m in enumerate(methods_7):
        ranking_results[m] = {
            "mean_rank": float(mean_ranks[j]),
            "ci_lower": float(ci_lower[j]),
            "ci_upper": float(ci_upper[j]),
            "mean_score": float(score_mat[:, j].mean()),
        }

    # Sort by mean rank
    sorted_methods = sorted(ranking_results.items(), key=lambda x: x[1]["mean_rank"])

    return {
        "ranking": {m: v for m, v in sorted_methods},
        "n_datasets": n_ds,
        "n_bootstrap": n_bootstrap,
        "complete_datasets": complete_datasets,
        "score_matrix": {
            complete_datasets[i]: {methods_7[j]: float(score_mat[i, j]) for j in range(n_m)}
            for i in range(n_ds)
        },
    }


# ── 7. Gap Analysis ─────────────────────────────────────────────────────

def compute_gap_analysis(
    exp1_scores: dict,
    exp2_info: dict,
) -> dict:
    """Per-dataset gap analysis: best_baseline_score - codebook_figs_score."""
    logger.info("Computing gap analysis")
    baselines = ["figs", "xgboost", "lightgbm"]
    results = {}
    failures = []

    for ds_name in DATASETS_ALL:
        if ds_name not in exp1_scores:
            continue
        ds_scores = exp1_scores[ds_name]

        # Best codebook_figs score (oracle K)
        best_cb = -float("inf")
        best_k = None
        for k in K_VALUES:
            k_key = f"codebook_figs_K{k}"
            if k_key in ds_scores and len(ds_scores[k_key]) > 0:
                m = np.mean(ds_scores[k_key])
                if m > best_cb:
                    best_cb = m
                    best_k = k

        # Best baseline score
        best_bl = -float("inf")
        best_bl_name = None
        for bl in baselines:
            if bl in ds_scores and len(ds_scores[bl]) > 0:
                m = np.mean(ds_scores[bl])
                if m > best_bl:
                    best_bl = m
                    best_bl_name = bl

        if best_cb == -float("inf") or best_bl == -float("inf"):
            continue

        gap = float(best_bl - best_cb)
        is_failure = gap > 0.03  # > 3% gap

        info = exp2_info.get(ds_name, {})
        task_type = "classification" if ds_name in CLASSIFICATION_DATASETS else "regression"
        n_features = info.get("n_features", 0)

        entry = {
            "codebook_figs_score": float(best_cb),
            "codebook_figs_best_K": best_k,
            "best_baseline": best_bl_name,
            "best_baseline_score": float(best_bl),
            "gap": gap,
            "gap_pct": float(gap * 100),
            "is_failure": is_failure,
            "task_type": task_type,
            "n_features": n_features,
        }
        results[ds_name] = entry

        if is_failure:
            failures.append({
                "dataset": ds_name,
                "gap_pct": float(gap * 100),
                "task_type": task_type,
                "n_features": n_features,
                "best_baseline": best_bl_name,
            })

    # Failure characterization
    n_failures = len(failures)
    failure_summary = {
        "n_failures": n_failures,
        "n_total": len(results),
        "failure_rate": n_failures / len(results) if results else 0.0,
        "failures": failures,
    }

    if failures:
        # Characterize by task type
        cls_failures = [f for f in failures if f["task_type"] == "classification"]
        reg_failures = [f for f in failures if f["task_type"] == "regression"]
        failure_summary["n_classification_failures"] = len(cls_failures)
        failure_summary["n_regression_failures"] = len(reg_failures)
        failure_summary["mean_failure_gap_pct"] = float(np.mean([f["gap_pct"] for f in failures]))
        failure_summary["mean_failure_n_features"] = float(np.mean([f["n_features"] for f in failures])) if all(f["n_features"] > 0 for f in failures) else 0.0

    return {
        "per_dataset": results,
        "summary": failure_summary,
        "mean_gap_all": float(np.mean([r["gap"] for r in results.values()])) if results else 0.0,
        "mean_gap_pct_all": float(np.mean([r["gap_pct"] for r in results.values()])) if results else 0.0,
    }


# ── 8. Hypothesis Verdict ───────────────────────────────────────────────

def compute_hypothesis_verdict(
    wilcoxon: dict,
    friedman: dict,
    elbow: dict,
    compression: dict,
    ablation: dict,
    gap: dict,
    unified: dict,
) -> dict:
    """Assess the 5 success criteria of the Codebook-FIGS hypothesis."""
    logger.info("Computing hypothesis verdict")
    criteria = {}

    # Criterion 1: Accuracy match within 1% (strict) / 3% (relaxed) of baselines
    gap_data = gap.get("per_dataset", {})
    n_within_1pct = sum(1 for d in gap_data.values() if abs(d["gap_pct"]) <= 1.0)
    n_within_3pct = sum(1 for d in gap_data.values() if abs(d["gap_pct"]) <= 3.0)
    n_total = len(gap_data)
    criteria["accuracy_match_1pct"] = {
        "n_pass": n_within_1pct,
        "n_total": n_total,
        "pass_rate": n_within_1pct / n_total if n_total > 0 else 0.0,
        "met": n_within_1pct / n_total >= 0.7 if n_total > 0 else False,
    }
    criteria["accuracy_match_3pct"] = {
        "n_pass": n_within_3pct,
        "n_total": n_total,
        "pass_rate": n_within_3pct / n_total if n_total > 0 else 0.0,
        "met": n_within_3pct / n_total >= 0.7 if n_total > 0 else False,
    }

    # Criterion 2: 3-10× direction reduction
    compression_ratios = []
    for ds_name, ds_data in compression.items():
        if ds_name.startswith("summary_") or not isinstance(ds_data, dict):
            continue
        for bl_name, bl_data in ds_data.get("baselines", {}).items():
            cr = bl_data.get("compression_ratio", 0)
            if not math.isnan(cr) and cr > 0:
                compression_ratios.append(cr)
    mean_cr = float(np.mean(compression_ratios)) if compression_ratios else 0.0
    criteria["direction_reduction_3_10x"] = {
        "mean_compression_ratio": mean_cr,
        "n_ratios": len(compression_ratios),
        "n_in_range_3_10": sum(1 for r in compression_ratios if 3 <= r <= 10),
        "n_above_3x": sum(1 for r in compression_ratios if r >= 3),
        "met": mean_cr >= 3.0,
    }

    # Criterion 3: Stability > 0.8
    stab = ablation.get("stability", {})
    n_above_08 = stab.get("n_datasets_above_08", 0)
    n_ds_total = stab.get("n_datasets_total", 0)
    mean_cos = stab.get("mean_cosine_all", 0)
    criteria["stability_above_08"] = {
        "n_above_threshold": n_above_08,
        "n_total": n_ds_total,
        "mean_cosine": float(mean_cos),
        "met": mean_cos > 0.8 or (n_ds_total > 0 and n_above_08 / n_ds_total >= 0.5),
    }

    # Criterion 4: Clear elbow existence
    n_clear_elbows = sum(1 for d in elbow.values() if d.get("has_clear_elbow", False))
    n_elbow_datasets = len(elbow)
    criteria["clear_elbow"] = {
        "n_clear_elbows": n_clear_elbows,
        "n_total": n_elbow_datasets,
        "elbow_rate": n_clear_elbows / n_elbow_datasets if n_elbow_datasets > 0 else 0.0,
        "met": n_elbow_datasets > 0 and n_clear_elbows / n_elbow_datasets >= 0.3,
    }

    # Criterion 5: Overall competitive ranking
    if isinstance(unified, dict) and "ranking" in unified:
        cb_rank_data = unified["ranking"].get("codebook_figs", {})
        cb_mean_rank = cb_rank_data.get("mean_rank", 99)
    else:
        cb_mean_rank = 99
    criteria["competitive_ranking"] = {
        "codebook_figs_mean_rank": float(cb_mean_rank),
        "met": cb_mean_rank <= 4.0,
    }

    # Composite verdict
    n_criteria_met = sum(1 for c in criteria.values() if c.get("met", False))
    n_criteria_total = len(criteria)

    verdict = "CONFIRMED" if n_criteria_met >= 4 else ("PARTIALLY_CONFIRMED" if n_criteria_met >= 2 else "DISCONFIRMED")

    return {
        "criteria": criteria,
        "n_criteria_met": n_criteria_met,
        "n_criteria_total": n_criteria_total,
        "verdict": verdict,
    }


# ── Output Formatting ────────────────────────────────────────────────────

def format_output(
    wilcoxon: dict,
    friedman: dict,
    elbow: dict,
    compression: dict,
    ablation: dict,
    unified: dict,
    gap: dict,
    verdict: dict,
    exp1_scores: dict,
    exp2_scores: dict,
    exp2_info: dict,
) -> dict:
    """Format all results into the exp_eval_sol_out.json schema."""
    logger.info("Formatting output to schema")

    # ── metrics_agg ──
    gap_summary = gap.get("summary", {})
    stab = ablation.get("stability", {})
    conv = ablation.get("convergence", {})

    # Collect all compression ratios for aggregate
    all_comp_ratios = []
    for ds_name, ds_data in compression.items():
        if ds_name.startswith("summary_") or not isinstance(ds_data, dict):
            continue
        for bl_name, bl_data in ds_data.get("baselines", {}).items():
            cr = bl_data.get("compression_ratio", 0)
            if not math.isnan(cr) and cr > 0:
                all_comp_ratios.append(cr)

    metrics_agg = {
        "friedman_chi2": float(friedman.get("chi2", 0)),
        "friedman_p_value": float(friedman.get("p_value", 1.0)),
        "friedman_n_datasets": int(friedman.get("n_datasets", 0)),
        "mean_gap_pct": float(gap.get("mean_gap_pct_all", 0)),
        "n_failures_above_3pct": int(gap_summary.get("n_failures", 0)),
        "failure_rate": float(gap_summary.get("failure_rate", 0)),
        "mean_compression_ratio": float(np.mean(all_comp_ratios)) if all_comp_ratios else 0.0,
        "median_compression_ratio": float(np.median(all_comp_ratios)) if all_comp_ratios else 0.0,
        "mean_stability_cosine": float(stab.get("mean_cosine_all", 0)),
        "n_datasets_stability_above_08": int(stab.get("n_datasets_above_08", 0)),
        "mean_convergence_rounds": float(conv.get("mean_rounds_all", 0)) if not math.isnan(conv.get("mean_rounds_all", 0)) else 0.0,
        "n_clear_elbows": int(sum(1 for d in elbow.values() if d.get("has_clear_elbow", False))),
        "n_criteria_met": int(verdict.get("n_criteria_met", 0)),
        "n_criteria_total": int(verdict.get("n_criteria_total", 0)),
        "hypothesis_confirmed": 1 if verdict.get("verdict") == "CONFIRMED" else 0,
    }

    # Add per-baseline Wilcoxon summary
    baselines = ["figs", "xgboost", "lightgbm"]
    for bl in baselines:
        sig_count = 0
        n_tests = 0
        for ds_name, ds_res in wilcoxon.items():
            tests = ds_res.get("tests", {})
            if bl in tests:
                n_tests += 1
                if tests[bl].get("p_value", 1.0) < 0.05:
                    sig_count += 1
        metrics_agg[f"wilcoxon_sig_vs_{bl}"] = sig_count
        metrics_agg[f"wilcoxon_n_tests_vs_{bl}"] = n_tests

    # Codebook-FIGS rank from unified ranking
    if isinstance(unified, dict) and "ranking" in unified:
        cb_rank = unified["ranking"].get("codebook_figs", {}).get("mean_rank", 0)
        metrics_agg["codebook_figs_unified_mean_rank"] = float(cb_rank)

    # Best ablation config rank
    if ablation.get("best_config"):
        metrics_agg["ablation_best_config_rank"] = float(
            ablation["config_mean_ranks"].get(ablation["best_config"], 0)
        )

    # ── datasets ──
    datasets_out = []
    for ds_name in DATASETS_ALL:
        examples = []

        # Build a single evaluation example per dataset
        input_dict = {
            "dataset": ds_name,
            "task_type": "classification" if ds_name in CLASSIFICATION_DATASETS else "regression",
            "n_features": exp2_info.get(ds_name, {}).get("n_features", 0),
        }

        # Compile evaluation results for this dataset into a summary
        ds_eval: dict[str, float] = {}
        ds_predict: dict[str, str] = {}

        # Wilcoxon results
        if ds_name in wilcoxon:
            w_res = wilcoxon[ds_name]
            ds_eval["eval_oracle_best_K"] = float(w_res.get("oracle_best_K", 0))
            ds_eval["eval_codebook_figs_mean_score"] = float(w_res.get("codebook_figs_mean", 0))
            for bl in baselines:
                if bl in w_res.get("tests", {}):
                    t = w_res["tests"][bl]
                    ds_eval[f"eval_wilcoxon_pval_vs_{bl}"] = float(t.get("p_value", 1.0))
                    ds_eval[f"eval_wilcoxon_mean_diff_vs_{bl}"] = float(t.get("mean_diff", 0))

        # Gap analysis
        if ds_name in gap.get("per_dataset", {}):
            g = gap["per_dataset"][ds_name]
            ds_eval["eval_gap_pct"] = float(g["gap_pct"])
            ds_eval["eval_is_failure"] = 1 if g["is_failure"] else 0

        # Elbow
        if ds_name in elbow:
            e = elbow[ds_name]
            ds_eval["eval_has_clear_elbow"] = 1 if e.get("has_clear_elbow", False) else 0
            ds_eval["eval_optimal_K"] = float(e.get("optimal_K", 0))
            ds_eval["eval_elbow_rss_improvement_pct"] = float(e.get("rss_improvement_pct", 0))

        # Compression
        if ds_name in compression and isinstance(compression[ds_name], dict):
            c = compression[ds_name]
            ds_eval["eval_codebook_figs_erank"] = float(c.get("codebook_figs_mean_erank", 0))
            for bl_name in ["sporf_matched", "sporf_full", "oblique_figs", "figs_axis_aligned"]:
                if bl_name in c.get("baselines", {}):
                    cr = c["baselines"][bl_name]["compression_ratio"]
                    ds_eval[f"eval_compression_ratio_{bl_name}"] = float(cr) if not math.isnan(cr) else 0.0

        # Ablation stability
        stab_ds = ablation.get("stability", {}).get("per_dataset", {}).get(ds_name, {})
        if stab_ds:
            ds_eval["eval_best_stability"] = float(stab_ds.get("best_stability", 0))
            ds_eval["eval_stability_above_08"] = 1 if stab_ds.get("above_threshold_08", False) else 0

        # Predict fields: mean scores per method
        if ds_name in exp1_scores:
            sc = exp1_scores[ds_name]
            for bl in baselines:
                if bl in sc and sc[bl]:
                    ds_predict[f"predict_{bl}"] = json.dumps({"mean_score": float(np.mean(sc[bl]))})
            # Best codebook_figs
            best_k_score = -float("inf")
            best_k_val = None
            for k in K_VALUES:
                k_key = f"codebook_figs_K{k}"
                if k_key in sc and sc[k_key]:
                    m = float(np.mean(sc[k_key]))
                    if m > best_k_score:
                        best_k_score = m
                        best_k_val = k
            if best_k_val is not None:
                ds_predict["predict_codebook_figs_best"] = json.dumps({
                    "mean_score": best_k_score, "best_K": best_k_val
                })

        if ds_name in exp2_scores:
            for m_name in ["sporf_matched", "sporf_full", "oblique_figs"]:
                if m_name in exp2_scores[ds_name] and exp2_scores[ds_name][m_name]["scores"]:
                    ds_predict[f"predict_{m_name}"] = json.dumps({
                        "mean_score": float(np.mean(exp2_scores[ds_name][m_name]["scores"]))
                    })

        # Build the output string: summary of all evaluations for this dataset
        output_summary = {
            "dataset": ds_name,
            "verdict_gap_pct": ds_eval.get("eval_gap_pct", 0),
            "oracle_best_K": ds_eval.get("eval_oracle_best_K", 0),
            "has_clear_elbow": ds_eval.get("eval_has_clear_elbow", 0),
            "codebook_figs_mean_score": ds_eval.get("eval_codebook_figs_mean_score", 0),
        }

        example = {
            "input": json.dumps(input_dict),
            "output": json.dumps(output_summary),
            **ds_predict,
            **{f"metadata_{k.replace('eval_', '')}": v for k, v in ds_eval.items() if k.startswith("eval_")},
        }
        # Add eval_ fields
        for k, v in ds_eval.items():
            example[k] = v

        examples.append(example)
        datasets_out.append({"dataset": ds_name, "examples": examples})

    # ── Full output ──
    output = {
        "metadata": {
            "evaluation": "codebook_figs_comprehensive_statistical_evaluation",
            "description": "Comprehensive statistical evaluation of Codebook-FIGS across 3 experiments",
            "experiments": ["exp_id1 (benchmark)", "exp_id2 (oblique baselines)", "exp_id3 (ablation)"],
            "n_datasets": len(DATASETS_ALL),
            "wilcoxon_tests": wilcoxon,
            "friedman_test": friedman,
            "accuracy_vs_K": elbow,
            "compression_ratios": compression,
            "ablation_analysis": ablation,
            "unified_ranking": unified,
            "gap_analysis": gap,
            "hypothesis_verdict": verdict,
        },
        "metrics_agg": metrics_agg,
        "datasets": datasets_out,
    }

    return output


# ── Main ─────────────────────────────────────────────────────────────────

@logger.catch
def main():
    logger.info("=" * 60)
    logger.info("Codebook-FIGS Comprehensive Statistical Evaluation")
    logger.info("=" * 60)

    # Load data
    exp1_data = load_json(EXP1_DIR / "full_method_out.json")
    exp2_data = load_json(EXP2_DIR / "full_method_out.json")
    exp3_data = load_json(EXP3_DIR / "full_method_out.json")

    logger.info(f"Exp1: {len(exp1_data['datasets'])} datasets")
    logger.info(f"Exp2: {len(exp2_data['datasets'])} datasets")
    logger.info(f"Exp3: {len(exp3_data['datasets'])} datasets")

    # Extract scores
    exp1_scores = extract_exp1_scores(exp1_data)
    exp1_eranks = extract_exp1_eranks(exp1_data)
    exp2_scores = extract_exp2_scores(exp2_data)
    exp2_info = extract_exp2_dataset_info(exp2_data)

    logger.info(f"Exp1 scores extracted for {len(exp1_scores)} datasets")
    logger.info(f"Exp2 scores extracted for {len(exp2_scores)} datasets")

    # 1. Wilcoxon tests
    wilcoxon = compute_wilcoxon_tests(exp1_scores)
    logger.info(f"Wilcoxon tests computed for {len(wilcoxon)} datasets")

    # 2. Friedman test
    friedman = compute_friedman_test(exp1_scores)
    logger.info(f"Friedman chi2={friedman['chi2']:.3f}, p={friedman['p_value']:.4f}")

    # 3. Accuracy-vs-K elbow
    elbow = compute_accuracy_vs_k(exp1_scores)
    n_elbows = sum(1 for d in elbow.values() if d.get("has_clear_elbow"))
    logger.info(f"Elbow detection: {n_elbows}/{len(elbow)} datasets have clear elbow")

    # 4. Compression ratios
    compression = compute_compression_ratios(exp1_scores, exp1_eranks, exp2_scores)
    logger.info(f"Compression ratios computed for {len([k for k in compression if not k.startswith('summary_')])} datasets")

    # 5. Ablation analysis
    ablation = compute_ablation_analysis(exp3_data)
    logger.info(f"Ablation best config: {ablation['best_config']}")

    # 6. Unified ranking
    unified = compute_unified_ranking(exp1_scores, exp2_scores)
    if "ranking" in unified:
        for m, v in list(unified["ranking"].items())[:3]:
            logger.info(f"  Rank {v['mean_rank']:.2f}: {m}")

    # 7. Gap analysis
    gap = compute_gap_analysis(exp1_scores, exp2_info)
    logger.info(f"Gap analysis: mean gap = {gap['mean_gap_pct_all']:.2f}%")

    # 8. Hypothesis verdict
    verdict = compute_hypothesis_verdict(
        wilcoxon, friedman, elbow, compression, ablation, gap, unified
    )
    logger.info(f"Hypothesis verdict: {verdict['verdict']} ({verdict['n_criteria_met']}/{verdict['n_criteria_total']} criteria met)")

    # Format output
    output = format_output(
        wilcoxon, friedman, elbow, compression, ablation, unified, gap, verdict,
        exp1_scores, exp2_scores, exp2_info,
    )

    # Save
    out_path = WORKSPACE / "eval_out.json"
    out_path.write_text(json.dumps(output, indent=2))
    logger.info(f"Saved evaluation to {out_path} ({out_path.stat().st_size / 1e6:.2f} MB)")

    logger.info("=" * 60)
    logger.info("DONE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
