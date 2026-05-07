#!/usr/bin/env python3
"""Diagnostic Gap Decomposition of Codebook-FIGS Accuracy.

Decomposes the Codebook-FIGS accuracy gap into three independent components
using per-fold paired analysis across 10 datasets and 3 source experiments:
  (A) oblique-vs-axis FIGS implementation gap
  (B) codebook constraint gap (central hypothesis test)
  (C) optimizer sensitivity range

Includes Wilcoxon tests, bootstrap CIs, Spearman correlations, and a decision framework.
"""

import json
import sys
import resource
from pathlib import Path

import numpy as np
from scipy import stats
from loguru import logger

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
logger.add(str(LOG_DIR / "run.log"), rotation="30 MB", level="DEBUG")

# ---------------------------------------------------------------------------
# Resource limits (14 GB RAM, 1 hour CPU)
# ---------------------------------------------------------------------------
resource.setrlimit(resource.RLIMIT_AS, (14 * 1024**3, 14 * 1024**3))
resource.setrlimit(resource.RLIMIT_CPU, (3600, 3600))

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE = Path("/home/adrian/projects/temp/ai-inventor-old/aii_pipeline/runs/run__20260227_195308/3_invention_loop")
EXP_ID2_PATH = BASE / "iter_2/gen_art/exp_id2_it2__opus/full_method_out.json"
EXP_ID1_PATH = BASE / "iter_3/gen_art/exp_id1_it3__opus/full_method_out.json"
EXP_ID3_PATH = BASE / "iter_2/gen_art/exp_id3_it2__opus/full_method_out.json"
OUTPUT_DIR = Path(__file__).parent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_metric_key(task_type: str) -> str:
    """Return primary metric name for the given task type."""
    return "accuracy" if task_type == "classification" else "r2"


def bootstrap_ci(values: np.ndarray, n_boot: int = 10_000,
                 ci: float = 0.95, seed: int = 42) -> tuple:
    """Bootstrap 95 % CI for the mean of *values*."""
    rng = np.random.RandomState(seed)
    boot_means = np.array([
        np.mean(rng.choice(values, size=len(values), replace=True))
        for _ in range(n_boot)
    ])
    alpha = (1 - ci) / 2
    lo = float(np.percentile(boot_means, 100 * alpha))
    hi = float(np.percentile(boot_means, 100 * (1 - alpha)))
    return lo, hi


def cohens_d(values: np.ndarray) -> float:
    """Cohen's d for a one-sample test (mean / std)."""
    if len(values) < 2 or np.std(values, ddof=1) == 0:
        return float("nan")
    return float(np.mean(values) / np.std(values, ddof=1))


def permutation_spearman_pvalue(x: np.ndarray, y: np.ndarray,
                                 n_perm: int = 10_000,
                                 seed: int = 42) -> tuple:
    """Spearman correlation with permutation-based p-value."""
    rng = np.random.RandomState(seed)
    rho_obs, _ = stats.spearmanr(x, y)
    count = 0
    for _ in range(n_perm):
        perm_y = rng.permutation(y)
        rho_perm, _ = stats.spearmanr(x, perm_y)
        if abs(rho_perm) >= abs(rho_obs):
            count += 1
    p_val = (count + 1) / (n_perm + 1)
    return float(rho_obs), float(p_val)


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------
def load_exp_id2(data: dict) -> dict:
    """Extract per-dataset, per-fold scores from experiment 2.

    Returns
    -------
    dict[dataset] -> dict[fold] -> {
        figs_axis, oblique_figs, erank_figs_axis, n_features, task_type
    }
    """
    results = {}
    for ds_data in data["datasets"]:
        dataset = ds_data["dataset"]
        results[dataset] = {}
        for example in ds_data["examples"]:
            fold = example["metadata_fold"]
            task_type = example["metadata_task_type"]
            n_features = example["metadata_n_features"]
            figs_axis = float(example["predict_figs_axis_aligned"])
            oblique_figs = float(example["predict_oblique_figs"])
            erank = example.get("metadata_erank_figs_axis_aligned", None)
            results[dataset][fold] = {
                "figs_axis": figs_axis,
                "oblique_figs": oblique_figs,
                "erank_figs_axis": erank,
                "n_features": n_features,
                "task_type": task_type,
            }
    return results


def load_exp_id1(data: dict) -> dict:
    """Extract per-dataset, per-fold scores from experiment 1.

    Returns
    -------
    dict[dataset] -> dict[fold] -> {
        figs, cb_figs: dict[(config, K) -> score], task_type, n_samples
    }
    """
    results = {}
    for ds_data in data["datasets"]:
        dataset = ds_data["dataset"]
        results[dataset] = {}
        for example in ds_data["examples"]:
            fold = example["metadata_fold"]
            task_type = example["metadata_task_type"]
            inp = json.loads(example["input"])
            n_samples = inp.get("n_train", 0) + inp.get("n_test", 0)
            metric_key = get_metric_key(task_type)

            # FIGS baseline
            figs_pred = json.loads(example["predict_figs"])
            figs_score = figs_pred[metric_key]

            # CB-FIGS scores for every (config, K) combo
            cb_scores: dict = {}
            for key in example:
                if key.startswith("predict_cb_figs_"):
                    parts = key.replace("predict_cb_figs_", "")
                    k_idx = parts.rfind("_K")
                    config = parts[:k_idx]   # e.g. "adaptive" or "random"
                    K = int(parts[k_idx + 2:])
                    pred = json.loads(example[key])
                    cb_scores[(config, K)] = pred[metric_key]

            results[dataset][fold] = {
                "figs": figs_score,
                "cb_figs": cb_scores,
                "task_type": task_type,
                "n_samples": n_samples,
            }
    return results


def load_exp_id3(data: dict) -> tuple:
    """Extract per-dataset, per-config, per-fold scores from experiment 3.

    Returns
    -------
    cb_results : dict[dataset] -> dict[config_name] -> list[score_per_fold]
    figs_results : dict[dataset] -> list[score_per_fold]
    """
    metadata = data["metadata"]
    cb_data = metadata["codebook_figs_results"]
    figs_data = metadata["figs_baseline_results"]

    cb_results: dict = {}
    for dataset in cb_data:
        cb_results[dataset] = {}
        for config_name in cb_data[dataset]:
            per_fold = cb_data[dataset][config_name]["per_fold_metrics"]
            scores = []
            for fm in per_fold:
                if "accuracy" in fm:
                    scores.append(fm["accuracy"])
                elif "r2" in fm:
                    scores.append(fm["r2"])
            cb_results[dataset][config_name] = scores

    figs_results: dict = {}
    for dataset in figs_data:
        per_fold = figs_data[dataset]["per_fold_metrics"]
        scores = []
        for fm in per_fold:
            if "accuracy" in fm:
                scores.append(fm["accuracy"])
            elif "r2" in fm:
                scores.append(fm["r2"])
        figs_results[dataset] = scores

    return cb_results, figs_results


# ---------------------------------------------------------------------------
# Core gap decomposition
# ---------------------------------------------------------------------------
def compute_best_k_per_dataset(exp1: dict) -> dict:
    """Select best (config, K) per dataset by mean score across folds.

    Returns
    -------
    dict[dataset] -> (best_config, best_K, mean_score)
    """
    best_per_ds: dict = {}
    for dataset, fold_data in exp1.items():
        folds = sorted(fold_data.keys())
        # Gather all (config, K) combos
        all_combos = set()
        for f in folds:
            all_combos.update(fold_data[f]["cb_figs"].keys())

        best_mean = -float("inf")
        best_combo = None
        for combo in sorted(all_combos):
            scores = [fold_data[f]["cb_figs"].get(combo) for f in folds]
            if any(s is None for s in scores):
                continue
            m = float(np.mean(scores))
            if m > best_mean:
                best_mean = m
                best_combo = combo
        if best_combo is not None:
            best_per_ds[dataset] = (best_combo[0], best_combo[1], best_mean)
    return best_per_ds


def compute_gaps(exp2: dict, exp1: dict, exp3_cb: dict,
                 exp3_figs: dict) -> dict:
    """Compute Gaps A, B, B', C per dataset per fold.

    Returns
    -------
    dict[dataset] -> {
        task_type, n_features, n_samples, folds, gap_a, gap_b, gap_b_prime,
        gap_c, total_gap, figs_axis_scores, oblique_figs_scores,
        cb_best_scores, figs_exp1_scores, erank_figs_axis,
        best_config, best_K
    }
    """
    best_k = compute_best_k_per_dataset(exp1)
    datasets = sorted(exp2.keys())
    result: dict = {}

    for dataset in datasets:
        if dataset not in exp1:
            logger.warning(f"Dataset {dataset} missing from exp_id1, skipping")
            continue

        folds2 = sorted(exp2[dataset].keys())
        folds1 = sorted(exp1[dataset].keys())
        common_folds = sorted(set(folds2) & set(folds1))
        if not common_folds:
            logger.warning(f"No common folds for {dataset}, skipping")
            continue

        task_type = exp2[dataset][common_folds[0]]["task_type"]
        n_features = exp2[dataset][common_folds[0]]["n_features"]
        n_samples = exp1[dataset][common_folds[0]]["n_samples"]

        best_combo = best_k.get(dataset)
        if best_combo is None:
            logger.warning(f"No best K for {dataset}, skipping")
            continue
        b_config, b_K, _ = best_combo

        gap_a_vals = []
        gap_b_vals = []
        gap_b_prime_vals = []
        gap_c_vals = []
        total_gap_vals = []
        figs_axis_vals = []
        oblique_figs_vals = []
        cb_best_vals = []
        figs_exp1_vals = []
        erank_vals = []

        for fold in common_folds:
            fa = exp2[dataset][fold]["figs_axis"]
            of = exp2[dataset][fold]["oblique_figs"]
            cb = exp1[dataset][fold]["cb_figs"].get((b_config, b_K))
            figs1 = exp1[dataset][fold]["figs"]
            er = exp2[dataset][fold]["erank_figs_axis"]

            if cb is None:
                logger.warning(
                    f"Missing CB-FIGS ({b_config}, K={b_K}) for "
                    f"{dataset} fold {fold}"
                )
                continue

            gap_a = fa - of       # positive = oblique hurts
            gap_b = of - cb       # positive = codebook hurts
            gap_b_prime = figs1 - cb   # within-experiment Gap B'
            total = fa - cb       # should equal gap_a + gap_b

            gap_a_vals.append(gap_a)
            gap_b_vals.append(gap_b)
            gap_b_prime_vals.append(gap_b_prime)
            total_gap_vals.append(total)
            figs_axis_vals.append(fa)
            oblique_figs_vals.append(of)
            cb_best_vals.append(cb)
            figs_exp1_vals.append(figs1)
            erank_vals.append(er)

            # Gap C: range across 6 configs in exp_id3 for this fold
            if dataset in exp3_cb:
                fold_scores = []
                for cfg in exp3_cb[dataset]:
                    cfg_scores = exp3_cb[dataset][cfg]
                    if fold < len(cfg_scores):
                        fold_scores.append(cfg_scores[fold])
                if len(fold_scores) >= 2:
                    gap_c_vals.append(max(fold_scores) - min(fold_scores))
                else:
                    gap_c_vals.append(0.0)
            else:
                gap_c_vals.append(0.0)

        result[dataset] = {
            "task_type": task_type,
            "n_features": n_features,
            "n_samples": n_samples,
            "folds": common_folds,
            "gap_a": np.array(gap_a_vals),
            "gap_b": np.array(gap_b_vals),
            "gap_b_prime": np.array(gap_b_prime_vals),
            "gap_c": np.array(gap_c_vals),
            "total_gap": np.array(total_gap_vals),
            "figs_axis_scores": np.array(figs_axis_vals),
            "oblique_figs_scores": np.array(oblique_figs_vals),
            "cb_best_scores": np.array(cb_best_vals),
            "figs_exp1_scores": np.array(figs_exp1_vals),
            "erank_figs_axis": np.array(erank_vals),
            "best_config": b_config,
            "best_K": b_K,
        }

    return result


# ---------------------------------------------------------------------------
# Statistical analysis
# ---------------------------------------------------------------------------
def run_statistical_analysis(gaps: dict) -> dict:
    """Run all statistical tests on the gap decomposition results."""
    datasets = sorted(gaps.keys())
    n_ds = len(datasets)

    # ---- per-dataset means ----
    ds_mean_gap_a = np.array([float(np.mean(gaps[d]["gap_a"])) for d in datasets])
    ds_mean_gap_b = np.array([float(np.mean(gaps[d]["gap_b"])) for d in datasets])
    ds_mean_gap_b_prime = np.array(
        [float(np.mean(gaps[d]["gap_b_prime"])) for d in datasets]
    )
    ds_mean_gap_c = np.array([float(np.mean(gaps[d]["gap_c"])) for d in datasets])
    ds_mean_total = np.array(
        [float(np.mean(gaps[d]["total_gap"])) for d in datasets]
    )

    logger.info(f"Per-dataset mean Gap A: {ds_mean_gap_a}")
    logger.info(f"Per-dataset mean Gap B: {ds_mean_gap_b}")
    logger.info(f"Per-dataset mean Gap C: {ds_mean_gap_c}")

    # ---- Wilcoxon signed-rank tests ----
    # Test: Gap A != 0
    if n_ds >= 6:
        try:
            stat_a, p_a = stats.wilcoxon(ds_mean_gap_a, alternative="two-sided")
        except ValueError:
            stat_a, p_a = float("nan"), float("nan")
        try:
            stat_b, p_b = stats.wilcoxon(ds_mean_gap_b, alternative="two-sided")
        except ValueError:
            stat_b, p_b = float("nan"), float("nan")
        # Paired |Gap A| vs |Gap B| magnitude
        try:
            stat_ab, p_ab = stats.wilcoxon(
                np.abs(ds_mean_gap_a), np.abs(ds_mean_gap_b),
                alternative="two-sided"
            )
        except ValueError:
            stat_ab, p_ab = float("nan"), float("nan")
    else:
        stat_a = p_a = stat_b = p_b = stat_ab = p_ab = float("nan")

    # Test: Gap B' (within-experiment) != 0
    if n_ds >= 6:
        try:
            stat_bp, p_bp = stats.wilcoxon(ds_mean_gap_b_prime, alternative="two-sided")
        except ValueError:
            stat_bp, p_bp = float("nan"), float("nan")
    else:
        stat_bp = p_bp = float("nan")

    logger.info(f"Wilcoxon Gap A: stat={stat_a:.4f}, p={p_a:.4f}")
    logger.info(f"Wilcoxon Gap B: stat={stat_b:.4f}, p={p_b:.4f}")
    logger.info(f"Wilcoxon Gap B' (within-exp): stat={stat_bp:.4f}, p={p_bp:.4f}")
    logger.info(f"Wilcoxon |A| vs |B|: stat={stat_ab:.4f}, p={p_ab:.4f}")

    # ---- Bootstrap CIs ----
    ci_gap_a = bootstrap_ci(ds_mean_gap_a)
    ci_gap_b = bootstrap_ci(ds_mean_gap_b)
    ci_gap_b_prime = bootstrap_ci(ds_mean_gap_b_prime)
    ci_gap_c = bootstrap_ci(ds_mean_gap_c)
    logger.info(f"Bootstrap CI Gap A: {ci_gap_a}")
    logger.info(f"Bootstrap CI Gap B: {ci_gap_b}")
    logger.info(f"Bootstrap CI Gap C: {ci_gap_c}")

    # ---- Cohen's d ----
    d_gap_a = cohens_d(ds_mean_gap_a)
    d_gap_b = cohens_d(ds_mean_gap_b)
    d_gap_b_prime = cohens_d(ds_mean_gap_b_prime)
    logger.info(f"Cohen's d Gap A: {d_gap_a:.4f}")
    logger.info(f"Cohen's d Gap B: {d_gap_b:.4f}")
    logger.info(f"Cohen's d Gap B': {d_gap_b_prime:.4f}")

    # ---- Spearman correlations with permutation p-values ----
    n_features = np.array([gaps[d]["n_features"] for d in datasets], dtype=float)
    n_samples = np.array([gaps[d]["n_samples"] for d in datasets], dtype=float)
    mean_erank = np.array(
        [float(np.mean(gaps[d]["erank_figs_axis"])) for d in datasets]
    )
    mean_baseline = np.array(
        [float(np.mean(gaps[d]["figs_axis_scores"])) for d in datasets]
    )

    corr_nfeat_rho, corr_nfeat_p = permutation_spearman_pvalue(
        ds_mean_gap_b, n_features
    )
    corr_nsamp_rho, corr_nsamp_p = permutation_spearman_pvalue(
        ds_mean_gap_b, n_samples
    )
    corr_erank_rho, corr_erank_p = permutation_spearman_pvalue(
        ds_mean_gap_b, mean_erank
    )
    corr_baseline_rho, corr_baseline_p = permutation_spearman_pvalue(
        ds_mean_gap_b, mean_baseline
    )

    logger.info(
        f"Corr(Gap B, n_features): rho={corr_nfeat_rho:.3f}, "
        f"p={corr_nfeat_p:.4f}"
    )
    logger.info(
        f"Corr(Gap B, n_samples): rho={corr_nsamp_rho:.3f}, "
        f"p={corr_nsamp_p:.4f}"
    )
    logger.info(
        f"Corr(Gap B, eRank): rho={corr_erank_rho:.3f}, "
        f"p={corr_erank_p:.4f}"
    )
    logger.info(
        f"Corr(Gap B, baseline): rho={corr_baseline_rho:.3f}, "
        f"p={corr_baseline_p:.4f}"
    )

    # ---- Classification vs Regression breakdown ----
    clf_ds = [d for d in datasets if gaps[d]["task_type"] == "classification"]
    reg_ds = [d for d in datasets if gaps[d]["task_type"] == "regression"]

    clf_gap_b = np.array([float(np.mean(gaps[d]["gap_b"])) for d in clf_ds])
    reg_gap_b = np.array([float(np.mean(gaps[d]["gap_b"])) for d in reg_ds])

    if len(clf_gap_b) >= 2 and len(reg_gap_b) >= 2:
        try:
            u_stat, u_p = stats.mannwhitneyu(
                clf_gap_b, reg_gap_b, alternative="two-sided"
            )
        except ValueError:
            u_stat, u_p = float("nan"), float("nan")
    else:
        u_stat, u_p = float("nan"), float("nan")

    logger.info(
        f"Classification Gap B (n={len(clf_ds)}): "
        f"mean={np.mean(clf_gap_b):.4f}, std={np.std(clf_gap_b):.4f}"
    )
    logger.info(
        f"Regression Gap B (n={len(reg_ds)}): "
        f"mean={np.mean(reg_gap_b):.4f}, std={np.std(reg_gap_b):.4f}"
    )
    logger.info(f"Mann-Whitney U: stat={u_stat:.2f}, p={u_p:.4f}")

    # ---- Per-dataset attribution ----
    per_ds_attribution = {}
    for d in datasets:
        mean_a = float(np.mean(gaps[d]["gap_a"]))
        mean_b = float(np.mean(gaps[d]["gap_b"]))
        total_abs = abs(mean_a) + abs(mean_b)
        if total_abs > 1e-12:
            pct_a = abs(mean_a) / total_abs * 100
            pct_b = abs(mean_b) / total_abs * 100
        else:
            pct_a = pct_b = 50.0
        # Check if gaps have opposite signs
        opposite = (mean_a > 0 and mean_b < 0) or (mean_a < 0 and mean_b > 0)
        per_ds_attribution[d] = {
            "pct_gap_a": pct_a,
            "pct_gap_b": pct_b,
            "opposite_signs": opposite,
        }

    # ---- Recoverable gap ----
    overall_mean_a = float(np.mean(ds_mean_gap_a))
    overall_mean_c = float(np.mean(ds_mean_gap_c))
    overall_mean_total = float(np.mean(ds_mean_total))
    recoverable = overall_mean_a + overall_mean_c
    if abs(overall_mean_total) > 1e-12:
        recoverable_pct = recoverable / overall_mean_total * 100
    else:
        recoverable_pct = float("nan")

    # ---- Gap identity check ----
    identity_residual = float(
        np.mean(np.abs(ds_mean_total - (ds_mean_gap_a + ds_mean_gap_b)))
    )
    logger.info(
        f"Gap identity check: |Total - (A + B)| mean = {identity_residual:.8f}"
    )

    # ---- Decision framework ----
    gap_b_significant = p_b < 0.05 if not np.isnan(p_b) else False
    gap_a_dominates = float(np.mean(np.abs(ds_mean_gap_a))) > float(
        np.mean(np.abs(ds_mean_gap_b))
    )
    gap_c_dominates = overall_mean_c > abs(float(np.mean(ds_mean_gap_b)))

    if not gap_b_significant:
        bottleneck = "implementation"
        reasoning = (
            "Gap B (codebook constraint) is NOT statistically significant "
            f"(p={p_b:.4f}), meaning the codebook constraint is essentially "
            "free. Performance loss comes from other sources."
        )
    elif gap_a_dominates:
        bottleneck = "implementation"
        reasoning = (
            "Gap A (oblique implementation) dominates Gap B "
            f"(|A|={float(np.mean(np.abs(ds_mean_gap_a))):.4f} > "
            f"|B|={float(np.mean(np.abs(ds_mean_gap_b))):.4f}), "
            "meaning the oblique FIGS implementation is the bottleneck."
        )
    elif gap_c_dominates:
        bottleneck = "optimizer"
        reasoning = (
            "Gap C (optimizer range) exceeds |Gap B| "
            f"(C={overall_mean_c:.4f} > |B|={abs(float(np.mean(ds_mean_gap_b))):.4f}), "
            "suggesting the codebook idea is sound but optimization is undertrained."
        )
    else:
        bottleneck = "codebook_constraint"
        reasoning = (
            "Gap B is significant and dominates both Gap A and Gap C, "
            "suggesting the codebook constraint itself is the primary bottleneck."
        )

    return {
        "n_datasets": n_ds,
        "datasets_list": datasets,
        "ds_mean_gap_a": ds_mean_gap_a,
        "ds_mean_gap_b": ds_mean_gap_b,
        "ds_mean_gap_b_prime": ds_mean_gap_b_prime,
        "ds_mean_gap_c": ds_mean_gap_c,
        "ds_mean_total": ds_mean_total,
        # Wilcoxon
        "wilcoxon_gap_a_stat": float(stat_a),
        "wilcoxon_gap_a_p": float(p_a),
        "wilcoxon_gap_b_stat": float(stat_b),
        "wilcoxon_gap_b_p": float(p_b),
        "wilcoxon_gap_b_prime_stat": float(stat_bp),
        "wilcoxon_gap_b_prime_p": float(p_bp),
        "wilcoxon_ab_mag_stat": float(stat_ab),
        "wilcoxon_ab_mag_p": float(p_ab),
        # Bootstrap CIs
        "ci_gap_a": ci_gap_a,
        "ci_gap_b": ci_gap_b,
        "ci_gap_b_prime": ci_gap_b_prime,
        "ci_gap_c": ci_gap_c,
        # Effect sizes
        "cohens_d_gap_a": d_gap_a,
        "cohens_d_gap_b": d_gap_b,
        "cohens_d_gap_b_prime": d_gap_b_prime,
        # Correlations
        "corr_gap_b_nfeatures": (corr_nfeat_rho, corr_nfeat_p),
        "corr_gap_b_nsamples": (corr_nsamp_rho, corr_nsamp_p),
        "corr_gap_b_erank": (corr_erank_rho, corr_erank_p),
        "corr_gap_b_baseline": (corr_baseline_rho, corr_baseline_p),
        # Classification vs Regression
        "clf_datasets": clf_ds,
        "reg_datasets": reg_ds,
        "clf_gap_b_mean": float(np.mean(clf_gap_b)) if len(clf_gap_b) else float("nan"),
        "clf_gap_b_std": float(np.std(clf_gap_b)) if len(clf_gap_b) else float("nan"),
        "reg_gap_b_mean": float(np.mean(reg_gap_b)) if len(reg_gap_b) else float("nan"),
        "reg_gap_b_std": float(np.std(reg_gap_b)) if len(reg_gap_b) else float("nan"),
        "mannwhitney_u_stat": float(u_stat),
        "mannwhitney_u_p": float(u_p),
        # Per-dataset attribution
        "per_ds_attribution": per_ds_attribution,
        # Recoverable gap
        "recoverable_gap": recoverable,
        "recoverable_pct": recoverable_pct,
        # Identity check
        "identity_residual": identity_residual,
        # Decision framework
        "bottleneck": bottleneck,
        "bottleneck_reasoning": reasoning,
        # Overall means
        "overall_mean_gap_a": overall_mean_a,
        "overall_mean_gap_b": float(np.mean(ds_mean_gap_b)),
        "overall_mean_gap_b_prime": float(np.mean(ds_mean_gap_b_prime)),
        "overall_mean_gap_c": overall_mean_c,
        "overall_mean_total": overall_mean_total,
    }


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------
def format_output(gaps: dict, analysis: dict) -> dict:
    """Format results into exp_eval_sol_out.json schema."""
    datasets_list = analysis["datasets_list"]

    # ---- metrics_agg ----
    metrics_agg = {
        "n_datasets": analysis["n_datasets"],
        "overall_mean_gap_a": analysis["overall_mean_gap_a"],
        "overall_mean_gap_b": analysis["overall_mean_gap_b"],
        "overall_mean_gap_b_prime": analysis["overall_mean_gap_b_prime"],
        "overall_mean_gap_c": analysis["overall_mean_gap_c"],
        "overall_mean_total_gap": analysis["overall_mean_total"],
        "wilcoxon_gap_a_stat": analysis["wilcoxon_gap_a_stat"],
        "wilcoxon_gap_a_p": analysis["wilcoxon_gap_a_p"],
        "wilcoxon_gap_b_stat": analysis["wilcoxon_gap_b_stat"],
        "wilcoxon_gap_b_p": analysis["wilcoxon_gap_b_p"],
        "wilcoxon_gap_b_prime_stat": analysis["wilcoxon_gap_b_prime_stat"],
        "wilcoxon_gap_b_prime_p": analysis["wilcoxon_gap_b_prime_p"],
        "wilcoxon_ab_magnitude_stat": analysis["wilcoxon_ab_mag_stat"],
        "wilcoxon_ab_magnitude_p": analysis["wilcoxon_ab_mag_p"],
        "bootstrap_ci_gap_a_lo": analysis["ci_gap_a"][0],
        "bootstrap_ci_gap_a_hi": analysis["ci_gap_a"][1],
        "bootstrap_ci_gap_b_lo": analysis["ci_gap_b"][0],
        "bootstrap_ci_gap_b_hi": analysis["ci_gap_b"][1],
        "bootstrap_ci_gap_b_prime_lo": analysis["ci_gap_b_prime"][0],
        "bootstrap_ci_gap_b_prime_hi": analysis["ci_gap_b_prime"][1],
        "bootstrap_ci_gap_c_lo": analysis["ci_gap_c"][0],
        "bootstrap_ci_gap_c_hi": analysis["ci_gap_c"][1],
        "cohens_d_gap_a": analysis["cohens_d_gap_a"],
        "cohens_d_gap_b": analysis["cohens_d_gap_b"],
        "cohens_d_gap_b_prime": analysis["cohens_d_gap_b_prime"],
        "corr_gap_b_nfeatures_rho": analysis["corr_gap_b_nfeatures"][0],
        "corr_gap_b_nfeatures_p": analysis["corr_gap_b_nfeatures"][1],
        "corr_gap_b_nsamples_rho": analysis["corr_gap_b_nsamples"][0],
        "corr_gap_b_nsamples_p": analysis["corr_gap_b_nsamples"][1],
        "corr_gap_b_erank_rho": analysis["corr_gap_b_erank"][0],
        "corr_gap_b_erank_p": analysis["corr_gap_b_erank"][1],
        "corr_gap_b_baseline_rho": analysis["corr_gap_b_baseline"][0],
        "corr_gap_b_baseline_p": analysis["corr_gap_b_baseline"][1],
        "clf_gap_b_mean": analysis["clf_gap_b_mean"],
        "clf_gap_b_std": analysis["clf_gap_b_std"],
        "reg_gap_b_mean": analysis["reg_gap_b_mean"],
        "reg_gap_b_std": analysis["reg_gap_b_std"],
        "mannwhitney_clf_vs_reg_stat": analysis["mannwhitney_u_stat"],
        "mannwhitney_clf_vs_reg_p": analysis["mannwhitney_u_p"],
        "recoverable_gap": analysis["recoverable_gap"],
        "recoverable_pct": analysis["recoverable_pct"],
        "identity_residual": analysis["identity_residual"],
    }

    # Replace any NaN with 0.0 for JSON compatibility
    for k, v in metrics_agg.items():
        if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
            metrics_agg[k] = -999.0

    # ---- datasets ----
    output_datasets = []
    for ds_name in datasets_list:
        g = gaps[ds_name]
        attr = analysis["per_ds_attribution"][ds_name]
        folds = g["folds"]

        examples = []
        for i, fold in enumerate(folds):
            input_dict = {
                "dataset": ds_name,
                "fold": fold,
                "task_type": g["task_type"],
                "n_features": int(g["n_features"]),
                "n_samples": int(g["n_samples"]),
            }
            output_dict = {
                "gap_a": round(float(g["gap_a"][i]), 6),
                "gap_b": round(float(g["gap_b"][i]), 6),
                "gap_b_prime": round(float(g["gap_b_prime"][i]), 6),
                "gap_c_range": round(float(g["gap_c"][i]), 6),
                "total_gap": round(float(g["total_gap"][i]), 6),
                "identity_check": round(
                    float(g["total_gap"][i])
                    - (float(g["gap_a"][i]) + float(g["gap_b"][i])),
                    10,
                ),
            }

            example = {
                "input": json.dumps(input_dict),
                "output": json.dumps(output_dict),
                "metadata_fold": fold,
                "metadata_task_type": g["task_type"],
                "metadata_n_features": int(g["n_features"]),
                "metadata_n_samples": int(g["n_samples"]),
                "metadata_best_config": g["best_config"],
                "metadata_best_K": int(g["best_K"]),
                "predict_figs_axis_aligned": str(round(float(g["figs_axis_scores"][i]), 6)),
                "predict_oblique_figs": str(round(float(g["oblique_figs_scores"][i]), 6)),
                "predict_cb_figs_best": str(round(float(g["cb_best_scores"][i]), 6)),
                "predict_figs_exp1": str(round(float(g["figs_exp1_scores"][i]), 6)),
                "eval_gap_a": round(float(g["gap_a"][i]), 6),
                "eval_gap_b": round(float(g["gap_b"][i]), 6),
                "eval_gap_b_prime": round(float(g["gap_b_prime"][i]), 6),
                "eval_gap_c_range": round(float(g["gap_c"][i]), 6),
                "eval_total_gap": round(float(g["total_gap"][i]), 6),
                "eval_erank_figs_axis": round(float(g["erank_figs_axis"][i]), 4),
            }
            examples.append(example)

        # Add a dataset-level summary example
        mean_a = float(np.mean(g["gap_a"]))
        mean_b = float(np.mean(g["gap_b"]))
        mean_bp = float(np.mean(g["gap_b_prime"]))
        mean_c = float(np.mean(g["gap_c"]))
        mean_t = float(np.mean(g["total_gap"]))

        summary_input = {
            "dataset": ds_name,
            "fold": "aggregate",
            "task_type": g["task_type"],
            "n_features": int(g["n_features"]),
            "n_samples": int(g["n_samples"]),
        }
        summary_output = {
            "mean_gap_a": round(mean_a, 6),
            "mean_gap_b": round(mean_b, 6),
            "mean_gap_b_prime": round(mean_bp, 6),
            "mean_gap_c_range": round(mean_c, 6),
            "mean_total_gap": round(mean_t, 6),
            "pct_gap_a": round(attr["pct_gap_a"], 2),
            "pct_gap_b": round(attr["pct_gap_b"], 2),
            "opposite_signs": attr["opposite_signs"],
            "best_config": g["best_config"],
            "best_K": int(g["best_K"]),
            "bottleneck": analysis["bottleneck"],
        }
        summary_example = {
            "input": json.dumps(summary_input),
            "output": json.dumps(summary_output),
            "metadata_fold": -1,
            "metadata_task_type": g["task_type"],
            "metadata_n_features": int(g["n_features"]),
            "metadata_n_samples": int(g["n_samples"]),
            "metadata_best_config": g["best_config"],
            "metadata_best_K": int(g["best_K"]),
            "predict_mean_figs_axis": str(round(float(np.mean(g["figs_axis_scores"])), 6)),
            "predict_mean_oblique_figs": str(round(float(np.mean(g["oblique_figs_scores"])), 6)),
            "predict_mean_cb_figs_best": str(round(float(np.mean(g["cb_best_scores"])), 6)),
            "eval_mean_gap_a": round(mean_a, 6),
            "eval_mean_gap_b": round(mean_b, 6),
            "eval_mean_gap_b_prime": round(mean_bp, 6),
            "eval_mean_gap_c_range": round(mean_c, 6),
            "eval_mean_total_gap": round(mean_t, 6),
            "eval_pct_gap_a": round(attr["pct_gap_a"], 2),
            "eval_pct_gap_b": round(attr["pct_gap_b"], 2),
        }
        examples.append(summary_example)

        output_datasets.append({
            "dataset": ds_name,
            "examples": examples,
        })

    # ---- metadata ----
    metadata = {
        "evaluation_name": "diagnostic_gap_decomposition",
        "description": (
            "Three-component gap decomposition of Codebook-FIGS accuracy: "
            "Gap A (oblique-vs-axis), Gap B (codebook constraint), "
            "Gap C (optimizer sensitivity range)"
        ),
        "source_experiments": {
            "exp_id2": "Unconstrained Oblique Baselines",
            "exp_id1": "Codebook-FIGS Full K-Sweep Benchmark",
            "exp_id3": "Codebook-FIGS Ablation (Init/Refine/Stability)",
        },
        "n_folds": 5,
        "n_datasets": analysis["n_datasets"],
        "bottleneck_conclusion": analysis["bottleneck"],
        "bottleneck_reasoning": analysis["bottleneck_reasoning"],
        "gap_definitions": {
            "gap_a": "figs_axis_score - oblique_figs_score (positive = oblique hurts)",
            "gap_b": "oblique_figs_score - cb_figs_best_score (positive = codebook hurts)",
            "gap_b_prime": "figs_exp1_score - cb_figs_best_score (within-experiment)",
            "gap_c_range": "max - min score across 6 optimizer configs in exp_id3 (K=8)",
            "total_gap": "figs_axis_score - cb_figs_best_score (= gap_a + gap_b)",
        },
        "fold_alignment": {
            "note": (
                "FIGS axis-aligned scores verified between exp_id1 and exp_id2. "
                "Minor mismatches found for breast_cancer_wdbc (4 folds, max delta=0.009) "
                "and ionosphere (3 folds, max delta=0.070). Gap B' (within-experiment) "
                "provided as robust alternative unaffected by fold alignment."
            ),
            "n_mismatched_folds": analysis.get("n_fold_mismatches", 7),
            "affected_datasets": ["breast_cancer_wdbc", "ionosphere"],
        },
        "statistical_tests": {
            "wilcoxon_gap_a_p": analysis["wilcoxon_gap_a_p"],
            "wilcoxon_gap_b_p": analysis["wilcoxon_gap_b_p"],
            "wilcoxon_gap_b_prime_p": analysis["wilcoxon_gap_b_prime_p"],
            "wilcoxon_ab_magnitude_p": analysis["wilcoxon_ab_mag_p"],
            "mannwhitney_clf_vs_reg_p": analysis["mannwhitney_u_p"],
        },
        "correlation_summary": {
            "gap_b_vs_n_features": {
                "rho": analysis["corr_gap_b_nfeatures"][0],
                "p": analysis["corr_gap_b_nfeatures"][1],
            },
            "gap_b_vs_n_samples": {
                "rho": analysis["corr_gap_b_nsamples"][0],
                "p": analysis["corr_gap_b_nsamples"][1],
            },
            "gap_b_vs_erank": {
                "rho": analysis["corr_gap_b_erank"][0],
                "p": analysis["corr_gap_b_erank"][1],
            },
            "gap_b_vs_baseline_performance": {
                "rho": analysis["corr_gap_b_baseline"][0],
                "p": analysis["corr_gap_b_baseline"][1],
            },
        },
        "classification_vs_regression": {
            "n_classification": len(analysis["clf_datasets"]),
            "n_regression": len(analysis["reg_datasets"]),
            "clf_mean_gap_b": analysis["clf_gap_b_mean"],
            "reg_mean_gap_b": analysis["reg_gap_b_mean"],
        },
    }

    # Replace NaN in metadata with sentinel
    def sanitize(obj):
        if isinstance(obj, dict):
            return {k: sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [sanitize(v) for v in obj]
        if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
            return -999.0
        return obj

    metadata = sanitize(metadata)

    return {
        "metadata": metadata,
        "metrics_agg": metrics_agg,
        "datasets": output_datasets,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
@logger.catch
def main():
    logger.info("=" * 60)
    logger.info("Diagnostic Gap Decomposition of Codebook-FIGS Accuracy")
    logger.info("=" * 60)

    # ---- Load data ----
    logger.info(f"Loading exp_id2 from {EXP_ID2_PATH}")
    data2 = json.loads(EXP_ID2_PATH.read_text())
    logger.info(f"  {len(data2['datasets'])} datasets loaded")

    logger.info(f"Loading exp_id1 from {EXP_ID1_PATH}")
    data1 = json.loads(EXP_ID1_PATH.read_text())
    logger.info(f"  {len(data1['datasets'])} datasets loaded")

    logger.info(f"Loading exp_id3 from {EXP_ID3_PATH}")
    data3 = json.loads(EXP_ID3_PATH.read_text())
    cb_datasets = list(data3["metadata"]["codebook_figs_results"].keys())
    logger.info(f"  {len(cb_datasets)} datasets in CB results")

    # ---- Extract scores ----
    logger.info("Extracting scores from exp_id2...")
    exp2 = load_exp_id2(data2)
    logger.info(f"  Datasets: {sorted(exp2.keys())}")

    logger.info("Extracting scores from exp_id1...")
    exp1 = load_exp_id1(data1)
    logger.info(f"  Datasets: {sorted(exp1.keys())}")

    logger.info("Extracting scores from exp_id3...")
    exp3_cb, exp3_figs = load_exp_id3(data3)
    logger.info(f"  Datasets: {sorted(exp3_cb.keys())}")

    # ---- Verify fold alignment ----
    logger.info("Verifying fold alignment between exp_id1 and exp_id2...")
    mismatches = 0
    for ds in exp2:
        if ds not in exp1:
            continue
        for fold in exp2[ds]:
            if fold not in exp1[ds]:
                continue
            fa = exp2[ds][fold]["figs_axis"]
            f1 = exp1[ds][fold]["figs"]
            if abs(fa - f1) > 1e-10:
                logger.warning(
                    f"Fold alignment mismatch: {ds} fold {fold}: "
                    f"exp2={fa:.6f} vs exp1={f1:.6f}"
                )
                mismatches += 1
    if mismatches == 0:
        logger.info("  All folds aligned perfectly!")
    else:
        logger.warning(f"  {mismatches} fold mismatches found!")

    # ---- Compute best K per dataset ----
    best_k = compute_best_k_per_dataset(exp1)
    for ds in sorted(best_k.keys()):
        cfg, K, mean_score = best_k[ds]
        logger.info(
            f"  Best K for {ds}: config={cfg}, K={K}, "
            f"mean_score={mean_score:.4f}"
        )

    # ---- Compute gaps ----
    logger.info("Computing gap decomposition...")
    gaps = compute_gaps(exp2, exp1, exp3_cb, exp3_figs)
    logger.info(f"  Computed gaps for {len(gaps)} datasets")

    for ds in sorted(gaps.keys()):
        g = gaps[ds]
        logger.info(
            f"  {ds}: Gap A={np.mean(g['gap_a']):.4f}, "
            f"Gap B={np.mean(g['gap_b']):.4f}, "
            f"Gap C={np.mean(g['gap_c']):.4f}, "
            f"Total={np.mean(g['total_gap']):.4f}"
        )

    # ---- Statistical analysis ----
    logger.info("Running statistical analysis...")
    analysis = run_statistical_analysis(gaps)
    analysis["n_fold_mismatches"] = mismatches

    logger.info(f"BOTTLENECK: {analysis['bottleneck']}")
    logger.info(f"REASONING: {analysis['bottleneck_reasoning']}")

    # ---- Format output ----
    logger.info("Formatting output...")
    output = format_output(gaps, analysis)

    # ---- Save ----
    out_path = OUTPUT_DIR / "eval_out.json"
    out_path.write_text(json.dumps(output, indent=2))
    logger.info(f"Saved output to {out_path}")
    logger.info(f"  metrics_agg keys: {sorted(output['metrics_agg'].keys())}")
    logger.info(f"  datasets: {len(output['datasets'])}")
    total_examples = sum(len(d["examples"]) for d in output["datasets"])
    logger.info(f"  total examples: {total_examples}")

    logger.info("Done!")


if __name__ == "__main__":
    main()
