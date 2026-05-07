# Oblique Trees

## Summary

Comprehensive survey of 10 oblique decision tree methods (SPORF, RO-FIGS, SREDT, FoLDTree, FC-ODT, FIGS, OC1, HHCART, CART-ELC, ODRF) covering code availability, benchmark results on 22 OpenML datasets, direction diversity metrics (eRank, stable rank, participation ratio), interpretability measures, and codebook-size K selection heuristics. Provides actionable recommendations for Codebook-FIGS evaluation design including primary benchmark suite, K-sweep strategy, and novel metrics.

## Research Findings

## Oblique Decision Tree Landscape for Codebook-FIGS Evaluation

### 1. Implementations and Code Availability

The oblique decision tree landscape includes 10 methods spanning 30 years. Among actively maintained Python packages, **SPORF** (Sparse Projection Oblique Randomer Forests) is available via `pip install treeple` (v0.10.3), using sparse random projections drawn from {+1, −1} and supporting both classification and regression [1, 2]. The treeple package is actively maintained with the last push in February 2026 [2], though the original SPORF repo by neurodata is effectively abandoned (last commit December 2019) [3]. The key class is `treeple.ObliqueRandomForestClassifier` with hyperparameters `max_features=p`, `feature_combinations=3.0` (λ=3/p), and `n_estimators=500` [1].

**RO-FIGS** (Random Oblique FIGS) extends FIGS with gradient-descent oblique splits using L-1/2 regularization for sparsity [4]. Code is available at https://github.com/um-k/rofigs as standalone Python (Apache-2.0) [5], though the repo has only 1 star and 10 commits, raising fragility concerns. It is classification-only and not integrated into imodels [4, 5].

**FIGS** (Fast Interpretable Greedy-Tree Sums) is the base method available via `pip install imodels` (v2.0.4) with sklearn-compatible `FIGSClassifier` and `FIGSRegressor` [6]. It uses axis-parallel CART stumps. The critical extension point for Codebook-FIGS is `_construct_node_with_stump()`, which internally calls `DecisionTreeRegressor(max_depth=1)` — this single method can be replaced with a codebook-constrained split search while preserving FIGS' priority queue, residual tracking, and tree competition infrastructure [6].

**SREDT** (Symbolic Regression Enhanced Decision Trees) uses gplearn-based genetic programming to evolve nonlinear split expressions [7]. It was published at AAAI 2024 but has **no public code** [7]. Key hyperparameters include parsimony coefficient 0.001, 40 generations, population size 400, and tournament size 200 with primitives {add, mul, sub, div} [7]. Because SREDT uses fundamentally nonlinear splits, it is not directly comparable to linear oblique methods.

**FoLDTree** uses Forward ULDA (Uncorrelated Linear Discriminant Analysis) for oblique splits [8]. It is available only as an R package (`LDATree` v0.2.0 on CRAN) with no Python wrapper [9]. Split direction vectors are accessible via the `folda` package's `scaling` matrix [9].

**FC-ODT** (Feature Concatenation Oblique Decision Tree) uses ridge regression with feature concatenation, achieving a provably better consistency rate of O(1/K²) vs O(1/K) for standard ODT [10]. It has **no public code** and is regression-only in published experiments [10]. FC-ODT is the closest existing work to direction sharing: it concatenates parent node projected scores ỹ = a^T x with features at child nodes, enabling vertical information reuse within a tree path [10]. This contrasts with Codebook-FIGS' horizontal reuse across trees.

Among classic methods, **OC1** (hill-climbing perturbation, 1994) has a Python wrapper via sklearn-oblique-tree but is abandoned since 2019 [11]. **HHCART** (Householder reflections) is available as scikit-obliquetree v0.1.4 but abandoned since May 2021 [12]. **CART-ELC** (exhaustive linear combinations) is actively maintained at https://github.com/andrewlaack/cart-elc with sklearn compatibility, created April 2025 [13]. **ODRF** is an R package on CRAN supporting customizable linear combinations [14].

### 2. Benchmark Results and Dataset Alignment

The most comprehensive recent benchmark comes from the RO-FIGS paper, which evaluates 11 methods across **22 OpenML classification datasets** spanning 4 to 503 features and 303 to 8,378 samples [4]. Full balanced accuracy results (10-fold CV) are provided in Table I of the paper [4]. Average ranks across all 22 datasets show RO-FIGS achieving rank 8.8 (best), followed by CatBoost at 8.5, Ens-ODT at 7.8, MLP at 6.5, and RF at 6.1, with FIGS at 4.5 [4]. The corrected Friedman test with Bonferroni-Dunn post-hoc (p<0.05, CD=2.81) shows RO-FIGS significantly outperforms FIGS, MT, ETC, DT, ODT, and OT, but is not significantly different from CatBoost, Ens-ODT, MLP, or RF [4].

SPORF benchmarks use 105 UCI datasets with Cohen's kappa via 5-fold CV [1], while SREDT uses 56 PMLB datasets with F-Score and accuracy via 70-30 splits with 100 repeats [7]. FoLDTree reports on 9 UCI datasets using accuracy [8], and FC-ODT uses 17 LIBSVM regression datasets with R² [10]. **Direct cross-paper comparison is impossible** due to metric incompatibilities (balanced accuracy vs kappa vs F-score vs R²) and different CV protocols [1, 4, 7, 8, 10].

The 22 RO-FIGS OpenML datasets with IDs are fully cataloged in research_out.json [4, 5], including datasets like blood (ID 1464, 4 features), diabetes (ID 37, 8 features), breast-w (ID 15, 9 features), spambase (ID 44, 57 features), USPS (ID 41964, 256 features), and SpeedDating (ID 40536, 503 features) [4]. The likely overlapping datasets across papers are diabetes, breast-w/breast-cancer, heart, and credit [1, 4, 7, 8].

**Recommendation**: Use the 22 RO-FIGS OpenML datasets as the primary Codebook-FIGS benchmark suite, as they provide the most comprehensive existing baseline numbers across both compact (FIGS, DT) and powerful (CatBoost, RF, MLP) methods [4].

### 3. Direction Diversity and Interpretability Metrics

**No existing oblique tree paper computes SVD-based diversity metrics on split directions** — this is a novel contribution opportunity for Codebook-FIGS [15, 16].

The primary recommended metric is **Effective Rank (eRank)**, defined as eRank(A) = exp(−Σᵢ ρᵢ ln ρᵢ) where ρᵢ = σᵢ / Σⱼ σⱼ are normalized singular values [15]. Properties include: 1 ≤ eRank(A) ≤ rank(A), eRank equals rank iff all nonzero singular values are equal, and eRank(A+B) ≤ eRank(A) + eRank(B) [15]. For Codebook-FIGS with K entries, eRank(W) ≤ K by construction since the split direction matrix W has at most K distinct rows, making eRank a natural measure of how uniformly the codebook entries are utilized [15].

The secondary metric is **Stable Rank**: srank(A) = ‖A‖²_F / ‖A‖²_2 = Σσ²ᵢ / max(σᵢ)² [16]. It is continuous, differentiable, and less sensitive to small singular values than eRank [16]. The tertiary metric is **Participation Ratio**: PR = (Σσ²ᵢ)² / Σσ⁴ᵢ, commonly used in neuroscience for neural population dimensionality [17]. A supplementary metric is **Average Pairwise Cosine Similarity**: mean(|cos(wᵢ, wⱼ)|) for all i ≠ j, which is intuitive and does not require SVD [4].

For interpretability, existing literature uses: number of splits (total across ensemble) [4, 6], number of trees [4, 6], tree depth [7, 8, 10], features per split [4], SHAP analysis [4], and inference time [7]. Novel codebook-specific metrics proposed include: **codebook entry sparsity** (average L0/L1 norms), **codebook stability across CV folds** (cosine similarity with Hungarian matching), **codebook coverage** (fraction of K entries used), **codebook interpretability score** (fraction with cos(entry, eⱼ) > 0.9 for some axis), and **domain alignment score** (alignment with known feature combinations) [4, 6, 15].

### 4. Codebook Size K Selection Heuristics

For Codebook-FIGS, K should be in the **undercomplete regime** (K ≪ d), unlike K-SVD dictionaries (K = 2–4d overcomplete) [18] or sparse autoencoders (4x–2,833x expansion) [19]. The goal is finding a small set of reusable directions, not overcomplete representation.

The **PCA 95% variance threshold** provides a data-driven upper bound: K ≥ number of PCA components explaining 95% of variance [20]. The **eRank of the data covariance matrix** provides an information-theoretic lower bound on the number of meaningful directions [15]. **Rate-distortion theory** suggests K ≥ 2^R(D) where R(D) is the rate-distortion function for target distortion D [21]. **MDL (Minimum Description Length)** selects K = argmin_K {K·d·log(bits) + n·s·log₂(K) + reconstruction_error} [22].

Intrinsic dimensionality estimation methods include the **MLE estimator** of Levina & Bickel [23] and **TWO-NN** [24], both providing local estimates of manifold dimension that can inform K selection.

Per-dataset guidance based on the RO-FIGS benchmark suite: for d=4–10 features, K=3–5 (blood, diabetes, breast-w); for d=11–30, K=5–10 (ilpd, climate, kc2, heart); for d=31–60, K=8–15 (pc3, biodeg, spambase); for d=61–100, K=10–20 (credit-g, friedman); for d=100+, K=15–30 (usps, bioresponse, speeddating) [4]. The default heuristic is K = min(d, 20) with an evaluation sweep over {3, 5, 8, 10, 15, 20}.

The recommended K-selection pipeline for experiments: (1) compute PCA 95% threshold as initial estimate, (2) compute eRank of data covariance as lower bound, (3) sweep over candidate K values via cross-validation, (4) report codebook coverage to detect over-specification [15, 20].

### 5. Key Gaps and Risks

Critical gaps include: SREDT and FC-ODT lack public code, so results can only be compared via reported numbers [7, 10]. FoLDTree is R-only, complicating pipeline integration [9]. SPORF is a full 500-tree ensemble while FIGS-style models use ≤5 trees — comparing requires careful framing of accuracy-interpretability tradeoffs [1, 4]. The treeple package does not expose split direction vectors via clean public API, requiring Cython internals access for direction diversity analysis [2]. The RO-FIGS repo's minimal development history (10 commits, 1 star) poses reproducibility risks [5]. No existing paper studies direction sharing/reuse quantitatively prior to FC-ODT's feature concatenation mechanism [10].

## Sources

[1] [Sparse Projection Oblique Randomer Forests (Tomita et al., JMLR 2020)](https://jmlr.org/papers/v21/18-664.html) — Original SPORF paper with 105 UCI benchmark datasets, Cohen's kappa metric, sparse random projection split methodology, and hyperparameter details (max_features, feature_combinations lambda=3/p, 500 trees).

[2] [treeple: Scikit-learn-compatible decision trees beyond axis-aligned splits](https://github.com/neurodata/treeple) — Active Python package (v0.10.3) implementing SPORF as ObliqueRandomForestClassifier, maintained by neurodata lab, last push Feb 2026. Split directions stored in Cython internals, not easily exposed.

[3] [SPORF original repository (neurodata)](https://github.com/neurodata/SPORF) — Original SPORF implementation, effectively abandoned (last commit Dec 2019). Superseded by treeple package.

[4] [RO-FIGS: Efficient and Expressive Tree-Based Ensembles for Tabular Data (Matjasec et al., IEEE CITREx 2025)](https://ieeexplore.ieee.org/document/10960202) — RO-FIGS paper with complete Table I balanced accuracy results for 11 methods on 22 OpenML datasets, average ranks, Friedman test with Bonferroni-Dunn post-hoc, L-1/2 oblique splits, model compactness analysis.

[5] [RO-FIGS GitHub Repository](https://github.com/um-k/rofigs) — Standalone Python implementation of RO-FIGS (Apache-2.0), 10 commits, 1 star, created Dec 2024. Contains ROFIGS class but not integrated into imodels.

[6] [imodels: Interpretable ML package including FIGS](https://github.com/csinva/imodels) — Python package (v2.0.4) with FIGSClassifier/FIGSRegressor. Key extension point _construct_node_with_stump() uses DecisionTreeRegressor(max_depth=1) for axis-parallel splits. Priority queue and residual tracking infrastructure preserved.

[7] [Symbolic Regression Enhanced Decision Trees for Classification Tasks (Fong & Motani, AAAI 2024)](https://ojs.aaai.org/index.php/AAAI/article/view/29091) — SREDT uses gplearn for GP-evolved nonlinear split expressions. 56 PMLB datasets, F-Score metric. No public code. Parsimony 0.001, 40 generations, pop 400, tournament 200.

[8] [FoLDTree: A ULDA-Based Decision Tree Framework (Wang, 2024)](https://arxiv.org/abs/2410.23147) — Forward ULDA-based oblique tree. 9 UCI datasets, accuracy metric. R package LDATree v0.2.0 on CRAN, no Python wrapper.

[9] [LDATree: Classification Trees with Linear Discriminant Analysis Splits (CRAN)](https://cran.r-project.org/package=LDATree) — R package v0.2.0 implementing FoLDTree. Split directions accessible via folda's scaling matrix. Active (updated Oct 2024).

[10] [Enhance Learning Efficiency of Oblique Decision Tree via Feature Concatenation (Lyu et al., 2025)](https://arxiv.org/abs/2502.00465) — FC-ODT with ridge regression and feature concatenation. O(1/K²) consistency rate vs O(1/K) for standard ODT. 17 LIBSVM datasets, R² metric. No public code. Closest existing work to direction sharing.

[11] [sklearn-oblique-tree: OC1 Python wrapper](https://github.com/AndriyMulyar/sklearn-oblique-tree) — Python wrapper for OC1 (Murthy et al., JAIR 1994). Abandoned since 2019, 48 stars. Sklearn-compatible ObliqueClassifier.

[12] [scikit-obliquetree: HHCART implementation (PyPI)](https://pypi.org/project/scikit-obliquetree/) — Python HHCART implementation v0.1.4 using Householder reflections. Abandoned since May 2021.

[13] [CART-ELC: Exhaustive Linear Combination Trees](https://github.com/andrewlaack/cart-elc) — Active C++/Python implementation (created Apr 2025) of exhaustive linear combination splits. Sklearn compatible.

[14] [ODRF: Oblique Decision Random Forest (CRAN)](https://cran.r-project.org/package=ODRF) — R package supporting customizable linear combination splits for both single trees (ODT) and ensembles (ODRF).

[15] [Effective Rank: A Measure of Effective Dimensionality (Roy & Vetterli, EUSIPCO 2007)](https://ieeexplore.ieee.org/document/4359079) — Defines eRank = exp(Shannon entropy of normalized singular values). Properties: 1 ≤ eRank ≤ rank, subadditive, scale-invariant. Corresponds to Hill number of order 1.

[16] [p-Stable Rank and Intrinsic Dimension (Ipsen & Saibaba, 2024)](https://arxiv.org/abs/2407.21594) — Defines stable rank as ||A||²_F / ||A||²_2 (2-stable rank). Framework unifying eRank and stable rank via p-stable ranks.

[17] [Scale-dependent dimensionality with participation ratio (Gao et al., 2022)](https://arxiv.org/abs/2112.13190) — Participation ratio PR = (Σσ²)² / Σσ⁴ used for neural population dimensionality. Sensitive to dominant directions.

[18] [K-SVD: An Algorithm for Designing Overcomplete Dictionaries (Aharon et al., IEEE TSP 2006)](https://ieeexplore.ieee.org/document/1710377) — K-SVD dictionary learning with typical 4x overcomplete dictionaries. Opposite regime from Codebook-FIGS which needs undercomplete K << d.

[19] [Scaling and Evaluating Sparse Autoencoders (Gao et al., ICLR 2025)](https://arxiv.org/abs/2404.16014) — SAE scaling laws showing 4x-2833x expansion factors. Provides scaling formula L₀ ∝ (N/L)^a for predicting feature usage with dictionary size.

[20] [scikit-learn PCA documentation](https://scikit-learn.org/stable/modules/decomposition.html#pca) — Standard PCA 95% variance threshold for dimensionality estimation. Provides data-driven upper bound on codebook size K.

[21] [Rate-Distortion Theory (Wikipedia)](https://en.wikipedia.org/wiki/Rate%E2%80%93distortion_theory) — Rate-distortion framework: K ≥ 2^R(D) for target distortion D. Provides information-theoretic lower bound on codebook size.

[22] [Minimum Description Length Principle (Wikipedia)](https://en.wikipedia.org/wiki/Minimum_description_length) — MDL framework for model selection: K = argmin {model_cost + data_cost}. Applicable to codebook size selection.

[23] [Maximum Likelihood Estimation of Intrinsic Dimension (Levina & Bickel, Annals of Statistics 2005)](https://projecteuclid.org/journals/annals-of-statistics/volume-33/issue-5/Maximum-likelihood-estimation-of-intrinsic-dimension/10.1214/009053605000000327.full) — MLE estimator for local intrinsic dimensionality using k-nearest neighbor distances. Can inform K selection for codebook.

[24] [Estimating the Intrinsic Dimension of Datasets by a Minimal Neighborhood Information (TWO-NN, Facco et al., 2017)](https://www.nature.com/articles/s41598-017-11873-y) — TWO-NN method using ratio of two nearest-neighbor distances for intrinsic dimension estimation. Simple and robust.

## Follow-up Questions

- How does Codebook-FIGS' accuracy-interpretability Pareto frontier compare to RO-FIGS when both are evaluated on the same 22 OpenML datasets with identical balanced accuracy / 10-fold CV protocol?
- What is the empirical relationship between the eRank of the data covariance matrix and the optimal codebook size K across datasets of varying dimensionality?
- Can FC-ODT's vertical direction reuse (parent-to-child feature concatenation) be combined with Codebook-FIGS' horizontal direction sharing (global codebook across trees) for further gains?

---
*Generated by AI Inventor Pipeline*
