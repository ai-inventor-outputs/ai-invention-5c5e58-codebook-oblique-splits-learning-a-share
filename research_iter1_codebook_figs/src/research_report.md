# Codebook-FIGS

## Summary

Comprehensive technical analysis of FIGS (imodels) and RO-FIGS source code architectures, identifying exact modification points for codebook-constrained oblique splits. Provides complete Codebook-FIGS algorithm specification including PCA-based codebook initialization, constrained greedy split search (O(K·n·log n) per node), K-SVD-inspired alternating optimization with three codebook refinement strategies, convergence criteria, and hyperparameter recommendations. Includes baseline method availability matrix (8 packages) and benchmark dataset survey (22 RO-FIGS + 15 FIGS datasets).

## Research Findings

## Technical Analysis of FIGS/RO-FIGS Codebases & Codebook-FIGS Algorithm Specification

### 1. FIGS Code Architecture (imodels)

The FIGS implementation in the imodels package follows a clean class hierarchy: `Node` → `FIGS(BaseEstimator)` → `FIGSRegressor(RegressorMixin, FIGS)` / `FIGSClassifier(ClassifierMixin, FIGS)`, with cross-validation wrappers `FIGSRegressorCV` and `FIGSClassifierCV` [1].

**Node Structure**: Each `Node` stores a single `feature` (int) and `threshold` (float) for axis-aligned splits, along with `idxs` (boolean mask of data points reaching this node), `impurity`, `impurity_reduction`, and child references (`left`, `right`, `left_temp`, `right_temp`) [1]. The temporary child pattern (`left_temp`/`right_temp` → `left`/`right`) implements a candidate-then-commit split mechanism.

**The Critical Split Method**: `_construct_node_with_stump()` is the KEY MODIFICATION POINT. It fits a `sklearn.tree.DecisionTreeRegressor(max_depth=1)` as a stump on `X[idxs], y[idxs]`, then extracts `feature`, `threshold`, and `impurity` from the sklearn tree internals. Impurity reduction is computed as `(parent_impurity - left_impurity * n_left/n_total - right_impurity * n_right/n_total) * n_total` [1]. This method evaluates O(d) features per split (one per feature dimension via the stump).

**The Fit Loop**: FIGS uses a greedy loop that maintains `potential_splits` sorted by impurity reduction. At each iteration it: (a) pops the best split, (b) checks stopping criteria (`min_impurity_decrease`, `max_rules`, `max_trees`, `max_depth`), (c) activates the split's children, (d) updates residuals for ALL trees (`y_residuals[t] = y - sum(predictions from other trees)`), and (e) recomputes ALL remaining potential splits on updated residuals [1]. This recomputation creates the O(m²) factor in total complexity, where m = number of splits.

**Prediction**: Sum predictions across all trees. Regression returns the raw sum; classification applies softmax (FIGS) [1]. Each tree is traversed via `_predict_tree_single_point()` which checks `x[feature] <= threshold` at each node.

**Runtime complexity**: O(d·m²·n²) where d=features, m=splits, n=samples [2].

---

### 2. RO-FIGS Code Architecture

The RO-FIGS repository at `github.com/um-k/rofigs` contains `src/rofigs.py` (main implementation) and `src/utils.py` (helpers), with Apache 2.0 license [3, 4].

**Key Structural Differences from FIGS**: The `Node` class replaces FIGS' single `feature` (int) with `features` (array of indices) and `weights` (array of coefficients), enabling oblique splits of the form `X[:, features] @ weights <= threshold` [3]. The `ROFIGS.__init__` signature uses `beam_size` (features per split), `max_splits` (instead of `max_rules`), and `num_repetitions` (retry mechanism, default=5) [3].

**The Oblique Split Pipeline** (`_construct_oblique_split`): This is the PRIMARY MODIFICATION POINT. It: (1) creates a `spyct.Model()` instance with specified feature subset, (2) fits a depth-1 oblique tree using gradient-based optimization (Adam optimizer with L₁/₂ regularization) on `X[idxs, splitting_features]`, (3) calls `extract_info()` to extract features, weights, threshold, and impurity values [3, 4]. The spyct library (installed from `git+https://gitlab.com/TStepi/spyct.git@0bc9808a`) provides the gradient-based oblique split learning with L₁/₂ norm `(Σ√|wᵢ|)²` for sparsity [5]. Error handling returns a dummy leaf node if the split fails.

**The `extract_info()` utility** computes `np.dot(X[:, features], weights)`, partitions samples by threshold, then calculates MSE impurity for each partition using `DummyRegressor` [4].

**Fit Loop Differences**: RO-FIGS adds random feature subset sampling at each iteration, a retry mechanism (`num_repetitions`), and uses `scipy.special.expit` (sigmoid) instead of softmax for classification probabilities [3].

**Dependencies**: Core dependencies are scikit-learn 1.3.2, scipy 1.10.1, numpy 1.24.4, pandas 2.0.3, and critically `spyct-tstepi` from GitLab which requires a C compiler [4, 5].

---

### 3. Codebook-FIGS Algorithm Specification

#### 3.1 Codebook Initialization

Three strategies are specified:

**PCA-based (RECOMMENDED DEFAULT)**: Compute `sklearn.decomposition.PCA(n_components=K)` on training data, take top-K principal components, normalize each to unit norm. For choosing K: start with `K = min(d, 10)`, or use explained variance ratio to find K capturing 90-95% variance, or use Minka's MLE via `PCA(n_components='mle')` [6]. Ablate K ∈ {3, 5, 8, 10, 15, 20}.

**Random unit vectors**: Sample K vectors from N(0, I_d) and normalize to unit sphere. Optionally use sparse random projections (as in SPORF) for sparsity [7].

**Warm-start from RO-FIGS**: Run unconstrained RO-FIGS, cluster the N learned split directions into K clusters via spherical K-means, use centroids as codebook [3].

#### 3.2 Codebook-Constrained Split Search

This method **replaces** `_construct_oblique_split()`. For each node:
1. Iterate over K codebook entries c_k
2. Project data onto each direction: `projections = X[idxs] @ c_k`
3. Find optimal threshold via sorted-scan (O(n log n)) using cumulative sum/squared-sum for efficient MSE computation
4. Select (k*, t*) maximizing impurity reduction
5. Create Node with `features=all, weights=c_k*, threshold=t*, codebook_idx=k*`

**Complexity**: O(K · n · log(n)) per node for sorted-scan vs O(d · n · log(n)) for FIGS stump. When K ≪ d, Codebook-FIGS is FASTER per split than standard FIGS.

#### 3.3 Alternating Optimization (K-SVD-Inspired)

Inspired by K-SVD [8], which alternates between sparse coding (fix dictionary, find representations) and dictionary atom update (fix representations, update atoms via rank-1 SVD of error matrix). The K-SVD algorithm iterates: (Step 1) fix D, solve for sparse codes X using OMP/BP; (Step 2) for each atom k, compute error matrix E_k = Y - Σ_{j≠k} d_j · x_j^T, restrict to signals using atom k, update d_k to first left singular vector of restricted E_k [8, 9].

**Codebook-FIGS Alternating Procedure (R rounds, typically R=3-5)**:

**Step A (Fix codebook → Grow trees)**: Run the standard FIGS/RO-FIGS greedy loop but with `_construct_codebook_split()` replacing `_construct_oblique_split()`. Each node selects `(codebook_idx, threshold)` maximizing impurity reduction. Apply all stopping criteria.

**Step B (Fix trees → Refine codebook)**: For each codebook entry c_k, collect all nodes N_k that used it, then refine c_k. Three refinement strategies:
- **Greedy re-optimization (RECOMMENDED)**: Pool data at all nodes using c_k, fit a stump + PCA(1) on pooled data, select direction yielding best total impurity reduction. O(|N_k| · d · n_k).
- **K-SVD-style SVD update**: Form error matrix from residuals at nodes using c_k, take first left singular vector. Most faithful to original K-SVD [8].
- **Gradient-based**: Define loss L(c_k) = total impurity at nodes using c_k, optimize via L-BFGS with unit-norm projection.

**Key simplification vs K-SVD**: Each tree node uses exactly ONE codebook entry (hard assignment, sparsity=1), while K-SVD allows T₀ > 1 atoms per signal [8, 9]. This makes the assignment step trivial (argmax over K entries) and the update step simpler.

#### 3.4 Convergence Criteria
1. Fixed rounds R=3-5 (default R=3)
2. Codebook stability: max_k |1 - |c_k^(r) · c_k^(r-1)|| < ε=0.01
3. Validation loss plateau: improvement < δ=0.001 × initial_loss

---

### 4. Precise Code Modification Points

| Component | File | Method/Location | Modification |
|-----------|------|----------------|-------------|
| Codebook storage | src/rofigs.py | ROFIGS.__init__() | Add self.codebook_ (K×d), self.K, self.n_alternation_rounds |
| Codebook init | src/rofigs.py | New _init_codebook(X, K) | New method before fit loop |
| Constrained split | src/rofigs.py | Replace _construct_oblique_split() | New _construct_codebook_split() |
| Alternating loop | src/rofigs.py | Wrap fit() loop | Outer loop + _refine_codebook() |
| Node tracking | src/rofigs.py | Node.__init__() | Add codebook_idx attribute |
| Diversity metric | src/utils.py | New direction_diversity() | Pairwise cosine similarity |
| Visualization | src/utils.py | New print_codebook() | Display learned directions |

**What NOT to change**: Prediction pipeline (still uses weights/threshold), residual computation, stopping criteria framework, print_split() [3].

---

### 5. Baseline Method Availability

| Method | Package | Install | Classes | Status |
|--------|---------|---------|---------|--------|
| FIGS | imodels | pip install imodels | FIGSClassifier/Regressor | Active [1] |
| RO-FIGS | rofigs (source) | git clone + pip install -r | ROFIGSClassifier/Regressor | Research code [3] |
| SPORF/RerF | rerf | pip install rerf | Custom API | Last release Aug 2019 [7] |
| treeple | treeple | pip install treeple | ObliqueDecisionTreeClassifier, ObliqueRandomForestClassifier | Active, v0.10.3 Mar 2025 [10] |
| XGBoost | xgboost | pip install xgboost | XGBClassifier/Regressor | Active |
| LightGBM | lightgbm | pip install lightgbm | LGBMClassifier/Regressor | Active |
| CatBoost | catboost | pip install catboost | CatBoostClassifier/Regressor | Active |
| sklearn | scikit-learn | pip install scikit-learn | DT/RF/ExtraTrees | Standard |

**Note**: SPORF (rerf) appears effectively unmaintained since 2019 [7]. **treeple** from the same NeuroData group is the actively maintained successor with full sklearn compatibility and `feature_combinations` parameter controlling oblique split sparsity [10, 11].

---

### 6. Benchmark Datasets

The RO-FIGS paper evaluated on 22 binary classification datasets from OpenML: blood, diabetes, breast-w, ilpd, monks2, climate, kc2, pc1, kc1, heart, tictactoe, wdbc, churn, pc3, biodeg, credit, spambase, credit-g, friedman, usps, bioresponse, speeddating [12]. The FIGS paper used 6 classification + 9 regression + 3 clinical datasets [2]. Both are accessible via `sklearn.datasets.fetch_openml()` or the `openml` Python package [13].

RO-FIGS baselines included DT, RF, ExtraTrees, FIGS, ODT, Ensemble-ODT, Optimal Tree, Model Tree, CatBoost, LightGBM, XGBoost, and MLP [12]. RO-FIGS achieved the highest average rank (8.8 out of 13 methods) with balanced accuracy as the primary metric [12].

---

### 7. Complexity Analysis

| Method | Per-Split Cost | Total (m splits) |
|--------|---------------|-------------------|
| FIGS | O(d·n·log n) | O(m²·d·n·log n) |
| RO-FIGS | O(b·I·n) where b=beam_size, I=max_iter | O(m²·b·I·n) |
| Codebook-FIGS (1 round) | O(K·n·log n) | O(m²·K·n·log n) |
| Codebook-FIGS (R rounds) | O(K·n·log n) | O(R·m²·K·n·log n + R·K·d·n_k) |

When K=10 and d=50+, Codebook-FIGS is 5x+ faster per split than FIGS, even with R=3 alternation rounds [1, 3].

---

### Confidence Assessment

**High confidence**: FIGS and RO-FIGS source code analysis (directly read from repositories), class hierarchies, method signatures, and modification points.

**High confidence**: Baseline method availability (verified via PyPI, GitHub repositories, and documentation).

**Medium-high confidence**: Codebook-FIGS algorithm specification — the constrained split search and alternating optimization are well-grounded in K-SVD theory [8, 9], but the specific codebook refinement strategies (especially the greedy re-optimization) have not been empirically validated yet.

**Medium confidence**: Hyperparameter recommendations (K, R values) — these are informed by the K-SVD literature and dimensional analysis but require empirical validation.

**Lower confidence**: Whether R=3 alternation rounds is sufficient — this depends on how well PCA initialization captures useful split directions, which is dataset-dependent.

## Sources

[1] [FIGS Source Code (imodels)](https://raw.githubusercontent.com/csinva/imodels/master/imodels/tree/figs.py) — Complete FIGS implementation including Node class, FIGS base class, _construct_node_with_stump(), fit() loop, and prediction methods. Key reference for class hierarchy and split evaluation pipeline.

[2] [FIGS Paper: Fast Interpretable Greedy-Tree Sums](https://arxiv.org/abs/2201.11931) — Original FIGS paper by Tan et al. describing the algorithm, theoretical analysis (O(dm²n²) complexity), benchmark datasets (6 classification + 9 regression), and comparison with CART, RuleFit, XGBoost, Random Forest.

[3] [RO-FIGS Source Code](https://raw.githubusercontent.com/um-k/rofigs/main/src/rofigs.py) — Complete RO-FIGS implementation showing oblique Node structure (features array + weights array), _construct_oblique_split() with spyct integration, and fit() loop with beam_size and num_repetitions parameters.

[4] [RO-FIGS Utilities Source Code](https://raw.githubusercontent.com/um-k/rofigs/main/src/utils.py) — Helper functions including extract_info() for spyct model output parsing, leaf_impurity(), check_fit_arguments(), print_split(), and data loading utilities.

[5] [spyct: Oblique Predictive Clustering Trees](https://github.com/knowledge-technologies/spyct) — Documentation for spyct library used by RO-FIGS for gradient-based oblique split optimization. Key API: Model(splitter='grad', max_features, lr, max_iter, C) with Adam optimizer and L1/2 regularization.

[6] [sklearn PCA Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html) — PCA implementation details including n_components selection (integer, float for variance ratio, 'mle' for Minka's MLE), explained_variance_ratio_ attribute, and SparsePCA for L1-penalized components.

[7] [SPORF Repository (Sparse Projection Oblique Randomer Forest)](https://github.com/neurodata/SPORF) — SPORF/RerF implementation with Python bindings (rerf package). Last Python release v2.0.5 (August 2019). Linux/Mac only. C++ core with Python wrappers.

[8] [K-SVD: An Algorithm for Designing Overcomplete Dictionaries for Sparse Representation (Aharon et al. 2006)](https://legacy.sites.fas.harvard.edu/~cs278/papers/ksvd.pdf) — Foundational K-SVD paper describing alternating minimization between sparse coding and atom update. Key algorithm: fix dictionary → OMP sparse coding; fix codes → rank-1 SVD update of each atom using restricted error matrix E_k^R.

[9] [K-SVD Wikipedia](https://en.wikipedia.org/wiki/K-SVD) — Overview of K-SVD algorithm: dictionary learning via alternating between sparse coding step and column-wise SVD-based dictionary update. Non-convex problem with monotonic decrease guarantee per iteration.

[10] [treeple: Advanced Decision Tree Library](https://github.com/neurodata/treeple) — Actively maintained sklearn-compatible library (v0.10.3, March 2025) providing ObliqueDecisionTreeClassifier, ObliqueRandomForestClassifier with feature_combinations parameter. Successor to SPORF for Python oblique trees.

[11] [ObliqueDecisionTreeClassifier API Documentation](https://docs.neurodata.io/treeple/v0.6/generated/sktree.tree.ObliqueDecisionTreeClassifier.html) — Full API documentation for treeple's oblique tree classifier including feature_combinations parameter (controls average features combined per split), criterion options, and all sklearn-compatible methods.

[12] [RO-FIGS Paper: Efficient and Expressive Tree-Based Ensembles for Tabular Data](https://arxiv.org/abs/2504.06927) — RO-FIGS paper evaluating on 22 OpenML binary classification datasets against 12 baselines. RO-FIGS achieved highest average rank (8.8) using balanced accuracy metric with 10-fold CV.

[13] [OpenML Documentation](https://docs.openml.org/) — OpenML platform for programmatic dataset access via openml-python package or sklearn.datasets.fetch_openml(). Supports benchmark suites and standardized train/test splits.

## Follow-up Questions

- How does the codebook refinement strategy (greedy re-optimization vs K-SVD-style SVD update vs gradient-based) affect convergence speed and final model quality — should we implement all three and ablate, or commit to one?
- What is the optimal handling of classification targets in the codebook-constrained split search — should impurity be Gini/entropy (like FIGS classification) or MSE on residuals (like the current regression-oriented design), and does this choice affect the alternating optimization convergence?
- Can the codebook be made adaptive in size (K) during alternating optimization — e.g., pruning unused codebook entries and splitting overloaded ones — and would this improve the interpretability-accuracy tradeoff?

---
*Generated by AI Inventor Pipeline*
