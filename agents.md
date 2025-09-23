# Agents: IncrementalLinearDiscriminantAnalysis (ILDA)

**Repo**: scikit-learn (fork)  
**Branch**: `feature/incremental-lda`  
**Owner**: <your-handle>  
**Start**: <YYYY-MM-DD>  
**Objective**: Add an incremental / online LDA classifier (`IncrementalLinearDiscriminantAnalysis`) with `partial_fit` supporting all solvers (`'svd'`, `'lsqr'`, `'eigen'`), plus tests and docs.

---

## 0) Full Implementation Prompt for the Coding Agent (authoritative spec)

### Mission

Implement `IncrementalLinearDiscriminantAnalysis` (ILDA) as an incremental / online estimator compatible with scikit‑learn. It must support **batch updates via `partial_fit`**, **all LDA solvers (`'svd'`, `'lsqr'`, `'eigen'`)** while preserving each solver’s important characteristics, and provide **unit tests** verifying correctness and closeness to `LinearDiscriminantAnalysis` (LDA) on equivalent data. Follow scikit‑learn’s contributor and API guidelines.

**Rationale**: For LDA, the sufficient statistics are **class counts, class means, and within‑class scatter**; these can be **updated exactly** in streaming fashion using numerically stable online moment formulas (Welford / Chan–Golub–LeVeque / Pébay) without storing past samples. From these, recompute the model parameters after each mini‑batch. Solver‑specific paths:

- **`'lsqr'`**: Solve pooled within‑class system \(\Sigma \omega_k = \mu_k\) with optional shrinkage; this maps naturally to sufficient statistics and matches batch solutions when shrinkage is fixed.
- **`'eigen'`**: Solve generalized eigenproblem \(S_b v = \lambda S_w v\) (with optional shrinkage) using between/within scatter from the stats.
- **`'svd'`**: Reconstruct what the `'svd'` path needs from **second‑moment matrices**: the centered data Gram (total scatter) and class‑means matrix, avoiding explicit covariance inversion and preserving decision/transform behavior. Optionally offer a sketch‑based path using randomized SVD for large \(p\).

---

### 1) Scope & Acceptance Criteria

**Deliverables**

1. **Code**
   - New estimator `IncrementalLinearDiscriminantAnalysis` in `sklearn/discriminant_analysis.py` (or a small private module imported by it), with API parity to `LinearDiscriminantAnalysis` (`predict`/`predict_proba`/`decision_function`/`transform`) and full estimator compliance.
   - Tests in `sklearn/discriminant_analysis/tests/test_incremental_lda.py`.
   - Docs: class docstring (NumPy style), brief “What’s New” entry, and a short addition to the LDA user guide referencing the incremental variant.

2. **Features**
   - `partial_fit(X, y, classes=None, sample_weight=None)` implements exact online updates of sufficient statistics across mini‑batches. On the first call, require `classes` per scikit‑learn’s convention.
   - Support **all solvers**: `'svd'`, `'lsqr'`, `'eigen'`.
   - Support `shrinkage=None | float | 'auto'` for `'lsqr'` and `'eigen'` (document that `'svd'` does not use shrinkage).
   - Support `priors`, `n_components`, `store_covariance`, `tol`, and metadata routing for `partial_fit` (e.g., `set_partial_fit_request` for `classes` and `sample_weight`).
   - Handle `sample_weight` in both `fit` and `partial_fit`.
   - Preserve/expose public attributes (`coef_`, `intercept_`, `means_`, `covariance_` (as applicable), `scalings_`, `xbar_` (for `'svd'`), `classes_`, `n_features_in_`, `feature_names_in_`) matching LDA semantics.

3. **Quality**
   - Estimator passes `check_estimator` (skip/xpass only where justified for incremental specifics).
   - Tests verify **predictions**, **decision_function**, and **transform/subspace** are **close** to batch `LinearDiscriminantAnalysis` trained on the full data (see §4 for metrics/tolerances).
   - Style & contributor compliance: PEP 8, high coverage, appropriate estimator tags, and clean static checks (ruff/mypy as used in the repo).

**Out‑of‑Scope (document with TODOs)**

- Incremental support for arbitrary `covariance_estimator` objects (accept the parameter for API parity, but raise `NotImplementedError` on `partial_fit` unless the estimator itself supports `partial_fit`). Keep shrinkage paths supported.
- Advanced drift/forgetting mechanisms beyond simple reweighting (possible future work).

---

### 2) Public API & Estimator Tags

**Class name**: `IncrementalLinearDiscriminantAnalysis`  
**Base classes**: `ClassifierMixin`, `LinearClassifierMixin`, `BaseEstimator`

**Parameters** (mirror LDA where applicable):

- `solver={'svd','lsqr','eigen'}` (default `'svd'`)
- `shrinkage=None | 'auto' | float in [0,1]` (only for `'lsqr'` and `'eigen'`)
- `priors=None | array-like (n_classes,)`
- `n_components=None | int` (≤ `min(n_classes - 1, n_features)`)
- `store_covariance=False`
- `tol=1e-4`
- `warm_start=False` (optional; calling `fit` repeatedly without resetting)
- `svd_method={'auto','cov','randomized'}` (new; §3.4), default `'auto'`
- `random_state=None` (for randomized SVD path)

**Methods**

- `fit(X, y, sample_weight=None)` — calls `_reset_stats()` then `partial_fit` once.
- `partial_fit(X, y, classes=None, sample_weight=None)` — streaming updates; require `classes` on first call.
- `predict`, `predict_proba`, `predict_log_proba`, `decision_function`, `transform`, `score` — same semantics as LDA.
- `__sklearn_is_fitted__`, `_more_tags` / `__sklearn_tags__` — set tags appropriate for classifier with `partial_fit`.
- Metadata routing helpers: `set_partial_fit_request(classes=True, sample_weight=True)`; ensure compatibility with `enable_metadata_routing=True`.

---

### 3) Streaming Sufficient Statistics (numerically stable)

Maintain (persist as learned attributes):

- **Global**: \(N\), \(\mu \in \mathbb{R}^p\), \(M2 \in \mathbb{R}^{p\times p}\) (total second central moment: \(\sum (x-\mu)(x-\mu)^\top\)).
- **Per class** \(k\): \(n_k\), \(\mu_k\), \(M2_k\).

For a mini‑batch \(X\) with labels \(y\), compute classwise batch stats \((n_k^b, \mu_k^b, M2_k^b)\), then **merge** into running stats using pairwise/Welford updates (vectorized):

- Means: \(\mu' = \mu + \delta\cdot n_b/(N+n_b)\) where \(\delta = \mu_b - \mu\).
- Second moments: \(M2' = M2 + M2_b + \frac{N\cdot n_b}{N+n_b}\, \delta\,\delta^\top\).

Apply the same *per class*. Then set \(S_w = \sum_k M2_k\), \(S_t = M2\), \(S_b = S_t - S_w\) (total scatter equals within + between).

**Weighted data**: compute weighted batch means and central second moments and use the **weighted** merge formulas (Pébay) to update `(N, mu, M2)` and `(n_k, mu_k, M2_k)`.

**Priors**: if `priors=None`, set \(\pi_k = n_k/N\); otherwise use fixed `priors` (still update means and scatters with the observed counts).

---

### 4) Parameter Estimation from Stats (per solver)

Recompute coefficients after each `partial_fit`. Keep attribute semantics consistent with `LinearDiscriminantAnalysis`.

#### 4.1 `'lsqr'` (classification only)

- Pooled within‑class **biased** covariance: \( \Sigma = \sum_k \pi_k \, C_k \) with \( C_k = M2_k / n_k \).
- **Shrinkage** if requested:  
  - Fixed \(\alpha\in[0,1]\): \(\Sigma \leftarrow (1-\alpha)\Sigma + \alpha\,\mathrm{diag}(\Sigma)\).  
  - `'auto'` (Ledoit–Wolf): estimate \(\alpha\) from cumulative stats if feasible; otherwise compute \(\alpha\) from a small **rolling buffer** of within‑class residuals (configurable, default a few thousand samples) and document as an approximation.
- Solve \(\Sigma \, \omega_k = \mu_k\) (e.g., Cholesky or least‑squares). Build `coef_` and `intercept_` as in LDA: \(\omega_{k0} = -\tfrac{1}{2}\mu_k^\top\Sigma^{-1}\mu_k + \log \pi_k\).

#### 4.2 `'eigen'`

- Solve \(S_b v = \lambda S_w v\) (with optional shrinkage applied to \(S_w\)); regularize \(S_w\) by adding \(\varepsilon I\) tied to `tol` if needed. Sort eigenpairs by descending \(\lambda\).
- `scalings_`: columns are eigenvectors for top `n_components`. Derive `coef_`/`intercept_` consistently with scikit‑learn’s implementation.

#### 4.3 `'svd'`

- Batch `'svd'` uses SVD of centered data and of the class‑means matrix, avoiding explicit covariance inversion and performing well when \(p\gg n\). Incrementally, reconstruct the required decompositions from **second‑moment accumulators**: the centered Gram equals \(S_t\), so eigendecomposition/SVD can be obtained from \(M2\) without storing samples. Build final directions from \(M2\) and the class‑means matrix built from \(\{\mu_k\}\). Do **not** apply shrinkage (consistent with LDA).
- **Optional approximate path** when `svd_method='randomized'`: maintain a streaming **sketch** (IPCA‑like) to update a low‑rank basis and then call `randomized_svd` to refine top components at update time. Document as *approximate* and cover with tests using relaxed tolerances.

---

### 5) Numerical Stability & Edge Cases

- Use **pairwise/online covariance updates** (Welford / Chan–Golub–LeVeque / Pébay); vectorize and avoid Python loops.
- Guard against singular \(S_w\): add small ridge \(\varepsilon I\) controlled by `tol` when solving linear systems or generalized eigenproblems.
- `classes` handling in `partial_fit`: require on first call; later calls must not introduce unseen classes (raise `ValueError`).
- `n_components`: cap to `min(n_features, n_classes−1)` (the cap can increase if new classes appear; recompute `scalings_` accordingly).
- `store_covariance`: for `'svd'`, honor flag as in LDA (store pooled within‑class covariance when requested).
- Metadata routing: implement `set_partial_fit_request` for `classes` and `sample_weight`.

---

### 6) Tests (pytest)

Create `sklearn/discriminant_analysis/tests/test_incremental_lda.py` with **fast**, **deterministic** tests.

**A. API & estimator checks**
- Use `parametrize_with_checks` on a small synthetic dataset (skip only checks truly incompatible with `partial_fit` semantics with justification).

**B. Equivalence / closeness to batch LDA**
- Generate datasets with `make_classification` (multiclass; include \(p \gg n\)) and Iris. Split into mini‑batches.
- For each solver and shrinkage option (`None`, `0.2`, `'auto'`):
  - Train **batch** `LinearDiscriminantAnalysis` on full data.
  - Train **incremental** with `partial_fit` across batches (supply `classes` first).
  - Assert:
    - `predict` equality rate ≥ **99.5%** (often exactly equal).  
    - `decision_function` closeness: `rtol=1e-5`, `atol=1e-8` (relax to `rtol=1e-3` for `'svd'` with `svd_method='randomized'`).  
    - `transform` subspace closeness: compute principal angles (e.g., `scipy.linalg.subspace_angles`); require max angle ≤ **1e‑6 rad** for exact paths; ≤ **5e‑3** for randomized.  
    - `coef_` / `intercept_` close within reasonable tolerance (allow sign/rotation ambiguity when applicable).
- Verify behavior with **`sample_weight`** matches batch LDA using the same weights.
- Verify **`n_components`** handling and shapes.
- Verify **`store_covariance`**: attribute presence and consistency when requested.
- Verify **error paths**: unseen classes in later `partial_fit`, invalid `n_components`, misuse of shrinkage with `'svd'`, unsupported `covariance_estimator` in `partial_fit` → `NotImplementedError`.

**C. Stress / numerical edge**
- High‑dimensional case (e.g., \(n=50, p=2000\)): ensure stability (no `LinAlgError`), runtime budgeted for CI.
- Ill‑conditioned covariance: ridge via `tol` should stabilize and keep predictions reasonable.

**D. Generic `partial_fit` compliance (mirror existing incremental classifiers)**
- Study the `partial_fit`-oriented suites for `SGDClassifier`, `PassiveAggressiveClassifier`, `Perceptron`, `GaussianNB`, and `MiniBatchKMeans` (see `sklearn/linear_model/tests/test_sgd.py`, `sklearn/naive_bayes/tests/test_naive_bayes.py`, etc.) and adapt their generic patterns.
- Implement dedicated tests (parameterized where possible) that assert:
  1. The first `partial_fit` call **requires** `classes`; omitting them raises `ValueError`.
  2. Subsequent `partial_fit` calls may omit `classes`, provided no unseen labels appear.
  3. Passing `classes` in different orders/permutations yields identical models and predictions (class-order invariance).
  4. Running `fit` on the full data produces the same results as streaming through `partial_fit` batches; ensure `fit` resets state and does **not** delegate to `partial_fit`.
  5. Sequentially observing classes one at a time (while always supplying the full `classes` list) still converges to the batch model.
  6. Model outputs (`predict`, `predict_proba`, `predict_log_proba`, `decision_function`, `transform`, `score`) remain available and numerically close after any number of `partial_fit` updates.
  7. Learned attributes preserve dtype stability (e.g., float32 vs. float64 inputs) in line with batch LDA behavior.
  8. Sparse input handling matches LDA expectations (raise a clear error if unsupported, or verify parity if support is added).
  9. Supplying batches with mismatched `n_features` raises `ValueError`.
  10. `sample_weight` updates scale statistics identically to their batch counterparts (check equivalence when weights are multiplied by a constant factor).
  11. Single-sample / tiny-batch updates remain numerically stable and produce deterministic results when re-run with the same mini-batch sequence and `random_state`.
  12. Later batches containing labels outside the initial `classes` argument trigger `ValueError` (no silent creation of new classes).

---

### 7) Documentation

- Class docstring (NumPy style) explaining streaming stats and solver behavior; cross‑reference LDA docs.
- Add a short subsection to LDA user guide describing the incremental variant and when to prefer it (large/out‑of‑core).  
- Note on shrinkage in streaming and the approximate `'auto'` strategy.
- Small example snippet showing `partial_fit` across chunks (consistent with scikit‑learn’s incremental learning guidance).

---

### 8) Code Organization & Style

- Follow scikit‑learn contributor guidelines, tests, and docstyle; ensure lints/static checks in the repo pass (ruff/mypy where applicable).
- Use existing helpers where appropriate (`validate_data`, `check_is_fitted`, `unique_labels`, `randomized_svd`, linear algebra from `scipy.linalg`).
- Keep solver‑specific code paths in private helpers (`_fit_lsqr_from_stats`, `_fit_eigen_from_stats`, `_fit_svd_from_stats`).
- Ensure attributes ending with `_` are set only after sufficient state is available (post `fit`/`partial_fit`).

---

### 9) File & API Checklist

- [ ] `sklearn/discriminant_analysis.py` (add new class; factor shared code if needed).
- [ ] `sklearn/discriminant_analysis/tests/test_incremental_lda.py`
- [ ] Update `__all__` / API docs index if appropriate.
- [ ] Docs build: ensure new class renders; add “What’s New” note.

---

### 10) Benchmarks (optional but valuable)

A small local benchmark script (not in repo) comparing batch LDA vs. ILDA across batch sizes, measuring walltime and accuracy for representative \(p/n\) regimes, including the `svd_method='randomized'` path. Use `random_state` for determinism. Reference `IncrementalPCA` docs for memory scaling discussion.

---

### 11) Mathematical References (for implementers; cite in docs)

- LDA API/attributes/solvers (including `svd`/`lsqr`/`eigen`, shrinkage support, `xbar_`, `scalings_`, `store_covariance`).  
- Incremental variance/covariance updates (Welford; Chan–Golub–LeVeque; Pébay).  
- Ledoit–Wolf shrinkage and scikit‑learn helpers.  
- Randomized/Incremental SVD.

(See “References” at the bottom of this file for canonical links.)

---

### 12) Definition of Done

- All unit tests pass locally; new tests cover ≥90% of added code.
- Lints/static checks pass; docstrings render; CI expected to pass.
- PR description enumerates design choices, limitations (notably `'auto'` shrinkage approximation and `covariance_estimator` support), and benchmark results.

---

## 1) Background & References

- LDA solvers, API, and attributes: scikit‑learn user guide and reference.  
- Partial‑fit conventions: require `classes` on the first call.  
- Online covariance updates: Welford; Chan–Golub–LeVeque; Pébay (weighted/pairwise).  
- Shrinkage: Ledoit–Wolf (`'auto'`), OAS (note: only `'lsqr'` and `'eigen'` support shrinkage).  
- Randomized/Incremental SVD for streaming (`randomized_svd`, `IncrementalPCA`).

Full links are collected in **§10 References** below.

---

## 2) Deliverables

- [x] `IncrementalLinearDiscriminantAnalysis` implementation.
- [x] Comprehensive tests in `sklearn/tests/test_incremental_lda.py` (batch vs. incremental, sample-weight, error handling).
- [x] Docstring + short user‑guide addition + What’s New entry.
- [ ] PR with benchmarks and limitations clearly stated.

---

## 3) Constraints & Non‑Goals (for this PR)

- `covariance_estimator` with `partial_fit`: **not implemented** unless the estimator itself supports `partial_fit`. If set, `partial_fit` raises `NotImplementedError` (documented).
- `'svd'` shrinkage unsupported (matches sklearn).
- `'svd'` path can use `svd_method='cov'` (exact from second moments) or `svd_method='randomized'` (approximate), with clear docs/tests.

---

## 4) Plan of Work (Milestones)

**M1 – Skeleton & Stats**
- [x] Scaffolding class + parameters + `_reset_stats`.
- [x] Implement numerically stable streaming stats (`N`, `mu`, `M2`; plus per‑class).
- [x] `partial_fit` first‑call (`classes=`) and subsequent‑call logic; validate inputs & tags.

**M2 – Solvers**
- [x] `'lsqr'`: pooled biased covariance + fixed shrinkage + solve; compute `coef_`, `intercept_`.
- [x] `'eigen'`: generalized eigenproblem + ridge; fill `scalings_`, `transform`.
- [x] `'svd'`: exact‑from‑moments path; optional `svd_method='randomized'` (randomized path deferred).

**M3 – Public API & Attributes**
- [x] `predict`, `predict_proba`, `decision_function`, `transform`, `score`.
- [x] `store_covariance`, `n_components` caps, `priors` handling, `sample_weight`.

**M4 – Tests & Docs**
- [x] Write tests (API, equivalence, edge, performance).
- [x] Docstrings; user‑guide addition.

**M5 – Polish**  
- [ ] Lints/static checks, coverage ≥90%.  
- [ ] Benchmarks + PR packaging.

_(Adjust milestones as needed; update dates in the Progress Log.)_

---

## 5) Decisions (Architecture & API)

- Online stats: pairwise/Welford with vectorized merges; weighted support.  
- Pooled covariance uses **biased** per‑class covariances combined by priors.  
- Shrinkage `'auto'`: first implementation estimates \(\alpha\) from a small rolling buffer of recent within‑class residuals (documented approximate); fixed \(\alpha\) uses exact formula.  
- Ridge stabilization via `tol` for singular systems.  
- Optional `svd_method` public parameter: `'auto'|'cov'|'randomized'`.

---

## 6) Open Questions

- Expose a public parameter to control the **rolling buffer size** for `'auto'` shrinkage in streaming? Default = 4096 samples.  
- Include the optional `svd_method='randomized'` path in the first PR, or ship exact‑from‑moments first?  
- Provide a public API to serialize/restore sufficient statistics (e.g., `get_state`/`set_state`) for checkpointing beyond pickling?

---

## 7) Risks & Mitigations

- **Large p** makes covariance eigendecomposition heavy → use `'svd_method="randomized"'` and respect `n_components ≤ C−1`.  
- **Numerical issues** in generalized eigenproblem → add small ridge to \(S_w\) via `tol`.  
- **Exact equivalence** with batch `'svd'` → expect decision/transform equivalence; tests check functional closeness rather than raw factor equality.

---

## 8) Test Matrix (quick checklist)

- [x] Solvers: `svd`, `lsqr`, `eigen` × shrinkage: `None`, `0.2` (covered in `test_incremental_matches_batch_solver`; `'auto'` pending implementation).
- [ ] Datasets: Iris; `make_classification` with `(n=200, p=20)` and `(n=50, p=2000)`; and a class‑imbalance case (additional synthetic regimes still TODO).
- [x] Batch vs. incremental: predictions ≥ 99.5% identical; decision_function & transform close (asserted against batch LDA).
- [x] `sample_weight` parity; `n_components` shape checks (fit + partial_fit equivalence and transform shape tests).
- [x] Error handling: unseen classes; invalid shrinkage with `'svd'`; unsupported `covariance_estimator` in `partial_fit`.

---

## 9) Progress Log

2025-09-22 — Implemented ILDA class skeleton, parameter wiring, and streaming statistics helpers (global + per-class) with stable merges.
2025-09-22 — Wired solver recomputation (`lsqr`, `eigen`, `svd`) using accumulated stats, including shrinkage handling and binary-shape adjustments.
2025-09-22 — Added incremental test suite (batch parity, sample-weight equivalence, error handling, n_components) plus user guide + What's New updates.

(Keep appending entries.)

---

## 10) How to Run Locally

```bash
# From repo root
pip install -r build_tools/requirements.txt
pip install pytest pytest-cov ruff mypy numpydoc

pytest sklearn/discriminant_analysis/tests/test_incremental_lda.py -q
ruff check sklearn
mypy sklearn
```

---

## 11) References (curated)

- **LDA API / attributes / solvers**: scikit‑learn reference for `LinearDiscriminantAnalysis` and LDA user guide.  
  - https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html  
  - https://scikit-learn.org/stable/modules/lda_qda.html
- **Estimator development & checks**: Developer guide and estimator checks.  
  - https://scikit-learn.org/stable/developers/develop.html  
  - https://scikit-learn.org/stable/modules/generated/sklearn.utils.estimator_checks.parametrize_with_checks.html  
  - https://scikit-learn.org/stable/modules/generated/sklearn.utils.estimator_checks.check_estimator.html
- **Incremental learning conventions** (`partial_fit`, `classes` on first call):  
  - Example docs showing `classes` required on first `partial_fit` (e.g., `GaussianNB.partial_fit`).  
    https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html
- **Randomized / Incremental SVD**:  
  - https://scikit-learn.org/stable/modules/generated/sklearn.utils.extmath.randomized_svd.html  
  - https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.IncrementalPCA.html
- **Online covariance / moments** (numerically stable & weighted):  
  - Pébay, P. (2008). *Formulas for Robust, One‑Pass Parallel Computation of Covariances and Arbitrary‑Order Statistical Moments*. Sandia report. https://www.osti.gov/servlets/purl/1028931  
  - Chan, T. F., Golub, G. H., & LeVeque, R. J. (1979/1982). *Updating Formulae and a Pairwise Algorithm for Computing Sample Variances*. Stanford CS‑TR‑79‑773; and subsequent publications. https://i.stanford.edu/pub/cstr/reports/cs/tr/79/773/CS-TR-79-773.pdf
- **Shrinkage (Ledoit–Wolf)** and helper in scikit‑learn:  
  - Ledoit‑Wolf estimator class: https://scikit-learn.org/stable/modules/generated/sklearn.covariance.LedoitWolf.html  
  - Convenience function: https://scikit-learn.org/stable/modules/generated/sklearn.covariance.ledoit_wolf_shrinkage.html
- **Metadata routing** (`set_partial_fit_request` examples in docs):  
  - Metadata routing guide: https://scikit-learn.org/stable/metadata_routing.html  
  - Example docs showing generated `set_partial_fit_request` (e.g., GaussianNB page above).

---

### Appendix A — Notes for PR Description

- Summarize design choices: sufficient statistics, solver‑specific estimation, stability safeguards, shrinkage handling, randomized SVD option.  
- Document limitations: `'auto'` shrinkage approximation in streaming; `covariance_estimator` support; randomized SVD is approximate.  
- Include small benchmark table (walltime & accuracy) versus batch LDA for two datasets and three batch sizes.
