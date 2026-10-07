# Cervical Cancer Risk - Decision Support System (DSS)
### Healthcare Information Systems · Carnegie Mellon University (Spring 2026)
**Team (Group 6):** Esha Pandya · Devavrath Sandeep · Abigail Torbatian

> Note: this is the original course write-up, kept for detail. File names in it predate the repo layout: `01_eda_v2.ipynb` is `notebooks/eda_v2.ipynb`, `03_sensitivity.ipynb` is `notebooks/cervical_cancer_model_sensitivity.ipynb`, and the bundle lives in `models/` (build it with `python scripts/train.py`). Sensitivity-analysis magnitudes were computed on the earlier (v1) model and should be read as directional only.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Repository Structure](#2-repository-structure)
3. [Setup & Installation](#3-setup--installation)
4. [Data Sources](#4-data-sources)
5. [Q1 - Exploratory Data Analysis (`01_eda_v2.ipynb`)](#5-q1--exploratory-data-analysis)
6. [Q2 - Predictive Modelling (`02_model_v2.ipynb`)](#6-q2--predictive-modelling)
7. [Q3 - Sensitivity Analysis (`03_sensitivity.ipynb`)](#7-q3--sensitivity-analysis)
8. [Confidence Score - Full Explainability](#8-confidence-score--full-explainability)
9. [Output Files & Plots](#9-output-files--plots)
10. [Key Findings](#10-key-findings)
11. [DSS Frontend Integration (Shiny)](#11-dss-frontend-integration)
12. [Known Limitations & Future Work](#12-known-limitations--future-work)

---

## 1. Project Overview

This project builds an end-to-end **Patient Risk Decision Support System (DSS)** for cervical cancer screening. Given a patient's demographic, lifestyle, and reproductive history, the system predicts their likelihood of requiring a biopsy and provides a confidence-scored, explainable recommendation to clinical staff.

**Clinical problem:** Cervical cancer is highly preventable with early detection, but screening guidelines are complex and follow-up rates are poor. A DSS that flags high-risk patients and explains *why* they are flagged can help staff prioritise interventions.

**Dataset:** UCI Cervical Cancer (Risk Factors) - 858 patients, 36 features, collected at Hospital Universitario de Caracas in Venezuela. Target variable: `Biopsy` (positive rate ~6.4%).

**System outputs per patient:**
- Binary prediction: HIGH RISK (biopsy recommended) / LOW RISK
- Ensemble probability (0–1)
- Confidence score (0–1) with tier label
- Model contribution waterfall (which model pushed probability up/down)
- Confidence component waterfall (which formula terms are helping/hurting)
- Natural language explanation for the clinician
- PCA scatter showing patient position in the full population

---

## 2. Repository Structure

```
project/
│
├── data/
│   └── risk_factors_cervical_cancer.csv        ← UCI dataset (primary)
│
├── notebooks/
│   ├── 01_eda_v2.ipynb                         ← EDA: cleaning, univariate, bivariate,
│   │                                               correlation, clustering, key insights
│   ├── 02_model_v2.ipynb                       ← Modelling: 4-model ensemble, ADASYN,
│   │                                               calibration, threshold tuning,
│   │                                               SHAP, confidence explainability,
│   │                                               patient scatter
│   └── 03_sensitivity.ipynb                    ← Sensitivity: misreporting, ablation,
│                                                   missing data, DSS intervention rules
│
├── model_plots_v2/                             ← Plots produced by 02_model_v2.ipynb
│   ├── calibration.png
│   ├── roc_pr_oof.png
│   ├── ensemble_vs_individual.png
│   ├── shap_all_models.png
│   ├── shap_rank_stability.png
│   ├── threshold_tuning.png
│   ├── holdout_evaluation.png
│   ├── subgroup_auc.png
│   ├── confidence_dist.png
│   ├── demo_confidence_waterfall.png
│   ├── demo_model_contribution.png
│   └── patient_scatter.png
│
├── sensitivity_plots/                          ← Plots produced by 03_sensitivity.ipynb
│   ├── baseline_distribution.png
│   ├── std_misreporting.png
│   ├── smoking_sensitivity.png
│   ├── iud_sensitivity.png
│   ├── age_sensitivity.png
│   ├── feature_ablation.png
│   ├── cluster_sensitivity.png
│   ├── confidence_stress.png
│   ├── missing_data_simulation.png
│   ├── hc_duration_sweep.png
│   └── findings_summary.png
│
├── cervical_model_bundle_v2.joblib             ← Serialised model bundle (produced by 02)
├── cervical_model_metadata_v2.json             ← Training metadata + scores
├── cervical_sensitivity_results.json          ← Sensitivity findings (produced by 03)
└── README.md
```

---

## 3. Setup & Installation

### Python environment

```bash
conda create -n dss python=3.11
conda activate dss
```

### Install dependencies

```bash
pip install \
  imbalanced-learn \
  xgboost \
  lightgbm \
  shap \
  scikit-learn \
  pandas \
  numpy \
  matplotlib \
  seaborn \
  joblib \
  scipy
```

**Optional - TabPFN** (requires free license from https://ux.priorlabs.ai):
```bash
pip install tabpfn
export TABPFN_TOKEN="your-api-key"
```
If TabPFN is not licensed, the notebook degrades gracefully to a 4-model ensemble (LR + RF + XGB + LGB).

### Run order

```bash
jupyter notebook notebooks/01_eda_v2.ipynb        # ~5 minutes
jupyter notebook notebooks/02_model_v2.ipynb       # ~15–25 minutes
jupyter notebook notebooks/03_sensitivity.ipynb    # ~10–20 minutes
```

Each notebook is self-contained - it reloads and cleans the dataset from scratch, so they can be run independently.

---

## 4. Data Sources

### Primary - UCI Cervical Cancer (Risk Factors)

| Property | Value |
|---|---|
| Source | UC Irvine Machine Learning Repository |
| Records | 858 patients |
| Features | 36 (after cleaning: 8 used for No-Dx model) |
| Target | `Biopsy` (binary: 0/1) |
| Positive rate | ~6.4% |
| Missing values | `?` placeholder - 2 columns dropped (>50% missing), remainder median/mode imputed |
| Origin | Hospital Universitario de Caracas, Venezuela |

**Outcome variables in the dataset:**
- `Hinselmann` - colposcopy result
- `Schiller` - Schiller's iodine test
- `Citology` - cytology (Pap smear)
- `Biopsy` - definitive diagnostic (primary target)
- `AnyAbnormal` - composite: 1 if any of the above is positive (engineered)

### Why this dataset was chosen over Mendeley

The team evaluated the Mendeley CER dataset as an alternative. It was ruled out for two reasons: sparse data (very few records per patient) and a narrow patient population that would not generalise to the broad clinic screening scenario the DSS targets.

---

## 5. Q1 - Exploratory Data Analysis

**Notebook:** `01_eda_v2.ipynb`

### 5.1 Data Cleaning Pipeline

1. Load raw CSV with `na_values='?'` to convert missing value placeholders immediately
2. Drop columns with >50% missing (`STDs: Time since first diagnosis`, `STDs: Time since last diagnosis` - both >80% missing)
3. Median impute remaining continuous columns
4. Mode impute remaining binary columns
5. Cast all columns to numeric

**Engineered features created in EDA:**
- `AnyAbnormal`: 1 if any of the four screening tests is positive
- `STD_burden`: sum of all individual STD indicator columns (reduces multicollinearity vs using raw STD columns)
- `AgeBucket`: categorical age groups (`<20`, `20-29`, `30-39`, `40-49`, `50+`)
- `Cluster`: K-Means risk cluster label (0=low, 1=medium, 2=high - sorted by Biopsy rate)

### 5.2 Missingness Analysis

A bar chart coloured by severity (red >50%, orange >20%, blue <20%) confirmed the two high-missingness columns and guided the imputation strategy. All other columns had <30% missing, making median/mode imputation appropriate.

### 5.3 Univariate Analysis

Key findings:
- **Age**: median ~26 years, right-skewed - population is young
- **Outcome prevalence**: Biopsy ~6.4%, AnyAbnormal ~13% - severe class imbalance, must be addressed in modelling
- **STD burden**: heavily zero-inflated - most patients report no STD history (stigma likely causes under-reporting, confirmed in Q3)
- **Smoking**: ~16% of patients smoke; most continuous features are zero-inflated

### 5.4 Bivariate Analysis

- Box plots of continuous features split by Biopsy outcome showed `Age` and `Hormonal Contraceptives (years)` have the most visible separation
- Grouped bar charts showed `Dx:Cancer` and `Dx:HPV` have drastically higher abnormal rates - flagged as near-leakage features
- Abnormal rate increases with age despite lower patient volume in older groups
- Group means table: `STDs (number)` and `STD_burden` show the highest relative lift for Biopsy=1

### 5.5 Correlation Analysis

Point-biserial correlations computed between all features and each of the four outcome variables. Key findings:
- `Dx:Cancer` and `Dx:HPV` are the strongest correlates (near-leakage - these are post-hoc diagnoses, not prospective risk factors)
- `STD_burden`, `Age`, and `Hormonal Contraceptives (years)` are the top non-leakage correlates
- `STDs`, `STDs (number)`, and `STD_burden` are highly intercorrelated - `STD_burden` chosen as composite to avoid multicollinearity in modelling

### 5.6 Cluster Analysis (K-Means, k=3)

- Elbow method used to select k=3
- Clusters sorted by Biopsy rate → labeled Low / Medium / High Risk
- PCA projection confirms visual separation between clusters
- Cluster assignment used as a feature in the predictive model

### 5.7 Feature Set Decision

Two feature sets defined for use in Q2:

**No-Dx (8 features - deployed in DSS):**
`Age`, `Number of sexual partners`, `Num of pregnancies`, `Smokes (years)`, `Hormonal Contraceptives (years)`, `IUD (years)`, `STD_burden`, `Cluster`, `Preg_x_Age` (interaction, added in v2)

**Full (10 features - comparison only):**
No-Dx + `Dx:Cancer`, `Dx:HPV`

The No-Dx set is what the DSS deploys because `Dx:Cancer` and `Dx:HPV` require prior diagnoses that are not available at initial screening intake.

---

## 6. Q2 - Predictive Modelling

**Notebook:** `02_model_v2.ipynb`  
**Output:** `cervical_model_bundle_v2.joblib`

### 6.1 Changes from v1

| Issue in v1 | Fix in v2 |
|---|---|
| AUC evaluated on training data (inflated 0.9969) | Proper 20% stratified holdout, locked until Section 10 |
| SMOTE distorted smoking-risk relationship (more smoking → lower risk) | ADASYN replaces SMOTE |
| 81% of patients flagged as high risk (threshold = 0.072) | Target flag rate ≤20%, maximise F2 within constraint |
| No calibration before ensemble weighting | Per-model isotonic calibration (3-fold CV) |
| SVM slow and inconsistent SHAP | SVM replaced by TabPFN (or LGB if TabPFN unlicensed) |
| No interaction features | `Preg_x_Age` added (largest individual AUC contributors) |
| No explainability on confidence score | Confidence waterfall + model contribution waterfall + NL explanation |
| No patient population context | PCA scatter with current patient marked |

### 6.2 Holdout Split

```python
X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_all, test_size=0.20, stratify=y_all, random_state=42
)
```

- `X_test` / `y_test` are not used again until Section 10 (final evaluation)
- All CV, OOF, weight optimisation, and threshold tuning use only `X_train`

### 6.3 Model Architecture

| Model | Resampling | Notes |
|---|---|---|
| Logistic Regression | ADASYN | Linear baseline, interpretable coefficients |
| Random Forest | ADASYN | 500 trees, `class_weight='balanced'` |
| XGBoost | ADASYN | `scale_pos_weight` for imbalance; fallback to GBM |
| LightGBM | ADASYN | Fast boosting, `class_weight='balanced'` |
| TabPFN | None | Meta-learned transformer; requires license |

**Why ADASYN over SMOTE:**
SMOTE interpolates uniformly between minority samples, which in this dataset created synthetic patients that inverted the smoking-risk relationship (more years of smoking → lower predicted risk). ADASYN focuses on harder-to-classify boundary samples, preserving the direction of feature relationships.

**Why TabPFN:**
TabPFN is a prior-fitted network trained on millions of synthetic tabular datasets. It is specifically designed for datasets under 1000 samples with fewer than 100 features - 858 rows and 9 features is its exact target regime. It requires no hyperparameter tuning and handles class imbalance through its prior distribution. It serves as the transformer-based component in the ensemble.

**Pipeline structure (ADASYN models):**
```
StandardScaler → ADASYN → Classifier
```
ADASYN runs inside the pipeline so it only operates on training folds during CV, never on validation data.

### 6.4 Cross-Validated Training

- 5-fold stratified CV on training set only
- OOF (out-of-fold) probabilities collected for each model
- TabPFN uses manual fold iteration (incompatible with `cross_val_predict`)
- Metrics at threshold=0.5: ROC-AUC, PR-AUC, F1, Recall, Precision, Brier score

### 6.5 Per-Model Isotonic Calibration

Before ensemble weight optimisation, each model is wrapped in `CalibratedClassifierCV(method='isotonic', cv=3)`. This ensures probability estimates are meaningful before being averaged - without calibration, tree models and neural models can output probabilities that cluster near 0 or 1, distorting the weighted average. TabPFN is skipped (it produces natively calibrated posteriors).

Calibration curves plotted before and after show the improvement for each model.

### 6.6 Ensemble Weight Optimisation

Soft-vote ensemble with OOF-optimised weights:

```python
# Objective: maximise PR-AUC on OOF predictions
def ensemble_pr_auc(weights):
    w = np.clip(weights, 0.01, 1.0)
    w = w / w.sum()
    return -average_precision_score(y_train, oof_matrix @ w)
```

Grid search provides a starting point, then Nelder-Mead refines to convergence. PR-AUC is used as the objective (rather than ROC-AUC) because it is more sensitive to performance on the rare positive class.

### 6.7 SHAP Analysis

TreeExplainer used for RF, XGB, LGB (exact, fast). LinearExplainer for LR. KernelExplainer for TabPFN (50-sample background, 150-row subset for speed).

For each model, two plots are generated:
- **Mean |SHAP| bar chart** - overall feature importance
- **SHAP beeswarm** - direction and distribution of each feature's impact

A **rank stability heatmap** across all models shows which features are consistently important. Features ranked consistently in the top 3 across all models are the most reliable candidates for required fields in the DSS intake form.

### 6.8 Threshold Tuning (v2 Fix)

**v1 problem:** `recall ≥ 0.85` constraint pushed threshold to 0.072, flagging 81% of patients - clinically meaningless.

**v2 approach:**
```python
TARGET_FLAG_RATE = 0.20   # flag at most 20% of patients

valid_mask     = flag_rates <= TARGET_FLAG_RATE
BEST_THRESHOLD = thresholds[valid_mask][np.argmax(fbeta_scores[valid_mask])]
```

Within the flag rate constraint, F2 (β=2) is maximised - recall is still weighted twice as much as precision, reflecting the clinical cost asymmetry (missing a true positive is worse than a false alarm), but within a range that is actionable for clinic staff.

Three plots: F2/recall/precision vs threshold, flag rate vs threshold, confusion matrix at chosen threshold.

### 6.9 Final Holdout Evaluation

Run once only in Section 10. Reports ROC-AUC, PR-AUC, F2, flag rate, and a full classification report on `X_test`. This is the number to report in the write-up.

### 6.10 Subgroup AUC

ROC-AUC computed separately for each age bucket and each risk cluster using a combined OOF+holdout probability array. Visualised as bar charts. A model with good global AUC can still fail for specific subgroups - particularly the `50+` age group and the `High Risk` cluster, where correct classification matters most clinically.

---

## 7. Q3 - Sensitivity Analysis

**Notebook:** `03_sensitivity.ipynb`  
**Output:** `cervical_sensitivity_results.json`

### 7.1 Purpose

Sensitivity analysis answers: *how much does the model's output change when input data is uncertain, mis-recorded, or missing?* Each scenario directly maps to a real clinical situation that clinic staff or system designers must plan for.

### 7.2 Analyses Performed

#### STD History Misreporting (Section 3)
**Scenario:** Patients frequently under-report STD history due to social stigma.

Two experiments:
- **Full zeroing:** Set `STD_burden = 0` for all patients. Measures maximum possible impact.
- **Misreport rate sweep:** Progressively blank STD_burden for 0–100% of patients. Plots classification flips (high→low risk) and mean probability shift as a function of misreport rate.
- **Partial misreport:** Scale STD_burden by 0.75, 0.50, 0.25, 0.00 to simulate partial disclosure.

**Key finding:** `STD_burden` has lower-than-expected individual impact on classification (few flips at 100% misreport). However, it compounds with other features - DSS rule: flag predictions where `STD_burden = 0` but ≥2 other risk factors are elevated.

#### Smoker Status Uncertainty (Section 4)
**Scenario:** Patients may not recall or disclose smoking duration.

Six scenarios tested: Baseline, Unknown (→0), Median imputed, +5 years, +10 years, Heavy smoker (20yr).

**Key finding from v1:** SMOTE had caused "more smoking → lower risk", which is clinically backwards. After switching to ADASYN in v2, this should be corrected. Re-run Section 4 after v2 training to verify the direction reverses.

Per-patient delta for current smokers also plotted - shows distribution of how much individual risk changes when smoking history is hidden.

#### IUD Protective Effect (Section 5)
**Scenario:** IUD duration is often poorly documented in clinical records.

EDA showed IUD has a slight protective (negative) association with abnormal outcomes. This section tests what happens when duration is zeroed (not recorded) vs doubled vs halved.

**Key finding:** Missing IUD duration slightly *inflates* predicted risk because the protective signal is lost. DSS rule: prompt staff to verify IUD status before finalising a prediction.

#### Age Sensitivity (Section 6)
**Scenario:** Age is the strongest non-leakage predictor. What does the full risk trajectory look like?

A synthetic median patient is created and age is varied from 15 to 74. Two plots:
- Probability vs age (all models + ensemble)
- Marginal effect dP/dAge per year of age

Also identifies the threshold-crossing age (the age at which a median patient first crosses the decision threshold) and the peak risk age.

#### Feature Ablation (Section 7)
**Method:** Replace each feature one at a time with its median value for all patients. Measure AUC drop.

This is the definitive importance measure - not model-internal (like SHAP), but prediction-performance-based. The feature causing the largest AUC drop when ablated is the most critical to collect accurately.

Also measures standard deviation of individual probability changes - features with high std are causing the most patient-level variation when ablated, indicating they provide individualised signal.

#### Cluster-Level Sensitivity (Section 8)
**Scenario:** Does the same input uncertainty affect low-risk and high-risk patients differently?

Each perturbation is run separately for Low/Medium/High Risk clusters. A grouped bar chart shows the mean probability change per cluster per perturbation. Typically high-risk cluster patients should show the largest drops when risk factors are hidden - if they don't, the model is not capturing the right interactions.

#### Confidence Score Stress Test (Section 9)
**Scenario:** Which features destabilise the confidence score most when ablated?

For each feature, measures: mean confidence change, standard deviation of change, % of patients with >0.10 confidence drop, % with >0.20 drop.

The feature causing the largest average confidence drop is the most important for the confidence score's reliability. DSS rule: make this a required field in the intake form.

#### Missing Data Simulation (Section 10)
**Scenario:** At a real clinic, not every field will be filled in.

For each count of missing features (1 through 8), 50 random feature subsets are blanked (replaced with median). AUC, PR-AUC, and mean confidence are plotted with ±1 std bands.

**Key finding from v1 run:** The model could tolerate only 1 missing feature before AUC dropped >5%. This is a fragility concern - the DSS UI should show a completeness score and warn when 2+ features are missing.

#### Hormonal Contraceptives Duration Sweep (Section 11)
HC use is one of the strongest non-leakage correlates. A synthetic median patient is swept across 0–20 years of HC use, plotting both predicted probability and confidence score. Identifies the HC duration at which a median patient crosses the risk threshold, and how confidence varies across the range.

### 7.3 Intervention Rules Generated

Section 12 collects all findings into a structured table saved to `cervical_sensitivity_results.json`. Each rule has a scenario, finding, DSS rule, and severity (High/Medium). These rules feed directly into the DSS frontend logic - e.g., warning banners when STD_burden=0, completeness score when fields are missing.

---

## 8. Confidence Score - Full Explainability

### 8.1 Formula

```
confidence = a·x + b·y − c·z + d·m + e·(1−H) + f·agreement
```

| Term | Symbol | Description |
|---|---|---|
| Cosine similarity to centroid | `x` | How close the patient is to the typical profile of the predicted class in scaled feature space. High = strong match |
| Ensemble probability | `y` | The ensemble's predicted probability for the predicted class. High = model is certain |
| Confusion risk | `z` | `(1 - ens_prob) × cosine_sim_to_opposite_centroid`. High z = the model was pulled toward the wrong class |
| Demographic match | `m` | How close the patient's age is to the mean age of positive cases. Proxy for clinical plausibility |
| Entropy bonus | `1 - H` | `H` = normalised binary entropy of the ensemble probability. High `1-H` = low entropy = high certainty |
| Model agreement | `agreement` | Fraction of individual models that agree with the ensemble prediction. Low agreement = unreliable |

Weights `(a, b, c, d, e, f)` are jointly optimised on OOF data using Nelder-Mead to maximise separation between confidence scores of correct and incorrect predictions.

### 8.2 Confidence Tiers

| Score | Tier | Clinical Action |
|---|---|---|
| ≥ 0.85 | High confidence | Act on prediction directly |
| 0.70 – 0.85 | Moderate confidence | Consider follow-up or additional data collection |
| 0.55 – 0.70 | Low confidence - borderline | Collect missing features, re-assess |
| < 0.55 | Very low confidence - inconclusive | Do not act on this prediction alone |

### 8.3 Confidence Component Waterfall

`plot_confidence_waterfall()` produces a running-total bar chart where each bar shows one term's weighted contribution. Green = boosts confidence, red = penalises. The bars stack from left to right and the final value is the confidence score. A second panel shows raw (unweighted) component values.

**Use in DSS:** Surface this chart in the clinician UI alongside the prediction. A prediction with confidence 0.30 due entirely to low model agreement (f·agreement near zero) tells a different story than one with confidence 0.30 because the patient is in a borderline probability region.

### 8.4 Model Contribution Waterfall

`plot_model_contribution_waterfall()` shows each model's contribution as `weight_i × prob_i`. The bars stack to the ensemble probability. Individual model probabilities are also shown as bars coloured green (agrees with ensemble prediction) or red (disagrees).

**Example interpretation:**
```
LR   (w=0.31):  0.31 × 0.39 = 0.121  ← votes HIGH RISK
RF   (w=0.28):  0.28 × 0.10 = 0.028  ← votes LOW RISK
XGB  (w=0.24):  0.24 × 0.06 = 0.014  ← votes LOW RISK
LGB  (w=0.17):  0.17 × 0.26 = 0.044  ← votes LOW RISK
─────────────────────────────────────
Ensemble:                    0.207  → crosses 0.XX threshold → HIGH RISK
Only 1/4 models agree → low confidence
```

### 8.5 Natural Language Explanation

`generate_nl_explanation()` returns a paragraph for the DSS UI. One sentence per confidence component, naming the agreeing and disagreeing models explicitly. Example:

> The model predicts high risk (biopsy recommended) with an ensemble probability of 20.7% (threshold: 18.0%). Overall confidence is 0.31 - Low confidence - borderline.
>
> Why this confidence score:
> - Centroid similarity (0.41): The patient is at moderate distance from the typical high-risk profile, which slightly reduces confidence.
> - Model probability (0.207): The ensemble assigns a low probability to this prediction, moderately reducing confidence.
> - Confusion risk (0.18): There is moderate pull toward the opposite class centroid, slightly reducing confidence.
> - Demographic match (0.72): Patient age 28 is 6.2 years from the typical positive-case age (34), slightly reducing confidence.
> - Prediction certainty (0.44): Ensemble entropy is 0.56 - moderate certainty, moderately increasing confidence.
> - Model agreement (0.25): 1/4 models agree (LR). RF, XGB, LGB disagree, strongly reducing confidence.
>
> Biggest confidence boost: ensemble probability (+0.12)
> Biggest confidence penalty: model agreement (-0.09)

### 8.6 Patient Population Scatter

`plot_patient_scatter()` takes any patient row and projects them onto a PCA scatterplot of all 858 patients. Two panels: left coloured by true Biopsy label, right coloured by ensemble risk score. Cluster centroids are marked with diamonds and labelled. The current patient appears as a gold star with a callout box showing probability, prediction, and confidence score.

This is the most clinically legible output - a clinician can instantly see whether a patient is in a high-risk region surrounded by confirmed positives, or whether they are an isolated outlier flagged by the threshold.

---

## 9. Output Files & Plots

### Model bundle (`cervical_model_bundle_v2.joblib`)

The bundle is a dictionary containing everything needed for inference:

```python
bundle = {
    'models':             # {name: fitted + calibrated pipeline}
    'model_names':        # ordered list matching ensemble_weights
    'ensemble_weights':   # OOF-optimised weight array
    'feature_names':      # 9-element No-Dx feature list
    'feature_names_full': # 11-element full feature list
    'threshold':          # clinically tuned decision threshold
    'target_flag_rate':   # 0.20
    'conf_weights':       # 6-tuple (a,b,c,d,e,f)
    'centroids':          # {0: neg_centroid, 1: pos_centroid} in scaled space
    'scaler_conf':        # StandardScaler for confidence computation
    'demo_pos':           # {age_mean, age_std} of positive cases
    'pca':                # fitted PCA object (2 components)
    'pca_scaler':         # StandardScaler for PCA
    'pca_coords':         # (858, 2) array - pre-computed for all patients
    'full_ens_proba':     # pre-computed ensemble probability for all 858 patients
    'y_all':              # full label array (for scatter colouring)
    'cluster_labels':     # cluster assignment for all 858 patients
    'km_model':           # fitted KMeans (k=3)
    'km_scaler':          # StandardScaler for cluster assignment
    'cluster_features':   # 7-element feature list used for clustering
    'holdout_auc':        # final holdout ROC-AUC (report this number)
    'holdout_prauc':      # final holdout PR-AUC
    'cv_scores':          # {model_name: {metric: value}} on OOF train set
    'conf_formula':       # string describing the formula
    'conf_components':    # dict of weight labels and values
}
```

### Sensitivity results (`cervical_sensitivity_results.json`)

Structured JSON containing all quantitative findings from Q3. Consumed by the Shiny frontend to populate warning banners and intervention rules.

---

## 10. Key Findings

### From EDA (Q1)

- `STD_burden` and `Dx:HPV` are the strongest correlates with Biopsy in univariate analysis
- `Dx:Cancer` and `Dx:HPV` are near-leakage features - they require prior diagnoses unavailable at initial screening
- K-Means (k=3) cleanly separates Low/Medium/High risk groups with distinct Biopsy rates
- `STDs`, `STDs (number)`, and `STD_burden` are highly collinear - composite `STD_burden` used
- Abnormal rate rises with age despite lower patient volume in older groups

### From Modelling (Q2)

- 4-model soft-vote ensemble (LR + RF + XGB + LGB) with OOF-optimised weights outperforms any individual model on both ROC-AUC and PR-AUC
- ADASYN corrects the SMOTE-induced smoking direction artifact from v1
- Per-model isotonic calibration improves ensemble probability estimates
- Threshold tuned to ≤20% flag rate resolves the v1 bug where 81% of patients were flagged
- `Preg_x_Age` interaction feature (largest combined AUC drop in ablation) improves subgroup performance
- Confidence separation gap (μ_correct − μ_wrong) should be reported from v2 run output

### From Sensitivity Analysis (Q3)

- `Age` causes the largest AUC drop when ablated - make it a required field
- The model tolerates ≤1 missing feature before AUC drops >5% - DSS must show completeness score
- STD misreporting (zeroing STD_burden for all patients) causes only moderate classification flips - the model does not rely excessively on a single feature, but STD_burden=0 with other elevated features should trigger a warning
- IUD not recorded slightly inflates risk estimates (protective signal lost)
- High-risk cluster patients show the largest probability drops under "all features → median" perturbation, confirming the model captures cluster-specific risk patterns

---

## 11. DSS Frontend Integration

The frontend is an R Shiny app built by a teammate (not included in this repository).

### Connecting Shiny to the Python model

The recommended integration pattern is a **FastAPI backend** that Shiny calls via `httr2`:

**Python side (FastAPI endpoint):**
```python
# api.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib, base64, io
import matplotlib.pyplot as plt

bundle = joblib.load('cervical_model_bundle_v2.joblib')
app = FastAPI()

class PatientInput(BaseModel):
    age: float
    sexual_partners: float
    pregnancies: float
    smokes_years: float
    hc_years: float
    iud_years: float
    std_burden: float
    cluster: int

@app.post("/predict")
def predict(patient: PatientInput):
    # Build feature row, run ensemble, compute confidence,
    # generate explanations, render PCA scatter as base64 PNG
    ...
    return {
        "prediction":       "HIGH RISK" or "LOW RISK",
        "probability":      float,
        "threshold":        float,
        "confidence":       float,
        "confidence_tier":  str,
        "model_probs":      dict,
        "conf_components":  dict,
        "nl_explanation":   str,
        "scatter_png_b64":  str,   # base64-encoded PNG
        "waterfall_png_b64":str,
    }
```

**R Shiny side:**
```r
library(httr2)

predict_patient <- function(patient_data) {
  req <- request("http://localhost:8001/predict") |>
    req_body_json(patient_data) |>
    req_perform()
  resp_body_json(req)
}

# Render the scatter plot from base64
output$scatter_plot <- renderImage({
  result <- predict_patient(input_values())
  img_data <- base64enc::base64decode(result$scatter_png_b64)
  tmpfile <- tempfile(fileext = ".png")
  writeBin(img_data, tmpfile)
  list(src = tmpfile, contentType = "image/png", width = "100%")
})
```

### What the Shiny UI should display

**Input panel:**
- Age (numeric)
- Number of sexual partners (numeric)
- Number of pregnancies (numeric)
- Smokes (years) - with warning if 0 and patient is flagged as smoker elsewhere
- Hormonal Contraceptives (years)
- IUD (years)
- STD burden (count)
- Completeness indicator (how many of 9 fields are filled)

**Output panel:**
- Primary prediction badge (HIGH RISK / LOW RISK) with colour
- Ensemble probability bar
- Confidence score with tier label and colour
- Natural language explanation paragraph
- Model contribution waterfall chart (which models agreed/disagreed)
- Confidence component waterfall chart (which formula terms are helping/hurting)
- PCA population scatter with patient marked
- Warning banners for: STD_burden=0 with elevated other factors, ≥2 missing features, low model agreement

---

## 12. Known Limitations & Future Work

### Limitations

**Dataset:**
- 858 patients is small for a machine learning study - all results should be interpreted as preliminary
- Data collected at a single hospital in Venezuela - may not generalise to other clinical settings or populations
- Significant class imbalance (6.4% positive) means recall and PR-AUC are more informative than accuracy
- Missing value imputation (median/mode) may introduce bias - the two dropped columns (>50% missing) contained potentially informative STD timing data

**Modelling:**
- `Dx:Cancer` and `Dx:HPV` were excluded from the deployed model but their exclusion reduces AUC vs the full-feature model - the full comparison is included in Section 14 of `02_model_v2.ipynb`
- Confidence score weights are optimised on the same dataset used for training - in a real deployment, a separate calibration dataset would be needed
- TabPFN requires a licensed API key - without it the ensemble falls back to 4 models

**Sensitivity analysis:**
- All perturbations use median imputation as the "unknown" value - in a real clinical setting, missing-at-random may not reflect the true distribution of unknown values
- The sensitivity analysis does not account for the correlation structure between features when perturbing one at a time

### Future Work

- Collect additional data from multiple clinical sites to improve generalisability
- Implement proper temporal validation (train on earlier appointments, test on later ones)
- Add `Hinselmann`, `Schiller`, `Citology` as secondary targets and build a multi-target model
- Integrate with EHR system for automatic feature extraction rather than manual data entry
- A/B test the DSS intervention recommendations in a pilot clinic to measure downstream impact on biopsy referral rates and missed diagnoses

---

*Last updated: April 2026 | Carnegie Mellon University, Spring 2026*
