# CxRisk: Cervical Cancer Risk Ensemble, FastAPI Service and Shiny Frontend

A calibrated four-model ensemble that flags patients for biopsy follow-up from screening-visit risk factors, served over a FastAPI endpoint that returns a probability, a confidence score, a plain-language explanation and three explanation charts, with an R Shiny clinician frontend that consumes it.

![Python](https://img.shields.io/badge/python-3.10-blue) ![R](https://img.shields.io/badge/R-Shiny-blue) ![License: MIT](https://img.shields.io/badge/license-MIT-green)

![Patient position in population risk space](assets/model_plots_v2/patient_scatter.png)

## What it does

- Predicts `Biopsy` on the UCI cervical cancer risk-factor cohort (858 patients, 6.4% positive) from 9 screening-time features, with prior HPV/cancer diagnoses deliberately left out as leakage.
- Ensembles Logistic Regression, Random Forest, XGBoost and LightGBM, each with ADASYN resampling and isotonic calibration; soft-vote weights are found by Nelder-Mead on out-of-fold PR-AUC (RF ends up at 0.75, XGB at 0.008).
- Picks the decision threshold (0.096) by maximizing F2 subject to flagging at most 20% of patients, down from an 81% flag rate in the first version.
- Scores a locked 20% stratified holdout once: ROC-AUC 0.7685, PR-AUC 0.1772.
- Serves `/predict`, `/health` and `/schema` with Pydantic validation; each prediction includes a six-term confidence score, per-model probabilities, clinical warnings and base64 PNG charts rendered server-side.
- Ships an R Shiny clinician app (patient search, risk assessment, model insights, clinician override, population dashboard) that runs in a no-login demo mode out of the box, or against Postgres with bcrypt logins.

## Architecture

```mermaid
flowchart LR
    A[UCI CSV] --> B[Clean + impute<br/>STD_burden, KMeans cluster,<br/>Preg_x_Age]
    B --> C[80/20 stratified split]
    C --> D[LR / RF / XGB / LGB<br/>ADASYN + isotonic calibration]
    D --> E[OOF weight search<br/>+ F2 threshold, flag rate <= 20%]
    E --> F[(models/cervical_model_bundle_v2.joblib)]
    F --> G[FastAPI api.py]
    H[R Shiny frontend<br/>frontend/app.R] -->|POST /predict| G
    G -->|JSON + base64 charts| H
    H -.->|DB mode only| I[(Postgres<br/>frontend/schema.sql)]
```

`scripts/train.py` (or `notebooks/02_model_v2.ipynb`) builds a single joblib bundle holding the calibrated models, ensemble weights, threshold, confidence-score weights, KMeans and PCA transforms, and the population coordinates used for the scatter chart. `api.py` loads that bundle once at startup, so no training code runs on the request path. The trained bundle is committed at `models/cervical_model_bundle_v2.joblib` (about 12 MB), so the API runs without retraining. The client sends seven raw fields; the service derives the risk cluster and the pregnancy-age interaction itself. The Shiny frontend calls `/predict` and falls back to a local logistic model in R if the API is unreachable. [docs/ASSESSMENT_WORKFLOW.md](docs/ASSESSMENT_WORKFLOW.md) traces one assessment end to end.

## Quickstart (end to end)

Requires Python 3.10 and, for the frontend, R 4.x.

```bash
git clone https://github.com/DevSoVague/cxrisk-cervical-dss.git
cd cxrisk-cervical-dss
python3.10 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# macOS only: XGBoost and LightGBM need OpenMP -> brew install libomp

# 1. (optional) retrain; overwrites the shipped models/cervical_model_bundle_v2.joblib (~40 s)
python scripts/train.py

# 2. start the scoring API (interactive docs at http://localhost:8001/docs)
uvicorn api:app --port 8001

# 3. in a second terminal, from the repo root: install R packages once, then start the app
Rscript frontend/install_packages.R --demo-only
Rscript -e "shiny::runApp('frontend', port = 3838)"   # open http://localhost:3838
```

With no database variables set the frontend starts in **demo mode**: no login, 60 synthetic demo patients drawn from the public UCI table, scoring through `http://localhost:8001/predict`, and the local R fallback model if the API is not running. Patients you add and overrides you save stay in memory for the session.

**DB mode** (the original deployment): create the tables with [frontend/schema.sql](frontend/schema.sql), install all R packages with `Rscript frontend/install_packages.R`, and set `SUPABASE_HOST`, `SUPABASE_USER` and `SUPABASE_PASSWORD`. The app then shows a shinymanager login checked against `app_user`, and `app_user.password_hash` must hold a bcrypt hash (for example `crypt('<password>', gen_salt('bf'))` with pgcrypto, or `bcrypt::hashpw()` in R). New patients are written to Postgres.

Environment variables (all optional for the demo flow; names only, no values are needed in the repo):

| Variable | Used by | Purpose |
|---|---|---|
| `CXRISK_BUNDLE_PATH` | api.py | Model bundle path (default `models/cervical_model_bundle_v2.joblib`) |
| `CERVICAL_API_URL` | frontend | Scoring endpoint; demo mode defaults to `http://localhost:8001/predict`, DB mode calls the API only when set |
| `CERVICAL_API_KEY` | frontend | Sent as a Bearer token if set |
| `CERVICAL_API_TIMEOUT` | frontend | Seconds, default 10 |
| `SUPABASE_HOST` | frontend | Postgres host; setting it switches to DB mode |
| `SUPABASE_USER`, `SUPABASE_PASSWORD` | frontend | Postgres credentials (DB mode) |
| `SUPABASE_DBNAME`, `SUPABASE_PORT`, `SUPABASE_SSLMODE` | frontend | Optional; default `postgres`, `6543`, `require` |
| `CXRISK_DATA_CSV` | frontend | Path to the UCI CSV if the app is not launched from the repo root |

Example request:

```bash
curl -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -d '{"age": 32, "sexual_partners": 3, "pregnancies": 2,
       "smokes_years": 0, "hc_years": 4, "iud_years": 0, "std_burden": 1}'
```

Response (charts truncated):

```json
{
  "prediction": "LOW RISK",
  "probability": 0.0814,
  "threshold": 0.0963,
  "confidence": 1.0,
  "confidence_tier": "High confidence",
  "model_probs": {"LR": 0.0912, "RF": 0.082, "XGB": 0.064, "LGB": 0.0623},
  "ensemble_weights": {"LR": 0.1518, "RF": 0.7474, "XGB": 0.0081, "LGB": 0.0928},
  "warnings": [],
  "nl_explanation": "The model predicts low risk with an ensemble probability of 8.1% ...",
  "scatter_png_b64": "iVBORw0...", "waterfall_png_b64": "...", "contribution_png_b64": "..."
}
```

`age`, `sexual_partners` and `pregnancies` are required; the rest default to 0, and `cluster` is inferred if omitted. Out-of-range input (for example `age` under 10) returns 422. Full endpoint and field reference: [docs/API.md](docs/API.md).

## Data

The public [UCI Cervical Cancer (Risk Factors)](https://archive.ics.uci.edu/dataset/383/cervical+cancer+risk+factors) dataset (CC BY 4.0) is included unmodified at `data/risk_factors_cervical_cancer.csv`, with citation in [data/README.md](data/README.md). It is de-identified public data; no other data is used, and the frontend's demo patients are synthetic identities attached to rows of this table.

## Results

All tuning used out-of-fold predictions on the 80% training split; the 20% holdout (172 patients, 11 positive) was scored once.

| Metric | Value |
|---|---|
| Holdout ROC-AUC | 0.7685 |
| Holdout PR-AUC (base rate 0.064) | 0.1772 |
| Holdout F2 at threshold 0.096 | 0.1613 |
| Holdout flag rate | 10.5% |
| Holdout recall / precision on Biopsy=1 | 0.18 / 0.11 |
| Ensemble OOF ROC-AUC / PR-AUC (train) | 0.5836 / 0.1031 |
| RF with prior Dx features, holdout ROC-AUC (comparison only) | 0.7143 |

The holdout has only 11 positives, so these numbers are noisy, and the gap between OOF and holdout AUC is worth keeping in mind. `scripts/train.py` reproduces the table and the published bundle metadata exactly with the pinned versions in `requirements.txt` (scikit-learn 1.7.2, imbalanced-learn 0.14.1, XGBoost 3.2.0, LightGBM 4.6.0); newer library versions run but give different numbers. Plots are in [assets/model_plots_v2/](assets/model_plots_v2/), and `notebooks/02_model_v2.ipynb` keeps its outputs (SHAP, calibration, subgroup AUC, threshold sweep).

This is a course project on a small single-site dataset, not a validated clinical tool.

## Project structure

```
cxrisk-cervical-dss/
├── api.py                    FastAPI service (/predict, /health, /schema)
├── scripts/train.py          Minimal training path, writes the model bundle
├── models/
│   ├── cervical_model_bundle_v2.joblib    Trained ensemble bundle (~12 MB)
│   └── cervical_model_metadata_v2.json    Weights, threshold, holdout metrics
├── frontend/
│   ├── app.R                 R Shiny clinician app (demo mode or DB mode)
│   ├── install_packages.R    R package installer
│   └── schema.sql            Postgres tables used in DB mode
├── notebooks/
│   ├── eda_v2.ipynb          EDA, cleaning, K-Means risk clusters
│   ├── 02_model_v2.ipynb     Full modelling notebook (outputs kept)
│   └── cervical_cancer_model_sensitivity.ipynb   Sensitivity / what-if analysis (v1 model)
├── data/                     UCI CSV + citation
├── assets/model_plots_v2/    Evaluation and explanation plots
├── docs/
│   ├── API.md                Endpoint and field reference
│   ├── ASSESSMENT_WORKFLOW.md   One assessment traced from login to override
│   └── model_readme.md       Original long-form course write-up
└── requirements.txt
```

## Team & credits

Healthcare Information Systems, Carnegie Mellon University (Spring 2026), Group 6: Esha Pandya, Devavrath Sandeep, Abigail Torbatian.

- Devavrath Sandeep: EDA, modelling pipeline, sensitivity analysis and the FastAPI service.
- The problem specification, process models, database design and the R Shiny clinician frontend are Group 6 team work. The demo mode, bcrypt login check and reconstructed `schema.sql` were added for this public release.

## License

MIT, see [LICENSE](LICENSE). The dataset is CC BY 4.0 from the UCI Machine Learning Repository.
