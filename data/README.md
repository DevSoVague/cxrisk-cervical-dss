# Data

`risk_factors_cervical_cancer.csv` is the public **UCI Cervical Cancer (Risk Factors)** dataset
(858 patients, 36 columns, missing values coded as `?`), collected at Hospital Universitario de
Caracas, Venezuela.

- Source: https://archive.ics.uci.edu/dataset/383/cervical+cancer+risk+factors
- License: CC BY 4.0 (redistributed here unmodified, with attribution)
- Citation: K. Fernandes, J. S. Cardoso, J. Fernandes. "Transfer Learning with Partial
  Observability Applied to Cervical Cancer Screening." Iberian Conference on Pattern
  Recognition and Image Analysis (IbPRIA), 2017.

`scripts/train.py`, the notebooks and the Shiny frontend (`frontend/app.R`, for its fallback model and demo-mode patients) read it from this folder. No other data files are used.
