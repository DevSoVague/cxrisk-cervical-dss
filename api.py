"""
api.py — FastAPI backend for the Cervical Cancer Risk DSS
CxRisk, CMU Healthcare Information Systems team project

Run (from the repo root; uses the shipped bundle, or rebuild it with `python scripts/train.py`):
    uvicorn api:app --port 8001

The model bundle path defaults to models/cervical_model_bundle_v2.joblib and can be
overridden with the CXRISK_BUNDLE_PATH environment variable.

Shiny calls:
    POST http://localhost:8001/predict   (patient data → full prediction + plots)
    GET  http://localhost:8001/health    (uptime + bundle metadata)
    GET  http://localhost:8001/schema    (field names, types, ranges)
"""

from __future__ import annotations

import base64
import io
import logging
import math
import os
from pathlib import Path
from typing import Optional

import joblib
import matplotlib
matplotlib.use("Agg")   # headless — no display needed
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from sklearn.metrics.pairwise import cosine_similarity

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger("dss_api")

# ── Bundle path ───────────────────────────────────────────────────────────────
BUNDLE_PATH = Path(os.environ.get("CXRISK_BUNDLE_PATH",
                                  "models/cervical_model_bundle_v2.joblib"))

# ── Load once at startup ──────────────────────────────────────────────────────
if not BUNDLE_PATH.exists():
    raise FileNotFoundError(
        f"Model bundle not found at {BUNDLE_PATH}. "
        "Run `python scripts/train.py` (or notebooks/02_model_v2.ipynb) first to generate it."
    )

log.info("Loading model bundle …")
BUNDLE: dict = joblib.load(BUNDLE_PATH)

# Pull everything out of the bundle into module-level names for speed
MODELS           = BUNDLE["models"]
MODEL_NAMES      = BUNDLE["model_names"]
ENS_WEIGHTS      = BUNDLE["ensemble_weights"]
FEATURE_NAMES    = BUNDLE["feature_names"]      # 9-element No-Dx list
THRESHOLD        = BUNDLE["threshold"]
CONF_WEIGHTS     = BUNDLE["conf_weights"]       # (a, b, c, d, e, f)
CENTROIDS        = BUNDLE["centroids"]          # {0: neg_arr, 1: pos_arr}
SCALER_CONF      = BUNDLE["scaler_conf"]
DEMO_POS         = BUNDLE["demo_pos"]           # {age_mean, age_std}
PCA              = BUNDLE["pca"]
PCA_SCALER       = BUNDLE["pca_scaler"]
PCA_COORDS       = BUNDLE["pca_coords"]         # (858, 2) pre-computed
FULL_ENS_PROBA   = BUNDLE["full_ens_proba"]     # (858,)
Y_ALL            = BUNDLE["y_all"]              # (858,)
CLUSTER_LABELS   = BUNDLE["cluster_labels"]     # (858,)
KM_MODEL         = BUNDLE["km_model"]
KM_SCALER        = BUNDLE["km_scaler"]
CLUSTER_FEATURES = BUNDLE["cluster_features"]
HOLDOUT_AUC      = BUNDLE["holdout_auc"]
HOLDOUT_PRAUC    = BUNDLE["holdout_prauc"]
CV_SCORES        = BUNDLE["cv_scores"]

log.info("Bundle loaded. Models: %s  |  Threshold: %.3f  |  Holdout AUC: %.4f",
         MODEL_NAMES, THRESHOLD, HOLDOUT_AUC)

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Cervical Cancer Risk DSS",
    description="Decision support system for cervical cancer screening. "
                "Returns risk prediction, confidence score, natural language "
                "explanation, and visualisation charts.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten for production
    allow_methods=["*"],
    allow_headers=["*"],
)


# ═══════════════════════════════════════════════════════════════════════════════
# Pydantic schemas
# ═══════════════════════════════════════════════════════════════════════════════

class PatientInput(BaseModel):
    """
    All nine No-Dx features used by the model.
    Cluster is optional — if omitted it is inferred from the other features.
    """
    age: float               = Field(...,  ge=0,  le=120, description="Patient age in years")
    sexual_partners: float   = Field(...,  ge=0,          description="Number of sexual partners")
    pregnancies: float       = Field(...,  ge=0,          description="Number of pregnancies")
    smokes_years: float      = Field(0.0,  ge=0,          description="Years of smoking (0 if non-smoker)")
    hc_years: float          = Field(0.0,  ge=0,          description="Years of hormonal contraceptive use")
    iud_years: float         = Field(0.0,  ge=0,          description="Years of IUD use")
    std_burden: float        = Field(0.0,  ge=0,          description="STD burden score (sum of STD indicators)")
    cluster: Optional[int]   = Field(None, ge=0,  le=2,   description="Risk cluster (0=low, 1=medium, 2=high). Inferred if omitted.")

    # Shiny sends JSON with snake_case — no alias needed, but validate ranges
    @field_validator("age")
    @classmethod
    def age_reasonable(cls, v: float) -> float:
        if v < 10:
            raise ValueError("Age below 10 is unlikely for this screening context.")
        return v


class PredictionResponse(BaseModel):
    # Core prediction
    prediction:       str    # "HIGH RISK" | "LOW RISK"
    probability:      float  # ensemble P(Biopsy=1)
    threshold:        float  # decision threshold used
    # Confidence
    confidence:       float  # 0–1
    confidence_tier:  str
    # Per-model
    model_probs:      dict   # {model_name: probability}
    ensemble_weights: dict   # {model_name: weight}
    # Explainability
    conf_components:  dict   # {component_name: {weighted, raw, description}}
    nl_explanation:   str
    # Warnings for clinician
    warnings:         list[str]
    # Charts (base64 PNG)
    scatter_png_b64:     str
    waterfall_png_b64:   str
    contribution_png_b64:str


class HealthResponse(BaseModel):
    status:       str
    models:       list[str]
    holdout_auc:  float
    holdout_prauc:float
    threshold:    float
    bundle_path:  str


class SchemaResponse(BaseModel):
    fields:  dict
    feature_order: list[str]


# ═══════════════════════════════════════════════════════════════════════════════
# Helper: cluster inference
# ═══════════════════════════════════════════════════════════════════════════════

def infer_cluster(row_raw: dict) -> int:
    """
    Assign a patient to a risk cluster using the saved KMeans model.
    row_raw must contain all CLUSTER_FEATURES.
    """
    vals = np.array([[row_raw[f] for f in CLUSTER_FEATURES]])
    scaled = KM_SCALER.transform(vals)
    label = int(KM_MODEL.predict(scaled)[0])

    # Re-map to risk-sorted label (same logic as notebook)
    # The bundle stores cluster_labels sorted by Biopsy rate.
    # Since the KM_MODEL was saved after the rank_map was applied,
    # the cluster IDs in the bundle are already 0=low, 1=med, 2=high.
    return label


# ═══════════════════════════════════════════════════════════════════════════════
# Helper: build feature row from PatientInput
# ═══════════════════════════════════════════════════════════════════════════════

def build_feature_row(patient: PatientInput) -> pd.DataFrame:
    """
    Convert PatientInput → single-row DataFrame matching FEATURE_NAMES.
    FEATURE_NAMES = ['Age','Number of sexual partners','Num of pregnancies',
                     'Smokes (years)','Hormonal Contraceptives (years)',
                     'IUD (years)','STD_burden','Cluster','Preg_x_Age']
    """
    raw = {
        "Age":                               patient.age,
        "Number of sexual partners":         patient.sexual_partners,
        "Num of pregnancies":                patient.pregnancies,
        "Smokes (years)":                    patient.smokes_years,
        "Hormonal Contraceptives (years)":   patient.hc_years,
        "IUD (years)":                       patient.iud_years,
        "STD_burden":                        patient.std_burden,
        "Preg_x_Age":                        patient.pregnancies * patient.age,
    }

    # Cluster: use provided value or infer
    cluster_input = {
        "Age":                             patient.age,
        "Number of sexual partners":       patient.sexual_partners,
        "Num of pregnancies":              patient.pregnancies,
        "Smokes (years)":                  patient.smokes_years,
        "Hormonal Contraceptives (years)": patient.hc_years,
        "IUD (years)":                     patient.iud_years,
        "STD_burden":                      patient.std_burden,
    }
    raw["Cluster"] = patient.cluster if patient.cluster is not None \
                     else infer_cluster(cluster_input)

    return pd.DataFrame([raw])[FEATURE_NAMES]


# ═══════════════════════════════════════════════════════════════════════════════
# Helper: ensemble prediction
# ═══════════════════════════════════════════════════════════════════════════════

def predict_ensemble(X: pd.DataFrame) -> tuple[np.ndarray, dict]:
    """
    Returns (ensemble_proba_array, {model_name: proba_array}).
    """
    ind_probs = {}
    for name, model in MODELS.items():
        if name == "TabPFN":
            ind_probs[name] = model.predict_proba(X.values)[:, 1]
        else:
            ind_probs[name] = model.predict_proba(X)[:, 1]

    mat = np.column_stack([ind_probs[n] for n in MODEL_NAMES])
    ens = mat @ ENS_WEIGHTS
    return ens, ind_probs


# ═══════════════════════════════════════════════════════════════════════════════
# Helper: confidence score
# ═══════════════════════════════════════════════════════════════════════════════

def _binary_entropy_norm(p: float) -> float:
    eps = 1e-9
    p = float(np.clip(p, eps, 1 - eps))
    H = -(p * math.log(p) + (1 - p) * math.log(1 - p))
    return H / math.log(2)


def compute_confidence(
    pred_class: int,
    ens_prob: float,
    ind_probs: dict,
    row_scaled: np.ndarray,
    row_raw: pd.Series,
) -> tuple[float, dict]:
    """
    Returns (score, components_dict).
    components_dict: {key: {weighted, raw, description}}
    """
    a, b, c, d, e, f = CONF_WEIGHTS

    x = max(0.0, float(cosine_similarity(
        row_scaled, CENTROIDS[pred_class].reshape(1, -1)
    )[0, 0]))

    y_s = float(ens_prob) if pred_class == 1 else 1.0 - float(ens_prob)

    opp = 1 - pred_class
    cos_opp = max(0.0, float(cosine_similarity(
        row_scaled, CENTROIDS[opp].reshape(1, -1)
    )[0, 0]))
    z = (1.0 - float(ens_prob)) * cos_opp

    age_z = abs(row_raw["Age"] - DEMO_POS["age_mean"]) / (DEMO_POS["age_std"] + 1e-6)
    m = max(0.0, 1.0 - age_z / 3.0) if pred_class == 1 else \
        float(np.clip(1.0 - age_z / (3.0 * DEMO_POS["age_std"] + 1e-6), 0, 1))

    H = _binary_entropy_norm(float(ens_prob))
    votes = [int((float(ind_probs[n][0]) >= 0.5) == pred_class) for n in ind_probs]
    agreement = float(np.mean(votes))

    components = {
        "a_cosine_sim":     {"weighted": round(a * x,          4), "raw": round(x,         4), "description": "cosine similarity to class centroid"},
        "b_model_prob":     {"weighted": round(b * y_s,        4), "raw": round(y_s,        4), "description": "ensemble probability"},
        "c_confusion_risk": {"weighted": round(-c * z,         4), "raw": round(z,          4), "description": "confusion risk (pull to opposite class)"},
        "d_demographic":    {"weighted": round(d * m,          4), "raw": round(m,          4), "description": "demographic profile match"},
        "e_entropy_bonus":  {"weighted": round(e * (1 - H),    4), "raw": round(1 - H,      4), "description": "certainty bonus (1 - entropy)"},
        "f_agreement":      {"weighted": round(f * agreement,  4), "raw": round(agreement,  4), "description": "model agreement fraction"},
    }
    score = float(np.clip(sum(v["weighted"] for v in components.values()), 0, 1))
    return score, components


def confidence_tier(score: float) -> str:
    if score >= 0.85: return "High confidence"
    if score >= 0.70: return "Moderate confidence — consider follow-up"
    if score >= 0.55: return "Low confidence — borderline"
    return "Very low confidence — inconclusive"


# ═══════════════════════════════════════════════════════════════════════════════
# Helper: natural language explanation
# ═══════════════════════════════════════════════════════════════════════════════

def generate_nl_explanation(
    pred_class: int,
    ens_prob: float,
    ind_probs: dict,
    score: float,
    components: dict,
    threshold: float,
    feature_row: pd.Series,
) -> str:
    pred_lbl = "high risk (biopsy recommended)" if pred_class == 1 else "low risk"
    tier = confidence_tier(score)
    model_preds = {n: int(float(p[0]) >= 0.5) for n, p in ind_probs.items()}
    agree_names    = [n for n, mp in model_preds.items() if mp == pred_class]
    disagree_names = [n for n, mp in model_preds.items() if mp != pred_class]

    lines = [
        f"The model predicts {pred_lbl} with an ensemble probability of "
        f"{ens_prob:.1%} (threshold: {threshold:.1%}).",
        f"Overall confidence is {score:.2f} ({tier}).",
        "",
        "Why this confidence score:",
    ]

    for key, info in components.items():
        weighted_val = info["weighted"]
        raw_val      = info["raw"]
        verb_s, verb_ing = ("increases", "increasing") if weighted_val >= 0 else ("reduces", "reducing")
        magnitude  = ("strongly"   if abs(weighted_val) > 0.10 else
                      "moderately" if abs(weighted_val) > 0.04 else "slightly")

        if key == "a_cosine_sim":
            quality = ("close to"               if raw_val > 0.6 else
                       "at moderate distance from" if raw_val > 0.3 else "far from")
            lines.append(
                f"  • Centroid similarity ({raw_val:.3f}): The patient is {quality} the "
                f"typical {'high' if pred_class == 1 else 'low'}-risk profile, "
                f"which {magnitude} {verb_s} confidence."
            )
        elif key == "b_model_prob":
            lines.append(
                f"  • Model probability ({raw_val:.3f}): The ensemble assigns a "
                f"{'high' if raw_val > 0.5 else 'low'} probability to this prediction, "
                f"{magnitude} {verb_ing} confidence."
            )
        elif key == "c_confusion_risk":
            risk_lbl = "high" if raw_val > 0.3 else "low"
            lines.append(
                f"  • Confusion risk ({raw_val:.3f}): There is {risk_lbl} pull toward the "
                f"opposite class centroid, {magnitude} {verb_ing} confidence."
            )
        elif key == "d_demographic":
            age      = feature_row["Age"]
            age_diff = abs(age - DEMO_POS["age_mean"])
            lines.append(
                f"  • Demographic match ({raw_val:.3f}): Patient age {age:.0f} is "
                f"{age_diff:.1f} years {'from' if age_diff > 5 else 'near'} the typical "
                f"positive-case age ({DEMO_POS['age_mean']:.0f}), "
                f"{magnitude} {verb_ing} confidence."
            )
        elif key == "e_entropy_bonus":
            entropy_val = 1 - raw_val
            certainty = ("high"     if entropy_val < 0.4 else
                         "moderate" if entropy_val < 0.7 else "low")
            lines.append(
                f"  • Prediction certainty ({raw_val:.3f}): Ensemble entropy is "
                f"{entropy_val:.3f} — {certainty} certainty in the probability estimate, "
                f"{magnitude} {verb_ing} confidence."
            )
        elif key == "f_agreement":
            n_agree = len(agree_names)
            n_total = len(ind_probs)
            if disagree_names:
                lines.append(
                    f"  • Model agreement ({raw_val:.2f}): {n_agree}/{n_total} models agree "
                    f"({', '.join(agree_names)}). "
                    f"{', '.join(disagree_names)} disagree{'s' if len(disagree_names)==1 else ''}, "
                    f"{magnitude} {verb_ing} confidence."
                )
            else:
                lines.append(
                    f"  • Model agreement ({raw_val:.2f}): All {n_total} models agree on "
                    f"this prediction, {magnitude} increasing confidence."
                )

    sorted_comps  = sorted(components.items(), key=lambda x: x[1]["weighted"])
    biggest_boost   = sorted_comps[-1]
    biggest_penalty = sorted_comps[0]
    lines += [
        "",
        f"Biggest confidence boost: {biggest_boost[1]['description']} (+{biggest_boost[1]['weighted']:.3f})",
        f"Biggest confidence penalty: {biggest_penalty[1]['description']} ({biggest_penalty[1]['weighted']:.3f})",
    ]
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# Helper: clinical warnings
# ═══════════════════════════════════════════════════════════════════════════════

def build_warnings(patient: PatientInput, ind_probs: dict, pred_class: int) -> list[str]:
    warnings = []

    # STD_burden = 0 but other elevated risk factors present
    elevated = (patient.age > 40 or patient.sexual_partners > 4
                or patient.smokes_years > 5 or patient.hc_years > 5)
    if patient.std_burden == 0 and elevated and pred_class == 1:
        warnings.append(
            "STD burden is recorded as 0 but other elevated risk factors are present. "
            "Consider verifying STD history — under-reporting is common in this dataset."
        )

    # Low model agreement
    model_preds = [int(float(p[0]) >= 0.5) for p in ind_probs.values()]
    agreement_rate = np.mean([mp == pred_class for mp in model_preds])
    if agreement_rate < 0.6:
        n_agree = int(round(agreement_rate * len(model_preds)))
        warnings.append(
            f"Low model agreement: only {n_agree}/{len(model_preds)} models agree with "
            "the ensemble decision. Treat this prediction with extra caution."
        )

    # IUD not recorded (0) — protective signal may be missing
    if patient.iud_years == 0 and pred_class == 1:
        warnings.append(
            "IUD use is recorded as 0 years. If IUD status is unknown rather than confirmed "
            "absent, the protective effect may be missing and risk could be overestimated."
        )

    return warnings


# ═══════════════════════════════════════════════════════════════════════════════
# Helper: charts → base64
# ═══════════════════════════════════════════════════════════════════════════════

def _fig_to_b64(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def plot_confidence_waterfall(
    components: dict,
    score: float,
    ens_prob: float,
    pred_class: int,
) -> str:
    labels        = [v["description"] for v in components.values()]
    contributions = [v["weighted"]    for v in components.values()]
    raw_vals      = [v["raw"]         for v in components.values()]

    colors_bar = ["#34d399" if v >= 0 else "#f87171" for v in contributions]
    running = [0.0]
    for v in contributions:
        running.append(running[-1] + v)
    bottoms = running[:-1]

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    for i, (lbl, val, bot, c) in enumerate(zip(labels, contributions, bottoms, colors_bar)):
        axes[0].bar(i, val, bottom=bot, color=c, edgecolor="white", linewidth=0.5, width=0.7)
        axes[0].text(i, bot + val + 0.005 * np.sign(val) if val != 0 else bot + 0.005,
                     f"{val:+.3f}", ha="center",
                     va="bottom" if val >= 0 else "top",
                     fontsize=8, fontweight="bold")

    axes[0].axhline(score, color="black", lw=2, ls="--", label=f"Final score = {score:.3f}")
    axes[0].axhline(0, color="k", lw=0.8, alpha=0.4)
    axes[0].set_xticks(range(len(labels)))
    axes[0].set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    axes[0].set_ylabel("Confidence contribution")
    axes[0].set_title(f"Confidence Score Waterfall\n{confidence_tier(score)}", fontweight="bold")
    axes[0].legend(fontsize=9)

    bar_colors = ["#60a5fa", "#34d399", "#f87171", "#fbbf24", "#a855f7", "#fb923c"]
    bars = axes[1].barh(labels, raw_vals, color=bar_colors, edgecolor="white", height=0.6)
    for bar, v in zip(bars, raw_vals):
        axes[1].text(v + 0.01, bar.get_y() + bar.get_height() / 2,
                     f"{v:.3f}", va="center", fontsize=9)
    axes[1].set_xlim(0, 1.2)
    axes[1].set_xlabel("Raw component value (before weighting)")
    axes[1].set_title("Raw Component Values", fontweight="bold")
    axes[1].tick_params(axis="y", labelsize=9)

    pred_lbl = "HIGH RISK" if pred_class == 1 else "LOW RISK"
    fig.suptitle(
        f"Prediction: {pred_lbl}  |  P(Biopsy=1) = {ens_prob:.4f}  |  Confidence = {score:.4f}",
        fontweight="bold", fontsize=11,
    )
    plt.tight_layout()
    return _fig_to_b64(fig)


def plot_model_contribution_waterfall(
    ind_probs: dict,
    ens_prob: float,
    threshold: float,
    pred_class: int,
) -> str:
    names       = MODEL_NAMES
    model_probs = np.array([float(ind_probs[n][0]) for n in names])
    contributions = ENS_WEIGHTS * model_probs
    model_preds = (model_probs >= 0.5).astype(int)
    bar_colors  = ["#34d399" if mp == pred_class else "#f87171" for mp in model_preds]

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    bars = axes[0].bar(names, model_probs, color=bar_colors, edgecolor="white", linewidth=0.5, width=0.6)
    for bar, v, w in zip(bars, model_probs, ENS_WEIGHTS):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f"p={v:.3f}\n(w={w:.2f})", ha="center", fontsize=9, fontweight="bold")
    axes[0].axhline(threshold, color="black", lw=2, ls="--", label=f"Threshold = {threshold:.3f}")
    axes[0].axhline(ens_prob,  color="crimson", lw=2, ls="-",  label=f"Ensemble = {ens_prob:.3f}")
    axes[0].set_ylim(0, 1.1)
    axes[0].set_ylabel("P(Biopsy=1)")
    axes[0].set_title("Individual Model Probabilities\n(green = agrees with ensemble)", fontweight="bold")
    axes[0].legend(fontsize=9)

    running = [0.0]
    for v in contributions:
        running.append(running[-1] + v)
    bottoms = running[:-1]

    for i, (n, val, bot) in enumerate(zip(names, contributions, bottoms)):
        axes[1].bar(i, val, bottom=bot, color="#4878CF", edgecolor="white", linewidth=0.5, width=0.7)
        axes[1].text(i, bot + val / 2,
                     f"w={ENS_WEIGHTS[i]:.2f}\n×{model_probs[i]:.3f}\n={val:.3f}",
                     ha="center", va="center", fontsize=8, color="white", fontweight="bold")

    axes[1].bar(len(names), ens_prob, color="crimson", edgecolor="white", linewidth=0.5, width=0.7)
    axes[1].text(len(names), ens_prob / 2, f"={ens_prob:.3f}",
                 ha="center", va="center", fontsize=9, color="white", fontweight="bold")
    axes[1].axhline(threshold, color="black", lw=2, ls="--", label=f"Threshold = {threshold:.3f}")
    axes[1].set_xticks(range(len(names) + 1))
    axes[1].set_xticklabels(names + ["Ensemble"], fontsize=10)
    axes[1].set_ylabel("Cumulative weighted probability")
    axes[1].set_title("Model Contribution Waterfall\n(each bar = weight × model_prob)", fontweight="bold")
    axes[1].legend(fontsize=9)

    n_agree = int((model_preds == pred_class).sum())
    pred_lbl = "HIGH RISK" if pred_class == 1 else "LOW RISK"
    fig.suptitle(
        f"Model Contributions — Prediction: {pred_lbl}  ({n_agree}/{len(names)} models agree)",
        fontweight="bold", fontsize=11,
    )
    plt.tight_layout()
    return _fig_to_b64(fig)


def plot_patient_scatter(
    patient_row: pd.DataFrame,
    ens_prob_patient: float,
    pred_class_patient: int,
    conf_score: float,
) -> str:
    patient_scaled = PCA_SCALER.transform(patient_row)
    patient_pca    = PCA.transform(patient_scaled)

    cluster_colors_map = {0: "#34d399", 1: "#fbbf24", 2: "#E74C3C"}
    cluster_labels_map = {0: "Low risk", 1: "Medium risk", 2: "High risk"}

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    for ax, color_vals, cbar_label, title in [
        (axes[0], Y_ALL.astype(float),  "True Biopsy Label",   "Population — True Labels"),
        (axes[1], FULL_ENS_PROBA,       "Ensemble Risk Score", "Population — Risk Score"),
    ]:
        sc = ax.scatter(
            PCA_COORDS[:, 0], PCA_COORDS[:, 1],
            c=color_vals, cmap="RdYlGn_r",
            s=np.clip(FULL_ENS_PROBA * 200 + 15, 10, 100),
            alpha=0.45, linewidths=0.2, edgecolors="white", vmin=0, vmax=1,
        )
        plt.colorbar(sc, ax=ax, label=cbar_label, shrink=0.75)

        for c_id in [0, 1, 2]:
            mask = (CLUSTER_LABELS == c_id)
            cx = PCA_COORDS[mask, 0].mean()
            cy = PCA_COORDS[mask, 1].mean()
            ax.scatter(cx, cy, marker="D", s=120,
                       color=cluster_colors_map[c_id], edgecolors="black", linewidths=1.2, zorder=4)
            ax.annotate(
                cluster_labels_map[c_id], (cx, cy),
                xytext=(cx + 0.3, cy + 0.3), fontsize=8,
                color=cluster_colors_map[c_id], fontweight="bold",
                path_effects=[pe.withStroke(linewidth=2, foreground="white")],
            )

        ax.scatter(patient_pca[0, 0], patient_pca[0, 1],
                   marker="*", s=600, c="gold", zorder=6, edgecolors="black", linewidths=1.5)

        pred_txt = "HIGH RISK" if pred_class_patient == 1 else "LOW RISK"
        callout  = (f"Current patient\n"
                    f"P = {ens_prob_patient:.3f}\n"
                    f"{pred_txt}\n"
                    f"Conf = {conf_score:.2f}")
        ax.annotate(
            callout,
            (patient_pca[0, 0], patient_pca[0, 1]),
            xytext=(patient_pca[0, 0] + 1.0, patient_pca[0, 1] + 1.0),
            fontsize=8, fontweight="bold", color="black",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="gold", alpha=0.9, edgecolor="black"),
            arrowprops=dict(arrowstyle="->", color="black", lw=1.5),
        )
        ax.set_xlabel(f"PC1 ({PCA.explained_variance_ratio_[0]:.1%} var)", fontsize=9)
        ax.set_ylabel(f"PC2 ({PCA.explained_variance_ratio_[1]:.1%} var)", fontsize=9)
        ax.set_title(title, fontweight="bold")

    fig.suptitle("Patient Position in Population Risk Space (PCA)", fontweight="bold", fontsize=12)
    plt.tight_layout()
    return _fig_to_b64(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# Routes
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/health", response_model=HealthResponse, tags=["Utility"])
def health():
    """Liveness check + bundle metadata. Call from Shiny on startup."""
    return HealthResponse(
        status="ok",
        models=MODEL_NAMES,
        holdout_auc=HOLDOUT_AUC,
        holdout_prauc=HOLDOUT_PRAUC,
        threshold=float(THRESHOLD),
        bundle_path=str(BUNDLE_PATH),
    )


@app.get("/schema", response_model=SchemaResponse, tags=["Utility"])
def schema():
    """Field names, types, and accepted ranges for the Shiny input form."""
    return SchemaResponse(
        fields={
            "age":              {"type": "float", "min": 10,  "max": 120, "required": True},
            "sexual_partners":  {"type": "float", "min": 0,   "max": None,"required": True},
            "pregnancies":      {"type": "float", "min": 0,   "max": None,"required": True},
            "smokes_years":     {"type": "float", "min": 0,   "max": None,"required": False, "default": 0},
            "hc_years":         {"type": "float", "min": 0,   "max": None,"required": False, "default": 0},
            "iud_years":        {"type": "float", "min": 0,   "max": None,"required": False, "default": 0},
            "std_burden":       {"type": "float", "min": 0,   "max": None,"required": False, "default": 0},
            "cluster":          {"type": "int",   "min": 0,   "max": 2,   "required": False,
                                 "note": "Inferred from other features if omitted"},
        },
        feature_order=FEATURE_NAMES,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(patient: PatientInput):
    """
    Main inference endpoint. Accepts patient features, returns full DSS output:
    prediction, probability, confidence score, explanations, and three charts.
    """
    try:
        # 1. Build feature row
        X = build_feature_row(patient)
        log.info("Predicting for patient: %s", X.to_dict(orient="records")[0])

        # 2. Ensemble prediction
        ens_proba_arr, ind_probs = predict_ensemble(X)
        ens_prob  = float(ens_proba_arr[0])
        pred_class = int(ens_prob >= THRESHOLD)

        # 3. Confidence score
        row_scaled = SCALER_CONF.transform(X)
        score, components = compute_confidence(
            pred_class, ens_prob, ind_probs,
            row_scaled, X.iloc[0],
        )

        # 4. Natural language explanation
        nl = generate_nl_explanation(
            pred_class, ens_prob, ind_probs,
            score, components, float(THRESHOLD), X.iloc[0],
        )

        # 5. Clinical warnings
        warnings = build_warnings(patient, ind_probs, pred_class)

        # 6. Charts
        waterfall_b64    = plot_confidence_waterfall(components, score, ens_prob, pred_class)
        contribution_b64 = plot_model_contribution_waterfall(ind_probs, ens_prob, float(THRESHOLD), pred_class)
        scatter_b64      = plot_patient_scatter(X, ens_prob, pred_class, score)

        return PredictionResponse(
            prediction        = "HIGH RISK" if pred_class == 1 else "LOW RISK",
            probability       = round(ens_prob, 4),
            threshold         = round(float(THRESHOLD), 4),
            confidence        = round(score, 4),
            confidence_tier   = confidence_tier(score),
            model_probs       = {n: round(float(p[0]), 4) for n, p in ind_probs.items()},
            ensemble_weights  = {n: round(float(w), 4) for n, w in zip(MODEL_NAMES, ENS_WEIGHTS)},
            conf_components   = components,
            nl_explanation    = nl,
            warnings          = warnings,
            scatter_png_b64      = scatter_b64,
            waterfall_png_b64    = waterfall_b64,
            contribution_png_b64 = contribution_b64,
        )

    except Exception as exc:
        log.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ═══════════════════════════════════════════════════════════════════════════════
# Dev entry point
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8001, reload=True)
