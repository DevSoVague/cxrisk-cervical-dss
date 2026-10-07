"""
Train the CxRisk v2 ensemble and write the model bundle that api.py loads.

This is the minimal training path extracted from notebooks/02_model_v2.ipynb
(plots, SHAP, subgroup analysis and the full-vs-No-Dx comparison are left in the
notebook). Same data cleaning, same seeds, same models, same bundle keys.

Usage (from the repo root):
    python scripts/train.py
    python scripts/train.py --data data/risk_factors_cervical_cancer.csv \
                            --out models/cervical_model_bundle_v2.joblib

TabPFN is not used here: the published v2 bundle was trained without it
(metadata "tabpfn": false), so the ensemble is LR + RF + XGB + LGB.
"""

from __future__ import annotations

import argparse
import json
import warnings
from itertools import product
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from imblearn.over_sampling import ADASYN
from imblearn.pipeline import Pipeline as ImbPipeline
from scipy.optimize import minimize
from sklearn.calibration import CalibratedClassifierCV
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             f1_score, fbeta_score, roc_auc_score)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import (StratifiedKFold, cross_val_predict,
                                     train_test_split)
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

SEED = 42
N_FOLDS = 5
TARGET = "Biopsy"
OUTCOMES = ["Hinselmann", "Schiller", "Citology", "Biopsy"]
TARGET_FLAG_RATE = 0.20
CLUSTER_FEATURES = ["Age", "Number of sexual partners", "Num of pregnancies",
                    "Smokes (years)", "Hormonal Contraceptives (years)",
                    "IUD (years)", "STD_burden"]
FEATURES_NODX = CLUSTER_FEATURES + ["Cluster", "Preg_x_Age"]
FEATURES_FULL = CLUSTER_FEATURES + ["Cluster", "Dx:Cancer", "Dx:HPV", "Preg_x_Age"]


# ── 1. Data loading and feature preparation ─────────────────────────────────
def load_data(path: Path):
    df_raw = pd.read_csv(path, na_values="?")
    missing_pct = df_raw.isnull().mean()
    df = df_raw.drop(columns=missing_pct[missing_pct > 0.50].index.tolist())

    feature_cols = [c for c in df.columns if c not in OUTCOMES]
    binary_cols = [c for c in feature_cols if df[c].dropna().isin([0.0, 1.0]).all()]
    continuous_cols = [c for c in feature_cols if c not in binary_cols]
    for col in continuous_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(df[col].median())
    for col in binary_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(df[col].mode()[0])

    std_cols = [c for c in df.columns
                if c.startswith("STDs:") and c != "STDs: Number of diagnosis"]
    df["STD_burden"] = df[std_cols].sum(axis=1)

    # K-Means risk clusters (k=3), relabelled 0/1/2 by ascending biopsy rate
    scaler_cluster = StandardScaler()
    X_cluster = scaler_cluster.fit_transform(df[CLUSTER_FEATURES])
    km = KMeans(n_clusters=3, random_state=SEED, n_init=10)
    km.fit(X_cluster)
    cluster_biopsy = {c: df[km.labels_ == c][TARGET].mean() for c in range(3)}
    rank_map = {c: r for r, c in enumerate(sorted(cluster_biopsy, key=cluster_biopsy.get))}
    df["Cluster"] = pd.Series(km.labels_).map(rank_map).values

    df["Preg_x_Age"] = df["Num of pregnancies"] * df["Age"]
    return df, km, scaler_cluster


# ── 3. Model definitions (ADASYN inside each pipeline) ──────────────────────
def make_adasyn_pipeline(classifier):
    return ImbPipeline([
        ("scaler", StandardScaler()),
        ("resample", ADASYN(random_state=SEED, n_neighbors=5)),
        ("clf", classifier),
    ])


def build_models(y_train):
    pos_weight = int((y_train == 0).sum() / (y_train == 1).sum())
    return {
        "LR": make_adasyn_pipeline(LogisticRegression(
            C=0.5, solver="lbfgs", max_iter=2000,
            class_weight="balanced", random_state=SEED)),
        "RF": make_adasyn_pipeline(RandomForestClassifier(
            n_estimators=500, max_depth=None, min_samples_leaf=2,
            max_features="sqrt", class_weight="balanced",
            n_jobs=-1, random_state=SEED)),
        "XGB": make_adasyn_pipeline(xgb.XGBClassifier(
            n_estimators=400, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, scale_pos_weight=pos_weight,
            eval_metric="logloss", tree_method="hist",
            random_state=SEED, verbosity=0)),
        "LGB": make_adasyn_pipeline(lgb.LGBMClassifier(
            n_estimators=400, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, class_weight="balanced",
            random_state=SEED, verbose=-1)),
    }


def _binary_entropy_norm(p):
    eps = 1e-9
    p = np.clip(p, eps, 1 - eps)
    return float(-(p * np.log(p) + (1 - p) * np.log(1 - p)) / np.log(2))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/risk_factors_cervical_cancer.csv")
    ap.add_argument("--out", default="models/cervical_model_bundle_v2.joblib")
    ap.add_argument("--metadata", default="models/cervical_model_metadata_v2.json",
                    help="where to write the metadata JSON (set to '' to skip)")
    args = ap.parse_args()
    np.random.seed(SEED)

    df, km, scaler_cluster = load_data(Path(args.data))
    X_all = df[FEATURES_NODX].copy()
    y_all = df[TARGET].values
    print(f"Data: {X_all.shape}, biopsy positive rate {y_all.mean():.1%}")

    # ── 2. Stratified 20% holdout, locked until the final evaluation ────────
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=0.20, stratify=y_all, random_state=SEED)
    train_idx, test_idx = X_train.index, X_test.index

    # ── 4. Cross-validated OOF predictions + fit on train ───────────────────
    models = build_models(y_train)
    model_names = list(models.keys())
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    cv_scores = {}
    for name, pipe in models.items():
        oof_p = cross_val_predict(pipe, X_train, y_train, cv=cv,
                                  method="predict_proba")[:, 1]
        pred_05 = (oof_p >= 0.5).astype(int)
        cv_scores[name] = {
            "ROC-AUC": round(roc_auc_score(y_train, oof_p), 4),
            "PR-AUC": round(average_precision_score(y_train, oof_p), 4),
            "F1": round(f1_score(y_train, pred_05), 4),
            "Recall": round((pred_05[y_train == 1] == 1).mean(), 4),
            "Precision": round((pred_05[pred_05 == 1] == y_train[pred_05 == 1]).mean(), 4)
            if pred_05.sum() > 0 else 0,
            "Brier": round(brier_score_loss(y_train, oof_p), 4),
        }
        print(f"  {name:4s} OOF ROC-AUC {cv_scores[name]['ROC-AUC']:.4f}  "
              f"PR-AUC {cv_scores[name]['PR-AUC']:.4f}")
        pipe.fit(X_train, y_train)

    # ── 5. Per-model isotonic calibration ───────────────────────────────────
    calibrated = {}
    oof_cal = {}
    for name, pipe in models.items():
        cal = CalibratedClassifierCV(pipe, cv=3, method="isotonic")
        cal.fit(X_train, y_train)
        calibrated[name] = cal
        oof_cal[name] = cross_val_predict(cal, X_train, y_train, cv=cv,
                                          method="predict_proba")[:, 1]

    # ── 8. Ensemble weights: grid seed, then Nelder-Mead on OOF PR-AUC ──────
    oof_matrix = np.column_stack([oof_cal[n] for n in model_names])

    def ensemble_pr_auc(w):
        w = np.clip(w, 0.01, 1.0)
        w = w / w.sum()
        return -average_precision_score(y_train, oof_matrix @ w)

    n_m = len(model_names)
    best_score, best_w = 999, np.ones(n_m) / n_m
    for combo in product(*[np.arange(0.1, 0.7, 0.1)] * n_m):
        combo = np.array(combo)
        s = ensemble_pr_auc(combo)
        if s < best_score:
            best_score, best_w = s, combo
    opt = minimize(ensemble_pr_auc, x0=best_w, method="Nelder-Mead",
                   options={"xatol": 1e-5, "fatol": 1e-5, "maxiter": 50000})
    ens_w = np.clip(opt.x, 0.01, 1.0)
    ens_w = ens_w / ens_w.sum()
    ens_oof_p = oof_matrix @ ens_w
    print("Ensemble weights:", {n: round(float(w), 4) for n, w in zip(model_names, ens_w)})
    print(f"Ensemble OOF ROC-AUC {roc_auc_score(y_train, ens_oof_p):.4f}  "
          f"PR-AUC {average_precision_score(y_train, ens_oof_p):.4f}")

    # ── 9. Threshold: maximise F2 subject to flag rate <= 20% ───────────────
    thresholds = np.linspace(0.01, 0.99, 1000)
    f2 = np.array([fbeta_score(y_train, (ens_oof_p >= t).astype(int), beta=2,
                               zero_division=0) for t in thresholds])
    flag_rates = np.array([(ens_oof_p >= t).mean() for t in thresholds])
    valid = flag_rates <= TARGET_FLAG_RATE
    threshold = thresholds[valid][np.argmax(f2[valid])] if valid.any() \
        else thresholds[np.argmax(f2)]
    print(f"Threshold {threshold:.4f}  (OOF flag rate "
          f"{(ens_oof_p >= threshold).mean():.1%})")

    # ── 10. Final holdout evaluation (touched once) ─────────────────────────
    def predict_ensemble(X):
        probs = {n: m.predict_proba(X)[:, 1] for n, m in calibrated.items()}
        return np.column_stack([probs[n] for n in model_names]) @ ens_w

    ens_test_p = predict_ensemble(X_test)
    ens_test_pred = (ens_test_p >= threshold).astype(int)
    holdout_auc = roc_auc_score(y_test, ens_test_p)
    holdout_prauc = average_precision_score(y_test, ens_test_p)
    print(f"HOLDOUT ROC-AUC {holdout_auc:.4f}  PR-AUC {holdout_prauc:.4f}  "
          f"F2 {fbeta_score(y_test, ens_test_pred, beta=2):.4f}  "
          f"flag rate {ens_test_pred.mean():.1%}")

    full_ens_p = np.zeros(len(X_all))
    full_ens_p[list(train_idx)] = ens_oof_p
    full_ens_p[list(test_idx)] = ens_test_p

    # ── 12. Confidence-score weights, optimised on OOF (train only) ─────────
    scaler_conf = StandardScaler()
    X_conf_s = scaler_conf.fit_transform(X_all)
    centroids = {0: X_conf_s[y_all == 0].mean(axis=0), 1: X_conf_s[y_all == 1].mean(axis=0)}
    pos = y_all == 1
    demo_pos = {"age_mean": float(X_all.loc[pos, "Age"].mean()),
                "age_std": float(X_all.loc[pos, "Age"].std())}

    recs = []
    for i in range(len(X_train)):
        row_raw = X_train.iloc[i]
        row_scaled = scaler_conf.transform(X_train.iloc[[i]])
        ens_p = float(ens_oof_p[i])
        pc = int(ens_p >= threshold)
        x = max(0.0, float(cosine_similarity(row_scaled, centroids[pc].reshape(1, -1))[0, 0]))
        y_s = ens_p if pc == 1 else 1.0 - ens_p
        cos_opp = max(0.0, float(cosine_similarity(row_scaled, centroids[1 - pc].reshape(1, -1))[0, 0]))
        z = (1.0 - ens_p) * cos_opp
        age_z = abs(row_raw["Age"] - demo_pos["age_mean"]) / (demo_pos["age_std"] + 1e-6)
        m = max(0.0, 1.0 - age_z / 3.0) if pc == 1 else \
            float(np.clip(1.0 - age_z / (3.0 * demo_pos["age_std"] + 1e-6), 0, 1))
        agreement = float(np.mean([int((oof_cal[n][i] >= 0.5) == pc) for n in model_names]))
        recs.append({"x": x, "y": y_s, "z": z, "m": m, "H": _binary_entropy_norm(ens_p),
                     "agreement": agreement, "correct": int(pc == y_train[i])})
    cdf = pd.DataFrame(recs)
    ok = cdf["correct"].values

    def _score_vec(w, d):
        a, b, c, dd, e, f = w
        return np.clip(a * d["x"] + b * d["y"] - c * d["z"] + dd * d["m"]
                       + e * (1 - d["H"]) + f * d["agreement"], 0, 1)

    def _obj(w, d, ok, alpha=0.65, beta=0.25, gamma=0.10):
        if any(v < 0 for v in w):
            return 999.0
        s = _score_vec(w, d)
        sc, sw = s[ok == 1], s[ok == 0]
        if len(sc) == 0 or len(sw) == 0:
            return 999.0
        return float(-alpha * (sc.mean() - sw.mean()) - beta * (sc > 0.70).mean()
                     + gamma * (sw > 0.50).mean())

    best_loss, best_w6 = 999.0, None
    for w6 in product(np.arange(0.10, 0.40, 0.10), np.arange(0.20, 0.50, 0.10),
                      np.arange(0.05, 0.25, 0.10), np.arange(0.05, 0.20, 0.08),
                      np.arange(0.05, 0.25, 0.10), np.arange(0.02, 0.15, 0.06)):
        loss = _obj(w6, cdf, ok)
        if loss < best_loss:
            best_loss, best_w6 = loss, w6
    opt_c = minimize(_obj, x0=list(best_w6), args=(cdf, ok), method="Nelder-Mead",
                     options={"xatol": 1e-5, "fatol": 1e-5, "maxiter": 100000})
    conf_weights = tuple(np.clip(opt_c.x, 0.01, 0.99))
    s_all = _score_vec(conf_weights, cdf)
    print(f"Confidence weights {tuple(round(float(v), 4) for v in conf_weights)}  "
          f"separation gap {float(s_all[ok == 1].mean() - s_all[ok == 0].mean()):.4f}")

    # ── 13. PCA for the population scatter ──────────────────────────────────
    pca_scaler = StandardScaler()
    pca = PCA(n_components=2, random_state=SEED)
    X_pca_2d = pca.fit_transform(pca_scaler.fit_transform(X_all))

    # ── 15. Refit calibrated models on the full dataset and save bundle ─────
    print("Refitting calibrated models on full dataset ...")
    final_models = {}
    for name, model in build_models(y_train).items():
        cal = CalibratedClassifierCV(model, cv=3, method="isotonic")
        cal.fit(X_all, y_all)
        final_models[name] = cal

    bundle = {
        "models": final_models,
        "model_names": model_names,
        "ensemble_weights": ens_w,
        "feature_names": FEATURES_NODX,
        "feature_names_full": FEATURES_FULL,
        "threshold": threshold,
        "target_flag_rate": TARGET_FLAG_RATE,
        "conf_weights": conf_weights,
        "centroids": centroids,
        "scaler_conf": scaler_conf,
        "demo_pos": demo_pos,
        "pca": pca,
        "pca_scaler": pca_scaler,
        "pca_coords": X_pca_2d,
        "full_ens_proba": full_ens_p,
        "y_all": y_all,
        "cluster_labels": df["Cluster"].values,
        "km_model": km,
        "km_scaler": scaler_cluster,
        "cluster_features": CLUSTER_FEATURES,
        "holdout_auc": holdout_auc,
        "holdout_prauc": holdout_prauc,
        "cv_scores": cv_scores,
        "conf_formula": "score = a*x + b*y - c*z + d*m + e*(1-H) + f*agreement",
        "conf_components": dict(zip(
            ["a_cosine_sim", "b_model_proba", "c_confusion_risk",
             "d_demographic", "e_entropy_bonus", "f_agreement"],
            [round(float(v), 4) for v in conf_weights])),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, out, compress=3)
    print(f"Bundle saved -> {out}")

    if args.metadata:
        meta = {
            "version": "v2", "target": TARGET, "features": FEATURES_NODX,
            "models": model_names, "tabpfn": False, "resampling": "ADASYN",
            "calibration": "isotonic",
            "ensemble_weights": {n: round(float(w), 4) for n, w in zip(model_names, ens_w)},
            "threshold": round(float(threshold), 4),
            "flag_rate_target": TARGET_FLAG_RATE,
            "holdout_auc": round(holdout_auc, 4),
            "holdout_prauc": round(holdout_prauc, 4),
            "confidence_formula": bundle["conf_formula"],
        }
        Path(args.metadata).write_text(json.dumps(meta, indent=2) + "\n")
        print(f"Metadata saved -> {args.metadata}")


if __name__ == "__main__":
    main()
