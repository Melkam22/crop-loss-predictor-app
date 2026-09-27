"""XGBoost tuning variations for HarvestGuard, judged honestly.

Stage 1 (search): each variant gets its own Optuna study on 04's exact 5 grouped
folds (StratifiedGroupKFold, random_state=42), PR-AUC objective.
Stage 2 (confirm): the best config of every variant is re-scored on 10 FRESH
grouped folds (two new splits, random_state=7 and 123) that no search ever saw.
That removes the "best of N trials" optimism of stage 1.
The test set is not touched here.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[2]  # repo root
OUT = sys.argv[1] if len(sys.argv) > 1 else Path(__file__).parent / "results.json"
N_TRIALS = int(sys.argv[2]) if len(sys.argv) > 2 else 60
RS = 42
MAX_TREES, EARLY_STOP = 3000, 100
optuna.logging.set_verbosity(optuna.logging.WARNING)

df = pd.read_csv(f"{ROOT}/data/processed/crop_loss_model_ready.csv", dtype={"household_id": str})
tr_idx, te_idx = next(GroupShuffleSplit(1, test_size=0.2, random_state=RS).split(df, groups=df["household_id"]))
train = df.iloc[tr_idx].reset_index(drop=True)
assert len(train) == 56395

CAT = ["crop_name", "region_code"]
NUM = ["household_size", "is_rural", "rainfall_belg_mm", "rainfall_belg_pct_of_avg",
       "rainfall_meher_mm", "rainfall_meher_pct_of_avg"]
y = train["loss_occurred"].to_numpy()
groups = train["household_id"].to_numpy()
SPW = float((y == 0).sum() / (y == 1).sum())
train["region_crop"] = train["region_code"].astype(str) + "|" + train["crop_name"]


def folds(seed):
    return list(StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(train, y, groups=groups))


SEARCH_FOLDS = folds(42)
CONFIRM_FOLDS = folds(7) + folds(123)


# ---------- feature builders: fit on the fitting part only, applied to any rows ----------
def onehot_builder(fit_rows, extra_cat=()):
    cats = CAT + list(extra_cat)
    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(train.loc[fit_rows, cats])
    names = enc.get_feature_names_out(cats)

    def f(rows):
        d = train.loc[rows]
        return pd.concat([d[NUM].reset_index(drop=True),
                          pd.DataFrame(enc.transform(d[cats]), columns=names)], axis=1).astype("float32")
    return f


def native_builder(fit_rows, extra_cat=()):
    cats = CAT + list(extra_cat)
    levels = {c: sorted(train.loc[fit_rows, c].astype(str).unique()) for c in cats}

    def f(rows):
        d = train.loc[rows, NUM + cats].reset_index(drop=True).copy()
        for c in cats:
            d[c] = pd.Categorical(d[c].astype(str), categories=levels[c])  # unseen -> NaN
        return d
    return f


def target_encode(fit_rows, cols, m):
    """Smoothed loss rate per level. Fitting rows get out-of-fold values (inner
    grouped 5-fold) so a row never sees its own label; other rows get values from
    all fitting rows."""
    fit_rows = np.asarray(fit_rows)
    prior = y[fit_rows].mean()

    def table(rows, col):
        s = pd.DataFrame({"k": train.loc[rows, col].to_numpy(), "y": y[rows]}).groupby("k")["y"].agg(["sum", "count"])
        return (s["sum"] + m * prior) / (s["count"] + m)

    fit_vals = {c: np.full(len(fit_rows), prior) for c in cols}
    inner = StratifiedGroupKFold(5, shuffle=True, random_state=RS)
    for a, b in inner.split(fit_rows, y[fit_rows], groups=groups[fit_rows]):
        for c in cols:
            t = table(fit_rows[a], c)
            fit_vals[c][b] = train.loc[fit_rows[b], c].map(t).fillna(prior).to_numpy()
    full = {c: table(fit_rows, c) for c in cols}

    def f(rows):
        out = {}
        for c in cols:
            if len(rows) == len(fit_rows) and np.array_equal(rows, fit_rows):
                out[f"te_{c}"] = fit_vals[c]
            else:
                out[f"te_{c}"] = train.loc[rows, c].map(full[c]).fillna(prior).to_numpy()
        return pd.DataFrame(out).astype("float32")
    return f


def build(variant, fit_rows, te_m=20):
    kind = variant["features"]
    if kind == "onehot":
        return onehot_builder(fit_rows)
    if kind == "onehot+pair":
        return onehot_builder(fit_rows, ["region_crop"])
    if kind == "native":
        return native_builder(fit_rows)
    if kind == "native+pair":
        return native_builder(fit_rows, ["region_crop"])
    if kind in ("onehot+te", "native+te"):
        base = (onehot_builder if kind.startswith("onehot") else native_builder)(fit_rows)
        te = target_encode(fit_rows, ["crop_name", "region_crop"], te_m)
        return lambda rows: pd.concat([base(rows), te(rows)], axis=1)
    raise ValueError(kind)


# ---------- CV ----------
FIXED = {"tree_method": "hist", "eval_metric": "aucpr", "n_jobs": 8}


def xgb_cv(variant, params, fold_list, trial=None, seeds=(RS,)):
    pr, roc, trees = [], [], []
    native = variant["features"].startswith("native")
    for k, (fit, val) in enumerate(fold_list):
        # early stopping on an inner 10% household split of the fitting part (never the scored fold)
        a, b = next(GroupShuffleSplit(1, test_size=0.1, random_state=RS).split(fit, groups=groups[fit]))
        es_fit, es_val = fit[a], fit[b]
        te_m = params.get("te_m", 20)
        f = build(variant, es_fit, te_m)
        Xa, Xb, Xv = f(es_fit), f(es_val), f(val)
        p = {k2: v for k2, v in params.items() if k2 != "te_m"}
        proba = np.zeros(len(val))
        for s in seeds:
            m = XGBClassifier(**FIXED, **p, random_state=s, n_estimators=MAX_TREES,
                              early_stopping_rounds=EARLY_STOP, enable_categorical=native)
            m.fit(Xa, y[es_fit], eval_set=[(Xb, y[es_val])], verbose=False)
            proba += m.predict_proba(Xv)[:, 1] / len(seeds)
            trees.append(m.best_iteration + 1)
        pr.append(average_precision_score(y[val], proba))
        roc.append(roc_auc_score(y[val], proba))
        if trial is not None:
            trial.report(float(np.mean(pr)), k)
            if trial.should_prune():
                raise optuna.TrialPruned()
    return {"pr": pr, "roc": roc, "trees": trees}


def rf_cv(params, fold_list):
    pr, roc = [], []
    for fit, val in fold_list:
        f = onehot_builder(fit)
        m = RandomForestClassifier(n_estimators=300, random_state=RS, n_jobs=8, **params).fit(f(fit), y[fit])
        proba = m.predict_proba(f(val))[:, 1]
        pr.append(average_precision_score(y[val], proba))
        roc.append(roc_auc_score(y[val], proba))
    return {"pr": pr, "roc": roc}


# ---------- search spaces ----------
def base_space(t, tune_spw, native):
    p = {
        "learning_rate": t.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "max_depth": t.suggest_int("max_depth", 2, 12),
        "min_child_weight": t.suggest_float("min_child_weight", 1, 200, log=True),
        "subsample": t.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": t.suggest_float("colsample_bytree", 0.3, 1.0),
        "gamma": t.suggest_float("gamma", 1e-3, 10, log=True),
        "reg_lambda": t.suggest_float("reg_lambda", 1e-2, 100, log=True),
        "reg_alpha": t.suggest_float("reg_alpha", 1e-3, 10, log=True),
    }
    p["scale_pos_weight"] = SPW ** t.suggest_float("spw_power", 0.0, 1.0) if tune_spw else SPW
    if native:
        p["max_cat_to_onehot"] = t.suggest_int("max_cat_to_onehot", 1, 16)
        p["max_cat_threshold"] = t.suggest_int("max_cat_threshold", 8, 64, log=True)
    return p


VARIANTS = {
    # A: 04's search space and features, but lower learning rates allowed, more trees, spw tuned
    "onehot, wider search + spw tuned": {"features": "onehot", "tune_spw": True},
    # B: XGBoost's own categorical splits instead of one-hot
    "native categorical": {"features": "native", "tune_spw": True},
    # C: explicit region x crop pair as a categorical
    "native + region×crop": {"features": "native+pair", "tune_spw": True},
    # D: smoothed out-of-fold loss rate per crop and per region×crop
    "onehot + target-encoded rates": {"features": "onehot+te", "tune_spw": True, "tune_te": True},
    "native + target-encoded rates": {"features": "native+te", "tune_spw": True, "tune_te": True},
}

# 04's winner, as the reference (its own early stopping had MAX_TREES 2000 / 50 rounds;
# trees are re-picked here the same way, so this is the fair comparison point)
XGB_04 = {"learning_rate": 0.055147, "max_depth": 10, "min_child_weight": 29.106359, "subsample": 0.799329,
          "colsample_bytree": 0.409213, "gamma": 0.004207, "reg_lambda": 0.017074, "reg_alpha": 2.915443,
          "scale_pos_weight": SPW}
RF_TUNED_04 = {"min_samples_leaf": 18, "max_features": 0.3, "class_weight": "balanced"}
RF_BASE_03 = {"min_samples_leaf": 10, "max_features": "sqrt", "class_weight": "balanced"}

results = {"search": {}, "best_params": {}, "confirm": {}}
t_all = time.time()

ref = xgb_cv({"features": "onehot"}, XGB_04, SEARCH_FOLDS)
results["search"]["04 tuned XGBoost (reference)"] = {"pr": ref["pr"], "roc": ref["roc"]}
print(f"reference on search folds: PR-AUC {np.mean(ref['pr']):.4f}", flush=True)

for name, v in VARIANTS.items():
    t0 = time.time()
    native = v["features"].startswith("native")

    def objective(t):
        p = base_space(t, v["tune_spw"], native)
        if v.get("tune_te"):
            p["te_m"] = t.suggest_float("te_m", 1, 200, log=True)
        r = xgb_cv(v, p, SEARCH_FOLDS, trial=t)
        t.set_user_attr("params", {k: (float(x) if isinstance(x, (float, np.floating)) else x) for k, x in p.items()})
        t.set_user_attr("pr", r["pr"])
        t.set_user_attr("roc", r["roc"])
        return float(np.mean(r["pr"]))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RS),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=1))
    # start from 04's winner so every search is at least as good as the reference on paper
    seed_trial = {k: XGB_04[k] for k in ["learning_rate", "max_depth", "min_child_weight", "subsample",
                                         "colsample_bytree", "gamma", "reg_lambda", "reg_alpha"]}
    seed_trial["learning_rate"] = min(seed_trial["learning_rate"], 0.2)
    seed_trial["spw_power"] = 1.0
    if native:
        seed_trial.update({"max_cat_to_onehot": 4, "max_cat_threshold": 64})
    if v.get("tune_te"):
        seed_trial["te_m"] = 20
    study.enqueue_trial(seed_trial)
    study.optimize(objective, n_trials=N_TRIALS)
    bt = study.best_trial
    results["search"][name] = {"pr": bt.user_attrs["pr"], "roc": bt.user_attrs["roc"]}
    results["best_params"][name] = bt.user_attrs["params"]
    print(f"{name}: best search PR-AUC {bt.value:.4f} (trial {bt.number}) in {(time.time()-t0)/60:.1f} min",
          flush=True)
    json.dump(results, open(OUT, "w"), indent=1)

# ---------- stage 2: confirm on fresh folds ----------
print("\nconfirming on 10 fresh folds...", flush=True)
results["confirm"]["RF baseline (03)"] = rf_cv(RF_BASE_03, CONFIRM_FOLDS)
results["confirm"]["RF tuned (04)"] = rf_cv(RF_TUNED_04, CONFIRM_FOLDS)
results["confirm"]["04 tuned XGBoost (reference)"] = xgb_cv({"features": "onehot"}, XGB_04, CONFIRM_FOLDS)
for name, v in VARIANTS.items():
    results["confirm"][name] = xgb_cv(v, results["best_params"][name], CONFIRM_FOLDS)
# seed-averaging (5 models) of the best stage-1 variant: does it reduce noise enough to matter?
best_name = max(VARIANTS, key=lambda n: np.mean(results["search"][n]["pr"]))
results["confirm"][f"{best_name}, 5-seed average"] = xgb_cv(VARIANTS[best_name], results["best_params"][best_name],
                                                            CONFIRM_FOLDS, seeds=(1, 2, 3, 4, 5))
json.dump(results, open(OUT, "w"), indent=1)

ref_pr = np.array(results["confirm"]["04 tuned XGBoost (reference)"]["pr"])
print(f"\n{'model':45s} {'search PR':>9s} {'confirm PR':>10s} {'confirm ROC':>11s} {'folds > 04 XGB':>14s}")
for name, r in results["confirm"].items():
    s = results["search"].get(name)
    s_pr = f"{np.mean(s['pr']):.4f}" if s else "-"
    wins = int((np.array(r["pr"]) > ref_pr).sum())
    print(f"{name:45s} {s_pr:>9s} {np.mean(r['pr']):.4f}±{np.std(r['pr']):.3f} {np.mean(r['roc']):.4f}     "
          f"{wins:>2d}/10")
print(f"\ntotal {(time.time()-t_all)/60:.1f} min")
