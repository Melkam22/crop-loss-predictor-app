"""Fairness check for xgb_variants.py: give Random Forest the same region×crop
information XGBoost got, tune it the same way, and confirm on the same 10
fresh folds.

Same two stages as xgb_variants.py:
Stage 1 (search): Optuna on 04's exact 5 grouped folds, PR-AUC objective.
Stage 2 (confirm): best config re-scored on the same 10 fresh grouped folds
(random_state=7 and 123), so results pair fold-by-fold with results.json.
The test set is not touched.
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

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = sys.argv[1] if len(sys.argv) > 1 else HERE / "results_rf.json"
N_TRIALS = int(sys.argv[2]) if len(sys.argv) > 2 else 50
# the one-hot pair variant adds ~700 columns and takes ~3 min per trial (vs ~0.6 min),
# so it gets a smaller budget; the target-encoded variant gets the same 50 as XGBoost
N_TRIALS_ONEHOT_PAIR = min(N_TRIALS, 20)
RS = 42
RF_TREES = 300
optuna.logging.set_verbosity(optuna.logging.WARNING)

df = pd.read_csv(ROOT / "data/processed/crop_loss_model_ready.csv", dtype={"household_id": str})
tr_idx, _ = next(GroupShuffleSplit(1, test_size=0.2, random_state=RS).split(df, groups=df["household_id"]))
train = df.iloc[tr_idx].reset_index(drop=True)
assert len(train) == 56395

CAT = ["crop_name", "region_code"]
NUM = ["household_size", "is_rural", "rainfall_belg_mm", "rainfall_belg_pct_of_avg",
       "rainfall_meher_mm", "rainfall_meher_pct_of_avg"]
y = train["loss_occurred"].to_numpy()
groups = train["household_id"].to_numpy()
train["region_crop"] = train["region_code"].astype(str) + "|" + train["crop_name"]


def folds(seed):
    return list(StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(train, y, groups=groups))


SEARCH_FOLDS = folds(42)
CONFIRM_FOLDS = folds(7) + folds(123)  # identical to xgb_variants.py's confirm folds


def onehot_builder(fit_rows, extra_cat=()):
    cats = CAT + list(extra_cat)
    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(train.loc[fit_rows, cats])
    names = enc.get_feature_names_out(cats)

    def f(rows):
        d = train.loc[rows]
        return pd.concat([d[NUM].reset_index(drop=True),
                          pd.DataFrame(enc.transform(d[cats]), columns=names)], axis=1).astype("float32")
    return f


def target_encode(fit_rows, cols, m):
    """Same as xgb_variants.py: smoothed loss rate per level, out-of-fold
    (inner grouped 5-fold) for fitting rows, full-fit table for other rows."""
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


def build(features, fit_rows, te_m=20):
    if features == "onehot":
        return onehot_builder(fit_rows)
    if features == "onehot+pair":
        return onehot_builder(fit_rows, ["region_crop"])
    if features == "onehot+te":
        base, te = onehot_builder(fit_rows), target_encode(fit_rows, ["crop_name", "region_crop"], te_m)
        return lambda rows: pd.concat([base(rows), te(rows)], axis=1)
    raise ValueError(features)


def rf_cv(features, params, fold_list, trial=None):
    pr, roc = [], []
    p = {k: v for k, v in params.items() if k != "te_m"}
    for k, (fit, val) in enumerate(fold_list):
        f = build(features, fit, params.get("te_m", 20))
        m = RandomForestClassifier(n_estimators=RF_TREES, random_state=RS, n_jobs=-1, **p).fit(f(fit), y[fit])
        proba = m.predict_proba(f(val))[:, 1]
        pr.append(average_precision_score(y[val], proba))
        roc.append(roc_auc_score(y[val], proba))
        if trial is not None:
            trial.report(float(np.mean(pr)), k)
            if trial.should_prune():
                raise optuna.TrialPruned()
    return {"pr": pr, "roc": roc}


# 04's RF search space, unchanged
MAX_FEATURES_CHOICES = {"sqrt": "sqrt", "log2": "log2", "0.1": 0.1, "0.2": 0.2, "0.3": 0.3, "0.5": 0.5}


def rf_space(t, tune_te):
    limit_depth = t.suggest_categorical("limit_depth", [False, True])
    p = {
        "max_depth": t.suggest_int("max_depth", 4, 30) if limit_depth else None,
        "min_samples_leaf": t.suggest_int("min_samples_leaf", 1, 200, log=True),
        "max_features": MAX_FEATURES_CHOICES[t.suggest_categorical("max_features", list(MAX_FEATURES_CHOICES))],
        "class_weight": t.suggest_categorical("class_weight", ["balanced", "balanced_subsample"]),
    }
    if tune_te:
        p["te_m"] = t.suggest_float("te_m", 1, 200, log=True)
    return p


VARIANTS = {
    "RF + region×crop (one-hot)": "onehot+pair",
    "RF + target-encoded rates": "onehot+te",
}
RF_TUNED_04 = {"min_samples_leaf": 18, "max_features": 0.3, "class_weight": "balanced"}

results = {"search": {}, "best_params": {}, "confirm": {}}
t_all = time.time()
for name, features in VARIANTS.items():
    t0 = time.time()
    tune_te = features == "onehot+te"

    def objective(t):
        p = rf_space(t, tune_te)
        r = rf_cv(features, p, SEARCH_FOLDS, trial=t)
        t.set_user_attr("params", p)
        t.set_user_attr("pr", r["pr"])
        t.set_user_attr("roc", r["roc"])
        return float(np.mean(r["pr"]))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RS),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=1))
    # start from 04's tuned RF, as the XGBoost searches started from 04's tuned XGBoost
    seed_trial = {"limit_depth": False, "min_samples_leaf": 18, "max_features": "0.3", "class_weight": "balanced"}
    if tune_te:
        seed_trial["te_m"] = 20
    study.enqueue_trial(seed_trial)
    study.optimize(objective, n_trials=N_TRIALS_ONEHOT_PAIR if features == "onehot+pair" else N_TRIALS)
    bt = study.best_trial
    results["search"][name] = {"pr": bt.user_attrs["pr"], "roc": bt.user_attrs["roc"]}
    results["best_params"][name] = bt.user_attrs["params"]
    print(f"{name}: best search PR-AUC {bt.value:.4f} (trial {bt.number}) in {(time.time()-t0)/60:.1f} min",
          flush=True)
    json.dump(results, open(OUT, "w"), indent=1)

print("\nconfirming on the same 10 fresh folds...", flush=True)
for name, features in VARIANTS.items():
    results["confirm"][name] = rf_cv(features, results["best_params"][name], CONFIRM_FOLDS)
json.dump(results, open(OUT, "w"), indent=1)

# fold-by-fold comparison against xgb_variants.py's results (same confirm folds)
xgb = json.load(open(HERE / "results.json"))["confirm"]
rf_tuned = np.array(xgb["RF tuned (04)"]["pr"])
xgb_best = np.array(xgb["native + region×crop"]["pr"])
assert np.allclose(rf_cv("onehot", RF_TUNED_04, CONFIRM_FOLDS[:1])["pr"][0], rf_tuned[0]), "folds differ"
print(f"\n{'model':38s} {'confirm PR':>12s} {'ROC':>7s} {'> RF tuned':>11s} {'XGB best >':>11s}")
for name, r in results["confirm"].items():
    p = np.array(r["pr"])
    print(f"{name:38s} {p.mean():.4f}±{p.std():.3f} {np.mean(r['roc']):.4f} {int((p > rf_tuned).sum()):>6d}/10 "
          f"{int((xgb_best > p).sum()):>8d}/10   (XGB best minus this: {np.mean(xgb_best - p):+.4f})")
print(f"\ntotal {(time.time()-t_all)/60:.1f} min")
