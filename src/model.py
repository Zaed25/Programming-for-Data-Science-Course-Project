"""Step 3: predict is_late at approval time.

* temporal split (train / validation / test) on order_approved_at, see config.py
* baseline + 3 models; hyper-parameters and the decision threshold are chosen on VALIDATION only
* the test period is used once, at the end, for the final numbers
"""
import json
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline

from config import PROCESSED, FIGURES, TABLES, RESULTS, SEED, TARGET, TRAIN_END, VAL_END, TEST_END
from features import FEATURES, preprocessor

# candidate settings (small grids, every one is written to the experiment log)
CANDIDATES = {
    "Baseline (always on time)": [
        ({}, lambda p: DummyClassifier(strategy="most_frequent"))],
    "Logistic Regression": [
        ({"C": c}, lambda p: LogisticRegression(C=p["C"], class_weight="balanced", max_iter=2000))
        for c in (0.01, 0.1, 1.0)],
    "Random Forest": [
        ({"max_depth": d, "min_samples_leaf": m},
         lambda p: RandomForestClassifier(n_estimators=300, max_depth=p["max_depth"],
                                          min_samples_leaf=p["min_samples_leaf"],
                                          class_weight="balanced_subsample", n_jobs=-1, random_state=SEED))
        for d in (12, None) for m in (5, 20)],
    "Gradient Boosting": [
        ({"learning_rate": lr, "max_leaf_nodes": l},
         lambda p: HistGradientBoostingClassifier(learning_rate=p["learning_rate"], max_leaf_nodes=p["max_leaf_nodes"],
                                                  max_iter=300, min_samples_leaf=50, l2_regularization=1.0,
                                                  class_weight="balanced", random_state=SEED))
        for lr in (0.03, 0.1) for l in (15, 31)],
}
SCALE = {"Logistic Regression": True}   # trees do not need scaling


def make_pipeline(name, est):
    return Pipeline([("prep", preprocessor(scale=SCALE.get(name, False))), ("model", est)])


def best_threshold(y, p):
    """Threshold that maximises F1 on the validation set (vectorised over all PR-curve points)."""
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(prec), where=(prec + rec) > 0)[:-1]
    i = int(np.argmax(f1))
    return float(thr[i]), float(f1[i])


def metrics(y, p, thr):
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"accuracy": accuracy_score(y, pred), "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred), "f1": f1_score(y, pred), "avg_precision": average_precision_score(y, p),
            "roc_auc": roc_auc_score(y, p), "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)}


def main():
    df = pd.read_csv(PROCESSED / "orders_model.csv", parse_dates=["order_approved_at"])
    assert df.order_id.is_unique
    parts = {s: df[df.split == s] for s in ("train", "val", "test")}
    X = {s: d[FEATURES] for s, d in parts.items()}
    y = {s: d[TARGET].astype(int).to_numpy() for s, d in parts.items()}
    split_info = pd.DataFrame([{"split": s, "from": d.order_approved_at.min(), "to": d.order_approved_at.max(),
                                "orders": len(d), "late": int(y[s].sum()), "late_rate": y[s].mean()}
                               for s, d in parts.items()])
    split_info.to_csv(TABLES / "split_info.csv", index=False)
    print(split_info.to_string(index=False))

    log, chosen = [], {}
    for name, cands in CANDIDATES.items():
        best = None
        for params, factory in cands:
            t0 = time.time()
            pipe = make_pipeline(name, factory(params)).fit(X["train"], y["train"])
            p_val = pipe.predict_proba(X["val"])[:, 1]
            ap = average_precision_score(y["val"], p_val)
            thr, f1v = best_threshold(y["val"], p_val) if name.startswith("Baseline") is False else (0.5, 0.0)
            log.append({"model": name, "params": json.dumps(params), "val_avg_precision": round(ap, 4),
                        "val_best_f1": round(f1v, 4), "threshold": round(thr, 4),
                        "fit_seconds": round(time.time() - t0, 1)})
            print(log[-1])
            if best is None or ap > best[0]:
                best = (ap, params, pipe, thr)
        chosen[name] = best
    pd.DataFrame(log).to_csv(RESULTS / "experiments_log.csv", index=False)

    # ---------------- final evaluation (test used once, with settings fixed on validation) ----
    rows, probs = [], {}
    for name, (ap, params, pipe, thr) in chosen.items():
        for s in ("val", "test"):
            p = pipe.predict_proba(X[s])[:, 1]
            probs[(name, s)] = p
            rows.append({"model": name, "split": s, "params": json.dumps(params), "threshold": thr,
                         **metrics(y[s], p, thr)})
    res = pd.DataFrame(rows)
    res.to_csv(TABLES / "model_results.csv", index=False)
    print(res[["model", "split", "accuracy", "precision", "recall", "f1", "avg_precision", "roc_auc"]]
          .round(3).to_string(index=False))

    # extra experiment: retrain the chosen settings on train+val (same threshold), score on test
    extra = []
    for name, (ap, params, pipe, thr) in chosen.items():
        if name.startswith("Baseline"):
            continue
        m2 = clone(pipe).fit(pd.concat([X["train"], X["val"]]), np.concatenate([y["train"], y["val"]]))
        p = m2.predict_proba(X["test"])[:, 1]
        extra.append({"model": name, "trained_on": "train+val", **metrics(y["test"], p, thr)})
    pd.DataFrame(extra).to_csv(TABLES / "extra_train_plus_val.csv", index=False)

    models = [m for m in chosen if not m.startswith("Baseline")]
    best_name = max(models, key=lambda m: chosen[m][0])        # chosen by VALIDATION average precision
    json.dump({"best_model": best_name, "params": chosen[best_name][1], "threshold": chosen[best_name][3]},
              open(RESULTS / "best_model.json", "w"), indent=2)

    # ---------------- figures --------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, s in zip(axes, ("val", "test")):
        for name in models:
            pr, rc, _ = precision_recall_curve(y[s], probs[(name, s)])
            ax.plot(rc, pr, label=f"{name} (AP={average_precision_score(y[s], probs[(name, s)]):.3f})")
        ax.axhline(y[s].mean(), color="grey", ls="--", label=f"baseline = late rate {y[s].mean():.3f}")
        ax.set_xlabel("Recall (late class)"); ax.set_ylabel("Precision (late class)")
        ax.set_title(f"Precision-Recall curve - {s} split"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIGURES / "11_pr_curves.png", dpi=150); plt.close(fig)

    fig, axes = plt.subplots(1, len(chosen), figsize=(15, 3.6))
    for ax, name in zip(axes, chosen):
        r = res[(res.model == name) & (res.split == "test")].iloc[0]
        cm = np.array([[r.tn, r.fp], [r.fn, r.tp]])
        ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax.set_xticks([0, 1], ["pred on time", "pred late"]); ax.set_yticks([0, 1], ["on time", "late"])
        ax.set_title(f"{name}\nF1={r.f1:.3f}", fontsize=10); ax.grid(False)
    fig.suptitle("Confusion matrices on the test period (threshold chosen on validation)")
    fig.tight_layout(); fig.savefig(FIGURES / "12_confusion_matrices.png", dpi=150); plt.close(fig)

    # permutation importance of the best model on (a sample of) validation data
    pipe = chosen[best_name][2]
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(X["val"]), size=min(8000, len(X["val"])), replace=False)
    pi = permutation_importance(pipe, X["val"].iloc[idx], y["val"][idx], scoring="average_precision",
                                n_repeats=5, random_state=SEED, n_jobs=-1)
    imp = pd.DataFrame({"feature": FEATURES, "importance": pi.importances_mean, "std": pi.importances_std}) \
            .sort_values("importance", ascending=False)
    imp.to_csv(TABLES / "permutation_importance.csv", index=False)
    top = imp.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(top.feature, top.importance, xerr=top["std"], color="#2E6DA4")
    ax.set_xlabel("Drop in average precision when the feature is shuffled")
    ax.set_title(f"Permutation importance - {best_name} (validation sample)")
    fig.tight_layout(); fig.savefig(FIGURES / "13_permutation_importance.png", dpi=150); plt.close(fig)

    # ---------------- error analysis on the test period ----------------------------------------
    t = parts["test"].copy()
    thr = chosen[best_name][3]
    t["p"] = probs[(best_name, "test")]
    t["pred"] = (t.p >= thr).astype(int)
    t["error"] = np.select([(t.pred == 1) & (t.is_late == 0), (t.pred == 0) & (t.is_late == 1)], ["FP", "FN"], "ok")
    t["month"] = t.order_approved_at.dt.to_period("M").astype(str)
    t["route"] = np.where(t.main_seller_same_state == 1, "same state", "different states")
    t["planned_grp"] = pd.cut(t.planned_days, [-1, 15, 25, 35, 200], labels=["<=15", "16-25", "26-35", ">35"])
    ea = []
    for col in ("month", "route", "planned_grp"):
        g = t.groupby(col, observed=True)
        ea.append(pd.DataFrame({"group_by": col, "value": g.size().index.astype(str), "orders": g.size().values,
                                "late_rate": g.is_late.mean().values, "predicted_late_rate": g.pred.mean().values,
                                "recall": g.apply(lambda d: d.pred[d.is_late == 1].mean(), include_groups=False).values,
                                "precision": g.apply(lambda d: d.is_late[d.pred == 1].mean() if d.pred.sum() else np.nan,
                                                     include_groups=False).values}))
    ea = pd.concat(ea)
    ea.to_csv(TABLES / "error_analysis.csv", index=False)
    # probability deciles: does a higher score really mean more late orders?
    t["decile"] = pd.qcut(t.p.rank(method="first"), 10, labels=range(1, 11))
    dec = t.groupby("decile", observed=True).agg(orders=("p", "size"), mean_score=("p", "mean"),
                                                 late_rate=("is_late", "mean"))
    dec.to_csv(TABLES / "score_deciles.csv")
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    ax[0].bar(dec.index.astype(int), dec.late_rate * 100, color="#C0392B")
    ax[0].axhline(t.is_late.mean() * 100, color="grey", ls="--", label="test late rate")
    ax[0].set_xlabel("Score decile (1 = lowest predicted risk)"); ax[0].set_ylabel("Actual late rate (%)")
    ax[0].set_title(f"{best_name}: actual late rate by score decile (test)"); ax[0].legend(fontsize=8)
    m = ea[ea.group_by == "month"]
    ax[1].plot(m.value, m.late_rate * 100, "-o", label="actual late rate")
    ax[1].plot(m.value, m.predicted_late_rate * 100, "-s", label="predicted late (share flagged)")
    ax[1].set_ylabel("%"); ax[1].set_xlabel("Approval month (test period)")
    ax[1].set_title("Flagged vs actual late orders by month"); ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIGURES / "14_error_analysis.png", dpi=150); plt.close(fig)
    print("best model (validation AP):", best_name)
    print(ea.round(3).to_string(index=False))
    print(dec.round(3).to_string())


if __name__ == "__main__":
    main()
