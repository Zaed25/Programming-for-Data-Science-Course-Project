"""Step 2: exploratory analysis. Six questions, figures in results/figures, tables in results/tables.

Populations (denominators) used, written on every figure:
  * ALL     = every order (99,441) -> volumes, statuses, missing values
  * ELIGIBLE = delivered orders with valid dates (the target population) -> every late rate
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from config import PROCESSED, FIGURES, TABLES

sns.set_theme(style="whitegrid", font_scale=0.95)
LATE = "#C0392B"; BLUE = "#2E6DA4"; GREY = "#8C8C8C"
MIN_N = 300   # groups smaller than this are shown in grey and not ranked


def wilson(k, n, z=1.96):
    """95% Wilson interval for a proportion, vectorised with NumPy (works for whole arrays)."""
    k, n = np.asarray(k, float), np.asarray(n, float)
    p = np.divide(k, n, out=np.zeros_like(k), where=n > 0)
    den = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    return p, centre - half, centre + half


def rate_table(df, by):
    t = df.groupby(by, observed=True).is_late.agg(n="size", late="sum").reset_index()
    t["late_rate"], t["ci_low"], t["ci_high"] = wilson(t.late, t.n)
    t["small_group"] = t.n < MIN_N
    return t


def bar_rates(ax, t, label_col, title, overall, ordered=False):
    """ordered=True keeps the natural order of ordinal groups instead of sorting by rate."""
    t = t.iloc[::-1] if ordered else t.sort_values("late_rate")
    colors = np.where(t.small_group, GREY, BLUE)
    err = np.vstack([t.late_rate - t.ci_low, t.ci_high - t.late_rate]) * 100
    ax.barh(t[label_col].astype(str), t.late_rate * 100, xerr=err, color=colors, capsize=2)
    for y, (r, n) in enumerate(zip(t.ci_high, t.n)):
        ax.text(r * 100 + 0.2, y, f"n={n:,}", va="center", fontsize=7.5)
    ax.axvline(overall * 100, color=LATE, ls="--", lw=1, label=f"overall {overall:.1%}")
    ax.set_xlabel("Late rate (%) with 95% CI"); ax.set_title(title); ax.legend(loc="lower right", fontsize=8)


def save(fig, name):
    fig.tight_layout(); fig.savefig(FIGURES / name, dpi=150); plt.close(fig)


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    full = pd.read_csv(PROCESSED / "orders_full.csv", parse_dates=[
        "order_purchase_timestamp", "order_approved_at", "order_delivered_customer_date",
        "order_estimated_delivery_date"], dtype={"customer_zip_code_prefix": str})
    el = full[full.is_late.notna()].copy()          # target population
    overall = el.is_late.mean()
    answers = {}

    # ---------------- F1: target distribution (imbalance) -------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    vc = el.is_late.map({0: "On time / early", 1: "Late"}).value_counts()
    ax[0].bar(vc.index, vc.values, color=[BLUE, LATE])
    for i, v in enumerate(vc.values):
        ax[0].text(i, v, f"{v:,}\n({v / len(el):.1%})", ha="center", va="bottom", fontsize=9)
    ax[0].set_ylabel("Orders"); ax[0].set_title(f"Target is_late (eligible orders, n={len(el):,})")
    ax[0].set_ylim(0, vc.max() * 1.18)
    ax[1].hist(el.delay_days.clip(-40, 40), bins=81, color=BLUE)
    ax[1].axvline(0.5, color=LATE, ls="--"); ax[1].set_yscale("log")
    ax[1].set_xlabel("Delivery day - estimated day (days, clipped to ±40)"); ax[1].set_ylabel("Orders (log)")
    ax[1].set_title("Most orders arrive well before the estimate")
    save(fig, "01_target_distribution.png")

    # ---------------- Q1: volume and late rate over time -----------------------------------
    full["month"] = full.order_purchase_timestamp.dt.to_period("M").dt.to_timestamp()
    el["month"] = el.order_purchase_timestamp.dt.to_period("M").dt.to_timestamp()
    vol = full.groupby("month").size()
    mt = rate_table(el, "month")
    mt = mt[mt.n >= 100]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(vol.index, vol.values, width=20, color="#BFD4EA", label="all orders (left axis)")
    ax.set_ylabel("Orders per month (all statuses)")
    ax2 = ax.twinx()
    ax2.plot(mt.month, mt.late_rate * 100, "-o", color=LATE, label="late rate, eligible (right axis)")
    ax2.fill_between(mt.month, mt.ci_low * 100, mt.ci_high * 100, color=LATE, alpha=0.15)
    ax2.set_ylabel("Late rate (%)"); ax2.grid(False)
    ax.set_title("Q1. Orders and late rate by purchase month (months with < 100 eligible orders hidden)")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8)
    save(fig, "02_q1_time_series.png")
    mt.to_csv(TABLES / "q1_monthly_late_rate.csv", index=False)
    top = mt.nlargest(3, "late_rate")
    answers["Q1"] = "highest months: " + ", ".join(f"{m:%Y-%m} {r:.1%}" for m, r in zip(top.month, top.late_rate))

    # ---------------- Q2: routes between states and multi-seller ------------------------------
    el["route"] = np.where(el.main_seller_same_state == 1, "same state", "different states")
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.4))
    bar_rates(ax[0], rate_table(el, "route"), "route", "Seller and customer state", overall)
    el["sellers_grp"] = np.where(el.n_sellers > 1, "2+ sellers", "1 seller")
    bar_rates(ax[1], rate_table(el, "sellers_grp"), "sellers_grp", "Number of sellers in the order", overall)
    fig.suptitle("Q2. Route and multi-seller orders (eligible orders)", fontsize=11)
    save(fig, "03_q2_route_sellers.png")

    top_s = el.main_seller_state.value_counts().head(6).index
    top_c = el.customer_state.value_counts().head(12).index
    sub = el[el.main_seller_state.isin(top_s) & el.customer_state.isin(top_c)]
    piv = sub.pivot_table(index="main_seller_state", columns="customer_state", values="is_late",
                          aggfunc=["mean", "size"])
    rate, size = piv["mean"] * 100, piv["size"]
    annot = rate.round(1).astype(str).where(size >= 100, "")
    fig, ax = plt.subplots(figsize=(11, 4.2))
    sns.heatmap(rate.where(size >= 30), annot=annot, fmt="", cmap="Reds", ax=ax,
                cbar_kws={"label": "Late rate (%)"}, linewidths=0.5)
    ax.set_xlabel("Customer state (12 largest)"); ax.set_ylabel("Main seller state (6 largest)")
    ax.set_title("Q2. Late rate by route (blank = fewer than 30 orders; number shown only when n >= 100)")
    save(fig, "04_q2_route_heatmap.png")
    rt = rate_table(el, ["main_seller_state", "customer_state"])
    rt[rt.n >= 100].sort_values("late_rate", ascending=False).to_csv(TABLES / "q2_routes.csv", index=False)
    r2 = rate_table(el, "route").set_index("route").late_rate
    r3 = rate_table(el, "sellers_grp").set_index("sellers_grp")
    answers["Q2"] = (f"same state {r2['same state']:.1%} vs different {r2['different states']:.1%}; "
                     f"1 seller {r3.late_rate['1 seller']:.1%} vs 2+ {r3.late_rate['2+ sellers']:.1%} "
                     f"(n={r3.n['2+ sellers']})")

    # ---------------- Q3: category and order size ------------------------------------------------
    ct = rate_table(el, "main_category")
    big = ct[ct.n >= MIN_N]
    show = pd.concat([big.nlargest(8, "late_rate"), big.nsmallest(8, "late_rate")]).drop_duplicates()
    fig, ax = plt.subplots(figsize=(9, 5.4))
    bar_rates(ax, show, "main_category", f"Q3. Categories with highest / lowest late rate (n >= {MIN_N})", overall)
    save(fig, "05_q3_categories.png")
    ct.sort_values("late_rate", ascending=False).to_csv(TABLES / "q3_categories.csv", index=False)

    el["items_grp"] = pd.cut(el.n_items, [0, 1, 2, 3, 100], labels=["1", "2", "3", "4+"])
    el["weight_grp"] = pd.qcut(el.total_weight_g, 5, labels=["Q1 light", "Q2", "Q3", "Q4", "Q5 heavy"])
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.4))
    bar_rates(ax[0], rate_table(el, "items_grp"), "items_grp", "Items in the order", overall, ordered=True)
    bar_rates(ax[1], rate_table(el, "weight_grp"), "weight_grp", "Total weight quintile", overall, ordered=True)
    fig.suptitle("Q3. Order size (eligible orders)", fontsize=11)
    save(fig, "06_q3_order_size.png")
    w = rate_table(el, "weight_grp").set_index("weight_grp").late_rate
    answers["Q3"] = (f"category range {big.late_rate.min():.1%}-{big.late_rate.max():.1%}; "
                     f"weight Q1 {w.iloc[0]:.1%} -> Q5 {w.iloc[-1]:.1%}")

    # ---------------- Q4: planned vs actual delivery time ----------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    smp = el.sample(min(15000, len(el)), random_state=1)
    ax[0].scatter(smp.planned_days, smp.actual_delivery_days, s=3, alpha=0.25,
                  c=np.where(smp.is_late == 1, LATE, BLUE))
    lim = [0, 80]; ax[0].plot(lim, lim, "k--", lw=1, label="actual = planned")
    ax[0].set_xlim(lim); ax[0].set_ylim(0, 100)
    ax[0].set_xlabel("Planned days (estimate - approval)"); ax[0].set_ylabel("Actual days (delivery - purchase)")
    ax[0].set_title("Q4. Planned vs actual (15k sample, red = late)"); ax[0].legend(fontsize=8)
    el["planned_grp"] = pd.cut(el.planned_days, [-1, 10, 15, 20, 25, 30, 40, 200],
                               labels=["<=10", "11-15", "16-20", "21-25", "26-30", "31-40", ">40"])
    sns.boxplot(data=el, x="planned_grp", y="delay_days", hue="planned_grp", legend=False,
                ax=ax[1], showfliers=False, palette="Blues")
    ax[1].axhline(0, color=LATE, ls="--", lw=1)
    ax[1].set_xlabel("Planned days group"); ax[1].set_ylabel("Delivery day - estimated day")
    ax[1].set_title("Q4. Margin before the estimate by planned time")
    save(fig, "07_q4_planned_vs_actual.png")
    pg = rate_table(el, "planned_grp")
    pg.to_csv(TABLES / "q4_planned_groups.csv", index=False)
    ok = el.planned_days.notna() & el.actual_delivery_days.notna()   # boolean mask: 14 orders have no approval date
    corr = np.corrcoef(el.planned_days[ok], el.actual_delivery_days[ok])[0, 1]
    answers["Q4"] = (f"corr(planned, actual)={corr:.2f}; late rate <=10 days {pg.late_rate.iloc[0]:.1%}, "
                     f">40 days {pg.late_rate.iloc[-1]:.1%}; median margin {el.delay_days.median():.0f} days")

    # ---------------- Q5: missing values and exclusions ----------------------------------------------
    excl = pd.read_csv(TABLES / "model_exclusions.csv")
    status = full.order_status.value_counts()
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    ax[0].barh(status.index, status.values, color=np.where(status.index == "delivered", BLUE, GREY))
    ax[0].set_xscale("log"); ax[0].set_xlabel("Orders (log)")
    ax[0].set_title("Order status (all orders) - only 'delivered' has an outcome")
    for y, v in enumerate(status.values):
        ax[0].text(v, y, f" {v:,}", va="center", fontsize=8)
    miss = full[["order_approved_at", "order_delivered_carrier_date", "order_delivered_customer_date",
                 "total_weight_g", "seller_prior_late_rate", "main_category"]].isna().sum()
    miss = miss.rename(lambda c: c.replace("order_", ""))
    ax[1].barh(miss.index, miss.values, color=BLUE)
    for y, v in enumerate(miss.values):
        ax[1].text(v, y, f" {v:,}", va="center", fontsize=8)
    ax[1].set_xlabel("Missing values (all orders)"); ax[1].set_title("Missing values in key columns")
    ax[1].set_xlim(0, miss.max() * 1.25)
    save(fig, "08_q5_missing_status.png")
    # how much would the late rate change if unknown outcomes were (wrongly) treated as on time?
    wrong = full.is_late.fillna(0).mean()
    # censoring at the end of the data: share of orders still not delivered per approval month
    full["appr_month"] = full.order_approved_at.dt.to_period("M").dt.to_timestamp()
    und = full.groupby("appr_month").apply(lambda g: (g.order_status.isin(["shipped", "processing", "invoiced",
                                                                              "approved", "created"])).mean(),
                                            include_groups=False)
    und = und[full.groupby("appr_month").size() >= 100]
    und.rename("share_not_finished").to_csv(TABLES / "q5_unfinished_by_month.csv")
    answers["Q5"] = (f"eligible late rate {overall:.2%} vs {wrong:.2%} if unknown were 'on time'; "
                     f"removed {int(excl.removed.sum())} orders; unfinished share last months: "
                     + ", ".join(f"{m:%Y-%m} {v:.1%}" for m, v in und.tail(4).items()))

    # ---------------- Q6: patterns worth follow-up: relations between variables --------------------
    num = ["planned_days", "approval_delay_h", "total_price", "total_freight", "freight_ratio",
           "total_weight_g", "n_items", "n_sellers", "seller_prior_late_rate", "seller_prior_orders",
           "main_seller_same_state", "is_late"]
    cm = el[num].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="RdBu_r", center=0, vmin=-1, vmax=1, ax=ax,
                annot_kws={"size": 7.5})
    ax.set_title("Q6. Spearman correlation between order features and is_late (eligible orders)")
    save(fig, "09_q6_correlation.png")

    el["seller_hist"] = pd.cut(el.seller_prior_late_rate, [-0.01, 0, 0.05, 0.1, 0.2, 1.0],
                               labels=["0%", "0-5%", "5-10%", "10-20%", ">20%"])
    el.loc[el.seller_prior_late_rate.isna(), "seller_hist"] = np.nan
    el["seller_hist"] = el.seller_hist.cat.add_categories("no history").fillna("no history")
    el["weekday"] = el.order_approved_at.dt.day_name().str[:3]
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    bar_rates(ax[0], rate_table(el, "seller_hist"), "seller_hist", "Seller's past late rate", overall, ordered=True)
    bar_rates(ax[1], rate_table(el, "weekday"), "weekday", "Approval weekday", overall)
    fig.suptitle("Q6. Patterns to follow up (eligible orders)", fontsize=11)
    save(fig, "10_q6_seller_history_weekday.png")
    sh = rate_table(el, "seller_hist").set_index("seller_hist").late_rate
    answers["Q6"] = (f"seller past late 0% -> {sh['0%']:.1%}, >20% -> {sh['>20%']:.1%}; "
                     f"spearman(is_late, planned_days)={cm.loc['is_late', 'planned_days']:.2f}, "
                     f"(is_late, seller_prior_late_rate)={cm.loc['is_late', 'seller_prior_late_rate']:.2f}")

    # pivot table used in the report: late rate by split and route
    el[el.split.isin(["train", "val", "test"])].pivot_table(index="split", columns="route", values="is_late", aggfunc="mean").round(4) \
      .to_csv(TABLES / "pivot_split_route.csv")
    pd.Series(answers).to_csv(TABLES / "eda_answers.csv", header=["answer"])
    for k, v in answers.items():
        print(k, v)


if __name__ == "__main__":
    main()
