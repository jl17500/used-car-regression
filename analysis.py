"""Used car price regression and auction buy/pass rule.

Run `python analysis.py` from this folder. It prints every result and writes
tables to outputs/ and charts to outputs/charts/.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor

SEED = 42                 # fixes the random 70/30 split
TEST_SHARE = 0.30         # share of the 240 cars held out for testing
PREP_COST = 1_500         # inspection, reconditioning, detailing, lot time
ALPHA = 0.05              # p-value cutoff for "this feature matters"

HERE = Path(__file__).parent
OUT = HERE / "outputs"
CHARTS = OUT / "charts"

FULL_FEATURES = ["Mileage_K", "Age_Years", "Horsepower", "Accident", "Num_Owners"]

INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
V1_COLOR = "#2a78d6"      # blue, the model
BASE_COLOR = "#eb6834"    # orange, the baseline


def load(path):
    cars = pd.read_csv(path)
    cars["Accident"] = (cars["Accident"].str.lower() == "yes").astype(int)
    return cars


def split(cars):
    order = np.random.default_rng(SEED).permutation(len(cars))
    n_test = round(TEST_SHARE * len(cars))
    return cars.iloc[order[n_test:]].copy(), cars.iloc[order[:n_test]].copy()


def fit(train, features):
    return smf.ols("Sale_Price ~ " + " + ".join(features), data=train).fit()


def r2(actual, predicted):
    return 1 - np.sum((actual - predicted) ** 2) / np.sum((actual - actual.mean()) ** 2)


def rmse(actual, predicted):
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def decide(car, auction_price, model):
    """Return "buy" if the predicted lot price beats auction price plus prep cost."""
    predicted = float(model.predict(pd.DataFrame([car]))[0])
    return "buy" if auction_price + PREP_COST < predicted else "pass"


def decide_baseline(auction_price, average_price):
    return "buy" if auction_price + PREP_COST < average_price else "pass"


def score_rule(lots, decision_col):
    bought = lots[lots[decision_col] == "buy"]
    return {
        "cars_bought": len(bought),
        "money_losers": int((bought["Profit"] < 0).sum()),
        "total_profit": int(bought["Profit"].sum()),
        "profit_per_car": round(bought["Profit"].mean()) if len(bought) else 0,
    }


def money(v, short=False):
    sign = "-" if v < 0 else ""
    return f"{sign}${abs(v) / 1000:g}k" if short else f"{sign}${abs(v):,.0f}"


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(INK_2)
    ax.tick_params(colors=INK_2, labelsize=10)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def chart_pred_vs_actual(test, base_pred, v1_pred, base_stats, v1_stats, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True, sharey=True)
    lo, hi = 0, 50_000
    panels = [
        (axes[0], base_pred, BASE_COLOR, "Baseline: average price for every car", base_stats),
        (axes[1], v1_pred, V1_COLOR, "Model v1", v1_stats),
    ]
    for ax, pred, color, title, stats in panels:
        style(ax)
        ax.plot([lo, hi], [lo, hi], color=INK_2, linewidth=1, linestyle="--", label="Perfect prediction")
        ax.scatter(test["Sale_Price"], pred, s=36, color=color, edgecolor="white", linewidth=1, zorder=3)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_title(title, color=INK, fontsize=12, loc="left")
        ax.text(0.03, 0.95, f"R² = {stats['r2']:.2f}\nTypical error = ${stats['rmse']:,.0f}",
                transform=ax.transAxes, va="top", color=INK, fontsize=11)
        ax.set_xlabel("Actual sale price", color=INK)
        ax.xaxis.set_major_formatter(lambda v, _: f"${v / 1000:.0f}k")
        ax.yaxis.set_major_formatter(lambda v, _: f"${v / 1000:.0f}k")
    axes[0].set_ylabel("Predicted sale price", color=INK)
    axes[1].legend(loc="lower right", frameon=False, fontsize=10, labelcolor=INK_2)
    fig.suptitle(f"Predicted vs actual sale price, {len(test)} test cars the model never saw",
                 color=INK, fontsize=13, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)


def chart_auction_results(results, path):
    metrics = [("cars_bought", "Cars bought"), ("money_losers", "Money-losing cars bought"),
               ("total_profit", "Total profit")]
    fig, axes = plt.subplots(1, 3, figsize=(11, 4))
    for ax, (key, label) in zip(axes, metrics):
        style(ax)
        ax.grid(axis="x", visible=False)
        values = [results["baseline"][key], results["v1"][key]]
        bars = ax.bar(["Baseline", "Model v1"], values, color=[BASE_COLOR, V1_COLOR], width=0.55)
        for bar, v in zip(bars, values):
            text = money(v) if key == "total_profit" else f"{v}"
            ax.text(bar.get_x() + bar.get_width() / 2, v, text, ha="center",
                    va="bottom" if v >= 0 else "top", color=INK, fontsize=12)
        ax.axhline(0, color=INK_2, linewidth=0.8)
        ax.set_title(label, color=INK, fontsize=12, loc="left")
        ax.margins(y=0.15)
        if key == "total_profit":
            ax.yaxis.set_major_formatter(lambda v, _: money(v, short=True))
        else:
            ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    fig.suptitle("Same 40 auction cars, two buying rules", color=INK, fontsize=13, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)


def chart_auction_lots(lots, path):
    lots = lots.sort_values("Profit")
    y = np.arange(len(lots))
    fig, ax = plt.subplots(figsize=(11, 8))
    style(ax)
    ax.grid(axis="y", visible=False)
    ax.barh(y, lots["Profit"], color=GRID, height=0.7)
    ax.scatter(np.where(lots["Baseline_Decision"] == "buy", lots["Profit"], np.nan), y - 0.15,
               marker="o", s=40, color=BASE_COLOR, label="Baseline buys", zorder=3)
    ax.scatter(np.where(lots["V1_Decision"] == "buy", lots["Profit"], np.nan), y + 0.15,
               marker="D", s=40, color=V1_COLOR, label="Model v1 buys", zorder=3)
    ax.axvline(0, color=INK_2, linewidth=0.8)
    ax.set_yticks(y, [f"Lot {n}" for n in lots["Lot"]], fontsize=8)
    ax.xaxis.set_major_formatter(lambda v, _: money(v))
    ax.set_xlabel("Profit if bought (later sale price minus auction price minus $1,500)", color=INK)
    ax.set_title("Which auction cars each rule buys", color=INK, fontsize=13, loc="left")
    ax.legend(loc="lower right", frameon=False, fontsize=10, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)


def main():
    sys.stdout.reconfigure(encoding="utf-8")  # Windows console otherwise garbles "R²"
    CHARTS.mkdir(parents=True, exist_ok=True)
    cars = load(HERE / "data" / "used_car_sales.csv")
    lots = load(HERE / "data" / "auction_lots.csv")
    train, test = split(cars)
    print(f"Data: {len(cars)} cars, {len(train)} train, {len(test)} test (seed {SEED})\n")

    # ---- step 2: fit on training cars ----
    full = fit(train, FULL_FEATURES)
    print("Full model (all five features)")
    print(full.summary(), "\n")

    X = train[FULL_FEATURES].assign(const=1.0)
    vif = pd.Series([variance_inflation_factor(X.values, i) for i in range(len(FULL_FEATURES))],
                    index=FULL_FEATURES)
    print("Variance inflation factors (above 5 means the feature overlaps with others)")
    print(vif.round(2).to_string(), "\n")

    weak = [f for f in FULL_FEATURES if full.pvalues[f] > ALPHA]
    reduced_features = [f for f in FULL_FEATURES if f not in weak]
    reduced = fit(train, reduced_features)
    print(f"Features with p > {ALPHA}: {weak or 'none'}")
    print("Reduced model (without them)")
    print(reduced.summary(), "\n")

    # v1: the version with higher adjusted R²; the simpler one wins a tie
    v1, v1_features = (reduced, reduced_features) if reduced.rsquared_adj >= full.rsquared_adj else (full, FULL_FEATURES)
    print(f"Adjusted R²: full {full.rsquared_adj:.4f}, reduced {reduced.rsquared_adj:.4f}")
    print(f"v1 uses: {v1_features}\n")

    coef_table = pd.DataFrame({
        "coef": v1.params, "p_value": v1.pvalues,
        "ci_low": v1.conf_int()[0], "ci_high": v1.conf_int()[1],
    }).round(4)
    coef_table.to_csv(OUT / "v1_coefficients.csv")
    (OUT / "model_summaries.txt").write_text(
        "FULL MODEL\n" + str(full.summary()) + "\n\nREDUCED MODEL (v1)\n" + str(reduced.summary())
        + "\n\nVIF\n" + vif.round(2).to_string() + "\n", encoding="utf-8")

    # ---- step 3: baseline vs v1 on test cars ----
    average_price = train["Sale_Price"].mean()
    base_pred = np.full(len(test), average_price)
    v1_pred = v1.predict(test).to_numpy()
    actual = test["Sale_Price"].to_numpy()
    fit_stats = {
        "baseline": {"r2": r2(actual, base_pred), "rmse": rmse(actual, base_pred)},
        "v1": {"r2": r2(actual, v1_pred), "rmse": rmse(actual, v1_pred)},
    }
    fit_table = pd.DataFrame(fit_stats).T
    fit_table.loc["v1", "train_adj_r2"] = v1.rsquared_adj
    fit_table.round(4).to_csv(OUT / "test_fit.csv")
    print(f"Training-set average sale price (baseline prediction): ${average_price:,.0f}")
    print("Fit on test cars")
    print(fit_table.round(4).to_string(), "\n")
    pd.DataFrame({"Sale_Price": actual, "Baseline_Pred": base_pred, "V1_Pred": v1_pred.round(0)},
                 index=test.index).to_csv(OUT / "test_predictions.csv")

    # ---- step 4: decision rules on the 40 auction lots ----
    lots["Predicted_Price"] = v1.predict(lots).round(0)
    lots["V1_Decision"] = [decide(row[v1_features].to_dict(), row["Auction_Price"], v1)
                           for _, row in lots.iterrows()]
    lots["Baseline_Decision"] = [decide_baseline(p, average_price) for p in lots["Auction_Price"]]
    lots["Profit"] = lots["Later_Sale_Price"] - lots["Auction_Price"] - PREP_COST
    lots.to_csv(OUT / "auction_decisions.csv", index=False)

    results = {"baseline": score_rule(lots, "Baseline_Decision"), "v1": score_rule(lots, "V1_Decision")}
    results["buy_every_car"] = {
        "cars_bought": len(lots), "money_losers": int((lots["Profit"] < 0).sum()),
        "total_profit": int(lots["Profit"].sum()), "profit_per_car": round(lots["Profit"].mean()),
    }
    results_table = pd.DataFrame(results).T
    results_table.to_csv(OUT / "auction_results.csv")
    print("Auction results (40 lots)")
    print(results_table.to_string(), "\n")
    print("Lot-by-lot decisions")
    print(lots[["Lot", "Auction_Price", "Predicted_Price", "Later_Sale_Price", "Profit",
                "Baseline_Decision", "V1_Decision"]].to_string(index=False))

    chart_pred_vs_actual(test, base_pred, v1_pred, fit_stats["baseline"], fit_stats["v1"],
                         CHARTS / "test_pred_vs_actual.png")
    chart_auction_results(results, CHARTS / "auction_results.png")
    chart_auction_lots(lots, CHARTS / "auction_lots.png")
    print("\nCharts written to outputs/charts")


if __name__ == "__main__":
    main()
