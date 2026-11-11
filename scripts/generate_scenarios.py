from itertools import product
from pathlib import Path
import platform
import json
import argparse

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# 1. Settings
SAMPLE_SEED = 42
PRESENTATION_SEED = 2026
PER_GROUP = 6

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description="Generate the Task 1 macroeconomic scenarios.")
parser.add_argument("--calibration", type=Path, default=ROOT / "results/task1/okun_calibration.json")
parser.add_argument("--out", type=Path, default=ROOT / "results/task1")
args = parser.parse_args()
calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
OKUN_INTERCEPT = calibration["intercept"]
OKUN_SLOPE = calibration["slope"]
RESIDUAL_LOWER, RESIDUAL_UPPER = calibration["residual_interval"]

LEVELS = {
    "inflation_gap": [-1, 0, 1, 2],
    "unemployment_gap": [-1.5, -0.5, 0.5, 1.5],
    "expectations_gap": [-0.5, 0, 0.5, 1],
    "output_gap": [-3, -1.5, 0, 1.5, 3],
}

COLUMNS = list(LEVELS)
GROUPS = ["inflation_gap", "unemployment_gap"]
LABELS = ["Inflation", "Unemployment", "Expectations", "Output"]
folder = args.out
folder.mkdir(parents=True, exist_ok=True)


# 2. Generate every combination of the four variable levels
candidates = pd.DataFrame(
    product(*LEVELS.values()),
    columns=COLUMNS,
)


# 3. Apply the supplied first-stage screening rule
# Predicted output gap = 0.156371 - 0.946960 * unemployment gap.
predicted_output = (
    OKUN_INTERCEPT + OKUN_SLOPE * candidates["unemployment_gap"]
)
residual = candidates["output_gap"] - predicted_output

# Keep both endpoints of the residual interval.
eligible = candidates.loc[
    residual.between(RESIDUAL_LOWER, RESIDUAL_UPPER)
].copy()


# 4. Randomly select six scenarios from each of the 16 groups
group_sizes = eligible.groupby(GROUPS).size()

if len(group_sizes) != 16 or (group_sizes < PER_GROUP).any():
    raise ValueError("Each of the 16 groups must have at least six candidates.")

base = (
    eligible.groupby(GROUPS, sort=True)
    .sample(n=PER_GROUP, replace=False, random_state=SAMPLE_SEED)
    .sort_values(COLUMNS)
    .reset_index(drop=True)
)

base.insert(
    0,
    "scenario_id",
    [f"S{i:03d}" for i in range(1, len(base) + 1)],
)


# 5. Add one neutral scenario outside the balanced base sample
# Neutral inputs do not impose a particular policy decision.
neutral = pd.DataFrame([
    {"scenario_id": "S000", **dict.fromkeys(COLUMNS, 0.0)}
])

final = pd.concat([base, neutral], ignore_index=True)

print(f"Candidates: {len(candidates)}")
print(f"Passed screening: {len(eligible)}")
print(f"Excluded: {len(candidates) - len(eligible)}")
print(f"Base sample: {len(base)}")
print(f"Added neutral: {len(neutral)}")
print(f"Final sample: {len(final)}")


# 6. Print frequencies for all four variables
# Include original grid levels even when their frequency is zero.
frequencies = {}

for column in COLUMNS:
    values = sorted(set(LEVELS[column]) | set(final[column]))

    frequency = pd.DataFrame({
        "base_n": base[column].value_counts(),
        "final_n": final[column].value_counts(),
    })
    frequency = frequency.reindex(values).fillna(0).astype(int)
    frequency.index.name = column
    frequencies[column] = frequency

    print(f"\nFREQUENCIES: {column}")
    print(frequency.to_string())


# 7. Print correlation matrices
corr_base = base[COLUMNS].corr()
corr_final = final[COLUMNS].corr()

print("\nCORRELATIONS: BASE SAMPLE")
print(corr_base.round(3).to_string())

print("\nCORRELATIONS: INCLUDING NEUTRAL")
print(corr_final.round(3).to_string())


# 8. Check expectations and output coverage within each base group
# Counts include eligible levels that were not selected.
# Coverage means selected distinct levels / eligible distinct levels.
coverage_rows = []

for (inflation, unemployment), group in base.groupby(GROUPS):
    allowed = eligible[
        (eligible["inflation_gap"] == inflation)
        & (eligible["unemployment_gap"] == unemployment)
    ]

    row = {
        "inflation_gap": inflation,
        "unemployment_gap": unemployment,
        "n": len(group),
    }

    for column, prefix in [
        ("expectations_gap", "e"),
        ("output_gap", "y"),
    ]:
        values = sorted(allowed[column].unique())
        counts = group[column].value_counts().reindex(values, fill_value=0)

        row[f"{prefix}_counts"] = "; ".join(
            f"{value:g}:{count}" for value, count in counts.items()
        )
        row[f"{prefix}_coverage"] = (
            f"{group[column].nunique()}/{len(values)}"
        )

    coverage_rows.append(row)

coverage = pd.DataFrame(coverage_rows)

print("\nWITHIN-GROUP COVERAGE: value:count")
print(coverage.to_string(index=False))
print("\nGlobal marginal balance is not enforced.")
print("Neutral S000 is excluded from the base-group diagnostics.")


# 9. Save one CSV with scenario IDs
final.to_csv(
    folder / "scenarios.csv",
    index=False,
    encoding="utf-8-sig",
)
coverage.to_csv(folder / "group_coverage.csv", index=False)
corr_final.to_csv(folder / "correlations.csv")
final[COLUMNS].describe().to_csv(folder / "descriptive_statistics.csv")


# 10. Save one figure with coverage, frequencies and correlations
fig, axes = plt.subplots(
    2, 3, figsize=(15, 9), layout="constrained"
)

# Inflation-unemployment coverage; labels show scenario counts.
pairs = final.groupby(GROUPS).size().reset_index(name="n")
ax = axes[0, 0]

ax.scatter(
    pairs["inflation_gap"],
    pairs["unemployment_gap"],
    s=pairs["n"] * 40,
    color="#286887",
    zorder=3,
)

for row in pairs.itertuples():
    ax.text(
        row.inflation_gap, row.unemployment_gap, str(row.n),
        ha="center", va="center", color="white", fontsize=8,
    )

ax.axhline(0, color="gray", linewidth=0.8)
ax.axvline(0, color="gray", linewidth=0.8)
ax.set(
    title="Inflation and unemployment coverage",
    xlabel="Inflation gap (pp)",
    ylabel="Unemployment gap (pp)",
    xlim=(-1.5, 2.5),
    ylim=(-2, 2),
)

# Four marginal frequency charts, including the neutral scenario.
frequency_axes = [axes[0, 1], axes[0, 2], axes[1, 0], axes[1, 1]]

for column, label, ax in zip(COLUMNS, LABELS, frequency_axes):
    frequency = frequencies[column]
    bars = ax.bar(
        [f"{value:g}" for value in frequency.index],
        frequency["final_n"],
        color="#286887",
    )
    ax.bar_label(bars, padding=3, fontsize=9)
    ax.set(
        title=f"{label} gap frequencies",
        xlabel="Gap (%)" if column == "output_gap" else "Gap (pp)",
        ylabel="Scenarios",
        ylim=(0, frequency["final_n"].max() * 1.2),
    )

# Correlations for the final sample.
ax = axes[1, 2]
ax.imshow(corr_final, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set(
    title="Pearson correlations",
    xticks=range(4),
    yticks=range(4),
    xticklabels=LABELS,
    yticklabels=LABELS,
)
plt.setp(ax.get_xticklabels(), rotation=25, ha="right", fontsize=8)

for i in range(4):
    for j in range(4):
        value = corr_final.iloc[i, j]
        ax.text(
            j, i, f"{value:.2f}",
            ha="center", va="center",
            color="white" if abs(value) > 0.65 else "black",
        )

fig.suptitle(
    f" {len(final)} scenarios",
    fontsize=15,
)
fig.savefig(folder / "coverage.png", dpi=200)
plt.close(fig)


# 11. Shuffle presentation order without changing scenario IDs
llm_order = final.sample(
    frac=1, random_state=PRESENTATION_SEED
).reset_index(drop=True)
llm_order.to_csv(folder / "presentation_order.csv", index=False, encoding="utf-8-sig")

print("\nFirst five IDs in LLM presentation order:")
print(", ".join(llm_order["scenario_id"].head()))


# 12. Print reproducibility information
print(f"\nPython: {platform.python_version()}")
print(f"pandas: {pd.__version__}")
print(f"Matplotlib: {matplotlib.__version__}")
print(f"Sampling seed: {SAMPLE_SEED}")
print(f"Presentation seed: {PRESENTATION_SEED}")
print(f"Saved scenarios.csv and coverage.png in: {folder}")
