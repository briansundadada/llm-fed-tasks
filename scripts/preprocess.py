from pathlib import Path

import pandas as pd


# ============================================================
# 1. Paths and assumptions
# ============================================================


PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_DIR / "data" / "raw"
OUTPUT_DIR = PROJECT_DIR / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# A common comparison benchmark for the entire sample.
INFLATION_TARGET = 2.0

QUARTERS = pd.period_range("2000Q1", "2025Q4", freq="Q")


# ============================================================
# 2. Helper functions
# ============================================================

def read_fred(filename, column):
    """Read a raw FRED CSV and return a date-indexed series."""
    frame = pd.read_csv(
        RAW_DIR / filename,
        parse_dates=["observation_date"],
        na_values=[".", "#N/A"],
    )

    series = frame.set_index("observation_date")[column]
    series = pd.to_numeric(series, errors="raise").sort_index()

    if series.index.has_duplicates:
        raise ValueError(f"Duplicate dates found in {filename}")

    return series


def monthly_to_quarterly(series):
    """
    Calculate the arithmetic mean of the three monthly values.

    Require all three months to be observed.
    An incomplete quarter remains missing; no interpolation is used.
    """
    groups = series.groupby(series.index.to_period("Q"))

    quarterly_mean = groups.mean()
    monthly_count = groups.count()

    return quarterly_mean.where(monthly_count == 3)


def label_quarterly(series):
    """Convert existing quarterly dates to labels such as 2000Q1."""
    result = series.copy()
    result.index = result.index.to_period("Q")

    if result.index.has_duplicates:
        raise ValueError("Multiple observations found within a quarter")

    return result


# ============================================================
# 3. Load unemployment and GDP series
# ============================================================

unemployment = read_fred("UNRATE_2000_2025.csv", "UNRATE")
natural_unemployment = read_fred("NROU_2000_2025.csv", "NROU")
real_gdp = read_fred("GDPC1_2000_2025.csv", "GDPC1")
potential_gdp = read_fred("GDPPOT_2000_2025.csv", "GDPPOT")

# Each row represents one US calendar quarter.
# Series are aligned automatically by their quarter labels.
data = pd.DataFrame(index=QUARTERS)
data.index.name = "quarter"

data["unemployment_rate"] = monthly_to_quarterly(unemployment)
data["natural_unemployment_rate"] = label_quarterly(
    natural_unemployment
)
data["real_gdp"] = label_quarterly(real_gdp)
data["potential_gdp"] = label_quarterly(potential_gdp)


# ============================================================
# 4. Calculate actual PCE inflation
# ============================================================

pce_monthly = read_fred("PCEPI_1999_2025.csv", "PCEPI")
pce_quarterly = monthly_to_quarterly(pce_monthly)

# Keep a continuous quarterly index so a four-period lag
# always means four calendar quarters.
pce_quarterly = pce_quarterly.reindex(
    pd.period_range("1999Q1", "2025Q4", freq="Q")
)

# Year-over-year inflation based on quarterly-average prices:
# inflation_t = 100 * (PCE_t / PCE_(t-4) - 1)
# Calculate BEFORE dropping 1999, which supplies the lag for 2000.
data["pce_price_index"] = pce_quarterly
data["inflation_rate"] = (
    pce_quarterly.pct_change(periods=4, fill_method=None) * 100
)


# ============================================================
# 5. Construct expected inflation over the next four quarters
# ============================================================

spf = pd.read_csv(
    RAW_DIR / "SPF_PCE_median_2000_2025.csv",
    na_values=["#N/A"],
)

# YEAR and QUARTER may appear as decimals in the raw file.
quarter_labels = (
    spf["YEAR"].astype(int).astype(str)
    + "Q"
    + spf["QUARTER"].astype(int).astype(str)
)
spf.index = pd.PeriodIndex(quarter_labels, freq="Q")

if spf.index.has_duplicates:
    raise ValueError("Duplicate survey quarters found in SPF data")

# PCE3, PCE4, PCE5 and PCE6 forecast quarters t+1 through t+4.
# Each is an annualized quarter-on-quarter percentage change.
forecasts = spf[["PCE3", "PCE4", "PCE5", "PCE6"]].apply(
    pd.to_numeric, errors="raise"
)

# Convert each annualized forecast into a quarterly growth factor.
quarterly_factors = (1 + forecasts / 100) ** 0.25

# Compound the four quarterly factors into a one-year rate.
# skipna=False requires all four forecasts to be available.
# This is a proxy constructed from horizon-specific medians,
# not the median of individual respondents' one-year forecasts.
data["expected_inflation_1y"] = (
    quarterly_factors.prod(axis=1, skipna=False) - 1
) * 100


# ============================================================
# 6. Construct the four gap variables
# ============================================================

data["inflation_target"] = INFLATION_TARGET

# Percentage points: positive means inflation exceeds the benchmark.
data["inflation_gap"] = (
    data["inflation_rate"] - data["inflation_target"]
)

# Percentage points: positive means unemployment exceeds NROU.
# NROU varies over time; it is not a fixed 4.5% assumption.
data["unemployment_gap"] = (
    data["unemployment_rate"] - data["natural_unemployment_rate"]
)

# Percent of potential GDP: positive means output exceeds potential.
# Both GDP series use the same chained-dollar units.
# This is the percentage-level gap, not GDP growth or a log gap.
data["output_gap"] = (
    data["real_gdp"] / data["potential_gdp"] - 1
) * 100

# Percentage points: positive means expected inflation exceeds 2%.
data["expected_inflation_gap"] = (
    data["expected_inflation_1y"] - data["inflation_target"]
)


# ============================================================
# 7. Save one merged dataset and print basic checks
# ============================================================

# Preserve all quarters, including rows with missing variables.
# In particular, missing early SPF forecasts must not remove
# otherwise usable observations for an Okun regression.
# No Okun screening, outlier removal or regression is performed.
output_file = OUTPUT_DIR / "macro_quarterly_2000_2025.csv"
data.to_csv(output_file, index=True, na_rep="", encoding="utf-8")

gap_columns = [
    "inflation_gap",
    "unemployment_gap",
    "output_gap",
    "expected_inflation_gap",
]

okun_complete = data[
    ["unemployment_gap", "output_gap"]
].notna().all(axis=1)

all_complete = data[gap_columns].notna().all(axis=1)

print(f"Saved: {output_file}")
print(f"Total quarters: {len(data)}")
print(f"Complete Okun observations: {okun_complete.sum()}")
print(f"Complete four-variable observations: {all_complete.sum()}")

print("\nMissing values by variable:")
print(data[gap_columns].isna().sum().to_string())

print("\nFirst five rows:")
print(data[gap_columns].head().round(3).to_string())