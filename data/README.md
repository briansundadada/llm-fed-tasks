# Data provenance and dictionary


## Raw inputs

| File in `raw/` | Source and series | Use |
| --- | --- | --- |
| `UNRATE_2000_2025.csv` | BLS via [FRED UNRATE](https://fred.stlouisfed.org/series/UNRATE) | Monthly unemployment rate, percent |
| `NROU_2000_2025.csv` | CBO via [FRED NROU](https://fred.stlouisfed.org/series/NROU) | Quarterly noncyclical unemployment reference, percent |
| `GDPC1_2000_2025.csv` | BEA via [FRED GDPC1](https://fred.stlouisfed.org/series/GDPC1) | Quarterly real GDP |
| `GDPPOT_2000_2025.csv` | CBO via [FRED GDPPOT](https://fred.stlouisfed.org/series/GDPPOT) | Quarterly real potential GDP in matching chained-dollar units |
| `PCEPI_1999_2025.csv` | BEA via [FRED PCEPI](https://fred.stlouisfed.org/series/PCEPI) | Monthly headline PCE price index; 1999 supplies year-over-year lags |
| `SPF_PCE_median_2000_2025.csv` | Philadelphia Fed [Survey of Professional Forecasters](https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/survey-of-professional-forecasters) | Survey medians; PCE3–PCE6 supply forecasts for t+1 through t+4 |



## Processed historical panel

`processed/macro_quarterly_2000_2025.csv` has one row per quarter, including incomplete rows. No interpolation or outlier removal is used.

| Column | Definition / unit |
| --- | --- |
| `quarter` | Calendar quarter, 2000Q1–2025Q4 |
| `unemployment_rate` | Mean of three monthly UNRATE observations, percent |
| `natural_unemployment_rate` | Quarterly NROU, percent |
| `real_gdp`, `potential_gdp` | Matching chained-dollar GDP series |
| `pce_price_index` | Mean of three monthly PCEPI observations |
| `inflation_rate` | 100 × (quarterly PCEPI / value four quarters earlier − 1) |
| `expected_inflation_1y` | 100 × (product of `(1 + PCEh / 100)^0.25` over h=3..6 − 1) |
| `inflation_target` | Common 2% comparison benchmark throughout the panel |
| `inflation_gap` | Inflation rate − 2, percentage points |
| `unemployment_gap` | Unemployment rate − time-varying NROU, percentage points |
| `output_gap` | 100 × (real GDP / potential GDP − 1), percent |
| `expected_inflation_gap` | Expected one-year inflation − 2, percentage points |

There are 104 quarters, 103 complete unemployment/output pairs, and 75 complete four-gap observations. The supplied UNRATE snapshot lacks October 2025, so 2025Q4 unemployment is missing even though November and December are present. There are 28 missing expectations-gap values. These missing values have different causes and do not justify dropping all incomplete quarters from the Okun regression.

The expectations measure compounds horizon-specific medians; it is not the median of respondent-level one-year forecasts. Historical GDP and unemployment estimates can be revised and are used for calibration, not as a real-time information set.

## Synthetic tables

Task 1 uses `expectations_gap` for the synthetic counterpart of historical `expected_inflation_gap`. Its unemployment reference is fixed at 4.5%, whereas historical gaps use time-varying NROU. All gaps are numeric; IDs `S000`–`S096` are strings and must retain their prefix and leading zeros.

Task 2 stores `decision_bps` (rounded observed action), `latent_rate_bps` (internal continuous rule), and `region` (internal coverage diagnostic). `history_rank` orders a common 20-scenario sequence; `history_lengths` lists the prefix lengths containing that row, separated by semicolons. It is a compact membership field, not separate independent samples. `type` is an internal ground-truth identifier.
