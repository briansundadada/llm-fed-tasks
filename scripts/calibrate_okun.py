"""Reproduce the Task 1 historical Okun calibration using bundled data."""
from pathlib import Path
import argparse
import hashlib
import json

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def regression(frame, hac_lags=4):
    rows = frame.dropna(subset=["unemployment_gap", "output_gap"])
    x = np.column_stack([np.ones(len(rows)), rows.unemployment_gap])
    y = rows.output_gap.to_numpy()
    coef = np.linalg.lstsq(x, y, rcond=None)[0]
    residual = y - x @ coef
    n, k = x.shape
    bread = np.linalg.inv(x.T @ x)
    ols_cov = (residual @ residual / (n - k)) * bread
    scores = x * residual[:, None]
    meat = scores.T @ scores
    for lag in range(1, hac_lags + 1):
        cross = scores[lag:].T @ scores[:-lag]
        meat += (1 - lag / (hac_lags + 1)) * (cross + cross.T)
    hac_cov = bread @ meat @ bread * n / (n - k)
    return {
        "n": n,
        "intercept": float(coef[0]),
        "slope": float(coef[1]),
        "slope_se_ols": float(np.sqrt(ols_cov[1, 1])),
        "slope_se_newey_west": float(np.sqrt(hac_cov[1, 1])),
        "residual_interval": np.percentile(residual, [2.5, 97.5], method="averaged_inverted_cdf").tolist(),
    }, residual


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/processed/macro_quarterly_2000_2025.csv")
    parser.add_argument("--out", type=Path, default=ROOT / "results/task1")
    args = parser.parse_args()
    frame = pd.read_csv(args.data)
    baseline, residual = regression(frame)
    no_2020, _ = regression(frame.loc[~frame.quarter.str.startswith("2020")])
    common, _ = regression(frame.dropna(subset=["inflation_gap", "expected_inflation_gap"]))
    baseline.update({
        "sample": "2000Q1-2025Q3; complete unemployment/output gap observations",
        "percentile_method": "averaged_inverted_cdf (empirical quantiles; matches supplied thresholds)",
        "newey_west": "4 lags, Bartlett kernel, n/(n-k) finite-sample correction",
        "source_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
        "robustness": {"excluding_2020": no_2020, "complete_four_gaps": common},
    })
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "okun_calibration.json").write_text(json.dumps(baseline, indent=2) + "\n", encoding="utf-8")
    complete = frame.dropna(subset=["unemployment_gap", "output_gap"]).copy()
    complete["residual"] = residual
    complete[["quarter", "unemployment_gap", "output_gap", "residual"]].to_csv(args.out / "okun_residuals.csv", index=False)
    print(json.dumps(baseline, indent=2))


if __name__ == "__main__":
    main()
