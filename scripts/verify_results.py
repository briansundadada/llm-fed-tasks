"""Check the scientific contract between Task 1 and Task 2 saved tables."""
from pathlib import Path
from itertools import product
import json

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
GAPS = ["inflation_gap", "unemployment_gap", "expectations_gap", "output_gap"]


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def main():
    output = ROOT / "results/verification.txt"
    # Clear old success before checking a new set of results.
    output.write_text("Verification in progress; no success recorded.\n", encoding="utf-8")
    scenarios = pd.read_csv(ROOT / "results/task1/scenarios.csv")
    base = scenarios.loc[scenarios.scenario_id != "S000"]
    require(len(scenarios) == 97 and scenarios.scenario_id.nunique() == 97, "97 unique scenarios required")
    neutral = scenarios.loc[scenarios.scenario_id == "S000", GAPS]
    require(len(neutral) == 1 and (neutral == 0).all().all(), "invalid neutral control")
    require(len(base.groupby(GAPS[:2])) == 16 and (base.groupby(GAPS[:2]).size() == 6).all(), "invalid 16 x 6 base design")
    calibration = json.loads((ROOT / "results/task1/okun_calibration.json").read_text())
    lo, hi = calibration["residual_interval"]
    residual = base.output_gap - calibration["intercept"] - calibration["slope"] * base.unemployment_gap
    require(residual.between(lo, hi).all(), "scenario violates calibrated screen")
    grid = pd.DataFrame(product([-1, 0, 1, 2], [-1.5, -.5, .5, 1.5], [-.5, 0, .5, 1], [-3, -1.5, 0, 1.5, 3]), columns=GAPS)
    screened = grid.output_gap - calibration["intercept"] - calibration["slope"] * grid.unemployment_gap
    require(int(screened.between(lo, hi).sum()) == 144, "expected 144 eligible candidates")
    order = pd.read_csv(ROOT / "results/task1/presentation_order.csv")
    pd.testing.assert_frame_equal(scenarios.sort_values("scenario_id").reset_index(drop=True), order.sort_values("scenario_id").reset_index(drop=True))
    provisional = ROOT / "results/task2/provisional"
    split = pd.read_csv(provisional / "scenario_split.csv")
    require(len(split) == 96 and split.scenario_id.nunique() == 96 and set(split.scenario_id) == set(base.scenario_id), "split must cover the base exactly once")
    require(set(split.split) == {"train", "test"}, "invalid split labels")
    require((split.groupby(GAPS[:2] + ["split"]).size() == 3).all(), "each cell must split 3/3")
    indexed = scenarios.set_index("scenario_id")
    for name in ("scenario_split.csv", "decision_histories.csv", "test_targets.csv", "neutral_control.csv"):
        table = pd.read_csv(provisional / name)
        require(set(table.status) == {"provisional"}, f"missing provisional status: {name}")
        require(np.allclose(table[GAPS].to_numpy(), indexed.loc[table.scenario_id, GAPS].to_numpy()), f"scenario inputs differ: {name}")
    histories = pd.read_csv(provisional / "decision_histories.csv")
    targets = pd.read_csv(provisional / "test_targets.csv")
    controls = pd.read_csv(provisional / "neutral_control.csv")
    types = pd.read_csv(ROOT / "results/task2/policymaker_types.csv").set_index("type")
    require(len(types) == 5 and set(types.index) == set("ABCDE"), "expected five policymaker types")
    require(len(histories) == 100 and len(targets) == 240 and len(controls) == 5, "incorrect output row counts")
    train_ids = set(split.loc[split.split == "train", "scenario_id"])
    test_ids = set(split.loc[split.split == "test", "scenario_id"])
    ratios = []
    for name in types.index:
        history = histories.loc[histories.type == name].sort_values("history_rank")
        require(history.history_rank.tolist() == list(range(1, 21)), "invalid history ranks")
        require(history.scenario_id.nunique() == 20 and set(history.scenario_id) <= train_ids, "history leakage/duplicates")
        require(history.groupby(GAPS[:2]).size().max() <= 2, "history cell cap exceeded")
        reference = histories.loc[histories.type == "A"].sort_values("history_rank").scenario_id.tolist()
        require(history.scenario_id.tolist() == reference, "types must share history scenarios")
        require(set(history.iloc[:5].region) >= {"tightening", "easing", "boundary"}, "missing first-five region coverage")
        for k in (2, 5, 10, 20):
            membership = history.history_lengths.astype(str).str.split(";").apply(lambda values: str(k) in values)
            require(history.loc[membership, "scenario_id"].tolist() == history.iloc[:k].scenario_id.tolist(), "histories are not nested prefixes")
            if k >= 5:
                require(np.linalg.matrix_rank(np.column_stack([np.ones(k), history.iloc[:k][GAPS]])) == 5, "history lacks full rank")
        test = targets.loc[targets.type == name]
        require(len(test) == 48 and test.scenario_id.nunique() == 48 and set(test.scenario_id) == test_ids, "test target coverage differs")
        control = controls.loc[controls.type == name]
        require(len(control) == 1 and control.scenario_id.iloc[0] == "S000" and control.decision_bps.iloc[0] == 0, "incorrect neutral control")
        for table in (history, test, control):
            spec = types.loc[name]
            latent = spec.alpha_bps + table[GAPS].to_numpy() @ np.array([spec.beta_pi, -spec.beta_u, spec.beta_exp, spec.beta_y])
            decision = np.clip(np.sign(latent) * np.floor(np.abs(latent) / 5 + .5) * 5, -100, 100)
            require(np.allclose(latent, table.latent_rate_bps) and np.array_equal(decision, table.decision_bps), "ground-truth calculation differs")
        coef = np.linalg.lstsq(np.column_stack([np.ones(len(test)), test[GAPS]]), test.decision_bps, rcond=None)[0]
        ratios.append(float(coef[1] / -coef[2]))
    require(all(a > b for a, b in zip(ratios, ratios[1:])), "held-out preference ordering is not strict")
    require("RESULT: PASSED" in (provisional / "check_report.txt").read_text(), "generator reported failure")
    historical = pd.read_csv(ROOT / "data/processed/macro_quarterly_2000_2025.csv")
    require(len(historical) == 104 and historical.quarter.nunique() == 104, "invalid historical panel")
    report = "\n".join([
        "RESULT: PASSED",
        "Historical panel: 104 quarters; calibration: 103 complete Okun observations.",
        "Task 1: 320 candidates, 144 eligible, 96 balanced base + S000 = 97 unique scenarios.",
        "Saved presentation order preserves all scenario IDs and inputs.",
        "Task 2: 48 train / 48 test; S000 held separately; no history/test leakage.",
        "Five shared scenario histories: 20 decisions each, nested prefixes 2/5/10/20.",
        "First 5/10/20 histories full rank; first five cover all required regions; cell cap <= 2.",
        "All saved decisions reproduce the reaction function and 5 bp rounding rule.",
        "Held-out recovered ratios (A-E): " + ", ".join(f"{r:.3f}" for r in ratios),
        "This validates generated experimental materials, not LLM performance.",
    ]) + "\n"
    output.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
