"""Construct synthetic policymaker types and provisional decision histories.

The reaction functions, rounding rule, train/test algorithm, and history
algorithm do not depend on which scenario sample is finally approved. Rows in
``provisional/`` are generated from whatever scenario file is passed in and
must be regenerated after that file is locked.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import tempfile
import platform
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="mpl-task2-"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

TYPES = (
    # alpha and betas are in basis points per percentage point.
    # beta_pi + beta_u is 30 for every type, so types differ in composition.
    # beta_exp and beta_y stay fixed.
    ("A", 0.0, 25.0, 5.0, 8.0, 2.0, "strongly inflation-sensitive"),
    ("B", 0.0, 20.0, 10.0, 8.0, 2.0, "moderately inflation-sensitive"),
    ("C", 0.0, 15.0, 15.0, 8.0, 2.0, "balanced inflation and employment"),
    ("D", 0.0, 10.0, 20.0, 8.0, 2.0, "moderately employment-sensitive"),
    ("E", 0.0, 5.0, 25.0, 8.0, 2.0, "strongly employment-sensitive"),
)

# 5 bp retains the expectations and output slopes. A 25 bp step erases them
# and pushes Type A's recovered inflation/unemployment ratio well above 5.
RATE_STEP = 5
RATE_MIN = -100
RATE_MAX = 100
HISTORY_LENGTHS = (2, 5, 10, 20)
CELL_CAP = 2
SPLIT_SEEDS = range(1000)
REQUIRED_REGIONS = ("tightening", "easing", "boundary")
VARIABLES = (
    "inflation_gap",
    "unemployment_gap",
    "expectations_gap",
    "output_gap",
)
# Ranges of the Task 1 design, used only to scale distances.
SCALES = {
    "inflation_gap": 3.0,
    "unemployment_gap": 3.0,
    "expectations_gap": 1.5,
    "output_gap": 6.0,
}


@dataclass(frozen=True)
class TypeSpec:
    name: str
    alpha: float
    beta_pi: float
    beta_u: float
    beta_exp: float
    beta_y: float
    internal_label: str

    @property
    def preference_ratio(self) -> float:
        return self.beta_pi / self.beta_u


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    inflation_gap: float
    unemployment_gap: float
    expectations_gap: float
    output_gap: float

    @property
    def cell(self) -> tuple[float, float]:
        return (self.inflation_gap, self.unemployment_gap)


def type_specs() -> list[TypeSpec]:
    return [TypeSpec(*row) for row in TYPES]


def round_rate(latent: float) -> int:
    """Round to the nearest rate step. Ties move away from zero, then clip."""
    if latent >= 0:
        steps = math.floor(latent / RATE_STEP + 0.5)
    else:
        steps = math.ceil(latent / RATE_STEP - 0.5)
    value = int(steps * RATE_STEP)
    return max(RATE_MIN, min(RATE_MAX, value))


def check_rounding_rule() -> None:
    expected = {
        0: 0,
        2.4: 0,
        2.5: 5,
        2.6: 5,
        -2.4: 0,
        -2.5: -5,
        -2.6: -5,
        7.5: 10,
        -7.5: -10,
        100: 100,
        -100: -100,
        102: 100,
        -102: -100,
    }
    for raw, target in expected.items():
        got = round_rate(raw)
        if got != target:
            raise AssertionError(f"round_rate({raw}) = {got}, expected {target}")


def latent_rate(scenario: Scenario, spec: TypeSpec) -> float:
    return (
        spec.alpha
        + spec.beta_pi * scenario.inflation_gap
        - spec.beta_u * scenario.unemployment_gap
        + spec.beta_exp * scenario.expectations_gap
        + spec.beta_y * scenario.output_gap
    )


def region_of(scenario: Scenario) -> str:
    """Classify a state without using policymaker coefficients.

    Tightening and easing are the clear corners. Boundary states are the
    remaining states whose inflation and unemployment gaps nearly offset.
    """
    pi = scenario.inflation_gap
    u = scenario.unemployment_gap
    if pi >= 1 and u <= -0.5:
        return "tightening"
    if pi <= 0 and u >= 0.5:
        return "easing"
    if abs(pi - u) <= 1:
        return "boundary"
    return "other"


def load_scenarios(path: Path) -> list[Scenario]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing = set(VARIABLES + ("scenario_id",)) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        scenarios = []
        for row in reader:
            scenarios.append(
                Scenario(
                    scenario_id=row["scenario_id"].strip(),
                    inflation_gap=float(row["inflation_gap"]),
                    unemployment_gap=float(row["unemployment_gap"]),
                    expectations_gap=float(row["expectations_gap"]),
                    output_gap=float(row["output_gap"]),
                )
            )
    ids = [s.scenario_id for s in scenarios]
    if any(not value for value in ids):
        raise ValueError("scenario_id values must not be empty")
    if any(not math.isfinite(getattr(s, name)) for s in scenarios for name in VARIABLES):
        raise ValueError("scenario values must be finite")
    if len(ids) != len(set(ids)):
        raise ValueError("scenario_id values are not unique")
    if len(scenarios) < HISTORY_LENGTHS[-1] * 2:
        raise ValueError("need at least 40 scenarios to hold a 20-row history and a test set")
    return sorted(scenarios, key=lambda s: s.scenario_id)


def design(rows: list[Scenario]) -> np.ndarray:
    return np.column_stack(
        [
            np.ones(len(rows)),
            [s.inflation_gap for s in rows],
            [s.unemployment_gap for s in rows],
            [s.expectations_gap for s in rows],
            [s.output_gap for s in rows],
        ]
    ).astype(float)


def numeric_rank(rows: list[Scenario]) -> int:
    if not rows:
        return 0
    singular = np.linalg.svd(design(rows), compute_uv=False)
    if singular[0] == 0:
        return 0
    return int(np.sum(singular > 1e-8 * singular[0]))


def standardized_condition(rows: list[Scenario]) -> float:
    if numeric_rank(rows) < design(rows).shape[1]:
        return math.inf
    scaled = design(rows)
    for col in range(1, scaled.shape[1]):
        sd = scaled[:, col].std()
        if sd < 1e-12:
            return math.inf
        scaled[:, col] = (scaled[:, col] - scaled[:, col].mean()) / sd
    return float(np.linalg.cond(scaled))


def fit_reaction(rows: list[Scenario], outcomes: np.ndarray) -> np.ndarray:
    coef, *_ = np.linalg.lstsq(design(rows), outcomes, rcond=None)
    return coef


def recover_ratio(coef: np.ndarray) -> float:
    beta_u = -coef[2]
    if abs(beta_u) < 1e-8:
        return math.inf
    return float(coef[1] / beta_u)


def split_once(scenarios: list[Scenario], seed: int) -> tuple[list[Scenario], list[Scenario]]:
    groups: dict[tuple[float, float], list[Scenario]] = defaultdict(list)
    for scenario in scenarios:
        groups[scenario.cell].append(scenario)
    rng = random.Random(seed)
    train: list[Scenario] = []
    test: list[Scenario] = []
    for cell in sorted(groups):
        items = groups[cell][:]
        if len(items) < 2:
            raise ValueError(f"cell {cell} has fewer than 2 scenarios")
        rng.shuffle(items)
        n_train = len(items) // 2
        train.extend(items[:n_train])
        test.extend(items[n_train:])
    return train, test


def imbalance(train: list[Scenario], test: list[Scenario]) -> float:
    total = 0
    n = len(train) + len(test)
    for attr in ("expectations_gap", "output_gap"):
        levels = {getattr(s, attr) for s in train + test}
        for level in levels:
            n_train = sum(getattr(s, attr) == level for s in train)
            n_test = sum(getattr(s, attr) == level for s in test)
            total += abs(n_train - n_test)
    return total / n


def split_score(train: list[Scenario], test: list[Scenario]) -> float:
    score = standardized_condition(train) + standardized_condition(test)
    score += 5 * imbalance(train, test)
    for part in (train, test):
        if numeric_rank(part) < 5:
            score += 1000
        present = {region_of(s) for s in part}
        if any(name not in present for name in REQUIRED_REGIONS):
            score += 1000
    return score


def choose_split(scenarios: list[Scenario]) -> tuple[int, list[Scenario], list[Scenario], float]:
    best: tuple[float, int] | None = None
    chosen: tuple[list[Scenario], list[Scenario]] | None = None
    for seed in SPLIT_SEEDS:
        train, test = split_once(scenarios, seed)
        score = split_score(train, test)
        if best is None or (score, seed) < best:
            best = (score, seed)
            chosen = (train, test)
    assert best is not None and chosen is not None
    return best[1], chosen[0], chosen[1], best[0]


def scaled_vector(scenario: Scenario) -> np.ndarray:
    return np.array([getattr(scenario, name) / SCALES[name] for name in VARIABLES])


def distance(left: Scenario, right: Scenario) -> float:
    return float(np.linalg.norm(scaled_vector(left) - scaled_vector(right)))


def level_bonus(candidate: Scenario, chosen: list[Scenario]) -> int:
    bonus = 0
    for name in VARIABLES:
        seen = {getattr(s, name) for s in chosen}
        if getattr(candidate, name) not in seen:
            bonus += 1
    return bonus


def candidate_score(candidate: Scenario, chosen: list[Scenario]) -> float:
    trial = chosen + [candidate]
    gain = numeric_rank(trial) - numeric_rank(chosen)
    nearest = min(distance(candidate, current) for current in chosen)
    region_bonus = 0.0
    region = region_of(candidate)
    if region in REQUIRED_REGIONS and all(region_of(s) != region for s in chosen):
        region_bonus = 3.0
    penalty = 0.0
    if numeric_rank(trial) == 5:
        penalty = 0.02 * standardized_condition(trial)
    return gain * 100 + nearest * 5 + level_bonus(candidate, chosen) * 2 + region_bonus - penalty


def id_tiebreak(scenario: Scenario) -> str:
    # Lexical IDs provide a stable ordering without converting S-prefixed IDs.
    return scenario.scenario_id


def choose_initial_pair(train: list[Scenario]) -> list[Scenario]:
    tightening = [s for s in train if region_of(s) == "tightening"]
    easing = [s for s in train if region_of(s) == "easing"]
    if not tightening or not easing:
        raise RuntimeError("training split does not contain both tightening and easing states")
    best: tuple[tuple[float, str, str], Scenario, Scenario] | None = None
    for left in tightening:
        for right in easing:
            key = (-distance(left, right), left.scenario_id, right.scenario_id)
            if best is None or key < best[0]:
                best = (key, left, right)
    assert best is not None
    return [best[1], best[2]]


def next_scenario(chosen: list[Scenario], pool: list[Scenario], force_boundary: bool) -> Scenario:
    counts = Counter(s.cell for s in chosen)
    feasible = [s for s in pool if counts[s.cell] < CELL_CAP]
    if not feasible:
        raise RuntimeError("cell cap leaves no scenario to add")
    if numeric_rank(chosen) < 5:
        rankers = [s for s in feasible if numeric_rank(chosen + [s]) > numeric_rank(chosen)]
        if rankers:
            feasible = rankers
    elif force_boundary and all(region_of(s) != "boundary" for s in chosen):
        boundaries = [
            s
            for s in feasible
            if region_of(s) == "boundary" and numeric_rank(chosen + [s]) == 5
        ]
        if boundaries:
            feasible = boundaries
    return min(feasible, key=lambda s: (-candidate_score(s, chosen), id_tiebreak(s)))


def repair_boundary(chosen: list[Scenario], train: list[Scenario]) -> list[Scenario]:
    if any(region_of(s) == "boundary" for s in chosen):
        return chosen
    counts = Counter(s.cell for s in chosen)
    best = None
    for index in (4, 3, 2):
        for candidate in train:
            if candidate in chosen or region_of(candidate) != "boundary":
                continue
            trial = chosen[:]
            removed = trial[index]
            trial[index] = candidate
            trial_counts = counts.copy()
            trial_counts[removed.cell] -= 1
            trial_counts[candidate.cell] += 1
            if trial_counts[candidate.cell] > CELL_CAP:
                continue
            if numeric_rank(trial) < 5:
                continue
            score = candidate_score(candidate, trial[:index] + trial[index + 1 :])
            key = (-score, candidate.scenario_id, index)
            if best is None or key < best[:3]:
                best = (*key, candidate)
                best_index = index
    if best is None:
        raise RuntimeError("could not place a boundary state in the first 5 histories")
    repaired = chosen[:]
    repaired[best_index] = best[3]
    return repaired


def select_history(train: list[Scenario]) -> list[Scenario]:
    chosen = choose_initial_pair(train)
    remaining = [s for s in train if s.scenario_id not in {x.scenario_id for x in chosen}]
    while len(chosen) < 5:
        pick = next_scenario(chosen, remaining, force_boundary=(len(chosen) == 4))
        chosen.append(pick)
        remaining = [s for s in remaining if s.scenario_id != pick.scenario_id]
    chosen = repair_boundary(chosen, train)
    remaining = [s for s in train if s.scenario_id not in {x.scenario_id for x in chosen}]
    while len(chosen) < HISTORY_LENGTHS[-1]:
        pick = next_scenario(chosen, remaining, force_boundary=False)
        chosen.append(pick)
        remaining = [s for s in remaining if s.scenario_id != pick.scenario_id]
    return chosen


def history_lengths(rank: int) -> str:
    return ";".join(str(k) for k in HISTORY_LENGTHS if rank <= k)


def evaluate(scenarios: list[Scenario], train: list[Scenario], test: list[Scenario], history: list[Scenario]) -> list[str]:
    problems: list[str] = []
    train_ids = {s.scenario_id for s in train}
    test_ids = {s.scenario_id for s in test}
    if train_ids & test_ids:
        problems.append("train and test share scenario ids")
    if train_ids | test_ids != {s.scenario_id for s in scenarios}:
        problems.append("train and test do not cover every scenario exactly once")
    for name, rows in (("train", train), ("test", test)):
        if numeric_rank(rows) < 5:
            problems.append(f"{name} design is not full rank")
        if not set(REQUIRED_REGIONS) <= {region_of(s) for s in rows}:
            problems.append(f"{name} is missing required regions")
    cells = {s.cell for s in scenarios}
    for cell in cells:
        n_train = sum(s.cell == cell for s in train)
        n_test = sum(s.cell == cell for s in test)
        if abs(n_train - n_test) > 1:
            problems.append(f"cell {cell} splits {n_train}/{n_test}")
    history_ids = [s.scenario_id for s in history]
    if len(history_ids) != len(set(history_ids)):
        problems.append("history contains duplicate scenarios")
    if set(history_ids) - train_ids:
        problems.append("history uses a scenario outside the training split")
    if numeric_rank(history[:2]) < 2:
        problems.append("first 2 histories are rank deficient")
    for length in HISTORY_LENGTHS:
        if length >= 5 and numeric_rank(history[:length]) < 5:
            problems.append(f"first {length} histories have rank below 5")
    present = {region_of(s) for s in history[:5]}
    for name in REQUIRED_REGIONS:
        if name not in present:
            problems.append(f"first 5 histories miss {name}")
    counts = Counter(s.cell for s in history)
    if max(counts.values()) > CELL_CAP:
        problems.append("history exceeds the per-cell cap")

    specs = type_specs()
    ratios = []
    for spec in specs:
        latent = np.array([latent_rate(s, spec) for s in scenarios])
        decisions = np.array([round_rate(v) for v in latent], dtype=float)
        if np.any(latent < RATE_MIN) or np.any(latent > RATE_MAX):
            problems.append(f"type {spec.name} has a latent rate outside the grid")
        if len(set(decisions.tolist())) < 3:
            problems.append(f"type {spec.name} uses fewer than 3 decision bins")
        recovered = fit_reaction(scenarios, latent)
        target = np.array([spec.alpha, spec.beta_pi, -spec.beta_u, spec.beta_exp, spec.beta_y])
        if np.max(np.abs(recovered - target)) > 1e-6:
            problems.append(f"type {spec.name} continuous coefficients were not recovered")
        ratios.append(recover_ratio(fit_reaction(scenarios, decisions)))
    if ratios != sorted(ratios, reverse=True) or len(set(np.round(ratios, 6))) < len(ratios):
        problems.append(f"discrete preference ratios are not strictly ordered: {ratios}")
    test_ratios = []
    for spec in specs:
        decisions = np.array([round_rate(latent_rate(s, spec)) for s in test], dtype=float)
        test_ratios.append(recover_ratio(fit_reaction(test, decisions)))
    if test_ratios != sorted(test_ratios, reverse=True):
        problems.append(f"test-set discrete ratios are not strictly ordered: {test_ratios}")
    return problems


def write_types(path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "type",
                "alpha_bps",
                "beta_pi",
                "beta_u",
                "beta_exp",
                "beta_y",
                "preference_ratio",
                "internal_label",
            ]
        )
        for spec in type_specs():
            writer.writerow(
                [
                    spec.name,
                    spec.alpha,
                    spec.beta_pi,
                    spec.beta_u,
                    spec.beta_exp,
                    spec.beta_y,
                    spec.preference_ratio,
                    spec.internal_label,
                ]
            )


def write_surfaces(path: Path) -> None:
    inflation = np.linspace(-1, 2, 161)
    unemployment = np.linspace(-1.5, 1.5, 161)
    grid_u, grid_pi = np.meshgrid(unemployment, inflation)
    surfaces = []
    for spec in type_specs():
        surfaces.append(
            spec.alpha
            + spec.beta_pi * grid_pi
            - spec.beta_u * grid_u
        )
    stacked = np.stack(surfaces)
    levels = np.linspace(stacked.min(), stacked.max(), 15)
    fig, axes = plt.subplots(1, 5, figsize=(16, 3.8), sharey=True)
    image = None
    for ax, spec, surface in zip(axes, type_specs(), surfaces):
        image = ax.contourf(grid_u, grid_pi, surface, levels=levels, cmap="RdBu_r")
        ax.contour(grid_u, grid_pi, surface, levels=[0], colors="black", linewidths=0.9)
        ax.set_title(f"Type {spec.name}\n$\\beta_\\pi/\\beta_u = {spec.preference_ratio:g}$")
        ax.set_xlabel("Unemployment gap")
        ax.set_xticks([-1.5, -0.5, 0.5, 1.5])
        ax.set_yticks([-1, 0, 1, 2])
    axes[0].set_ylabel("Inflation gap")
    fig.colorbar(image, ax=axes.tolist(), fraction=0.02, pad=0.02, label="Latent rate change (bp)")
    fig.suptitle("True policy surfaces at zero expectations gap and zero output gap", y=1.05)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def fmt(value: float) -> str:
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.4f}".rstrip("0").rstrip(".")


def scenario_fields(scenario: Scenario) -> list[str]:
    return [
        str(scenario.scenario_id),
        fmt(scenario.inflation_gap),
        fmt(scenario.unemployment_gap),
        fmt(scenario.expectations_gap),
        fmt(scenario.output_gap),
        region_of(scenario),
    ]


def write_provisional(
    directory: Path,
    scenarios: list[Scenario],
    train: list[Scenario],
    test: list[Scenario],
    history: list[Scenario],
    metadata: dict,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    split_of = {s.scenario_id: "train" for s in train}
    split_of.update({s.scenario_id: "test" for s in test})
    with (directory / "scenario_split.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["status", "scenario_id", *VARIABLES, "region", "split"]
        )
        for scenario in scenarios:
            writer.writerow(["provisional", *scenario_fields(scenario), split_of[scenario.scenario_id]])

    with (directory / "decision_histories.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "status",
                "type",
                "history_rank",
                "history_lengths",
                "scenario_id",
                *VARIABLES,
                "region",
                "latent_rate_bps",
                "decision_bps",
            ]
        )
        for spec in type_specs():
            for rank, scenario in enumerate(history, start=1):
                latent = latent_rate(scenario, spec)
                writer.writerow(
                    [
                        "provisional",
                        spec.name,
                        rank,
                        history_lengths(rank),
                        *scenario_fields(scenario),
                        fmt(latent),
                        round_rate(latent),
                    ]
                )

    with (directory / "test_targets.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "status",
                "type",
                "scenario_id",
                *VARIABLES,
                "region",
                "latent_rate_bps",
                "decision_bps",
            ]
        )
        for spec in type_specs():
            for scenario in sorted(test, key=lambda s: s.scenario_id):
                latent = latent_rate(scenario, spec)
                writer.writerow(
                    [
                        "provisional",
                        spec.name,
                        *scenario_fields(scenario),
                        fmt(latent),
                        round_rate(latent),
                    ]
                )
    (directory / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    (directory / "README.txt").write_text(
        "These files were generated from a scenario table that has not been signed off.\n"
        "Regenerate them after the Task 1 scenario file is locked.\n"
        "Do not treat scenario ids, the split, or the histories as final.\n",
        encoding="utf-8",
    )


def decision_summary(scenarios: list[Scenario]) -> dict[str, dict[int, int]]:
    summary = {}
    for spec in type_specs():
        counts = Counter(round_rate(latent_rate(s, spec)) for s in scenarios)
        summary[spec.name] = dict(sorted(counts.items()))
    return summary


def build_report(
    problems: list[str],
    scenarios: list[Scenario],
    train: list[Scenario],
    test: list[Scenario],
    history: list[Scenario],
    seed: int,
    score: float,
) -> str:
    lines = [
        "PROVISIONAL CHECK REPORT",
        "Regenerate after the Task 1 scenario file is locked.",
        f"split_seed: {seed}",
        f"split_score: {score:.4f}",
        f"n_scenarios: {len(scenarios)}",
        f"n_train: {len(train)}",
        f"n_test: {len(test)}",
        f"train_condition: {standardized_condition(train):.4f}",
        f"test_condition: {standardized_condition(test):.4f}",
        f"history_condition_5: {standardized_condition(history[:5]):.4f}",
        f"history_condition_10: {standardized_condition(history[:10]):.4f}",
        f"history_condition_20: {standardized_condition(history[:20]):.4f}",
        "history_regions_first5: " + ",".join(region_of(s) for s in history[:5]),
        "history_ranks: "
        + ", ".join(f"{k}:{numeric_rank(history[:k])}" for k in (2, 5, 10, 20)),
    ]
    for spec in type_specs():
        latent = np.array([latent_rate(s, spec) for s in scenarios])
        decisions = np.array([round_rate(v) for v in latent], dtype=float)
        ratio = recover_ratio(fit_reaction(scenarios, decisions))
        test_ratio = recover_ratio(
            fit_reaction(test, np.array([round_rate(latent_rate(s, spec)) for s in test], dtype=float))
        )
        lines.append(
            f"type {spec.name}: true_ratio={spec.preference_ratio:g} "
            f"discrete_ratio={ratio:.3f} test_discrete_ratio={test_ratio:.3f} "
            f"latent=[{latent.min():.1f}, {latent.max():.1f}] "
            f"bins={decision_summary(scenarios)[spec.name]}"
        )
    if problems:
        lines.append("RESULT: FAILED")
        lines.extend(f"- {problem}" for problem in problems)
    else:
        lines.append("RESULT: PASSED")
    return "\n".join(lines) + "\n"


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    check_rounding_rule()
    parser = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parents[1]
    parser.add_argument(
        "--scenarios",
        type=Path,
        default=here / "results/task1/scenarios.csv",
        help="scenario CSV to read; current file is only a provisional input",
    )
    parser.add_argument("--out", type=Path, default=here / "results/task2")
    args = parser.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    write_types(out / "policymaker_types.csv")
    write_surfaces(out / "true_policy_surfaces.png")

    all_scenarios = load_scenarios(args.scenarios)
    neutral = [s for s in all_scenarios if s.scenario_id == "S000"]
    if len(neutral) != 1 or any(getattr(neutral[0], name) != 0 for name in VARIABLES):
        raise ValueError("Expected one all-zero neutral control with ID S000")
    scenarios = [s for s in all_scenarios if s.scenario_id != "S000"]
    seed, train, test, score = choose_split(scenarios)
    history = select_history(train)
    problems = evaluate(scenarios, train, test, history)
    report = build_report(problems, scenarios, train, test, history, seed, score)
    provisional = out / "provisional"
    provisional.mkdir(parents=True, exist_ok=True)
    with (provisional / "neutral_control.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["status", "type", "scenario_id", *VARIABLES, "region", "latent_rate_bps", "decision_bps"])
        for spec in type_specs():
            latent = latent_rate(neutral[0], spec)
            writer.writerow(["provisional", spec.name, *scenario_fields(neutral[0]), fmt(latent), round_rate(latent)])
    (provisional / "check_report.txt").write_text(report, encoding="utf-8")
    print(report)
    if problems:
        raise SystemExit(1)

    metadata = {
        "status": "provisional",
        "generated_on": date.today().isoformat(),
        "scenarios_file": os.path.relpath(args.scenarios, here).replace(os.sep, "/"),
        "scenarios_sha256": file_sha256(args.scenarios),
        "split_seed": seed,
        "split_score": score,
        "history_lengths": list(HISTORY_LENGTHS),
        "n_input": len(all_scenarios),
        "n_base": len(scenarios),
        "n_train": len(train),
        "n_test": len(test),
        "neutral_control": "S000 excluded from train/test and histories; separate control table",
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "matplotlib": matplotlib.__version__},
        "generator_sha256": file_sha256(Path(__file__)),
        "note": "Regenerate after the Task 1 scenario file is locked.",
    }
    write_provisional(provisional, scenarios, train, test, history, metadata)


if __name__ == "__main__":
    main()
