"""Regenerate the bundled Task 1/2 results and check cross-file consistency."""
from pathlib import Path
import hashlib
import json
import os
import platform
import subprocess
import sys

import matplotlib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
STEPS = ("preprocess.py", "calibrate_okun.py", "generate_scenarios.py", "generate_policymakers.py", "verify_results.py")


def main():
    results = ROOT / "results"
    results.mkdir(exist_ok=True)
    manifest = results / "run_manifest.json"
    # Do not leave a stale success manifest if a new run fails.
    if manifest.exists():
        manifest.unlink()
    with (results / "pipeline.log").open("w", encoding="utf-8") as log:
        for step in STEPS:
            print(f"Running {step}", flush=True)
            run = subprocess.run([sys.executable, str(ROOT / "scripts" / step)], cwd=ROOT,
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 encoding="utf-8", errors="replace",
                                 env={**os.environ, "PYTHONIOENCODING": "utf-8"})
            log.write(f"\n--- {step} ---\n{run.stdout}")
            log.flush()
            if run.returncode:
                print(run.stdout)
                raise SystemExit(run.returncode)
    paths = sorted(set((ROOT / "data").rglob("*.csv")) | set((ROOT / "scripts").glob("*.py")) |
                   {ROOT / "requirements.txt"} |
                   {p for p in results.rglob("*") if p.is_file() and p.suffix in (".csv", ".json", ".txt", ".png")})
    record = {
        "status": "passed; Task 2 remains provisional",
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "matplotlib": matplotlib.__version__},
        "steps": list(STEPS),
        "sha256": {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
    }
    manifest.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print("All steps passed. See results/verification.txt and results/run_manifest.json.")


if __name__ == "__main__":
    main()
