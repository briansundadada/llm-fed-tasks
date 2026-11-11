# Historical references

`preprocess_new.py.txt` is the supplied alternate preprocessing script, preserved as text. Its original paths expected raw CSVs beside the script and do not match this repository. The maintained entry point is `scripts/preprocess.py`, which also validates duplicate dates, parses numeric values explicitly, and retains the quarterly PCE price index.

`original_checksums.json` records SHA-256 values of the supplied scenario table and processed historical table before regeneration, using their locations immediately after directory organization. The scenario CSV is unchanged. The processed table has been regenerated from the unchanged raw inputs, with possible floating-point serialization differences.

The original Word documents are one directory above. Consult `docs/methods.md` and generated reports for current run-specific values.
