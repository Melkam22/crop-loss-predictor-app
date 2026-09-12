# Working conventions

## Environment

- Conda env `lewagon` is the active Python environment. Run Python and
  Jupyter through it; don't create a new venv.
- Re-run the pipeline notebook end to end with:
  `jupyter nbconvert --to notebook --execute --inplace notebooks/01_data_exploration.ipynb`

## Data

- `data/raw/` is gitignored and must be restored manually. A pipeline failure
  on a missing raw wave folder is a missing-data problem, not a code bug —
  don't work around it in code.
- `data/processed/` is tracked. Regenerate it from the notebook rather than
  editing CSVs by hand.

## Notebooks

- Executed notebooks carry embedded outputs and can exceed read limits. If a
  notebook is too large to read or edit directly, manipulate the `.ipynb`
  JSON with a short Python script (`json.load` → mutate `cell["source"]` as a
  list of lines → clear `outputs`/`execution_count` on code cells →
  `json.dump`), then re-execute via `nbconvert`.
- This repo commits notebooks **with** their outputs (01 carries ~225KB of
  embedded output). Executed outputs are the evidence the pipeline actually
  ran, so don't strip them to tidy up diffs.

## Git

- Work on `main` with remote `origin` (`Melkam22/crop-loss-predictor-app`).
- Don't commit unless asked. Stage specific files rather than `git add .`,
  since raw data and notebook outputs are easy to sweep in by accident.
