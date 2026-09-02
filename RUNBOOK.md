# Climate Risk & Health Prediction — Operations Runbook

This file is the operational memory for this repository. Read it before changing
the workflow, running experiments, or helping from a future ChatGPT session.

## Non-negotiable operating rule

**Do not use the user's PC for model training, evaluation, dependency installs,
data processing, or long-running computation.** Local access is only for small
repository/file operations explicitly requested by the user.

All competition compute is intended to run on **Kaggle**.

## Source-of-truth model

The project deliberately separates code, data, and runtime artifacts:

```text
GitHub repository
  code, configs, orchestration, experiment definitions

Kaggle Input Dataset
  Zindi competition input files

Private Kaggle Dataset
  checkpoints, reports, models, submissions, runtime state

Kaggle VM
  disposable compute only
```

Do not commit competition data, generated submissions, models, or local virtual
environments to Git.

## The one command layer

The project is governed by the `crhp` CLI. Do not turn notebook cells into the
primary orchestration system.

Main commands:

```bash
crhp bootstrap
crhp status
crhp update
crhp hydrate
crhp run baseline
crhp run diagnose-shift
crhp run ablate
crhp run robust-validate
crhp run robust-candidates
crhp run suite
crhp run candidates
crhp run tune-catboost
crhp resume
crhp snapshot
crhp restore
crhp sync -m "message"
```

The stage registry is in:

```text
climateops/stages.py
```

Add new experiments as normal Python modules/scripts and then register them in
that file. Do not add ad-hoc notebook-only workflows.

Current diagnostic stages:

- `diagnose-shift`: adversarial train-vs-test validation, including progressively
  less spatial information, to identify distribution shift before leaderboard use.
- `ablate`: location-group CV across full, no-location, no-spatial, no-interaction,
  no-climate, and demographics/time-only feature views.
- `robust-validate`: repeated location-group CV across multiple split seeds for the
  two preferred low-shift feature spaces (`demographics_time` and `no_spatial`),
  comparing CatBoost, LightGBM, XGBoost, Logistic Regression, and coarse blends.
- `robust-candidates`: generate CV-bagged test predictions using the same 3x5
  location-group split design as `robust-validate`. The primary file is a 50/50
  blend of demographics/time CatBoost and no-spatial Logistic Regression; a 75/25
  CatBoost/XGBoost diversity file and pure demographics/time CatBoost reference
  are also written. Use this stage instead of the older `candidates` stage for
  leaderboard-ready files.

## Fresh Kaggle session

For a public GitHub repository:

```bash
python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/AstralJugs69/Zindi_climate/main/bootstrap_kaggle.py','/kaggle/working/bootstrap_kaggle.py')"
python /kaggle/working/bootstrap_kaggle.py --repo-url https://github.com/AstralJugs69/Zindi_climate.git
cd /kaggle/working/climate-risk-health-prediction
crhp status
```

For a private repository, attach `bootstrap_kaggle.py` as a tiny Kaggle Dataset
and add a Kaggle Secret named `GITHUB_TOKEN`, then run the bootstrapper from the
attached dataset.

## Normal start / update / resume cycle

### Start work

```bash
crhp status
crhp run <stage>
```

`crhp run` automatically updates code and hydrates data before executing the
stage unless explicitly told otherwise.

### Pull code only

```bash
crhp update
```

### Re-discover competition files

```bash
crhp hydrate
```

### Resume after Kaggle reset/disconnect

Bootstrap a fresh VM, then:

```bash
crhp resume
```

`resume` restores persisted state/artifacts and reruns the interrupted or failed
stage. A cleanly completed stage is not rerun automatically.

## Save progress

Generated artifacts are checkpointed to a private Kaggle Dataset when
`CRHP_ARTIFACT_DATASET` is configured.

```bash
crhp snapshot --notes "what was completed"
```

Restore manually with:

```bash
crhp restore
```

The checkpoint store should contain runtime state and generated files such as:

```text
.crhp/state.json
models/
reports/
submissions/
```

It should not contain competition input data or repository source code.

## Commit and push changes from Kaggle

Use:

```bash
crhp sync -m "describe the change"
```

This performs the safe Git lifecycle:

```text
pull/rebase -> stage allowed files -> commit -> push
```

To push code and save runtime artifacts together:

```bash
crhp sync -m "describe the change" --snapshot
```

Never put GitHub credentials directly in a remote URL or tracked config file.
Authentication must come from the Kaggle Secret `GITHUB_TOKEN`.

## Expected Kaggle configuration

Environment/secrets:

```text
CRHP_REPO_URL=https://github.com/AstralJugs69/Zindi_climate.git
CRHP_BRANCH=main
GITHUB_TOKEN=<Kaggle Secret, required for private clone or push>
CRHP_ARTIFACT_DATASET=YOUR_KAGGLE_USER/YOUR_PRIVATE_CHECKPOINT_DATASET
```

Never commit real secret values.

## Competition data contract

The six canonical files are:

```text
Train.csv
Test.csv
SampleSubmission.csv
data_dictionary.csv
downloaded_climate_features_data_dictionary.csv
climate_features.csv
```

`crhp hydrate` searches `/kaggle/input` recursively and exposes them under:

```text
data/raw/
```

Modeling code should always use these canonical paths rather than hard-coding a
Kaggle Dataset slug.

## Competition-specific modeling rules

The official prediction columns are:

```text
TargetF1
TargetRAUC
```

`TargetRAUC` is the predicted probability.

`TargetF1` must always be derived with the fixed threshold:

```python
TargetF1 = (TargetRAUC >= 0.5).astype(int)
```

Do not optimize or change the classification threshold.

The local/offline validation score used by the project is:

```text
0.60 * F1@0.5 + 0.40 * ROC-AUC
```

Important data facts discovered during the initial audit:

- Train rows: 3,146
- Test rows: 1,030
- Positive rate in train: about 65.1%
- Train has 39 named locations; test has 11
- Only one location name overlaps train/test
- Exact coordinate clusters have zero train/test overlap
- Years overlap heavily between train and test
- Age is the strongest single raw predictor found in the initial audit
- Supplied `climate_features.csv` already contains CHIRPS, ERA5-Land, MODIS,
  elevation, and slope features for all train/test IDs

Because train/test geography is strongly shifted, do not trust random CV alone.
Grouped geographic validation is a required second view.

## Research direction already established

The strongest planned improvement path is:

1. robust spatial/group validation;
2. CatBoost / LightGBM / XGBoost tabular baselines;
3. interaction features, especially age × lagged climate;
4. longer climate windows such as 180-day and 365-day rainfall/temperature;
5. fold-safe calibration and ensemble selection;
6. avoid target-derived leakage from global year/location target statistics.

The March 2026 predecessor challenge appears to use the same bundle. Public
historical solutions are useful for ideas, but any target-rate encodings or
similar engineered statistics must be recomputed strictly inside folds.

## Files that define the infrastructure

Before editing orchestration, inspect these files:

```text
bootstrap_kaggle.py
climateops/cli.py
climateops/config.py
climateops/gitops.py
climateops/data.py
climateops/state.py
climateops/artifacts.py
climateops/stages.py
configs/kaggle.example.env
.gitignore
```

The main project overview is in `README.md`.

## Future ChatGPT session checklist

When resuming this project in another conversation:

1. Read `RUNBOOK.md` first.
2. Read `README.md` second.
3. Inspect `git status` and `git log -5` before making changes.
4. Do not run local model workloads on the user's PC.
5. Treat GitHub as the code source of truth.
6. Make all competition execution Kaggle-compatible.
7. Add new experiments to `climateops/stages.py` so they are resumable.
8. Use `crhp sync` rather than inventing another Git workflow.
9. Keep secrets and Zindi files out of Git.
10. Preserve compatibility with `crhp bootstrap` + `crhp resume` on a fresh VM.
