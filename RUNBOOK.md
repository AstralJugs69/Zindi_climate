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
crhp run apc-prior-validate
crhp run diagnose-shift
crhp run ablate
crhp run robust-validate
crhp run robust-candidates
crhp run structured-validate
crhp run teleconnection-validate
crhp run power-validate
crhp run chirps-validate
crhp run cohort-candidates
crhp run demographic-validate
crhp run fine-demo-validate
crhp run interaction-validate
crhp run interaction-select
crhp run interaction-candidates
crhp run low-shift-select
crhp run lagged-climate-validate
crhp run profile-validate
crhp run age-expert-validate
crhp run temporal-density-validate
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

- `apc-prior-validate`: fold-safe continuous empirical-Bayes age/period/cohort risk
  surfaces. Unlike the rejected categorical target encodings, these priors use
  Gaussian smoothing over age/year/birth-cohort and are blended as a separate OOF
  probability component rather than injected into CatBoost. Validation locations are
  completely excluded when each prior is built.
- `teleconnection-validate`: climate-only NOAA Indian Ocean Dipole (DMI) and Nino 3.4
  experiment motivated by East African malaria literature. It tests monthly DMI/ENSO
  lags and 3-month means on the low-shift cohort model and reports adversarial
  train/test AUC. NOAA series are cached under `reports/teleconnection_validation/`.

- `diagnose-shift`: adversarial train-vs-test validation, including progressively
  less spatial information, to identify distribution shift before leaderboard use.
- `ablate`: location-group CV across full, no-location, no-spatial, no-interaction,
  no-climate, and demographics/time-only feature views.
- `robust-validate`: repeated location-group CV across multiple split seeds for the
  two preferred low-shift feature spaces (`demographics_time` and `no_spatial`),
  comparing CatBoost, LightGBM, XGBoost, Logistic Regression, and coarse blends.
- `robust-candidates`: generate CV-bagged test predictions using the same 3x5
  location-group split design as `robust-validate`. The primary file is pure
  demographics/time CatBoost because it has both the best robust score and the
  lowest train/test shift. A 50/50 CatBoost + no-spatial Logistic Regression blend
  and a 75/25 CatBoost/XGBoost diversity file are also written. Use this stage
  instead of the older `candidates` stage for leaderboard-ready files.
- `structured-validate`: repeated location-group validation of strictly fold-safe
  smoothed target priors for year, age, season and demographic interactions, with
  optional distance-weighted target priors from *other* training locations. Fit-row
  target encodings are leave-one-out; spatial priors exclude the row's whole own
  location so validation still simulates unseen test locations.
- `power-validate`: the current external-signal branch. Kaggle downloads and caches
  NASA POWER daily meteorology for the competition's half-degree climate cells,
  builds 14/30/56/84/180/365-day temperature, rainfall, humidity, wind, wet-bulb,
  solar and pressure histories plus relative anomalies, and evaluates both repeated
  location-group target CV and adversarial train/test shift. Generated API cache and
  feature tables live under `reports/power_climate/` and are not committed to Git.
- `chirps-validate`: high-resolution rainfall branch using the public ClimateSERV
  CHIRPS API. It caches one long daily rainfall series per unique competition
  coordinate, then derives 7/14/30/56/84/90/120/180/365-day accumulation/intensity,
  wet/heavy-rain fractions, recent-vs-background ratios, and matched prior-year
  anomalies. It compares relative-only versus all CHIRPS features using repeated
  location-group CV and train/test adversarial AUC. Cache/output lives under
  `reports/chirps/` and is intentionally ignored by Git.
- `cohort-candidates`: CV-bagged leaderboard generator for the low-shift cohort branch.
  It trains the `all_ctr1` reference and `cohort_calendar` feature sets over the same
  3x5 location-group folds, then writes 25/75, 50/50, and 75/25 probability blends.
  All submissions keep the fixed 0.5 classification threshold required by Zindi.
- `demographic-validate`: low-shift model-shape search on the demographics/time view.
  It compares the existing CatBoost reference with spline-regularized logistic
  regression, ExtraTrees, histogram gradient boosting, and a deliberately smooth
  100-neighbour classifier, all under the same repeated location-group folds, then
  evaluates coarse probability blends. Use this after `power-validate` if external
  climate does not beat the low-shift CatBoost baseline.
- `fine-demo-validate`: low-shift refinement of the successful interaction branch.
  It keeps all rows in one CatBoost model while adding fine age categories,
  approximate birth cohort, categorical calendar/season fields, and selected
  age/cohort × zone/year/season interactions. No labels, location names, or
  coordinates are used to construct these features; repeated location-group CV and
  adversarial train/test AUC are reported for every configuration.
- `interaction-select`: zero-retraining selection step that reads the OOF files from
  `interaction-validate` and evaluates coarse blends among base, categorical,
  numeric, and all-interaction CatBoost configurations. Run this before creating a
  new leaderboard candidate so blend weights are chosen from OOF evidence.
- `interaction-candidates`: leaderboard candidate generator for the selected
  interaction branch. It CV-bags categorical, numeric, and all-interaction CatBoost
  components over the same 3x5 location-group folds and writes the selected 75/25
  categorical/numeric blend, the higher-AUC 50/50 categorical/all blend, and the
  low-shift `all_ctr1` reference submission.
- `low-shift-select`: zero-retraining cross-family OOF blend search across the
  successful interaction and fine-demographic branches. It evaluates coarse pairwise
  blends plus a small pre-declared set of three-way blends centered on `all_ctr1`.
  Run this after both `interaction-validate` and `fine-demo-validate` before creating
  another leaderboard candidate.
- `lagged-climate-validate`: literature-driven climate validation based on published
  Iganga-Mayuge HDSS malaria studies. Kaggle downloads one common site-centroid
  ERA5-Land daily series from Open-Meteo, then creates non-overlapping weekly lags
  0-12 for maximum/mean/minimum temperature and rainfall. Focused features encode
  the published 2-8 week rainfall case window, 5-11 week maximum-temperature
  mortality window, under-5 lag-8 temperature effect, 5-14-year lag-4-8 effect,
  male 5-14 effect modification, plus short-lag extreme heat/rain anomalies. Using
  one site-wide series deliberately avoids village geography as a Train/Test marker.
  Every mode is checked with repeated location-group CV and adversarial shift AUC.
- `interaction-validate`: target-free demographic interaction search. It adds
  age-band × zone/year/month, gender × age/time, age × year, and vulnerable-group ×
  seasonal interactions, and also tests CatBoost categorical-combination depth.
  It uses the same repeated location-group CV and reports train/test adversarial AUC
  so any apparent gain that comes from distribution shift is rejected.
- `profile-validate`: target-free transductive location-profile search. It represents
  unseen locations through unlabeled case-mix summaries (age, gender, season and
  record-count structure), then optionally adds collection-period summaries and
  within-location climate deviations. A full-profile variant also includes absolute
  location climate summaries. Train and test location profiles are built separately,
  no target values are used, and every configuration is checked with repeated
  location-group CV plus adversarial train/test AUC.
- `age-expert-validate`: low-shift specialist-model search on the current
  `all_ctr1` demographic interaction feature space. It trains separate CatBoost
  experts for under-5 / older, three-stage, and four-stage age partitions inside
  each location-group fold, then evaluates expert-only and global/expert blends.
  Routing uses observed age only; no target-derived routing or geography is added.
- `temporal-density-validate`: research-driven, target-free mortality-event context.
  It uses only the provided Train/Test covariates to describe how many *other* deaths
  occur near each death date over 7/14/28/56/84/168-day windows, including age/sex
  composition, recent-vs-background burst ratios, and a seasonal 28-day event z-score.
  A symmetric retrospective variant and a lean zone-context variant are tested
  separately. The row's own death is subtracted from direct count windows, no target
  values or external health records are used, and every configuration is evaluated
  with repeated location-group CV plus adversarial train/test AUC.

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
