# Climate Risk & Health Prediction — Kaggle-first workspace

> **Operational memory:** before working on this repository, read
> [`RUNBOOK.md`](RUNBOOK.md). It documents the exact Kaggle bootstrap/update/
> resume/sync lifecycle, project invariants, and competition-specific rules.

This repository is designed so a Kaggle VM can be treated as disposable compute.
The repository is the source of truth for **code**; Kaggle Inputs provide the Zindi
files; an optional private Kaggle Dataset stores **runtime checkpoints, models,
reports, and submissions**.

No workflow depends on notebook state. The control plane is the `crhp` command.

## Architecture

```text
GitHub repository
  └─ code/configs/CLI ───────────────┐
                                     │ bootstrap/update/sync
Kaggle Notebook VM                   ▼
  /kaggle/working/climate-risk-health-prediction
      ├─ climateops/    orchestration
      ├─ src/           competition code
      ├─ data/raw/      symlinks to attached Kaggle Inputs
      ├─ models/        generated artifacts
      ├─ submissions/   generated submissions
      └─ .crhp/         runtime/resume state
                                     │
                                     │ snapshot/restore (optional)
                                     ▼
                           Private Kaggle Dataset
```

Kaggle's Python image supports notebook-side secrets through
`kaggle_secrets.UserSecretsClient`. `climateops` will look for a Kaggle Secret
named `GITHUB_TOKEN` when a private GitHub repo or push authentication is needed.
The token is passed to Git only for the individual network command and is never
written into `.git/config`.

## 1. Repository setup

The project expects a normal GitHub remote named `origin` and branch `main`.
Once this folder has been pushed to GitHub, set these Kaggle values:

- `CRHP_REPO_URL` — clean HTTPS clone URL:
  `https://github.com/AstralJugs69/Zindi_climate.git`
- `CRHP_BRANCH` — normally `main`
- `GITHUB_TOKEN` — **Kaggle Secret**, only required for a private repo or pushes
- `CRHP_ARTIFACT_DATASET` — optional private Kaggle Dataset handle such as
  `your-kaggle-user/climate-risk-health-checkpoints`

Never put a token in `CRHP_REPO_URL`.

## 2. Fresh Kaggle session: bootstrap

For a public repo, download the standalone bootstrapper from the repository's raw
GitHub URL and run it. Replace the URL once with your actual repository URL:

```bash
python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/AstralJugs69/Zindi_climate/main/bootstrap_kaggle.py','/kaggle/working/bootstrap_kaggle.py')"
python /kaggle/working/bootstrap_kaggle.py --repo-url https://github.com/AstralJugs69/Zindi_climate.git
```

For a private repository, make `bootstrap_kaggle.py` available to the notebook as
a tiny Kaggle Dataset (or paste only that bootstrap file once), attach the
`GITHUB_TOKEN` secret, then run:

```bash
python /kaggle/input/YOUR_BOOTSTRAP_DATASET/bootstrap_kaggle.py --repo-url https://github.com/YOUR_USER/YOUR_REPO.git
```

The bootstrapper performs all of the following:

1. clones the repo or pulls the latest `main`;
2. installs the repo as an editable Python package plus Kaggle requirements;
3. scans `/kaggle/input` for the six canonical competition files;
4. creates `data/raw/*` links to those read-only inputs;
5. restores the latest checkpoint when `CRHP_ARTIFACT_DATASET` is configured;
6. records runtime state in `.crhp/state.json`.

After bootstrap, change into the repo once:

```bash
cd /kaggle/working/climate-risk-health-prediction
```

## 3. Commands that govern the project

### See exactly what Kaggle has

```bash
crhp status
```

### Pull the newest code and refresh dependencies

```bash
crhp update
```

`run` and `resume` automatically update first, so a manual `update` is usually
only needed when you want the latest files without launching work.

### Re-discover attached Zindi data

```bash
crhp hydrate
```

### Run a named stage

```bash
crhp run baseline
crhp run diagnose-shift
crhp run ablate
crhp run suite
crhp run candidates
crhp run tune-catboost
```

Every `crhp run ...` does this by default:

```text
git update -> hydrate data -> mark RUNNING -> execute stage
           -> mark COMPLETED/FAILED -> snapshot if configured
```

The official stage registry lives in `climateops/stages.py`; adding a new
experiment means adding a normal Python script/module and one explicit registry
entry. This keeps notebook cells from becoming the orchestration layer.

### Resume after a Kaggle disconnect/session reset

Run the fresh-session bootstrap, then:

```bash
crhp resume
```

`resume` updates code, hydrates data, restores the latest Kaggle checkpoint, and
automatically reruns the stage whose persisted state is `running`, `failed`, or
`interrupted`. If the last stage completed cleanly, it does not rerun it.

### Save generated artifacts without touching Git

With `CRHP_ARTIFACT_DATASET` configured:

```bash
crhp snapshot --notes "catboost cv iteration 12"
```

The snapshot contains runtime state plus `models/`, `submissions/`, and
`reports/`. It intentionally excludes the competition input data and repository
source files.

Restore explicitly with:

```bash
crhp restore
```

KaggleHub supports creating a Dataset or a new Dataset version from Python, so
the same handle can act as our versioned checkpoint store.

### Commit and push code from Kaggle

```bash
crhp sync -m "add 180d rainfall features"
```

`sync` is deliberately safer than a raw `git push`:

1. pulls/rebases from `origin` first;
2. stages files allowed by `.gitignore`;
3. commits only when something changed;
4. pushes the current branch using the Kaggle `GITHUB_TOKEN` secret when needed.

To commit/push code **and** checkpoint generated artifacts:

```bash
crhp sync -m "finish calibrated blend" --snapshot
```

## 4. Data contract

`crhp hydrate` requires these filenames somewhere below `/kaggle/input`:

```text
Train.csv
Test.csv
SampleSubmission.csv
data_dictionary.csv
downloaded_climate_features_data_dictionary.csv
climate_features.csv
```

It creates canonical links below `data/raw/`, so modeling code never needs to
know the Kaggle Dataset slug or mount folder. The files are ignored by Git.

## 5. Source-of-truth rules

1. **Code/config:** Git + `crhp sync`.
2. **Competition input:** Kaggle Input Dataset + `crhp hydrate`.
3. **Models/submissions/runtime state:** private Kaggle Dataset +
   `crhp snapshot`/`restore`.
4. **Compute:** Kaggle only. Nothing in this design needs a persistent local VM.
5. **No secrets in Git:** GitHub credentials come from Kaggle Secrets.

This separation is what makes a completely new Kaggle VM recoverable using only
the bootstrap command and `crhp resume`.
