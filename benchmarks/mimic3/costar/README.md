# COSTAR MIMIC-III validation

This directory contains the evaluation code used for the five-seed MIMIC-III results in the COSTAR PR. It includes the experiment's model components, prepared-data reader, factual validation, target counterfactual evaluation, and result aggregation. It does not include a training launcher or a raw-data converter.

The experiment uses the repository's `COSTAREncoder` and `COSTARLoss`, with additional code matching the [official COSTAR implementation](https://github.com/google-research/google-research/tree/5b09c22d73a9d35eb6c5d2a99b95677a45053466/COSTAR). Its group pooling, augmentation, `2 * temperature` loss scaling, outcome head, and EMA updates during training and validation forwards differ from the public `COSTAR` method's default training path. The public method updates EMA after optimizer steps.

## Rebuild the results table

From the repository root, install the package and run:

```bash
python -m pip install -e ".[dev]"
python -m benchmarks.mimic3.costar.validate summarize --evaluation benchmarks/mimic3/costar/reference_metrics.json
```

`reference_metrics.json` contains only the saved aggregate squared-error sums and valid-origin counts for the 20 evaluations, plus their recorded six-horizon means. It contains no patient records, patient identifiers, data paths, or model weights. The command recomputes the table without model inference or access to MIMIC data. Use `--output summary.json` to save full-precision aggregate and per-seed results to a new file.

| Target-domain evaluation | Without SSL | With SSL pretraining | Relative RMSE change |
|---|---:|---:|---:|
| Zero-shot | 0.52925 ± 0.08948 | 0.52479 ± 0.09116 | −0.842% |
| After target fine-tuning | 0.52540 ± 0.09206 | 0.52423 ± 0.08975 | −0.223% |

Seeds are 17, 43, 44, 91, and 95. For each seed and horizon, the metric is `sqrt((SSE_y1 + SSE_y2) / valid_origin_count)` in standardized outcome units. It sums the two outcome errors without dividing by two or multiplying by 100, then takes the arithmetic mean over horizons 1–6. The table gives the mean and sample standard deviation (`ddof=1`) across the five seeds. Relative change is `100 * (SSL mean / without-SSL mean - 1)`. All eleven horizons are retained in the saved statistics.

Source and target supervision used at most 100 epochs with patience 20. The recorded SSL configuration allowed 200 epochs, but its five runs first reached patience 20 at epochs 79, 54, 62, 37, and 79. Thus all stages completed within 100 epochs. Reducing the SSL cap to 100 would not change those completed trajectories: the learning rate, EMA coefficient, random streams, and checkpoint selection do not depend on the cap. This does not claim that the original SSL configuration was 100 or that a different runtime will reproduce identical weights.

These are descriptive results on previously examined test populations. The published simulation seeds can share clinical patients. The results do not establish statistical significance.

## Validate an existing experiment

The commands below require a local copy of the recorded run's prepared arrays, reports, certificates, and checkpoints. Obtain the [official pre-generated data](https://zenodo.org/record/8412226/files/all_needed_data.tar.gz) through its provider. Raw archive files are not accepted directly by this validator; conversion to the schema checked in `data.py` is a separate step. No data or checkpoints are distributed here.

```bash
python -m benchmarks.mimic3.costar.validate check --run-dir /path/to/run --data-catalog /path/to/catalog.json
python -m benchmarks.mimic3.costar.validate evaluate --run-dir /path/to/run --data-catalog /path/to/catalog.json --device cuda
```

`check` verifies the existing artifact certificate, the five SSL stopping histories, the 20 supervised checkpoint selections, the full source-model transfer to target training, and the prepared-data hashes. It does not run a model. Data manifest hashes must match the certified evaluation plan; a different but internally consistent dataset is rejected. Metadata hashes provide consistency checks for trusted local artifacts, not a signature authenticating an external publisher.

`evaluate` performs all 20 factual validation replays first. Counts, squared-error sums, float32 selection losses, and batch proofs must match the saved validation records before any counterfactual test begins. It then evaluates each selected model on the fixed target test data once. Source checkpoints give zero-shot results; target checkpoints give fine-tuned results. One-step and sequential counterfactual histories use separate caches. The evaluator refuses an existing `run/evaluation` directory, including an interrupted evaluation. Use `summarize` for a completed run; do not delete the directory to repeat tests.

Validation replay uses the original strict numerical comparisons. The recorded environment was Python 3.12.14 and PyTorch 2.10.0+cu128. This code requires Python 3.11 or newer. A different device or numerical environment may fail replay even when it implements the same equations. Failed replay does not open the test gate.

## Required input layout

The evaluator reads the original artifact format:

```text
run/
  development.json
  protocol.json
  evaluation_plan.json
  verification.json
  seed17/                         # also seed43, seed44, seed91, seed95
    ssl/report.json
    ssl/curve.json
    ssl/last.pt
    full/random/source/          # also random/target, ssl/source, ssl/target
      report.json
      curve.json
      best.pt
      initial.pt
```

`development.json` contains one record per seed with `ssl` and `budgets.full.{random,ssl}.{source,target}` stage reports. Supervised reports retain the selected validation statistics, selection epoch, and checkpoint/curve SHA-256 hashes. The recorded verification certificate binds `development.json`, `protocol.json`, and `evaluation_plan.json`. Checkpoints are loaded with `weights_only=True`; complete encoder-plus-head state is required.

The catalog uses schema version 1, lists all five published seeds, and maps each seed to a prepared array directory, its manifest, and the manifest SHA-256:

```json
{
  "schema_version": 1,
  "published_seeds": [17, 43, 44, 91, 95],
  "cohorts": {
    "17": {
      "data_dir": "arrays_seed17",
      "data_manifest": "arrays_seed17/data_manifest.json",
      "data_manifest_sha256": "<SHA-256 of the unchanged manifest>"
    }
  }
}
```

The example omits the other four cohort entries for brevity; all five are required. Relative paths resolve from the catalog's directory. Absolute paths are also accepted. Checkpoint paths are resolved from `--run-dir`, so the run directory can be moved without rewriting certified reports. Catalog paths may be updated for relocation, but certified manifest and array contents must remain unchanged.

Each manifest specifies dimensions, source-derived outcome normalization, disjoint patient splits, the 10% target-training subset, and `array_groups`. The groups contain `.npy` records with `path`, `size_bytes`, `sha256`, `shape`, and `dtype`. The reader preserves the released cached-next-covariate alignment: `X_(j+1)`, previous treatments `A_(j-1)`, and observed outcomes `Y_j`. `data.py` defines the complete required schema and checks it before loading numerical arrays with `allow_pickle=False`.

## Tests

```bash
python -m pytest benchmarks/mimic3/costar/tests -q
```

Tests use synthetic inputs. They cover numerical model behavior, temporal causality, complete checkpoint transfer, data-path and hash checks, validation-selected checkpoints, test gating, refusal of repeated evaluation, and reconstruction of metrics from SSE/counts. They do not run the patient dataset or repeat the recorded counterfactual tests.
