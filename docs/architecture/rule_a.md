// docs/architecture/rule_a.md

# Rule A: Canonical Run Artifacts Specification

Rule A defines a **single, stable, scalable** artifact layout for every run in GenCyberSynth.

A “run” is the atomic unit of reproducibility:
- one dataset
- one model family + variant
- one config (resolved + frozen)
- one seed / run_id
- outputs: checkpoints, synthetic samples, metrics, eval summaries, plots

This spec exists so we can:
- add many datasets without changing code
- add many model families/variants without changing evaluation/reporting
- re_run a historical experiment and get the same structure
- aggregate across runs reliably

---

## 1) Design Goals

### G1 — Dataset_scalable
Artifacts MUST be namespaced by dataset identity (not just “data/” vs “USTC/”).
Dataset identity must be stable across machines.

### G2 — Run_scalable
Each run MUST have an isolated root folder keyed by a stable run id.
Multiple runs must coexist without collisions.

### G3 — Model_scalable
Model families/variants MUST be encoded in the artifact path so we can:
- filter runs by family
- group by variant
- compare families across the same dataset

### G4 — Orchestrator_first
The orchestrator owns the artifact root. Individual models never invent paths.
Model code receives a `RunPaths`/`RunIO` object (or a resolved root path) and writes inside it.

### G5 — Schema_backed
Canonical outputs are validated against JSON schemas:
- run_manifest.schema.json
- run_events.schema.jsonl
- eval_summary.schema.json
- dataset_fingerprint.schema.json
- dataset_registry.schema.json

---

## 2) Key Concepts

### 2.1 Dataset ID
A short stable string (example: `ustc_tfc2016_npy`, `mnist`, `custom_folder_v1`).

Dataset ID must map to:
- dataset spec (shape, num_classes, splits)
- dataset fingerprint (hashes of raw files/splits)

Dataset adapters are responsible for producing `DatasetSpec` and fingerprints.

### 2.2 Run ID
A stable identifier derived from (at minimum):
- dataset fingerprint id (or dataset_id + fingerprint hash)
- family + variant
- resolved config hash
- seed
- (optional) code version hash

The run id must be deterministic when inputs are identical.

### 2.3 Family + Variant
- family: high_level model family (gan, vae, diffusion, autoregressive, ...)
- variant: the sub_implementation (dcgan, c_vae, c_ddpm, c_rbm_bernoulli, ...)

Adapters are responsible for mapping (family, variant) to the correct implementation.

---

## 3) Canonical Artifact Root

Rule A defines the run root as:

<ARTIFACTS_ROOT>/
datasets/<dataset_id>/
runs/<run_id>/


Where:
- `ARTIFACTS_ROOT` is configured at runtime (default `artifacts/`)
- `dataset_id` is from the dataset registry
- `run_id` is produced by the orchestrator (deterministic)

### 3.1 Canonical Tree Under a Run

datasets/<dataset_id>/
runs/<run_id>/
run_manifest.json
run_events.jsonl

resolved_config.yaml
config_hash.txt
code_hash.txt                  # optional but recommended
environment.txt                # optional (pip freeze, conda env export)

checkpoints/
  ... model checkpoints ...

synthetic/
  index.json                   # manifest of synthetic items (paths + labels)
  npy/
    x_synth.npy
    y_synth.npy
    gen_class_<k>.npy          # optional (traceability)
    labels_class_<k>.npy       # optional
  png/
    <class>/<seed>/....png     # optional (visual gallery mode)

metrics/
  features/
    real_features.npz
    synth_features.npz
  distribution/
    fid.json
    kid.json
    mmd.json
    js_kl.json
  diversity/
    duplicates.json
    coverage.json
  calibration/
    ece.json
    brier.json
  privacy/
    nn_distance.json           # optional

eval/
  eval_summary.json            # schema: eval_summary.schema.json
  downstream/
    classifier_metrics.json
    confusion_matrix.npy
    ...

reporting/
  plots/
    summary.png
    training_curves.png
    ...
  tables/
    metrics.csv
    ...

_tmp/ 


---

## 4) Canonical Files (Required vs Optional)

### 4.1 Required Files
Every successful run MUST produce:

1) `run_manifest.json`
- what was run
- what dataset
- what model variant
- what config hash
- pointers to outputs

2) `run_events.jsonl`
- event stream with timestamps
- stage boundaries (train start/end, synth start/end, eval start/end)
- metrics emissions (optional to include in events, but recommended)

3) `resolved_config.yaml`
- final config after merging defaults + dataset + family + variant + CLI overrides
- this is the exact config that produced artifacts

4) `synthetic/index.json`
- canonical index of synthetic outputs:
  - file path
  - label
  - seed
  - any per_item metadata

5) `eval/eval_summary.json`
- canonical summary with:
  - dataset id/fingerprint id
  - run id
  - primary metrics
  - downstream utility metrics
  - calibration metrics if enabled

### 4.2 Optional But Recommended
- `code_hash.txt`
- `environment.txt`
- `metrics/features/*.npz` (for re_computing distribution metrics without re_encoding)
- `reporting/plots/*`

---

## 5) Path Authority (Who Creates Paths?)

Rule A requires a strict authority boundary:

### 5.1 Orchestrator / RunIO is the authority
Only orchestrator code may compute canonical paths:
- `RunPaths` / `RunIO` (thin wrappers over `gencysynth.utils.paths`)
- `gencysynth.adapters.run_io` should expose helpers like:
  - `run_root(artifacts_root, dataset_id, run_id)`
  - `checkpoints_dir(...)`
  - `synthetic_dir(...)`
  - `metrics_dir(...)`
  - `eval_dir(...)`
  - `reporting_dir(...)`

### 5.2 Model implementations do not invent folder structures
Model code should accept:
- an explicit `output_root` for synthesis
- a `RunPaths` object (or `paths` dict) during train/synth/eval

If a model wants to save:
- it uses `paths.checkpoints_dir` etc.
- never hardcodes `artifacts/<family>/...`

---

## 6) Config Rules Under Rule A

### 6.1 Config layering
Resolved config comes from ordered merges:
1) `configs/base.yaml`
2) `configs/datasets/<dataset>.yaml`
3) `configs/families/<family>.yaml`
4) `.../variants/<variant>/defaults.yaml`
5) suite_level override (configs/suites/*.yaml)
6) CLI overrides

### 6.2 Resolved config must be persisted
The orchestrator must write:
- `resolved_config.yaml`
- `config_hash.txt`

The hash must be computed from a canonical serialization (stable YAML or JSON).

---

## 7) Synthetic Output Contract

All synthetic outputs must be evaluatable uniformly:

### 7.1 Minimal contract (arrays)
- `x_synth.npy`: float32, shape (N, H, W, C)
- `y_synth.npy`: either
  - int labels shape (N,), or
  - one_hot labels shape (N, K)
- values must be one of:
  - [0, 1] for image_space metrics and saving PNGs
  - [-1, 1] is allowed internally but must be converted before writing

### 7.2 PNG contract (optional)
When enabled (gallery mode), PNGs are stored under:
- `synthetic/png/<class>/<seed>/...png`
and indexed in `synthetic/index.json`.

---

## 8) Validation & Compliance

Rule A compliance is verified by:
- schema validation of manifest/events/summary
- sanity checks (shape checks, basic stats)
- smoke tests (short training) that assert outputs land in canonical locations

See:
- `tools/smoke/smoke_onepass.py`
- `tools/validate/validate_artifacts.py`

---

## 9) Rule A Checklist (for new model family/variant)

When adding a new model:
- [ ] Add adapter registration (family, variant) -> implementation
- [ ] Ensure train saves checkpoints under `run_root/checkpoints/`
- [ ] Ensure synth writes under `run_root/synthetic/` and produces `index.json`
- [ ] Ensure evaluation reads from synthetic contract and writes `eval_summary.json`
- [ ] Ensure `run_manifest.json` and `run_events.jsonl` are written
- [ ] Pass smoke test for <5 epochs

---

## 10) Non_goals

Rule A does NOT require:
- the same checkpoint format for every model
- PNGs for every run
- plotting in every run

Rule A DOES require:
- canonical run root
- canonical minimal artifacts (manifest + events + summary)
- reproducible resolved config + hashes# optional: scratch space (safe to delete)
