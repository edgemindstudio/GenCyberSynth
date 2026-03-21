# src/gencysynth/cli/main.py
"""
GenCyberSynth CLI (Unified Entry Point)
======================================

This is the **single command-line entrypoint** for the GenCyberSynth dissertation repo.
It wires together three core layers:

1) **Adapters** (src/gencysynth/adapters/)
   - Responsible for *synthesis* (generate synthetic samples and write a manifest).
   - Adapters are the boundary between the "model implementation" and the "unified runner".

2) **Evaluator** (src/gencysynth/eval/)
   - Responsible for *evaluation* (compute metrics on REAL, SYNTH, REAL+SYNTH).
   - Evaluator consumes the manifest emitted by adapters.

3) **Config Loader** (YAML)
   - Loads the experiment config and (optionally) merges an overrides YAML.
   - The CLI attaches provenance metadata in cfg["run_meta"] for auditability.

Variant / Identity (Important)
------------------------------
This repo supports multiple **families** and **variants** (e.g., gan/dcgan, gan/wgan_gp).

This CLI exposes that identity explicitly via flags:

  --family   (e.g., gan, diffusion, vae)
  --variant  (e.g., dcgan, wgan_gp)  [optional for families with only one implementation]

The adapter registry should accept both patterns:
  - make_adapter("gan")              (family-level adapter/router)
  - make_adapter("gan:dcgan")        (variant-level adapter)
  - make_adapter("gan/dcgan")        (variant-level adapter)
Choose *one* canonical scheme and keep it consistent across the repo.

I/O Conventions (Paths)
-----------------------
All outputs go under the **artifacts root**, resolved by:

  1) --artifacts (CLI override)
  2) cfg["paths"]["artifacts"]
  3) "artifacts" (default)

Synthesis outputs should be written to:

  <artifacts>/<family>/<variant>/synthetic/
      manifest.json                     (shared / latest)
      <family>_<variant>_<CFG>_seed<SEED>/manifest.json    (per-run immutable copy)

Evaluation outputs are written by the evaluator under:

  <artifacts>/eval/<family>/<variant>/...                 (exact structure defined in eval.runner)

Design Goals
------------
- Lightweight imports: the CLI should start even if some model deps are missing.
- Defensive behavior: missing configs or missing adapters => clear messages, no silent failure.
- Audit-friendly: attach run_meta (config hash, git commit, caps, budgets).

"""

from __future__ import annotations
from gencysynth.cli.datasets import register_dataset_subcommands
import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from gencysynth.adapters.models.registry import register_builtin_adapters
register_builtin_adapters()

# ----------------------------
# Local imports (repo modules)
# ----------------------------
from gencysynth.adapters.registry import SKIPPED_IMPORTS, list_adapters  # legacy "global" registry diagnostics only
from gencysynth.adapters.models.registry import register_builtin_adapters, resolve_model_adapter, list_model_adapters
from gencysynth.models.registry import register_builtin_models

# Optional dependency: only required when a config path is provided.
try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None


# =============================================================================
# Logging helpers
# =============================================================================
def _info(msg: str) -> None:
    """Standard informational log."""
    print(f"[info] {msg}")


def _warn(msg: str) -> None:
    """Standard warning log."""
    print(f"[warn] {msg}")


def _err(msg: str) -> None:
    """Standard error log (stderr)."""
    print(f"[error] {msg}", file=sys.stderr)


# =============================================================================
# Config utilities
# =============================================================================
def load_config(path: Optional[str]) -> Dict[str, Any]:
    """
    Load a YAML config file into a dict.

    Behavior:
    - If `path` is None/empty -> return {}.
    - If file doesn't exist -> warn and return {} (allows CLI to run with defaults).
    - If PyYAML is missing and a path is provided -> exit with code 2.

    This function does **not** enforce schema; downstream code must be robust to missing keys.
    """
    if not path:
        return {}

    if not os.path.exists(path):
        _warn(f"Config file not found: {path} (continuing with defaults)")
        return {}

    if yaml is None:
        _err("PyYAML is required to read config files. Install with: pip install pyyaml")
        raise SystemExit(2)

    with open(path, "r") as f:
        data = yaml.safe_load(f) or {}

    if not isinstance(data, dict):
        _warn(f"Config loaded but top-level YAML is not a mapping/dict: {path}. Using defaults.")
        return {}

    return data


def deep_update(base: dict, upd: dict) -> dict:
    """
    Recursively merge upd into base (dict->dict deep merge).

    - Dict values merge recursively.
    - Non-dict values overwrite.
    """
    for k, v in (upd or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base


def artifacts_root(cfg: Dict[str, Any], override: Optional[str] = None) -> str:
    """
    Resolve artifacts root directory in priority order:
      1) explicit override (--artifacts)
      2) cfg['paths']['artifacts']
      3) 'artifacts' (default)

    NOTE: Directory creation is done by downstream components when needed.
    """
    if override:
        return override
    return str(cfg.get("paths", {}).get("artifacts", "artifacts"))


# =============================================================================
# Model identity helpers (family/variant)
# =============================================================================
def _normalize_family_variant(args: argparse.Namespace, cfg: Dict[str, Any]) -> Tuple[str, Optional[str]]:
    """
    Determine (family, variant) for the current run.

    Priority:
      1) CLI flags: --family, --variant
      2) cfg["model"]["family"], cfg["model"]["variant"]
      3) Legacy CLI: --model (treated as family)

    Returns:
      (family, variant_or_None)
    """
    # New preferred flags
    family = getattr(args, "family", None)
    variant = getattr(args, "variant", None)

    # Legacy compatibility: --model behaves like "family"
    legacy_model = getattr(args, "model", None)
    if not family and legacy_model:
        family = legacy_model

    # Config fallback
    m = cfg.get("model", {})
    if not family and isinstance(m, dict):
        if isinstance(m.get("family"), str) and m["family"]:
            family = m["family"]
    if not variant and isinstance(m, dict):
        if isinstance(m.get("variant"), str) and m["variant"]:
            variant = m["variant"]

    # Final fallback
    family = family or "gan"
    variant = variant if (isinstance(variant, str) and variant) else None

    return family, variant


def _adapter_key(family: str, variant: Optional[str]) -> str:
    """
    Canonical model_tag used across artifacts + registries.

    We use slash form:
      gan/dcgan
      gaussianmixture/gmm_diag
    """
    return f"{family}/{variant}" if variant else family


# =============================================================================
# Manifest path helpers (shared + per-run)
# =============================================================================
def _infer_config_variant(cfg: Dict[str, Any]) -> Optional[str]:
    """
    Best-effort inference of config variant letter (A/B/...) from cfg["run_meta"].

    Priority:
      1) cfg["run_meta"]["config_variant"] (preferred)
      2) cfg["run_meta"]["config_id"]      (e.g. "gan_A" -> "A")
    """
    rm = cfg.get("run_meta")
    if not isinstance(rm, dict):
        return None

    v = rm.get("config_variant")
    if isinstance(v, str) and v:
        return v

    cid = rm.get("config_id")
    if isinstance(cid, str) and "_" in cid:
        maybe = cid.split("_")[-1]
        return maybe if maybe else None

    return None


def _shared_manifest_path(arts_root: str, family: str, variant: Optional[str]) -> str:
    """
    Conventional *shared* manifest location (latest run):

      <artifacts>/<family>/<variant>/synthetic/manifest.json
      <artifacts>/<family>/synthetic/manifest.json           (if variant is None)

    This is useful for tooling that just wants "latest output for this adapter".
    """
    if variant:
        return os.path.join(arts_root, family, variant, "synthetic", "manifest.json")
    return os.path.join(arts_root, family, "synthetic", "manifest.json")


def _per_run_manifest_path(arts_root: str, family: str, variant: Optional[str], cfg: Dict[str, Any]) -> Optional[str]:
    """
    Seed/config-specific immutable manifest copy:

      <artifacts>/<family>/<variant>/synthetic/<family>_<variant>_<CFG>_seed<SEED>/manifest.json

    Returns None if CFG/SEED cannot be determined.

    Notes:
    - CFG is intended to separate "A/B" configs per family for tuning-lite.
    - SEED should be a single seed for this run (not the whole random_seeds list).
    """
    cfg_variant = _infer_config_variant(cfg)

    # Prefer cfg["SEED"], else first element in cfg["random_seeds"], else None
    seed = cfg.get("SEED")
    if seed is None:
        rs = cfg.get("random_seeds")
        if isinstance(rs, (list, tuple)) and len(rs) > 0:
            seed = rs[0]

    # Validate
    if not (isinstance(cfg_variant, str) and cfg_variant):
        return None

    try:
        seed_i = int(seed)
    except Exception:
        return None

    # Build run directory name
    if variant:
        run_dir = os.path.join(
            arts_root,
            family,
            variant,
            "synthetic",
            f"{family}_{variant}_{cfg_variant}_seed{seed_i}",
        )
    else:
        run_dir = os.path.join(
            arts_root,
            family,
            "synthetic",
            f"{family}_{cfg_variant}_seed{seed_i}",
        )

    return os.path.join(run_dir, "manifest.json")


# =============================================================================
# Provenance / audit metadata
# =============================================================================
def attach_run_meta(cfg: Dict[str, Any], args: argparse.Namespace) -> None:
    """
    Attach audit metadata to cfg so downstream summaries can prove which config was used.

    Safety:
      - Never raises outward.
      - Stores None on failures.

    Fields:
      cfg["run_meta"] = {
        "config_path": <abs path or None>,
        "config_sha1": <sha1 or None>,
        "git_commit": <commit or None>,
        "caps": {...},
        "budget_per_class": <int or None>,
      }

    IMPORTANT:
      The evaluator summary writer must copy cfg["run_meta"] into final summary JSON
      for full auditability.
    """

    def _sha1(path: str) -> Optional[str]:
        try:
            b = Path(path).expanduser().read_bytes()
            return hashlib.sha1(b).hexdigest()
        except Exception:
            return None

    def _git_commit(repo_root: str) -> Optional[str]:
        try:
            return (
                subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root)
                .decode()
                .strip()
            )
        except Exception:
            return None

    # Resolve config path (absolute)
    cfg_path: Optional[str] = None
    try:
        if getattr(args, "config", None):
            cfg_path = str(Path(args.config).expanduser().resolve())
    except Exception:
        cfg_path = None

    # src/gencysynth/cli/main.py -> parents[3] should be repo root
    #   main.py
    #   cli/
    #   gencysynth/
    #   src/
    #   <repo_root>   <-- parents[3]
    try:
        repo_root = str(Path(__file__).resolve().parents[3])
    except Exception:
        repo_root = os.getcwd()

    # Pull commonly audited knobs (best-effort)
    per_class_cap = None
    try:
        per_class_cap = cfg.get("evaluator", {}).get("per_class_cap", None)
    except Exception:
        per_class_cap = None

    budget_per_class = None
    try:
        budget_per_class = cfg.get("synth", {}).get("n_per_class", None)
    except Exception:
        budget_per_class = None

    existing = cfg.get("run_meta")
    existing = existing if isinstance(existing, dict) else {}

    rm = dict(existing)  # copy
    rm.setdefault("config_path", cfg_path)
    rm.setdefault("config_sha1", _sha1(cfg_path) if cfg_path else None)
    rm.setdefault("git_commit", _git_commit(repo_root))

    # caps always refresh (safe)
    rm["caps"] = {
        "manifest_cap_per_class": per_class_cap,
        "fid_cap_per_class": per_class_cap,
    }

    # budget: do not overwrite if already set by overrides downstream
    if rm.get("budget_per_class") is None:
        rm["budget_per_class"] = budget_per_class

    cfg["run_meta"] = rm


# =============================================================================
# Command handlers
# =============================================================================
def cmd_train(args: argparse.Namespace) -> int:
    """
    Train a model (best-effort routing).

    This CLI does not implement training. It imports the trainer module and calls:
      - main(argv)  OR
      - train(cfg)

    Trainer module resolution strategy:
      - If variant is provided, prefer:
          gencysynth.models.<family>.variants.<variant>.train
      - Else fallback:
          <family>.train   (legacy / external package style)
          gencysynth.models.<family>.train  (if you later add family-level trainers)
    """
    cfg = load_config(args.config)

    if args.overrides:
        ov = load_config(args.overrides)
        deep_update(cfg, ov)

    # Resolve artifacts root override early so all downstream code writes consistently
    cfg.setdefault("paths", {})
    if args.artifacts:
        cfg["paths"]["artifacts"] = args.artifacts

    attach_run_meta(cfg, args)

    family, variant = _normalize_family_variant(args, cfg)

    _info(f"Train family : {family}")
    _info(f"Train variant: {variant or '<none>'}")
    _info(f"Config       : {args.config or '<defaults>'}")
    _info(f"Artifacts    : {artifacts_root(cfg, args.artifacts)}")

    # Prefer new structured trainer location
    candidate_modules = []
    if variant:
        candidate_modules.append(f"gencysynth.models.{family}.variants.{variant}.train")
    candidate_modules.append(f"{family}.train")  # legacy fallback
    candidate_modules.append(f"gencysynth.models.{family}.train")  # optional future path

    mod = None
    for module_name in candidate_modules:
        try:
            mod = __import__(module_name, fromlist=["*"])
            _info(f"Using trainer module: {module_name}")
            break
        except Exception:
            continue

    if mod is None:
        _warn(
            "No trainer module found. Tried:\n  - " + "\n  - ".join(candidate_modules)
        )
        _info(
            "Tip: add train.py for this family/variant (expose `main(argv)` or `train(config)`), "
            "or skip training and just run synth/eval."
        )
        return 1

    has_main = hasattr(mod, "main") and callable(getattr(mod, "main"))
    has_train = hasattr(mod, "train") and callable(getattr(mod, "train"))
    if not (has_main or has_train):
        _warn("Trainer module has no callable main()/train(). Nothing to do.")
        return 1

    # Prefer main(argv) if available
    if has_main:
        try:
            _info(f"Calling trainer.main(['--config', '{args.config}'])")
            ret = mod.main(["--config", args.config])  # type: ignore[attr-defined]
            return int(ret) if isinstance(ret, int) else 0
        except TypeError:
            # Some trainers accept dict config instead of argv
            try:
                _info("Trainer main() signature mismatch. Falling back to main(config_dict).")
                ret = mod.main(cfg)  # type: ignore[attr-defined]
                return int(ret) if isinstance(ret, int) else 0
            except Exception as e:
                _err(f"Training failed: {type(e).__name__}: {e}")
                return 1
        except Exception as e:
            _err(f"Training failed: {type(e).__name__}: {e}")
            return 1

    # Else train(cfg)
    try:
        _info("Calling trainer.train(config_dict)")
        ret = mod.train(cfg)  # type: ignore[attr-defined]
        return int(ret) if isinstance(ret, int) else 0
    except TypeError:
        # Some trainers accept argv-like input
        try:
            _info(f"Trainer train() signature mismatch. Falling back to train(['--config','{args.config}']).")
            ret = mod.train(["--config", args.config])  # type: ignore[attr-defined]
            return int(ret) if isinstance(ret, int) else 0
        except Exception as e:
            _err(f"Training failed: {type(e).__name__}: {e}")
            return 1
    except Exception as e:
        _err(f"Training failed: {type(e).__name__}: {e}")
        return 1


def cmd_synth(args: argparse.Namespace) -> int:
    """
    Synthesize (generate) synthetic samples using the *model adapter registry*.

    Flow:
      1) Load config (+ optional overrides)
      2) Resolve (family, variant) identity -> model_tag "family/variant"
      3) Load dataset arrays via dataset registry
      4) Convert arrays -> DatasetSplits (Rule A contract)
      5) Resolve model adapter and run adapter.synthesize(...)
    """
    cfg = load_config(args.config)

    if args.overrides:
        ov = load_config(args.overrides)
        deep_update(cfg, ov)

    cfg.setdefault("paths", {})
    if args.artifacts:
        cfg["paths"]["artifacts"] = args.artifacts

    attach_run_meta(cfg, args)

    # ---- identity ----
    family, variant = _normalize_family_variant(args, cfg)
    if not variant:
        _err("cmd_synth requires --variant (family-only adapters are not supported in the model-adapter path).")
        return 2

    model_tag = _adapter_key(family, variant)  # "family/variant"

    _info(f"Adapter family : {family}")
    _info(f"Adapter variant: {variant}")
    _info(f"Model tag      : {model_tag}")
    _info(f"Config         : {args.config or '<defaults>'}")
    _info(f"Artifacts      : {artifacts_root(cfg, args.artifacts)}")

    # ---- dataset load ----
    from gencysynth.data.datasets.registry import make_dataset_from_config
    from gencysynth.adapters.datasets.splits import DatasetSplits, SplitArrays

    ds = make_dataset_from_config(cfg)
    arrays = ds.load_arrays(cfg)  # DatasetArrays

    # ---- build DatasetSplits ----
    # Your DatasetArrays uses x_train/y_train/x_val/y_val/x_test/y_test.
    # y may be ints (N,) or one-hot (N,K). We normalize to BOTH.
    def _to_int_and_onehot(y, K: int):
        import numpy as np
        y = np.asarray(y)
        if y.ndim == 2 and y.shape[1] == K:
            y_onehot = y.astype("float32", copy=False)
            y_int = y_onehot.argmax(axis=1).astype("int64", copy=False)
            return y_int, y_onehot
        # assume integer labels
        y_int = y.astype("int64", copy=False)
        y_onehot = np.zeros((len(y_int), K), dtype="float32")
        y_onehot[np.arange(len(y_int)), y_int.astype("int64")] = 1.0
        return y_int, y_onehot

    dataset_id = cfg.get("dataset", {}).get("id", "unknown_dataset")
    K = int(cfg.get("dataset", {}).get("num_classes", 9))
    seed = int(cfg.get("SEED", 42))
    bpc = int(cfg.get("synth", {}).get("n_per_class", 10))
    run_id = f"smoke_seed{seed}_pc{bpc}"

    import numpy as np
    x_train = np.asarray(arrays.x_train).astype("float32", copy=False)
    y_train_int, y_train_1h = _to_int_and_onehot(arrays.y_train, K)

    train = SplitArrays(x01=x_train, y_int=y_train_int, y_onehot=y_train_1h)

    val = None
    if getattr(arrays, "x_val", None) is not None and getattr(arrays, "y_val", None) is not None:
        x_val = np.asarray(arrays.x_val).astype("float32", copy=False)
        y_val_int, y_val_1h = _to_int_and_onehot(arrays.y_val, K)
        val = SplitArrays(x01=x_val, y_int=y_val_int, y_onehot=y_val_1h)

    test = None
    if getattr(arrays, "x_test", None) is not None and getattr(arrays, "y_test", None) is not None:
        x_test = np.asarray(arrays.x_test).astype("float32", copy=False)
        y_test_int, y_test_1h = _to_int_and_onehot(arrays.y_test, K)
        test = SplitArrays(x01=x_test, y_int=y_test_int, y_onehot=y_test_1h)

    splits = DatasetSplits(train=train, val=val, test=test)

    # ---- resolve model adapter ----
    try:
        adapter = resolve_model_adapter(family=family, variant=variant)
    except Exception as e:
        _err(f"Model adapter not found for {family}/{variant}: {type(e).__name__}: {e}")
        _info(f"Known model adapters: {', '.join(list_model_adapters())}")
        return 2

    # ---- run_ctx (minimal) ----
    # Adapters expect run_ctx to exist; we provide the fields commonly used.
    from types import SimpleNamespace
    try:
        from gencysynth.adapters.run_io import RunIO
        io = RunIO(artifacts_root=artifacts_root(cfg, args.artifacts), dataset_id=dataset_id, model_tag=model_tag, run_id=run_id)
    except Exception:
        io = None  # adapters that don't require io will still work

    run_ctx = SimpleNamespace(
        artifacts_root=artifacts_root(cfg, args.artifacts),
        dataset_id=dataset_id,
        model_tag=model_tag,
        run_id=run_id,
        seed=seed,
        io=io,
        log=lambda stage, msg: print(f"[{stage}] {msg}"),
    )

    # Record key provenance
    cfg.setdefault("run_meta", {})
    cfg["run_meta"].update({
        "dataset_id": dataset_id,
        "model_tag": model_tag,
        "run_id": run_id,
        "seed": seed,
        "budget_per_class": bpc,
    })

    # ---- synthesize ----
    _info(f"Running synth via model adapter: {family}/{variant}")
    _ = adapter.synthesize(run_ctx, cfg, splits)

    _info("Synthesis complete.")
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    """
    Evaluate a model using the evaluator runner.

    Assumptions:
      - If --no-synth is NOT set, you should have already run `synth` so that:
          <artifacts>/<family>/<variant>/synthetic/manifest.json
        exists (or the adapter/evaluator handles missing manifests gracefully).

    This is the stage where summary JSON files are written by eval.runner.

    Audit requirement:
      Ensure eval summary writing copies cfg["run_meta"] for provenance.
    """
    from gencysynth.eval.runner import evaluate_model_suite
    cfg = load_config(args.config)

    if args.overrides:
        ov = load_config(args.overrides)
        deep_update(cfg, ov)

    cfg.setdefault("paths", {})
    if args.artifacts:
        cfg["paths"]["artifacts"] = args.artifacts

    attach_run_meta(cfg, args)

    family, variant = _normalize_family_variant(args, cfg)
    akey = _adapter_key(family, variant)

    _info(f"Evaluate family : {family}")
    _info(f"Evaluate variant: {variant or '<none>'}")
    _info(f"Adapter key     : {akey}")
    _info(f"Config          : {args.config or '<defaults>'}")
    _info(f"Artifacts       : {artifacts_root(cfg, args.artifacts)}")
    _info(f"No-synth flag   : {args.no_synth}")

    try:
        # NOTE: evaluate_model_suite currently takes a single "model_name".
        # We pass the adapter key so evaluator can find correct manifests and paths.
        evaluate_model_suite(cfg, model_name=akey, no_synth=args.no_synth)
    except FileNotFoundError as e:
        _err(str(e))
        return 2
    except Exception as e:
        _err(f"Evaluation failed: {type(e).__name__}: {e}")
        return 1

    return 0


def cmd_list(_: argparse.Namespace) -> int:
    """List registered adapters (global + model-adapters) and any skipped imports."""
    # Global adapter registry (older path)
    names = list_adapters()
    if not names:
        _info("No global adapters registered.")
    else:
        _info("Registered global adapters:")
        for n in names:
            print(f"  - {n}")

    # Model adapter registry (family/variant) — this is what we use now
    try:
        from gencysynth.adapters.models.registry import list_model_adapters, register_builtin_adapters

        register_builtin_adapters()
        m = list_model_adapters()
        if not m:
            _info("No model adapters registered.")
        else:
            _info("Registered model adapters (family/variant):")
            for x in m:
                print(f"  - {x}")
    except Exception as e:
        _warn(f"Could not list model adapters: {type(e).__name__}: {e}")

    if SKIPPED_IMPORTS:
        _warn("Adapters skipped during import (non-fatal):")
        for k, v in SKIPPED_IMPORTS.items():
            print(f"  * {k}: {v}")

    return 0


# =============================================================================
# CLI parser
# =============================================================================
def build_parser() -> argparse.ArgumentParser:
    """
    Build the CLI parser + subcommands.

    Subcommands:
      - train : routes into a trainer module (family/variant aware)
      - synth : synthesis via adapter registry (family/variant aware)
      - eval  : evaluation via evaluator runner (family/variant aware)
      - list  : list registered adapters
    """
    p = argparse.ArgumentParser(
        prog="gencs",
        description="GenCyberSynth – unified CLI for training, synthesis & evaluation",
    )
    sub = p.add_subparsers(dest="cmd", required=True)
    register_dataset_subcommands(sub)

    # Shared args: we support both legacy --model and new --family/--variant
    def add_identity_args(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--family", default=None, help="Model family (gan, diffusion, vae, ...)")
        sp.add_argument("--variant", default=None, help="Model variant (dcgan, wgan_gp, ...)")

        # Legacy compatibility: --model behaves like --family
        sp.add_argument("--model", default=None, help="(legacy) Same as --family")

    # train
    p_t = sub.add_parser("train", help="Train (routes into trainer module if available)")
    p_t.add_argument("--overrides", default=None, help="Path to a YAML overrides file")
    add_identity_args(p_t)
    p_t.add_argument("--config", default="configs/config.yaml", help="Path to YAML config")
    p_t.add_argument("--artifacts", default=None, help="Override artifacts root directory")
    p_t.set_defaults(func=cmd_train)

    # synth
    p_s = sub.add_parser("synth", help="Generate synthetic images via an adapter")
    p_s.add_argument("--overrides", default=None, help="Path to a YAML overrides file")
    add_identity_args(p_s)
    p_s.add_argument("--config", default="configs/config.yaml", help="Path to YAML config")
    p_s.add_argument("--artifacts", default=None, help="Override artifacts root directory")
    p_s.set_defaults(func=cmd_synth)

    # eval
    p_e = sub.add_parser("eval", help="Run evaluation on manifest + real data")
    p_e.add_argument("--overrides", default=None, help="Path to a YAML overrides file")
    add_identity_args(p_e)
    p_e.add_argument("--config", default="configs/config.yaml", help="Path to YAML config")
    p_e.add_argument("--artifacts", default=None, help="Override artifacts root directory")
    p_e.add_argument("--no-synth", action="store_true", help="Skip metrics that require synthetic images")
    p_e.set_defaults(func=cmd_eval)

    # list
    p_l = sub.add_parser("list", help="List registered adapters")
    p_l.set_defaults(func=cmd_list)

    return p


def main(argv: Optional[list[str]] = None) -> int:
    """
    CLI entry point.

    Returns:
      0 = success
      1 = general failure
      2 = bad config / missing resource / adapter not found
    """
    # Ensure model registry is populated, but do it at runtime (not import-time).
    register_builtin_models()
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)  # type: ignore[attr-defined]


if __name__ == "__main__":
    raise SystemExit(main())

