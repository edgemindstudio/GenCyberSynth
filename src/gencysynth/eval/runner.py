# src/gencysynth/eval/runner.py
"""
GenCyberSynth – Evaluation Runner (dataset_scalable)
===================================================

What this runner does
---------------------
Evaluate a *single run* (one model_tag + one run_id) by:
  1) Locating the synthetic manifest that describes generated samples
  2) Loading synthetic images (local robust loader)
  3) Computing lightweight "synthetic_only" metrics (KID/cFID/etc. only if available;
     MS_SSIM fallback always possible)
  4) Writing run_scoped evaluation outputs under a dataset_aware artifacts layout.

Why this file was refactored
----------------------------
The legacy runner assumed a single dataset and stored outputs under:
    artifacts/<model_name>/synthetic/...
    artifacts/<model_name>/summaries/...

GenCyberSynth now supports multiple datasets and needs scalable, collision_free paths.
This runner therefore writes to:

    artifacts/eval/<dataset_id>/<model_tag>/<run_id>/
        summary_<timestamp>.json   (optional historical snapshots)
        latest.json                (most recent snapshot for this run)
        summary.jsonl              (optional append log for HPC aggregation)

And resolves manifests preferably from:
    config["run_meta"]["manifest_path"]  (truth; best for audit)

Or (standard per_run convention):
    artifacts/runs/<dataset_id>/<model_tag>/<run_id>/manifest.json

With legacy fallback:
    artifacts/<model_name>/synthetic/manifest.json

Inputs expected in config
-------------------------
paths:
  artifacts: "artifacts"

dataset:
  id: "USTC_TFC2016_40x40_gray"     # REQUIRED for scalable layout

run_meta (injected by CLI/orchestrator; recommended):
  model_tag: "gan/dcgan"
  config_variant: "A"
  config_id: "gan_A"
  manifest_path: ".../manifest.json"
  budget_per_class: 1000
  ...

Seed:
  SEED: 42

Notes
-----
- This runner intentionally does *not* hard_require TensorFlow or any old gcs_core code.
- It is robust: missing optional metrics do not crash evaluation; they become warnings.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import os

# --- New structure utilities (single source of truth for IO/paths/repro/run_id)
from gencysynth.utils.io import read_json, write_json, write_text, append_jsonl
from gencysynth.utils.paths import (
    ensure_dir,
    resolve_eval_paths,
    resolve_run_manifest_paths,
)
from gencysynth.utils.reproducibility import now_iso
from gencysynth.utils.run_id import make_run_id


# =============================================================================
# Warnings (collected, persisted in output)
# =============================================================================
_WARNINGS: List[str] = []


# =============================================================================
# Config helpers
# =============================================================================
def _cfg_get(cfg: Dict[str, Any], dotted: str, default: Any = None) -> Any:
    """Fetch a nested config value by dotted path, e.g. 'evaluator.per_class_cap'."""
    cur: Any = cfg
    for key in dotted.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _deep_update(base: dict, upd: dict) -> dict:
    """Recursive dict merge used by CLI overrides."""
    for k, v in (upd or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def _now_ts() -> str:
    """Timestamp used in filenames: YYYYMMDD_HHMMSS."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


# =============================================================================
# Manifest loading + normalization
# =============================================================================
def _load_manifest_local(manifest_path: str) -> Dict[str, Any]:
    """
    Load a manifest JSON and normalize the schema.

    Supported input shapes:
      - {"paths": [{"path": "...", "label": int}, ...], "per_class_counts": {...}}
      - {"samples": [{"path": "...", "label": int}, ...]}  -> normalized to "paths"
    """
    man = read_json(manifest_path)

    # Normalize common field name
    if "paths" not in man and isinstance(man.get("samples"), list):
        man["paths"] = man["samples"]

    man.setdefault("paths", [])
    man.setdefault("per_class_counts", {})
    return man


def _manifest_for_meta(manifest_path: str) -> Dict[str, Any] | None:
    """
    Load + normalize manifest then add best_effort derived fields:
      - num_fake
      - budget_per_class (min per_class count)
      - per_class_counts if missing
    """
    try:
        man = _load_manifest_local(manifest_path)
    except Exception:
        return None

    # Derive per_class_counts if missing
    try:
        pcc = man.get("per_class_counts")
        if not isinstance(pcc, dict) or len(pcc) == 0:
            counts: Dict[str, int] = {}
            for item in man.get("paths", []) or []:
                y = item.get("label", None)
                if y is None:
                    continue
                y = str(int(y))
                counts[y] = counts.get(y, 0) + 1
            man["per_class_counts"] = counts
    except Exception:
        pass

    # Derived stable fields
    try:
        paths_list = man.get("paths", [])
        man["num_fake"] = int(len(paths_list)) if isinstance(paths_list, list) else None

        pcc = man.get("per_class_counts")
        if isinstance(pcc, dict) and len(pcc) > 0:
            vals = [int(v) for v in pcc.values() if v is not None]
            man["budget_per_class"] = min(vals) if vals else None
        else:
            man["budget_per_class"] = None
    except Exception:
        pass

    return man


# =============================================================================
# Synthetic image loading (robust local fallback)
# =============================================================================
def _read_image(
    path: Path,
    *,
    min_hw: int = 11,
    target_hw: tuple[int, int] | None = (40, 40),
) -> Optional["np.ndarray"]:
    """
    Minimal image reader -> float32 HWC in [0,1].

    - min_hw=11 matters because TF SSIM/MS_SSIM uses an 11x11 window by default.
    - target_hw defaults to (40,40) for USTC_TFC2016 malware images; adjust for new datasets.
    """
    try:
        from PIL import Image  # type: ignore
        import numpy as np

        img = Image.open(path).convert("RGB")
        w, h = img.size

        if target_hw is not None:
            img = img.resize(target_hw, Image.NEAREST)
        elif min(w, h) < min_hw:
            img = img.resize((max(min_hw, w), max(min_hw, h)), Image.NEAREST)

        arr = np.asarray(img).astype("float32") / 255.0
        return arr
    except Exception:
        return None


def _load_images_local(
    manifest: Dict[str, Any],
    per_class_cap: int = 200,
    *,
    target_hw: tuple[int, int] | None = (40, 40),
) -> Tuple["np.ndarray", "np.ndarray"]:
    """
    Load up to `per_class_cap` images per class from the manifest using local IO.

    Returns:
      imgs   : (N, H, W, 3) float32 in [0,1]
      labels : (N,) int32
    """
    import numpy as np

    xs: List[np.ndarray] = []
    ys: List[int] = []
    counts: Dict[int, int] = {}

    for item in manifest.get("paths", []):
        try:
            y = int(item["label"])
            p = Path(item["path"])
        except Exception:
            continue

        if counts.get(y, 0) >= per_class_cap:
            continue

        arr = _read_image(p, target_hw=target_hw)
        if arr is None:
            continue

        xs.append(arr)
        ys.append(y)
        counts[y] = counts.get(y, 0) + 1

    if not xs:
        return (
            np.zeros((0, 0, 0, 0), dtype="float32"),
            np.zeros((0,), dtype="int32"),
        )

    return np.stack(xs, axis=0).astype("float32"), np.asarray(ys, dtype="int32")


# =============================================================================
# Robust local MS_SSIM with SSIM fallback
# =============================================================================
def _ms_ssim_intra_class_local(imgs, labels, max_pairs_per_class: int = 200) -> float | None:
    """
    Intra_class diversity proxy:
      - Higher similarity (MS_SSIM) => lower diversity
      - Lower similarity => higher diversity

    Returns:
      mean similarity in [0,1], or None if insufficient pairs.

    This is a robust fallback that works without any GenCyberSynth_specific loaders.
    """
    try:
        import numpy as np
        import tensorflow as tf
    except Exception:
        return None

    if imgs is None or getattr(imgs, "size", 0) == 0 or labels is None:
        return None

    x = imgs.astype("float32", copy=False)
    if x.max() > 1.5:
        x = x / 255.0

    if x.ndim == 3:
        x = x[..., None]
    if x.shape[-1] == 1:
        x = np.repeat(x, 3, axis=-1)

    H, W = int(x.shape[1]), int(x.shape[2])
    if min(H, W) < 11:
        x = tf.image.resize(tf.convert_to_tensor(x), [max(11, H), max(11, W)], method="nearest").numpy()
        H, W = int(x.shape[1]), int(x.shape[2])

    fs = min(11, H, W)
    if fs % 2 == 0:
        fs -= 1
    fs = max(fs, 3)

    y = np.asarray(labels).astype("int32", copy=False)
    vals: list[float] = []
    rng = np.random.default_rng(42)

    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        if len(idx) < 2:
            continue

        n_pairs = min(max_pairs_per_class, len(idx) * (len(idx) - 1) // 2)
        for _ in range(n_pairs):
            i, j = rng.choice(idx, size=2, replace=False)
            a = tf.convert_to_tensor(x[i : i + 1])
            b = tf.convert_to_tensor(x[j : j + 1])

            v: float | None = None
            try:
                v_tf = tf.image.ssim_multiscale(a, b, max_val=1.0, filter_size=fs)
                v = float(tf.reduce_mean(v_tf).numpy())
            except Exception:
                try:
                    v_tf = tf.image.ssim(a, b, max_val=1.0, filter_size=fs)
                    v = float(tf.reduce_mean(v_tf).numpy())
                except Exception:
                    v = None

            if v is not None and np.isfinite(v):
                vals.append(float(v))

    return float(np.mean(vals)) if vals else None


# =============================================================================
# Run identity + path resolution (dataset_scalable)
# =============================================================================
@dataclass(frozen=True)
class RunIdentity:
    """
    Canonical identity for a single evaluation run.
    This is what makes multi_dataset scaling work reliably.
    """
    artifacts_root: str
    dataset_id: str
    model_tag: str
    run_id: str
    seed: int


def _resolve_run_identity(config: Dict[str, Any], model_name: str) -> RunIdentity:
    """
    Resolve artifacts_root, dataset_id, model_tag, run_id and seed.

    Priority:
      - model_tag from run_meta.model_tag (preferred; variant_aware)
      - else model_tag = model_name (fallback)

      - run_id:
          1) run_meta.run_id (if provided)
          2) make_run_id(config_variant, seed, extra=?)
          3) fallback: "<model_name>_seed<seed>"
    """
    artifacts_root = str(_cfg_get(config, "paths.artifacts", "artifacts"))

    dataset_id = _cfg_get(config, "dataset.id", None)
    if not isinstance(dataset_id, str) or not dataset_id:
        # Dataset id is essential for multi_dataset scaling.
        # We do not hard_crash; we fall back to a stable placeholder.
        dataset_id = "unknown_dataset"
        _WARNINGS.append("config['dataset']['id'] missing; using dataset_id='unknown_dataset' (not recommended).")

    seed = int(config.get("SEED", config.get("seed", 0)))

    rm = config.get("run_meta") if isinstance(config.get("run_meta"), dict) else {}
    model_tag = rm.get("model_tag") or model_name
    if not isinstance(model_tag, str) or not model_tag:
        model_tag = model_name

    # Prefer explicit run_id if orchestrator provided it
    rid = rm.get("run_id")
    if isinstance(rid, str) and rid:
        run_id = rid
    else:
        cfg_variant = rm.get("config_variant") or rm.get("config_id") or "run"
        extra = None
        bpc = rm.get("budget_per_class") or config.get("budget_per_class")
        if bpc is not None:
            try:
                extra = f"pc{int(bpc)}"
            except Exception:
                extra = None
        run_id = make_run_id(str(cfg_variant), seed, extra=extra)

    return RunIdentity(
        artifacts_root=artifacts_root,
        dataset_id=str(dataset_id),
        model_tag=str(model_tag),
        run_id=str(run_id),
        seed=seed,
    )


def _resolve_manifest_path(config: Dict[str, Any], ident: RunIdentity, model_name: str) -> Optional[str]:
    """
    Locate the synthetic manifest for this run.

    Priority:
      1) config.run_meta.manifest_path (truth; best for audit)
      2) artifacts/runs/<dataset_id>/<model_tag>/<run_id>/manifest.json (standard new layout)
      3) legacy: artifacts/<model_name>/synthetic/manifest.json
    """
    rm = config.get("run_meta") if isinstance(config.get("run_meta"), dict) else {}

    # 1) explicit manifest_path (preferred)
    mp = rm.get("manifest_path")
    if isinstance(mp, str) and mp and Path(mp).exists():
        return mp

    # 2) new standard per_run manifest location
    new_paths = resolve_run_manifest_paths(
        artifacts_root=ident.artifacts_root,
        dataset_id=ident.dataset_id,
        model_tag=ident.model_tag,
        run_id=ident.run_id,
    )
    if new_paths.manifest_path.exists():
        return str(new_paths.manifest_path)

    # 3) legacy fallback (old repo layout)
    legacy = Path(ident.artifacts_root) / model_name / "synthetic" / "manifest.json"
    if legacy.exists():
        _WARNINGS.append(f"Using legacy manifest path: {legacy}")
        return str(legacy)

    return None


# =============================================================================
# Public API
# =============================================================================
def evaluate_model_suite(
    config: Dict[str, Any],
    model_name: str,
    no_synth: bool = False,
) -> Dict[str, Any]:
    """
    Evaluate a single model run, write outputs to dataset_scalable paths, return record.

    Important:
    - This runner currently focuses on synth_only metrics + bookkeeping.
    - Full Phase_1 suite that also uses REAL splits belongs in eval_common.py.
    """
    # -------------------------------------------------------------------------
    # 0) Resolve canonical run identity (dataset_aware)
    # -------------------------------------------------------------------------
    ident = _resolve_run_identity(config, model_name)

    # Persist identity into run_meta for auditability (non_destructive).
    rm = config.get("run_meta") if isinstance(config.get("run_meta"), dict) else {}
    rm.setdefault("dataset_id", ident.dataset_id)
    rm.setdefault("model_tag", ident.model_tag)
    rm.setdefault("run_id", ident.run_id)
    rm.setdefault("seed", ident.seed)
    config["run_meta"] = rm

    # -------------------------------------------------------------------------
    # 1) Resolve output locations (dataset_scalable)
    # -------------------------------------------------------------------------
    eval_paths = resolve_eval_paths(
        artifacts_root=ident.artifacts_root,
        dataset_id=ident.dataset_id,
        model_tag=ident.model_tag,
        run_id=ident.run_id,
    )
    ensure_dir(eval_paths.root_dir)

    # We keep historical snapshots for debugging / audit trails.
    snapshot_path = eval_paths.root_dir / f"summary_{_now_ts()}.json"
    latest_path = eval_paths.root_dir / "latest.json"
    jsonl_path = eval_paths.jsonl_path  # summary.jsonl
    console_path = eval_paths.console_path  # summary.txt

    # -------------------------------------------------------------------------
    # 2) Resolve manifest path
    # -------------------------------------------------------------------------
    manifest_path = _resolve_manifest_path(config, ident, model_name)
    have_synth = (not no_synth) and (manifest_path is not None) and Path(manifest_path).exists()

    if not have_synth:
        _WARNINGS.append("No synthetic manifest found (or --no_synth used); synth metrics skipped.")

    # -------------------------------------------------------------------------
    # 3) Evaluator parameters (configurable)
    # -------------------------------------------------------------------------
    per_class_cap = int(_cfg_get(config, "evaluator.per_class_cap", 200))

    # Dataset_specific resizing (scalable):
    # - default (40,40) is correct for USTC_TFC2016 malware images
    # - for future datasets, set config['dataset']['image_hw'] = [H,W]
    hw = _cfg_get(config, "dataset.image_hw", [40, 40])
    target_hw = None
    try:
        if isinstance(hw, (list, tuple)) and len(hw) == 2:
            target_hw = (int(hw[0]), int(hw[1]))
    except Exception:
        target_hw = (40, 40)

    # -------------------------------------------------------------------------
    # 4) Load manifest + synthetic images
    # -------------------------------------------------------------------------
    metrics: Dict[str, Any] = {"_warnings": list(_WARNINGS)} if _WARNINGS else {}

    man: Optional[Dict[str, Any]] = None
    imgs = labels = None

    if have_synth and manifest_path is not None:
        man = _load_manifest_local(manifest_path)

        imgs, labels = _load_images_local(
            man,
            per_class_cap=per_class_cap,
            target_hw=target_hw,
        )

        if getattr(imgs, "size", 0) == 0:
            metrics.setdefault("_warnings", []).append("No images loaded from manifest; synth metrics may be empty.")

        # MS_SSIM diversity proxy (local robust)
        try:
            mss_val = _ms_ssim_intra_class_local(imgs, labels, max_pairs_per_class=200)
            metrics["ms_ssim"] = mss_val
            if mss_val is None:
                metrics.setdefault("_warnings", []).append("MS_SSIM returned no value (insufficient pairs per class?).")
        except Exception as e:
            metrics["ms_ssim"] = None
            metrics.setdefault("_warnings", []).append(f"MS_SSIM failed: {type(e).__name__}: {e}")
    else:
        metrics["ms_ssim"] = None

    # -------------------------------------------------------------------------
    # 5) Counts (best_effort; scalable)
    # -------------------------------------------------------------------------
    num_fake = None
    budget_per_class = None
    if manifest_path and Path(manifest_path).exists():
        meta = _manifest_for_meta(manifest_path)
        if isinstance(meta, dict):
            num_fake = meta.get("num_fake")
            budget_per_class = meta.get("budget_per_class")

    # Allow run_meta override (orchestrator truth)
    rm = config.get("run_meta") if isinstance(config.get("run_meta"), dict) else {}
    if rm.get("num_fake") is not None:
        try:
            num_fake = int(rm["num_fake"])
        except Exception:
            pass
    if rm.get("budget_per_class") is not None:
        try:
            budget_per_class = int(rm["budget_per_class"])
        except Exception:
            pass

    counts_map = {
        "train_real": rm.get("num_real"),   # runner does not infer real counts reliably; orchestrator should set this
        "synthetic": num_fake,
    }

    # -------------------------------------------------------------------------
    # 6) Assemble a clean record (schema_friendly + legacy shims)
    # -------------------------------------------------------------------------
    rec: Dict[str, Any] = {
        "timestamp": now_iso(),
        "dataset_id": ident.dataset_id,
        "model": model_name,
        "model_tag": ident.model_tag,
        "seed": ident.seed,
        "run_id": ident.run_id,
        "manifest_path": manifest_path,

        "images": {
            "train_real": counts_map.get("train_real"),
            "synthetic": counts_map.get("synthetic"),
        },

        "generative": {
            # This runner currently provides MS_SSIM only (robust local).
            # Full FID/cFID/KID suite that uses REAL splits is owned by eval_common.py.
            "ms_ssim": metrics.get("ms_ssim"),
            "fid": None,
            "fid_macro": None,
            "cfid_macro": None,
            "kid": None,
        },

        # Keep a dedicated warnings list for auditability.
        "_warnings": metrics.get("_warnings", []),
    }

    # Legacy flattened keys (keep compatibility with older scripts)
    rec["metrics.ms_ssim"] = rec["generative"]["ms_ssim"]
    rec["metrics.fid"] = rec["generative"]["fid"]
    rec["metrics.fid_macro"] = rec["generative"]["fid_macro"]
    rec["metrics.cfid_macro"] = rec["generative"]["cfid_macro"]
    rec["metrics.kid"] = rec["generative"]["kid"]
    rec["counts.num_real"] = counts_map.get("train_real")
    rec["counts.num_fake"] = counts_map.get("synthetic")

    # Attach useful audit hints
    if budget_per_class is not None:
        rec.setdefault("run_meta", {})
        rec["run_meta"]["budget_per_class"] = budget_per_class

    # -------------------------------------------------------------------------
    # 7) Write outputs (dataset_scalable paths)
    # -------------------------------------------------------------------------
    # 7.1) Human_readable console log
    lines = [
        f"[eval] dataset_id={ident.dataset_id}",
        f"[eval] model_tag={ident.model_tag}",
        f"[eval] run_id={ident.run_id}  seed={ident.seed}",
        f"[eval] manifest={manifest_path}",
        f"[eval] out_dir={eval_paths.root_dir}",
        f"[eval] ms_ssim={rec['generative']['ms_ssim']}",
    ]
    if rec.get("_warnings"):
        lines.append("[eval] warnings:")
        for w in rec["_warnings"]:
            lines.append(f"  - {w}")
    lines.append("")
    write_text(console_path, "\n".join(lines), append=False, atomic=True)

    # 7.2) Snapshot (immutable record for debugging/audit)
    write_json(snapshot_path, rec, indent=2, sort_keys=True, atomic=True)

    # 7.3) latest.json (most recent summary for this run)
    write_json(latest_path, rec, indent=2, sort_keys=True, atomic=True)

    # 7.4) summary.jsonl (append log — good for HPC aggregation)
    append_jsonl(jsonl_path, rec)

    print(f"[eval] Saved snapshot → {snapshot_path}")
    print(f"[eval] Updated latest  → {latest_path}")
    print(f"[eval] Appended JSONL  → {jsonl_path}")
    print(f"[eval] Console summary → {console_path}")

    return rec


__all__ = ["evaluate_model_suite"]


# =============================================================================
# CLI entrypoint (kept for local runs)
# =============================================================================
def main():
    import argparse
    import yaml

    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--overrides", default=None)
    p.add_argument("--model", required=True)
    p.add_argument("--no_synth", action="store_true")
    args = p.parse_args()

    cfg = yaml.safe_load(open(args.config, "r"))
    cfg = cfg if isinstance(cfg, dict) else {}
    cfg.setdefault("run_meta", {})
    cfg["run_meta"]["config_path"] = args.config  # provenance hint

    if args.overrides:
        ov = yaml.safe_load(open(args.overrides, "r"))
        ov = ov if isinstance(ov, dict) else {}
        _deep_update(cfg, ov)

    rec = evaluate_model_suite(cfg, model_name=args.model, no_synth=args.no_synth)
    print("[runner] done:", rec.get("run_id"))


if __name__ == "__main__":
    main()
