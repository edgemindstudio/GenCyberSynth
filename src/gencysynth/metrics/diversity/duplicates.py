# src/gencysynth/metrics/diversity/duplicates.py
"""
Duplicate / near_duplicate detection for synthetic samples.

Why this metric exists
----------------------
A common failure mode in generative models is memorization or mode collapse:
- exact duplicates: identical images repeated
- near_duplicates: visually almost identical (tiny noise differences)

This file implements a *fast, deterministic, dependency_minimal* approach:
- Convert each image to a compact hash (8x8 average hash by default)
- Count duplicates by hash equality (exact in hash_space)
- Optionally estimate near_duplicates by Hamming distance threshold

Rule A
------
- No artifact paths, no saving — returns MetricResult only.
- Uses run.seed for deterministic subsampling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from ..types import DatasetMeta, MetricResult, RunMeta


def _to_grayscale01(x01: np.ndarray) -> np.ndarray:
    """
    Convert image batch to grayscale in [0,1].
    Accepts:
      - (N,H,W,1)
      - (N,H,W,3)
      - (N,H,W,C) where C != 1/3 -> uses first channel
    Returns: (N,H,W) float32 in [0,1]
    """
    x = np.asarray(x01, dtype=np.float32)
    if x.ndim != 4:
        raise ValueError(f"Expected (N,H,W,C); got {x.shape}")
    if x.shape[-1] == 1:
        g = x[..., 0]
    elif x.shape[-1] == 3:
        # Standard luminance weights
        g = 0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2]
    else:
        g = x[..., 0]
    return np.clip(g, 0.0, 1.0).astype(np.float32, copy=False)


def _resize_nn(img: np.ndarray, out_hw: Tuple[int, int]) -> np.ndarray:
    """
    Nearest_neighbor resize for a single grayscale image using NumPy only.
    img: (H,W) -> (oh,ow)
    """
    H, W = img.shape
    oh, ow = out_hw
    # Pick source indices for each destination pixel
    r_idx = (np.linspace(0, H - 1, oh)).astype(np.int64)
    c_idx = (np.linspace(0, W - 1, ow)).astype(np.int64)
    return img[np.ix_(r_idx, c_idx)]


def _ahash_bits(img_gray01: np.ndarray, hash_hw: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """
    Average hash (aHash) bit_vector for one grayscale image.
    - Resize to hash_hw
    - Threshold by mean

    Returns: (H*W,) uint8 in {0,1}
    """
    small = _resize_nn(img_gray01, hash_hw).astype(np.float32)
    m = float(np.mean(small))
    bits = (small >= m).astype(np.uint8).reshape(-1)
    return bits


def _packbits(bits01: np.ndarray) -> bytes:
    """
    Pack a {0,1} bit vector into bytes for dictionary keys.
    """
    return np.packbits(bits01.astype(np.uint8), bitorder="little").tobytes()


def _hamming_bytes(a: bytes, b: bytes) -> int:
    """
    Hamming distance between two packed bitstrings.
    """
    aa = np.frombuffer(a, dtype=np.uint8)
    bb = np.frombuffer(b, dtype=np.uint8)
    x = np.bitwise_xor(aa, bb)
    # popcount lookup
    return int(np.unpackbits(x).sum())


@dataclass
class DuplicatesMetric:
    """
    diversity.duplicates

    Options (cfg.metrics.options.diversity.duplicates):
      max_samples: 20000         # cap for speed
      hash_size: 8               # aHash size => hash_size x hash_size bits
      near_hamming: 0            # 0 => exact hash duplicates only; >0 => near_duplicates threshold
      per_class: false           # compute per_class duplicate rates if labels provided
    """

    def __call__(
        self,
        *,
        x_real01: np.ndarray,
        y_real: Optional[np.ndarray],
        x_synth01: np.ndarray,
        y_synth: Optional[np.ndarray],
        dataset: DatasetMeta,
        run: RunMeta,
        cfg: Dict,
    ) -> MetricResult:
        name = "diversity.duplicates"
        opts = (((cfg.get("metrics") or {}).get("options") or {}).get(name) or {})

        max_samples = int(opts.get("max_samples", 20000))
        hash_size = int(opts.get("hash_size", 8))
        near_hamming = int(opts.get("near_hamming", 0))
        per_class = bool(opts.get("per_class", False))

        rng = np.random.default_rng(int(run.seed))

        xs = np.asarray(x_synth01, dtype=np.float32)
        ys = None if y_synth is None else np.asarray(y_synth)

        # Subsample for speed, deterministically
        n_total = xs.shape[0]
        if n_total > max_samples:
            idx = rng.choice(n_total, size=max_samples, replace=False)
            xs = xs[idx]
            ys = ys[idx] if ys is not None else None

        # Hash all samples
        g = _to_grayscale01(xs)
        keys = []
        for i in range(g.shape[0]):
            bits = _ahash_bits(g[i], hash_hw=(hash_size, hash_size))
            keys.append(_packbits(bits))

        # Exact duplicates by key frequency
        from collections import Counter
        ctr = Counter(keys)
        n = len(keys)
        n_unique = len(ctr)
        dup_count = int(sum(c - 1 for c in ctr.values() if c > 1))
        dup_rate = float(dup_count / max(1, n))

        details = {
            "max_samples": max_samples,
            "hash_size": hash_size,
            "near_hamming": near_hamming,
            "n_used": int(n),
            "n_unique": int(n_unique),
            "duplicate_count": int(dup_count),
            "duplicate_rate": float(dup_rate),
        }

        # Optional near_duplicate estimate:
        # We do a lightweight pass over only the hash *unique* set.
        # Complexity O(U^2) can be big, so we cap U for this mode.
        if near_hamming > 0:
            # Cap unique hashes to keep this safe in big runs
            max_unique_for_near = int(opts.get("max_unique_for_near", 6000))
            uniq = list(ctr.keys())
            if len(uniq) > max_unique_for_near:
                uniq = list(rng.choice(uniq, size=max_unique_for_near, replace=False))
            near_pairs = 0
            for i in range(len(uniq)):
                ai = uniq[i]
                for j in range(i + 1, len(uniq)):
                    if _hamming_bytes(ai, uniq[j]) <= near_hamming:
                        near_pairs += 1
            details["near_duplicates"] = {
                "max_unique_for_near": max_unique_for_near,
                "unique_checked": int(len(uniq)),
                "near_pairs": int(near_pairs),
                "note": "near_pairs counts unique_hash pairs within threshold (not weighted by frequency).",
            }

        # Optional per_class duplicate rate (exact only; near_dup optional but expensive)
        if per_class and ys is not None:
            K = int(dataset.num_classes)
            per = {}
            for k in range(K):
                mk = (ys == k)
                nk = int(np.sum(mk))
                if nk <= 1:
                    per[str(k)] = {"status": "skipped", "n": nk}
                    continue
                keys_k = [keys[i] for i in range(n) if mk[i]]
                ctr_k = Counter(keys_k)
                dup_k = int(sum(c - 1 for c in ctr_k.values() if c > 1))
                per[str(k)] = {
                    "n": nk,
                    "n_unique": int(len(ctr_k)),
                    "duplicate_count": int(dup_k),
                    "duplicate_rate": float(dup_k / max(1, nk)),
                }
            details["per_class"] = per

        # Primary scalar is duplicate_rate (lower is better)
        return MetricResult(
            name=name,
            value=float(dup_rate),
            details=details,
            status="ok",
        )