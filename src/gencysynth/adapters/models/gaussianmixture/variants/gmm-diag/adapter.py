from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

from gencysynth.adapters.models.gaussianmixture.base import GaussianMixtureAdapterBase
from gencysynth.adapters.models.base import TrainResult
from gencysynth.adapters.datasets.splits import DatasetSplits


class GMMDiagAdapter(GaussianMixtureAdapterBase):
    def __init__(self):
        super().__init__(variant="gmm_diag", description="Class-conditional sklearn GaussianMixture (diag cov)")

    def _train_impl(
        self,
        run_ctx: Any,
        cfg: Dict[str, Any],
        data: DatasetSplits,
        *,
        x_flat: np.ndarray,
        y_onehot: np.ndarray,
        img_shape: Tuple[int, int, int],
        ckpt_dir: Path,
        summaries_dir: Path,
    ) -> TrainResult:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        (ckpt_dir / "trained.txt").write_text("trained (fit happens in synth)\n")
        return TrainResult(checkpoints_dir=str(ckpt_dir), summaries_dir=str(summaries_dir))

    def _synth_impl(
        self,
        run_ctx: Any,
        cfg: Dict[str, Any],
        data: DatasetSplits,
        *,
        out_dir: Path,
    ) -> Tuple[np.ndarray, np.ndarray]:
        from sklearn.mixture import GaussianMixture

        x_flat, y1h, img_shape = self.standardize_train_inputs(cfg, data)
        y_int = np.argmax(y1h, axis=1).astype(int)

        K = int(y1h.shape[1])
        n_per_class = int(cfg.get("synth", {}).get("n_per_class", 10))
        gm_cfg = cfg.get("gaussianmixture", {}) if isinstance(cfg.get("gaussianmixture"), dict) else {}
        n_components = int(gm_cfg.get("n_components", 8))
        max_iter = int(gm_cfg.get("max_iter", 25))
        seed = int(cfg.get("SEED", 42))

        rng = np.random.default_rng(seed)

        xs, ys = [], []
        for k in range(K):
            idx = np.where(y_int == k)[0]
            if len(idx) < max(10, n_components):
                if len(idx) == 0:
                    continue
                pick = rng.choice(idx, size=n_per_class, replace=True)
                xk = x_flat[pick]
            else:
                gmm = GaussianMixture(
                    n_components=n_components,
                    covariance_type="diag",
                    max_iter=max_iter,
                    random_state=seed + k,
                )
                gmm.fit(x_flat[idx])
                xk, _ = gmm.sample(n_per_class)

            xs.append(xk)
            ys.append(np.full((xk.shape[0],), k, dtype=np.int32))

        if not xs:
            H, W, C = img_shape
            return np.zeros((0, H, W, C), dtype=np.float32), np.zeros((0, K), dtype=np.float32)

        x_all = np.concatenate(xs, axis=0).astype(np.float32)
        y_all = np.concatenate(ys, axis=0).astype(np.int32)

        H, W, C = img_shape
        x01 = x_all.reshape((-1, H, W, C))
        x01 = np.clip(x01, 0.0, 1.0)

        y1h_out = np.zeros((len(y_all), K), dtype=np.float32)
        y1h_out[np.arange(len(y_all)), y_all] = 1.0

        return x01, y1h_out