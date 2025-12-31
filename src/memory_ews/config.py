### `src/memory_ews/config.py`
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

Bounds = Tuple[float, float, float, float]  # (lat_min, lat_max, lon_min, lon_max)

@dataclass(frozen=True)
class PipelineConfig:
    # ---- core ----
    var_name: str = "precip"
    dry_thr_z: float = -1.0
    min_valid_months: int = 360

    # ---- EWS ----
    ews_win_years: int = 10
    ews_min_frac: float = 0.8
    smooth_span_ews_months: int = 9

    # ---- baseline mu(t) ----
    mu_slow_win_years: int = 10
    mu_clip: tuple[float, float] = (1e-5, 0.25)

    # ---- memory index ----
    tau_months: float = 6.0
    kernel_max_lag_months: int = 60
    y_short_win_months: int = 12
    alpha_min: float = 1e-3
    alpha_cap: float = 50.0
    smooth_span_M_months: int = 9

    # ---- phase plot ----
    marker_scale: tuple[int, int] = (20, 350)

    # ---- lead-lag ----
    max_lag_leadlag_months: int = 60

    # trailing onset-rate for causal interpretation
    onset_rate_trailing: bool = True

    # Regions (bounding boxes)
    regions: Dict[str, Bounds] = None

    def with_default_regions(self) -> "PipelineConfig":
        if self.regions is not None:
            return self
        return PipelineConfig(
            var_name=self.var_name,
            dry_thr_z=self.dry_thr_z,
            min_valid_months=self.min_valid_months,
            ews_win_years=self.ews_win_years,
            ews_min_frac=self.ews_min_frac,
            smooth_span_ews_months=self.smooth_span_ews_months,
            mu_slow_win_years=self.mu_slow_win_years,
            mu_clip=self.mu_clip,
            tau_months=self.tau_months,
            kernel_max_lag_months=self.kernel_max_lag_months,
            y_short_win_months=self.y_short_win_months,
            alpha_min=self.alpha_min,
            alpha_cap=self.alpha_cap,
            smooth_span_M_months=self.smooth_span_M_months,
            marker_scale=self.marker_scale,
            max_lag_leadlag_months=self.max_lag_leadlag_months,
            onset_rate_trailing=self.onset_rate_trailing,
            regions={
                "ChileCentral": (-38.0, -28.0, -74.5, -69.0),
                "AustraliaSE": (-40.0, -28.0, 140.0, 155.0),
                "Mediterranean": (34.0, 46.0, -10.0, 36.0),
                "California": (32.0, 42.0, -124.5, -114.0),
            },
        )
