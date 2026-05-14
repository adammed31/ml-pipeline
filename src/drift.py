import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


class DataDriftDetector:

    PSI_THRESHOLD_WARNING = 0.1
    PSI_THRESHOLD_ALERT = 0.2
    KS_PVALUE_THRESHOLD = 0.05

    def compute_reference_stats(self, df: pd.DataFrame, target_col: str | None = None) -> dict:
        X = df.drop(columns=[target_col], errors="ignore") if target_col else df.copy()

        stats_ref = {"numerical": {}, "categorical": {}}

        for col in X.select_dtypes(include="number").columns:
            vals = X[col].dropna().tolist()
            stats_ref["numerical"][col] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
                "q25": float(np.percentile(vals, 25)),
                "q75": float(np.percentile(vals, 75)),
                "values_sample": vals[:500],
            }

        for col in X.select_dtypes(include=["object", "category"]).columns:
            vc = X[col].value_counts(normalize=True)
            stats_ref["categorical"][col] = vc.to_dict()

        return stats_ref

    def detect(self, df_new: pd.DataFrame, stats_ref: dict, target_col: str | None = None) -> dict:
        X = df_new.drop(columns=[target_col], errors="ignore") if target_col else df_new.copy()

        results = {"features": {}, "summary": {}}
        drift_count = 0
        total = 0

        for col, ref in stats_ref.get("numerical", {}).items():
            if col not in X.columns:
                continue
            new_vals = X[col].dropna().tolist()
            if len(new_vals) < 5:
                continue

            ks_stat, p_value = stats.ks_2samp(ref["values_sample"], new_vals)
            psi = self._compute_psi(ref["values_sample"], new_vals)
            drift = p_value < self.KS_PVALUE_THRESHOLD or psi > self.PSI_THRESHOLD_WARNING

            results["features"][col] = {
                "type": "numerical",
                "ks_stat": round(ks_stat, 4),
                "p_value": round(p_value, 4),
                "psi": round(psi, 4),
                "drift_detected": drift,
                "severity": self._severity(psi),
                "ref_mean": round(ref["mean"], 4),
                "new_mean": round(float(np.mean(new_vals)), 4),
                "ref_std": round(ref["std"], 4),
                "new_std": round(float(np.std(new_vals)), 4),
            }
            total += 1
            if drift:
                drift_count += 1

        for col, ref_dist in stats_ref.get("categorical", {}).items():
            if col not in X.columns:
                continue
            new_dist = X[col].value_counts(normalize=True).to_dict()
            all_cats = list(set(ref_dist) | set(new_dist))

            ref_freq = np.array([ref_dist.get(c, 1e-6) for c in all_cats])
            new_freq = np.array([new_dist.get(c, 1e-6) for c in all_cats])
            ref_freq /= ref_freq.sum()
            new_freq /= new_freq.sum()

            chi2, p_value = stats.chisquare(new_freq, ref_freq)
            new_cats = set(new_dist) - set(ref_dist)
            missing_cats = set(ref_dist) - set(new_dist)
            drift = p_value < self.KS_PVALUE_THRESHOLD or bool(new_cats)

            results["features"][col] = {
                "type": "categorical",
                "chi2": round(float(chi2), 4),
                "p_value": round(float(p_value), 4),
                "drift_detected": drift,
                "severity": "alert" if p_value < 0.01 else ("warning" if drift else "ok"),
                "new_categories": list(new_cats),
                "missing_categories": list(missing_cats),
            }
            total += 1
            if drift:
                drift_count += 1

        drift_rate = round(drift_count / total, 3) if total > 0 else 0.0
        results["summary"] = {
            "total_features": total,
            "drifted_features": drift_count,
            "drift_rate": drift_rate,
            "overall_status": "alert" if drift_rate > 0.3 else ("warning" if drift_rate > 0.1 else "ok"),
        }
        return results

    def save_reference(self, stats_ref: dict, path: str = "models/training_stats.json") -> None:
        clean = {"numerical": {}, "categorical": stats_ref.get("categorical", {})}
        for col, s in stats_ref.get("numerical", {}).items():
            clean["numerical"][col] = {k: v for k, v in s.items() if k != "values_sample"}
        with open(path, "w") as f:
            json.dump(clean, f)
        logger.info("Reference stats saved to %s", path)

    @classmethod
    def load_reference(cls, path: str = "models/training_stats.json") -> dict:
        with open(path) as f:
            return json.load(f)

    @staticmethod
    def _compute_psi(ref: list, new: list, bins: int = 10) -> float:
        breakpoints = np.percentile(ref, np.linspace(0, 100, bins + 1))
        breakpoints = np.unique(breakpoints)
        if len(breakpoints) < 2:
            return 0.0
        ref_pct = np.histogram(ref, bins=breakpoints)[0] / len(ref)
        new_pct = np.histogram(new, bins=breakpoints)[0] / len(new)
        ref_pct = np.where(ref_pct == 0, 1e-6, ref_pct)
        new_pct = np.where(new_pct == 0, 1e-6, new_pct)
        return float(np.sum((new_pct - ref_pct) * np.log(new_pct / ref_pct)))

    @staticmethod
    def _severity(psi: float) -> str:
        if psi > DataDriftDetector.PSI_THRESHOLD_ALERT:
            return "alert"
        if psi > DataDriftDetector.PSI_THRESHOLD_WARNING:
            return "warning"
        return "ok"
