"""
AI progress tracking model: benchmarks, compute scaling, capability indices.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from src.models.scenarios import ScenarioParams


# Benchmark baselines (as of early 2026)
AI_BENCHMARKS = {
    "swe_bench": {
        "description": "Software Engineering Tasks (% human-level)",
        "current_score": 0.55,  # 55% as of early 2026
        "human_level": 1.0,
    },
    "mmlu": {
        "description": "Massive Multitask Language Understanding (percentile)",
        "current_score": 0.92,
        "human_level": 1.0,
    },
    "humaneval": {
        "description": "Code Generation (pass@1)",
        "current_score": 0.85,
        "human_level": 1.0,
    },
    "robotics_manipulation": {
        "description": "Manipulation Success Rate",
        "current_score": 0.45,
        "human_level": 1.0,
    },
    "robotics_navigation": {
        "description": "Navigation Accuracy",
        "current_score": 0.70,
        "human_level": 1.0,
    },
    "recursive_improvement": {
        "description": "Training Efficiency Gains per Generation",
        "current_score": 0.20,  # 20% improvement per generation
        "human_level": None,     # No human baseline
    },
}

# Compute scaling (estimated FLOPS in AI training)
COMPUTE_BASELINE = {
    "training_flops_2025": 1e26,   # ~10^26 FLOPS for frontier model
    "inference_cost_per_1m_tokens": 0.50,  # $ per 1M tokens
    "chip_manufacturing_capacity_wafers_month": 200_000,
}


class AIProgressModel:
    """Models AI capability progress across benchmarks and compute."""

    def __init__(self):
        self.benchmarks = AI_BENCHMARKS.copy()
        self.compute = COMPUTE_BASELINE.copy()

    def project_benchmarks(
        self,
        scenario: ScenarioParams,
        months: int = 60,
        rng: Optional[np.random.Generator] = None,
    ) -> pd.DataFrame:
        """
        Project AI benchmark scores over time under a scenario.

        Benchmark progress follows logistic curves modulated by scenario's
        ai_progress_rate.
        """
        if rng is None:
            rng = np.random.default_rng(42)

        agi_month = self._q_to_m(scenario.agi_arrival_quarter)
        records = []

        for m in range(1, months + 1):
            for bench_name, params in self.benchmarks.items():
                current = params["current_score"]
                ceiling = params["human_level"] if params["human_level"] else 2.0

                # Progress rate depends on scenario
                if agi_month and m >= agi_month:
                    # Post-AGI: rapid improvement
                    t = m - agi_month
                    k = 0.15 * scenario.ai_progress_rate
                    mid = 12 / scenario.ai_progress_rate
                    score = current + (ceiling - current) / (1 + np.exp(-k * (t - mid)))
                else:
                    # Pre-AGI: gradual improvement
                    monthly_improvement = (
                        0.005 * scenario.ai_progress_rate
                        + rng.normal(0, 0.002)
                    )
                    score = current + monthly_improvement * m
                    score = min(score, ceiling * 0.9)  # Cap below human level pre-AGI

                score = min(score, ceiling) + rng.normal(0, 0.005)
                score = np.clip(score, 0, ceiling)

                records.append({
                    "month": m,
                    "benchmark": bench_name,
                    "score": round(score, 4),
                    "human_level": params["human_level"],
                    "pct_human": round(score / ceiling, 4) if ceiling else None,
                })

        return pd.DataFrame(records)

    def project_compute(
        self,
        scenario: ScenarioParams,
        months: int = 60,
        rng: Optional[np.random.Generator] = None,
    ) -> pd.DataFrame:
        """Project compute scaling: FLOPS, cost per token, chip capacity."""
        if rng is None:
            rng = np.random.default_rng(42)

        records = []
        flops = self.compute["training_flops_2025"]
        cost_per_token = self.compute["inference_cost_per_1m_tokens"]
        chip_capacity = self.compute["chip_manufacturing_capacity_wafers_month"]

        for m in range(1, months + 1):
            # Compute scales exponentially (doubling every ~6 months in aggressive scenarios)
            doubling_months = 8 / scenario.ai_progress_rate
            flops *= 2 ** (1 / doubling_months)

            # Cost per token decreases (Moore's Law + algorithmic improvements)
            cost_per_token *= (1 - 0.05 * scenario.ai_progress_rate / 12)
            cost_per_token = max(cost_per_token, 0.001)

            # Chip manufacturing capacity grows
            chip_capacity *= (1 + 0.03 / 12)  # ~3% annual growth in fab capacity

            records.append({
                "month": m,
                "training_flops": flops,
                "cost_per_1m_tokens": round(cost_per_token, 4),
                "chip_capacity_wafers": int(chip_capacity),
                "log10_flops": round(np.log10(flops), 2),
            })

        return pd.DataFrame(records)

    def composite_ai_index(
        self,
        benchmark_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Compute a composite AI progress index from benchmark scores.

        Returns monthly composite score (0-100 scale).
        """
        # Weight each benchmark
        weights = {
            "swe_bench": 0.25,
            "mmlu": 0.15,
            "humaneval": 0.20,
            "robotics_manipulation": 0.15,
            "robotics_navigation": 0.10,
            "recursive_improvement": 0.15,
        }

        records = []
        for m in benchmark_df["month"].unique():
            month_data = benchmark_df[benchmark_df["month"] == m]
            composite = 0.0
            for _, row in month_data.iterrows():
                bench = row["benchmark"]
                if bench in weights:
                    pct = row["pct_human"] if row["pct_human"] is not None else row["score"]
                    composite += weights[bench] * pct * 100

            records.append({
                "month": int(m),
                "ai_composite_index": round(composite, 2),
            })

        return pd.DataFrame(records)

    @staticmethod
    def _q_to_m(quarter_str: Optional[str]) -> Optional[int]:
        if not quarter_str or quarter_str == "never":
            return None
        try:
            year, q = quarter_str.split("-Q")
            return (int(year) - 2026) * 12 + (int(q) - 1) * 3 + 1
        except (ValueError, AttributeError):
            return None
