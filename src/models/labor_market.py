"""
Labor market model: occupations, vulnerability indices, displacement forecasting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Occupation data
# ---------------------------------------------------------------------------

@dataclass
class Occupation:
    """Single occupation with automation vulnerability data."""
    name: str
    us_employment: int           # Total US employment
    ai_vulnerability: float      # 0.0 – 1.0 (AI/cognitive automation)
    robot_vulnerability: float   # 0.0 – 1.0 (physical robot automation)
    avg_annual_wage: float       # Average annual wage in $
    category: str                # "cognitive", "physical", "mixed"

    # Task composition (fractions summing to 1.0)
    cognitive_routine: float = 0.0
    physical_routine: float = 0.0
    cognitive_non_routine: float = 0.0
    physical_non_routine: float = 0.0
    human_interaction: float = 0.0

    @property
    def composite_vulnerability(self) -> float:
        """Weighted automation vulnerability index (0-100 scale)."""
        return (
            self.cognitive_routine * 0.8
            + self.physical_routine * 0.7
            + self.cognitive_non_routine * 0.4
            + self.physical_non_routine * 0.3
            + self.human_interaction * 0.1
        ) * 100


# Top 20 critical occupations from the spec
CRITICAL_OCCUPATIONS: List[Occupation] = [
    Occupation("Software Developers", 4_400_000, 0.70, 0.05, 130_000, "cognitive",
               0.35, 0.0, 0.50, 0.0, 0.15),
    Occupation("Customer Service Reps", 2_900_000, 0.90, 0.10, 38_000, "cognitive",
               0.60, 0.0, 0.15, 0.0, 0.25),
    Occupation("Truck Drivers", 1_800_000, 0.10, 0.60, 52_000, "physical",
               0.10, 0.50, 0.05, 0.30, 0.05),
    Occupation("Data Entry Clerks", 1_500_000, 0.95, 0.05, 36_000, "cognitive",
               0.80, 0.0, 0.10, 0.0, 0.10),
    Occupation("Accountants", 1_400_000, 0.65, 0.05, 82_000, "cognitive",
               0.45, 0.0, 0.35, 0.0, 0.20),
    Occupation("Warehouse Workers", 1_700_000, 0.15, 0.80, 36_000, "physical",
               0.10, 0.60, 0.05, 0.20, 0.05),
    Occupation("Factory Assemblers", 2_500_000, 0.10, 0.75, 38_000, "physical",
               0.10, 0.55, 0.05, 0.25, 0.05),
    Occupation("Retail Salespersons", 3_700_000, 0.50, 0.30, 32_000, "mixed",
               0.25, 0.10, 0.20, 0.10, 0.35),
    Occupation("Food Prep Workers", 2_800_000, 0.15, 0.50, 30_000, "physical",
               0.10, 0.40, 0.05, 0.25, 0.20),
    Occupation("Paralegals", 400_000, 0.75, 0.05, 60_000, "cognitive",
               0.50, 0.0, 0.30, 0.0, 0.20),
    Occupation("Financial Analysts", 500_000, 0.70, 0.05, 100_000, "cognitive",
               0.40, 0.0, 0.40, 0.0, 0.20),
    Occupation("Market Research Analysts", 800_000, 0.80, 0.05, 75_000, "cognitive",
               0.50, 0.0, 0.30, 0.0, 0.20),
    Occupation("Radiologists", 30_000, 0.60, 0.10, 350_000, "cognitive",
               0.40, 0.0, 0.40, 0.0, 0.20),
    Occupation("Junior Lawyers", 200_000, 0.65, 0.05, 90_000, "cognitive",
               0.35, 0.0, 0.35, 0.0, 0.30),
    Occupation("Insurance Underwriters", 100_000, 0.85, 0.05, 78_000, "cognitive",
               0.60, 0.0, 0.25, 0.0, 0.15),
    Occupation("Tax Preparers", 300_000, 0.90, 0.05, 48_000, "cognitive",
               0.65, 0.0, 0.20, 0.0, 0.15),
    Occupation("Telemarketers", 200_000, 0.99, 0.05, 30_000, "cognitive",
               0.70, 0.0, 0.10, 0.0, 0.20),
    Occupation("Bookkeepers", 1_500_000, 0.80, 0.05, 46_000, "cognitive",
               0.60, 0.0, 0.20, 0.0, 0.20),
    Occupation("Claims Adjusters", 300_000, 0.70, 0.05, 72_000, "cognitive",
               0.45, 0.0, 0.30, 0.0, 0.25),
    Occupation("Construction Laborers", 1_500_000, 0.10, 0.45, 42_000, "physical",
               0.05, 0.35, 0.05, 0.40, 0.15),
]


# ---------------------------------------------------------------------------
# Labor Market Model
# ---------------------------------------------------------------------------

class LaborMarketModel:
    """Models employment displacement over time under different scenarios."""

    def __init__(self, occupations: Optional[List[Occupation]] = None):
        self.occupations = occupations or CRITICAL_OCCUPATIONS
        self._total_tracked = sum(o.us_employment for o in self.occupations)

    @property
    def total_tracked_employment(self) -> int:
        return self._total_tracked

    def displacement_curve(
        self,
        ai_progress_rate: float,
        robot_deployment_rate_enum: str,
        months: int = 60,
    ) -> pd.DataFrame:
        """
        Compute monthly cumulative job displacement for each occupation.

        Uses a logistic (S-curve) model:
            displaced_frac(t) = vulnerability / (1 + exp(-k*(t - t_mid)))

        where:
            - vulnerability: occupation-specific automation probability
            - k: steepness (higher ai_progress → steeper)
            - t_mid: midpoint month (lower ai_progress → later midpoint)
        """
        robot_speed = {"slow_100k_yr": 0.6, "medium_500k_yr": 1.0, "fast_2m_yr": 1.8}
        r_mult = robot_speed.get(robot_deployment_rate_enum, 1.0)

        records = []
        month_range = np.arange(1, months + 1)

        for occ in self.occupations:
            # Combined vulnerability depends on whether job is cognitive vs physical
            if occ.category == "cognitive":
                vuln = occ.ai_vulnerability
                speed = ai_progress_rate
            elif occ.category == "physical":
                vuln = occ.robot_vulnerability
                speed = ai_progress_rate * r_mult
            else:  # mixed
                vuln = max(occ.ai_vulnerability, occ.robot_vulnerability)
                speed = ai_progress_rate * (1 + r_mult) / 2

            # S-curve parameters
            k = 0.08 * speed           # steepness
            t_mid = 30 / speed          # midpoint month

            for m in month_range:
                frac = vuln / (1 + np.exp(-k * (m - t_mid)))
                displaced = int(occ.us_employment * frac)
                records.append({
                    "month": int(m),
                    "occupation": occ.name,
                    "category": occ.category,
                    "total_employment": occ.us_employment,
                    "displaced": displaced,
                    "displacement_fraction": frac,
                    "remaining": occ.us_employment - displaced,
                })

        return pd.DataFrame(records)

    def aggregate_displacement(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate displacement across all occupations by month."""
        agg = df.groupby("month").agg(
            total_displaced=("displaced", "sum"),
            total_remaining=("remaining", "sum"),
        ).reset_index()
        agg["total_tracked"] = self._total_tracked
        agg["displacement_rate"] = agg["total_displaced"] / self._total_tracked
        return agg

    def vulnerability_ranking(self) -> pd.DataFrame:
        """Return occupations ranked by composite vulnerability."""
        rows = []
        for occ in self.occupations:
            rows.append({
                "occupation": occ.name,
                "employment": occ.us_employment,
                "ai_vulnerability": occ.ai_vulnerability,
                "robot_vulnerability": occ.robot_vulnerability,
                "composite_index": occ.composite_vulnerability,
                "category": occ.category,
                "avg_wage": occ.avg_annual_wage,
            })
        df = pd.DataFrame(rows)
        return df.sort_values("composite_index", ascending=False).reset_index(drop=True)

    def wage_impact(
        self,
        displacement_df: pd.DataFrame,
        month: int,
    ) -> Dict[str, float]:
        """Estimate wage pressure by category at a given month."""
        snapshot = displacement_df[displacement_df["month"] == month]
        results = {}
        for cat in ["cognitive", "physical", "mixed"]:
            cat_data = snapshot[snapshot["category"] == cat]
            if cat_data.empty:
                results[cat] = 0.0
                continue
            # Weighted average displacement drives wage decline
            total_emp = cat_data["total_employment"].sum()
            total_disp = cat_data["displaced"].sum()
            disp_rate = total_disp / total_emp if total_emp > 0 else 0
            # Wage elasticity: -0.5 (10% displacement → ~5% wage decline)
            results[cat] = -0.5 * disp_rate
        return results
