"""
Macroeconomic model: GDP, unemployment, inflation dynamics.

Implements:
- Okun's Law (unemployment → GDP)
- Modified Phillips Curve (inflation ↔ unemployment, broken in AI era)
- Bifurcated CPI (deflationary AI basket + inflationary scarce basket)
- Technology adoption → productivity (Solow residual)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config.settings import EconomicDefaults
from src.models.scenarios import ScenarioParams


@dataclass
class MacroState:
    """Snapshot of macroeconomic state at a point in time."""
    period: str                 # e.g. "2026-M03" or "2027-Q2"
    unemployment_rate: float
    gdp_growth_annual: float    # Annualized growth rate
    gdp_level_trillion: float
    cpi_overall: float          # Year-over-year inflation
    cpi_deflation_basket: float # AI-impacted goods/services inflation (negative = deflation)
    cpi_inflation_basket: float # Scarce physical assets inflation
    fed_rate: float
    labor_force_participation: float
    ai_adoption_pct: float      # % of firms that have adopted AI meaningfully
    robot_fleet_size: int       # Cumulative robots deployed


class MacroEconomicsModel:
    """
    Projects macroeconomic trajectory under a given scenario.
    """

    def __init__(self, defaults: Optional[EconomicDefaults] = None):
        self.d = defaults or EconomicDefaults()

    # ------------------------------------------------------------------
    # Core projection
    # ------------------------------------------------------------------

    def project(
        self,
        scenario: ScenarioParams,
        months: int = 60,
        rng: Optional[np.random.Generator] = None,
    ) -> pd.DataFrame:
        """
        Generate month-by-month macroeconomic projections.

        Returns DataFrame with columns matching MacroState fields.
        """
        if rng is None:
            rng = np.random.default_rng(42)

        records: List[Dict] = []

        # Initial state
        unemp = self.d.baseline_unemployment
        gdp = self.d.total_gdp_trillion
        inflation = self.d.baseline_inflation
        fed_rate = self.d.baseline_fed_rate
        lfpr = 0.624  # labor force participation rate
        ai_adopt = 0.15  # 15% of firms have adopted AI meaningfully (2025)
        robots = 0

        # Scenario-specific parameters
        agi_month = self._quarter_to_month(scenario.agi_arrival_quarter)
        asi_month = self._quarter_to_month(scenario.asi_arrival_quarter) if scenario.asi_arrival_quarter else None

        for m in range(1, months + 1):
            # Time label
            year = 2026 + (m - 1) // 12
            month_in_year = ((m - 1) % 12) + 1
            period = f"{year}-M{month_in_year:02d}"

            # ---- AI adoption S-curve ----
            if agi_month and m >= agi_month:
                months_since_agi = m - agi_month
                ai_adopt = self._logistic(
                    months_since_agi,
                    ceiling=0.95,
                    midpoint=self.d.ai_adoption_midpoint_years * 12,
                    steepness=0.1 * scenario.ai_progress_rate,
                    floor=ai_adopt,
                )
            else:
                # Pre-AGI: slow linear adoption
                ai_adopt = min(ai_adopt + 0.002, 0.30)

            # ---- Robot deployment ----
            monthly_robots = scenario.robot_units_year1 / 12
            if asi_month and m >= asi_month:
                accel = 1 + 0.05 * (m - asi_month)  # Accelerating production
                monthly_robots *= min(accel, 5.0)
            robots += int(monthly_robots)

            # ---- Unemployment dynamics ----
            # Base: displacement pressure from AI + robots
            displacement_pressure = self._displacement_pressure(
                m, agi_month, asi_month, scenario
            )
            # Natural adjustment: some workers retrain / new jobs created
            adjustment = 0.003 * ai_adopt  # New AI-economy jobs
            unemp_target = self.d.baseline_unemployment + displacement_pressure - adjustment
            unemp_target = np.clip(unemp_target, 0.02, scenario.peak_unemployment_rate)
            # Smooth adjustment toward target
            unemp += 0.15 * (unemp_target - unemp) + rng.normal(0, 0.001)
            unemp = np.clip(unemp, 0.02, 0.30)

            # ---- GDP dynamics (modified Okun's Law) ----
            unemployment_gap = unemp - self.d.nairu
            okun_effect = self.d.okun_coefficient * unemployment_gap
            productivity_boost = scenario.gdp_growth_modifier * ai_adopt
            gdp_growth_annual = (
                self.d.baseline_gdp_growth + okun_effect + productivity_boost
                + rng.normal(0, 0.002)
            )
            gdp_growth_annual = np.clip(gdp_growth_annual, -0.10, 0.20)
            gdp *= (1 + gdp_growth_annual / 12)

            # ---- Inflation dynamics (bifurcated) ----
            # Phillips curve component
            phillips = self.d.phillips_slope * (unemp - self.d.nairu)

            # Deflation basket: AI crushes prices for digital goods
            defl_basket = -0.02 * ai_adopt * scenario.ai_progress_rate
            if asi_month and m >= asi_month:
                defl_basket *= 2.0  # ASI accelerates deflation

            # Inflation basket: scarce assets + energy
            infl_basket = (
                0.03 + 0.02 * ai_adopt  # Energy demand from AI
                + scenario.inflation_modifier
            )
            if scenario.policy_response.value == "ubi_early" and scenario.ubi_start_quarter:
                ubi_month = self._quarter_to_month(scenario.ubi_start_quarter)
                if ubi_month and m >= ubi_month:
                    infl_basket += 0.015  # UBI adds demand-side inflation

            # Overall CPI = weighted average
            cpi_overall = 0.45 * defl_basket + 0.55 * infl_basket + phillips
            cpi_overall += rng.normal(0, 0.001)

            # ---- Fed rate (Taylor Rule approximation) ----
            inflation_gap = cpi_overall - self.d.inflation_expectations
            output_gap = gdp_growth_annual - self.d.baseline_gdp_growth
            fed_rate_target = (
                self.d.inflation_expectations + 0.02  # real neutral rate
                + 1.5 * inflation_gap
                + 0.5 * output_gap
            )
            fed_rate += 0.05 * (fed_rate_target - fed_rate)  # Gradual adjustment
            fed_rate = np.clip(fed_rate, 0.0, 0.08)

            # ---- LFPR ----
            lfpr -= 0.0005 * displacement_pressure  # Discouraged workers leave
            lfpr = np.clip(lfpr, 0.55, 0.65)

            records.append({
                "period": period,
                "month": m,
                "year": year,
                "unemployment_rate": round(unemp, 4),
                "gdp_growth_annual": round(gdp_growth_annual, 4),
                "gdp_level_trillion": round(gdp, 3),
                "cpi_overall": round(cpi_overall, 4),
                "cpi_deflation_basket": round(defl_basket, 4),
                "cpi_inflation_basket": round(infl_basket, 4),
                "fed_rate": round(fed_rate, 4),
                "labor_force_participation": round(lfpr, 4),
                "ai_adoption_pct": round(ai_adopt, 4),
                "robot_fleet_size": robots,
            })

        return pd.DataFrame(records)

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    @staticmethod
    def _logistic(
        t: float,
        ceiling: float = 1.0,
        midpoint: float = 36,
        steepness: float = 0.1,
        floor: float = 0.0,
    ) -> float:
        """Logistic (S-curve) function."""
        return floor + (ceiling - floor) / (1 + np.exp(-steepness * (t - midpoint)))

    @staticmethod
    def _quarter_to_month(quarter_str: Optional[str]) -> Optional[int]:
        """Convert '2026-Q4' to month offset from 2026-Q1 (month 1)."""
        if not quarter_str or quarter_str == "never":
            return None
        try:
            year, q = quarter_str.split("-Q")
            year, q = int(year), int(q)
            return (year - 2026) * 12 + (q - 1) * 3 + 1
        except (ValueError, AttributeError):
            return None

    def _displacement_pressure(
        self,
        month: int,
        agi_month: Optional[int],
        asi_month: Optional[int],
        scenario: ScenarioParams,
    ) -> float:
        """
        Calculate unemployment displacement pressure at a given month.
        Returns a rate (e.g. 0.05 = 5 percentage points above baseline).
        """
        if agi_month is None or month < agi_month:
            # Pre-AGI: minimal displacement
            return 0.005 * scenario.ai_progress_rate * (month / 60)

        months_since_agi = month - agi_month
        base_pressure = scenario.peak_unemployment_rate - self.d.baseline_unemployment

        # Ramp up displacement following logistic curve
        pressure = base_pressure * self._logistic(
            months_since_agi,
            ceiling=1.0,
            midpoint=18 / scenario.ai_progress_rate,
            steepness=0.12 * scenario.ai_progress_rate,
            floor=0.0,
        )

        # ASI accelerates displacement
        if asi_month and month >= asi_month:
            asi_boost = 0.3 * (1 - np.exp(-0.05 * (month - asi_month)))
            pressure *= (1 + asi_boost)

        return min(pressure, base_pressure * 1.2)
