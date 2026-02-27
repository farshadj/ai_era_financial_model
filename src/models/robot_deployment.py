"""
Robot deployment model: production tracking, cost curves, capability benchmarks.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from src.models.scenarios import ScenarioParams


class RobotDeploymentModel:
    """Models robot production, cost curves, and capability evolution."""

    def __init__(
        self,
        initial_cost: float = 50_000.0,       # $/unit in 2025
        target_cost: float = 15_000.0,          # $/unit by 2030
        learning_rate: float = 0.18,            # 18% cost reduction per doubling
        initial_capability: float = 0.3,        # 0-1 scale vs human baseline
        initial_reliability_hours: float = 500, # Hours between maintenance
    ):
        self.initial_cost = initial_cost
        self.target_cost = target_cost
        self.learning_rate = learning_rate
        self.initial_capability = initial_capability
        self.initial_reliability_hours = initial_reliability_hours

    def project(
        self,
        scenario: ScenarioParams,
        months: int = 60,
        rng: Optional[np.random.Generator] = None,
    ) -> pd.DataFrame:
        """
        Project robot deployment metrics over time.

        Returns monthly DataFrame with:
        - cumulative_units: Total robots deployed
        - monthly_production: New units produced that month
        - unit_cost: Manufacturing cost per unit
        - capability_score: 0-1 tasks vs human baseline
        - reliability_hours: Mean hours between maintenance
        - tasks_per_hour: Adjusted TPH relative to human baseline
        - energy_efficiency: kWh per task
        """
        if rng is None:
            rng = np.random.default_rng(42)

        records = []
        cumulative = 0
        monthly_base = scenario.robot_units_year1 / 12

        agi_month = self._q_to_m(scenario.agi_arrival_quarter)
        asi_month = self._q_to_m(scenario.asi_arrival_quarter) if scenario.asi_arrival_quarter else None

        for m in range(1, months + 1):
            # Production rate (accelerates post-ASI)
            production = monthly_base
            if asi_month and m >= asi_month:
                accel = 1 + 0.08 * (m - asi_month)   # 8% acceleration per month post-ASI
                production *= min(accel, 6.0)
            elif agi_month and m >= agi_month:
                accel = 1 + 0.02 * (m - agi_month)
                production *= min(accel, 2.0)

            production = int(production * (1 + rng.normal(0, 0.05)))
            production = max(0, production)
            cumulative += production

            # Cost curve (Wright's Law / experience curve)
            if cumulative > 0:
                doublings = np.log2(max(cumulative, 1) / max(scenario.robot_units_year1, 1))
                doublings = max(doublings, 0)
                unit_cost = self.initial_cost * (1 - self.learning_rate) ** doublings
                unit_cost = max(unit_cost, self.target_cost)
            else:
                unit_cost = self.initial_cost

            # Capability (logistic improvement)
            cap_ceiling = 0.95  # Near human-level
            if asi_month and m >= asi_month:
                cap_steepness = 0.15
                cap_mid = 12
                t = m - asi_month
            elif agi_month and m >= agi_month:
                cap_steepness = 0.08
                cap_mid = 24
                t = m - agi_month
            else:
                cap_steepness = 0.03
                cap_mid = 36
                t = m

            capability = self.initial_capability + (
                (cap_ceiling - self.initial_capability)
                / (1 + np.exp(-cap_steepness * (t - cap_mid)))
            )

            # Reliability improves with deployment
            reliability = self.initial_reliability_hours * (1 + 0.5 * capability)

            # Tasks per hour (relative to human = 1.0)
            tph = capability * 1.2  # Robots can work faster when capable

            # Energy efficiency (improves over time)
            energy_kwh = 2.0 * (1 - 0.3 * capability)  # kWh per task

            year = 2026 + (m - 1) // 12
            records.append({
                "month": m,
                "year": year,
                "cumulative_units": cumulative,
                "monthly_production": production,
                "unit_cost": round(unit_cost, 0),
                "capability_score": round(capability, 3),
                "reliability_hours": round(reliability, 0),
                "tasks_per_hour": round(tph, 3),
                "energy_kwh_per_task": round(energy_kwh, 3),
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
