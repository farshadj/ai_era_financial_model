"""
Firm agents: Representatives of US businesses that make hiring, AI adoption, and pricing decisions.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from src.agents.base import BaseAgent


INDUSTRIES = [
    {"name": "technology", "ai_roi_multiplier": 2.0, "labor_intensity": 0.3},
    {"name": "financial_services", "ai_roi_multiplier": 1.5, "labor_intensity": 0.4},
    {"name": "healthcare", "ai_roi_multiplier": 1.3, "labor_intensity": 0.6},
    {"name": "manufacturing", "ai_roi_multiplier": 1.4, "labor_intensity": 0.5},
    {"name": "retail", "ai_roi_multiplier": 1.0, "labor_intensity": 0.7},
    {"name": "logistics", "ai_roi_multiplier": 1.2, "labor_intensity": 0.6},
    {"name": "professional_services", "ai_roi_multiplier": 1.6, "labor_intensity": 0.5},
    {"name": "construction", "ai_roi_multiplier": 0.8, "labor_intensity": 0.8},
    {"name": "education", "ai_roi_multiplier": 0.7, "labor_intensity": 0.7},
    {"name": "hospitality", "ai_roi_multiplier": 0.6, "labor_intensity": 0.8},
]


class FirmAgent(BaseAgent):
    """
    Firm agent that makes AI adoption, hiring, and pricing decisions.

    Properties:
        industry: str
        size: str ('small', 'medium', 'large')
        employee_count: int
        revenue: float (annual)
        ai_adopted: bool
        ai_adoption_month: int (when adopted)
        ai_roi_multiplier: float
        labor_intensity: float (0-1, how dependent on human labor)
    """

    def __init__(self, agent_id: int, **kwargs):
        super().__init__(agent_id, "firm", **kwargs)
        self.ai_adopted = False
        self.ai_adoption_month = 0
        self.robot_adopted = False
        self.employees_displaced = 0

    def step(self, env: Dict[str, Any]) -> Dict[str, Any]:
        """
        Firm behavior per month:
        1. Evaluate AI adoption ROI
        2. Adopt AI if payback < 3 years
        3. Displace workers, hire AI-complementary roles
        4. Adjust pricing & revenue
        """
        rng: np.random.Generator = env.get("rng", np.random.default_rng())
        month = env.get("month", 1)
        ai_cost_per_worker_month = env.get("ai_cost_per_worker_month", 500)
        robot_cost_per_unit = env.get("robot_cost_per_unit", 50_000)
        robot_capability = env.get("robot_capability", 0.3)
        ai_capability = env.get("ai_capability", 0.5)

        actions = {"hired": 0, "fired": 0, "ai_adopted": False, "revenue_change": 0}

        employee_count = self.properties.get("employee_count", 100)
        revenue = self.properties.get("revenue", 10_000_000)
        labor_intensity = self.properties.get("labor_intensity", 0.5)
        ai_roi_mult = self.properties.get("ai_roi_multiplier", 1.0)

        # --- AI Adoption Decision ---
        if not self.ai_adopted:
            # ROI calculation: compare AI cost vs. labor cost savings
            avg_worker_cost_month = 5_000  # ~$60K/year loaded
            potential_displaced = int(employee_count * labor_intensity * ai_capability * 0.3)
            annual_savings = potential_displaced * avg_worker_cost_month * 12
            annual_ai_cost = potential_displaced * ai_cost_per_worker_month * 12

            if annual_savings > 0:
                payback_years = annual_ai_cost / (annual_savings * ai_roi_mult) if annual_savings > 0 else 99

                # Firms adopt AI when payback period < 3 years
                if payback_years < 3.0 and rng.random() < 0.10:  # 10% monthly adoption probability
                    self.ai_adopted = True
                    self.ai_adoption_month = month
                    actions["ai_adopted"] = True

        # --- Post-adoption: displace workers, boost revenue ---
        if self.ai_adopted:
            months_since = month - self.ai_adoption_month
            # Gradual displacement over 12 months
            target_displaced = int(
                employee_count * labor_intensity * ai_capability * 0.4
                * min(months_since / 12, 1.0)
            )
            new_displaced = max(0, target_displaced - self.employees_displaced)
            self.employees_displaced = target_displaced
            actions["fired"] = new_displaced

            # Revenue boost from AI productivity
            productivity_boost = 0.002 * ai_roi_mult * min(months_since / 12, 1.0)
            revenue_change = revenue * productivity_boost
            self.properties["revenue"] = revenue + revenue_change
            actions["revenue_change"] = revenue_change

        # --- Robot adoption for physical-labor firms ---
        if not self.robot_adopted and labor_intensity > 0.5:
            if robot_capability > 0.5 and robot_cost_per_unit < 30_000:
                if rng.random() < 0.05:
                    self.robot_adopted = True

        return actions


def create_firm_population(
    n_agents: int,
    rng: Optional[np.random.Generator] = None,
) -> List[FirmAgent]:
    """Create a representative population of firm agents."""
    if rng is None:
        rng = np.random.default_rng(42)

    agents = []
    for i in range(n_agents):
        industry = rng.choice(INDUSTRIES)
        size = rng.choice(["small", "medium", "large"], p=[0.75, 0.20, 0.05])
        size_map = {"small": (5, 50), "medium": (50, 500), "large": (500, 10_000)}
        lo, hi = size_map[size]
        emp = int(rng.uniform(lo, hi))

        agents.append(FirmAgent(
            agent_id=i,
            industry=industry["name"],
            size=size,
            employee_count=emp,
            revenue=float(emp * rng.uniform(80_000, 300_000)),  # Revenue per employee
            ai_roi_multiplier=industry["ai_roi_multiplier"],
            labor_intensity=industry["labor_intensity"],
        ))

    return agents
