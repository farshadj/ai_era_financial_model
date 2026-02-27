"""
Government agent: Policy decisions (UBI, regulation, fiscal policy).
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from src.agents.base import BaseAgent


class GovernmentAgent(BaseAgent):
    """
    Government agent that makes policy decisions based on economic conditions.

    Properties:
        jurisdiction: str (e.g. "US_federal")
        political_orientation: float (-1 = progressive, 0 = centrist, 1 = conservative)
        tax_revenue_annual: float
        debt_level_trillion: float
        ubi_active: bool
        ubi_monthly_amount: float
    """

    def __init__(self, agent_id: int, **kwargs):
        super().__init__(agent_id, "government", **kwargs)
        self.ubi_active = False
        self.ubi_monthly_amount = 0.0
        self.regulation_level = 0.0  # 0-1 scale
        self.emergency_spending = 0.0

    def step(self, env: Dict[str, Any]) -> Dict[str, Any]:
        """
        Government behavior per month:
        1. Monitor unemployment, social stability
        2. Decide on UBI: implement if unemployment > 12% AND protests > threshold
        3. Decide on regulation: implement if AI safety concerns or political pressure
        4. Adjust fiscal policy (deficit spending in crises)
        """
        rng: np.random.Generator = env.get("rng", np.random.default_rng())
        unemployment_rate = env.get("unemployment_rate", 0.04)
        protest_cities = env.get("protest_cities", 0)
        ai_adoption_pct = env.get("ai_adoption_pct", 0.15)
        month = env.get("month", 1)

        orientation = self.properties.get("political_orientation", 0.0)
        actions = {
            "ubi_implemented": False,
            "ubi_amount": 0.0,
            "regulation_change": 0.0,
            "emergency_spending": 0.0,
        }

        # --- UBI Decision ---
        # Triggers: unemployment > 12% AND protests in > 5 cities
        # OR: unemployment > 8% AND progressive government
        ubi_trigger_hard = unemployment_rate > 0.12 and protest_cities > 5
        ubi_trigger_soft = unemployment_rate > 0.08 and orientation < -0.3

        if not self.ubi_active and (ubi_trigger_hard or ubi_trigger_soft):
            # Implement UBI with probability based on severity
            ubi_prob = 0.10 if ubi_trigger_soft else 0.25
            if rng.random() < ubi_prob:
                self.ubi_active = True
                # Amount scales with unemployment severity
                base_amount = 1_000 + 500 * max(0, (unemployment_rate - 0.08) / 0.04)
                self.ubi_monthly_amount = min(base_amount, 2_500)
                actions["ubi_implemented"] = True

        if self.ubi_active:
            actions["ubi_amount"] = self.ubi_monthly_amount
            # Increase amount if conditions worsen
            if unemployment_rate > 0.15 and self.ubi_monthly_amount < 2_500:
                self.ubi_monthly_amount += 100

        # --- Regulation Decision ---
        # Regulate if AI adoption is fast and social costs are high
        regulation_pressure = (
            0.3 * max(0, unemployment_rate - 0.06)
            + 0.2 * max(0, ai_adoption_pct - 0.5)
            + 0.1 * (protest_cities / 10)
            + 0.2 * max(0, -orientation)  # Progressive governments regulate more
        )

        if rng.random() < regulation_pressure * 0.05:
            self.regulation_level = min(1.0, self.regulation_level + 0.1)
            actions["regulation_change"] = 0.1

        # --- Emergency fiscal response ---
        if unemployment_rate > 0.10:
            self.emergency_spending = 100_000_000_000 * (unemployment_rate - 0.10)  # $100B per pct point
            actions["emergency_spending"] = self.emergency_spending

        return actions


class AICompanyAgent(BaseAgent):
    """
    AI company agent (OpenAI, Google, Anthropic, etc.)

    Properties:
        company_name: str
        compute_capacity_flops: float
        researcher_count: int
        funding_billion: float
    """

    def __init__(self, agent_id: int, **kwargs):
        super().__init__(agent_id, "ai_company", **kwargs)
        self.capability_level = kwargs.get("capability_level", 0.5)
        self.breakthrough_achieved = False

    def step(self, env: Dict[str, Any]) -> Dict[str, Any]:
        """
        AI company behavior:
        1. Invest in R&D
        2. Roll for breakthrough (Monte Carlo)
        3. Deploy models, capture revenue
        4. AGI race dynamics
        """
        rng: np.random.Generator = env.get("rng", np.random.default_rng())
        month = env.get("month", 1)
        regulation_level = env.get("regulation_level", 0.0)

        compute = self.properties.get("compute_capacity_flops", 1e24)
        researchers = self.properties.get("researcher_count", 500)
        funding = self.properties.get("funding_billion", 10)

        actions = {"breakthrough": False, "capability_improvement": 0, "revenue": 0}

        # Capability improvement: function of compute, researchers, funding
        improvement_rate = (
            0.01 * np.log10(compute / 1e24)
            + 0.005 * (researchers / 1000)
            + 0.003 * funding
        ) * (1 - 0.3 * regulation_level)  # Regulation slows progress

        self.capability_level = min(1.0, self.capability_level + improvement_rate)
        actions["capability_improvement"] = improvement_rate

        # Breakthrough probability (logistic based on capability)
        if not self.breakthrough_achieved and self.capability_level > 0.8:
            breakthrough_prob = 0.02 * (self.capability_level - 0.8) / 0.2
            if rng.random() < breakthrough_prob:
                self.breakthrough_achieved = True
                actions["breakthrough"] = True

        # Revenue: grows with capability and market adoption
        base_revenue = funding * 1e9 * 0.1  # 10% of funding as revenue baseline
        actions["revenue"] = base_revenue * (1 + self.capability_level)

        # Scale compute
        self.properties["compute_capacity_flops"] = compute * 1.05  # 5% monthly growth

        return actions
