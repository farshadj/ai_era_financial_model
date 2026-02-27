"""
Worker agents: Representative members of the US labor force.

Each worker agent represents ~15,000 real workers (scale factor from config).
Workers have occupations with AI/robot vulnerability and make decisions about
job search, consumption, savings, and migration.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from src.agents.base import BaseAgent


class WorkerAgent(BaseAgent):
    """
    Worker agent in the AI economy simulation.

    Properties:
        occupation: str
        skill_level: float (0-1)
        age: int
        geography: str
        ai_vulnerability: float (0-1)
        robot_vulnerability: float (0-1)
        annual_wage: float
        can_retrain: bool
    """

    def __init__(self, agent_id: int, **kwargs):
        super().__init__(agent_id, "worker", **kwargs)
        self.months_unemployed = 0
        self.retrained = False

    def step(self, env: Dict[str, Any]) -> Dict[str, Any]:
        """
        Worker behavior per month:
        1. Check if displaced by AI/robots
        2. If unemployed: job search, retrain, or leave labor force
        3. Consume and save based on income
        """
        rng: np.random.Generator = env.get("rng", np.random.default_rng())
        ai_adoption = env.get("ai_adoption_pct", 0.15)
        unemployment_rate = env.get("unemployment_rate", 0.04)
        ubi_amount = env.get("ubi_monthly_amount", 0.0)

        actions = {"displaced": False, "migrated": False, "retrained": False, "consumption": 0}

        # --- Displacement check ---
        if self.state.employed:
            vuln = self.properties.get("ai_vulnerability", 0.0) if self.properties.get("category") == "cognitive" \
                else self.properties.get("robot_vulnerability", 0.0)

            # Probability of displacement increases with AI adoption
            displacement_prob = vuln * ai_adoption * 0.02  # Monthly probability
            if rng.random() < displacement_prob:
                self.state.employed = False
                self.months_unemployed = 0
                actions["displaced"] = True

        # --- Unemployed behavior ---
        if not self.state.employed:
            self.months_unemployed += 1

            # Job search: harder in high-unemployment environment
            rehire_prob = max(0.02, 0.10 - 0.5 * unemployment_rate)

            # Retrain if unemployed > 6 months and has skill
            if self.months_unemployed > 6 and not self.retrained:
                skill = self.properties.get("skill_level", 0.5)
                age = self.properties.get("age", 35)
                retrain_prob = 0.05 * skill * max(0, (60 - age) / 30)
                if rng.random() < retrain_prob:
                    self.retrained = True
                    rehire_prob *= 2.0  # Retrained workers find jobs easier
                    actions["retrained"] = True

            if rng.random() < rehire_prob:
                self.state.employed = True
                self.months_unemployed = 0
                # Retrained workers may earn less initially
                if self.retrained:
                    self.state.income = self.properties.get("annual_wage", 50_000) / 12 * 0.8
                else:
                    self.state.income = self.properties.get("annual_wage", 50_000) / 12

            # Migration: if unemployed > 12 months, consider moving to AI hub
            if self.months_unemployed > 12:
                migration_prob = 0.02
                if rng.random() < migration_prob:
                    self.properties["geography"] = "AI_hub"
                    actions["migrated"] = True

        # --- Income ---
        if self.state.employed:
            self.state.income = self.properties.get("annual_wage", 50_000) / 12
        else:
            # Unemployment insurance + UBI
            self.state.income = 2_000 + ubi_amount  # Basic UI + UBI if any

        # --- Consumption and savings ---
        if self.state.employed:
            # Employed: save more when fearful
            fear = 1 - self.state.sentiment
            self.state.savings_rate = 0.10 + 0.15 * fear
        else:
            self.state.savings_rate = 0.0  # Spend everything when unemployed

        self.state.consumption = self.state.income * (1 - self.state.savings_rate)
        self.state.wealth += self.state.income * self.state.savings_rate

        # Sentiment update
        if self.state.employed:
            self.state.sentiment = min(1, self.state.sentiment + 0.02)
        else:
            self.state.sentiment = max(0, self.state.sentiment - 0.05)

        actions["consumption"] = self.state.consumption
        return actions


def create_worker_population(
    n_agents: int,
    occupations: List[Dict[str, Any]],
    rng: Optional[np.random.Generator] = None,
) -> List[WorkerAgent]:
    """
    Create a representative population of worker agents.

    Distributes agents across occupations proportional to employment.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    total_employment = sum(o["us_employment"] for o in occupations)
    agents = []
    agent_id = 0

    for occ in occupations:
        n_from_occ = max(1, int(n_agents * occ["us_employment"] / total_employment))
        for _ in range(n_from_occ):
            age = int(rng.normal(40, 10))
            age = max(22, min(65, age))
            agents.append(WorkerAgent(
                agent_id=agent_id,
                occupation=occ["name"],
                skill_level=rng.uniform(0.2, 1.0),
                age=age,
                geography=rng.choice(["AI_hub", "Major_city", "Suburban", "Rural"],
                                      p=[0.15, 0.35, 0.35, 0.15]),
                ai_vulnerability=occ.get("ai_vulnerability", 0.5),
                robot_vulnerability=occ.get("robot_vulnerability", 0.3),
                annual_wage=occ.get("avg_annual_wage", 50_000) * rng.uniform(0.7, 1.3),
                category=occ.get("category", "mixed"),
                employed=True,
                wealth=float(rng.exponential(50_000)),
                sentiment=rng.uniform(0.3, 0.8),
            ))
            agent_id += 1
            if agent_id >= n_agents:
                break
        if agent_id >= n_agents:
            break

    return agents
