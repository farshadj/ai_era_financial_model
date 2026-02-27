"""
Investor agents: Asset allocation, rebalancing, and sentiment-driven behavior.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from src.agents.base import BaseAgent


RISK_PROFILES = [
    {"name": "conservative", "equity_target": 0.30, "gold_target": 0.15, "bond_target": 0.45},
    {"name": "moderate", "equity_target": 0.50, "gold_target": 0.10, "bond_target": 0.30},
    {"name": "aggressive", "equity_target": 0.70, "gold_target": 0.05, "bond_target": 0.15},
    {"name": "ai_believer", "equity_target": 0.80, "gold_target": 0.02, "bond_target": 0.08},
]


class InvestorAgent(BaseAgent):
    """
    Investor agent managing a portfolio.

    Properties:
        risk_profile: str
        sophistication: float (0-1)
        portfolio: dict of asset → value
    """

    def __init__(self, agent_id: int, **kwargs):
        super().__init__(agent_id, "investor", **kwargs)
        self.portfolio = kwargs.get("portfolio", {
            "ai_equities": 0, "broad_equities": 0, "bonds": 0,
            "gold": 0, "bitcoin": 0, "real_estate": 0, "cash": 0,
        })

    def step(self, env: Dict[str, Any]) -> Dict[str, Any]:
        """
        Investor behavior per month:
        1. Update portfolio values based on market returns
        2. Adjust sentiment based on news/unemployment
        3. Rebalance if triggers hit
        4. Panic sell if extreme fear
        """
        rng: np.random.Generator = env.get("rng", np.random.default_rng())
        unemployment_rate = env.get("unemployment_rate", 0.04)
        vix = env.get("vix", 18)
        ai_return = env.get("ai_equity_return", 0.01)
        broad_return = env.get("broad_equity_return", 0.007)
        bond_return = env.get("bond_return", 0.003)
        gold_return = env.get("gold_return", 0.002)
        btc_return = env.get("btc_return", 0.005)
        re_return = env.get("re_return", 0.003)

        # --- Update portfolio values ---
        self.portfolio["ai_equities"] *= (1 + ai_return)
        self.portfolio["broad_equities"] *= (1 + broad_return)
        self.portfolio["bonds"] *= (1 + bond_return)
        self.portfolio["gold"] *= (1 + gold_return)
        self.portfolio["bitcoin"] *= (1 + btc_return)
        self.portfolio["real_estate"] *= (1 + re_return)

        total_value = sum(self.portfolio.values())
        actions = {"total_value": total_value, "rebalanced": False, "panic_sold": False}

        if total_value <= 0:
            return actions

        # --- Sentiment update ---
        # Fear increases with unemployment and volatility
        fear_signal = (unemployment_rate - 0.04) * 5 + (vix - 18) / 50
        self.state.sentiment = np.clip(
            self.state.sentiment - 0.1 * fear_signal + rng.normal(0, 0.05),
            0, 1
        )

        # --- Panic selling (VIX > 40 or unemployment > 10%) ---
        if (vix > 40 or unemployment_rate > 0.10) and self.state.sentiment < 0.2:
            # Shift to safety: sell equities, buy gold/cash
            sell_amount = total_value * 0.1  # Sell 10% of portfolio
            equity_sell = min(sell_amount, self.portfolio["ai_equities"] + self.portfolio["broad_equities"])
            self.portfolio["ai_equities"] -= equity_sell * 0.6
            self.portfolio["broad_equities"] -= equity_sell * 0.4
            self.portfolio["gold"] += equity_sell * 0.5
            self.portfolio["cash"] += equity_sell * 0.5
            actions["panic_sold"] = True

        # --- Rebalance trigger: unemployment > 10% → shift to gold/Bitcoin ---
        elif unemployment_rate > 0.10 and not actions["panic_sold"]:
            if rng.random() < 0.05:  # 5% monthly probability of rebalancing
                gold_target = 0.25  # Increase gold
                current_gold_pct = self.portfolio["gold"] / total_value
                if current_gold_pct < gold_target:
                    shift = total_value * 0.05
                    self.portfolio["broad_equities"] -= min(shift, self.portfolio["broad_equities"])
                    self.portfolio["gold"] += shift
                    actions["rebalanced"] = True

        # --- Greed: shift to AI equities when sentiment high ---
        elif self.state.sentiment > 0.7 and unemployment_rate < 0.06:
            if rng.random() < 0.03:
                shift = total_value * 0.03
                self.portfolio["cash"] -= min(shift, self.portfolio["cash"])
                self.portfolio["ai_equities"] += shift
                actions["rebalanced"] = True

        actions["total_value"] = sum(self.portfolio.values())
        return actions


def create_investor_population(
    n_agents: int,
    rng: Optional[np.random.Generator] = None,
) -> List[InvestorAgent]:
    """Create representative investor population."""
    if rng is None:
        rng = np.random.default_rng(42)

    agents = []
    for i in range(n_agents):
        profile = rng.choice(RISK_PROFILES)
        wealth = float(rng.lognormal(12, 1.5))  # Log-normal wealth distribution
        wealth = max(10_000, min(wealth, 100_000_000))  # Cap at $100M

        portfolio = {
            "ai_equities": wealth * profile["equity_target"] * 0.3,
            "broad_equities": wealth * profile["equity_target"] * 0.7,
            "bonds": wealth * profile["bond_target"],
            "gold": wealth * profile["gold_target"],
            "bitcoin": wealth * 0.03,
            "real_estate": wealth * 0.15,
            "cash": wealth * (1 - profile["equity_target"] - profile["bond_target"]
                              - profile["gold_target"] - 0.03 - 0.15),
        }

        agents.append(InvestorAgent(
            agent_id=i,
            risk_profile=profile["name"],
            sophistication=rng.uniform(0.1, 1.0),
            portfolio=portfolio,
            wealth=wealth,
            sentiment=rng.uniform(0.3, 0.7),
        ))

    return agents
