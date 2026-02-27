"""
Portfolio optimization engine.

Implements:
- Mean-Variance Optimization (Markowitz)
- Black-Litterman model with scenario-based views
- Risk parity allocation
- Kelly Criterion for position sizing
- Scenario-weighted optimal allocation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from config.settings import PortfolioConfig
from src.simulation.engine import SimulationResults
from src.models.scenarios import ScenarioParams, ScenarioTree


# ---------------------------------------------------------------------------
# Asset class definitions with expected characteristics
# ---------------------------------------------------------------------------

ASSET_PROFILES = {
    "ai_equities": {
        "label": "AI Pure-Play Equities",
        "base_return": 0.20,
        "base_vol": 0.35,
        "liquidity": 0.95,
    },
    "broad_equities": {
        "label": "Broad Market (S&P 500)",
        "base_return": 0.10,
        "base_vol": 0.18,
        "liquidity": 0.98,
    },
    "international_eq": {
        "label": "International Equities",
        "base_return": 0.08,
        "base_vol": 0.20,
        "liquidity": 0.90,
    },
    "treasuries": {
        "label": "US Treasuries",
        "base_return": 0.04,
        "base_vol": 0.06,
        "liquidity": 1.00,
    },
    "tips": {
        "label": "TIPS (Inflation-Protected)",
        "base_return": 0.035,
        "base_vol": 0.07,
        "liquidity": 0.95,
    },
    "corporate_bonds": {
        "label": "Corporate Bonds (IG+HY)",
        "base_return": 0.06,
        "base_vol": 0.10,
        "liquidity": 0.85,
    },
    "ai_hub_real_estate": {
        "label": "AI Hub Real Estate",
        "base_return": 0.15,
        "base_vol": 0.20,
        "liquidity": 0.30,
    },
    "reits": {
        "label": "Diversified REITs",
        "base_return": 0.08,
        "base_vol": 0.18,
        "liquidity": 0.80,
    },
    "gold": {
        "label": "Gold",
        "base_return": 0.06,
        "base_vol": 0.15,
        "liquidity": 0.95,
    },
    "uranium": {
        "label": "Uranium / Nuclear",
        "base_return": 0.12,
        "base_vol": 0.30,
        "liquidity": 0.60,
    },
    "copper": {
        "label": "Copper",
        "base_return": 0.08,
        "base_vol": 0.25,
        "liquidity": 0.70,
    },
    "bitcoin": {
        "label": "Bitcoin",
        "base_return": 0.15,
        "base_vol": 0.60,
        "liquidity": 0.90,
    },
    "private_ai": {
        "label": "Private AI Equity / VC",
        "base_return": 0.25,
        "base_vol": 0.40,
        "liquidity": 0.05,
    },
    "cash": {
        "label": "Cash / Money Market",
        "base_return": 0.045,
        "base_vol": 0.01,
        "liquidity": 1.00,
    },
}

# Base correlation matrix (simplified)
_ASSETS = list(ASSET_PROFILES.keys())
_N = len(_ASSETS)

# Build a reasonable correlation matrix
_BASE_CORR = np.eye(_N)
_idx = {a: i for i, a in enumerate(_ASSETS)}

def _set_corr(a1, a2, val):
    _BASE_CORR[_idx[a1], _idx[a2]] = val
    _BASE_CORR[_idx[a2], _idx[a1]] = val

# Equity correlations
_set_corr("ai_equities", "broad_equities", 0.75)
_set_corr("ai_equities", "international_eq", 0.55)
_set_corr("broad_equities", "international_eq", 0.70)
_set_corr("ai_equities", "private_ai", 0.80)

# Bond-equity inverse
_set_corr("treasuries", "broad_equities", -0.30)
_set_corr("treasuries", "ai_equities", -0.25)
_set_corr("tips", "broad_equities", -0.10)

# Gold as hedge
_set_corr("gold", "broad_equities", -0.15)
_set_corr("gold", "ai_equities", -0.10)
_set_corr("gold", "treasuries", 0.30)
_set_corr("gold", "bitcoin", 0.20)

# Real estate
_set_corr("ai_hub_real_estate", "ai_equities", 0.50)
_set_corr("reits", "broad_equities", 0.60)

# Bitcoin
_set_corr("bitcoin", "ai_equities", 0.35)
_set_corr("bitcoin", "broad_equities", 0.25)

# Commodities
_set_corr("copper", "broad_equities", 0.40)
_set_corr("uranium", "ai_equities", 0.30)


# ---------------------------------------------------------------------------
# Portfolio Optimizer
# ---------------------------------------------------------------------------

@dataclass
class PortfolioAllocation:
    """Result of portfolio optimization."""
    weights: Dict[str, float]
    expected_return: float
    expected_volatility: float
    sharpe_ratio: float
    scenario: str
    method: str
    var_95: float = 0.0       # Value at Risk (95%)
    cvar_95: float = 0.0      # Conditional VaR (95%)
    max_drawdown: float = 0.0
    rebalancing_triggers: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "weights": self.weights,
            "expected_return": self.expected_return,
            "expected_volatility": self.expected_volatility,
            "sharpe_ratio": self.sharpe_ratio,
            "scenario": self.scenario,
            "method": self.method,
            "var_95": self.var_95,
            "cvar_95": self.cvar_95,
            "max_drawdown": self.max_drawdown,
            "rebalancing_triggers": self.rebalancing_triggers,
        }


class PortfolioOptimizer:
    """
    Multi-method portfolio optimizer with scenario-based views.
    """

    def __init__(self, config: Optional[PortfolioConfig] = None):
        self.config = config or PortfolioConfig()
        self.assets = list(ASSET_PROFILES.keys())
        self.n_assets = len(self.assets)
        self.base_corr = _BASE_CORR.copy()

    def optimize(
        self,
        scenario: ScenarioParams,
        sim_results: Optional[SimulationResults] = None,
        method: str = "mean_variance",
        capital: float = 500_000,
        risk_tolerance: str = "moderate",
    ) -> PortfolioAllocation:
        """
        Optimize portfolio allocation for a scenario.

        Methods: 'mean_variance', 'risk_parity', 'black_litterman', 'kelly'
        """
        # Build expected returns and covariance for this scenario
        expected_returns = self._scenario_expected_returns(scenario, sim_results)
        cov_matrix = self._scenario_covariance(scenario)

        if method == "mean_variance":
            weights = self._mean_variance_optimize(expected_returns, cov_matrix, risk_tolerance)
        elif method == "risk_parity":
            weights = self._risk_parity(cov_matrix)
        elif method == "black_litterman":
            weights = self._black_litterman(expected_returns, cov_matrix, scenario)
        elif method == "kelly":
            weights = self._kelly_criterion(expected_returns, cov_matrix)
        else:
            weights = self._mean_variance_optimize(expected_returns, cov_matrix, risk_tolerance)

        # Apply constraints
        weights = self._apply_constraints(weights, risk_tolerance)

        # Calculate metrics
        w = np.array([weights.get(a, 0) for a in self.assets])
        mu = np.array([expected_returns[a] for a in self.assets])
        exp_ret = float(w @ mu)
        exp_vol = float(np.sqrt(w @ cov_matrix @ w))
        sharpe = (exp_ret - self.config.risk_free_rate) / exp_vol if exp_vol > 0 else 0

        # Risk metrics
        var_95 = capital * (exp_ret - 1.645 * exp_vol)
        cvar_95 = capital * (exp_ret - 2.063 * exp_vol)  # Approximate CVaR

        # Rebalancing triggers
        triggers = self._generate_rebalancing_triggers(scenario, weights)

        return PortfolioAllocation(
            weights={k: round(v, 4) for k, v in weights.items() if v > 0.001},
            expected_return=round(exp_ret, 4),
            expected_volatility=round(exp_vol, 4),
            sharpe_ratio=round(sharpe, 3),
            scenario=scenario.code,
            method=method,
            var_95=round(var_95, 0),
            cvar_95=round(cvar_95, 0),
            rebalancing_triggers=triggers,
        )

    def optimize_all_scenarios(
        self,
        all_results: Dict[str, SimulationResults],
        method: str = "mean_variance",
        capital: float = 500_000,
        risk_tolerance: str = "moderate",
    ) -> Dict[str, PortfolioAllocation]:
        """Optimize for all scenarios."""
        allocations = {}
        for code, result in all_results.items():
            allocations[code] = self.optimize(
                result.scenario, result, method, capital, risk_tolerance
            )
        return allocations

    def scenario_weighted_allocation(
        self,
        allocations: Dict[str, PortfolioAllocation],
    ) -> Dict[str, float]:
        """
        Compute probability-weighted allocation across all scenarios.
        """
        tree = ScenarioTree()
        weighted = {a: 0.0 for a in self.assets}

        for code, alloc in allocations.items():
            prob = tree.get(code).probability
            for asset, weight in alloc.weights.items():
                if asset in weighted:
                    weighted[asset] += prob * weight

        # Normalize
        total = sum(weighted.values())
        if total > 0:
            weighted = {k: round(v / total, 4) for k, v in weighted.items()}

        return {k: v for k, v in weighted.items() if v > 0.001}

    # ------------------------------------------------------------------
    # Optimization methods
    # ------------------------------------------------------------------

    def _mean_variance_optimize(
        self,
        expected_returns: Dict[str, float],
        cov_matrix: np.ndarray,
        risk_tolerance: str,
    ) -> Dict[str, float]:
        """Markowitz mean-variance optimization."""
        mu = np.array([expected_returns[a] for a in self.assets])
        n = self.n_assets

        # Risk aversion parameter
        risk_aversion = {"conservative": 5.0, "moderate": 2.5, "aggressive": 1.0,
                         "moderate-aggressive": 1.5}.get(risk_tolerance, 2.5)

        # Objective: maximize (return - risk_aversion * variance)
        def objective(w):
            ret = w @ mu
            vol = np.sqrt(w @ cov_matrix @ w)
            return -(ret - risk_aversion * vol ** 2)

        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},  # Weights sum to 1
        ]
        bounds = [(0, 0.40) for _ in range(n)]  # Max 40% per asset

        x0 = np.ones(n) / n
        result = minimize(objective, x0, method="SLSQP", bounds=bounds, constraints=constraints)

        if result.success:
            weights = result.x
        else:
            weights = x0  # Fallback to equal weight

        return {a: float(w) for a, w in zip(self.assets, weights)}

    def _risk_parity(self, cov_matrix: np.ndarray) -> Dict[str, float]:
        """Risk parity: equal risk contribution from each asset."""
        n = self.n_assets

        def risk_contribution(w):
            port_vol = np.sqrt(w @ cov_matrix @ w)
            if port_vol < 1e-10:
                return np.ones(n) / n
            marginal = cov_matrix @ w / port_vol
            return w * marginal

        def objective(w):
            rc = risk_contribution(w)
            target = np.mean(rc)
            return np.sum((rc - target) ** 2)

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        bounds = [(0.02, 0.30) for _ in range(n)]
        x0 = np.ones(n) / n

        result = minimize(objective, x0, method="SLSQP", bounds=bounds, constraints=constraints)
        weights = result.x if result.success else x0

        return {a: float(w) for a, w in zip(self.assets, weights)}

    def _black_litterman(
        self,
        expected_returns: Dict[str, float],
        cov_matrix: np.ndarray,
        scenario: ScenarioParams,
    ) -> Dict[str, float]:
        """
        Black-Litterman model: blend market equilibrium with scenario views.
        """
        n = self.n_assets
        tau = 0.05  # Scaling factor

        # Market implied equilibrium returns (from market cap weights)
        market_weights = np.array([
            0.15, 0.30, 0.10, 0.10, 0.03, 0.05,
            0.05, 0.05, 0.05, 0.02, 0.02, 0.03, 0.03, 0.02,
        ])
        market_weights = market_weights[:n]
        market_weights /= market_weights.sum()

        risk_aversion = 2.5
        pi = risk_aversion * cov_matrix @ market_weights  # Equilibrium returns

        # Scenario views (P matrix and Q vector)
        views = self._scenario_views(scenario)
        if not views:
            return {a: float(w) for a, w in zip(self.assets, market_weights)}

        P = np.zeros((len(views), n))
        Q = np.zeros(len(views))
        omega_diag = []

        for i, (view_assets, view_return, confidence) in enumerate(views):
            for asset, direction in view_assets:
                if asset in self.assets:
                    P[i, self.assets.index(asset)] = direction
            Q[i] = view_return
            omega_diag.append((1 - confidence) / confidence * 0.01)

        Omega = np.diag(omega_diag)

        # Black-Litterman formula
        tau_sigma = tau * cov_matrix
        M1 = np.linalg.inv(tau_sigma)
        M2 = P.T @ np.linalg.inv(Omega) @ P
        bl_returns = np.linalg.inv(M1 + M2) @ (M1 @ pi + P.T @ np.linalg.inv(Omega) @ Q)

        # Optimize with BL returns
        mu_bl = {a: float(bl_returns[i]) for i, a in enumerate(self.assets)}
        return self._mean_variance_optimize(mu_bl, cov_matrix, "moderate")

    def _kelly_criterion(
        self,
        expected_returns: Dict[str, float],
        cov_matrix: np.ndarray,
    ) -> Dict[str, float]:
        """Kelly Criterion: size bets based on edge and confidence."""
        mu = np.array([expected_returns[a] for a in self.assets])
        # Full Kelly: w = Sigma^{-1} * (mu - rf)
        try:
            inv_cov = np.linalg.inv(cov_matrix)
            excess = mu - self.config.risk_free_rate
            kelly_weights = inv_cov @ excess
            # Half Kelly for safety
            kelly_weights *= 0.5
            # Clip and normalize
            kelly_weights = np.clip(kelly_weights, 0, None)
            total = kelly_weights.sum()
            if total > 0:
                kelly_weights /= total
            else:
                kelly_weights = np.ones(self.n_assets) / self.n_assets
        except np.linalg.LinAlgError:
            kelly_weights = np.ones(self.n_assets) / self.n_assets

        return {a: float(w) for a, w in zip(self.assets, kelly_weights)}

    # ------------------------------------------------------------------
    # Scenario adjustments
    # ------------------------------------------------------------------

    def _scenario_expected_returns(
        self,
        scenario: ScenarioParams,
        sim_results: Optional[SimulationResults],
    ) -> Dict[str, float]:
        """Adjust expected returns based on scenario."""
        returns = {}
        for asset, profile in ASSET_PROFILES.items():
            base = profile["base_return"]

            if "equit" in asset:
                mult = scenario.tech_equity_multiplier if "ai" in asset else 1.0
                base *= mult ** 0.2  # Annualized

            if "gold" in asset:
                # Gold benefits from uncertainty and inflation
                base *= (1 + scenario.peak_unemployment_rate)

            if "real_estate" in asset or "reit" in asset:
                if "ai_hub" in asset:
                    base = max(base, scenario.ai_hub_re_appreciation)
                else:
                    base *= (1 + scenario.at_risk_re_depreciation)

            if "bitcoin" in asset:
                base *= (1 + 0.5 * scenario.peak_unemployment_rate)  # Fear hedge

            if "uranium" in asset or "copper" in asset:
                base *= scenario.ai_progress_rate  # AI infrastructure demand

            returns[asset] = base

        return returns

    def _scenario_covariance(self, scenario: ScenarioParams) -> np.ndarray:
        """Adjust covariance matrix for scenario."""
        vols = np.array([ASSET_PROFILES[a]["base_vol"] for a in self.assets])

        # Increase volatilities in disruptive scenarios
        disruption = scenario.ai_progress_rate * scenario.peak_unemployment_rate * 10
        vol_mult = 1 + 0.2 * min(disruption, 3)
        vols *= vol_mult

        # Covariance = D * Corr * D where D is diagonal volatility matrix
        D = np.diag(vols)
        cov = D @ self.base_corr @ D

        return cov

    def _scenario_views(
        self,
        scenario: ScenarioParams,
    ) -> List[Tuple[List[Tuple[str, float]], float, float]]:
        """
        Generate Black-Litterman views from scenario.

        Returns: list of ([(asset, direction)], expected_return, confidence)
        """
        views = []

        # View 1: AI equities outperform
        if scenario.ai_progress_rate > 1.0:
            views.append((
                [("ai_equities", 1.0), ("broad_equities", -1.0)],
                0.10 * scenario.ai_progress_rate,
                0.7,
            ))

        # View 2: Gold benefits from disruption
        if scenario.peak_unemployment_rate > 0.08:
            views.append((
                [("gold", 1.0)],
                0.10,
                0.6,
            ))

        # View 3: AI hub real estate appreciates
        if scenario.ai_hub_re_appreciation > 0.15:
            views.append((
                [("ai_hub_real_estate", 1.0), ("reits", -1.0)],
                scenario.ai_hub_re_appreciation - 0.05,
                0.5,
            ))

        # View 4: Treasuries benefit from flight to safety
        if scenario.peak_unemployment_rate > 0.10:
            views.append((
                [("treasuries", 1.0)],
                0.05,
                0.6,
            ))

        return views

    def _apply_constraints(
        self,
        weights: Dict[str, float],
        risk_tolerance: str,
    ) -> Dict[str, float]:
        """Apply liquidity and concentration constraints."""
        # Ensure minimum cash/liquid allocation
        liquid_assets = ["cash", "treasuries", "broad_equities", "gold"]
        liquid_weight = sum(weights.get(a, 0) for a in liquid_assets)
        if liquid_weight < self.config.min_liquidity:
            deficit = self.config.min_liquidity - liquid_weight
            weights["cash"] = weights.get("cash", 0) + deficit
            # Reduce illiquid proportionally
            illiquid = [a for a in weights if a not in liquid_assets]
            total_illiquid = sum(weights.get(a, 0) for a in illiquid)
            if total_illiquid > 0:
                for a in illiquid:
                    weights[a] = weights.get(a, 0) * (1 - deficit / total_illiquid)

        # Normalize
        total = sum(max(0, v) for v in weights.values())
        if total > 0:
            weights = {k: max(0, v) / total for k, v in weights.items()}

        return weights

    def _generate_rebalancing_triggers(
        self,
        scenario: ScenarioParams,
        weights: Dict[str, float],
    ) -> List[Dict[str, Any]]:
        """Generate scenario-specific rebalancing triggers."""
        triggers = []

        triggers.append({
            "condition": "unemployment_rate > 0.06",
            "action": "Increase gold allocation to 20%, reduce equities",
            "urgency": "medium",
        })
        triggers.append({
            "condition": "unemployment_rate > 0.10",
            "action": "Increase gold to 30%, increase cash to 15%, reduce AI equities",
            "urgency": "high",
        })
        triggers.append({
            "condition": "VIX > 40",
            "action": "Reduce all equities by 20%, increase treasuries and gold",
            "urgency": "high",
        })
        triggers.append({
            "condition": "AI benchmark breakthrough (SWE-bench > 95%)",
            "action": "Increase AI equities allocation by 10%",
            "urgency": "medium",
        })
        triggers.append({
            "condition": "UBI announced",
            "action": "Increase TIPS allocation (inflation hedge), reduce treasuries",
            "urgency": "medium",
        })

        if scenario.ai_progress_rate > 1.2:
            triggers.append({
                "condition": "NVIDIA earnings beat > 20%",
                "action": "Increase AI equities by 5%, funded from cash",
                "urgency": "low",
            })

        return triggers
