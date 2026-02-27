"""
Financial markets model: equities, fixed income, commodities, real estate.

Projects asset class returns under different AI scenarios.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.models.scenarios import ScenarioParams


# ---------------------------------------------------------------------------
# Sector definitions
# ---------------------------------------------------------------------------

EQUITY_SECTORS = {
    "technology":    {"ticker": "XLK", "ai_beta": 2.5, "base_return": 0.12},
    "financials":    {"ticker": "XLF", "ai_beta": 0.5, "base_return": 0.08},
    "industrials":   {"ticker": "XLI", "ai_beta": 1.0, "base_return": 0.07},
    "energy":        {"ticker": "XLE", "ai_beta": 0.8, "base_return": 0.06},
    "healthcare":    {"ticker": "XLV", "ai_beta": 1.2, "base_return": 0.09},
    "consumer_disc": {"ticker": "XLY", "ai_beta": 0.6, "base_return": 0.07},
    "consumer_stap": {"ticker": "XLP", "ai_beta": 0.3, "base_return": 0.05},
    "utilities":     {"ticker": "XLU", "ai_beta": 0.7, "base_return": 0.05},
    "real_estate":   {"ticker": "XLRE", "ai_beta": 0.4, "base_return": 0.06},
    "materials":     {"ticker": "XLB", "ai_beta": 0.6, "base_return": 0.06},
    "communication": {"ticker": "XLC", "ai_beta": 1.5, "base_return": 0.08},
}

AI_STOCKS = {
    "NVIDIA":    {"ai_beta": 4.0, "base_return": 0.25},
    "Microsoft": {"ai_beta": 2.5, "base_return": 0.15},
    "Google":    {"ai_beta": 2.5, "base_return": 0.14},
    "Tesla":     {"ai_beta": 3.0, "base_return": 0.20},
    "TSMC":      {"ai_beta": 3.5, "base_return": 0.18},
    "Palantir":  {"ai_beta": 3.0, "base_return": 0.15},
    "OpenAI*":   {"ai_beta": 5.0, "base_return": 0.30},  # Private
    "Anthropic*": {"ai_beta": 4.5, "base_return": 0.28},  # Private
}

COMMODITIES = {
    "gold":        {"base_price": 2950, "ai_fear_beta": 1.5, "unit": "$/oz"},
    "copper":      {"base_price": 4.50, "ai_infra_beta": 1.2, "unit": "$/lb"},
    "uranium":     {"base_price": 85, "ai_energy_beta": 2.0, "unit": "$/lb"},
    "natural_gas": {"base_price": 3.2, "ai_energy_beta": 1.5, "unit": "$/MMBtu"},
    "lithium":     {"base_price": 15000, "ai_infra_beta": 1.0, "unit": "$/ton"},
}

# Geographic real estate baseline appreciation rates
METRO_RE_PROFILES = {
    "SF Bay Area":     {"base_appreciation": 0.05, "ai_hub": True, "ai_beta": 2.0},
    "Seattle":         {"base_appreciation": 0.04, "ai_hub": True, "ai_beta": 1.8},
    "Austin":          {"base_appreciation": 0.06, "ai_hub": True, "ai_beta": 1.5},
    "Boston":          {"base_appreciation": 0.03, "ai_hub": True, "ai_beta": 1.4},
    "NYC":             {"base_appreciation": 0.02, "ai_hub": True, "ai_beta": 1.2},
    "Irvine":          {"base_appreciation": 0.04, "ai_hub": False, "ai_beta": 1.0},
    "Los Angeles":     {"base_appreciation": 0.03, "ai_hub": False, "ai_beta": 0.8},
    "Phoenix":         {"base_appreciation": 0.04, "ai_hub": False, "ai_beta": 0.6},
    "Detroit":         {"base_appreciation": 0.01, "ai_hub": False, "ai_beta": -0.5},
    "Cleveland":       {"base_appreciation": 0.00, "ai_hub": False, "ai_beta": -0.6},
    "Omaha":           {"base_appreciation": 0.02, "ai_hub": False, "ai_beta": -0.3},
    "Tampa":           {"base_appreciation": 0.03, "ai_hub": False, "ai_beta": -0.2},
}


class FinancialMarketsModel:
    """Projects financial asset returns under AI scenarios."""

    def __init__(self):
        pass

    # ------------------------------------------------------------------
    # Equity projections
    # ------------------------------------------------------------------

    def project_sector_returns(
        self,
        scenario: ScenarioParams,
        months: int = 60,
        rng: Optional[np.random.Generator] = None,
    ) -> pd.DataFrame:
        """
        Project monthly sector returns.

        Returns:
            DataFrame with columns: month, sector, monthly_return, cumulative_return
        """
        if rng is None:
            rng = np.random.default_rng(42)

        agi_month = self._q_to_m(scenario.agi_arrival_quarter)
        asi_month = self._q_to_m(scenario.asi_arrival_quarter) if scenario.asi_arrival_quarter else None

        records = []
        for sector, params in EQUITY_SECTORS.items():
            cum_return = 1.0
            for m in range(1, months + 1):
                # Base monthly return
                base_monthly = params["base_return"] / 12

                # AI boost/drag based on scenario
                ai_factor = self._ai_return_factor(
                    m, agi_month, asi_month, scenario, params["ai_beta"]
                )

                # Volatility scales with AI disruption
                vol = 0.04 * (1 + 0.5 * abs(ai_factor))
                noise = rng.normal(0, vol)

                monthly_ret = base_monthly * ai_factor + noise
                cum_return *= (1 + monthly_ret)

                year = 2026 + (m - 1) // 12
                records.append({
                    "month": m,
                    "year": year,
                    "sector": sector,
                    "monthly_return": round(monthly_ret, 5),
                    "cumulative_return": round(cum_return, 4),
                    "annualized_vol": round(vol * np.sqrt(12), 4),
                })

        return pd.DataFrame(records)

    def project_ai_stock_returns(
        self,
        scenario: ScenarioParams,
        months: int = 60,
        rng: Optional[np.random.Generator] = None,
    ) -> pd.DataFrame:
        """Project individual AI stock returns."""
        if rng is None:
            rng = np.random.default_rng(42)

        agi_month = self._q_to_m(scenario.agi_arrival_quarter)
        asi_month = self._q_to_m(scenario.asi_arrival_quarter) if scenario.asi_arrival_quarter else None

        records = []
        for stock, params in AI_STOCKS.items():
            cum_return = 1.0
            for m in range(1, months + 1):
                base_monthly = params["base_return"] / 12
                ai_factor = self._ai_return_factor(
                    m, agi_month, asi_month, scenario, params["ai_beta"]
                )
                vol = 0.08 * (1 + 0.3 * abs(ai_factor))  # Higher vol for individual stocks
                noise = rng.normal(0, vol)
                monthly_ret = base_monthly * ai_factor + noise
                cum_return *= (1 + monthly_ret)
                records.append({
                    "month": m,
                    "stock": stock,
                    "monthly_return": round(monthly_ret, 5),
                    "cumulative_return": round(cum_return, 4),
                })

        return pd.DataFrame(records)

    # ------------------------------------------------------------------
    # Fixed income
    # ------------------------------------------------------------------

    def project_bond_yields(
        self,
        scenario: ScenarioParams,
        months: int = 60,
        rng: Optional[np.random.Generator] = None,
    ) -> pd.DataFrame:
        """Project Treasury and corporate bond yields."""
        if rng is None:
            rng = np.random.default_rng(42)

        agi_month = self._q_to_m(scenario.agi_arrival_quarter)
        records = []
        tsy_10y = 0.043
        ig_spread = 0.012
        hy_spread = 0.035

        for m in range(1, months + 1):
            # Flight to safety if high unemployment
            if agi_month and m >= agi_month:
                months_post = m - agi_month
                safety_demand = 0.002 * min(months_post, 24)  # Gradually drives yields down
                tsy_10y = max(0.01, tsy_10y - safety_demand / 100 + rng.normal(0, 0.001))

                # Credit spreads widen with disruption
                ig_spread = min(0.04, ig_spread + 0.0003 * months_post / 12 + rng.normal(0, 0.0005))
                hy_spread = min(0.10, hy_spread + 0.001 * months_post / 12 + rng.normal(0, 0.001))
            else:
                tsy_10y += rng.normal(0, 0.001)
                ig_spread += rng.normal(0, 0.0003)
                hy_spread += rng.normal(0, 0.0005)

            records.append({
                "month": m,
                "treasury_10y": round(tsy_10y, 4),
                "ig_spread": round(ig_spread, 4),
                "hy_spread": round(hy_spread, 4),
                "ig_yield": round(tsy_10y + ig_spread, 4),
                "hy_yield": round(tsy_10y + hy_spread, 4),
            })

        return pd.DataFrame(records)

    # ------------------------------------------------------------------
    # Commodities
    # ------------------------------------------------------------------

    def project_commodities(
        self,
        scenario: ScenarioParams,
        months: int = 60,
        rng: Optional[np.random.Generator] = None,
    ) -> pd.DataFrame:
        """Project commodity prices."""
        if rng is None:
            rng = np.random.default_rng(42)

        agi_month = self._q_to_m(scenario.agi_arrival_quarter)
        records = []

        prices = {k: v["base_price"] for k, v in COMMODITIES.items()}

        for m in range(1, months + 1):
            for commodity, params in COMMODITIES.items():
                price = prices[commodity]

                if commodity == "gold":
                    # Gold: fear-driven + inflation hedge
                    fear_factor = 0.0
                    if agi_month and m >= agi_month:
                        fear_factor = params["ai_fear_beta"] * 0.01 * scenario.ai_progress_rate
                    drift = 0.002 + fear_factor
                else:
                    # Infrastructure commodities: AI demand
                    ai_demand = 0.0
                    beta_key = "ai_infra_beta" if "ai_infra_beta" in params else "ai_energy_beta"
                    if agi_month and m >= agi_month:
                        ai_demand = params[beta_key] * 0.005 * scenario.ai_progress_rate
                    drift = 0.001 + ai_demand

                vol = 0.06
                price *= np.exp(drift + rng.normal(0, vol / np.sqrt(12)))
                prices[commodity] = price

                records.append({
                    "month": m,
                    "commodity": commodity,
                    "price": round(price, 2),
                    "unit": params["unit"],
                })

        return pd.DataFrame(records)

    # ------------------------------------------------------------------
    # Real estate
    # ------------------------------------------------------------------

    def project_real_estate(
        self,
        scenario: ScenarioParams,
        months: int = 60,
        rng: Optional[np.random.Generator] = None,
    ) -> pd.DataFrame:
        """Project metro-level real estate price indices (base=100)."""
        if rng is None:
            rng = np.random.default_rng(42)

        agi_month = self._q_to_m(scenario.agi_arrival_quarter)
        records = []
        indices = {metro: 100.0 for metro in METRO_RE_PROFILES}

        for m in range(1, months + 1):
            for metro, params in METRO_RE_PROFILES.items():
                idx = indices[metro]

                # Base appreciation
                monthly_base = params["base_appreciation"] / 12

                # AI impact
                ai_effect = 0.0
                if agi_month and m >= agi_month:
                    if params["ai_hub"]:
                        ai_effect = (
                            scenario.ai_hub_re_appreciation
                            * params["ai_beta"]
                            / 12
                            * min((m - agi_month) / 24, 1.0)  # Ramp up
                        )
                    else:
                        ai_effect = (
                            scenario.at_risk_re_depreciation
                            * abs(params["ai_beta"])
                            / 12
                            * min((m - agi_month) / 24, 1.0)
                        )

                vol = 0.01
                monthly_change = monthly_base + ai_effect + rng.normal(0, vol)
                idx *= (1 + monthly_change)
                indices[metro] = idx

                records.append({
                    "month": m,
                    "metro": metro,
                    "price_index": round(idx, 2),
                    "is_ai_hub": params["ai_hub"],
                    "monthly_change": round(monthly_change, 5),
                })

        return pd.DataFrame(records)

    # ------------------------------------------------------------------
    # VIX / volatility
    # ------------------------------------------------------------------

    def project_vix(
        self,
        scenario: ScenarioParams,
        months: int = 60,
        rng: Optional[np.random.Generator] = None,
    ) -> pd.DataFrame:
        """Project VIX (volatility index)."""
        if rng is None:
            rng = np.random.default_rng(42)

        agi_month = self._q_to_m(scenario.agi_arrival_quarter)
        records = []
        vix = 18.0  # Baseline VIX

        for m in range(1, months + 1):
            # Mean-reverting process with AI shock
            mean_vix = 18.0
            if agi_month and m >= agi_month:
                months_post = m - agi_month
                # VIX spikes on AGI announcement, then elevated
                spike = 25 * np.exp(-0.1 * months_post)
                elevated = 5 * scenario.ai_progress_rate
                mean_vix = 18 + spike + elevated

            vix += 0.1 * (mean_vix - vix) + rng.normal(0, 2)
            vix = max(10, min(80, vix))

            records.append({"month": m, "vix": round(vix, 1)})

        return pd.DataFrame(records)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _q_to_m(quarter_str: Optional[str]) -> Optional[int]:
        if not quarter_str or quarter_str == "never":
            return None
        try:
            year, q = quarter_str.split("-Q")
            return (int(year) - 2026) * 12 + (int(q) - 1) * 3 + 1
        except (ValueError, AttributeError):
            return None

    @staticmethod
    def _ai_return_factor(
        month: int,
        agi_month: Optional[int],
        asi_month: Optional[int],
        scenario: ScenarioParams,
        ai_beta: float,
    ) -> float:
        """Compute AI-driven return multiplier for a given month."""
        factor = 1.0

        if agi_month and month >= agi_month:
            months_post = month - agi_month
            # Initial hype / repricing
            hype = 1 + ai_beta * 0.3 * np.exp(-0.05 * months_post)
            # Sustained growth from productivity
            sustained = 1 + ai_beta * 0.1 * scenario.ai_progress_rate * min(months_post / 36, 1.0)
            factor = hype * sustained * scenario.tech_equity_multiplier ** (1/60)

        if asi_month and month >= asi_month:
            months_post_asi = month - asi_month
            asi_boost = 1 + ai_beta * 0.2 * (1 - np.exp(-0.03 * months_post_asi))
            factor *= asi_boost

        return factor
