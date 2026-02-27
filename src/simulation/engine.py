"""
Main simulation engine: orchestrates agents, models, and scenario execution.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config.settings import SimulationConfig, EconomicDefaults
from src.models.scenarios import ScenarioParams, ScenarioTree
from src.models.macro_economics import MacroEconomicsModel
from src.models.labor_market import LaborMarketModel, CRITICAL_OCCUPATIONS
from src.models.financial_markets import FinancialMarketsModel
from src.models.robot_deployment import RobotDeploymentModel
from src.models.ai_progress import AIProgressModel
from src.agents.workers import WorkerAgent, create_worker_population
from src.agents.firms import FirmAgent, create_firm_population
from src.agents.investors import InvestorAgent, create_investor_population
from src.agents.government import GovernmentAgent, AICompanyAgent


class SimulationResults:
    """Container for all simulation outputs."""

    def __init__(self, scenario: ScenarioParams):
        self.scenario = scenario
        self.macro: Optional[pd.DataFrame] = None
        self.labor: Optional[pd.DataFrame] = None
        self.labor_aggregate: Optional[pd.DataFrame] = None
        self.sectors: Optional[pd.DataFrame] = None
        self.ai_stocks: Optional[pd.DataFrame] = None
        self.etf_returns: Optional[pd.DataFrame] = None
        self.bonds: Optional[pd.DataFrame] = None
        self.commodities: Optional[pd.DataFrame] = None
        self.real_estate: Optional[pd.DataFrame] = None
        self.vix: Optional[pd.DataFrame] = None
        self.robots: Optional[pd.DataFrame] = None
        self.ai_benchmarks: Optional[pd.DataFrame] = None
        self.ai_compute: Optional[pd.DataFrame] = None
        self.ai_index: Optional[pd.DataFrame] = None
        self.agent_summary: Optional[pd.DataFrame] = None

    def summary_metrics(self) -> Dict[str, Any]:
        """Extract key summary metrics."""
        metrics = {"scenario": self.scenario.code, "scenario_name": self.scenario.name}

        if self.macro is not None and not self.macro.empty:
            metrics["peak_unemployment"] = round(float(self.macro["unemployment_rate"].max()), 4)
            metrics["min_unemployment"] = round(float(self.macro["unemployment_rate"].min()), 4)
            metrics["final_unemployment"] = round(float(self.macro["unemployment_rate"].iloc[-1]), 4)
            metrics["avg_gdp_growth"] = round(float(self.macro["gdp_growth_annual"].mean()), 4)
            metrics["final_gdp_trillion"] = round(float(self.macro["gdp_level_trillion"].iloc[-1]), 2)
            metrics["avg_inflation"] = round(float(self.macro["cpi_overall"].mean()), 4)
            metrics["final_ai_adoption"] = round(float(self.macro["ai_adoption_pct"].iloc[-1]), 4)
            metrics["final_robot_fleet"] = int(self.macro["robot_fleet_size"].iloc[-1])

        if self.labor_aggregate is not None and not self.labor_aggregate.empty:
            final_la = self.labor_aggregate.iloc[-1]
            metrics["total_displaced"] = int(final_la.get("total_displaced", 0))
            metrics["displacement_rate"] = round(float(final_la.get("displacement_rate", 0)), 4)

        if self.real_estate is not None and not self.real_estate.empty:
            final_re = self.real_estate[self.real_estate["month"] == self.real_estate["month"].max()]
            ai_hub_re = final_re[final_re["is_ai_hub"] == True]
            at_risk_re = final_re[final_re["is_ai_hub"] == False]
            if not ai_hub_re.empty:
                metrics["ai_hub_re_index"] = round(float(ai_hub_re["price_index"].mean()), 1)
            if not at_risk_re.empty:
                metrics["at_risk_re_index"] = round(float(at_risk_re["price_index"].mean()), 1)

        if self.etf_returns is not None and not self.etf_returns.empty:
            final_etf = self.etf_returns[self.etf_returns["month"] == self.etf_returns["month"].max()]
            for _, row in final_etf.iterrows():
                ticker = row["etf"]
                metrics[f"{ticker}_cumulative_return"] = round(float(row["cumulative_return"]), 4)
                metrics[f"{ticker}_annualized_return"] = round(float(row["annualized_return"]), 4)
                metrics[f"{ticker}_max_drawdown"] = round(
                    float(self.etf_returns[self.etf_returns["etf"] == ticker]["drawdown"].min()), 4
                )

        return metrics


class SimulationEngine:
    """
    Orchestrates a full simulation run for a given scenario.

    Combines:
    1. Macro-economic model (GDP, unemployment, inflation)
    2. Labor market model (occupation-level displacement)
    3. Financial markets model (equities, bonds, commodities, real estate)
    4. Robot deployment model
    5. AI progress model
    6. Agent-based interactions (workers, firms, investors, government)
    """

    def __init__(
        self,
        config: Optional[SimulationConfig] = None,
        defaults: Optional[EconomicDefaults] = None,
    ):
        self.config = config or SimulationConfig()
        self.defaults = defaults or EconomicDefaults()

        # Initialize models
        self.macro_model = MacroEconomicsModel(self.defaults)
        self.labor_model = LaborMarketModel()
        self.financial_model = FinancialMarketsModel()
        self.robot_model = RobotDeploymentModel()
        self.ai_model = AIProgressModel()

    def run(
        self,
        scenario: ScenarioParams,
        months: int = 60,
        run_agents: bool = True,
        seed: Optional[int] = None,
    ) -> SimulationResults:
        """
        Execute full simulation for a scenario.

        Args:
            scenario: The scenario parameters
            months: Simulation horizon in months
            run_agents: Whether to run agent-based simulation (slower but richer)
            seed: Random seed for reproducibility

        Returns:
            SimulationResults with all output DataFrames
        """
        rng = np.random.default_rng(seed or self.config.random_seed)
        results = SimulationResults(scenario)

        # 1. Macro projections
        results.macro = self.macro_model.project(scenario, months, rng)

        # 2. Labor market
        results.labor = self.labor_model.displacement_curve(
            ai_progress_rate=scenario.ai_progress_rate,
            robot_deployment_rate_enum=scenario.robot_deployment_rate.value,
            months=months,
        )
        results.labor_aggregate = self.labor_model.aggregate_displacement(results.labor)

        # 3. Financial markets (use separate RNG streams for independence)
        results.sectors = self.financial_model.project_sector_returns(scenario, months, np.random.default_rng(rng.integers(1e9)))
        results.ai_stocks = self.financial_model.project_ai_stock_returns(scenario, months, np.random.default_rng(rng.integers(1e9)))
        results.etf_returns = self.financial_model.project_etf_returns(
            scenario, months, np.random.default_rng(rng.integers(1e9)),
            sector_returns=results.sectors,
        )
        results.bonds = self.financial_model.project_bond_yields(scenario, months, np.random.default_rng(rng.integers(1e9)))
        results.commodities = self.financial_model.project_commodities(scenario, months, np.random.default_rng(rng.integers(1e9)))
        results.real_estate = self.financial_model.project_real_estate(scenario, months, np.random.default_rng(rng.integers(1e9)))
        results.vix = self.financial_model.project_vix(scenario, months, np.random.default_rng(rng.integers(1e9)))

        # 4. Robot deployment
        results.robots = self.robot_model.project(scenario, months, np.random.default_rng(rng.integers(1e9)))

        # 5. AI progress
        results.ai_benchmarks = self.ai_model.project_benchmarks(scenario, months, np.random.default_rng(rng.integers(1e9)))
        results.ai_compute = self.ai_model.project_compute(scenario, months, np.random.default_rng(rng.integers(1e9)))
        results.ai_index = self.ai_model.composite_ai_index(results.ai_benchmarks)

        # 6. Agent-based simulation (lighter version for ABM insights)
        if run_agents:
            results.agent_summary = self._run_agent_simulation(
                scenario, results, months, rng
            )

        return results

    def _run_agent_simulation(
        self,
        scenario: ScenarioParams,
        model_results: SimulationResults,
        months: int,
        rng: np.random.Generator,
    ) -> pd.DataFrame:
        """
        Run the agent-based model for additional behavioral dynamics.

        Uses macro model outputs as environment, runs agent decisions monthly.
        """
        # Create agents (scaled down for performance)
        n_workers = min(self.config.worker_agents, 5_000)
        n_firms = min(self.config.firm_agents, 500)
        n_investors = min(self.config.investor_agents, 500)

        occ_dicts = [{
            "name": o.name,
            "us_employment": o.us_employment,
            "ai_vulnerability": o.ai_vulnerability,
            "robot_vulnerability": o.robot_vulnerability,
            "avg_annual_wage": o.avg_annual_wage,
            "category": o.category,
        } for o in CRITICAL_OCCUPATIONS]

        workers = create_worker_population(n_workers, occ_dicts, rng)
        firms = create_firm_population(n_firms, rng)
        investors = create_investor_population(n_investors, rng)

        # Government
        gov = GovernmentAgent(
            agent_id=0,
            jurisdiction="US_federal",
            political_orientation=0.0,   # Centrist
            tax_revenue_annual=4.4e12,   # $4.4T
            debt_level_trillion=35.0,
        )

        records = []
        for m in range(1, min(months + 1, 61)):  # Cap ABM at 60 months for performance
            # Get environment from macro model
            macro_row = model_results.macro.iloc[min(m - 1, len(model_results.macro) - 1)]
            ai_idx_row = model_results.ai_index.iloc[min(m - 1, len(model_results.ai_index) - 1)] if model_results.ai_index is not None else None
            robot_row = model_results.robots.iloc[min(m - 1, len(model_results.robots) - 1)] if model_results.robots is not None else None
            vix_row = model_results.vix.iloc[min(m - 1, len(model_results.vix) - 1)] if model_results.vix is not None else None

            env = {
                "rng": rng,
                "month": m,
                "unemployment_rate": float(macro_row["unemployment_rate"]),
                "ai_adoption_pct": float(macro_row["ai_adoption_pct"]),
                "ubi_monthly_amount": scenario.ubi_monthly_amount if scenario.policy_response.value == "ubi_early" else 0,
                "ai_capability": float(ai_idx_row["ai_composite_index"] / 100) if ai_idx_row is not None else 0.5,
                "robot_capability": float(robot_row["capability_score"]) if robot_row is not None else 0.3,
                "robot_cost_per_unit": float(robot_row["unit_cost"]) if robot_row is not None else 50_000,
                "ai_cost_per_worker_month": max(50, 500 * (1 - float(macro_row["ai_adoption_pct"]))),
                "vix": float(vix_row["vix"]) if vix_row is not None else 18,
                "protest_cities": max(0, int((float(macro_row["unemployment_rate"]) - 0.08) * 50)),
                "regulation_level": gov.regulation_level,
            }

            # Get market returns for investors
            if model_results.sectors is not None:
                tech_row = model_results.sectors[
                    (model_results.sectors["month"] == m) &
                    (model_results.sectors["sector"] == "technology")
                ]
                broad_rows = model_results.sectors[model_results.sectors["month"] == m]
                env["ai_equity_return"] = float(tech_row["monthly_return"].iloc[0]) if not tech_row.empty else 0.01
                env["broad_equity_return"] = float(broad_rows["monthly_return"].mean()) if not broad_rows.empty else 0.005
                env["bond_return"] = 0.003
                env["gold_return"] = 0.002
                env["btc_return"] = 0.005
                env["re_return"] = 0.003

            # Step agents
            worker_results = [w.step(env) for w in workers]
            firm_results = [f.step(env) for f in firms]
            investor_results = [inv.step(env) for inv in investors]
            gov_result = gov.step(env)

            # Aggregate
            n_employed = sum(1 for w in workers if w.state.employed)
            n_displaced_this_month = sum(1 for r in worker_results if r["displaced"])
            n_retrained = sum(1 for r in worker_results if r["retrained"])
            n_migrated = sum(1 for r in worker_results if r["migrated"])
            total_consumption = sum(r["consumption"] for r in worker_results)
            n_firms_adopted_ai = sum(1 for f in firms if f.ai_adopted)
            total_fired = sum(r["fired"] for r in firm_results)
            total_portfolio = sum(r["total_value"] for r in investor_results)
            n_panic_sold = sum(1 for r in investor_results if r["panic_sold"])

            records.append({
                "month": m,
                "abm_employment_rate": n_employed / len(workers),
                "abm_displaced_this_month": n_displaced_this_month,
                "abm_retrained_cumulative": n_retrained,
                "abm_migrated_cumulative": n_migrated,
                "abm_total_consumption": total_consumption * self.config.scale_factor,
                "abm_firms_ai_adopted_pct": n_firms_adopted_ai / len(firms),
                "abm_firms_total_fired": total_fired * self.config.scale_factor,
                "abm_investor_total_portfolio": total_portfolio * self.config.scale_factor,
                "abm_investor_panic_sells": n_panic_sold,
                "abm_ubi_active": gov.ubi_active,
                "abm_ubi_amount": gov.ubi_monthly_amount,
                "abm_regulation_level": gov.regulation_level,
                "abm_emergency_spending": gov.emergency_spending,
            })

        return pd.DataFrame(records)

    def run_all_scenarios(
        self,
        months: int = 60,
        run_agents: bool = True,
        seed: Optional[int] = None,
    ) -> Dict[str, SimulationResults]:
        """Run simulation for all scenarios in the tree."""
        tree = ScenarioTree()
        all_results = {}

        for code, scenario in tree.scenarios.items():
            print(f"  Running scenario {code}: {scenario.name}...")
            all_results[code] = self.run(
                scenario, months, run_agents,
                seed=seed if seed else hash(code) % (2**31)
            )

        return all_results
