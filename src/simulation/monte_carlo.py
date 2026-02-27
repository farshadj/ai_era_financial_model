"""
Monte Carlo simulation engine: run thousands of parameter-randomized simulations
to generate probability distributions of outcomes.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from dataclasses import dataclass

from config.settings import SimulationConfig, ScenarioProbabilities
from src.models.scenarios import ScenarioTree, ScenarioParams
from src.simulation.engine import SimulationEngine, SimulationResults


@dataclass
class MonteCarloConfig:
    """Configuration for Monte Carlo runs."""
    n_simulations: int = 1_000      # Per scenario (use 10K for production)
    months: int = 60
    run_agents: bool = False         # Disable ABM for speed in MC
    randomize_params: bool = True

    # Parameter randomization ranges (multiplicative factors)
    ai_progress_range: Tuple[float, float] = (0.5, 2.0)
    robot_adoption_range: Tuple[float, float] = (0.5, 2.0)
    peak_unemployment_range: Tuple[float, float] = (0.7, 1.3)
    gdp_modifier_range: Tuple[float, float] = (0.5, 1.5)


class MonteCarloResult:
    """Container for Monte Carlo simulation results."""

    def __init__(self, scenario_code: str, n_sims: int):
        self.scenario_code = scenario_code
        self.n_sims = n_sims

        # Key metric distributions
        self.peak_unemployment: List[float] = []
        self.final_unemployment: List[float] = []
        self.avg_gdp_growth: List[float] = []
        self.final_gdp: List[float] = []
        self.avg_inflation: List[float] = []
        self.final_ai_adoption: List[float] = []
        self.final_robot_fleet: List[int] = []
        self.ai_hub_re_index: List[float] = []
        self.at_risk_re_index: List[float] = []

        # Track individual simulation summaries
        self.sim_summaries: List[Dict[str, Any]] = []

    def add_result(self, summary: Dict[str, Any]):
        """Add a single simulation result."""
        self.sim_summaries.append(summary)
        self.peak_unemployment.append(summary.get("peak_unemployment", 0))
        self.final_unemployment.append(summary.get("final_unemployment", 0))
        self.avg_gdp_growth.append(summary.get("avg_gdp_growth", 0))
        self.final_gdp.append(summary.get("final_gdp_trillion", 0))
        self.avg_inflation.append(summary.get("avg_inflation", 0))
        self.final_ai_adoption.append(summary.get("final_ai_adoption", 0))
        self.final_robot_fleet.append(summary.get("final_robot_fleet", 0))
        self.ai_hub_re_index.append(summary.get("ai_hub_re_index", 100))
        self.at_risk_re_index.append(summary.get("at_risk_re_index", 100))

    def percentiles(self, metric: str, pcts: List[float] = None) -> Dict[str, float]:
        """Compute percentiles for a metric."""
        if pcts is None:
            pcts = [10, 25, 50, 75, 90]
        data = getattr(self, metric, [])
        if not data:
            return {}
        arr = np.array(data)
        return {f"p{p}": round(float(np.percentile(arr, p)), 4) for p in pcts}

    def fat_tail_risks(self) -> Dict[str, float]:
        """Calculate probability of extreme outcomes."""
        n = len(self.peak_unemployment)
        if n == 0:
            return {}
        return {
            "P(unemployment > 15%)": round(sum(1 for u in self.peak_unemployment if u > 0.15) / n, 4),
            "P(unemployment > 20%)": round(sum(1 for u in self.peak_unemployment if u > 0.20) / n, 4),
            "P(GDP decline)": round(sum(1 for g in self.avg_gdp_growth if g < 0) / n, 4),
            "P(AI hub RE > 150)": round(sum(1 for r in self.ai_hub_re_index if r > 150) / n, 4),
            "P(at-risk RE < 80)": round(sum(1 for r in self.at_risk_re_index if r < 80) / n, 4),
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Convert all simulation summaries to a DataFrame."""
        return pd.DataFrame(self.sim_summaries)

    def summary(self) -> Dict[str, Any]:
        """Full summary with percentiles and fat tails."""
        return {
            "scenario": self.scenario_code,
            "n_simulations": self.n_sims,
            "peak_unemployment": self.percentiles("peak_unemployment"),
            "final_unemployment": self.percentiles("final_unemployment"),
            "avg_gdp_growth": self.percentiles("avg_gdp_growth"),
            "final_gdp": self.percentiles("final_gdp"),
            "avg_inflation": self.percentiles("avg_inflation"),
            "ai_hub_re_index": self.percentiles("ai_hub_re_index"),
            "at_risk_re_index": self.percentiles("at_risk_re_index"),
            "fat_tail_risks": self.fat_tail_risks(),
        }


class MonteCarloEngine:
    """
    Runs Monte Carlo simulations across scenarios.

    For each scenario, randomizes key parameters and runs the simulation
    engine multiple times to build probability distributions.
    """

    def __init__(
        self,
        mc_config: Optional[MonteCarloConfig] = None,
        sim_config: Optional[SimulationConfig] = None,
    ):
        self.mc = mc_config or MonteCarloConfig()
        self.sim_config = sim_config or SimulationConfig()
        self.engine = SimulationEngine(self.sim_config)

    def run_scenario(
        self,
        scenario: ScenarioParams,
        n_sims: Optional[int] = None,
        seed: int = 42,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> MonteCarloResult:
        """
        Run Monte Carlo simulations for a single scenario.

        Args:
            scenario: Base scenario parameters
            n_sims: Number of simulations (overrides config)
            seed: Base random seed
            progress_callback: Called with (completed, total) counts
        """
        n = n_sims or self.mc.n_simulations
        result = MonteCarloResult(scenario.code, n)
        rng = np.random.default_rng(seed)

        for i in range(n):
            # Randomize parameters
            if self.mc.randomize_params:
                sim_scenario = self._randomize_scenario(scenario, rng)
            else:
                sim_scenario = scenario

            # Run simulation
            sim_seed = int(rng.integers(0, 2**31))
            sim_result = self.engine.run(
                sim_scenario,
                months=self.mc.months,
                run_agents=self.mc.run_agents,
                seed=sim_seed,
            )

            # Collect summary
            summary = sim_result.summary_metrics()
            summary["sim_id"] = i
            result.add_result(summary)

            if progress_callback and (i + 1) % max(1, n // 20) == 0:
                progress_callback(i + 1, n)

        return result

    def run_all_scenarios(
        self,
        n_sims: Optional[int] = None,
        seed: int = 42,
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
    ) -> Dict[str, MonteCarloResult]:
        """Run Monte Carlo for all scenarios in the tree."""
        tree = ScenarioTree()
        results = {}

        for code, scenario in tree.scenarios.items():
            print(f"  MC for scenario {code}: {scenario.name} ({n_sims or self.mc.n_simulations} sims)...")

            def _cb(done, total, _code=code):
                if progress_callback:
                    progress_callback(_code, done, total)

            results[code] = self.run_scenario(
                scenario,
                n_sims=n_sims,
                seed=seed + hash(code) % 10000,
                progress_callback=_cb,
            )

        return results

    def sensitivity_analysis(
        self,
        scenario: ScenarioParams,
        n_sims: int = 200,
        seed: int = 42,
    ) -> pd.DataFrame:
        """
        Analyze which input variable has the highest impact on outcomes.

        Tests each parameter at ±50% while holding others constant.
        Returns DataFrame with parameter sensitivities.
        """
        params_to_test = [
            ("ai_progress_rate", 0.5, 1.5),
            ("robot_units_year1", 0.5, 2.0),
            ("peak_unemployment_rate", 0.7, 1.3),
            ("gdp_growth_modifier", 0.5, 1.5),
            ("ai_hub_re_appreciation", 0.5, 2.0),
        ]

        records = []
        rng = np.random.default_rng(seed)

        for param_name, low_mult, high_mult in params_to_test:
            for label, mult in [("low", low_mult), ("base", 1.0), ("high", high_mult)]:
                test_scenario = self._copy_scenario_with_multiplier(
                    scenario, param_name, mult
                )
                result = self.run_scenario(test_scenario, n_sims=n_sims, seed=seed)
                summary = result.summary()

                records.append({
                    "parameter": param_name,
                    "variant": label,
                    "multiplier": mult,
                    "peak_unemployment_p50": summary["peak_unemployment"].get("p50", 0),
                    "avg_gdp_growth_p50": summary["avg_gdp_growth"].get("p50", 0),
                    "avg_inflation_p50": summary["avg_inflation"].get("p50", 0),
                    "ai_hub_re_p50": summary["ai_hub_re_index"].get("p50", 100),
                })

        return pd.DataFrame(records)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _randomize_scenario(
        self,
        scenario: ScenarioParams,
        rng: np.random.Generator,
    ) -> ScenarioParams:
        """Create a copy of the scenario with randomized parameters."""
        import copy
        s = copy.deepcopy(scenario)

        # Randomize AI progress rate
        lo, hi = self.mc.ai_progress_range
        s.ai_progress_rate *= rng.uniform(lo, hi)

        # Randomize robot deployment
        lo, hi = self.mc.robot_adoption_range
        s.robot_units_year1 = int(s.robot_units_year1 * rng.uniform(lo, hi))

        # Randomize peak unemployment
        lo, hi = self.mc.peak_unemployment_range
        s.peak_unemployment_rate *= rng.uniform(lo, hi)
        s.peak_unemployment_rate = min(s.peak_unemployment_rate, 0.30)

        # Randomize GDP modifier
        lo, hi = self.mc.gdp_modifier_range
        if s.gdp_growth_modifier != 0:
            s.gdp_growth_modifier *= rng.uniform(lo, hi)

        # Small random shifts to other params
        s.inflation_modifier += rng.normal(0, 0.005)
        s.ai_hub_re_appreciation *= rng.uniform(0.8, 1.2)
        s.at_risk_re_depreciation *= rng.uniform(0.8, 1.2)

        return s

    @staticmethod
    def _copy_scenario_with_multiplier(
        scenario: ScenarioParams,
        param_name: str,
        multiplier: float,
    ) -> ScenarioParams:
        """Create scenario copy with one parameter multiplied."""
        import copy
        s = copy.deepcopy(scenario)
        current = getattr(s, param_name)
        if isinstance(current, (int, float)):
            setattr(s, param_name, type(current)(current * multiplier))
        return s
