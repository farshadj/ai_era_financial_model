"""Tests for simulation engine and model integration."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import pandas as pd
from src.models.scenarios import ScenarioTree
from src.simulation.engine import SimulationEngine, SimulationResults
from src.models.labor_market import LaborMarketModel
from src.models.macro_economics import MacroEconomicsModel
from src.models.financial_markets import FinancialMarketsModel
from src.models.robot_deployment import RobotDeploymentModel
from src.models.ai_progress import AIProgressModel


class TestLaborMarketModel:
    def setup_method(self):
        self.model = LaborMarketModel()

    def test_occupations_loaded(self):
        assert len(self.model.occupations) == 20

    def test_vulnerability_ranking(self):
        df = self.model.vulnerability_ranking()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 20
        assert "composite_index" in df.columns
        assert df["composite_index"].max() <= 100
        assert df["composite_index"].min() >= 0

    def test_displacement_curve_returns_dataframe(self):
        df = self.model.displacement_curve(ai_progress_rate=0.8, robot_deployment_rate_enum="fast_2m_yr", months=24)
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_aggregate_displacement(self):
        df = self.model.displacement_curve(ai_progress_rate=0.8, robot_deployment_rate_enum="fast_2m_yr", months=24)
        agg = self.model.aggregate_displacement(df)
        assert isinstance(agg, pd.DataFrame)


class TestMacroEconomicsModel:
    def test_projection(self):
        model = MacroEconomicsModel()
        scenario = ScenarioTree().get("A1")
        df = model.project(scenario, months=24)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 24
        assert "gdp_growth_annual" in df.columns
        assert "unemployment_rate" in df.columns
        assert "cpi_overall" in df.columns

    def test_unemployment_bounds(self):
        model = MacroEconomicsModel()
        scenario = ScenarioTree().get("A1")
        df = model.project(scenario, months=60)
        assert df["unemployment_rate"].min() >= 0
        assert df["unemployment_rate"].max() <= 1.0


class TestFinancialMarketsModel:
    def test_sector_returns(self):
        model = FinancialMarketsModel()
        scenario = ScenarioTree().get("A1")
        df = model.project_sector_returns(scenario, months=12)
        assert isinstance(df, pd.DataFrame)
        assert "sector" in df.columns
        assert "cumulative_return" in df.columns

    def test_real_estate(self):
        model = FinancialMarketsModel()
        scenario = ScenarioTree().get("A1")
        df = model.project_real_estate(scenario, months=12)
        assert len(df) > 0
        assert "metro" in df.columns


class TestSimulationEngine:
    def test_run_basic(self):
        engine = SimulationEngine()
        scenario = ScenarioTree().get("A1")
        results = engine.run(scenario, months=12, run_agents=False)
        assert isinstance(results, SimulationResults)
        assert results.macro is not None
        assert results.sectors is not None

    def test_summary_metrics(self):
        engine = SimulationEngine()
        scenario = ScenarioTree().get("A1")
        results = engine.run(scenario, months=12, run_agents=False)
        m = results.summary_metrics()
        assert "peak_unemployment" in m
        assert "avg_gdp_growth" in m
        assert "final_ai_adoption" in m

    def test_all_scenarios_run(self):
        engine = SimulationEngine()
        all_results = engine.run_all_scenarios(months=12, run_agents=False)
        assert len(all_results) == 7
