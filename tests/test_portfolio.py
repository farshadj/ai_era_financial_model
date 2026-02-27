"""Tests for portfolio optimization and risk metrics."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
from src.models.scenarios import ScenarioTree
from src.simulation.engine import SimulationEngine
from src.portfolio.optimizer import PortfolioOptimizer, PortfolioAllocation
from src.portfolio.risk_metrics import RiskMetrics


class TestPortfolioOptimizer:
    def setup_method(self):
        self.optimizer = PortfolioOptimizer()
        self.tree = ScenarioTree()
        self.engine = SimulationEngine()
        self.scenario = self.tree.get("A1")
        self.results = self.engine.run(self.scenario, months=12, run_agents=False)

    def test_mean_variance(self):
        alloc = self.optimizer.optimize(self.scenario, self.results,
                                         "mean_variance", 500_000, "moderate")
        assert isinstance(alloc, PortfolioAllocation)
        assert abs(sum(alloc.weights.values()) - 1.0) < 0.01
        assert alloc.expected_return != 0
        assert alloc.expected_volatility > 0

    def test_risk_parity(self):
        alloc = self.optimizer.optimize(self.scenario, self.results,
                                         "risk_parity", 500_000, "moderate")
        assert abs(sum(alloc.weights.values()) - 1.0) < 0.01

    def test_black_litterman(self):
        alloc = self.optimizer.optimize(self.scenario, self.results,
                                         "black_litterman", 500_000, "moderate")
        assert abs(sum(alloc.weights.values()) - 1.0) < 0.01

    def test_kelly(self):
        alloc = self.optimizer.optimize(self.scenario, self.results,
                                         "kelly", 500_000, "moderate")
        assert abs(sum(alloc.weights.values()) - 1.0) < 0.01

    def test_weights_positive(self):
        alloc = self.optimizer.optimize(self.scenario, self.results,
                                         "mean_variance", 500_000, "moderate")
        for w in alloc.weights.values():
            assert w >= -0.01  # allow tiny numerical errors

    def test_rebalancing_triggers_exist(self):
        alloc = self.optimizer.optimize(self.scenario, self.results,
                                         "mean_variance", 500_000, "moderate")
        assert len(alloc.rebalancing_triggers) > 0

    def test_scenario_weighted(self):
        allocs = {}
        for code, scen in self.tree.scenarios.items():
            res = self.engine.run(scen, months=12, run_agents=False)
            allocs[code] = self.optimizer.optimize(scen, res, "mean_variance",
                                                     500_000, "moderate")
        weighted = self.optimizer.scenario_weighted_allocation(allocs)
        assert abs(sum(weighted.values()) - 1.0) < 0.01


class TestRiskMetrics:
    def test_var(self):
        returns = np.random.normal(0.001, 0.02, 252)
        var = RiskMetrics.value_at_risk(returns, 500_000, 0.95)
        assert var < 0  # VaR should be negative (loss)

    def test_cvar(self):
        returns = np.random.normal(0.001, 0.02, 252)
        cvar = RiskMetrics.conditional_var(returns, 500_000, 0.95)
        var = RiskMetrics.value_at_risk(returns, 500_000, 0.95)
        assert cvar <= var  # CVaR should be worse than VaR

    def test_sharpe(self):
        returns = np.random.normal(0.001, 0.02, 252)
        sharpe = RiskMetrics.sharpe_ratio(returns, 0.05)
        assert isinstance(sharpe, float)

    def test_max_drawdown(self):
        returns = np.random.normal(0.001, 0.02, 252)
        dd = RiskMetrics.max_drawdown(returns)
        assert dd <= 0
        assert dd >= -1.0
