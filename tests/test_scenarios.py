"""Tests for scenario definitions and probability tree."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from src.models.scenarios import ScenarioTree, ScenarioParams, AGITiming


class TestScenarioTree:
    def setup_method(self):
        self.tree = ScenarioTree()

    def test_seven_scenarios_exist(self):
        assert len(self.tree.scenarios) == 7

    def test_probabilities_sum_to_one(self):
        total = sum(s.probability for s in self.tree.scenarios.values())
        assert abs(total - 1.0) < 0.01, f"Probabilities sum to {total}"

    def test_get_valid_scenario(self):
        s = self.tree.get("A1")
        assert isinstance(s, ScenarioParams)
        assert s.code == "A1"
        assert s.probability == pytest.approx(0.21, abs=0.01)

    def test_get_invalid_scenario_raises(self):
        with pytest.raises(KeyError):
            self.tree.get("Z9")

    def test_scenario_a1_parameters(self):
        s = self.tree.get("A1")
        assert s.agi_timing == AGITiming.Q4_2026
        assert s.agi_arrival_quarter == "2026-Q4"

    def test_scenario_d_no_agi(self):
        s = self.tree.get("D")
        assert s.agi_timing == AGITiming.NEVER

    def test_list_scenarios(self):
        listing = self.tree.list_scenarios()
        assert len(listing) == 7
        codes = {s.code for s in listing}
        assert codes == {"A1", "A2", "B1", "B2", "C1", "C2", "D"}

    def test_expected_value(self):
        ev = self.tree.expected_value(lambda s: 0.03)
        assert ev == pytest.approx(0.03, abs=0.001)


class TestScenarioParams:
    def test_a1_has_economic_shock(self):
        tree = ScenarioTree()
        s = tree.get("A1")
        assert s.peak_unemployment_rate > 0.05
        assert s.gdp_growth_modifier != 0 or s.tech_equity_multiplier > 1

    def test_d_minimal_shocks(self):
        tree = ScenarioTree()
        s = tree.get("D")
        assert s.peak_unemployment_rate <= 0.06
        assert abs(s.gdp_growth_modifier) < 0.02
