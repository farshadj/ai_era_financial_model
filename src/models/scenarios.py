"""
Scenario definitions and scenario tree for AI economy model.

Implements the 7-scenario decision tree:
  A1: AGI Q4 2026 → ASI Q2 2027 → Fast Robot Deployment
  A2: AGI Q4 2026 → ASI Q2 2027 → Slow Robot Deployment
  B1: AGI Q4 2026 → ASI delayed  → UBI Implemented Early
  B2: AGI Q4 2026 → ASI delayed  → No UBI / Social Unrest
  C1: AGI 2027-28 → Gradual Transition
  C2: AGI 2027-28 → Regulatory Slowdown
  D:  AGI Never / Plateau (base case)
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from config.settings import ScenarioProbabilities


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class AGITiming(enum.Enum):
    Q4_2026 = "agi_q4_2026"
    DELAYED_2027_2028 = "agi_2027_2028"
    NEVER = "agi_never"


class ASITiming(enum.Enum):
    FAST_3MO = "asi_3mo_post_agi"
    MEDIUM_6MO = "asi_6mo_post_agi"
    SLOW_12MO = "asi_12mo_post_agi"
    VERY_SLOW_24MO = "asi_24mo_post_agi"
    NEVER = "asi_never"


class RobotDeploymentRate(enum.Enum):
    SLOW = "slow_100k_yr"          # 100K units/year
    MEDIUM = "medium_500k_yr"      # 500K units/year
    FAST = "fast_2m_yr"            # 2M+ units/year


class PolicyResponse(enum.Enum):
    UBI_EARLY = "ubi_early"
    UBI_LATE = "ubi_late"
    REGULATION = "regulation_heavy"
    LAISSEZ_FAIRE = "laissez_faire"


# ---------------------------------------------------------------------------
# Scenario parameter sets
# ---------------------------------------------------------------------------

@dataclass
class ScenarioParams:
    """Full parameter set for a single scenario path."""
    name: str
    code: str   # e.g. "A1", "B2", "D"
    description: str

    # Timing
    agi_timing: AGITiming
    asi_timing: ASITiming
    agi_arrival_quarter: str        # e.g. "2026-Q4"
    asi_arrival_quarter: Optional[str] = None

    # Deployment
    robot_deployment_rate: RobotDeploymentRate = RobotDeploymentRate.MEDIUM
    robot_units_year1: int = 500_000
    robot_cost_2025: float = 50_000.0
    robot_cost_learning_rate: float = 0.18  # 18% cost reduction per doubling

    # Policy
    policy_response: PolicyResponse = PolicyResponse.LAISSEZ_FAIRE
    ubi_monthly_amount: float = 0.0   # $ per person per month
    ubi_start_quarter: Optional[str] = None

    # AI progress multiplier (1.0 = baseline expectation)
    ai_progress_rate: float = 1.0

    # Compound probability from the scenario tree
    probability: float = 0.0

    # Economic shock parameters (deviations from baseline)
    peak_unemployment_rate: float = 0.06
    gdp_growth_modifier: float = 0.0     # Added to baseline GDP growth
    tech_equity_multiplier: float = 1.0   # Multiplier on tech return
    inflation_modifier: float = 0.0       # Added to baseline inflation
    ai_hub_re_appreciation: float = 0.05  # Annual real estate appreciation in AI hubs
    at_risk_re_depreciation: float = -0.02  # Annual depreciation in at-risk metros


# ---------------------------------------------------------------------------
# Build the 7 scenarios
# ---------------------------------------------------------------------------

def _build_scenario_a1(probs: ScenarioProbabilities) -> ScenarioParams:
    """A1: AGI Q4 2026 → ASI Q2 2027 → Fast Robot Deployment"""
    p = probs.agi_by_q4_2026 * probs.asi_by_q2_2027 * probs.fast_robot_deployment
    return ScenarioParams(
        name="AGI 2026 + Fast ASI + Fast Robots",
        code="A1",
        description=(
            "AGI arrives Q4 2026, ASI follows within 6 months (Q2 2027). "
            "Robot production scales rapidly to 2M+ units/year. Maximum "
            "disruption scenario with massive productivity gains but severe "
            "labor displacement."
        ),
        agi_timing=AGITiming.Q4_2026,
        asi_timing=ASITiming.FAST_3MO,
        agi_arrival_quarter="2026-Q4",
        asi_arrival_quarter="2027-Q2",
        robot_deployment_rate=RobotDeploymentRate.FAST,
        robot_units_year1=2_000_000,
        policy_response=PolicyResponse.LAISSEZ_FAIRE,
        ai_progress_rate=1.5,
        probability=p,
        peak_unemployment_rate=0.18,
        gdp_growth_modifier=0.06,
        tech_equity_multiplier=2.5,
        inflation_modifier=-0.02,
        ai_hub_re_appreciation=0.35,
        at_risk_re_depreciation=-0.15,
    )


def _build_scenario_a2(probs: ScenarioProbabilities) -> ScenarioParams:
    """A2: AGI Q4 2026 → ASI Q2 2027 → Slow Robot Deployment"""
    p = probs.agi_by_q4_2026 * probs.asi_by_q2_2027 * probs.slow_robot_deployment
    return ScenarioParams(
        name="AGI 2026 + Fast ASI + Slow Robots",
        code="A2",
        description=(
            "AGI arrives Q4 2026, ASI follows quickly, but physical robot "
            "deployment is constrained by manufacturing capacity. AI disrupts "
            "cognitive work first; physical labor displacement is slower."
        ),
        agi_timing=AGITiming.Q4_2026,
        asi_timing=ASITiming.FAST_3MO,
        agi_arrival_quarter="2026-Q4",
        asi_arrival_quarter="2027-Q2",
        robot_deployment_rate=RobotDeploymentRate.SLOW,
        robot_units_year1=100_000,
        policy_response=PolicyResponse.LAISSEZ_FAIRE,
        ai_progress_rate=1.4,
        probability=p,
        peak_unemployment_rate=0.12,
        gdp_growth_modifier=0.04,
        tech_equity_multiplier=2.2,
        inflation_modifier=-0.01,
        ai_hub_re_appreciation=0.25,
        at_risk_re_depreciation=-0.08,
    )


def _build_scenario_b1(probs: ScenarioProbabilities) -> ScenarioParams:
    """B1: AGI Q4 2026 → ASI delayed → UBI early"""
    p = probs.agi_by_q4_2026 * probs.asi_delayed_2028_plus * probs.ubi_implemented_early
    return ScenarioParams(
        name="AGI 2026 + Delayed ASI + UBI",
        code="B1",
        description=(
            "AGI arrives Q4 2026, but ASI takes longer (2028+). Government "
            "implements UBI proactively, cushioning labor displacement. "
            "Moderate disruption with policy safety net."
        ),
        agi_timing=AGITiming.Q4_2026,
        asi_timing=ASITiming.SLOW_12MO,
        agi_arrival_quarter="2026-Q4",
        asi_arrival_quarter="2028-Q1",
        robot_deployment_rate=RobotDeploymentRate.MEDIUM,
        robot_units_year1=500_000,
        policy_response=PolicyResponse.UBI_EARLY,
        ubi_monthly_amount=1_500.0,
        ubi_start_quarter="2027-Q3",
        ai_progress_rate=1.2,
        probability=p,
        peak_unemployment_rate=0.09,
        gdp_growth_modifier=0.02,
        tech_equity_multiplier=1.8,
        inflation_modifier=0.015,
        ai_hub_re_appreciation=0.18,
        at_risk_re_depreciation=-0.05,
    )


def _build_scenario_b2(probs: ScenarioProbabilities) -> ScenarioParams:
    """B2: AGI Q4 2026 → ASI delayed → No UBI / Social Unrest"""
    p = probs.agi_by_q4_2026 * probs.asi_delayed_2028_plus * probs.no_ubi_social_unrest
    return ScenarioParams(
        name="AGI 2026 + Delayed ASI + Social Unrest",
        code="B2",
        description=(
            "AGI arrives Q4 2026, ASI delayed. No UBI leads to rising "
            "unemployment, social unrest, and political instability. "
            "Markets experience high volatility."
        ),
        agi_timing=AGITiming.Q4_2026,
        asi_timing=ASITiming.SLOW_12MO,
        agi_arrival_quarter="2026-Q4",
        asi_arrival_quarter="2028-Q1",
        robot_deployment_rate=RobotDeploymentRate.MEDIUM,
        robot_units_year1=500_000,
        policy_response=PolicyResponse.LAISSEZ_FAIRE,
        ai_progress_rate=1.2,
        probability=p,
        peak_unemployment_rate=0.14,
        gdp_growth_modifier=-0.01,
        tech_equity_multiplier=1.3,
        inflation_modifier=0.02,
        ai_hub_re_appreciation=0.10,
        at_risk_re_depreciation=-0.12,
    )


def _build_scenario_c1(probs: ScenarioProbabilities) -> ScenarioParams:
    """C1: AGI delayed to 2027-2028 → Gradual Transition"""
    p = probs.agi_delayed_2027_2028 * probs.gradual_transition
    return ScenarioParams(
        name="AGI Delayed + Gradual Transition",
        code="C1",
        description=(
            "AGI arrives 2027-2028, giving more time for adjustment. "
            "Gradual automation with manageable displacement. "
            "Most orderly transition scenario."
        ),
        agi_timing=AGITiming.DELAYED_2027_2028,
        asi_timing=ASITiming.VERY_SLOW_24MO,
        agi_arrival_quarter="2028-Q1",
        asi_arrival_quarter="2030-Q1",
        robot_deployment_rate=RobotDeploymentRate.MEDIUM,
        robot_units_year1=500_000,
        policy_response=PolicyResponse.REGULATION,
        ai_progress_rate=0.8,
        probability=p,
        peak_unemployment_rate=0.07,
        gdp_growth_modifier=0.015,
        tech_equity_multiplier=1.5,
        inflation_modifier=0.005,
        ai_hub_re_appreciation=0.12,
        at_risk_re_depreciation=-0.03,
    )


def _build_scenario_c2(probs: ScenarioProbabilities) -> ScenarioParams:
    """C2: AGI delayed to 2027-2028 → Regulatory Slowdown"""
    p = probs.agi_delayed_2027_2028 * probs.regulatory_slowdown
    return ScenarioParams(
        name="AGI Delayed + Regulatory Slowdown",
        code="C2",
        description=(
            "AGI arrives 2027-2028 but heavy regulation slows deployment. "
            "Lower disruption but also lower productivity gains. "
            "US may lose AI leadership to less regulated markets."
        ),
        agi_timing=AGITiming.DELAYED_2027_2028,
        asi_timing=ASITiming.NEVER,
        agi_arrival_quarter="2028-Q2",
        policy_response=PolicyResponse.REGULATION,
        robot_deployment_rate=RobotDeploymentRate.SLOW,
        robot_units_year1=100_000,
        ai_progress_rate=0.5,
        probability=p,
        peak_unemployment_rate=0.06,
        gdp_growth_modifier=0.005,
        tech_equity_multiplier=1.1,
        inflation_modifier=0.01,
        ai_hub_re_appreciation=0.08,
        at_risk_re_depreciation=-0.02,
    )


def _build_scenario_d(probs: ScenarioProbabilities) -> ScenarioParams:
    """D: AGI Never / Plateau — Base Case"""
    p = probs.agi_never
    return ScenarioParams(
        name="AGI Never / AI Plateau",
        code="D",
        description=(
            "Current AI capabilities plateau. No AGI breakthrough. "
            "Incremental automation continues at historical pace. "
            "Traditional economic relationships hold."
        ),
        agi_timing=AGITiming.NEVER,
        asi_timing=ASITiming.NEVER,
        agi_arrival_quarter="never",
        robot_deployment_rate=RobotDeploymentRate.SLOW,
        robot_units_year1=50_000,
        policy_response=PolicyResponse.LAISSEZ_FAIRE,
        ai_progress_rate=0.3,
        probability=p,
        peak_unemployment_rate=0.05,
        gdp_growth_modifier=0.0,
        tech_equity_multiplier=1.0,
        inflation_modifier=0.0,
        ai_hub_re_appreciation=0.05,
        at_risk_re_depreciation=-0.01,
    )


# ---------------------------------------------------------------------------
# Scenario Tree
# ---------------------------------------------------------------------------

class ScenarioTree:
    """Container for all scenarios with probability weighting."""

    def __init__(self, probs: Optional[ScenarioProbabilities] = None):
        self.probs = probs or ScenarioProbabilities()
        self.scenarios: Dict[str, ScenarioParams] = {}
        self._build()

    def _build(self):
        builders = [
            _build_scenario_a1, _build_scenario_a2,
            _build_scenario_b1, _build_scenario_b2,
            _build_scenario_c1, _build_scenario_c2,
            _build_scenario_d,
        ]
        for builder in builders:
            s = builder(self.probs)
            self.scenarios[s.code] = s

    @property
    def total_probability(self) -> float:
        return sum(s.probability for s in self.scenarios.values())

    def get(self, code: str) -> ScenarioParams:
        return self.scenarios[code]

    def list_scenarios(self) -> List[ScenarioParams]:
        return sorted(self.scenarios.values(), key=lambda s: s.code)

    def expected_value(self, metric_fn) -> float:
        """Compute probability-weighted expected value of a metric function.

        Args:
            metric_fn: Callable(ScenarioParams) → float
        """
        return sum(s.probability * metric_fn(s) for s in self.scenarios.values())

    def __repr__(self):
        lines = [f"ScenarioTree (total_prob={self.total_probability:.4f})"]
        for s in self.list_scenarios():
            lines.append(f"  [{s.code}] {s.name} — P={s.probability:.4f}")
        return "\n".join(lines)
