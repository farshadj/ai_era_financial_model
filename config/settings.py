"""
AI Economy Model Configuration
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SimulationConfig:
    """Core simulation parameters."""
    # Time horizon
    start_year: int = 2026
    start_quarter: int = 1
    monthly_horizon_months: int = 24       # Monthly granularity first 2 years
    quarterly_horizon_quarters: int = 12   # Quarterly for years 3-5
    annual_horizon_years: int = 5          # Annual for years 6-10

    # Monte Carlo
    num_simulations: int = 10_000
    random_seed: Optional[int] = 42

    # Agent-based model scale (use representative agents for performance)
    worker_agents: int = 10_000       # Representative of 150M workforce
    firm_agents: int = 1_000          # Representative of 10M businesses
    investor_agents: int = 1_000      # Representative of households + institutions
    ai_company_agents: int = 50       # Major AI companies
    scale_factor: float = 15_000.0    # worker_agents * scale_factor ≈ 150M


@dataclass
class ScenarioProbabilities:
    """Probabilities for the scenario tree."""
    # Level 1: AGI timing
    agi_by_q4_2026: float = 0.60
    agi_delayed_2027_2028: float = 0.30
    agi_never: float = 0.10

    # Level 2a: ASI timing (if AGI by Q4 2026)
    asi_by_q2_2027: float = 0.70
    asi_delayed_2028_plus: float = 0.30

    # Level 2b: Robot deployment (if ASI by Q2 2027)
    fast_robot_deployment: float = 0.50
    slow_robot_deployment: float = 0.50

    # Level 2c: Policy response (if ASI delayed)
    ubi_implemented_early: float = 0.40
    no_ubi_social_unrest: float = 0.60

    # Level 2d: Transition pace (if AGI delayed)
    gradual_transition: float = 0.60
    regulatory_slowdown: float = 0.40


@dataclass
class EconomicDefaults:
    """Default economic parameters."""
    # US baseline (2025)
    baseline_gdp_growth: float = 0.025        # 2.5% annual
    baseline_unemployment: float = 0.042       # 4.2%
    baseline_inflation: float = 0.028          # 2.8%
    baseline_fed_rate: float = 0.0450          # 4.50%
    baseline_sp500_pe: float = 22.0            # P/E ratio
    labor_force: int = 168_000_000             # US labor force
    total_gdp_trillion: float = 29.5           # US GDP in trillions

    # Okun's Law coefficient (each 1% unemployment above NAIRU → -2% GDP)
    okun_coefficient: float = -2.0
    nairu: float = 0.042  # Non-accelerating inflation rate of unemployment

    # Phillips Curve parameters (modified for AI era)
    phillips_slope: float = -0.5    # Inflation sensitivity to unemployment gap
    inflation_expectations: float = 0.025

    # Technology adoption S-curve parameters
    ai_adoption_midpoint_years: float = 3.0   # Years to 50% adoption after ROI > 0
    ai_adoption_steepness: float = 1.5


@dataclass
class GeographicConfig:
    """Geographic analysis configuration."""
    # National
    nations: list = field(default_factory=lambda: ["US", "China", "EU", "Emerging"])

    # US States (focus)
    us_states: list = field(default_factory=lambda: ["CA", "TX", "NY", "WA"])

    # Metro areas (AI hubs)
    ai_hubs: list = field(default_factory=lambda: [
        "SF Bay Area", "Seattle", "Austin", "Boston", "NYC"
    ])

    # At-risk metros
    at_risk_metros: list = field(default_factory=lambda: [
        "Detroit", "Cleveland", "Phoenix-suburbs", "Tampa", "Omaha"
    ])


@dataclass
class PortfolioConfig:
    """Portfolio optimization configuration."""
    # Constraints
    max_drawdown: float = 0.40         # 40% max drawdown
    min_liquidity: float = 0.10        # 10% liquid at all times
    risk_free_rate: float = 0.045      # Current risk-free rate

    # Asset classes
    asset_classes: list = field(default_factory=lambda: [
        "ai_equities",         # NVIDIA, Microsoft, Google, etc.
        "broad_equities",      # S&P 500
        "international_eq",    # International developed + EM
        "treasuries",          # US Treasuries
        "tips",                # Treasury Inflation-Protected
        "corporate_bonds",     # Investment grade + high yield
        "ai_hub_real_estate",  # SF, Seattle, Austin properties
        "reits",               # Diversified REITs
        "gold",                # Gold
        "uranium",             # Uranium / nuclear
        "copper",              # Copper
        "bitcoin",             # Bitcoin
        "private_ai",          # Private AI equity / VC
        "cash",                # Cash / money market
    ])


# Singleton configuration
DEFAULT_CONFIG = SimulationConfig()
DEFAULT_SCENARIO_PROBS = ScenarioProbabilities()
DEFAULT_ECONOMICS = EconomicDefaults()
DEFAULT_GEO = GeographicConfig()
DEFAULT_PORTFOLIO = PortfolioConfig()
