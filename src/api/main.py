"""
FastAPI application for the AI Economy Forecasting Model.

Endpoints:
  GET /api/v1/forecast/gdp
  GET /api/v1/forecast/unemployment
  GET /api/v1/forecast/sector_returns
  GET /api/v1/forecast/real_estate
  GET /api/v1/forecast/inflation
  GET /api/v1/portfolio/optimal_allocation
  GET /api/v1/portfolio/scenario_weighted
  GET /api/v1/scenarios
  GET /api/v1/labor/vulnerability
  GET /api/v1/labor/displacement
  GET /api/v1/monte_carlo/summary
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config.settings import SimulationConfig
from src.models.scenarios import ScenarioTree
from src.models.labor_market import LaborMarketModel
from src.simulation.engine import SimulationEngine, SimulationResults
from src.simulation.monte_carlo import MonteCarloEngine, MonteCarloConfig
from src.portfolio.optimizer import PortfolioOptimizer

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="AI Economy Forecasting Model",
    description=(
        "Predicts macro/microeconomic outcomes under AGI, ASI, "
        "and mass robot automation scenarios."
    ),
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Shared state (lazy initialization)
# ---------------------------------------------------------------------------

_cache: Dict[str, Any] = {}


def _get_scenario_tree() -> ScenarioTree:
    if "tree" not in _cache:
        _cache["tree"] = ScenarioTree()
    return _cache["tree"]


def _get_engine() -> SimulationEngine:
    if "engine" not in _cache:
        _cache["engine"] = SimulationEngine()
    return _cache["engine"]


def _get_sim_results(scenario_code: str, months: int = 60) -> SimulationResults:
    key = f"sim_{scenario_code}_{months}"
    if key not in _cache:
        tree = _get_scenario_tree()
        engine = _get_engine()
        scenario = tree.get(scenario_code)
        _cache[key] = engine.run(scenario, months=months, run_agents=False)
    return _cache[key]


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

class ScenarioInfo(BaseModel):
    code: str
    name: str
    description: str
    probability: float


class ForecastPoint(BaseModel):
    period: str
    value: float


class SectorReturn(BaseModel):
    sector: str
    cumulative_return: float
    annualized_vol: float


class AllocationResult(BaseModel):
    weights: Dict[str, float]
    expected_return: float
    expected_volatility: float
    sharpe_ratio: float
    method: str
    rebalancing_triggers: List[Dict[str, Any]]


# ---------------------------------------------------------------------------
# Endpoints: Scenarios
# ---------------------------------------------------------------------------

@app.get("/api/v1/scenarios", response_model=List[ScenarioInfo])
def list_scenarios():
    """List all available scenarios with probabilities."""
    tree = _get_scenario_tree()
    return [
        ScenarioInfo(
            code=s.code, name=s.name,
            description=s.description, probability=round(s.probability, 4)
        )
        for s in tree.list_scenarios()
    ]


# ---------------------------------------------------------------------------
# Endpoints: Forecasts
# ---------------------------------------------------------------------------

@app.get("/api/v1/forecast/gdp")
def forecast_gdp(
    scenario: str = Query("A1", description="Scenario code (A1, A2, B1, B2, C1, C2, D)"),
    horizon: str = Query("5y", description="Forecast horizon (e.g. 2y, 5y)"),
):
    """Forecast GDP growth rate and level."""
    months = _parse_horizon(horizon)
    results = _get_sim_results(scenario, months)
    if results.macro is None:
        raise HTTPException(404, "No macro data")

    data = results.macro[["period", "gdp_growth_annual", "gdp_level_trillion"]].to_dict("records")
    return {
        "scenario": scenario,
        "horizon_months": months,
        "forecasts": data,
    }


@app.get("/api/v1/forecast/unemployment")
def forecast_unemployment(
    scenario: str = Query("A1"),
    geography: str = Query("national", description="Geography (national, metro name)"),
    horizon: str = Query("2y"),
):
    """Forecast unemployment rate."""
    months = _parse_horizon(horizon)
    results = _get_sim_results(scenario, months)
    if results.macro is None:
        raise HTTPException(404, "No macro data")

    data = results.macro[["period", "unemployment_rate"]].to_dict("records")
    return {
        "scenario": scenario,
        "geography": geography,
        "horizon_months": months,
        "forecasts": data,
    }


@app.get("/api/v1/forecast/sector_returns")
def forecast_sector_returns(
    scenario: str = Query("A1"),
    sector: str = Query("all", description="Sector name or 'all'"),
    horizon: str = Query("5y"),
):
    """Forecast equity sector returns."""
    months = _parse_horizon(horizon)
    results = _get_sim_results(scenario, months)
    if results.sectors is None:
        raise HTTPException(404, "No sector data")

    df = results.sectors
    if sector != "all":
        df = df[df["sector"] == sector]

    # Get final cumulative returns
    final = df[df["month"] == df["month"].max()]
    data = final[["sector", "cumulative_return", "annualized_vol"]].to_dict("records")
    return {"scenario": scenario, "sector_returns": data}


@app.get("/api/v1/forecast/real_estate")
def forecast_real_estate(
    scenario: str = Query("A1"),
    metro: str = Query("all", description="Metro name or 'all'"),
    horizon: str = Query("5y"),
):
    """Forecast real estate price indices by metro."""
    months = _parse_horizon(horizon)
    results = _get_sim_results(scenario, months)
    if results.real_estate is None:
        raise HTTPException(404, "No real estate data")

    df = results.real_estate
    if metro != "all":
        df = df[df["metro"] == metro]

    final = df[df["month"] == df["month"].max()]
    data = final[["metro", "price_index", "is_ai_hub"]].to_dict("records")
    return {"scenario": scenario, "real_estate": data}


@app.get("/api/v1/forecast/inflation")
def forecast_inflation(
    scenario: str = Query("A1"),
    horizon: str = Query("5y"),
):
    """Forecast bifurcated inflation (deflation basket + inflation basket)."""
    months = _parse_horizon(horizon)
    results = _get_sim_results(scenario, months)
    if results.macro is None:
        raise HTTPException(404, "No macro data")

    data = results.macro[[
        "period", "cpi_overall", "cpi_deflation_basket", "cpi_inflation_basket"
    ]].to_dict("records")
    return {"scenario": scenario, "inflation": data}


# ---------------------------------------------------------------------------
# Endpoints: Labor Market
# ---------------------------------------------------------------------------

@app.get("/api/v1/labor/vulnerability")
def labor_vulnerability():
    """Get automation vulnerability ranking for tracked occupations."""
    model = LaborMarketModel()
    df = model.vulnerability_ranking()
    return {"occupations": df.to_dict("records")}


@app.get("/api/v1/labor/displacement")
def labor_displacement(
    scenario: str = Query("A1"),
    horizon: str = Query("5y"),
):
    """Forecast job displacement by occupation."""
    months = _parse_horizon(horizon)
    results = _get_sim_results(scenario, months)
    if results.labor_aggregate is None:
        raise HTTPException(404, "No labor data")

    agg = results.labor_aggregate.tail(12)  # Last 12 months
    return {
        "scenario": scenario,
        "aggregate_displacement": agg.to_dict("records"),
    }


# ---------------------------------------------------------------------------
# Endpoints: Portfolio
# ---------------------------------------------------------------------------

@app.get("/api/v1/portfolio/optimal_allocation")
def optimal_allocation(
    scenario: str = Query("A1"),
    capital: float = Query(500_000),
    risk_tolerance: str = Query("moderate", description="conservative, moderate, aggressive"),
    method: str = Query("mean_variance", description="mean_variance, risk_parity, black_litterman, kelly"),
):
    """Get optimal portfolio allocation for a scenario."""
    tree = _get_scenario_tree()
    results = _get_sim_results(scenario, 60)
    optimizer = PortfolioOptimizer()

    alloc = optimizer.optimize(
        tree.get(scenario), results, method, capital, risk_tolerance
    )
    return alloc.to_dict()


@app.get("/api/v1/portfolio/scenario_weighted")
def scenario_weighted_allocation(
    capital: float = Query(500_000),
    risk_tolerance: str = Query("moderate"),
):
    """Get probability-weighted allocation across all scenarios."""
    tree = _get_scenario_tree()
    optimizer = PortfolioOptimizer()

    allocations = {}
    for code in tree.scenarios:
        results = _get_sim_results(code, 60)
        allocations[code] = optimizer.optimize(
            tree.get(code), results, "mean_variance", capital, risk_tolerance
        )

    weighted = optimizer.scenario_weighted_allocation(allocations)
    return {
        "method": "scenario_probability_weighted",
        "weights": weighted,
        "scenario_allocations": {k: v.to_dict() for k, v in allocations.items()},
    }


# ---------------------------------------------------------------------------
# Endpoints: Monte Carlo
# ---------------------------------------------------------------------------

@app.get("/api/v1/monte_carlo/summary")
def monte_carlo_summary(
    scenario: str = Query("A1"),
    n_sims: int = Query(200, description="Number of simulations (max 1000)"),
):
    """Run Monte Carlo simulation and return distributional summary."""
    n_sims = min(n_sims, 1000)
    tree = _get_scenario_tree()
    mc_config = MonteCarloConfig(n_simulations=n_sims, run_agents=False)
    mc_engine = MonteCarloEngine(mc_config)
    result = mc_engine.run_scenario(tree.get(scenario), n_sims=n_sims)
    return result.summary()


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "model": "AI Economy Forecasting v0.1.0"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_horizon(horizon: str) -> int:
    """Parse horizon string like '2y', '5y', '18m' to months."""
    h = horizon.strip().lower()
    if h.endswith("y"):
        return int(h[:-1]) * 12
    elif h.endswith("m"):
        return int(h[:-1])
    else:
        try:
            return int(h)
        except ValueError:
            return 60  # Default 5 years
