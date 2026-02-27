# AI Superintelligence & Robot Automation Economic Forecasting Model

A comprehensive Python system that forecasts macro and microeconomic outcomes under scenarios of accelerated AGI, ASI, and mass robot automation — with Monte Carlo simulation, agent-based modeling, portfolio optimization, and interactive visualization.

---

## 🏗️ Architecture

```
financial model/
├── config/
│   └── settings.py              # Central configuration (simulation, economic, portfolio)
├── src/
│   ├── models/
│   │   ├── scenarios.py         # 7-scenario decision tree with probability weights
│   │   ├── labor_market.py      # 20 occupations, vulnerability index, displacement curves
│   │   ├── macro_economics.py   # GDP, unemployment, bifurcated CPI, Fed rate
│   │   ├── financial_markets.py # 11 equity sectors, 8 AI stocks, bonds, commodities, real estate
│   │   ├── robot_deployment.py  # Wright's Law cost curves, capability evolution
│   │   └── ai_progress.py       # 6 benchmarks, compute scaling, composite AI index
│   ├── agents/
│   │   ├── base.py              # Base agent class
│   │   ├── workers.py           # Worker agents (displacement, retraining, migration)
│   │   ├── firms.py             # Firm agents (AI adoption, robot deployment)
│   │   ├── investors.py         # Investor agents (portfolio, panic selling)
│   │   └── government.py        # Government agent (UBI, regulation) + AI Company agent
│   ├── simulation/
│   │   ├── engine.py            # Main orchestrator (macro → labor → markets → ABM)
│   │   └── monte_carlo.py       # Monte Carlo engine with sensitivity analysis
│   ├── portfolio/
│   │   ├── optimizer.py         # Mean-Variance, Risk Parity, Black-Litterman, Kelly
│   │   └── risk_metrics.py      # VaR, CVaR, Sharpe, Sortino, max drawdown
│   ├── api/
│   │   └── main.py              # FastAPI REST API (12+ endpoints)
│   ├── dashboard/
│   │   └── app.py               # Streamlit interactive dashboard (5 tabs)
│   └── reports/
│       └── generator.py         # Markdown report generator
├── scripts/
│   ├── run_simulation.py        # CLI: run forecasts
│   ├── run_monte_carlo.py       # CLI: Monte Carlo analysis
│   └── generate_reports.py      # CLI: produce Markdown reports
├── tests/
│   ├── test_scenarios.py
│   ├── test_simulation.py
│   └── test_portfolio.py
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## 🚀 Quick Start

### 1. Prerequisites

- Python 3.11+
- pip or conda

### 2. Install Dependencies

```bash
cd "financial model"
pip install -r requirements.txt
```

### 3. Run a Simulation

```bash
# All 7 scenarios
python scripts/run_simulation.py

# Single scenario with 120-month horizon
python scripts/run_simulation.py --scenario A1 --months 120

# With agent-based modeling (slower, richer dynamics)
python scripts/run_simulation.py --scenario A1 --agents
```

### 4. Monte Carlo Analysis

```bash
python scripts/run_monte_carlo.py --scenario A1 --sims 500

# With sensitivity analysis
python scripts/run_monte_carlo.py --scenario A1 --sims 200 --sensitivity
```

### 5. Generate Reports

```bash
# All scenarios + blended report
python scripts/generate_reports.py

# Single scenario with custom parameters
python scripts/generate_reports.py --scenario A1 --months 120 --capital 1000000 --risk aggressive
```

### 6. Launch Interactive Dashboard

```bash
streamlit run src/dashboard/app.py
```

### 7. Start REST API

```bash
uvicorn src.api.main:app --reload --port 8000
```

Then visit: `http://localhost:8000/docs` for interactive Swagger UI.

---

## 📊 Scenarios

| Code | Name | AGI | ASI | Robots | Policy | Probability |
|------|------|-----|-----|--------|--------|-------------|
| A1 | AGI 2026 → Fast ASI, Fast Robots | Q4 2026 | Q2 2028 | Fast | Mixed | 21.0% |
| A2 | AGI 2026 → Fast ASI, Slow Robots | Q4 2026 | Q2 2028 | Slow | Mixed | 21.0% |
| B1 | AGI 2026 → Delayed ASI, UBI | Q4 2026 | Q4 2030 | Moderate | UBI | 7.2% |
| B2 | AGI 2026 → Delayed ASI, No UBI, Unrest | Q4 2026 | Q4 2030 | Moderate | Austerity | 10.8% |
| C1 | Delayed AGI – Gradual | Q4 2029 | Q4 2035 | Slow | Progressive | 18.0% |
| C2 | Delayed AGI – Heavy Regulation | Q4 2029 | Q4 2035 | Slow | Restrictive | 12.0% |
| D  | AGI Never (Diminishing Returns) | Never | Never | Minimal | Status quo | 10.0% |

---

## 🔧 Key Features

### Macro-Economic Modeling
- GDP via Okun's Law + AI productivity boost + robot capital deepening
- Bifurcated CPI: deflation basket (AI-produced) vs. inflation basket (scarce/human)
- Unemployment dynamics: logistic displacement S-curve × AI adoption
- Taylor Rule Fed rate response

### Labor Market
- 20 critical occupations with composite vulnerability index
- Task-composition weighting: cognitive routine (0.8), physical routine (0.7), cognitive non-routine (0.4), physical non-routine (0.3), human interaction (0.1)
- Displacement curves with occupation-specific timing

### Financial Markets
- 11 equity sectors with AI beta exposure
- 8 AI stocks (NVIDIA, MSFT, GOOG, META, AMZN, TSLA, Anthropic*, OpenAI*)
- Bond yields (term premium, flight to safety)
- 5 commodities (gold, oil, uranium, copper, lithium)
- 12 metro real estate profiles (AI hub / stable / at-risk)

### Agent-Based Model
- Worker agents: displacement probability, job search, retraining, migration
- Firm agents: AI adoption decision (payback period), gradual worker displacement
- Investor agents: sentiment-driven behavior, panic selling triggers
- Government agent: UBI activation logic, regulation pressure
- AI Company agents: R&D investment, breakthrough probability

### Portfolio Optimization
- **Mean-Variance** (Markowitz efficient frontier)
- **Risk Parity** (equal risk contribution)
- **Black-Litterman** (scenario views → BL formula)
- **Kelly Criterion** (half-Kelly for safety)
- 14 asset classes with full correlation matrix
- Scenario-probability-weighted blended allocation

### Monte Carlo Engine
- Log-normal parameter perturbations
- Fan charts with percentile bands
- Fat-tail risk probabilities (depression, hyperinflation, mass unemployment)
- Sensitivity analysis (one-at-a-time ±50%)

---

## 🌐 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/scenarios` | List all scenarios with parameters |
| GET | `/api/v1/forecast/gdp?scenario=A1` | GDP growth forecast |
| GET | `/api/v1/forecast/unemployment?scenario=A1` | Unemployment forecast |
| GET | `/api/v1/forecast/inflation?scenario=A1` | Bifurcated CPI forecast |
| GET | `/api/v1/forecast/sector_returns?scenario=A1` | Equity sector returns |
| GET | `/api/v1/forecast/real_estate?scenario=A1` | Metro real estate prices |
| GET | `/api/v1/labor/vulnerability` | Occupation vulnerability ranking |
| GET | `/api/v1/labor/displacement?scenario=A1` | Job displacement forecast |
| POST | `/api/v1/portfolio/optimal_allocation` | Portfolio optimization |
| POST | `/api/v1/portfolio/scenario_weighted` | Probability-weighted allocation |
| POST | `/api/v1/monte_carlo/summary` | Monte Carlo distribution stats |
| GET | `/health` | Health check |

---

## 📈 Dashboard Tabs

1. **Macro Overview** – GDP, unemployment, bifurcated CPI, AI adoption, robot fleet
2. **Sector Analysis** – Equity sector returns, AI stock performance, VIX
3. **Geographic Heatmap** – Real estate by metro (AI hubs vs at-risk), commodities
4. **Labor Market** – Vulnerability ranking, displacement curves, ABM insights
5. **Portfolio** – Allocation pie chart, rebalancing triggers, Monte Carlo fan chart

---

## ⚠️ Disclaimer

This model is for **educational and research purposes only**. All projections are based on synthetic scenarios and should not be construed as investment advice. The model uses simplified approximations of complex economic dynamics. Past performance does not predict future results.

---

## License

MIT
