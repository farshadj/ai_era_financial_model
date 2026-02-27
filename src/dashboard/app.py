"""
Streamlit Interactive Dashboard for AI Economy Forecasting Model.

5 Tabs:
  1. Macro Overview (GDP, unemployment, inflation, UBI probability)
  2. Sector Analysis (equity returns, AI valuations, bankruptcy risk)
  3. Geographic Heatmap (real estate by metro)
  4. Labor Market (displacement, vulnerability, wages)
  5. Portfolio Recommendations (allocation, rebalancing, Monte Carlo fan chart)
"""

import sys
import os

# Ensure project root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

from config.settings import SimulationConfig
from src.models.scenarios import ScenarioTree
from src.models.labor_market import LaborMarketModel
from src.simulation.engine import SimulationEngine
from src.simulation.monte_carlo import MonteCarloEngine, MonteCarloConfig
from src.portfolio.optimizer import PortfolioOptimizer


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="AI Economy Forecasting Model",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🤖 AI Economy Forecasting Model")
st.caption("Predicting macro/microeconomic outcomes under AGI, ASI, and robot automation scenarios")


# ---------------------------------------------------------------------------
# Sidebar: Scenario selection
# ---------------------------------------------------------------------------

st.sidebar.header("⚙️ Configuration")

tree = ScenarioTree()
scenario_options = {f"{s.code}: {s.name}": s.code for s in tree.list_scenarios()}
selected_label = st.sidebar.selectbox(
    "Select Scenario",
    list(scenario_options.keys()),
    index=0,
)
scenario_code = scenario_options[selected_label]
scenario = tree.get(scenario_code)

months = st.sidebar.slider("Forecast Horizon (months)", 12, 120, 60, 6)
run_agents = st.sidebar.checkbox("Run Agent-Based Model", value=False,
                                   help="Enables ABM (slower but richer dynamics)")

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Scenario Probability:** {scenario.probability:.1%}")
st.sidebar.markdown(f"**AGI Arrival:** {scenario.agi_arrival_quarter}")
st.sidebar.markdown(f"**ASI Arrival:** {scenario.asi_arrival_quarter or 'N/A'}")
st.sidebar.markdown(f"**Robot Rate:** {scenario.robot_deployment_rate.value}")
st.sidebar.markdown(f"**Policy:** {scenario.policy_response.value}")


# ---------------------------------------------------------------------------
# Run simulation (cached)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=600, show_spinner="Running simulation...")
def run_simulation(code: str, _months: int, _run_agents: bool):
    engine = SimulationEngine()
    s = ScenarioTree().get(code)
    return engine.run(s, months=_months, run_agents=_run_agents)


results = run_simulation(scenario_code, months, run_agents)


# ---------------------------------------------------------------------------
# Tab 1: Macro Overview
# ---------------------------------------------------------------------------

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Macro Overview",
    "📈 Sector Analysis",
    "🗺️ Geographic Heatmap",
    "👷 Labor Market",
    "💼 Portfolio",
])

with tab1:
    st.header("Macroeconomic Overview")

    if results.macro is not None:
        macro = results.macro

        # KPI row
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            peak_u = macro["unemployment_rate"].max()
            st.metric("Peak Unemployment", f"{peak_u:.1%}",
                       delta=f"{peak_u - 0.042:.1%} vs baseline")
        with col2:
            avg_gdp = macro["gdp_growth_annual"].mean()
            st.metric("Avg GDP Growth", f"{avg_gdp:.2%}")
        with col3:
            avg_inf = macro["cpi_overall"].mean()
            st.metric("Avg Inflation", f"{avg_inf:.2%}")
        with col4:
            final_ai = macro["ai_adoption_pct"].iloc[-1]
            st.metric("AI Adoption", f"{final_ai:.0%}")

        # Unemployment chart
        fig_unemp = go.Figure()
        fig_unemp.add_trace(go.Scatter(
            x=macro["period"], y=macro["unemployment_rate"] * 100,
            mode="lines", name="Unemployment Rate (%)",
            line=dict(color="red", width=2),
        ))
        fig_unemp.add_hline(y=4.2, line_dash="dash", annotation_text="Baseline (4.2%)")
        fig_unemp.add_hline(y=8.0, line_dash="dot", line_color="orange",
                             annotation_text="Policy trigger (8%)")
        fig_unemp.update_layout(title="Unemployment Rate Forecast", yaxis_title="%",
                                 height=400)
        st.plotly_chart(fig_unemp, use_container_width=True)

        # GDP and Inflation side by side
        col_left, col_right = st.columns(2)
        with col_left:
            fig_gdp = px.line(macro, x="period", y="gdp_growth_annual",
                               title="GDP Growth (Annualized)")
            fig_gdp.update_layout(yaxis_tickformat=".1%", height=350)
            st.plotly_chart(fig_gdp, use_container_width=True)

        with col_right:
            fig_inf = go.Figure()
            fig_inf.add_trace(go.Scatter(x=macro["period"], y=macro["cpi_deflation_basket"],
                                          name="Deflation Basket (AI)", line=dict(color="blue")))
            fig_inf.add_trace(go.Scatter(x=macro["period"], y=macro["cpi_inflation_basket"],
                                          name="Inflation Basket (Scarce)", line=dict(color="red")))
            fig_inf.add_trace(go.Scatter(x=macro["period"], y=macro["cpi_overall"],
                                          name="Overall CPI", line=dict(color="black", dash="dash")))
            fig_inf.update_layout(title="Bifurcated CPI", yaxis_tickformat=".1%", height=350)
            st.plotly_chart(fig_inf, use_container_width=True)

        # AI adoption and robot fleet
        col_l, col_r = st.columns(2)
        with col_l:
            fig_ai = px.area(macro, x="period", y="ai_adoption_pct",
                              title="AI Adoption Rate")
            fig_ai.update_layout(yaxis_tickformat=".0%", height=300)
            st.plotly_chart(fig_ai, use_container_width=True)
        with col_r:
            fig_rob = px.area(macro, x="period", y="robot_fleet_size",
                               title="Cumulative Robot Fleet")
            fig_rob.update_layout(height=300)
            st.plotly_chart(fig_rob, use_container_width=True)


# ---------------------------------------------------------------------------
# Tab 2: Sector Analysis
# ---------------------------------------------------------------------------

with tab2:
    st.header("Sector & Stock Analysis")

    if results.sectors is not None:
        # Final returns by sector
        final_sectors = results.sectors[results.sectors["month"] == results.sectors["month"].max()]
        final_sectors = final_sectors.sort_values("cumulative_return", ascending=True)

        fig_sector = px.bar(
            final_sectors, x="cumulative_return", y="sector",
            orientation="h",
            title=f"Cumulative Sector Returns ({months} months)",
            color="cumulative_return",
            color_continuous_scale="RdYlGn",
        )
        fig_sector.update_layout(height=500, xaxis_tickformat=".0%")
        st.plotly_chart(fig_sector, use_container_width=True)

    if results.ai_stocks is not None:
        st.subheader("AI Stock Performance")
        final_stocks = results.ai_stocks[results.ai_stocks["month"] == results.ai_stocks["month"].max()]
        final_stocks = final_stocks.sort_values("cumulative_return", ascending=False)

        fig_stocks = px.bar(
            final_stocks, x="stock", y="cumulative_return",
            title="AI Stock Cumulative Returns",
            color="cumulative_return",
            color_continuous_scale="Viridis",
        )
        fig_stocks.update_layout(yaxis_tickformat=".0%", height=400)
        st.plotly_chart(fig_stocks, use_container_width=True)

    # ------------------------------------------------------------------
    # ETF Performance Predictions (SPY & QQQ)
    # ------------------------------------------------------------------

    if results.etf_returns is not None:
        st.subheader("📊 SPY & QQQ Performance Forecast")

        etf_df = results.etf_returns
        final_etf = etf_df[etf_df["month"] == etf_df["month"].max()]

        # KPI cards
        etf_cols = st.columns(len(final_etf))
        for i, (_, row) in enumerate(final_etf.iterrows()):
            ticker = row["etf"]
            with etf_cols[i]:
                cum_ret = row["cumulative_return"] - 1  # Convert from multiplier
                ann_ret = row["annualized_return"]
                max_dd = float(etf_df[etf_df["etf"] == ticker]["drawdown"].min())
                st.metric(
                    f"{ticker} ({row['etf_name']})",
                    f"{cum_ret:+.1%}",
                    delta=f"Ann: {ann_ret:+.1%} | Max DD: {max_dd:.1%}",
                )

        # Cumulative price chart
        fig_etf_price = go.Figure()
        for ticker in etf_df["etf"].unique():
            tk_data = etf_df[etf_df["etf"] == ticker]
            fig_etf_price.add_trace(go.Scatter(
                x=tk_data["month"], y=tk_data["price"],
                mode="lines", name=ticker,
                line=dict(width=2.5),
            ))
        fig_etf_price.add_hline(y=100, line_dash="dash", line_color="gray",
                                 annotation_text="Starting price (100)")
        fig_etf_price.update_layout(
            title=f"SPY vs QQQ Price Forecast ({months} months, base=100)",
            yaxis_title="Price Index", xaxis_title="Month",
            height=420,
        )
        st.plotly_chart(fig_etf_price, use_container_width=True)

        # Monthly returns comparison + drawdown
        col_ret, col_dd = st.columns(2)
        with col_ret:
            fig_monthly = px.line(
                etf_df, x="month", y="monthly_return", color="etf",
                title="Monthly Returns",
            )
            fig_monthly.update_layout(yaxis_tickformat=".1%", height=300)
            st.plotly_chart(fig_monthly, use_container_width=True)
        with col_dd:
            fig_dd = px.area(
                etf_df, x="month", y="drawdown", color="etf",
                title="Drawdown From Peak",
            )
            fig_dd.update_layout(yaxis_tickformat=".0%", height=300)
            st.plotly_chart(fig_dd, use_container_width=True)

        # Return range table
        st.markdown("**Predicted Return Ranges**")
        range_records = []
        for ticker in etf_df["etf"].unique():
            tk = etf_df[etf_df["etf"] == ticker]
            # Rolling 12-month returns for range estimation
            monthly_rets = tk["monthly_return"].values
            cum_final = float(tk["cumulative_return"].iloc[-1])
            ann_vol = float(monthly_rets.std() * np.sqrt(12))
            ann_ret = cum_final ** (12 / months) - 1
            range_records.append({
                "ETF": ticker,
                "Cumulative Return": f"{(cum_final - 1):+.1%}",
                "Annualized Return": f"{ann_ret:+.1%}",
                "Annualized Volatility": f"{ann_vol:.1%}",
                "Best Month": f"{monthly_rets.max():+.1%}",
                "Worst Month": f"{monthly_rets.min():+.1%}",
                "Max Drawdown": f"{float(tk['drawdown'].min()):.1%}",
            })
        st.dataframe(pd.DataFrame(range_records), use_container_width=True, hide_index=True)

    if results.vix is not None:
        st.subheader("Market Volatility (VIX)")
        fig_vix = px.line(results.vix, x="month", y="vix", title="VIX Forecast")
        fig_vix.add_hline(y=40, line_dash="dash", line_color="red",
                           annotation_text="Extreme Fear (40)")
        fig_vix.update_layout(height=300)
        st.plotly_chart(fig_vix, use_container_width=True)


# ---------------------------------------------------------------------------
# Tab 3: Geographic Heatmap
# ---------------------------------------------------------------------------

with tab3:
    st.header("Geographic Real Estate Analysis")

    if results.real_estate is not None:
        re = results.real_estate
        final_re = re[re["month"] == re["month"].max()].copy()
        final_re["appreciation"] = (final_re["price_index"] - 100).round(1)
        final_re = final_re.sort_values("appreciation", ascending=True)

        # Color-coded bar chart (proxy for geographic heatmap without map tiles)
        fig_re = px.bar(
            final_re, x="appreciation", y="metro",
            orientation="h",
            color="appreciation",
            color_continuous_scale="RdYlGn",
            title=f"Real Estate Price Change (%, base=100, {months} months)",
        )
        fig_re.update_layout(height=500)
        st.plotly_chart(fig_re, use_container_width=True)

        # Time series by category
        st.subheader("Price Index Over Time")
        col_hub, col_risk = st.columns(2)

        ai_hubs = re[re["is_ai_hub"] == True]
        at_risk = re[re["is_ai_hub"] == False]

        with col_hub:
            fig_hub = px.line(ai_hubs, x="month", y="price_index", color="metro",
                               title="AI Hub Metros (🟢)")
            fig_hub.update_layout(height=350)
            st.plotly_chart(fig_hub, use_container_width=True)

        with col_risk:
            fig_risk = px.line(at_risk, x="month", y="price_index", color="metro",
                                title="Other Metros (⚠️)")
            fig_risk.update_layout(height=350)
            st.plotly_chart(fig_risk, use_container_width=True)

    if results.commodities is not None:
        st.subheader("Commodity Prices")
        fig_comm = px.line(results.commodities, x="month", y="price",
                            color="commodity", title="Commodity Price Trajectories")
        fig_comm.update_layout(height=400)
        st.plotly_chart(fig_comm, use_container_width=True)


# ---------------------------------------------------------------------------
# Tab 4: Labor Market
# ---------------------------------------------------------------------------

with tab4:
    st.header("Labor Market & Automation")

    # Vulnerability ranking
    labor_model = LaborMarketModel()
    vuln_df = labor_model.vulnerability_ranking()

    st.subheader("Automation Vulnerability Index")
    fig_vuln = px.bar(
        vuln_df.sort_values("composite_index", ascending=True),
        x="composite_index", y="occupation",
        orientation="h", color="category",
        title="Occupation Vulnerability Ranking (0-100)",
        hover_data=["employment", "ai_vulnerability", "robot_vulnerability", "avg_wage"],
    )
    fig_vuln.update_layout(height=600)
    st.plotly_chart(fig_vuln, use_container_width=True)

    if results.labor_aggregate is not None:
        st.subheader("Cumulative Job Displacement")

        agg = results.labor_aggregate
        fig_disp = go.Figure()
        fig_disp.add_trace(go.Scatter(
            x=agg["month"], y=agg["total_displaced"],
            fill="tozeroy", name="Displaced",
            line=dict(color="red"),
        ))
        fig_disp.add_trace(go.Scatter(
            x=agg["month"], y=agg["total_remaining"],
            name="Remaining",
            line=dict(color="green"),
        ))
        fig_disp.update_layout(
            title="Tracked Occupations: Displacement Over Time",
            yaxis_title="Workers",
            height=400,
        )
        st.plotly_chart(fig_disp, use_container_width=True)

        # Displacement rate
        fig_rate = px.area(agg, x="month", y="displacement_rate",
                            title="Displacement Rate (% of tracked workers)")
        fig_rate.update_layout(yaxis_tickformat=".1%", height=300)
        st.plotly_chart(fig_rate, use_container_width=True)

    # ABM insights
    if results.agent_summary is not None:
        st.subheader("Agent-Based Model Insights")
        abm = results.agent_summary

        col1, col2 = st.columns(2)
        with col1:
            fig_abm_emp = px.line(abm, x="month", y="abm_employment_rate",
                                   title="ABM Employment Rate")
            fig_abm_emp.update_layout(yaxis_tickformat=".1%", height=300)
            st.plotly_chart(fig_abm_emp, use_container_width=True)
        with col2:
            fig_abm_ai = px.line(abm, x="month", y="abm_firms_ai_adopted_pct",
                                  title="ABM: Firms AI Adoption Rate")
            fig_abm_ai.update_layout(yaxis_tickformat=".0%", height=300)
            st.plotly_chart(fig_abm_ai, use_container_width=True)


# ---------------------------------------------------------------------------
# Tab 5: Portfolio Recommendations
# ---------------------------------------------------------------------------

with tab5:
    st.header("Portfolio Optimization")

    p_col1, p_col2, p_col3 = st.columns(3)
    with p_col1:
        capital = st.number_input("Investment Capital ($)", value=500_000, step=50_000)
    with p_col2:
        risk_tol = st.selectbox("Risk Tolerance",
                                 ["conservative", "moderate", "moderate-aggressive", "aggressive"])
    with p_col3:
        opt_method = st.selectbox("Optimization Method",
                                   ["mean_variance", "risk_parity", "black_litterman", "kelly"])

    optimizer = PortfolioOptimizer()
    alloc = optimizer.optimize(scenario, results, opt_method, capital, risk_tol)

    # Allocation pie chart
    weights_df = pd.DataFrame([
        {"Asset": k, "Weight": v}
        for k, v in alloc.weights.items()
    ]).sort_values("Weight", ascending=False)

    col_pie, col_metrics = st.columns([1, 1])
    with col_pie:
        fig_pie = px.pie(weights_df, names="Asset", values="Weight",
                          title=f"Optimal Allocation ({opt_method})")
        fig_pie.update_layout(height=400)
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_metrics:
        st.subheader("Portfolio Metrics")
        st.metric("Expected Annual Return", f"{alloc.expected_return:.1%}")
        st.metric("Expected Volatility", f"{alloc.expected_volatility:.1%}")
        st.metric("Sharpe Ratio", f"{alloc.sharpe_ratio:.2f}")
        st.metric("Value at Risk (95%)", f"${alloc.var_95:,.0f}")

    # Scenario-weighted allocation
    st.subheader("Probability-Weighted Allocation (All Scenarios)")

    @st.cache_data(ttl=600, show_spinner="Computing scenario-weighted allocation...")
    def compute_scenario_weighted(_months: int, _capital: float, _risk_tol: str):
        _tree = ScenarioTree()
        _optimizer = PortfolioOptimizer()
        _engine = SimulationEngine()
        allocs = {}
        for code, scen in _tree.scenarios.items():
            res = _engine.run(scen, months=_months, run_agents=False)
            allocs[code] = _optimizer.optimize(scen, res, "mean_variance", _capital, _risk_tol)
        return _optimizer.scenario_weighted_allocation(allocs)

    weighted = compute_scenario_weighted(months, capital, risk_tol)
    weighted_df = pd.DataFrame([
        {"Asset": k, "Weight": v} for k, v in weighted.items()
    ]).sort_values("Weight", ascending=False)

    fig_weighted = px.bar(weighted_df, x="Asset", y="Weight",
                           title="Scenario-Probability-Weighted Allocation",
                           color="Weight", color_continuous_scale="Viridis")
    fig_weighted.update_layout(yaxis_tickformat=".1%", height=400)
    st.plotly_chart(fig_weighted, use_container_width=True)

    # Rebalancing triggers
    st.subheader("⚡ Rebalancing Triggers")
    for trigger in alloc.rebalancing_triggers:
        urgency_color = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(trigger["urgency"], "⚪")
        st.markdown(f"{urgency_color} **{trigger['condition']}** → {trigger['action']}")

    # Monte Carlo fan chart
    st.subheader("Monte Carlo Distribution")
    mc_sims = st.slider("Number of MC simulations", 50, 500, 100, 50)

    @st.cache_data(ttl=600, show_spinner="Running Monte Carlo...")
    def run_mc(_code: str, _n: int):
        mc_config = MonteCarloConfig(n_simulations=_n, run_agents=False)
        mc_engine = MonteCarloEngine(mc_config)
        _tree = ScenarioTree()
        return mc_engine.run_scenario(_tree.get(_code), n_sims=_n)

    mc_result = run_mc(scenario_code, mc_sims)
    mc_summary = mc_result.summary()

    col_mc1, col_mc2 = st.columns(2)
    with col_mc1:
        st.markdown("**Peak Unemployment Distribution**")
        unemp_data = pd.DataFrame({"peak_unemployment": mc_result.peak_unemployment})
        fig_mc_u = px.histogram(unemp_data, x="peak_unemployment", nbins=30,
                                 title="Peak Unemployment (MC Distribution)")
        fig_mc_u.update_layout(xaxis_tickformat=".1%", height=300)
        st.plotly_chart(fig_mc_u, use_container_width=True)

        st.json(mc_summary["peak_unemployment"])

    with col_mc2:
        st.markdown("**GDP Growth Distribution**")
        gdp_data = pd.DataFrame({"avg_gdp_growth": mc_result.avg_gdp_growth})
        fig_mc_g = px.histogram(gdp_data, x="avg_gdp_growth", nbins=30,
                                 title="Avg GDP Growth (MC Distribution)")
        fig_mc_g.update_layout(xaxis_tickformat=".1%", height=300)
        st.plotly_chart(fig_mc_g, use_container_width=True)

        st.json(mc_summary["avg_gdp_growth"])

    # Fat tail risks
    st.subheader("🚨 Fat Tail Risks")
    fat_tails = mc_summary.get("fat_tail_risks", {})
    for risk, prob in fat_tails.items():
        color = "🔴" if prob > 0.2 else "🟡" if prob > 0.05 else "🟢"
        st.markdown(f"{color} **{risk}**: {prob:.1%}")
