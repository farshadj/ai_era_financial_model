"""
Report Generator – Produces Markdown scenario reports.

Sections:
  1. Executive Summary
  2. Detailed Forecast  (macro, labor, financial)
  3. Investment Playbook (portfolio allocation, rebalancing triggers)
  4. Appendices         (Monte Carlo stats, sensitivity, methodology)
"""

from __future__ import annotations

import os
import datetime
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from src.models.scenarios import ScenarioTree, ScenarioParams
from src.models.labor_market import LaborMarketModel
from src.simulation.engine import SimulationEngine, SimulationResults
from src.simulation.monte_carlo import MonteCarloEngine, MonteCarloConfig
from src.portfolio.optimizer import PortfolioOptimizer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pct(v: float, decimals: int = 1) -> str:
    return f"{v * 100:.{decimals}f}%"


def _money(v: float) -> str:
    if abs(v) >= 1e12:
        return f"${v / 1e12:.1f}T"
    if abs(v) >= 1e9:
        return f"${v / 1e9:.1f}B"
    if abs(v) >= 1e6:
        return f"${v / 1e6:.1f}M"
    return f"${v:,.0f}"


# ---------------------------------------------------------------------------
# Report generator
# ---------------------------------------------------------------------------

@dataclass
class ReportConfig:
    months: int = 60
    mc_simulations: int = 200
    run_agents: bool = False
    capital: float = 500_000
    risk_tolerance: str = "moderate"
    output_dir: str = "reports"


class ReportGenerator:
    """Generates comprehensive Markdown reports per scenario or blended."""

    def __init__(self, config: Optional[ReportConfig] = None):
        self.config = config or ReportConfig()
        self.tree = ScenarioTree()
        self.engine = SimulationEngine()
        self.labor_model = LaborMarketModel()
        self.optimizer = PortfolioOptimizer()
        self.mc_engine = MonteCarloEngine(
            MonteCarloConfig(n_simulations=self.config.mc_simulations,
                             run_agents=self.config.run_agents)
        )

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def generate_scenario_report(self, scenario_code: str) -> str:
        """Generate a full Markdown report for a single scenario."""
        scenario = self.tree.get(scenario_code)
        results = self.engine.run(scenario, months=self.config.months,
                                   run_agents=self.config.run_agents)
        mc = self.mc_engine.run_scenario(scenario, n_sims=self.config.mc_simulations)
        mc_summary = mc.summary()
        alloc = self.optimizer.optimize(scenario, results, "mean_variance",
                                         self.config.capital, self.config.risk_tolerance)

        parts = [
            self._header(scenario),
            self._executive_summary(scenario, results, mc_summary),
            self._detailed_forecast(scenario, results),
            self._investment_playbook(alloc, results),
            self._appendices(mc_summary),
            self._footer(),
        ]
        return "\n\n".join(parts)

    def generate_all_reports(self) -> dict[str, str]:
        """Generate reports for every scenario and a master blended report."""
        reports: dict[str, str] = {}
        for code in self.tree.scenarios:
            reports[code] = self.generate_scenario_report(code)
        reports["BLENDED"] = self._blended_report()
        return reports

    def save_reports(self, reports: dict[str, str] | None = None):
        """Write all reports to disk."""
        if reports is None:
            reports = self.generate_all_reports()
        os.makedirs(self.config.output_dir, exist_ok=True)
        for name, content in reports.items():
            path = os.path.join(self.config.output_dir, f"report_{name}.md")
            with open(path, "w") as f:
                f.write(content)
        print(f"  ✓ Saved {len(reports)} reports to {self.config.output_dir}/")

    # ------------------------------------------------------------------
    # Section builders
    # ------------------------------------------------------------------

    def _header(self, scenario: ScenarioParams) -> str:
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        return f"""# AI Economy Forecast Report
## Scenario {scenario.code}: {scenario.name}
> Generated: {now}  |  Horizon: {self.config.months} months  |  Monte Carlo: {self.config.mc_simulations} sims

---"""

    def _executive_summary(self, scenario: ScenarioParams,
                            results: SimulationResults,
                            mc_summary: dict) -> str:
        metrics = results.summary_metrics()
        peak_u = mc_summary.get("peak_unemployment", {})
        gdp = mc_summary.get("avg_gdp_growth", {})
        fat_tails = mc_summary.get("fat_tail_risks", {})

        lines = [
            "## 1. Executive Summary\n",
            f"**Scenario probability:** {_pct(scenario.probability)}\n",
            "### Key Forecasts (median ± range)\n",
            f"| Metric | p10 | Median | p90 |",
            f"|--------|-----|--------|-----|",
            f"| Peak Unemployment | {_pct(peak_u.get('p10', 0))} | {_pct(peak_u.get('p50', 0))} | {_pct(peak_u.get('p90', 0))} |",
            f"| Avg GDP Growth | {_pct(gdp.get('p10', 0))} | {_pct(gdp.get('p50', 0))} | {_pct(gdp.get('p90', 0))} |",
            "",
            "### Fat-Tail Risks\n",
        ]
        for risk, prob in fat_tails.items():
            emoji = "🔴" if prob > 0.2 else "🟡" if prob > 0.05 else "🟢"
            lines.append(f"- {emoji} **{risk}**: {_pct(prob)}")

        return "\n".join(lines)

    def _detailed_forecast(self, scenario: ScenarioParams,
                            results: SimulationResults) -> str:
        lines = ["## 2. Detailed Forecast\n"]

        # Macro
        if results.macro is not None:
            m = results.macro
            lines.append("### 2.1 Macroeconomics\n")
            lines.append(f"- Peak unemployment: {_pct(m['unemployment_rate'].max())}")
            lines.append(f"- Final unemployment: {_pct(m['unemployment_rate'].iloc[-1])}")
            lines.append(f"- Avg annual GDP growth: {_pct(m['gdp_growth_annual'].mean())}")
            lines.append(f"- Final AI adoption: {_pct(m['ai_adoption_pct'].iloc[-1])}")
            lines.append(f"- Final robot fleet: {m['robot_fleet_size'].iloc[-1]:,.0f}")
            lines.append(f"- Final CPI overall: {_pct(m['cpi_overall'].iloc[-1])}")
            lines.append(f"- Fed funds rate (final): {_pct(m['fed_rate'].iloc[-1])}")

        # Labor
        if results.labor_aggregate is not None:
            la = results.labor_aggregate
            lines.append("\n### 2.2 Labor Market\n")
            final = la.iloc[-1]
            lines.append(f"- Workers displaced: {final['total_displaced']:,.0f}")
            lines.append(f"- Displacement rate: {_pct(final['displacement_rate'])}")
            vuln = self.labor_model.vulnerability_ranking()
            top3 = vuln.nlargest(3, "composite_index")
            lines.append("\n**Most vulnerable occupations:**\n")
            for _, row in top3.iterrows():
                lines.append(f"  1. {row['occupation']} (index {row['composite_index']:.1f})")

        # Financial
        if results.sectors is not None:
            lines.append("\n### 2.3 Financial Markets\n")
            final_s = results.sectors[results.sectors["month"] == results.sectors["month"].max()]
            top = final_s.nlargest(3, "cumulative_return")
            bot = final_s.nsmallest(3, "cumulative_return")
            lines.append("**Top sectors:**")
            for _, r in top.iterrows():
                lines.append(f"  - {r['sector']}: {_pct(r['cumulative_return'])}")
            lines.append("**Bottom sectors:**")
            for _, r in bot.iterrows():
                lines.append(f"  - {r['sector']}: {_pct(r['cumulative_return'])}")

        # Real estate
        if results.real_estate is not None:
            lines.append("\n### 2.4 Real Estate\n")
            re = results.real_estate
            final_re = re[re["month"] == re["month"].max()]
            final_re = final_re.copy()
            final_re["change"] = final_re["price_index"] - 100
            top_re = final_re.nlargest(3, "change")
            bot_re = final_re.nsmallest(3, "change")
            lines.append("**Strongest metros:**")
            for _, r in top_re.iterrows():
                lines.append(f"  - {r['metro']}: {r['change']:+.1f}%")
            lines.append("**Weakest metros:**")
            for _, r in bot_re.iterrows():
                lines.append(f"  - {r['metro']}: {r['change']:+.1f}%")

        return "\n".join(lines)

    def _investment_playbook(self, alloc, results: SimulationResults) -> str:
        lines = [
            "## 3. Investment Playbook\n",
            f"**Capital: {_money(self.config.capital)}**  |  Risk tolerance: {self.config.risk_tolerance}\n",
            "### 3.1 Optimal Allocation\n",
            "| Asset | Weight | Dollar Value |",
            "|-------|--------|-------------|",
        ]
        for asset, wt in sorted(alloc.weights.items(), key=lambda x: -x[1]):
            dollar = wt * self.config.capital
            lines.append(f"| {asset} | {_pct(wt)} | {_money(dollar)} |")

        lines.append(f"\n**Expected return:** {_pct(alloc.expected_return)}")
        lines.append(f"**Expected volatility:** {_pct(alloc.expected_volatility)}")
        lines.append(f"**Sharpe ratio:** {alloc.sharpe_ratio:.2f}")
        lines.append(f"**VaR (95%):** {_money(alloc.var_95)}")

        lines.append("\n### 3.2 Rebalancing Triggers\n")
        for t in alloc.rebalancing_triggers:
            icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(t["urgency"], "⚪")
            lines.append(f"- {icon} **{t['condition']}** → {t['action']}")

        return "\n".join(lines)

    def _appendices(self, mc_summary: dict) -> str:
        lines = [
            "## 4. Appendices\n",
            "### 4.1 Monte Carlo Statistics\n",
            "```",
        ]
        for metric, stats in mc_summary.items():
            if isinstance(stats, dict):
                lines.append(f"\n{metric}:")
                for k, v in stats.items():
                    if isinstance(v, float):
                        lines.append(f"  {k}: {v:.4f}")
                    else:
                        lines.append(f"  {k}: {v}")
        lines.append("```")

        lines.append("\n### 4.2 Methodology Notes\n")
        lines.append("- GDP: Okun's Law + AI productivity boost + robot capital deepening")
        lines.append("- Unemployment: Displacement logistic S-curve × AI adoption, adjusted by natural adjustment rate")
        lines.append("- Bifurcated CPI: Deflation basket (AI-produced goods) and inflation basket (scarce/human services)")
        lines.append("- Portfolio: Mean-Variance / Risk Parity / Black-Litterman / Kelly Criterion")
        lines.append("- ABM: Mesa-based agent model with worker, firm, investor, and government agents")
        lines.append("- Monte Carlo: Parameter randomization using log-normal perturbations")

        return "\n".join(lines)

    def _footer(self) -> str:
        return """---

*Disclaimer: This model is for educational and research purposes only.
All projections are based on synthetic scenarios and should not be construed
as investment advice. Past performance does not predict future results.*
"""

    # ------------------------------------------------------------------
    # Blended report
    # ------------------------------------------------------------------

    def _blended_report(self) -> str:
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        lines = [
            "# AI Economy Forecast – Blended Scenario Report",
            f"> Generated: {now}  |  Probability-weighted across all 7 scenarios\n",
            "---\n",
            "## Scenario Probabilities\n",
            "| Code | Name | Probability |",
            "|------|------|------------|",
        ]
        for code, scen in self.tree.scenarios.items():
            lines.append(f"| {code} | {scen.name} | {_pct(scen.probability)} |")

        # Run each scenario and compute weighted metrics
        lines.append("\n## Probability-Weighted Key Metrics\n")
        weighted_peak_u = 0.0
        weighted_gdp = 0.0
        allocs = {}
        for code, scen in self.tree.scenarios.items():
            res = self.engine.run(scen, months=self.config.months, run_agents=False)
            m = res.summary_metrics()
            weighted_peak_u += scen.probability * m["peak_unemployment"]
            weighted_gdp += scen.probability * m["avg_gdp_growth"]
            allocs[code] = self.optimizer.optimize(scen, res, "mean_variance",
                                                     self.config.capital, self.config.risk_tolerance)

        lines.append(f"- **Expected peak unemployment:** {_pct(weighted_peak_u)}")
        lines.append(f"- **Expected avg GDP growth:** {_pct(weighted_gdp)}")

        # Blended portfolio
        weighted_alloc = self.optimizer.scenario_weighted_allocation(allocs)
        lines.append("\n## Probability-Weighted Portfolio Allocation\n")
        lines.append("| Asset | Weight |")
        lines.append("|-------|--------|")
        for asset, wt in sorted(weighted_alloc.items(), key=lambda x: -x[1]):
            lines.append(f"| {asset} | {_pct(wt)} |")

        lines.append(self._footer())
        return "\n".join(lines)
