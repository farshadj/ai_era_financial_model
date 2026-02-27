#!/usr/bin/env python3
"""
Run a full simulation for one or all scenarios and print summary metrics.

Usage:
  python scripts/run_simulation.py                   # all scenarios
  python scripts/run_simulation.py --scenario A1      # single scenario
  python scripts/run_simulation.py --months 120 --agents
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.scenarios import ScenarioTree
from src.simulation.engine import SimulationEngine


def main():
    parser = argparse.ArgumentParser(description="Run AI economy simulation")
    parser.add_argument("--scenario", "-s", type=str, default=None,
                        help="Scenario code (A1, A2, B1, B2, C1, C2, D). Omit to run all.")
    parser.add_argument("--months", "-m", type=int, default=60,
                        help="Forecast horizon in months (default: 60)")
    parser.add_argument("--agents", action="store_true",
                        help="Enable agent-based model (slower)")
    args = parser.parse_args()

    tree = ScenarioTree()
    engine = SimulationEngine()

    if args.scenario:
        scenarios = {args.scenario: tree.get(args.scenario)}
    else:
        scenarios = tree.scenarios

    print("=" * 70)
    print("  AI ECONOMY FORECASTING MODEL – Simulation Run")
    print(f"  Horizon: {args.months} months  |  ABM: {'ON' if args.agents else 'OFF'}")
    print("=" * 70)

    for code, scen in scenarios.items():
        print(f"\n{'─' * 60}")
        print(f"  Scenario {code}: {scen.name}")
        print(f"  Probability: {scen.probability:.1%}")
        print(f"{'─' * 60}")

        results = engine.run(scen, months=args.months, run_agents=args.agents)
        metrics = results.summary_metrics()

        print(f"  Peak Unemployment:      {metrics['peak_unemployment']:.1%}")
        print(f"  Final Unemployment:     {metrics['final_unemployment']:.1%}")
        print(f"  Avg GDP Growth:         {metrics['avg_gdp_growth']:.2%}")
        print(f"  Final AI Adoption:      {metrics['final_ai_adoption']:.0%}")
        print(f"  Final Robot Fleet:      {metrics['final_robot_fleet']:,.0f}")
        print(f"  Total Displaced:        {metrics['total_displaced']:,.0f}")
        print(f"  Displacement Rate:      {metrics['displacement_rate']:.1%}")

    print(f"\n{'=' * 70}")
    print("  Simulation complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
