#!/usr/bin/env python3
"""
Run Monte Carlo analysis for a given scenario and print summary statistics.

Usage:
  python scripts/run_monte_carlo.py --scenario A1
  python scripts/run_monte_carlo.py --scenario A1 --sims 500 --months 120
"""

import sys
import os
import argparse
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.scenarios import ScenarioTree
from src.simulation.monte_carlo import MonteCarloEngine, MonteCarloConfig


def main():
    parser = argparse.ArgumentParser(description="Run Monte Carlo simulation")
    parser.add_argument("--scenario", "-s", type=str, required=True,
                        help="Scenario code (A1, A2, B1, B2, C1, C2, D)")
    parser.add_argument("--sims", "-n", type=int, default=200,
                        help="Number of MC simulations (default: 200)")
    parser.add_argument("--months", "-m", type=int, default=60,
                        help="Forecast horizon (default: 60)")
    parser.add_argument("--sensitivity", action="store_true",
                        help="Run sensitivity analysis")
    args = parser.parse_args()

    tree = ScenarioTree()
    scenario = tree.get(args.scenario)

    print("=" * 60)
    print(f"  Monte Carlo Analysis – Scenario {args.scenario}: {scenario.name}")
    print(f"  Simulations: {args.sims}  |  Months: {args.months}")
    print("=" * 60)

    mc_config = MonteCarloConfig(n_simulations=args.sims, run_agents=False)
    mc = MonteCarloEngine(mc_config)
    result = mc.run_scenario(scenario, n_sims=args.sims)
    summary = result.summary()

    print("\n  Summary Statistics:")
    print("-" * 60)
    for metric, stats in summary.items():
        if isinstance(stats, dict):
            print(f"\n  {metric}:")
            for k, v in stats.items():
                if isinstance(v, float):
                    print(f"    {k}: {v:.4f}")
                else:
                    print(f"    {k}: {v}")

    if args.sensitivity:
        print(f"\n{'=' * 60}")
        print("  Sensitivity Analysis")
        print("=" * 60)
        sensitivity = mc.sensitivity_analysis(scenario, args.months)
        for param, data in sensitivity.items():
            print(f"\n  {param}:")
            print(f"    Low  → peak_unemp={data['low']['peak_unemployment']:.3f}, "
                  f"gdp={data['low']['avg_gdp_growth']:.4f}")
            print(f"    High → peak_unemp={data['high']['peak_unemployment']:.3f}, "
                  f"gdp={data['high']['avg_gdp_growth']:.4f}")

    print(f"\n{'=' * 60}")
    print("  Done.")


if __name__ == "__main__":
    main()
