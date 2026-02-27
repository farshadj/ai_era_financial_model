#!/usr/bin/env python3
"""
Generate Markdown reports for all scenarios.

Usage:
  python scripts/generate_reports.py
  python scripts/generate_reports.py --scenario A1 --months 120
  python scripts/generate_reports.py --output reports/custom/
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.reports.generator import ReportGenerator, ReportConfig


def main():
    parser = argparse.ArgumentParser(description="Generate AI economy forecast reports")
    parser.add_argument("--scenario", "-s", type=str, default=None,
                        help="Generate report for one scenario (e.g. A1). Omit for all.")
    parser.add_argument("--months", "-m", type=int, default=60,
                        help="Forecast horizon (default: 60)")
    parser.add_argument("--mc-sims", type=int, default=200,
                        help="Monte Carlo simulations (default: 200)")
    parser.add_argument("--capital", type=float, default=500_000,
                        help="Portfolio capital (default: 500000)")
    parser.add_argument("--risk", type=str, default="moderate",
                        choices=["conservative", "moderate", "moderate-aggressive", "aggressive"])
    parser.add_argument("--output", "-o", type=str, default="reports",
                        help="Output directory (default: reports/)")
    args = parser.parse_args()

    config = ReportConfig(
        months=args.months,
        mc_simulations=args.mc_sims,
        run_agents=False,
        capital=args.capital,
        risk_tolerance=args.risk,
        output_dir=args.output,
    )

    gen = ReportGenerator(config)

    print("=" * 60)
    print("  AI Economy Forecast – Report Generator")
    print("=" * 60)

    if args.scenario:
        print(f"\n  Generating report for scenario {args.scenario}...")
        report = gen.generate_scenario_report(args.scenario)
        os.makedirs(config.output_dir, exist_ok=True)
        path = os.path.join(config.output_dir, f"report_{args.scenario}.md")
        with open(path, "w") as f:
            f.write(report)
        print(f"  ✓ Saved to {path}")
    else:
        print("\n  Generating reports for all 7 scenarios + blended report...")
        reports = gen.generate_all_reports()
        gen.save_reports(reports)

    print("\n  Done.")


if __name__ == "__main__":
    main()
