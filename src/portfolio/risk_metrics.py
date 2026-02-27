"""
Risk metrics: VaR, CVaR, Sharpe, max drawdown, etc.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def calculate_var(
    returns: np.ndarray,
    confidence: float = 0.95,
    capital: float = 1_000_000,
) -> float:
    """Calculate Value at Risk (parametric and historical)."""
    sorted_returns = np.sort(returns)
    idx = int((1 - confidence) * len(sorted_returns))
    return float(-sorted_returns[idx] * capital)


def calculate_cvar(
    returns: np.ndarray,
    confidence: float = 0.95,
    capital: float = 1_000_000,
) -> float:
    """Calculate Conditional Value at Risk (Expected Shortfall)."""
    sorted_returns = np.sort(returns)
    idx = int((1 - confidence) * len(sorted_returns))
    tail = sorted_returns[:idx]
    if len(tail) == 0:
        return 0.0
    return float(-np.mean(tail) * capital)


def calculate_sharpe(
    returns: np.ndarray,
    risk_free_rate: float = 0.045,
    periods_per_year: int = 12,
) -> float:
    """Calculate annualized Sharpe ratio."""
    excess = returns - risk_free_rate / periods_per_year
    if np.std(excess) < 1e-10:
        return 0.0
    return float(np.mean(excess) / np.std(excess) * np.sqrt(periods_per_year))


def calculate_max_drawdown(cumulative_returns: np.ndarray) -> float:
    """Calculate maximum drawdown from a cumulative return series."""
    peak = np.maximum.accumulate(cumulative_returns)
    drawdown = (cumulative_returns - peak) / peak
    return float(np.min(drawdown))


def calculate_sortino(
    returns: np.ndarray,
    risk_free_rate: float = 0.045,
    periods_per_year: int = 12,
) -> float:
    """Calculate annualized Sortino ratio (downside deviation only)."""
    excess = returns - risk_free_rate / periods_per_year
    downside = excess[excess < 0]
    if len(downside) == 0 or np.std(downside) < 1e-10:
        return 0.0
    return float(np.mean(excess) / np.std(downside) * np.sqrt(periods_per_year))


def portfolio_metrics(
    weights: Dict[str, float],
    returns_df: pd.DataFrame,
    capital: float = 500_000,
    risk_free_rate: float = 0.045,
) -> Dict[str, float]:
    """
    Calculate comprehensive risk metrics for a portfolio.

    Args:
        weights: Asset → weight mapping
        returns_df: DataFrame with columns matching asset names, rows = monthly returns
        capital: Total capital
        risk_free_rate: Annual risk-free rate
    """
    # Compute portfolio returns
    port_returns = np.zeros(len(returns_df))
    for asset, weight in weights.items():
        if asset in returns_df.columns:
            port_returns += weight * returns_df[asset].values

    cumulative = np.cumprod(1 + port_returns)

    return {
        "annualized_return": float(np.mean(port_returns) * 12),
        "annualized_volatility": float(np.std(port_returns) * np.sqrt(12)),
        "sharpe_ratio": calculate_sharpe(port_returns, risk_free_rate),
        "sortino_ratio": calculate_sortino(port_returns, risk_free_rate),
        "var_95": calculate_var(port_returns, 0.95, capital),
        "cvar_95": calculate_cvar(port_returns, 0.95, capital),
        "max_drawdown": calculate_max_drawdown(cumulative),
        "total_return": float(cumulative[-1] - 1) if len(cumulative) > 0 else 0,
        "final_value": float(capital * cumulative[-1]) if len(cumulative) > 0 else capital,
    }


class RiskMetrics:
    """Class-based convenience wrapper around risk metric functions."""

    @staticmethod
    def value_at_risk(returns: np.ndarray, capital: float = 1_000_000,
                       confidence: float = 0.95) -> float:
        sorted_r = np.sort(returns)
        idx = int((1 - confidence) * len(sorted_r))
        return float(sorted_r[idx] * capital)  # negative number

    @staticmethod
    def conditional_var(returns: np.ndarray, capital: float = 1_000_000,
                         confidence: float = 0.95) -> float:
        sorted_r = np.sort(returns)
        idx = int((1 - confidence) * len(sorted_r))
        tail = sorted_r[:max(idx, 1)]
        return float(np.mean(tail) * capital)

    @staticmethod
    def sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.045,
                      periods_per_year: int = 252) -> float:
        excess = returns - risk_free_rate / periods_per_year
        if np.std(excess) < 1e-10:
            return 0.0
        return float(np.mean(excess) / np.std(excess) * np.sqrt(periods_per_year))

    @staticmethod
    def sortino_ratio(returns: np.ndarray, risk_free_rate: float = 0.045,
                       periods_per_year: int = 252) -> float:
        excess = returns - risk_free_rate / periods_per_year
        downside = excess[excess < 0]
        if len(downside) == 0 or np.std(downside) < 1e-10:
            return 0.0
        return float(np.mean(excess) / np.std(downside) * np.sqrt(periods_per_year))

    @staticmethod
    def max_drawdown(returns: np.ndarray) -> float:
        cumulative = np.cumprod(1 + returns)
        peak = np.maximum.accumulate(cumulative)
        dd = (cumulative - peak) / peak
        return float(np.min(dd))
