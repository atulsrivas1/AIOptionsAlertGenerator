import numpy as np
import pandas as pd

def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0):
    """
    Calculates the annualized Sharpe Ratio of a returns series.

    Args:
        returns (pd.Series): A series of periodic returns.
        risk_free_rate (float, optional): The annualized risk-free rate. Defaults to 0.0.

    Returns:
        float: The annualized Sharpe Ratio.
    """
    # Assuming daily returns, 252 trading days in a year
    annualized_return = returns.mean() * 252
    annualized_volatility = returns.std() * np.sqrt(252)
    
    if annualized_volatility == 0:
        return 0.0

    sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility
    return sharpe_ratio

def calculate_max_drawdown(returns: pd.Series):
    """
    Calculates the Maximum Drawdown of a returns series.

    Args:
        returns (pd.Series): A series of periodic returns.

    Returns:
        float: The Maximum Drawdown.
    """
    cumulative_returns = (1 + returns).cumprod()
    peak = cumulative_returns.expanding(min_periods=1).max()
    drawdown = (cumulative_returns - peak) / peak
    max_drawdown = drawdown.min()
    return max_drawdown
