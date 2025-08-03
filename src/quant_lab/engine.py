import pandas as pd
from .data_loader import get_historical_option_data
from .metrics import calculate_sharpe_ratio, calculate_max_drawdown

class BacktestEngine:
    """
    A simple backtesting calculator.
    It takes a series of signals and calculates performance metrics.
    """
    def run(self, price_data: pd.Series, signals: pd.Series):
        """
        Calculates performance metrics based on price data and signals.

        Args:
            price_data (pd.Series): A series of asset prices.
            signals (pd.Series): A series of trading signals (1, -1, or 0).

        Returns:
            dict: A dictionary containing the performance metrics.
        """
        print("Calculating backtest results...")

        # Ensure signals and prices are aligned
        if not isinstance(signals.index, pd.DatetimeIndex):
            signals.index = price_data.index[:len(signals)]
        signals = signals.reindex(price_data.index, method='ffill').fillna(0)

        positions = signals.shift(1)
        returns = price_data.pct_change() * positions
        returns = returns.dropna()

        if returns.empty or returns.sum() == 0:
            print("No trades were executed or no profit was made.")
            return {"sharpe_ratio": 0, "max_drawdown": 0, "total_return": 0}

        cumulative_returns = (1 + returns).cumprod()

        sharpe = calculate_sharpe_ratio(returns)
        max_dd = calculate_max_drawdown(returns)

        return {
            "sharpe_ratio": sharpe,
            "max_drawdown": max_dd,
            "total_return": cumulative_returns.iloc[-1]
        }