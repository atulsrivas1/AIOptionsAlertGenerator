import pandas as pd

def get_triple_barrier_labels(prices: pd.Series, profit_target: float, stop_loss: float, time_horizon: int):
    """
    Applies the Triple Barrier Method to a price series.

    Args:
        prices (pd.Series): A series of asset prices.
        profit_target (float): The percentage increase for the profit-taking barrier.
        stop_loss (float): The percentage decrease for the stop-loss barrier.
        time_horizon (int): The maximum number of periods to hold the position.

    Returns:
        pd.Series: A series containing the outcome labels (1 for win, -1 for loss, 0 for timeout).
    """
    outcomes = pd.Series(index=prices.index, data=0)

    for i in range(len(prices) - time_horizon):
        entry_price = prices.iloc[i]
        upper_barrier = entry_price * (1 + profit_target)
        lower_barrier = entry_price * (1 - stop_loss)

        for j in range(1, time_horizon + 1):
            current_price = prices.iloc[i + j]

            if current_price >= upper_barrier:
                outcomes.iloc[i] = 1  # Win
                break
            elif current_price <= lower_barrier:
                outcomes.iloc[i] = -1  # Loss
                break
        # If the loop completes without hitting a barrier, the outcome remains 0 (timeout)

    return outcomes
