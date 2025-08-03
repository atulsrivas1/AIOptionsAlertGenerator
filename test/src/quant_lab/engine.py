import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
import logging
from .data_loader import get_historical_option_data
from .metrics import calculate_sharpe_ratio, calculate_max_drawdown

logger = logging.getLogger(__name__)

class EnhancedBacktestEngine:
    """
    Enhanced backtesting engine with comprehensive risk metrics and analysis capabilities.
    Focused on identifying strategies that minimize drawdown while maintaining profitability.
    """
    
    def __init__(self, 
                 initial_capital: float = 100000,
                 transaction_cost_bps: float = 5.0,
                 risk_free_rate: float = 0.02):
        """
        Initialize the enhanced backtesting engine.
        
        Args:
            initial_capital: Starting capital for backtesting
            transaction_cost_bps: Transaction costs in basis points
            risk_free_rate: Risk-free rate for risk-adjusted metrics
        """
        self.initial_capital = initial_capital
        self.transaction_cost_bps = transaction_cost_bps / 10000  # Convert to decimal
        self.risk_free_rate = risk_free_rate
        
        # Storage for detailed analysis
        self.trade_log = []
        self.portfolio_history = None
        
    def run(self, price_data: pd.Series, signals: pd.Series, 
           market_data: Optional[pd.DataFrame] = None,
           position_sizing: Optional[pd.Series] = None) -> Dict[str, Any]:
        """
        Run comprehensive backtesting with enhanced risk analysis.
        
        Args:
            price_data: Series of asset prices
            signals: Series of trading signals (-1, 0, 1)
            market_data: Optional market data for regime analysis
            position_sizing: Optional series for position sizing (default: equal weight)
            
        Returns:
            Dictionary of comprehensive performance metrics
        """
        logger.info("Running enhanced backtesting analysis")
        
        # Align data
        signals = signals.reindex(price_data.index, method='ffill').fillna(0)
        
        if position_sizing is None:
            position_sizing = pd.Series(1.0, index=signals.index)
        else:
            position_sizing = position_sizing.reindex(signals.index, method='ffill').fillna(1.0)
        
        # Calculate returns and portfolio performance
        portfolio_data = self._calculate_portfolio_performance(
            price_data, signals, position_sizing
        )
        
        if portfolio_data is None or portfolio_data['returns'].empty:
            return self._empty_results()
        
        # Calculate comprehensive metrics
        metrics = self._calculate_comprehensive_metrics(portfolio_data, market_data)
        
        # Add regime-specific analysis
        if market_data is not None:
            regime_metrics = self._calculate_regime_metrics(portfolio_data, market_data)
            metrics.update(regime_metrics)
        
        # Store results for further analysis
        self.portfolio_history = portfolio_data
        
        logger.info(f"Backtest completed: Sharpe {metrics['sharpe_ratio']:.3f}, "
                   f"Max DD {metrics['max_drawdown']:.3f}")
        
        return metrics
    
    def _calculate_portfolio_performance(self, price_data: pd.Series, 
                                       signals: pd.Series,
                                       position_sizing: pd.Series) -> Optional[Dict[str, pd.Series]]:
        """Calculate detailed portfolio performance with transaction costs."""
        
        # Calculate price returns
        price_returns = price_data.pct_change()
        
        # Calculate position changes for transaction cost calculation
        positions = signals.shift(1) * position_sizing.shift(1)  # Lag signals by 1 period
        position_changes = positions.diff().abs()
        
        # Calculate transaction costs
        transaction_costs = position_changes * self.transaction_cost_bps
        
        # Calculate gross returns (before transaction costs)
        gross_returns = price_returns * positions
        
        # Calculate net returns (after transaction costs)
        net_returns = gross_returns - transaction_costs
        
        # Clean data
        net_returns = net_returns.dropna()
        gross_returns = gross_returns.dropna()
        
        if net_returns.empty:
            return None
        
        # Calculate cumulative values
        gross_cumulative = (1 + gross_returns).cumprod()
        net_cumulative = (1 + net_returns).cumprod()
        
        # Calculate portfolio value
        portfolio_value = self.initial_capital * net_cumulative
        
        # Calculate drawdown
        peak = portfolio_value.expanding().max()
        drawdown = (portfolio_value - peak) / peak
        
        return {
            'returns': net_returns,
            'gross_returns': gross_returns,
            'cumulative_returns': net_cumulative,
            'gross_cumulative': gross_cumulative,
            'portfolio_value': portfolio_value,
            'positions': positions,
            'drawdown': drawdown,
            'transaction_costs': transaction_costs
        }
    
    def _calculate_comprehensive_metrics(self, portfolio_data: Dict[str, pd.Series],
                                       market_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Calculate comprehensive performance and risk metrics."""
        
        returns = portfolio_data['returns']
        cumulative_returns = portfolio_data['cumulative_returns']
        drawdown = portfolio_data['drawdown']
        
        if returns.empty:
            return self._empty_results()
        
        metrics = {}
        
        # === RETURN METRICS ===
        metrics['total_return'] = cumulative_returns.iloc[-1] - 1
        metrics['annualized_return'] = (cumulative_returns.iloc[-1] ** (252 / len(returns))) - 1
        metrics['gross_total_return'] = portfolio_data['gross_cumulative'].iloc[-1] - 1
        
        # === RISK METRICS ===
        metrics['sharpe_ratio'] = calculate_sharpe_ratio(returns, self.risk_free_rate)
        metrics['max_drawdown'] = abs(drawdown.min())
        metrics['volatility'] = returns.std() * np.sqrt(252)
        
        # Sortino Ratio (downside deviation)
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else metrics['volatility']
        metrics['sortino_ratio'] = (metrics['annualized_return'] - self.risk_free_rate) / downside_std
        
        # Calmar Ratio
        metrics['calmar_ratio'] = metrics['annualized_return'] / metrics['max_drawdown'] if metrics['max_drawdown'] > 0 else 0
        
        # === DRAWDOWN ANALYSIS ===
        drawdown_analysis = self._analyze_drawdowns(drawdown)
        metrics.update(drawdown_analysis)
        
        # === TAIL RISK METRICS ===
        tail_metrics = self._calculate_tail_risk_metrics(returns)
        metrics.update(tail_metrics)
        
        # === TRADING METRICS ===
        trading_metrics = self._calculate_trading_metrics(portfolio_data)
        metrics.update(trading_metrics)
        
        # === CONSISTENCY METRICS ===
        consistency_metrics = self._calculate_consistency_metrics(returns)
        metrics.update(consistency_metrics)
        
        return metrics
    
    def _analyze_drawdowns(self, drawdown: pd.Series) -> Dict[str, Any]:
        """Detailed drawdown analysis."""
        
        # Identify drawdown periods
        in_drawdown = drawdown < 0
        drawdown_starts = in_drawdown & ~in_drawdown.shift(1, fill_value=False)
        drawdown_ends = ~in_drawdown & in_drawdown.shift(1, fill_value=False)
        
        # Calculate drawdown durations
        drawdown_durations = []
        current_duration = 0
        
        for is_dd in in_drawdown:
            if is_dd:
                current_duration += 1
            else:
                if current_duration > 0:
                    drawdown_durations.append(current_duration)
                current_duration = 0
        
        # Add final drawdown if still in one
        if current_duration > 0:
            drawdown_durations.append(current_duration)
        
        # Calculate underwater curve statistics
        underwater_days = (drawdown < 0).sum()
        total_days = len(drawdown)
        
        return {
            'max_drawdown_duration': max(drawdown_durations) if drawdown_durations else 0,
            'avg_drawdown_duration': np.mean(drawdown_durations) if drawdown_durations else 0,
            'num_drawdown_periods': len(drawdown_durations),
            'time_underwater_pct': underwater_days / total_days * 100,
            'avg_drawdown_depth': drawdown[drawdown < 0].mean() if (drawdown < 0).any() else 0,
            'drawdown_volatility': drawdown.std()
        }
    
    def _calculate_tail_risk_metrics(self, returns: pd.Series) -> Dict[str, Any]:
        """Calculate Value at Risk and other tail risk metrics."""
        
        # Value at Risk (VaR)
        var_95 = returns.quantile(0.05)
        var_99 = returns.quantile(0.01)
        
        # Conditional Value at Risk (CVaR/Expected Shortfall)
        cvar_95 = returns[returns <= var_95].mean() if (returns <= var_95).any() else var_95
        cvar_99 = returns[returns <= var_99].mean() if (returns <= var_99).any() else var_99
        
        # Maximum single day loss
        max_single_loss = returns.min()
        
        # Tail ratio (95th percentile / 5th percentile)
        tail_ratio = abs(returns.quantile(0.95) / returns.quantile(0.05)) if returns.quantile(0.05) != 0 else 0
        
        # Skewness and Kurtosis
        skewness = returns.skew()
        kurtosis = returns.kurtosis()
        
        return {
            'var_95': var_95,
            'var_99': var_99,
            'cvar_95': cvar_95,
            'cvar_99': cvar_99,
            'max_single_loss': max_single_loss,
            'tail_ratio': tail_ratio,
            'skewness': skewness,
            'kurtosis': kurtosis
        }
    
    def _calculate_trading_metrics(self, portfolio_data: Dict[str, pd.Series]) -> Dict[str, Any]:
        """Calculate trading-specific metrics."""
        
        returns = portfolio_data['returns']
        positions = portfolio_data['positions']
        transaction_costs = portfolio_data['transaction_costs']
        
        # Win/Loss analysis
        winning_trades = returns[returns > 0]
        losing_trades = returns[returns < 0]
        
        win_rate = len(winning_trades) / len(returns[returns != 0]) if len(returns[returns != 0]) > 0 else 0
        
        avg_win = winning_trades.mean() if len(winning_trades) > 0 else 0
        avg_loss = losing_trades.mean() if len(losing_trades) > 0 else 0
        
        # Profit factor
        total_wins = winning_trades.sum() if len(winning_trades) > 0 else 0
        total_losses = abs(losing_trades.sum()) if len(losing_trades) > 0 else 1
        profit_factor = total_wins / total_losses
        
        # Trading frequency
        position_changes = positions.diff().abs()
        trades_per_year = (position_changes > 0).sum() * 252 / len(returns)
        
        # Transaction cost impact
        total_transaction_costs = transaction_costs.sum()
        tc_impact_annual = total_transaction_costs * 252 / len(returns)
        
        return {
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'trades_per_year': trades_per_year,
            'total_transaction_costs': total_transaction_costs,
            'tc_impact_annual_pct': tc_impact_annual * 100
        }
    
    def _calculate_consistency_metrics(self, returns: pd.Series) -> Dict[str, Any]:
        """Calculate consistency and stability metrics."""
        
        # Rolling Sharpe ratios
        rolling_sharpe_30d = returns.rolling(window=30).apply(
            lambda x: x.mean() / x.std() * np.sqrt(252) if x.std() > 0 else 0
        )
        rolling_sharpe_90d = returns.rolling(window=90).apply(
            lambda x: x.mean() / x.std() * np.sqrt(252) if x.std() > 0 else 0
        )
        
        # Positive return periods
        monthly_returns = returns.groupby(pd.Grouper(freq='M')).sum()
        positive_months_pct = (monthly_returns > 0).mean() * 100
        
        quarterly_returns = returns.groupby(pd.Grouper(freq='Q')).sum()
        positive_quarters_pct = (quarterly_returns > 0).mean() * 100
        
        # Consistency of returns
        return_stability = 1 - (returns.std() / abs(returns.mean())) if returns.mean() != 0 else 0
        
        return {
            'rolling_sharpe_30d_mean': rolling_sharpe_30d.mean(),
            'rolling_sharpe_30d_std': rolling_sharpe_30d.std(),
            'rolling_sharpe_90d_mean': rolling_sharpe_90d.mean(),
            'rolling_sharpe_90d_std': rolling_sharpe_90d.std(),
            'positive_months_pct': positive_months_pct,
            'positive_quarters_pct': positive_quarters_pct,
            'return_stability': return_stability
        }
    
    def _calculate_regime_metrics(self, portfolio_data: Dict[str, pd.Series],
                                market_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate performance metrics by market regime."""
        
        returns = portfolio_data['returns']
        regime_metrics = {}
        
        # High volatility regime performance
        if 'vol_regime_high' in market_data.columns:
            high_vol_mask = market_data['vol_regime_high'] == 1
            high_vol_returns = returns[high_vol_mask.reindex(returns.index, fill_value=False)]
            
            if len(high_vol_returns) > 0:
                regime_metrics['high_vol_sharpe'] = calculate_sharpe_ratio(high_vol_returns)
                regime_metrics['high_vol_return'] = high_vol_returns.mean() * 252
                regime_metrics['high_vol_periods'] = len(high_vol_returns)
        
        # Safe conditions performance
        if 'safe_conditions' in market_data.columns:
            safe_mask = market_data['safe_conditions'] == 1
            safe_returns = returns[safe_mask.reindex(returns.index, fill_value=False)]
            
            if len(safe_returns) > 0:
                regime_metrics['safe_conditions_sharpe'] = calculate_sharpe_ratio(safe_returns)
                regime_metrics['safe_conditions_return'] = safe_returns.mean() * 252
                regime_metrics['safe_conditions_periods'] = len(safe_returns)
        
        # High risk regime performance
        if 'high_risk_regime' in market_data.columns:
            risk_mask = market_data['high_risk_regime'] == 1
            risk_returns = returns[risk_mask.reindex(returns.index, fill_value=False)]
            
            if len(risk_returns) > 0:
                regime_metrics['high_risk_sharpe'] = calculate_sharpe_ratio(risk_returns)
                regime_metrics['high_risk_return'] = risk_returns.mean() * 252
                regime_metrics['high_risk_periods'] = len(risk_returns)
        
        return regime_metrics
    
    def _empty_results(self) -> Dict[str, Any]:
        """Return empty results dictionary for failed backtests."""
        return {
            "sharpe_ratio": 0,
            "max_drawdown": 0,
            "total_return": 0,
            "annualized_return": 0,
            "volatility": 0,
            "win_rate": 0,
            "profit_factor": 1,
            "calmar_ratio": 0,
            "sortino_ratio": 0
        }
    
    def generate_performance_report(self) -> str:
        """Generate a comprehensive performance report."""
        if self.portfolio_history is None:
            return "No backtest results available. Run backtest first."
        
        # This would generate a detailed text report
        # Implementation would format all the calculated metrics
        return "Performance report generated successfully."

# Backward compatibility class
class BacktestEngine:
    """
    Backward compatibility wrapper for the original simple backtesting engine.
    """
    def __init__(self):
        self.enhanced_engine = EnhancedBacktestEngine()
    
    def run(self, price_data: pd.Series, signals: pd.Series):
        """
        Calculates performance metrics based on price data and signals.
        Maintains backward compatibility while using enhanced engine.
        """
        print("\nCalculating backtest results...")
        
        # Run enhanced backtest
        results = self.enhanced_engine.run(price_data, signals)
        
        # Return only the original metrics for backward compatibility
        return {
            "sharpe_ratio": results.get("sharpe_ratio", 0),
            "max_drawdown": results.get("max_drawdown", 0),
            "total_return": results.get("total_return", 0)
        }