import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, Union
import logging

logger = logging.getLogger(__name__)

class AdaptiveTripleBarrierLabeler:
    """
    Advanced triple barrier labeling with dynamic parameters that adapt to market conditions.
    Designed to improve risk management and reduce drawdown.
    """
    
    def __init__(self, 
                 base_profit_target: float = 0.10,
                 base_stop_loss: float = 0.05,
                 base_time_horizon: int = 10,
                 volatility_adjustment: bool = True,
                 regime_adjustment: bool = True,
                 min_profit_target: float = 0.03,
                 max_profit_target: float = 0.25,
                 min_stop_loss: float = 0.02,
                 max_stop_loss: float = 0.15):
        """
        Initialize adaptive triple barrier labeler.
        
        Args:
            base_profit_target: Base profit target percentage
            base_stop_loss: Base stop loss percentage
            base_time_horizon: Base time horizon in periods
            volatility_adjustment: Whether to adjust barriers based on volatility
            regime_adjustment: Whether to adjust barriers based on market regime
            min_profit_target: Minimum allowed profit target
            max_profit_target: Maximum allowed profit target
            min_stop_loss: Minimum allowed stop loss
            max_stop_loss: Maximum allowed stop loss
        """
        self.base_profit_target = base_profit_target
        self.base_stop_loss = base_stop_loss
        self.base_time_horizon = base_time_horizon
        self.volatility_adjustment = volatility_adjustment
        self.regime_adjustment = regime_adjustment
        
        # Bounds for adaptive adjustments
        self.min_profit_target = min_profit_target
        self.max_profit_target = max_profit_target
        self.min_stop_loss = min_stop_loss
        self.max_stop_loss = max_stop_loss
        
        # Volatility lookback periods
        self.short_vol_window = 5
        self.long_vol_window = 20
        
    def get_adaptive_labels(self, 
                           prices: pd.Series,
                           market_data: Optional[pd.DataFrame] = None,
                           **kwargs) -> pd.Series:
        """
        Generate triple barrier labels with adaptive parameters.
        
        Args:
            prices: Price series
            market_data: Additional market data for adaptive adjustments
            **kwargs: Additional parameters
            
        Returns:
            Series of labels (1=win, -1=loss, 0=timeout)
        """
        logger.info("Generating adaptive triple barrier labels")
        
        # Calculate adaptive parameters for each period
        adaptive_params = self._calculate_adaptive_parameters(prices, market_data)
        
        # Apply triple barrier method with adaptive parameters
        outcomes = self._apply_adaptive_barriers(prices, adaptive_params)
        
        return outcomes
    
    def _calculate_adaptive_parameters(self, 
                                     prices: pd.Series,
                                     market_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Calculate adaptive parameters for each time period."""
        
        # Initialize with base parameters
        params = pd.DataFrame(index=prices.index)
        params['profit_target'] = self.base_profit_target
        params['stop_loss'] = self.base_stop_loss
        params['time_horizon'] = self.base_time_horizon
        
        if self.volatility_adjustment:
            params = self._adjust_for_volatility(params, prices)
        
        if self.regime_adjustment and market_data is not None:
            params = self._adjust_for_regime(params, market_data)
        
        # Apply bounds
        params['profit_target'] = params['profit_target'].clip(
            lower=self.min_profit_target, 
            upper=self.max_profit_target
        )
        params['stop_loss'] = params['stop_loss'].clip(
            lower=self.min_stop_loss, 
            upper=self.max_stop_loss
        )
        
        return params
    
    def _adjust_for_volatility(self, params: pd.DataFrame, prices: pd.Series) -> pd.DataFrame:
        """Adjust barrier parameters based on volatility regime."""
        
        # Calculate rolling volatility
        returns = prices.pct_change()
        short_vol = returns.rolling(window=self.short_vol_window).std() * np.sqrt(252)
        long_vol = returns.rolling(window=self.long_vol_window).std() * np.sqrt(252)
        
        # Volatility ratio (current vs historical)
        vol_ratio = short_vol / long_vol
        vol_ratio = vol_ratio.fillna(1.0)
        
        # Volatility z-score
        vol_zscore = (short_vol - long_vol.rolling(window=60).mean()) / long_vol.rolling(window=60).std()
        vol_zscore = vol_zscore.fillna(0.0)
        
        # Adjust parameters based on volatility
        # High volatility: wider barriers, shorter time horizon
        # Low volatility: tighter barriers, longer time horizon
        
        vol_adjustment_factor = np.clip(vol_ratio, 0.5, 2.0)
        
        params['profit_target'] *= vol_adjustment_factor
        params['stop_loss'] *= vol_adjustment_factor
        
        # Adjust time horizon inversely with volatility
        time_adj_factor = np.clip(1.0 / vol_adjustment_factor, 0.5, 2.0)
        params['time_horizon'] = (params['time_horizon'] * time_adj_factor).astype(int)
        
        logger.debug(f"Applied volatility adjustments: avg vol_ratio={vol_adjustment_factor.mean():.3f}")
        
        return params
    
    def _adjust_for_regime(self, params: pd.DataFrame, market_data: pd.DataFrame) -> pd.DataFrame:
        """Adjust parameters based on market regime."""
        
        # Identify high-risk periods
        high_risk_conditions = []
        
        if 'high_risk_regime' in market_data.columns:
            high_risk_conditions.append(market_data['high_risk_regime'] == 1)
        
        if 'stress_indicator' in market_data.columns:
            high_risk_conditions.append(market_data['stress_indicator'] > 0.5)
        
        if 'vol_breakout' in market_data.columns:
            high_risk_conditions.append(market_data['vol_breakout'] == 1)
        
        # Combine risk conditions
        if high_risk_conditions:
            high_risk = pd.concat(high_risk_conditions, axis=1).any(axis=1)
        else:
            high_risk = pd.Series(False, index=params.index)
        
        # Identify safe trading conditions
        safe_conditions = []
        
        if 'safe_conditions' in market_data.columns:
            safe_conditions.append(market_data['safe_conditions'] == 1)
        
        if 'low_vol_regime' in market_data.columns:
            safe_conditions.append(market_data['low_vol_regime'] == 1)
        
        # Combine safe conditions
        if safe_conditions:
            safe_trading = pd.concat(safe_conditions, axis=1).all(axis=1)
        else:
            safe_trading = pd.Series(False, index=params.index)
        
        # Adjust parameters based on regime
        # High risk: wider stop losses, tighter profit targets, shorter horizon
        params.loc[high_risk, 'stop_loss'] *= 1.5
        params.loc[high_risk, 'profit_target'] *= 0.8
        params.loc[high_risk, 'time_horizon'] = (params.loc[high_risk, 'time_horizon'] * 0.7).astype(int)
        
        # Safe conditions: tighter stops, wider profits, longer horizon
        params.loc[safe_trading, 'stop_loss'] *= 0.8
        params.loc[safe_trading, 'profit_target'] *= 1.2
        params.loc[safe_trading, 'time_horizon'] = (params.loc[safe_trading, 'time_horizon'] * 1.3).astype(int)
        
        logger.debug(f"High risk periods: {high_risk.sum()}, Safe periods: {safe_trading.sum()}")
        
        return params
    
    def _apply_adaptive_barriers(self, prices: pd.Series, params: pd.DataFrame) -> pd.Series:
        """Apply triple barrier method with adaptive parameters."""
        
        outcomes = pd.Series(index=prices.index, data=0, dtype=int)
        
        max_horizon = int(params['time_horizon'].max())
        
        for i in range(len(prices) - max_horizon):
            if i >= len(params):
                break
                
            entry_price = prices.iloc[i]
            profit_target = params.iloc[i]['profit_target']
            stop_loss = params.iloc[i]['stop_loss']
            time_horizon = int(params.iloc[i]['time_horizon'])
            
            upper_barrier = entry_price * (1 + profit_target)
            lower_barrier = entry_price * (1 - stop_loss)
            
            # Look forward for barrier hits
            for j in range(1, min(time_horizon + 1, len(prices) - i)):
                current_price = prices.iloc[i + j]
                
                if current_price >= upper_barrier:
                    outcomes.iloc[i] = 1  # Win
                    break
                elif current_price <= lower_barrier:
                    outcomes.iloc[i] = -1  # Loss
                    break
        
        return outcomes
    
    def analyze_label_distribution(self, labels: pd.Series, 
                                  market_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Analyze the distribution of generated labels."""
        
        analysis = {
            'total_labels': len(labels),
            'wins': (labels == 1).sum(),
            'losses': (labels == -1).sum(),
            'timeouts': (labels == 0).sum(),
            'win_rate': (labels == 1).mean(),
            'loss_rate': (labels == -1).mean(),
            'timeout_rate': (labels == 0).mean()
        }
        
        # Add regime-specific analysis if market data available
        if market_data is not None:
            if 'high_risk_regime' in market_data.columns:
                high_risk_mask = market_data['high_risk_regime'] == 1
                high_risk_labels = labels[high_risk_mask]
                
                analysis['high_risk_analysis'] = {
                    'periods': high_risk_mask.sum(),
                    'win_rate': (high_risk_labels == 1).mean() if len(high_risk_labels) > 0 else 0,
                    'loss_rate': (high_risk_labels == -1).mean() if len(high_risk_labels) > 0 else 0
                }
            
            if 'safe_conditions' in market_data.columns:
                safe_mask = market_data['safe_conditions'] == 1
                safe_labels = labels[safe_mask]
                
                analysis['safe_conditions_analysis'] = {
                    'periods': safe_mask.sum(),
                    'win_rate': (safe_labels == 1).mean() if len(safe_labels) > 0 else 0,
                    'loss_rate': (safe_labels == -1).mean() if len(safe_labels) > 0 else 0
                }
        
        return analysis

class RiskAdjustedLabeler:
    """
    Specialized labeler focused on risk-adjusted outcomes to reduce drawdown.
    """
    
    def __init__(self, risk_free_rate: float = 0.02):
        self.risk_free_rate = risk_free_rate
    
    def get_risk_adjusted_labels(self, 
                                prices: pd.Series,
                                volatility: pd.Series,
                                time_horizon: int = 10) -> pd.Series:
        """
        Generate labels based on risk-adjusted returns (Sharpe-like measure).
        
        Args:
            prices: Price series
            volatility: Volatility series (annualized)
            time_horizon: Forward-looking horizon
            
        Returns:
            Labels based on risk-adjusted performance
        """
        returns = prices.pct_change(time_horizon).shift(-time_horizon)
        
        # Calculate risk-adjusted returns
        risk_adjusted_returns = (returns * 252 - self.risk_free_rate) / volatility
        
        # Generate labels based on risk-adjusted performance
        labels = pd.Series(0, index=prices.index)
        
        # Positive risk-adjusted return above threshold
        labels[risk_adjusted_returns > 0.5] = 1  # Good risk-adjusted return
        labels[risk_adjusted_returns < -0.5] = -1  # Poor risk-adjusted return
        
        return labels

# Backward compatibility function
def get_triple_barrier_labels(prices: pd.Series, 
                             profit_target: float, 
                             stop_loss: float, 
                             time_horizon: int,
                             market_data: Optional[pd.DataFrame] = None) -> pd.Series:
    """
    Backward compatibility wrapper with optional adaptive features.
    
    Args:
        prices: Price series
        profit_target: Base profit target
        stop_loss: Base stop loss
        time_horizon: Base time horizon
        market_data: Optional market data for adaptive adjustments
        
    Returns:
        Series of triple barrier labels
    """
    if market_data is not None:
        # Use adaptive labeler if market data is provided
        labeler = AdaptiveTripleBarrierLabeler(
            base_profit_target=profit_target,
            base_stop_loss=stop_loss,
            base_time_horizon=time_horizon
        )
        return labeler.get_adaptive_labels(prices, market_data)
    else:
        # Use original simple method
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

        return outcomes
