import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
import logging
from scipy.stats import zscore

logger = logging.getLogger(__name__)

class AdvancedFeatureEngineer:
    """
    Advanced feature engineering for options trading with focus on risk-adjusted features
    that can help reduce drawdown while maintaining profitability.
    """
    
    def __init__(self):
        self.feature_categories = {
            'technical': self._add_technical_features,
            'volatility': self._add_volatility_features,
            'options_specific': self._add_options_specific_features,
            'risk_management': self._add_risk_management_features,
            'market_structure': self._add_market_structure_features,
            'regime_detection': self._add_regime_detection_features
        }
    
    def engineer_features(self, data: pd.DataFrame, 
                         categories: Optional[list] = None) -> pd.DataFrame:
        """
        Apply comprehensive feature engineering.
        
        Args:
            data: Input OHLCV data with implied_volatility
            categories: List of feature categories to apply (default: all)
            
        Returns:
            DataFrame with engineered features
        """
        if categories is None:
            categories = list(self.feature_categories.keys())
        
        logger.info(f"Engineering features for categories: {categories}")
        
        # Ensure datetime index
        if not isinstance(data.index, pd.DatetimeIndex):
            data.index = pd.to_datetime(data.index)
        
        # Apply feature engineering by category
        for category in categories:
            if category in self.feature_categories:
                try:
                    data = self.feature_categories[category](data)
                    logger.info(f"Applied {category} features")
                except Exception as e:
                    logger.error(f"Error applying {category} features: {e}")
        
        return data
    
    def _add_technical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add technical analysis features with multiple timeframes."""
        # Multi-timeframe moving averages
        data['mavg_5d'] = data['close'].rolling(window=5).mean()
        data['mavg_10d'] = data['close'].rolling(window=10).mean()
        data['mavg_20d'] = data['close'].rolling(window=20).mean()
        data['mavg_50d'] = data['close'].rolling(window=50).mean()
        
        # Weekly and monthly resampled features
        weekly_close = data['close'].resample('W').last()
        monthly_close = data['close'].resample('ME').last()
        
        data['mavg_4w'] = weekly_close.rolling(window=4).mean().reindex(data.index, method='ffill')
        data['mavg_3m'] = monthly_close.rolling(window=3).mean().reindex(data.index, method='ffill')
        
        # Price position indicators
        data['close_vs_mavg_5d'] = data['close'] / data['mavg_5d']
        data['close_vs_mavg_20d'] = data['close'] / data['mavg_20d']
        data['close_vs_mavg_50d'] = data['close'] / data['mavg_50d']
        
        # Moving average convergence/divergence
        data['ma_spread_5_20'] = (data['mavg_5d'] - data['mavg_20d']) / data['mavg_20d']
        data['ma_spread_20_50'] = (data['mavg_20d'] - data['mavg_50d']) / data['mavg_50d']
        
        # RSI (Relative Strength Index)
        data['rsi_14'] = self._calculate_rsi(data['close'], window=14)
        data['rsi_30'] = self._calculate_rsi(data['close'], window=30)
        
        # Bollinger Bands
        bb_upper, bb_lower, bb_middle = self._calculate_bollinger_bands(data['close'])
        data['bb_position'] = (data['close'] - bb_lower) / (bb_upper - bb_lower)
        data['bb_width'] = (bb_upper - bb_lower) / bb_middle
        
        # Price momentum
        data['momentum_5d'] = data['close'].pct_change(5)
        data['momentum_10d'] = data['close'].pct_change(10)
        data['momentum_20d'] = data['close'].pct_change(20)
        
        return data
    
    def _add_volatility_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add sophisticated volatility features."""
        # Historical volatility (multiple windows)
        data['realized_vol_5d'] = data['close'].pct_change().rolling(window=5).std() * np.sqrt(252)
        data['realized_vol_10d'] = data['close'].pct_change().rolling(window=10).std() * np.sqrt(252)
        data['realized_vol_20d'] = data['close'].pct_change().rolling(window=20).std() * np.sqrt(252)
        data['realized_vol_60d'] = data['close'].pct_change().rolling(window=60).std() * np.sqrt(252)
        
        # Implied vs Realized volatility relationships
        data['iv_rv_ratio_20d'] = data['implied_volatility'] / data['realized_vol_20d']
        data['iv_rv_spread_20d'] = data['implied_volatility'] - data['realized_vol_20d']
        
        # Volatility regime indicators
        data['vol_regime_20d'] = self._calculate_volatility_regime(data['realized_vol_20d'])
        data['iv_regime'] = self._calculate_volatility_regime(data['implied_volatility'])
        
        # Volatility of volatility
        data['iv_vol_10d'] = data['implied_volatility'].rolling(window=10).std()
        data['iv_vol_20d'] = data['implied_volatility'].rolling(window=20).std()
        data['rv_vol_20d'] = data['realized_vol_20d'].rolling(window=20).std()
        
        # Add missing iv_std_20d for backward compatibility
        data['iv_std_20d'] = data['iv_vol_20d']  # Same as iv_vol_20d
        
        # Volatility mean reversion indicators
        data['iv_zscore_20d'] = self._calculate_zscore(data['implied_volatility'], window=20)
        data['iv_zscore_60d'] = self._calculate_zscore(data['implied_volatility'], window=60)
        
        # GARCH-like volatility proxy
        data['garch_vol'] = self._calculate_garch_proxy(data['close'].pct_change())
        
        return data
    
    def _add_options_specific_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add options-specific features for enhanced risk management."""
        # Volume-based features
        data['volume_ma_20d'] = data['volume'].rolling(window=20).mean()
        data['volume_ratio'] = data['volume'] / data['volume_ma_20d']
        data['volume_spike'] = (data['volume'] > data['volume_ma_20d'] * 2).astype(int)
        
        # Implied volatility interaction features
        data['iv_x_volume'] = data['implied_volatility'] * data['volume']
        data['iv_x_volume_normalized'] = data['iv_x_volume'] / data['iv_x_volume'].rolling(window=20).mean()
        
        # Options flow proxy (simplified)
        data['unusual_volume'] = (data['volume'] > data['volume'].rolling(window=20).quantile(0.95)).astype(int)
        data['high_iv_volume'] = ((data['implied_volatility'] > data['implied_volatility'].rolling(window=20).quantile(0.8)) & 
                                 (data['volume'] > data['volume_ma_20d'])).astype(int)
        
        # Time decay proxy (assuming daily data)
        # This would be more accurate with actual theta values
        data['time_decay_proxy'] = np.exp(-0.01 * np.arange(len(data)) % 252)  # Simplified time decay
        
        # Volatility surface stability (simplified)
        data['iv_stability'] = 1 / (1 + data['iv_vol_10d'])  # Inverse of IV volatility
        
        return data
    
    def _add_risk_management_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add features specifically designed for risk management and drawdown reduction."""
        # Drawdown indicators
        data['peak'] = data['close'].expanding().max()
        data['drawdown'] = (data['close'] - data['peak']) / data['peak']
        data['drawdown_duration'] = self._calculate_drawdown_duration(data['drawdown'])
        
        # Risk regime indicators
        data['high_risk_regime'] = (
            (data['realized_vol_20d'] > data['realized_vol_20d'].rolling(window=60).quantile(0.8)) |
            (abs(data['drawdown']) > 0.10)
        ).astype(int)
        
        # Volatility breakout indicators
        data['vol_breakout'] = (data['realized_vol_20d'] > data['realized_vol_20d'].rolling(window=60).quantile(0.95)).astype(int)
        data['iv_spike'] = (data['implied_volatility'] > data['implied_volatility'].rolling(window=20).quantile(0.9)).astype(int)
        
        # Market stress indicators
        data['stress_indicator'] = (
            data['vol_breakout'] + 
            data['iv_spike'] + 
            (abs(data['drawdown']) > 0.05).astype(int)
        ) / 3
        
        # Safe trading conditions
        data['low_vol_regime'] = (data['realized_vol_20d'] < data['realized_vol_20d'].rolling(window=60).quantile(0.3)).astype(int)
        data['stable_trend'] = (abs(data['ma_spread_5_20']) < 0.02).astype(int)
        data['safe_conditions'] = (data['low_vol_regime'] & data['stable_trend']).astype(int)
        
        return data
    
    def _add_market_structure_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add market microstructure and regime features."""
        # Price action patterns
        data['higher_highs'] = (data['high'] > data['high'].shift(1)).rolling(window=5).sum()
        data['lower_lows'] = (data['low'] < data['low'].shift(1)).rolling(window=5).sum()
        data['trend_strength'] = data['higher_highs'] - data['lower_lows']
        
        # Gap analysis
        data['gap_up'] = ((data['open'] - data['close'].shift(1)) / data['close'].shift(1) > 0.01).astype(int)
        data['gap_down'] = ((data['close'].shift(1) - data['open']) / data['close'].shift(1) > 0.01).astype(int)
        
        # Intraday range features
        data['true_range'] = np.maximum(
            data['high'] - data['low'],
            np.maximum(
                abs(data['high'] - data['close'].shift(1)),
                abs(data['low'] - data['close'].shift(1))
            )
        )
        data['atr_14'] = data['true_range'].rolling(window=14).mean()
        data['range_position'] = (data['close'] - data['low']) / (data['high'] - data['low'])
        
        # Volume profile indicators
        data['volume_price_trend'] = ((data['close'] - data['close'].shift(1)) * data['volume']).rolling(window=10).sum()
        data['price_volume_correlation'] = data['close'].pct_change().rolling(window=20).corr(data['volume'].pct_change())
        
        return data
    
    def _add_regime_detection_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add features for market regime detection."""
        # Volatility regimes
        vol_20d = data['realized_vol_20d'].rolling(window=60)
        data['vol_regime_low'] = (data['realized_vol_20d'] < vol_20d.quantile(0.33)).astype(int)
        data['vol_regime_med'] = ((data['realized_vol_20d'] >= vol_20d.quantile(0.33)) & 
                                 (data['realized_vol_20d'] <= vol_20d.quantile(0.67))).astype(int)
        data['vol_regime_high'] = (data['realized_vol_20d'] > vol_20d.quantile(0.67)).astype(int)
        
        # Trend regimes
        momentum_20d = data['momentum_20d'].rolling(window=60)
        data['trend_regime_bear'] = (data['momentum_20d'] < momentum_20d.quantile(0.33)).astype(int)
        data['trend_regime_neutral'] = ((data['momentum_20d'] >= momentum_20d.quantile(0.33)) & 
                                       (data['momentum_20d'] <= momentum_20d.quantile(0.67))).astype(int)
        data['trend_regime_bull'] = (data['momentum_20d'] > momentum_20d.quantile(0.67)).astype(int)
        
        # Combined regime indicator
        data['market_regime'] = (
            data['vol_regime_high'] * 4 +  # High vol regime
            data['trend_regime_bear'] * 2 + # Bear trend
            data['trend_regime_bull'] * 1    # Bull trend
        )
        
        return data
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_bollinger_bands(self, prices: pd.Series, window: int = 20, std_dev: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands."""
        middle = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        return upper, lower, middle
    
    def _calculate_volatility_regime(self, vol_series: pd.Series, window: int = 60) -> pd.Series:
        """Calculate volatility regime (0=low, 1=medium, 2=high)."""
        rolling_quantiles = vol_series.rolling(window=window)
        low_threshold = rolling_quantiles.quantile(0.33)
        high_threshold = rolling_quantiles.quantile(0.67)
        
        regime = pd.Series(1, index=vol_series.index)  # Default to medium
        regime[vol_series < low_threshold] = 0  # Low volatility
        regime[vol_series > high_threshold] = 2  # High volatility
        
        return regime
    
    def _calculate_zscore(self, series: pd.Series, window: int = 20) -> pd.Series:
        """Calculate rolling z-score."""
        rolling_mean = series.rolling(window=window).mean()
        rolling_std = series.rolling(window=window).std()
        return (series - rolling_mean) / rolling_std
    
    def _calculate_garch_proxy(self, returns: pd.Series) -> pd.Series:
        """Calculate simple GARCH-like volatility proxy."""
        # Simple GARCH(1,1) proxy using exponential weighting
        alpha = 0.1
        beta = 0.85
        
        volatility = pd.Series(index=returns.index, dtype=float)
        volatility.iloc[0] = returns.std()
        
        for i in range(1, len(returns)):
            if not pd.isna(returns.iloc[i-1]):
                volatility.iloc[i] = np.sqrt(
                    alpha * returns.iloc[i-1]**2 + 
                    beta * volatility.iloc[i-1]**2
                )
            else:
                volatility.iloc[i] = volatility.iloc[i-1]
        
        return volatility * np.sqrt(252)  # Annualize
    
    def _calculate_drawdown_duration(self, drawdown: pd.Series) -> pd.Series:
        """Calculate duration of current drawdown."""
        duration = pd.Series(0, index=drawdown.index)
        current_duration = 0
        
        for i, dd in enumerate(drawdown):
            if dd < 0:
                current_duration += 1
            else:
                current_duration = 0
            duration.iloc[i] = current_duration
        
        return duration

# Backward compatibility function
def add_multi_timeframe_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Backward compatibility wrapper for the enhanced feature engineering.
    Now includes advanced features designed to reduce drawdown.
    """
    engineer = AdvancedFeatureEngineer()
    return engineer.engineer_features(data)
