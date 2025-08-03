"""
Enhanced Main Trading Strategy Runner
More aggressive signal generation while maintaining risk control
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import logging
import mlflow
import mlflow.xgboost
from datetime import datetime
from src.quant_lab.fixed_spy_data_loader import FixedSPYDataLoader
from src.quant_lab.labeling import AdaptiveTripleBarrierLabeler
from src.quant_lab.engine import EnhancedBacktestEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def enhanced_feature_engineering(data: pd.DataFrame) -> pd.DataFrame:
    """Enhanced feature engineering leveraging real Greeks data."""
    featured_data = data.copy()
    
    # Price-based features
    featured_data['returns'] = featured_data['close'].pct_change()
    featured_data['log_returns'] = np.log(featured_data['close'] / featured_data['close'].shift(1))
    
    # Multi-timeframe moving averages
    for window in [5, 10, 20]:
        featured_data[f'mavg_{window}d'] = featured_data['close'].rolling(window=window).mean()
        featured_data[f'close_vs_mavg_{window}d'] = featured_data['close'] / featured_data[f'mavg_{window}d'] - 1
    
    # Volatility features
    featured_data['realized_vol_5d'] = featured_data['returns'].rolling(window=5).std() * np.sqrt(252)
    featured_data['realized_vol_20d'] = featured_data['returns'].rolling(window=20).std() * np.sqrt(252)
    
    # Volume features
    featured_data['volume_ma'] = featured_data['volume'].rolling(window=20).mean()
    featured_data['volume_ratio'] = featured_data['volume'] / (featured_data['volume_ma'] + 1e-10)
    
    # Enhanced Greeks features (using real data!)
    if 'delta' in featured_data.columns:
        featured_data['delta_change'] = featured_data['delta'].diff()
        featured_data['delta_ma'] = featured_data['delta'].rolling(window=5).mean()
        featured_data['delta_momentum'] = featured_data['delta'].pct_change(3)
        featured_data['delta_volatility'] = featured_data['delta'].rolling(window=10).std()
    
    if 'gamma' in featured_data.columns:
        featured_data['gamma_change'] = featured_data['gamma'].diff()
        featured_data['gamma_ma'] = featured_data['gamma'].rolling(window=5).mean()
        # Gamma exposure proxy
        featured_data['gamma_exposure'] = featured_data['gamma'] * featured_data['close'] * featured_data['close']
        featured_data['gamma_spike'] = (featured_data['gamma'] > featured_data['gamma'].rolling(window=20).quantile(0.75)).astype(int)  # More sensitive
    
    if 'vega' in featured_data.columns:
        featured_data['vega_change'] = featured_data['vega'].diff()
        featured_data['vega_ma'] = featured_data['vega'].rolling(window=5).mean()
        # Volatility risk exposure
        featured_data['vega_exposure'] = featured_data['vega'] * featured_data['realized_vol_20d']
    
    if 'theta' in featured_data.columns:
        featured_data['theta_change'] = featured_data['theta'].diff()
        featured_data['theta_ma'] = featured_data['theta'].rolling(window=5).mean()
        # Time decay pressure
        featured_data['theta_pressure'] = abs(featured_data['theta']) / (featured_data['close'] + 1e-10)
    
    # Greeks interactions (unique to real options data!)
    if 'delta' in featured_data.columns and 'gamma' in featured_data.columns:
        featured_data['delta_gamma_product'] = featured_data['delta'] * featured_data['gamma']
        featured_data['gamma_delta_ratio'] = featured_data['gamma'] / (abs(featured_data['delta']) + 1e-10)
    
    if 'vega' in featured_data.columns and 'theta' in featured_data.columns:
        featured_data['vega_theta_ratio'] = featured_data['vega'] / (abs(featured_data['theta']) + 1e-10)
        featured_data['risk_adjusted_theta'] = featured_data['theta'] / (featured_data['vega'] + 1e-10)
    
    # Options market structure features
    if 'put_call_ratio' in featured_data.columns:
        featured_data['pcr_ma'] = featured_data['put_call_ratio'].rolling(window=10).mean()
        featured_data['pcr_divergence'] = featured_data['put_call_ratio'] - featured_data['pcr_ma']
        featured_data['pcr_spike'] = (featured_data['put_call_ratio'] > featured_data['put_call_ratio'].rolling(window=20).quantile(0.75)).astype(int)  # More sensitive
    
    if 'gex_estimate' in featured_data.columns:
        featured_data['gex_change'] = featured_data['gex_estimate'].diff()
        featured_data['gex_ma'] = featured_data['gex_estimate'].rolling(window=5).mean()
        featured_data['gex_normalized'] = featured_data['gex_estimate'] / (featured_data['close'] * featured_data['close'] + 1e-10)
    
    # Risk management features
    featured_data['peak'] = featured_data['close'].expanding().max()
    featured_data['drawdown'] = (featured_data['close'] - featured_data['peak']) / featured_data['peak']
    
    # Momentum features
    for period in [3, 5, 10]:
        featured_data[f'momentum_{period}d'] = featured_data['close'].pct_change(period)
    
    # Market regime features
    featured_data['vol_regime'] = (featured_data['realized_vol_20d'] > featured_data['realized_vol_20d'].rolling(window=60).quantile(0.65)).astype(int)  # More sensitive
    
    # Options regime features (unique!)
    if 'data_quality_score' in featured_data.columns:
        featured_data['high_quality_greeks'] = (featured_data['data_quality_score'] >= 2).astype(int)  # Lower threshold
    
    return featured_data

def train_and_predict_enhanced(train_data: pd.DataFrame, test_data: pd.DataFrame, 
                              triple_barrier_params: dict) -> pd.Series:
    """Train model with enhanced Greeks features - more aggressive signal generation."""
    
    # Core features optimized for real Greeks
    core_features = [
        # Price & momentum
        'close_vs_mavg_5d', 'close_vs_mavg_20d', 'momentum_10d', 'realized_vol_20d',
        'volume_ratio', 'drawdown',
        
        # Real Greeks features  
        'delta', 'gamma', 'vega', 'theta',
        'delta_change', 'gamma_change', 'vega_change', 'theta_change',
        'delta_momentum', 'gamma_exposure', 'theta_pressure',
        
        # Greeks interactions
        'delta_gamma_product', 'vega_theta_ratio', 'gamma_delta_ratio',
        
        # Options market structure
        'put_call_ratio', 'pcr_divergence', 'gex_estimate', 'gex_normalized',
        
        # Regime features
        'vol_regime', 'high_quality_greeks', 'pcr_spike', 'gamma_spike'
    ]
    
    # Filter to available features
    available_features = [f for f in core_features if f in train_data.columns]
    logger.info(f"Using {len(available_features)} features including real Greeks")
    
    if len(available_features) < 8:
        logger.warning("Insufficient features for training")
        return pd.Series(0, index=test_data.index)
    
    # Generate adaptive labels
    labeler = AdaptiveTripleBarrierLabeler(
        base_profit_target=triple_barrier_params['profit_target'],
        base_stop_loss=triple_barrier_params['stop_loss'],
        base_time_horizon=triple_barrier_params['time_horizon'],
        volatility_adjustment=True,
        regime_adjustment=True
    )
    
    labels = labeler.get_adaptive_labels(
        prices=train_data['close'],
        market_data=train_data
    )
    
    # Prepare training data
    X_train = train_data.loc[labels.index, available_features].fillna(method='ffill').fillna(0)
    y_train = labels.loc[X_train.index]
    
    # Remove rows with insufficient data
    mask = ~X_train.isnull().all(axis=1)
    X_train = X_train[mask]
    y_train = y_train[mask]
    
    if len(X_train) < 10 or y_train.nunique() <= 1:
        logger.warning("Insufficient training data")
        return pd.Series(0, index=test_data.index)
    
    # Enhanced XGBoost model for Greeks data
    model = xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=3,
        eval_metric='mlogloss',
        n_estimators=150,  # Slightly reduced for faster training
        max_depth=5,       # Reduced to prevent overfitting
        learning_rate=0.12, # Slightly higher learning rate
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.02,    # Reduced regularization for more signals
        reg_lambda=0.02,
        random_state=42
    )
    
    # Map labels for XGBoost
    y_train_mapped = y_train.map({-1: 0, 0: 1, 1: 2})
    model.fit(X_train, y_train_mapped)
    
    # Generate predictions
    X_test = test_data[available_features].fillna(method='ffill').fillna(0)
    if X_test.empty:
        return pd.Series(0, index=test_data.index)
    
    probabilities = model.predict_proba(X_test)
    signals = pd.Series(0, index=X_test.index)
    
    if len(probabilities) > 0:
        # Use win probability with Greeks-adjusted threshold
        win_probs = probabilities[:, 2] if probabilities.shape[1] > 2 else probabilities[:, 1]
        
        # More aggressive threshold for increased signal generation
        base_threshold = 0.60  # Reduced from 0.70
        
        # Adjust based on Greeks data quality
        if 'high_quality_greeks' in test_data.columns:
            quality_bonus = test_data['high_quality_greeks'].mean() * 0.08  # Increased bonus
            threshold = np.percentile(win_probs, (base_threshold - quality_bonus) * 100)
        else:
            threshold = np.percentile(win_probs, base_threshold * 100)
        
        signals[win_probs >= threshold] = 1
        
        # More generous fallback for signal generation
        if signals.sum() == 0 and len(win_probs) > 0:
            threshold = np.percentile(win_probs, 55)  # More generous
            signals[win_probs >= threshold] = 1
    
    # Enhanced but less restrictive risk filters
    signals = apply_balanced_risk_filters(signals, test_data)
    
    return signals.reindex(test_data.index, fill_value=0)

def apply_balanced_risk_filters(signals: pd.Series, test_data: pd.DataFrame) -> pd.Series:
    """Apply balanced risk filters - less restrictive for more signals."""
    filtered_signals = signals.copy()
    
    # Don't trade during extreme volatility (more lenient)
    if 'realized_vol_20d' in test_data.columns:
        high_vol = test_data['realized_vol_20d'] > test_data['realized_vol_20d'].quantile(0.95)  # Only extreme cases
        filtered_signals[high_vol] = 0
    
    # Don't trade during severe drawdowns (more lenient)
    if 'drawdown' in test_data.columns:
        extreme_dd = test_data['drawdown'] < -0.15  # More lenient threshold
        filtered_signals[extreme_dd] = 0
    
    # Don't trade when Greeks are very unstable (more lenient)
    if 'gamma_change' in test_data.columns:
        gamma_instability = test_data['gamma_change'].abs() > test_data['gamma_change'].abs().quantile(0.90)  # More lenient
        filtered_signals[gamma_instability] = 0
    
    # Allow some lower quality Greeks (more lenient)
    # Removed the high_quality_greeks filter to allow more signals
    
    # Don't trade during extreme Gamma exposure (more lenient)
    if 'gamma_exposure' in test_data.columns:
        extreme_gex = test_data['gamma_exposure'].abs() > test_data['gamma_exposure'].abs().quantile(0.98)  # More lenient
        filtered_signals[extreme_gex] = 0
    
    return filtered_signals

def run_enhanced_strategy():
    """Run the enhanced trading strategy with more signal generation."""
    
    print("=" * 80)
    print("ENHANCED SPY OPTIONS TRADING STRATEGY - MORE SIGNALS")
    print("=" * 80)
    
    # Initialize MLflow
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("SPY Options Strategy - Enhanced")
    
    # Load real SPY data with Greeks
    loader = FixedSPYDataLoader()
    
    # Use full available date range
    spy_data = loader.get_spy_trading_data_enhanced("2023-08-01", "2023-12-29")
    
    if spy_data.empty:
        print("ERROR: No SPY data loaded")
        return
    
    print(f"Loaded {len(spy_data)} days of real SPY options data")
    print(f"Greeks coverage: {spy_data['delta'].notna().sum()}/{len(spy_data)} days ({spy_data['delta'].notna().sum()/len(spy_data)*100:.1f}%)")
    
    # Enhanced feature engineering
    featured_data = enhanced_feature_engineering(spy_data)
    featured_data = featured_data.dropna(thresh=len(featured_data.columns) * 0.6)  # More lenient
    
    print(f"After feature engineering: {len(featured_data)} days, {len(featured_data.columns)} features")
    
    # Optimized parameters - slightly more aggressive
    ENHANCED_PARAMS = {
        'profit_target': 0.04,  # 4% (slightly lower for more signals)
        'stop_loss': 0.02,      # 2% (keep tight)
        'time_horizon': 5,      # 5 days
        'volatility_threshold': 0.05
    }
    
    print(f"\nUsing enhanced parameters for more signals:")
    print(f"  Profit Target: {ENHANCED_PARAMS['profit_target']:.1%}")
    print(f"  Stop Loss: {ENHANCED_PARAMS['stop_loss']:.1%}")
    print(f"  Time Horizon: {ENHANCED_PARAMS['time_horizon']} days")
    
    # Split data for training and testing
    split_point = int(len(featured_data) * 0.7)  # 70% train, 30% test
    train_data = featured_data.iloc[:split_point]
    test_data = featured_data.iloc[split_point:]
    
    print(f"\nData split:")
    print(f"  Training: {len(train_data)} days ({train_data.index.min().date()} to {train_data.index.max().date()})")
    print(f"  Testing: {len(test_data)} days ({test_data.index.min().date()} to {test_data.index.max().date()})")
    
    # Start MLflow run
    with mlflow.start_run(run_name=f"Enhanced Strategy - {datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        
        # Log parameters
        mlflow.log_params(ENHANCED_PARAMS)
        mlflow.log_param("strategy_type", "enhanced_aggressive")
        mlflow.log_param("data_source", "real_spy_greeks")
        mlflow.log_param("train_days", len(train_data))
        mlflow.log_param("test_days", len(test_data))
        mlflow.log_param("greeks_coverage_pct", spy_data['delta'].notna().sum()/len(spy_data)*100)
        mlflow.log_param("total_features", len(featured_data.columns))
        mlflow.log_param("signal_threshold", "60% (vs 70% conservative)")
        mlflow.log_param("risk_filters", "balanced_less_restrictive")
        
        # Train and predict
        print(f"\nTraining enhanced model...")
        signals = train_and_predict_enhanced(train_data, test_data, ENHANCED_PARAMS)
        
        print(f"Generated {signals.sum()} signals out of {len(signals)} test days ({signals.sum()/len(signals)*100:.1f}%)")
        
        # Log signal metrics
        mlflow.log_metric("total_signals", int(signals.sum()))
        mlflow.log_metric("signal_frequency_pct", signals.sum()/len(signals)*100)
        
        if signals.sum() == 0:
            print("WARNING: No signals generated - try even more aggressive parameters")
            mlflow.log_metric("sharpe_ratio", 0.0)
            mlflow.log_metric("total_return", 0.0)
            mlflow.log_metric("max_drawdown", 0.0)
            mlflow.log_metric("win_rate", 0.0)
            return
        
        # Run backtest
        print(f"\nRunning backtest...")
        backtest_engine = EnhancedBacktestEngine(
            initial_capital=100000,
            transaction_cost_bps=5.0,
            risk_free_rate=0.02
        )
        
        results = backtest_engine.run(
            price_data=test_data['close'],
            signals=signals,
            market_data=test_data
        )
        
        # Log performance metrics to MLflow
        mlflow.log_metric("sharpe_ratio", results['sharpe_ratio'])
        mlflow.log_metric("total_return", results['total_return'])
        mlflow.log_metric("annualized_return", results['annualized_return'])
        mlflow.log_metric("max_drawdown", results['max_drawdown'])
        mlflow.log_metric("win_rate", results['win_rate'])
        mlflow.log_metric("calmar_ratio", results.get('calmar_ratio', 0))
        mlflow.log_metric("sortino_ratio", results.get('sortino_ratio', 0))
        
        # Log additional metrics
        signal_dates = signals[signals == 1].index
        mlflow.log_metric("days_between_signals", len(test_data) / max(signals.sum(), 1))
        mlflow.log_metric("test_period_days", len(test_data))
        
        # Create and log enhanced signal summary with more Greeks data
        signal_summary = []
        for date in signal_dates[:15]:  # First 15 signals for enhanced
            price = test_data.loc[date, 'close']
            delta = test_data.loc[date, 'delta'] if 'delta' in test_data.columns else None
            gamma = test_data.loc[date, 'gamma'] if 'gamma' in test_data.columns else None
            pcr = test_data.loc[date, 'put_call_ratio'] if 'put_call_ratio' in test_data.columns else None
            gex = test_data.loc[date, 'gex_estimate'] if 'gex_estimate' in test_data.columns else None
            signal_summary.append({
                'date': date.strftime('%Y-%m-%d'),
                'price': float(price),
                'delta': float(delta) if delta is not None else None,
                'gamma': float(gamma) if gamma is not None else None,
                'put_call_ratio': float(pcr) if pcr is not None else None,
                'gex_estimate': float(gex) if gex is not None else None
            })
        
        # Save enhanced signal summary as artifact
        import json
        with open('enhanced_signal_summary.json', 'w') as f:
            json.dump(signal_summary, f, indent=2)
        mlflow.log_artifact('enhanced_signal_summary.json')
        
        # Compare with conservative baseline (if available)
        conservative_signals = 8  # Previous conservative result
        mlflow.log_metric("signal_increase_vs_conservative", signals.sum() - conservative_signals)
        mlflow.log_metric("signal_increase_pct", ((signals.sum() / conservative_signals) - 1) * 100 if conservative_signals > 0 else 0)
        
        # Log strategy notes
        mlflow.set_tag("strategy_notes", "Enhanced aggressive approach - more signals while maintaining risk control")
        mlflow.set_tag("data_quality", f"{spy_data['delta'].notna().sum()}/{len(spy_data)} days with Greeks")
        mlflow.set_tag("enhancement_features", "Lower thresholds, balanced filters, 4% profit target")
        mlflow.set_tag("success_criteria", f"Sharpe > 0.3: {'✓' if results['sharpe_ratio'] > 0.3 else '✗'}, DD < 15%: {'✓' if results['max_drawdown'] < 0.15 else '✗'}, Signals > 8: {'✓' if signals.sum() > 8 else '✗'}")
    
    # Display results
    print(f"\n" + "=" * 60)
    print("ENHANCED STRATEGY PERFORMANCE RESULTS")
    print("=" * 60)
    print(f"PERFORMANCE METRICS:")
    print(f"  Sharpe Ratio:        {results['sharpe_ratio']:.3f}")
    print(f"  Total Return:        {results['total_return']:.2%}")
    print(f"  Annualized Return:   {results['annualized_return']:.2%}")
    print(f"  Maximum Drawdown:    {results['max_drawdown']:.2%}")
    print(f"  Win Rate:            {results['win_rate']:.1%}")
    print(f"  Total Signals:       {int(signals.sum())}")
    
    # Show signal dates with Greeks
    signal_dates = signals[signals == 1].index
    print(f"\nSIGNAL DATES WITH GREEKS:")
    for i, date in enumerate(signal_dates[:15]):  # Show first 15
        price = test_data.loc[date, 'close']
        delta = test_data.loc[date, 'delta'] if 'delta' in test_data.columns else 'N/A'
        gamma = test_data.loc[date, 'gamma'] if 'gamma' in test_data.columns else 'N/A'
        pcr = test_data.loc[date, 'put_call_ratio'] if 'put_call_ratio' in test_data.columns else 'N/A'
        print(f"  {date.date()}: ${price:.2f} (Δ:{delta:.4f}, Γ:{gamma:.4f}, PCR:{pcr:.2f})" if delta != 'N/A' else f"  {date.date()}: ${price:.2f}")
    
    if len(signal_dates) > 15:
        print(f"  ... and {len(signal_dates) - 15} more signals")
    
    print(f"\n" + "=" * 60)
    print("STRATEGY COMPARISON:")
    print("Enhanced vs Conservative approach:")
    print(f"• Signal Generation: {signals.sum()} vs 8 (previous)")
    print(f"• Signal Frequency: {signals.sum()/len(signals)*100:.1f}% vs {8/len(signals)*100:.1f}%")
    print(f"• Expected trade-off: More signals but potentially lower Sharpe ratio")
    
    print("=" * 80)

if __name__ == "__main__":
    run_enhanced_strategy()