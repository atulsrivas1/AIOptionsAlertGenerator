"""
Fixed SPY Data Loader - Properly handles your options data with Greeks
Optimized for your specific data structure and quality patterns
"""

import pandas as pd
import numpy as np
import json
import os
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple, Union
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class FixedSPYDataLoader:
    """
    Fixed SPY data loader that properly extracts Greeks from your data.
    
    Key improvements:
    - Handles your specific JSON structure
    - Prioritizes contracts with Greeks data
    - Better ATM contract selection
    - Improved data quality handling
    """
    
    def __init__(self, data_dir: str = "data_cache"):
        """Initialize the fixed SPY data loader."""
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise ValueError(f"Data directory not found: {data_dir}")
        
        self.price_data = {}
        self.options_data = {}
        
        logger.info(f"FixedSPYDataLoader initialized with data_dir: {self.data_dir}")
        self._validate_data_files()
    
    def _validate_data_files(self):
        """Validate that required data files exist."""
        # Check price data files
        price_files = [
            "SPY_day_2022-01-01_2023-12-31.json",
            "SPY_hour_2022-01-01_2023-12-31.json", 
            "SPY_minute_2022-01-01_2023-12-31.json"
        ]
        
        for file in price_files:
            if (self.data_dir / file).exists():
                logger.info(f"Found price file: {file}")
        
        # Check options data files  
        options_files = list(self.data_dir.glob("SPY_202*.json"))
        options_files = [f for f in options_files if f.name not in price_files]
        logger.info(f"Found {len(options_files)} daily options chain files")
        
        # Check data quality across time
        self._analyze_data_quality(options_files[:10])  # Sample first 10 files
    
    def _analyze_data_quality(self, sample_files: List[Path]):
        """Analyze data quality across time periods."""
        logger.info("Analyzing options data quality...")
        
        quality_stats = []
        for file_path in sample_files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                if data:
                    total_contracts = len(data)
                    contracts_with_delta = sum(1 for c in data if c.get('delta') is not None)
                    contracts_with_iv = sum(1 for c in data if c.get('implied_volatility') is not None)
                    
                    quality_stats.append({
                        'date': file_path.stem.replace('SPY_', ''),
                        'total': total_contracts,
                        'with_delta': contracts_with_delta,
                        'with_iv': contracts_with_iv,
                        'delta_pct': contracts_with_delta / total_contracts * 100
                    })
                    
            except Exception as e:
                logger.warning(f"Error analyzing {file_path}: {e}")
        
        if quality_stats:
            avg_delta_pct = np.mean([s['delta_pct'] for s in quality_stats])
            logger.info(f"Average Delta availability: {avg_delta_pct:.1f}%")
            
            # Find best quality period
            best_period = max(quality_stats, key=lambda x: x['delta_pct'])
            logger.info(f"Best data quality: {best_period['date']} ({best_period['delta_pct']:.1f}% Greeks)")
    
    def load_price_data(self, timeframe: str = "day") -> pd.DataFrame:
        """Load SPY price data for specified timeframe."""
        if timeframe in self.price_data:
            return self.price_data[timeframe]
        
        filename = f"SPY_{timeframe}_2022-01-01_2023-12-31.json"
        filepath = self.data_dir / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"Price data file not found: {filename}")
        
        logger.info(f"Loading {timeframe} price data...")
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        df.sort_index(inplace=True)
        
        # Ensure proper data types
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        self.price_data[timeframe] = df
        return df
    
    def load_options_chain(self, date: str) -> pd.DataFrame:
        """Load options chain data for a specific date with improved Greeks extraction."""
        date_key = date.replace('-', '_')
        
        if date_key in self.options_data:
            return self.options_data[date_key]
        
        filename = f"SPY_{date}.json"
        filepath = self.data_dir / filename
        
        if not filepath.exists():
            logger.debug(f"Options data not found for {date}")
            return pd.DataFrame()
        
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            if not data:
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(data)
            
            # Enhanced data processing
            df = self._process_options_data_enhanced(df, date)
            
            # Cache the data
            self.options_data[date_key] = df
            return df
            
        except Exception as e:
            logger.error(f"Error loading options data for {date}: {e}")
            return pd.DataFrame()
    
    def _process_options_data_enhanced(self, df: pd.DataFrame, date: str) -> pd.DataFrame:
        """Enhanced processing of options chain data."""
        if df.empty:
            return df
        
        # Add date column
        df['date'] = pd.to_datetime(date)
        
        # Ensure proper data types
        numeric_cols = ['strike_price', 'open_interest', 'delta', 'gamma', 'vega', 'theta']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Parse expiration date
        if 'expiration_date' in df.columns:
            df['expiration_date'] = pd.to_datetime(df['expiration_date'])
            df['days_to_expiry'] = (df['expiration_date'] - df['date']).dt.days
        
        # Remove rows with missing critical data
        df = df.dropna(subset=['strike_price', 'contract_type'])
        
        # Calculate moneyness (we'll estimate spot price)
        if len(df) > 0:
            # Estimate spot price from ATM options
            calls = df[df['contract_type'] == 'call']
            if not calls.empty:
                # Find call with delta closest to 0.5 (ATM)
                calls_with_delta = calls[calls['delta'].notna()]
                if not calls_with_delta.empty:
                    atm_call = calls_with_delta.loc[(calls_with_delta['delta'] - 0.5).abs().idxmin()]
                    estimated_spot = atm_call['strike_price']
                    df['moneyness'] = df['strike_price'] / estimated_spot
                    df['estimated_spot'] = estimated_spot
        
        # Add data quality indicators
        df['has_greeks'] = df[['delta', 'gamma', 'vega', 'theta']].notna().all(axis=1)
        df['greeks_count'] = df[['delta', 'gamma', 'vega', 'theta']].notna().sum(axis=1)
        
        # Calculate additional derived metrics for contracts with Greeks
        mask = df['delta'].notna() & df['gamma'].notna()
        if mask.any():
            df.loc[mask, 'delta_gamma_ratio'] = df.loc[mask, 'delta'] / (df.loc[mask, 'gamma'].abs() + 1e-10)
        
        mask = df['vega'].notna() & df['theta'].notna()
        if mask.any():
            df.loc[mask, 'vega_theta_ratio'] = df.loc[mask, 'vega'] / (df.loc[mask, 'theta'].abs() + 1e-10)
        
        return df
    
    def get_spy_trading_data_enhanced(self, start_date: str, end_date: str, 
                                    prefer_greeks_quality: bool = True) -> pd.DataFrame:
        """
        Get enhanced SPY trading data with better Greeks integration.
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)  
            prefer_greeks_quality: If True, prioritize dates with good Greeks data
        """
        logger.info(f"Building enhanced SPY trading data from {start_date} to {end_date}")
        
        # Load daily price data as base
        price_df = self.load_price_data('day')
        # Convert date strings to datetime for proper comparison
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date) + pd.Timedelta(days=1)  # Include end date
        price_df = price_df[(price_df.index >= start_dt) & (price_df.index < end_dt)].copy()
        
        if price_df.empty:
            logger.warning("No price data found for date range")
            return pd.DataFrame()
        
        # Build enhanced options features
        options_features = []
        
        for idx, price_date in enumerate(price_df.index):
            date_str = price_date.strftime('%Y-%m-%d')
            options_df = self.load_options_chain(date_str)
            
            if options_df.empty:
                # Fill with NaN if no options data
                options_features.append(self._create_empty_options_row(date_str, price_date))
                continue
            
            # Get the best available option data for this date
            spot_price = price_df.iloc[idx]['close']
            
            # Enhanced option selection with quality prioritization
            target_option, market_metrics = self._select_best_option_enhanced(
                options_df, spot_price, prefer_greeks_quality
            )
            
            if target_option is not None:
                # Remove the date key from target_option to avoid overwriting our price_date
                target_option_clean = {k: v for k, v in target_option.items() if k != 'date'}
                feature_dict = {
                    'date': price_date,  # Use the actual price_date timestamp
                    **target_option_clean,
                    **market_metrics
                }
                options_features.append(feature_dict)
            else:
                options_features.append(self._create_empty_options_row(date_str, price_date))
        
        # Convert to DataFrame
        options_df = pd.DataFrame(options_features)
        options_df.set_index('date', inplace=True)
        
        # Merge with price data
        combined_df = price_df.join(options_df, how='left')
        
        # Add derived features
        combined_df = self._add_enhanced_derived_features(combined_df)
        
        logger.info(f"Enhanced trading data built: {len(combined_df)} days with {len(combined_df.columns)} features")
        
        # Log data quality
        if 'delta' in combined_df.columns:
            delta_coverage = combined_df['delta'].notna().sum()
            logger.info(f"Greeks coverage: {delta_coverage}/{len(combined_df)} days ({delta_coverage/len(combined_df)*100:.1f}%)")
        
        return combined_df
    
    def _create_empty_options_row(self, date_str: str, price_date: pd.Timestamp = None) -> Dict:
        """Create empty options row for dates with no data."""
        return {
            'date': price_date if price_date is not None else pd.to_datetime(date_str),
            'delta': np.nan,
            'gamma': np.nan,
            'vega': np.nan,
            'theta': np.nan,
            'open_interest': np.nan,
            'strike_price': np.nan,
            'days_to_expiry': np.nan,
            'put_call_ratio': np.nan,
            'total_oi': np.nan,
            'gex_estimate': np.nan,
            'data_quality_score': 0
        }
    
    def _select_best_option_enhanced(self, options_df: pd.DataFrame, spot_price: float,
                                   prefer_greeks_quality: bool) -> Tuple[Optional[Dict], Dict]:
        """Enhanced option selection prioritizing data quality."""
        if options_df.empty:
            return None, {}
        
        # Filter to calls for primary tracking
        calls_df = options_df[options_df['contract_type'] == 'call'].copy()
        
        if calls_df.empty:
            return None, {}
        
        # If preferring Greeks quality, filter to contracts with good Greeks
        if prefer_greeks_quality:
            good_greeks = calls_df[calls_df['has_greeks'] == True]
            if not good_greeks.empty:
                calls_df = good_greeks
        
        # Filter by expiry (prefer 1-6 weeks out)
        if 'days_to_expiry' in calls_df.columns:
            calls_df = calls_df[
                (calls_df['days_to_expiry'] >= 7) & 
                (calls_df['days_to_expiry'] <= 45)
            ]
        
        if calls_df.empty:
            # Fallback to any available calls
            calls_df = options_df[options_df['contract_type'] == 'call'].copy()
        
        # Find ATM option
        calls_df['strike_distance'] = abs(calls_df['strike_price'] - spot_price)
        
        # If we have deltas, use delta to find ATM (delta closest to 0.5)
        calls_with_delta = calls_df[calls_df['delta'].notna()]
        if not calls_with_delta.empty:
            calls_with_delta['delta_distance'] = abs(calls_with_delta['delta'] - 0.5)
            best_option = calls_with_delta.loc[calls_with_delta['delta_distance'].idxmin()]
        else:
            # Fallback to strike distance
            best_option = calls_df.loc[calls_df['strike_distance'].idxmin()]
        
        # Calculate market metrics
        market_metrics = self._calculate_enhanced_market_metrics(options_df, spot_price)
        
        # Add data quality score
        target_dict = best_option.to_dict()
        target_dict['data_quality_score'] = best_option.get('greeks_count', 0)
        
        return target_dict, market_metrics
    
    def _calculate_enhanced_market_metrics(self, options_df: pd.DataFrame, spot_price: float) -> Dict:
        """Calculate enhanced market metrics."""
        if options_df.empty:
            return {
                'put_call_ratio': np.nan, 
                'total_oi': np.nan, 
                'gex_estimate': np.nan,
                'avg_iv': np.nan,
                'skew_proxy': np.nan
            }
        
        # Basic metrics
        calls = options_df[options_df['contract_type'] == 'call']
        puts = options_df[options_df['contract_type'] == 'put']
        
        call_oi = calls['open_interest'].sum() if not calls.empty else 0
        put_oi = puts['open_interest'].sum() if not puts.empty else 0
        put_call_ratio = put_oi / (call_oi + 1e-10)
        
        total_oi = options_df['open_interest'].sum()
        
        # GEX approximation (only for contracts with gamma)
        gex_contracts = options_df[options_df['gamma'].notna()]
        if not gex_contracts.empty:
            gex_contributions = (
                gex_contracts['gamma'] * 
                gex_contracts['open_interest'] * 
                spot_price * spot_price * 0.01  # Scaling factor
            )
            total_gex = gex_contributions.sum()
        else:
            total_gex = np.nan
        
        # IV metrics (skip if no IV data)
        avg_iv = np.nan
        skew_proxy = np.nan
        
        if 'implied_volatility' in options_df.columns:
            iv_data = options_df['implied_volatility'].dropna()
            avg_iv = iv_data.mean() if not iv_data.empty else np.nan
            
            # Simple skew proxy (difference between OTM put and call IV)
            if not iv_data.empty:
                otm_puts = puts[(puts['strike_price'] < spot_price * 0.95) & puts['implied_volatility'].notna()]
                otm_calls = calls[(calls['strike_price'] > spot_price * 1.05) & calls['implied_volatility'].notna()]
                
                if not otm_puts.empty and not otm_calls.empty:
                    put_iv = otm_puts['implied_volatility'].mean()
                    call_iv = otm_calls['implied_volatility'].mean()
                    skew_proxy = put_iv - call_iv
        
        return {
            'put_call_ratio': put_call_ratio,
            'total_oi': total_oi,
            'gex_estimate': total_gex,
            'avg_iv': avg_iv,
            'skew_proxy': skew_proxy
        }
    
    def _add_enhanced_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add enhanced derived features."""
        # Price-based features
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        df['realized_vol_20d'] = df['returns'].rolling(window=20).std() * np.sqrt(252)
        
        # Moving averages
        for window in [5, 10, 20]:
            df[f'mavg_{window}d'] = df['close'].rolling(window=window).mean()
            df[f'close_vs_mavg_{window}d'] = df['close'] / df[f'mavg_{window}d'] - 1
        
        # Momentum
        for period in [3, 5, 10, 20]:
            df[f'momentum_{period}d'] = df['close'].pct_change(period)
        
        # Volume features
        df['volume_ma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        # Enhanced Greeks features (only if Greeks are available)
        if 'delta' in df.columns:
            df['delta_change'] = df['delta'].diff()
            df['delta_ma'] = df['delta'].rolling(window=5).mean()
            df['delta_momentum'] = df['delta'].pct_change(3)
        
        if 'gamma' in df.columns:
            df['gamma_change'] = df['gamma'].diff()
            df['gamma_ma'] = df['gamma'].rolling(window=5).mean()
            
            # Enhanced GEX features
            if 'gex_estimate' in df.columns:
                df['gex_change'] = df['gex_estimate'].diff()
                df['gex_ma'] = df['gex_estimate'].rolling(window=5).mean()
        
        # IV features (only if available)
        if 'avg_iv' in df.columns:
            df['iv_ma'] = df['avg_iv'].rolling(window=5).mean()
            df['iv_change'] = df['avg_iv'].diff()
            
            # IV-RV relationships using avg_iv instead of implied_volatility
            if 'realized_vol_20d' in df.columns:
                df['iv_rv_ratio'] = df['avg_iv'] / (df['realized_vol_20d'] + 1e-10)
                df['iv_rv_spread'] = df['avg_iv'] - df['realized_vol_20d']
        
        # Risk features
        df['peak'] = df['close'].expanding().max()
        df['drawdown'] = (df['close'] - df['peak']) / df['peak']
        
        # Market regime features
        if 'data_quality_score' in df.columns:
            df['high_quality_data'] = (df['data_quality_score'] >= 3).astype(int)
        
        return df

# Convenience function for backward compatibility
def get_enhanced_spy_data(start_date: str, end_date: str) -> pd.DataFrame:
    """Get enhanced SPY data with proper Greeks integration."""
    loader = FixedSPYDataLoader()
    return loader.get_spy_trading_data_enhanced(start_date, end_date, prefer_greeks_quality=True)

if __name__ == "__main__":
    # Test the enhanced loader
    loader = FixedSPYDataLoader()
    
    # Test with a period that should have good Greeks data
    test_data = loader.get_spy_trading_data_enhanced("2023-12-01", "2023-12-29")
    
    print(f"Enhanced data loaded: {len(test_data)} days")
    print(f"Features: {len(test_data.columns)}")
    
    if not test_data.empty:
        # Check Greeks coverage
        greeks_cols = ['delta', 'gamma', 'vega', 'theta']
        for col in greeks_cols:
            if col in test_data.columns:
                coverage = test_data[col].notna().sum()
                print(f"{col} coverage: {coverage}/{len(test_data)} ({coverage/len(test_data)*100:.1f}%)")
        
        print(f"\nSample data:")
        print(test_data[['close', 'delta', 'gamma', 'put_call_ratio', 'data_quality_score']].tail())