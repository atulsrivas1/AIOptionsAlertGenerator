"""
SPY Data Loader - Optimized for Real SPY Options Data
Handles multi-timeframe price data + rich options chains with real Greeks
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

class SPYDataLoader:
    """
    Advanced data loader specifically designed for your SPY data structure.
    
    Features:
    - Multi-timeframe price data (minute, hour, day)
    - Rich options chains with real Greeks
    - Smart contract selection and filtering
    - Advanced feature engineering from options data
    - Optimized for trading strategy development
    """
    
    def __init__(self, data_dir: str = "data_cache"):
        """
        Initialize the SPY data loader.
        
        Args:
            data_dir: Directory containing SPY data files
        """
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise ValueError(f"Data directory not found: {data_dir}")
        
        self.price_data = {}  # Cache for price data
        self.options_data = {}  # Cache for options data
        
        logger.info(f"SPYDataLoader initialized with data_dir: {self.data_dir}")
        self._validate_data_files()
    
    def _validate_data_files(self):
        """Validate that required data files exist."""
        required_files = [
            "SPY_day_2022-01-01_2023-12-31.json",
            "SPY_hour_2022-01-01_2023-12-31.json", 
            "SPY_minute_2022-01-01_2023-12-31.json"
        ]
        
        missing_files = []
        for file in required_files:
            if not (self.data_dir / file).exists():
                missing_files.append(file)
        
        if missing_files:
            logger.warning(f"Missing files: {missing_files}")
        else:
            logger.info("All required price data files found")
        
        # Check for options data files
        options_files = list(self.data_dir.glob("SPY_202*.json"))
        options_files = [f for f in options_files if f.name not in required_files]
        logger.info(f"Found {len(options_files)} daily options chain files")
    
    def load_price_data(self, timeframe: str = "day") -> pd.DataFrame:
        """
        Load SPY price data for specified timeframe.
        
        Args:
            timeframe: 'minute', 'hour', or 'day'
            
        Returns:
            DataFrame with OHLCV data
        """
        if timeframe in self.price_data:
            logger.info(f"Using cached {timeframe} price data")
            return self.price_data[timeframe]
        
        filename = f"SPY_{timeframe}_2022-01-01_2023-12-31.json"
        filepath = self.data_dir / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"Price data file not found: {filename}")
        
        logger.info(f"Loading {timeframe} price data from {filename}")
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        df.sort_index(inplace=True)
        
        # Ensure proper data types
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        logger.info(f"Loaded {len(df)} {timeframe} price records from {df.index.min()} to {df.index.max()}")
        
        # Cache the data
        self.price_data[timeframe] = df
        return df
    
    def load_options_chain(self, date: str) -> pd.DataFrame:
        """
        Load options chain data for a specific date.
        
        Args:
            date: Date in YYYY-MM-DD format
            
        Returns:
            DataFrame with options chain data
        """
        date_key = date.replace('-', '_')
        
        if date_key in self.options_data:
            return self.options_data[date_key]
        
        filename = f"SPY_{date}.json"
        filepath = self.data_dir / filename
        
        if not filepath.exists():
            logger.warning(f"Options data not found for {date}")
            return pd.DataFrame()
        
        logger.debug(f"Loading options chain for {date}")
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        if df.empty:
            return df
        
        # Clean and process options data
        df = self._process_options_data(df, date)
        
        # Cache the data
        self.options_data[date_key] = df
        return df
    
    def _process_options_data(self, df: pd.DataFrame, date: str) -> pd.DataFrame:
        """Process and clean options chain data."""
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
        
        # Calculate additional metrics
        if 'delta' in df.columns and 'gamma' in df.columns:
            df['delta_gamma_ratio'] = df['delta'] / (df['gamma'] + 1e-10)
        
        if 'vega' in df.columns and 'theta' in df.columns:
            df['vega_theta_ratio'] = df['vega'] / (abs(df['theta']) + 1e-10)
        
        return df
    
    def get_spy_trading_data(self, start_date: str, end_date: str, 
                           target_strike: Optional[float] = None,
                           target_expiry_days: int = 30) -> pd.DataFrame:
        """
        Get comprehensive SPY trading data combining price and options data.
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            target_strike: Specific strike price to track (if None, uses ATM)
            target_expiry_days: Target days to expiration for option selection
            
        Returns:
            DataFrame with combined price + options features
        """
        logger.info(f"Building SPY trading data from {start_date} to {end_date}")
        
        # Load daily price data as base
        price_df = self.load_price_data('day')
        price_df = price_df[(price_df.index >= start_date) & (price_df.index <= end_date)].copy()
        
        if price_df.empty:
            logger.warning("No price data found for date range")
            return pd.DataFrame()
        
        # Add options data for each trading day
        options_features = []
        
        for date in price_df.index.strftime('%Y-%m-%d'):
            options_df = self.load_options_chain(date)
            
            if options_df.empty:
                # Fill with NaN if no options data
                options_features.append({
                    'date': pd.to_datetime(date),
                    'implied_volatility': np.nan,
                    'delta': np.nan,
                    'gamma': np.nan,
                    'vega': np.nan,
                    'theta': np.nan,
                    'open_interest': np.nan,
                    'put_call_ratio': np.nan
                })
                continue
            
            # Select target option contract
            target_option = self._select_target_option(
                options_df, price_df.loc[date, 'close'], 
                target_strike, target_expiry_days
            )
            
            if target_option is not None:
                # Calculate aggregate options metrics
                options_metrics = self._calculate_options_metrics(options_df, price_df.loc[date, 'close'])
                
                # Combine target option + aggregate metrics
                feature_dict = {
                    'date': pd.to_datetime(date),
                    'implied_volatility': target_option.get('implied_volatility', np.nan),
                    'delta': target_option.get('delta', np.nan),
                    'gamma': target_option.get('gamma', np.nan),
                    'vega': target_option.get('vega', np.nan),
                    'theta': target_option.get('theta', np.nan),
                    'open_interest': target_option.get('open_interest', np.nan),
                    'days_to_expiry': target_option.get('days_to_expiry', np.nan),
                    'strike_price': target_option.get('strike_price', np.nan),
                    **options_metrics
                }
                
                options_features.append(feature_dict)
            else:
                # No suitable option found
                options_features.append({
                    'date': pd.to_datetime(date),
                    'implied_volatility': np.nan,
                    'delta': np.nan,
                    'gamma': np.nan,
                    'vega': np.nan,
                    'theta': np.nan,
                    'open_interest': np.nan,
                    'put_call_ratio': np.nan
                })
        
        # Convert options features to DataFrame
        options_df = pd.DataFrame(options_features)
        options_df.set_index('date', inplace=True)
        
        # Merge price and options data
        combined_df = price_df.join(options_df, how='left')
        
        # Add derived features
        combined_df = self._add_derived_features(combined_df)
        
        logger.info(f"Built trading data: {len(combined_df)} days with {len(combined_df.columns)} features")
        return combined_df
    
    def _select_target_option(self, options_df: pd.DataFrame, spot_price: float,
                            target_strike: Optional[float], target_expiry_days: int) -> Optional[Dict]:
        """Select the most appropriate option contract for tracking."""
        if options_df.empty:
            return None
        
        # Filter to calls only for now
        calls_df = options_df[options_df['contract_type'] == 'call'].copy()
        
        if calls_df.empty:
            return None
        
        # Filter by expiry range (target_expiry_days ± 10 days)
        if 'days_to_expiry' in calls_df.columns:
            expiry_mask = (calls_df['days_to_expiry'] >= target_expiry_days - 10) & \
                         (calls_df['days_to_expiry'] <= target_expiry_days + 10)
            calls_df = calls_df[expiry_mask]
        
        if calls_df.empty:
            return None
        
        if target_strike is not None:
            # Use specific strike if provided
            strike_options = calls_df[calls_df['strike_price'] == target_strike]
            if not strike_options.empty:
                return strike_options.iloc[0].to_dict()
        
        # Find ATM option (closest to spot price)
        calls_df['strike_distance'] = abs(calls_df['strike_price'] - spot_price)
        atm_option = calls_df.loc[calls_df['strike_distance'].idxmin()]
        
        return atm_option.to_dict()
    
    def _calculate_options_metrics(self, options_df: pd.DataFrame, spot_price: float) -> Dict:
        """Calculate aggregate options market metrics."""
        if options_df.empty:
            return {'put_call_ratio': np.nan, 'total_oi': np.nan, 'avg_iv': np.nan}
        
        # Split calls and puts
        calls = options_df[options_df['contract_type'] == 'call']
        puts = options_df[options_df['contract_type'] == 'put']
        
        # Put/Call ratio by open interest
        call_oi = calls['open_interest'].sum() if not calls.empty else 0
        put_oi = puts['open_interest'].sum() if not puts.empty else 0
        put_call_ratio = put_oi / (call_oi + 1e-10)
        
        # Total open interest
        total_oi = options_df['open_interest'].sum()
        
        # Average IV (if available)
        avg_iv = options_df['implied_volatility'].mean() if 'implied_volatility' in options_df.columns else np.nan
        
        # GEX approximation (simplified)
        if not options_df.empty and 'gamma' in options_df.columns and 'open_interest' in options_df.columns:
            # Simplified GEX calculation: sum(gamma * open_interest * spot^2)
            options_df['gex_contribution'] = options_df['gamma'] * options_df['open_interest'] * spot_price * spot_price
            total_gex = options_df['gex_contribution'].sum()
        else:
            total_gex = np.nan
        
        return {
            'put_call_ratio': put_call_ratio,
            'total_oi': total_oi,
            'avg_iv': avg_iv,
            'gex_estimate': total_gex
        }
    
    def _add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived features from price and options data."""
        # Price-based features
        df['returns'] = df['close'].pct_change()
        df['realized_vol_20d'] = df['returns'].rolling(window=20).std() * np.sqrt(252)
        df['momentum_5d'] = df['close'].pct_change(5)
        df['momentum_20d'] = df['close'].pct_change(20)
        
        # Volume features
        df['volume_ma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        # Options-specific features
        if 'implied_volatility' in df.columns and 'realized_vol_20d' in df.columns:
            df['iv_rv_ratio'] = df['implied_volatility'] / df['realized_vol_20d']
            df['iv_rv_spread'] = df['implied_volatility'] - df['realized_vol_20d']
        
        # Greeks momentum
        for greek in ['delta', 'gamma', 'vega', 'theta']:
            if greek in df.columns:
                df[f'{greek}_change'] = df[greek].diff()
                df[f'{greek}_ma'] = df[greek].rolling(window=5).mean()
        
        # Risk metrics
        df['peak'] = df['close'].expanding().max()
        df['drawdown'] = (df['close'] - df['peak']) / df['peak']
        
        return df
    
    def get_contract_specific_data(self, contract_ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Get data for a specific options contract across multiple days.
        
        Args:
            contract_ticker: Full option ticker (e.g., "O:SPY251219C00500000")
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            DataFrame with contract-specific data over time
        """
        logger.info(f"Loading data for contract {contract_ticker}")
        
        # Parse contract details
        contract_info = self._parse_contract_ticker(contract_ticker)
        
        # Load price data as base
        price_df = self.load_price_data('day')
        price_df = price_df[(price_df.index >= start_date) & (price_df.index <= end_date)].copy()
        
        contract_data = []
        
        for date in price_df.index.strftime('%Y-%m-%d'):
            options_df = self.load_options_chain(date)
            
            if not options_df.empty:
                # Find the specific contract
                contract_row = options_df[options_df['ticker'] == contract_ticker]
                
                if not contract_row.empty:
                    contract_data.append({
                        'date': pd.to_datetime(date),
                        'spot_price': price_df.loc[date, 'close'],
                        **contract_row.iloc[0].to_dict()
                    })
        
        if not contract_data:
            logger.warning(f"No data found for contract {contract_ticker}")
            return pd.DataFrame()
        
        # Convert to DataFrame
        result_df = pd.DataFrame(contract_data)
        result_df.set_index('date', inplace=True)
        
        # Add moneyness and other derived metrics
        if 'strike_price' in result_df.columns and 'spot_price' in result_df.columns:
            result_df['moneyness'] = result_df['spot_price'] / result_df['strike_price']
            result_df['intrinsic_value'] = np.maximum(result_df['spot_price'] - result_df['strike_price'], 0)
        
        logger.info(f"Found {len(result_df)} days of data for {contract_ticker}")
        return result_df
    
    def _parse_contract_ticker(self, ticker: str) -> Dict:
        """Parse option ticker to extract contract details."""
        # Example: "O:SPY251219C00500000" 
        # O: = Option, SPY = underlying, 251219 = exp date, C = call, 00500000 = strike
        parts = ticker.split(':')[1] if ':' in ticker else ticker
        
        return {
            'underlying': parts[:3] if len(parts) > 10 else 'SPY',
            'expiry': parts[3:9] if len(parts) > 10 else None,
            'option_type': parts[9] if len(parts) > 10 else 'C',
            'strike': int(parts[10:]) / 1000 if len(parts) > 10 else None
        }
    
    def get_data_summary(self) -> Dict:
        """Get summary statistics of available data."""
        summary = {}
        
        # Price data summary
        for timeframe in ['day', 'hour', 'minute']:
            try:
                df = self.load_price_data(timeframe)
                summary[f'{timeframe}_price'] = {
                    'records': len(df),
                    'date_range': f"{df.index.min().date()} to {df.index.max().date()}",
                    'avg_volume': f"{df['volume'].mean():,.0f}"
                }
            except FileNotFoundError:
                summary[f'{timeframe}_price'] = {'status': 'not_found'}
        
        # Options data summary
        options_files = list(self.data_dir.glob("SPY_202*.json"))
        options_files = [f for f in options_files if 'day_' not in f.name and 'hour_' not in f.name and 'minute_' not in f.name]
        
        summary['options'] = {
            'daily_files': len(options_files),
            'date_range': f"{min(f.stem.split('_')[1] for f in options_files)} to {max(f.stem.split('_')[1] for f in options_files)}"
        }
        
        return summary

# Convenience function for backward compatibility
def get_historical_option_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Backward-compatible function that uses the new SPY data loader.
    """
    loader = SPYDataLoader()
    
    if ticker.startswith('O:SPY'):
        # Specific contract request
        return loader.get_contract_specific_data(ticker, start_date, end_date)
    else:
        # General SPY trading data
        return loader.get_spy_trading_data(start_date, end_date)

if __name__ == "__main__":
    # Test the data loader
    loader = SPYDataLoader()
    
    # Print data summary
    summary = loader.get_data_summary()
    print("Data Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    # Test loading trading data
    trading_data = loader.get_spy_trading_data("2023-01-01", "2023-01-31")
    print(f"\nTrading data loaded: {len(trading_data)} days")
    print(f"Columns: {list(trading_data.columns)}")
    
    if not trading_data.empty:
        print(f"Sample data:\n{trading_data.head()}")