import os
import pandas as pd
import numpy as np
from polygon import RESTClient
from datetime import date, datetime
import logging
from typing import Optional, Dict, Any

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataValidationError(Exception):
    """Custom exception for data validation failures"""
    pass

class PolygonDataLoader:
    """Enhanced data loader with comprehensive validation and error handling"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("POLYGON_API_KEY")
        if not self.api_key:
            raise ValueError("POLYGON_API_KEY environment variable not set or api_key not provided.")
        
        self.client = RESTClient(self.api_key)
        self.required_columns = ['open', 'high', 'low', 'close', 'volume', 'implied_volatility']
        self.data_quality_thresholds = {
            'min_data_points': 50,
            'max_missing_data_pct': 0.10,
            'min_volume': 1,
            'max_price_change_pct': 0.50,  # 50% max daily change filter
            'min_iv': 0.01,
            'max_iv': 5.0
        }
    
    def get_historical_option_data(self, option_ticker: str, start_date: str, end_date: str, 
                                 validate_data: bool = True) -> pd.DataFrame:
        """
        Fetches historical OHLCV data for a given option contract from Polygon.io with enhanced validation.

        Args:
            option_ticker (str): The option ticker symbol (e.g., O:SPY251219C00500000).
            start_date (str): The start date in YYYY-MM-DD format.
            end_date (str): The end date in YYYY-MM-DD format.
            validate_data (bool): Whether to perform data quality validation.

        Returns:
            pd.DataFrame: A DataFrame containing the OHLCV data, with the date as the index.
            
        Raises:
            DataValidationError: If data quality checks fail.
        """
        logger.info(f"Fetching data for {option_ticker} from {start_date} to {end_date}")
        
        # Validate input parameters
        self._validate_inputs(option_ticker, start_date, end_date)
        
        try:
            # Fetch data from Polygon API
            raw_data = self._fetch_raw_data(option_ticker, start_date, end_date)
            
            if raw_data.empty:
                logger.warning(f"No data returned for {option_ticker}")
                return pd.DataFrame()
            
            # Process and clean the data
            processed_data = self._process_raw_data(raw_data)
            
            # Validate data quality if requested
            if validate_data:
                self._validate_data_quality(processed_data, option_ticker)
            
            logger.info(f"Successfully loaded {len(processed_data)} data points for {option_ticker}")
            return processed_data
            
        except Exception as e:
            logger.error(f"Error fetching data for {option_ticker}: {str(e)}")
            if isinstance(e, DataValidationError):
                raise
            return pd.DataFrame()
    
    def _validate_inputs(self, option_ticker: str, start_date: str, end_date: str) -> None:
        """Validate input parameters"""
        if not option_ticker or not isinstance(option_ticker, str):
            raise ValueError("option_ticker must be a non-empty string")
        
        if not option_ticker.startswith('O:'):
            raise ValueError("option_ticker must start with 'O:' for options contracts")
        
        try:
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        except ValueError:
            raise ValueError("Dates must be in YYYY-MM-DD format")
        
        if start_dt >= end_dt:
            raise ValueError("start_date must be before end_date")
        
        if end_dt > datetime.now():
            logger.warning("end_date is in the future, may result in incomplete data")
    
    def _fetch_raw_data(self, option_ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch raw data from Polygon API with retry logic"""
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                aggs = self.client.get_aggs(
                    ticker=option_ticker,
                    multiplier=1,
                    timespan="day",
                    from_=start_date,
                    to=end_date,
                    params={"greeks": "true"}
                )
                
                df = pd.DataFrame(aggs)
                return df
                
            except Exception as e:
                retry_count += 1
                logger.warning(f"Attempt {retry_count} failed for {option_ticker}: {e}")
                if retry_count >= max_retries:
                    raise e
        
        return pd.DataFrame()
    
    def _process_raw_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process and clean raw data from Polygon API"""
        if df.empty:
            return df
        
        # Convert timestamp to datetime and set as index
        df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.set_index('date')
        
        # Ensure all required columns are present
        for col in self.required_columns:
            if col not in df.columns:
                logger.warning(f"Missing column {col}, filling with appropriate default")
                if col == 'implied_volatility':
                    df[col] = 0.20  # Default IV of 20%
                else:
                    df[col] = 0
        
        # Select only required columns
        df = df[self.required_columns]
        
        # Clean and validate OHLC relationships
        df = self._clean_ohlc_data(df)
        
        # Handle missing values
        df = self._handle_missing_values(df)
        
        # Sort by date
        df = df.sort_index()
        
        return df
    
    def _clean_ohlc_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean OHLC data and fix inconsistencies"""
        # Ensure OHLC relationships are valid
        df.loc[df['high'] < df['low'], 'high'] = df.loc[df['high'] < df['low'], 'low']
        df.loc[df['open'] < df['low'], 'open'] = df.loc[df['open'] < df['low'], 'low']
        df.loc[df['open'] > df['high'], 'open'] = df.loc[df['open'] > df['high'], 'high']
        df.loc[df['close'] < df['low'], 'close'] = df.loc[df['close'] < df['low'], 'low']
        df.loc[df['close'] > df['high'], 'close'] = df.loc[df['close'] > df['high'], 'high']
        
        # Remove rows with invalid prices (negative or zero)
        price_cols = ['open', 'high', 'low', 'close']
        df = df[(df[price_cols] > 0).all(axis=1)]
        
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values with appropriate strategies"""
        # Forward fill missing values (common in financial data)
        df = df.fillna(method='ffill')
        
        # For remaining NaNs at the beginning, use backward fill
        df = df.fillna(method='bfill')
        
        # If still NaNs, drop those rows
        initial_length = len(df)
        df = df.dropna()
        
        if len(df) < initial_length:
            logger.warning(f"Dropped {initial_length - len(df)} rows due to missing values")
        
        return df
    
    def _validate_data_quality(self, df: pd.DataFrame, ticker: str) -> None:
        """Comprehensive data quality validation"""
        if df.empty:
            raise DataValidationError(f"No data available for {ticker}")
        
        # Check minimum data points
        if len(df) < self.data_quality_thresholds['min_data_points']:
            raise DataValidationError(
                f"Insufficient data points: {len(df)} < {self.data_quality_thresholds['min_data_points']}"
            )
        
        # Check for excessive missing data
        missing_pct = df.isnull().sum().sum() / (len(df) * len(df.columns))
        if missing_pct > self.data_quality_thresholds['max_missing_data_pct']:
            raise DataValidationError(
                f"Too much missing data: {missing_pct:.2%} > {self.data_quality_thresholds['max_missing_data_pct']:.2%}"
            )
        
        # Check for reasonable price movements
        price_changes = df['close'].pct_change().abs()
        extreme_moves = price_changes > self.data_quality_thresholds['max_price_change_pct']
        if extreme_moves.sum() > len(df) * 0.05:  # More than 5% of days have extreme moves
            logger.warning(f"High number of extreme price movements detected: {extreme_moves.sum()} days")
        
        # Check implied volatility bounds
        iv_out_of_bounds = (
            (df['implied_volatility'] < self.data_quality_thresholds['min_iv']) |
            (df['implied_volatility'] > self.data_quality_thresholds['max_iv'])
        )
        if iv_out_of_bounds.sum() > 0:
            logger.warning(f"IV out of bounds detected: {iv_out_of_bounds.sum()} days")
        
        # Check for data gaps
        date_gaps = pd.date_range(start=df.index.min(), end=df.index.max(), freq='D')
        missing_dates = set(date_gaps) - set(df.index)
        weekdays_missing = [d for d in missing_dates if d.weekday() < 5]  # Only count weekdays
        
        if len(weekdays_missing) > len(date_gaps) * 0.20:  # More than 20% weekdays missing
            logger.warning(f"Significant data gaps detected: {len(weekdays_missing)} missing weekdays")
        
        logger.info(f"Data quality validation passed for {ticker}")
    
    def get_data_quality_report(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate a comprehensive data quality report"""
        if df.empty:
            return {"error": "No data to analyze"}
        
        report = {
            "data_points": len(df),
            "date_range": {
                "start": df.index.min().strftime('%Y-%m-%d'),
                "end": df.index.max().strftime('%Y-%m-%d'),
                "days": (df.index.max() - df.index.min()).days
            },
            "missing_data": {
                "total_missing": df.isnull().sum().sum(),
                "missing_percentage": df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100
            },
            "price_statistics": {
                "mean_close": df['close'].mean(),
                "volatility": df['close'].pct_change().std() * np.sqrt(252),
                "max_daily_change": df['close'].pct_change().abs().max()
            },
            "volume_statistics": {
                "mean_volume": df['volume'].mean(),
                "zero_volume_days": (df['volume'] == 0).sum()
            },
            "iv_statistics": {
                "mean_iv": df['implied_volatility'].mean(),
                "iv_range": [df['implied_volatility'].min(), df['implied_volatility'].max()]
            }
        }
        
        return report

# Backward compatibility function
def get_historical_option_data(option_ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Backward compatibility wrapper for the enhanced data loader.
    """
    loader = PolygonDataLoader()
    return loader.get_historical_option_data(option_ticker, start_date, end_date)
