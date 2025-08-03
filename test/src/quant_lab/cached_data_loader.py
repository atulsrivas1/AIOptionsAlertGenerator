"""
Enhanced Data Loader with Local Caching and Parallel Downloads
Supports local file cache with fallback to Polygon.io API
"""

import pandas as pd
import numpy as np
import os
import json
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple
from pathlib import Path
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from dotenv import load_dotenv
import pickle
import warnings
warnings.filterwarnings('ignore')

load_dotenv()
logger = logging.getLogger(__name__)

class CachedDataLoader:
    """
    Enhanced data loader with local caching and parallel downloading capabilities.
    
    Features:
    - Checks local cache directory first
    - Falls back to Polygon.io API if not found
    - Parallel downloading for multiple contracts/dates
    - Automatic caching of downloaded data
    - Supports both CSV and pickle formats
    """
    
    def __init__(self, cache_dir: str = "options_data_cache", max_workers: int = 5):
        """
        Initialize the cached data loader.
        
        Args:
            cache_dir: Directory to store cached data files
            max_workers: Maximum number of parallel download workers
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.max_workers = max_workers
        self.api_key = os.getenv('POLYGON_API_KEY')
        
        if not self.api_key:
            logger.warning("No POLYGON_API_KEY found in environment variables")
        
        logger.info(f"CachedDataLoader initialized with cache_dir: {self.cache_dir}")
    
    def _get_cache_filename(self, ticker: str, start_date: str, end_date: str) -> Path:
        """Generate standardized cache filename."""
        # Clean ticker for filename
        clean_ticker = ticker.replace(":", "_").replace("/", "_")
        filename = f"{clean_ticker}_{start_date}_{end_date}.pkl"
        return self.cache_dir / filename
    
    def _load_from_cache(self, cache_file: Path) -> Optional[pd.DataFrame]:
        """Load data from cache file."""
        try:
            if cache_file.exists():
                logger.info(f"Loading data from cache: {cache_file}")
                
                if cache_file.suffix == '.csv':
                    df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
                elif cache_file.suffix == '.pkl':
                    df = pd.read_pickle(cache_file)
                else:
                    logger.warning(f"Unsupported cache file format: {cache_file}")
                    return None
                
                logger.info(f"Loaded {len(df)} rows from cache")
                return df
            
        except Exception as e:
            logger.error(f"Error loading cache file {cache_file}: {e}")
            
        return None
    
    def _save_to_cache(self, data: pd.DataFrame, cache_file: Path) -> None:
        """Save data to cache file."""
        try:
            # Save as pickle for better performance and data type preservation
            data.to_pickle(cache_file)
            logger.info(f"Cached data to: {cache_file} ({len(data)} rows)")
            
            # Also save as CSV for manual inspection
            csv_file = cache_file.with_suffix('.csv')
            data.to_csv(csv_file)
            logger.info(f"Also saved CSV: {csv_file}")
            
        except Exception as e:
            logger.error(f"Error saving to cache {cache_file}: {e}")
    
    def _download_from_polygon(self, ticker: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """Download data from Polygon.io API (fallback method)."""
        try:
            logger.info(f"Downloading {ticker} from Polygon.io API...")
            
            # Import the original data loader as fallback
            from .data_loader import PolygonDataLoader
            
            polygon_loader = PolygonDataLoader()
            data = polygon_loader.get_historical_option_data(ticker, start_date, end_date)
            
            if not data.empty:
                logger.info(f"Downloaded {len(data)} rows from Polygon.io")
                return data
            else:
                logger.warning(f"No data received from Polygon.io for {ticker}")
                return None
                
        except Exception as e:
            logger.error(f"Error downloading from Polygon.io: {e}")
            return None
    
    def get_historical_option_data(self, ticker: str, start_date: str, end_date: str, 
                                 force_download: bool = False) -> pd.DataFrame:
        """
        Get historical option data with caching support.
        
        Args:
            ticker: Option ticker symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD) 
            force_download: If True, skip cache and download fresh data
            
        Returns:
            DataFrame with historical option data
        """
        cache_file = self._get_cache_filename(ticker, start_date, end_date)
        
        # Try to load from cache first (unless force_download is True)
        if not force_download:
            cached_data = self._load_from_cache(cache_file)
            if cached_data is not None:
                return cached_data
        
        # Check for manual data files (CSV format)
        manual_csv_files = list(self.cache_dir.glob(f"*{ticker.replace(':', '_')}*.csv"))
        if manual_csv_files and not force_download:
            logger.info(f"Found manual CSV files: {manual_csv_files}")
            try:
                # Try to load the most recent manual file
                manual_file = max(manual_csv_files, key=os.path.getctime)
                logger.info(f"Loading manual data from: {manual_file}")
                data = pd.read_csv(manual_file, index_col=0, parse_dates=True)
                
                # Filter by date range if needed
                data = data[(data.index >= start_date) & (data.index <= end_date)]
                
                if not data.empty:
                    # Cache this data for future use
                    self._save_to_cache(data, cache_file)
                    return data
                    
            except Exception as e:
                logger.error(f"Error loading manual CSV file: {e}")
        
        # Download from Polygon.io as fallback
        logger.info(f"Cache miss - downloading {ticker} from Polygon.io...")
        data = self._download_from_polygon(ticker, start_date, end_date)
        
        if data is not None and not data.empty:
            # Cache the downloaded data
            self._save_to_cache(data, cache_file)
            return data
        else:
            logger.error(f"Failed to get data for {ticker}")
            return pd.DataFrame()
    
    def get_multiple_contracts_parallel(self, contracts: List[Dict[str, str]]) -> Dict[str, pd.DataFrame]:
        """
        Download multiple option contracts in parallel.
        
        Args:
            contracts: List of dicts with keys: 'ticker', 'start_date', 'end_date'
            
        Returns:
            Dictionary mapping ticker to DataFrame
        """
        logger.info(f"Starting parallel download of {len(contracts)} contracts...")
        results = {}
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all download tasks
            future_to_contract = {
                executor.submit(
                    self.get_historical_option_data,
                    contract['ticker'],
                    contract['start_date'], 
                    contract['end_date']
                ): contract for contract in contracts
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_contract):
                contract = future_to_contract[future]
                ticker = contract['ticker']
                
                try:
                    data = future.result()
                    results[ticker] = data
                    logger.info(f"✓ Completed {ticker}: {len(data)} rows")
                    
                except Exception as e:
                    logger.error(f"✗ Failed {ticker}: {e}")
                    results[ticker] = pd.DataFrame()
        
        logger.info(f"Parallel download completed: {len(results)} contracts processed")
        return results
    
    def get_date_range_parallel(self, ticker: str, start_date: str, end_date: str, 
                              chunk_days: int = 30) -> pd.DataFrame:
        """
        Download data for a large date range by splitting into parallel chunks.
        
        Args:
            ticker: Option ticker symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            chunk_days: Number of days per chunk for parallel downloading
            
        Returns:
            Combined DataFrame with all data
        """
        logger.info(f"Downloading {ticker} in parallel chunks of {chunk_days} days...")
        
        # Generate date chunks
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        
        chunks = []
        current_start = start_dt
        
        while current_start < end_dt:
            current_end = min(current_start + timedelta(days=chunk_days), end_dt)
            chunks.append({
                'ticker': ticker,
                'start_date': current_start.strftime('%Y-%m-%d'),
                'end_date': current_end.strftime('%Y-%m-%d')
            })
            current_start = current_end + timedelta(days=1)
        
        logger.info(f"Split into {len(chunks)} chunks for parallel download")
        
        # Download chunks in parallel
        chunk_results = self.get_multiple_contracts_parallel(chunks)
        
        # Combine all chunks
        all_data = []
        for i, chunk in enumerate(chunks):
            chunk_key = chunk['ticker']  # All chunks have same ticker
            if chunk_key in chunk_results and not chunk_results[chunk_key].empty:
                all_data.append(chunk_results[chunk_key])
        
        if all_data:
            combined_data = pd.concat(all_data, ignore_index=False)
            combined_data = combined_data.sort_index().drop_duplicates()
            logger.info(f"Combined parallel chunks: {len(combined_data)} total rows")
            
            # Cache the combined result
            cache_file = self._get_cache_filename(ticker, start_date, end_date)
            self._save_to_cache(combined_data, cache_file)
            
            return combined_data
        else:
            logger.warning("No data retrieved from any chunks")
            return pd.DataFrame()
    
    def list_cached_files(self) -> List[Dict[str, str]]:
        """List all cached data files with metadata."""
        cached_files = []
        
        for file_path in self.cache_dir.glob("*.pkl"):
            try:
                stat = file_path.stat()
                cached_files.append({
                    'filename': file_path.name,
                    'size_mb': round(stat.st_size / 1024 / 1024, 2),
                    'modified': datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S'),
                    'path': str(file_path)
                })
            except Exception as e:
                logger.error(f"Error reading file info for {file_path}: {e}")
        
        return sorted(cached_files, key=lambda x: x['modified'], reverse=True)
    
    def clear_cache(self, pattern: str = "*") -> int:
        """Clear cached files matching pattern."""
        files_removed = 0
        
        for file_path in self.cache_dir.glob(f"{pattern}.pkl"):
            try:
                file_path.unlink()
                # Also remove corresponding CSV if it exists
                csv_path = file_path.with_suffix('.csv')
                if csv_path.exists():
                    csv_path.unlink()
                files_removed += 1
                logger.info(f"Removed cached file: {file_path}")
            except Exception as e:
                logger.error(f"Error removing {file_path}: {e}")
        
        logger.info(f"Cleared {files_removed} cached files")
        return files_removed
    
    def get_cache_stats(self) -> Dict[str, any]:
        """Get cache directory statistics."""
        cache_files = list(self.cache_dir.glob("*.pkl"))
        csv_files = list(self.cache_dir.glob("*.csv"))
        
        total_size = sum(f.stat().st_size for f in cache_files + csv_files)
        
        return {
            'cache_directory': str(self.cache_dir),
            'pickle_files': len(cache_files),
            'csv_files': len(csv_files),
            'total_files': len(cache_files) + len(csv_files),
            'total_size_mb': round(total_size / 1024 / 1024, 2),
            'oldest_file': min((f.stat().st_mtime for f in cache_files + csv_files), default=None),
            'newest_file': max((f.stat().st_mtime for f in cache_files + csv_files), default=None)
        }

# Example usage and testing
if __name__ == "__main__":
    # Example usage
    loader = CachedDataLoader(cache_dir="options_data_cache")
    
    # Print cache stats
    stats = loader.get_cache_stats()
    print("Cache Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Example: Load single contract
    ticker = "O:SPY251219C00500000"
    start_date = "2022-01-01"
    end_date = "2023-12-31"
    
    data = loader.get_historical_option_data(ticker, start_date, end_date)
    print(f"\nLoaded {len(data)} rows for {ticker}")
    
    # Example: Load multiple contracts in parallel
    contracts = [
        {"ticker": "O:SPY251219C00500000", "start_date": "2022-01-01", "end_date": "2023-12-31"},
        {"ticker": "O:SPY251219C00550000", "start_date": "2022-01-01", "end_date": "2023-12-31"},
    ]
    
    # parallel_results = loader.get_multiple_contracts_parallel(contracts)
    # print(f"\nParallel results: {len(parallel_results)} contracts loaded")