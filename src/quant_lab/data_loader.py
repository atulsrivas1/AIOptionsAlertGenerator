import os
import pandas as pd
from polygon import RESTClient
from alpha_vantage.timeseries import TimeSeries
from datetime import date, timedelta
import json
import time

CACHE_DIR = "data_cache"

def get_multi_timeframe_data(underlying_ticker: str, start_date: str, end_date: str, timeframes: list) -> dict:
    """
    Fetches historical OHLCV data for a given underlying asset across multiple timeframes.
    Uses a file-based cache.

    Args:
        underlying_ticker (str): The underlying stock ticker (e.g., SPY).
        start_date (str): The start date in YYYY-MM-DD format.
        end_date (str): The end date in YYYY-MM-DD format.
        timeframes (list): A list of timeframes (e.g., ["day", "hour", "minute"]).

    Returns:
        dict: A dictionary where keys are timeframes and values are DataFrames 
              containing the OHLCV data, with the date as the index.
    """
    multi_tf_data = {}
    API_KEY = os.getenv("POLYGON_API_KEY")
    if not API_KEY:
        raise ValueError("POLYGON_API_KEY environment variable not set.")

    client = RESTClient(API_KEY)

    for tf in timeframes:
        cache_file_path = os.path.join(CACHE_DIR, f"{underlying_ticker}_{tf}_{start_date}_{end_date}.json")

        if os.path.exists(cache_file_path):
            print(f"Loading {tf} data for {underlying_ticker} from cache...")
            try:
                with open(cache_file_path, 'r') as f:
                    # Check if the file is empty
                    if os.fstat(f.fileno()).st_size == 0:
                        print(f"Cache file for {tf} is empty. Refetching...")
                        os.remove(cache_file_path)
                    else:
                        data_json = json.load(f)
                        df = pd.DataFrame(data_json)
                        df['date'] = pd.to_datetime(df['date'])
                        df = df.set_index('date')
                        multi_tf_data[tf] = df
                        continue
            except json.JSONDecodeError:
                print(f"Error decoding JSON from cache file for {tf}. Deleting and refetching...")
                os.remove(cache_file_path)

        print(f"Fetching {tf} data for {underlying_ticker} from API...")
        try:
            aggs = client.get_aggs(
                ticker=underlying_ticker,
                multiplier=1,
                timespan=tf,
                from_=start_date,
                to=end_date,
                limit=50000 # Max limit
            )

            df = pd.DataFrame(aggs)
            if df.empty:
                print(f"No {tf} data found for {underlying_ticker}.")
                multi_tf_data[tf] = pd.DataFrame()
                continue
                
            df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.set_index('date')
            df = df[['open', 'high', 'low', 'close', 'volume']]
            multi_tf_data[tf] = df

            # Save to cache
            df_to_cache = df.reset_index()
            df_to_cache['date'] = df_to_cache['date'].astype(str)
            with open(cache_file_path, 'w') as f:
                json.dump(df_to_cache.to_dict(orient='records'), f)

            time.sleep(1) # To respect API rate limits

        except Exception as e:
            print(f"An error occurred while fetching {tf} data for {underlying_ticker}: {e}")
            multi_tf_data[tf] = pd.DataFrame()

    return multi_tf_data

def get_full_options_chain(client: RESTClient, underlying_ticker: str, target_date: str) -> pd.DataFrame:
    """
    Fetches the full options chain for a given underlying asset on a specific date.
    Uses a file-based cache to avoid re-fetching data.

    Args:
        client (RESTClient): An existing Polygon RESTClient object.
        underlying_ticker (str): The underlying stock ticker (e.g., SPY).
        target_date (str): The date for which to fetch the chain (YYYY-MM-DD).

    Returns:
        pd.DataFrame: A DataFrame containing the full options chain data.
    """
    cache_file_path = os.path.join(CACHE_DIR, f"{underlying_ticker}_{target_date}.json")

    # Check if data is in cache
    if os.path.exists(cache_file_path):
        print(f"Loading options chain for {target_date} from cache...")
        with open(cache_file_path, 'r') as f:
            data = json.load(f)
        chain_df = pd.DataFrame(data)
    else:
        # If not in cache, fetch from API
        print(f"Fetching options chain for {target_date} from API...")
        all_contracts = []
        try:
            for contract in client.list_snapshot_options_chain(underlying_ticker, params={"snapshot_date": target_date}):
                if hasattr(contract, 'details') and hasattr(contract, 'greeks') and contract.greeks and hasattr(contract, 'open_interest'):
                    # Fetch daily open, close, volume for the contract
                    daily_agg = client.get_daily_open_close_agg(contract.details.ticker, target_date)
                    volume = daily_agg.volume if hasattr(daily_agg, 'volume') else 0

                    contract_details = {
                        'ticker': contract.details.ticker,
                        'strike_price': contract.details.strike_price,
                        'contract_type': contract.details.contract_type.lower(),
                        'expiration_date': contract.details.expiration_date,
                        'open_interest': contract.open_interest,
                        'implied_volatility': contract.iv,
                        'gamma': contract.greeks.gamma,
                        'delta': contract.greeks.delta,
                        'vega': contract.greeks.vega,
                        'theta': contract.greeks.theta,
                        'volume': volume,
                        'underlying_price': contract.underlying_asset.price if hasattr(contract, 'underlying_asset') and hasattr(contract.underlying_asset, 'price') else 0
                    }
                    all_contracts.append(contract_details)
            
            chain_df = pd.DataFrame(all_contracts)
            
            # Save to cache
            with open(cache_file_path, 'w') as f:
                json.dump(all_contracts, f)

        except Exception as e:
            print(f"An error occurred while fetching the options chain for {underlying_ticker} on {target_date}: {e}")
            return pd.DataFrame()

    # Calculate days_to_expiry
    if not chain_df.empty and 'expiration_date' in chain_df.columns:
        chain_df['expiration_date'] = pd.to_datetime(chain_df['expiration_date'])
        chain_df['days_to_expiry'] = (chain_df['expiration_date'] - pd.to_datetime(target_date)).dt.days
    elif 'days_to_expiry' not in chain_df.columns:
        chain_df['days_to_expiry'] = np.nan

    return chain_df


def get_historical_option_data(option_ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetches historical OHLCV data for a given option contract from Polygon.io.

    Args:
        option_ticker (str): The option ticker symbol (e.g., O:SPY251219C00500000).
        start_date (str): The start date in YYYY-MM-DD format.
        end_date (str): The end date in YYYY-MM-DD format.

    Returns:
        pd.DataFrame: A DataFrame containing the OHLCV data, with the date as the index.
    """
    API_KEY = os.getenv("POLYGON_API_KEY")
    if not API_KEY:
        raise ValueError("POLYGON_API_KEY environment variable not set.")

    client = RESTClient(API_KEY)

    try:
        # Using get_aggs for an option ticker
        aggs = client.get_aggs(
            ticker=option_ticker,
            multiplier=1,
            timespan="day",
            from_=start_date,
            to=end_date,
            params={"greeks": "true"}
        )

        df = pd.DataFrame(aggs)
        if df.empty:
            return df
            
        df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.set_index('date')
        # Ensure all required columns are present, fill missing greeks with 0
        required_cols = ['open', 'high', 'low', 'close', 'volume', 'implied_volatility']
        for col in required_cols:
            if col not in df.columns:
                df[col] = 0
        df = df[required_cols]
        return df

    except Exception as e:
        print(f"An error occurred while fetching data for {option_ticker}: {e}")
        return pd.DataFrame()

def get_historical_stock_data_alpha_vantage(stock_ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetches historical OHLCV data for a given stock from Alpha Vantage.

    Args:
        stock_ticker (str): The stock ticker symbol (e.g., SPY).
        start_date (str): The start date in YYYY-MM-DD format.
        end_date (str): The end date in YYYY-MM-DD format.

    Returns:
        pd.DataFrame: A DataFrame containing the OHLCV data, with the date as the index.
    """
    API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
    if not API_KEY:
        raise ValueError("ALPHA_VANTAGE_API_KEY environment variable not set.")

    ts = TimeSeries(key=API_KEY, output_format='pandas')
    try:
        data, meta_data = ts.get_daily_adjusted(symbol=stock_ticker, outputsize='full')
        data = data.rename(columns={
            '1. open': 'open', 
            '2. high': 'high', 
            '3. low': 'low', 
            '4. close': 'close', 
            '6. volume': 'volume'
        })
        data = data[['open', 'high', 'low', 'close', 'volume']]
        data.index = pd.to_datetime(data.index)
        data = data[(data.index >= start_date) & (data.index <= end_date)]
        return data.sort_index()

    except Exception as e:
        print(f"An error occurred while fetching data for {stock_ticker} from Alpha Vantage: {e}")
        return pd.DataFrame()
