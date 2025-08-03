
import os
import pandas as pd
from alpha_vantage.timeseries import TimeSeries

# --- Configuration ---
API_KEY = "2IB4C01DYDDILE3T" # Your Alpha Vantage API Key
VIX_TICKER = "^VIX"
OUTPUT_FILE = "vix_data.csv"

def download_vix_data():
    """
    Downloads historical VIX data from Alpha Vantage and saves it to a CSV file.
    """
    print(f"Attempting to download VIX data for {VIX_TICKER}...")

    ts = TimeSeries(key=API_KEY, output_format='pandas')

    try:
        # Get daily adjusted VIX data
        vix_data, meta_data = ts.get_daily_adjusted(symbol=VIX_TICKER, outputsize='full')
        print("Successfully retrieved VIX data.")

        # Rename columns for clarity
        vix_data.rename(columns={
            '1. open': 'open',
            '2. high': 'high',
            '3. low': 'low',
            '4. close': 'close',
            '5. adjusted close': 'adjusted_close',
            '6. volume': 'volume',
            '7. dividend amount': 'dividend_amount',
            '8. split coefficient': 'split_coefficient'
        }, inplace=True)

        # Save to CSV
        vix_data.to_csv(OUTPUT_FILE)
        print(f"Successfully saved VIX data to {OUTPUT_FILE}")

    except Exception as e:
        print(f"An error occurred while trying to download the data: {e}")
        print("Please ensure your API key is correct and has not reached its limit.")

if __name__ == "__main__":
    download_vix_data()
