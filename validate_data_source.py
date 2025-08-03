import os
from polygon import RESTClient
from datetime import date

# --- Configuration ---
# IMPORTANT: Set your Polygon API key as an environment variable named POLYGON_API_KEY
# or replace the os.getenv call with your actual key: client = RESTClient("YOUR_API_KEY")
API_KEY = os.getenv("POLYGON_API_KEY")

# The underlying stock ticker to query
UNDERLYING_TICKER = "SPY"

# A specific expiration date to look for
# Format: YYYY-MM-DD
EXPIRATION_DATE = "2025-12-19"

# --- Main Script ---

def main():
    """Connects to Polygon.io and fetches a sample of the options chain."""
    if not API_KEY:
        print("Error: POLYGON_API_KEY environment variable not set.")
        print("Please set the environment variable or hardcode your API key in the script.")
        return

    print(f"Connecting to Polygon.io with API key...\n")
    client = RESTClient(API_KEY)

    print(f"Fetching options chain for {UNDERLYING_TICKER} expiring on or after {EXPIRATION_DATE}...\n")

    try:
        # Use the list_snapshot_options_chain method to get the data
        # We will limit the results to 5 for this validation script
        chain = client.list_snapshot_options_chain(UNDERLYING_TICKER, params={
            "expiration_date": EXPIRATION_DATE,
            "limit": 5
        })

        print("--- Successfully fetched options chain data --- \n")

        for i, contract in enumerate(chain):
            print(f"--- Contract {i+1} ---")
            print(f"  Ticker: {contract.ticker}")
            print(f"  Type: {contract.contract_type}")
            print(f"  Strike Price: {contract.details.strike_price}")
            print(f"  Expiration Date: {contract.details.expiration_date}")
            print(f"  Open Interest: {contract.open_interest}")
            
            # Print the Greeks and IV, which are crucial for our features
            if hasattr(contract, 'greeks') and contract.greeks:
                print(f"  Greeks:")
                print(f"    Delta: {contract.greeks.delta}")
                print(f"    Gamma: {contract.greeks.gamma}")
                print(f"    Vega: {contract.greeks.vega}")
                print(f"    Theta: {contract.greeks.theta}")
            else:
                print("  Greeks: Not available for this contract.")

            if hasattr(contract, 'implied_volatility'):
                 print(f"  Implied Volatility: {contract.implied_volatility}")
            else:
                print("  Implied Volatility: Not available for this contract.")

            print("\n")

    except Exception as e:
        print(f"An error occurred: {e}")
        print("Please ensure your API key is correct and has permissions for options data.")

if __name__ == "__main__":
    main()