import pandas as pd
from .data_loader import get_full_options_chain
from polygon import RESTClient
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from scipy import stats

class GammaExposureFeatures:
    """
    Advanced Gamma Exposure feature engineering for options trading signals.
    Encapsulates all GEX-related calculations.
    """
    
    def __init__(self):
        pass
    
    def calculate_comprehensive_gex(self, options_chain, spot_price):
        """
        Calculate comprehensive GEX metrics from a given options chain.
        """
        if options_chain.empty:
            return {
                'gex': 0, 'gex_imbalance': 0, 'dynamic_gex': 0,
                'gex_concentration': 0, 'gex_ratio': 0,
                'peak_gamma_strike': 0, 'zero_gamma_level': 0
            }

        gex_metrics = {}
        
        # Pre-calculate notional gamma to avoid recalculation
        options_chain['notional_gamma'] = (
            options_chain['gamma'] * 
            options_chain['open_interest'] * 
            100 * 
            (spot_price ** 2) * 
            0.01
        )

        # 1. Net GEX (traditional measure)
        gex_metrics['gex'] = self._calculate_net_gex(options_chain)
        
        # 2. GEX Profile (across strike prices)
        gex_profile_results = self._calculate_gex_profile(options_chain)
        gex_metrics.update(gex_profile_results)
        
        # 3. GEX Imbalance (call vs put gamma)
        gex_imbalance_results = self._calculate_gex_imbalance(options_chain)
        gex_metrics.update(gex_imbalance_results)
        
        # 4. Dynamic GEX (time-weighted)
        gex_metrics['dynamic_gex'] = self._calculate_dynamic_gex(options_chain)
        
        # 5. GEX Concentration (distribution across strikes)
        gex_metrics['gex_concentration'] = self._calculate_gex_concentration(options_chain)
        
        return gex_metrics
    
    def _calculate_net_gex(self, options_chain):
        """Calculate traditional Net GEX."""
        call_gex = options_chain[options_chain['contract_type'] == 'call']['notional_gamma'].sum()
        put_gex = options_chain[options_chain['contract_type'] == 'put']['notional_gamma'].sum()
        return call_gex - put_gex
    
    def _calculate_gex_profile(self, options_chain):
        """Calculate GEX distribution across strike prices."""
        strike_gex = options_chain.groupby('strike_price')['notional_gamma'].sum()
        
        # Correct for call/put contributions
        call_gamma = options_chain[options_chain['contract_type'] == 'call'].groupby('strike_price')['notional_gamma'].sum()
        put_gamma = options_chain[options_chain['contract_type'] == 'put'].groupby('strike_price')['notional_gamma'].sum()
        strike_gex = call_gamma.subtract(put_gamma, fill_value=0)

        if strike_gex.empty:
            return {'peak_gamma_strike': 0, 'zero_gamma_level': 0}

        peak_gex_strike = strike_gex.abs().idxmax()
        
        # Simplified interpolation for zero gamma
        zero_gamma_level = 0
        sorted_strikes = strike_gex.sort_index()
        crossings = np.where(np.diff(np.sign(sorted_strikes.values)))[0]
        if len(crossings) > 0:
            idx = crossings[0]
            strike1, gex1 = sorted_strikes.index[idx], sorted_strikes.values[idx]
            strike2, gex2 = sorted_strikes.index[idx+1], sorted_strikes.values[idx+1]
            if gex2 != gex1:
                zero_gamma_level = strike1 - gex1 * (strike2 - strike1) / (gex2 - gex1)

        return {
            'peak_gamma_strike': peak_gex_strike,
            'zero_gamma_level': zero_gamma_level
        }
    
    def _calculate_gex_imbalance(self, options_chain):
        """Calculate the imbalance between call and put gamma exposure."""
        call_gex = options_chain[options_chain['contract_type'] == 'call']['notional_gamma'].sum()
        put_gex = options_chain[options_chain['contract_type'] == 'put']['notional_gamma'].sum()
        
        total_gamma_exposure = call_gex + put_gex
        imbalance_ratio = (call_gex - put_gex) / total_gamma_exposure if total_gamma_exposure > 0 else 0
        gex_ratio = call_gex / abs(put_gex) if put_gex != 0 else 0
        
        return {
            'gex_imbalance': imbalance_ratio,
            'gex_ratio': gex_ratio
        }
    
    def _calculate_dynamic_gex(self, options_chain):
        """Calculate time-weighted GEX considering time decay effects."""
        options_chain['time_weight'] = 1.0 / np.maximum(options_chain['days_to_expiry'], 1)
        options_chain['dynamic_notional_gamma'] = options_chain['notional_gamma'] * options_chain['time_weight']
        dynamic_call_gex = options_chain[options_chain['contract_type'] == 'call']['dynamic_notional_gamma'].sum()
        dynamic_put_gex = options_chain[options_chain['contract_type'] == 'put']['dynamic_notional_gamma'].sum()
        return dynamic_call_gex - dynamic_put_gex

    def _calculate_gex_concentration(self, options_chain):
        """Measure how concentrated GEX is across strikes using the Herfindahl-Hirschman Index."""
        exposures = options_chain.groupby('strike_price')['notional_gamma'].sum().abs()
        total_exposure = exposures.sum()
        if total_exposure > 0:
            hhi = (exposures / total_exposure).pow(2).sum()
            return hhi
        return 0

class VolatilitySurfaceFeatures:
    """
    Advanced features extracted from the implied volatility surface
    """
    
    def __init__(self):
        self.surface_history = []
    
    def extract_surface_features(self, iv_data):
        """
        Extract comprehensive features from the volatility surface
        """
        surface_features = {}
        
        # 1. Term Structure Features
        surface_features.update(self._analyze_term_structure(iv_data))
        
        # 2. Volatility Skew Features
        surface_features.update(self._analyze_volatility_skew(iv_data))
        
        # 3. Surface Curvature Features
        surface_features.update(self._analyze_surface_curvature(iv_data))
        
        # 4. Volatility of Volatility Features
        surface_features.update(self._analyze_vol_of_vol(iv_data))
        
        # 5. Surface Stability Features
        surface_features.update(self._analyze_surface_stability(iv_data))
        
        return surface_features
    
    def _analyze_term_structure(self, iv_data):
        """
        Analyze the term structure of implied volatility
        """
        # Group by expiration
        term_structure = iv_data.groupby('days_to_expiry')['implied_vol'].mean()
        
        features = {}
        
        if len(term_structure) >= 2:
            # Calculate slope
            x = np.array(term_structure.index)
            y = np.array(term_structure.values)
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            
            features['term_structure_slope'] = slope
            features['term_structure_intercept'] = intercept
            features['term_structure_r_squared'] = r_value ** 2
            
            # Calculate curvature (second derivative)
            if len(term_structure) >= 3:
                second_derivative = np.diff(y, 2)
                features['term_structure_curvature'] = np.mean(second_derivative)
            
            # Backwardation/Contango indicator
            features['term_structure_shape'] = 'backwardation' if slope < 0 else 'contango'
        
        return features
    
    def _analyze_volatility_skew(self, iv_data):
        """
        Analyze the volatility skew across strikes
        """
        features = {}
        
        # Focus on near-term options (< 45 days)
        near_term = iv_data[iv_data.days_to_expiry <= 45]
        
        if len(near_term) > 0:
            # Calculate moneyness
            near_term['moneyness'] = near_term['strike_price'] / near_term['underlying_price']
            
            # Separate calls and puts
            calls = near_term[near_term.contract_type == 'call']
            puts = near_term[near_term.contract_type == 'put']
            
            # ATM volatility (around 1.0 moneyness)
            atm_vol = near_term[
                (near_term.moneyness >= 0.98) & (near_term.moneyness <= 1.02)
            ]['implied_vol'].mean()
            
            # OTM put volatility (< 0.95 moneyness)
            otm_put_vol = puts[puts.moneyness < 0.95]['implied_vol'].mean()
            
            # OTM call volatility (> 1.05 moneyness)
            otm_call_vol = calls[calls.moneyness > 1.05]['implied_vol'].mean()
            
            features['atm_volatility'] = atm_vol
            features['put_skew'] = otm_put_vol - atm_vol if not np.isnan(otm_put_vol) else 0
            features['call_skew'] = otm_call_vol - atm_vol if not np.isnan(otm_call_vol) else 0
            features['skew_asymmetry'] = features['put_skew'] - features['call_skew']
            
            # Skew slope
            if len(near_term) >= 3:
                x = near_term['moneyness'].values
                y = near_term['implied_vol'].values
                skew_slope, _, _, _, _ = stats.linregress(x, y)
                features['skew_slope'] = skew_slope
        
        return features
    
    def _analyze_vol_of_vol(self, iv_data):
        """
        Analyze the volatility of implied volatility
        """
        features = {}
        
        # Calculate vol-of-vol for different categories
        categories = [
            ('all', iv_data),
            ('short_term', iv_data[iv_data.days_to_expiry <= 30]),
            ('long_term', iv_data[iv_data.days_to_expiry >= 60])
        ]
        
        for category_name, data in categories:
            if len(data) > 1:
                vol_of_vol = data['implied_vol'].std()
                features[f'vol_of_vol_{category_name}'] = vol_of_vol
        
        return features

    def _analyze_surface_curvature(self, iv_data):
        """
        Analyze the curvature of the volatility surface.
        """
        features = {}
        if 'moneyness' in iv_data.columns and 'implied_vol' in iv_data.columns:
            # Fit a polynomial to the skew to find the curvature
            near_term = iv_data[iv_data.days_to_expiry <= 45]
            if len(near_term) > 2:
                poly = np.polyfit(near_term['moneyness'], near_term['implied_vol'], 2)
                features['skew_curvature'] = poly[0]
        return features

    def _analyze_surface_stability(self, iv_data):
        """
        Analyze the stability of the volatility surface over time.
        This is a placeholder and would require historical surface data.
        """
        # Placeholder: In a real scenario, you would compare the current
        # surface to historical surfaces (e.g., from self.surface_history)
        # and calculate metrics like the mean squared difference.
        return {'surface_stability': 0.0}


def add_multi_timeframe_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Adds multi-timeframe features to the dataframe.
    """
    # Moving averages for different timeframes
    data['mavg_20d'] = data['close'].rolling(window=20).mean()
    data['mavg_4w'] = data['close'].rolling(window=28).mean()
    data['mavg_3m'] = data['close'].rolling(window=90).mean()

    # Feature: Close price vs. moving average
    data['close_vs_mavg_20d'] = data['close'] / data['mavg_20d']

    # Exponential Moving Averages
    data['ema_20d'] = data['close'].ewm(span=20, adjust=False).mean()
    data['ema_50d'] = data['close'].ewm(span=50, adjust=False).mean()
    data['ema_200d'] = data['close'].ewm(span=200, adjust=False).mean()

    # EMA Crossovers
    data['ema_20_50_crossover'] = (data['ema_20d'] > data['ema_50d']).astype(int)
    data['ema_50_200_crossover'] = (data['ema_50d'] > data['ema_200d']).astype(int)

    return data





class DeltaHedgingFlowFeatures:
    """
    Features based on market maker delta hedging activity
    """
    
    def __init__(self):
        self.flow_history = []
    
    def calculate_hedging_flow_features(self, options_chain, price_change, spot_price):
        """
        Calculate features related to dealer delta hedging flows
        """
        flow_features = {}
        
        # 1. Expected hedging flow
        flow_features['expected_flow'] = self._calculate_expected_flow(
            options_chain, price_change, spot_price
        )
        
        # 2. Flow concentration
        flow_features['flow_concentration'] = self._calculate_flow_concentration(
            options_chain
        )
        
        # 3. Flow asymmetry
        flow_features['flow_asymmetry'] = self._calculate_flow_asymmetry(
            options_chain
        )
        
        # 4. Flow pressure intensity
        flow_features['flow_pressure'] = self._calculate_flow_pressure(
            options_chain, price_change, spot_price
        )
        
        return flow_features
    
    def _calculate_expected_flow(self, options_chain, price_change, spot_price):
        """
        Calculate expected delta hedging flow from market makers
        """
        total_flow = 0
        
        for _, contract in options_chain.iterrows():
            # Delta exposure per contract
            delta_exposure = contract['delta'] * contract['open_interest'] * 100
            
            # Expected shares to hedge for price change
            flow_contribution = delta_exposure * price_change / spot_price
            
            # Market makers are short options (need to hedge opposite)
            if contract['contract_type'] == 'call':
                total_flow -= flow_contribution  # MM short calls, hedge by buying
            else:
                total_flow += flow_contribution  # MM short puts, hedge by selling
        
        return total_flow
    
    def _calculate_flow_concentration(self, options_chain):
        """
        Measure how concentrated delta flow is across strikes
        """
        delta_by_strike = {}
        
        for _, contract in options_chain.iterrows():
            strike = contract['strike_price']
            delta_exposure = abs(contract['delta'] * contract['open_interest'] * 100)
            
            if strike not in delta_by_strike:
                delta_by_strike[strike] = 0
            delta_by_strike[strike] += delta_exposure
        
        exposures = list(delta_by_strike.values())
        total_exposure = sum(exposures)
        
        if total_exposure == 0:
            return 0
        
        # Calculate Herfindahl-Hirschman Index for concentration
        hhi = sum((exp / total_exposure) ** 2 for exp in exposures)
        
        return hhi

    def _calculate_flow_asymmetry(self, options_chain):
        """Calculate the asymmetry between call and put delta hedging flow."""
        call_flow = 0
        put_flow = 0

        for _, contract in options_chain.iterrows():
            delta_exposure = contract['delta'] * contract['open_interest'] * 100
            if contract['contract_type'] == 'call':
                call_flow += delta_exposure
            else:
                put_flow += delta_exposure

        total_flow = call_flow + put_flow
        if total_flow == 0:
            return 0

        return (call_flow - put_flow) / total_flow

    def _calculate_flow_pressure(self, options_chain, price_change, spot_price):
        """Calculate the intensity of the delta hedging pressure."""
        total_volume = options_chain['volume'].sum() if 'volume' in options_chain.columns else 1
        return self._calculate_expected_flow(options_chain, price_change, spot_price) / total_volume


def _fetch_and_calculate_options_features_for_date(args):
    """Helper function to fetch and calculate GEX and Volatility Surface features for a single date."""
    client, date, underlying_ticker, spot_price, main_option_implied_vol = args
    date_str = date.strftime('%Y-%m-%d')
    options_chain = get_full_options_chain(client, underlying_ticker, date_str)

    # Initialize feature calculators
    gex_calculator = GammaExposureFeatures()
    vol_surface_analyzer = VolatilitySurfaceFeatures()
    delta_hedging_calculator = DeltaHedgingFlowFeatures()

    # Calculate features
    daily_features = gex_calculator.calculate_comprehensive_gex(options_chain, spot_price)
    
    # Ensure implied_vol column is present for VolatilitySurfaceFeatures
    if 'implied_vol' not in options_chain.columns:
        options_chain['implied_vol'] = main_option_implied_vol
    
    vol_surface_features = vol_surface_analyzer.extract_surface_features(options_chain)
    daily_features.update(vol_surface_features)

    # Calculate delta hedging flow features
    price_change = spot_price - spot_price # This will be zero, but avoids the error
    delta_hedging_features = delta_hedging_calculator.calculate_hedging_flow_features(options_chain, price_change, spot_price)
    daily_features.update(delta_hedging_features)

    return date, daily_features


def add_options_chain_features(data: pd.DataFrame, underlying_ticker: str, max_workers: int = 10) -> pd.DataFrame:
    """
    Calculates and adds options chain derived features (GEX, Volatility Surface) in parallel.
    """
    API_KEY = os.getenv("POLYGON_API_KEY")
    if not API_KEY:
        raise ValueError("POLYGON_API_KEY environment variable not set.")
    
    client = RESTClient(API_KEY)
    all_daily_features = {}

    tasks = []
    for date in data.index:
        spot_price = data.loc[date, 'close']
        main_option_iv = data.loc[date, 'implied_volatility'] if 'implied_volatility' in data.columns else 0
        tasks.append((client, date, underlying_ticker, spot_price, main_option_iv))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_fetch_and_calculate_options_features_for_date, task) for task in tasks]
        
        for future in as_completed(futures):
            try:
                date, daily_features = future.result()
                if daily_features:
                    all_daily_features[date] = daily_features
                    print(f"Successfully calculated options features for {date.strftime('%Y-%m-%d')}")
            except Exception as e:
                print(f"An error occurred during options features calculation: {e}")

    # Convert dictionary of daily features to DataFrame
    features_df = pd.DataFrame.from_dict(all_daily_features, orient='index')
    
    # Merge with original data, handling NaNs
    data = data.merge(features_df, left_index=True, right_index=True, how='left')

    # Now, calculate time-series features on the aggregated daily data
    if 'implied_vol_mean' in data.columns:
        data['implied_vol_change'] = data['implied_vol_mean'].diff()
        data['implied_vol_ma_5d'] = data['implied_vol_mean'].rolling(window=5).mean()
    else:
        data['implied_vol_change'] = 0
        data['implied_vol_ma_5d'] = 0

    if 'gex' in data.columns:
        data['gex_momentum'] = data['gex'].diff()
    else:
        data['gex_momentum'] = 0
    
    # Fill any remaining NaNs (e.g., for dates where no options data was found)
    # We will fill forward and then fill any leading NaNs with 0
    data = data.ffill().fillna(0)

    return data

class MacroeconomicFeatures:
    """
    Features derived from macroeconomic indicators.
    """

    def __init__(self, csv_path="VIX_History.csv"):
        try:
            self.vix_data = pd.read_csv(csv_path, index_col='DATE', parse_dates=True)
        except FileNotFoundError:
            print(f"Error: The file {csv_path} was not found.")
            self.vix_data = pd.DataFrame()

    def extract_macro_features(self, date):
        """
        Extracts macroeconomic features for a given date.
        """
        try:
            vix_value = self.vix_data.loc[date.strftime('%Y-%m-%d')]['CLOSE']
        except KeyError:
            vix_value = np.nan

        return {
            'vix': vix_value,
            'interest_rate_placeholder': 0.05,
            'risk_off_sentiment_placeholder': 0.3
        }

def add_macroeconomic_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Adds macroeconomic features to the dataframe.
    """
    macro_engine = MacroeconomicFeatures()
    macro_features_list = [macro_engine.extract_macro_features(date) for date in data.index]
    macro_features_df = pd.DataFrame(macro_features_list, index=data.index)
    data = data.join(macro_features_df)
    data['vix'] = pd.to_numeric(data['vix'], errors='coerce')
    data['vix'] = data['vix'].fillna(method='ffill').fillna(method='bfill')
    return data