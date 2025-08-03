

import pandas as pd
import numpy as np
import xgboost as xgb
import mlflow
from dotenv import load_dotenv
from src.quant_lab.data_loader import get_historical_option_data
from src.quant_lab.feature_engineering import add_multi_timeframe_features, add_options_chain_features, add_macroeconomic_features
from src.quant_lab.labeling import get_triple_barrier_labels
from src.quant_lab.engine import BacktestEngine
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint

load_dotenv()  # Load environment variables from .env file

def train_and_predict(train_data: pd.DataFrame, test_data: pd.DataFrame, triple_barrier_params: dict):
    """
    Trains a model on train_data and returns predictions for test_data.
    """
    feature_columns = [
        'mavg_20d', 
        'gex',
        'ema_20d',
        'ema_20_50_crossover', 'ema_50_200_crossover',
        'vix',
        'gex_ratio', 'peak_gamma_strike', 'zero_gamma_level',
        'implied_vol_std', 'implied_vol_change'
    ]
    
    # 1. Generate Triple Barrier Labels for the training data
    labels = get_triple_barrier_labels(
        prices=train_data['close'], 
        profit_target=triple_barrier_params['profit_target'],
        stop_loss=triple_barrier_params['stop_loss'],
        time_horizon=triple_barrier_params['time_horizon']
    )
    
    # Align features and labels
    X_train = train_data.loc[labels.index, feature_columns]
    y_train = labels

    # Map labels for XGBoost
    y_train_mapped = y_train.map({-1: 0, 0: 1, 1: 2})

    # 2. Train a multi-class XGBoost model
    model = xgb.XGBClassifier(objective='multi:softprob', num_class=3, eval_metric='mlogloss', enable_categorical=True)
    model.fit(X_train, y_train_mapped)

    # Get feature importances
    feature_importances = pd.Series(model.feature_importances_, index=feature_columns)

    # 3. Generate predictions on the out-of-sample test data
    X_test = test_data[feature_columns]
    predictions_mapped = model.predict(X_test)
    predictions = pd.Series(predictions_mapped, index=test_data.index).map({0: -1, 1: 0, 2: 1})

    # 4. Generate final signals based on model prediction and volatility filter
    # For this strategy, we only take a long position if the model predicts a "win" (1)
    signals = predictions.apply(lambda x: 1 if x == 1 else 0)

    # Apply volatility filter
    # We need to calculate volatility on the test data to align indices
    volatility = test_data['close'].pct_change().rolling(window=20).std()
    # Reindex to align with signals, and fill initial NaNs
    aligned_volatility = volatility.reindex(signals.index, method='bfill').fillna(0)

    # --- Volatility Filter Debug ---
    if not aligned_volatility.index.equals(signals.index):
        print("WARNING: Index mismatch between aligned_volatility and signals!")
    # --- End Debug ---

    signals[aligned_volatility > triple_barrier_params['volatility_threshold']] = 0 # Do not trade if volatility is too high

    return signals, feature_importances

if __name__ == "__main__":
    # --- MLflow Configuration ---
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("Options Signal Generation")

    with mlflow.start_run(run_name="Volatility Surface Feature Experiment - Reverted Tuning"):
        # --- Configuration ---
        UNDERLYING_TICKER = "SPY"
        OPTION_TICKER = "O:SPY251219C00500000"
        START_DATE = "2022-01-01"
        END_DATE = "2023-12-31"
        TRAIN_PERIOD_DAYS = 120
        TEST_PERIOD_DAYS = 30
        TRIPLE_BARRIER_PARAMS = {
            "profit_target": 0.10,
            "stop_loss": 0.05,
            "time_horizon": 10,
            "volatility_threshold": 0.02 # Max 2% daily volatility over last 20 days
        }

        # Log parameters
        mlflow.log_params(TRIPLE_BARRIER_PARAMS)
        mlflow.log_param("training_days", TRAIN_PERIOD_DAYS)
        mlflow.log_param("testing_days", TEST_PERIOD_DAYS)
        mlflow.log_param("underlying_ticker", UNDERLYING_TICKER)

        # --- Data Loading and Feature Engineering ---
        print("Loading and preparing data...")
        raw_data = get_historical_option_data(OPTION_TICKER, START_DATE, END_DATE)
        print(f"Raw data points: {len(raw_data)}")
        print(f"NaNs in raw data: \n{raw_data.isna().sum()}")
        featured_data = add_multi_timeframe_features(raw_data)
        print(f"Data points after multi-timeframe features: {len(featured_data)}")
        print(f"NaNs after multi-timeframe features: \n{featured_data.isna().sum()}")
        featured_data = add_options_chain_features(featured_data, UNDERLYING_TICKER)
        print(f"Data points after options chain features: {len(featured_data)}")
        print(f"NaNs after options chain features: \n{featured_data.isna().sum()}")
        featured_data = add_macroeconomic_features(featured_data)
        print(f"Data points after macroeconomic features: {len(featured_data)}")
        print(f"NaNs after macroeconomic features: \n{featured_data.isna().sum()}")
        featured_data = featured_data.dropna()
        print(f"Date range of final data: {featured_data.index.min().date()} to {featured_data.index.max().date()}")
        print(f"Total data points after feature engineering: {len(featured_data)}")

        # --- Feature Correlation Analysis ---
        print("\n--- Feature Correlation Matrix ---")
        feature_columns = [
            'mavg_20d', 'mavg_4w', 'mavg_3m', 
            'close_vs_mavg_20d', 'gex',
            'ema_20d', 'ema_50d', 'ema_200d',
            'ema_20_50_crossover', 'ema_50_200_crossover',
            'vix',
            'gex_ratio', 'peak_gamma_strike', 'zero_gamma_level',
            'implied_vol_mean', 'implied_vol_std', 'implied_vol_change', 'implied_vol_ma_5d'
        ]
        # Ensure all feature columns exist in the dataframe before calculating correlation
        existing_features = [col for col in feature_columns if col in featured_data.columns]
        correlation_matrix = featured_data[existing_features].corr()
        print(correlation_matrix)
        # --- End Feature Correlation Analysis ---

        # --- Walk-Forward Loop ---
        print("Starting walk-forward validation...")
        all_signals = []
        all_feature_importances = []
        num_folds = (len(featured_data) - TRAIN_PERIOD_DAYS) // TEST_PERIOD_DAYS

        for i in range(num_folds):
            train_end_idx = (i * TEST_PERIOD_DAYS) + TRAIN_PERIOD_DAYS
            test_start_idx = train_end_idx
            test_end_idx = test_start_idx + TEST_PERIOD_DAYS

            train_data = featured_data.iloc[0:train_end_idx]
            test_data_slice = featured_data.iloc[test_start_idx:test_end_idx]

            if test_data_slice.empty:
                continue

            print(f"  Processing Fold {i+1}/{num_folds}...")
            fold_signals, fold_importances = train_and_predict(train_data, test_data_slice, TRIPLE_BARRIER_PARAMS)
            all_signals.append(fold_signals)
            all_feature_importances.append(fold_importances)

        if not all_signals:
            print("No signals were generated.")
        else:
            # --- Final Signal Series and Backtest ---
            final_signals = pd.concat(all_signals)
            
            engine = BacktestEngine()
            results = engine.run(price_data=featured_data['close'], signals=final_signals)

            # --- Feature Importance Analysis ---
            avg_feature_importance = pd.concat(all_feature_importances, axis=1).mean(axis=1)
            avg_feature_importance.sort_values(ascending=False, inplace=True)
            
            # Log feature importances as a dictionary
            mlflow.log_params({"feature_importance_" + k: v for k, v in avg_feature_importance.to_dict().items()})

            # Log metrics
            mlflow.log_metric("sharpe_ratio", results['sharpe_ratio'])
            mlflow.log_metric("max_drawdown", results['max_drawdown'])
            mlflow.log_metric("total_return", results['total_return'])

            # --- Output ---
            print("\n--- Triple Barrier Strategy Results ---")
            print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
            print(f"Maximum Drawdown: {results['max_drawdown']:.2%}")
            print(f"Total Return: {results['total_return']:.2%}")

            print("\n--- Average Feature Importance ---")
            print(avg_feature_importance)
