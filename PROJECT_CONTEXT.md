# Project Context & Session Summary

This document serves as a living summary of the project to ensure continuity between sessions.

## 1. Project Goal

To research, design, and build a quantitative trading system to generate profitable buy signals for options contracts.

## 2. Core Philosophy: The "Two-Stack" Approach

After initial planning, we made a critical strategic decision to split the project into two distinct phases and technology stacks:

1.  **The Research Stack (Current Phase):** Optimized for rapid, local-first experimentation to find a profitable strategy ("alpha"). The primary goal is speed of iteration.
2.  **The Production Stack (Future Phase):** A full-scale, production-grade MLOps architecture (Kafka, Kubeflow, etc.) to deploy a *proven* alpha with high reliability and low latency.

We are currently in the **Research Phase**.

## 3. Key Accomplishments & Timeline

1.  **Initial Design & Research:** We began with broad research into options trading, technical indicators, and AI applications. This led to an initial design for a microservices-based system.
2.  **Advanced Strategy Refinement:** Through a collaborative process, we significantly upgraded the design to include professional-grade concepts:
    -   **Advanced Features:** Multi-timeframe analysis, options-specific features (Greeks, IV), and alternative data.
    -   **Sophisticated Training:** Implementation of the **Triple Barrier Method** for labeling and plans for advanced validation techniques (Walk-Forward, Purging & Embargoing).
3.  **Data Provider Vetting:** We investigated several data providers. We first considered Alpha Vantage but discovered it did not provide the necessary options data. We then successfully vetted **Polygon.io** and selected their "Options Starter" plan as the ideal data source for our research phase.
4.  **Backtesting Engine Development (The Bug Hunt):** This was the most challenging phase. We built a `quant_lab` library with a walk-forward backtesting engine. We encountered a persistent "No trades were executed" bug due to subtle pandas indexing issues. After several failed attempts at patching the complex engine, we made the strategic decision to **radically simplify the design**. The final, working architecture consists of a simple "calculator" engine and an explicit walk-forward loop in the main script. This fixed the bug and provided our first reliable backtest results.
5.  **Alpha Discovery:** We started with a simple moving average strategy (negative Sharpe). We then added multi-timeframe features and finally more sophisticated interaction features (`iv_x_volume`, `close_vs_mavg_20d`, etc.). This process led to our first **positive Sharpe Ratio (0.76)**, validating that our feature engineering approach is on the right track.
6.  **Scientific Workflow:** We successfully integrated **MLflow** into our main script, allowing us to log experiment parameters and metrics. This transitions us from ad-hoc scripting to a reproducible, scientific research process.
7.  **Debugging Volatility Filter:** We encountered an `IndexingError` when applying the volatility filter. The root cause was an index misalignment between the `volatility` Series (calculated on `train_data`) and the `signals` Series (generated for `test_data`). The fix involved calculating volatility on the `test_data` and explicitly reindexing it to align with the signals.
8.  **Research Sprint 7: Multi-Timeframe Fractal Analysis Features**
    *   **Objective:** Explore Multi-Timeframe Fractal Analysis features to capture market structure across different time scales, aiming to improve the strategy's risk-adjusted returns.
    *   **Actions Taken:**
        *   Implemented `FractalAnalysisFeatures` class and integrated it into `add_multi_timeframe_features`.
        *   Fixed `TypeError` in `data_loader.py` related to JSON serialization of Timestamps for caching.
        *   Added robustness to `data_loader.py` for empty/corrupted cache files.
        *   Fixed `UnicodeEncodeError` in `main.py` by setting `PYTHONIOENCODING=UTF-8`.
    *   **Results:**
        *   **Sharpe Ratio:** 0.05
        *   **Maximum Drawdown:** -35.04%
    *   **Analysis & Next Steps:** The fractal analysis features, as implemented, degraded performance. We reverted these changes.
9.  **Research Sprint 8: Moving Average Refinement & GEX Feature Exploration**
    *   **Objective:** Improve the strategy's risk profile by refining moving average features and exploring more sophisticated GEX-based features.
    *   **Actions Taken:**
        *   **Removed Non-Contributing Volatility Features:** Identified and removed volatility surface features from the model (Sharpe Ratio: 1.24, Max Drawdown: -11.01%).
        *   **Added Exponential Moving Averages (EMAs) & Crossovers:** Introduced 20-day, 50-day, and 200-day EMAs, along with EMA crossover features. This significantly improved performance.
        *   **Explored GEX Ratio:** Added `gex_ratio` feature. Performance slightly decreased (Sharpe Ratio: 0.96, Max Drawdown: -12.36%).
        *   **Explored GEX Profile (Peak Gamma & Zero Gamma):** Added `peak_gamma_strike` and `zero_gamma_level` features. Performance slightly decreased (Sharpe Ratio: 0.84, Max Drawdown: -12.36%).
        *   **Removed Non-Contributing GEX Features:** Removed `gex_ratio`, `peak_gamma_strike`, and `zero_gamma_level` to simplify the model, returning to our best baseline.
        *   **Attempted Hyperparameter Tuning (XGBoost):** Re-implemented `RandomizedSearchCV` with `n_iter=10` and then `n_iter=50`. Neither attempt improved performance over the untuned model. These changes were reverted.
        *   **Explored Placeholder Unusual Options Activity (UOA) Features:** Added placeholder features for UOA. Performance was unchanged (Sharpe Ratio: 1.24, Max Drawdown: -11.01%). These features were removed.
        *   **Explored Placeholder Sentiment Analysis Features:** Added placeholder features for news, social, and combined sentiment. Performance slightly decreased (Sharpe Ratio: 1.01, Max Drawdown: -10.47%). These features were removed.
    *   **Current Best Performance (after reverting non-contributing features):**
        *   **Sharpe Ratio:** 1.24
        *   **Maximum Drawdown:** -11.01%
        *   **Total Return:** 121.42%
10. **Research Sprint 9: Macroeconomic Features & Simplified Volatility Surface Features**
    *   **Objective:** Explore the impact of macroeconomic indicators on the options trading strategy and ensure consistent volatility features.
    *   **Actions Taken:**
        *   Implemented `MacroeconomicFeatures` class with placeholder data.
        *   Integrated macroeconomic features into `main.py`.
        *   Fixed `UnicodeEncodeError` by setting `PYTHONIOENCODING=UTF-8`.
        *   Attempted to fetch real VIX data using `yfinance` and `Alpha Vantage`, but encountered network/timeout issues.
        *   Provided a `download_vix.py` script to download VIX data locally.
        *   Modified `MacroeconomicFeatures` to read VIX data from `VIX_History.csv`.
        *   Modified `_fetch_and_calculate_options_features_for_date` to pass `implied_volatility` of the main option to `VolatilitySurfaceFeatures`.
        *   Simplified `VolatilitySurfaceFeatures` to calculate features (mean, std, change, MA) solely from the main option's `implied_volatility` to ensure consistent data availability.
        *   Updated `feature_columns` in `main.py` to use the new simplified volatility features.
    *   **Results:**
        *   **Sharpe Ratio:** 0.35 (after integrating real VIX and simplified IV features)
        *   **Maximum Drawdown:** -10.47%
        *   **Total Return:** 103.56%
    *   **Analysis:** The Sharpe Ratio improved from 0.14 to 0.35 with the integration of real VIX data and simplified implied volatility features. However, the Sharpe Ratio is still low, and the model's performance needs significant improvement.

11. **Research Sprint 11: Two-Model Strategy Implementation**
    *   **Objective:** To create two specialized models for different trading horizons and to establish a new baseline performance.
    *   **Actions Taken:**
        *   **Created Two-Model Structure:** Created `main_long_term.py` and `main_short_term.py` to house the two distinct strategies.
        *   **Pivoted Labeling Strategy:** Switched from using the option's price to the underlying stock's price (SPY) for generating triple-barrier labels. This was a critical change that resolved the issue of models not training due to a lack of diverse outcomes.
        *   **Configured Long-Term Model:** Set the `time_horizon` to 30 days, increased the `TRAIN_PERIOD_DAYS` to 252, and adjusted the triple-barrier parameters to be more suitable for a longer-term strategy.
        *   **Configured Short-Term Model:** Set the `time_horizon` to 2 days to focus on short-term price movements.
    *   **Results:**
        *   **Long-Term Model Sharpe Ratio:** 0.71
        *   **Short-Term Model Sharpe Ratio:** 0.00
    *   **Analysis & Next Steps:** The long-term model shows a strong initial performance. The short-term model, while not yet profitable, is now correctly structured for further development. The next step is to enhance both models by implementing more advanced, options-native features.

12. **Research Sprint 12: Advanced GEX and Delta-Hedging Flow Features**
    *   **Objective:** To improve the performance of both models by implementing more sophisticated, options-native features.
    *   **Actions Taken:**
        *   Refactored the GEX feature calculations into a dedicated `GammaExposureFeatures` class.
        *   Added a `DeltaHedgingFlowFeatures` class to calculate features related to dealer delta hedging flows.
        *   Integrated the new GEX and delta-hedging flow features into the main data processing pipeline.
        *   Adjusted the `profit_target` for the short-term model to a more realistic value (0.027) based on historical data analysis.
        *   Fixed a `KeyError: 'days_to_expiry'` bug in the `data_loader.py` by calculating the `days_to_expiry` column.
        *   Fixed a `UnicodeEncodeError` in the `docker-compose.yml` file by setting the `PYTHONIOENCODING` environment variable to `UTF-8`.
    *   **Results:**
        *   **Long-Term Model Sharpe Ratio:** 0.71
        *   **Short-Term Model Sharpe Ratio:** -0.52
    *   **Analysis & Next Steps:** The long-term model's performance remained the same. The short-term model, while still not profitable, is now generating trades and has a more balanced set of labels for the model to train on. The next step is to implement more advanced volatility features to further improve the short-term model.

13. **Research Sprint 13: MLflow and Backtesting Stability**
    *   **Objective:** To stabilize the MLflow and backtesting workflow to ensure reliable experiment tracking and analysis.
    *   **Actions Taken:**
        *   **Resolved `UnicodeEncodeError`:** Reconfigured `sys.stdout` to use UTF-8 encoding in the backtesting scripts, which resolved the `UnicodeEncodeError` that was crashing the script and preventing MLflow from saving run data.
        *   **Fixed MLflow Networking:** Corrected the MLflow container's `default-artifact-root` to use an absolute path and explicitly set the `MLFLOW_TRACKING_URI` in the backtesting scripts to ensure a stable connection to the MLflow server.
        *   **Diagnosed Empty `mlruns` Directory:** Determined that the `mlruns` directory was empty because the script was not generating any artifacts, and that all parameters and metrics were being correctly logged to the PostgreSQL backend.
        *   **Enabled MLflow CLI:** Successfully used `docker-compose exec` to run MLflow CLI commands from within the MLflow container, allowing for direct interaction with the MLflow server.
    *   **Current Blocker:** The backtesting script is now failing with a `getaddrinfo failed` error when trying to download data from Alpha Vantage. This is a new, likely transient, network error.

## 4. Current State of the Codebase

-   **`main_long_term.py` / `main_short_term.py`:** The primary entry points for our two new strategies.
-   **`src/quant_lab/`:** Our core library.
    -   `engine.py`: A simple "calculator" that takes prices and signals and computes metrics.
    -   `data_loader.py`: Fetches historical stock data from Alpha Vantage and option data from Polygon.io.
    -   `feature_engineering.py`: Contains functions to create new features, including multi-timeframe, GEX, and simplified volatility features.
    -   `metrics.py`: Calculates Sharpe Ratio, Max Drawdown, etc.
    -   `labeling.py`: Implements the Triple Barrier Method.
-   **`docker-compose.yml`:** Manages the local MLflow and PostgreSQL services.
-   **`.env` file:** Used for storing the `POLYGON_API_KEY` and `ALPHA_VANTAGE_API_KEY`.
-   **`VIX_History.csv`:** Local VIX data file.

## 5. Next Research Sprint: Advanced Volatility Features

*   **Objective:** To significantly improve the performance of the short-term model by implementing more sophisticated, options-native volatility features.
*   **Plan:**
    1.  Replace the current simple `VolatilitySurfaceFeatures` class in `src/quant_lab/feature_engineering.py` with the advanced version from `feature_engineering_breakthroughs.md`.
    2.  Integrate the new, more detailed volatility features into the main data processing pipeline.
    3.  Update the feature lists in both `main_long_term.py` and `main_short_term.py` to include these new features.
    4.  Run the backtests for both models to evaluate the impact of these new features.
