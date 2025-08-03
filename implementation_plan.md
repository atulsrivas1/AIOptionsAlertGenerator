# Implementation Plan: A Phased Approach to Quantitative Trading

## Part 1: The Research Phase - The Hunt for Alpha

**Objective:** To rapidly iterate and discover a statistically robust and profitable trading strategy using a lean, local-first technology stack.

### Research Sprint 1: Environment Setup & Data Validation

-   **Goal:** Build the foundational tools for research and validate our chosen data provider.
-   **Tasks:**
    -   [x] Initialize Git repository & create `.gitignore`.
    -   [x] Set up a local Python environment with Jupyter, Pandas, etc.
    -   [x] Create and run a `docker-compose.yml` file to stand up **PostgreSQL** and **MLflow** containers.
    -   [x] Acquire an API key for the **Polygon.io "Options Starter"** plan.
    -   [x] Create and execute a `validate_data_source.py` script to fetch a sample options chain using the `polygon-python` library and confirm it contains Greeks and Open Interest.

### Research Sprint 2: Core Backtesting Engine

-   **Goal:** Develop the core backtesting engine with robust validation and labeling.
-   **Tasks:**
    -   [x] Develop the core **Backtesting Engine** as a Python library.
    -   [x] Implement **Walk-Forward Validation**.
    -   [x] Implement the **Triple Barrier Labeling** method.
    -   [x] Implement the comprehensive **Evaluation Metrics Framework** (Financial, Prediction, Risk).
    -   [x] Ensure the engine logs all parameters, code versions, and metrics to the local MLflow server on every run.

### Research Sprint 3: Baseline Model & Feature Engineering

-   **Goal:** Establish a baseline performance and begin feature engineering.
-   **Tasks:**
    -   [x] Create a data-loading module within the backtester to pull data from Polygon.io.
    -   [x] Build a baseline model (e.g., simple XGBoost) on a single timeframe.
    -   [x] Run and log the baseline backtest to MLflow.
    -   [x] Implement feature engineering for **Multi-Timeframe Features**.
    -   [x] Run new backtests with these features and compare results to the baseline in MLflow.

### Research Sprint 4: Advanced Feature Engineering & Modeling

-   **Goal:** Systematically explore more advanced features and models to find a consistent edge.
-   **Tasks (Iterative):**
    -   [x] **Options-Specific Features:** Engineer features from the Greeks and IV data (e.g., GEX).
    -   [x] **Alternative Data:** Integrate and test features from macro indicators (VIX).
    -   [x] **Feature Selection:** Analyze feature correlation and remove redundant features.

### Research Sprint 5: The Two-Model Strategy

-   **Goal:** Create two specialized models for different trading horizons.
-   **Tasks:**
    -   [x] Create `main_long_term.py` and `main_short_term.py` scripts.
    -   [x] Configure the `time_horizon` and feature set for the long-term model (30+ days).
    -   [x] Configure the `time_horizon` and feature set for the short-term model (1-3 days).
    -   [x] Refactor the data loader to use **Alpha Vantage** for historical stock data.
    -   [x] Switched to using the underlying's price for labeling to resolve training issues.

### Research Sprint 6: Advanced GEX and Delta-Hedging Flow Features

-   **Goal:** To significantly improve the performance of both the long-term and short-term models by implementing more sophisticated, options-native features.
-   **Tasks:**
    -   [x] Refactored the GEX feature calculations into a dedicated `GammaExposureFeatures` class.
    -   [x] Added a `DeltaHedgingFlowFeatures` class to calculate features related to dealer delta hedging flows.
    -   [x] Integrated the new GEX and delta-hedging flow features into the main data processing pipeline.
    -   [x] Adjusted the `profit_target` for the short-term model to a more realistic value (0.027) based on historical data analysis.
    -   [x] Fixed a `KeyError: 'days_to_expiry'` bug in the `data_loader.py` by calculating the `days_to_expiry` column.
    -   [x] Fixed a `UnicodeEncodeError` in the `docker-compose.yml` file by setting the `PYTHONIOENCODING` environment variable to `UTF-8`.

### Research Sprint 7: Advanced Volatility Features

-   **Goal:** To significantly improve the performance of the short-term model by implementing more sophisticated, options-native volatility features.
-   **Tasks:**
    -   [ ] Replace the current simple `VolatilitySurfaceFeatures` class in `src/quant_lab/feature_engineering.py` with the advanced version from `feature_engineering_breakthroughs.md`.
    -   [ ] Integrate the new, more detailed volatility features into the main data processing pipeline.
    -   [ ] Update the feature lists in both `main_long_term.py` and `main_short_term.py` to include these new features.
    -   [ ] Run the backtests for both models to evaluate the impact of these new features.

### Research Sprint 8: MLflow and Backtesting Stability

-   **Goal:** To stabilize the MLflow and backtesting workflow to ensure reliable experiment tracking and analysis.
-   **Tasks:**
    -   [x] Resolved `UnicodeEncodeError` in backtesting scripts.
    -   [x] Fixed MLflow networking and container configuration issues.
    -   [x] Established a reliable workflow for running backtests and analyzing results with MLflow.
    -   [ ] **Current Blocker:** Resolve the transient `getaddrinfo failed` error in the data loader.

**Go/No-Go Decision:** We only proceed to Part 2 when we have a model that consistently meets our predefined success criteria (e.g., Sharpe > 1.5, Max Drawdown < 20%) across multiple walk-forward validation periods.

---

## Part 2: The Production Phase - Industrializing the Alpha

**Objective:** To build a scalable, reliable, and secure system to deploy and manage the profitable strategy discovered in Part 1.

### Production Sprint 1: MLOps & Data Pipeline Foundation

-   **Goal:** Set up the production-grade infrastructure backbone on the cloud.
-   **Tasks:**
    -   [ ] Set up a Kubernetes cluster (EKS, GKE, etc.).
    -   [ ] Deploy and configure **Apache Kafka** and **Apache Airflow**.
    -   [ ] Deploy and configure **MLflow** (production instance) and **Feast**.
    -   [ ] Build Airflow DAGs to automate the ingestion of all required data sources into Kafka and a data lake/warehouse.

### Production Sprint 2: Feature Store and Training Pipelines

-   **Goal:** Automate the feature engineering and model training processes.
-   **Tasks:**
    -   [ ] Define and populate the **Feast Feature Repository** using Airflow DAGs.
    -   [ ] Build a **Kubeflow Pipeline** that replicates the training logic of our best model from the research phase.
    -   [ ] The pipeline will pull data from Feast, train the model, and register the validated model in the production MLflow Model Registry.

### Production Sprint 3: High-Performance Inference & Risk Management

-   **Goal:** Deploy the model for live inference and wrap it in a safety layer.
-   **Tasks:**
    -   [ ] Set up **Seldon Core** for model serving.
    -   [ ] Build a CI/CD pipeline to automatically convert, optimize (ONNX), and deploy models from the MLflow Registry to Seldon.
    -   [ ] Implement and deploy the **Risk Management Service**.
    -   [ ] Implement the **Signal Generation Service** to orchestrate inference and risk checks.

### Production Sprint 4: Live Deployment & Monitoring

-   **Goal:** Go live with the full system and establish monitoring.
-   **Tasks:**
    -   [ ] Perform end-to-end testing of the entire production pipeline.
    -   [ ] Deploy the user-facing API and UI.
    -   [ ] Implement comprehensive monitoring and alerting for all system components.
    -   [ ] Begin running the system in a paper-trading mode before committing real capital.