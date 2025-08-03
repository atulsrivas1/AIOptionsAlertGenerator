
# Design Document: A Two-Stack Approach for Quantitative Research and Production Trading

## 1. Core Philosophy: Research First, Then Scale

The primary risk in any quantitative trading project is not infrastructure failure, but the failure to find a profitable, robust, and non-obvious trading strategy (alpha). Therefore, our design is split into two distinct stacks, each optimized for a different purpose:

1.  **The Research Stack:** Optimized for **speed of iteration, experimentation, and discovery**. Its goal is to find alpha quickly and cheaply.
2.  **The Production Stack:** Optimized for **scalability, reliability, low-latency, and security**. Its goal is to deploy and manage a *proven* alpha with real capital.

We will only proceed with building the Production Stack *after* we have validated a profitable strategy with the Research Stack.

---

## 2. The Research Stack: The Alpha Factory

### 2.1. Objective

To enable rapid, rigorous, and reproducible financial research. The entire environment is designed to run on a single powerful machine (local or cloud VM).

### 2.2. Architecture & Technology

-   **Environment:** Local machine or single cloud VM.
-   **Core Tools:** Python, Jupyter Notebooks, Docker Compose.
-   **Data Providers:**
    -   **Historical Stock Data:** **Alpha Vantage** will be used for fetching historical daily OHLCV data for the underlying asset.
    -   **Historical Options Data:** **Polygon.io** (specifically, the "Options Starter" plan) will be the primary source for historical options data, including chains, open interest, and Greeks.
-   **Core Libraries:** `alpha_vantage` and `polygon-python` for data access, `pandas` for data manipulation, `py_vollib` for any custom Greek calculations, and the core ML stack (Scikit-learn, XGBoost, etc.).
-   **Data Storage:**
    -   Local PostgreSQL or InfluxDB instance (managed via Docker Compose) for structured and time-series data.
    -   Simple file system (e.g., Parquet files) for datasets.
-   **Key MLOps Tool:** **MLflow** (managed via Docker Compose). This is the only piece of complex MLOps we use here, and it is non-negotiable. It will track every experiment, parameter, model, and metric.
-   **Backtesting Engine:** A custom Python library built to the following specifications:
    -   **Validation Strategy:** Must implement **Walk-Forward Validation with Purged and Embargoed Cross-Validation**.
    -   **Labeling Strategy:** Must use the **Triple Barrier Method**.
    -   **Evaluation:** Must compute and log the full suite of **Financial, Prediction, and Risk metrics** to MLflow.

### 2.3. The Two-Model Strategy

To capture a wider range of market opportunities, we will develop two specialized models:

1.  **Long-Term Directional Model:**
    *   **Objective:** To identify and capitalize on fundamental market trends over a longer time horizon.
    *   **Time Horizon:** 30+ days.
    *   **Features:** Will focus on features that are less sensitive to daily noise, such as longer-term moving averages and macroeconomic indicators.
    *   **Initial Implementation:** Will be a directional forecasting model for the underlying stock (SPY).

2.  **Short-Term Options-Native Model:**
    *   **Objective:** To exploit short-term volatility, 0DTE opportunities, and other options-specific phenomena.
    *   **Time Horizon:** 1-3 days.
    *   **Features:** Will focus on options-native features like GEX, IV dynamics, and short-term moving averages.
    *   **Evolution:** This model will be evolved to become a true options-aware strategy that can dynamically select the optimal contract and account for volatility and time decay.

### 2.4. Workflow

1.  **Data Collection:** Manually or with simple Python scripts, acquire data from Alpha Vantage and Polygon.io and store it locally.
2.  **Experimentation:** In a Jupyter Notebook or Python script, define a hypothesis for either the long-term or short-term model.
3.  **Feature Engineering:** Create the new features using Pandas/NumPy.
4.  **Training & Backtesting:** Run the backtesting engine for the chosen model. The engine will automatically log all results to the MLflow server.
5.  **Analysis:** Use the MLflow UI to compare the results of the new experiment against previous runs.
6.  **Iteration:** Repeat. The goal is to find a model that meets our predefined success criteria (e.g., Sharpe > 1.5, Max Drawdown < 20%).

---

## 3. The Production Stack: The Money-Making Machine

### 3.1. Objective

To deploy, monitor, and manage a *proven* trading strategy from the Research Stack in a live market. This architecture is designed for high-availability and low-latency.

### 3.2. Architecture & Technology

-   **Orchestration:** Kubernetes (Cloud-managed: EKS, GKE, AKS).
-   **Data Ingestion:**
    -   **Real-time Stock Data:** **Alpaca** will be used for its real-time streaming API.
    -   **Real-time Options Data:** **Polygon.io** will be used for its real-time options data feed.
    -   **Streaming Platform:** **Apache Kafka** for real-time data streams.
    -   **Batch Processing:** **Apache Airflow** for scheduling batch data processing and feature engineering pipelines.
-   **Feature Store:** **Feast** for serving versioned features consistently across training and inference. **Redis** provides the low-latency online store.
-   **ML Pipelines:** **Kubeflow Pipelines** for orchestrating complex, multi-step model training and validation workflows.
-   **Experiment Tracking:** **MLflow** (production instance) for tracking, versioning, and registering production-candidate models.
-   **Distributed Compute:** **Dask** or **Ray** for large-scale, distributed retraining and backtesting jobs.
-   **Model Serving:** **Seldon Core** or **TF Serving** for high-performance, low-latency inference, with models optimized using **ONNX**.
-   **Risk Management:** A dedicated **Risk Management Service** acts as a final gatekeeper for all generated signals.

### 3.3. Workflow

1.  **Model Promotion:** A model that has been validated in the Research Stack is promoted in the MLflow Model Registry.
2.  **CI/CD Pipeline:** This promotion triggers a CI/CD pipeline (e.g., GitOps) that:
    -   Packages the model into a container.
    -   Converts it to ONNX format.
    -   Deploys it to the Seldon Core inference graph on Kubernetes.
3.  **Live Operation:**
    -   The live Kafka stream feeds data into the system.
    -   Airflow DAGs continuously compute and update features in the Feast Feature Store.
    -   The Seldon Core server pulls the latest features from Redis, generates a prediction, and publishes it to another Kafka topic.
    -   The Signal Generation Service consumes the prediction and validates it with the Risk Management Service.
    -   An approved signal is sent to a trader or an automated execution engine.
4.  **Monitoring:** The system is continuously monitored for performance, data drift, and model decay. Airflow automatically triggers retraining pipelines (orchestrated by Kubeflow) on a schedule or when performance degrades.

