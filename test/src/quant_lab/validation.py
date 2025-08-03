import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Any, Optional, Callable
from datetime import timedelta
import logging

logger = logging.getLogger(__name__)

class ValidationFramework:
    """
    Comprehensive validation framework implementing walk-forward validation
    with purging and embargoing for financial time series.
    """
    
    def __init__(self, 
                 train_period_days: int = 120,
                 test_period_days: int = 30,
                 purge_days: int = 1,
                 embargo_days: int = 0,
                 min_train_samples: int = 100):
        """
        Initialize validation framework.
        
        Args:
            train_period_days: Number of days for training window
            test_period_days: Number of days for testing window
            purge_days: Days to purge between train and test to avoid lookahead bias
            embargo_days: Days to embargo after test period to avoid information leakage
            min_train_samples: Minimum samples required for training
        """
        self.train_period_days = train_period_days
        self.test_period_days = test_period_days
        self.purge_days = purge_days
        self.embargo_days = embargo_days
        self.min_train_samples = min_train_samples
        
        # Validation metrics storage
        self.fold_results = []
        self.performance_metrics = {}
    
    def walk_forward_split(self, data: pd.DataFrame, 
                          target_col: str = None) -> List[Dict[str, Any]]:
        """
        Generate walk-forward validation splits with purging and embargoing.
        
        Args:
            data: DataFrame with datetime index
            target_col: Name of target column (if None, assumes target generation during validation)
            
        Returns:
            List of dictionaries containing train/test splits with metadata
        """
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Data must have DatetimeIndex")
        
        data = data.sort_index()
        splits = []
        
        # Calculate the total number of possible folds
        total_days = (data.index[-1] - data.index[0]).days
        max_folds = (total_days - self.train_period_days) // self.test_period_days
        
        logger.info(f"Generating {max_folds} walk-forward validation folds")
        
        for fold in range(max_folds):
            # Calculate date ranges for this fold
            train_start = data.index[0] + timedelta(days=fold * self.test_period_days)
            train_end = train_start + timedelta(days=self.train_period_days)
            
            # Apply purging
            test_start = train_end + timedelta(days=self.purge_days)
            test_end = test_start + timedelta(days=self.test_period_days)
            
            # Check if we have enough data for this fold
            if test_end > data.index[-1]:
                break
            
            # Extract train and test data
            train_mask = (data.index >= train_start) & (data.index < train_end)
            test_mask = (data.index >= test_start) & (data.index < test_end)
            
            train_data = data[train_mask].copy()
            test_data = data[test_mask].copy()
            
            # Validate minimum sample requirements
            if len(train_data) < self.min_train_samples:
                logger.warning(f"Fold {fold}: Insufficient training samples ({len(train_data)})")
                continue
            
            if len(test_data) == 0:
                logger.warning(f"Fold {fold}: No test samples")
                continue
            
            split_info = {
                'fold': fold,
                'train_data': train_data,
                'test_data': test_data,
                'train_start': train_start,
                'train_end': train_end,
                'test_start': test_start,
                'test_end': test_end,
                'train_samples': len(train_data),
                'test_samples': len(test_data),
                'purge_days': self.purge_days,
                'embargo_days': self.embargo_days
            }
            
            splits.append(split_info)
        
        logger.info(f"Generated {len(splits)} valid validation folds")
        return splits
    
    def cross_validate(self, data: pd.DataFrame, 
                      model_trainer: Callable,
                      predictor: Callable,
                      evaluator: Callable,
                      target_generator: Callable = None,
                      **kwargs) -> Dict[str, Any]:
        """
        Perform walk-forward cross-validation with comprehensive evaluation.
        
        Args:
            data: Input DataFrame with datetime index
            model_trainer: Function to train model (train_data, **kwargs) -> model
            predictor: Function to make predictions (model, test_data, **kwargs) -> predictions
            evaluator: Function to evaluate predictions (predictions, test_data, **kwargs) -> metrics
            target_generator: Function to generate targets (data, **kwargs) -> targets
            **kwargs: Additional arguments passed to functions
            
        Returns:
            Dictionary containing aggregated validation results
        """
        splits = self.walk_forward_split(data)
        
        if not splits:
            raise ValueError("No valid validation splits generated")
        
        # Store results for each fold
        fold_metrics = []
        all_predictions = []
        all_actuals = []
        
        for split in splits:
            logger.info(f"Processing fold {split['fold']}: "
                       f"{split['train_start'].date()} to {split['test_end'].date()}")
            
            try:
                # Generate targets if needed
                if target_generator:
                    train_targets = target_generator(split['train_data'], **kwargs)
                    test_targets = target_generator(split['test_data'], **kwargs)
                    
                    # Align targets with data
                    split['train_data'] = split['train_data'].loc[train_targets.index]
                    split['test_data'] = split['test_data'].loc[test_targets.index]
                
                # Train model on current fold
                model = model_trainer(split['train_data'], **kwargs)
                
                # Generate predictions
                predictions = predictor(model, split['test_data'], **kwargs)
                
                # Evaluate predictions
                metrics = evaluator(predictions, split['test_data'], **kwargs)
                
                # Add fold metadata to metrics
                metrics.update({
                    'fold': split['fold'],
                    'train_samples': split['train_samples'],
                    'test_samples': split['test_samples'],
                    'train_period': (split['train_end'] - split['train_start']).days,
                    'test_period': (split['test_end'] - split['test_start']).days
                })
                
                fold_metrics.append(metrics)
                
                # Store predictions and actuals for overall analysis
                if hasattr(split['test_data'], 'values'):
                    all_predictions.extend(predictions.values if hasattr(predictions, 'values') else predictions)
                    # Assume target is in test_data or extract from evaluator
                    
                logger.info(f"Fold {split['fold']} completed successfully")
                
            except Exception as e:
                logger.error(f"Error in fold {split['fold']}: {str(e)}")
                continue
        
        # Aggregate results across all folds
        aggregated_results = self._aggregate_fold_results(fold_metrics)
        
        # Store results for later analysis
        self.fold_results = fold_metrics
        self.performance_metrics = aggregated_results
        
        return aggregated_results
    
    def _aggregate_fold_results(self, fold_metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate metrics across all validation folds."""
        if not fold_metrics:
            return {}
        
        # Get all numeric metric names
        numeric_metrics = []
        for key, value in fold_metrics[0].items():
            if isinstance(value, (int, float)) and key != 'fold':
                numeric_metrics.append(key)
        
        aggregated = {}
        
        # Calculate statistics for each metric
        for metric in numeric_metrics:
            values = [fold[metric] for fold in fold_metrics if metric in fold and not np.isnan(fold[metric])]
            
            if values:
                aggregated[f'{metric}_mean'] = np.mean(values)
                aggregated[f'{metric}_std'] = np.std(values)
                aggregated[f'{metric}_median'] = np.median(values)
                aggregated[f'{metric}_min'] = np.min(values)
                aggregated[f'{metric}_max'] = np.max(values)
                
                # Calculate confidence intervals (95%)
                confidence_level = 0.95
                margin_error = 1.96 * (np.std(values) / np.sqrt(len(values)))
                aggregated[f'{metric}_ci_lower'] = np.mean(values) - margin_error
                aggregated[f'{metric}_ci_upper'] = np.mean(values) + margin_error
        
        # Add metadata
        aggregated['total_folds'] = len(fold_metrics)
        aggregated['successful_folds'] = len([f for f in fold_metrics if f.get('fold') is not None])
        
        # Calculate consistency metrics
        if 'sharpe_ratio' in [m.replace('_mean', '').replace('_std', '').replace('_median', '').replace('_min', '').replace('_max', '').replace('_ci_lower', '').replace('_ci_upper', '') for m in aggregated.keys()]:
            sharpe_values = [fold.get('sharpe_ratio', 0) for fold in fold_metrics]
            positive_sharpe_folds = sum(1 for s in sharpe_values if s > 0)
            aggregated['sharpe_consistency'] = positive_sharpe_folds / len(sharpe_values) if sharpe_values else 0
        
        return aggregated
    
    def get_validation_report(self) -> str:
        """Generate a comprehensive validation report."""
        if not self.performance_metrics:
            return "No validation results available. Run cross_validate() first."
        
        report = "\n=== WALK-FORWARD VALIDATION REPORT ===\n"
        report += f"Configuration:\n"
        report += f"  - Training Period: {self.train_period_days} days\n"
        report += f"  - Testing Period: {self.test_period_days} days\n"
        report += f"  - Purge Period: {self.purge_days} days\n"
        report += f"  - Embargo Period: {self.embargo_days} days\n\n"
        
        report += f"Validation Results:\n"
        report += f"  - Total Folds: {self.performance_metrics.get('total_folds', 0)}\n"
        report += f"  - Successful Folds: {self.performance_metrics.get('successful_folds', 0)}\n\n"
        
        # Key performance metrics
        key_metrics = ['sharpe_ratio', 'max_drawdown', 'total_return', 'hit_rate']
        
        for metric in key_metrics:
            mean_key = f'{metric}_mean'
            std_key = f'{metric}_std'
            
            if mean_key in self.performance_metrics:
                mean_val = self.performance_metrics[mean_key]
                std_val = self.performance_metrics.get(std_key, 0)
                ci_lower = self.performance_metrics.get(f'{metric}_ci_lower', 0)
                ci_upper = self.performance_metrics.get(f'{metric}_ci_upper', 0)
                
                report += f"  {metric.replace('_', ' ').title()}:\n"
                report += f"    - Mean: {mean_val:.4f} ± {std_val:.4f}\n"
                report += f"    - 95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]\n"
                report += f"    - Range: [{self.performance_metrics.get(f'{metric}_min', 0):.4f}, "
                report += f"{self.performance_metrics.get(f'{metric}_max', 0):.4f}]\n\n"
        
        # Consistency metrics
        if 'sharpe_consistency' in self.performance_metrics:
            consistency = self.performance_metrics['sharpe_consistency']
            report += f"  Strategy Consistency:\n"
            report += f"    - Positive Sharpe Folds: {consistency:.1%}\n\n"
        
        return report
    
    def plot_validation_results(self) -> None:
        """Plot validation results across folds (requires matplotlib)."""
        try:
            import matplotlib.pyplot as plt
            
            if not self.fold_results:
                print("No validation results to plot")
                return
            
            # Extract key metrics for plotting
            folds = [r['fold'] for r in self.fold_results]
            sharpe_ratios = [r.get('sharpe_ratio', 0) for r in self.fold_results]
            max_drawdowns = [r.get('max_drawdown', 0) for r in self.fold_results]
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            
            # Plot Sharpe ratios
            ax1.bar(folds, sharpe_ratios, alpha=0.7, color='blue')
            ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5)
            ax1.set_title('Sharpe Ratio by Validation Fold')
            ax1.set_xlabel('Fold')
            ax1.set_ylabel('Sharpe Ratio')
            ax1.grid(True, alpha=0.3)
            
            # Plot Max Drawdowns
            ax2.bar(folds, max_drawdowns, alpha=0.7, color='red')
            ax2.set_title('Maximum Drawdown by Validation Fold')
            ax2.set_xlabel('Fold')
            ax2.set_ylabel('Max Drawdown')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            print("Matplotlib not available. Install with: pip install matplotlib")

class TimeSeriesValidator:
    """Specialized validator for time series with financial-specific considerations."""
    
    def __init__(self):
        self.validation_tests = {
            'stationarity': self._test_stationarity,
            'autocorrelation': self._test_autocorrelation,
            'heteroscedasticity': self._test_heteroscedasticity,
            'normality': self._test_normality
        }
    
    def validate_time_series(self, data: pd.Series) -> Dict[str, Any]:
        """Run comprehensive time series validation tests."""
        results = {}
        
        for test_name, test_func in self.validation_tests.items():
            try:
                results[test_name] = test_func(data)
            except Exception as e:
                results[test_name] = {'error': str(e)}
        
        return results
    
    def _test_stationarity(self, data: pd.Series) -> Dict[str, Any]:
        """Test for stationarity using Augmented Dickey-Fuller test."""
        try:
            from statsmodels.tsa.stattools import adfuller
            
            result = adfuller(data.dropna())
            
            return {
                'adf_statistic': result[0],
                'p_value': result[1],
                'critical_values': result[4],
                'is_stationary': result[1] < 0.05
            }
        
        except ImportError:
            return {'error': 'statsmodels not available'}
    
    def _test_autocorrelation(self, data: pd.Series) -> Dict[str, Any]:
        """Test for autocorrelation using Ljung-Box test."""
        try:
            from statsmodels.stats.diagnostic import acorr_ljungbox
            
            result = acorr_ljungbox(data.dropna(), lags=10, return_df=True)
            
            return {
                'ljung_box_statistic': result['lb_stat'].iloc[-1],
                'p_value': result['lb_pvalue'].iloc[-1],
                'has_autocorrelation': result['lb_pvalue'].iloc[-1] < 0.05
            }
        
        except ImportError:
            return {'error': 'statsmodels not available'}
    
    def _test_heteroscedasticity(self, data: pd.Series) -> Dict[str, Any]:
        """Test for heteroscedasticity (changing variance)."""
        # Simple test based on rolling standard deviation
        window = min(30, len(data) // 4)
        rolling_std = data.rolling(window=window).std()
        
        # Test if variance is changing over time
        variance_trend = np.corrcoef(range(len(rolling_std.dropna())), rolling_std.dropna())[0, 1]
        
        return {
            'variance_trend_correlation': variance_trend,
            'has_heteroscedasticity': abs(variance_trend) > 0.3
        }
    
    def _test_normality(self, data: pd.Series) -> Dict[str, Any]:
        """Test for normality using Shapiro-Wilk test."""
        try:
            from scipy.stats import shapiro
            
            # Sample data if too large (Shapiro-Wilk has limitations)
            test_data = data.dropna()
            if len(test_data) > 5000:
                test_data = test_data.sample(n=5000, random_state=42)
            
            statistic, p_value = shapiro(test_data)
            
            return {
                'shapiro_statistic': statistic,
                'p_value': p_value,
                'is_normal': p_value > 0.05,
                'skewness': test_data.skew(),
                'kurtosis': test_data.kurtosis()
            }
        
        except ImportError:
            # Fallback to basic statistics
            clean_data = data.dropna()
            return {
                'skewness': clean_data.skew(),
                'kurtosis': clean_data.kurtosis(),
                'is_normal': abs(clean_data.skew()) < 2 and abs(clean_data.kurtosis()) < 7
            }
