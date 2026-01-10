"""
Unpaired t-test implementation for independent two-group comparisons.
"""

import pandas as pd
import numpy as np
from scipy.stats import ttest_ind, mannwhitneyu
import math
import logging
import os
from typing import List, Dict, Tuple, Optional

import statsmodels.stats.multitest as multi

from ...shared.data_models import ExperimentalDesign, StatisticalResult, DataContainer
from ...shared.utils import clean_dataframe, categorize_measurement, get_measurement_categories, should_use_parametric

logger = logging.getLogger(__name__)


class UnpairedTTest:
    """Implementation of unpaired t-test for two independent groups."""
    
    def __init__(self):
        self.name = "Unpaired t-test"
        
    def run_analysis(self, design: ExperimentalDesign, data_container: DataContainer, 
                    base_path: str) -> List[StatisticalResult]:
        """Run unpaired t-test analysis for two groups."""
        
        if len(design.groups) != 2:
            raise ValueError("Unpaired t-test requires exactly 2 groups")
            
        group1, group2 = design.groups
        
        logger.info(f"Running unpaired t-test: {group1.name} vs {group2.name}")
        
        # Load and combine data for each group
        combined_group1 = self._load_combined_data(group1.name, base_path)
        combined_group2 = self._load_combined_data(group2.name, base_path)
        
        if combined_group1.empty or combined_group2.empty:
            raise ValueError("No data found for one or both groups")
        
        # Clean the data
        combined_group1 = clean_dataframe(combined_group1)
        combined_group2 = clean_dataframe(combined_group2)
        
        # Get column names for analysis (exclude metadata columns)
        analysis_columns = self._get_analysis_columns(combined_group1)
        
        # Run t-tests for each measurement
        results = []
        for column in analysis_columns:
            result = self._run_single_test(column, combined_group1, combined_group2, 
                                         group1.name, group2.name)
            if result:
                results.append(result)
                
        # Apply multiple comparison correction
        results = self._apply_multiple_comparison_correction(results)
        
        logger.info(f"Completed {len(results)} statistical tests")
        return results
    
    def _load_combined_data(self, group_name: str, base_path: str) -> pd.DataFrame:
        """Load and combine all data types for a group."""
        import os
        
        dfs = []
        
        # Load each data type
        data_types = [
            f"Calc_{group_name}_resting_potential.csv",
            f"Calc_{group_name}_membrane_properties.csv", 
            f"Calc_{group_name}_current_step_parameters.csv",
            f"Calc_{group_name}_afterhyperpolarization.csv"
        ]
        
        results_dir = os.path.join(base_path, "Results")
        
        for filename in data_types:
            filepath = os.path.join(results_dir, filename)
            if os.path.exists(filepath):
                try:
                    df = pd.read_csv(filepath)
                    dfs.append(df)
                    logger.debug(f"Loaded {filename}: {len(df)} records")
                except Exception as e:
                    logger.warning(f"Failed to load {filename}: {e}")
                    
        if dfs:
            combined = pd.concat(dfs, ignore_index=True)
            logger.info(f"Combined data for {group_name}: {len(combined)} records")
            return combined
        else:
            logger.warning(f"No data files found for group {group_name}")
            return pd.DataFrame()
    
    def _get_analysis_columns(self, df: pd.DataFrame) -> List[str]:
        """Get columns that should be analyzed statistically."""
        exclude_columns = ["filename", "Group", "geno"]
        return [col for col in df.columns if col not in exclude_columns]
    
    def _run_single_test(self, column: str, df1: pd.DataFrame, df2: pd.DataFrame,
                        group1_name: str, group2_name: str) -> Optional[StatisticalResult]:
        """Run a single t-test for one measurement."""
        
        # Extract data, dropping NaN values
        data1 = df1[column].dropna()
        data2 = df2[column].dropna()
        
        # Check if we have sufficient data
        if len(data1) <= 1 or len(data2) <= 1:
            logger.warning(f"Insufficient data for {column}: {len(data1)}, {len(data2)}")
            return None
            
        # Calculate descriptive statistics
        mean1 = data1.mean()
        mean2 = data2.mean()
        se1 = data1.std(ddof=1) / math.sqrt(len(data1))
        se2 = data2.std(ddof=1) / math.sqrt(len(data2))
        
        # Decide parametric vs nonparametric using residuals (values centered by group)
        residuals = np.concatenate([
            (data1.values - mean1),
            (data2.values - mean2)
        ])
        use_parametric = should_use_parametric([residuals])

        try:
            if use_parametric:
                t_stat, p_value = ttest_ind(data1, data2, nan_policy="omit", equal_var=False)
            else:
                mw = mannwhitneyu(data1, data2, alternative="two-sided")
                p_value = mw.pvalue
            
            return StatisticalResult(
                test_name=self.name if use_parametric else "Mann-Whitney U",
                measurement=column,
                group1_name=group1_name,
                group1_mean=mean1,
                group1_stderr=se1,
                group1_n=len(data1),
                group2_name=group2_name,
                group2_mean=mean2,
                group2_stderr=se2,
                group2_n=len(data2),
                p_value=p_value,
                measurement_type=categorize_measurement(column)
            )
            
        except Exception as e:
            logger.error(f"Error running t-test for {column}: {e}")
            return None
    
    def _apply_multiple_comparison_correction(self, results: List[StatisticalResult]) -> List[StatisticalResult]:
        """Apply FDR correction within measurement categories."""
        
        # Group results by measurement type
        categories = get_measurement_categories()
        
        for category_name in categories.keys():
            # Get results for this category
            category_results = [r for r in results if r.measurement_type == category_name]
            
            if len(category_results) > 1:
                # Extract p-values and track valid indices (handling NaN)
                valid_indices = [i for i, r in enumerate(category_results) if not np.isnan(r.p_value)]
                valid_p_values = [category_results[i].p_value for i in valid_indices]
                
                if len(valid_p_values) > 1:
                    # Apply FDR correction
                    try:
                        rejected, corrected_p, _, _ = multi.multipletests(valid_p_values, method="fdr_bh")
                        
                        # Map corrected p-values back to valid indices
                        for i, corrected_val in zip(valid_indices, corrected_p):
                            category_results[i].corrected_p = corrected_val
                        
                        # Set NaN for invalid p-values
                        for i in range(len(category_results)):
                            if i not in valid_indices:
                                category_results[i].corrected_p = np.nan
                            
                        logger.info(f"Applied FDR correction to {len(valid_p_values)} tests in {category_name}")
                        
                    except Exception as e:
                        logger.error(f"Error applying FDR correction to {category_name}: {e}")
                        # Fallback: use uncorrected p-values
                        for r in category_results:
                            r.corrected_p = r.p_value
                else:
                    # Only 1 valid p-value - no correction needed, use original p-value
                    for r in category_results:
                        if not np.isnan(r.p_value):
                            r.corrected_p = r.p_value
                        else:
                            r.corrected_p = np.nan
            elif len(category_results) == 1:
                # Single result - no correction needed, use original p-value
                if not np.isnan(category_results[0].p_value):
                    category_results[0].corrected_p = category_results[0].p_value
                else:
                    category_results[0].corrected_p = np.nan
        
        return results
    
    def save_results(self, results: List[StatisticalResult], output_path: str) -> None:
        """Save statistical results to CSV file."""
        
        if not results:
            logger.warning("No results to save")
            return
            
        # Convert results to DataFrame
        data = []
        for result in results:
            row = {
                "Measurement": result.measurement,
                "MeasurementType": result.measurement_type,
                "Test_Type": result.test_name,
                f"{result.group1_name}_mean": result.group1_mean,
                f"{result.group1_name}_stderr": result.group1_stderr,
                f"{result.group1_name}_n": result.group1_n,
                f"{result.group2_name}_mean": result.group2_mean,
                f"{result.group2_name}_stderr": result.group2_stderr,
                f"{result.group2_name}_n": result.group2_n,
                "p-value": result.p_value,
                "corrected_p": result.corrected_p
            }
            data.append(row)
            
        df = pd.DataFrame(data)
        
        # Explicitly set column order: base info, test type, then group stats, then p-values
        base_columns = ["Measurement", "MeasurementType", "Test_Type"]
        group_columns = []
        p_value_columns = ["p-value", "corrected_p"]
        
        # Get all group-related columns dynamically
        for col in df.columns:
            if col not in base_columns + p_value_columns:
                group_columns.append(col)
        
        # Reorder columns: base + groups (sorted) + p-values
        column_order = base_columns + sorted(group_columns) + p_value_columns
        df = df[column_order]
        
        # Save with Stats_parameters.csv filename
        results_dir = os.path.dirname(output_path)
        parameters_output = os.path.join(results_dir, "Stats_parameters.csv")
        df.to_csv(parameters_output, index=False)
        logger.info(f"Saved results to {parameters_output}")
        
        # Log summary
        significant = df[df["corrected_p"] < 0.05] if "corrected_p" in df.columns else df[df["p-value"] < 0.05]
        logger.info(f"Summary: {len(significant)}/{len(df)} measurements significant (p < 0.05)")
