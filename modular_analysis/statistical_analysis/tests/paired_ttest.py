"""
Paired t-test implementation for paired two-group comparisons.
"""

import pandas as pd
import numpy as np
import pingouin as pg
import math
import logging
import os
from typing import List, Dict, Tuple, Optional

import statsmodels.stats.multitest as multi

from ...shared.data_models import ExperimentalDesign, StatisticalResult, DataContainer
from ...shared.utils import clean_dataframe, categorize_measurement, get_measurement_categories

logger = logging.getLogger(__name__)


class PairedTTest:
    """Implementation of paired t-test for two dependent groups."""
    
    def __init__(self):
        self.name = "Paired t-test"
        
    def run_analysis(self, design: ExperimentalDesign, data_container: DataContainer, 
                    base_path: str) -> List[StatisticalResult]:
        """Run paired t-test analysis for two conditions."""
        
        if len(design.groups) != 2:
            raise ValueError("Paired t-test requires exactly 2 groups")
        
        # Extract manifest info
        manifest = design.pairing_manifest
        if manifest is None or manifest.empty:
            raise ValueError("Paired design requires a pairing manifest")
        
        conditions = sorted(manifest['Condition'].unique())
        if len(conditions) != 2:
            raise ValueError(f"Expected 2 conditions, found {len(conditions)}")
        
        cond1_name, cond2_name = conditions[0], conditions[1]
        group1, group2 = design.groups
        
        logger.info(f"Running paired t-test: {cond1_name} vs {cond2_name}")
        
        # Load data for both groups
        combined_group1 = self._load_combined_data(group1.name, base_path)
        combined_group2 = self._load_combined_data(group2.name, base_path)
        
        if combined_group1.empty or combined_group2.empty:
            raise ValueError("No data found for one or both conditions")
        
        # Clean the data
        combined_group1 = clean_dataframe(combined_group1)
        combined_group2 = clean_dataframe(combined_group2)
        
        # Create unified dataframe with Subject_ID labels
        unified_df = self._create_unified_dataframe(
            combined_group1, combined_group2, manifest, cond1_name, cond2_name
        )
        
        # Get column names for analysis (exclude metadata columns)
        analysis_columns = self._get_analysis_columns(unified_df)
        
        # Run paired t-tests for each measurement
        results = []
        for column in analysis_columns:
            result = self._run_single_paired_test(
                column, unified_df, cond1_name, cond2_name
            )
            if result:
                results.append(result)
                
        # Apply multiple comparison correction
        results = self._apply_multiple_comparison_correction(results)
        
        logger.info(f"Completed {len(results)} paired statistical tests")
        return results
    
    def _load_combined_data(self, group_name: str, base_path: str) -> pd.DataFrame:
        """Load and combine all data types for a group."""
        import os
        
        dfs = []
        
        # Load each data type
        data_types = [
            f"Calc_{group_name}_resting_potential.csv",
            f"Calc_{group_name}_input_resistance.csv", 
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
    
    def _create_unified_dataframe(self, df1: pd.DataFrame, df2: pd.DataFrame, 
                                  manifest: pd.DataFrame, cond1_name: str, 
                                  cond2_name: str) -> pd.DataFrame:
        """
        Create unified dataframe with Subject_ID labels for both conditions.
        
        Args:
            df1: Data for condition 1
            df2: Data for condition 2
            manifest: Long-format manifest with Subject_ID, Condition, Filename
            cond1_name: Name of condition 1
            cond2_name: Name of condition 2
            
        Returns:
            DataFrame with columns: Subject_ID, Condition, and all measurements
        """
        # Create filename -> Subject_ID mapping
        filename_to_subject = dict(zip(manifest['Filename'], manifest['Subject_ID']))
        
        # Map filenames to Subject_IDs for both conditions
        # Check if Subject_ID already present (from add_subject_ids_to_extracted_data)
        if 'Subject_ID' not in df1.columns:
            df1['Subject_ID'] = df1['filename'].map(filename_to_subject)
        if 'Subject_ID' not in df2.columns:
            df2['Subject_ID'] = df2['filename'].map(filename_to_subject)
        
        # Add condition labels
        df1['Condition'] = cond1_name
        df2['Condition'] = cond2_name
        
        # Combine into single dataframe
        unified_df = pd.concat([df1, df2], ignore_index=True)
        
        # Log matching statistics
        n_subjects_cond1 = df1['Subject_ID'].notna().sum()
        n_subjects_cond2 = df2['Subject_ID'].notna().sum()
        logger.info(f"Matched {n_subjects_cond1} files in {cond1_name}, {n_subjects_cond2} files in {cond2_name}")
        
        return unified_df
    
    def _get_analysis_columns(self, df: pd.DataFrame) -> List[str]:
        """Get columns that should be analyzed statistically."""
        exclude_columns = ["filename", "Group", "geno", "Subject_ID", "Condition"]
        return [col for col in df.columns if col not in exclude_columns]
    
    def _run_single_paired_test(self, column: str, unified_df: pd.DataFrame,
                                cond1_name: str, cond2_name: str) -> Optional[StatisticalResult]:
        """Run a single paired t-test for one measurement."""
        
        # Get data for both conditions
        cond1_data = unified_df[unified_df['Condition'] == cond1_name][['Subject_ID', column]].dropna()
        cond2_data = unified_df[unified_df['Condition'] == cond2_name][['Subject_ID', column]].dropna()
        
        # Average repeated observations per subject (if multiple files map to same subject)
        cond1_data = cond1_data.groupby('Subject_ID', as_index=True)[column].mean()
        cond2_data = cond2_data.groupby('Subject_ID', as_index=True)[column].mean()
        
        # Find common subjects (intersection)
        common_subjects = cond1_data.index.intersection(cond2_data.index)
        
        if len(common_subjects) <= 1:
            logger.warning(f"Insufficient paired data for {column}: {len(common_subjects)} pairs")
            return None
        
        # Align data by Subject_ID
        cond1_aligned = cond1_data.loc[common_subjects].sort_index()
        cond2_aligned = cond2_data.loc[common_subjects].sort_index()
        
        # Extract values
        data1 = cond1_aligned.values
        data2 = cond2_aligned.values
        
        # Calculate descriptive statistics
        mean1 = data1.mean()
        mean2 = data2.mean()
        se1 = data1.std(ddof=1) / math.sqrt(len(data1)) if len(data1) > 1 else 0.0
        se2 = data2.std(ddof=1) / math.sqrt(len(data2)) if len(data2) > 1 else 0.0
        
        # Run paired t-test using pingouin
        try:
            result = pg.ttest(data1, data2, paired=True)
            p_value = result['p-val'].values[0]
            
            return StatisticalResult(
                test_name=self.name,
                measurement=column,
                group1_name=cond1_name,
                group1_mean=mean1,
                group1_stderr=se1,
                group1_n=len(data1),
                group2_name=cond2_name,
                group2_mean=mean2,
                group2_stderr=se2,
                group2_n=len(data2),
                p_value=p_value,
                measurement_type=categorize_measurement(column)
            )
            
        except Exception as e:
            logger.error(f"Error running paired t-test for {column}: {e}")
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
        
        # Explicitly set column order: descriptive stats first, then p-values last
        base_columns = ["Measurement", "MeasurementType"]
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

