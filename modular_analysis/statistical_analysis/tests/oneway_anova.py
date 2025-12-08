"""
One-way ANOVA implementation for multiple independent groups.
"""

import pandas as pd
import numpy as np
from scipy.stats import kruskal, mannwhitneyu
from scipy import stats
import math
import logging
import os
from typing import List, Dict, Tuple, Optional
from itertools import combinations

import statsmodels.stats.multitest as multi
from statsmodels.stats.anova import anova_lm
from statsmodels.formula.api import ols
from statsmodels.stats.oneway import anova_oneway
import scikit_posthocs as sp

from ...shared.data_models import ExperimentalDesign, StatisticalResult, DataContainer
from ...shared.utils import clean_dataframe, categorize_measurement, get_measurement_categories, should_use_parametric

logger = logging.getLogger(__name__)


class OneWayANOVA:
    """Implementation of one-way ANOVA for multiple independent groups."""
    
    def __init__(self):
        self.name = "One-way ANOVA"
        
    def run_analysis(self, design: ExperimentalDesign, data_container: DataContainer, 
                    base_path: str) -> List[StatisticalResult]:
        """Run one-way ANOVA analysis for multiple groups."""
        
        if len(design.groups) < 3:
            raise ValueError("One-way ANOVA requires 3 or more groups")
            
        logger.info(f"Running one-way ANOVA: {[g.name for g in design.groups]}")
        
        # Load and combine data for all groups
        group_data = {}
        for group in design.groups:
            combined_data = self._load_combined_data(group.name, base_path)
            if not combined_data.empty:
                group_data[group.name] = clean_dataframe(combined_data)
        
        if len(group_data) < 3:
            raise ValueError("No data found for sufficient groups")
        
        # Get column names for analysis
        analysis_columns = self._get_analysis_columns(list(group_data.values())[0])
        
        # Run ANOVA for each measurement
        results = []
        for column in analysis_columns:
            result = self._run_single_anova(column, group_data, design.groups)
            if result:
                results.append(result)
        
        # Apply multiple comparison correction to ANOVA results
        results = self._apply_multiple_comparison_correction(results)
        
        # Run post-hoc pairwise comparisons ONLY for significant ANOVA results AND if more than 2 groups
        if len(design.groups) > 2:
            pairwise_results = self._run_pairwise_comparisons_if_significant(results, group_data, design.groups)
        else:
            logger.info("Skipping pairwise comparisons - only 2 groups (ANOVA directly tests the difference)")
            pairwise_results = []
        
        # Apply multiple comparison correction to pairwise results
        if pairwise_results:
            pairwise_results = self._apply_pairwise_correction(pairwise_results)
        
        # Combine ANOVA and pairwise results
        all_results = results + pairwise_results
        
        logger.info(f"Completed {len(results)} ANOVA tests and {len(pairwise_results)} pairwise comparisons")
        return all_results
    
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
    
    def _get_analysis_columns(self, df: pd.DataFrame) -> List[str]:
        """Get columns that should be analyzed statistically."""
        exclude_columns = ["filename", "Group", "geno"]
        return [col for col in df.columns if col not in exclude_columns]
    
    def _run_single_anova(self, column: str, group_data: Dict[str, pd.DataFrame], 
                         groups: List) -> Optional[StatisticalResult]:
        """Run a single ANOVA for one measurement."""
        
        # Extract data for all groups, dropping NaN values
        group_arrays = []
        group_stats = {}
        
        for group in groups:
            if group.name in group_data:
                data = group_data[group.name][column].dropna()
                if len(data) > 1:
                    group_arrays.append(data)
                    group_stats[group.name] = {
                        'mean': data.mean(),
                        'stderr': data.std(ddof=1) / math.sqrt(len(data)),
                        'n': len(data)
                    }
        
        # Check if we have sufficient data
        if len(group_arrays) < 3:
            logger.warning(f"Insufficient groups with data for {column}: {len(group_arrays)}")
            return None
        
        # Decide parametric vs nonparametric using skewness/kurtosis
        use_parametric = should_use_parametric(group_arrays)
            
        # Run test
        try:
            if use_parametric:
                # Welch one-way ANOVA (unequal variances)
                res = anova_oneway(group_arrays, use_var="unequal", welch_correction=True)
                p_value = res.pvalue
                test_label = self.name + " (Welch)"
            else:
                h_stat, p_value = kruskal(*group_arrays)
                test_label = "Kruskal-Wallis"
            
            # Create result (using first two groups for compatibility, but includes ANOVA p-value)
            first_group = groups[0].name
            second_group = groups[1].name
            
            return StatisticalResult(
                test_name=test_label,
                measurement=column,
                group1_name=first_group,
                group1_mean=group_stats[first_group]['mean'] if first_group in group_stats else 0,
                group1_stderr=group_stats[first_group]['stderr'] if first_group in group_stats else 0,
                group1_n=group_stats[first_group]['n'] if first_group in group_stats else 0,
                group2_name=second_group,
                group2_mean=group_stats[second_group]['mean'] if second_group in group_stats else 0,
                group2_stderr=group_stats[second_group]['stderr'] if second_group in group_stats else 0,
                group2_n=group_stats[second_group]['n'] if second_group in group_stats else 0,
                p_value=p_value,
                measurement_type=categorize_measurement(column)
            )
            
        except Exception as e:
            logger.error(f"Error running ANOVA for {column}: {e}")
            return None
    
    def _run_pairwise_comparisons_if_significant(self, anova_results: List[StatisticalResult],
                                               group_data: Dict[str, pd.DataFrame], groups: List) -> List[StatisticalResult]:
        """Run pairwise t-tests ONLY for measurements with significant ANOVA results."""
        
        # Get measurements with significant ANOVA results
        significant_measurements = set()
        alpha = 0.05
        
        for result in anova_results:
            # Use corrected p-value if available, otherwise use raw p-value
            corrected_p = result.corrected_p if result.corrected_p is not None else result.p_value
            if corrected_p < alpha:
                significant_measurements.add(result.measurement)
                logger.info(f"ANOVA significant for {result.measurement} (corrected p = {corrected_p:.4f}) - running post-hoc tests")
            else:
                logger.info(f"ANOVA not significant for {result.measurement} (corrected p = {corrected_p:.4f}) - skipping post-hoc tests")
        
        if not significant_measurements:
            logger.info("No significant ANOVA results - skipping all post-hoc pairwise comparisons")
            return []
        
        logger.info(f"Running post-hoc pairwise comparisons for {len(significant_measurements)} significant measurements")
        
        pairwise_results = []
        group_names = [g.name for g in groups if g.name in group_data]
        
        for measurement in significant_measurements:
            # Look up which omnibus test was used for this measurement
            anova_result = next((r for r in anova_results if r.measurement == measurement), None)
            is_nonparametric = anova_result and "Kruskal-Wallis" in anova_result.test_name
            
            if is_nonparametric and sp is not None:
                # Use Dunn for all pairwise comparisons, then FDR later
                values = []
                labels = []
                group_stats = {}
                for gname in group_names:
                    if measurement in group_data[gname].columns:
                        data = group_data[gname][measurement].dropna()
                        if len(data) > 0:
                            values.extend(list(data.values))
                            labels.extend([gname] * len(data))
                            group_stats[gname] = {
                                'mean': data.mean(),
                                'stderr': data.std(ddof=1) / math.sqrt(len(data)) if len(data) > 1 else 0.0,
                                'n': len(data)
                            }
                if len(set(labels)) >= 2 and len(values) > 0:
                    dunn_input = pd.DataFrame({'val': values, 'grp': labels})
                    dunn_df = sp.posthoc_dunn(dunn_input, val_col='val', group_col='grp')
                    # Ensure ordering matches group_names
                    dunn_df = dunn_df.reindex(index=group_names, columns=group_names)
                    for g1, g2 in combinations(group_names, 2):
                        p_val = dunn_df.loc[g1, g2]
                        pairwise_results.append(
                            StatisticalResult(
                                test_name=f"Pairwise Dunn ({g1} vs {g2})",
                                measurement=measurement,
                                group1_name=g1,
                                group1_mean=group_stats.get(g1, {}).get('mean', 0),
                                group1_stderr=group_stats.get(g1, {}).get('stderr', 0),
                                group1_n=group_stats.get(g1, {}).get('n', 0),
                                group2_name=g2,
                                group2_mean=group_stats.get(g2, {}).get('mean', 0),
                                group2_stderr=group_stats.get(g2, {}).get('stderr', 0),
                                group2_n=group_stats.get(g2, {}).get('n', 0),
                                p_value=p_val,
                                measurement_type=anova_result.measurement_type if anova_result else ""
                            )
                        )
            else:
                # Parametric or fallback: run pairwise tests as before
                for group1_name, group2_name in combinations(group_names, 2):
                    result = self._run_pairwise_test(
                        measurement,
                        group_data[group1_name],
                        group_data[group2_name],
                        group1_name,
                        group2_name,
                        force_parametric=not is_nonparametric if anova_result else None
                    )
                    if result:
                        pairwise_results.append(result)
        
        return pairwise_results
    
    def _run_pairwise_test(self, column: str, df1: pd.DataFrame, df2: pd.DataFrame,
                          group1_name: str, group2_name: str,
                          force_parametric: Optional[bool] = None) -> Optional[StatisticalResult]:
        """Run a single pairwise test with parametric/nonparametric decision."""
        
        # Extract data, dropping NaN values
        data1 = df1[column].dropna()
        data2 = df2[column].dropna()
        
        # Check if we have sufficient data
        if len(data1) <= 1 or len(data2) <= 1:
            return None
            
        # Calculate descriptive statistics
        mean1 = data1.mean()
        mean2 = data2.mean()
        se1 = data1.std(ddof=1) / math.sqrt(len(data1))
        se2 = data2.std(ddof=1) / math.sqrt(len(data2))
        
        # Decide parametric vs nonparametric
        # Follow the omnibus choice: if Kruskal-Wallis was used, force nonparametric;
        # otherwise use parametric. No per-pair Shapiro needed.
        if force_parametric is not None:
            use_parametric = force_parametric
        else:
            use_parametric = True
        
        try:
            if use_parametric:
                from scipy.stats import ttest_ind
                t_stat, p_value = ttest_ind(data1, data2, nan_policy="omit", equal_var=False)
                test_label = f"Pairwise t-test ({group1_name} vs {group2_name})"
            else:
                mw = mannwhitneyu(data1, data2, alternative="two-sided")
                p_value = mw.pvalue
                test_label = f"Pairwise Mann-Whitney ({group1_name} vs {group2_name})"
            
            return StatisticalResult(
                test_name=test_label,
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
            logger.error(f"Error running pairwise test for {column} ({group1_name} vs {group2_name}): {e}")
            return None
    
    def _apply_multiple_comparison_correction(self, results: List[StatisticalResult]) -> List[StatisticalResult]:
        """Apply FDR correction within measurement categories for ANOVA tests."""
        
        # Group results by measurement type
        categories = get_measurement_categories()
        
        for category_name in categories.keys():
            # Get ANOVA results for this category
            category_results = [
                r for r in results
                if r.measurement_type == category_name
                and (r.test_name.startswith(self.name) or "Kruskal-Wallis" in r.test_name)
            ]
            
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
                            
                        logger.info(f"Applied FDR correction to {len(valid_p_values)} ANOVA tests in {category_name}")
                        
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
    
    def _apply_pairwise_correction(self, pairwise_results: List[StatisticalResult]) -> List[StatisticalResult]:
        """Apply FDR correction to pairwise comparisons within each measurement's family."""
        
        # Group pairwise results by measurement (each ANOVA family)
        measurement_groups = {}
        for result in pairwise_results:
            if "Pairwise" in result.test_name:
                if result.measurement not in measurement_groups:
                    measurement_groups[result.measurement] = []
                measurement_groups[result.measurement].append(result)
        
        # Apply FDR correction within each measurement's pairwise tests
        for measurement, results_list in measurement_groups.items():
            if len(results_list) > 1:
                # Extract p-values (filter out NaN)
                p_values = [r.p_value for r in results_list if not np.isnan(r.p_value)]
                valid_indices = [i for i, r in enumerate(results_list) if not np.isnan(r.p_value)]
                
                if len(p_values) > 1:
                    # Apply FDR correction
                    try:
                        rejected, corrected_p, _, _ = multi.multipletests(p_values, method="fdr_bh")
                        
                        # Update results with corrected p-values
                        for i, corrected_val in zip(valid_indices, corrected_p):
                            results_list[i].corrected_p = corrected_val
                        
                        # Set NaN corrected_p for any NaN p-values
                        for i, result in enumerate(results_list):
                            if np.isnan(result.p_value) and not hasattr(result, 'corrected_p'):
                                result.corrected_p = np.nan
                        
                        logger.info(f"Applied FDR correction to {len(p_values)} pairwise tests for {measurement}")
                        
                    except Exception as e:
                        logger.error(f"Error applying pairwise FDR correction to {measurement}: {e}")
                else:
                    # Only 1 valid p-value - no correction needed, use original p-value
                    for result in results_list:
                        if not np.isnan(result.p_value):
                            result.corrected_p = result.p_value
                        else:
                            result.corrected_p = np.nan
            elif len(results_list) == 1:
                # Single result - no correction needed, use original p-value
                if not np.isnan(results_list[0].p_value):
                    results_list[0].corrected_p = results_list[0].p_value
                else:
                    results_list[0].corrected_p = np.nan
        
        return pairwise_results
    
    def save_results(self, results: List[StatisticalResult], output_path: str, 
                    design: ExperimentalDesign = None, base_path: str = None) -> None:
        """Save statistical results to CSV file - combined format with all stats in one row per measurement."""
        
        if not results:
            logger.warning("No results to save")
            return
            
        # Separate ANOVA and pairwise results
        anova_results = [
            r for r in results
            if r.test_name.startswith(self.name) or "Kruskal-Wallis" in r.test_name
        ]
        pairwise_results = [r for r in results if "Pairwise" in r.test_name]
        
        # Get all unique groups - prefer from design if available, otherwise from results
        if design and design.groups:
            all_groups = sorted([g.name for g in design.groups])
        else:
            all_groups = set()
            for result in pairwise_results:
                all_groups.add(result.group1_name)
                all_groups.add(result.group2_name)
            # Also check ANOVA results in case there are no pairwise results at all
            for result in anova_results:
                all_groups.add(result.group1_name)
                all_groups.add(result.group2_name)
            all_groups = sorted(list(all_groups))
        
        # Load raw group data if design and base_path provided (for complete stats)
        raw_group_data = {}
        if design and base_path:
            for group in design.groups:
                combined_data = self._load_combined_data(group.name, base_path)
                if not combined_data.empty:
                    raw_group_data[group.name] = clean_dataframe(combined_data)
        
        # Determine all possible pairwise comparisons
        all_comparisons = []
        for i, group1 in enumerate(all_groups):
            for group2 in all_groups[i+1:]:
                all_comparisons.append(f"{group1} vs {group2}")
        
        # Build combined data: one row per measurement
        combined_data = []
        
        # Get unique measurements
        measurements = set()
        for result in anova_results:
            measurements.add(result.measurement)
        
        for measurement in sorted(measurements):
            # Start row with measurement info
            row = {
                "Measurement": measurement,
                "MeasurementType": None,
                "Test_Type": None,
                "Omnibus_p-value": np.nan,
                "Omnibus_corrected_p": np.nan
            }
            
            # Get ANOVA/Welch/KW result for this measurement
            anova_result = next((r for r in anova_results if r.measurement == measurement), None)
            if anova_result:
                row["MeasurementType"] = anova_result.measurement_type
                row["Test_Type"] = anova_result.test_name
                row["Omnibus_p-value"] = anova_result.p_value
                row["Omnibus_corrected_p"] = anova_result.corrected_p
            
            # Add group statistics (mean, stderr, n) for each group
            # Priority: pairwise results > raw data calculation > ANOVA result (fallback)
            group_stats = {}
            
            # First, get from pairwise results if available
            for result in pairwise_results:
                if result.measurement == measurement:
                    if result.group1_name not in group_stats:
                        group_stats[result.group1_name] = {
                            'mean': result.group1_mean,
                            'stderr': result.group1_stderr,
                            'n': result.group1_n
                        }
                    if result.group2_name not in group_stats:
                        group_stats[result.group2_name] = {
                            'mean': result.group2_mean,
                            'stderr': result.group2_stderr,
                            'n': result.group2_n
                        }
            
            # For any missing groups, calculate from raw data if available
            if raw_group_data:
                for group_name in all_groups:
                    if group_name not in group_stats and group_name in raw_group_data:
                        group_df = raw_group_data[group_name]
                        if measurement in group_df.columns:
                            data = group_df[measurement].dropna()
                            if len(data) > 0:
                                group_stats[group_name] = {
                                    'mean': data.mean(),
                                    'stderr': data.std(ddof=1) / math.sqrt(len(data)) if len(data) > 1 else 0,
                                    'n': len(data)
                                }
            
            # If still no group_stats, try to get from ANOVA result as last resort
            if not group_stats and anova_result:
                group_stats[anova_result.group1_name] = {
                    'mean': anova_result.group1_mean,
                    'stderr': anova_result.group1_stderr,
                    'n': anova_result.group1_n
                }
                group_stats[anova_result.group2_name] = {
                    'mean': anova_result.group2_mean,
                    'stderr': anova_result.group2_stderr,
                    'n': anova_result.group2_n
                }
            
            # Add columns for each group
            for group in all_groups:
                if group in group_stats:
                    row[f"{group}_mean"] = group_stats[group]['mean']
                    row[f"{group}_stderr"] = group_stats[group]['stderr']
                    row[f"{group}_n"] = group_stats[group]['n']
                else:
                    row[f"{group}_mean"] = np.nan
                    row[f"{group}_stderr"] = np.nan
                    row[f"{group}_n"] = 0
            
            # Add ANOVA p-values
            if anova_result:
                row["ANOVA_p-value"] = anova_result.p_value
                row["ANOVA_corrected_p"] = anova_result.corrected_p
            else:
                row["ANOVA_p-value"] = np.nan
                row["ANOVA_corrected_p"] = np.nan
            
            # Pre-initialize ALL pairwise comparison columns to NaN
            for comparison in all_comparisons:
                row[f"{comparison}_p-value"] = np.nan
                row[f"{comparison}_corrected_p"] = np.nan
            
            # Fill in pairwise p-values if they exist (ANOVA was significant)
            pairwise_for_measurement = [r for r in pairwise_results if r.measurement == measurement]
            for pw_result in pairwise_for_measurement:
                comparison = f"{pw_result.group1_name} vs {pw_result.group2_name}"
                row[f"{comparison}_p-value"] = pw_result.p_value
                row[f"{comparison}_corrected_p"] = pw_result.corrected_p
            
            combined_data.append(row)
        
        # Create DataFrame and organize columns
        combined_df = pd.DataFrame(combined_data)
        
        # Order columns: Measurement, MeasurementType, Test_Type, then groups (mean, stderr, n), then p-values, then pairwise
        ordered_columns = ["Measurement", "MeasurementType", "Test_Type"]
        
        # Add group columns in order
        for group in all_groups:
            ordered_columns.extend([f"{group}_mean", f"{group}_stderr", f"{group}_n"])
        
        # Add omnibus columns (p-value first, then corrected_p)
        ordered_columns.extend(["Omnibus_p-value", "Omnibus_corrected_p"])
        
        # Add pairwise columns in correct order (p-value before corrected_p for each comparison)
        for comparison in all_comparisons:
            ordered_columns.append(f"{comparison}_p-value")
            ordered_columns.append(f"{comparison}_corrected_p")
        
        # Reorder
        combined_df = combined_df[ordered_columns]
        
        # Save combined results
        # Replace filename entirely with Stats_parameters.csv
        results_dir = os.path.dirname(output_path)
        combined_output = os.path.join(results_dir, "Stats_parameters.csv")
        combined_df.to_csv(combined_output, index=False)
        logger.info(f"Saved combined parameters to {combined_output}")
        
        # Log summary
        if anova_results:
            significant_anova = [r for r in anova_results if (r.corrected_p or r.p_value) < 0.05]
            logger.info(f"ANOVA Summary: {len(significant_anova)}/{len(anova_results)} measurements significant (p < 0.05)")
        
        if pairwise_results:
            significant_pairwise = [r for r in pairwise_results if (r.corrected_p or r.p_value) < 0.05]
            logger.info(f"Pairwise Summary: {len(significant_pairwise)}/{len(pairwise_results)} comparisons significant (p < 0.05)")
