"""
Repeated measures ANOVA implementation for 3+ dependent groups (within-subjects design).
"""

import pandas as pd
import numpy as np
import pingouin as pg
import math
import logging
import os
from typing import List, Dict, Tuple, Optional
from itertools import combinations

import statsmodels.stats.multitest as multi

from ...shared.data_models import ExperimentalDesign, StatisticalResult, DataContainer
from ...shared.utils import clean_dataframe, categorize_measurement, get_measurement_categories

logger = logging.getLogger(__name__)


class RepeatedMeasuresANOVA:
    """Implementation of repeated measures ANOVA for 3+ dependent groups."""
    
    def __init__(self):
        self.name = "Repeated Measures ANOVA"
        
    def run_analysis(self, design: ExperimentalDesign, data_container: DataContainer, 
                    base_path: str) -> List[StatisticalResult]:
        """Run repeated measures ANOVA analysis for 3+ conditions."""
        
        if len(design.groups) < 3:
            raise ValueError("Repeated measures ANOVA requires 3 or more groups")
        
        # Extract manifest info
        manifest = design.pairing_manifest
        if manifest is None or manifest.empty:
            raise ValueError("Repeated measures design requires a pairing manifest")
        
        conditions = sorted(manifest['Condition'].unique())
        if len(conditions) < 3:
            raise ValueError(f"Expected 3+ conditions, found {len(conditions)}")
        
        logger.info(f"Running repeated measures ANOVA: {conditions}")
        
        # Load data for all groups
        all_group_data = {}
        for group in design.groups:
            combined_data = self._load_combined_data(group.name, base_path)
            if not combined_data.empty:
                all_group_data[group.name] = clean_dataframe(combined_data)
        
        if len(all_group_data) < 3:
            raise ValueError("No data found for sufficient groups")
        
        # Get column names for analysis
        analysis_columns = self._get_analysis_columns(list(all_group_data.values())[0])
        
        # Run RM-ANOVA for each measurement
        results = []
        for column in analysis_columns:
            result = self._run_single_rm_anova(column, all_group_data, conditions, manifest)
            if result:
                results.append(result)
        
        # Apply multiple comparison correction to RM-ANOVA results
        results = self._apply_multiple_comparison_correction(results)
        
        # Run post-hoc pairwise comparisons ONLY for significant RM-ANOVA results
        pairwise_results = self._run_pairwise_comparisons_if_significant(
            results, all_group_data, conditions, manifest
        )
        
        # Apply multiple comparison correction to pairwise results
        if pairwise_results:
            pairwise_results = self._apply_pairwise_correction(pairwise_results)
        
        # Combine RM-ANOVA and pairwise results
        all_results = results + pairwise_results
        
        logger.info(f"Completed {len(results)} RM-ANOVA tests and {len(pairwise_results)} pairwise comparisons")
        return all_results
    
    def _load_combined_data(self, group_name: str, base_path: str) -> pd.DataFrame:
        """Load and combine all data types for a group."""
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
        exclude_columns = ["filename", "Group", "geno", "Subject_ID"]
        return [col for col in df.columns if col not in exclude_columns]
    
    def _run_single_rm_anova(self, column: str, all_group_data: Dict[str, pd.DataFrame],
                            conditions: List[str], manifest: pd.DataFrame) -> Optional[StatisticalResult]:
        """Run a single repeated measures ANOVA for one measurement."""
        
        # Build long-format DataFrame for pingouin
        long_data = []
        
        # Map condition names to group folder names - require exact match
        condition_to_group = {}
        for condition in conditions:
            if condition in all_group_data:
                condition_to_group[condition] = condition
            else:
                logger.warning(f"Condition '{condition}' from manifest does not match any group folder name")
                logger.warning(f"Available groups: {list(all_group_data.keys())}")
        
        for condition in conditions:
            if condition not in condition_to_group:
                logger.warning(f"No group data found for condition {condition}")
                continue
            
            group_df = all_group_data[condition_to_group[condition]]
            
            if column not in group_df.columns:
                continue
            
            # Get subject IDs for this condition from manifest
            condition_manifest = manifest[manifest['Condition'] == condition]
            
            # Match subjects with their data
            for _, row in condition_manifest.iterrows():
                subject_id = row['Subject_ID']
                filename = row['Filename']
                
                # Find matching data by filename
                matching_rows = group_df[group_df['filename'].str.contains(
                    filename.replace('.abf', ''), case=False, na=False
                )]
                
                if not matching_rows.empty:
                    value = matching_rows[column].iloc[0]
                    if pd.notna(value):
                        long_data.append({
                            'Subject_ID': subject_id,
                            'Condition': condition,
                            'Value': value
                        })
        
        if len(long_data) < len(conditions) * 3:  # Need at least 3 subjects with complete data
            logger.warning(f"Insufficient data for RM-ANOVA on {column}: {len(long_data)} observations")
            return None
        
        df_long = pd.DataFrame(long_data)
        
        # Check if we have complete data (all subjects measured in all conditions)
        subjects_per_condition = df_long.groupby('Condition')['Subject_ID'].nunique()
        if subjects_per_condition.min() < 3:
            logger.warning(f"Insufficient subjects per condition for {column}")
            return None
        
        # Run repeated measures ANOVA using pingouin
        try:
            aov = pg.rm_anova(
                dv='Value',
                within='Condition',
                subject='Subject_ID',
                data=df_long
            )
            
            p_value = aov['p-unc'].values[0]
            
            # Calculate group statistics for first two conditions (for StatisticalResult compatibility)
            cond1, cond2 = conditions[0], conditions[1]
            cond1_data = df_long[df_long['Condition'] == cond1]['Value']
            cond2_data = df_long[df_long['Condition'] == cond2]['Value']
            
            return StatisticalResult(
                test_name=self.name,
                measurement=column,
                group1_name=cond1,
                group1_mean=cond1_data.mean(),
                group1_stderr=cond1_data.std() / math.sqrt(len(cond1_data)),
                group1_n=len(cond1_data),
                group2_name=cond2,
                group2_mean=cond2_data.mean(),
                group2_stderr=cond2_data.std() / math.sqrt(len(cond2_data)),
                group2_n=len(cond2_data),
                p_value=p_value,
                measurement_type=categorize_measurement(column)
            )
            
        except Exception as e:
            logger.error(f"Error running RM-ANOVA for {column}: {e}")
            return None
    
    def _run_pairwise_comparisons_if_significant(self, anova_results: List[StatisticalResult],
                                                all_group_data: Dict[str, pd.DataFrame],
                                                conditions: List[str],
                                                manifest: pd.DataFrame) -> List[StatisticalResult]:
        """Run pairwise paired t-tests ONLY for measurements with significant RM-ANOVA results."""
        
        # Get measurements with significant RM-ANOVA results
        significant_measurements = set()
        alpha = 0.05
        
        for result in anova_results:
            # Use corrected p-value if available, otherwise use raw p-value
            corrected_p = result.corrected_p if result.corrected_p is not None else result.p_value
            if corrected_p < alpha:
                significant_measurements.add(result.measurement)
                logger.info(f"RM-ANOVA significant for {result.measurement} (corrected p = {corrected_p:.4f}) - running post-hoc tests")
            else:
                logger.info(f"RM-ANOVA not significant for {result.measurement} (corrected p = {corrected_p:.4f}) - skipping post-hoc tests")
        
        if not significant_measurements:
            logger.info("No significant RM-ANOVA results - skipping all post-hoc pairwise comparisons")
            return []
        
        logger.info(f"Running post-hoc pairwise comparisons for {len(significant_measurements)} significant measurements")
        
        # Run pairwise comparisons only for significant measurements
        pairwise_results = []
        
        # Generate all pairwise combinations of conditions
        for cond1, cond2 in combinations(conditions, 2):
            for measurement in significant_measurements:
                result = self._run_pairwise_paired_test(
                    measurement, all_group_data, cond1, cond2, manifest
                )
                if result:
                    pairwise_results.append(result)
        
        return pairwise_results
    
    def _run_pairwise_paired_test(self, column: str, all_group_data: Dict[str, pd.DataFrame],
                                  cond1: str, cond2: str, manifest: pd.DataFrame) -> Optional[StatisticalResult]:
        """Run a single pairwise paired t-test between two conditions."""
        
        # Map condition names to group folder names - require exact match
        group1_name = cond1 if cond1 in all_group_data else None
        group2_name = cond2 if cond2 in all_group_data else None
        
        if group1_name is None or group2_name is None:
            return None
        
        group1_df = all_group_data[group1_name]
        group2_df = all_group_data[group2_name]
        
        if column not in group1_df.columns or column not in group2_df.columns:
            return None
        
        # Match subjects across conditions
        data1_list = []
        data2_list = []
        
        # Get all subjects that appear in both conditions
        subjects_cond1 = set(manifest[manifest['Condition'] == cond1]['Subject_ID'])
        subjects_cond2 = set(manifest[manifest['Condition'] == cond2]['Subject_ID'])
        common_subjects = subjects_cond1.intersection(subjects_cond2)
        
        for subject_id in common_subjects:
            # Get filenames for this subject in both conditions
            filename1 = manifest[(manifest['Subject_ID'] == subject_id) & 
                                (manifest['Condition'] == cond1)]['Filename'].iloc[0]
            filename2 = manifest[(manifest['Subject_ID'] == subject_id) & 
                                (manifest['Condition'] == cond2)]['Filename'].iloc[0]
            
            # Find matching data
            match1 = group1_df[group1_df['filename'].str.contains(
                filename1.replace('.abf', ''), case=False, na=False
            )]
            match2 = group2_df[group2_df['filename'].str.contains(
                filename2.replace('.abf', ''), case=False, na=False
            )]
            
            if not match1.empty and not match2.empty:
                val1 = match1[column].iloc[0]
                val2 = match2[column].iloc[0]
                
                if pd.notna(val1) and pd.notna(val2):
                    data1_list.append(val1)
                    data2_list.append(val2)
        
        # Check if we have sufficient paired data
        if len(data1_list) < 3:
            return None
        
        # Run paired t-test using pingouin
        try:
            data1_arr = np.array(data1_list)
            data2_arr = np.array(data2_list)
            
            t_result = pg.ttest(data1_arr, data2_arr, paired=True)
            p_value = t_result['p-val'].values[0]
            
            return StatisticalResult(
                test_name=f"Pairwise paired t-test ({cond1} vs {cond2})",
                measurement=column,
                group1_name=cond1,
                group1_mean=data1_arr.mean(),
                group1_stderr=data1_arr.std() / math.sqrt(len(data1_arr)),
                group1_n=len(data1_arr),
                group2_name=cond2,
                group2_mean=data2_arr.mean(),
                group2_stderr=data2_arr.std() / math.sqrt(len(data2_arr)),
                group2_n=len(data2_arr),
                p_value=p_value,
                measurement_type=categorize_measurement(column)
            )
            
        except Exception as e:
            logger.error(f"Error running pairwise test for {column} ({cond1} vs {cond2}): {e}")
            return None
    
    def _apply_multiple_comparison_correction(self, results: List[StatisticalResult]) -> List[StatisticalResult]:
        """Apply FDR correction within measurement categories for RM-ANOVA tests."""
        
        # Group results by measurement type
        categories = get_measurement_categories()
        
        for category_name in categories.keys():
            # Get RM-ANOVA results for this category
            category_results = [r for r in results 
                              if r.measurement_type == category_name and r.test_name == self.name]
            
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
                            
                        logger.info(f"Applied FDR correction to {len(valid_p_values)} RM-ANOVA tests in {category_name}")
                        
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
        
        # Group pairwise results by measurement (each RM-ANOVA family)
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
            
        # Separate RM-ANOVA and pairwise results
        anova_results = [r for r in results if r.test_name == self.name]
        pairwise_results = [r for r in results if "Pairwise" in r.test_name]
        
        # Get all unique conditions from design
        if design and design.pairing_manifest is not None:
            all_conditions = sorted(design.pairing_manifest['Condition'].unique())
        else:
            all_conditions = set()
            for result in pairwise_results:
                all_conditions.add(result.group1_name)
                all_conditions.add(result.group2_name)
            all_conditions = sorted(list(all_conditions))
        
        # Load raw group data if design and base_path provided (for complete stats)
        raw_group_data = {}
        if design and base_path:
            for group in design.groups:
                combined_data = self._load_combined_data(group.name, base_path)
                if not combined_data.empty:
                    raw_group_data[group.name] = clean_dataframe(combined_data)
        
        # Determine all possible pairwise comparisons
        all_comparisons = []
        for i, cond1 in enumerate(all_conditions):
            for cond2 in all_conditions[i+1:]:
                all_comparisons.append(f"{cond1} vs {cond2}")
        
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
                "MeasurementType": None
            }
            
            # Get RM-ANOVA result for this measurement
            anova_result = next((r for r in anova_results if r.measurement == measurement), None)
            if anova_result:
                row["MeasurementType"] = anova_result.measurement_type
            
            # Add group statistics (mean, stderr, n) for each condition
            condition_stats = {}
            
            # First, get from RM-ANOVA result itself (has stats for first 2 conditions)
            if anova_result:
                condition_stats[anova_result.group1_name] = {
                    'mean': anova_result.group1_mean,
                    'stderr': anova_result.group1_stderr,
                    'n': anova_result.group1_n
                }
                condition_stats[anova_result.group2_name] = {
                    'mean': anova_result.group2_mean,
                    'stderr': anova_result.group2_stderr,
                    'n': anova_result.group2_n
                }
            
            # Then, get from pairwise results if available (overrides ANOVA stats)
            for result in pairwise_results:
                if result.measurement == measurement:
                    if result.group1_name not in condition_stats:
                        condition_stats[result.group1_name] = {
                            'mean': result.group1_mean,
                            'stderr': result.group1_stderr,
                            'n': result.group1_n
                        }
                    if result.group2_name not in condition_stats:
                        condition_stats[result.group2_name] = {
                            'mean': result.group2_mean,
                            'stderr': result.group2_stderr,
                            'n': result.group2_n
                        }
            
            # For any STILL missing conditions, calculate from raw data if available
            # Need to match condition names to group folder names
            if raw_group_data:
                for cond_name in all_conditions:
                    if cond_name not in condition_stats:
                        # Try to find matching group data - require exact match
                        if cond_name in raw_group_data and measurement in raw_group_data[cond_name].columns:
                            matching_group = raw_group_data[cond_name]
                            data = matching_group[measurement].dropna()
                            if len(data) > 0:
                                condition_stats[cond_name] = {
                                    'mean': data.mean(),
                                    'stderr': data.std() / math.sqrt(len(data)) if len(data) > 1 else 0,
                                    'n': len(data)
                                }
            
            # Add columns for each condition
            for cond in all_conditions:
                if cond in condition_stats:
                    row[f"{cond}_mean"] = condition_stats[cond]['mean']
                    row[f"{cond}_stderr"] = condition_stats[cond]['stderr']
                    row[f"{cond}_n"] = condition_stats[cond]['n']
                else:
                    row[f"{cond}_mean"] = np.nan
                    row[f"{cond}_stderr"] = np.nan
                    row[f"{cond}_n"] = 0
            
            # Add RM-ANOVA p-values
            if anova_result:
                row["RM_ANOVA_p-value"] = anova_result.p_value
                row["RM_ANOVA_corrected_p"] = anova_result.corrected_p
            else:
                row["RM_ANOVA_p-value"] = np.nan
                row["RM_ANOVA_corrected_p"] = np.nan
            
            # Pre-initialize ALL pairwise comparison columns to NaN
            for comparison in all_comparisons:
                row[f"{comparison}_p-value"] = np.nan
                row[f"{comparison}_corrected_p"] = np.nan
            
            # Fill in pairwise p-values if they exist (RM-ANOVA was significant)
            pairwise_for_measurement = [r for r in pairwise_results if r.measurement == measurement]
            for pw_result in pairwise_for_measurement:
                comparison = f"{pw_result.group1_name} vs {pw_result.group2_name}"
                row[f"{comparison}_p-value"] = pw_result.p_value
                row[f"{comparison}_corrected_p"] = pw_result.corrected_p
            
            combined_data.append(row)
        
        # Create DataFrame and organize columns
        combined_df = pd.DataFrame(combined_data)
        
        # Order columns: Measurement, MeasurementType, then conditions (mean, stderr, n), then RM-ANOVA, then pairwise
        ordered_columns = ["Measurement", "MeasurementType"]
        
        # Add condition columns in order
        for cond in all_conditions:
            ordered_columns.extend([f"{cond}_mean", f"{cond}_stderr", f"{cond}_n"])
        
        # Add RM-ANOVA columns
        ordered_columns.extend(["RM_ANOVA_p-value", "RM_ANOVA_corrected_p"])
        
        # Add pairwise columns
        for comparison in all_comparisons:
            ordered_columns.append(f"{comparison}_p-value")
            ordered_columns.append(f"{comparison}_corrected_p")
        
        # Reorder
        combined_df = combined_df[ordered_columns]
        
        # Save combined results
        results_dir = os.path.dirname(output_path)
        combined_output = os.path.join(results_dir, "Stats_parameters.csv")
        combined_df.to_csv(combined_output, index=False)
        logger.info(f"Saved combined parameters to {combined_output}")
        
        # Log summary
        if anova_results:
            significant_anova = [r for r in anova_results if (r.corrected_p or r.p_value) < 0.05]
            logger.info(f"RM-ANOVA Summary: {len(significant_anova)}/{len(anova_results)} measurements significant (p < 0.05)")
        
        if pairwise_results:
            significant_pairwise = [r for r in pairwise_results if (r.corrected_p or r.p_value) < 0.05]
            logger.info(f"Pairwise Summary: {len(significant_pairwise)}/{len(pairwise_results)} comparisons significant (p < 0.05)")

