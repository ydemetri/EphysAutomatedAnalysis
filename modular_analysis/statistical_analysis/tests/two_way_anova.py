"""
Two-way ANOVA implementation for N×M factorial designs.
"""

import pandas as pd
import numpy as np
from scipy.stats import ttest_ind, t
import math
import logging
import os
from itertools import combinations
from typing import List, Dict, Tuple, Optional

import statsmodels.stats.multitest as multi
from statsmodels.stats.anova import anova_lm
from statsmodels.formula.api import ols

from ...shared.data_models import ExperimentalDesign, StatisticalResult, DataContainer
from ...shared.utils import clean_dataframe, categorize_measurement, get_measurement_categories
from .posthoc_utils import (
    should_run_posthocs,
    get_simple_effect_comparisons,
    compute_lsmean_summaries,
)

logger = logging.getLogger(__name__)


class TwoWayANOVA:
    """Implementation of two-way ANOVA for N×M factorial designs."""
    
    def __init__(self):
        self.name = "Two-Way ANOVA"
        self._measurement_fit_info = {}
        
    def run_analysis(self, design: ExperimentalDesign, container: DataContainer, 
                    base_path: str) -> List[StatisticalResult]:
        """Run two-way ANOVA for N×M factorial design."""

        # Reset cached model info for this run
        self._measurement_fit_info = {}
        
        # Calculate expected groups from factor levels
        factor1_levels = set(design.factor_mapping[g.name]['factor1'] for g in design.groups)
        factor2_levels = set(design.factor_mapping[g.name]['factor2'] for g in design.groups)
        expected_groups = len(factor1_levels) * len(factor2_levels)
        
        if len(design.groups) != expected_groups:
            raise ValueError(
                f"Two-way ANOVA requires {expected_groups} groups for "
                f"{len(factor1_levels)}×{len(factor2_levels)} factorial (found {len(design.groups)})"
            )
            
        logger.info(f"Running {len(factor1_levels)}×{len(factor2_levels)} two-way ANOVA: {[g.name for g in design.groups]}")
        
        # Load and combine data for all groups
        group_data = {}
        for group in design.groups:
            combined_data = self._load_combined_data(group.name, base_path)
            if not combined_data.empty:
                group_data[group.name] = clean_dataframe(combined_data)
        
        if len(group_data) != expected_groups:
            raise ValueError(f"No data found for all {expected_groups} groups (found data for {len(group_data)})")
        
        # Get column names for analysis
        analysis_columns = self._get_analysis_columns(list(group_data.values())[0])
        
        # Run two-way ANOVA for each measurement
        anova_results = []
        for column in analysis_columns:
            result = self._run_single_two_way_anova(column, group_data, design)
            if result:
                anova_results.extend(result)  # Returns 3 results (Factor1, Factor2, Interaction)
        
        # Apply multiple comparison correction (separate by effect type)
        anova_results = self._apply_multiple_comparison_correction(anova_results, design)
        
        # Run post-hoc comparisons ONLY for measurements with significant interaction
        posthoc_results = self._run_posthoc_if_interaction_significant(
            anova_results, group_data, design, analysis_columns
        )
        
        # Apply FDR correction to post-hoc tests (within measurement families)
        if posthoc_results:
            posthoc_results = self._apply_posthoc_correction(posthoc_results)
        
        # Combine all results
        all_results = anova_results + posthoc_results
        
        logger.info(f"Two-way ANOVA complete: {len(anova_results)} ANOVA effects, {len(posthoc_results)} post-hoc comparisons")
        
        return all_results
    
    def _run_single_two_way_anova(self, column: str, group_data: Dict[str, pd.DataFrame],
                                  design: ExperimentalDesign) -> Optional[List[StatisticalResult]]:
        """Run two-way ANOVA for a single measurement."""
        
        # Build dataframe with factor labels
        data_list = []
        for group in design.groups:
            if group.name in group_data:
                group_df = group_data[group.name]
                if column in group_df.columns:
                    values = group_df[column].dropna()
                    
                    # Get factor levels for this group
                    factor1_level = design.factor_mapping[group.name]['factor1']
                    factor2_level = design.factor_mapping[group.name]['factor2']
                    
                    for val in values:
                        data_list.append({
                            'measurement': val,
                            'factor1': factor1_level,
                            'factor2': factor2_level,
                            'group': group.name
                        })
        
        if len(data_list) < 4:
            logger.warning(f"Insufficient data for {column}")
            return None
        
        df = pd.DataFrame(data_list)
        
        # Check we have data for all factor combinations
        factor_combos = df.groupby(['factor1', 'factor2']).size()
        factor1_levels = df['factor1'].nunique()
        factor2_levels = df['factor2'].nunique()
        expected_combos = factor1_levels * factor2_levels
        
        if len(factor_combos) < expected_combos:
            logger.warning(f"Missing factor combinations for {column} (have {len(factor_combos)}/{expected_combos})")
            return None
        
        # Run two-way ANOVA using statsmodels
        try:
            model = ols('measurement ~ C(factor1) * C(factor2)', data=df).fit()
            anova_table = anova_lm(model, typ=2)
            cell_stats_df = (
                df.groupby(['factor1', 'factor2'])['measurement']
                .agg(['mean', 'count'])
                .reset_index()
            )
            cell_stats = {
                (row['factor1'], row['factor2']): {
                    'mean': row['mean'],
                    'count': int(row['count'])
                }
                for _, row in cell_stats_df.iterrows()
            }

            self._measurement_fit_info[column] = {
                'mse': model.mse_resid,
                'df_resid': int(model.df_resid),
                'cell_stats': cell_stats,
                'factor1_levels': sorted(df['factor1'].unique()),
                'factor2_levels': sorted(df['factor2'].unique())
            }
            
            # Extract p-values for each effect
            factor1_p = anova_table.loc['C(factor1)', 'PR(>F)']
            factor2_p = anova_table.loc['C(factor2)', 'PR(>F)']
            interaction_p = anova_table.loc['C(factor1):C(factor2)', 'PR(>F)']
            
            measurement_type = categorize_measurement(column)
            
            # Create three StatisticalResult objects (one per effect)
            results = []
            
            # Factor 1 main effect
            results.append(StatisticalResult(
                test_name=f"{self.name} - {design.factor1_name}",
                measurement=column,
                group1_name=design.groups[0].name,
                group1_mean=0,  # Will be filled in save_results
                group1_stderr=0,
                group1_n=0,
                group2_name=design.groups[1].name,
                group2_mean=0,
                group2_stderr=0,
                group2_n=0,
                p_value=factor1_p,
                measurement_type=measurement_type
            ))
            
            # Factor 2 main effect
            results.append(StatisticalResult(
                test_name=f"{self.name} - {design.factor2_name}",
                measurement=column,
                group1_name=design.groups[0].name,
                group1_mean=0,
                group1_stderr=0,
                group1_n=0,
                group2_name=design.groups[1].name,
                group2_mean=0,
                group2_stderr=0,
                group2_n=0,
                p_value=factor2_p,
                measurement_type=measurement_type
            ))
            
            # Interaction effect
            results.append(StatisticalResult(
                test_name=f"{self.name} - Interaction",
                measurement=column,
                group1_name=design.groups[0].name,
                group1_mean=0,
                group1_stderr=0,
                group1_n=0,
                group2_name=design.groups[1].name,
                group2_mean=0,
                group2_stderr=0,
                group2_n=0,
                p_value=interaction_p,
                measurement_type=measurement_type
            ))
            
            return results
            
        except Exception as e:
            logger.error(f"Error running two-way ANOVA for {column}: {e}")
            return None
    
    def _apply_multiple_comparison_correction(self, results: List[StatisticalResult],
                                             design: ExperimentalDesign) -> List[StatisticalResult]:
        """Apply FDR correction separately by measurement category AND effect type.
        
        Corrects Factor1, Factor2, and Interaction effects separately within each measurement category.
        This acknowledges that these are different hypotheses being tested.
        """
        
        categories = get_measurement_categories()
        
        # Define effect types based on test name patterns
        effect_types = [design.factor1_name, design.factor2_name, "Interaction"]
        
        for category_name in categories.keys():
            for effect_type in effect_types:
                # Get results for this category AND effect type
                if effect_type == "Interaction":
                    category_effect_results = [
                        r for r in results 
                        if r.measurement_type == category_name 
                        and r.test_name.startswith("Two-Way ANOVA")
                        and "Interaction" in r.test_name
                    ]
                else:
                    category_effect_results = [
                        r for r in results 
                        if r.measurement_type == category_name 
                        and r.test_name.startswith("Two-Way ANOVA")
                        and effect_type in r.test_name
                        and "Interaction" not in r.test_name
                    ]
                
                if len(category_effect_results) > 1:
                    # Extract p-values (filter out NaN)
                    p_values = [r.p_value for r in category_effect_results if not np.isnan(r.p_value)]
                    valid_indices = [i for i, r in enumerate(category_effect_results) if not np.isnan(r.p_value)]
                    
                    if len(p_values) > 1:
                        try:
                            rejected, corrected_p, _, _ = multi.multipletests(p_values, method="fdr_bh")
                            
                            # Update results with corrected p-values
                            for i, corrected_val in zip(valid_indices, corrected_p):
                                category_effect_results[i].corrected_p = corrected_val
                            
                            # Set NaN for any NaN p-values
                            for i, result in enumerate(category_effect_results):
                                if np.isnan(result.p_value) and not hasattr(result, 'corrected_p'):
                                    result.corrected_p = np.nan
                            
                            logger.info(f"Applied FDR correction to {len(p_values)} {effect_type} effects in {category_name}")
                            
                        except Exception as e:
                            logger.error(f"Error applying FDR correction to {category_name} {effect_type}: {e}")
                    else:
                        # Only 1 valid p-value - no correction needed, use original p-value
                        for result in category_effect_results:
                            if not np.isnan(result.p_value):
                                result.corrected_p = result.p_value
                            else:
                                result.corrected_p = np.nan
                elif len(category_effect_results) == 1:
                    # Single result - no correction needed, use original p-value
                    if not np.isnan(category_effect_results[0].p_value):
                        category_effect_results[0].corrected_p = category_effect_results[0].p_value
                    else:
                        category_effect_results[0].corrected_p = np.nan
        
        return results
    
    def _run_posthoc_if_interaction_significant(self, anova_results: List[StatisticalResult],
                                                group_data: Dict[str, pd.DataFrame],
                                                design: ExperimentalDesign,
                                                analysis_columns: List[str]) -> List[StatisticalResult]:
        """Run post-hoc comparisons for measurements with significant effects that require post-hocs.
        
        Post-hoc logic:
        - Main effect with 2 levels: No post-hoc needed (effect IS the comparison)
        - Main effect with 3+ levels: Run post-hocs
        - Significant interaction: Run simple effects comparisons
        """
        
        # Get measurements that need post-hocs based on corrected p-values and factor levels
        measurements_needing_posthocs = set()
        measurement_reasons = {}  # Track which effects require post-hocs
        
        # Group results by measurement
        results_by_measurement = {}
        for result in anova_results:
            if result.measurement not in results_by_measurement:
                results_by_measurement[result.measurement] = {}
            
            # Store corrected p-values for each effect
            if design.factor1_name in result.test_name:
                results_by_measurement[result.measurement]['factor1_corrected_p'] = result.corrected_p if result.corrected_p is not None else result.p_value
            elif design.factor2_name in result.test_name:
                results_by_measurement[result.measurement]['factor2_corrected_p'] = result.corrected_p if result.corrected_p is not None else result.p_value
            elif "Interaction" in result.test_name:
                results_by_measurement[result.measurement]['Interaction_corrected_p'] = result.corrected_p if result.corrected_p is not None else result.p_value
        
        # For each measurement, determine if post-hocs are needed
        group_names = [g.name for g in design.groups]
        for measurement, effect_ps in results_by_measurement.items():
            # Create a result dict with the naming expected by should_run_posthocs
            anova_result = {
                f'{design.factor1_name}_corrected_p': effect_ps.get('factor1_corrected_p', 1),
                f'{design.factor2_name}_corrected_p': effect_ps.get('factor2_corrected_p', 1),
                'Interaction_corrected_p': effect_ps.get('Interaction_corrected_p', 1)
            }
            
            posthoc_decision = should_run_posthocs(anova_result, design, group_names)
            
            if posthoc_decision['run_posthocs']:
                measurements_needing_posthocs.add(measurement)
                measurement_reasons[measurement] = posthoc_decision['reasons']
                logger.info(f"{measurement}: Running post-hocs for {posthoc_decision['reasons']}")
        
        if not measurements_needing_posthocs:
            logger.info("No measurements require post-hoc comparisons (all significant effects have 2 levels)")
            return []
        
        logger.info(f"Running post-hoc comparisons for {len(measurements_needing_posthocs)} measurements")
        
        # Run simple effects post-hoc tests
        posthoc_results = []
        for measurement in measurements_needing_posthocs:
            if measurement in analysis_columns:
                reasons = measurement_reasons.get(measurement, [])
                if 'interaction' in reasons:
                    results = self._run_logical_posthoc(measurement, group_data, design)
                    if results:
                        posthoc_results.extend(results)
                else:
                    if 'factor1_main' in reasons:
                        results = self._run_marginal_posthoc(
                            measurement, group_data, design, factor_key='factor1'
                        )
                        if results:
                            posthoc_results.extend(results)
                    if 'factor2_main' in reasons:
                        results = self._run_marginal_posthoc(
                            measurement, group_data, design, factor_key='factor2'
                        )
                        if results:
                            posthoc_results.extend(results)
        
        return posthoc_results

    def _run_marginal_posthoc(self, measurement: str, group_data: Dict[str, pd.DataFrame],
                              design: ExperimentalDesign, factor_key: str) -> List[StatisticalResult]:
        """Run marginal mean comparisons for a main effect when interaction is not significant."""

        fit_info = self._measurement_fit_info.get(measurement)
        if not fit_info:
            logger.warning(f"No cached model info for {measurement}, skipping marginal posthoc")
            return []

        mse = fit_info.get('mse')
        df_resid = fit_info.get('df_resid')
        cell_stats = fit_info.get('cell_stats')
        factor1_levels = fit_info.get('factor1_levels', [])
        factor2_levels = fit_info.get('factor2_levels', [])
        if mse is None or np.isnan(mse) or df_resid is None or df_resid <= 0 or not cell_stats:
            logger.warning(f"Invalid model info (mse={mse}, df={df_resid}) for {measurement}")
            return []

        factor_label = design.factor1_name if factor_key == 'factor1' else design.factor2_name

        target_levels = factor1_levels if factor_key == 'factor1' else factor2_levels
        other_levels = factor2_levels if factor_key == 'factor1' else factor1_levels
        level_summaries = compute_lsmean_summaries(
            cell_stats,
            target_levels,
            other_levels,
            mse,
            factor_key,
        )

        if len(level_summaries) < 2:
            logger.info(f"Insufficient level summaries for {measurement} ({factor_label})")
            return []

        measurement_type = categorize_measurement(measurement)
        results = []
        levels = sorted(level_summaries.keys())

        for level1, level2 in combinations(levels, 2):
            stats1 = level_summaries[level1]
            stats2 = level_summaries[level2]

            if stats1['count'] == 0 or stats2['count'] == 0:
                continue

            se1 = math.sqrt(stats1['variance'])
            se2 = math.sqrt(stats2['variance'])
            se_diff = math.sqrt(stats1['variance'] + stats2['variance'])
            if not np.isfinite(se_diff) or se_diff == 0:
                continue

            diff = stats1['mean'] - stats2['mean']
            t_stat = diff / se_diff
            p_value = 2 * t.sf(abs(t_stat), df_resid)

            result = StatisticalResult(
                test_name=f"Marginal t-test ({factor_label})",
                measurement=measurement,
                group1_name=f"{factor_label}={level1}",
                group1_mean=stats1['mean'],
                group1_stderr=se1,
                group1_n=stats1['count'],
                group2_name=f"{factor_label}={level2}",
                group2_mean=stats2['mean'],
                group2_stderr=se2,
                group2_n=stats2['count'],
                p_value=p_value,
                measurement_type=measurement_type
            )
            results.append(result)

        return results
    
    def _run_logical_posthoc(self, measurement: str, group_data: Dict[str, pd.DataFrame],
                            design: ExperimentalDesign) -> List[StatisticalResult]:
        """Run logical simple effects comparisons for N×M factorial design.
        
        For an N×M design, compares:
        - All pairs of factor2 levels within each factor1 level: N × C(M,2) comparisons
        - All pairs of factor1 levels within each factor2 level: M × C(N,2) comparisons
        
        Examples:
        - 2×2: 2×1 + 2×1 = 4 comparisons
        - 2×3: 2×3 + 3×1 = 9 comparisons
        - 3×3: 3×3 + 3×3 = 18 comparisons
        """
        
        # Get factor levels
        factor1_levels = sorted(set(design.factor_mapping[g.name]['factor1'] for g in design.groups))
        factor2_levels = sorted(set(design.factor_mapping[g.name]['factor2'] for g in design.groups))
        
        posthoc_results = []
        
        # Within each factor1 level, compare all pairs of factor2 levels
        for f1_level in factor1_levels:
            # Find groups with this factor1 level
            groups_f1 = [g for g in design.groups if design.factor_mapping[g.name]['factor1'] == f1_level]
            
            if len(groups_f1) >= 2:
                # Sort by factor2 to ensure consistent ordering
                groups_f1 = sorted(groups_f1, key=lambda g: design.factor_mapping[g.name]['factor2'])
                
                # Compare all pairs
                for g1, g2 in combinations(groups_f1, 2):
                    result = self._run_pairwise_test(
                        measurement, 
                        group_data[g1.name],
                        group_data[g2.name],
                        g1.name,
                        g2.name,
                        f"Within {design.factor1_name}={f1_level}"
                    )
                    if result:
                        posthoc_results.append(result)
        
        # Within each factor2 level, compare all pairs of factor1 levels
        for f2_level in factor2_levels:
            # Find groups with this factor2 level
            groups_f2 = [g for g in design.groups if design.factor_mapping[g.name]['factor2'] == f2_level]
            
            if len(groups_f2) >= 2:
                # Sort by factor1 to ensure consistent ordering
                groups_f2 = sorted(groups_f2, key=lambda g: design.factor_mapping[g.name]['factor1'])
                
                # Compare all pairs
                for g1, g2 in combinations(groups_f2, 2):
                    result = self._run_pairwise_test(
                        measurement,
                        group_data[g1.name],
                        group_data[g2.name],
                        g1.name,
                        g2.name,
                        f"Within {design.factor2_name}={f2_level}"
                    )
                    if result:
                        posthoc_results.append(result)
        
        return posthoc_results
    
    def _run_pairwise_test(self, column: str, df1: pd.DataFrame, df2: pd.DataFrame,
                          group1_name: str, group2_name: str, 
                          context: str) -> Optional[StatisticalResult]:
        """Run a single pairwise t-test for post-hoc comparison."""
        
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
        
        # Run pooled-variance t-test (aligned with homoscedastic ANOVA)
        try:
            t_stat, p_value = ttest_ind(data1, data2, nan_policy="omit", equal_var=True)
            
            return StatisticalResult(
                test_name=f"Pairwise t-test ({context})",
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
            logger.error(f"Error running pairwise test for {column}: {e}")
            return None
    
    def _apply_posthoc_correction(self, posthoc_results: List[StatisticalResult]) -> List[StatisticalResult]:
        """Apply FDR correction to post-hoc comparisons within each measurement's family."""
        
        # Group post-hoc results by measurement (each interaction family)
        measurement_groups = {}
        for result in posthoc_results:
            if "Pairwise" in result.test_name:
                if result.measurement not in measurement_groups:
                    measurement_groups[result.measurement] = []
                measurement_groups[result.measurement].append(result)
        
        # Apply FDR correction within each measurement's post-hoc tests
        for measurement, results_list in measurement_groups.items():
            if len(results_list) > 1:
                # Extract p-values (filter out NaN)
                p_values = [r.p_value for r in results_list if not np.isnan(r.p_value)]
                valid_indices = [i for i, r in enumerate(results_list) if not np.isnan(r.p_value)]
                
                if len(p_values) > 1:
                    try:
                        rejected, corrected_p, _, _ = multi.multipletests(p_values, method="fdr_bh")
                        
                        # Update results with corrected p-values
                        for i, corrected_val in zip(valid_indices, corrected_p):
                            results_list[i].corrected_p = corrected_val
                        
                        # Set NaN corrected_p for any NaN p-values
                        for i, result in enumerate(results_list):
                            if np.isnan(result.p_value) and not hasattr(result, 'corrected_p'):
                                result.corrected_p = np.nan
                            
                        logger.info(f"Applied FDR correction to {len(p_values)} post-hoc tests for {measurement}")
                        
                    except Exception as e:
                        logger.error(f"Error applying post-hoc FDR correction to {measurement}: {e}")
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
        
        return posthoc_results
    
    def _load_combined_data(self, group_name: str, base_path: str) -> pd.DataFrame:
        """Load and combine all data types for a group."""
        
        dfs = []
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
                except Exception as e:
                    logger.warning(f"Failed to load {filename}: {e}")
                    
        if dfs:
            combined = pd.concat(dfs, ignore_index=True)
            return clean_dataframe(combined)
        else:
            return pd.DataFrame()
    
    def _get_analysis_columns(self, df: pd.DataFrame) -> List[str]:
        """Get columns that should be analyzed."""
        exclude_columns = ["filename", "Group", "geno"]
        return [col for col in df.columns if col not in exclude_columns]
    
    def save_results(self, results: List[StatisticalResult], output_path: str,
                    design: ExperimentalDesign = None, base_path: str = None) -> None:
        """Save two-way ANOVA results to CSV file."""
        
        if not results:
            logger.warning("No results to save")
            return
        
        # Separate ANOVA effects and post-hoc results
        anova_results = [r for r in results if "ANOVA" in r.test_name]
        posthoc_results = [r for r in results if "Pairwise" in r.test_name]
        
        # Get all groups
        if design and design.groups:
            all_groups = sorted([g.name for g in design.groups])
        else:
            all_groups = []
        
        # Load raw group data if design and base_path provided (for complete stats)
        raw_group_data = {}
        if design and base_path:
            for group in design.groups:
                combined_data = self._load_combined_data(group.name, base_path)
                if not combined_data.empty:
                    raw_group_data[group.name] = clean_dataframe(combined_data)
        
        # Determine all possible post-hoc comparisons (4 for factorial)
        all_posthoc_comparisons = set()
        for result in posthoc_results:
            comparison = f"{result.group1_name} vs {result.group2_name}"
            all_posthoc_comparisons.add(comparison)
        all_posthoc_comparisons = sorted(list(all_posthoc_comparisons))
        
        # Build combined data: one row per measurement
        combined_data = []
        
        # Get unique measurements from ANOVA results
        measurements = set()
        for result in anova_results:
            if "Interaction" in result.test_name:  # Use interaction as the key
                measurements.add(result.measurement)
        
        for measurement in sorted(measurements):
            # Start row with measurement info
            row = {
                "Measurement": measurement,
                "MeasurementType": None
            }
            
            # Get the three ANOVA effects for this measurement
            factor1_result = next((r for r in anova_results 
                                  if r.measurement == measurement and design.factor1_name in r.test_name), None)
            factor2_result = next((r for r in anova_results 
                                  if r.measurement == measurement and design.factor2_name in r.test_name), None)
            interaction_result = next((r for r in anova_results 
                                      if r.measurement == measurement and "Interaction" in r.test_name), None)
            
            if interaction_result:
                row["MeasurementType"] = interaction_result.measurement_type
            
            # Calculate group stats from raw data
            group_stats = {}
            if raw_group_data:
                for group_name in all_groups:
                    if group_name in raw_group_data:
                        group_df = raw_group_data[group_name]
                        if measurement in group_df.columns:
                            data = group_df[measurement].dropna()
                            if len(data) > 0:
                                group_stats[group_name] = {
                                    'mean': data.mean(),
                                    'stderr': data.std(ddof=1) / math.sqrt(len(data)) if len(data) > 1 else 0,
                                    'n': len(data)
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
            
            # Add ANOVA p-values for all three effects
            if factor1_result:
                row[f"{design.factor1_name}_p-value"] = factor1_result.p_value
                row[f"{design.factor1_name}_corrected_p"] = factor1_result.corrected_p
            else:
                row[f"{design.factor1_name}_p-value"] = np.nan
                row[f"{design.factor1_name}_corrected_p"] = np.nan
            
            if factor2_result:
                row[f"{design.factor2_name}_p-value"] = factor2_result.p_value
                row[f"{design.factor2_name}_corrected_p"] = factor2_result.corrected_p
            else:
                row[f"{design.factor2_name}_p-value"] = np.nan
                row[f"{design.factor2_name}_corrected_p"] = np.nan
            
            if interaction_result:
                row["Interaction_p-value"] = interaction_result.p_value
                row["Interaction_corrected_p"] = interaction_result.corrected_p
            else:
                row["Interaction_p-value"] = np.nan
                row["Interaction_corrected_p"] = np.nan
            
            # Pre-initialize ALL post-hoc comparison columns to NaN
            for comparison in all_posthoc_comparisons:
                row[f"{comparison}_p-value"] = np.nan
                row[f"{comparison}_corrected_p"] = np.nan
            
            # Fill in post-hoc p-values if they exist (interaction was significant)
            posthoc_for_measurement = [r for r in posthoc_results if r.measurement == measurement]
            for ph_result in posthoc_for_measurement:
                comparison = f"{ph_result.group1_name} vs {ph_result.group2_name}"
                row[f"{comparison}_p-value"] = ph_result.p_value
                row[f"{comparison}_corrected_p"] = ph_result.corrected_p
            
            combined_data.append(row)
        
        # Create DataFrame and organize columns
        combined_df = pd.DataFrame(combined_data)
        
        # Order columns: Measurement, MeasurementType, groups, ANOVA effects, post-hoc
        ordered_columns = ["Measurement", "MeasurementType"]
        
        # Add group columns in order
        for group in all_groups:
            ordered_columns.extend([f"{group}_mean", f"{group}_stderr", f"{group}_n"])
        
        # Add ANOVA effect columns (p-value first, then corrected_p for each effect)
        ordered_columns.extend([
            f"{design.factor1_name}_p-value", f"{design.factor1_name}_corrected_p",
            f"{design.factor2_name}_p-value", f"{design.factor2_name}_corrected_p",
            "Interaction_p-value", "Interaction_corrected_p"
        ])
        
        # Add post-hoc columns in correct order (p-value before corrected_p for each comparison)
        for comparison in all_posthoc_comparisons:
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
            significant_interactions = [r for r in anova_results 
                                       if "Interaction" in r.test_name 
                                       and (r.corrected_p or r.p_value) < 0.05]
            logger.info(f"Two-way ANOVA Summary: {len(significant_interactions)}/{len(measurements)} interactions significant (p < 0.05)")
        
        if posthoc_results:
            significant_posthoc = [r for r in posthoc_results if (r.corrected_p or r.p_value) < 0.05]
            logger.info(f"Post-hoc Summary: {len(significant_posthoc)}/{len(posthoc_results)} comparisons significant (p < 0.05)")
