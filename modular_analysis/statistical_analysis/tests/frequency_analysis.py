"""
Statistical analysis for frequency-related measurements (current vs frequency, fold rheobase vs frequency).
"""

import pandas as pd
import numpy as np
from scipy.stats import ttest_ind, t
import os
import logging
import math
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

import statsmodels.stats.multitest as multi
from statsmodels.formula.api import mixedlm

from ...shared.data_models import ExperimentalDesign, StatisticalResult, DesignType
from ...shared.config import ProtocolConfig
from .posthoc_utils import (
    should_run_posthocs,
    get_simple_effect_comparisons,
    compute_lsmean_summaries,
    compute_global_curve_posthocs,
)

logger = logging.getLogger(__name__)


def _get_effect_pvalue(model, param_names: list, data: pd.DataFrame, formula: str, 
                       groups_col: str, re_formula: str, effect_name: str = "") -> float:
    """
    Get p-value for a block of fixed-effect parameters using a Wald test.
    (Extra arguments are kept for API compatibility but are unused.)
    """
    if not param_names:
        logger.warning(f"No parameters found for {effect_name}")
        return 1.0
    
    param_index = model.params.index
    present_params = [name for name in param_names if name in param_index]
    if len(present_params) != len(param_names):
        missing = set(param_names) - set(present_params)
        if missing:
            logger.debug(f"{effect_name}: missing parameters in model: {missing}")
    
    if not present_params:
        logger.warning(f"{effect_name}: No matching parameters for Wald test")
        return np.nan
    
    R = np.zeros((len(present_params), len(param_index)))
    for row_idx, name in enumerate(present_params):
        col_idx = param_index.get_loc(name)
        R[row_idx, col_idx] = 1.0
    
    try:
        wald_res = model.wald_test(R, scalar=True)
        p_value = float(np.asarray(wald_res.pvalue).squeeze())
        logger.debug(f"Wald test for {effect_name}: statistic={wald_res.statistic}, p={p_value:.4g}")
        return p_value
    except Exception as e:
        logger.error(f"Wald test failed for {effect_name}: {e}")
        return np.nan


class FrequencyAnalyzer:
    """Analyzes frequency vs current and frequency vs fold rheobase data."""
    
    def __init__(self, protocol_config: ProtocolConfig):
        self.protocol_config = protocol_config
        
    def analyze_current_vs_frequency(self, design: ExperimentalDesign, base_path: str) -> Dict[str, any]:
        """Analyze current vs frequency data with ANOVA and unified mixed-effects model."""
        
        results = {
            'point_by_point_stats': [],
            'mixed_effects_result': None,
            'success': False
        }
        
        try:
            # Load frequency data for all groups
            all_group_data = {}
            
            for group in design.groups:
                freq_file = os.path.join(base_path, "Results", f"Calc_{group.name}_frequency_vs_current.csv")
                
                if os.path.exists(freq_file):
                    try:
                        group_data = self._make_cvf_df(freq_file)
                        if not group_data.empty:
                            all_group_data[group.name] = group_data
                            logger.info(f"Loaded current vs frequency data for {group.name}")
                    except Exception as e:
                        logger.warning(f"Error loading frequency data for {group.name}: {e}")
                        
            if len(all_group_data) < 2:
                logger.warning("Need at least 2 groups with frequency data")
                return results
            
            # Point-by-point ANOVA for each current step (factorial or regular)
            point_stats = self._run_pointwise_anova_frequency(all_group_data, 'current', design)
            if point_stats:
                results['point_by_point_stats'] = point_stats
                self._save_point_by_point_results(point_stats, base_path, "Current", design)
            
            # Unified mixed-effects model (factorial or regular)
            mixed_effects_result = self._run_unified_mixed_effects_frequency(all_group_data, 'current', design)
            if mixed_effects_result:
                results['mixed_effects_result'] = mixed_effects_result
            
            results['success'] = True
            logger.info(f"Current vs frequency analysis completed for {len(all_group_data)} groups")
            
        except Exception as e:
            logger.error(f"Error in current vs frequency analysis: {e}")
            results['error'] = str(e)
            
        return results
    
    def analyze_fold_rheobase_vs_frequency(self, design: ExperimentalDesign, base_path: str) -> Dict[str, any]:
        """Analyze fold rheobase vs frequency data with ANOVA and unified mixed-effects model."""
        
        results = {
            'point_by_point_stats': [],
            'mixed_effects_result': None,
            'success': False
        }
        
        try:
            # Load fold rheobase data for all groups (integers only for point-by-point ANOVA)
            all_group_data_integers = {}
            
            for group in design.groups:
                freq_file = os.path.join(base_path, "Results", f"Calc_{group.name}_frequency_vs_current.csv")
                
                if os.path.exists(freq_file):
                    try:
                        # Convert to fold rheobase data (integers only for point-by-point)
                        group_data = self._make_rheo_df(freq_file, group.name, base_path, integers_only=True)
                        if not group_data.empty:
                            all_group_data_integers[group.name] = group_data
                            logger.info(f"Loaded fold rheobase vs frequency data for {group.name}")
                    except Exception as e:
                        logger.warning(f"Error loading fold rheobase data for {group.name}: {e}")
                        
            if len(all_group_data_integers) < 2:
                logger.warning("Need at least 2 groups with fold rheobase data")
                return results
            
            # Point-by-point ANOVA for each fold rheobase step (factorial or regular) - uses integer data
            point_stats = self._run_pointwise_anova_frequency(all_group_data_integers, 'fold_rheobase', design)
            if point_stats:
                results['point_by_point_stats'] = point_stats
                self._save_point_by_point_results(point_stats, base_path, "Fold_Rheobase", design)
            
            # Load ALL fold rheobase data (including fractional values) for mixed model - better convergence
            all_group_data_full = {}
            for group in design.groups:
                freq_file = os.path.join(base_path, "Results", f"Calc_{group.name}_frequency_vs_current.csv")
                if os.path.exists(freq_file):
                    try:
                        group_data = self._make_rheo_df(freq_file, group.name, base_path, integers_only=False)
                        if not group_data.empty:
                            all_group_data_full[group.name] = group_data
                            logger.info(f"Loaded FULL fold rheobase data for {group.name}: {len(group_data)} steps (vs {len(all_group_data_integers[group.name])} integer steps)")
                    except Exception as e:
                        logger.warning(f"Error loading full fold rheobase data for {group.name}: {e}")
            
            # Unified mixed-effects model (factorial or regular) - uses ALL data points
            if len(all_group_data_full) >= 2:
                mixed_effects_result = self._run_unified_mixed_effects_frequency(all_group_data_full, 'fold_rheobase', design)
                if mixed_effects_result:
                    results['mixed_effects_result'] = mixed_effects_result
            
            results['success'] = True
            logger.info(f"Fold rheobase vs frequency analysis completed for {len(all_group_data_integers)} groups")
            
        except Exception as e:
            logger.error(f"Error in fold rheobase vs frequency analysis: {e}")
            results['error'] = str(e)
            
        return results
    
    def _make_cvf_df(self, file_path: str) -> pd.DataFrame:
        """Make current vs frequency dataframe from file."""
        
        try:
            if not os.path.exists(file_path):
                logger.warning(f"File not found: {file_path}")
                return pd.DataFrame()
                
            c_vs_f_data = pd.read_csv(file_path, index_col=False)
            vals_df = c_vs_f_data.filter(regex='Values').copy()  # Explicit copy to avoid SettingWithCopyWarning
            val_cols = vals_df.columns
            vals_df['n'] = vals_df.count(axis=1)

            # Get current
            max_steps = vals_df.count().max()
            min_curr = self.protocol_config.min_current
            curr_step = self.protocol_config.step_size
            max_curr = min_curr + (curr_step * max_steps)
            currs = np.arange(min_curr, max_curr, curr_step)
            vals_df['Current'] = currs

            # Calculate mean and stderr
            vals_df['mean'] = vals_df[val_cols].mean(axis=1)
            vals_df['se'] = vals_df[val_cols].sem(axis=1)

            return vals_df
            
        except Exception as e:
            logger.error(f"Error creating current vs frequency data: {e}")
            return pd.DataFrame()
    
    def _make_rheo_df(self, file_path: str, group_name: str = "", base_path: str = "", integers_only: bool = True) -> pd.DataFrame:
        """
        Make fold rheobase dataframe from file.
        
        Args:
            file_path: Path to frequency vs current CSV
            group_name: Name of group (for saving)
            base_path: Base path (for saving)
            integers_only: If True, filter to integer fold rheobase 1-10 (for plots/point-by-point).
                          If False, include all fold rheobase values (for mixed model).
        """
        
        try:
            if not os.path.exists(file_path):
                logger.warning(f"File not found: {file_path}")
                return pd.DataFrame()
                
            c_vs_f_data = pd.read_csv(file_path, index_col=False)
            val_cols = c_vs_f_data.filter(regex='Values').columns
            curr_cols = c_vs_f_data.filter(regex='.abf').columns

            # Get current
            max_steps = c_vs_f_data.count().max()
            curr_step = self.protocol_config.step_size
            min_curr = self.protocol_config.min_current
            max_curr = min_curr + (curr_step * max_steps)
            currs = np.arange(min_curr, max_curr, curr_step)

            # Convert currents to fold rheobase
            i = 0
            rheobase = c_vs_f_data[val_cols].ne(0).idxmax()  # get index of first non-zero frequency for each cell
            for col in curr_cols:
                c_vs_f_data.loc[:, col] = currs
                c_vs_f_data.loc[:, col] /= ((rheobase.iloc[i] * curr_step) + min_curr)
                c_vs_f_data.loc[:, col] = c_vs_f_data.loc[:, col].shift(periods=-rheobase.iloc[i])
                c_vs_f_data.loc[:, val_cols[i]] = c_vs_f_data.loc[:, val_cols[i]].shift(periods=-rheobase.iloc[i])
                i += 1
            
            # Save fold-rheobase data (like original)
            if group_name and base_path:
                fold_rheo_path = os.path.join(base_path, "Results", f"Calc_{group_name}_frequency_vs_fold_rheobase.csv")
                c_vs_f_data.to_csv(fold_rheo_path, index=False)
            
            # Make fold-rheobase data for analysis
            points = []
            for i in range(len(curr_cols)):
                for j in range(c_vs_f_data.loc[:, curr_cols[i]].shape[0]):
                    curr = c_vs_f_data.loc[:, curr_cols[i]][j]
                    points.append((curr, c_vs_f_data.loc[:, val_cols[i]][j]))

            point_dict = defaultdict(list)
            folds = []
            for i, j in points:
                if integers_only:
                    # Only integer folds 1-10 (for plotting and point-by-point ANOVA)
                    if i in [1,2,3,4,5,6,7,8,9,10]:
                        point_dict[i].append(j)
                        folds.append(i)
                else:
                    # All fold rheobase values (for mixed model - better convergence)
                    if not np.isnan(i) and not np.isnan(j):
                        point_dict[i].append(j)
                        folds.append(i)

            folds = list(set(folds))
            folds.sort()
            final_df = pd.DataFrame.from_dict(point_dict, orient='index')
            new_col_names = []
            for i in range(len(final_df.columns)):
                new_col_names.append("Values_{}".format(i))
            final_df.columns = new_col_names

            # Calculate mean and stderr
            final_df['n'] = final_df[new_col_names].count(axis=1)
            final_df['mean'] = final_df[new_col_names].mean(axis=1)
            final_df['se'] = final_df[new_col_names].sem(axis=1)

            final_df.insert(0, 'Fold Rheobase', folds)
            
            return final_df
            
        except Exception as e:
            logger.error(f"Error creating fold rheobase data: {e}")
            return pd.DataFrame()
    
    def _run_point_by_point_tests(self, df_group1: pd.DataFrame, df_group2: pd.DataFrame, 
                                 group1_name: str, group2_name: str, x_name: str) -> List[Dict]:
        """Run t-tests at each x-axis point with FDR correction."""
        
        # FDR corrected stats
        num_tests = min(df_group1.shape[0], df_group2.shape[0])
        group1_vals = df_group1.filter(regex="Values")
        group2_vals = df_group2.filter(regex="Values")
        stat_data = []
        
        for i in range(num_tests):
            try:
                t_stat, p_val = ttest_ind(
                    np.array(group1_vals.iloc[i]), np.array(group2_vals.iloc[i]),
                    nan_policy="omit", equal_var=False
                )
                
                if not np.isnan(p_val):
                    stat_data.append({
                        x_name: df_group1[x_name].iloc[i],
                        f"{group2_name}_mean": df_group2['mean'].iloc[i],
                        f"{group2_name}_stderr": df_group2['se'].iloc[i],
                        f"{group2_name}_n": df_group2['n'].iloc[i],
                        f"{group1_name}_mean": df_group1['mean'].iloc[i],
                        f"{group1_name}_stderr": df_group1['se'].iloc[i],
                        f"{group1_name}_n": df_group1['n'].iloc[i],
                        "p-value": p_val
                    })
                    
            except Exception as e:
                logger.warning(f"Error in t-test at {x_name} point {i}: {e}")
                continue
        
        # Apply FDR correction (handling NaN p-values)
        if stat_data:
            if len(stat_data) > 1:
                # Extract p-values and track valid indices
                valid_indices = [i for i, s in enumerate(stat_data) if not np.isnan(s["p-value"])]
                valid_p_values = [stat_data[i]["p-value"] for i in valid_indices]
                
                if len(valid_p_values) > 1:
                    rejected, corrected_p, _, _ = multi.multipletests(valid_p_values, method="fdr_bh")
                    # Map corrected p-values back to valid indices
                    for i, corrected_val in zip(valid_indices, corrected_p):
                        stat_data[i]["corrected_p"] = corrected_val
                    # Set NaN for invalid p-values
                    for i in range(len(stat_data)):
                        if i not in valid_indices:
                            stat_data[i]["corrected_p"] = np.nan
                else:
                    # Only 1 or 0 valid p-values - copy raw p-values for valid tests
                    for idx in valid_indices:
                        stat_data[idx]["corrected_p"] = stat_data[idx]["p-value"]
                    for i in range(len(stat_data)):
                        if i not in valid_indices:
                            stat_data[i]["corrected_p"] = np.nan
            else:
                # Single result - copy raw p-value if valid
                for s in stat_data:
                    s["corrected_p"] = s["p-value"] if not np.isnan(s["p-value"]) else np.nan
        
        return stat_data
    
    def _save_point_by_point_results(self, stat_data: List[Dict], base_path: str, x_name: str, 
                                     design: ExperimentalDesign = None) -> None:
        """Save point-by-point statistical results to CSV."""
        
        if not stat_data:
            logger.warning("No statistical data to save")
            return
        
        # Check if factorial design
        is_factorial = design and design.design_type == DesignType.FACTORIAL_2X2
            
        # Separate ANOVA and post-hoc results for proper column organization
        if is_factorial:
            anova_results = [r for r in stat_data if r.get('Test_Type') == 'Two-Way ANOVA']
        else:
            anova_results = [r for r in stat_data if r.get('Test_Type') == 'ANOVA']
        posthoc_results = [r for r in stat_data if r.get('Test_Type') == 'Post-hoc t-test']
        
        # Save ANOVA results
        if anova_results:
            anova_df = pd.DataFrame(anova_results)
            
            if is_factorial:
                # For factorial: one row per step with all effects as columns (matching Stats_parameters structure)
                base_cols = ['Step_Value']
                
                # Add group statistics columns (mean, SEM, n for each group) - FIRST after Step_Value
                # Extract group names and sort them, then add mean, SEM, n for each group in that order
                group_names_set = set()
                for col in anova_df.columns:
                    if col.endswith('_mean'):
                        group_names_set.add(col.replace('_mean', ''))
                
                group_stat_cols = []
                for group_name in sorted(group_names_set):
                    if f'{group_name}_mean' in anova_df.columns:
                        group_stat_cols.append(f'{group_name}_mean')
                    if f'{group_name}_SEM' in anova_df.columns:
                        group_stat_cols.append(f'{group_name}_SEM')
                    if f'{group_name}_n' in anova_df.columns:
                        group_stat_cols.append(f'{group_name}_n')
                
                # Find effect columns dynamically, ensuring p_value comes before corrected_p
                effect_names = set()
                for col in anova_df.columns:
                    # Check _corrected_p FIRST since it also ends with _p
                    if col.endswith('_corrected_p'):
                        effect_names.add(col.replace('_corrected_p', ''))
                    elif col.endswith('_p'):
                        effect_names.add(col.replace('_p', ''))
                
                # Build columns in correct order: effect_p, then effect_corrected_p
                effect_cols = []
                for effect in sorted(effect_names):
                    if f'{effect}_p' in anova_df.columns:
                        effect_cols.append(f'{effect}_p')
                    if f'{effect}_corrected_p' in anova_df.columns:
                        effect_cols.append(f'{effect}_corrected_p')
                
                # Order: Step_Value -> Group stats -> ANOVA effects
                available_cols = base_cols + group_stat_cols + effect_cols
            else:
                # For one-way ANOVA: simpler format (matching Stats_parameters structure)
                base_cols = ['Step_Value']
                
                # Add group statistics columns (mean, SEM, n for each group) - FIRST after Step_Value
                # Extract group names and sort them, then add mean, SEM, n for each group in that order
                group_names_set = set()
                for col in anova_df.columns:
                    if col.endswith('_mean'):
                        group_names_set.add(col.replace('_mean', ''))
                
                group_stat_cols = []
                for group_name in sorted(group_names_set):
                    if f'{group_name}_mean' in anova_df.columns:
                        group_stat_cols.append(f'{group_name}_mean')
                    if f'{group_name}_SEM' in anova_df.columns:
                        group_stat_cols.append(f'{group_name}_SEM')
                    if f'{group_name}_n' in anova_df.columns:
                        group_stat_cols.append(f'{group_name}_n')
                
                # ANOVA statistics last
                stat_cols = ['F_statistic', 'p_value']
                if 'corrected_p' in anova_df.columns:
                    stat_cols.append('corrected_p')
                
                # Order: Step_Value -> Group stats -> ANOVA stats
                available_cols = [col for col in base_cols + group_stat_cols + stat_cols if col in anova_df.columns]
            
            anova_df = anova_df[available_cols]
            
            anova_path = os.path.join(base_path, "Results", f"Stats_{x_name}_vs_frequency_each_point_ANOVA.csv")
            anova_df.to_csv(anova_path, index=False)
            logger.info(f"Saved ANOVA results to {anova_path}")
        
        # Save post-hoc results with organized columns - only if more than 2 groups
        # Count unique groups from the data
        unique_groups = set()
        for result in stat_data:
            if 'Group1' in result:
                unique_groups.add(result['Group1'])
            if 'Group2' in result:
                unique_groups.add(result['Group2'])
        
        if posthoc_results and len(unique_groups) > 2:
            posthoc_df = pd.DataFrame(posthoc_results)
            
            # Organize columns: step info, comparison, group stats, then p-values
            base_cols = ['Step_Value', 'Step_Label', 'Test_Type', 'Comparison']
            group_cols = []
            stat_cols = ['t_statistic', 'p_value']
            if 'corrected_p' in posthoc_df.columns:
                stat_cols.append('corrected_p')
            
            # Columns to exclude from group_cols
            exclude_cols = base_cols + stat_cols + ['measurement_type']
            
            # Get group-related columns dynamically
            for col in posthoc_df.columns:
                if col not in exclude_cols and ('Group' in col or '_mean' in col or '_stderr' in col or '_n' in col):
                    group_cols.append(col)
            
            # Reorder columns: base + groups (sorted) + statistics (don't include measurement_type)
            column_order = base_cols + sorted(group_cols) + stat_cols
            available_cols = [col for col in column_order if col in posthoc_df.columns]
            posthoc_df = posthoc_df[available_cols]
            
            posthoc_path = os.path.join(base_path, "Results", f"Stats_{x_name}_vs_frequency_each_point_pairwise.csv")
            posthoc_df.to_csv(posthoc_path, index=False)
            logger.info(f"Saved pairwise results to {posthoc_path}")
        elif len(unique_groups) <= 2:
            logger.info(f"Skipping pairwise CSV creation - only {len(unique_groups)} groups")
        
        logger.info(f"Saved organized {x_name} vs frequency statistics")
    
    def _run_pointwise_anova_frequency(self, all_group_data: Dict[str, pd.DataFrame], analysis_type: str, 
                                       design: ExperimentalDesign = None) -> List[Dict]:
        """Run point-by-point ANOVA for each current/fold rheobase step with post-hoc if significant.
        Handles both regular one-way ANOVA and two-way ANOVA for factorial designs."""
        from scipy.stats import f_oneway
        from itertools import combinations
        import statsmodels.stats.multitest as multi
        from statsmodels.stats.anova import anova_lm
        from statsmodels.formula.api import ols
        
        # Check if this is a factorial design
        is_factorial = design and design.design_type == DesignType.FACTORIAL_2X2
        
        all_results = []
        group_names = list(all_group_data.keys())
        
        logger.info(f"Running point-by-point ANOVA for {analysis_type} across {len(group_names)} groups")
        
        # Get the number of measurement steps
        max_steps = max(len(df) for df in all_group_data.values())
        
        # Run ANOVA for each step
        for step_idx in range(max_steps):
            # Determine step value based on analysis type
            if analysis_type == 'current':
                step_value = self.protocol_config.min_current + (step_idx * self.protocol_config.step_size)
                if step_value <= 0:
                    continue
                step_label = 'Current (pA)'
            else:  # fold_rheobase
                # For fold rheobase, need to get the actual fold values from the data
                first_group_data = list(all_group_data.values())[0]
                if 'Fold Rheobase' in first_group_data.columns and step_idx < len(first_group_data):
                    step_value = first_group_data['Fold Rheobase'].iloc[step_idx]
                else:
                    step_value = 0.5 + (step_idx * 0.5)  # Default fallback
                step_label = 'Fold Rheobase'
            
            # Collect data for this step from all groups
            group_data_lists = []
            group_stats = {}
            
            for group_name in group_names:
                group_data = all_group_data[group_name]
                if step_idx < len(group_data):
                    # Get frequency values for this step (across all cells)
                    value_cols = [col for col in group_data.columns if 'Value' in col]
                    step_data = group_data.iloc[step_idx][value_cols].dropna()
                    
                    if len(step_data) > 1:
                        group_data_lists.append(step_data)
                        group_stats[group_name] = {
                            'mean': step_data.mean(),
                            'stderr': step_data.std() / np.sqrt(len(step_data)),
                            'n': len(step_data)
                        }
            
            if len(group_data_lists) < 2:
                continue
                
            # Run ANOVA (one-way or two-way depending on design)
            try:
                if is_factorial:
                    # Build dataframe with factor labels for two-way ANOVA
                    data_list = []
                    for group_name in group_names:
                        if group_name in all_group_data:
                            group_data = all_group_data[group_name]
                            if step_idx < len(group_data):
                                value_cols = [col for col in group_data.columns if 'Value' in col]
                                step_data = group_data.iloc[step_idx][value_cols].dropna()
                                
                                factor1_level = design.factor_mapping[group_name]['factor1']
                                factor2_level = design.factor_mapping[group_name]['factor2']
                                
                                for val in step_data:
                                    data_list.append({
                                        'frequency': val,
                                        'factor1': factor1_level,
                                        'factor2': factor2_level
                                    })
                    
                    if len(data_list) >= 4:
                        df = pd.DataFrame(data_list)
                        
                        # Check we have data for all factor combinations
                        factor_combos = df.groupby(['factor1', 'factor2']).size()
                        factor1_levels_count = df['factor1'].nunique()
                        factor2_levels_count = df['factor2'].nunique()
                        expected_combos = factor1_levels_count * factor2_levels_count
                        
                        if len(factor_combos) < expected_combos:
                            logger.warning(f"Missing factor combinations at {step_label} {step_value} (have {len(factor_combos)}/{expected_combos})")
                            continue
                        
                        model = ols('frequency ~ C(factor1) * C(factor2)', data=df).fit()
                        anova_table = anova_lm(model, typ=2)
                        
                        factor1_p = anova_table.loc['C(factor1)', 'PR(>F)']
                        factor2_p = anova_table.loc['C(factor2)', 'PR(>F)']
                        interaction_p = anova_table.loc['C(factor1):C(factor2)', 'PR(>F)']
                        
                        # Create one result per step with all effects as columns (matching dependent format)
                        cell_stats_df = (
                            df.groupby(['factor1', 'factor2'])['frequency']
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

                        result = {
                            'Step_Value': step_value,
                            'Step_Label': step_label,
                            'Test_Type': 'Two-Way ANOVA',
                            f'{design.factor1_name}_p': factor1_p,
                            f'{design.factor2_name}_p': factor2_p,
                            'Interaction_p': interaction_p,
                            'measurement_type': f'frequency_{analysis_type}',
                            '_run_posthoc': True,  # Mark for potential post-hoc if interaction significant
                            '_mse_resid': model.mse_resid,
                            '_df_resid': int(model.df_resid),
                            '_cell_stats': cell_stats,
                            '_factor1_levels': sorted(df['factor1'].unique()),
                            '_factor2_levels': sorted(df['factor2'].unique())
                        }
                        
                        # Add group statistics (mean, SEM, n for each group)
                        for group_name in sorted(group_stats.keys()):
                            result[f'{group_name}_mean'] = group_stats[group_name]['mean']
                            result[f'{group_name}_SEM'] = group_stats[group_name]['stderr']
                            result[f'{group_name}_n'] = group_stats[group_name]['n']
                        
                        all_results.append(result)
                else:
                    # Regular one-way ANOVA
                    f_stat, anova_p = f_oneway(*group_data_lists)
                    
                    # Only include results with valid p-values (like t-test approach)
                    if not np.isnan(anova_p):
                        # Create ANOVA result
                        anova_result = {
                            'Step_Value': step_value,
                            'Step_Label': step_label,
                            'Test_Type': 'ANOVA',
                            'Comparison': f"Overall ({' vs '.join(group_names)})",
                            'F_statistic': f_stat,
                            'p_value': anova_p,
                            'measurement_type': f'frequency_{analysis_type}'
                        }
                        
                        # Add group statistics (mean, SEM, n for each group)
                        for group_name in sorted(group_stats.keys()):
                            anova_result[f'{group_name}_mean'] = group_stats[group_name]['mean']
                            anova_result[f'{group_name}_SEM'] = group_stats[group_name]['stderr']
                            anova_result[f'{group_name}_n'] = group_stats[group_name]['n']
                        
                        all_results.append(anova_result)
                        
                        # Store this for later post-hoc decision (will check corrected p-value)
                        anova_result['_run_posthoc'] = len(group_names) > 2  # Mark for potential post-hoc
                        
                        if len(group_names) <= 2:
                            # For 2 groups, ANOVA directly tests the difference
                            if anova_p < 0.05:
                                logger.info(f"{step_label} {step_value} ANOVA significant (p={anova_p:.4f}) - no post-hoc needed (only 2 groups)")
                            else:
                                logger.info(f"{step_label} {step_value} ANOVA not significant (p={anova_p:.4f})")
                        else:
                            logger.info(f"{step_label} {step_value} ANOVA (p={anova_p:.4f}) - will check corrected p-value for post-hoc")
                    else:
                        logger.info(f"{step_label} {step_value} ANOVA returned NaN p-value - skipping")
                    
            except Exception as e:
                logger.warning(f"Error running ANOVA for {step_label} {step_value}: {e}")
                continue
        
        # Apply FDR correction to ANOVA results first
        if all_results:
            self._apply_fdr_correction_frequency(all_results)
            
            # Now run post-hoc tests based on corrected ANOVA p-values
            posthoc_results = []
            for result in all_results:
                # For factorial designs, check any effect; for one-way, check main ANOVA
                should_run_posthoc = False
                posthoc_decision = None
                
                if result.get('Test_Type') == 'Two-Way ANOVA':
                    # For factorial, use helper function to determine if post-hocs needed
                    posthoc_decision = should_run_posthocs(result, design, group_names)
                    should_run_posthoc = posthoc_decision['run_posthocs']
                elif result.get('Test_Type') == 'ANOVA':
                    # For one-way ANOVA with 3+ groups
                    corrected_p = result.get('corrected_p', result.get('p_value', 1))
                    should_run_posthoc = result.get('_run_posthoc', False) and corrected_p < 0.05
                
                if should_run_posthoc:
                    step_value = result['Step_Value']
                    step_label = result['Step_Label']
                    
                    # Log why we're running post-hocs
                    if posthoc_decision:
                        logger.info(f"{step_label} {step_value}: Running post-hocs for {posthoc_decision['reasons']}")
                    else:
                        corrected_p = result.get('corrected_p', result.get('p_value', 1))
                        logger.info(f"{step_label} {step_value} ANOVA significant (corrected p={corrected_p:.4f}) - running post-hoc tests")
                    
                    # Find the corresponding step data for post-hoc tests
                    step_idx = int((step_value - self.protocol_config.min_current) / self.protocol_config.step_size) if analysis_type == 'current' else int(step_value - 1)
                    
                    if step_idx < max_steps:
                        # Collect data for this step from all groups
                        step_group_data = {}
                        step_group_stats = {}
                        
                        for group_name in group_names:
                            group_data = all_group_data[group_name]
                            if step_idx < len(group_data):
                                value_cols = [col for col in group_data.columns if 'Value' in col]
                                step_data = group_data.iloc[step_idx][value_cols].dropna()
                                
                                if len(step_data) > 1:
                                    step_group_data[group_name] = step_data
                                    step_group_stats[group_name] = {
                                        'mean': step_data.mean(),
                                        'stderr': step_data.std() / np.sqrt(len(step_data)),
                                        'n': len(step_data)
                                    }
                        
                        if result.get('Test_Type') == 'Two-Way ANOVA':
                            reasons = posthoc_decision.get('reasons', []) if posthoc_decision else []
                            if 'interaction' in reasons:
                                comparisons_to_run = get_simple_effect_comparisons(design, list(step_group_data.keys()))
                                logger.info(f"Running {len(comparisons_to_run)} simple effects comparisons")
                                
                                for group1_name, group2_name in comparisons_to_run:
                                    if group1_name in step_group_data and group2_name in step_group_data:
                                        t_stat, pairwise_p = ttest_ind(
                                            step_group_data[group1_name], step_group_data[group2_name],
                                            nan_policy="omit", equal_var=False
                                        )
                                        
                                        pairwise_result = {
                                            'Step_Value': step_value,
                                            'Step_Label': step_label,
                                            'Test_Type': 'Post-hoc t-test',
                                            'Comparison': f"{group1_name} vs {group2_name}",
                                            'Group1': group1_name,
                                            'Group1_mean': step_group_stats[group1_name]['mean'],
                                            'Group1_stderr': step_group_stats[group1_name]['stderr'],
                                            'Group1_n': step_group_stats[group1_name]['n'],
                                            'Group2': group2_name,
                                            'Group2_mean': step_group_stats[group2_name]['mean'],
                                            'Group2_stderr': step_group_stats[group2_name]['stderr'],
                                            'Group2_n': step_group_stats[group2_name]['n'],
                                            't_statistic': t_stat,
                                            'p_value': pairwise_p,
                                            'measurement_type': f'frequency_{analysis_type}'
                                        }
                                        posthoc_results.append(pairwise_result)
                            else:
                                mse = result.get('_mse_resid')
                                df_resid = result.get('_df_resid')
                                cell_stats = result.get('_cell_stats')
                                factor1_levels = result.get('_factor1_levels')
                                factor2_levels = result.get('_factor2_levels')
                                if 'factor1_main' in reasons:
                                    marginal = self._run_frequency_marginal_posthoc(
                                        step_value, step_label, f'frequency_{analysis_type}',
                                        design, 'factor1', mse, df_resid,
                                        cell_stats, factor1_levels, factor2_levels
                                    )
                                    posthoc_results.extend(marginal)
                                if 'factor2_main' in reasons:
                                    marginal = self._run_frequency_marginal_posthoc(
                                        step_value, step_label, f'frequency_{analysis_type}',
                                        design, 'factor2', mse, df_resid,
                                        cell_stats, factor1_levels, factor2_levels
                                    )
                                    posthoc_results.extend(marginal)
                        else:
                            from itertools import combinations
                            comparisons_to_run = list(combinations(step_group_data.keys(), 2))
                            
                            for group1_name, group2_name in comparisons_to_run:
                                if group1_name in step_group_data and group2_name in step_group_data:
                                    t_stat, pairwise_p = ttest_ind(
                                        step_group_data[group1_name], step_group_data[group2_name],
                                        nan_policy="omit", equal_var=False
                                    )
                                    
                                    pairwise_result = {
                                        'Step_Value': step_value,
                                        'Step_Label': step_label,
                                        'Test_Type': 'Post-hoc t-test',
                                        'Comparison': f"{group1_name} vs {group2_name}",
                                        'Group1': group1_name,
                                        'Group1_mean': step_group_stats[group1_name]['mean'],
                                        'Group1_stderr': step_group_stats[group1_name]['stderr'],
                                        'Group1_n': step_group_stats[group1_name]['n'],
                                        'Group2': group2_name,
                                        'Group2_mean': step_group_stats[group2_name]['mean'],
                                        'Group2_stderr': step_group_stats[group2_name]['stderr'],
                                        'Group2_n': step_group_stats[group2_name]['n'],
                                        't_statistic': t_stat,
                                        'p_value': pairwise_p,
                                        'measurement_type': f'frequency_{analysis_type}'
                                    }
                                    posthoc_results.append(pairwise_result)
            
            # Add post-hoc results to all results and apply FDR correction to them
            all_results.extend(posthoc_results)
            if posthoc_results:
                self._apply_fdr_correction_frequency(all_results)  # Re-apply to include post-hoc
        
        logger.info(f"Completed point-by-point ANOVA for {analysis_type}: {len(all_results)} statistical tests")
        return all_results

    def _run_frequency_marginal_posthoc(self, step_value: float, step_label: str, measurement_type: str,
                                        design: ExperimentalDesign, factor_key: str, mse: Optional[float],
                                        df_resid: Optional[int], cell_stats: Optional[Dict],
                                        factor1_levels: Optional[List[str]], factor2_levels: Optional[List[str]]) -> List[Dict]:
        """Run marginal mean comparisons for factorial frequency analyses."""
        if mse is None or not np.isfinite(mse):
            return []
        if df_resid is None or df_resid <= 0:
            return []
        if not cell_stats:
            return []

        factor_label = design.factor1_name if factor_key == 'factor1' else design.factor2_name
        target_levels = factor1_levels if factor_key == 'factor1' else factor2_levels
        other_levels = factor2_levels if factor_key == 'factor1' else factor1_levels

        level_summaries = compute_lsmean_summaries(
            cell_stats,
            target_levels or [],
            other_levels or [],
            mse,
            factor_key,
        )

        if len(level_summaries) < 2:
            return []

        from itertools import combinations
        results = []
        levels = sorted(level_summaries.keys())

        for level1, level2 in combinations(levels, 2):
            stats1 = level_summaries[level1]
            stats2 = level_summaries[level2]

            if stats1['count'] == 0 or stats2['count'] == 0:
                continue

            se_diff = math.sqrt(stats1['variance'] + stats2['variance'])
            if not np.isfinite(se_diff) or se_diff == 0:
                continue

            diff = stats1['mean'] - stats2['mean']
            t_statistic = diff / se_diff
            p_value = 2 * t.sf(abs(t_statistic), df_resid)

            result = {
                'Step_Value': step_value,
                'Step_Label': step_label,
                'Test_Type': 'Marginal t-test',
                'Comparison': f"{factor_label}={level1} vs {factor_label}={level2}",
                'Group1': f"{factor_label}={level1}",
                'Group1_mean': stats1['mean'],
                'Group1_stderr': math.sqrt(stats1['variance']),
                'Group1_n': stats1['count'],
                'Group2': f"{factor_label}={level2}",
                'Group2_mean': stats2['mean'],
                'Group2_stderr': math.sqrt(stats2['variance']),
                'Group2_n': stats2['count'],
                't_statistic': t_statistic,
                'p_value': p_value,
                'measurement_type': measurement_type
            }
            results.append(result)

        return results
    
    def _run_unified_mixed_effects_frequency(self, all_group_data: Dict[str, pd.DataFrame], analysis_type: str, 
                                             design: ExperimentalDesign = None) -> Optional[Dict]:
        """Run unified mixed-effects model for frequency data with all groups and proper cell IDs.
        Handles both regular mixed-effects and factorial designs with 3-way interactions."""
        
        # Check if factorial design
        is_factorial = design and design.design_type == DesignType.FACTORIAL_2X2
        
        try:
            # Combine data from all groups with consistent cell IDs
            steps = []
            frequencies = []
            groups_list = []
            cell_ids = []
            
            # For factorial designs, also track factor levels
            if is_factorial:
                factor1_list = []
                factor2_list = []
            
            for group_name, group_data in all_group_data.items():
                # Get value columns (cells)
                value_cols = [col for col in group_data.columns if 'Value' in col]
                
                # For each cell, add all its measurements across steps
                for cell_idx, col in enumerate(value_cols):
                    if col in group_data.columns:
                        # Create consistent cell ID across groups
                        cell_id = f"{group_name}_{cell_idx}"
                        
                        # Add data for this cell across all steps
                        for step_idx, freq_val in enumerate(group_data[col].dropna()):
                            if analysis_type == 'current':
                                step_value = self.protocol_config.min_current + (step_idx * self.protocol_config.step_size)
                                # CRITICAL: Match original behavior - only include Current > 0 in mixed effects model
                                if step_value <= 0:
                                    continue
                            else:  # fold_rheobase
                                # Get fold rheobase value from data or use default
                                if 'Fold Rheobase' in group_data.columns and step_idx < len(group_data):
                                    step_value = group_data['Fold Rheobase'].iloc[step_idx]
                                else:
                                    step_value = 0.5 + (step_idx * 0.5)
                                
                            steps.append(step_value)
                            frequencies.append(freq_val)
                            groups_list.append(group_name)
                            cell_ids.append(cell_id)
                            
                            # For factorial designs, add factor levels
                            if is_factorial:
                                factor1_list.append(design.factor_mapping[group_name]['factor1'])
                                factor2_list.append(design.factor_mapping[group_name]['factor2'])
            
            if len(set(groups_list)) < 2:
                logger.warning("Need at least 2 groups for mixed-effects model")
                return None
                
            # Create unified DataFrame
            step_col_name = 'Current' if analysis_type == 'current' else 'FoldRheobase'
            
            if is_factorial:
                final_df = pd.DataFrame({
                    step_col_name: steps,
                    'Frequency': frequencies,
                    'Factor1': factor1_list,
                    'Factor2': factor2_list,
                    'Cell_id': cell_ids
                }).dropna()
                
                # Standardize continuous variable (z-score) to help convergence
                step_col_z = f"{step_col_name}_z"
                final_df[step_col_z] = (final_df[step_col_name] - final_df[step_col_name].mean()) / final_df[step_col_name].std()
                
                logger.info(f"Factorial mixed-effects model ({analysis_type}): {len(final_df)} observations, {len(final_df['Cell_id'].unique())} cells")
                logger.info(f"Running factorial mixed-effects model with random slopes for standardized {step_col_name}...")
                
                # 3-way interaction model for factorial design with standardized variable
                formula = f"Frequency ~ C(Factor1) * C(Factor2) * {step_col_z}"
                model = mixedlm(formula, final_df, groups=final_df["Cell_id"],
                              re_formula=f"1 + {step_col_z}").fit(method='lbfgs', maxiter=5000)
                
                # Check convergence
                if not model.converged:
                    logger.warning(f"Factorial frequency model ({analysis_type}) did not converge after 5000 iterations")
                else:
                    logger.info(f"Factorial frequency model ({analysis_type}) converged successfully")
                
                # Extract p-values for all 7 effects using proper statistical tests (LRT for multiple contrasts)
                factor1_params = [p for p in model.pvalues.index if 'C(Factor1)' in p and ':' not in p]
                factor1_p = _get_effect_pvalue(model, factor1_params, final_df, formula, 'Cell_id', f"1 + {step_col_z}", "Factor1")
                
                factor2_params = [p for p in model.pvalues.index if 'C(Factor2)' in p and ':' not in p]
                factor2_p = _get_effect_pvalue(model, factor2_params, final_df, formula, 'Cell_id', f"1 + {step_col_z}", "Factor2")
                
                step_p = model.pvalues.get(step_col_z, 1.0)
                
                # Two-way interactions - use LRT for multiple contrasts
                f1_f2_params = [p for p in model.pvalues.index if 'C(Factor1)' in p and 'C(Factor2)' in p and step_col_z not in p]
                f1_f2_p = _get_effect_pvalue(model, f1_f2_params, final_df, formula, 'Cell_id', f"1 + {step_col_z}", "Factor1:Factor2")
                
                f1_step_params = [p for p in model.pvalues.index if 'C(Factor1)' in p and step_col_z in p and 'C(Factor2)' not in p]
                f1_step_p = _get_effect_pvalue(model, f1_step_params, final_df, formula, 'Cell_id', f"1 + {step_col_z}", f"Factor1:{step_col_name}")
                
                f2_step_params = [p for p in model.pvalues.index if 'C(Factor2)' in p and step_col_z in p and 'C(Factor1)' not in p]
                f2_step_p = _get_effect_pvalue(model, f2_step_params, final_df, formula, 'Cell_id', f"1 + {step_col_z}", f"Factor2:{step_col_name}")
                
                # Three-way interaction
                three_way_params = [p for p in model.pvalues.index if 'C(Factor1)' in p and 'C(Factor2)' in p and step_col_z in p]
                three_way_p = _get_effect_pvalue(model, three_way_params, final_df, formula, 'Cell_id', f"1 + {step_col_z}", f"Factor1:Factor2:{step_col_name}")
                
                x_var_label = 'Current' if analysis_type == 'current' else 'Fold Rheobase'
                
                # Create result with all 7 effects
                result = {
                    'Effect': [
                        design.factor1_name,
                        design.factor2_name,
                        x_var_label,
                        f'{design.factor1_name}:{design.factor2_name}',
                        f'{design.factor1_name}:{x_var_label}',
                        f'{design.factor2_name}:{x_var_label}',
                        f'{design.factor1_name}:{design.factor2_name}:{x_var_label}'
                    ],
                    'p-value': [
                        factor1_p,
                        factor2_p,
                        step_p,
                        f1_f2_p,
                        f1_step_p,
                        f2_step_p,
                        three_way_p
                    ]
                }
                
                logger.info(f"Factorial mixed-effects {analysis_type}: F1 p={factor1_p:.4f}, F2 p={factor2_p:.4f}, 3-way interaction p={three_way_p:.4f}")
                
                factor1_levels_unique = sorted(final_df['Factor1'].dropna().unique())
                factor2_levels_unique = sorted(final_df['Factor2'].dropna().unique())
                analysis_label = 'Current_vs_Frequency' if analysis_type == 'current' else 'Fold_Rheobase_vs_Frequency'
                posthoc_rows = compute_global_curve_posthocs(
                    model=model,
                    step_col_z=step_col_z,
                    analysis_label=analysis_label,
                    factor_a_col='Factor1',
                    factor_b_col='Factor2',
                    factor_a_label=design.factor1_name,
                    factor_b_label=design.factor2_name,
                    factor_a_levels=factor1_levels_unique,
                    factor_b_levels=factor2_levels_unique,
                    main_a_p=factor1_p,
                    main_b_p=factor2_p,
                    interaction_ab_p=f1_f2_p,
                    interaction_a_step_p=f1_step_p,
                    interaction_b_step_p=f2_step_p,
                    interaction_three_way_p=three_way_p
                )
                
                return {
                    'effects': result,
                    'posthocs': posthoc_rows
                }
                
            else:
                # Regular non-factorial design
                final_df = pd.DataFrame({
                    step_col_name: steps,
                    'Frequency': frequencies,
                    'Genotype': groups_list,
                    'Cell_id': cell_ids
                }).dropna()
                
                # Standardize continuous variable (z-score) to help convergence
                step_col_z = f"{step_col_name}_z"
                final_df[step_col_z] = (final_df[step_col_name] - final_df[step_col_name].mean()) / final_df[step_col_name].std()
                
                logger.info(f"Unified mixed-effects model ({analysis_type}): {len(final_df)} observations, {len(final_df['Cell_id'].unique())} cells, {len(set(groups_list))} groups")
                logger.info(f"Running unified mixed-effects model with random slopes for standardized {step_col_name}...")
                
                # Unified mixed-effects model: Genotype effect, step effect, and interaction with random slopes
                formula = f"Frequency ~ C(Genotype) * {step_col_z}"
                model = mixedlm(formula, final_df, groups=final_df["Cell_id"],
                              re_formula=f"1 + {step_col_z}").fit(method='lbfgs', maxiter=5000)
                
                # Check convergence
                if not model.converged:
                    logger.warning(f"Unified frequency model ({analysis_type}) did not converge after 5000 iterations")
                else:
                    logger.info(f"Unified frequency model ({analysis_type}) converged successfully")
                
                # Extract overall Genotype effect p-value using proper statistical test (LRT for 3+)
                genotype_params = [p for p in model.pvalues.index if 'C(Genotype)' in p and ':' not in p]
                overall_group_p = _get_effect_pvalue(model, genotype_params, final_df, formula, 'Cell_id', f"1 + {step_col_z}", "Genotype")
                
                # Format results like original system (DataFrame with Effect and p-value columns)
                if analysis_type == 'current':
                    x_var_label = 'Current'
                else:  # fold_rheobase
                    x_var_label = 'Fold Rheobase'
                
                # Get interaction p-value (use LRT for 3+ contrasts)
                interaction_params = [p for p in model.pvalues.index if 'C(Genotype)' in p and step_col_z in p]
                interaction_p = _get_effect_pvalue(model, interaction_params, final_df, formula, 'Cell_id', f"1 + {step_col_z}", f"Genotype:{x_var_label}")
                
                # Create DataFrame in original format (use standardized variable name)
                result = {
                    'Effect': ['Genotype', x_var_label, f'Genotype:{x_var_label}'],
                    'p-value': [
                        overall_group_p,
                        model.pvalues.get(step_col_z, 1.0),
                        interaction_p
                    ]
                }
                
                logger.info(f"Mixed-effects {analysis_type}: Group p={overall_group_p:.4f}, {step_col_name} p={model.pvalues.get(step_col_z, 1.0):.4f}")
                
                group_levels = sorted(final_df['Genotype'].dropna().unique())
                analysis_label = 'Current_vs_Frequency' if analysis_type == 'current' else 'Fold_Rheobase_vs_Frequency'
                continuous_label = 'Current' if analysis_type == 'current' else 'Fold Rheobase'
                posthoc_rows = compute_global_curve_posthocs(
                    model=model,
                    step_col_z=step_col_z,
                    analysis_label=analysis_label,
                    factor_a_col='Genotype',
                    factor_b_col=None,
                    factor_a_label='Genotype',
                    factor_b_label=continuous_label,
                    factor_a_levels=group_levels,
                    factor_b_levels=[],
                    main_a_p=overall_group_p,
                    main_b_p=None,
                    interaction_ab_p=None,
                    interaction_a_step_p=interaction_p,
                    interaction_b_step_p=None,
                    interaction_three_way_p=None
                )
            
            return {
                'effects': result,
                'posthocs': posthoc_rows
            }
            
        except Exception as e:
            logger.error(f"Error running unified mixed-effects model for {analysis_type}: {e}")
            return None
    
    def _apply_fdr_correction_frequency(self, results: List[Dict]):
        """Apply FDR correction to frequency analysis results.
        Handles both regular one-way ANOVA and factorial two-way ANOVA."""
        import statsmodels.stats.multitest as multi
        
        # Check if we have factorial results
        has_factorial = any('Two-Way ANOVA' in r.get('Test_Type', '') for r in results)
        
        if has_factorial:
            # For factorial: Apply FDR separately for each effect type (Factor1, Factor2, Interaction)
            # This acknowledges these are different hypotheses
            
            # For factorial designs, correct each effect type separately across steps
            # Results now have format: {Factor1}_p, {Factor2}_p, Interaction_p as columns
            anova_results = [r for r in results if r.get('Test_Type') == 'Two-Way ANOVA']
            
            if anova_results:
                # Find the effect column names dynamically
                effect_cols = [col for col in anova_results[0].keys() if col.endswith('_p') and not col.endswith('corrected_p')]
                
                # Correct each effect type separately
                for effect_col in effect_cols:
                    pvals = [r[effect_col] for r in anova_results if not np.isnan(r.get(effect_col, np.nan))]
                    valid_indices = [i for i, r in enumerate(anova_results) if not np.isnan(r.get(effect_col, np.nan))]
                    
                    if len(pvals) > 1:
                        _, corrected_p, _, _ = multi.multipletests(pvals, method="fdr_bh")
                        corrected_col = effect_col.replace('_p', '_corrected_p')
                        
                        for i, corrected_val in zip(valid_indices, corrected_p):
                            anova_results[i][corrected_col] = corrected_val
                        
                        # Set NaN for invalid results
                        for result in anova_results:
                            if np.isnan(result.get(effect_col, np.nan)) and corrected_col not in result:
                                result[corrected_col] = np.nan
                        
                        logger.info(f"Applied FDR correction to {len(pvals)} {effect_col} effects across frequency points")
                    else:
                        # Only 1 or 0 valid p-values - copy raw p-values for valid tests
                        corrected_col = effect_col.replace('_p', '_corrected_p')
                        for result in anova_results:
                            raw_p = result.get(effect_col, np.nan)
                            result[corrected_col] = raw_p if not np.isnan(raw_p) else np.nan
        else:
            # Regular one-way ANOVA
            anova_results = [r for r in results if r['Test_Type'] == 'ANOVA']
            
            if len(anova_results) > 1:
                anova_pvals = [r['p_value'] for r in anova_results if not np.isnan(r['p_value'])]
                valid_indices = [i for i, r in enumerate(anova_results) if not np.isnan(r['p_value'])]
                
                if len(anova_pvals) > 1:
                    _, corrected_p, _, _ = multi.multipletests(anova_pvals, method="fdr_bh")
                    for i, corrected_val in zip(valid_indices, corrected_p):
                        anova_results[i]['corrected_p'] = corrected_val
                        
                    # Set NaN corrected p-values for invalid results
                    for i, result in enumerate(anova_results):
                        if np.isnan(result['p_value']) and 'corrected_p' not in result:
                            result['corrected_p'] = np.nan
                else:
                    # Only 1 or 0 valid p-values - copy raw p-values for valid tests
                    for result in anova_results:
                        raw_p = result.get('p_value', np.nan)
                        result['corrected_p'] = raw_p if not np.isnan(raw_p) else np.nan
                
                logger.info(f"Applied FDR correction: {len(anova_results)} ANOVA tests")
        
        # Apply FDR to post-hoc p-values within each step's family (same for both designs)
        posthoc_results = [r for r in results if r.get('Test_Type') == 'Post-hoc t-test']
        
        # Group post-hoc results by step value (each ANOVA family)
        step_groups = {}
        for result in posthoc_results:
            step_value = result.get('Step_Value')
            if step_value not in step_groups:
                step_groups[step_value] = []
            step_groups[step_value].append(result)
        
        # Apply FDR correction within each step's post-hoc tests
        for step_value, results_list in step_groups.items():
            if len(results_list) > 1:
                # Extract p-values (filter out NaN)
                pvals = [r['p_value'] for r in results_list if not np.isnan(r['p_value'])]
                valid_indices = [i for i, r in enumerate(results_list) if not np.isnan(r['p_value'])]
                
                if len(pvals) > 1:
                    _, corrected_p, _, _ = multi.multipletests(pvals, method="fdr_bh")
                    for i, corrected_val in zip(valid_indices, corrected_p):
                        results_list[i]['corrected_p'] = corrected_val
                    
                    # Set NaN corrected p-values for invalid results
                    for i, result in enumerate(results_list):
                        if np.isnan(result['p_value']) and 'corrected_p' not in result:
                            result['corrected_p'] = np.nan
                    
                    logger.info(f"Applied FDR correction to {len(pvals)} post-hoc tests at step {step_value}")
                else:
                    # Only 1 or 0 valid p-values - copy raw p-values for valid tests
                    for idx in valid_indices:
                        results_list[idx]['corrected_p'] = results_list[idx]['p_value']
                    for i in range(len(results_list)):
                        if i not in valid_indices:
                            results_list[i]['corrected_p'] = np.nan
            elif len(results_list) == 1:
                # Single result - copy raw p-value if valid
                results_list[0]['corrected_p'] = (
                    results_list[0]['p_value']
                    if not np.isnan(results_list[0]['p_value'])
                    else np.nan
                )
