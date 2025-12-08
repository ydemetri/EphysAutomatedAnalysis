"""
Statistical analysis for AP attenuation (AP number vs peak).
"""

import pandas as pd
import numpy as np
from scipy.stats import ttest_ind, t, kruskal
import os
import logging
import math
from typing import List, Dict, Tuple, Optional

import statsmodels.stats.multitest as multi
from statsmodels.formula.api import mixedlm
from statsmodels.stats.oneway import anova_oneway
import scikit_posthocs as sp

from ...shared.data_models import ExperimentalDesign, StatisticalResult, DesignType
from ...shared.utils import should_use_parametric
from .posthoc_utils import (
    should_run_posthocs,
    get_simple_effect_comparisons,
    compute_lsmean_summaries,
    compute_global_curve_posthocs,
)

logger = logging.getLogger(__name__)

MIN_CELLS_PER_GROUP = 3


def _get_effect_pvalue(model, param_names: list, data: pd.DataFrame, formula: str, 
                       groups_col: str, re_formula: str, effect_name: str = "") -> float:
    """
    Compute a Wald-test p-value for a block of fixed-effect parameters.
    (Extra arguments are retained for compatibility but unused here.)
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


class AttenuationAnalyzer:
    """Analyzes AP attenuation (AP number vs peak)."""
    
    def analyze_attenuation(self, design: ExperimentalDesign, base_path: str) -> Dict[str, any]:
        """Analyze AP attenuation with t-tests and mixed-effects model."""
        
        results = {
            'point_by_point_stats': [],
            'mixed_effects_result': None,
            'attenuation_plot_data': None,
            'success': False
        }
        
        try:
            # Load and combine data from all groups
            all_group_data = {}
            all_plot_data = {}
            
            for group in design.groups:
                # Load attenuation data for each group
                attenuation_file = os.path.join(base_path, "Results", f"Calc_{group.name}_attenuation.csv")
                
                if not os.path.exists(attenuation_file):
                    logger.warning(f"No attenuation data found for {group.name}")
                    continue
                    
                try:
                    # Read CSV exactly like the original code does
                    group_data = pd.read_csv(attenuation_file, index_col=False)
                    
                    # Filter for only the "Value" columns using regex (exactly like original)
                    group_data_values_only = group_data.filter(regex='Value').copy()  # Explicit copy to avoid SettingWithCopyWarning
                    
                    # Filter for cells with 10+ APs
                    group_filtered = group_data_values_only[group_data_values_only.columns[group_data_values_only.count() >= 10]]
                    
                    # Check if this group has at least 5 cells with >=10 APs
                    if not group_filtered.empty and group_filtered.shape[0] >= 10 and group_filtered.shape[1] >= 5:
                        all_group_data[group.name] = group_filtered
                        # Calculate plot data for this group
                        group_plot_data = self._calculate_group_plot_data(group_filtered, group.name)
                        all_plot_data[group.name] = group_plot_data
                        logger.info(f"Loaded attenuation data for {group.name}: {group_filtered.shape[1]} cells with >=10 APs")
                    else:
                        if group_filtered.shape[1] < 5:
                            logger.warning(f"Insufficient cells for {group.name}: only {group_filtered.shape[1]} cells with >=10 APs (need at least 5)")
                        else:
                            logger.warning(f"Insufficient attenuation data for {group.name}: shape={group_data.shape}")
                        
                except Exception as e:
                    logger.warning(f"Error loading attenuation data for {group.name}: {e}")
                    continue
            
            # Check if ALL groups have sufficient data (at least 5 cells with >=10 APs each)
            if len(all_group_data) < len(design.groups):
                missing_groups = [g.name for g in design.groups if g.name not in all_group_data]
                logger.warning(f"Skipping attenuation analysis - insufficient data for groups: {missing_groups}")
                logger.warning("All groups must have at least 5 cells with >=10 APs for attenuation analysis")
                return results
                
            if len(all_group_data) < 2:
                logger.warning("Need at least 2 groups with sufficient attenuation data")
                return results
            
            # Set plot data
            if all_plot_data:
                results['attenuation_plot_data'] = all_plot_data
            
            # Point-by-point ANOVA for each AP position (factorial or regular)
            point_stats = self._run_pointwise_anova_attenuation(all_group_data, design)
            if point_stats:
                results['point_by_point_stats'] = point_stats
                self._save_attenuation_results(point_stats, base_path, design)
            
            # Unified mixed-effects model with all groups (factorial or regular)
            mixed_effects_result = self._run_unified_mixed_effects_attenuation(all_group_data, design)
            if mixed_effects_result:
                results['mixed_effects_result'] = mixed_effects_result
            
            results['success'] = True
            logger.info(f"Attenuation analysis completed for {len(all_group_data)} groups")
            
        except Exception as e:
            logger.error(f"Error in attenuation analysis: {e}")
            results['error'] = str(e)
            
        return results
    
    def _calculate_group_plot_data(self, group_data: pd.DataFrame, group_name: str) -> Dict:
        """Calculate plot data for a single group."""
        
        # Use first 10 APs - fix pandas slicing issue
        data_10ap = group_data.iloc[:10]
        
        logger.info(f"DEBUG: Calculating plot data for {group_name}")
        logger.info(f"DEBUG: data_10ap shape: {data_10ap.shape}, columns: {list(data_10ap.columns[:5])}")
        
        # All columns are already value columns (filtered during loading)
        value_cols = list(data_10ap.columns)
        logger.info(f"DEBUG: Found {len(value_cols)} value columns")
        
        ap_numbers = list(range(1, 11))
        means = []
        sems = []
        
        for i in range(10):  # First 10 APs
            if i < len(data_10ap):
                ap_data = data_10ap.iloc[i][value_cols].dropna()
                if len(ap_data) > 0:
                    mean_val = ap_data.mean()
                    means.append(mean_val)
                    sems.append(ap_data.std(ddof=1) / np.sqrt(len(ap_data)))
                    if i < 3:  # Log first 3 APs
                        logger.info(f"DEBUG: AP {i}: {len(ap_data)} cells, mean={mean_val:.2f}, values={list(ap_data[:3])}")
                else:
                    means.append(np.nan)
                    sems.append(np.nan)
            else:
                means.append(np.nan)
                sems.append(np.nan)
        
        logger.info(f"DEBUG: Calculated means for {group_name}: {means[:5]}")
        
        return {
            'ap_number': ap_numbers,
            'mean': means,
            'sem': sems
        }
    
    def _run_pointwise_anova_attenuation(self, all_group_data: Dict[str, pd.DataFrame],
                                         design: ExperimentalDesign = None) -> List[Dict]:
        """Run point-by-point ANOVA for each AP position with post-hoc if significant.
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
        
        logger.info(f"Running point-by-point ANOVA for attenuation across {len(group_names)} groups")
        
        # Run ANOVA for each AP position (1-10)
        for ap_pos in range(1, 11):
            ap_index = ap_pos - 1
            
            # Collect data for this AP position from all groups
            group_data_lists = []
            group_stats = {}
            group_ap_values = {}
            insufficient_groups = set()
            
            for group_name in group_names:
                group_data = all_group_data[group_name]
                if ap_index < len(group_data):
                    # All columns are already value columns (filtered during loading)
                    ap_data = group_data.iloc[ap_index].dropna()
                    
                    if len(ap_data) >= MIN_CELLS_PER_GROUP:
                        group_ap_values[group_name] = ap_data
                        group_stats[group_name] = {
                            'mean': ap_data.mean(),
                            'stderr': ap_data.std(ddof=1) / np.sqrt(len(ap_data)),
                            'n': len(ap_data)
                        }
                    else:
                        insufficient_groups.add(group_name)
                else:
                    insufficient_groups.add(group_name)
            
            if len(group_ap_values) != len(group_names):
                logger.info(
                    f"Skipping AP {ap_pos}: insufficient data for "
                    f"{sorted(insufficient_groups)} (need >= {MIN_CELLS_PER_GROUP} cells per group)"
                )
                continue
            
            group_data_lists = [group_ap_values[name] for name in group_names]
            
            # Decide parametric vs nonparametric using skewness/kurtosis
            use_parametric = should_use_parametric(group_data_lists)
            
            # Run ANOVA (one-way or two-way depending on design)
            try:
                if is_factorial:
                    # Build dataframe with factor labels for two-way ANOVA
                    data_list = []
                    for group_name in group_names:
                        ap_data = group_ap_values.get(group_name)
                        if ap_data is None:
                            continue
                        
                        factor1_level = design.factor_mapping[group_name]['factor1']
                        factor2_level = design.factor_mapping[group_name]['factor2']
                        
                        for val in ap_data:
                            data_list.append({
                                'peak': val,
                                'factor1': factor1_level,
                                'factor2': factor2_level
                            })
                    
                    if len(data_list) < 4:
                        continue
                    
                    df_anova = pd.DataFrame(data_list)
                    
                    # Check we have data for all factor combinations
                    factor_combos = df_anova.groupby(['factor1', 'factor2']).size()
                    factor1_levels_count = df_anova['factor1'].nunique()
                    factor2_levels_count = df_anova['factor2'].nunique()
                    expected_combos = factor1_levels_count * factor2_levels_count
                    
                    if len(factor_combos) < expected_combos:
                        logger.warning(f"Missing factor combinations at AP {ap_pos} (have {len(factor_combos)}/{expected_combos})")
                        continue
                    
                    model = ols('peak ~ C(factor1) * C(factor2)', data=df_anova).fit()
                    anova_table = anova_lm(model, typ=2)
                    
                    # Extract p-values for Factor1, Factor2, and Interaction
                    factor1_p = anova_table.loc['C(factor1)', 'PR(>F)'] if 'C(factor1)' in anova_table.index else 1.0
                    factor2_p = anova_table.loc['C(factor2)', 'PR(>F)'] if 'C(factor2)' in anova_table.index else 1.0
                    interaction_p = anova_table.loc['C(factor1):C(factor2)', 'PR(>F)'] if 'C(factor1):C(factor2)' in anova_table.index else 1.0
                    
                    # Create one result per AP with all effects as columns (matching dependent format)
                    cell_stats_df = (
                        df_anova.groupby(['factor1', 'factor2'])['peak']
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
                        'AP_Number': ap_pos,
                        'Test_Type': 'Two-Way ANOVA',
                        f'{design.factor1_name}_p': factor1_p,
                        f'{design.factor2_name}_p': factor2_p,
                        'Interaction_p': interaction_p,
                        'measurement_type': 'attenuation',
                        '_run_posthoc': True,  # Mark for potential post-hoc if interaction significant
                        '_mse_resid': model.mse_resid,
                        '_df_resid': int(model.df_resid),
                        '_cell_stats': cell_stats,
                        '_factor1_levels': sorted(df_anova['factor1'].unique()),
                        '_factor2_levels': sorted(df_anova['factor2'].unique())
                    }
                    
                    # Add group statistics (mean, SEM, n for each group)
                    for group_name in sorted(group_stats.keys()):
                        result[f'{group_name}_mean'] = group_stats[group_name]['mean']
                        result[f'{group_name}_SEM'] = group_stats[group_name]['stderr']
                        result[f'{group_name}_n'] = group_stats[group_name]['n']
                    
                    all_results.append(result)
                    
                else:
                    # Standard one-way: choose Welch ANOVA or Kruskal-Wallis based on use_parametric
                    if use_parametric:
                        try:
                            welch_res = anova_oneway(group_data_lists, use_var="unequal", welch_correction=True)
                            anova_p = welch_res.pvalue
                            test_type = 'Welch_ANOVA'
                            f_stat = np.nan
                        except Exception as e:
                            logger.warning(f"Welch ANOVA failed at AP {ap_pos}: {e}")
                            anova_p = np.nan
                            test_type = 'Welch_ANOVA'
                            f_stat = np.nan
                    else:
                        h_stat, anova_p = kruskal(*group_data_lists)
                        test_type = 'Kruskal_Wallis'
                        f_stat = h_stat
                    
                    # Create ANOVA result
                    anova_result = {
                        'AP_Number': ap_pos,
                        'Test_Type': test_type,
                        'Comparison': f"Overall ({' vs '.join(group_names)})",
                        'p_value': anova_p,
                        'measurement_type': 'attenuation'
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
                        if anova_p < 0.05:
                            logger.info(f"AP {ap_pos} {test_type} significant (p={anova_p:.4f}) - no post-hoc needed (only 2 groups)")
                        else:
                            logger.info(f"AP {ap_pos} {test_type} not significant (p={anova_p:.4f})")
                    else:
                        logger.info(f"AP {ap_pos} {test_type} (p={anova_p:.4f}) - will check corrected p-value for post-hoc")
                    
            except Exception as e:
                logger.warning(f"Error running ANOVA for AP {ap_pos}: {e}")
                continue
        
        # Apply FDR correction to ANOVA results first
        if all_results:
            self._apply_fdr_correction_attenuation(all_results)
            
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
                    ap_pos = result['AP_Number']  
                    
                    # Log why we're running post-hocs
                    if posthoc_decision:
                        logger.info(f"AP {ap_pos}: Running post-hocs for {posthoc_decision['reasons']}")
                    else:
                        corrected_p = result.get('corrected_p', result.get('p_value', 1))
                        logger.info(f"AP {ap_pos} ANOVA significant (corrected p={corrected_p:.4f}) - running post-hoc tests")
                    
                    # Find the corresponding AP data for post-hoc tests
                    ap_idx = ap_pos - 1  # Convert to 0-based index
                    
                    if ap_idx < 10:  # We only analyze first 10 APs
                        # Collect data for this AP from all groups
                        ap_group_data = {}
                        ap_group_stats = {}
                        missing_groups = set()
                        
                        for group_name in group_names:
                            group_data = all_group_data[group_name]
                            if ap_idx < len(group_data):
                                # Get values for this AP across all cells
                                ap_values = group_data.iloc[ap_idx].dropna()
                                
                                if len(ap_values) >= MIN_CELLS_PER_GROUP:
                                    ap_group_data[group_name] = ap_values
                                    ap_group_stats[group_name] = {
                                        'mean': ap_values.mean(),
                                        'stderr': ap_values.std(ddof=1) / np.sqrt(len(ap_values)),
                                        'n': len(ap_values)
                                    }
                                else:
                                    missing_groups.add(group_name)
                            else:
                                missing_groups.add(group_name)
                        
                        if len(ap_group_data) != len(group_names):
                            logger.info(
                                f"Skipping attenuation post-hocs at AP {ap_pos}: insufficient data for "
                                f"{sorted(missing_groups)} (need >= {MIN_CELLS_PER_GROUP} cells per group)"
                            )
                            continue
                        
                        if result.get('Test_Type') == 'Two-Way ANOVA':
                            reasons = posthoc_decision.get('reasons', []) if posthoc_decision else []
                            if 'interaction' in reasons:
                                comparisons_to_run = get_simple_effect_comparisons(design, list(ap_group_data.keys()))
                                logger.info(f"Running {len(comparisons_to_run)} simple effects comparisons")
                                
                                for group1_name, group2_name in comparisons_to_run:
                                    if group1_name in ap_group_data and group2_name in ap_group_data:
                                        t_stat, pairwise_p = ttest_ind(
                                            ap_group_data[group1_name], ap_group_data[group2_name],
                                            nan_policy="omit", equal_var=False
                                        )
                                        
                                        pairwise_result = {
                                            'AP_Number': ap_pos, 
                                            'Test_Type': 'Post-hoc t-test',
                                            'Comparison': f"{group1_name} vs {group2_name}",
                                            'Group1': group1_name,
                                            'Group1_mean': ap_group_stats[group1_name]['mean'],
                                            'Group1_stderr': ap_group_stats[group1_name]['stderr'],
                                            'Group1_n': ap_group_stats[group1_name]['n'],
                                            'Group2': group2_name,
                                            'Group2_mean': ap_group_stats[group2_name]['mean'],
                                            'Group2_stderr': ap_group_stats[group2_name]['stderr'],
                                            'Group2_n': ap_group_stats[group2_name]['n'],
                                            'p_value': pairwise_p,
                                            'measurement_type': 'attenuation'
                                        }
                                        posthoc_results.append(pairwise_result)
                            else:
                                mse = result.get('_mse_resid')
                                df_resid = result.get('_df_resid')
                                cell_stats = result.get('_cell_stats')
                                factor1_levels = result.get('_factor1_levels')
                                factor2_levels = result.get('_factor2_levels')
                                if 'factor1_main' in reasons:
                                    marginal = self._run_attenuation_marginal_posthoc(
                                        ap_pos, design, 'factor1', mse, df_resid,
                                        cell_stats, factor1_levels, factor2_levels
                                    )
                                    posthoc_results.extend(marginal)
                                if 'factor2_main' in reasons:
                                    marginal = self._run_attenuation_marginal_posthoc(
                                        ap_pos, design, 'factor2', mse, df_resid,
                                        cell_stats, factor1_levels, factor2_levels
                                    )
                                    posthoc_results.extend(marginal)
                        else:
                            from itertools import combinations
                            comparisons_to_run = list(combinations(ap_group_data.keys(), 2))
                            
                            if result.get('Test_Type') == 'Kruskal_Wallis':
                                # Dunn posthocs for KW
                                values = []
                                labels = []
                                for gname, data in ap_group_data.items():
                                    values.extend(list(data.values))
                                    labels.extend([gname] * len(data))
                                dunn_input = pd.DataFrame({'val': values, 'grp': labels})
                                dunn_df = sp.posthoc_dunn(dunn_input, val_col='val', group_col='grp')
                                dunn_df = dunn_df.reindex(index=ap_group_data.keys(), columns=ap_group_data.keys())
                                
                                for group1_name, group2_name in comparisons_to_run:
                                    p_val = dunn_df.loc[group1_name, group2_name]
                                    pairwise_result = {
                                        'AP_Number': ap_pos, 
                                        'Test_Type': 'Post-hoc Dunn',
                                        'Comparison': f"{group1_name} vs {group2_name}",
                                        'Group1': group1_name,
                                        'Group1_mean': ap_group_stats[group1_name]['mean'],
                                        'Group1_stderr': ap_group_stats[group1_name]['stderr'],
                                        'Group1_n': ap_group_stats[group1_name]['n'],
                                        'Group2': group2_name,
                                        'Group2_mean': ap_group_stats[group2_name]['mean'],
                                        'Group2_stderr': ap_group_stats[group2_name]['stderr'],
                                        'Group2_n': ap_group_stats[group2_name]['n'],
                                        'p_value': p_val,
                                        'measurement_type': 'attenuation'
                                    }
                                    posthoc_results.append(pairwise_result)
                            else:
                                for group1_name, group2_name in comparisons_to_run:
                                    if group1_name in ap_group_data and group2_name in ap_group_data:
                                        t_stat, pairwise_p = ttest_ind(
                                            ap_group_data[group1_name], ap_group_data[group2_name],
                                            nan_policy="omit", equal_var=False
                                        )
                                        
                                        pairwise_result = {
                                            'AP_Number': ap_pos, 
                                            'Test_Type': 'Post-hoc t-test',
                                            'Comparison': f"{group1_name} vs {group2_name}",
                                            'Group1': group1_name,
                                            'Group1_mean': ap_group_stats[group1_name]['mean'],
                                            'Group1_stderr': ap_group_stats[group1_name]['stderr'],
                                            'Group1_n': ap_group_stats[group1_name]['n'],
                                            'Group2': group2_name,
                                            'Group2_mean': ap_group_stats[group2_name]['mean'],
                                            'Group2_stderr': ap_group_stats[group2_name]['stderr'],
                                            'Group2_n': ap_group_stats[group2_name]['n'],
                                            'p_value': pairwise_p,
                                            'measurement_type': 'attenuation'
                                        }
                                        posthoc_results.append(pairwise_result)
            
            # Add post-hoc results to all results and apply FDR correction to them
            all_results.extend(posthoc_results)
            if posthoc_results:
                self._apply_fdr_correction_attenuation(all_results)  # Re-apply to include post-hoc
        
        logger.info(f"Completed point-by-point ANOVA: {len(all_results)} statistical tests")
        return all_results

    def _run_attenuation_marginal_posthoc(self, ap_pos: int, design: ExperimentalDesign, factor_key: str,
                                          mse: Optional[float], df_resid: Optional[int],
                                          cell_stats: Optional[Dict],
                                          factor1_levels: Optional[List[str]],
                                          factor2_levels: Optional[List[str]]) -> List[Dict]:
        """Run marginal mean comparisons for attenuation factorial designs."""
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
                'AP_Number': ap_pos,
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
                'p_value': p_value,
                'measurement_type': 'attenuation'
            }
            results.append(result)

        return results
    
    def _apply_fdr_correction_attenuation(self, results: List[Dict]):
        """Apply FDR correction to attenuation results.
        Handles both regular one-way ANOVA and factorial two-way ANOVA."""
        import statsmodels.stats.multitest as multi
        
        def _valid_p(val: float) -> bool:
            return val is not None and np.isfinite(val)
        
        # Check if we have factorial results
        has_factorial = any('Two-Way ANOVA' in r.get('Test_Type', '') for r in results)
        
        if has_factorial:
            # For factorial: Apply FDR separately for each effect type (Factor1, Factor2, Interaction)
            # Results now have format: {Factor1}_p, {Factor2}_p, Interaction_p as columns
            anova_results = [r for r in results if r.get('Test_Type') == 'Two-Way ANOVA']
            
            if anova_results:
                # Find the effect column names dynamically
                effect_cols = [col for col in anova_results[0].keys() if col.endswith('_p') and not col.endswith('corrected_p')]
                
                # Correct each effect type separately
                for effect_col in effect_cols:
                    pvals = [r.get(effect_col) for r in anova_results if _valid_p(r.get(effect_col, np.nan))]
                    valid_indices = [i for i, r in enumerate(anova_results) if _valid_p(r.get(effect_col, np.nan))]
                    
                    if len(pvals) > 1:
                        _, corrected_p, _, _ = multi.multipletests(pvals, method="fdr_bh")
                        corrected_col = effect_col.replace('_p', '_corrected_p')
                        
                        for i, corrected_val in zip(valid_indices, corrected_p):
                            anova_results[i][corrected_col] = corrected_val
                        
                        # Set NaN for invalid results
                        for result in anova_results:
                            if np.isnan(result.get(effect_col, np.nan)) and corrected_col not in result:
                                result[corrected_col] = np.nan
                        
                        logger.info(f"Applied FDR correction to {len(pvals)} {effect_col} effects across AP numbers")
                    else:
                        # Only 1 or 0 valid p-values - copy raw p-values for valid tests
                        corrected_col = effect_col.replace('_p', '_corrected_p')
                        for result in anova_results:
                            raw_p = result.get(effect_col, np.nan)
                            result[corrected_col] = raw_p if not np.isnan(raw_p) else np.nan
        else:
            # Regular one-way tests (ANOVA/Welch/KW)
            anova_results = [r for r in results if r.get('Test_Type') in ['ANOVA', 'Welch_ANOVA', 'Kruskal_Wallis']]
            
            if len(anova_results) > 1:
                # Extract p-values and track valid indices
                valid_indices = [i for i, r in enumerate(anova_results) if _valid_p(r.get('p_value'))]
                valid_p_values = [anova_results[i]['p_value'] for i in valid_indices]
                
                if len(valid_p_values) > 1:
                    _, corrected_p, _, _ = multi.multipletests(valid_p_values, method="fdr_bh")
                    # Map corrected p-values back to valid indices
                    for i, corrected_val in zip(valid_indices, corrected_p):
                        anova_results[i]['corrected_p'] = corrected_val
                    # Set NaN for invalid p-values
                    for i in range(len(anova_results)):
                        if i not in valid_indices:
                            anova_results[i]['corrected_p'] = np.nan
                else:
                    # Only 1 or 0 valid p-values - copy raw p-values for valid tests
                    for idx in valid_indices:
                        anova_results[idx]['corrected_p'] = anova_results[idx]['p_value']
                    for i in range(len(anova_results)):
                        if i not in valid_indices:
                            anova_results[i]['corrected_p'] = np.nan
            
            logger.info(f"Applied FDR correction: {len(anova_results)} one-way tests")
        
        # Apply FDR to post-hoc p-values within each AP number's family (same for both designs)
        posthoc_results = [r for r in results if r.get('Test_Type') in ['Post-hoc t-test', 'Post-hoc Dunn']]
        
        # Group post-hoc results by AP number (each ANOVA family)
        ap_groups = {}
        for result in posthoc_results:
            ap_num = result.get('AP_Number')
            if ap_num not in ap_groups:
                ap_groups[ap_num] = []
            ap_groups[ap_num].append(result)
        
        # Apply FDR correction within each AP number's post-hoc tests
        for ap_num, results_list in ap_groups.items():
            if len(results_list) > 1:
                # Extract p-values (filter out NaN)
                pvals = [r['p_value'] for r in results_list if _valid_p(r['p_value'])]
                valid_indices = [i for i, r in enumerate(results_list) if _valid_p(r['p_value'])]
                
                if len(pvals) > 1:
                    _, corrected_p, _, _ = multi.multipletests(pvals, method="fdr_bh")
                    for i, corrected_val in zip(valid_indices, corrected_p):
                        results_list[i]['corrected_p'] = corrected_val
                    
                    # Set NaN corrected p-values for invalid results
                    for i, result in enumerate(results_list):
                        if np.isnan(result['p_value']) and 'corrected_p' not in result:
                            result['corrected_p'] = np.nan
                    
                    logger.info(f"Applied FDR correction to {len(pvals)} post-hoc tests at AP {ap_num}")
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
    
    def _run_unified_mixed_effects_attenuation(self, all_group_data: Dict[str, pd.DataFrame], 
                                               design: ExperimentalDesign = None) -> Optional[Dict]:
        """Run unified mixed-effects model with all groups and proper cell IDs.
        Handles both regular mixed-effects and factorial designs with 3-way interactions."""
        
        # Check if factorial design
        is_factorial = design and design.design_type == DesignType.FACTORIAL_2X2
        
        try:
            # Combine data from all groups with consistent cell IDs
            ap_nums = []
            peaks = []
            groups_list = []
            cell_ids = []
            
            # For factorial designs, also track factor levels
            if is_factorial:
                factor1_list = []
                factor2_list = []
            
            for group_name, group_data in all_group_data.items():
                # Use first 10 APs - fix pandas slicing issue
                data_10ap = group_data.iloc[:10].copy()
                
                # All columns are already value columns (filtered during loading)
                value_cols = list(data_10ap.columns)
                
                # For each cell, add all its AP measurements
                for cell_idx, col in enumerate(value_cols):
                    if col in data_10ap.columns:
                        cell_data = data_10ap[col].dropna()
                        
                        # Create consistent cell ID across groups
                        cell_id = f"{group_name}_cell_{cell_idx:03d}"
                        
                        # Add data for this cell across all APs
                        for ap_idx, peak_val in enumerate(cell_data):
                            if ap_idx < 10:  # Only first 10 APs
                                ap_nums.append(ap_idx + 1)
                                peaks.append(peak_val)
                                groups_list.append(group_name)
                                cell_ids.append(cell_id)
                                
                                # For factorial designs, add factor levels
                                if is_factorial:
                                    factor1_list.append(design.factor_mapping[group_name]['factor1'])
                                    factor2_list.append(design.factor_mapping[group_name]['factor2'])
            
            if len(set(groups_list)) < 2:
                logger.warning("Need at least 2 groups for mixed-effects model")
                return None
                
            # Create unified DataFrame (with or without factors)
            if is_factorial:
                final_df = pd.DataFrame({
                    'AP_num': ap_nums, 
                    'Peak': peaks, 
                    'Factor1': factor1_list,
                    'Factor2': factor2_list,
                    'Cell_id': cell_ids
                }).dropna()
                
                # Standardize continuous variable (z-score) to help convergence
                final_df['AP_num_z'] = (final_df['AP_num'] - final_df['AP_num'].mean()) / final_df['AP_num'].std()
                
                logger.info(f"Factorial mixed-effects model: {len(final_df)} observations, {len(final_df['Cell_id'].unique())} cells")
                logger.info(f"Running factorial mixed-effects model with random slopes for standardized AP_num...")
                
                # 3-way interaction model for factorial design
                model = mixedlm("Peak ~ C(Factor1) * C(Factor2) * AP_num_z", final_df, groups=final_df["Cell_id"],
                              re_formula="1 + AP_num_z").fit(method='lbfgs', maxiter=5000)
                
                # Check convergence
                if not model.converged:
                    logger.warning(f"Factorial attenuation model did not converge after 5000 iterations")
                else:
                    logger.info(f"Factorial attenuation model converged successfully")
                
                # Extract p-values for all 7 effects using proper statistical tests (LRT for multiple contrasts)
                formula = "Peak ~ C(Factor1) * C(Factor2) * AP_num_z"
                
                factor1_params = [p for p in model.pvalues.index if 'C(Factor1)' in p and ':' not in p]
                factor1_p = _get_effect_pvalue(model, factor1_params, final_df, formula, 'Cell_id', "1 + AP_num_z", "Factor1")
                
                factor2_params = [p for p in model.pvalues.index if 'C(Factor2)' in p and ':' not in p]
                factor2_p = _get_effect_pvalue(model, factor2_params, final_df, formula, 'Cell_id', "1 + AP_num_z", "Factor2")
                
                ap_p = model.pvalues.get('AP_num_z', 1.0)
                
                # Two-way interactions - use LRT for multiple contrasts
                f1_f2_params = [p for p in model.pvalues.index if 'C(Factor1)' in p and 'C(Factor2)' in p and 'AP_num_z' not in p]
                f1_f2_p = _get_effect_pvalue(model, f1_f2_params, final_df, formula, 'Cell_id', "1 + AP_num_z", "Factor1:Factor2")
                
                f1_ap_params = [p for p in model.pvalues.index if 'C(Factor1)' in p and 'AP_num_z' in p and 'C(Factor2)' not in p]
                f1_ap_p = _get_effect_pvalue(model, f1_ap_params, final_df, formula, 'Cell_id', "1 + AP_num_z", "Factor1:AP_Number")
                
                f2_ap_params = [p for p in model.pvalues.index if 'C(Factor2)' in p and 'AP_num_z' in p and 'C(Factor1)' not in p]
                f2_ap_p = _get_effect_pvalue(model, f2_ap_params, final_df, formula, 'Cell_id', "1 + AP_num_z", "Factor2:AP_Number")
                
                # Three-way interaction
                three_way_params = [p for p in model.pvalues.index if 'C(Factor1)' in p and 'C(Factor2)' in p and 'AP_num_z' in p]
                three_way_p = _get_effect_pvalue(model, three_way_params, final_df, formula, 'Cell_id', "1 + AP_num_z", "Factor1:Factor2:AP_Number")
                
                # Create result with all 7 effects
                result = {
                    'Effect': [
                        design.factor1_name,
                        design.factor2_name,
                        'AP Number',
                        f'{design.factor1_name}:{design.factor2_name}',
                        f'{design.factor1_name}:AP Number',
                        f'{design.factor2_name}:AP Number',
                        f'{design.factor1_name}:{design.factor2_name}:AP Number'
                    ],
                    'p-value': [
                        factor1_p,
                        factor2_p,
                        ap_p,
                        f1_f2_p,
                        f1_ap_p,
                        f2_ap_p,
                        three_way_p
                    ]
                }
                
                logger.info(f"Factorial mixed-effects attenuation: F1 p={factor1_p:.4f}, F2 p={factor2_p:.4f}, 3-way interaction p={three_way_p:.4f}")
                
                factor1_levels_unique = sorted(final_df['Factor1'].dropna().unique())
                factor2_levels_unique = sorted(final_df['Factor2'].dropna().unique())
                posthoc_rows = compute_global_curve_posthocs(
                    model=model,
                    step_col_z='AP_num_z',
                    analysis_label='AP_Number_vs_Peak',
                    factor_a_col='Factor1',
                    factor_b_col='Factor2',
                    factor_a_label=design.factor1_name,
                    factor_b_label=design.factor2_name,
                    factor_a_levels=factor1_levels_unique,
                    factor_b_levels=factor2_levels_unique,
                    main_a_p=factor1_p,
                    main_b_p=factor2_p,
                    interaction_ab_p=f1_f2_p,
                    interaction_a_step_p=f1_ap_p,
                    interaction_b_step_p=f2_ap_p,
                    interaction_three_way_p=three_way_p
                )
                
                return {
                    'effects': result,
                    'posthocs': posthoc_rows
                }
                
            else:
                # Regular non-factorial design
                final_df = pd.DataFrame({
                    'AP_num': ap_nums, 
                    'Peak': peaks, 
                    'Group': groups_list, 
                    'Cell_id': cell_ids
                }).dropna()
                
                # Standardize continuous variable (z-score) to help convergence
                final_df['AP_num_z'] = (final_df['AP_num'] - final_df['AP_num'].mean()) / final_df['AP_num'].std()
                
                logger.info(f"Unified mixed-effects model: {len(final_df)} observations, {len(final_df['Cell_id'].unique())} cells, {len(set(groups_list))} groups")
                logger.info(f"Running unified mixed-effects model with random slopes for standardized AP_num...")
                
                # Unified mixed-effects model: Group effect, AP_num effect, and interaction
                model = mixedlm("Peak ~ C(Group) * AP_num_z", final_df, groups=final_df["Cell_id"],
                              re_formula="1 + AP_num_z").fit(method='lbfgs', maxiter=5000)
                
                # Check convergence
                if not model.converged:
                    logger.warning(f"Unified attenuation model did not converge after 5000 iterations")
                else:
                    logger.info(f"Unified attenuation model converged successfully")
                
                # Extract overall Group effect p-value using proper statistical test (LRT for 3+)
                formula = "Peak ~ C(Group) * AP_num_z"
                group_params = [p for p in model.pvalues.index if 'C(Group)' in p and ':' not in p]
                overall_group_p = _get_effect_pvalue(model, group_params, final_df, formula, 'Cell_id', "1 + AP_num_z", "Group")
                
                # Get interaction p-value (use LRT for 3+ contrasts)
                interaction_params = [p for p in model.pvalues.index if 'C(Group)' in p and 'AP_num_z' in p]
                interaction_p = _get_effect_pvalue(model, interaction_params, final_df, formula, 'Cell_id', "1 + AP_num_z", "Group:AP_Number")
                
                # Create DataFrame in original format
                result = {
                    'Effect': ['Genotype', 'AP Number', 'Genotype:AP Number'],
                    'p-value': [
                        overall_group_p,
                        model.pvalues.get('AP_num_z', 1.0),
                        interaction_p
                    ]
                }
                
                logger.info(f"Mixed-effects attenuation: Group p={overall_group_p:.4f}, AP_num p={model.pvalues.get('AP_num_z', 1.0):.4f}")
                
                group_levels = sorted(final_df['Group'].dropna().unique())
                posthoc_rows = compute_global_curve_posthocs(
                    model=model,
                    step_col_z='AP_num_z',
                    analysis_label='AP_Number_vs_Peak',
                    factor_a_col='Group',
                    factor_b_col=None,
                    factor_a_label='Genotype',
                    factor_b_label='AP Number',
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
            logger.error(f"Error running unified mixed-effects model: {e}")
            return None
    
    def _load_attenuation_data(self, group1_name: str, group2_name: str, base_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load attenuation data for both groups."""
        
        try:
            group1_file = os.path.join(base_path, "Results", f"Calc_{group1_name}_attenuation.csv")
            group2_file = os.path.join(base_path, "Results", f"Calc_{group2_name}_attenuation.csv")
            
            group1_data = pd.DataFrame()
            group2_data = pd.DataFrame()
            
            if os.path.exists(group1_file):
                group1_full = pd.read_csv(group1_file, index_col=False)
                group1_data = group1_full.filter(regex='Value').copy()  # Explicit copy to avoid SettingWithCopyWarning
                
            if os.path.exists(group2_file):
                group2_full = pd.read_csv(group2_file, index_col=False)
                group2_data = group2_full.filter(regex='Value').copy()  # Explicit copy to avoid SettingWithCopyWarning
                
            return group1_data, group2_data
            
        except Exception as e:
            logger.error(f"Error loading attenuation data: {e}")
            return pd.DataFrame(), pd.DataFrame()
    
    def _calculate_attenuation_summary(self, group1_data: pd.DataFrame, group2_data: pd.DataFrame,
                                      group1_name: str, group2_name: str) -> Dict:
        """Calculate summary statistics for attenuation plotting."""
        
        g1_val_cols = group1_data.columns
        g2_val_cols = group2_data.columns
        
        summary = {
            'group1': {
                'name': group1_name,
                'mean': group1_data[g1_val_cols].mean(axis=1),
                'sem': group1_data[g1_val_cols].sem(axis=1),
                'n': group1_data[g1_val_cols].count(axis=1),
                'AP_num': [i for i in range(1, group1_data.shape[0] + 1)]
            },
            'group2': {
                'name': group2_name,
                'mean': group2_data[g2_val_cols].mean(axis=1),
                'sem': group2_data[g2_val_cols].sem(axis=1),
                'n': group2_data[g2_val_cols].count(axis=1),
                'AP_num': [i for i in range(1, group2_data.shape[0] + 1)]
            }
        }
        
        return summary
    
    def _run_attenuation_tests(self, group1_data: pd.DataFrame, group2_data: pd.DataFrame,
                              group1_name: str, group2_name: str) -> List[Dict]:
        """Run t-tests for first 10 APs with FDR correction."""
        
        num_tests = min(10, group1_data.shape[0], group2_data.shape[0])
        stat_data = []
        
        for i in range(num_tests):
            try:
                t_stat, p_val = ttest_ind(
                    np.array(group2_data.iloc[i]), np.array(group1_data.iloc[i]),
                    nan_policy="omit", equal_var=False
                )
                
                if not np.isnan(p_val):
                    stat_data.append({
                        "AP Number": i + 1,
                        f"{group2_name}_mean": group2_data.iloc[i].mean(),
                        f"{group2_name}_stderr": group2_data.iloc[i].sem(),
                        f"{group2_name}_n": group2_data.iloc[i].count(),
                        f"{group1_name}_mean": group1_data.iloc[i].mean(),
                        f"{group1_name}_stderr": group1_data.iloc[i].sem(),
                        f"{group1_name}_n": group1_data.iloc[i].count(),
                        "p-value": p_val
                    })
                    
            except Exception as e:
                logger.warning(f"Error in attenuation t-test at AP {i+1}: {e}")
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
                    # Only 1 or 0 valid p-values - no correction possible
                    for s in stat_data:
                        s["corrected_p"] = np.nan
            else:
                # Single result - no correction possible
                for s in stat_data:
                    s["corrected_p"] = np.nan
        
        return stat_data
    
    def _save_attenuation_results(self, stat_data: List[Dict], base_path: str, 
                                  design: ExperimentalDesign = None) -> None:
        """Save attenuation statistical results to CSV."""
        
        if not stat_data:
            logger.warning("No attenuation statistical data to save")
            return
        
        # Check if this is a factorial design to determine the ANOVA result identification
        is_factorial = design and design.design_type == DesignType.FACTORIAL_2X2
        
        # Separate ANOVA and post-hoc results
        if is_factorial:
            anova_results = [r for r in stat_data if r.get('Test_Type') == 'Two-Way ANOVA']
        else:
            anova_results = [r for r in stat_data if r.get('Test_Type') in ['ANOVA', 'Welch_ANOVA', 'Kruskal_Wallis']]
        posthoc_results = [r for r in stat_data if r.get('Test_Type') in ['Post-hoc t-test', 'Post-hoc Dunn']]
        
        # Save ANOVA results
        if anova_results:
            df_anova = pd.DataFrame(anova_results)
            
            if is_factorial:
                # For factorial: one row per AP with all effects as columns (matching Stats_parameters structure)
                base_cols = ['AP_Number', 'Test_Type']
                
                # Add group statistics columns (mean, SEM, n for each group) - FIRST after AP_Number
                # Extract group names and sort them, then add mean, SEM, n for each group in that order
                group_names_set = set()
                for col in df_anova.columns:
                    if col.endswith('_mean'):
                        group_names_set.add(col.replace('_mean', ''))
                
                group_stat_cols = []
                for group_name in sorted(group_names_set):
                    if f'{group_name}_mean' in df_anova.columns:
                        group_stat_cols.append(f'{group_name}_mean')
                    if f'{group_name}_SEM' in df_anova.columns:
                        group_stat_cols.append(f'{group_name}_SEM')
                    if f'{group_name}_n' in df_anova.columns:
                        group_stat_cols.append(f'{group_name}_n')
                
                # Find effect columns dynamically, ensuring p_value comes before corrected_p
                effect_names = set()
                for col in df_anova.columns:
                    # Check _corrected_p FIRST since it also ends with _p
                    if col.endswith('_corrected_p'):
                        effect_names.add(col.replace('_corrected_p', ''))
                    elif col.endswith('_p'):
                        effect_names.add(col.replace('_p', ''))
                
                # Build columns in correct order: effect_p, then effect_corrected_p
                effect_cols = []
                for effect in sorted(effect_names):
                    if f'{effect}_p' in df_anova.columns:
                        effect_cols.append(f'{effect}_p')
                    if f'{effect}_corrected_p' in df_anova.columns:
                        effect_cols.append(f'{effect}_corrected_p')
                
                # Order: AP_Number -> Group stats -> ANOVA effects
                available_cols = base_cols + group_stat_cols + effect_cols
            else:
                # For one-way ANOVA: simpler format (matching Stats_parameters structure)
                base_cols = ['AP_Number', 'Test_Type']
                
                # Add group statistics columns (mean, SEM, n for each group) - FIRST after AP_Number
                # Extract group names and sort them, then add mean, SEM, n for each group in that order
                group_names_set = set()
                for col in df_anova.columns:
                    if col.endswith('_mean'):
                        group_names_set.add(col.replace('_mean', ''))
                
                group_stat_cols = []
                for group_name in sorted(group_names_set):
                    if f'{group_name}_mean' in df_anova.columns:
                        group_stat_cols.append(f'{group_name}_mean')
                    if f'{group_name}_SEM' in df_anova.columns:
                        group_stat_cols.append(f'{group_name}_SEM')
                    if f'{group_name}_n' in df_anova.columns:
                        group_stat_cols.append(f'{group_name}_n')
                
                # ANOVA statistics last (drop F-stat to avoid NaNs for Welch/KW)
                stat_cols = ['p_value']
                if 'corrected_p' in df_anova.columns:
                    stat_cols.append('corrected_p')
                
                # Order: AP_Number -> Group stats -> ANOVA stats
                available_cols = [col for col in base_cols + group_stat_cols + stat_cols if col in df_anova.columns]
            
            df_anova = df_anova[available_cols]
            anova_path = os.path.join(base_path, "Results", "Stats_AP_Num_vs_Peak_each_point_ANOVA.csv")
            df_anova.to_csv(anova_path, index=False)
            logger.info(f"Saved AP attenuation ANOVA statistics to {anova_path}")
        
        # Save post-hoc results
        if posthoc_results:
            df_posthoc = pd.DataFrame(posthoc_results)
            
            # Organize columns: AP info, comparison, group stats, then p-values
            base_cols = ['AP_Number', 'Test_Type', 'Comparison']
            group_cols = []
            stat_cols = ['p_value']
            if 'corrected_p' in df_posthoc.columns:
                stat_cols.append('corrected_p')
            
            # Columns to exclude from group_cols
            exclude_cols = base_cols + stat_cols + ['measurement_type']
            
            # Get group-related columns dynamically
            for col in df_posthoc.columns:
                if col not in exclude_cols and ('Group' in col or '_mean' in col or '_stderr' in col or '_n' in col):
                    group_cols.append(col)
            
            # Reorder columns: base + groups (sorted) + statistics (don't include measurement_type)
            column_order = base_cols + sorted(group_cols) + stat_cols
            available_cols = [col for col in column_order if col in df_posthoc.columns]
            df_posthoc = df_posthoc[available_cols]
            
            posthoc_path = os.path.join(base_path, "Results", "Stats_AP_Num_vs_Peak_each_point_pairwise.csv")
            df_posthoc.to_csv(posthoc_path, index=False)
            logger.info(f"Saved AP attenuation post-hoc statistics to {posthoc_path}")
    
    def _run_attenuation_mixed_effects(self, group1_data: pd.DataFrame, group2_data: pd.DataFrame,
                                      group1_name: str, group2_name: str) -> Optional[Dict]:
        """Run mixed-effects model for AP attenuation."""
        
        try:
            # Use first 10 APs
            g1 = group1_data[:10].copy()
            g2 = group2_data[:10].copy()
            
            g1['Genotype'] = group1_name
            g2['Genotype'] = group2_name
            g1['AP_num'] = [i for i in range(1, g1.shape[0] + 1)]
            g2['AP_num'] = [i for i in range(1, g2.shape[0] + 1)]
            
            # All columns are already value columns (filtered during loading)
            cols = list(g1.columns)
            
            ap_nums = []
            peaks = []
            genos = []
            cell_ids = []
            
            # Add group 2 data with consistent cell IDs
            for i, col in enumerate(cols):
                if col in g2.columns:
                    ap_nums += list(g2['AP_num'])
                    peaks += list(g2[col])
                    genos += list(g2['Genotype'])
                    cell_ids += [f"{group2_name}_cell_{i:03d}" for j in range(len(g2))]
            
            # Add group 1 data with consistent cell IDs
            for i, col in enumerate(cols):
                if col in g1.columns:
                    ap_nums += list(g1['AP_num'])
                    peaks += list(g1[col])
                    genos += list(g1['Genotype'])
                    cell_ids += [f"{group1_name}_cell_{i:03d}" for j in range(len(g1))]
            
            data = {'AP_num': ap_nums, 'Peak': peaks, 'Genotype': genos, 'Cell_id': cell_ids}
            final_df = pd.DataFrame(data)
            final_df = final_df.dropna()
            
            # Standardize continuous variable (z-score) to help convergence
            final_df['AP_num_z'] = (final_df['AP_num'] - final_df['AP_num'].mean()) / final_df['AP_num'].std()
            
            logger.info(f"Running independent mixed-effects model with random slopes for standardized AP_num...")
            
            # Mixed-effects model: Fixed effects for Genotype, AP_num, and interaction; Random slopes for AP_num
            model = mixedlm("Peak ~ C(Genotype) * AP_num_z", final_df, groups=final_df["Cell_id"],
                          re_formula="1 + AP_num_z").fit(method='lbfgs', maxiter=5000)
            
            # Check convergence
            if not model.converged:
                logger.warning(f"Independent attenuation model did not converge after 5000 iterations")
            else:
                logger.info(f"Independent attenuation model converged successfully")
            
            # Extract key statistics using proper statistical test (LRT for 3+ contrasts)
            formula = "Peak ~ C(Genotype) * AP_num_z"
            genotype_params = [p for p in model.pvalues.index if 'C(Genotype)' in p and ':' not in p]
            genotype_p = _get_effect_pvalue(model, genotype_params, final_df, formula, 'Cell_id', "1 + AP_num_z", "Genotype")
            
            interaction_params = [p for p in model.pvalues.index if 'C(Genotype)' in p and 'AP_num_z' in p]
            interaction_p = _get_effect_pvalue(model, interaction_params, final_df, formula, 'Cell_id', "1 + AP_num_z", "Genotype:AP_Number")
            
            result = {
                'Effect': ['Genotype', 'AP Number', 'Genotype:AP Number'],
                'p-value': [
                    genotype_p,
                    model.pvalues['AP_num_z'],
                    interaction_p
                ]
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error running attenuation mixed-effects model: {e}")
            return None
