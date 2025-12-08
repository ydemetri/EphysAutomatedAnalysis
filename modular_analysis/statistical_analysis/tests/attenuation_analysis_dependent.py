"""
Attenuation analysis for dependent (paired/repeated measures/mixed factorial) designs.
Handles AP number vs peak voltage using mixed effects models.
"""

import pandas as pd
import numpy as np
import logging
import os
from typing import List, Dict, Tuple
from itertools import combinations

import statsmodels.stats.multitest as multi
import statsmodels.formula.api as smf
import pingouin as pg
from scipy.stats import wilcoxon, friedmanchisquare

from ...shared.data_models import ExperimentalDesign
from ...shared.utils import should_use_parametric
from .posthoc_utils import (
    should_run_posthocs,
    get_simple_effect_comparisons_dependent,
    compute_global_curve_posthocs,
)

logger = logging.getLogger(__name__)

MIN_CELLS_PER_UNIT = 3
MIN_SUBJECTS = 10

def _valid_p(val: float) -> bool:
    return val is not None and np.isfinite(val)


def _pair_subject_value_columns(df: pd.DataFrame) -> List[Tuple[str, str]]:
    subject_cols = [col for col in df.columns if col.startswith('Subject_')]
    value_cols = [col for col in df.columns if col.startswith('Values_')]
    if len(subject_cols) != len(value_cols):
        logger.warning(
            "Subject/value column mismatch (subjects=%d, values=%d); pairing by index",
            len(subject_cols), len(value_cols)
        )
    min_len = min(len(subject_cols), len(value_cols))
    return list(zip(subject_cols[:min_len], value_cols[:min_len]))


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


class AttenuationAnalyzerDependent:
    """Attenuation analyzer for dependent designs using mixed models."""
    
    def __init__(self):
        self.name = "Attenuation Analyzer (Dependent)"
        
    def analyze_attenuation(self, design: ExperimentalDesign, base_path: str) -> Dict:
        """Analyze attenuation for dependent designs (paired or mixed factorial)."""
        
        results = {
            'point_by_point_stats': [],
            'mixed_effects_result': None,
            'attenuation_plot_data': None,
            'success': False
        }
        
        try:
            logger.info("Running dependent attenuation analysis...")
            
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
                    # Read CSV
                    group_data = pd.read_csv(attenuation_file, index_col=False)
                    
                    if not group_data.empty:
                        all_group_data[group.name] = group_data
                        # Calculate plot data for this group
                        group_plot_data = self._calculate_group_plot_data(group_data, group.name)
                        all_plot_data[group.name] = group_plot_data
                        logger.info(f"Loaded attenuation data for {group.name}")
                        
                except Exception as e:
                    logger.warning(f"Error loading attenuation data for {group.name}: {e}")
                    continue
            
            if len(all_group_data) < 2:
                logger.warning("Need at least 2 groups with sufficient attenuation data")
                return results
            
            # Set plot data
            if all_plot_data:
                results['attenuation_plot_data'] = all_plot_data
            
            # Detect design type: paired (1 group, 2 conditions) vs repeated measures (1 group, 3+ conditions) vs mixed factorial (2+ groups)
            manifest = design.pairing_manifest
            between_levels = sorted(manifest['Group'].unique())
            within_levels = sorted(manifest['Condition'].unique())
            
            if len(between_levels) == 1:
                if len(within_levels) == 2:
                    # Paired 2-group design
                    logger.info("Detected paired two-group design for attenuation analysis")
                    return self._analyze_paired_attenuation(design, base_path, all_group_data, within_levels, all_plot_data)
                else:
                    # Repeated measures (3+ conditions)
                    logger.info(f"Detected repeated measures design ({len(within_levels)} conditions) for attenuation analysis")
                    return self._analyze_repeated_measures_attenuation(design, base_path, all_group_data, within_levels, all_plot_data)
            else:
                # Mixed factorial design
                logger.info(f"Detected mixed factorial design ({len(between_levels)}×{len(within_levels)}) for attenuation analysis")
                return self._analyze_mixed_factorial_attenuation(design, base_path, all_group_data, between_levels, within_levels, all_plot_data)
            
        except Exception as e:
            logger.error(f"Error in attenuation analysis: {e}")
            results['error'] = str(e)
        return results
    
    def _calculate_group_plot_data(self, group_data: pd.DataFrame, group_name: str) -> Dict:
        """Calculate plot data for a single group (first 10 APs)."""
        
        # Use first 10 APs
        data_10ap = group_data.iloc[:10]
        
        # Get Subject and Values columns
        subject_cols = [col for col in data_10ap.columns if col.startswith('Subject_')]
        value_cols = [col for col in data_10ap.columns if col.startswith('Values_')]
        
        ap_numbers = list(range(1, 11))
        means = []
        sems = []
        
        for ap_idx in range(min(10, len(data_10ap))):
            # Collect all values for this AP across subjects
            ap_values = []
            for value_col in value_cols:
                val = data_10ap[value_col].iloc[ap_idx]
                if pd.notna(val):
                    ap_values.append(val)
            
            if ap_values:
                means.append(np.mean(ap_values))
                sems.append(np.std(ap_values) / np.sqrt(len(ap_values)) if len(ap_values) > 1 else 0)
            else:
                means.append(None)
                sems.append(None)
        
        return {
            'ap_numbers': ap_numbers[:len(means)],
            'mean': means,
            'sem': sems
        }
    
    def _analyze_paired_attenuation(self, design: ExperimentalDesign, base_path: str,
                                    all_group_data: Dict[str, pd.DataFrame],
                                    within_levels: List[str], all_plot_data: Dict) -> Dict:
        """Analyze attenuation for paired two-group design (simpler than mixed factorial)."""
        
        # Point-by-point paired t-tests for each AP
        anova_stats = self._run_pointwise_paired_ttest_attenuation(all_group_data, within_levels)
        if anova_stats:
            self._save_paired_attenuation_results(anova_stats, base_path)
        
        # Unified mixed-effects model (simpler formula for paired design)
        mixed_effects_result = self._run_global_mixed_effects_paired_attenuation(all_group_data, within_levels, design)
        
        logger.info(f"Paired attenuation analysis completed for {len(all_group_data)} groups")
        
        return {
            'point_by_point_stats': anova_stats,
            'mixed_effects_result': mixed_effects_result,
            'attenuation_plot_data': all_plot_data,
            'success': True
        }
    
    def _analyze_mixed_factorial_attenuation(self, design: ExperimentalDesign, base_path: str,
                                             all_group_data: Dict[str, pd.DataFrame],
                                             between_levels: List[str], within_levels: List[str],
                                             all_plot_data: Dict) -> Dict:
        """Analyze attenuation for mixed factorial design (existing logic)."""
        
        # Point-by-point mixed ANOVA for each AP position
        anova_stats, posthoc_stats = self._run_pointwise_mixed_model_attenuation(all_group_data, design)
        if anova_stats:
            self._save_attenuation_results(anova_stats, base_path, design)
        if posthoc_stats:
            self._save_posthoc_results(posthoc_stats, base_path)
        
        # Unified mixed-effects model
        mixed_effects_result = self._run_unified_mixed_effects_attenuation(all_group_data, design)
        
        logger.info(f"Mixed factorial attenuation analysis completed for {len(all_group_data)} groups")
        
        return {
            'point_by_point_stats': anova_stats,
            'mixed_effects_result': mixed_effects_result,
            'attenuation_plot_data': all_plot_data,
            'success': True
        }
    
    def _run_pointwise_paired_ttest_attenuation(self, all_group_data: Dict[str, pd.DataFrame], 
                                                 within_levels: List[str]) -> List[Dict]:
        """Run paired t-tests at each AP position for paired design."""
        
        results = []
        ap_frames = {}
        cond1_name, cond2_name = within_levels[0], within_levels[1]
        
        # Get data for both conditions - require exact match
        if cond1_name not in all_group_data or cond2_name not in all_group_data:
            logger.error(f"Conditions not found in group data. Expected: {within_levels}, Available: {list(all_group_data.keys())}")
            return []
        
        cond1_data = all_group_data[cond1_name]
        cond2_data = all_group_data[cond2_name]
        
        # Analyze first 10 APs
        max_ap = min(10, len(cond1_data), len(cond2_data))
        logger.info(f"Analyzing attenuation at {max_ap} AP positions (paired t-tests)")
        
        for ap_idx in range(max_ap):
            # Extract data for this AP
            pairs = _pair_subject_value_columns(cond1_data)
            
            # Match subjects and collect data
            data1_list = []
            data2_list = []
            
            for subject_col, value_col in pairs:
                if value_col in cond1_data.columns and value_col in cond2_data.columns:
                    val1 = cond1_data[value_col].iloc[ap_idx] if ap_idx < len(cond1_data) else np.nan
                    val2 = cond2_data[value_col].iloc[ap_idx] if ap_idx < len(cond2_data) else np.nan
                    
                    if pd.notna(val1) and pd.notna(val2):
                        data1_list.append(val1)
                        data2_list.append(val2)
            
            if len(data1_list) < MIN_SUBJECTS:
                continue
            
            # Choose paired t vs Wilcoxon based on skewness/kurtosis of differences
            diffs = np.array(data1_list) - np.array(data2_list)
            use_parametric = should_use_parametric([diffs])
            
            try:
                data1_arr = np.array(data1_list)
                data2_arr = np.array(data2_list)
                
                if use_parametric:
                    t_result = pg.ttest(data1_arr, data2_arr, paired=True)
                    p_value = t_result['p-val'].values[0]
                    test_label = "Paired t-test"
                else:
                    w_stat, p_value = wilcoxon(data1_arr, data2_arr, zero_method="wilcox", alternative="two-sided")
                    test_label = "Wilcoxon signed-rank"
                
                results.append({
                    'ap_number': ap_idx + 1,  # 1-indexed
                    'Test': test_label,
                    f'{cond1_name}_mean': data1_arr.mean(),
                    f'{cond1_name}_SEM': data1_arr.std(ddof=1) / np.sqrt(len(data1_arr)),
                    f'{cond1_name}_n': len(data1_arr),
                    f'{cond2_name}_mean': data2_arr.mean(),
                    f'{cond2_name}_SEM': data2_arr.std(ddof=1) / np.sqrt(len(data2_arr)),
                    f'{cond2_name}_n': len(data2_arr),
                    'p_value': p_value
                })
                
            except Exception as e:
                logger.warning(f"Error in paired t-test at AP {ap_idx + 1}: {e}")
        # Apply FDR correction (handling NaN p-values)
        if len(results) > 1:
            # Extract p-values and track valid indices
            valid_indices = [i for i, r in enumerate(results) if _valid_p(r['p_value'])]
            valid_p_values = [results[i]['p_value'] for i in valid_indices]
            
            if len(valid_p_values) > 1:
                try:
                    _, corrected_p, _, _ = multi.multipletests(valid_p_values, method="fdr_bh")
                    # Map corrected p-values back to valid indices
                    for i, corrected_val in zip(valid_indices, corrected_p):
                        results[i]['corrected_p'] = corrected_val
                    # Set NaN for invalid p-values
                    for i in range(len(results)):
                        if i not in valid_indices:
                            results[i]['corrected_p'] = np.nan
                    logger.info(f"Applied FDR correction to {len(valid_p_values)} paired t-tests")
                except Exception as e:
                    logger.error(f"Error applying FDR correction: {e}")
                    # Fallback: use uncorrected p-values
                    for result in results:
                        result['corrected_p'] = result['p_value']
            else:
                # Only 1 or 0 valid p-values - copy raw p-values for valid tests
                for idx in valid_indices:
                    results[idx]['corrected_p'] = results[idx]['p_value']
                for i in range(len(results)):
                    if i not in valid_indices:
                        results[i]['corrected_p'] = np.nan
        else:
            # Single result - copy raw p-value if valid
            for result in results:
                result['corrected_p'] = (
                    result['p_value'] if _valid_p(result['p_value']) else np.nan
                )
        
        return results
    
    def _run_global_mixed_effects_paired_attenuation(self, all_group_data: Dict[str, pd.DataFrame], 
                                                      within_levels: List[str],
                                                      design: ExperimentalDesign):
        """Run global mixed-effects model for paired attenuation design (simpler formula)."""
        
        all_data = []
        cond1_name, cond2_name = within_levels[0], within_levels[1]
        factor_name = design.within_factor_name or "Condition"  # Use custom name or default
        
        # Get data for both conditions - require exact match
        if cond1_name not in all_group_data or cond2_name not in all_group_data:
            logger.error(f"Conditions not found in group data. Expected: {within_levels}, Available: {list(all_group_data.keys())}")
            return None
        
        cond1_data = all_group_data[cond1_name]
        cond2_data = all_group_data[cond2_name]
        
        # Collect data from first 10 APs
        max_ap = min(10, len(cond1_data), len(cond2_data))
        
        for ap_idx in range(max_ap):
            for subject_col, value_col in _pair_subject_value_columns(cond1_data):
                if value_col in cond1_data.columns and value_col in cond2_data.columns:
                    peak1 = cond1_data[value_col].iloc[ap_idx] if ap_idx < len(cond1_data) else np.nan
                    peak2 = cond2_data[value_col].iloc[ap_idx] if ap_idx < len(cond2_data) else np.nan
                    
                    if pd.notna(peak1):
                        all_data.append({
                            'Subject_ID': subject_col,
                            'Condition': cond1_name,
                            'AP_num': ap_idx + 1,
                            'Peak': peak1
                        })
                    if pd.notna(peak2):
                        all_data.append({
                            'Subject_ID': subject_col,
                            'Condition': cond2_name,
                            'AP_num': ap_idx + 1,
                            'Peak': peak2
                        })
        
        if not all_data:
            logger.warning("No data for paired attenuation global mixed-effects model")
            return None
        
        unified_df = pd.DataFrame(all_data)
        
        try:
            # Drop NaN
            unified_df = unified_df.dropna(subset=['Peak', 'AP_num', 'Subject_ID', 'Condition'])
            
            # Standardize AP number
            unified_df['AP_num_z'] = (unified_df['AP_num'] - unified_df['AP_num'].mean()) / unified_df['AP_num'].std()
            
            logger.info("Running global mixed-effects model for paired attenuation design")
            
            # Simpler formula for paired: no between factor
            formula = "Peak ~ C(Condition) * AP_num_z"
            model = smf.mixedlm(
                formula,
                unified_df,
                groups=unified_df["Subject_ID"],
                re_formula="1 + AP_num_z"
            ).fit(method='lbfgs', maxiter=5000)
            
            if not model.converged:
                logger.warning("Paired attenuation global model did not converge")
            else:
                logger.info("Paired attenuation global model converged successfully")
            
            # Extract p-values (simpler: only Condition, AP, and Condition:AP)
            # For paired designs, there's only 1 contrast for Condition - extract directly
            condition_params = [p for p in model.pvalues.index if 'C(Condition)' in p and ':' not in p]
            condition_p = model.pvalues[condition_params[0]] if len(condition_params) == 1 else 1.0
            
            ap_p = model.pvalues.get('AP_num_z', 1.0)
            
            interaction_params = [p for p in model.pvalues.index if 'C(Condition)' in p and 'AP_num_z' in p]
            interaction_p = model.pvalues[interaction_params[0]] if len(interaction_params) == 1 else 1.0
            
            rows = [
                {'Effect': factor_name, 'p-value': condition_p},
                {'Effect': 'AP Number', 'p-value': ap_p},
                {'Effect': f'{factor_name}:AP Number', 'p-value': interaction_p}
            ]
            
            logger.info("Paired attenuation global mixed-effects model complete")
            return rows
            
        except Exception as e:
            logger.error(f"Error running paired attenuation global mixed-effects model: {e}")
            return None
    
    def _save_paired_attenuation_results(self, results: List[Dict], base_path: str) -> None:
        """Save point-by-point paired t-test attenuation results to CSV."""
        
        if not results:
            logger.warning("No paired attenuation results to save")
            return
        
        try:
            df = pd.DataFrame(results)
            
            # Reorder columns: AP_Number -> Group stats -> p-values
            base_cols = ['ap_number']
            
            # Extract group names and build stat columns
            group_names_set = set()
            for col in df.columns:
                if col.endswith('_mean'):
                    group_names_set.add(col.replace('_mean', ''))
            
            group_stat_cols = []
            for group_name in sorted(group_names_set):
                for suffix in ['_mean', '_SEM', '_n']:
                    col_name = f'{group_name}{suffix}'
                    if col_name in df.columns:
                        group_stat_cols.append(col_name)
            
            p_value_cols = ['p_value', 'corrected_p']
            
            # Rename ap_number to AP_Number
            df = df.rename(columns={'ap_number': 'AP_Number'})
            
            column_order = ['AP_Number'] + group_stat_cols + p_value_cols
            available_cols = [col for col in column_order if col in df.columns]
            df = df[available_cols]
            
            output_path = os.path.join(base_path, "Results", "Stats_AP_Num_vs_Peak_each_point_Paired.csv")
            df.to_csv(output_path, index=False)
            logger.info(f"Saved paired attenuation results to {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving paired attenuation results: {e}")
    
    def _analyze_repeated_measures_attenuation(self, design: ExperimentalDesign, base_path: str,
                                              all_group_data: Dict[str, pd.DataFrame],
                                              within_levels: List[str], all_plot_data: Dict) -> Dict:
        """Analyze attenuation for repeated measures design (3+ conditions, single group)."""
        
        # Point-by-point RM-ANOVA for each AP
        anova_stats = self._run_pointwise_rm_anova_attenuation(all_group_data, within_levels, design)
        if anova_stats:
            self._save_rm_anova_attenuation_results(anova_stats, base_path)
        
        # Global mixed-effects model
        mixed_effects_result = self._run_global_mixed_effects_rm_attenuation(all_group_data, within_levels, design)
        
        logger.info(f"Repeated measures attenuation analysis completed for {len(all_group_data)} groups")
        
        return {
            'point_by_point_stats': anova_stats,
            'mixed_effects_result': mixed_effects_result,
            'attenuation_plot_data': all_plot_data,
            'success': True
        }
    
    def _run_pointwise_rm_anova_attenuation(self, all_group_data: Dict[str, pd.DataFrame],
                                           within_levels: List[str], design: ExperimentalDesign) -> List[Dict]:
        """Run RM-ANOVA at each AP number for repeated measures design."""
        
        results = []
        ap_frames = {}
        max_ap = 10
        
        # Map condition names to group folder names - require exact match
        condition_data = {}
        for condition in within_levels:
            if condition in all_group_data:
                condition_data[condition] = all_group_data[condition]
            else:
                logger.warning(f"Condition '{condition}' not found in group data. Available: {list(all_group_data.keys())}")
        
        if len(condition_data) < len(within_levels):
            logger.warning(f"Could not match all conditions to group data. Expected {len(within_levels)}, found {len(condition_data)}")
        
        if len(condition_data) < 3:
            logger.error(f"Insufficient conditions mapped: {len(condition_data)}/3 required")
            return []
        
        logger.info(f"Analyzing attenuation at {max_ap} AP positions (RM-ANOVA)")
        
        for ap_idx in range(max_ap):
            # Build long-format DataFrame for this AP
            long_data = []
            
            for condition in within_levels:
                if condition not in condition_data:
                    continue
                
                cond_df = condition_data[condition]
                
                if ap_idx >= len(cond_df):
                    continue
                
                # Extract data for this AP across all subjects
                for subject_col, value_col in _pair_subject_value_columns(cond_df):
                    if value_col in cond_df.columns:
                        peak = cond_df[value_col].iloc[ap_idx]
                        
                        if pd.notna(peak):
                            long_data.append({
                                'Subject_ID': subject_col,
                                'Condition': condition,
                                'Peak': peak
                            })
            
            if len(long_data) < len(within_levels) * MIN_CELLS_PER_UNIT:
                continue
            
            df_long = pd.DataFrame(long_data)
            
            # Decide parametric vs nonparametric using residuals (value minus subject mean)
            subject_means = (
                df_long.groupby('Subject_ID')['Peak'].mean().rename('SubjectMean')
            )
            df_long = df_long.merge(subject_means, left_on='Subject_ID', right_index=True, how='left')
            residuals = (df_long['Peak'] - df_long['SubjectMean']).values
            use_parametric = should_use_parametric([residuals])
            
            try:
                if use_parametric:
                    aov = pg.rm_anova(
                        dv='Peak',
                        within='Condition',
                        subject='Subject_ID',
                        data=df_long
                    )
                    p_value = aov['p-unc'].values[0]
                    test_label = "RM_ANOVA"
                else:
                    pivot = df_long.pivot(index='Subject_ID', columns='Condition', values='Peak').dropna()
                    if pivot.shape[0] < 2:
                        p_value = np.nan
                        test_label = "Friedman"
                    else:
                        stat, p_value = friedmanchisquare(*[pivot[c].values for c in pivot.columns])
                        test_label = "Friedman"
                
                # Calculate group statistics
                result = {
                    'ap_number': ap_idx + 1,
                    'rm_anova_p': p_value,
                    'Test': test_label
                }
                
                # Add statistics for each condition
                for condition in within_levels:
                    cond_data = df_long[df_long['Condition'] == condition]['Peak']
                    result[f'{condition}_mean'] = cond_data.mean()
                    result[f'{condition}_SEM'] = cond_data.std(ddof=1) / np.sqrt(len(cond_data)) if len(cond_data) > 1 else 0
                    result[f'{condition}_n'] = len(cond_data)
                
                if not np.isnan(p_value):
                    results.append(result)
                    ap_frames[ap_idx + 1] = df_long.copy()
                
            except Exception as e:
                logger.warning(f"Error in RM-ANOVA/Friedman at AP {ap_idx + 1}: {e}")
        
        # Apply FDR correction to RM-ANOVA/Friedman p-values
        if len(results) > 1:
            p_values = [r['rm_anova_p'] for r in results if _valid_p(r.get('rm_anova_p'))]
            try:
                if len(p_values) > 1:
                    _, corrected_p, _, _ = multi.multipletests(p_values, method="fdr_bh")
                    idx = 0
                    for result in results:
                        if _valid_p(result.get('rm_anova_p')):
                            result['rm_anova_corrected_p'] = corrected_p[idx]
                            idx += 1
                        else:
                            result['rm_anova_corrected_p'] = np.nan
                    logger.info(f"Applied FDR correction to {len(p_values)} RM-ANOVA/Friedman tests")
                else:
                    for result in results:
                        if _valid_p(result.get('rm_anova_p')):
                            result['rm_anova_corrected_p'] = result['rm_anova_p']
                        else:
                            result['rm_anova_corrected_p'] = np.nan
            except Exception as e:
                logger.error(f"Error applying FDR correction: {e}")
                for result in results:
                    result['rm_anova_corrected_p'] = result['rm_anova_p']
        
        # Run post-hoc tests only where corrected p-values remain significant
        for result in results:
            corrected = result.get('rm_anova_corrected_p', result.get('rm_anova_p'))
            if corrected is None or np.isnan(corrected) or corrected >= 0.05:
                continue
            ap_number = result.get('ap_number')
            df_long = ap_frames.get(ap_number)
            if df_long is None:
                continue
            logger.info(f"RM-ANOVA/Friedman significant at AP {ap_number} (corrected p = {corrected:.4f}) - running post-hoc tests")
            is_nonparam = result.get('Test') == 'Friedman'
            for cond1, cond2 in combinations(within_levels, 2):
                try:
                    paired_data1 = []
                    paired_data2 = []
                    for subject in df_long['Subject_ID'].unique():
                        subj_data = df_long[df_long['Subject_ID'] == subject]
                        val1 = subj_data[subj_data['Condition'] == cond1]['Peak']
                        val2 = subj_data[subj_data['Condition'] == cond2]['Peak']
                        if len(val1) > 0 and len(val2) > 0:
                            paired_data1.append(val1.iloc[0])
                            paired_data2.append(val2.iloc[0])
                    if len(paired_data1) >= MIN_CELLS_PER_UNIT:
                        if is_nonparam:
                            w_stat, posthoc_p = wilcoxon(
                                np.array(paired_data1),
                                np.array(paired_data2),
                                zero_method="wilcox",
                                alternative="two-sided"
                            )
                        else:
                            t_result = pg.ttest(np.array(paired_data1), np.array(paired_data2), paired=True)
                            posthoc_p = t_result['p-val'].values[0]
                        result[f'{cond1}_vs_{cond2}_posthoc_p'] = posthoc_p
                except Exception as e:
                    logger.warning(f"Error in post-hoc test at AP {ap_number} ({cond1} vs {cond2}): {e}")
        
        # Apply FDR correction to post-hoc p-values (within each comparison)
        if results:
            # Get all unique pairwise comparisons
            posthoc_keys = set()
            for result in results:
                for key in result.keys():
                    if '_posthoc_p' in key:
                        posthoc_keys.add(key)
            
            # For each pairwise comparison, collect and correct p-values across APs
            for posthoc_key in posthoc_keys:
                posthoc_p_values = []
                valid_indices = []
                
                for i, result in enumerate(results):
                    if posthoc_key in result:
                        posthoc_p_values.append(result[posthoc_key])
                        valid_indices.append(i)
                
                if len(posthoc_p_values) > 1:
                    try:
                        _, corrected_p, _, _ = multi.multipletests(posthoc_p_values, method="fdr_bh")
                        corrected_key = posthoc_key.replace('_posthoc_p', '_posthoc_corrected_p')
                        for i, idx in enumerate(valid_indices):
                            results[idx][corrected_key] = corrected_p[i]
                    except Exception as e:
                        logger.error(f"Error applying post-hoc FDR correction for {posthoc_key}: {e}")
        
        return results
    
    def _run_global_mixed_effects_rm_attenuation(self, all_group_data: Dict[str, pd.DataFrame],
                                                 within_levels: List[str],
                                                 design: ExperimentalDesign):
        """Run global mixed-effects model for repeated measures attenuation design."""
        
        all_data = []
        factor_name = design.within_factor_name or "Condition"
        
        # Collect data from first 10 APs
        max_ap = 10
        
        for condition in within_levels:
            # Require exact match between condition name and folder name
            if condition not in all_group_data:
                logger.warning(f"Condition '{condition}' not found in group data. Available: {list(all_group_data.keys())}")
                continue
            
            cond_data = all_group_data[condition]
            
            for ap_idx in range(min(max_ap, len(cond_data))):
                for subject_col, value_col in _pair_subject_value_columns(cond_data):
                    if value_col in cond_data.columns:
                        peak = cond_data[value_col].iloc[ap_idx]
                        
                        if pd.notna(peak):
                            all_data.append({
                                'Subject_ID': subject_col,
                                'Condition': condition,
                                'AP_num': ap_idx + 1,
                                'Peak': peak
                            })
        
        if not all_data:
            logger.warning("No data for repeated measures attenuation global mixed-effects model")
            return None
        
        unified_df = pd.DataFrame(all_data)
        
        try:
            # Drop NaN
            unified_df = unified_df.dropna(subset=['Peak', 'AP_num', 'Subject_ID', 'Condition'])
            
            # Standardize AP number
            unified_df['AP_num_z'] = (unified_df['AP_num'] - unified_df['AP_num'].mean()) / unified_df['AP_num'].std()
            
            logger.info(f"Running global mixed-effects model for repeated measures attenuation design ({len(within_levels)} conditions)")
            
            # Formula: Peak ~ C(Condition) * AP_num_z
            formula = "Peak ~ C(Condition) * AP_num_z"
            model = smf.mixedlm(
                formula,
                unified_df,
                groups=unified_df["Subject_ID"],
                re_formula="1 + AP_num_z"
            ).fit(method='lbfgs', maxiter=5000)
            
            if not model.converged:
                logger.warning("Repeated measures attenuation global model did not converge")
            else:
                logger.info("Repeated measures attenuation global model converged successfully")
            
            # Extract p-values using proper statistical tests (LRT for multiple contrasts)
            condition_params = [p for p in model.pvalues.index if 'C(Condition)' in p and ':' not in p]
            condition_p = _get_effect_pvalue(model, condition_params, unified_df, formula, 'Subject_ID', "1 + AP_num_z", "Condition")
            
            ap_p = model.pvalues.get('AP_num_z', 1.0)
            
            interaction_params = [p for p in model.pvalues.index if 'C(Condition)' in p and 'AP_num_z' in p]
            interaction_p = _get_effect_pvalue(model, interaction_params, unified_df, formula, 'Subject_ID', "1 + AP_num_z", "Condition:AP_Number")
            
            rows = [
                {'Effect': factor_name, 'p-value': condition_p},
                {'Effect': 'AP Number', 'p-value': ap_p},
                {'Effect': f'{factor_name}:AP Number', 'p-value': interaction_p}
            ]
            
            logger.info("Repeated measures attenuation global mixed-effects model complete")
            return rows
            
        except Exception as e:
            logger.error(f"Error running repeated measures attenuation global mixed-effects model: {e}")
            return None
    
    def _save_rm_anova_attenuation_results(self, results: List[Dict], base_path: str) -> None:
        """Save point-by-point RM-ANOVA attenuation results to CSV - split into main test and post-hocs."""
        
        if not results:
            logger.warning("No RM-ANOVA attenuation results to save")
            return
        
        try:
            df = pd.DataFrame(results)
            
            # Extract condition names and build stat columns
            condition_names_set = set()
            for col in df.columns:
                if col.endswith('_mean'):
                    condition_names_set.add(col.replace('_mean', ''))
            
            condition_stat_cols = []
            for cond_name in sorted(condition_names_set):
                for suffix in ['_mean', '_SEM', '_n']:
                    col_name = f'{cond_name}{suffix}'
                    if col_name in df.columns:
                        condition_stat_cols.append(col_name)
            
            # RM-ANOVA p-value columns
            anova_cols = ['rm_anova_p', 'rm_anova_corrected_p']
            
            # Post-hoc p-value columns (all remaining columns with posthoc in name)
            # Sort with uncorrected p-values before corrected p-values for each comparison
            posthoc_cols = [col for col in df.columns if 'posthoc' in col and col not in anova_cols]
            
            # Custom sort: group by comparison, put uncorrected before corrected
            def posthoc_sort_key(col):
                # Replace _corrected_p with _p for sorting, then add a suffix to sort corrected after uncorrected
                if '_posthoc_corrected_p' in col:
                    comparison = col.replace('_posthoc_corrected_p', '')
                    return (comparison, 1)  # Sort corrected second
                elif '_posthoc_p' in col:
                    comparison = col.replace('_posthoc_p', '')
                    return (comparison, 0)  # Sort uncorrected first
                return (col, 0)
            
            posthoc_cols = sorted(posthoc_cols, key=posthoc_sort_key)
            
            # Rename ap_number to AP_Number
            df = df.rename(columns={'ap_number': 'AP_Number'})
            
            # Save main RM-ANOVA results file (without post-hocs)
            anova_column_order = ['AP_Number'] + condition_stat_cols + anova_cols
            available_anova_cols = [col for col in anova_column_order if col in df.columns]
            df_anova = df[available_anova_cols].copy()
            
            anova_path = os.path.join(base_path, "Results", "Stats_AP_Num_vs_Peak_each_point_RM_ANOVA.csv")
            df_anova.to_csv(anova_path, index=False)
            logger.info(f"Saved RM-ANOVA attenuation results to {anova_path}")
            
            # Save post-hoc results file (only if there are post-hocs)
            if posthoc_cols:
                posthoc_column_order = ['AP_Number'] + condition_stat_cols + posthoc_cols
                available_posthoc_cols = [col for col in posthoc_column_order if col in df.columns]
                df_posthoc = df[available_posthoc_cols].copy()
                
                posthoc_path = os.path.join(base_path, "Results", "Stats_AP_Num_vs_Peak_each_point_pairwise.csv")
                df_posthoc.to_csv(posthoc_path, index=False)
                logger.info(f"Saved RM-ANOVA attenuation post-hoc results to {posthoc_path}")
            else:
                logger.info("No post-hoc comparisons to save for RM-ANOVA attenuation")
            
        except Exception as e:
            logger.error(f"Error saving RM-ANOVA attenuation results: {e}")
    
    def _run_pointwise_mixed_model_attenuation(self, all_group_data: Dict[str, pd.DataFrame], design: ExperimentalDesign) -> Tuple[List[Dict], List[Dict]]:
        """Run point-by-point mixed ANOVA for each AP position with post-hocs.
        
        Returns:
            Tuple of (anova_results, posthoc_results)
        """
        manifest = design.pairing_manifest
        anova_results = []
        posthoc_results = []
        
        # Analyze first 10 APs only (matching independent design)
        max_ap = min(10, max(len(df) for df in all_group_data.values()))
        logger.info(f"Analyzing attenuation for first {max_ap} AP positions")
        
        for ap_idx in range(max_ap):
            # Collect data for this AP from all groups
            ap_data_list = []
            
            for group_name, group_data in all_group_data.items():
                if ap_idx < len(group_data):
                    between_level, within_level = self._parse_group_name(group_name, manifest)
                    subject_cols = [col for col in group_data.columns if col.startswith('Subject_')]
                    value_cols = [col for col in group_data.columns if col.startswith('Values_')]
                    
                    for subject_col, value_col in zip(subject_cols, value_cols):
                        subject_id = subject_col
                        peak_voltage = group_data[value_col].iloc[ap_idx]
                        if pd.notna(peak_voltage):
                            ap_data_list.append({
                                'Subject_ID': str(subject_id),
                                'Between_Factor': between_level,
                                'Within_Factor': within_level,
                                'Peak_Voltage': peak_voltage
                            })
            
            if len(ap_data_list) < 4:  # Reduced threshold - need at least 4 for 2x2
                logger.debug(f"Skipping AP {ap_idx + 1}: insufficient data (n={len(ap_data_list)})")
                continue
            
            df_ap = pd.DataFrame(ap_data_list)
            
            # Run mixed ANOVA
            try:
                aov = pg.mixed_anova(
                    dv='Peak_Voltage',
                    within='Within_Factor',
                    between='Between_Factor',
                    subject='Subject_ID',
                    data=df_ap
                )
                
                # Safely extract p-values, handling missing columns
                if 'p-unc' not in aov.columns:
                    logger.warning(f"Skipping AP {ap_idx + 1}: ANOVA table missing p-unc column (likely unbalanced data)")
                    continue
                
                between_p = aov[aov['Source'] == 'Between_Factor']['p-unc'].values[0] if 'Between_Factor' in aov['Source'].values else np.nan
                within_p = aov[aov['Source'] == 'Within_Factor']['p-unc'].values[0] if 'Within_Factor' in aov['Source'].values else np.nan
                interaction_p = aov[aov['Source'] == 'Interaction']['p-unc'].values[0] if 'Interaction' in aov['Source'].values else np.nan
                
                # Calculate group statistics
                result = {
                    'ap_number': ap_idx + 1,  # 1-indexed
                    'between_p': between_p,
                    'within_p': within_p,
                    'interaction_p': interaction_p,
                    '_df_ap': df_ap  # Store for later post-hoc analysis
                }
                
                # Add group statistics (mean, SEM, n for each Between x Within combination)
                for (between_level, within_level), group_df in df_ap.groupby(['Between_Factor', 'Within_Factor']):
                    group_name = f"{between_level}: {within_level}"
                    result[f'{group_name}_mean'] = group_df['Peak_Voltage'].mean()
                    result[f'{group_name}_SEM'] = group_df['Peak_Voltage'].std(ddof=1) / np.sqrt(len(group_df))
                    result[f'{group_name}_n'] = len(group_df)
                
                anova_results.append(result)
                
            except Exception as e:
                logger.warning(f"Error at AP {ap_idx + 1}: {e}")
        
        # Apply FDR correction to ANOVA results
        anova_results = self._apply_fdr_to_results(anova_results)
        
        # Now run post-hocs based on corrected p-values (only if needed)
        posthoc_results = []
        for result in anova_results:
            # Check if any corrected effect is significant AND requires post-hocs
            between_p_corr = result.get('between_corrected_p', result.get('between_p', 1))
            within_p_corr = result.get('within_corrected_p', result.get('within_p', 1))
            interaction_p_corr = result.get('interaction_corrected_p', result.get('interaction_p', 1))
            
            # Create result dict for helper function
            anova_result = {
                f'{design.between_factor_name}_corrected_p': between_p_corr,
                f'{design.within_factor_name}_corrected_p': within_p_corr,
                'Interaction_corrected_p': interaction_p_corr
            }
            
            posthoc_decision = should_run_posthocs(anova_result, design, [g.name for g in design.groups])
            
            if posthoc_decision['run_posthocs']:
                df_ap = result.pop('_df_ap', None)  # Remove and retrieve stored data
                if df_ap is not None:
                    ap_number = result.get('ap_number')
                    logger.info(f"AP {ap_number}: Running post-hocs for {posthoc_decision['reasons']}")
                    ap_posthocs = self._run_posthocs_at_ap(
                        df_ap, ap_number, design,
                        between_p_corr, within_p_corr, interaction_p_corr,
                        posthoc_decision.get('reasons', [])
                    )
                    posthoc_results.extend(ap_posthocs)
            else:
                # Remove stored data even if not running post-hocs
                result.pop('_df_ap', None)
        
        # Apply FDR correction to post-hoc results (separately for paired and independent)
        posthoc_results = self._apply_posthoc_fdr(posthoc_results)
        
        logger.info(f"Attenuation analysis complete: {len(anova_results)} AP positions, {len(posthoc_results)} post-hocs")
        return anova_results, posthoc_results
    
    def _run_unified_mixed_effects_attenuation(self, all_group_data: Dict[str, pd.DataFrame], design: ExperimentalDesign):
        """Run unified mixed-effects model across first 10 AP positions."""
        manifest = design.pairing_manifest
        all_data = []
        
        # Analyze first 10 APs only (matching independent design)
        max_ap = min(10, max(len(df) for df in all_group_data.values()))
        
        for ap_idx in range(max_ap):
            for group_name, group_data in all_group_data.items():
                if ap_idx < len(group_data):
                    between_level, within_level = self._parse_group_name(group_name, manifest)
                    subject_cols = [col for col in group_data.columns if col.startswith('Subject_')]
                    value_cols = [col for col in group_data.columns if col.startswith('Values_')]
                    
                    for subject_col, value_col in zip(subject_cols, value_cols):
                        # Subject_ID is the column name itself (e.g., "Subject_Scn1a_1")
                        subject_id = subject_col
                        peak_voltage = group_data[value_col].iloc[ap_idx]
                        if pd.notna(peak_voltage):
                            all_data.append({
                                'Subject_ID': str(subject_id),
                                'Between_Factor': between_level,
                                'Within_Factor': within_level,
                                'AP_num': ap_idx + 1,
                                'Peak': peak_voltage
                            })
        
        unified_df = pd.DataFrame(all_data)
        return self._run_global_mixed_effects(unified_df, design)
    
    def _parse_group_name(self, group_name: str, manifest: pd.DataFrame) -> Tuple[str, str]:
        """Parse group name to extract between and within factor levels.
        
        Uses exact matching to avoid issues with underscores/spaces in factor level names.
        Tries all combinations of manifest Group × Condition values.
        
        Format: "{Condition}_{Group}" or "{Condition} {Group}"
        For example: "32_Scn1a", "32 Scn1a", "treatment_1_WT", "temp_high Scn1a"
        """
        
        between_levels = list(manifest['Group'].unique())
        within_levels = list(manifest['Condition'].unique())
        
        # Try all combinations of between × within with both separators
        # This handles cases where condition or group names contain underscores/spaces
        for between in between_levels:
            for within in within_levels:
                # Try: {Condition}_{Group}
                if group_name == f"{within}_{between}":
                    return (between, within)
                # Try: {Condition} {Group}  
                if group_name == f"{within} {between}":
                    return (between, within)
        
        # No match found
        raise ValueError(
            f"Could not parse group name '{group_name}'. "
            f"Expected format: {{Condition}}_{{Group}} or {{Condition}} {{Group}} "
            f"where Condition is one of {within_levels} and Group is one of {between_levels}"
        )
    
    def _run_posthocs_at_ap(self, df_ap: pd.DataFrame, ap_idx: int, design: ExperimentalDesign,
                            between_p: float, within_p: float, interaction_p: float, 
                            reasons: List[str], alpha: float = 0.05) -> List[Dict]:
        """Run post-hoc t-tests at a specific AP position if ANOVA was significant (corrected p-values)."""
        
        posthocs = []
        
        # Run post-hocs if any effect is significant (using corrected p-values)
        if not (between_p < alpha or within_p < alpha or interaction_p < alpha):
            return posthocs
        
        between_levels = sorted(df_ap['Between_Factor'].unique())
        within_levels = sorted(df_ap['Within_Factor'].unique())
        
        if 'interaction' in reasons:
            # Within-factor comparisons at each between-level (PAIRED t-tests)
            for between_level in between_levels:
                subset = df_ap[df_ap['Between_Factor'] == between_level]
                
                for wlevel1, wlevel2 in combinations(within_levels, 2):
                    try:
                        data1 = subset[subset['Within_Factor'] == wlevel1].set_index('Subject_ID')['Peak_Voltage']
                        data2 = subset[subset['Within_Factor'] == wlevel2].set_index('Subject_ID')['Peak_Voltage']
                        
                        common_subjects = data1.index.intersection(data2.index)
                        if len(common_subjects) < MIN_CELLS_PER_UNIT:
                            continue
                        
                        data1_paired = data1.loc[common_subjects]
                        data2_paired = data2.loc[common_subjects]
                        
                        t_result = pg.ttest(data1_paired, data2_paired, paired=True)
                        
                        posthocs.append({
                            'ap_number': ap_idx,
                            'Test_Type': 'Paired t-test',
                            'Comparison': f"{between_level}: {wlevel1} vs {wlevel2}",
                            'Group1': f"{between_level}: {wlevel1}",
                            'Group1_mean': data1_paired.mean(),
                            'Group1_stderr': data1_paired.std(ddof=1) / np.sqrt(len(data1_paired)),
                            'Group1_n': len(common_subjects),
                            'Group2': f"{between_level}: {wlevel2}",
                            'Group2_mean': data2_paired.mean(),
                            'Group2_stderr': data2_paired.std(ddof=1) / np.sqrt(len(data2_paired)),
                            'Group2_n': len(common_subjects),
                            'p_value': t_result['p-val'].values[0]
                        })
                    except Exception as e:
                        logger.debug(f"Error in paired t-test at AP {ap_idx}: {e}")
            
            # Between-factor comparisons at each within-level (INDEPENDENT t-tests)
            for within_level in within_levels:
                subset = df_ap[df_ap['Within_Factor'] == within_level]
                
                for blevel1, blevel2 in combinations(between_levels, 2):
                    try:
                        data1 = subset[subset['Between_Factor'] == blevel1]['Peak_Voltage']
                        data2 = subset[subset['Between_Factor'] == blevel2]['Peak_Voltage']
                        
                        if len(data1) < MIN_CELLS_PER_UNIT or len(data2) < MIN_CELLS_PER_UNIT:
                            continue
                        
                        t_result = pg.ttest(data1, data2, paired=False)
                        
                        posthocs.append({
                            'ap_number': ap_idx,
                            'Test_Type': 'Independent t-test',
                            'Comparison': f"{within_level}: {blevel1} vs {blevel2}",
                            'Group1': f"{within_level}: {blevel1}",
                            'Group1_mean': data1.mean(),
                            'Group1_stderr': data1.std(ddof=1) / np.sqrt(len(data1)),
                            'Group1_n': len(data1),
                            'Group2': f"{within_level}: {blevel2}",
                            'Group2_mean': data2.mean(),
                            'Group2_stderr': data2.std(ddof=1) / np.sqrt(len(data2)),
                            'Group2_n': len(data2),
                            'p_value': t_result['p-val'].values[0]
                        })
                    except Exception as e:
                        logger.debug(f"Error in independent t-test at AP {ap_idx}: {e}")
        else:
            if 'factor2_main' in reasons:
                posthocs.extend(self._run_marginal_within_ap(df_ap, ap_idx, design))
            if 'factor1_main' in reasons:
                posthocs.extend(self._run_marginal_between_ap(df_ap, ap_idx, design))
        
        return posthocs
    
    def _run_marginal_within_ap(self, df_ap: pd.DataFrame, ap_idx: int,
                                design: ExperimentalDesign) -> List[Dict]:
        """Marginal paired comparisons across within-factor levels."""
        results = []
        df_local = df_ap.copy()
        df_local['Subject_Key'] = df_local['Between_Factor'].astype(str) + "__" + df_local['Subject_ID'].astype(str)
        within_levels = sorted(df_local['Within_Factor'].unique())
        pivot = df_local.pivot_table(
            index='Subject_Key',
            columns='Within_Factor',
            values='Peak_Voltage',
            aggfunc='mean'
        )
        
        for wlevel1, wlevel2 in combinations(within_levels, 2):
            if wlevel1 not in pivot.columns or wlevel2 not in pivot.columns:
                continue
            paired = pivot[[wlevel1, wlevel2]].dropna()
            if len(paired) < 3:
                continue
            try:
                t_result = pg.ttest(paired[wlevel1], paired[wlevel2], paired=True)
                results.append({
                    'ap_number': ap_idx,
                    'Test_Type': 'Marginal paired t-test',
                    'Comparison': f"{design.within_factor_name}={wlevel1} vs {design.within_factor_name}={wlevel2}",
                    'Group1': f"{design.within_factor_name}={wlevel1}",
                    'Group1_mean': paired[wlevel1].mean(),
                    'Group1_stderr': paired[wlevel1].std(ddof=1) / np.sqrt(len(paired)) if len(paired) > 1 else 0.0,
                    'Group1_n': len(paired),
                    'Group2': f"{design.within_factor_name}={wlevel2}",
                    'Group2_mean': paired[wlevel2].mean(),
                    'Group2_stderr': paired[wlevel2].std(ddof=1) / np.sqrt(len(paired)) if len(paired) > 1 else 0.0,
                    'Group2_n': len(paired),
                    'p_value': t_result['p-val'].values[0]
                })
            except Exception as e:
                logger.debug(f"Marginal within comparison failed at AP {ap_idx}: {e}")
        return results
    
    def _run_marginal_between_ap(self, df_ap: pd.DataFrame, ap_idx: int,
                                 design: ExperimentalDesign) -> List[Dict]:
        """Marginal independent comparisons across between-factor levels."""
        results = []
        df_local = df_ap.copy()
        df_local['Subject_Key'] = df_local['Between_Factor'].astype(str) + "__" + df_local['Subject_ID'].astype(str)
        subject_means = (
            df_local.groupby(['Between_Factor', 'Subject_Key'])['Peak_Voltage']
            .mean()
            .reset_index()
        )
        between_levels = sorted(subject_means['Between_Factor'].unique())
        
        for blevel1, blevel2 in combinations(between_levels, 2):
            data1 = subject_means[subject_means['Between_Factor'] == blevel1]['Peak_Voltage']
            data2 = subject_means[subject_means['Between_Factor'] == blevel2]['Peak_Voltage']
            if len(data1) < MIN_CELLS_PER_UNIT or len(data2) < MIN_CELLS_PER_UNIT:
                continue
            try:
                t_result = pg.ttest(data1, data2, paired=False)
                results.append({
                    'ap_number': ap_idx,
                    'Test_Type': 'Marginal independent t-test',
                    'Comparison': f"{design.between_factor_name}={blevel1} vs {design.between_factor_name}={blevel2}",
                    'Group1': f"{design.between_factor_name}={blevel1}",
                    'Group1_mean': data1.mean(),
                    'Group1_stderr': data1.std(ddof=1) / np.sqrt(len(data1)) if len(data1) > 1 else 0.0,
                    'Group1_n': len(data1),
                    'Group2': f"{design.between_factor_name}={blevel2}",
                    'Group2_mean': data2.mean(),
                    'Group2_stderr': data2.std(ddof=1) / np.sqrt(len(data2)) if len(data2) > 1 else 0.0,
                    'Group2_n': len(data2),
                    'p_value': t_result['p-val'].values[0]
                })
            except Exception as e:
                logger.debug(f"Marginal between comparison failed at AP {ap_idx}: {e}")
        return results
    
    def _apply_posthoc_fdr(self, posthoc_results: List[Dict]) -> List[Dict]:
        """Apply FDR correction per AP (within each ANOVA family) to match independent design."""
        
        if not posthoc_results:
            return posthoc_results
        
        # Group post-hoc results by AP number (each ANOVA family)
        ap_groups = {}
        for result in posthoc_results:
            ap_num = result.get('ap_number')
            if ap_num not in ap_groups:
                ap_groups[ap_num] = []
            ap_groups[ap_num].append(result)
        
        # Apply FDR correction within each AP's post-hoc tests
        for ap_num, results_list in ap_groups.items():
            if len(results_list) > 1:
                # Extract p-values (filter out NaN)
                p_values = [r['p_value'] for r in results_list if _valid_p(r['p_value'])]
                valid_indices = [i for i, r in enumerate(results_list) if _valid_p(r['p_value'])]
                
                if len(p_values) > 1:
                    try:
                        _, corrected_p, _, _ = multi.multipletests(p_values, method="fdr_bh")
                        
                        # Update results with corrected p-values
                        for i, corrected_val in zip(valid_indices, corrected_p):
                            results_list[i]['corrected_p'] = corrected_val
                        
                        # Set NaN corrected_p for any NaN p-values
                        for i, result in enumerate(results_list):
                            if np.isnan(result['p_value']) and 'corrected_p' not in result:
                                result['corrected_p'] = np.nan
                        
                        logger.debug(f"Applied FDR correction to {len(p_values)} post-hoc tests at AP {ap_num}")
                        
                    except Exception as e:
                        logger.error(f"Error applying post-hoc FDR correction at AP {ap_num}: {e}")
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
        
        logger.info(f"Applied FDR correction per AP to {len(posthoc_results)} post-hoc tests across {len(ap_groups)} APs")
        
        return posthoc_results
    
    def _extract_factor_p(self, pvalues_dict: Dict, factor_name: str, 
                          exclude_interaction: bool = False) -> float:
        """Extract the minimum p-value for a factor's main effect."""
        
        relevant_p = []
        
        for param, p in pvalues_dict.items():
            if factor_name in param:
                if exclude_interaction and ':' in param:
                    continue
                relevant_p.append(p)
        
        if relevant_p:
            return min(relevant_p)
        return None
    
    def _extract_interaction_p(self, pvalues_dict: Dict) -> float:
        """Extract the minimum p-value for interaction terms."""
        
        interaction_p = []
        
        for param, p in pvalues_dict.items():
            if ':' in param:
                interaction_p.append(p)
        
        if interaction_p:
            return min(interaction_p)
        return None
    
    def _apply_fdr_to_results(self, results: List[Dict]) -> List[Dict]:
        """Apply FDR correction separately for each effect type."""
        
        effect_types = ['between_p', 'within_p', 'interaction_p']
        
        for effect_type in effect_types:
            p_values = [r[effect_type] for r in results if _valid_p(r.get(effect_type))]
            valid_indices = [i for i, r in enumerate(results) if _valid_p(r.get(effect_type))]
            
            if len(p_values) > 1:
                try:
                    rejected, corrected_p, _, _ = multi.multipletests(p_values, method="fdr_bh")
                    
                    corrected_key = effect_type.replace('_p', '_corrected_p')
                    for i, corrected_val in zip(valid_indices, corrected_p):
                        results[i][corrected_key] = corrected_val
                    
                    logger.info(f"Applied FDR to {len(p_values)} {effect_type} tests for attenuation")
                except Exception as e:
                    logger.error(f"Error applying FDR to {effect_type}: {e}")
            else:
                # Only 1 or 0 valid p-values - copy raw p-values for valid tests
                corrected_key = effect_type.replace('_p', '_corrected_p')
                for idx in valid_indices:
                    results[idx][corrected_key] = results[idx][effect_type]
                for i in range(len(results)):
                    if i not in valid_indices:
                        results[i][corrected_key] = np.nan
        
        return results
    
    def _save_attenuation_results(self, results: List[Dict], base_path: str,
                                     design: ExperimentalDesign = None) -> None:
        """Save point-by-point mixed model results to CSV (matching independent format)."""
        
        if not results:
            logger.warning("No attenuation results to save")
            return
        
        try:
            # Prepare data for CSV
            rows = []
            for result in results:
                ap_num = result['ap_number']
                
                # Base row data
                row = {
                    'AP_Number': ap_num,
                    'Test_Type': result.get('Test', 'Mixed ANOVA')
                }
                
                # Add effect p-values
                if design:
                    row[f'{design.between_factor_name}_p_value'] = result.get('between_p', np.nan)
                    row[f'{design.between_factor_name}_corrected_p'] = result.get('between_corrected_p', np.nan)
                    row[f'{design.within_factor_name}_p_value'] = result.get('within_p', np.nan)
                    row[f'{design.within_factor_name}_corrected_p'] = result.get('within_corrected_p', np.nan)
                else:
                    row['Between_p_value'] = result.get('between_p', np.nan)
                    row['Between_corrected_p'] = result.get('between_corrected_p', np.nan)
                    row['Within_p_value'] = result.get('within_p', np.nan)
                    row['Within_corrected_p'] = result.get('within_corrected_p', np.nan)
                
                row['Interaction_p_value'] = result.get('interaction_p', np.nan)
                row['Interaction_corrected_p'] = result.get('interaction_corrected_p', np.nan)
                
                # Add group statistics (mean, SEM, n for each group)
                for key, value in result.items():
                    if key.endswith('_mean') or key.endswith('_SEM') or key.endswith('_n'):
                        row[key] = value
                
                rows.append(row)
            
            # Create DataFrame
            df = pd.DataFrame(rows)
            
            # Reorder columns to ensure correct order (matching Stats_parameters structure)
            # Order: AP_Number -> Group stats -> ANOVA effects
            base_cols = ['AP_Number', 'Test_Type']
            
            # Extract group names and sort them, then add mean, SEM, n for each group in that order
            group_names_set = set()
            for col in df.columns:
                if col.endswith('_mean'):
                    group_names_set.add(col.replace('_mean', ''))
            
            group_stat_cols = []
            for group_name in sorted(group_names_set):
                for suffix in ['_mean', '_SEM', '_n']:
                    col_name = f'{group_name}{suffix}'
                    if col_name in df.columns:
                        group_stat_cols.append(col_name)
            
            if design:
                effect_cols = [
                    f'{design.between_factor_name}_p_value', f'{design.between_factor_name}_corrected_p',
                    f'{design.within_factor_name}_p_value', f'{design.within_factor_name}_corrected_p',
                    'Interaction_p_value', 'Interaction_corrected_p'
                ]
            else:
                effect_cols = [
                    'Between_p_value', 'Between_corrected_p',
                    'Within_p_value', 'Within_corrected_p',
                    'Interaction_p_value', 'Interaction_corrected_p'
                ]
            
            # Reorder columns
            column_order = base_cols + group_stat_cols + effect_cols
            available_cols = [col for col in column_order if col in df.columns]
            df = df[available_cols]
            
            # Save to CSV
            output_path = os.path.join(base_path, "Results", "Stats_AP_Num_vs_Peak_each_point_Mixed_ANOVA.csv")
            df.to_csv(output_path, index=False)
            logger.info(f"Saved point-by-point mixed model results to {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving point-by-point attenuation results: {e}")
    
    def _save_posthoc_results(self, posthoc_results: List[Dict], base_path: str) -> None:
        """Save post-hoc results to a separate CSV file (matching independent format)."""
        
        if not posthoc_results:
            return
        
        try:
            df = pd.DataFrame(posthoc_results)
            
            # Reorder columns to match independent design format
            base_cols = ['ap_number', 'Test_Type', 'Comparison']
            group_cols = ['Group1', 'Group1_mean', 'Group1_stderr', 'Group1_n',
                         'Group2', 'Group2_mean', 'Group2_stderr', 'Group2_n']
            stat_cols = ['p_value']
            if 'corrected_p' in df.columns:
                stat_cols.append('corrected_p')
            
            # Reorder using only columns that exist
            column_order = base_cols + group_cols + stat_cols
            available_cols = [col for col in column_order if col in df.columns]
            df = df[available_cols]
            
            output_path = os.path.join(base_path, "Results", "Stats_AP_Num_vs_Peak_each_point_pairwise.csv")
            df.to_csv(output_path, index=False)
            logger.info(f"Saved {len(posthoc_results)} post-hoc results to {output_path}")
        except Exception as e:
            logger.error(f"Error saving post-hoc results: {e}")
    
    def _run_global_mixed_effects(self, unified_df: pd.DataFrame, design: ExperimentalDesign):
        """Run global mixed-effects model across all APs.
        
        Formula: Peak ~ C(Between_Factor) * C(Within_Factor) * AP_Number_z (standardized)
        This tests for all main effects, 2-way, and 3-way interactions with standardized continuous variable.
        """
        
        try:
            if unified_df.empty or 'AP_num' not in unified_df.columns:
                logger.warning("Cannot run global model: insufficient data")
                return None
            
            # Drop any rows with NaN in critical columns
            unified_df = unified_df.dropna(subset=['Peak', 'AP_num', 'Subject_ID', 'Between_Factor', 'Within_Factor'])
            
            if len(unified_df) < 20:
                logger.warning(f"Not enough data for global model: {len(unified_df)} observations")
                return None
            
            # Standardize continuous variable (z-score) to help convergence
            unified_df['AP_num_z'] = (unified_df['AP_num'] - unified_df['AP_num'].mean()) / unified_df['AP_num'].std()
            
            logger.info(f"Running global mixed-effects model for attenuation with {len(unified_df)} observations across {unified_df['AP_num'].nunique()} AP positions...")
            
            # Full 3-way interaction model with standardized continuous variable
            formula = "Peak ~ C(Between_Factor) * C(Within_Factor) * AP_num_z"
            model = smf.mixedlm(
                formula, 
                unified_df, 
                groups=unified_df["Subject_ID"],
                re_formula="1 + AP_num_z"  # Random slopes for standardized continuous predictor
            ).fit(method='lbfgs', maxiter=5000)
            
            # Check convergence
            if not model.converged:
                logger.warning(f"Global attenuation model did not converge after 5000 iterations")
            else:
                logger.info(f"Global attenuation model converged successfully")
            
            # Extract p-values for all 7 effects using proper statistical tests (LRT for multiple contrasts)
            between_params = [p for p in model.pvalues.index if 'C(Between_Factor)' in p and ':' not in p]
            between_p = _get_effect_pvalue(model, between_params, unified_df, formula, 'Subject_ID', "1 + AP_num_z", "Between_Factor")
            
            within_params = [p for p in model.pvalues.index if 'C(Within_Factor)' in p and ':' not in p]
            within_p = _get_effect_pvalue(model, within_params, unified_df, formula, 'Subject_ID', "1 + AP_num_z", "Within_Factor")
            
            # Use standardized variable name
            if 'AP_num_z' not in model.pvalues.index:
                logger.warning(f"AP_num_z parameter not found in model. Available parameters: {list(model.pvalues.index)}")
            ap_p = model.pvalues.get('AP_num_z', 1.0)
            
            # Two-way interactions - use LRT for multiple contrasts
            between_within_params = [p for p in model.pvalues.index if 'C(Between_Factor)' in p and 'C(Within_Factor)' in p and 'AP_num_z' not in p]
            between_within_p = _get_effect_pvalue(model, between_within_params, unified_df, formula, 'Subject_ID', "1 + AP_num_z", "Between:Within")
            
            between_ap_params = [p for p in model.pvalues.index if 'C(Between_Factor)' in p and 'AP_num_z' in p and 'C(Within_Factor)' not in p]
            between_ap_p = _get_effect_pvalue(model, between_ap_params, unified_df, formula, 'Subject_ID', "1 + AP_num_z", "Between:AP_Number")
            
            within_ap_params = [p for p in model.pvalues.index if 'C(Within_Factor)' in p and 'AP_num_z' in p and 'C(Between_Factor)' not in p]
            within_ap_p = _get_effect_pvalue(model, within_ap_params, unified_df, formula, 'Subject_ID', "1 + AP_num_z", "Within:AP_Number")
            
            # Three-way interaction
            three_way_params = [p for p in model.pvalues.index if 'C(Between_Factor)' in p and 'C(Within_Factor)' in p and 'AP_num_z' in p]
            three_way_p = _get_effect_pvalue(model, three_way_params, unified_df, formula, 'Subject_ID', "1 + AP_num_z", "Between:Within:AP_Number")
            
            between_levels = sorted(unified_df['Between_Factor'].dropna().unique())
            within_levels = sorted(unified_df['Within_Factor'].dropna().unique())
            
            posthoc_rows = compute_global_curve_posthocs(
                model=model,
                step_col_z='AP_num_z',
                analysis_label='AP_Number_vs_Peak',
                factor_a_col='Between_Factor',
                factor_b_col='Within_Factor',
                factor_a_label=design.between_factor_name,
                factor_b_label=design.within_factor_name,
                factor_a_levels=between_levels,
                factor_b_levels=within_levels,
                main_a_p=between_p,
                main_b_p=within_p,
                interaction_ab_p=between_within_p,
                interaction_a_step_p=between_ap_p,
                interaction_b_step_p=within_ap_p,
                interaction_three_way_p=three_way_p
            )
            
            rows = [
                {'Effect': design.between_factor_name, 'p-value': between_p},
                {'Effect': design.within_factor_name, 'p-value': within_p},
                {'Effect': 'AP Number', 'p-value': ap_p},
                {'Effect': f'{design.between_factor_name}:{design.within_factor_name}', 'p-value': between_within_p},
                {'Effect': f'{design.between_factor_name}:AP Number', 'p-value': between_ap_p},
                {'Effect': f'{design.within_factor_name}:AP Number', 'p-value': within_ap_p},
                {'Effect': f'{design.between_factor_name}:{design.within_factor_name}:AP Number', 'p-value': three_way_p}
            ]
            
            logger.info("Global mixed-effects model complete for attenuation")
            return {
                'effects': rows,
                'posthocs': posthoc_rows
            }
            
        except Exception as e:
            logger.error(f"Error running global mixed-effects model for attenuation: {e}")
            return None

