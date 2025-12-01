"""
Frequency analysis for dependent (paired/repeated measures/mixed factorial) designs.
Handles current vs frequency and fold rheobase vs frequency using mixed effects models.
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

from ...shared.data_models import ExperimentalDesign, StatisticalResult
from .posthoc_utils import (
    should_run_posthocs,
    get_simple_effect_comparisons_dependent,
    compute_global_curve_posthocs,
)

logger = logging.getLogger(__name__)


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


class FrequencyAnalyzerDependent:
    """Frequency analyzer for dependent designs using mixed models."""
    
    def __init__(self, protocol_config=None):
        self.name = "Frequency Analyzer (Dependent)"
        self.protocol_config = protocol_config
        
    def analyze_current_vs_frequency(self, design: ExperimentalDesign, base_path: str) -> Dict:
        """Analyze frequency vs current for dependent designs (paired or mixed factorial)."""
        
        logger.info("Running dependent frequency analysis...")
        
        # Load frequency data for all groups
        all_group_data = {}
        for group in design.groups:
            freq_file = os.path.join(base_path, "Results", f"Calc_{group.name}_frequency_vs_current.csv")
            if os.path.exists(freq_file):
                try:
                    group_data = pd.read_csv(freq_file, index_col=False)
                    if not group_data.empty:
                        all_group_data[group.name] = group_data
                        logger.info(f"Loaded current vs frequency data for {group.name}")
                except Exception as e:
                    logger.warning(f"Error loading frequency data for {group.name}: {e}")
        
        if len(all_group_data) < 2:
            logger.warning("Need at least 2 groups with frequency data")
            return {'point_by_point_stats': [], 'mixed_effects_result': None, 'success': False}
        
        # Detect design type: paired (1 group, 2 conditions) vs repeated measures (1 group, 3+ conditions) vs mixed factorial (2+ groups)
        manifest = design.pairing_manifest
        between_levels = sorted(manifest['Group'].unique())
        within_levels = sorted(manifest['Condition'].unique())
        
        if len(between_levels) == 1:
            if len(within_levels) == 2:
                # Paired 2-group design
                logger.info("Detected paired two-group design for frequency analysis")
                return self._analyze_paired_frequency(design, base_path, all_group_data, within_levels, 'current')
            else:
                # Repeated measures (3+ conditions)
                logger.info(f"Detected repeated measures design ({len(within_levels)} conditions) for frequency analysis")
                return self._analyze_repeated_measures_frequency(design, base_path, all_group_data, within_levels, 'current')
        else:
            # Mixed factorial design
            logger.info(f"Detected mixed factorial design ({len(between_levels)}×{len(within_levels)}) for frequency analysis")
            return self._analyze_mixed_factorial_frequency(design, base_path, all_group_data, between_levels, within_levels, 'current')
    
    def analyze_fold_rheobase_vs_frequency(self, design: ExperimentalDesign, base_path: str) -> Dict:
        """Analyze fold rheobase vs frequency for dependent designs (paired or mixed factorial)."""
        
        logger.info("Running dependent fold rheobase analysis...")
        
        if not self.protocol_config:
            logger.warning("Protocol config required for fold rheobase conversion")
            return {'point_by_point_stats': [], 'mixed_effects_result': None, 'success': False}
        
        # Load frequency data for all groups
        all_group_data = {}
        
        for group in design.groups:
            freq_file = os.path.join(base_path, "Results", f"Calc_{group.name}_frequency_vs_current.csv")
            if os.path.exists(freq_file):
                try:
                    group_data = pd.read_csv(freq_file, index_col=False)
                    if not group_data.empty:
                        all_group_data[group.name] = group_data
                        logger.info(f"Loaded fold rheobase vs frequency data for {group.name}")
                except Exception as e:
                    logger.warning(f"Error loading fold rheobase data for {group.name}: {e}")
        
        if len(all_group_data) < 2:
            logger.warning("Need at least 2 groups with fold rheobase data")
            return {'point_by_point_stats': [], 'mixed_effects_result': None, 'success': False}
        
        # Detect design type: paired (1 group, 2 conditions) vs repeated measures (1 group, 3+ conditions) vs mixed factorial (2+ groups)
        manifest = design.pairing_manifest
        between_levels = sorted(manifest['Group'].unique())
        within_levels = sorted(manifest['Condition'].unique())
        
        if len(between_levels) == 1:
            if len(within_levels) == 2:
                # Paired 2-group design
                logger.info("Detected paired two-group design for fold rheobase analysis")
                return self._analyze_paired_frequency(design, base_path, all_group_data, within_levels, 'fold_rheobase')
            else:
                # Repeated measures (3+ conditions)
                logger.info(f"Detected repeated measures design ({len(within_levels)} conditions) for fold rheobase analysis")
                return self._analyze_repeated_measures_frequency(design, base_path, all_group_data, within_levels, 'fold_rheobase')
        else:
            # Mixed factorial design
            logger.info(f"Detected mixed factorial design ({len(between_levels)}×{len(within_levels)}) for fold rheobase analysis")
            return self._analyze_mixed_factorial_frequency(design, base_path, all_group_data, between_levels, within_levels, 'fold_rheobase')
    
    def _analyze_paired_frequency(self, design: ExperimentalDesign, base_path: str, 
                                   all_group_data: Dict[str, pd.DataFrame], 
                                   within_levels: List[str], analysis_type: str) -> Dict:
        """Analyze frequency for paired two-group design (simpler than mixed factorial)."""
        
        # Point-by-point paired t-tests for each current step
        anova_stats = self._run_pointwise_paired_ttest_frequency(all_group_data, analysis_type, within_levels)
        if anova_stats:
            # Save with "Paired" in filename to distinguish from "Mixed_ANOVA"
            x_name = "Current" if analysis_type == 'current' else "Fold_Rheobase"
            self._save_paired_point_by_point_results(anova_stats, base_path, x_name)
        
        # Unified mixed-effects model (simpler formula for paired design)
        mixed_effects_result = self._run_global_mixed_effects_paired(all_group_data, analysis_type, within_levels, design)
        
        logger.info(f"Paired frequency analysis completed for {len(all_group_data)} groups")
        
        return {
            'point_by_point_stats': anova_stats,
            'mixed_effects_result': mixed_effects_result,
            'success': True
        }
    
    def _analyze_mixed_factorial_frequency(self, design: ExperimentalDesign, base_path: str,
                                           all_group_data: Dict[str, pd.DataFrame],
                                           between_levels: List[str], within_levels: List[str],
                                           analysis_type: str) -> Dict:
        """Analyze frequency for mixed factorial design (existing logic)."""
        
        # Point-by-point mixed ANOVA for each current step
        anova_stats, posthoc_stats = self._run_pointwise_mixed_model_frequency(all_group_data, analysis_type, design)
        if anova_stats:
            x_name = "Current" if analysis_type == 'current' else "Fold_Rheobase"
            self._save_point_by_point_results(anova_stats, base_path, x_name, design)
        if posthoc_stats:
            analysis_name = 'Current_vs_frequency' if analysis_type == 'current' else 'Fold_rheobase_vs_frequency'
            self._save_posthoc_results(posthoc_stats, base_path, analysis_name)
        
        # Unified mixed-effects model
        mixed_effects_result = self._run_unified_mixed_effects_frequency(all_group_data, analysis_type, design)
        
        logger.info(f"Mixed factorial frequency analysis completed for {len(all_group_data)} groups")
        
        return {
            'point_by_point_stats': anova_stats,
            'mixed_effects_result': mixed_effects_result,
            'success': True
        }
    
    def _run_pointwise_paired_ttest_frequency(self, all_group_data: Dict[str, pd.DataFrame], 
                                               analysis_type: str, within_levels: List[str]) -> List[Dict]:
        """Run paired t-tests at each current step/fold rheobase for paired design."""
        
        results = []
        step_frames = {}
        cond1_name, cond2_name = within_levels[0], within_levels[1]
        
        # Get data for both conditions - require exact match
        if cond1_name not in all_group_data or cond2_name not in all_group_data:
            logger.error(f"Conditions not found in group data. Expected: {within_levels}, Available: {list(all_group_data.keys())}")
            return []
        
        cond1_data = all_group_data[cond1_name]
        cond2_data = all_group_data[cond2_name]
        
        if analysis_type == 'current':
            max_steps = min(len(cond1_data), len(cond2_data))
            logger.info(f"Analyzing frequency at {max_steps} current steps (paired t-tests)")
            
            for step_idx in range(max_steps):
                step_value = self.protocol_config.min_current + (step_idx * self.protocol_config.step_size)
                if step_value <= 0:
                    continue
                
                pairs = _pair_subject_value_columns(cond1_data)
                
                # Match subjects and collect data
                data1_list = []
                data2_list = []
                subjects = []
                
                for subject_col, value_col in pairs:
                    if value_col in cond1_data.columns and value_col in cond2_data.columns:
                        val1 = cond1_data[value_col].iloc[step_idx] if step_idx < len(cond1_data) else np.nan
                        val2 = cond2_data[value_col].iloc[step_idx] if step_idx < len(cond2_data) else np.nan
                        
                        if pd.notna(val1) and pd.notna(val2):
                            data1_list.append(val1)
                            data2_list.append(val2)
                            subjects.append(subject_col)
                
                if len(data1_list) < 3:
                    continue
                
                # Run paired t-test
                try:
                    data1_arr = np.array(data1_list)
                    data2_arr = np.array(data2_list)
                    
                    t_result = pg.ttest(data1_arr, data2_arr, paired=True)
                    p_value = t_result['p-val'].values[0]
                    
                    results.append({
                        'current_step': step_value,
                        f'{cond1_name}_mean': data1_arr.mean(),
                        f'{cond1_name}_SEM': data1_arr.std() / np.sqrt(len(data1_arr)),
                        f'{cond1_name}_n': len(data1_arr),
                        f'{cond2_name}_mean': data2_arr.mean(),
                        f'{cond2_name}_SEM': data2_arr.std() / np.sqrt(len(data2_arr)),
                        f'{cond2_name}_n': len(data2_arr),
                        'p_value': p_value
                    })
                    
                except Exception as e:
                    logger.warning(f"Error in paired t-test at current step {step_value}: {e}")
        
        else:  # fold_rheobase
            fold_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            logger.info(f"Analyzing frequency at up to {len(fold_values)} fold rheobase steps (paired t-tests)")
            
            for fold_value in fold_values:
                pairs = _pair_subject_value_columns(cond1_data)
                
                data1_list = []
                data2_list = []
                
                for subject_col, value_col in pairs:
                    if value_col in cond1_data.columns and value_col in cond2_data.columns:
                        # Find rheobase for each condition
                        frequencies1 = cond1_data[value_col].values
                        frequencies2 = cond2_data[value_col].values
                        
                        rheobase_idx1 = np.argmax(frequencies1 > 0) if np.any(frequencies1 > 0) else 0
                        rheobase_idx2 = np.argmax(frequencies2 > 0) if np.any(frequencies2 > 0) else 0
                        
                        target_idx1 = rheobase_idx1 + (fold_value - 1)
                        target_idx2 = rheobase_idx2 + (fold_value - 1)
                        
                        if target_idx1 < len(cond1_data) and target_idx2 < len(cond2_data):
                            val1 = cond1_data[value_col].iloc[target_idx1]
                            val2 = cond2_data[value_col].iloc[target_idx2]
                            
                            if pd.notna(val1) and pd.notna(val2):
                                data1_list.append(val1)
                                data2_list.append(val2)
                
                if len(data1_list) < 3:
                    continue
                
                # Run paired t-test
                try:
                    data1_arr = np.array(data1_list)
                    data2_arr = np.array(data2_list)
                    
                    t_result = pg.ttest(data1_arr, data2_arr, paired=True)
                    p_value = t_result['p-val'].values[0]
                    
                    results.append({
                        'fold_step': fold_value,
                        f'{cond1_name}_mean': data1_arr.mean(),
                        f'{cond1_name}_SEM': data1_arr.std() / np.sqrt(len(data1_arr)),
                        f'{cond1_name}_n': len(data1_arr),
                        f'{cond2_name}_mean': data2_arr.mean(),
                        f'{cond2_name}_SEM': data2_arr.std() / np.sqrt(len(data2_arr)),
                        f'{cond2_name}_n': len(data2_arr),
                        'p_value': p_value
                    })
                    
                except Exception as e:
                    logger.warning(f"Error in paired t-test at fold rheobase {fold_value}: {e}")
        
        # Apply FDR correction (handling NaN p-values)
        if len(results) > 1:
            # Extract valid p-values and track their indices
            valid_indices = [i for i, r in enumerate(results) if not np.isnan(r['p_value'])]
            valid_p_values = [results[i]['p_value'] for i in valid_indices]
            
            if len(valid_p_values) > 1:
                # Apply FDR correction to valid p-values
                try:
                    _, corrected_p, _, _ = multi.multipletests(valid_p_values, method="fdr_bh")
                    # Map corrected p-values back to valid indices
                    for idx, corrected_val in zip(valid_indices, corrected_p):
                        results[idx]['corrected_p'] = corrected_val
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
                    result['p_value'] if not np.isnan(result['p_value']) else np.nan
                )
        
        return results
    
    def _run_global_mixed_effects_paired(self, all_group_data: Dict[str, pd.DataFrame], 
                                         analysis_type: str, within_levels: List[str],
                                         design: ExperimentalDesign):
        """Run global mixed-effects model for paired design (simpler formula)."""
        
        all_data = []
        cond1_name, cond2_name = within_levels[0], within_levels[1]
        factor_name = design.within_factor_name or "Condition"  # Use custom name or default
        
        # Get data for both conditions - require exact match
        if cond1_name not in all_group_data or cond2_name not in all_group_data:
            logger.error(f"Conditions not found in group data. Expected: {within_levels}, Available: {list(all_group_data.keys())}")
            return None
        
        cond1_data = all_group_data[cond1_name]
        cond2_data = all_group_data[cond2_name]
        
        if analysis_type == 'current':
            max_steps = min(len(cond1_data), len(cond2_data))
            for step_idx in range(max_steps):
                current_value = self.protocol_config.min_current + (step_idx * self.protocol_config.step_size)
                if current_value <= 0:
                    continue
                
                pairs = _pair_subject_value_columns(cond1_data)
                for subject_col, value_col in pairs:
                    if value_col in cond1_data.columns and value_col in cond2_data.columns:
                        freq1 = cond1_data[value_col].iloc[step_idx] if step_idx < len(cond1_data) else np.nan
                        freq2 = cond2_data[value_col].iloc[step_idx] if step_idx < len(cond2_data) else np.nan
                        
                        if pd.notna(freq1):
                            all_data.append({
                                'Subject_ID': subject_col,
                                'Condition': cond1_name,
                                'Current': current_value,
                                'Frequency': freq1
                            })
                        if pd.notna(freq2):
                            all_data.append({
                                'Subject_ID': subject_col,
                                'Condition': cond2_name,
                                'Current': current_value,
                                'Frequency': freq2
                            })
        
        else:  # fold_rheobase - use ALL data for better convergence (not just integers 1-10)
            pairs = _pair_subject_value_columns(cond1_data)
            for subject_col, value_col in pairs:
                if value_col in cond1_data.columns and value_col in cond2_data.columns:
                    # Calculate for condition 1 - ALL fold rheobase values
                    frequencies1 = cond1_data[value_col].values
                    rheobase_idx1 = np.argmax(frequencies1 > 0) if np.any(frequencies1 > 0) else 0
                    rheobase_current1 = (rheobase_idx1 * self.protocol_config.step_size) + self.protocol_config.min_current
                    
                    # Loop through ALL available steps from rheobase to end
                    for target_idx1 in range(rheobase_idx1, len(cond1_data)):
                        freq1 = cond1_data[value_col].iloc[target_idx1]
                        if pd.notna(freq1):
                            # Calculate actual fold rheobase as current ratio (includes fractional values)
                            current_at_step1 = (target_idx1 * self.protocol_config.step_size) + self.protocol_config.min_current
                            actual_fold1 = current_at_step1 / rheobase_current1 if rheobase_current1 != 0 else 1.0
                            all_data.append({
                                'Subject_ID': subject_col,
                                'Condition': cond1_name,
                                'FoldRheobase': actual_fold1,
                                'Frequency': freq1
                            })
                    
                    # Calculate for condition 2 - ALL fold rheobase values
                    frequencies2 = cond2_data[value_col].values
                    rheobase_idx2 = np.argmax(frequencies2 > 0) if np.any(frequencies2 > 0) else 0
                    rheobase_current2 = (rheobase_idx2 * self.protocol_config.step_size) + self.protocol_config.min_current
                    
                    # Loop through ALL available steps from rheobase to end
                    for target_idx2 in range(rheobase_idx2, len(cond2_data)):
                        freq2 = cond2_data[value_col].iloc[target_idx2]
                        if pd.notna(freq2):
                            # Calculate actual fold rheobase as current ratio (includes fractional values)
                            current_at_step2 = (target_idx2 * self.protocol_config.step_size) + self.protocol_config.min_current
                            actual_fold2 = current_at_step2 / rheobase_current2 if rheobase_current2 != 0 else 1.0
                            all_data.append({
                                'Subject_ID': subject_col,
                                'Condition': cond2_name,
                                'FoldRheobase': actual_fold2,
                                'Frequency': freq2
                            })
        
        if not all_data:
            logger.warning("No data for global mixed-effects model (paired)")
            return None
        
        unified_df = pd.DataFrame(all_data)
        
        # Log data points collected (helpful for debugging)
        if analysis_type == 'fold_rheobase':
            n_subjects = unified_df['Subject_ID'].nunique()
            n_obs = len(unified_df)
            logger.info(f"Collected {n_obs} observations for paired global mixed model (fold rheobase) from {n_subjects} subjects")
        
        # Determine predictor column
        step_col = 'Current' if analysis_type == 'current' else 'FoldRheobase'
        
        try:
            # Drop NaN
            unified_df = unified_df.dropna(subset=['Frequency', step_col, 'Subject_ID', 'Condition'])
            
            # Standardize continuous variable
            step_col_z = f"{step_col}_z"
            unified_df[step_col_z] = (unified_df[step_col] - unified_df[step_col].mean()) / unified_df[step_col].std()
            
            logger.info(f"Running global mixed-effects model for paired design ({analysis_type})")
            
            # Simpler formula for paired: no between factor
            formula = f"Frequency ~ C(Condition) * {step_col_z}"
            model = smf.mixedlm(
                formula,
                unified_df,
                groups=unified_df["Subject_ID"],
                re_formula=f"1 + {step_col_z}"
            ).fit(method='lbfgs', maxiter=5000)
            
            if not model.converged:
                logger.warning(f"Paired global model for {analysis_type} did not converge")
            else:
                logger.info(f"Paired global model for {analysis_type} converged successfully")
            
            # Extract p-values (simpler: only Condition, Step, and Condition:Step)
            # For paired designs, there's only 1 contrast for Condition - extract directly
            condition_params = [p for p in model.pvalues.index if 'C(Condition)' in p and ':' not in p]
            condition_p = model.pvalues[condition_params[0]] if len(condition_params) == 1 else 1.0
            
            step_p = model.pvalues.get(step_col_z, 1.0)
            
            interaction_params = [p for p in model.pvalues.index if 'C(Condition)' in p and step_col_z in p]
            interaction_p = model.pvalues[interaction_params[0]] if len(interaction_params) == 1 else 1.0
            
            x_var_label = 'Current' if analysis_type == 'current' else 'Fold Rheobase'
            
            rows = [
                {'Effect': factor_name, 'p-value': condition_p},
                {'Effect': x_var_label, 'p-value': step_p},
                {'Effect': f'{factor_name}:{x_var_label}', 'p-value': interaction_p}
            ]
            
            logger.info(f"Paired global mixed-effects model complete for {analysis_type}")
            return rows
            
        except Exception as e:
            logger.error(f"Error running paired global mixed-effects model for {analysis_type}: {e}")
            return None
    
    def _save_paired_point_by_point_results(self, results: List[Dict], base_path: str, x_name: str) -> None:
        """Save point-by-point paired t-test results to CSV."""
        
        if not results:
            logger.warning("No paired results to save")
            return
        
        try:
            df = pd.DataFrame(results)
            
            # Reorder columns: Step_Value -> Group stats -> p-values
            step_key = 'current_step' if 'current_step' in df.columns else 'fold_step'
            base_cols = [step_key]
            
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
            
            # Rename step_key to 'Step_Value'
            df = df.rename(columns={step_key: 'Step_Value'})
            
            column_order = ['Step_Value'] + group_stat_cols + p_value_cols
            available_cols = [col for col in column_order if col in df.columns]
            df = df[available_cols]
            
            output_path = os.path.join(base_path, "Results", f"Stats_{x_name}_vs_frequency_each_point_Paired.csv")
            df.to_csv(output_path, index=False)
            logger.info(f"Saved paired point-by-point results to {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving paired point-by-point results: {e}")
    
    def _analyze_repeated_measures_frequency(self, design: ExperimentalDesign, base_path: str,
                                            all_group_data: Dict[str, pd.DataFrame],
                                            within_levels: List[str], analysis_type: str) -> Dict:
        """Analyze frequency for repeated measures design (3+ conditions, single group)."""
        
        # Point-by-point RM-ANOVA for each current step
        anova_stats = self._run_pointwise_rm_anova_frequency(all_group_data, analysis_type, within_levels, design)
        if anova_stats:
            x_name = "Current" if analysis_type == 'current' else "Fold_Rheobase"
            self._save_rm_anova_frequency_results(anova_stats, base_path, x_name)
        
        # Global mixed-effects model
        mixed_effects_result = self._run_global_mixed_effects_rm_frequency(all_group_data, analysis_type, within_levels, design)
        
        logger.info(f"Repeated measures frequency analysis completed for {len(all_group_data)} groups")
        
        return {
            'point_by_point_stats': anova_stats,
            'mixed_effects_result': mixed_effects_result,
            'success': True
        }
    
    def _run_pointwise_rm_anova_frequency(self, all_group_data, analysis_type, within_levels, design):
        """Run RM-ANOVA at each current step/fold rheobase for repeated measures design."""
        
        results = []
        step_frames = {}
        
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
        
        if analysis_type == 'current':
            max_steps = max(len(df) for df in condition_data.values())
            logger.info(f"Analyzing frequency at {max_steps} current steps (RM-ANOVA)")
            
            for step_idx in range(max_steps):
                step_value = self.protocol_config.min_current + (step_idx * self.protocol_config.step_size)
                if step_value <= 0:
                    continue
                
                # Build long-format DataFrame for this step
                long_data = []
                
                for condition in within_levels:
                    if condition not in condition_data:
                        continue
                    
                    cond_df = condition_data[condition]
                    
                    if step_idx >= len(cond_df):
                        continue
                    
                    # Extract data for this step across all subjects
                    for subject_col, value_col in _pair_subject_value_columns(cond_df):
                        if value_col in cond_df.columns:
                            freq = cond_df[value_col].iloc[step_idx]
                            
                            if pd.notna(freq):
                                long_data.append({
                                    'Subject_ID': subject_col,
                                    'Condition': condition,
                                    'Frequency': freq
                                })
                
                if len(long_data) < len(within_levels) * 3:  # Need at least 3 subjects with complete data
                    continue
                
                df_long = pd.DataFrame(long_data)
                
                # Run RM-ANOVA using pingouin
                try:
                    aov = pg.rm_anova(
                        dv='Frequency',
                        within='Condition',
                        subject='Subject_ID',
                        data=df_long
                    )
                    
                    p_value = aov['p-unc'].values[0]
                    
                    # Calculate group statistics
                    result = {
                        'current_step': step_value,
                        'rm_anova_p': p_value
                    }
                    
                    # Add statistics for each condition
                    for condition in within_levels:
                        cond_data = df_long[df_long['Condition'] == condition]['Frequency']
                        result[f'{condition}_mean'] = cond_data.mean()
                        result[f'{condition}_SEM'] = cond_data.std() / np.sqrt(len(cond_data)) if len(cond_data) > 1 else 0
                        result[f'{condition}_n'] = len(cond_data)
                    
                    results.append(result)
                    step_frames[('current', step_value)] = df_long.copy()
                    
                except Exception as e:
                    logger.warning(f"Error in RM-ANOVA at step {step_value}: {e}")
        
        else:  # fold_rheobase
            # Get all unique fold values across conditions
            all_fold_values = set()
            for cond_df in condition_data.values():
                if not cond_df.empty:
                    all_fold_values.update(cond_df.index.tolist())
            
            fold_values = sorted(all_fold_values)
            logger.info(f"Analyzing frequency at {len(fold_values)} fold rheobase values (RM-ANOVA)")
            
            for fold_value in fold_values:
                # Build long-format DataFrame for this fold value
                long_data = []
                
                for condition in within_levels:
                    if condition not in condition_data:
                        continue
                    
                    cond_df = condition_data[condition]
                    
                    if fold_value not in cond_df.index:
                        continue
                    
                    # Extract data for this fold value across all subjects
                    for subject_col, value_col in _pair_subject_value_columns(cond_df):
                        if value_col in cond_df.columns:
                            freq = cond_df.loc[fold_value, value_col]
                            
                            if pd.notna(freq):
                                long_data.append({
                                    'Subject_ID': subject_col,
                                    'Condition': condition,
                                    'Frequency': freq
                                })
                
                if len(long_data) < len(within_levels) * 3:
                    continue
                
                df_long = pd.DataFrame(long_data)
                
                # Run RM-ANOVA using pingouin
                try:
                    aov = pg.rm_anova(
                        dv='Frequency',
                        within='Condition',
                        subject='Subject_ID',
                        data=df_long
                    )
                    
                    p_value = aov['p-unc'].values[0]
                    
                    # Calculate group statistics
                    result = {
                        'fold_step': fold_value,
                        'rm_anova_p': p_value
                    }
                    
                    # Add statistics for each condition
                    for condition in within_levels:
                        cond_data = df_long[df_long['Condition'] == condition]['Frequency']
                        result[f'{condition}_mean'] = cond_data.mean()
                        result[f'{condition}_SEM'] = cond_data.std() / np.sqrt(len(cond_data)) if len(cond_data) > 1 else 0
                        result[f'{condition}_n'] = len(cond_data)
                    
                    results.append(result)
                    step_frames[('fold', fold_value)] = df_long.copy()
                    
                except Exception as e:
                    logger.warning(f"Error in RM-ANOVA at fold {fold_value}: {e}")
        
        # Apply FDR correction to RM-ANOVA p-values
        if len(results) > 1:
            p_values = [r['rm_anova_p'] for r in results]
            try:
                _, corrected_p, _, _ = multi.multipletests(p_values, method="fdr_bh")
                for i, result in enumerate(results):
                    result['rm_anova_corrected_p'] = corrected_p[i]
                logger.info(f"Applied FDR correction to {len(p_values)} RM-ANOVA tests")
            except Exception as e:
                logger.error(f"Error applying FDR correction: {e}")
                for result in results:
                    result['rm_anova_corrected_p'] = result['rm_anova_p']
        
        # Run post-hoc tests only for steps that remain significant after correction
        for result in results:
            corrected = result.get('rm_anova_corrected_p', result.get('rm_anova_p'))
            if corrected is None or np.isnan(corrected) or corrected >= 0.05:
                continue
            step_key = 'current_step' if 'current_step' in result else 'fold_step'
            step_value = result.get(step_key)
            frame_key = ('current', step_value) if step_key == 'current_step' else ('fold', step_value)
            df_long = step_frames.get(frame_key)
            if df_long is None:
                continue
            logger.info(f"RM-ANOVA significant at {step_key} {step_value} (corrected p = {corrected:.4f}) - running post-hoc tests")
            for cond1, cond2 in combinations(within_levels, 2):
                try:
                    paired_data1 = []
                    paired_data2 = []
                    for subject in df_long['Subject_ID'].unique():
                        subj_data = df_long[df_long['Subject_ID'] == subject]
                        val1 = subj_data[subj_data['Condition'] == cond1]['Frequency']
                        val2 = subj_data[subj_data['Condition'] == cond2]['Frequency']
                        if len(val1) > 0 and len(val2) > 0:
                            paired_data1.append(val1.iloc[0])
                            paired_data2.append(val2.iloc[0])
                    if len(paired_data1) >= 3:
                        t_result = pg.ttest(np.array(paired_data1), np.array(paired_data2), paired=True)
                        posthoc_p = t_result['p-val'].values[0]
                        result[f'{cond1}_vs_{cond2}_posthoc_p'] = posthoc_p
                except Exception as e:
                    logger.warning(f"Error in post-hoc test at {step_key} {step_value} ({cond1} vs {cond2}): {e}")
        
        # Apply FDR correction to post-hoc p-values (within each comparison)
        if results:
            # Get all unique pairwise comparisons
            posthoc_keys = set()
            for result in results:
                for key in result.keys():
                    if '_posthoc_p' in key:
                        posthoc_keys.add(key)
            
            # For each pairwise comparison, collect and correct p-values across steps
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
    
    def _run_global_mixed_effects_rm_frequency(self, all_group_data, analysis_type, within_levels, design):
        """Run global mixed-effects model for repeated measures frequency design."""
        
        all_data = []
        factor_name = design.within_factor_name or "Condition"
        
        # Determine step column
        step_col = "Current_step" if analysis_type == 'current' else "Fold_rheobase"
        
        for condition in within_levels:
            # Require exact match between condition name and folder name
            if condition not in all_group_data:
                logger.warning(f"Condition '{condition}' not found in group data. Available: {list(all_group_data.keys())}")
                continue
            
            cond_data = all_group_data[condition]
            
            if analysis_type == 'current':
                for step_idx in range(len(cond_data)):
                    step_value = self.protocol_config.min_current + (step_idx * self.protocol_config.step_size)
                    if step_value <= 0:
                        continue
                    
                    for subject_col, value_col in _pair_subject_value_columns(cond_data):
                        if value_col in cond_data.columns:
                            freq = cond_data[value_col].iloc[step_idx]
                            
                            if pd.notna(freq):
                                all_data.append({
                                    'Subject_ID': subject_col,
                                    'Condition': condition,
                                    step_col: step_value,
                                    'Frequency': freq
                                })
            else:  # fold_rheobase - use ALL data for better convergence (not just integers 1-10)
                for subject_col, value_col in _pair_subject_value_columns(cond_data):
                    if value_col in cond_data.columns:
                        # Calculate rheobase for this subject
                        frequencies = cond_data[value_col].values
                        rheobase_idx = np.argmax(frequencies > 0) if np.any(frequencies > 0) else 0
                        rheobase_current = (rheobase_idx * self.protocol_config.step_size) + self.protocol_config.min_current
                        
                        # Loop through ALL available steps from rheobase to end
                        for target_idx in range(rheobase_idx, len(cond_data)):
                            freq = cond_data[value_col].iloc[target_idx]
                            
                            if pd.notna(freq):
                                # Calculate actual fold rheobase as current ratio (includes fractional values)
                                current_at_step = (target_idx * self.protocol_config.step_size) + self.protocol_config.min_current
                                actual_fold = current_at_step / rheobase_current if rheobase_current != 0 else 1.0
                                all_data.append({
                                    'Subject_ID': subject_col,
                                    'Condition': condition,
                                    step_col: actual_fold,
                                    'Frequency': freq
                                })
        
        if not all_data:
            logger.warning(f"No data for repeated measures frequency global mixed-effects model ({analysis_type})")
            return None
        
        unified_df = pd.DataFrame(all_data)
        
        # Log data points collected (helpful for debugging)
        if analysis_type == 'fold_rheobase':
            n_subjects = unified_df['Subject_ID'].nunique()
            n_obs = len(unified_df)
            n_conditions = len(within_levels)
            logger.info(f"Collected {n_obs} observations for repeated measures global mixed model (fold rheobase) from {n_subjects} subjects across {n_conditions} conditions")
        
        try:
            # Drop NaN
            unified_df = unified_df.dropna(subset=['Frequency', step_col, 'Subject_ID', 'Condition'])
            
            # Standardize continuous variable
            step_col_z = f"{step_col}_z"
            unified_df[step_col_z] = (unified_df[step_col] - unified_df[step_col].mean()) / unified_df[step_col].std()
            
            logger.info(f"Running global mixed-effects model for repeated measures frequency design ({analysis_type}, {len(within_levels)} conditions)")
            
            # Formula: Frequency ~ C(Condition) * Step_z
            formula = f"Frequency ~ C(Condition) * {step_col_z}"
            model = smf.mixedlm(
                formula,
                unified_df,
                groups=unified_df["Subject_ID"],
                re_formula=f"1 + {step_col_z}"
            ).fit(method='lbfgs', maxiter=5000)
            
            if not model.converged:
                logger.warning(f"Repeated measures global model for {analysis_type} did not converge")
            else:
                logger.info(f"Repeated measures global model for {analysis_type} converged successfully")
            
            # Extract p-values using proper statistical tests (LRT for multiple contrasts)
            condition_params = [p for p in model.pvalues.index if 'C(Condition)' in p and ':' not in p]
            condition_p = _get_effect_pvalue(model, condition_params, unified_df, formula, 'Subject_ID', f"1 + {step_col_z}", "Condition")
            
            step_p = model.pvalues.get(step_col_z, 1.0)
            
            step_name = "Current" if analysis_type == 'current' else "Fold Rheobase"
            
            interaction_params = [p for p in model.pvalues.index if 'C(Condition)' in p and step_col_z in p]
            interaction_p = _get_effect_pvalue(model, interaction_params, unified_df, formula, 'Subject_ID', f"1 + {step_col_z}", f"Condition:{step_name}")
            
            rows = [
                {'Effect': factor_name, 'p-value': condition_p},
                {'Effect': step_name, 'p-value': step_p},
                {'Effect': f'{factor_name}:{step_name}', 'p-value': interaction_p}
            ]
            
            condition_levels = sorted(unified_df['Condition'].dropna().unique())
            analysis_label = 'Current_vs_Frequency' if analysis_type == 'current' else 'Fold_Rheobase_vs_Frequency'
            posthoc_rows = compute_global_curve_posthocs(
                model=model,
                step_col_z=step_col_z,
                analysis_label=analysis_label,
                factor_a_col='Condition',
                factor_b_col=None,
                factor_a_label=factor_name,
                factor_b_label=step_name,
                factor_a_levels=condition_levels,
                factor_b_levels=[],
                main_a_p=condition_p,
                main_b_p=None,
                interaction_ab_p=None,
                interaction_a_step_p=interaction_p,
                interaction_b_step_p=None,
                interaction_three_way_p=None
            )
            
            logger.info(f"Repeated measures frequency global mixed-effects model complete ({analysis_type})")
            return {
                'effects': rows,
                'posthocs': posthoc_rows
            }
            
        except Exception as e:
            logger.error(f"Error running repeated measures frequency global mixed-effects model ({analysis_type}): {e}")
            return None
    
    def _save_rm_anova_frequency_results(self, results, base_path, x_name):
        """Save point-by-point RM-ANOVA frequency results to CSV - split into main test and post-hocs."""
        
        if not results:
            logger.warning("No RM-ANOVA frequency results to save")
            return
        
        try:
            df = pd.DataFrame(results)
            
            # Reorder columns: Step_Value -> Condition stats -> RM-ANOVA p-values -> Post-hoc p-values
            step_key = 'current_step' if 'current_step' in df.columns else 'fold_step'
            
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
            
            # Rename step column
            step_col_name = f"{x_name}_step" if x_name == "Current" else x_name
            df = df.rename(columns={step_key: step_col_name})
            
            # Save main RM-ANOVA results file (without post-hocs)
            anova_column_order = [step_col_name] + condition_stat_cols + anova_cols
            available_anova_cols = [col for col in anova_column_order if col in df.columns]
            df_anova = df[available_anova_cols].copy()
            
            anova_filename = f"Stats_{x_name}_vs_frequency_each_point_RM_ANOVA.csv"
            anova_path = os.path.join(base_path, "Results", anova_filename)
            df_anova.to_csv(anova_path, index=False)
            logger.info(f"Saved RM-ANOVA frequency results to {anova_path}")
            
            # Save post-hoc results file (only if there are post-hocs)
            if posthoc_cols:
                posthoc_column_order = [step_col_name] + condition_stat_cols + posthoc_cols
                available_posthoc_cols = [col for col in posthoc_column_order if col in df.columns]
                df_posthoc = df[available_posthoc_cols].copy()
                
                posthoc_filename = f"Stats_{x_name}_vs_frequency_each_point_pairwise.csv"
                posthoc_path = os.path.join(base_path, "Results", posthoc_filename)
                df_posthoc.to_csv(posthoc_path, index=False)
                logger.info(f"Saved RM-ANOVA frequency post-hoc results to {posthoc_path}")
            else:
                logger.info("No post-hoc comparisons to save for RM-ANOVA frequency")
            
        except Exception as e:
            logger.error(f"Error saving RM-ANOVA frequency results: {e}")
    
    def _run_pointwise_mixed_model_frequency(self, all_group_data: Dict[str, pd.DataFrame], analysis_type: str, design: ExperimentalDesign) -> Tuple[List[Dict], List[Dict]]:
        """Run point-by-point mixed ANOVA for each current/fold rheobase step with post-hocs.
        
        Returns:
            Tuple of (anova_results, posthoc_results)
        """
        manifest = design.pairing_manifest
        anova_results = []
        posthoc_results = []
        
        if analysis_type == 'current':
            # Get max steps
            max_steps = max(len(df) for df in all_group_data.values())
            logger.info(f"Analyzing frequency at {max_steps} current steps")
            
            for step_idx in range(max_steps):
                step_value = self.protocol_config.min_current + (step_idx * self.protocol_config.step_size)
                if step_value <= 0:
                    continue
                
                # Collect data for this step
                step_data_list = []
                for group_name, group_data in all_group_data.items():
                    if step_idx < len(group_data):
                        between_level, within_level = self._parse_group_name(group_name, manifest)
                        subject_cols = [col for col in group_data.columns if col.startswith('Subject_')]
                        value_cols = [col for col in group_data.columns if col.startswith('Values_')]
                        
                        for subject_col, value_col in zip(subject_cols, value_cols):
                            subject_id = subject_col
                            frequency = group_data[value_col].iloc[step_idx]
                            if pd.notna(frequency):
                                step_data_list.append({
                                    'Subject_ID': str(subject_id),
                                    'Between_Factor': between_level,
                                    'Within_Factor': within_level,
                                    'Frequency': frequency
                                })
                
                if len(step_data_list) < 4:
                    continue
                
                df_step = pd.DataFrame(step_data_list)
                
                # Run mixed ANOVA
                try:
                    aov = pg.mixed_anova(
                        dv='Frequency',
                        within='Within_Factor',
                        between='Between_Factor',
                        subject='Subject_ID',
                        data=df_step
                    )
                    
                    # Safely extract p-values, handling missing columns
                    if 'p-unc' not in aov.columns:
                        logger.warning(f"Skipping current step {step_value}: ANOVA table missing p-unc column (likely unbalanced data)")
                        continue
                    
                    between_p = aov[aov['Source'] == 'Between_Factor']['p-unc'].values[0] if 'Between_Factor' in aov['Source'].values else np.nan
                    within_p = aov[aov['Source'] == 'Within_Factor']['p-unc'].values[0] if 'Within_Factor' in aov['Source'].values else np.nan
                    interaction_p = aov[aov['Source'] == 'Interaction']['p-unc'].values[0] if 'Interaction' in aov['Source'].values else np.nan
                    
                    # Calculate group statistics
                    result = {
                        'current_step': step_value,
                        'between_p': between_p,
                        'within_p': within_p,
                        'interaction_p': interaction_p,
                        '_df_step': df_step  # Store for later post-hoc analysis
                    }
                    
                    # Add group statistics (mean, SEM, n for each Between x Within combination)
                    for (between_level, within_level), group_df in df_step.groupby(['Between_Factor', 'Within_Factor']):
                        group_name = f"{between_level}: {within_level}"
                        result[f'{group_name}_mean'] = group_df['Frequency'].mean()
                        result[f'{group_name}_SEM'] = group_df['Frequency'].std() / np.sqrt(len(group_df))
                        result[f'{group_name}_n'] = len(group_df)
                    
                    anova_results.append(result)
                    
                except Exception as e:
                    logger.warning(f"Error at current step {step_value}: {e}")
        
        else:  # fold_rheobase
            fold_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            logger.info(f"Analyzing frequency at up to {len(fold_values)} fold rheobase steps")
            
            for fold_value in fold_values:
                step_data_list = []
                for group_name, group_data in all_group_data.items():
                    between_level, within_level = self._parse_group_name(group_name, manifest)
                    subject_cols = [col for col in group_data.columns if col.startswith('Subject_')]
                    value_cols = [col for col in group_data.columns if col.startswith('Values_')]
                    
                    for subject_col, value_col in zip(subject_cols, value_cols):
                        frequencies = group_data[value_col].values
                        rheobase_idx = np.argmax(frequencies > 0) if np.any(frequencies > 0) else 0
                        target_idx = rheobase_idx + (fold_value - 1)
                        
                        if target_idx < len(group_data):
                            subject_id = subject_col
                            frequency = group_data[value_col].iloc[target_idx]
                            if pd.notna(frequency):
                                step_data_list.append({
                                    'Subject_ID': str(subject_id),
                                    'Between_Factor': between_level,
                                    'Within_Factor': within_level,
                                    'Frequency': frequency
                                })
                
                if len(step_data_list) < 4:
                    continue
                
                df_step = pd.DataFrame(step_data_list)
                
                # Run mixed ANOVA
                try:
                    aov = pg.mixed_anova(
                        dv='Frequency',
                        within='Within_Factor',
                        between='Between_Factor',
                        subject='Subject_ID',
                        data=df_step
                    )
                    
                    # Safely extract p-values, handling missing columns
                    if 'p-unc' not in aov.columns:
                        logger.warning(f"Skipping fold rheobase {fold_value}: ANOVA table missing p-unc column (likely unbalanced data)")
                        continue
                    
                    between_p = aov[aov['Source'] == 'Between_Factor']['p-unc'].values[0] if 'Between_Factor' in aov['Source'].values else np.nan
                    within_p = aov[aov['Source'] == 'Within_Factor']['p-unc'].values[0] if 'Within_Factor' in aov['Source'].values else np.nan
                    interaction_p = aov[aov['Source'] == 'Interaction']['p-unc'].values[0] if 'Interaction' in aov['Source'].values else np.nan
                    
                    # Calculate group statistics
                    result = {
                        'fold_step': fold_value,
                        'between_p': between_p,
                        'within_p': within_p,
                        'interaction_p': interaction_p,
                        '_df_step': df_step  # Store for later post-hoc analysis
                    }
                    
                    # Add group statistics (mean, SEM, n for each Between x Within combination)
                    for (between_level, within_level), group_df in df_step.groupby(['Between_Factor', 'Within_Factor']):
                        group_name = f"{between_level}: {within_level}"
                        result[f'{group_name}_mean'] = group_df['Frequency'].mean()
                        result[f'{group_name}_SEM'] = group_df['Frequency'].std() / np.sqrt(len(group_df))
                        result[f'{group_name}_n'] = len(group_df)
                    
                    anova_results.append(result)
                    
                except Exception as e:
                    logger.warning(f"Error at fold rheobase step {fold_value}: {e}")
        
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
                df_step = result.pop('_df_step', None)  # Remove and retrieve stored data
                if df_step is not None:
                    step_value = result.get('current_step') or result.get('fold_step')
                    logger.info(f"Step {step_value}: Running post-hocs for {posthoc_decision['reasons']}")
                    step_posthocs = self._run_posthocs_at_step(
                        df_step, step_value, analysis_type, design,
                        between_p_corr, within_p_corr, interaction_p_corr,
                        posthoc_decision.get('reasons', [])
                    )
                    posthoc_results.extend(step_posthocs)
            else:
                # Remove stored data even if not running post-hocs
                result.pop('_df_step', None)
        
        # Apply FDR correction to post-hoc results (separately for paired and independent)
        posthoc_results = self._apply_posthoc_fdr(posthoc_results)
        
        logger.info(f"{'Frequency' if analysis_type == 'current' else 'Fold rheobase'} analysis complete: {len(anova_results)} steps, {len(posthoc_results)} post-hocs")
        return anova_results, posthoc_results
    
    def _run_unified_mixed_effects_frequency(self, all_group_data: Dict[str, pd.DataFrame], analysis_type: str, design: ExperimentalDesign):
        """Run unified mixed-effects model across all steps."""
        manifest = design.pairing_manifest
        all_data = []
        
        if analysis_type == 'current':
            max_steps = max(len(df) for df in all_group_data.values())
            for step_idx in range(max_steps):
                current_value = self.protocol_config.min_current + (step_idx * self.protocol_config.step_size)
                # Skip negative currents but keep 0 (needed for model convergence)
                if current_value <= 0:
                    continue
                for group_name, group_data in all_group_data.items():
                    if step_idx < len(group_data):
                        between_level, within_level = self._parse_group_name(group_name, manifest)
                        subject_cols = [col for col in group_data.columns if col.startswith('Subject_')]
                        value_cols = [col for col in group_data.columns if col.startswith('Values_')]
                        for subject_col, value_col in zip(subject_cols, value_cols):
                            # Subject_ID is the column name itself (e.g., "Subject_Scn1a_1")
                            subject_id = subject_col
                            frequency = group_data[value_col].iloc[step_idx]
                            if pd.notna(frequency):
                                all_data.append({
                                    'Subject_ID': str(subject_id),
                                    'Between_Factor': between_level,
                                    'Within_Factor': within_level,
                                    'Current': current_value,
                                    'Frequency': frequency
                                })
        else:  # fold_rheobase - use ALL data for better convergence (not just integers 1-10)
            for group_name, group_data in all_group_data.items():
                between_level, within_level = self._parse_group_name(group_name, manifest)
                subject_cols = [col for col in group_data.columns if col.startswith('Subject_')]
                value_cols = [col for col in group_data.columns if col.startswith('Values_')]
                for subject_col, value_col in zip(subject_cols, value_cols):
                    frequencies = group_data[value_col].values
                    rheobase_idx = np.argmax(frequencies > 0) if np.any(frequencies > 0) else 0
                    rheobase_current = (rheobase_idx * self.protocol_config.step_size) + self.protocol_config.min_current
                    
                    # Loop through ALL available steps from rheobase to end
                    for target_idx in range(rheobase_idx, len(group_data)):
                        subject_id = subject_col
                        frequency = group_data[value_col].iloc[target_idx]
                        if pd.notna(frequency):
                            # Calculate actual fold rheobase as current ratio (includes fractional values)
                            current_at_step = (target_idx * self.protocol_config.step_size) + self.protocol_config.min_current
                            actual_fold = current_at_step / rheobase_current if rheobase_current != 0 else 1.0
                            all_data.append({
                                'Subject_ID': str(subject_id),
                                'Between_Factor': between_level,
                                'Within_Factor': within_level,
                                'FoldRheobase': actual_fold,
                                'Frequency': frequency
                            })
        
        unified_df = pd.DataFrame(all_data)
        return self._run_global_mixed_effects(unified_df, design, analysis_type)
    
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
    
    def _run_posthocs_at_step(self, df_step: pd.DataFrame, step_value: float, analysis_type: str,
                               design: ExperimentalDesign, between_p: float, within_p: float, 
                               interaction_p: float, reasons: List[str], alpha: float = 0.05) -> List[Dict]:
        """Run post-hoc t-tests at a specific step if ANOVA was significant (corrected p-values)."""
        
        posthocs = []
        
        # Run post-hocs if any effect is significant (using corrected p-values)
        if not (between_p < alpha or within_p < alpha or interaction_p < alpha):
            return posthocs
        
        between_levels = sorted(df_step['Between_Factor'].unique())
        within_levels = sorted(df_step['Within_Factor'].unique())
        
        step_key = 'current_step' if analysis_type == 'current' else 'fold_step'
        
        if 'interaction' in reasons:
            # Within-factor comparisons at each between-level (PAIRED t-tests)
            for between_level in between_levels:
                subset = df_step[df_step['Between_Factor'] == between_level]
                
                for wlevel1, wlevel2 in combinations(within_levels, 2):
                    try:
                        data1 = subset[subset['Within_Factor'] == wlevel1].set_index('Subject_ID')['Frequency']
                        data2 = subset[subset['Within_Factor'] == wlevel2].set_index('Subject_ID')['Frequency']
                        
                        common_subjects = data1.index.intersection(data2.index)
                        if len(common_subjects) < 3:
                            continue
                        
                        data1_paired = data1.loc[common_subjects]
                        data2_paired = data2.loc[common_subjects]
                        
                        t_result = pg.ttest(data1_paired, data2_paired, paired=True)
                        
                        posthocs.append({
                            step_key: step_value,
                            'Test_Type': 'Paired t-test',
                            'Comparison': f"{between_level}: {wlevel1} vs {wlevel2}",
                            'Group1': f"{between_level}: {wlevel1}",
                            'Group1_mean': data1_paired.mean(),
                            'Group1_stderr': data1_paired.std() / np.sqrt(len(data1_paired)),
                            'Group1_n': len(common_subjects),
                            'Group2': f"{between_level}: {wlevel2}",
                            'Group2_mean': data2_paired.mean(),
                            'Group2_stderr': data2_paired.std() / np.sqrt(len(data2_paired)),
                            'Group2_n': len(common_subjects),
                            't_statistic': t_result['T'].values[0],
                            'p_value': t_result['p-val'].values[0]
                        })
                    except Exception as e:
                        logger.debug(f"Error in paired t-test at step {step_value}: {e}")
            
            # Between-factor comparisons at each within-level (INDEPENDENT t-tests)
            for within_level in within_levels:
                subset = df_step[df_step['Within_Factor'] == within_level]
                
                for blevel1, blevel2 in combinations(between_levels, 2):
                    try:
                        data1 = subset[subset['Between_Factor'] == blevel1]['Frequency']
                        data2 = subset[subset['Between_Factor'] == blevel2]['Frequency']
                        
                        if len(data1) < 2 or len(data2) < 2:
                            continue
                        
                        t_result = pg.ttest(data1, data2, paired=False)
                        
                        posthocs.append({
                            step_key: step_value,
                            'Test_Type': 'Independent t-test',
                            'Comparison': f"{within_level}: {blevel1} vs {blevel2}",
                            'Group1': f"{within_level}: {blevel1}",
                            'Group1_mean': data1.mean(),
                            'Group1_stderr': data1.std() / np.sqrt(len(data1)),
                            'Group1_n': len(data1),
                            'Group2': f"{within_level}: {blevel2}",
                            'Group2_mean': data2.mean(),
                            'Group2_stderr': data2.std() / np.sqrt(len(data2)),
                            'Group2_n': len(data2),
                            't_statistic': t_result['T'].values[0],
                            'p_value': t_result['p-val'].values[0]
                        })
                    except Exception as e:
                        logger.debug(f"Error in independent t-test at step {step_value}: {e}")
        else:
            if 'factor2_main' in reasons:
                posthocs.extend(self._run_marginal_within_step(df_step, step_value, step_key, design, analysis_type))
            if 'factor1_main' in reasons:
                posthocs.extend(self._run_marginal_between_step(df_step, step_value, step_key, design, analysis_type))
        
        return posthocs
    
    def _run_marginal_within_step(self, df_step: pd.DataFrame, step_value: float, step_key: str,
                                  design: ExperimentalDesign, analysis_type: str) -> List[Dict]:
        """Run marginal paired comparisons collapsing across between-factor levels."""
        results = []
        measurement_type = f'frequency_{analysis_type}'
        df_local = df_step.copy()
        df_local['Subject_Key'] = df_local['Between_Factor'].astype(str) + "__" + df_local['Subject_ID'].astype(str)
        within_levels = sorted(df_local['Within_Factor'].unique())
        pivot = df_local.pivot_table(
            index='Subject_Key',
            columns='Within_Factor',
            values='Frequency',
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
                    step_key: step_value,
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
                    't_statistic': t_result['T'].values[0],
                    'p_value': t_result['p-val'].values[0],
                    'measurement_type': measurement_type
                })
            except Exception as e:
                logger.debug(f"Marginal within comparison failed at step {step_value}: {e}")
        return results
    
    def _run_marginal_between_step(self, df_step: pd.DataFrame, step_value: float, step_key: str,
                                   design: ExperimentalDesign, analysis_type: str) -> List[Dict]:
        """Run marginal independent comparisons collapsing across within-factor levels."""
        results = []
        measurement_type = f'frequency_{analysis_type}'
        df_local = df_step.copy()
        df_local['Subject_Key'] = df_local['Between_Factor'].astype(str) + "__" + df_local['Subject_ID'].astype(str)
        subject_means = (
            df_local.groupby(['Between_Factor', 'Subject_Key'])['Frequency']
            .mean()
            .reset_index()
        )
        between_levels = sorted(subject_means['Between_Factor'].unique())
        
        for blevel1, blevel2 in combinations(between_levels, 2):
            data1 = subject_means[subject_means['Between_Factor'] == blevel1]['Frequency']
            data2 = subject_means[subject_means['Between_Factor'] == blevel2]['Frequency']
            if len(data1) < 2 or len(data2) < 2:
                continue
            try:
                t_result = pg.ttest(data1, data2, paired=False)
                results.append({
                    step_key: step_value,
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
                    't_statistic': t_result['T'].values[0],
                    'p_value': t_result['p-val'].values[0],
                    'measurement_type': measurement_type
                })
            except Exception as e:
                logger.debug(f"Marginal between comparison failed at step {step_value}: {e}")
        return results
    
    def _apply_posthoc_fdr(self, posthoc_results: List[Dict]) -> List[Dict]:
        """Apply FDR correction per step (within each ANOVA family) to match independent design."""
        
        if not posthoc_results:
            return posthoc_results
        
        # Group post-hoc results by step value (each ANOVA family)
        step_key = 'current_step' if 'current_step' in posthoc_results[0] else 'fold_step'
        step_groups = {}
        for result in posthoc_results:
            step_value = result.get(step_key)
            if step_value not in step_groups:
                step_groups[step_value] = []
            step_groups[step_value].append(result)
        
        # Apply FDR correction within each step's post-hoc tests
        for step_value, results_list in step_groups.items():
            if len(results_list) > 1:
                # Extract p-values (filter out NaN)
                p_values = [r['p_value'] for r in results_list if not np.isnan(r['p_value'])]
                valid_indices = [i for i, r in enumerate(results_list) if not np.isnan(r['p_value'])]
                
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
                        
                        logger.debug(f"Applied FDR correction to {len(p_values)} post-hoc tests at step {step_value}")
                        
                    except Exception as e:
                        logger.error(f"Error applying post-hoc FDR correction at step {step_value}: {e}")
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
        
        logger.info(f"Applied FDR correction per step to {len(posthoc_results)} post-hoc tests across {len(step_groups)} steps")
        
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
            p_values = [r[effect_type] for r in results if not np.isnan(r[effect_type])]
            valid_indices = [i for i, r in enumerate(results) if not np.isnan(r[effect_type])]
            
            if len(p_values) > 1:
                try:
                    rejected, corrected_p, _, _ = multi.multipletests(p_values, method="fdr_bh")
                    
                    corrected_key = effect_type.replace('_p', '_corrected_p')
                    for i, corrected_val in zip(valid_indices, corrected_p):
                        results[i][corrected_key] = corrected_val
                    
                    logger.info(f"Applied FDR to {len(p_values)} {effect_type} tests")
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
    
    def _save_point_by_point_results(self, results: List[Dict], base_path: str, x_name: str,
                                     design: ExperimentalDesign = None) -> None:
        """Save point-by-point mixed model results to CSV (matching independent format)."""
        
        if not results:
            logger.warning("No results to save")
            return
        
        try:
            # Prepare data for CSV
            rows = []
            for result in results:
                step_key = 'current_step' if 'current_step' in result else 'fold_step'
                step_value = result[step_key]
                
                # Base row data
                row = {
                    'Step_Value': step_value,
                    'Step_Label': f"{x_name}={step_value}",
                    'Test_Type': 'Mixed ANOVA'
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
            # Order: Step_Value -> Group stats -> ANOVA effects
            base_cols = ['Step_Value']
            
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
            import os
            output_path = os.path.join(base_path, "Results", f"Stats_{x_name}_vs_frequency_each_point_Mixed_ANOVA.csv")
            df.to_csv(output_path, index=False)
            logger.info(f"Saved point-by-point mixed model results to {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving point-by-point results: {e}")
    
    def _save_posthoc_results(self, posthoc_results: List[Dict], base_path: str, analysis_name: str) -> None:
        """Save post-hoc results to a separate CSV file (matching independent format)."""
        
        if not posthoc_results:
            return
        
        try:
            df = pd.DataFrame(posthoc_results)
            
            # Reorder columns to match independent design format
            step_key = 'current_step' if 'current_step' in df.columns else 'fold_step'
            base_cols = [step_key, 'Test_Type', 'Comparison']
            group_cols = ['Group1', 'Group1_mean', 'Group1_stderr', 'Group1_n',
                         'Group2', 'Group2_mean', 'Group2_stderr', 'Group2_n']
            stat_cols = ['t_statistic', 'p_value']
            if 'corrected_p' in df.columns:
                stat_cols.append('corrected_p')
            
            # Reorder using only columns that exist
            column_order = base_cols + group_cols + stat_cols
            available_cols = [col for col in column_order if col in df.columns]
            df = df[available_cols]
            
            output_path = os.path.join(base_path, "Results", f"Stats_{analysis_name}_each_point_pairwise.csv")
            df.to_csv(output_path, index=False)
            logger.info(f"Saved {len(posthoc_results)} post-hoc results to {output_path}")
        except Exception as e:
            logger.error(f"Error saving post-hoc results: {e}")
    
    def _run_global_mixed_effects(self, unified_df: pd.DataFrame, design: ExperimentalDesign, analysis_type: str):
        """Run global mixed-effects model across all steps/APs.
        
        Formula: Frequency ~ C(Between_Factor) * C(Within_Factor) * Step_Variable_z (standardized)
        This tests for all main effects, 2-way, and 3-way interactions with standardized continuous variable.
        """
        
        # Determine column name based on data type (same as independent analysis)
        if analysis_type == 'current':
            step_col = 'Current'
        elif analysis_type == 'fold_rheobase':
            step_col = 'FoldRheobase'
        else:
            logger.warning(f"Unknown analysis type: {analysis_type}")
            return None
        
        try:
            if unified_df.empty or step_col not in unified_df.columns:
                logger.warning(f"Cannot run global model: {step_col} column not found")
                return None
            
            # Drop any rows with NaN in critical columns
            unified_df = unified_df.dropna(subset=['Frequency', step_col, 'Subject_ID', 'Between_Factor', 'Within_Factor'])
            
            # Standardize continuous variable (z-score) to help convergence
            step_col_z = f"{step_col}_z"
            unified_df[step_col_z] = (unified_df[step_col] - unified_df[step_col].mean()) / unified_df[step_col].std()
            
            logger.info(f"Running global mixed-effects model for {analysis_type} with random slopes for standardized {step_col}...")
            
            # Full 3-way interaction model with standardized continuous variable
            formula = f"Frequency ~ C(Between_Factor) * C(Within_Factor) * {step_col_z}"
            model = smf.mixedlm(
                formula, 
                unified_df, 
                groups=unified_df["Subject_ID"],
                re_formula=f"1 + {step_col_z}"  # Random slopes for standardized continuous predictor
            ).fit(method='lbfgs', maxiter=5000)
            
            # Check convergence
            if not model.converged:
                logger.warning(f"Global model for {analysis_type} did not converge after 5000 iterations")
            else:
                logger.info(f"Global model for {analysis_type} converged successfully")
            
            # Extract p-values for all 7 effects using proper statistical tests (LRT for multiple contrasts)
            between_params = [p for p in model.pvalues.index if 'C(Between_Factor)' in p and ':' not in p]
            between_p = _get_effect_pvalue(model, between_params, unified_df, formula, 'Subject_ID', f"1 + {step_col_z}", "Between_Factor")
            
            within_params = [p for p in model.pvalues.index if 'C(Within_Factor)' in p and ':' not in p]
            within_p = _get_effect_pvalue(model, within_params, unified_df, formula, 'Subject_ID', f"1 + {step_col_z}", "Within_Factor")
            
            # Use standardized variable name
            step_p = model.pvalues.get(step_col_z, 1.0)
            
            # Create display label (matching independent)
            x_var_label = 'Current' if analysis_type == 'current' else 'Fold Rheobase'
            
            # Two-way interactions - use LRT for multiple contrasts
            between_within_params = [p for p in model.pvalues.index if 'C(Between_Factor)' in p and 'C(Within_Factor)' in p and step_col_z not in p]
            between_within_p = _get_effect_pvalue(model, between_within_params, unified_df, formula, 'Subject_ID', f"1 + {step_col_z}", "Between:Within")
            
            between_step_params = [p for p in model.pvalues.index if 'C(Between_Factor)' in p and step_col_z in p and 'C(Within_Factor)' not in p]
            between_step_p = _get_effect_pvalue(model, between_step_params, unified_df, formula, 'Subject_ID', f"1 + {step_col_z}", f"Between:{x_var_label}")
            
            within_step_params = [p for p in model.pvalues.index if 'C(Within_Factor)' in p and step_col_z in p and 'C(Between_Factor)' not in p]
            within_step_p = _get_effect_pvalue(model, within_step_params, unified_df, formula, 'Subject_ID', f"1 + {step_col_z}", f"Within:{x_var_label}")
            
            # Three-way interaction
            three_way_params = [p for p in model.pvalues.index if 'C(Between_Factor)' in p and 'C(Within_Factor)' in p and step_col_z in p]
            three_way_p = _get_effect_pvalue(model, three_way_params, unified_df, formula, 'Subject_ID', f"1 + {step_col_z}", f"Between:Within:{x_var_label}")
            
            between_levels = sorted(unified_df['Between_Factor'].dropna().unique())
            within_levels = sorted(unified_df['Within_Factor'].dropna().unique())
            
            posthoc_rows = compute_global_curve_posthocs(
                model=model,
                step_col_z=step_col_z,
                analysis_label='Current_vs_Frequency' if analysis_type == 'current' else 'Fold_Rheobase_vs_Frequency',
                factor_a_col='Between_Factor',
                factor_b_col='Within_Factor',
                factor_a_label=design.between_factor_name,
                factor_b_label=design.within_factor_name,
                factor_a_levels=between_levels,
                factor_b_levels=within_levels,
                main_a_p=between_p,
                main_b_p=within_p,
                interaction_ab_p=between_within_p,
                interaction_a_step_p=between_step_p,
                interaction_b_step_p=within_step_p,
                interaction_three_way_p=three_way_p
            )
            
            # Create result with all 7 effects
            rows = [
                {'Effect': design.between_factor_name, 'p-value': between_p},
                {'Effect': design.within_factor_name, 'p-value': within_p},
                {'Effect': x_var_label, 'p-value': step_p},
                {'Effect': f'{design.between_factor_name}:{design.within_factor_name}', 'p-value': between_within_p},
                {'Effect': f'{design.between_factor_name}:{x_var_label}', 'p-value': between_step_p},
                {'Effect': f'{design.within_factor_name}:{x_var_label}', 'p-value': within_step_p},
                {'Effect': f'{design.between_factor_name}:{design.within_factor_name}:{x_var_label}', 'p-value': three_way_p}
            ]
            
            logger.info(f"Global mixed-effects model complete for {analysis_type}")
            return {
                'effects': rows,
                'posthocs': posthoc_rows
            }
            
        except Exception as e:
            logger.error(f"Error running global mixed-effects model for {analysis_type}: {e}")
            return None
    

