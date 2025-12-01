"""
Mixed ANOVA for dependent (paired/repeated measures/mixed factorial) designs.
Handles general parameter analysis using mixed effects models.
"""

import pandas as pd
import numpy as np
import logging
import os
from itertools import combinations
from typing import List, Dict, Tuple, Optional
import scipy.stats as stats
import statsmodels.stats.multitest as multi
import statsmodels.formula.api as smf
import pingouin as pg

from ...shared.data_models import ExperimentalDesign, StatisticalResult, DataContainer
from ...shared.utils import clean_dataframe, categorize_measurement, get_measurement_categories
from .posthoc_utils import should_run_posthocs, get_simple_effect_comparisons_dependent

logger = logging.getLogger(__name__)


class MixedANOVA:
    """Mixed ANOVA for dependent designs (paired, RM, mixed factorial)."""
    
    def __init__(self):
        self.name = "Mixed ANOVA"
        
    def run_analysis(self, design: ExperimentalDesign, container: DataContainer, 
                    base_path: str) -> List[StatisticalResult]:
        """
        Run mixed model analysis for general parameters.
        
        Args:
            design: ExperimentalDesign with dependent design type
            container: DataContainer (not used, kept for consistency)
            base_path: Base directory path
            
        Returns:
            List of StatisticalResult objects
        """
        
        if not design.is_paired():
            raise ValueError("Mixed model test requires a dependent design")
            
        manifest = design.pairing_manifest
        between_levels = sorted(manifest['Group'].unique())
        within_levels = sorted(manifest['Condition'].unique())
        expected_groups = len(between_levels) * len(within_levels)
        
        if len(design.groups) != expected_groups:
            raise ValueError(
                f"Mixed design requires {expected_groups} groups for "
                f"{len(between_levels)}×{len(within_levels)} design (found {len(design.groups)})"
            )
            
        logger.info(f"Running {len(between_levels)}×{len(within_levels)} mixed model test: "
                   f"{design.between_factor_name} × {design.within_factor_name}")
        
        # Load and combine data for all groups
        group_data = {}
        for group in design.groups:
            combined_data = self._load_combined_data(group.name, base_path)
            if not combined_data.empty:
                group_data[group.name] = clean_dataframe(combined_data)
        
        if len(group_data) != expected_groups:
            raise ValueError(f"No data found for all {expected_groups} groups (found data for {len(group_data)})")
        
        # Create unified dataframe with factor labels
        unified_df = self._create_unified_dataframe(group_data, design, manifest)
        
        # Get column names for analysis
        analysis_columns = self._get_analysis_columns(unified_df)
        
        # Run mixed model for each measurement
        mixed_results = []
        for column in analysis_columns:
            result = self._run_single_mixed_model(column, unified_df, design)
            if result:
                mixed_results.extend(result)  # Returns 3 results (Between, Within, Interaction)
        
        # Apply multiple comparison correction
        mixed_results = self._apply_multiple_comparison_correction(mixed_results, design)
        
        # Run post-hoc comparisons using model contrasts
        posthoc_results = self._run_posthoc_if_significant(
            mixed_results, unified_df, design, analysis_columns
        )
        
        # Apply FDR correction to post-hoc tests
        if posthoc_results:
            posthoc_results = self._apply_posthoc_correction(posthoc_results)
        
        # Combine all results
        all_results = mixed_results + posthoc_results
        
        logger.info(f"Mixed model test complete: {len(mixed_results)} effects, {len(posthoc_results)} post-hoc comparisons")
        
        return all_results
    
    def _load_combined_data(self, group_name: str, base_path: str) -> pd.DataFrame:
        """Load and combine all data types for a group."""
        
        dfs = []
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
                except Exception as e:
                    logger.warning(f"Failed to load {filename}: {e}")
                    
        if dfs:
            combined = pd.concat(dfs, ignore_index=True)
            return clean_dataframe(combined)
        else:
            return pd.DataFrame()
    
    def _get_analysis_columns(self, df: pd.DataFrame) -> List[str]:
        """Get columns suitable for statistical analysis (excluding metadata)."""
        
        exclude_columns = ['Filename', 'Subject_ID', 'Between_Factor', 'Within_Factor', 'Group_Name', 'filename']
        
        analysis_cols = []
        for col in df.columns:
            if col in exclude_columns or col.startswith('Subject_'):
                continue
            
            if df[col].dtype in [np.float64, np.int64, np.float32, np.int32]:
                if df[col].notna().sum() >= 5:
                    analysis_cols.append(col)
        
        return analysis_cols
    
    def _create_unified_dataframe(self, group_data: Dict[str, pd.DataFrame],
                                  design: ExperimentalDesign, 
                                  manifest: pd.DataFrame) -> pd.DataFrame:
        """Create unified dataframe with factor labels and subject IDs."""
        
        unified_dfs = []
        
        for group_name, group_df in group_data.items():
            # Parse group name to extract factor levels
            between_level, within_level = self._parse_group_name(group_name, manifest)
            
            # Add factor columns
            group_df = group_df.copy()
            group_df['Between_Factor'] = between_level
            group_df['Within_Factor'] = within_level
            group_df['Group_Name'] = group_name
            
            # Ensure Subject_ID exists
            if 'Subject_ID' not in group_df.columns:
                logger.warning(f"No Subject_ID column in {group_name}, cannot perform mixed model")
                continue
            
            # Ensure Subject_ID is string type
            group_df['Subject_ID'] = group_df['Subject_ID'].astype(str)
            
            unified_dfs.append(group_df)
        
        if not unified_dfs:
            raise ValueError("No valid data with Subject_ID found")
        
        unified_df = pd.concat(unified_dfs, ignore_index=True)
        
        logger.info(f"Created unified dataframe: {len(unified_df)} total observations")
        logger.info(f"  Between levels: {sorted(unified_df['Between_Factor'].unique())}")
        logger.info(f"  Within levels: {sorted(unified_df['Within_Factor'].unique())}")
        logger.info(f"  Subjects: {unified_df['Subject_ID'].nunique()}")
        
        return unified_df
    
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
    
    def _run_single_mixed_model(self, column: str, unified_df: pd.DataFrame,
                                design: ExperimentalDesign) -> Optional[List[StatisticalResult]]:
        """Run mixed ANOVA using pingouin for a single measurement."""
        
        if column not in unified_df.columns:
            return None
        
        df_clean = unified_df[['Between_Factor', 'Within_Factor', 'Subject_ID', column]].copy()
        df_clean = df_clean.dropna(subset=[column])
        
        # Collapse to one observation per subject/condition (average across repeats)
        if not df_clean.empty:
            before_count = len(df_clean)
            df_clean = (
                df_clean.groupby(['Subject_ID', 'Between_Factor', 'Within_Factor'], as_index=False)[column]
                .mean()
            )
            if len(df_clean) < before_count:
                logger.debug(
                    f"{column}: collapsed {before_count - len(df_clean)} duplicate rows into subject means"
                )
        
        if len(df_clean) < 4:
            logger.warning(f"Insufficient data for {column} (n={len(df_clean)})")
            return None
        
        n_subjects = df_clean['Subject_ID'].nunique()
        if n_subjects < 2:
            logger.warning(f"Need at least 2 subjects for mixed ANOVA (found {n_subjects})")
            return None
        
        try:
            # Run mixed ANOVA using pingouin
            aov = pg.mixed_anova(
                dv=column,
                within='Within_Factor',
                between='Between_Factor',
                subject='Subject_ID',
                data=df_clean
            )
            
            logger.info(f"Mixed ANOVA for {column} completed successfully")
            
            # Extract p-values from the ANOVA table
            between_p = aov[aov['Source'] == 'Between_Factor']['p-unc'].values[0] if 'Between_Factor' in aov['Source'].values else np.nan
            within_p = aov[aov['Source'] == 'Within_Factor']['p-unc'].values[0] if 'Within_Factor' in aov['Source'].values else np.nan
            interaction_p = aov[aov['Source'] == 'Interaction']['p-unc'].values[0] if 'Interaction' in aov['Source'].values else np.nan
            
            # Create results
            results = []
            
            results.append(StatisticalResult(
                test_name="Mixed ANOVA - Between Effect",
                measurement=column,
                group1_name=f"{design.between_factor_name}",
                group1_mean=0, group1_stderr=0, group1_n=0,
                group2_name="",
                group2_mean=0, group2_stderr=0, group2_n=0,
                p_value=between_p,
                measurement_type=categorize_measurement(column)
            ))
            
            results.append(StatisticalResult(
                test_name="Mixed ANOVA - Within Effect",
                measurement=column,
                group1_name=f"{design.within_factor_name}",
                group1_mean=0, group1_stderr=0, group1_n=0,
                group2_name="",
                group2_mean=0, group2_stderr=0, group2_n=0,
                p_value=within_p,
                measurement_type=categorize_measurement(column)
            ))
            
            results.append(StatisticalResult(
                test_name="Mixed ANOVA - Interaction",
                measurement=column,
                group1_name=f"{design.between_factor_name} × {design.within_factor_name}",
                group1_mean=0, group1_stderr=0, group1_n=0,
                group2_name="",
                group2_mean=0, group2_stderr=0, group2_n=0,
                p_value=interaction_p,
                measurement_type=categorize_measurement(column)
            ))
            
            return results
            
        except Exception as e:
            logger.error(f"Error running mixed ANOVA for {column}: {e}")
            return None
    
    def _extract_factor_p(self, pvalues_dict: Dict, factor_name: str, 
                          exclude_interaction: bool = False) -> Optional[float]:
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
    
    def _extract_interaction_p(self, pvalues_dict: Dict) -> Optional[float]:
        """Extract the minimum p-value for interaction terms."""
        
        interaction_p = []
        
        for param, p in pvalues_dict.items():
            if ':' in param:
                interaction_p.append(p)
        
        if interaction_p:
            return min(interaction_p)
        return None
    
    def _run_posthoc_if_significant(self, mixed_results: List[StatisticalResult],
                                    unified_df: pd.DataFrame,
                                    design: ExperimentalDesign,
                                    analysis_columns: List[str],
                                    alpha: float = 0.05) -> List[StatisticalResult]:
        """Run simple effects post-hocs for measurements with any significant effect (matching independent approach)."""
        
        # Get measurements that need post-hocs based on corrected p-values and factor levels
        measurements_needing_posthocs = set()
        measurement_reasons = {}  # Track which effects require post-hocs
        
        # Group results by measurement
        results_by_measurement = {}
        for result in mixed_results:
            if result.measurement not in results_by_measurement:
                results_by_measurement[result.measurement] = {}
            
            # Store corrected p-values for each effect
            p = result.corrected_p if hasattr(result, 'corrected_p') and result.corrected_p is not None else result.p_value
            
            if 'Between' in result.test_name:
                results_by_measurement[result.measurement]['between_corrected_p'] = p
            elif 'Within' in result.test_name:
                results_by_measurement[result.measurement]['within_corrected_p'] = p
            elif 'Interaction' in result.test_name:
                results_by_measurement[result.measurement]['Interaction_corrected_p'] = p
        
        # For each measurement, determine if post-hocs are needed
        group_names = [g.name for g in design.groups]
        for measurement, effect_ps in results_by_measurement.items():
            # Create a result dict with the naming expected by should_run_posthocs
            anova_result = {
                f'{design.between_factor_name}_corrected_p': effect_ps.get('between_corrected_p', 1),
                f'{design.within_factor_name}_corrected_p': effect_ps.get('within_corrected_p', 1),
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
        
        # Run appropriate post-hocs for each measurement
        all_posthoc_results = []
        for measurement in measurements_needing_posthocs:
            reasons = measurement_reasons.get(measurement, [])
            if 'interaction' in reasons:
                posthoc = self._run_posthoc_for_measurement(
                    measurement, unified_df, design, sig_dict=None
                )
                all_posthoc_results.extend(posthoc)
            else:
                if 'factor1_main' in reasons:
                    posthoc = self._run_marginal_posthoc(
                        measurement, unified_df, design, factor_key='between'
                    )
                    all_posthoc_results.extend(posthoc)
                if 'factor2_main' in reasons:
                    posthoc = self._run_marginal_posthoc(
                        measurement, unified_df, design, factor_key='within'
                    )
                    all_posthoc_results.extend(posthoc)
        
        return all_posthoc_results
    
    def _run_posthoc_for_measurement(self, measurement: str, unified_df: pd.DataFrame,
                                    design: ExperimentalDesign,
                                    sig_dict: Optional[Dict[str, bool]] = None) -> List[StatisticalResult]:
        """Run simple effects post-hocs using paired/independent t-tests."""
        
        results = []
        
        df_clean = unified_df[['Between_Factor', 'Within_Factor', 'Subject_ID', measurement]].copy()
        df_clean = df_clean.dropna(subset=[measurement])
        
        between_levels = sorted(df_clean['Between_Factor'].unique())
        within_levels = sorted(df_clean['Within_Factor'].unique())
        
        # 1. Within-factor comparisons at each between-level (PAIRED t-tests)
        # These compare different conditions for the same subjects
        for between_level in between_levels:
            subset = df_clean[df_clean['Between_Factor'] == between_level]
            
            for wlevel1, wlevel2 in combinations(within_levels, 2):
                try:
                    # Get data for each within-factor level, indexed by subject
                    data1 = subset[subset['Within_Factor'] == wlevel1].set_index('Subject_ID')[measurement]
                    data2 = subset[subset['Within_Factor'] == wlevel2].set_index('Subject_ID')[measurement]
                    
                    # Find common subjects (paired data)
                    common_subjects = data1.index.intersection(data2.index)
                    
                    if len(common_subjects) < 3:
                        logger.debug(f"Insufficient paired data for {between_level}: {wlevel1} vs {wlevel2} (n={len(common_subjects)})")
                        continue
                    
                    data1_paired = data1.loc[common_subjects]
                    data2_paired = data2.loc[common_subjects]
                    
                    # Run paired t-test using pingouin
                    t_result = pg.ttest(data1_paired, data2_paired, paired=True)
                    p_value = t_result['p-val'].values[0]
                    
                    results.append(StatisticalResult(
                        test_name="Paired t-test (simple effect)",
                        measurement=measurement,
                        group1_name=f"{between_level}: {wlevel1}",
                        group1_mean=data1_paired.mean(),
                        group1_stderr=data1_paired.sem(),
                        group1_n=len(data1_paired),
                        group2_name=f"{between_level}: {wlevel2}",
                        group2_mean=data2_paired.mean(),
                        group2_stderr=data2_paired.sem(),
                        group2_n=len(data2_paired),
                        p_value=p_value,
                        measurement_type=categorize_measurement(measurement)
                    ))
                    
                except Exception as e:
                    logger.warning(f"Error in paired t-test for {between_level} ({wlevel1} vs {wlevel2}): {e}")
        
        # 2. Between-factor comparisons at each within-level (INDEPENDENT t-tests)
        # These compare different subject groups at the same condition
        for within_level in within_levels:
            subset = df_clean[df_clean['Within_Factor'] == within_level]
            
            for blevel1, blevel2 in combinations(between_levels, 2):
                try:
                    # Get data for each between-factor level (different subjects)
                    data1 = subset[subset['Between_Factor'] == blevel1][measurement]
                    data2 = subset[subset['Between_Factor'] == blevel2][measurement]
                    
                    if len(data1) < 2 or len(data2) < 2:
                        logger.debug(f"Insufficient data for {within_level}: {blevel1} vs {blevel2} (n1={len(data1)}, n2={len(data2)})")
                        continue
                    
                    # Run independent t-test using pingouin
                    t_result = pg.ttest(data1, data2, paired=False)
                    p_value = t_result['p-val'].values[0]
                    
                    results.append(StatisticalResult(
                        test_name="Independent t-test (simple effect)",
                        measurement=measurement,
                        group1_name=f"{within_level}: {blevel1}",
                        group1_mean=data1.mean(),
                        group1_stderr=data1.sem(),
                        group1_n=len(data1),
                        group2_name=f"{within_level}: {blevel2}",
                        group2_mean=data2.mean(),
                        group2_stderr=data2.sem(),
                        group2_n=len(data2),
                        p_value=p_value,
                        measurement_type=categorize_measurement(measurement)
                    ))
                    
                except Exception as e:
                    logger.warning(f"Error in independent t-test for {within_level} ({blevel1} vs {blevel2}): {e}")
        
        return results
    
    def _run_marginal_posthoc(self, measurement: str, unified_df: pd.DataFrame,
                              design: ExperimentalDesign, factor_key: str) -> List[StatisticalResult]:
        """Run marginal mean comparisons for mixed designs (between or within factor)."""
        
        df_clean = unified_df[['Between_Factor', 'Within_Factor', 'Subject_ID', measurement]].copy()
        df_clean = df_clean.dropna(subset=[measurement])
        if df_clean.empty:
            return []
        
        results = []
        measurement_type = categorize_measurement(measurement)
        
        # Build unique subject identifier to avoid collisions across groups
        df_clean['Subject_Key'] = df_clean['Between_Factor'].astype(str) + "__" + df_clean['Subject_ID'].astype(str)
        
        if factor_key == 'within':
            within_levels = sorted(df_clean['Within_Factor'].unique())
            if len(within_levels) < 2:
                return []
            
            pivot = df_clean.pivot_table(
                index='Subject_Key',
                columns='Within_Factor',
                values=measurement,
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
                    p_value = t_result['p-val'].values[0]
                    results.append(StatisticalResult(
                        test_name="Marginal paired t-test",
                        measurement=measurement,
                        group1_name=f"{design.within_factor_name}={wlevel1}",
                        group1_mean=paired[wlevel1].mean(),
                        group1_stderr=paired[wlevel1].std(ddof=1) / np.sqrt(len(paired)) if len(paired) > 1 else 0.0,
                        group1_n=len(paired),
                        group2_name=f"{design.within_factor_name}={wlevel2}",
                        group2_mean=paired[wlevel2].mean(),
                        group2_stderr=paired[wlevel2].std(ddof=1) / np.sqrt(len(paired)) if len(paired) > 1 else 0.0,
                        group2_n=len(paired),
                        p_value=p_value,
                        measurement_type=measurement_type
                    ))
                except Exception as e:
                    logger.debug(f"Marginal within comparison failed for {measurement} ({wlevel1} vs {wlevel2}): {e}")
        else:
            between_levels = sorted(df_clean['Between_Factor'].unique())
            if len(between_levels) < 2:
                return []
            
            subject_means = (
                df_clean.groupby(['Between_Factor', 'Subject_Key'])[measurement]
                .mean()
                .reset_index()
            )
            
            for blevel1, blevel2 in combinations(between_levels, 2):
                data1 = subject_means[subject_means['Between_Factor'] == blevel1][measurement]
                data2 = subject_means[subject_means['Between_Factor'] == blevel2][measurement]
                
                if len(data1) < 2 or len(data2) < 2:
                    continue
                
                try:
                    t_result = pg.ttest(data1, data2, paired=False)
                    p_value = t_result['p-val'].values[0]
                    results.append(StatisticalResult(
                        test_name="Marginal independent t-test",
                        measurement=measurement,
                        group1_name=f"{design.between_factor_name}={blevel1}",
                        group1_mean=data1.mean(),
                        group1_stderr=data1.std(ddof=1) / np.sqrt(len(data1)) if len(data1) > 1 else 0.0,
                        group1_n=len(data1),
                        group2_name=f"{design.between_factor_name}={blevel2}",
                        group2_mean=data2.mean(),
                        group2_stderr=data2.std(ddof=1) / np.sqrt(len(data2)) if len(data2) > 1 else 0.0,
                        group2_n=len(data2),
                        p_value=p_value,
                        measurement_type=measurement_type
                    ))
                except Exception as e:
                    logger.debug(f"Marginal between comparison failed for {measurement} ({blevel1} vs {blevel2}): {e}")
        
        return results
    
    def _calculate_simple_effect_within(self, wlevel1, wlevel2, between_level,
                                       within_coeffs, interaction_coeffs, between_levels, within_levels,
                                       params, cov_matrix, fitted_model) -> Tuple[float, float]:
        """Calculate estimate and SE for within-factor simple effect (wlevel1 vs wlevel2 at specific between_level)."""
        
        # Get cell means for each combination
        est1 = self._get_cell_mean_coef(between_level, wlevel1, between_levels, within_levels,
                                       within_coeffs, params, interaction_coeffs)
        est2 = self._get_cell_mean_coef(between_level, wlevel2, between_levels, within_levels,
                                       within_coeffs, params, interaction_coeffs)
        
        estimate = est1 - est2
        
        # Calculate SE using covariance matrix
        se = self._calculate_contrast_se(
            between_level, wlevel1, between_level, wlevel2,
            between_levels, within_levels, within_coeffs, interaction_coeffs,
            cov_matrix, fitted_model
        )
        
        return estimate, se
    
    def _calculate_simple_effect_between(self, blevel1, blevel2, within_level,
                                        between_coeffs, interaction_coeffs, between_levels, within_levels,
                                        params, cov_matrix, fitted_model) -> Tuple[float, float]:
        """Calculate estimate and SE for between-factor simple effect (blevel1 vs blevel2 at specific within_level)."""
        
        # Get cell means for each combination
        est1 = self._get_cell_mean_coef(blevel1, within_level, between_levels, within_levels,
                                       between_coeffs, params, interaction_coeffs, is_between=True)
        est2 = self._get_cell_mean_coef(blevel2, within_level, between_levels, within_levels,
                                       between_coeffs, params, interaction_coeffs, is_between=True)
        
        estimate = est1 - est2
        
        # Calculate SE using covariance matrix
        se = self._calculate_contrast_se(
            blevel1, within_level, blevel2, within_level,
            between_levels, within_levels, between_coeffs, interaction_coeffs,
            cov_matrix, fitted_model, is_between=True
        )
        
        return estimate, se
    
    def _calculate_main_effect_between(self, blevel1, blevel2, between_coeffs,
                                      params, cov_matrix, fitted_model) -> Tuple[float, float]:
        """Calculate estimate and SE for between-factor main effect."""
        
        coef1 = between_coeffs.get(blevel1)
        coef2 = between_coeffs.get(blevel2)
        
        # Get estimates
        est1 = params[coef1] if coef1 else 0.0
        est2 = params[coef2] if coef2 else 0.0
        estimate = est1 - est2
        
        # Calculate SE
        if coef1 and coef2:
            se = np.sqrt(
                cov_matrix.loc[coef1, coef1] +
                cov_matrix.loc[coef2, coef2] -
                2 * cov_matrix.loc[coef1, coef2]
            )
        elif coef1:
            se = fitted_model.bse[coef1]
        elif coef2:
            se = fitted_model.bse[coef2]
        else:
            se = np.nan
        
        return estimate, se
    
    def _calculate_main_effect_within(self, wlevel1, wlevel2, within_coeffs,
                                     params, cov_matrix, fitted_model) -> Tuple[float, float]:
        """Calculate estimate and SE for within-factor main effect."""
        
        coef1 = within_coeffs.get(wlevel1)
        coef2 = within_coeffs.get(wlevel2)
        
        # Get estimates
        est1 = params[coef1] if coef1 else 0.0
        est2 = params[coef2] if coef2 else 0.0
        estimate = est1 - est2
        
        # Calculate SE
        if coef1 and coef2:
            se = np.sqrt(
                cov_matrix.loc[coef1, coef1] +
                cov_matrix.loc[coef2, coef2] -
                2 * cov_matrix.loc[coef1, coef2]
            )
        elif coef1:
            se = fitted_model.bse[coef1]
        elif coef2:
            se = fitted_model.bse[coef2]
        else:
            se = np.nan
        
        return estimate, se
    
    def _get_cell_mean_coef(self, factor_level, other_level, between_levels, within_levels,
                           factor_coeffs, params, interaction_coeffs, is_between=False) -> float:
        """Get the coefficient value for a specific cell mean (between_level, within_level combination)."""
        
        if is_between:
            # Comparing between levels at a specific within level
            between_level = factor_level
            within_level = other_level
            between_coef = factor_coeffs.get(between_level)
        else:
            # Comparing within levels at a specific between level
            between_level = factor_level
            within_level = other_level
            within_coef = factor_coeffs.get(within_level)
        
        # Intercept (reference cell)
        estimate = params['Intercept']
        
        # Add between effect if not reference
        if is_between:
            if between_coef:
                estimate += params[between_coef]
            # Add within effect if not reference
            within_coef = [v for k, v in factor_coeffs.items() if k == within_level]
            if within_coef and within_coef[0]:
                # Need to find within coefficient separately for between comparisons
                for k, v in interaction_coeffs.items():
                    if within_level in str(k):
                        estimate += params[v] if v else 0.0
                        break
        else:
            if within_coef:
                estimate += params[within_coef]
        
        # Add interaction if both are non-reference
        if is_between:
            interaction_key = (between_level, within_level)
        else:
            interaction_key = (between_level, within_level)
        
        interaction_coef = interaction_coeffs.get(interaction_key)
        if interaction_coef:
            estimate += params[interaction_coef]
        
        return estimate
    
    def _calculate_contrast_se(self, between1, within1, between2, within2,
                              between_levels, within_levels, factor_coeffs, interaction_coeffs,
                              cov_matrix, fitted_model, is_between=False) -> float:
        """Calculate SE for a contrast between two cells using covariance matrix."""
        
        # Collect all coefficients involved in the contrast
        coefs_cell1 = []
        coefs_cell2 = []
        
        # Add between coefficients
        if is_between:
            coef1_between = factor_coeffs.get(between1)
            coef2_between = factor_coeffs.get(between2)
            if coef1_between:
                coefs_cell1.append(coef1_between)
            if coef2_between:
                coefs_cell2.append(coef2_between)
        else:
            if between1 != between_levels[0]:
                # Find between coefficient
                for k, v in factor_coeffs.items():
                    if k == between1 and v:
                        coefs_cell1.append(v)
                        break
            if between2 != between_levels[0]:
                for k, v in factor_coeffs.items():
                    if k == between2 and v:
                        coefs_cell2.append(v)
                        break
        
        # Add within coefficients
        if not is_between:
            coef1_within = factor_coeffs.get(within1)
            coef2_within = factor_coeffs.get(within2)
            if coef1_within:
                coefs_cell1.append(coef1_within)
            if coef2_within:
                coefs_cell2.append(coef2_within)
        else:
            if within1 != within_levels[0]:
                # Find within coefficient
                for k, v in interaction_coeffs.items():
                    if within1 in str(k) and v:
                        coefs_cell1.append(v)
                        break
            if within2 != within_levels[0]:
                for k, v in interaction_coeffs.items():
                    if within2 in str(k) and v:
                        coefs_cell2.append(v)
                        break
        
        # Add interaction coefficients
        interaction_key1 = (between1, within1)
        interaction_key2 = (between2, within2)
        
        interaction_coef1 = interaction_coeffs.get(interaction_key1)
        interaction_coef2 = interaction_coeffs.get(interaction_key2)
        
        if interaction_coef1:
            coefs_cell1.append(interaction_coef1)
        if interaction_coef2:
            coefs_cell2.append(interaction_coef2)
        
        # Calculate variance of the contrast
        variance = 0.0
        
        # Variance from cell1 coefficients
        for coef in coefs_cell1:
            if coef in cov_matrix.index:
                variance += cov_matrix.loc[coef, coef]
        
        # Variance from cell2 coefficients
        for coef in coefs_cell2:
            if coef in cov_matrix.index:
                variance += cov_matrix.loc[coef, coef]
        
        # Covariance between cell1 and cell2 coefficients (subtract 2*)
        for coef1 in coefs_cell1:
            for coef2 in coefs_cell2:
                if coef1 in cov_matrix.index and coef2 in cov_matrix.columns:
                    variance -= 2 * cov_matrix.loc[coef1, coef2]
        
        # Covariance within cell1 coefficients (add 2*)
        for i, coef_i in enumerate(coefs_cell1):
            for coef_j in coefs_cell1[i+1:]:
                if coef_i in cov_matrix.index and coef_j in cov_matrix.columns:
                    variance += 2 * cov_matrix.loc[coef_i, coef_j]
        
        # Covariance within cell2 coefficients (add 2*)
        for i, coef_i in enumerate(coefs_cell2):
            for coef_j in coefs_cell2[i+1:]:
                if coef_i in cov_matrix.index and coef_j in cov_matrix.columns:
                    variance += 2 * cov_matrix.loc[coef_i, coef_j]
        
        se = np.sqrt(variance) if variance >= 0 else np.nan
        
        return se
    
    def _apply_multiple_comparison_correction(self, results: List[StatisticalResult],
                                             design: ExperimentalDesign) -> List[StatisticalResult]:
        """Apply FDR correction separately for each effect type within each measurement category."""
        
        categories = get_measurement_categories()
        
        for category, measurements in categories.items():
            effect_groups = {
                'Between': [],
                'Within': [],
                'Interaction': []
            }
            
            for result in results:
                if result.measurement in measurements:
                    if 'Between' in result.test_name:
                        effect_groups['Between'].append(result)
                    elif 'Within' in result.test_name:
                        effect_groups['Within'].append(result)
                    elif 'Interaction' in result.test_name:
                        effect_groups['Interaction'].append(result)
            
            # Apply FDR within each effect type
            for effect_type, effect_results in effect_groups.items():
                if len(effect_results) > 1:
                    p_values = [r.p_value for r in effect_results if not np.isnan(r.p_value)]
                    valid_indices = [i for i, r in enumerate(effect_results) if not np.isnan(r.p_value)]
                    
                    if len(p_values) > 1:
                        try:
                            rejected, corrected_p, _, _ = multi.multipletests(p_values, method="fdr_bh")
                            
                            for i, corrected_val in zip(valid_indices, corrected_p):
                                effect_results[i].corrected_p = corrected_val
                            
                            logger.info(f"Applied FDR to {len(p_values)} {effect_type} effects in {category}")
                        except Exception as e:
                            logger.error(f"Error applying FDR correction: {e}")
                    else:
                        # Only 1 valid p-value - no correction needed, use original p-value
                        for result in effect_results:
                            if not np.isnan(result.p_value):
                                result.corrected_p = result.p_value
                            else:
                                result.corrected_p = np.nan
                elif len(effect_results) == 1:
                    # Single result - no correction needed, use original p-value
                    if not np.isnan(effect_results[0].p_value):
                        effect_results[0].corrected_p = effect_results[0].p_value
                    else:
                        effect_results[0].corrected_p = np.nan
        
        return results
    
    def _apply_posthoc_correction(self, results: List[StatisticalResult]) -> List[StatisticalResult]:
        """Apply FDR correction to post-hoc tests separately for paired and independent tests within each measurement."""
        
        # Group by measurement AND test type (paired vs independent)
        measurement_test_groups = {}
        for result in results:
            # Create key: (measurement, test_type)
            test_type = "Paired" if "Paired" in result.test_name else "Independent"
            key = (result.measurement, test_type)
            
            if key not in measurement_test_groups:
                measurement_test_groups[key] = []
            measurement_test_groups[key].append(result)
        
        # Apply FDR correction within each measurement-test_type family
        for (measurement, test_type), results_list in measurement_test_groups.items():
            if len(results_list) > 1:
                p_values = [r.p_value for r in results_list if not np.isnan(r.p_value)]
                valid_indices = [i for i, r in enumerate(results_list) if not np.isnan(r.p_value)]
                
                if len(p_values) > 1:
                    try:
                        rejected, corrected_p, _, _ = multi.multipletests(p_values, method="fdr_bh")
                        
                        for i, corrected_val in zip(valid_indices, corrected_p):
                            results_list[i].corrected_p = corrected_val
                        
                        logger.info(f"Applied FDR to {len(p_values)} {test_type} post-hoc tests for {measurement}")
                    except Exception as e:
                        logger.error(f"Error applying post-hoc FDR correction: {e}")
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
        
        return results
    
    def save_results(self, results: List[StatisticalResult], output_path: str,
                    design: ExperimentalDesign = None, base_path: str = None) -> None:
        """Save mixed model results to CSV (matching independent factorial format)."""
        
        if not results:
            logger.warning("No results to save")
            return
        
        # Separate main effects and post-hoc results
        main_effects = [r for r in results if "Mixed ANOVA -" in r.test_name]
        posthoc_results = [r for r in results if "t-test" in r.test_name]
        
        # Get all groups
        if design and design.groups:
            all_groups = sorted([g.name for g in design.groups])
        else:
            all_groups = []
        
        # Load raw group data if provided
        raw_group_data = {}
        if design and base_path:
            for group in design.groups:
                combined_data = self._load_combined_data(group.name, base_path)
                if not combined_data.empty:
                    raw_group_data[group.name] = clean_dataframe(combined_data)
        
        # Determine all possible post-hoc comparisons
        all_posthoc_comparisons = set()
        for result in posthoc_results:
            comparison = f"{result.group1_name} vs {result.group2_name}"
            all_posthoc_comparisons.add(comparison)
        all_posthoc_comparisons = sorted(list(all_posthoc_comparisons))
        
        # Build combined data: one row per measurement
        combined_data = []
        
        # Get unique measurements from main effects
        measurements = set()
        for result in main_effects:
            if "Interaction" in result.test_name:  # Use interaction as the key
                measurements.add(result.measurement)
        
        for measurement in sorted(measurements):
            # Start row with measurement info
            row = {
                "Measurement": measurement,
                "MeasurementType": None
            }
            
            # Get the three effects for this measurement
            between_result = next((r for r in main_effects 
                                  if r.measurement == measurement and "Between" in r.test_name), None)
            within_result = next((r for r in main_effects 
                                  if r.measurement == measurement and "Within" in r.test_name), None)
            interaction_result = next((r for r in main_effects 
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
                                import math
                                group_stats[group_name] = {
                                    'mean': data.mean(),
                                    'stderr': data.std() / math.sqrt(len(data)) if len(data) > 1 else 0,
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
            
            # Add effect p-values (use factor names from design)
            if between_result:
                row[f"{design.between_factor_name}_p-value"] = between_result.p_value
                row[f"{design.between_factor_name}_corrected_p"] = between_result.corrected_p
            else:
                row[f"{design.between_factor_name}_p-value"] = np.nan
                row[f"{design.between_factor_name}_corrected_p"] = np.nan
            
            if within_result:
                row[f"{design.within_factor_name}_p-value"] = within_result.p_value
                row[f"{design.within_factor_name}_corrected_p"] = within_result.corrected_p
            else:
                row[f"{design.within_factor_name}_p-value"] = np.nan
                row[f"{design.within_factor_name}_corrected_p"] = np.nan
            
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
            
            # Fill in post-hoc p-values if they exist
            posthoc_for_measurement = [r for r in posthoc_results if r.measurement == measurement]
            for ph_result in posthoc_for_measurement:
                comparison = f"{ph_result.group1_name} vs {ph_result.group2_name}"
                row[f"{comparison}_p-value"] = ph_result.p_value
                row[f"{comparison}_corrected_p"] = ph_result.corrected_p
            
            combined_data.append(row)
        
        # Create DataFrame and organize columns
        combined_df = pd.DataFrame(combined_data)
        
        # Order columns: Measurement, MeasurementType, groups, effects, post-hoc
        ordered_columns = ["Measurement", "MeasurementType"]
        
        # Add group columns in order
        for group in all_groups:
            ordered_columns.extend([f"{group}_mean", f"{group}_stderr", f"{group}_n"])
        
        # Add effect columns
        ordered_columns.extend([
            f"{design.between_factor_name}_p-value", f"{design.between_factor_name}_corrected_p",
            f"{design.within_factor_name}_p-value", f"{design.within_factor_name}_corrected_p",
            "Interaction_p-value", "Interaction_corrected_p"
        ])
        
        # Add post-hoc columns
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
        if main_effects:
            significant_interactions = [r for r in main_effects 
                                       if "Interaction" in r.test_name 
                                       and (r.corrected_p or r.p_value) < 0.05]
            logger.info(f"Mixed model Summary: {len(significant_interactions)}/{len(measurements)} interactions significant (p < 0.05)")
        
        if posthoc_results:
            significant_posthoc = [r for r in posthoc_results if (r.corrected_p or r.p_value) < 0.05]
            logger.info(f"Post-hoc Summary: {len(significant_posthoc)}/{len(posthoc_results)} comparisons significant (p < 0.05)")

