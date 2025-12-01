"""
Utilities for determining when and which post-hoc comparisons to run in factorial designs.

This module implements correct statistical practice for N×M factorial ANOVA post-hoc testing:
- Main effects with 2 levels: No post-hoc needed (the main effect IS the comparison)
- Main effects with 3+ levels: Run pairwise comparisons for that factor
- Significant interaction: Run simple effects (comparisons where one factor is held constant)
"""

import logging
import math
from itertools import combinations
from typing import Dict, List, Set, Tuple, Optional

import numpy as np
import pandas as pd
from patsy import build_design_matrices
import statsmodels.stats.multitest as multi

logger = logging.getLogger(__name__)


def count_factor_levels(design, group_names: List[str]) -> Tuple[int, int]:
    """
    Count unique levels for each factor in a factorial design.
    
    Args:
        design: ExperimentalDesign object with factor_mapping
        group_names: List of group names
        
    Returns:
        (n_factor1_levels, n_factor2_levels)
    """
    factor1_levels = set()
    factor2_levels = set()
    
    for group_name in group_names:
        mapping = design.factor_mapping.get(group_name)
        if mapping:
            factor1_levels.add(mapping['factor1'])
            factor2_levels.add(mapping['factor2'])
    
    return len(factor1_levels), len(factor2_levels)


def should_run_posthocs(anova_result: Dict, design, group_names: List[str]) -> Dict:
    """
    Determine if post-hocs are needed based on:
    - Which effects are significant (corrected p < 0.05)
    - Number of levels in each factor
    
    Post-hoc logic:
    - Main effect of Factor with 2 levels: No post-hoc (effect IS the comparison)
    - Main effect of Factor with 3+ levels: Run pairwise comparisons within that factor
    - Interaction significant: Run simple effects (one factor held constant)
    
    Args:
        anova_result: Dictionary with corrected p-values for each effect
        design: ExperimentalDesign object
        group_names: List of group names
        
    Returns:
        Dictionary with:
            - 'run_posthocs': bool - whether any post-hocs should be run
            - 'reasons': List[str] - which effects require post-hocs
                ('factor1_main', 'factor2_main', 'interaction')
    """
    n_f1, n_f2 = count_factor_levels(design, group_names)
    
    # Check corrected p-values (try both naming conventions)
    f1_key = f'{design.factor1_name}_corrected_p'
    f2_key = f'{design.factor2_name}_corrected_p'
    
    # For mixed models, might use 'Between_corrected_p', 'Within_corrected_p'
    if hasattr(design, 'between_factor_name') and hasattr(design, 'within_factor_name'):
        # Try between/within naming
        f1_key_alt = f'{design.between_factor_name}_corrected_p'
        f2_key_alt = f'{design.within_factor_name}_corrected_p'
        f1_sig = anova_result.get(f1_key_alt, anova_result.get(f1_key, 1)) < 0.05
        f2_sig = anova_result.get(f2_key_alt, anova_result.get(f2_key, 1)) < 0.05
    else:
        f1_sig = anova_result.get(f1_key, 1) < 0.05
        f2_sig = anova_result.get(f2_key, 1) < 0.05
    
    int_sig = anova_result.get('Interaction_corrected_p', 1) < 0.05
    
    # Determine if post-hocs needed
    reasons = []
    if f1_sig and n_f1 >= 3:
        reasons.append('factor1_main')
    if f2_sig and n_f2 >= 3:
        reasons.append('factor2_main')
    if int_sig:
        reasons.append('interaction')
    
    return {
        'run_posthocs': len(reasons) > 0,
        'reasons': reasons
    }


def get_simple_effect_comparisons(design, group_names: List[str]) -> List[Tuple[str, str]]:
    """
    Generate only logical simple effects comparisons for factorial designs.
    
    Simple effects = comparisons where ONE factor is held constant.
    This excludes diagonal comparisons where both factors change.
    
    For a 2×2 design (WT/Scn1a × Heat/NoHeat):
    - WT Heat vs WT NoHeat (Factor1=WT held constant)
    - Scn1a Heat vs Scn1a NoHeat (Factor1=Scn1a held constant)
    - WT Heat vs Scn1a Heat (Factor2=Heat held constant)
    - WT NoHeat vs Scn1a NoHeat (Factor2=NoHeat held constant)
    
    Excludes: WT Heat vs Scn1a NoHeat (diagonal - both factors change)
    
    Args:
        design: ExperimentalDesign object with factor_mapping
        group_names: List of group names to compare
        
    Returns:
        List of (group1_name, group2_name) tuples for logical comparisons
    """
    # Get all factor levels
    factor1_levels = set()
    factor2_levels = set()
    
    for group_name in group_names:
        mapping = design.factor_mapping.get(group_name)
        if mapping:
            factor1_levels.add(mapping['factor1'])
            factor2_levels.add(mapping['factor2'])
    
    comparisons = []
    
    # For each level of Factor1, compare all levels of Factor2 (Factor1 held constant)
    for f1_level in factor1_levels:
        groups_at_f1 = [g for g in group_names 
                       if design.factor_mapping.get(g, {}).get('factor1') == f1_level]
        
        for g1, g2 in combinations(sorted(groups_at_f1), 2):
            comparisons.append((g1, g2))
    
    # For each level of Factor2, compare all levels of Factor1 (Factor2 held constant)
    for f2_level in factor2_levels:
        groups_at_f2 = [g for g in group_names 
                       if design.factor_mapping.get(g, {}).get('factor2') == f2_level]
        
        for g1, g2 in combinations(sorted(groups_at_f2), 2):
            comparisons.append((g1, g2))
    
    # Remove duplicates (keep order for reproducibility)
    seen = set()
    unique_comparisons = []
    for comp in comparisons:
        # Normalize comparison order for deduplication
        normalized = tuple(sorted(comp))
        if normalized not in seen:
            seen.add(normalized)
            unique_comparisons.append(comp)
    
    logger.info(f"Generated {len(unique_comparisons)} simple effects comparisons "
                f"({len(factor1_levels)} Factor1 levels × {len(factor2_levels)} Factor2 levels)")
    
    return unique_comparisons


def get_simple_effect_comparisons_dependent(design, between_levels: List[str], 
                                           within_levels: List[str]) -> Dict[str, List[Tuple]]:
    """
    Generate logical simple effects comparisons for dependent (mixed) factorial designs.
    
    Returns separate lists for paired and independent comparisons:
    - Paired: Within-factor comparisons at each between-level (same subjects)
    - Independent: Between-factor comparisons at each within-level (different subjects)
    
    Args:
        design: ExperimentalDesign object
        between_levels: List of between-subject factor levels
        within_levels: List of within-subject factor levels
        
    Returns:
        Dictionary with:
            - 'paired': List of (between_level, within1, within2) tuples
            - 'independent': List of (within_level, between1, between2) tuples
    """
    paired_comparisons = []
    independent_comparisons = []
    
    # Within-factor comparisons at each between-level (PAIRED)
    # E.g., for each genotype, compare heat vs no heat
    for between_level in between_levels:
        for w1, w2 in combinations(sorted(within_levels), 2):
            paired_comparisons.append((between_level, w1, w2))
    
    # Between-factor comparisons at each within-level (INDEPENDENT)
    # E.g., for each temperature, compare WT vs Scn1a
    for within_level in within_levels:
        for b1, b2 in combinations(sorted(between_levels), 2):
            independent_comparisons.append((within_level, b1, b2))
    
    logger.info(f"Generated {len(paired_comparisons)} paired comparisons and "
                f"{len(independent_comparisons)} independent comparisons for mixed design")
    
    return {
        'paired': paired_comparisons,
        'independent': independent_comparisons
    }


def compute_lsmean_summaries(
    cell_stats: Dict[Tuple[str, str], Dict[str, float]],
    target_levels: List[str],
    other_levels: List[str],
    mse: float,
    orientation: str,
) -> Dict[str, Dict[str, float]]:
    """
    Compute least-squares mean (marginal mean) summaries for a factor in a 2-way ANOVA.

    Args:
        cell_stats: Mapping {(factor1_level, factor2_level): {'mean': ..., 'count': ...}}
        target_levels: Levels of the factor being compared (e.g., factor1 levels)
        other_levels: Levels of the other factor (held constant)
        mse: Mean squared error from the ANOVA model
        orientation: 'factor1' if target_levels correspond to factor1, else 'factor2'

    Returns:
        Dict[level] = {
            'mean': LS-mean value,
            'variance': variance of LS-mean,
            'count': total observations contributing to the LS-mean
        }
    """
    if not cell_stats or not target_levels or not other_levels or mse is None or mse <= 0:
        return {}

    weight = 1.0 / len(other_levels)
    summaries: Dict[str, Dict[str, float]] = {}

    for target in target_levels:
        contributions = []
        for other in other_levels:
            key = (target, other) if orientation == 'factor1' else (other, target)
            stats = cell_stats.get(key)
            if not stats or stats.get('count', 0) <= 0:
                contributions = []
                break
            contributions.append(stats)

        if not contributions:
            continue

        mean_value = sum(entry['mean'] for entry in contributions) * weight
        variance = mse * sum((weight ** 2) / entry['count'] for entry in contributions)
        if not math.isfinite(variance):
            continue

        summaries[target] = {
            'mean': mean_value,
            'variance': variance,
            'count': sum(entry['count'] for entry in contributions)
        }

    return summaries


def compute_global_curve_posthocs(
    model,
    step_col_z: str,
    analysis_label: str,
    factor_a_col: str,
    factor_b_col: Optional[str],
    factor_a_label: str,
    factor_b_label: Optional[str],
    factor_a_levels: List[str],
    factor_b_levels: Optional[List[str]],
    main_a_p: Optional[float],
    main_b_p: Optional[float],
    interaction_ab_p: Optional[float] = None,
    interaction_a_step_p: Optional[float] = None,
    interaction_b_step_p: Optional[float] = None,
    interaction_three_way_p: Optional[float] = None,
    alpha: float = 0.05,
) -> List[Dict]:
    """Run Wald-test contrasts for global mixed models (entire-curve comparisons)."""
    factor_a_levels = factor_a_levels or []
    factor_b_levels = factor_b_levels or []
    has_factor_b = factor_b_col is not None and len(factor_b_levels) >= 1
    has_factor_b_two_levels = factor_b_col is not None and len(factor_b_levels) >= 2
    single_factor = factor_b_col is None
    collapse_levels_for_a = factor_b_levels if has_factor_b else [None]
    design_info = getattr(getattr(model.model.data, 'orig_exog', None), 'design_info', None)
    if design_info is None:
        logger.debug("Design information unavailable for global mixed-effects post-hocs")
        return []

    if len(factor_a_levels) < 2 and len(factor_b_levels) < 2:
        return []
    if not has_factor_b and len(factor_a_levels) <= 2:
        return []

    fe_names = list(model.fe_params.index)
    param_names = list(model.params.index)
    try:
        fe_param_indices = [param_names.index(name) for name in fe_names]
    except ValueError as exc:
        logger.warning(f"Could not align fixed-effect parameters for global post-hocs: {exc}")
        return []

    fe_beta = model.fe_params.values
    total_params = len(model.params)

    def _is_sig(p_val: Optional[float]) -> bool:
        return p_val is not None and not np.isnan(p_val) and p_val < alpha

    interaction_sig = any(
        _is_sig(p_val)
        for p_val in [interaction_ab_p, interaction_a_step_p, interaction_b_step_p, interaction_three_way_p]
    )
    factor_a_sig = _is_sig(main_a_p)
    factor_b_sig = _is_sig(main_b_p)

    posthocs: List[Dict] = []
    if interaction_sig and len(factor_a_levels) >= 2:
        if has_factor_b_two_levels:
            posthocs.extend(
                _build_global_simple_effects(
                    model,
                    design_info,
                    step_col_z,
                    factor_a_col,
                    factor_b_col,
                    factor_a_label,
                    factor_b_label,
                    factor_a_levels,
                    factor_b_levels,
                    fe_param_indices,
                    fe_beta,
                    total_params,
                    analysis_label,
                )
            )
        elif single_factor:
            posthocs.extend(
                _build_global_marginal_effects(
                    model,
                    design_info,
                    step_col_z,
                    factor_key='a',
                    factor_label=factor_a_label,
                    other_label=factor_b_label,
                    target_levels=factor_a_levels,
                    collapse_levels=collapse_levels_for_a,
                    factor_a_col=factor_a_col,
                    factor_b_col=factor_b_col,
                    fe_param_indices=fe_param_indices,
                    fe_beta=fe_beta,
                    total_params=total_params,
                    analysis_label=analysis_label,
                    posthoc_mode='Interaction',
                )
            )
    else:
        if factor_a_sig and len(factor_a_levels) >= 2:
            posthocs.extend(
                _build_global_marginal_effects(
                    model,
                    design_info,
                    step_col_z,
                    factor_key='a',
                    factor_label=factor_a_label,
                    other_label=factor_b_label,
                    target_levels=factor_a_levels,
                    collapse_levels=collapse_levels_for_a,
                    factor_a_col=factor_a_col,
                    factor_b_col=factor_b_col,
                    fe_param_indices=fe_param_indices,
                    fe_beta=fe_beta,
                    total_params=total_params,
                    analysis_label=analysis_label,
                )
            )
        if has_factor_b_two_levels and factor_b_sig and len(factor_a_levels) >= 1:
            posthocs.extend(
                _build_global_marginal_effects(
                    model,
                    design_info,
                    step_col_z,
                    factor_key='b',
                    factor_label=factor_b_label,
                    other_label=factor_a_label,
                    target_levels=factor_b_levels,
                    collapse_levels=factor_a_levels,
                    factor_a_col=factor_a_col,
                    factor_b_col=factor_b_col,
                    fe_param_indices=fe_param_indices,
                    fe_beta=fe_beta,
                    total_params=total_params,
                    analysis_label=analysis_label,
                )
            )

    return _apply_global_posthoc_fdr(posthocs)


def _build_curve_rows(design_info, step_col_z: str, factor_a_col: str, factor_b_col: str,
                      level_a: str, level_b: str) -> Dict[str, np.ndarray]:
    row_dict = {
        factor_a_col: level_a,
        step_col_z: 0.0
    }
    if factor_b_col is not None:
        row_dict[factor_b_col] = level_b
    data_zero = pd.DataFrame([row_dict])
    data_one = data_zero.copy()
    data_one[step_col_z] = 1.0

    row_zero = np.asarray(build_design_matrices([design_info], data_zero)[0])[0]
    row_one = np.asarray(build_design_matrices([design_info], data_one)[0])[0]
    return {'z0': row_zero, 'z1': row_one}


def _embed_fe_row(row: np.ndarray, fe_param_indices: List[int], total_params: int) -> np.ndarray:
    full = np.zeros(total_params)
    for value, idx in zip(row, fe_param_indices):
        full[idx] = value
    return full


def _evaluate_curve_contrast(
    model,
    curve_a: Dict[str, np.ndarray],
    curve_b: Dict[str, np.ndarray],
    fe_param_indices: List[int],
    fe_beta: np.ndarray,
    total_params: int,
    description: str = "",
) -> Dict:
    intercept_fe = curve_a['z0'] - curve_b['z0']
    slope_fe = (curve_a['z1'] - curve_a['z0']) - (curve_b['z1'] - curve_b['z0'])

    if np.allclose(intercept_fe, 0) and np.allclose(slope_fe, 0):
        return {}

    R = np.vstack([
        _embed_fe_row(intercept_fe, fe_param_indices, total_params),
        _embed_fe_row(slope_fe, fe_param_indices, total_params),
    ])

    try:
        wald_res = model.wald_test(R, scalar=True)
        chi2_stat = float(np.asarray(wald_res.statistic).squeeze())
        p_value = float(np.asarray(wald_res.pvalue).squeeze())
    except Exception as exc:
        logger.debug(f"Wald test failed for {description}: {exc}")
        return {}

    delta_mean = float(intercept_fe @ fe_beta)
    delta_slope = float(slope_fe @ fe_beta)

    return {
        'Delta_at_mean': delta_mean,
        'Delta_per_SD': delta_slope,
        'chi2': chi2_stat,
        'df': R.shape[0],
        'p_value': p_value,
    }


def _build_global_simple_effects(
    model,
    design_info,
    step_col_z: str,
    factor_a_col: str,
    factor_b_col: str,
    factor_a_label: str,
    factor_b_label: str,
    factor_a_levels: List[str],
    factor_b_levels: List[str],
    fe_param_indices: List[int],
    fe_beta: np.ndarray,
    total_params: int,
    analysis_label: str,
) -> List[Dict]:
    results: List[Dict] = []
    label_a = factor_a_label or "Factor A"
    label_b = factor_b_label or "Factor B"

    for level_b in factor_b_levels:
        for level_a1, level_a2 in combinations(factor_a_levels, 2):
            try:
                curve_a = _build_curve_rows(design_info, step_col_z, factor_a_col, factor_b_col, level_a1, level_b)
                curve_b = _build_curve_rows(design_info, step_col_z, factor_a_col, factor_b_col, level_a2, level_b)
            except Exception as exc:
                logger.debug(f"Failed to build curve for {label_b}={level_b} simple effect: {exc}")
                continue

            contrast = _evaluate_curve_contrast(
                model,
                curve_a,
                curve_b,
                fe_param_indices,
                fe_beta,
                total_params,
                description=f"{label_b}={level_b}: {level_a1} vs {level_a2}",
            )
            if not contrast:
                continue

            contrast.update({
                'Analysis': analysis_label,
                'Posthoc_Mode': 'Simple',
                'Factor': label_a,
                'Context': f"{label_b}={level_b}",
                'Comparison': f"{level_a1} vs {level_a2}",
                'Level1': level_a1,
                'Level2': level_a2,
                'family_id': f"{analysis_label}_interaction_{label_b}_{level_b}_{label_a}",
            })
            results.append(contrast)

    for level_a in factor_a_levels:
        for level_b1, level_b2 in combinations(factor_b_levels, 2):
            try:
                curve_a = _build_curve_rows(design_info, step_col_z, factor_a_col, factor_b_col, level_a, level_b1)
                curve_b = _build_curve_rows(design_info, step_col_z, factor_a_col, factor_b_col, level_a, level_b2)
            except Exception as exc:
                logger.debug(f"Failed to build curve for {label_a}={level_a} simple effect: {exc}")
                continue

            contrast = _evaluate_curve_contrast(
                model,
                curve_a,
                curve_b,
                fe_param_indices,
                fe_beta,
                total_params,
                description=f"{label_a}={level_a}: {level_b1} vs {level_b2}",
            )
            if not contrast:
                continue

            contrast.update({
                'Analysis': analysis_label,
                'Posthoc_Mode': 'Simple',
                'Factor': label_b,
                'Context': f"{label_a}={level_a}",
                'Comparison': f"{level_b1} vs {level_b2}",
                'Level1': level_b1,
                'Level2': level_b2,
                'family_id': f"{analysis_label}_interaction_{label_a}_{level_a}_{label_b}",
            })
            results.append(contrast)

    return results


def _build_global_marginal_effects(
    model,
    design_info,
    step_col_z: str,
    factor_key: str,
    factor_label: str,
    other_label: str,
    target_levels: List[str],
    collapse_levels: List[str],
    factor_a_col: str,
    factor_b_col: str,
    fe_param_indices: List[int],
    fe_beta: np.ndarray,
    total_params: int,
    analysis_label: str,
    posthoc_mode: str = 'Marginal',
) -> List[Dict]:
    if not collapse_levels:
        logger.debug(f"No levels available to average across for {factor_label or factor_key} marginal contrasts")
        return []

    results: List[Dict] = []
    label_self = factor_label or ("Factor A" if factor_key == 'a' else "Factor B")
    label_other = other_label or ("Factor B" if factor_key == 'a' else "Factor A")

    for level_a, level_b in combinations(target_levels, 2):
        curves_self_a = []
        curves_self_b = []
        try:
            for other_level in collapse_levels:
                if factor_key == 'a':
                    curves_self_a.append(_build_curve_rows(design_info, step_col_z, factor_a_col, factor_b_col, level_a, other_level))
                    curves_self_b.append(_build_curve_rows(design_info, step_col_z, factor_a_col, factor_b_col, level_b, other_level))
                else:
                    curves_self_a.append(_build_curve_rows(design_info, step_col_z, factor_a_col, factor_b_col, other_level, level_a))
                    curves_self_b.append(_build_curve_rows(design_info, step_col_z, factor_a_col, factor_b_col, other_level, level_b))
        except Exception as exc:
            logger.debug(f"Failed to build marginal curves for {label_self}: {level_a} vs {level_b}: {exc}")
            continue

        if not curves_self_a or not curves_self_b:
            continue

        avg_curve_a = {
            'z0': np.mean([curve['z0'] for curve in curves_self_a], axis=0),
            'z1': np.mean([curve['z1'] for curve in curves_self_a], axis=0),
        }
        avg_curve_b = {
            'z0': np.mean([curve['z0'] for curve in curves_self_b], axis=0),
            'z1': np.mean([curve['z1'] for curve in curves_self_b], axis=0),
        }

        contrast = _evaluate_curve_contrast(
            model,
            avg_curve_a,
            avg_curve_b,
            fe_param_indices,
            fe_beta,
            total_params,
            description=f"marginal {label_self}: {level_a} vs {level_b}",
        )
        if not contrast:
            continue

        family_id = f"{analysis_label}_marginal_{label_self.replace(' ', '_')}"
        contrast.update({
            'Analysis': analysis_label,
            'Posthoc_Mode': posthoc_mode,
            'Factor': label_self,
            'Context': f"Averaged over {label_other}",
            'Comparison': f"{level_a} vs {level_b}",
            'Level1': level_a,
            'Level2': level_b,
            'family_id': family_id,
        })
        results.append(contrast)

    return results


def _apply_global_posthoc_fdr(posthocs: List[Dict]) -> List[Dict]:
    if not posthocs:
        return posthocs

    families: Dict[str, List[int]] = {}
    for idx, item in enumerate(posthocs):
        family_id = item.get('family_id')
        if not family_id:
            continue
        families.setdefault(family_id, []).append(idx)

    for family_id, indices in families.items():
        pvals = [
            posthocs[idx]['p_value']
            for idx in indices
            if not np.isnan(posthocs[idx].get('p_value', np.nan))
        ]
        valid_indices = [
            idx for idx in indices
            if not np.isnan(posthocs[idx].get('p_value', np.nan))
        ]

        if len(pvals) > 1:
            try:
                _, corrected, _, _ = multi.multipletests(pvals, method="fdr_bh")
                for idx, corr_val in zip(valid_indices, corrected):
                    posthocs[idx]['corrected_p'] = corr_val
                for idx in indices:
                    if idx not in valid_indices:
                        posthocs[idx]['corrected_p'] = np.nan
            except Exception as exc:
                logger.error(f"Global post-hoc FDR failed for {family_id}: {exc}")
        else:
            for idx in indices:
                raw_p = posthocs[idx].get('p_value', np.nan)
                posthocs[idx]['corrected_p'] = raw_p if not np.isnan(raw_p) else np.nan

    for item in posthocs:
        item.pop('family_id', None)

    return posthocs

