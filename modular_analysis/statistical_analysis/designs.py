"""
Experimental design definitions and management.
"""

from typing import List, Dict, Optional
import pandas as pd
import os
import logging
from ..shared.data_models import ExperimentalDesign, DesignType, GroupInfo
from ..shared.utils import convert_manifest_wide_to_long, validate_manifest

logger = logging.getLogger(__name__)


class DesignManager:
    """Manages experimental design creation and validation."""
    
    @staticmethod
    def create_two_group_independent(group1: GroupInfo, group2: GroupInfo, 
                                   name: str = "") -> ExperimentalDesign:
        """Create a two-group independent design (A vs B)."""
        return ExperimentalDesign(
            design_type=DesignType.INDEPENDENT_TWO_GROUP,
            groups=[group1, group2],
            name=name or f"{group1.name} vs {group2.name}",
            description=f"Independent comparison between {group1.name} and {group2.name}"
        )
    
    @staticmethod
    def create_multi_group_independent(groups: List[GroupInfo], name: str = "") -> ExperimentalDesign:
        """Create a multi-group independent design (A vs B vs C vs ...)."""
        if len(groups) < 3:
            raise ValueError("Multi-group design requires at least 3 groups")
            
        group_names = [g.name for g in groups]
        return ExperimentalDesign(
            design_type=DesignType.INDEPENDENT_MULTI_GROUP,
            groups=groups,
            name=name or " vs ".join(group_names),
            description=f"Independent comparison between {', '.join(group_names)}"
        )
    
    @staticmethod
    def create_factorial_2x2(groups: List[GroupInfo], factor1_name: str, factor2_name: str,
                            factor_mapping: Dict[str, Dict[str, str]], name: str = "",
                            level_italic: Dict[str, bool] = None) -> ExperimentalDesign:
        """Create an N×M factorial design (2 factors with N and M levels).
        
        Note: Method name kept as 'create_factorial_2x2' for backwards compatibility,
        but now supports any N×M design (2×2, 2×3, 3×3, etc.).
        """
        # Validate factor mapping exists and matches group count
        if not factor_mapping or len(factor_mapping) != len(groups):
            raise ValueError(f"Factor mapping must specify factor levels for all {len(groups)} groups")
        
        for group in groups:
            if group.name not in factor_mapping:
                raise ValueError(f"Factor mapping missing for group {group.name}")
            if 'factor1' not in factor_mapping[group.name] or 'factor2' not in factor_mapping[group.name]:
                raise ValueError(f"Factor mapping for {group.name} must include both factor1 and factor2")
        
        # Determine number of levels for each factor
        factor1_levels = set(factor_mapping[g.name]['factor1'] for g in groups)
        factor2_levels = set(factor_mapping[g.name]['factor2'] for g in groups)
        
        # Validate at least 2 levels per factor
        if len(factor1_levels) < 2:
            raise ValueError(f"Factor1 ({factor1_name}) must have at least 2 levels (found {len(factor1_levels)})")
        if len(factor2_levels) < 2:
            raise ValueError(f"Factor2 ({factor2_name}) must have at least 2 levels (found {len(factor2_levels)})")
        
        # Validate group count matches factor levels
        expected_groups = len(factor1_levels) * len(factor2_levels)
        if len(groups) != expected_groups:
            raise ValueError(
                f"{len(factor1_levels)}×{len(factor2_levels)} factorial design requires {expected_groups} groups "
                f"(found {len(groups)})"
            )
        
        group_names = [g.name for g in groups]
        design_notation = f"{len(factor1_levels)}×{len(factor2_levels)}"
        
        return ExperimentalDesign(
            design_type=DesignType.FACTORIAL_2X2,
            groups=groups,
            factor1_name=factor1_name,
            factor2_name=factor2_name,
            factor_mapping=factor_mapping,
            level_italic=level_italic or {},
            name=name or f"{design_notation} Factorial: {factor1_name} × {factor2_name}",
            description=f"{design_notation} factorial design with factors {factor1_name} and {factor2_name}"
        )
    
    @staticmethod
    def create_mixed_factorial(groups: List[GroupInfo], manifest_path: str,
                               between_factor_name: str, within_factor_name: str,
                               base_path: str = "", name: str = "",
                               level_italic: Dict[str, bool] = None) -> ExperimentalDesign:
        """
        Create a mixed factorial design (between × within factors).
        
        Args:
            groups: List of selected group folders (e.g., ["32 WT", "37 WT", "42 WT", "32 Scn1a", ...])
            manifest_path: Path to Excel manifest file
            between_factor_name: Name of between-subjects factor (e.g., "Genotype")
            within_factor_name: Name of within-subjects factor (e.g., "Temperature")
            base_path: Base directory path for validation (optional)
            name: Custom design name (optional)
            level_italic: Dictionary mapping factor levels to italic boolean (optional)
            
        Returns:
            ExperimentalDesign with MIXED_FACTORIAL type
        """
        if level_italic is None:
            level_italic = {}
        # Load and convert manifest
        df_wide = pd.read_excel(manifest_path)
        df_long = convert_manifest_wide_to_long(df_wide)
        
        # Validate manifest
        if base_path:
            is_valid, errors = validate_manifest(df_long, base_path)
            if not is_valid:
                error_msg = "Manifest validation failed:\n" + "\n".join(errors)
                raise ValueError(error_msg)
        
        # Extract unique levels from manifest
        between_levels = sorted(df_long['Group'].unique())
        within_levels = sorted(df_long['Condition'].unique())
        
        # Validate at least 2 levels per factor
        if len(between_levels) < 2:
            raise ValueError(f"{between_factor_name} must have at least 2 levels (found {len(between_levels)})")
        if len(within_levels) < 2:
            raise ValueError(f"{within_factor_name} must have at least 2 levels (found {len(within_levels)})")
        
        # Validate group count matches factor structure
        expected_groups = len(between_levels) * len(within_levels)
        if len(groups) != expected_groups:
            raise ValueError(
                f"{len(between_levels)}×{len(within_levels)} mixed design requires {expected_groups} groups "
                f"(found {len(groups)})"
            )
        
        # Create design notation
        design_notation = f"{len(between_levels)}×{len(within_levels)}"
        
        # Create factor_mapping by parsing group folder names
        # Group folders are named like "32 WT", "32_2 Scn1a", "37 Scn1a", etc.
        # We need to map each folder to its between (Group) and within (Condition) levels
        factor_mapping = {}
        for group in groups:
            # Try to match this group folder to manifest entries using exact matching
            # This avoids issues with underscores/spaces in factor level names
            matched = False
            for between_level in between_levels:
                for within_level in within_levels:
                    # Try: {Condition}_{Group}
                    if group.name == f"{within_level}_{between_level}":
                        factor_mapping[group.name] = {
                            'factor1': between_level,   # Between factor (Group)
                            'factor2': within_level     # Within factor (Condition)
                        }
                        matched = True
                        break
                    # Try: {Condition} {Group}
                    if group.name == f"{within_level} {between_level}":
                        factor_mapping[group.name] = {
                            'factor1': between_level,   # Between factor (Group)
                            'factor2': within_level     # Within factor (Condition)
                        }
                        matched = True
                        break
                if matched:
                    break
            
            if not matched:
                raise ValueError(
                    f"Could not determine factor levels for group '{group.name}'. "
                    f"Expected format: {{Condition}}_{{Group}} or {{Condition}} {{Group}} "
                    f"where Group is one of {between_levels} and Condition is one of {within_levels}. "
                    f"Examples: '{within_levels[0]}_{between_levels[0]}' or '{within_levels[0]} {between_levels[0]}'"
                )
        
        return ExperimentalDesign(
            design_type=DesignType.MIXED_FACTORIAL,
            groups=groups,
            factor1_name=between_factor_name,  # Store as factor1 for consistency
            factor2_name=within_factor_name,   # Store as factor2 for consistency
            factor_mapping=factor_mapping,      # Add factor mapping for plotting
            level_italic=level_italic,          # Use provided italic mapping
            between_factor_name=between_factor_name,
            within_factor_name=within_factor_name,
            pairing_manifest=df_long,
            name=name or f"{design_notation} Mixed Factorial: {between_factor_name} × {within_factor_name}",
            description=f"{design_notation} mixed factorial design: "
                       f"{between_factor_name} (between-subjects) × {within_factor_name} (within-subjects)"
        )
    
    @staticmethod
    def validate_design(design: ExperimentalDesign) -> List[str]:
        """Validate an experimental design and return any error messages."""
        errors = []
        
        if not design.groups:
            errors.append("Design must have at least one group")
            
        if design.design_type == DesignType.INDEPENDENT_TWO_GROUP:
            if len(design.groups) != 2:
                errors.append("Two-group design must have exactly 2 groups")
        elif design.design_type == DesignType.INDEPENDENT_MULTI_GROUP:
            if len(design.groups) < 3:
                errors.append("Multi-group design must have at least 3 groups")
        elif design.design_type == DesignType.FACTORIAL_2X2:
            if not design.factor1_name or not design.factor2_name:
                errors.append("Factorial design must specify both factor names")
            if not design.factor_mapping:
                errors.append("Factorial design must include factor mapping")
            else:
                # Validate factorial structure
                factor1_levels = set(design.factor_mapping[g.name]['factor1'] for g in design.groups if g.name in design.factor_mapping)
                factor2_levels = set(design.factor_mapping[g.name]['factor2'] for g in design.groups if g.name in design.factor_mapping)
                
                if len(factor1_levels) < 2:
                    errors.append(f"Factor1 must have at least 2 levels (found {len(factor1_levels)})")
                if len(factor2_levels) < 2:
                    errors.append(f"Factor2 must have at least 2 levels (found {len(factor2_levels)})")
                
                expected_groups = len(factor1_levels) * len(factor2_levels)
                if len(design.groups) != expected_groups:
                    errors.append(
                        f"{len(factor1_levels)}×{len(factor2_levels)} factorial design requires "
                        f"{expected_groups} groups (found {len(design.groups)})"
                    )
        elif design.design_type == DesignType.MIXED_FACTORIAL:
            if not design.between_factor_name or not design.within_factor_name:
                errors.append("Mixed factorial design must specify both between and within factor names")
            if design.pairing_manifest is None or design.pairing_manifest.empty:
                errors.append("Mixed factorial design must include pairing manifest")
            else:
                # Validate manifest structure
                required_cols = ['Subject_ID', 'Group', 'Condition', 'Filename']
                manifest_cols = design.pairing_manifest.columns.tolist()
                missing_cols = [c for c in required_cols if c not in manifest_cols]
                if missing_cols:
                    errors.append(f"Manifest missing required columns: {', '.join(missing_cols)}")
                else:
                    # Validate factor structure
                    between_levels = design.pairing_manifest['Group'].unique()
                    within_levels = design.pairing_manifest['Condition'].unique()
                    
                    if len(between_levels) < 2:
                        errors.append(f"Between-subjects factor must have at least 2 levels (found {len(between_levels)})")
                    if len(within_levels) < 2:
                        errors.append(f"Within-subjects factor must have at least 2 levels (found {len(within_levels)})")
                    
                    expected_groups = len(between_levels) * len(within_levels)
                    if len(design.groups) != expected_groups:
                        errors.append(
                            f"{len(between_levels)}×{len(within_levels)} mixed design requires "
                            f"{expected_groups} groups (found {len(design.groups)})"
                        )
                
        # Check for duplicate group names
        group_names = [g.name for g in design.groups]
        if len(group_names) != len(set(group_names)):
            errors.append("Group names must be unique")
            
        # Check that all groups have valid folder paths
        for group in design.groups:
            if not group.folder_path:
                errors.append(f"Group {group.name} missing folder path")
                
        return errors
    
    @staticmethod
    def create_paired_two_group(group1: GroupInfo, group2: GroupInfo, manifest_path: str,
                                factor_name: str = "Condition", base_path: str = "", 
                                name: str = "") -> ExperimentalDesign:
        """
        Create a paired two-group design (A vs B, repeated measures).
        
        Args:
            group1: First condition (e.g., "No_Heat")
            group2: Second condition (e.g., "Heat")
            manifest_path: Path to Excel manifest file with subject pairings
            factor_name: Name for the within-subjects factor (e.g., "Temperature", "Treatment")
            base_path: Base directory path for validation (optional)
            name: Custom design name (optional)
            
        Returns:
            ExperimentalDesign with PAIRED_TWO_GROUP type
        """
        # Load and convert manifest from wide to long format
        df_wide = pd.read_excel(manifest_path)
        df_long = convert_manifest_wide_to_long(df_wide)
        
        # Validate manifest structure (skip validate_manifest() since it checks for 2+ groups)
        # Instead, do paired-specific validation
        required_cols = ['Subject_ID', 'Group', 'Condition', 'Filename']
        missing_cols = [col for col in required_cols if col not in df_long.columns]
        if missing_cols:
            raise ValueError(f"Manifest missing required columns: {', '.join(missing_cols)}")
        
        # Check for empty values
        for col in required_cols:
            if df_long[col].isna().any():
                n_missing = df_long[col].isna().sum()
                raise ValueError(f"Manifest column '{col}' has {n_missing} missing values")
        
        # Verify exactly 2 conditions
        conditions = sorted(df_long['Condition'].unique())
        if len(conditions) != 2:
            raise ValueError(f"Paired design requires exactly 2 conditions (found {len(conditions)})")
        
        # For paired design, all subjects belong to a single dummy group
        groups_in_manifest = df_long['Group'].unique()
        if len(groups_in_manifest) != 1:
            raise ValueError(
                f"Paired design manifest should have all subjects in same 'Group' column "
                f"(found {len(groups_in_manifest)} groups)"
            )
        
        # Verify each subject has measurements in both conditions
        subject_counts = df_long.groupby('Subject_ID')['Condition'].nunique()
        incomplete_subjects = subject_counts[subject_counts != 2].index.tolist()
        if incomplete_subjects:
            raise ValueError(
                f"All subjects must have measurements in both conditions. "
                f"Incomplete subjects: {incomplete_subjects}"
            )
        
        # Optional: Check that files exist if base_path provided
        if base_path:
            missing_files = []
            for _, row in df_long.iterrows():
                # Try to find the file in any of the condition folders
                file_found = False
                for cond_folder in [group1.folder_path, group2.folder_path]:
                    for protocol in ['Brief_current', 'Membrane_test_vc', 'Gap_free', 'Current_steps']:
                        file_path = os.path.join(cond_folder, protocol, row['Filename'])
                        if os.path.exists(file_path):
                            file_found = True
                            break
                    if file_found:
                        break
                
                if not file_found:
                    missing_files.append(row['Filename'])
            
            if missing_files:
                logger.warning(f"Could not find {len(missing_files)} files in manifest (may be in different folders)")
                # Don't raise error - files might be organized differently
        
        logger.info(f"Paired design validated: {len(df_long['Subject_ID'].unique())} subjects across 2 conditions")
        
        return ExperimentalDesign(
            design_type=DesignType.PAIRED_TWO_GROUP,
            groups=[group1, group2],
            pairing_manifest=df_long,
            within_factor_name=factor_name,  # Store factor name for use in analysis
            name=name or f"Paired: {group1.name} vs {group2.name}",
            description=f"Paired comparison between {group1.name} and {group2.name}"
        )
    
    @staticmethod
    def create_repeated_measures_multi_group(groups: List[GroupInfo], manifest_path: str,
                                            factor_name: str = "Temperature", 
                                            base_path: str = "", name: str = "") -> ExperimentalDesign:
        """
        Create a repeated measures design with 3+ conditions (single group, multiple measurements).
        
        Args:
            groups: List of GroupInfo for each condition (3+)
            manifest_path: Excel file with Subject ID and conditions in wide format
            factor_name: Name for within-subjects factor (e.g., "Temperature", "Time")
            base_path: Base directory for validation
            name: Custom design name
            
        Returns:
            ExperimentalDesign with REPEATED_MEASURES type
        """
        # Validate input
        if len(groups) < 3:
            raise ValueError("Repeated measures design requires 3+ groups (conditions)")
        
        # Load and convert manifest from wide to long format
        df_wide = pd.read_excel(manifest_path)
        df_long = convert_manifest_wide_to_long(df_wide)
        
        # Validate manifest structure
        required_cols = ['Subject_ID', 'Group', 'Condition', 'Filename']
        missing_cols = [col for col in required_cols if col not in df_long.columns]
        if missing_cols:
            raise ValueError(f"Manifest missing required columns: {', '.join(missing_cols)}")
        
        # Check for empty values
        for col in required_cols:
            if df_long[col].isna().any():
                n_missing = df_long[col].isna().sum()
                raise ValueError(f"Manifest column '{col}' has {n_missing} missing values")
        
        # Verify 3+ conditions
        conditions = sorted(df_long['Condition'].unique())
        if len(conditions) < 3:
            raise ValueError(f"Repeated measures design requires 3+ conditions (found {len(conditions)})")
        
        # For repeated measures, all subjects belong to a single dummy group
        groups_in_manifest = df_long['Group'].unique()
        if len(groups_in_manifest) != 1:
            raise ValueError(
                f"Repeated measures manifest should have all subjects in same 'Group' column "
                f"(found {len(groups_in_manifest)} groups)"
            )
        
        # Verify each subject has measurements in all conditions
        subject_counts = df_long.groupby('Subject_ID')['Condition'].nunique()
        expected_conditions = len(conditions)
        incomplete_subjects = subject_counts[subject_counts != expected_conditions].index.tolist()
        if incomplete_subjects:
            raise ValueError(
                f"All subjects must have measurements in all {expected_conditions} conditions. "
                f"Incomplete subjects: {incomplete_subjects}"
            )
        
        # Optional: Check that files exist if base_path provided
        if base_path:
            missing_files = []
            for _, row in df_long.iterrows():
                # Try to find the file in any of the condition folders
                file_found = False
                for group in groups:
                    for protocol in ['Brief_current', 'Membrane_test_vc', 'Gap_free', 'Current_steps']:
                        file_path = os.path.join(group.folder_path, protocol, row['Filename'])
                        if os.path.exists(file_path):
                            file_found = True
                            break
                    if file_found:
                        break
                
                if not file_found:
                    missing_files.append(row['Filename'])
            
            if missing_files:
                logger.warning(f"Could not find {len(missing_files)} files in manifest (may be in different folders)")
                # Don't raise error - files might be organized differently
        
        logger.info(f"Repeated measures design validated: {len(df_long['Subject_ID'].unique())} subjects across {len(conditions)} conditions")
        
        condition_names = ' vs '.join([g.name for g in groups])
        
        return ExperimentalDesign(
            design_type=DesignType.REPEATED_MEASURES,
            groups=groups,
            pairing_manifest=df_long,
            within_factor_name=factor_name,  # Store factor name for use in analysis
            name=name or f"Repeated Measures: {condition_names}",
            description=f"Repeated measures comparison across {len(conditions)} conditions"
        )
    
    @staticmethod
    def get_supported_designs() -> Dict[str, DesignType]:
        """Get dictionary of supported design types."""
        return {
            # Independent Groups
            "2 independent groups - 1 factor (between), 2 levels (A vs B)": DesignType.INDEPENDENT_TWO_GROUP,
            "3+ independent groups - 1 factor (between), 3+ levels (A vs B vs C...)": DesignType.INDEPENDENT_MULTI_GROUP,
            "Factorial design - 2+ factors (between), 2+ levels each (A1 vs A2 vs B1 vs B2)": DesignType.FACTORIAL_2X2,
            # Repeated Measures
            "Paired design - 1 factor (within), 2 levels (A vs A')": DesignType.PAIRED_TWO_GROUP,
            "Repeated measures - 1 factor (within), 3+ levels (A vs A' vs A''...)": DesignType.REPEATED_MEASURES,
            "Mixed factorial - 1 factor (between) + 1 factor (within) (A vs A' vs B vs B')": DesignType.MIXED_FACTORIAL,
        }
