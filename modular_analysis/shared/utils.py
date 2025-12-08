"""
Utility functions for the analysis system.
"""

import os
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
from typing import List, Dict, Optional, Tuple
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_directories(base_path: str, subdirs: List[str]) -> None:
    """Create directory structure if it doesn't exist."""
    for subdir in subdirs:
        dir_path = os.path.join(base_path, subdir)
        os.makedirs(dir_path, exist_ok=True)


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and preprocess dataframe."""
    # Make a copy to avoid modifying original
    df_clean = df.copy()
    
    # Replace zero time constants with NaN (couldn't be calculated)
    if "Time Constant (ms)" in df_clean.columns:
        df_clean["Time Constant (ms)"] = df_clean["Time Constant (ms)"].replace(0, np.nan)
    
    # Replace "None" strings with NaN for SFA columns
    sfa_cols = ["SFA10", "SFAn"]
    for col in sfa_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].replace("None", np.nan)
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    
    # Remove AHP data points with 'nan' strings
    if "AHP Amplitude (mV)" in df_clean.columns:
        df_clean = df_clean.loc[~(df_clean['AHP Amplitude (mV)'] == 'nan')]
    
    return df_clean


def validate_file_exists(filepath: str) -> bool:
    """Check if file exists and is accessible."""
    try:
        return os.path.isfile(filepath) and os.access(filepath, os.R_OK)
    except Exception as e:
        logger.warning(f"Error checking file {filepath}: {e}")
        return False


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers, returning default if division by zero."""
    try:
        if denominator == 0:
            return default
        return numerator / denominator
    except (TypeError, ZeroDivisionError):
        return default


def format_group_label(label: str, italic: bool = False) -> str:
    """Format group label for plots, with optional italics."""
    if italic:
        return rf'$\it{{{label}}}$'
    return label


def format_factorial_label(factor1_level: str, factor2_level: str, 
                           level_italic: Dict[str, bool] = None) -> str:
    """Format factorial design label with factors on separate lines and per-level italics."""
    level_italic = level_italic or {}
    
    # Format each level with or without italics based on level name
    # Note: Don't escape underscores - matplotlib handles them fine in regular text
    if level_italic.get(factor1_level, False):
        f1 = rf'$\mathit{{{factor1_level}}}$'
    else:
        f1 = factor1_level
    
    if level_italic.get(factor2_level, False):
        f2 = rf'$\mathit{{{factor2_level}}}$'
    else:
        f2 = factor2_level
    
    # Return with newline separator
    return f"{f1}\n{f2}"


def get_measurement_categories() -> Dict[str, List[str]]:
    """Get the standard measurement categories for multiple comparison correction."""
    return {
        "Intrinsic Property": [
            "Vm (mV)",
            "Rm (MOhm)", 
            "Time Constant (ms)",
            "Sag",
            "Rheobase (pA)"
        ],
        "Repetitive AP Property": [
            "Max Instantaneous (Hz)",
            "Max Steady-state (Hz)",
            "ISI_CoV",
            "SFA10", 
            "SFAn",
            "Burst_length (ms)"
        ],
        "Individual AP Property": [
            "AP Peak (mV)",
            "AP Threshold (mV)",
            "AP Amplitude (mV)",
            "AP Rise Time (ms)",
            "APD 50 (ms)",
            "APD 90 (ms)",
            "AHP Amplitude (mV)"
        ]
    }


def categorize_measurement(measurement: str) -> str:
    """Categorize a measurement for multiple comparison correction."""
    categories = get_measurement_categories()
    
    for category, measurements in categories.items():
        if measurement in measurements:
            return category
    
    # Default category if not found
    return "Other"


# Constants for parametric/nonparametric decision
# Based on Curran, West & Finch (1996) thresholds
MIN_N_PARAMETRIC = 10  # Below this, use non-parametric (can't assess assumptions reliably)
SKEW_THRESHOLD = 2.0   # |skewness| > 2 indicates severe non-normality
KURT_THRESHOLD = 7.0   # |kurtosis| > 7 indicates severe non-normality


def should_use_parametric(data_arrays: List[np.ndarray]) -> bool:
    """
    Decide whether to use parametric or non-parametric tests based on
    sample size and distribution shape (skewness/kurtosis).
    
    This replaces Shapiro-Wilk testing, which is problematic because:
    - Overpowered for large samples (rejects normality for trivial deviations)
    - Underpowered for small samples (can't detect non-normality)
    
    Decision logic:
    1. If any group has n < MIN_N_PARAMETRIC: use non-parametric
    2. If any group has |skewness| > 2 or |kurtosis| > 7: use non-parametric
    3. Otherwise: use parametric (with Welch correction for unequal variances)
    
    Thresholds based on Curran, West & Finch (1996) guidelines.
    
    Args:
        data_arrays: List of numpy arrays, one per group
        
    Returns:
        True if parametric tests should be used, False for non-parametric
    """
    for arr in data_arrays:
        # Convert to numpy array if needed
        arr = np.asarray(arr)
        arr = arr[~np.isnan(arr)]  # Remove NaN values
        
        # Small samples: can't reliably assess assumptions
        if len(arr) < MIN_N_PARAMETRIC:
            return False
        
        # Check distribution shape (need at least 3 values for skew/kurtosis)
        if len(arr) >= 3:
            try:
                arr_skew = abs(skew(arr))
                arr_kurt = abs(kurtosis(arr))  # excess kurtosis (normal = 0)
                
                # Severe violations: use non-parametric
                if arr_skew > SKEW_THRESHOLD or arr_kurt > KURT_THRESHOLD:
                    return False
            except Exception:
                # If calculation fails, be conservative
                return False
    
    return True


def create_output_paths(base_path: str, group_names: List[str]) -> Dict[str, Dict[str, str]]:
    """Create standard output file paths for all protocols and groups."""
    paths = {}
    
    protocols = {
        'brief_current': 'afterhyperpolarization',
        'membrane_test': 'input_resistance', 
        'gap_free': 'resting_potential',
        'current_steps': ['current_step_parameters', 'frequency_vs_current', 'attenuation']
    }
    
    for group in group_names:
        paths[group] = {}
        
        for protocol, suffix in protocols.items():
            if protocol == 'current_steps':
                # Current steps has multiple outputs
                for i, suf in enumerate(suffix):
                    key = f"{protocol}_{i+1}" if i > 0 else protocol
                    paths[group][key] = os.path.join(base_path, "Results", f"Calc_{group}_{suf}.csv")
            else:
                paths[group][protocol] = os.path.join(base_path, "Results", f"Calc_{group}_{suffix}.csv")
    
    return paths


def convert_manifest_wide_to_long(df_wide: pd.DataFrame) -> pd.DataFrame:
    """
    Convert wide format Excel manifest to long format for mixed designs.
    
    Input (wide format):
        Subject ID | condition 1: 32 | condition 2: 37 | condition 3: 42 | ...
        Scn1a_1    | file1           | file2           | file3           | ...
        WT_1       | file4           | file5           | file6           | ...
    
    Output (long format):
        Subject_ID | Group  | Condition | Filename
        Scn1a_1    | Scn1a  | 32        | file1.abf
        Scn1a_1    | Scn1a  | 37        | file2.abf
        WT_1       | WT     | 32        | file4.abf
        WT_1       | WT     | 37        | file5.abf
    
    Args:
        df_wide: Wide format DataFrame from Excel
        
    Returns:
        Long format DataFrame with Subject_ID, Group, Condition, Filename columns
    """
    # Identify subject ID column (assume first column)
    subject_col = df_wide.columns[0]
    # Exclude Subject ID and Group columns (if present) from condition columns
    condition_cols = [col for col in df_wide.columns if col != subject_col and col != 'Group']
    
    # Melt to long format
    df_long = df_wide.melt(
        id_vars=[subject_col],
        value_vars=condition_cols,
        var_name='Condition_Raw',
        value_name='Filename'
    )
    
    # Rename Subject ID column
    df_long = df_long.rename(columns={subject_col: 'Subject_ID'})
    
    # Check if Group column exists in original data
    if 'Group' in df_wide.columns:
        # Merge Group info from original wide format
        subject_to_group = df_wide.set_index(subject_col)['Group'].to_dict()
        df_long['Group'] = df_long['Subject_ID'].map(subject_to_group)
    else:
        # Extract group from Subject ID (before underscore or hyphen)
        df_long['Group'] = df_long['Subject_ID'].str.split('[_-]', n=1, expand=True)[0]
    
    # Extract condition name from column header (after colon if present)
    def extract_condition(col_name):
        """Extract condition name from column header."""
        # Ensure col_name is a string (Excel might parse it as int/float)
        col_name_str = str(col_name).strip()
        
        if ':' in col_name_str:
            # Format: "condition 1: 32" -> "32"
            condition = col_name_str.split(':', 1)[1].strip()
        else:
            # Format: "32" -> "32"
            condition = col_name_str
        
        # Normalize parentheses to underscores to match folder naming convention
        # E.g., "32(2)" -> "32_2"
        condition = condition.replace('(', '_').replace(')', '')
        
        return condition
    
    df_long['Condition'] = df_long['Condition_Raw'].apply(extract_condition)
    
    # Add .abf extension if missing
    def ensure_abf_extension(filename):
        """Ensure filename has .abf extension."""
        if pd.isna(filename):
            return filename
        filename_str = str(filename).strip()
        if not filename_str.lower().endswith('.abf'):
            return f"{filename_str}.abf"
        return filename_str
    
    df_long['Filename'] = df_long['Filename'].apply(ensure_abf_extension)
    
    # Clean up - remove intermediate columns and NaN filenames
    df_long = df_long[['Subject_ID', 'Group', 'Condition', 'Filename']]
    df_long = df_long.dropna(subset=['Filename'])
    
    logger.info(f"Converted manifest: {len(df_long)} file entries from {df_long['Subject_ID'].nunique()} subjects")
    
    return df_long


def validate_manifest(df_long: pd.DataFrame, base_path: str) -> Tuple[bool, List[str]]:
    """
    Validate manifest for completeness and consistency.
    
    Args:
        df_long: Long format manifest DataFrame
        base_path: Base directory path for finding files
        
    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []
    
    # Check required columns
    required_cols = ['Subject_ID', 'Group', 'Condition', 'Filename']
    missing_cols = [col for col in required_cols if col not in df_long.columns]
    if missing_cols:
        errors.append(f"Missing required columns: {', '.join(missing_cols)}")
        return False, errors
    
    # Check for empty values
    for col in required_cols:
        if df_long[col].isna().any():
            n_missing = df_long[col].isna().sum()
            errors.append(f"Column '{col}' has {n_missing} missing values")
    
    # Check that each subject has same number of conditions
    conditions_per_subject = df_long.groupby('Subject_ID')['Condition'].nunique()
    expected_conditions = conditions_per_subject.mode()[0] if len(conditions_per_subject) > 0 else 0
    
    incomplete_subjects = conditions_per_subject[conditions_per_subject != expected_conditions]
    if len(incomplete_subjects) > 0:
        errors.append(f"Incomplete data: {len(incomplete_subjects)} subjects don't have all conditions. "
                     f"Expected {expected_conditions} conditions per subject.")
        for subj in incomplete_subjects.index[:5]:  # Show first 5
            n_cond = incomplete_subjects[subj]
            errors.append(f"  - {subj}: has {n_cond} conditions (expected {expected_conditions})")
    
    # Check for duplicate subject-condition combinations
    duplicates = df_long.groupby(['Subject_ID', 'Condition']).size()
    duplicates = duplicates[duplicates > 1]
    if len(duplicates) > 0:
        errors.append(f"Found {len(duplicates)} duplicate subject-condition combinations")
        for (subj, cond), count in duplicates.head(5).items():
            errors.append(f"  - {subj} at {cond}: {count} entries")
    
    # Check that each subject belongs to only one group
    groups_per_subject = df_long.groupby('Subject_ID')['Group'].nunique()
    multi_group_subjects = groups_per_subject[groups_per_subject > 1]
    if len(multi_group_subjects) > 0:
        errors.append(f"Inconsistent groups: {len(multi_group_subjects)} subjects assigned to multiple groups")
        for subj in multi_group_subjects.index[:5]:
            groups = df_long[df_long['Subject_ID'] == subj]['Group'].unique()
            errors.append(f"  - {subj}: assigned to {', '.join(groups)}")
    
    # Check that at least 2 groups and 2 conditions exist
    n_groups = df_long['Group'].nunique()
    n_conditions = df_long['Condition'].nunique()
    
    if n_groups < 2:
        errors.append(f"Need at least 2 groups for mixed design (found {n_groups})")
    if n_conditions < 2:
        errors.append(f"Need at least 2 conditions for mixed design (found {n_conditions})")
    
    is_valid = len(errors) == 0
    
    if is_valid:
        logger.info(f"Manifest validation passed: {n_groups} groups × {n_conditions} conditions = "
                   f"{n_groups * n_conditions} expected cells")
    else:
        logger.warning(f"Manifest validation failed with {len(errors)} errors")
    
    return is_valid, errors
