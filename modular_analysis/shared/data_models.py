"""
Data models and structures for the analysis system.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum
import pandas as pd


class DesignType(Enum):
    """Types of experimental designs."""
    INDEPENDENT_TWO_GROUP = "independent_2_group"
    INDEPENDENT_MULTI_GROUP = "independent_multi_group"
    FACTORIAL_2X2 = "factorial_2x2"  # N×M factorial design (2 factors with N and M levels each)
    MIXED_FACTORIAL = "mixed_factorial"  # N×M mixed factorial (between × within factors, repeated measures)
    PAIRED_TWO_GROUP = "paired_2_group"
    REPEATED_MEASURES = "repeated_measures"
    MIXED_EFFECTS = "mixed_effects"


@dataclass
class GroupInfo:
    """Information about a data group."""
    name: str
    folder_path: str
    color: str = "blue"
    marker: str = "o"
    italic: bool = False
    n_files: int = 0


@dataclass
class ExperimentalDesign:
    """Experimental design specification."""
    design_type: DesignType
    groups: List[GroupInfo]
    name: str = ""
    description: str = ""
    # Factorial design fields
    factor1_name: Optional[str] = None
    factor2_name: Optional[str] = None
    factor_mapping: Optional[Dict[str, Dict[str, str]]] = None  # Maps group_name -> {factor1: level, factor2: level}
    level_italic: Optional[Dict[str, bool]] = None  # Maps level name -> italic status
    # Mixed factorial design fields
    pairing_manifest: Optional[pd.DataFrame] = None  # Subject-level pairing info (long format)
    between_factor_name: Optional[str] = None  # Between-subjects factor (e.g., "Genotype")
    within_factor_name: Optional[str] = None  # Within-subjects factor (e.g., "Temperature")
    
    def get_group_by_name(self, name: str) -> Optional[GroupInfo]:
        """Get group info by name."""
        for group in self.groups:
            if group.name == name:
                return group
        return None
    
    def is_independent(self) -> bool:
        """Check if this is an independent groups design."""
        return self.design_type in [
            DesignType.INDEPENDENT_TWO_GROUP,
            DesignType.INDEPENDENT_MULTI_GROUP,
            DesignType.FACTORIAL_2X2
        ]
    
    def is_paired(self) -> bool:
        """Check if this is a paired groups design."""
        return self.design_type in [
            DesignType.PAIRED_TWO_GROUP,
            DesignType.REPEATED_MEASURES,
            DesignType.MIXED_EFFECTS,
            DesignType.MIXED_FACTORIAL
        ]


@dataclass
class DataContainer:
    """Container for extracted data."""
    current_steps: Dict[str, pd.DataFrame] = field(default_factory=dict)
    brief_current: Dict[str, pd.DataFrame] = field(default_factory=dict)
    membrane_test: Dict[str, pd.DataFrame] = field(default_factory=dict)
    gap_free: Dict[str, pd.DataFrame] = field(default_factory=dict)
    
    def get_combined_data(self, group_name: str) -> pd.DataFrame:
        """Get combined data for a specific group."""
        dfs = []
        
        if group_name in self.gap_free:
            dfs.append(self.gap_free[group_name])
        if group_name in self.membrane_test:
            dfs.append(self.membrane_test[group_name])
        if group_name in self.current_steps:
            dfs.append(self.current_steps[group_name])
        if group_name in self.brief_current:
            dfs.append(self.brief_current[group_name])
            
        if dfs:
            return pd.concat(dfs, ignore_index=True)
        else:
            return pd.DataFrame()


@dataclass
class StatisticalResult:
    """Container for statistical test results."""
    test_name: str
    measurement: str
    group1_name: str
    group1_mean: float
    group1_stderr: float
    group1_n: int
    group2_name: str
    group2_mean: float
    group2_stderr: float
    group2_n: int
    p_value: float
    corrected_p: Optional[float] = None
    measurement_type: str = ""
