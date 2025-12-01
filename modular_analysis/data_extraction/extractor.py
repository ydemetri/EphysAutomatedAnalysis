"""
Main data extraction engine for patch clamp data.
"""

import os
import logging
from typing import List, Dict, Optional, Tuple
from pathlib import Path

# Import the existing analysis modules (relative to project root)
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from analysis_code import joannas_vc_test
from analysis_code import joannas_IC_gap_free 
from analysis_code import joannas_current_steps
from analysis_code import joannas_brief_current

from ..shared.config import AnalysisConfig
from ..shared.data_models import GroupInfo, DataContainer
from ..shared.utils import setup_directories, validate_file_exists, convert_manifest_wide_to_long, validate_manifest
import pandas as pd

logger = logging.getLogger(__name__)


class DataExtractor:
    """Main class for extracting patch clamp data from folders."""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.extracted_data = DataContainer()
        
    def scan_data_directory(self, base_path: str) -> List[GroupInfo]:
        """Scan base directory and identify potential data groups."""
        groups = []
        
        if not os.path.isdir(base_path):
            raise ValueError(f"Directory does not exist: {base_path}")
        
        # Look for subdirectories that might contain data
        for item in os.listdir(base_path):
            folder_path = os.path.join(base_path, item)
            if os.path.isdir(folder_path):
                # Check if this folder has the expected protocol subdirectories
                n_files = self._count_protocol_files(folder_path)
                if n_files > 0:
                    groups.append(GroupInfo(
                        name=item,
                        folder_path=folder_path,
                        n_files=n_files
                    ))
                    
        logger.info(f"Found {len(groups)} potential data groups: {[g.name for g in groups]}")
        return groups
    
    def _count_protocol_files(self, folder_path: str) -> int:
        """Count total number of .abf files in protocol subdirectories."""
        total_files = 0
        protocol_dirs = ["Brief_current", "Membrane_test_vc", "Gap_free", "Current_steps"]
        
        for protocol_dir in protocol_dirs:
            protocol_path = os.path.join(folder_path, protocol_dir)
            if os.path.isdir(protocol_path):
                # Count .abf files
                abf_files = [f for f in os.listdir(protocol_path) if f.endswith('.abf')]
                total_files += len(abf_files)
                
        return total_files
    
    def extract_group_data(self, group: GroupInfo, base_path: str) -> bool:
        """Extract all data for a single group."""
        logger.info(f"Extracting data for group: {group.name}")
        
        # Setup output directory
        results_dir = os.path.join(base_path, self.config.output_dir)
        os.makedirs(results_dir, exist_ok=True)
        
        success = True
        
        # Extract Brief Current (AHP) data
        if self._extract_brief_current(group, base_path):
            logger.info(f"✓ Brief current data extracted for {group.name}")
        else:
            logger.warning(f"✗ Failed to extract brief current data for {group.name}")
            success = False
            
        # Extract Membrane Test (Input resistance) data
        if self._extract_membrane_test(group, base_path):
            logger.info(f"✓ Membrane test data extracted for {group.name}")
        else:
            logger.warning(f"✗ Failed to extract membrane test data for {group.name}")
            success = False
            
        # Extract Gap Free (Resting potential) data
        if self._extract_gap_free(group, base_path):
            logger.info(f"✓ Gap free data extracted for {group.name}")
        else:
            logger.warning(f"✗ Failed to extract gap free data for {group.name}")
            success = False
            
        # Extract Current Steps data
        if self._extract_current_steps(group, base_path):
            logger.info(f"✓ Current steps data extracted for {group.name}")
        else:
            logger.warning(f"✗ Failed to extract current steps data for {group.name}")
            success = False
            
        return success
    
    def _extract_brief_current(self, group: GroupInfo, base_path: str) -> bool:
        """Extract brief current (AHP) data for a group."""
        try:
            input_path = os.path.join(group.folder_path, "Brief_current")
            output_path = os.path.join(base_path, self.config.output_dir, 
                                     f"Calc_{group.name}_afterhyperpolarization.csv")
            
            if not os.path.isdir(input_path):
                logger.warning(f"Brief current directory not found: {input_path}")
                return False
                
            joannas_brief_current.analyze_bc(input_path, output_path)
            
            return validate_file_exists(output_path)
            
        except Exception as e:
            logger.error(f"Error extracting brief current data for {group.name}: {e}")
            return False
    
    def _extract_membrane_test(self, group: GroupInfo, base_path: str) -> bool:
        """Extract membrane test (input resistance) data for a group."""
        try:
            input_path = os.path.join(group.folder_path, "Membrane_test_vc")
            output_path = os.path.join(base_path, self.config.output_dir,
                                     f"Calc_{group.name}_input_resistance.csv")
            
            if not os.path.isdir(input_path):
                logger.warning(f"Membrane test directory not found: {input_path}")
                return False
                
            joannas_vc_test.get_input_resistance_from_vc(input_path, output_path)
            
            return validate_file_exists(output_path)
            
        except Exception as e:
            logger.error(f"Error extracting membrane test data for {group.name}: {e}")
            return False
    
    def _extract_gap_free(self, group: GroupInfo, base_path: str) -> bool:
        """Extract gap free (resting potential) data for a group."""
        try:
            input_path = os.path.join(group.folder_path, "Gap_free") 
            output_path = os.path.join(base_path, self.config.output_dir,
                                     f"Calc_{group.name}_resting_potential.csv")
            
            if not os.path.isdir(input_path):
                logger.warning(f"Gap free directory not found: {input_path}")
                return False
                
            joannas_IC_gap_free.get_resting_potential_from_gf(input_path, output_path)
            
            return validate_file_exists(output_path)
            
        except Exception as e:
            logger.error(f"Error extracting gap free data for {group.name}: {e}")
            return False
    
    def _extract_current_steps(self, group: GroupInfo, base_path: str) -> bool:
        """Extract current steps data for a group."""
        try:
            input_path = os.path.join(group.folder_path, "Current_steps")
            output_path1 = os.path.join(base_path, self.config.output_dir,
                                      f"Calc_{group.name}_frequency_vs_current.csv")
            output_path2 = os.path.join(base_path, self.config.output_dir,
                                      f"Calc_{group.name}_current_step_parameters.csv")
            output_path3 = os.path.join(base_path, self.config.output_dir,
                                      f"Calc_{group.name}_attenuation.csv")
            
            if not os.path.isdir(input_path):
                logger.warning(f"Current steps directory not found: {input_path}")
                return False
                
            joannas_current_steps.analyze_cc(input_path, output_path1, output_path2, output_path3)
            
            # Check that all outputs were created
            return all(validate_file_exists(p) for p in [output_path1, output_path2, output_path3])
            
        except Exception as e:
            logger.error(f"Error extracting current steps data for {group.name}: {e}")
            return False
    
    def extract_all_groups(self, groups: List[GroupInfo], base_path: str) -> Dict[str, bool]:
        """Extract data for all groups."""
        results = {}
        
        logger.info(f"Starting data extraction for {len(groups)} groups")
        
        for group in groups:
            results[group.name] = self.extract_group_data(group, base_path)
            
        successful = sum(results.values())
        logger.info(f"Data extraction completed: {successful}/{len(groups)} groups successful")
        
        return results
    
    def add_subject_ids_to_extracted_data(self, base_path: str, manifest: pd.DataFrame, 
                                          selected_groups: List[str]) -> None:
        """
        Retrospectively add Subject_ID columns to already-extracted CSV files.
        
        This is called during analysis (not extraction) for both MIXED_FACTORIAL 
        and PAIRED_TWO_GROUP designs. The manifest links filenames to subject IDs 
        across multiple group folders (mixed) or condition folders (paired).
        
        Args:
            base_path: Base directory containing Results folder
            manifest: Long-format manifest with Subject_ID, Group, Condition, Filename columns
            selected_groups: List of group folder names that were extracted
        """
        results_dir = os.path.join(base_path, self.config.output_dir)
        
        if not os.path.isdir(results_dir):
            logger.error(f"Results directory not found: {results_dir}")
            return
        
        # Detect design type from manifest structure
        n_groups = len(manifest['Group'].unique())
        if n_groups == 1:
            logger.info("Detected paired two-group design (single group in manifest)")
        else:
            logger.info(f"Detected mixed factorial design ({n_groups} groups in manifest)")
        
        # Create filename to subject mapping (filename -> subject_id)
        filename_to_subject = dict(zip(manifest['Filename'], manifest['Subject_ID']))
        
        logger.info(f"Adding Subject_IDs to extracted CSVs for {len(selected_groups)} groups...")
        
        # Process each selected group
        for group_name in selected_groups:
            # Handle current_step_parameters (has 'filename' column)
            self._add_subject_id_to_parameters_csv(
                results_dir, group_name, filename_to_subject
            )
            
            # Handle frequency_vs_current and attenuation (filenames in header)
            self._add_subject_id_to_wide_csv(
                results_dir, group_name, 'frequency_vs_current', filename_to_subject
            )
            self._add_subject_id_to_wide_csv(
                results_dir, group_name, 'attenuation', filename_to_subject
            )
    
    def _add_subject_id_to_parameters_csv(self, results_dir: str, group_name: str, 
                                          filename_to_subject: dict) -> None:
        """Add Subject_ID to current_step_parameters CSV (has 'filename' column)."""
        csv_file = f"Calc_{group_name}_current_step_parameters.csv"
        csv_path = os.path.join(results_dir, csv_file)
        
        if not os.path.exists(csv_path):
            return
        
        try:
            df = pd.read_csv(csv_path, index_col=False)
            
            # Check if filename column exists (case-insensitive)
            filename_col = None
            for col in df.columns:
                if col.lower() == 'filename':
                    filename_col = col
                    break
            
            if filename_col is None:
                logger.warning(f"No filename column in {csv_file}, skipping Subject_ID addition")
                return
            
            # Check if Subject_ID already present
            if 'Subject_ID' in df.columns:
                logger.info(f"{csv_file}: Subject_ID already present, skipping")
                return
            
            # Add Subject_ID (direct mapping - manifest already has .abf extensions)
            df['Subject_ID'] = df[filename_col].map(filename_to_subject)
            
            n_matched = df['Subject_ID'].notna().sum()
            n_total = len(df)
            
            if n_matched < n_total:
                unmatched = df[df['Subject_ID'].isna()][filename_col].tolist()
                logger.warning(f"{csv_file}: Only {n_matched}/{n_total} files matched. Unmatched: {unmatched}")
            
            df.to_csv(csv_path, index=False)
            logger.info(f"Added Subject_ID to {csv_file} ({n_matched}/{n_total} matched)")
                
        except Exception as e:
            logger.error(f"Error adding Subject_ID to {csv_file}: {e}")
    
    def _add_subject_id_to_wide_csv(self, results_dir: str, group_name: str, 
                                    suffix: str, filename_to_subject: dict) -> None:
        """
        Add Subject_ID to wide-format CSVs (frequency_vs_current, attenuation).
        In these files, filenames are in the header as column names.
        """
        csv_file = f"Calc_{group_name}_{suffix}.csv"
        csv_path = os.path.join(results_dir, csv_file)
        
        if not os.path.exists(csv_path):
            return
        
        try:
            # Read the CSV
            df = pd.read_csv(csv_path, index_col=False)
            
            # Extract filenames from column headers (exclude "Values_X" columns)
            # Header format: filename1, Values_0, filename2, Values_1, ...
            filenames = [col for col in df.columns if not col.startswith('Values_')]
            
            if not filenames:
                logger.warning(f"{csv_file}: No filenames found in header")
                return
            
            # Check if already processed (columns already have Subject_ prefix)
            if any(col.startswith('Subject_') for col in filenames):
                logger.info(f"{csv_file}: Already has Subject_ID format, skipping")
                return
            
            # Map filenames to Subject_IDs (direct mapping - manifest already has .abf extensions)
            subject_ids = [filename_to_subject.get(fn) for fn in filenames]
            
            # Create new header row with Subject_IDs
            # New structure: Subject_ID1, Values_0, Subject_ID2, Values_1, ...
            new_columns = []
            subject_idx = 0
            unmatched_files = []
            for i, col in enumerate(df.columns):
                if not col.startswith('Values_'):
                    # This is a filename column, replace with Subject_ID
                    if subject_ids[subject_idx] is not None:
                        new_columns.append(f"Subject_{subject_ids[subject_idx]}")
                    else:
                        new_columns.append(col)  # Keep original if no match
                        unmatched_files.append(col)
                    subject_idx += 1
                else:
                    # Keep Values_X columns as is
                    new_columns.append(col)
            
            # Rename columns
            df.columns = new_columns
            
            # Save back
            df.to_csv(csv_path, index=False)
            n_matched = sum(1 for sid in subject_ids if sid is not None)
            
            if n_matched < len(filenames):
                logger.warning(f"{csv_file}: Only {n_matched}/{len(filenames)} matched. Unmatched: {unmatched_files}")
            
            logger.info(f"Added Subject_ID to {csv_file} ({n_matched}/{len(filenames)} matched)")
            
        except Exception as e:
            logger.error(f"Error adding Subject_ID to {csv_file}: {e}")
