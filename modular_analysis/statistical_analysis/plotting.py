"""
Plotting functions for statistical analysis results.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import os
import logging
from typing import List, Dict, Optional
from collections import defaultdict

from ..shared.data_models import ExperimentalDesign, StatisticalResult, DesignType
from ..shared.config import PlotConfig
from ..shared.utils import format_group_label, format_factorial_label, clean_dataframe

# Set matplotlib backend
matplotlib.use('agg')

logger = logging.getLogger(__name__)


class PlotGenerator:
    """Generates plots for statistical analysis results."""
    
    def __init__(self, config: PlotConfig, protocol_config=None):
        self.config = config
        self.protocol_config = protocol_config
        
    def create_measurement_plots(self, design: ExperimentalDesign, base_path: str) -> Dict[str, str]:
        """Create scatter plots for each measurement comparing groups."""
        
        plot_files = {}
        
        # Load combined data for all groups
        group_data = {}
        for group in design.groups:
            group_data[group.name] = self._load_combined_data(group.name, base_path)
            
        # Combine all data with group labels
        full_df = self._create_combined_dataframe(group_data, design)
        
        if full_df.empty:
            logger.warning("No data available for plotting")
            return plot_files
            
        # Get analysis columns
        analysis_columns = self._get_analysis_columns(full_df)
        
        # Create plot for each measurement
        for column in analysis_columns:
            plot_path = self._create_measurement_plot(column, full_df, design, base_path)
            if plot_path:
                plot_files[column] = plot_path
                
        logger.info(f"Created {len(plot_files)} measurement plots")
        return plot_files
        
    def create_frequency_plots(self, design: ExperimentalDesign, base_path: str) -> Dict[str, str]:
        """Create frequency vs current and fold-rheobase plots."""
        
        plot_files = {}
        
        try:
            # Current vs Frequency plot
            current_plot = self._create_current_vs_frequency_plot(design, base_path)
            if current_plot:
                plot_files["Current_vs_frequency"] = current_plot
                
            # Fold rheobase vs Frequency plot  
            rheobase_plot = self._create_fold_rheobase_plot(design, base_path)
            if rheobase_plot:
                plot_files["Fold_Rheobase_vs_frequency"] = rheobase_plot
                
        except Exception as e:
            logger.error(f"Error creating frequency plots: {e}")
            
        return plot_files
        
    def create_attenuation_plot(self, design: ExperimentalDesign, base_path: str, attenuation_data: Dict) -> Optional[str]:
        """Create AP number vs peak attenuation plot."""
        
        try:
            # Check if we have attenuation plot data
            if not attenuation_data or not attenuation_data.get('attenuation_plot_data'):
                logger.warning("No attenuation plot data available")
                return None
                
            plot_data = attenuation_data['attenuation_plot_data']
            
            # Create figure with main plot and legend subplot below
            # Adjust width based on number of groups
            n_groups = len(plot_data)
            fig_width = max(14, 1.8 * n_groups)
            legend_height = 2.5 if n_groups > 4 else 2.0
            fig, (ax, ax_legend) = plt.subplots(2, 1, figsize=(fig_width, self.config.figure_size[1]+legend_height),
                                                gridspec_kw={'height_ratios': [5, legend_height/2.0]})
            
            # Dynamic color palette for N groups
            default_colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan']
            
            # Plot first 10 APs for each group
            ap_num = [i for i in range(1, 11)]
            group_index = 0
            
            is_factorial = design.design_type in (DesignType.FACTORIAL_2X2, DesignType.MIXED_FACTORIAL)
            
            for group_name, group_data in plot_data.items():
                # Get group info for colors and formatting
                group_info = next((g for g in design.groups if g.name == group_name), None)
                color = group_info.color if (group_info and hasattr(group_info, 'color') and group_info.color) else default_colors[group_index % len(default_colors)]
                
                # Use factorial labeling for factorial designs
                if is_factorial and design.factor_mapping and group_name in design.factor_mapping:
                    # Both independent and dependent factorial
                    factor1_level = design.factor_mapping[group_name]['factor1']
                    factor2_level = design.factor_mapping[group_name]['factor2']
                    label = format_factorial_label(
                        factor1_level, factor2_level,
                        design.level_italic
                    )
                else:
                    label = format_group_label(group_name, group_info.italic if (group_info and hasattr(group_info, 'italic')) else False)
                
                # Plot line with error bars for first 10 APs
                mean_vals = group_data['mean'].iloc[:10] if hasattr(group_data['mean'], 'iloc') else group_data['mean'][:10]
                sem_vals = group_data['sem'].iloc[:10] if hasattr(group_data['sem'], 'iloc') else group_data['sem'][:10]
                
                # Convert to numpy arrays and replace None with NaN
                mean_array = np.array([val if val is not None else np.nan for val in mean_vals])
                sem_array = np.array([val if val is not None else np.nan for val in sem_vals])
                
                # Use only the AP numbers that have data
                actual_ap_num = [i for i in range(1, len(mean_array) + 1)]
                
                ax.plot(actual_ap_num, mean_array, color=color, label=label, linewidth=2)
                ax.fill_between(actual_ap_num, 
                              mean_array - sem_array,
                              mean_array + sem_array, 
                              facecolor=color, alpha=0.3)
                
                group_index += 1
                          
            ax.set_xlabel("AP Number", fontsize=self.config.font_size_label)
            ax.set_ylabel("AP Peak (mV)", fontsize=self.config.font_size_label)
            ax.spines[['right', 'top']].set_visible(False)
            ax.tick_params(axis='both', which='major', labelsize=self.config.font_size_tick)
            ax.set_xticks([0, 5, 10])
            
            # Put legend in separate subplot below
            handles, labels = ax.get_legend_handles_labels()
            legend_fontsize = self.config.font_size_label / 1.5
            n_cols = min(4, n_groups)
            leg = ax_legend.legend(handles, labels, loc='center', ncol=n_cols, frameon=False, 
                                  fontsize=legend_fontsize, handlelength=2.5, handleheight=2.5, columnspacing=1.0)
            for line in leg.get_lines():
                line.set_linewidth(3.0)
            ax_legend.axis('off')
            
            plt.tight_layout()
            
            output_path = os.path.join(base_path, "Results", "Plot_AP_Num_vs_Peak.png")
            plt.savefig(output_path)
            plt.close('all')
            
            logger.info("Created attenuation plot")
            return output_path
            
        except Exception as e:
            logger.error(f"Error creating attenuation plot: {e}")
            return None
        
    def create_burst_analysis_plot(self, design: ExperimentalDesign, base_path: str) -> Optional[str]:
        """Create ISI CoV vs Burst Length scatter plot."""
        
        try:
            group_data = {}
            for group in design.groups:
                data = self._load_combined_data(group.name, base_path)
                if not data.empty and 'ISI_CoV' in data.columns and 'Burst_length (ms)' in data.columns:
                    group_data[group.name] = data
                    
            if len(group_data) < 2:
                logger.warning("Insufficient data for burst analysis plot")
                return None
                
            # Create figure with main plot and legend subplot below
            # Adjust width based on number of groups
            n_groups = len(group_data)
            fig_width = max(14, 1.8 * n_groups)
            legend_height = 2.5 if n_groups > 4 else 2.0
            fig, (ax, ax_legend) = plt.subplots(2, 1, figsize=(fig_width, self.config.figure_size[1]+legend_height),
                                                gridspec_kw={'height_ratios': [5, legend_height/2.0]})
            is_factorial = design.design_type in (DesignType.FACTORIAL_2X2, DesignType.MIXED_FACTORIAL)
            
            for i, (group_name, data) in enumerate(group_data.items()):
                group_info = design.get_group_by_name(group_name)
                color = group_info.color if group_info else self.config.colors[f'group{i+1}']
                marker = group_info.marker if group_info else self.config.markers[f'group{i+1}']
                
                # Use factorial labeling for factorial designs
                if is_factorial and design.factor_mapping and group_name in design.factor_mapping:
                    factor1_level = design.factor_mapping[group_name]['factor1']
                    factor2_level = design.factor_mapping[group_name]['factor2']
                    label = format_factorial_label(
                        factor1_level, factor2_level,
                        design.level_italic
                    )
                else:
                    label = format_group_label(group_name, group_info.italic if group_info else False)
                
                ax.scatter(data['ISI_CoV'], data['Burst_length (ms)'], 
                         color=color, label=label, s=110, marker=marker)
                         
            ax.set_xlabel("ISI CoV", fontsize=self.config.font_size_label)
            ax.set_ylabel("Burst Length (ms)", fontsize=self.config.font_size_label)
            ax.spines[['right', 'top']].set_visible(False)
            ax.tick_params(axis='both', which='major', labelsize=self.config.font_size_tick)
            ax.axhline(800.0, linestyle='--', c="black")
            ax.axvline(1.0, linestyle='--', c='black')
            
            # Put legend in separate subplot below
            handles, labels = ax.get_legend_handles_labels()
            legend_fontsize = self.config.font_size_label / 1.5
            n_cols = min(4, n_groups)
            leg = ax_legend.legend(handles, labels, loc='center', ncol=n_cols, frameon=False, 
                                  fontsize=legend_fontsize, handlelength=1.5, handleheight=2.5, columnspacing=1.0)
            for line in leg.get_lines():
                line.set_linewidth(3.0)
            ax_legend.axis('off')
            
            plt.tight_layout()
            
            output_path = os.path.join(base_path, "Results", "Plot_ISI_cov_vs_burst_length.png")
            plt.savefig(output_path)
            plt.close('all')
            
            logger.info("Created burst analysis plot")
            return output_path
            
        except Exception as e:
            logger.error(f"Error creating burst analysis plot: {e}")
            return None
    
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
    
    def _create_combined_dataframe(self, group_data: Dict[str, pd.DataFrame], 
                                  design: ExperimentalDesign) -> pd.DataFrame:
        """Create combined dataframe with group labels."""
        
        dfs = []
        is_factorial = design.design_type in (DesignType.FACTORIAL_2X2, DesignType.MIXED_FACTORIAL)
        
        for group in design.groups:
            if group.name in group_data and not group_data[group.name].empty:
                df = group_data[group.name].copy()
                
                # Use factorial labeling for factorial designs
                if is_factorial and design.factor_mapping:
                    factor1_level = design.factor_mapping[group.name]['factor1']
                    factor2_level = design.factor_mapping[group.name]['factor2']
                    df['Group'] = format_factorial_label(
                        factor1_level, factor2_level,
                        design.level_italic
                    )
                else:
                    df['Group'] = format_group_label(group.name, group.italic)
                    
                dfs.append(df)
                
        if dfs:
            return pd.concat(dfs, ignore_index=True)
        else:
            return pd.DataFrame()
    
    def _get_analysis_columns(self, df: pd.DataFrame) -> List[str]:
        """Get columns that should be plotted."""
        exclude_columns = ["filename", "Group", "geno", "Subject_ID", "Filename"]
        # Also exclude any columns that start with "Subject_" (from wide-format Subject_1, Subject_2, etc.)
        return [col for col in df.columns if col not in exclude_columns and not col.startswith('Subject_')]
    
    def _create_measurement_plot(self, column: str, full_df: pd.DataFrame, 
                                design: ExperimentalDesign, base_path: str) -> Optional[str]:
        """Create a scatter plot for a single measurement."""
        
        try:
            # Filter data for this measurement
            plot_data = full_df[['Group', column]].dropna()
            if plot_data.empty:
                logger.warning(f"No data available for {column}")
                return None
                
            plt.close('all')
            # Create figure without legend (legend removed per user request)
            # Adjust width based on number of groups to prevent crowding
            n_groups = len(design.groups)
            # Scale figure width: 3 inches base + 2.5 inches per group
            fig_width = 3 + 2.5 * n_groups
            fig, ax = plt.subplots(figsize=(fig_width, 10))
            
            # Create color and size mappings for seaborn
            color_map = {}
            size_map = {}
            is_factorial = design.design_type in (DesignType.FACTORIAL_2X2, DesignType.MIXED_FACTORIAL)
            
            for group in design.groups:
                if is_factorial and design.factor_mapping:
                    factor1_level = design.factor_mapping[group.name]['factor1']
                    factor2_level = design.factor_mapping[group.name]['factor2']
                    formatted_name = format_factorial_label(
                        factor1_level, factor2_level,
                        design.level_italic
                    )
                else:
                    formatted_name = format_group_label(group.name, group.italic)
                    
                color_map[formatted_name] = group.color
                size_map[formatted_name] = 150
            
            # Determine group order
            if is_factorial and design.factor_mapping:
                group_order = [format_factorial_label(
                    design.factor_mapping[g.name]['factor1'],
                    design.factor_mapping[g.name]['factor2'],
                    design.level_italic
                ) for g in design.groups]
            else:
                group_order = [format_group_label(g.name, g.italic) for g in design.groups]
            
            # Create scatter plot with stripplot directly on ax
            sns.stripplot(x="Group", y=column, hue="Group", data=plot_data,
                         palette=color_map, legend=False, size=10, ax=ax,
                         order=group_order, jitter=0.3)
            
            # Adjust x-axis limits with consistent padding
            xticks = ax.get_xticks()
            # Use fixed padding of 0.5 for clean, consistent margins
            ax.set_xlim(xticks[0] - 0.5, xticks[-1] + 0.5)
                
            # Add error bars for all groups
            for group in design.groups:
                if is_factorial and design.factor_mapping:
                    factor1_level = design.factor_mapping[group.name]['factor1']
                    factor2_level = design.factor_mapping[group.name]['factor2']
                    formatted_name = format_factorial_label(
                        factor1_level, factor2_level,
                        design.level_italic
                    )
                else:
                    formatted_name = format_group_label(group.name, group.italic)
                    
                group_data = plot_data[plot_data['Group'] == formatted_name][column].dropna()
                
                if len(group_data) > 0:
                    mean_val = group_data.mean()
                    se_val = group_data.std() / np.sqrt(len(group_data))
                    ax.errorbar(formatted_name, mean_val, yerr=se_val, fmt="_", 
                              color="black", capsize=10, markersize=50, 
                              markeredgewidth=6, zorder=5)
            
            # Set axis properties
            if column not in ['AP Threshold (mV)', 'Vm (mV)', "Velocity Downstroke (mV_per_ms)"]:
                # Get the y-axis range to calculate margin
                current_ylim = ax.get_ylim()
                y_range = current_ylim[1] - current_ylim[0]
                # Set bottom with small negative margin (2% of range) so 0 doesn't overlap with axis
                ax.set_ylim(bottom=-0.02 * y_range, top=current_ylim[1])
                
            ax.spines[['right', 'top']].set_visible(False)
            ax.set(xlabel=None)
            plt.tick_params(axis='x', which='both', bottom=False, labelbottom=True)
            ax.tick_params(axis='both', which='major', labelsize=self.config.font_size_tick)
            
            # Explicitly set x-tick positions and labels to ensure newlines are preserved
            # This avoids matplotlib warning about using set_xticklabels without set_xticks
            tick_positions = range(len(group_order))
            ax.set_xticks(tick_positions)
            # rotation_mode='anchor' ensures the label rotates around the anchor point (ha, va)
            # This centers the label under the tick when rotated
            ax.set_xticklabels(group_order, rotation=45, ha='right', rotation_mode='anchor')
            ax.set_ylabel(column, fontsize=self.config.font_size_label)
            
            plt.tight_layout()
            
            # Save plot
            output_path = os.path.join(base_path, "Results", f"Plot_{column}.png")
            plt.savefig(output_path)
            plt.close('all')
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error creating plot for {column}: {e}")
            return None
    
    def _create_current_vs_frequency_plot(self, design: ExperimentalDesign, base_path: str) -> Optional[str]:
        """Create current vs frequency plot."""
        
        try:
            # Create figure with main plot and legend subplot below
            # Adjust width based on number of groups
            n_groups = len(design.groups)
            fig_width = max(14, 1.8 * n_groups)
            legend_height = 2.5 if n_groups > 4 else 2.0
            fig, (ax, ax_legend) = plt.subplots(2, 1, figsize=(fig_width, self.config.figure_size[1]+legend_height),
                                                gridspec_kw={'height_ratios': [5, legend_height/2.0]})
            
            # Dynamic color palette for N groups
            default_colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan']
            is_factorial = design.design_type in (DesignType.FACTORIAL_2X2, DesignType.MIXED_FACTORIAL)
            
            for i, group in enumerate(design.groups):
                # Load frequency vs current data
                freq_file = os.path.join(base_path, "Results", 
                                       f"Calc_{group.name}_frequency_vs_current.csv")
                if not os.path.exists(freq_file):
                    continue
                    
                df = self._make_cvf_df(freq_file)
                if df.empty:
                    continue
                    
                # Use group color or default color palette
                color = group.color if hasattr(group, 'color') and group.color else default_colors[i % len(default_colors)]
                
                # Use factorial labeling for factorial designs
                if is_factorial and design.factor_mapping and group.name in design.factor_mapping:
                    factor1_level = design.factor_mapping[group.name]['factor1']
                    factor2_level = design.factor_mapping[group.name]['factor2']
                    label = format_factorial_label(
                        factor1_level, factor2_level,
                        design.level_italic
                    )
                else:
                    label = format_group_label(group.name, group.italic)
                
                ax.plot(df['Current'], df['mean'], color=color, label=label)
                ax.fill_between(df['Current'], (df['mean']-df['se']), 
                              (df['mean']+df['se']), facecolor=color, alpha=0.3)
                              
            ax.set_xlabel('Current (pA)', fontsize=self.config.font_size_label)
            ax.set_ylabel("Frequency (Hz)", fontsize=self.config.font_size_label)
            ax.spines[['right', 'top']].set_visible(False)
            ax.tick_params(axis='both', which='major', labelsize=self.config.font_size_tick)
            
            # Put legend in separate subplot below
            handles, labels = ax.get_legend_handles_labels()
            legend_fontsize = self.config.font_size_label / 1.5
            n_cols = min(4, n_groups)
            leg = ax_legend.legend(handles, labels, loc='center', ncol=n_cols, frameon=False, 
                                  fontsize=legend_fontsize, handlelength=2.5, handleheight=2.5, columnspacing=1.0)
            for line in leg.get_lines():
                line.set_linewidth(3.0)
            ax_legend.axis('off')
            
            plt.tight_layout()
            
            output_path = os.path.join(base_path, "Results", "Plot_Current_vs_frequency.png")
            plt.savefig(output_path)
            plt.close('all')
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error creating current vs frequency plot: {e}")
            return None
    
    def _create_fold_rheobase_plot(self, design: ExperimentalDesign, base_path: str) -> Optional[str]:
        """Create fold rheobase vs frequency plot."""
        
        try:
            # Create figure with main plot and legend subplot below
            # Adjust width based on number of groups
            n_groups = len(design.groups)
            fig_width = max(14, 1.8 * n_groups)
            legend_height = 2.5 if n_groups > 4 else 2.0
            fig, (ax, ax_legend) = plt.subplots(2, 1, figsize=(fig_width, self.config.figure_size[1]+legend_height),
                                                gridspec_kw={'height_ratios': [5, legend_height/2.0]})
            
            # Dynamic color palette for N groups
            default_colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan']
            is_factorial = design.design_type in (DesignType.FACTORIAL_2X2, DesignType.MIXED_FACTORIAL)
            
            for i, group in enumerate(design.groups):
                # Create rheobase data
                freq_file = os.path.join(base_path, "Results", 
                                       f"Calc_{group.name}_frequency_vs_current.csv")
                if not os.path.exists(freq_file):
                    continue
                    
                df = self._make_rheo_df(freq_file, group.name, base_path)
                if df.empty:
                    continue
                    
                # Use group color or default color palette
                color = group.color if hasattr(group, 'color') and group.color else default_colors[i % len(default_colors)]
                
                # Use factorial labeling for factorial designs
                if is_factorial and design.factor_mapping and group.name in design.factor_mapping:
                    factor1_level = design.factor_mapping[group.name]['factor1']
                    factor2_level = design.factor_mapping[group.name]['factor2']
                    label = format_factorial_label(
                        factor1_level, factor2_level,
                        design.level_italic
                    )
                else:
                    label = format_group_label(group.name, group.italic)
                
                ax.plot(df['Fold Rheobase'], df['mean'], color=color, label=label)
                ax.fill_between(df['Fold Rheobase'], (df['mean']-df['se']), 
                              (df['mean']+df['se']), facecolor=color, alpha=0.3)
                              
            ax.set_xlabel('Fold Rheobase', fontsize=self.config.font_size_label)
            ax.set_ylabel("Frequency (Hz)", fontsize=self.config.font_size_label)
            ax.spines[['right', 'top']].set_visible(False)
            ax.tick_params(axis='both', which='major', labelsize=self.config.font_size_tick)
            ax.set_xticks([2, 4, 6, 8, 10])
            
            # Put legend in separate subplot below
            handles, labels = ax.get_legend_handles_labels()
            legend_fontsize = self.config.font_size_label / 1.5
            n_cols = min(4, n_groups)
            leg = ax_legend.legend(handles, labels, loc='center', ncol=n_cols, frameon=False, 
                                  fontsize=legend_fontsize, handlelength=2.5, handleheight=2.5, columnspacing=1.0)
            for line in leg.get_lines():
                line.set_linewidth(3.0)
            ax_legend.axis('off')
            
            plt.tight_layout()
            
            output_path = os.path.join(base_path, "Results", "Plot_Fold_Rheobase_vs_frequency.png")
            plt.savefig(output_path)
            plt.close('all')
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error creating fold rheobase plot: {e}")
            return None
    
    def _make_cvf_df(self, file_path: str) -> pd.DataFrame:
        """Make current vs frequency dataframe from file."""
        # This is extracted from the original code
        try:
            c_vs_f_data = pd.read_csv(file_path, index_col=False)
            vals_df = c_vs_f_data.filter(regex='Values').copy()  # Explicit copy to avoid SettingWithCopyWarning
            val_cols = vals_df.columns
            vals_df['n'] = vals_df.count(axis=1)

            # Get current - using default values for now
            max_steps = vals_df.count().max()
            min_curr = self.protocol_config.min_current if self.protocol_config else -60.0
            curr_step = self.protocol_config.step_size if self.protocol_config else 20.0
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
    
    def _make_rheo_df(self, file_path: str, group_name: str = "", base_path: str = "") -> pd.DataFrame:
        """Make fold rheobase dataframe from file."""
        # This is extracted from the original code
        try:
            c_vs_f_data = pd.read_csv(file_path, index_col=False)
            val_cols = c_vs_f_data.filter(regex='Values').columns
            # Get ID columns - either .abf (independent) or Subject_ (dependent)
            curr_cols = c_vs_f_data.filter(regex='.abf').columns
            if len(curr_cols) == 0:
                curr_cols = c_vs_f_data.filter(regex='Subject_').columns

            # Get current
            max_steps = c_vs_f_data.count().max()
            curr_step = self.protocol_config.step_size if self.protocol_config else 20.0
            min_curr = self.protocol_config.min_current if self.protocol_config else -60.0
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
            
            # Make fold-rheobase data for plotting (only integer folds, up to 10)
            points = []
            for i in range(len(curr_cols)):
                for j in range(c_vs_f_data.loc[:, curr_cols[i]].shape[0]):
                    curr = c_vs_f_data.loc[:, curr_cols[i]][j]
                    points.append((curr, c_vs_f_data.loc[:, val_cols[i]][j]))

            point_dict = defaultdict(list)
            folds = []
            for i, j in points: 
                if i in [1,2,3,4,5,6,7,8,9,10]:
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
