"""
Main statistical analyzer that coordinates the analysis process.
"""

import os
import logging
from typing import List, Dict, Optional
import pandas as pd
from datetime import datetime

from .designs import DesignManager
from .tests.unpaired_ttest import UnpairedTTest
from .tests.paired_ttest import PairedTTest
from .tests.oneway_anova import OneWayANOVA
from .tests.two_way_anova import TwoWayANOVA
from .tests.mixed_anova import MixedANOVA
from .tests.frequency_analysis import FrequencyAnalyzer
from .tests.frequency_analysis_dependent import FrequencyAnalyzerDependent
from .tests.attenuation_analysis import AttenuationAnalyzer
from .tests.attenuation_analysis_dependent import AttenuationAnalyzerDependent
from .plotting import PlotGenerator
from ..shared.data_models import ExperimentalDesign, DesignType, StatisticalResult
from ..shared.config import AnalysisConfig
from ..data_extraction.extractor import DataExtractor

logger = logging.getLogger(__name__)


class StatisticalAnalyzer:
    """Main class for coordinating statistical analysis."""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.plot_generator = PlotGenerator(config.plotting, config.protocol)
        self.frequency_analyzer = FrequencyAnalyzer(config.protocol)
        self.frequency_analyzer_dependent = FrequencyAnalyzerDependent(config.protocol)
        self.attenuation_analyzer = AttenuationAnalyzer()
        self.attenuation_analyzer_dependent = AttenuationAnalyzerDependent()
        self.extractor = DataExtractor(config)  # For adding Subject_IDs in mixed designs
        
    def run_analysis(self, design: ExperimentalDesign, base_path: str, 
                     selected_measurements: Optional[List[str]] = None) -> Dict[str, any]:
        """Run complete statistical analysis for the given design.
        
        Parameters
        ----------
        design : ExperimentalDesign
            The experimental design to analyze
        base_path : str
            Path to the data directory
        selected_measurements : Optional[List[str]]
            List of measurement names to include in analysis. If None, all measurements are included.
        """
        
        logger.info(f"Starting analysis: {design.name}")
        if selected_measurements:
            logger.info(f"Analyzing {len(selected_measurements)} selected measurements")
        
        log_handler = None
        log_file_path = None
        root_logger = logging.getLogger()
        
        # Validate design
        errors = DesignManager.validate_design(design)
        if errors:
            raise ValueError(f"Invalid design: {', '.join(errors)}")
        
        results = {
            'design': design,
            'statistical_results': [],
            'frequency_analysis': {},
            'attenuation_analysis': {},
            'mixed_effects_results': [],
            'plot_files': {},
            'success': False,
            'log_file': None
        }
        
        try:
            # Set up run-specific log file
            results_dir = os.path.join(base_path, self.config.output_dir)
            os.makedirs(results_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file_path = os.path.join(results_dir, f"analysis_{timestamp}.log")
            log_handler = logging.FileHandler(log_file_path, mode='w', encoding='utf-8')
            log_handler.setLevel(logging.INFO)
            formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            log_handler.setFormatter(formatter)
            root_logger.addHandler(log_handler)
            results['log_file'] = log_file_path
            logger.info(f"Analysis log file: {log_file_path}")
            
            # For dependent designs (mixed factorial, paired, and repeated measures), add Subject_IDs to extracted CSVs first
            if design.design_type in [DesignType.MIXED_FACTORIAL, DesignType.PAIRED_TWO_GROUP, DesignType.REPEATED_MEASURES]:
                design_label = "paired design" if design.design_type == DesignType.PAIRED_TWO_GROUP else "mixed design"
                logger.info(f"Adding Subject_IDs to extracted data for {design_label}...")
                selected_group_names = [g.name for g in design.groups]
                self.extractor.add_subject_ids_to_extracted_data(
                    base_path, 
                    design.pairing_manifest,
                    selected_group_names
                )
            
            # Run basic statistical tests (measurement comparisons)
            statistical_results = self._run_statistical_tests(design, base_path, selected_measurements)
            results['statistical_results'] = statistical_results
            
            # Save basic statistical results
            self._save_statistical_results(statistical_results, design, base_path)
            
            # Run frequency analyses (current and fold rheobase vs frequency)
            frequency_results = self._run_frequency_analyses(design, base_path)
            results['frequency_analysis'] = frequency_results
            
            # Run attenuation analysis (AP number vs peak)
            attenuation_results = self._run_attenuation_analysis(design, base_path)
            results['attenuation_analysis'] = attenuation_results
            
            # Collect mixed-effects results
            mixed_effects_results = self._collect_mixed_effects_results(frequency_results, attenuation_results)
            results['mixed_effects_results'] = mixed_effects_results
            
            # Save mixed-effects results
            self._save_mixed_effects_results(mixed_effects_results, base_path)
            self._save_mixed_effects_posthoc_results(mixed_effects_results, base_path)
            
            # Generate plots (including attenuation plot)
            plot_files = self._generate_plots(design, base_path, attenuation_results)
            results['plot_files'] = plot_files
            
            results['success'] = True
            logger.info(f"Analysis completed successfully for {design.name}")
            
        except Exception as e:
            logger.error(f"Error during analysis: {e}")
            results['error'] = str(e)
            
        finally:
            if log_handler:
                root_logger.removeHandler(log_handler)
                log_handler.close()
            
        return results
    
    def _run_statistical_tests(self, design: ExperimentalDesign, base_path: str,
                               selected_measurements: Optional[List[str]] = None) -> List[StatisticalResult]:
        """Select and run appropriate statistical tests based on design type."""
        
        if design.design_type == DesignType.INDEPENDENT_TWO_GROUP:
            test = UnpairedTTest()
            return test.run_analysis(design, None, base_path, selected_measurements)
        elif design.design_type == DesignType.PAIRED_TWO_GROUP:
            test = PairedTTest()
            return test.run_analysis(design, None, base_path, selected_measurements)
        elif design.design_type == DesignType.INDEPENDENT_MULTI_GROUP:
            test = OneWayANOVA()
            return test.run_analysis(design, None, base_path, selected_measurements)
        elif design.design_type == DesignType.REPEATED_MEASURES:
            from .tests.repeated_measures_anova import RepeatedMeasuresANOVA
            test = RepeatedMeasuresANOVA()
            return test.run_analysis(design, None, base_path, selected_measurements)
        elif design.design_type == DesignType.FACTORIAL_2X2:
            test = TwoWayANOVA()
            return test.run_analysis(design, None, base_path, selected_measurements)
        elif design.design_type == DesignType.MIXED_FACTORIAL:
            test = MixedANOVA()
            return test.run_analysis(design, None, base_path, selected_measurements)
        else:
            raise NotImplementedError(f"Design type {design.design_type} not yet implemented")
    
    def _save_statistical_results(self, results: List[StatisticalResult], 
                                 design: ExperimentalDesign, base_path: str) -> None:
        """Save statistical results to files."""
        
        if not results:
            logger.warning("No statistical results to save")
            return
            
        # Create output directory
        results_dir = os.path.join(base_path, self.config.output_dir)
        os.makedirs(results_dir, exist_ok=True)
        
        # Save main results file
        if design.design_type == DesignType.INDEPENDENT_TWO_GROUP:
            output_file = os.path.join(results_dir, "Stats_Group1_vs_Group2.csv")
            test = UnpairedTTest()
            test.save_results(results, output_file)
        elif design.design_type == DesignType.PAIRED_TWO_GROUP:
            output_file = os.path.join(results_dir, "Stats_Group1_vs_Group2.csv")
            test = PairedTTest()
            test.save_results(results, output_file)
        elif design.design_type == DesignType.REPEATED_MEASURES:
            output_file = os.path.join(results_dir, "Stats_Group_comparison.csv")
            from .tests.repeated_measures_anova import RepeatedMeasuresANOVA
            test = RepeatedMeasuresANOVA()
            test.save_results(results, output_file, design, base_path)
        elif design.design_type == DesignType.INDEPENDENT_MULTI_GROUP:
            output_file = os.path.join(results_dir, "Stats_Group_comparison.csv")  
            test = OneWayANOVA()
            test.save_results(results, output_file, design, base_path)
        elif design.design_type == DesignType.FACTORIAL_2X2:
            # Dynamic filename based on actual N×M dimensions
            n_factor1_levels = len(set(design.factor_mapping[g]['factor1'] for g in design.factor_mapping))
            n_factor2_levels = len(set(design.factor_mapping[g]['factor2'] for g in design.factor_mapping))
            output_file = os.path.join(results_dir, f"Stats_Parameters.csv")
            test = TwoWayANOVA()
            test.save_results(results, output_file, design, base_path)
        elif design.design_type == DesignType.MIXED_FACTORIAL:
            # Dynamic filename based on actual N×M dimensions
            manifest = design.pairing_manifest
            n_between = len(manifest['Group'].unique())
            n_within = len(manifest['Condition'].unique())
            output_file = os.path.join(results_dir, f"Stats_Parameters.csv")
            # Save mixed factorial results using new format (matches independent factorial)
            test = MixedANOVA()
            test.save_results(results, output_file, design, base_path)
            
    def _run_frequency_analyses(self, design: ExperimentalDesign, base_path: str) -> Dict[str, any]:
        """Run frequency-related statistical analyses."""
        
        frequency_results = {}
        
        try:
            # Route to dependent or independent analyzer based on design type
            if design.design_type in [DesignType.MIXED_FACTORIAL, DesignType.PAIRED_TWO_GROUP, DesignType.REPEATED_MEASURES]:
                # Dependent designs (mixed factorial and paired) use the dependent frequency analyzer
                # The analyzer internally detects paired vs mixed factorial
                current_analysis = self.frequency_analyzer_dependent.analyze_current_vs_frequency(design, base_path)
                frequency_results['current_vs_frequency'] = current_analysis
                
                rheobase_analysis = self.frequency_analyzer_dependent.analyze_fold_rheobase_vs_frequency(design, base_path)
                frequency_results['fold_rheobase_vs_frequency'] = rheobase_analysis
            else:
                # Independent designs use the standard frequency analyzer
                current_analysis = self.frequency_analyzer.analyze_current_vs_frequency(design, base_path)
                frequency_results['current_vs_frequency'] = current_analysis
                
                rheobase_analysis = self.frequency_analyzer.analyze_fold_rheobase_vs_frequency(design, base_path)
                frequency_results['fold_rheobase_vs_frequency'] = rheobase_analysis
            
            logger.info("Frequency analyses completed")
            
        except Exception as e:
            logger.error(f"Error in frequency analyses: {e}")
            frequency_results['error'] = str(e)
            
        return frequency_results
    
    def _run_attenuation_analysis(self, design: ExperimentalDesign, base_path: str) -> Dict[str, any]:
        """Run attenuation analysis."""
        
        try:
            if design.design_type in [DesignType.MIXED_FACTORIAL, DesignType.PAIRED_TWO_GROUP, DesignType.REPEATED_MEASURES]:
                # Dependent designs (mixed factorial and paired) use dependent attenuation analyzer
                # The analyzer internally detects paired vs mixed factorial
                attenuation_results = self.attenuation_analyzer_dependent.analyze_attenuation(design, base_path)
            else:
                # Independent designs use standard attenuation analyzer
                attenuation_results = self.attenuation_analyzer.analyze_attenuation(design, base_path)
            
            logger.info("Attenuation analysis completed")
            return attenuation_results
            
        except Exception as e:
            logger.error(f"Error in attenuation analysis: {e}")
            return {'error': str(e), 'success': False}
    
    def _collect_mixed_effects_results(self, frequency_results: Dict, attenuation_results: Dict) -> List[Dict]:
        """Collect all mixed-effects model results."""
        
        mixed_effects = []
        
        def _append_result(name: str, payload):
            if not payload:
                return
            
            if isinstance(payload, dict):
                if 'effects' in payload:
                    effect_rows = payload.get('effects')
                    posthoc_rows = payload.get('posthocs', [])
                else:
                    effect_rows = payload
                    posthoc_rows = payload.get('posthocs', [])
            else:
                effect_rows = payload
                posthoc_rows = []
            
            if not effect_rows:
                return
            
            mixed_effects.append({
                'name': name,
                'results': effect_rows,
                'posthocs': posthoc_rows or []
            })
        
        # Current vs frequency mixed effects
        if 'current_vs_frequency' in frequency_results:
            current_me = frequency_results['current_vs_frequency'].get('mixed_effects_result')
            _append_result('Current_vs_Frequency', current_me)
        
        # Fold rheobase vs frequency mixed effects
        if 'fold_rheobase_vs_frequency' in frequency_results:
            rheobase_me = frequency_results['fold_rheobase_vs_frequency'].get('mixed_effects_result')
            _append_result('Fold_Rheobase_vs_Frequency', rheobase_me)
        
        # AP number vs peak mixed effects
        if attenuation_results.get('success'):
            attenuation_me = attenuation_results.get('mixed_effects_result')
            _append_result('AP_Number_vs_Peak', attenuation_me)
        
        return mixed_effects
    
    def _save_mixed_effects_results(self, mixed_effects_results: List[Dict], base_path: str) -> None:
        """Save mixed-effects model results to CSV."""
        
        if not mixed_effects_results:
            logger.warning("No mixed-effects results to save")
            return
            
        try:
            output_path = os.path.join(base_path, self.config.output_dir, "Stats_across_frequencies_global_mixed_effects.csv")
            
            with open(output_path, 'w') as f:
                for i, result in enumerate(mixed_effects_results):
                    name = result['name']
                    data = result['results']
                    
                    f.write(f"=== {name} ===\n")
                    
                    # Convert to DataFrame and save
                    df = pd.DataFrame(data)
                    df.to_csv(f, index=False, lineterminator='\n')
                    f.write("\n\n")
                    
            logger.info(f"Saved mixed-effects results to {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving mixed-effects results: {e}")
    
    def _save_mixed_effects_posthoc_results(self, mixed_effects_results: List[Dict], base_path: str) -> None:
        """Save aggregated global mixed-effects post-hoc contrasts."""
        if not mixed_effects_results:
            return
        
        rows = []
        for entry in mixed_effects_results:
            analysis_name = entry.get('name')
            for posthoc in entry.get('posthocs', []) or []:
                record = {'Analysis': analysis_name}
                record.update(posthoc)
                rows.append(record)
        
        if not rows:
            logger.info("No global mixed-effects post-hoc results to save")
            return
        
        try:
            output_dir = os.path.join(base_path, self.config.output_dir)
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, "Stats_across_frequencies_global_mixed_effects_posthocs.csv")
            
            df = pd.DataFrame(rows)
            column_order = [
                'Analysis', 'Posthoc_Mode', 'Factor', 'Context', 'Comparison',
                'Level1', 'Level2', 'Delta_at_mean', 'Delta_per_SD',
                'chi2', 'df', 'p_value', 'corrected_p'
            ]
            available_cols = [col for col in column_order if col in df.columns]
            df = df[available_cols]
            df.to_csv(output_path, index=False)
            logger.info(f"Saved global mixed-effects post-hoc results to {output_path}")
        except Exception as e:
            logger.error(f"Error saving global mixed-effects post-hoc results: {e}")
    
    def _generate_plots(self, design: ExperimentalDesign, base_path: str, attenuation_data: Optional[Dict] = None) -> Dict[str, str]:
        """Generate all plots for the analysis."""
        
        plot_files = {}
        
        try:
            # Measurement scatter plots
            measurement_plots = self.plot_generator.create_measurement_plots(design, base_path)
            plot_files.update(measurement_plots)
            
            # Frequency plots
            frequency_plots = self.plot_generator.create_frequency_plots(design, base_path)
            plot_files.update(frequency_plots)
            
            # Burst analysis plot
            burst_plot = self.plot_generator.create_burst_analysis_plot(design, base_path)
            if burst_plot:
                plot_files['burst_analysis'] = burst_plot
            
            # Attenuation plot
            if attenuation_data and attenuation_data.get('attenuation_plot_data'):
                attenuation_plot = self.plot_generator.create_attenuation_plot(design, base_path, attenuation_data)
                if attenuation_plot:
                    plot_files['attenuation'] = attenuation_plot
                else:
                    logger.warning("Failed to create attenuation plot")
            else:
                logger.warning("No attenuation plot data available - check if attenuation analysis succeeded")
                
            logger.info(f"Generated {len(plot_files)} plots")
            
        except Exception as e:
            logger.error(f"Error generating plots: {e}")
            
        return plot_files
