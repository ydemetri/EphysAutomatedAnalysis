"""
Main application for the modular patch clamp analysis system.
Implements A vs B independent groups analysis.
"""

import sys
import os
sys.dont_write_bytecode = True

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading
import logging
import io
from typing import Optional, Dict
from contextlib import redirect_stdout, redirect_stderr

# Add the modular_analysis to path
sys.path.insert(0, os.path.dirname(__file__))

from modular_analysis.data_extraction.extractor import DataExtractor
from modular_analysis.statistical_analysis.analyzer import StatisticalAnalyzer
from modular_analysis.statistical_analysis.designs import DesignManager
from modular_analysis.shared.config import AnalysisConfig
from modular_analysis.shared.data_models import GroupInfo, DesignType
from modular_analysis.shared.utils import convert_manifest_wide_to_long, validate_manifest
import pandas as pd

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class GUIOutputCapture:
    """Captures stdout/stderr and redirects to GUI text widget."""
    
    def __init__(self, text_widget, root):
        self.text_widget = text_widget
        self.root = root
        self.buffer = io.StringIO()
        
    def write(self, text):
        if text:  # Include all text, including newlines
            # Schedule GUI update in main thread
            self.root.after(0, lambda t=text: self._add_text(t))
        return len(text)
    
    def flush(self):
        pass
    
    def _add_text(self, text):
        self.text_widget.config(state='normal')
        self.text_widget.insert(tk.END, text)
        self.text_widget.see(tk.END)  # Auto-scroll to bottom
        self.text_widget.config(state='disabled')


class ModularAnalysisApp:
    """Main application for modular patch clamp analysis."""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Modular Patch Clamp Analysis")
        self.root.geometry("900x600")
        
        # Configuration
        self.config = AnalysisConfig()
        self.extractor = DataExtractor(self.config)
        self.analyzer = StatisticalAnalyzer(self.config)
        
        # State variables
        self.base_path = tk.StringVar()
        self.available_groups = []
        self.selected_groups = []
        self.analysis_running = False
        
        # Measurement selection state (will be populated in setup_measurement_tab)
        self.measurement_vars = {}  # Dict[str, tk.BooleanVar] for each measurement
        self.category_vars = {}  # Dict[str, tk.BooleanVar] for Select All per category
        
        self.setup_ui()
        
    def setup_ui(self):
        """Set up the user interface."""
        
        # Main notebook for tabs
        notebook = ttk.Notebook(self.root)
        
        # Tab 1: Data Discovery
        discovery_frame = ttk.Frame(notebook)
        notebook.add(discovery_frame, text="Data Discovery")
        self.setup_discovery_tab(discovery_frame)
        
        # Tab 2: Data Extraction
        extraction_frame = ttk.Frame(notebook)
        notebook.add(extraction_frame, text="Data Extraction")
        self.setup_extraction_tab(extraction_frame)
        
        # Tab 3: Measurement Selection
        measurement_frame = ttk.Frame(notebook)
        notebook.add(measurement_frame, text="Measurement Selection")
        self.setup_measurement_tab(measurement_frame)
        
        # Tab 4: Statistical Analysis  
        analysis_frame = ttk.Frame(notebook)
        notebook.add(analysis_frame, text="Statistical Analysis")
        self.setup_analysis_tab(analysis_frame)
        
        notebook.pack(expand=True, fill='both', padx=10, pady=10)
        
    def setup_discovery_tab(self, parent):
        """Set up the data discovery tab."""
        
        # Path selection
        path_frame = ttk.LabelFrame(parent, text="Data Directory", padding=10)
        path_frame.pack(fill='x', padx=5, pady=5)
        
        ttk.Label(path_frame, text="Select folder containing group data:").pack(anchor='w')
        
        path_entry_frame = ttk.Frame(path_frame)
        path_entry_frame.pack(fill='x', pady=5)
        
        ttk.Entry(path_entry_frame, textvariable=self.base_path, width=60).pack(side='left', fill='x', expand=True)
        ttk.Button(path_entry_frame, text="Browse...", command=self.browse_directory).pack(side='right', padx=(5, 0))
        
        # Group detection (compact)
        groups_frame = ttk.LabelFrame(parent, text="Detected Groups", padding=10)
        groups_frame.pack(fill='x', padx=5, pady=5)
        
        self.groups_text = tk.Text(groups_frame, height=4, state='disabled')
        scrollbar = ttk.Scrollbar(groups_frame, orient="vertical", command=self.groups_text.yview)
        self.groups_text.configure(yscrollcommand=scrollbar.set)
        
        self.groups_text.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')
        
        # Scan button (large and centered)
        button_frame = ttk.Frame(parent)
        button_frame.pack(fill='x', padx=5, pady=20)
        
        scan_button = ttk.Button(button_frame, text="Scan Directory", command=self.scan_directory)
        scan_button.configure(width=20)  # Make button wider
        scan_button.pack(anchor='center')  # Center the button
        
    def setup_extraction_tab(self, parent):
        """Set up the data extraction tab."""
        
        # Instruction
        instruction_frame = ttk.LabelFrame(parent, text="Instructions", padding=10)
        instruction_frame.pack(fill='x', padx=5, pady=5)
        
        ttk.Label(instruction_frame, text="First, discover groups in the Data Discovery tab, then extract data here.").pack(anchor='w')
        
        # Extract button (centered)
        button_frame = ttk.Frame(parent)
        button_frame.pack(fill='x', padx=5, pady=30)
        
        extract_button = ttk.Button(button_frame, text="Extract Data", command=self.extract_data, width=20)
        extract_button.pack(anchor='center')
        
        # Progress bar (centered and with percentage)
        progress_frame = ttk.Frame(parent)
        progress_frame.pack(fill='x', padx=50, pady=20)
        
        self.extraction_progress = ttk.Progressbar(progress_frame, mode='determinate', length=400)
        self.extraction_progress.pack(anchor='center', pady=(0, 5))
        
        # Progress label
        self.extraction_progress_label = ttk.Label(progress_frame, text="Ready to extract data", font=('TkDefaultFont', 10))
        self.extraction_progress_label.pack(anchor='center')
        
        # Detailed progress output (scrollable)
        details_frame = ttk.LabelFrame(parent, text="Extraction Details", padding=10)
        details_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        self.extraction_details = tk.Text(details_frame, height=8, state='disabled', font=('Consolas', 9))
        details_scrollbar = ttk.Scrollbar(details_frame, orient="vertical", command=self.extraction_details.yview)
        self.extraction_details.configure(yscrollcommand=details_scrollbar.set)
        
        self.extraction_details.pack(side='left', fill='both', expand=True)
        details_scrollbar.pack(side='right', fill='y')
    
    def setup_measurement_tab(self, parent):
        """Set up the measurement selection tab."""
        from modular_analysis.shared.utils import get_measurement_categories
        
        # Center everything vertically and horizontally
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=1)
        
        # Main centered frame
        center_frame = ttk.Frame(parent)
        center_frame.grid(row=0, column=0)
        
        # Instructions
        instruction_frame = ttk.LabelFrame(center_frame, text="Instructions", padding=10)
        instruction_frame.pack(pady=(20, 10))
        
        ttk.Label(instruction_frame, 
                  text="Select which measurements to include in statistical analysis.\n"
                       "Reducing the number of measurements reduces the multiple comparison correction penalty.").pack()
        
        # Get measurement categories
        categories = get_measurement_categories()
        
        # Horizontal frame for categories side by side
        categories_frame = ttk.Frame(center_frame)
        categories_frame.pack(pady=10)
        
        # Create checkboxes for each category - side by side
        for category_name, measurements in categories.items():
            # Category frame
            cat_frame = ttk.LabelFrame(categories_frame, text=category_name, padding=10)
            cat_frame.pack(side='left', padx=10, anchor='n')
            
            # Select All checkbox for this category
            self.category_vars[category_name] = tk.BooleanVar(value=True)
            select_all_cb = ttk.Checkbutton(
                cat_frame, 
                text="Select All", 
                variable=self.category_vars[category_name],
                command=lambda cat=category_name, meas=measurements: self._toggle_category(cat, meas)
            )
            select_all_cb.pack(anchor='w')
            
            # Separator
            ttk.Separator(cat_frame, orient='horizontal').pack(fill='x', pady=5)
            
            # Individual measurement checkboxes
            for measurement in measurements:
                self.measurement_vars[measurement] = tk.BooleanVar(value=True)
                cb = ttk.Checkbutton(
                    cat_frame,
                    text=measurement,
                    variable=self.measurement_vars[measurement],
                    command=lambda cat=category_name, meas=measurements: self._update_category_checkbox(cat, meas)
                )
                cb.pack(anchor='w', padx=20)
        
        # Buttons frame - centered below
        button_frame = ttk.Frame(center_frame)
        button_frame.pack(pady=20)
        
        ttk.Button(button_frame, text="Select All", command=self._select_all_measurements).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Deselect All", command=self._deselect_all_measurements).pack(side='left', padx=5)
        
        # Summary label
        self.measurement_summary = ttk.Label(button_frame, text="")
        self.measurement_summary.pack(side='left', padx=20)
        self._update_measurement_summary()
    
    def _toggle_category(self, category_name: str, measurements: list):
        """Toggle all measurements in a category when Select All is clicked."""
        select_all = self.category_vars[category_name].get()
        for measurement in measurements:
            if measurement in self.measurement_vars:
                self.measurement_vars[measurement].set(select_all)
        self._update_measurement_summary()
    
    def _update_category_checkbox(self, category_name: str, measurements: list):
        """Update the Select All checkbox based on individual measurement selections."""
        all_selected = all(
            self.measurement_vars[m].get() 
            for m in measurements 
            if m in self.measurement_vars
        )
        self.category_vars[category_name].set(all_selected)
        self._update_measurement_summary()
    
    def _select_all_measurements(self):
        """Select all measurements."""
        for var in self.measurement_vars.values():
            var.set(True)
        for var in self.category_vars.values():
            var.set(True)
        self._update_measurement_summary()
    
    def _deselect_all_measurements(self):
        """Deselect all measurements."""
        for var in self.measurement_vars.values():
            var.set(False)
        for var in self.category_vars.values():
            var.set(False)
        self._update_measurement_summary()
    
    def _update_measurement_summary(self):
        """Update the summary label showing how many measurements are selected."""
        selected = sum(1 for var in self.measurement_vars.values() if var.get())
        total = len(self.measurement_vars)
        self.measurement_summary.config(text=f"{selected}/{total} measurements selected")
    
    def get_selected_measurements(self) -> list:
        """Get list of selected measurement names."""
        return [name for name, var in self.measurement_vars.items() if var.get()]
        
    def setup_analysis_tab(self, parent):
        """Set up the statistical analysis tab."""
        
        # Protocol parameters (moved from extraction tab since they're only used in analysis)
        protocol_frame = ttk.LabelFrame(parent, text="Protocol Parameters (for Analysis)", padding=10)
        protocol_frame.pack(fill='x', padx=5, pady=5)
        
        ttk.Label(protocol_frame, text="These settings are used to calculate current values during analysis:").pack(anchor='w', pady=(0,5))
        
        param_frame1 = ttk.Frame(protocol_frame)
        param_frame1.pack(fill='x')
        
        ttk.Label(param_frame1, text="Min Current (pA):").pack(side='left')
        self.min_current = tk.StringVar(value=str(self.config.protocol.min_current))
        ttk.Entry(param_frame1, textvariable=self.min_current, width=10).pack(side='left', padx=(5, 20))
        
        ttk.Label(param_frame1, text="Step Size (pA):").pack(side='left')
        self.step_size = tk.StringVar(value=str(self.config.protocol.step_size))
        ttk.Entry(param_frame1, textvariable=self.step_size, width=10).pack(side='left', padx=5)
        
        # Design selection
        design_frame = ttk.LabelFrame(parent, text="Experimental Design", padding=10)
        design_frame.pack(fill='x', padx=5, pady=5)
        
        ttk.Label(design_frame, text="Select experimental design:").pack(anchor='w')
        
        self.design_type = tk.StringVar(value="2 independent groups - 1 factor (between), 2 levels (A vs B)")
        design_combo = ttk.Combobox(design_frame, textvariable=self.design_type, 
                                   values=list(DesignManager.get_supported_designs().keys()),
                                   state='readonly', width=85)
        design_combo.pack(pady=5)
        
        # Group assignment
        assignment_frame = ttk.LabelFrame(parent, text="Group Assignment", padding=10)
        assignment_frame.pack(fill='x', padx=5, pady=5)
        
        # Available groups (left side)
        left_frame = ttk.Frame(assignment_frame)
        left_frame.pack(side='left', padx=5)
        
        ttk.Label(left_frame, text="Available Groups:").pack(anchor='w')
        self.available_listbox = tk.Listbox(left_frame, height=4, width=25)
        self.available_listbox.pack(pady=5)
        # Bind click to move to selected
        self.available_listbox.bind('<Button-1>', self.on_available_click)
        
        # Buttons (middle)
        button_frame = ttk.Frame(assignment_frame)
        button_frame.pack(side='left', padx=15, pady=20)
        
        ttk.Button(button_frame, text="→", command=self.add_group).pack(pady=5)
        ttk.Button(button_frame, text="←", command=self.remove_group).pack(pady=5)
        
        # Selected groups (right side)
        right_frame = ttk.Frame(assignment_frame)
        right_frame.pack(side='left', padx=5)
        
        ttk.Label(right_frame, text="Selected Groups:").pack(anchor='w')
        self.selected_listbox = tk.Listbox(right_frame, height=4, width=25)
        self.selected_listbox.pack(pady=5)
        # Bind click to move to available
        self.selected_listbox.bind('<Button-1>', self.on_selected_click)
        
        # Group properties (dynamic based on selected groups) - with scrolling
        self.properties_frame = ttk.LabelFrame(parent, text="Group Properties", padding=10)
        self.properties_frame.pack(fill='x', padx=5, pady=5)
        
        # Create a canvas with scrollbar for group properties
        properties_canvas = tk.Canvas(self.properties_frame, height=60)  # Fixed height - reduced to make room for button
        properties_scrollbar = ttk.Scrollbar(self.properties_frame, orient="vertical", command=properties_canvas.yview)
        self.properties_scrollable_frame = ttk.Frame(properties_canvas)
        
        self.properties_scrollable_frame.bind(
            "<Configure>",
            lambda e: properties_canvas.configure(scrollregion=properties_canvas.bbox("all"))
        )
        
        properties_canvas.create_window((0, 0), window=self.properties_scrollable_frame, anchor="nw")
        properties_canvas.configure(yscrollcommand=properties_scrollbar.set)
        
        properties_canvas.pack(side="left", fill="both", expand=True)
        properties_scrollbar.pack(side="right", fill="y")
        
        # Enable mousewheel scrolling (bind to canvas and scrollable frame)
        def on_mousewheel(event):
            properties_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        properties_canvas.bind("<Enter>", lambda e: properties_canvas.bind_all("<MouseWheel>", on_mousewheel))
        properties_canvas.bind("<Leave>", lambda e: properties_canvas.unbind_all("<MouseWheel>"))
        
        # Store references
        self.properties_canvas = properties_canvas
        
        # This will be populated dynamically based on selected groups
        self.group_color_vars = {}
        self.group_italic_vars = {}
        self.color_widgets = []
        
        # Analysis buttons
        analysis_button_frame = ttk.Frame(parent)
        analysis_button_frame.pack(fill='x', padx=5, pady=15)
        
        # Center the run analysis button
        run_button = ttk.Button(analysis_button_frame, text="Run Analysis", command=self.run_analysis, width=20)
        run_button.pack(anchor='center')
        
        # Progress bar (centered below button)
        progress_frame = ttk.Frame(parent)
        progress_frame.pack(fill='x', padx=50, pady=10)
        
        self.analysis_progress = ttk.Progressbar(progress_frame, mode='determinate', length=300)
        self.analysis_progress.pack(anchor='center')
        self.analysis_progress['value'] = 0  # Start completely grey/empty
        self.analysis_progress['maximum'] = 100
        
    def browse_directory(self):
        """Browse for data directory."""
        directory = filedialog.askdirectory(title="Select Data Directory")
        if directory:
            self.base_path.set(directory)
            
    def _update_extraction_progress(self, value, text, color="blue"):
        """Update extraction progress from any thread."""
        self.extraction_progress['value'] = value
        self.extraction_progress_label.config(text=text, foreground=color)
        
    def _add_extraction_detail(self, message, clear_first=False):
        """Add a detailed message to the extraction details area."""
        if clear_first:
            self.extraction_details.config(state='normal')
            self.extraction_details.delete(1.0, tk.END)
            self.extraction_details.config(state='disabled')
        
        self.extraction_details.config(state='normal')
        self.extraction_details.insert(tk.END, f"{message}\n")
        self.extraction_details.see(tk.END)  # Auto-scroll to bottom
        self.extraction_details.config(state='disabled')
            
    def scan_directory(self):
        """Scan directory for available groups."""
        if not self.base_path.get():
            messagebox.showerror("Error", "Please select a data directory first")
            return
            
        try:
            self.available_groups = self.extractor.scan_data_directory(self.base_path.get())
            
            # Update groups display
            self.groups_text.config(state='normal')
            self.groups_text.delete(1.0, tk.END)
            
            for group in self.available_groups:
                self.groups_text.insert(tk.END, f"✓ {group.name} ({group.n_files} files)\n")
                
            self.groups_text.config(state='disabled')
            
            # Update available groups listbox
            self.available_listbox.delete(0, tk.END)
            for group in self.available_groups:
                self.available_listbox.insert(tk.END, f"{group.name} ({group.n_files} files)")
                
            logger.info(f"Found {len(self.available_groups)} groups")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error scanning directory: {e}")
            logger.error(f"Error scanning directory: {e}")
            
    def extract_data(self):
        """Extract data from all groups."""
        if not self.available_groups:
            messagebox.showerror("Error", "Please scan directory first")
            return
            
        if self.analysis_running:
            messagebox.showwarning("Warning", "Analysis already running")
            return
            
        # Update configuration
        self.config.protocol.min_current = float(self.min_current.get())
        self.config.protocol.step_size = float(self.step_size.get())
        
        def extraction_worker():
            self.analysis_running = True
            
            try:
                # Initialize progress
                total_groups = len(self.available_groups)
                self.extraction_progress['maximum'] = 100
                self.extraction_progress['value'] = 0
                
                # Update UI and clear details
                self.root.after(0, lambda: self.extraction_progress_label.config(text=f"Starting extraction... (0/{total_groups} groups)", foreground="blue"))
                self.root.after(0, lambda: self._add_extraction_detail("Starting data extraction...\n", clear_first=True))
                
                # Set up output capture
                output_capture = GUIOutputCapture(self.extraction_details, self.root)
                
                # Extract groups one by one with real progress updates
                results = {}
                
                for i, group in enumerate(self.available_groups):
                    # Update progress before extraction
                    progress_percent = int((i / total_groups) * 100)
                    progress_text = f"Extracting group {group.name}... (group {i+1}/{total_groups}) - {progress_percent}%"
                    self.root.after(0, lambda p=progress_percent, text=progress_text: 
                                   self._update_extraction_progress(p, text))
                    
                    # Add group header to output
                    self.root.after(0, lambda name=group.name: self._add_extraction_detail(f"\n=== Processing Group: {name} ===\n"))
                    
                    # Capture stdout/stderr during extraction
                    old_stdout = sys.stdout
                    old_stderr = sys.stderr
                    
                    try:
                        # Redirect output to GUI
                        sys.stdout = output_capture
                        sys.stderr = output_capture
                        
                        # Extract this group (will capture all print statements)
                        extraction_success = self.extractor.extract_group_data(group, self.base_path.get())
                        results[group.name] = extraction_success if extraction_success is not None else False
                        
                    except Exception as e:
                        error_msg = f"Error processing {group.name}: {str(e)}\n"
                        self.root.after(0, lambda msg=error_msg: self._add_extraction_detail(msg))
                        logger.error(f"Exception during extraction of {group.name}: {e}", exc_info=True)
                        results[group.name] = False
                    finally:
                        # Restore stdout/stderr
                        sys.stdout = old_stdout
                        sys.stderr = old_stderr
                    
                    # Update progress after extraction
                    progress_percent = int(((i + 1) / total_groups) * 100)
                    success_status = "✓ Success" if results[group.name] else "✗ Failed"
                    progress_text = f"Completed {i+1}/{total_groups} groups - {progress_percent}%"
                    self.root.after(0, lambda p=progress_percent, text=progress_text: 
                                   self._update_extraction_progress(p, text))
                    self.root.after(0, lambda name=group.name, status=success_status: 
                                   self._add_extraction_detail(f"\n{name}: {status}\n"))
                
                # Count successful extractions (True = 1, False = 0)
                successful = sum(1 for v in results.values() if v is True)
                total = len(results)
                
                # Add final summary
                self.root.after(0, lambda: self._add_extraction_detail(f"\n=== EXTRACTION COMPLETE ===\n"))
                self.root.after(0, lambda succ=successful, tot=total: 
                               self._add_extraction_detail(f"Successfully processed: {succ}/{tot} groups\n"))
                
                # Update final status
                if successful == total:
                    status_text = f"✓ All {total} groups extracted successfully! (100%)"
                    status_color = "green"
                else:
                    status_text = f"⚠ {successful}/{total} groups extracted successfully (100%)"
                    status_color = "orange"
                    failed_groups = [name for name, success in results.items() if not success]
                    self.root.after(0, lambda failed=failed_groups: 
                                   self._add_extraction_detail(f"Failed groups: {', '.join(failed)}\n"))
                
                self.root.after(0, lambda text=status_text, color=status_color: 
                               self._update_extraction_progress(100, text, color))
                self.root.after(0, lambda: messagebox.showinfo("Extraction Complete", 
                                                              f"Data extraction completed\n"
                                                              f"Successful: {successful}/{total} groups"))
                
            except Exception as e:
                error_msg = f"Error during extraction: {e}"
                self.root.after(0, lambda: self._update_extraction_progress(0, "Extraction failed!", "red"))
                self.root.after(0, lambda msg=error_msg: self._add_extraction_detail(f"\n✗ EXTRACTION FAILED: {msg}\n"))
                self.root.after(0, lambda msg=error_msg: messagebox.showerror("Error", msg))
                logger.error(f"Error during extraction: {e}")
            finally:
                self.analysis_running = False
                
        thread = threading.Thread(target=extraction_worker)
        thread.daemon = True
        thread.start()
        
    def add_group(self):
        """Add selected group to analysis."""
        selection = self.available_listbox.curselection()
        if not selection:
            return
            
        # Check max groups based on selected design
        max_groups = self._get_max_groups_for_design()
        if len(self.selected_groups) >= max_groups:
            design_name = self.design_type.get()
            messagebox.showwarning("Warning", f"Maximum {max_groups} groups allowed for {design_name}")
            return
            
        idx = selection[0]
        group = self.available_groups[idx]
        
        if group not in self.selected_groups:
            self.selected_groups.append(group)
            # Assign default colors
            self._assign_default_colors()
            self.update_selected_groups()
            
    def remove_group(self):
        """Remove selected group from analysis."""
        selection = self.selected_listbox.curselection()
        if not selection:
            return
            
        idx = selection[0]
        if 0 <= idx < len(self.selected_groups):
            self.selected_groups.pop(idx)
            self._assign_default_colors()  # Reassign colors after removal
            self.update_selected_groups()
    
    def on_available_click(self, event):
        """Handle click on available groups listbox - move to selected."""
        # Get the clicked item index
        listbox = event.widget
        selection = listbox.nearest(event.y)
        if selection < 0 or selection >= len(self.available_groups):
            return
            
        listbox.selection_clear(0, tk.END)
        listbox.selection_set(selection)
        
        # Check max groups based on selected design
        max_groups = self._get_max_groups_for_design()
        if len(self.selected_groups) >= max_groups:
            design_name = self.design_type.get()
            messagebox.showwarning("Warning", f"Maximum {max_groups} groups allowed for {design_name}")
            return
            
        group = self.available_groups[selection]
        
        if group not in self.selected_groups:
            self.selected_groups.append(group)
            # Assign default colors
            self._assign_default_colors()
            self.update_selected_groups()
    
    def on_selected_click(self, event):
        """Handle click on selected groups listbox - move to available."""
        # Get the clicked item index
        listbox = event.widget
        selection = listbox.nearest(event.y)
        if selection < 0 or selection >= len(self.selected_groups):
            return
            
        listbox.selection_clear(0, tk.END)
        listbox.selection_set(selection)
        
        if 0 <= selection < len(self.selected_groups):
            self.selected_groups.pop(selection)
            self._assign_default_colors()  # Reassign colors after removal
            self.update_selected_groups()
            
    def update_selected_groups(self):
        """Update the selected groups display."""
        self.selected_listbox.delete(0, tk.END)
        for group in self.selected_groups:
            self.selected_listbox.insert(tk.END, f"{group.name} ({group.n_files} files)")
        
        # Update group properties interface
        self._update_group_properties_interface()
            
    def run_analysis(self):
        """Run statistical analysis."""
        design_name = self.design_type.get()
        min_groups = self._get_min_groups_for_design()
        max_groups = self._get_max_groups_for_design()
        
        if len(self.selected_groups) < min_groups:
            messagebox.showerror("Error", f"Please select at least {min_groups} groups for {design_name}")
            return
        elif len(self.selected_groups) > max_groups:
            messagebox.showerror("Error", f"Please select at most {max_groups} groups for {design_name}")
            return
            
        if self.analysis_running:
            messagebox.showwarning("Warning", "Analysis already running")
            return
            
        # Update protocol configuration from GUI before analysis
        try:
            self.config.protocol.min_current = float(self.min_current.get())
            self.config.protocol.step_size = float(self.step_size.get())
            logger.info(f"Updated protocol: min_current={self.config.protocol.min_current}, step_size={self.config.protocol.step_size}")
        except ValueError as e:
            messagebox.showerror("Error", f"Invalid protocol values: {e}")
            return
            
        if not self.base_path.get():
            messagebox.showerror("Error", "Please select a data directory first")
            return
        
        # Get factorial mapping BEFORE starting worker thread (if needed)
        factor_mapping = None
        mixed_mapping = None
        paired_manifest = None
        
        if "Mixed factorial" in design_name:
            # Mixed factorial design - get manifest and factor names
            mixed_mapping = self._get_mixed_design_mapping()
            if not mixed_mapping:
                messagebox.showerror("Error", "Mixed factorial design requires manifest")
                return
        elif "Paired design" in design_name:
            # Paired two-group design - get manifest
            paired_manifest = self._get_paired_design_info()
            if not paired_manifest:
                messagebox.showerror("Error", "Paired design requires manifest")
                return
        elif "Repeated measures" in design_name:
            # Repeated measures design - get manifest
            rm_manifest = self._get_repeated_measures_design_info()
            if not rm_manifest:
                messagebox.showerror("Error", "Repeated measures design requires manifest")
                return
        elif "Factorial design" in design_name and len(self.selected_groups) >= 4:
            # Independent factorial design (N×M)
            factor_mapping = self._get_factorial_mapping()
            if not factor_mapping:
                messagebox.showerror("Error", "Factorial design requires factor assignment")
                return
            
        def analysis_worker():
            self.analysis_running = True
            # Set progress bar to show activity (indeterminate-like animation)
            self.analysis_progress.configure(mode='indeterminate')
            self.analysis_progress.start()
            
            try:
                # Color mapping (display name to actual color value)
                color_display_to_value = {
                    'blue': 'blue',
                    'red': 'red',
                    'green': 'green',
                    'purple': 'purple',
                    'orange': 'orange',
                    'brown': 'brown',
                    'pink': 'pink',
                    'gray': 'gray',
                    'olive': 'olive',
                    'cyan': 'cyan',
                    'medium blue': '#0080FF',
                    'light blue': '#87CEEB',
                    'light red': '#FFA07A'
                }
                
                # Set group properties
                for i, group in enumerate(self.selected_groups):
                    if group.name in self.group_color_vars:
                        display_name = self.group_color_vars[group.name].get()
                        group.color = color_display_to_value.get(display_name, 'blue')
                    if group.name in self.group_italic_vars:
                        group.italic = self.group_italic_vars[group.name].get() == "yes"
                
                # Create experimental design based on design type and number of groups
                design_name = self.design_type.get()
                
                if "Mixed factorial" in design_name:
                    # Dependent factorial design with manifest
                    design = DesignManager.create_mixed_factorial(
                        self.selected_groups,
                        mixed_mapping['manifest_path'],
                        mixed_mapping['between_factor'],
                        mixed_mapping['within_factor'],
                        base_path=self.base_path.get(),
                        level_italic=mixed_mapping.get('level_italic', {})
                    )
                elif "Paired design" in design_name:
                    # Paired two-group design with manifest
                    design = DesignManager.create_paired_two_group(
                        self.selected_groups[0],
                        self.selected_groups[1],
                        paired_manifest['manifest_path'],
                        factor_name=paired_manifest['factor_name'],
                        base_path=self.base_path.get()
                    )
                elif "Repeated measures" in design_name:
                    # Repeated measures design with manifest
                    design = DesignManager.create_repeated_measures_multi_group(
                        self.selected_groups,
                        rm_manifest['manifest_path'],
                        factor_name=rm_manifest['factor_name'],
                        base_path=self.base_path.get()
                    )
                elif "2 independent groups" in design_name:
                    design = DesignManager.create_two_group_independent(
                        self.selected_groups[0], self.selected_groups[1]
                    )
                elif "3+ independent groups" in design_name:
                    design = DesignManager.create_multi_group_independent(
                        self.selected_groups
                    )
                elif "Factorial design" in design_name and len(self.selected_groups) >= 4:
                    # Use the factor mapping obtained before thread started
                    design = DesignManager.create_factorial_2x2(
                        self.selected_groups,
                        factor_mapping['factor1_name'],
                        factor_mapping['factor2_name'],
                        factor_mapping['mapping'],
                        level_italic=factor_mapping.get('level_italic', {})
                    )
                
                # Get selected measurements from GUI
                selected_measurements = self.get_selected_measurements()
                
                # Run analysis
                results = self.analyzer.run_analysis(design, self.base_path.get(), selected_measurements)
                
                if results['success']:
                    self.root.after(0, lambda: messagebox.showinfo("Analysis Complete", "Done"))
                else:
                    error_msg = results.get('error', 'Unknown error')
                    self.root.after(0, lambda: messagebox.showerror("Analysis Failed", f"Analysis failed: {error_msg}"))
                    
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", f"Error during analysis: {e}"))
                logger.error(f"Error during analysis: {e}")
            finally:
                self.analysis_running = False
                self.analysis_progress.stop()
                # Reset progress bar to grey/empty state
                self.analysis_progress.configure(mode='determinate')
                self.analysis_progress['value'] = 0
                
        thread = threading.Thread(target=analysis_worker)
        thread.daemon = True
        thread.start()
        
    def _get_min_groups_for_design(self) -> int:
        """Get minimum number of groups for selected design type."""
        design_name = self.design_type.get()
        if "2 independent groups" in design_name or "Paired design" in design_name:
            return 2
        elif "Mixed factorial" in design_name:
            return 4  # Minimum N×M is 2×2 = 4 groups
        elif "Factorial design" in design_name:
            return 4  # Minimum N×M is 2×2 = 4 groups
        elif "3+ independent groups" in design_name:
            return 3
        elif "Repeated measures" in design_name:
            return 3
        else:
            return 2  # Default
    
    def _get_max_groups_for_design(self) -> int:
        """Get maximum number of groups for selected design type."""
        design_name = self.design_type.get()
        if "2 independent groups" in design_name or "Paired design" in design_name:
            return 2
        elif "Repeated measures" in design_name:
            return 20  # Allow up to many repeated measures conditions
        elif "Mixed factorial" in design_name:
            return 20  # Allow up to large mixed designs (e.g., 2×10, 4×5)
        elif "Factorial design" in design_name:
            return 20  # Allow up to large factorial designs (e.g., 4×5, 2×10)
        elif "3+ independent groups" in design_name:
            return 10  # Reasonable upper limit
        else:
            return 2  # Default
    
    def _assign_default_colors(self):
        """Assign default colors to selected groups."""
        default_colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan', '#87CEEB', '#FFA07A']
        
        for i, group in enumerate(self.selected_groups):
            if i < len(default_colors):
                group.color = default_colors[i]
            else:
                group.color = default_colors[i % len(default_colors)]
            group.italic = False  # Default to no italics
    
    def _update_group_properties_interface(self):
        """Update the group properties interface based on selected groups."""
        # Clear existing widgets
        for widget in self.color_widgets:
            widget.destroy()
        self.color_widgets.clear()
        
        if not self.selected_groups:
            return
            
        # Create color/italic controls for each selected group
        # Map display names to actual color values (hex codes for custom colors)
        color_display_to_value = {
            'blue': 'blue',
            'red': 'red',
            'green': 'green',
            'purple': 'purple',
            'orange': 'orange',
            'brown': 'brown',
            'pink': 'pink',
            'gray': 'gray',
            'olive': 'olive',
            'cyan': 'cyan',
            'medium blue': '#0080FF',
            'light blue': '#87CEEB',
            'light red': '#FFA07A'
        }
        color_options = list(color_display_to_value.keys())
        
        for i, group in enumerate(self.selected_groups):
            # Create frame for this group's properties
            group_frame = ttk.Frame(self.properties_scrollable_frame)
            group_frame.pack(fill='x', pady=2)
            self.color_widgets.append(group_frame)
            
            # Group name label
            name_label = ttk.Label(group_frame, text=f"{group.name}:")
            name_label.pack(side='left')
            
            # Color selection
            color_label = ttk.Label(group_frame, text="Color:")
            color_label.pack(side='left', padx=(20, 5))
            
            if group.name not in self.group_color_vars:
                # Convert current color value to display name
                current_color = group.color or 'blue'
                # Find display name for this color value
                display_name = 'blue'  # default
                for name, value in color_display_to_value.items():
                    if value == current_color:
                        display_name = name
                        break
                self.group_color_vars[group.name] = tk.StringVar(value=display_name)
                
            color_combo = ttk.Combobox(group_frame, textvariable=self.group_color_vars[group.name],
                                     values=color_options, width=12, state='readonly')
            color_combo.pack(side='left', padx=5)
            
            # Italic selection
            italic_label = ttk.Label(group_frame, text="Italic:")
            italic_label.pack(side='left', padx=(20, 5))
            
            if group.name not in self.group_italic_vars:
                self.group_italic_vars[group.name] = tk.StringVar(value='no')
                
            italic_combo = ttk.Combobox(group_frame, textvariable=self.group_italic_vars[group.name],
                                      values=['no', 'yes'], width=8, state='readonly')
            italic_combo.pack(side='left', padx=5)
    
    def _get_factorial_mapping(self) -> Optional[Dict]:
        """Show dialog to get factorial design mappings."""
        dialog = tk.Toplevel(self.root)
        dialog.title("Factorial Design Setup - Step 2 of 2")
        dialog.geometry("700x650")
        dialog.transient(self.root)
        dialog.grab_set()
        
        result = {}
        
        # Factor names
        ttk.Label(dialog, text="Factor Names", font=('TkDefaultFont', 12, 'bold')).pack(pady=(10, 5))
        
        factor_frame = ttk.Frame(dialog)
        factor_frame.pack(pady=10)
        
        ttk.Label(factor_frame, text="Factor 1 Name:").grid(row=0, column=0, padx=5, pady=5, sticky='e')
        factor1_entry = ttk.Entry(factor_frame, width=20)
        factor1_entry.grid(row=0, column=1, padx=5, pady=5)
        factor1_entry.insert(0, "Genotype")
        
        ttk.Label(factor_frame, text="Factor 2 Name:").grid(row=1, column=0, padx=5, pady=5, sticky='e')
        factor2_entry = ttk.Entry(factor_frame, width=20)
        factor2_entry.grid(row=1, column=1, padx=5, pady=5)
        factor2_entry.insert(0, "Treatment")
        
        # Group assignments
        ttk.Label(dialog, text="Group Factor Assignments", font=('TkDefaultFont', 12, 'bold')).pack(pady=(20, 5))
        
        mapping_frame = ttk.Frame(dialog)
        mapping_frame.pack(pady=10)
        
        ttk.Label(mapping_frame, text="Group", font=('TkDefaultFont', 10, 'bold')).grid(row=0, column=0, padx=10, pady=5)
        ttk.Label(mapping_frame, text="Factor 1 Level", font=('TkDefaultFont', 10, 'bold')).grid(row=0, column=1, padx=10, pady=5)
        ttk.Label(mapping_frame, text="Factor 2 Level", font=('TkDefaultFont', 10, 'bold')).grid(row=0, column=2, padx=10, pady=5)
        
        group_factor1_vars = {}
        group_factor2_vars = {}
        level_italic_vars = {}  # Track italic checkboxes for each unique level
        
        for i, group in enumerate(self.selected_groups):
            ttk.Label(mapping_frame, text=group.name).grid(row=i+1, column=0, padx=10, pady=5)
            
            f1_var = tk.StringVar()
            f1_entry = ttk.Entry(mapping_frame, textvariable=f1_var, width=15)
            f1_entry.grid(row=i+1, column=1, padx=10, pady=5)
            group_factor1_vars[group.name] = f1_var
            
            # Add italic checkbox for this factor1 level (if not already added)
            # We'll dynamically manage these after levels are entered
            
            f2_var = tk.StringVar()
            f2_entry = ttk.Entry(mapping_frame, textvariable=f2_var, width=15)
            f2_entry.grid(row=i+1, column=2, padx=10, pady=5)
            group_factor2_vars[group.name] = f2_var
        
        # Italics section (will be populated as levels are entered)
        ttk.Label(dialog, text="Italicize Levels", font=('TkDefaultFont', 12, 'bold')).pack(pady=(20, 5))
        ttk.Label(dialog, text="(Checkboxes appear automatically as you type levels above)", font=('TkDefaultFont', 8)).pack()
        
        italic_frame = ttk.Frame(dialog)
        italic_frame.pack(pady=10)
        
        # We'll dynamically populate this based on entered levels
        level_italic_checkboxes = {}
        
        def update_italic_checkboxes(*args):
            """Update italic checkboxes based on current level entries, preserving existing selections."""
            # Save current selections
            current_selections = {level: var.get() for level, var in level_italic_checkboxes.items()}
            
            # Clear existing checkboxes
            for widget in italic_frame.winfo_children():
                widget.destroy()
            level_italic_checkboxes.clear()
            
            # Collect unique levels
            all_levels = set()
            for group in self.selected_groups:
                f1_level = group_factor1_vars[group.name].get().strip()
                f2_level = group_factor2_vars[group.name].get().strip()
                if f1_level:
                    all_levels.add(f1_level)
                if f2_level:
                    all_levels.add(f2_level)
            
            # Create checkboxes for each unique level, restoring previous selections
            if all_levels:
                for i, level in enumerate(sorted(all_levels)):
                    # Restore previous selection if level existed before
                    var = tk.BooleanVar(value=current_selections.get(level, False))
                    checkbox = ttk.Checkbutton(italic_frame, text=level, variable=var)
                    checkbox.grid(row=i//2, column=i%2, padx=20, pady=2, sticky='w')
                    level_italic_checkboxes[level] = var
        
        # Bind update function to all entry fields
        for group in self.selected_groups:
            group_factor1_vars[group.name].trace_add('write', update_italic_checkboxes)
            group_factor2_vars[group.name].trace_add('write', update_italic_checkboxes)
        
        # Status label
        status_label = ttk.Label(dialog, text="", foreground="red")
        status_label.pack(pady=5)
        
        def on_ok():
            factor1_name = factor1_entry.get().strip()
            factor2_name = factor2_entry.get().strip()
            
            if not factor1_name or not factor2_name:
                status_label.config(text="Please enter both factor names")
                return
            
            # Build mapping
            mapping = {}
            for group in self.selected_groups:
                f1_level = group_factor1_vars[group.name].get().strip()
                f2_level = group_factor2_vars[group.name].get().strip()
                
                if not f1_level or not f2_level:
                    status_label.config(text=f"Please assign factor levels for {group.name}")
                    return
                
                mapping[group.name] = {
                    'factor1': f1_level,
                    'factor2': f2_level
                }
            
            # Validate at least 2 levels per factor
            f1_levels = set(m['factor1'] for m in mapping.values())
            f2_levels = set(m['factor2'] for m in mapping.values())
            
            if len(f1_levels) < 2:
                status_label.config(text=f"{factor1_name} must have at least 2 levels (found {len(f1_levels)})")
                return
            if len(f2_levels) < 2:
                status_label.config(text=f"{factor2_name} must have at least 2 levels (found {len(f2_levels)})")
                return
            
            # Validate group count matches N×M structure
            expected_groups = len(f1_levels) * len(f2_levels)
            if len(self.selected_groups) != expected_groups:
                status_label.config(text=f"{len(f1_levels)}×{len(f2_levels)} design requires {expected_groups} groups (selected {len(self.selected_groups)})")
                return
            
            # Validate each group has a unique combination
            level_combinations = set((m['factor1'], m['factor2']) for m in mapping.values())
            if len(level_combinations) != len(self.selected_groups):
                status_label.config(text="Each group must have a unique combination of factor levels")
                return
            
            # Build level_italic mapping
            level_italic = {}
            for level, var in level_italic_checkboxes.items():
                level_italic[level] = var.get()
            
            result['factor1_name'] = factor1_name
            result['factor2_name'] = factor2_name
            result['level_italic'] = level_italic
            result['mapping'] = mapping
            dialog.destroy()
        
        def on_cancel():
            dialog.destroy()
        
        # Buttons
        button_frame = ttk.Frame(dialog)
        button_frame.pack(pady=20)
        
        ttk.Button(button_frame, text="Run Analysis", command=on_ok, width=15).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Cancel", command=on_cancel, width=15).pack(side='left', padx=5)
        
        # Wait for dialog to close
        self.root.wait_window(dialog)
        
        return result if result else None
    
    def _get_paired_design_info(self) -> Optional[str]:
        """Show dialog to upload manifest for paired two-group design."""
        dialog = tk.Toplevel(self.root)
        dialog.title("Paired Design Setup")
        dialog.geometry("600x550")
        dialog.transient(self.root)
        dialog.grab_set()
        
        main_frame = ttk.Frame(dialog, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Display selected groups (conditions)
        ttk.Label(main_frame, text="Selected Conditions:", font=('TkDefaultFont', 10, 'bold')).pack(anchor='w', pady=(0,5))
        for i, group in enumerate(self.selected_groups, 1):
            ttk.Label(main_frame, text=f"  {i}. {group.name}").pack(anchor='w')
        
        ttk.Separator(main_frame, orient='horizontal').pack(fill='x', pady=10)
        
        # Factor name input
        ttk.Label(main_frame, text="Factor Name:", font=('TkDefaultFont', 10, 'bold')).pack(anchor='w', pady=(0,5))
        ttk.Label(main_frame, text="Name for the within-subjects factor (e.g., 'Temperature', 'Treatment', 'Heat'):", 
                 wraplength=550).pack(anchor='w', padx=(10,0))
        
        factor_frame = ttk.Frame(main_frame)
        factor_frame.pack(fill='x', pady=(5,10))
        factor_name = tk.StringVar(value="Condition")
        ttk.Entry(factor_frame, textvariable=factor_name, width=30).pack(side=tk.LEFT)
        
        ttk.Separator(main_frame, orient='horizontal').pack(fill='x', pady=10)
        
        # Manifest file upload
        ttk.Label(main_frame, text="Upload Pairing Manifest:", font=('TkDefaultFont', 10, 'bold')).pack(anchor='w', pady=(5,5))
        ttk.Label(main_frame, text="The manifest should be an Excel file with:", wraplength=550).pack(anchor='w', padx=(10,0))
        ttk.Label(main_frame, text="• Subject_ID column", wraplength=550).pack(anchor='w', padx=(20,0))
        ttk.Label(main_frame, text=f"• Columns matching condition names: {self.selected_groups[0].name}, {self.selected_groups[1].name}", 
                 wraplength=550).pack(anchor='w', padx=(20,0))
        ttk.Label(main_frame, text="• File names in the cells (e.g., 'cell1.abf')", wraplength=550).pack(anchor='w', padx=(20,0))
        ttk.Label(main_frame, text="• IMPORTANT: Condition column names must EXACTLY match folder names!", 
                 wraplength=550, foreground="red", font=('TkDefaultFont', 9, 'bold')).pack(anchor='w', padx=(20,0))
        
        manifest_frame = ttk.Frame(main_frame)
        manifest_frame.pack(fill='x', pady=10)
        
        manifest_path = tk.StringVar()
        ttk.Entry(manifest_frame, textvariable=manifest_path, width=50).pack(side=tk.LEFT, padx=(0,5))
        
        def browse_manifest():
            filename = filedialog.askopenfilename(
                title="Select Pairing Manifest",
                filetypes=[("Excel files", "*.xlsx *.xls"), ("All files", "*.*")]
            )
            if filename:
                manifest_path.set(filename)
        
        ttk.Button(manifest_frame, text="Browse...", command=browse_manifest).pack(side=tk.LEFT)
        
        # Result storage
        result = [None]
        
        def on_ok():
            if not manifest_path.get():
                messagebox.showerror("Error", "Please select a manifest file")
                return
            
            if not factor_name.get().strip():
                messagebox.showerror("Error", "Please provide a factor name")
                return
            
            # Validate manifest columns match folder names
            try:
                import pandas as pd
                df = pd.read_excel(manifest_path.get())
                
                # Get condition columns (exclude Subject_ID and Group columns)
                exclude_cols = {'subject_id', 'subject id', 'group'}
                condition_cols = [col for col in df.columns if str(col).strip().lower() not in exclude_cols]
                
                # Extract condition names (parse "label: value" format if present, e.g., "condition 1: 32" -> "32")
                manifest_conditions = []
                for col in condition_cols:
                    col_str = str(col).strip()
                    if ':' in col_str:
                        condition = col_str.split(':', 1)[1].strip()
                    else:
                        condition = col_str
                    manifest_conditions.append(condition)
                
                # Check if manifest conditions match selected folder names
                selected_folder_names = [g.name for g in self.selected_groups]
                
                missing_folders = []
                for cond in manifest_conditions:
                    if cond not in selected_folder_names:
                        missing_folders.append(cond)
                
                if missing_folders:
                    messagebox.showerror(
                        "Folder Name Mismatch",
                        f"Manifest condition(s) do not match selected folder names:\n\n"
                        f"Missing folders: {', '.join(missing_folders)}\n\n"
                        f"Manifest conditions: {', '.join(manifest_conditions)}\n"
                        f"Selected folders: {', '.join(selected_folder_names)}\n\n"
                        f"Please ensure folder names EXACTLY match manifest condition names."
                    )
                    return
         
            except Exception as e:
                messagebox.showerror("Error", f"Failed to validate manifest: {str(e)}")
                return
            
            result[0] = {
                'manifest_path': manifest_path.get(),
                'factor_name': factor_name.get().strip()
            }
            dialog.destroy()
        
        def on_cancel():
            dialog.destroy()
        
        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(side=tk.BOTTOM, pady=10)
        ttk.Button(button_frame, text="OK", command=on_ok).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=on_cancel).pack(side=tk.LEFT, padx=5)
        
        # Wait for dialog to close
        self.root.wait_window(dialog)
        
        return result[0]
    
    def _get_repeated_measures_design_info(self) -> Optional[Dict]:
        """Show dialog to upload manifest for repeated measures (3+ groups) design."""
        dialog = tk.Toplevel(self.root)
        dialog.title("Repeated Measures Design Setup")
        dialog.geometry("600x600")
        dialog.transient(self.root)
        dialog.grab_set()
        
        main_frame = ttk.Frame(dialog, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Display selected groups (conditions)
        ttk.Label(main_frame, text="Selected Conditions:", font=('TkDefaultFont', 10, 'bold')).pack(anchor='w', pady=(0,5))
        for i, group in enumerate(self.selected_groups, 1):
            ttk.Label(main_frame, text=f"  {i}. {group.name}").pack(anchor='w')
        
        ttk.Separator(main_frame, orient='horizontal').pack(fill='x', pady=10)
        
        # Factor name input
        ttk.Label(main_frame, text="Factor Name:", font=('TkDefaultFont', 10, 'bold')).pack(anchor='w', pady=(0,5))
        ttk.Label(main_frame, text="Name for the within-subjects factor (e.g., 'Temperature', 'Time', 'Dose'):", 
                 wraplength=550).pack(anchor='w', padx=(10,0))
        
        factor_frame = ttk.Frame(main_frame)
        factor_frame.pack(fill='x', pady=(5,10))
        factor_name = tk.StringVar(value="Condition")
        ttk.Entry(factor_frame, textvariable=factor_name, width=30).pack(side=tk.LEFT)
        
        ttk.Separator(main_frame, orient='horizontal').pack(fill='x', pady=10)
        
        # Manifest file upload
        ttk.Label(main_frame, text="Upload Pairing Manifest:", font=('TkDefaultFont', 10, 'bold')).pack(anchor='w', pady=(5,5))
        ttk.Label(main_frame, text="The manifest should be an Excel file with:", wraplength=550).pack(anchor='w', padx=(10,0))
        ttk.Label(main_frame, text="• Subject_ID column", wraplength=550).pack(anchor='w', padx=(20,0))
        condition_names = ", ".join([g.name for g in self.selected_groups])
        ttk.Label(main_frame, text=f"• Columns for each condition: {condition_names}", 
                 wraplength=550).pack(anchor='w', padx=(20,0))
        ttk.Label(main_frame, text="• IMPORTANT: Condition column names must EXACTLY match folder names!", 
                 wraplength=550, foreground="red", font=('TkDefaultFont', 9, 'bold')).pack(anchor='w', padx=(20,0))
        ttk.Label(main_frame, text="• File names in the cells (e.g., 'cell1.abf')", wraplength=550).pack(anchor='w', padx=(20,0))
        ttk.Label(main_frame, text="• Each subject must have measurements in ALL conditions", wraplength=550).pack(anchor='w', padx=(20,0))
        
        manifest_frame = ttk.Frame(main_frame)
        manifest_frame.pack(fill='x', pady=10)
        
        manifest_path = tk.StringVar()
        ttk.Entry(manifest_frame, textvariable=manifest_path, width=50).pack(side=tk.LEFT, padx=(0,5))
        
        def browse_manifest():
            filename = filedialog.askopenfilename(
                title="Select Pairing Manifest",
                filetypes=[("Excel files", "*.xlsx *.xls"), ("All files", "*.*")]
            )
            if filename:
                manifest_path.set(filename)
        
        ttk.Button(manifest_frame, text="Browse...", command=browse_manifest).pack(side=tk.LEFT)
        
        # Result storage
        result = [None]
        
        def on_ok():
            if not manifest_path.get():
                messagebox.showerror("Error", "Please select a manifest file")
                return
            
            if not factor_name.get().strip():
                messagebox.showerror("Error", "Please enter a factor name")
                return
            
            # Validate manifest columns match folder names
            try:
                import pandas as pd
                df = pd.read_excel(manifest_path.get())
                
                # Get condition columns (exclude Subject_ID and Group columns)
                exclude_cols = {'subject_id', 'subject id', 'group'}
                condition_cols = [col for col in df.columns if str(col).strip().lower() not in exclude_cols]
                
                # Extract condition names (parse "label: value" format if present, e.g., "condition 1: 32" -> "32")
                manifest_conditions = []
                for col in condition_cols:
                    col_str = str(col).strip()
                    if ':' in col_str:
                        condition = col_str.split(':', 1)[1].strip()
                    else:
                        condition = col_str
                    manifest_conditions.append(condition)
                
                # Check if manifest conditions match selected folder names
                selected_folder_names = [g.name for g in self.selected_groups]
                
                missing_folders = []
                for cond in manifest_conditions:
                    if cond not in selected_folder_names:
                        missing_folders.append(cond)
                
                if missing_folders:
                    messagebox.showerror(
                        "Folder Name Mismatch",
                        f"Manifest condition(s) do not match selected folder names:\n\n"
                        f"Missing folders: {', '.join(missing_folders)}\n\n"
                        f"Manifest conditions: {', '.join(manifest_conditions)}\n"
                        f"Selected folders: {', '.join(selected_folder_names)}\n\n"
                        f"Please ensure folder names EXACTLY match manifest condition names."
                    )
                    return
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to validate manifest: {str(e)}")
                return
            
            result[0] = {
                'manifest_path': manifest_path.get(),
                'factor_name': factor_name.get().strip()
            }
            dialog.destroy()
        
        def on_cancel():
            dialog.destroy()
        
        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(side=tk.BOTTOM, pady=10)
        ttk.Button(button_frame, text="OK", command=on_ok).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=on_cancel).pack(side=tk.LEFT, padx=5)
        
        # Wait for dialog to close
        self.root.wait_window(dialog)
        
        return result[0]
    
    def _get_mixed_design_mapping(self) -> Optional[Dict]:
        """Show dialog to upload manifest and specify factor names for dependent design."""
        dialog = tk.Toplevel(self.root)
        dialog.title("Mixed Factorial Design Setup")
        dialog.geometry("800x750")
        dialog.transient(self.root)
        dialog.grab_set()
        
        result = {}
        manifest_df = None
        
        # File selection section
        ttk.Label(dialog, text="1. Select Pairing Manifest", font=('TkDefaultFont', 12, 'bold')).pack(pady=(10, 5))
        
        # Instructions
        instructions_frame = ttk.Frame(dialog)
        instructions_frame.pack(fill='x', padx=30, pady=(0,5))
        ttk.Label(instructions_frame, text="Required: Excel file with Group, Condition, Subject_ID, and Filename columns", 
                 font=('TkDefaultFont', 9), wraplength=700).pack(anchor='w')
        ttk.Label(instructions_frame, text="IMPORTANT: Folder names must be in format {Condition}_{Group} or {Condition} {Group}", 
                 foreground="red", font=('TkDefaultFont', 9, 'bold'), wraplength=700).pack(anchor='w')
        ttk.Label(instructions_frame, text="Examples: '32_Scn1a', '32 Scn1a', 'Baseline_WT', 'Baseline WT'", 
                 font=('TkDefaultFont', 8), wraplength=700).pack(anchor='w', padx=(10,0))
        
        file_frame = ttk.Frame(dialog)
        file_frame.pack(fill='x', padx=20, pady=10)
        
        manifest_path_var = tk.StringVar()
        ttk.Entry(file_frame, textvariable=manifest_path_var, width=60).pack(side='left', fill='x', expand=True)
        
        def browse_manifest():
            path = filedialog.askopenfilename(
                title="Select Pairing Manifest",
                filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")]
            )
            if path:
                manifest_path_var.set(path)
                load_and_preview_manifest(path)
        
        ttk.Button(file_frame, text="Browse...", command=browse_manifest).pack(side='left', padx=(5, 0))
        
        # Factor names section
        ttk.Label(dialog, text="2. Specify Factor Names", font=('TkDefaultFont', 12, 'bold')).pack(pady=(10, 5))
        
        factor_frame = ttk.Frame(dialog)
        factor_frame.pack(pady=10)
        
        ttk.Label(factor_frame, text="Between-subjects factor (e.g., Genotype):").grid(row=0, column=0, padx=5, pady=5, sticky='e')
        between_entry = ttk.Entry(factor_frame, width=25)
        between_entry.grid(row=0, column=1, padx=5, pady=5)
        between_entry.insert(0, "Genotype")
        
        ttk.Label(factor_frame, text="Within-subjects factor (e.g., Temperature):").grid(row=1, column=0, padx=5, pady=5, sticky='e')
        within_entry = ttk.Entry(factor_frame, width=25)
        within_entry.grid(row=1, column=1, padx=5, pady=5)
        within_entry.insert(0, "Temperature")
        
        # Italics section (will be populated after manifest is loaded)
        ttk.Label(dialog, text="3. Italicize Levels", font=('TkDefaultFont', 12, 'bold')).pack(pady=(10, 5))
        ttk.Label(dialog, text="(Checkboxes will appear after loading manifest)", font=('TkDefaultFont', 8)).pack()
        
        italic_frame = ttk.Frame(dialog)
        italic_frame.pack(pady=10)
        
        # We'll dynamically populate this based on loaded manifest
        level_italic_checkboxes = {}
        
        def populate_italic_checkboxes():
            """Populate italic checkboxes based on manifest levels (called once after manifest loads)."""
            # Clear existing checkboxes
            for widget in italic_frame.winfo_children():
                widget.destroy()
            level_italic_checkboxes.clear()
            
            if manifest_df is not None:
                # Collect unique levels from manifest
                all_levels = set()
                all_levels.update(manifest_df['Group'].unique())
                all_levels.update(manifest_df['Condition'].unique())
                
                # Create checkboxes for each unique level
                if all_levels:
                    for i, level in enumerate(sorted(all_levels)):
                        var = tk.BooleanVar(value=False)
                        checkbox = ttk.Checkbutton(italic_frame, text=level, variable=var)
                        checkbox.grid(row=i//2, column=i%2, padx=20, pady=2, sticky='w')
                        level_italic_checkboxes[level] = var
        
        # Status/validation section
        status_frame = ttk.LabelFrame(dialog, text="Validation", padding=10)
        status_frame.pack(fill='x', padx=20, pady=10)
        
        status_text = tk.Text(status_frame, height=4, state='disabled', font=('TkDefaultFont', 9))
        status_text.pack(fill='x')
        
        def load_and_preview_manifest(path):
            """Load manifest and validate."""
            nonlocal manifest_df
            
            try:
                # Load wide format
                df_wide = pd.read_excel(path)
                
                # Convert to long format
                df_long = convert_manifest_wide_to_long(df_wide)
                manifest_df = df_long
                
                # Auto-detect factor names
                groups = sorted(df_long['Group'].unique())
                conditions = sorted(df_long['Condition'].unique())
                
                # Update factor entries with detected info
                between_entry.delete(0, tk.END)
                between_entry.insert(0, f"Genotype ({len(groups)} levels)")
                
                within_entry.delete(0, tk.END)
                within_entry.insert(0, f"Temperature ({len(conditions)} levels)")
                
                # Validate manifest
                is_valid, errors = validate_manifest(df_long, self.base_path.get())
                
                # Show validation results
                status_text.config(state='normal')
                status_text.delete(1.0, tk.END)
                
                if is_valid:
                    status_text.insert(tk.END, f"✓ Manifest valid\n", 'success')
                    status_text.insert(tk.END, f"  Groups: {', '.join(groups)}\n")
                    status_text.insert(tk.END, f"  Conditions: {', '.join(conditions)}\n")
                    status_text.insert(tk.END, f"  Total: {len(groups)}×{len(conditions)} = {len(df_long['Subject_ID'].unique())} subjects\n")
                    status_text.tag_config('success', foreground='green')
                else:
                    status_text.insert(tk.END, "✗ Validation errors:\n", 'error')
                    for error in errors[:5]:  # Show first 5 errors
                        status_text.insert(tk.END, f"  - {error}\n", 'error')
                    status_text.tag_config('error', foreground='red')
                
                status_text.config(state='disabled')
                
                # Auto-populate italic checkboxes
                populate_italic_checkboxes()
                
            except Exception as e:
                status_text.config(state='normal')
                status_text.delete(1.0, tk.END)
                status_text.insert(tk.END, f"✗ Error: {e}\n", 'error')
                status_text.tag_config('error', foreground='red')
                status_text.config(state='disabled')
        
        def on_ok():
            manifest_path = manifest_path_var.get()
            between_name = between_entry.get().strip().split('(')[0].strip()  # Remove auto-added info
            within_name = within_entry.get().strip().split('(')[0].strip()
            
            if not manifest_path:
                messagebox.showerror("Error", "Please select a manifest file")
                return
            
            if not between_name or not within_name:
                messagebox.showerror("Error", "Please specify both factor names")
                return
            
            if manifest_df is None:
                messagebox.showerror("Error", "Please load a valid manifest file")
                return
            
            # Validate folder names match {Condition}_{Group} or {Condition} {Group} format
            try:
                between_levels = list(manifest_df['Group'].unique())
                within_levels = list(manifest_df['Condition'].unique())
                selected_folder_names = [g.name for g in self.selected_groups]
                
                # Generate all expected folder name combinations
                expected_combinations = []
                for within in within_levels:
                    for between in between_levels:
                        expected_combinations.append(f"{within}_{between}")
                        expected_combinations.append(f"{within} {between}")
                
                # Check each selected folder matches at least one expected combination
                missing_folders = []
                for folder_name in selected_folder_names:
                    if folder_name not in expected_combinations:
                        missing_folders.append(folder_name)
                
                # Check each expected combination has at least one matching folder
                unmatched_combinations = []
                for within in within_levels:
                    for between in between_levels:
                        comb1 = f"{within}_{between}"
                        comb2 = f"{within} {between}"
                        if comb1 not in selected_folder_names and comb2 not in selected_folder_names:
                            unmatched_combinations.append(f"{within}_{between} or {within} {between}")
                
                if missing_folders or unmatched_combinations:
                    error_msg = "Folder name validation failed:\n\n"
                    if unmatched_combinations:
                        error_msg += f"Expected folder combinations not found:\n"
                        for combo in unmatched_combinations[:5]:  # Show first 5
                            error_msg += f"  - {combo}\n"
                    if missing_folders:
                        error_msg += f"\nSelected folders that don't match expected format:\n"
                        for folder in missing_folders[:5]:  # Show first 5
                            error_msg += f"  - {folder}\n"
                    error_msg += f"\nExpected format: {{Condition}}_{{Group}} or {{Condition}} {{Group}}\n"
                    error_msg += f"Groups: {', '.join(between_levels)}\n"
                    error_msg += f"Conditions: {', '.join(within_levels)}"
                    
                    messagebox.showerror("Folder Name Mismatch", error_msg)
                    return
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to validate folder names: {str(e)}")
                return
            
            # Build level_italic mapping
            level_italic = {}
            for level, var in level_italic_checkboxes.items():
                level_italic[level] = var.get()
            
            result['manifest_path'] = manifest_path
            result['manifest'] = manifest_df
            result['between_factor'] = between_name
            result['within_factor'] = within_name
            result['level_italic'] = level_italic
            dialog.destroy()
        
        def on_cancel():
            dialog.destroy()
        
        # Buttons
        button_frame = ttk.Frame(dialog)
        button_frame.pack(pady=20)
        
        ttk.Button(button_frame, text="Run Analysis", command=on_ok, width=15).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Cancel", command=on_cancel, width=15).pack(side='left', padx=5)
        
        # Wait for dialog to close
        self.root.wait_window(dialog)
        
        return result if result else None
    
    def run(self):
        """Start the application."""
        self.root.mainloop()


if __name__ == "__main__":
    app = ModularAnalysisApp()
    app.run()
