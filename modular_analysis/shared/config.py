"""
Configuration management for the analysis system.
"""

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


@dataclass
class ProtocolConfig:
    """Configuration for patch clamp protocols."""
    min_current: float = -60.0
    step_size: float = 20.0
    

@dataclass
class PlotConfig:
    """Configuration for plotting aesthetics."""
    figure_size: tuple = (10, 10)
    font_size_label: int = 47
    font_size_tick: int = 40
    font_size_legend: int = 32
    colors: Dict[str, str] = field(default_factory=lambda: {
        'group1': 'blue',
        'group2': 'red',
        'group3': 'green'
    })
    markers: Dict[str, str] = field(default_factory=lambda: {
        'group1': 'o',
        'group2': '^',
        'group3': 's'
    })


@dataclass
class AnalysisConfig:
    """Main configuration for analysis."""
    protocol: ProtocolConfig = field(default_factory=ProtocolConfig)
    plotting: PlotConfig = field(default_factory=PlotConfig)
    output_dir: str = "Results"
    
    def save(self, filepath: str):
        """Save configuration to JSON file."""
        config_dict = {
            'protocol': {
                'min_current': self.protocol.min_current,
                'step_size': self.protocol.step_size
            },
            'plotting': {
                'figure_size': self.plotting.figure_size,
                'font_size_label': self.plotting.font_size_label,
                'font_size_tick': self.plotting.font_size_tick,
                'font_size_legend': self.plotting.font_size_legend,
                'colors': self.plotting.colors,
                'markers': self.plotting.markers
            },
            'output_dir': self.output_dir
        }
        
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'AnalysisConfig':
        """Load configuration from JSON file."""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        
        protocol = ProtocolConfig(
            min_current=config_dict['protocol']['min_current'],
            step_size=config_dict['protocol']['step_size']
        )
        
        plotting = PlotConfig(
            figure_size=tuple(config_dict['plotting']['figure_size']),
            font_size_label=config_dict['plotting']['font_size_label'],
            font_size_tick=config_dict['plotting']['font_size_tick'],
            font_size_legend=config_dict['plotting']['font_size_legend'],
            colors=config_dict['plotting']['colors'],
            markers=config_dict['plotting']['markers']
        )
        
        return cls(
            protocol=protocol,
            plotting=plotting,
            output_dir=config_dict['output_dir']
        )
