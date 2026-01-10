import os
import sys 
sys.dont_write_bytecode = True
from pyabf import ABF
from .analyze_abf import VCTestData
import numpy as np

def get_membrane_properties_from_vc(ABF_LOCATION, VC_TEST_OUTPUT_FILE):
    """
    Extract membrane resistance and capacitance from VC test recordings.
    
    Rm is calculated from steady-state current response to voltage step.
    Cm is calculated from the capacitive transient using charge integration (Q/V).
    """
    if os.path.isdir(ABF_LOCATION):
        abf_files = [os.path.join(ABF_LOCATION, f) for f in os.listdir(ABF_LOCATION) if f.endswith('.abf')]
    else:
        abf_files = [ABF_LOCATION]

    # Print the files we're analyzing as a sanity check
    print('Analyzing the following files:\n{}'.format(abf_files))

    print("Extracting membrane resistance and capacitance")

    # Gathering data from the abf files
    input_resistance_output = {}
    capacitance_output = {}

    for filepath in abf_files:
        abf = ABF(filepath)
        experiment = VCTestData(abf)

        filename = os.path.basename(filepath)
        print('Analyzing {}'.format(filename))
        input_resistance_output[filename] = experiment.get_input_resistance()
        capacitance_output[filename] = experiment.get_capacitance()

    # Writing the additional analysis to output file
    with open(VC_TEST_OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write("filename,Rm (MOhm),Cm (pF)\n")
        for filename in input_resistance_output:
            rm_mohm = 1000 * np.nanmean(input_resistance_output[filename])
            cm_pf = np.nanmean(capacitance_output[filename])
            f.write('{},{},{}\n'.format(filename, rm_mohm, cm_pf))


# Keep old function name for backwards compatibility
def get_input_resistance_from_vc(ABF_LOCATION, VC_TEST_OUTPUT_FILE):
    """Deprecated: Use get_membrane_properties_from_vc instead."""
    return get_membrane_properties_from_vc(ABF_LOCATION, VC_TEST_OUTPUT_FILE)
