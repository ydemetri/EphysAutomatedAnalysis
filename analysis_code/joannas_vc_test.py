import os
import sys 
sys.dont_write_bytecode = True
from pyabf import ABF
from .analyze_abf import VCTestData
import numpy as np

def get_input_resistance_from_vc(ABF_LOCATION, VC_TEST_OUTPUT_FILE):
    if os.path.isdir(ABF_LOCATION):
        abf_files = [os.path.join(ABF_LOCATION, f) for f in os.listdir(ABF_LOCATION) if f.endswith('.abf')]
    else:
        abf_files = [ABF_LOCATION]

    # Print the files we're analyzing as a sanity check
    print('Analyzing the following files:\n{}'.format(abf_files))

    print("Extracting resistance")

    # Gathering data from the abf files
    input_resistance_output = {}

    for filepath in abf_files:
        abf = ABF(filepath)
        experiment = VCTestData(abf)

        filename = os.path.basename(filepath)
        print('Analyzing {}'.format(filename))
        input_resistance_output[os.path.basename(filename)] = []
        input_resistance_output[filename] = experiment.get_input_resistance()

    # Writing the additional analysis to output file
    with open(VC_TEST_OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write("filename,Rm (MOhm)\n")
        for filename in input_resistance_output:
            f.write('{},{}\n'.format(filename, 1000*np.mean(input_resistance_output[filename])))
