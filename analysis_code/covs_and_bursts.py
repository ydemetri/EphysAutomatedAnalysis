import os
import sys 
sys.dont_write_bytecode = True
from pyabf import ABF
from .analyze_abf import CurrentStepsData

def analyze_cc(ABF_LOCATION, COV_OUTPUT_FILE, BURSTS_OUTPUT_FILE):
    if os.path.isdir(ABF_LOCATION):
        abf_files = [os.path.join(ABF_LOCATION, f) for f in os.listdir(ABF_LOCATION) if f.endswith('.abf')]
    else:
        abf_files = [ABF_LOCATION]

    # Print the files we're analyzing as a sanity check
    print('Analyzing the following files:\n{}'.format(abf_files))

    print("Extracting current steps data")

    # Gathering data from the abf files
    
    covs_output = {}
    bursts_output = {}

    for filepath in abf_files:
        abf = ABF(filepath)
        experiment = CurrentStepsData(abf)

        filename = os.path.basename(filepath)
        print('Analyzing {}'.format(filename))

        print('{} contains {} sweeps'.format(filename, len(experiment.sweeps)))        
        currents = experiment.get_current_step_sizes()
        covs, bursts = experiment.get_isi_cov_and_burst_length()
        covs_output[filename] = list(zip(currents, covs))
        bursts_output[filename] = list(zip(currents, bursts))

    #Writing the data to output file
    def write_data(output_name, output_file):
        max_sweeps = len(max(output_name.values(), key=lambda x: len(x)))
        filenames = sorted(output_name.keys())
        with open(output_file, 'w') as f:
            header = []
            index = 0
            for s in filenames:
                header.append(s)
                header.append("Values_{}".format(index))
                index += 1
            f.write(','.join(header))
            f.write('\n')

            for i in range(max_sweeps):
                for filename in filenames:
                    try:
                        f.write('{},{},'.format(*output_name[filename][i]))
                    except IndexError:
                        f.write(',,')
                f.write('\n')

    write_data(covs_output, COV_OUTPUT_FILE)
    write_data(bursts_output, BURSTS_OUTPUT_FILE)
