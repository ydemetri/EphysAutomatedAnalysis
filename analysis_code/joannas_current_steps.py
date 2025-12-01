import os
import sys 
sys.dont_write_bytecode = True
from pyabf import ABF
from .analyze_abf import CurrentStepsData

def analyze_cc(ABF_LOCATION, CURRENT_VS_APS_OUTPUT_FILE, ANALYSIS_OUTPUT_FILE, ATTN_OUTPUT_FILE):
    if os.path.isdir(ABF_LOCATION):
        abf_files = [os.path.join(ABF_LOCATION, f) for f in os.listdir(ABF_LOCATION) if f.endswith('.abf')]
    else:
        abf_files = [ABF_LOCATION]

    # Print the files we're analyzing as a sanity check
    print('Analyzing the following files:\n{}'.format(abf_files))

    print("Extracting current steps data")

    # Gathering data from the abf files
    current_vs_aps_output = {}
    ap_half_width_output = {}
    ap_peak_output = {}
    ap_amplitude_output = {}
    ap_rise_time_output = {}
    ap_threshold_1_output = {}
    rheobase_output = {}
    max_instantaneous_firing_frequency_output = {}
    min_instantaneous_firing_frequency_output = {}
    max_steady_state_firing_frequency_output = {}
    spike_frequency_adaptation_10_output = {}
    spike_frequency_adaptation_N_output = {}
    time_constant_output = {}
    sag_per_current_output = {}
    upstroke_vel_output = {}
    downstroke_vel_output = {}
    apd_output = {}
    apd50_output = {}
    apd90_output = {}
    attenuation_output = {}
    isi_cov_output = {}
    burst_length_output = {}

    for filepath in abf_files:
        abf = ABF(filepath)
        experiment = CurrentStepsData(abf)

        filename = os.path.basename(filepath)
        print('Analyzing {}'.format(filename))
        current_vs_aps_output[os.path.basename(filename)] = []

        print('{} contains {} sweeps'.format(filename, len(experiment.sweeps)))


        # individual AP characteristics
        print("Extracting half-width and peak")
        ap_half_width_output[filename], ap_peak_output[filename] = experiment.get_ap_half_width_and_peak()
        
        print("Extracting threshold")
        ap_threshold_1_output[filename] = experiment.get_ap_threshold()
        
        print("Extracting amplitude")
        ap_amplitude_output[filename] = experiment.get_ap_amplitude()
        
        print("Extracting rise time")
        ap_rise_time_output[filename] = experiment.get_ap_rise_time()
        
        #upstroke_vel_output[filename], downstroke_vel_output[filename] = experiment.get_velocities()

        print("Extracting APD50 and 90")
        apd_output[filename], apd50_output[filename], apd90_output[filename] = experiment.get_apd50_90()
        

        
        # characteristic of cell
        print("Extracting time constant")
        time_constant_output[filename] = experiment.get_time_constant()
        
        print("Extrating rheobase")
        rheobase_output[filename] = experiment.get_rheobase()
        

        # characteristics of spike train
        print("Extracting max IFF")
        max_instantaneous_firing_frequency_output[filename] = experiment.get_max_instantaneous_firing_frequency()
        
        print("Extracting max SSFF")
        max_steady_state_firing_frequency_output[filename], isi_cov_output[filename], burst_length_output[filename] = experiment.get_max_steady_state_firing_frequency()
       
        print("Extracting SFA10 and SFAn")
        spike_frequency_adaptation_10_output[filename], spike_frequency_adaptation_N_output[filename] = experiment.get_spike_frequency_adaptation()
        
        print("Extracting current vs frequency data")
        current_vs_aps_output[filename] = list(zip(experiment.get_current_step_sizes(), experiment.get_ap_counts()[1]))
        

        print("Extracting sag data")
        sag_per_current_output[filename] = experiment.get_sag()

        print("Extracting attenuation data")
        nums, peaks = experiment.get_attenuation()
        attenuation_output[filename] = list(zip(nums, peaks))


    #Writing the % spikes per current step data to output file
    max_sweeps = len(max(current_vs_aps_output.values(), key=lambda x: len(x)))
    filenames = sorted(current_vs_aps_output.keys())
    # print('max_sweeps is {}'.format(max_sweeps))
    with open(CURRENT_VS_APS_OUTPUT_FILE, 'w') as f:
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
                    f.write('{},{},'.format(*current_vs_aps_output[filename][i]))
                except IndexError:
                    f.write(',,')
            f.write('\n')

    #Writing the attenuation data to output file
    max_sweeps = len(max(attenuation_output.values(), key=lambda x: len(x)))
    filenames = sorted(attenuation_output.keys())
    with open(ATTN_OUTPUT_FILE, 'w') as f:
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
                    f.write('{},{},'.format(*attenuation_output[filename][i]))
                except IndexError:
                    f.write(',,')
            f.write('\n')


    
    #Writing the sag data to output file
    # max_sweeps = len(max(sag_per_current_output.values(), key=lambda x: len(x)))
    # filenames = sorted(sag_per_current_output.keys())
    # # print('max_sweeps is {}'.format(max_sweeps))
    # with open(SAG_OUTPUT_FILE, 'w') as f:
    #     f.write(','.join(['{}, '.format(s) for s in filenames]))
    #     f.write('\n')

    #     for i in range(max_sweeps):
    #         for filename in filenames:
    #             try:
    #                 f.write('{},{},'.format(*sag_per_current_output[filename][i]))
    #             except IndexError:
    #                 f.write(',,')
    #         f.write('\n')


    # Writing the additional analysis to output file
    with open(ANALYSIS_OUTPUT_FILE, 'w') as f:
        f.write("filename,Rheobase (pA),Time Constant (ms),Sag,Max Steady-state (Hz),Max Instantaneous (Hz),SFA10,SFAn,ISI_CoV,Burst_length (ms),AP Threshold (mV),AP Peak (mV),AP Amplitude (mV),AP Rise Time (ms),APD 50 (ms),APD 90 (ms)\n")
        for filename in ap_half_width_output:
            f.write('{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n'.format(
                filename,
                rheobase_output[filename],
                time_constant_output[filename],
                sag_per_current_output[filename],
                max_steady_state_firing_frequency_output[filename],
                max_instantaneous_firing_frequency_output[filename],
                spike_frequency_adaptation_10_output[filename],
                spike_frequency_adaptation_N_output[filename],
                isi_cov_output[filename],
                burst_length_output[filename],
                ap_threshold_1_output[filename],
                ap_peak_output[filename],
                ap_amplitude_output[filename],
                ap_rise_time_output[filename],
                #ap_half_width_output[filename],
                #apd_output[filename],
                apd50_output[filename],
                apd90_output[filename]
                #upstroke_vel_output[filename],
                #downstroke_vel_output[filename]
            ))
