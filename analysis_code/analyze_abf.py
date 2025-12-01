import sys 
sys.dont_write_bytecode = True

import os
import logging
from functools import lru_cache
from collections import defaultdict
from typing import List

from pyabf import ABF
import matplotlib.pyplot as plt
import numpy as np

from .trace_analysis import fit_tophat, find_peaks, get_derivative
from sys import float_info
from .subthresh_features import time_constant
from .subthresh_features import sag
from .subthresh_features import baseline_voltage
from scipy import ndimage

ABF_FILE_EXTENSION = '.abf'
EXPERIMENT_TYPE_CURRENT_STEPS = 'current_steps'

EXPERIMENT_TYPES = [
    EXPERIMENT_TYPE_CURRENT_STEPS
]

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class InvalidSweep(Exception):
    pass


class Sweep(object):
    """
    The data related to one sweep in an `ExperimentData` (i.e. in an abf file)
    """

    def __init__(
            self,
            time_steps,
            input_signal,
            output_signal,
            time_steps_units=None,
            input_signal_units=None,
            output_signal_units=None,
            sweep_name=None):
        """

        :param time_steps: The time steps of the sweep (list)
        :param input_signal: The input signal values (list)
        :param output_signal: The output signal values (list)
        :param time_steps_units: units of `time`
        :param input_signal_units: units of `input_signal`
        :param output_signal_units: units of `output_signal`
        """
        # time, input signal and output signal must have the same number of
        # data points
        assert len(time_steps) == len(input_signal)
        assert len(time_steps) == len(output_signal)

        self.time_steps = time_steps
        self.input_signal = input_signal
        self.output_signal = output_signal
        self.time_steps_units = time_steps_units
        self.input_signal_units = input_signal_units
        self.output_signal_units = output_signal_units
        self.sweep_name = sweep_name
        self.analysis_cache = {}  # TODO replace this with functools.lru_cache

        # logger.debug('{} input units {}'.format(sweep_name, input_signal_units))
        # logger.debug('{} output units {}'.format(sweep_name, output_signal_units))

    def __str__(self):
        return 'Data from a single sweep, containing {} data points'.format(len(self.time_steps))

    def fit_input_tophat(self, verify=False):
        """
        Fit a tophat function to the input signal and cache the result
        :return: (base_level, hat_level, hat_mid, hat_width)
        """
        if 'fit_input_tophat' in self.analysis_cache:
            logger.info('Found tophat params in cache')
            return self.analysis_cache['fit_input_tophat']

        tophat_params = fit_tophat(
            self.time_steps, self.input_signal, verify=verify)
        self.analysis_cache['fit_input_tophat'] = tophat_params

        # TODO Try to spot zero height tophats, and return something sensible

        return tophat_params

    def find_output_peaks(self, threshold=0, verify=False):
        """
        Find the peaks in the output and return a list of (t, V) values

        :param threshold:
        :param verify:
        :return: list of tuples (peak time, peak value)
        """
        if 'find_output_peaks' in self.analysis_cache:
            logger.info('Found peak counts in cache')
            return self.analysis_cache['find_output_peaks']

        peaks = find_peaks(
            self.sweep_name,
            self.time_steps,
            self.output_signal,
            threshold=threshold,
            verify=verify)
        self.analysis_cache['find_output_peaks'] = peaks

        return peaks

    @lru_cache(maxsize=2)
    def find_input_peaks(self, threshold=1, verify=False):
        """
        Find the peaks in the input and return a list of (t, V) values

        :param threshold:
        :param verify:
        :return:
        """
        peaks = find_peaks(
            self.sweep_name,
            self.time_steps,
            self.input_signal,
            threshold=threshold,
            verify=verify)
        return peaks

    @lru_cache(maxsize=2)
    def get_output_derivative(self, verify=False):
        """
        return d/dt values of output signal. List indices correspond to
        self.time_steps

        :return: list of dV/dt values
        """
        d_dt_output = get_derivative(self.output_signal, self.time_steps)

        if verify:
            plt.close('all')
            plt.figure(figsize=(16, 10))
            fig, ax1 = plt.subplots()
            ax1.plot(self.time_steps, self.output_signal, color='b', label="Output Signal (mV)")
            ax1.legend(loc="upper right")
            ax1.set_xlabel("Time (s)")
            ax2 = ax1.twinx()
            ax2.plot(self.time_steps, d_dt_output, color='g', label="Voltage Derivative")
            ax2.set_xlabel("Time (s)")
            ax2.set_ylabel("Voltage (mV)")
            ax2.set_title("Output Signal and its Derivative for {}".format(self.sweep_name))
            ax2.legend(loc="center right")
            plt.savefig("C:\\Users\\Yiannos\\Documents\\Mattis Lab\\MattisLab\\Test3\\derivative_{}.png".format(self.sweep_name))

        return d_dt_output

    def get_output_second_derivative(self):
        """
        return d2V/dt2 values of output signal. List indices correspond to
        self.time_steps

        :return: list of d2V/dt2 values
        """
        if 'get_output_second_derivative' in self.analysis_cache:
            logger.info('Found output second derivative in cache')
            return self.analysis_cache['get_output_second_derivative']

        d2_dt2_output = get_derivative(self.get_output_derivative(), self.time_steps)
        self.analysis_cache['get_output_second_derivative'] = d2_dt2_output
        return d2_dt2_output

    def show_plot(self):
        """
        Plot input vs time and output vs time on overlapping axes
        :return: None
        """
        plt.close(fig='all')  # Make sure there are no unwanted figures
        fig, ax1 = plt.subplots()
        ax1.plot(self.time_steps, self.input_signal, color='red', label="Input Signal (pA)")
        ax1.set_ylabel("Current (pA)")
        ax1.set_xlabel("Time (s)")
        ax1.legend(loc='upper right')
        ax2 = ax1.twinx()
        ax2.plot(self.time_steps, self.output_signal, label="Output Signal (mV)")
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Voltage (mV)")
        ax2.legend(loc='center right')
        ax2.set_title("Input and Output Signals for {}".format(self.sweep_name))

        plt.savefig("io_{}.png".format(self.sweep_name))


class CurrentStepsSweep(Sweep):
    """
    Functions to extract relevant data from a current steps sweep
    """
    DEFAULT_GOOD_AP_AMPLITUDE = 0
    DEFAULT_FAILED_AP_AMPLITUDE = -20
    DEFAULT_CEILING_AP_AMPLITUDE = 100
    DEFAULT_AP_TO_ANALYZE = 0 # 0 for 1st AP, 4 for 5th AP
    DEFAULT_MIN_APS_IN_ANALYSIS_SWEEP = 1

    def __init__(self, *args, **kwargs):
        self._has_failed_aps = None
        # These should be overwritten after instantiation to use different values
        self.good_ap_amplitude = self.DEFAULT_GOOD_AP_AMPLITUDE
        self.failed_ap_amplitude = self.DEFAULT_FAILED_AP_AMPLITUDE
        self.ceiling_ap_amplitude = self.DEFAULT_CEILING_AP_AMPLITUDE
        self.default_ap_to_analyze = self.DEFAULT_AP_TO_ANALYZE
        self.default_min_ap = self.DEFAULT_MIN_APS_IN_ANALYSIS_SWEEP

        super().__init__(*args, **kwargs)

    def has_aps(self):
        return len(self.get_aps()) > 0

    def has_failed_aps(self):
        """
        Boolean specifying whether the sweep has failed aps. See get_aps for
        definition.
        """
        if self._has_failed_aps is None:
            # Call get_aps with default args to ensure _has_failed_aps is set
            #logger.warning('Checking for failed APs with default threshold values')
            self.get_aps()

        return self._has_failed_aps

    @lru_cache(maxsize=1)
    def _get_output_peaks(self):
        """
        Get all peaks from the output signal that are above threshold
        self.failed_ap_amplitude

        :return:
        """
        return find_peaks(
            self.sweep_name, self.time_steps, self.output_signal, threshold=self.failed_ap_amplitude)

    @lru_cache(maxsize=1)
    def _get_aps_with_idx(self):
        """
        Like get_aps, but the returned list includes list index of the peaks

        :return: [(idx, time, voltage), ...]
        """
        low_boundary_peaks = self._get_output_peaks()
        aps = [peak for peak in low_boundary_peaks if (peak[2] > self.good_ap_amplitude and peak[2] < self.ceiling_ap_amplitude)]
        current = self.get_drive_current()
        if len(low_boundary_peaks) > len(aps):
            # logger.warning('Sweep {} at {}{} has failed APs'.format(
            #     self.sweep_name,
            #     current,
            #     self.input_signal_units
            # ))
            self._has_failed_aps = True
        else:
            self._has_failed_aps = False


        # Discount APs from negative or no current injections (current reading not fully accurate so check <5)
        if current < 5.0:
            return []

        # Only use peaks within stimulus interval
        stim_start, stim_end = self.get_input_start_end()
        aps = [peak for peak in aps if peak[1] >= stim_start and peak[1] <= stim_end]

        return aps

    @lru_cache(maxsize=1)
    def get_aps(self, verify=False):
        """
        Look for peaks in the output voltage. Any peak above self.good_ap_amplitude
        is an AP. Peaks that pass self.failed_ap_amplitude, but not
        self.good_ap_amplitude are failed peaks, which make the sweep invalid.

        :return: [(ap_time, ap_voltage), ...]
        """
        if verify:
            raise NotImplementedError

        aps_with_idx = self._get_aps_with_idx()
        return [ap[1:3] for ap in aps_with_idx]

    @lru_cache(maxsize=1)
    def get_drive_current(self, verify=False):
        """
        Drive current is a tophat function.
        :return:
        """
        base_level, hat_level, hat_start, hat_end = fit_tophat(
            self.time_steps, self.input_signal, verify=verify)
        current_step = hat_level - base_level
        if np.isnan(current_step):
            current_step = 0
        return current_step
    
    def get_input_start_end(self):
        base_level, hat_level, hat_start, hat_end = fit_tophat(
            self.time_steps, self.input_signal)
        return hat_start, hat_end
        

    def get_ap_count(self, verify=False):
        return len(self.get_aps(verify))


    @lru_cache(maxsize=1)
    def get_steady_state_ap_frequency(self):

        aps = self.get_aps()
        start, end = self.get_input_start_end()
        freq = (len(aps)) / (end - start)
        logger.debug('Steady state firing frequency of {} is {}'.format(self.sweep_name, freq))

        return freq

    def get_min_instantaneous_ap_frequency(self):
        """
        Get the frequency corresponding to the shortest gap between APs in this sweep

        :return:
        """
        if self.get_ap_count() < 2:
            logger.debug('{} has < 2 APs. Cannot get frequency'.format(self.sweep_name))
            raise InvalidSweep()

        aps = self.get_aps()
        minimum_ap_interval = float_info.min
        minimum_ap_interval_first_ap = None
        for idx, ap in enumerate(aps[1:]):
            ap_interval = ap[0] - aps[idx][0]
            logger.debug('{}, {} to {} peak interval is {}'.format(
                self.sweep_name, idx, idx + 1, ap_interval))
            if ap_interval > minimum_ap_interval:
                logger.debug('Found a smaller peak interval in {}, peaks {} to {}'.format(
                        self.sweep_name, idx, idx + 1))
                minimum_ap_interval = ap_interval
                minimum_ap_interval_first_ap = idx

        return 1 / minimum_ap_interval, minimum_ap_interval_first_ap
    
    def get_max_instantaneous_ap_frequency(self):
        """
        Get the frequency corresponding to the shortest gap between APs in this sweep

        :return:
        """
        if self.get_ap_count() < 2:
            logger.debug('{} has < 2 APs. Cannot get frequency'.format(self.sweep_name))
            raise InvalidSweep()

        aps = self.get_aps()
        minimum_ap_interval = float_info.max
        minimum_ap_interval_first_ap = None
        for idx, ap in enumerate(aps[1:]):
            ap_interval = ap[0] - aps[idx][0]
            logger.debug('{}, {} to {} peak interval is {}'.format(
                self.sweep_name, idx, idx + 1, ap_interval))
            if ap_interval < minimum_ap_interval:
                logger.debug('Found a smaller peak interval in {}, peaks {} to {}'.format(
                        self.sweep_name, idx, idx + 1))
                minimum_ap_interval = ap_interval
                minimum_ap_interval_first_ap = idx

        return 1 / minimum_ap_interval, minimum_ap_interval_first_ap

    def get_ap_frequency_adaptation(self, min_ap_count):
        """
        Spike frequency adaptation: ratio of first to Nth interspike interval
        (ISI1/ISI10) and ratio of first to last interspike interval
        (ISI1/ISIn) for the first suprathreshold current injection to elicit
        sustained firing

        :return:
        """
        if self.get_ap_count() < min_ap_count:
            raise InvalidSweep('Not enough APs for spike frequency adaptation')

        aps = self.get_aps()
        isi_1 = aps[1][0] - aps[0][0]
        isi_10 = aps[10][0] - aps[9][0]
        isi_n = aps[-1][0] - aps[-2][0]

        return isi_1/isi_10, isi_1/isi_n

    def get_first_ap_peak_data_index(self):
        """
        Get the index of the time_steps / input_signal / output_signal iterables
        where the peak of the first AP occurs

        :return:
        """
        if self.get_ap_count() < self.default_min_ap:
            raise InvalidSweep('No APs in {}'.format(self.sweep_name))

        aps_with_idx = self._get_aps_with_idx()
        return aps_with_idx[self.default_ap_to_analyze][0]

    def get_first_ap_amplitude(self):
        """

        :return:
        """
        threshold_voltage, threshold_time = self.get_first_ap_threshold()
        first_peak_voltage = self.get_aps()[self.default_ap_to_analyze][1]

        return first_peak_voltage - threshold_voltage

    def get_first_ap_rise_time(self):
        """

        :return:
        """
        threshold_voltage, threshold_time = self.get_first_ap_threshold()
        first_peak_time = self.get_aps()[self.default_ap_to_analyze][0]

        return 1000*(first_peak_time - threshold_time)

    @lru_cache(maxsize=1)
    def get_first_ap_threshold(self, threshold=10000):
        """
        Get the AP threshold of the first AP. Defined as the point at which its
        rate of change is greated than a threshold value.
        # TODO Lots of functions call this function, but cannot specify a threshold

        :param threshold
        :return:
        """
        if self.get_ap_count() < self.default_min_ap:
            raise InvalidSweep('Sweep {} has not APs'.format(self.sweep_name))
        dv_dt = self.get_output_derivative()
        aps = self._get_aps_with_idx()
        start_idx = aps[self.default_ap_to_analyze][0] # make sure we are getting the threshold of the first good ap, not a previous failed one
        for idx, gradient in enumerate(dv_dt):
            if gradient >= threshold and idx >= start_idx - 20:
                # return the value of the voltage at the timestamp that
                # we cross the threshold in gradient, and the time it happened
                return self.output_signal[idx], self.time_steps[idx]


class VCTestSweep(Sweep):
    """
    Functions to extract relevant data from a current steps sweep
    """
    def get_input_resistance(self, verify=False):
        voltage_base, applied_voltage, voltage_start, voltage_end = \
            self.fit_input_tophat(verify=verify)
        logger.debug('Voltage starts at t={}'.format(voltage_start))
        start_idx = None
        end_idx = None
        for idx, t in enumerate(self.time_steps):
            if t >= voltage_start and start_idx is None:
                start_idx = idx
            if t >= voltage_end and end_idx is None:
                end_idx = idx

        # Measure current for the middle half of the driven part of the sweep
        logger.debug('Driven slice is {} to {}'.format(start_idx, end_idx))
        measurement_slice_start = start_idx + (end_idx - start_idx) // 4
        measurement_slice_end = start_idx + 3 * (end_idx - start_idx) // 4

        mean_current_in_measurement_slice = np.mean(
            self.output_signal[measurement_slice_start: measurement_slice_end])

        # Measure current for the middle half post drive part of the sweep
        last_idx = len(self.input_signal) - 1
        resting_slice_start = end_idx + (last_idx - end_idx) // 4
        resting_slice_end = end_idx + 3 * (last_idx - end_idx) // 4

        mean_current_in_resting_slice = np.mean(
            self.output_signal[resting_slice_start: resting_slice_end])

        logger.debug('{} applied voltage: {} {}'.format(
            self.sweep_name, applied_voltage, self.input_signal_units))
        logger.debug('{} mean driven current: {} {}'.format(
            self.sweep_name, mean_current_in_measurement_slice, self.output_signal_units))
        logger.debug('{} resting current is: {} {}'.format(
            self.sweep_name, mean_current_in_resting_slice, self.input_signal_units))

        change_in_current = mean_current_in_measurement_slice - mean_current_in_resting_slice
        resistance = (applied_voltage - voltage_base) / change_in_current
        logger.info('Resistance from sweep {} is {}'.format(self.sweep_name, resistance))

        return resistance


class EToIRatioSweep(Sweep):
    """Functions to extract relevant data from an EtoIRatio sweep"""

    def _find_pulse_baseline(self, n, pre_pulse_artifact=0.0005):
        """

        :param n:
        :return:
        """
        if n == 1:
            # Baseline for first pulse is start of trace to input time - padding time
            baseline_end_t = self.find_input_peaks()[0][1] - pre_pulse_artifact
            baseline_end_t_step = min(self.time_steps, key=lambda x: abs(x - baseline_end_t))
            baseline_end_idx = np.where(self.time_steps == baseline_end_t_step)[0][0]
            baseline_value = np.mean(self.output_signal[:baseline_end_idx])
        else:
            # Baseline for subsequent pulses is average of a number of timesteps preceding the next peak
            pulse_t = self.find_input_peaks()[n-1][1]
            pulse_idx = self.find_input_peaks()[n-1][0]
            for t in self.time_steps[pulse_idx::-1]:
                if t < pulse_t - pre_pulse_artifact:
                    mean_end_idx_arr = np.where(self.time_steps == t)
                    assert(len(mean_end_idx_arr[0]) == 1)
                    mean_end_idx = mean_end_idx_arr[0][0]
                    break
            baseline_value = np.mean(self.output_signal[mean_end_idx - 100:mean_end_idx])

        return baseline_value

    def find_post_synaptic_potential(self, n, post_pulse_artifact=0.001, verify=False):
        """
        Peak in absolute output after stimulation.

        :param n: find potential after nth peak
        :param post_pulse_artifact:
        :param verify:
        :return: time, value of the post synaptic peak
        """
        def verification_plot():
            plt.figure(figsize=(32, 20))
            plt.close('all')
            plot_idx_padding = input_pulses[1][0] - input_pulses[0][0]
            plot_idx_start = input_pulses[0][0] - plot_idx_padding
            plot_idx_end = input_pulses[-1][0] + plot_idx_padding

            plt.plot(
                self.time_steps[plot_idx_start:plot_idx_end],
                self.output_signal[plot_idx_start:plot_idx_end])

            # Show peak position
            plt.axvline(x=peak[0], color='red')
            plt.axhline(y=peak[1], color='red')

            # Show search range for peak
            plt.axvline(x=search_range_start_t, color='yellow')
            plt.axvline(x=search_range_end_t, color='yellow')

            # Show baseline output
            plt.axhline(y=baseline_value, color='blue')

            plt.show()

        # Times of input signals
        input_pulses = self.find_input_peaks()
        input_time = input_pulses[n-1][1]
        input_idx = input_pulses[n-1][0]

        if n < len(input_pulses):
            next_input_time = input_pulses[n][1]
            next_input_index = input_pulses[n][0]
        elif n == len(input_pulses):
            # If this is the last input pulse, search for peaks until t + last inter-pulse interval
            next_input_time = 2*input_pulses[n - 1][1] - input_pulses[n - 2][1]
            next_input_index = 2*input_pulses[n - 1][0] - input_pulses[n - 2][0]
        else:
            raise ValueError('There are less peaks than the requested peak number')

        # Find the range between the first input pulse and second.
        search_range = zip(
            self.time_steps[input_idx:next_input_index],
            self.output_signal[input_idx:next_input_index])

        # Ignore the first period after input signal and last period before next
        # input signal, to avoid artifacts
        search_range = [d for d in search_range if (d[0] > input_time + post_pulse_artifact) and (d[0] < next_input_time - post_pulse_artifact)]

        # Find baseline output value
        baseline_value = self._find_pulse_baseline(n)

        # Search for max absolute value
        search_range_start_t = search_range[0][0]
        search_range_end_t = search_range[-1][0]
        peak = max(search_range, key=lambda x: abs(x[1]-baseline_value))

        if verify:
            verification_plot()

        return peak[0] - input_time, peak[1] - baseline_value, baseline_value

    def find_total_integrated_current(
            self, integration_interval=.04, post_pulse_artifact=.001, verify=False):
        """

        :param post_pulse_artifact: Time to ignore after pulse to avoid artifacts
        :param integration_interval: Interval over which to integrate current
                                     (measured from pulse_time, not from pulse_time+post_pulse_artifact)
        :param verify: Show verification plot
        :return:
        """
        def verification_plot():
            plt.figure(figsize=(32, 20))
            plt.close('all')
            plot_idx_padding = input_pulses[1][0] - input_pulses[0][0]
            plot_idx_start = input_pulses[0][0] - plot_idx_padding
            plot_idx_end = input_pulses[-1][0] + 6 * plot_idx_padding

            plt.plot(
                self.time_steps[plot_idx_start:plot_idx_end],
                self.output_signal[plot_idx_start:plot_idx_end])

            # Integral ranges
            for start, end in verify_integral_ranges:
                plt.axvline(x=start, color='C0')
                plt.axvline(x=end, color='C0')

            # Show baselines used to check for excitatory artifacts
            for i, baseline_value in enumerate(verify_excitatory_artifact_baselines):
                input_pulse_t = input_pulses[i][1]
                plt.axhline(y=baseline_value, color='C{}'.format(i+1))
                plt.axvline(x=input_pulse_t, color='C{}'.format(i+1))

            plt.show()

        # Times of input signals
        input_pulses = self.find_input_peaks()

        # Determine if post pulse potential is +ve or -ve
        first_peak_time, first_peak_value, baseline = \
            self.find_first_post_synaptic_potential(post_pulse_artifact)

        # Create vars to save some data for a verification plot
        verify_excitatory_artifact_baselines = []
        verify_integral_ranges = []

        # Perform integral
        pulse_current_integrals = []
        excitatory_artifact_baseline = baseline
        for pulse_idx, pulse in enumerate(input_pulses):
            logger.info('Integrating after pulse {}'.format(pulse_idx))
            logger.debug('Pulse datapoint index: {}'.format(pulse[0]))
            verify_excitatory_artifact_baselines.append(excitatory_artifact_baseline)

            # Start the integral dt=post_pulse_artifact after the first pulse
            integral_start_t = pulse[1] + post_pulse_artifact
            logger.debug('Integral start time: {}'.format(integral_start_t))
            for t_idx, t in enumerate(self.time_steps[pulse[0]:]):
                if t > integral_start_t:
                    integral_start_idx = pulse[0] + t_idx
                    logger.debug('Integral start index: {}'.format(integral_start_idx))
                    break

            # End the integral post_pulse_artifact before the next pulse,
            # or tail_length*pulse-delta-t after the last pulse
            integral_end_t = pulse[1] + integration_interval
            logger.debug('Integral end time: {}'.format(integral_end_t))

            for t_idx, t in enumerate(self.time_steps[pulse[0]:]):
                if t > integral_end_t:
                    integral_end_idx = pulse[0] + t_idx
                    logger.debug('Integral end index: {}'.format(integral_end_idx))
                    break

            verify_integral_ranges.append((integral_start_t, integral_end_t))

            # Derive arrays of time gaps and current-baseline values
            t_gaps = np.diff(self.time_steps[integral_start_idx:integral_end_idx])
            i_values = np.subtract(self.output_signal[integral_start_idx:integral_end_idx-1], baseline)

            # If the peak is +ve, set i_values less than baseline to zero to eliminate
            # artifactorial -ve spike.
            i_values = i_values.clip(min=excitatory_artifact_baseline)

            # Set the artifact baseline for the next pulse
            excitatory_artifact_baseline = np.mean(self.output_signal[integral_end_idx-100:integral_end_idx])

            current_integral = np.dot(t_gaps, i_values)
            logger.debug('Pulse {} integral = {}'.format(pulse_idx, current_integral))

            pulse_current_integrals.append(current_integral)

        if verify:
            verification_plot()

        return tuple(pulse_current_integrals)


class ExperimentData(object):
    """The set of traces in one abf file (a colloquial, not mathematical set)"""

    def __init__(self, abf, input_signal_channel=1, output_signal_channel=0):
        """

        :param abf: The abf file, as loaded by pyabf
        """
        self.abf = abf
        self.filename = os.path.basename(abf.abfFilePath)
        self.sweep_count = abf.sweepCount
        self.experiment_type = 'experiment'  # TODO This should be set by subclasses
        #logger.info('{} sweeps in {}'.format(self.sweep_count, self.filename))

        # Extract all the sweeps into
        self.sweeps = []
        for sweep_num in self.abf.sweepList:
            self.abf.setSweep(sweep_num, channel=input_signal_channel)
            input_signal = self.abf.sweepY
            input_signal_units = self.abf.sweepUnitsY

            self.abf.setSweep(sweep_num, channel=output_signal_channel)
            output_signal_units = self.abf.sweepUnitsY
            output_signal = self.abf.sweepY

            time_steps = self.abf.sweepX
            time_units = self.abf.sweepUnitsX
            self.sweeps.append(
                self._sweep(
                    time_steps,
                    input_signal,
                    output_signal,
                    time_units,
                    input_signal_units,
                    output_signal_units,
                    '{}_{}'.format(self.filename[:-4], sweep_num)
                )
            )

    @staticmethod
    def _sweep(*args):
        return Sweep(*args)

    def __str__(self):
        return('Experiment data from {} containing {} sweeps of {} data'.format(
            self.filename, self.sweep_count, self.experiment_type
        ))


class EToIRatioData(ExperimentData):
    """Functions to get relevant metrics for 'E to I ratio' experiments"""
    sweeps: List[EToIRatioSweep]

    def _sweep(self, *args, **kwargs):
        return EToIRatioSweep(*args, **kwargs)

    def average_sweeps(self, verify=False):
        """

        :return:
        """
        def verification_plot():
            plt.close('all')
            fig, ax1 = plt.subplots()
            ax1.plot(t_steps, inp, color='red')
            ax2 = ax1.twinx()
            ax2.plot(t_steps, outp)
            plt.show()

        inp = np.mean([sweep.input_signal for sweep in self.sweeps], axis=0)
        outp = np.mean([sweep.output_signal for sweep in self.sweeps], axis=0)
        t_steps = self.sweeps[0].time_steps

        # Verify all timesteps are the same
        for sweep in self.sweeps:
            assert np.array_equal(t_steps, sweep.time_steps)

        if verify:
            verification_plot()

        self.sweeps = [self._sweep(
            t_steps,
            inp,
            outp,
            self.sweeps[0].time_steps_units,
            self.sweeps[0].input_signal_units,
            self.sweeps[0].output_signal_units,
            '{}_{}'.format(self.filename[:-4], 'mean')
        )]


class VCTestData(ExperimentData):
    """Functions to get relevant metrics for 'VC test' experiments"""
    sweeps: List[VCTestSweep]

    def _sweep(self, *args, **kwargs):
        return VCTestSweep(*args, **kwargs)

    def get_input_resistance(self, verify=False):
        """
        Input resistance: calculate using change in steady state current
        in response to small hyperpolarizing voltage step

        :return:
        """
        resistances = []
        for sweep in self.sweeps:
            resistances.append(sweep.get_input_resistance())

        if verify:
            raise NotImplementedError

        return resistances


class CurrentClampGapFreeData(ExperimentData):
    """Functions to get relevant metrics for 'current clamp gap free' experiments"""
    def get_resting_potential(self, verify=False):
        """
        Resting potential is in the output trace. Just average it. There should
        be just one trace

        :return:
        """
        def verification_plot():
            plt.close(fig='all')  # Make sure there are no unwanted figures
            plt.figure(figsize=(32, 20))
            plt.plot(self.sweeps[0].time_steps, self.sweeps[0].output_signal)
            plt.title("Resting Poential for {}".format(self.sweeps[0].sweep_name))
            plt.xlabel("Time (s)")
            plt.ylabel("Voltage (mV)")
            plt.axhline(mean_voltage, label="Average Voltage")
            plt.plot()

        assert len(self.sweeps) == 1
        mean_voltage = np.mean(self.sweeps[0].output_signal)

        if verify:
            verification_plot()

        return mean_voltage


class CurrentStepsData(ExperimentData):
    """Functions to get relevant metrics for 'current steps' experiments"""
    # This is a type hint for pycharm - not functional
    sweeps: List[CurrentStepsSweep]

    @staticmethod
    def _sweep(*args):
        return CurrentStepsSweep(*args)

    def get_current_step_sizes(self, verify=False):
        """
        Get a list of the step sizes of the driving current in the same order
        as self.sweeps

        :return: list of floats
        """
        #logger.info('Getting current step sizes for {}'.format(self.filename))
        step_sizes = []
        for sweep in self.sweeps:
            step_sizes.append(sweep.get_drive_current(verify))

        return step_sizes

    @lru_cache(maxsize=1)
    def get_ap_counts(self, verify=False):
        """
        Get a list of the number of action potentials in the output in the
        same order as self.sweeps. Also get frequency

        :return: list of ints
        """
        #logger.info("Getting counts of APs in {}".format(self.filename))
        ap_counts = [] 
        freqs = []
        for sweep in self.sweeps:
            count = sweep.get_ap_count(verify=verify)
            ap_counts.append(count)
            start, end = sweep.get_input_start_end()
            freqs.append(count / (end - start))


        if len(freqs) == 0:
            logger.warning('No sweep had enough peaks to calculate a frequency')

        return ap_counts, freqs

    def get_rheobase(self, verify=False):
        """
        Get the rheobase - the minimum current that elicits at least one peak
        :return:
        """
        def verification_plot():
            plt.close(fig='all')  # Make sure there are no unwanted figures
            plt.figure(figsize=(32, 20))
            for i, sweep in enumerate(self.sweeps):
                offset = 140 * i  # TODO Derive offset from data
                plt.plot(sweep.time_steps, sweep.output_signal + offset)
                plt.title("Rheobase for {}".format(sweep.sweep_name))
                text_height = sweep.output_signal[0] + offset
                plt.text(0, text_height, '{:.0f}'.format(sweep.get_drive_current()), fontsize=8)
                if i == rheobase_sweep_num:
                    plt.text(
                        .95,
                        text_height,
                        'Rheobase from this sweep',
                        horizontalalignment='right',
                        fontsize=8
                    )

            plt.gca().get_yaxis().set_visible(False)
            plt.show()

        rheobase_sweep_num = self._get_rheobase_sweep_num()
        rheobase = self.sweeps[rheobase_sweep_num].get_drive_current()

        if verify:
            verification_plot()

        return rheobase

    def _get_rheobase_sweep_num(self):
        """
        get the sweep number that the rheobase is obtained from. i.e. the first
        sweep with an AP
        :return:
        """
        for idx, sweep in enumerate(self.sweeps):
            if sweep.get_ap_count() >= sweep.default_min_ap:
                return idx
        else:
            logger.warning('No sweep in {} had a peak'.format(self.filename))
            return None

    def get_spike_frequency_adaptation(self, min_ap_count=11, verify=False):
        """
        Spike frequency adaptation: ratio of first to 10th interspike interval
        (ISI1/ISI10) and ratio of first to last interspike interval
        (ISI1/ISIn) for the Sweep with most APs and at least `min_ap_count` APs

        :return: (isi_1/isi_10, isi_1/isi_n)
        """
        # Find the sweep with most APs. Breaking ties with higher sweep number.
        ap_counts = defaultdict(list)
        for idx, sweep in enumerate(self.sweeps):
            ap_counts[sweep.get_ap_count()].append(idx)

        max_aps = max(ap_counts)
        if max_aps < min_ap_count:
            logger.warning('{} had no sweeps with sustained firing'.format(self.filename))
            return None, None

        sfa_sweep = max(ap_counts[max_aps])
        sfa = self.sweeps[sfa_sweep].get_ap_frequency_adaptation(min_ap_count)

        if verify:
            raise NotImplementedError

        return sfa

    def get_max_steady_state_firing_frequency(self, verify=False):
        """
         Max steady state firing frequency:
         max mean firing frequency in response to current injection with no
         failures (AP amplitude at least 40mV and overshooting 0mV)

        # TODO what should be returned. frequency. Driving voltage eliciting that frequency?
        # TODO do we have to check for "missing" peaks
        :return: frequency, inverse of timesteps units
        """
        def verification_plot():
            plt.figure(figsize=(32, 20))
            plt.close('all')
            for i, sweep in enumerate(self.sweeps):
                offset = 140 * i  # TODO Derive offset from data
                text_height = sweep.output_signal[0] + offset
                plot_color = 'b'
                for freq, idx in frequencies.items():
                    if idx == i:
                        plt.text(0, text_height, '{:.2f}Hz'.format(freq), fontsize=8)
                if i == max_frequency_idx:
                    max_freq_peaks = self.sweeps[i].get_aps()
                    first_ap_time = max_freq_peaks[0][0]
                    last_ap_time = max_freq_peaks[-1][0]
                    time_diff = last_ap_time - first_ap_time
                    plt.text(
                        .95,
                        text_height,
                        'Max SSFF: {} APs in {:.2f}s = {:.0f}Hz'.format(
                            len(max_freq_peaks), time_diff, max_frequency),
                        horizontalalignment='right',
                        fontsize=8
                    )
                    plt.axvline(first_ap_time)
                    plt.axvline(last_ap_time)
                    plot_color = 'g'
                plt.plot(sweep.time_steps, sweep.output_signal + offset, color=plot_color)

            plt.gca().get_yaxis().set_visible(False)
            plt.savefig("max_ss_{}.png".format(sweep.sweep_name))

        def group_pv_cells(sweep):
            aps = sweep.get_aps()
            all_isi = []
            burst_length = None

            for i in range(len(aps) - 1):
                isi = (aps[i + 1][0] - aps[i][0]) * 1000
                all_isi.append(isi)

            isi_mean = np.mean(np.array(all_isi))
            isi_stdev = np.std(np.array(all_isi))

            for i in range(len(all_isi)):
                if all_isi[i] > (isi_mean + (3*isi_stdev)): #is above 3 stdevs
                    burst_length = (aps[i][0] - aps[0][0]) * 1000
                    break
            if burst_length == None and len(aps) != 0:
                burst_length = (aps[-1][0] - aps[0][0]) * 1000

            isi_cov = isi_stdev / isi_mean
            if len(aps) < 2:
                isi_cov = np.nan
                burst_length = np.nan
            return isi_cov, burst_length
            


        # Start of function
        frequencies = {}  # {<frequency>: sweep_num, ... }
        for i, sweep in enumerate(self.sweeps):
            sweep_ap_freq = sweep.get_steady_state_ap_frequency()
            frequencies[sweep_ap_freq] = i

        if len(frequencies) == 0:
            logger.warning('No sweep had enough peaks to calculate a frequency')
            return 0.0
        else:
            max_frequency = max(frequencies)
            max_frequency_idx = frequencies[max_frequency]
            #logger.info('Max SSFF is {} from sweep {}'.format(max_frequency, max_frequency_idx))

        if verify:
            verification_plot()

        isi_cov, burst_length = group_pv_cells(self.sweeps[max_frequency_idx])

        return max_frequency, isi_cov, burst_length

    def get_min_instantaneous_firing_frequency(self, verify=False):
        """
        Max instantaneous firing frequency:
        inverse of smallest interspike interval in response to current
        injection (AP amplitude at least 40mV and overshooting 0mV)


        :return:
        """
        max_frequency = float_info.max
        max_frequency_sweep = None
        for i, sweep in enumerate(self.sweeps):
            try:
                sweep_max_freq, max_freq_ap_num = sweep.get_min_instantaneous_ap_frequency()
            except InvalidSweep:
                continue

            if sweep_max_freq < max_frequency:
                max_frequency = sweep_max_freq
                max_frequency_sweep = i

        if verify:
            logger.info('max frequency is in sweep {}'.format(max_frequency_sweep))
            raise NotImplementedError

        return max_frequency
    
    def get_max_instantaneous_firing_frequency(self, verify=False):
        """
        Max instantaneous firing frequency:
        inverse of smallest interspike interval in response to current
        injection (AP amplitude at least 40mV and overshooting 0mV)


        :return:
        """
        max_frequency = float_info.min
        max_frequency_sweep = None
        for i, sweep in enumerate(self.sweeps):
            try:
                sweep_max_freq, max_freq_ap_num = sweep.get_max_instantaneous_ap_frequency()
            except InvalidSweep:
                continue

            if sweep_max_freq > max_frequency:
                max_frequency = sweep_max_freq
                max_frequency_sweep = i

        if verify:
            logger.info('max frequency is in sweep {}'.format(max_frequency_sweep))
            raise NotImplementedError

        return max_frequency

    def _get_ap_threshold_1_details(self, dvdt_threshold=10000):
        """
        AP threshold #1:
        for first spike obtained at suprathreshold current injection, the
        voltage at which first derivative (dV/dt) of the AP waveform reaches
        10V/s = 10000mV/s

        :return: sweep number of threshold measurement, V or threshold, t of threshold
        """

        # iterate through sweeps and peaks until we find the first peak. We will
        # return a result based on that peak.
        for sweep_num, sweep in enumerate(self.sweeps):
            try:
                threshold_voltage, threshold_time = sweep.get_first_ap_threshold(dvdt_threshold)
                return sweep_num, threshold_voltage, threshold_time
            except InvalidSweep:
                pass
        else:
            # If there are no peaks in any sweep we will hit this. This shouldn't happen.
            raise Exception('No sweep had an AP')

    def _get_ap_threshold_1_time(self):
        """


        :return:
        """
        return self._get_ap_threshold_1_details()[2]

    def get_ap_threshold(self):
        """

        :return:
        """
        return self._get_ap_threshold_1_details()[1]

    def get_ahp_amplitude(self):
        """
        Returns a tuple (AHP amp, AHP time)
        Should only be used on brief current files with one action potential
        """
        # Use baseline instead of threshold to calculate AHP amplitutde

        # Find first sweep with 1 AP
        sweep_1_ap = self.sweeps[0]
        sweep_found = False
        for sweep in self.sweeps:
            if sweep.get_ap_count() == 1:
                sweep_1_ap = sweep
                sweep_found = True
                break
        if not sweep_found:
            return np.nan # No sweep with one action potential
        
        smoothed = ndimage.gaussian_filter1d(sweep_1_ap.output_signal, sigma=3)        
        v_smooth = np.array(smoothed)
        v = np.array(sweep_1_ap.output_signal)
        t = np.array(sweep_1_ap.time_steps)
        stimulus_start, stim_end = sweep_1_ap.get_input_start_end()
        baseline_value = baseline_voltage(t, v, stimulus_start, baseline_interval=stimulus_start)
        max_v_idx = np.argmax(v_smooth)
        end_t_idx = np.asarray(t >= t[max_v_idx] + 0.1).nonzero()[0][0]
        new_v = v_smooth[max_v_idx:end_t_idx]
        min_v_idx = np.argmin(new_v)
        lowest_v_pre_ap_idx = np.argmin(v_smooth[:max_v_idx])
        if new_v[min_v_idx] >= v_smooth[lowest_v_pre_ap_idx]: #if lowest post-ap v isn't lower than lowest pre-ap v then it's not hyperpolarization
            return 0
        # if np.asarray(new_v <= baseline_value).nonzero()[0].size == 0: # never dropped to baseline
        #     return 0
        #dropped_to_base_idx = np.asarray(new_v <= baseline_value).nonzero()[0][0]
        ahp = baseline_value - new_v[min_v_idx]
        return ahp


    def get_ap_rise_time(self, verify=False):
        """
        AP rise time:
        for first spike obtained at suprathreshold current injection, time
        from AP threshold 1 to peak

        :return:
        """
        sweep_num, ap_threshold, ap_threshold_time = self._get_ap_threshold_1_details()
        rise_time = self.sweeps[sweep_num].get_first_ap_rise_time()

        if verify:
            raise NotImplementedError

        return rise_time

    def get_ap_amplitude(self, verify=False):
        """
        AP amplitude:
        for first spike obtained at suprathreshold current injection, change
        in mV from AP threshold #1 to peak

        :return:
        """
        sweep_num, ap_threshold_voltage, ap_threshold_time = \
            self._get_ap_threshold_1_details()
        ap_amplitude = self.sweeps[sweep_num].get_first_ap_amplitude()

        if verify:
            raise NotImplementedError

        return ap_amplitude
    
    def get_apd50_90(self):
        """
        Returns Action Potential Duration (APD) 50 and 90 (time it takes to get to 50% repolarization and 90% repolarization 
        from threshold)

        """
        sweep_num, threshold_voltage, ap_threshold_time = self._get_ap_threshold_1_details()
        rheobase_sweep = self.sweeps[sweep_num]
        thresh1_idx = list(rheobase_sweep.time_steps).index(ap_threshold_time)
        thresh2_idx = [i for i, n in enumerate(rheobase_sweep.output_signal) if n <= threshold_voltage and i > thresh1_idx][0]
        aps = rheobase_sweep._get_aps_with_idx()
        ap_idx = aps[rheobase_sweep.default_ap_to_analyze][0]
        ap_voltage = aps[rheobase_sweep.default_ap_to_analyze][2]
        amp50 = ap_voltage - (0.5 * (ap_voltage - threshold_voltage))
        amp90 = ap_voltage - (0.9 * (ap_voltage - threshold_voltage))
        ap_time = rheobase_sweep.time_steps[ap_idx:thresh2_idx + 1]
        ap_vol = rheobase_sweep.output_signal[ap_idx:thresh2_idx + 1]
        amp50_idx = [i for i, n in enumerate(ap_vol) if n <= amp50][0]
        amp90_idx = [i for i, n in enumerate(ap_vol) if n <= amp90][0]
        apd = rheobase_sweep.time_steps[thresh2_idx] - ap_threshold_time
        apd50 = ap_time[amp50_idx] - ap_threshold_time
        apd90 = ap_time[amp90_idx] - ap_threshold_time

        # plots to verify
        # voltage_at_time = dict(zip(rheobase_sweep.time_steps, rheobase_sweep.output_signal))
        # start_idx50 = thresh1_idx
        # # Iterate forward through the data to find the half peak time
        # for idx_diff, time_step in enumerate(rheobase_sweep.time_steps[ap_idx:]):
        #     if voltage_at_time[time_step] <= amp50:
        #         end_idx50 = ap_idx + idx_diff
        #         break

        # plt.close('all')
        # plot_slice_start_idx = start_idx50 - 3 * (end_idx50 - start_idx50)
        # plot_slice_end_idx = end_idx50 + 5 * (end_idx50 - start_idx50)
        # plt.plot(
        #     rheobase_sweep.time_steps[plot_slice_start_idx: plot_slice_end_idx],
        #     rheobase_sweep.output_signal[plot_slice_start_idx: plot_slice_end_idx]
        # )

        # plt.axhline(threshold_voltage, color='b')
        # plt.axhline(rheobase_sweep.output_signal[ap_idx], color='b')

        # half_width_amp = 0.5 * (rheobase_sweep.output_signal[ap_idx] + threshold_voltage)
        # plt.axhline(half_width_amp, color='r')

        # plt.axvline(rheobase_sweep.time_steps[ap_idx], color='g')
        # plt.axvline(rheobase_sweep.time_steps[start_idx50], color='r')
        # plt.axvline(rheobase_sweep.time_steps[end_idx50], color='r')


        # plt.savefig("zoomed_{}".format(rheobase_sweep.sweep_name))


        return apd * 1000, apd50 * 1000, apd90 * 1000
    
    def get_velocities(self):
        """
        Returns maximum derivative on the upstroke and downstroke

        """
        sweep_num, ap_threshold, ap_threshold_time = self._get_ap_threshold_1_details()
        rheo_sweep = self.sweeps[sweep_num]
        thresh1_idx = list(rheo_sweep.time_steps).index(ap_threshold_time)
        thresh2_idx = [i for i, n in enumerate(rheo_sweep.output_signal) if n <= ap_threshold and i >= thresh1_idx][1]
        dv_dt = rheo_sweep.get_output_derivative()
        aps = rheo_sweep._get_aps_with_idx()
        ap_idx = aps[rheo_sweep.default_ap_to_analyze][0]
        upstroke_vel = 0
        for item in dv_dt[thresh1_idx:ap_idx]:
            if item > upstroke_vel:
                upstroke_vel = item
        downstroke_vel = 0
        for item in dv_dt[ap_idx:thresh2_idx + 1]:
            if item < downstroke_vel:
                downstroke_vel = item

        # Baseline during firing
        # start_idx = None
        # end_idx = None
        # start, end = rheo_sweep.get_input_start_end()
        # for idx, t in enumerate(rheo_sweep.time_steps):
        #     if t >= start and start_idx is None:
        #         start_idx = idx
        #     if t >= end and end_idx is None:
        #         end_idx = idx

        # med_v = np.median(np.asarray(rheo_sweep.output_signal[start_idx:end_idx]))

        return upstroke_vel / 1000, downstroke_vel / 1000

    def get_ap_half_width_and_peak(self, verify=False):
        """
        AP half-width:
        for first spike obtained at suprathreshold current injection, width
        of the AP (in ms) at 1/2 maximal amplitude, using AP threshold #1 and
        AP amplitude

        :return:
        """
        # Check if we have any sweeps with APs
        rheobase_sweep_num = self._get_rheobase_sweep_num()
        if rheobase_sweep_num is None:
            # No sweep has sufficient APs - return NaN values
            return np.nan, np.nan
        def verification_plot():
            plt.close('all')
            plot_slice_start_idx = peak_start_idx - 3 * (peak_end_idx - peak_start_idx)
            plot_slice_end_idx = peak_end_idx + 5 * (peak_end_idx - peak_start_idx)
            plt.plot(
                rheobase_sweep.time_steps[plot_slice_start_idx: plot_slice_end_idx],
                rheobase_sweep.output_signal[plot_slice_start_idx: plot_slice_end_idx]
            )

            # TODO add text for threshold, peak, half, horizontal lines
            plt.axhline(threshold_voltage, color='b')
            plt.axhline(rheobase_sweep.output_signal[ap_idx], color='b')

            half_width_amp = 0.5 * (rheobase_sweep.output_signal[ap_idx] + threshold_voltage)
            plt.axhline(half_width_amp, color='r')

            plt.axvline(rheobase_sweep.time_steps[peak_start_idx])
            plt.axvline(rheobase_sweep.time_steps[peak_end_idx])

            plt.text(
                rheobase_sweep.time_steps[plot_slice_end_idx],
                0.5 * (ap_peak_voltage + half_width_amp),
                'AP half width ={:.2g}'.format(ap_half_width),
                horizontalalignment='right',
                verticalalignment='top')

            plt.savefig("{}".format(rheobase_sweep.sweep_name))

        #logger.info('Getting AP half-width from the first AP elicited')

        rheobase_sweep = self.sweeps[rheobase_sweep_num]
        
        # Check if this sweep actually has APs
        aps = rheobase_sweep.get_aps()
        if len(aps) == 0 or len(aps) <= rheobase_sweep.default_ap_to_analyze:
            # No APs or not enough APs for analysis - return NaN values
            return np.nan, np.nan
            
        half_width_ap = aps[rheobase_sweep.default_ap_to_analyze]
        
        # Get threshold voltage (this method handles its own error cases)
        try:
            threshold_voltage = self.get_ap_threshold()
            if np.isnan(threshold_voltage):
                return np.nan, np.nan
        except Exception:
            # If threshold calculation fails, return NaN
            return np.nan, np.nan
        ap_time, ap_peak_voltage = half_width_ap
        half_peak_voltage = 0.5 * (ap_peak_voltage + threshold_voltage)
        
        # Get AP index (this may fail if no valid APs)
        try:
            ap_idx = rheobase_sweep.get_first_ap_peak_data_index()
        except Exception:
            # If AP index calculation fails, return NaN
            return np.nan, np.nan
        voltage_at_time = dict(zip(rheobase_sweep.time_steps, rheobase_sweep.output_signal))

        peak_start = None
        peak_end = None
        # Iterate back through the data to find the half peak time
        for idx_diff, time_step in enumerate(rheobase_sweep.time_steps[ap_idx::-1]):
            if voltage_at_time[time_step] <= half_peak_voltage:
                #logger.info('Found peak start at {}'.format(time_step))
                peak_start = time_step
                peak_start_idx = ap_idx - idx_diff
                break

        # Iterate forward through the data to find the half peak time
        for idx_diff, time_step in enumerate(rheobase_sweep.time_steps[ap_idx:]):
            if voltage_at_time[time_step] <= half_peak_voltage:
                #logger.info('Found peak end at {}'.format(time_step))
                peak_end = time_step
                peak_end_idx = ap_idx + idx_diff
                break

        # Check if we found valid peak boundaries
        if peak_start is None or peak_end is None:
            # Could not determine half-width boundaries - return NaN
            return np.nan, np.nan
            
        ap_half_width = peak_end - peak_start

        if verify:
            verification_plot()
        
        return 1000*ap_half_width, ap_peak_voltage
    
    def get_time_constant(self):
        time_const = 0
        num_added = 0
        sweep = self.sweeps[0]
        current = sweep.get_drive_current()
        if current < -5:
            t = np.array(sweep.time_steps)
            v = np.array(sweep.output_signal)
            # Calculate stimulus interval
            stimulus_start, stimulus_end = sweep.get_input_start_end()
            tau = time_constant(t, v, current, stimulus_start, stimulus_end,
                                                    baseline_interval=stimulus_start, frac=0.01)
            if(not np.isnan(tau)):
                time_const += tau
                num_added += 1

        if num_added == 0:
            return np.nan # Fitting error is too large or no good window was found to fit or could not calculate current
        else:
            return 1000 * (time_const / num_added)
    
    def get_sag(self):
        # Check if first sweep has hyperpolarizing current for sag analysis
        first_sweep = self.sweeps[0]
        current = first_sweep.get_drive_current()
        
        # Only calculate sag if current is sufficiently hyperpolarizing
        if current < -5.0 and not np.isnan(current):
            t = np.array(first_sweep.time_steps)
            v = np.array(first_sweep.output_signal)
            # Calculate stimulus interval
            stimulus_start, stimulus_end = first_sweep.get_input_start_end()
            curr_sag = sag(t, v, current, stimulus_start, stimulus_end)
            return curr_sag
        else:
            # No hyperpolarizing current available for sag analysis
            return np.nan
    
    def get_attenuation(self):
        if len(self.sweeps) < 10: #change these numbers to adjust which sweep this is calculated on
            return [np.nan], [np.nan]
        sweep = self.sweeps[9] 
        aps = sweep.get_aps()
        peaks = [peak[1] for peak in aps]
        nums = [i for i in range(len(peaks))]

        return nums, peaks
    
    def get_isi_cov_and_burst_length(self):
        isi_covs = []
        burst_lengths = []
        for sweep in self.sweeps:
            aps = sweep.get_aps()
            all_isi = []
            burst_length = None

            for i in range(len(aps) - 1):
                isi = (aps[i + 1][0] - aps[i][0]) * 1000
                all_isi.append(isi)

            isi_mean = np.mean(np.array(all_isi))
            isi_stdev = np.std(np.array(all_isi))

            for i in range(len(all_isi)):
                if all_isi[i] > (isi_mean + (3*isi_stdev)): #is above 3 stdevs
                    burst_length = (aps[i][0] - aps[0][0]) * 1000
                    break
            if burst_length == None and len(aps) != 0:
                burst_length = (aps[-1][0] - aps[0][0]) * 1000

            isi_cov = isi_stdev / isi_mean
            if len(aps) < 2:
                isi_cov = np.nan
                burst_length = np.nan
            isi_covs.append(isi_cov)
            burst_lengths.append(burst_length)
        
        return isi_covs, burst_lengths
                

    def plot_v_vs_i(self, sweep_num):
        """
        Just for testing

        :param sweep_num:
        :return:
        """
        sweep = self.sweeps[sweep_num]
        fig, ax1 = plt.subplots()
        ax1.plot(sweep.output_signal, sweep.input_signal)
        plt.show()


def get_file_list(abf_location):
    """
    Figure out which file(s) to analyze based on the location(s) specified in the config file.
    Locations can be strings with the name of a file / folder path, or a list of strings.

    :param abf_location: The abf location as extracted from config
    :return:
    """
    if isinstance(abf_location, list):
        abf_location_list = abf_location
    else:
        abf_location_list = [abf_location]

    abf_files = []
    error = False
    for path in abf_location_list:
        if not os.path.exists(path):
            logger.error('File {} not found'.format(path))
            error = True
        if os.path.isdir(path):
            abf_files += [f for f in os.listdir(path) if f.endswith(ABF_FILE_EXTENSION)]
        elif os.path.isfile(path):
            if path.endswith(ABF_FILE_EXTENSION):
                abf_files.append(path)

    if error:
        raise ValueError('Specified location for abd files does not exist')

    logger.info('Found {} files to analyze'.format(len(abf_files)))
    return abf_files

