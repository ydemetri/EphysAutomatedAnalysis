from scipy.optimize import minimize
import matplotlib.pyplot as plt
import numpy as np
import logging
import os.path

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


def fit_tophat(x, y, verify=False):
    """
    Fit the x and y data to a tophat function, returning:
    base_level - the y-value inside the tophat
    hat_level - the y-value outside the tophat
    hat_start - start of the tophat
    hat_end - end of the tophat

    :param x: iterable of x values
    :param y: corresponding iterable of y values
    :param verify: Show a plot of the fit, blocking progress until it is dismissed
    :param verify_file:
    :return: (base_level, hat_level, hat_start, hat_end)
    """
    # TODO Try to spot close to zero height tophats, which may confuse the algorithm

    def top_hat(x, base_level, hat_level, hat_start, hat_end):
        return np.where((hat_start < x) & (x < hat_end), hat_level, base_level)

    gradient = list(get_derivative(y, x))
    max_gradient = max(gradient)
    min_gradient = min(gradient)

    # The tophat could be upside down, so we don't know which of these comes
    # in the x direction
    max_gradient_index = gradient.index(max_gradient)
    min_gradient_index = gradient.index(min_gradient)
    step_indices = (max_gradient_index, min_gradient_index)

    max_gradient_x = x[max_gradient_index]
    min_gradient_x = x[min_gradient_index]
    step_xs = (max_gradient_x, min_gradient_x)

    base_level = np.mean(y[:min(step_indices)])
    hat_level = np.mean(y[min(*step_indices):max(*step_indices)])
    hat_start = min(*step_xs)
    hat_end = max(*step_xs)

    if verify:
        plt.close('all')
        plt.figure(figsize=(8, 5))
        plt.plot(x, y)
        plt.plot(x, top_hat(x, base_level, hat_level, hat_start, hat_end))
        plt.show()

    return base_level, hat_level, hat_start, hat_end


def find_peaks(sweep, x, y, threshold=0, verify=False):
    """
    Count signficant spikes in y values above threshold, for some definition
    of "significant". This is a very naive approach - any time the trace goes
    then below the threshold, find the highest value while it is above. That
    is the peak. It works so far. scipy.signal.findpeaks may provide a more
    robust approach if needed, using a peaks "prominence".

    :param x: list of x values
    :param y: list of y values
    :param threshold: y threshold that neesd to be crossed to count as a peak
    :param verify: If True, a plot showing peaks found will be shown, or saved to file if verify=='offline'
    :return: list of (idx, x, y) values of peaks at point of maximum y
    """
    logger.debug('Peak threshold is {}'.format(threshold))

    in_peak = False
    peak = []  # list of data points that are part of a peak
    peaks = []  # list or peaks
    for idx, data_point in enumerate(zip(x, y)):
        if in_peak:
            if data_point[1] > threshold:
                peak.append((idx, data_point[0], data_point[1]))
            else:
                in_peak = False
                peaks.append(peak)
        elif data_point[1] > threshold:
            in_peak = True
            peak = [(idx, *data_point)]

    # print([peak[0] for peak in peaks])
    # if len(peaks) > 0:
    #     print([max(peak, key=lambda d: d[1]) for peak in peaks])
    # else:
    #     print('No peaks')
    # print(len(peaks))

    maximums = [max(peak, key=lambda d: d[2]) for peak in peaks]

    if verify:
        plt.close(fig='all')  # Make sure there are no unwanted figures
        plt.figure(figsize=(16, 10))
        plt.plot(x, y)
        plt.xlabel("Time (s)")
        plt.ylabel("Voltage (mV)")
        plt.title("Action Potentials Identified for {}".format(sweep))
        for m in maximums:
            plt.axvline(x=m[1], color='red', label="peak")
        plt.show()
        #plt.savefig("C:\\Users\\Yiannos\\Documents\\Mattis Lab\\MattisLab\\Test3\\io_{}.png".format(self.sweep_name))

    logger.info('Found {} peaks'.format(len(maximums)))
    return maximums

def get_derivative(y, x):
    """
    Return the numerical derivatice of the data dy/dx
    :param y: y values list
    :param x: x values list
    :return: dy/dx
    """
    return np.gradient(y, x)

