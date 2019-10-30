#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Resource Utilization Anomaly Detection

Description
In a large data center there are many different devices that make up the entire network fabric.
The health of these devices and their statuses need to be closely monitored so that early device failures
 can be detected in order to prevent any unnecessary downtime.
 One of the important health indicators is CPU utilization.

In this project you are required to analyze CPU utilization data (CUD) of a machine.
Additionally, we ask that you implement a feature that will automatically
detect any CUD anomalies in real-time without the need of configuring a threshold.

Bonus Points:
* Implement your solution using a big data analytics framework such as Spark Streaming
* Provide a front-end UI implementation with the following features (in order of difficulty):
o illustration of CPU utilization
o illustration of anomalies collected
o ability to choose a specific time window and view CPU utilization and/or anomalies for that time window
o real-time updates for time-series data and anomalies detected

Date Files:
* Please download datasets from:
https://www.dropbox.com/s/tud9bp35fukegah/data.csv?dl=0
* data.csv: the training set. It has two fields:
o  first column: timestamp
o second column: CPU utilization in percentage

Rules
* You need to finish the project independently.
* You are welcome to use Google for inspiration, however we have zero tolerance for plagiarism.
* Sharing code or data outside of this project is not permitted.
  This includes making code or data available to other candidates.
* You have up to a week to finish your project and get ready for the onsite interview.
  Once your project is done, please send us your code for reviewing.
  It can be hosted in github, bitbucket, etc.
  Or you can simply compress the project and send it to us at: ksheng@equinix.com
* You are welcome to email us if you need clarification on the project requirements.
* You can write your solution in any programming language that you choose.
  Your coding skills will be evaluated further when you come on site.
"""
from __future__ import absolute_import, division, print_function

__copyright__ = 'Public Domain'
__author__ = 'Inga Kuznetsova'


"""
This code depends on the following packages:
    pip install pandas matplotlib statsmodels sklearn numpy pandas
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA
import numpy as np
from datetime import datetime
from sklearn.metrics import mean_squared_error

import warnings

"""The order (number of time lags)
    of the non-seazonal autoregressive model ARIMA(p,d,q)

    Must be a non-negative integer
"""
ARIMA_P = 0

"""The degree of differencing (the number of times the data have had past values subtracted)
    of the non-seazonal autoregressive model ARIMA(p,d,q)

Must be a non-negative integer
"""
ARIMA_D = 1

"""The order of the moving-average model
   of the non-seazonal autoregressive model ARIMA(p,d,q)

Must be a non-negative integer
"""
ARIMA_Q = 0

"""Test file"""
TEST_FILE = os.path.abspath(os.path.dirname(__file__) + '/../tests/data.csv')

"""Default data chunk size"""
DEFAULT_CHUNK_SIZE = 1000

"""Dafault rolling windows size"""
DEFAULT_ROLLING_WINDOW_SIZE = 5


class AnomalyDetector(object):
    """Resource Utilization Anomaly Detection

    Implements resource utilization anomaly detection using non-seasonal
    autoregressive integrated moving average method ARIMA(p,d,q)
    Non-seasonal ARIMA models are generally denoted ARIMA(p,d,q)
    where parameters p, d, and q are non-negative integers,
       - p is the order (number of time lags) of the autoregressive model,
       - d is the degree of differencing (the number of times the data have had past values subtracted),
       - q is the order of the moving-average model.
    See more information at https://en.wikipedia.org/wiki/Autoregressive_integrated_moving_average

     How to test this code:
        Usage:

                python anomaly_detection.py stream-test
                    Prints outliers from a "real-time" input stream of (timestamp, cpu_usage) pairs
                    The model auto-tunes up its parameters from the data stream
                    The input stream comes from the CSV 'tests/data.csv'
                    Sample printout:
                        Outlier = (1476400708.0, 0.18241668092452884)
                        Outlier = (1476400931.0, 0.8239113982345683)
                        Outlier = (1476401114.0, 0.21467013167017163)
                        ...
                    Outliers are printed in as soon as they are detected

                python anomaly_detection.py file-test
                    Prints outliers from the CSV file'tests/data.csv'
                    The file is processed in bulk (faster than processing from a real-time)
                    IMPORTANT: Outliers are printed at the very end.
                                Please be patient and wait until the whole file is processed

    """

    def get_quantile_outliers(
        self,
        file_name=TEST_FILE,
        quantile=0.05,
        rolling_window_size=DEFAULT_ROLLING_WINDOW_SIZE,
        chunksize=DEFAULT_CHUNK_SIZE
    ):
        """Computes quantile-based outliers in the input data sed

        :param file_name: Input data set file name
        :type file_name: str
        :param quantile: input quantile. Default value: 0.05 (5%)
        :type quantile: float
        :param rolling_window_size: Rolling window size
        :type rolling_window_size: int
        :param chunksize: Input file reading chunk size
        :type chunksize: int
        :return: A tuple of two numpy arrays containing low/high end outliers
        :rtype: tuple
        """
        anom_min = pd.DataFrame()
        anom_max = pd.DataFrame()
        data_anom_max = pd.DataFrame()
        data_anom_min = pd.DataFrame()
        for chunk in pd.read_csv(file_name, chunksize=chunksize, date_parser=True):
            cpu_usage_min = pd.rolling_quantile(
                chunk.cpu_usage, window=rolling_window_size, quantile=quantile)
            cpu_usage_max = pd.rolling_quantile(
                chunk.cpu_usage, window=rolling_window_size, quantile=1 - quantile)
            # print(cpu_usage_max)
            anom_min = chunk.loc[chunk.cpu_usage <
                                 cpu_usage_min, ['time', 'cpu_usage']]
            anom_max = chunk.loc[chunk.cpu_usage >
                                 cpu_usage_max, ['time', 'cpu_usage']]
            data_anom_min = data_anom_min.append(anom_min).dropna()
            data_anom_max = data_anom_max.append(anom_max).dropna()
            print (data_anom_max)
        return data_anom_max, data_anom_min

    def get_n_sigma_outliers(
        self,
        file_name=TEST_FILE,
        rolling_window_size=DEFAULT_ROLLING_WINDOW_SIZE,
        n_sigma=3.0,
        chunksize=DEFAULT_CHUNK_SIZE,
    ):
        """Computes outliers using rolling n-sigma deviation from the rolling average

        :param file_name: Input data set file name
        :type file_name: str
        :param n_sigma: n-sigma outlier detection parameter
        :type rolling_window_size: float. Default value: 3.00
        :param chunksize: Input file reading chunk size
        :type chunksize: int
        :return: numpy array containing outliers
        :rtype: numpy array
        """
        input_data_frame = pd.DataFrame()
        outliers = pd.DataFrame()
        for chunk in pd.read_csv(file_name, chunksize=chunksize, date_parser=True):
            # Compute rolling mean
            cpu_usage_mean = chunk.cpu_usage.rolling(
                window=rolling_window_size).mean()
            cpu_usage_mean = cpu_usage_mean.fillna(chunk['cpu_usage'].mean())
            # Compute rolling standard deviation
            cpu_usage_std = chunk.cpu_usage.rolling(
                window=rolling_window_size).std()
            # rolling std
            cpu_usage_std = cpu_usage_std.fillna(chunk['cpu_usage'].std())
            # Compute anomaly detection condition
            anomaly_condition = (
                chunk.loc[abs(chunk.cpu_usage - cpu_usage_mean) >
                          n_sigma * cpu_usage_std, ['time', 'cpu_usage']]
            )
            outliers = outliers.append(anomaly_condition)
        return outliers

    def get_arima_outliers_from_file(
        self,
        file_name=TEST_FILE,
        rolling_window_size=DEFAULT_ROLLING_WINDOW_SIZE,
        n_sigma=3.0,
        arima_p=ARIMA_P,
        arima_d=ARIMA_D,
        arima_q=ARIMA_Q,
        chunksize=DEFAULT_CHUNK_SIZE
    ):
        """Computes outliers using rolling n-sigma deviation from the rolling average

        :param file_name: Input data set file name
        :type file_name: str
        :param n_sigma: n-sigma outlier detection parameter
        :type rolling_window_size: float. Default value: 3.00
        :param arima_p: The order (number of time lags) of the non-seazonal autoregressive model ARIMA(p,d,q)
        :type int       Must be a non-negative integer
        :param arima_d: The degree of differencing (the number of times the data have had past values subtracted)
                        of the non-seazonal autoregressive model ARIMA(p,d,q)
        :type int       Must be a non-negative integer
        :param arima_q: The order of the moving-average model of the non-seazonal autoregressive model ARIMA(p,d,q)
        :type int       Must be a non-negative integer
        :param chunksize: Input file reading chunk size
        :type chunksize: int
        :return: numpy array of outlier points
        :rtype: numpy array
        """
        predicted_data_points = pd.DataFrame()
        outliers = pd.DataFrame()
        for chunk in pd.read_csv(file_name, chunksize=chunksize):
            # Set outlier indexes
            outlier_indexes, prediction_difference = self.get_arima_outlier_indexes(
                input_points=chunk,
                rolling_window_size=rolling_window_size,
                n_sigma=n_sigma,
                arima_p=arima_p,
                arima_d=arima_d,
                arima_q=arima_q,
            )
            outliers = outliers.append(chunk.shift(-1).iloc[outlier_indexes])

        return outliers

    def plot_outliers(self, input_points, cpu_usage_mean, outlier_indexes):
        """Plots outliers

        :param input_points: input data points
        :param cpu_usage_mean: rolling mean of the data points
        :param outlier_indexes: Outlier indexes
        """
        plot_input = input_points.shift(-1).iloc[outlier_indexes]
        plt.plot(
            input_points['time'].iloc[len(input_points['time']) - 400:],
            cpu_usage_mean.iloc[len(input_points['time']) - 400:]
        )
        plt.plot(
            input_points['time'].iloc[len(input_points['time']) - 400:],
            input_points['cpu_usage'].iloc[len(input_points['time']) - 400:]
        )
        #ax.Axes.set_xlim(0, len(chunk['time'])-400)
        plt.scatter(['time'].iloc[len(plot_input['time']) - 2:],
                    plot_input['cpu_usage'].iloc[len(plot_input['time']) - 2:]
                    )

    def get_arima_outlier_indexes(
            self,
            input_points,
            rolling_window_size=DEFAULT_ROLLING_WINDOW_SIZE,
            n_sigma=3.0,
            arima_p=ARIMA_P,
            arima_d=ARIMA_D,
            arima_q=ARIMA_Q,
    ):
        """Computes outliers using rolling n-sigma deviation from the rolling average

        :param input_points: input data frame
        :type input_points:  DataFrame
        :param n_sigma: n-sigma outlier detection parameter
        :type rolling_window_size: float. Default value: 3.00
        :param arima_p: The order (number of time lags) of the non-seazonal autoregressive model ARIMA(p,d,q)
        :type int       Must be a non-negative integer
        :param arima_d: The degree of differencing (the number of times the data have had past values subtracted)
                        of the non-seazonal autoregressive model ARIMA(p,d,q)
        :type int       Must be a non-negative integer
        :param arima_q: The order of the moving-average model of the non-seazonal autoregressive model ARIMA(p,d,q)
        :type int       Must be a non-negative integer
        :return: numpy array of outlier points
        :rtype: a pair of numpy arrays (outliers, prediction_difference)
        """
        predicted_data_points = pd.DataFrame()
        outliers = pd.DataFrame()
        anom = pd.DataFrame()
        input_points['time'] = input_points['time'].apply(
            lambda x: datetime.fromtimestamp(x))
        cpu_usage_mean = input_points.cpu_usage.rolling(
            window=rolling_window_size).mean()
        cpu_usage_mean = cpu_usage_mean.fillna(input_points['cpu_usage'].mean())
        input_points.index = pd.DatetimeIndex(input_points.time)
        del input_points['time']

        np_chunk = np.array(input_points['cpu_usage'].iloc[1:])
        np_cpu_usage_mean = np.array(cpu_usage_mean.iloc[1:])

        # shift cpu_usage by mean to compare to ARIMA fit
        mean_centered_data_points = np_chunk - np_cpu_usage_mean

        # Fit the model
        predicted_data_points = ARIMA(input_points, order=(
            arima_p, arima_d, arima_q)).fit(disp=-1, iprint=0)

        # Find difference between predicted and input data points
        prediction_difference = abs(
            np.array(predicted_data_points.fittedvalues) - mean_centered_data_points)

        # Compute error
        rmse = np.sqrt(
            mean_squared_error(
                mean_centered_data_points,
                np.array(predicted_data_points.fittedvalues)
            ))

        # Compute outlier_indexes
        outlier_indexes = np.where(prediction_difference > n_sigma * rmse)
        return outlier_indexes, prediction_difference

    def get_arima_outliers_from_stream(
            self,
            input_stream=sys.stdin,
            min_rolling_window_size=DEFAULT_ROLLING_WINDOW_SIZE,
            n_sigma=3.0,
            arima_p=ARIMA_P,
            arima_d=ARIMA_D,
            arima_q=ARIMA_Q,
            min_chunksize=300,
            tuneup=True,
            debug=True,
    ):
        """Computes outliers using rolling n-sigma deviation from the rolling average

        :param input_stream: Input data stream
        :type input_stream: point sequence
        :param n_sigma: n-sigma outlier detection parameter
        :type min_rolling_window_size: float. Default value: 3.00
        :param arima_p: The order (number of time lags) of the non-seazonal autoregressive model ARIMA(p,d,q)
        :type int       Must be a non-negative integer
        :param arima_d: The degree of differencing (the number of times the data have had past values subtracted)
                        of the non-seazonal autoregressive model ARIMA(p,d,q)
        :type int       Must be a non-negative integer
        :param arima_q: The order of the moving-average model of the non-seazonal autoregressive model ARIMA(p,d,q)
        :type int       Must be a non-negative integer
        :param min_chunksize: Input file reading chunk size
        :type min_chunksize: int
        :param tuneup: Auto-tuneup mode (if True, find the best chunksize)
        :type tuneup: bool
        :return: numpy array of outlier points
        :rtype: numpy array
        """

        max_chunksize = min_chunksize * 10                     # 10x optimization range
        max_rolling_window_size = min_rolling_window_size * 10 # 10x optimization range

        current_chunksize = min_chunksize
        current_rolling_window_size = min_rolling_window_size

        chunksize_tuneup_in_progress = tuneup
        rolling_window_tuneup_in_progress = tuneup

        self.tuneup_map = {}

        for timestamp, cpu_usage in input_stream:
            new_data = pd.DataFrame([[timestamp, cpu_usage]], columns=[
                                    'time', 'cpu_usage']).set_index('time')

            self.rolling_points = self.rolling_points.append(new_data)

            if len(self.rolling_points) <= current_chunksize:
                continue

            # remove the oldest point
            self.rolling_points = self.rolling_points[-current_chunksize:]

            chunk = self.rolling_points.reset_index()

            predicted_data_points = pd.DataFrame()
            outliers = pd.DataFrame()
            # Set outlier indexes
            outlier_indexes, prediction_difference = self.get_arima_outlier_indexes(
                input_points=chunk,
                rolling_window_size=current_rolling_window_size,
                n_sigma=n_sigma,
                arima_p=arima_p,
                arima_d=arima_d,
                arima_q=arima_q,
            )

            for timestamp, cpu_usage in self.rolling_points.shift(-1).iloc[outlier_indexes].reset_index().values.tolist():
                result = (timestamp, cpu_usage)
                if result in self.duplicates_cache:
                    continue

                self.duplicates_cache.add(result)

                # Adjust duplicates cache
                if len(self.duplicates_cache) > 10 * current_chunksize:
                    self.duplicates_cache = set(sorted(self.duplicates_cache)[5 * current_chunksize:])
                yield result

            if not tuneup:
                continue

            if chunksize_tuneup_in_progress:
                # Optimize chunk size
                if current_chunksize < max_chunksize:
                    # Collect tuneup stats
                    self.tuneup_map[current_chunksize] = prediction_difference.mean()
                    if debug:
                        print ('DEBUG: chunksize tuneup-in-progress:', dict(
                            current_chunksize=current_chunksize,
                            tuneup_map=self.tuneup_map
                        ))
                    current_chunksize += min_chunksize
                else:
                    # Choose the best chunksize
                    stats = sorted((v, k) for k, v in self.tuneup_map.items())
                    min_difference, best_chunksize = stats[0]
                    current_chunksize = best_chunksize
                    chunksize_tuneup_in_progress = False

                    if debug:
                        print ('DEBUG: chunksize tuneup-completed:', dict(
                            current_chunksize=current_chunksize,
                            tuneup_map=self.tuneup_map,
                        ))

                    self.tuneup_map = {}
            else:
                # Optimize rolling window
                if rolling_window_tuneup_in_progress:
                    if current_rolling_window_size < max_rolling_window_size:
                        # Collect tuneup stats
                        self.tuneup_map[current_rolling_window_size] = prediction_difference.mean()
                        if debug:
                            print ('DEBUG: window tuneup-in-progress:', dict(
                                current_rolling_window_size=current_rolling_window_size,
                                tuneup_map=self.tuneup_map
                            ))
                        current_rolling_window_size += min_rolling_window_size
                    else:
                        # Choose the best rolling window size
                        stats = sorted((v, k) for k, v in self.tuneup_map.items())
                        min_difference, best_rolling_window_size = stats[0]
                        current_rolling_window_size = best_rolling_window_size
                        rolling_window_tuneup_in_progress = False

                        if debug:
                            print ('DEBUG: window tuneup-completed:', dict(
                                current_rolling_window_size=current_rolling_window_size,
                                tuneup_map=self.tuneup_map,
                            ))

    def __init__(self, min_rolling_points=60 * 5):
        super(AnomalyDetector, self).__init__()
        self.rolling_points = pd.DataFrame([[pd.Timestamp(0), 0]], columns=[
                                           'time', 'cpu_usage']).set_index('time')[:0]
        self.duplicates_cache = set()
        self.tuneup_map = {}


def data_stream(csv_stream):
    """Parses stream of CSV data, converts it into a stream of (timestamp, cpu_usage) tuples

    :param csv_stream: input csv stream
    :type csv_stream: stream of CSV strings
    :return: stream of (timestamp, cpu_usage) tuples
    :rtype: tuple
    """
    for line in csv_stream:
        line = line.strip()
        if line.startswith('time'):  # ignore header
            continue
        timestamp, cpu_usage = line.split(',')
        timestamp = int(timestamp)
        cpu_usage = float(cpu_usage)
        yield timestamp, cpu_usage


def file_test():
    """File processing test"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        anomaly_detector = AnomalyDetector()
        print (anomaly_detector.get_arima_outliers_from_file())


def stream_test():
    """Stream processing test"""
    anomaly_detector = AnomalyDetector()

    # print(anomaly_detector.get_arima_outliers_from_file())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for outlier in anomaly_detector.get_arima_outliers_from_stream(data_stream(open(TEST_FILE))):
            print('..... Outlier =', outlier)


if __name__ == '__main__':
    if len(sys.argv) == 2 and sys.argv[1] == 'stream-test':
        stream_test()
    if len(sys.argv) == 2 and sys.argv[1] == 'file-test':
        file_test()
    else:
        print(
            """
            Usage: 

                python anomaly_detection.py stream-test
                    Prints outliers from a "real-time" input stream of (timestamp, cpu_usage) pairs
                    The model auto-tunes up its parameters from the data stream
                    The input stream comes from the CSV 'tests/data.csv'
                    Sample printout:
                        Outlier = (1476400708.0, 0.18241668092452884)
                        Outlier = (1476400931.0, 0.8239113982345683)
                        Outlier = (1476401114.0, 0.21467013167017163)
                        ...
                    Outliers are printed in as soon as they are detected
                        
                python anomaly_detection.py file-test
                    Prints outliers from the CSV file'tests/data.csv'
                    The file is processed in bulk (faster than processing from a real-time)
                    IMPORTANT: Outliers are printed at the very end. 
                                Please be patient and wait until the whole file is processed
            """
        )
