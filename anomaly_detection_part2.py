from __future__ import absolute_import, division, print_function
__copyright__ = 'Public Domain'
__author__ = 'Inga Kuznetsova'
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 11:01:00 2017

@author: inga_
"""

#!/usr/bin/env python
# -*- coding: utf-8 -*-


import sys
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn import linear_model as lm




"""Test file"""
TEST_FILE = 'tests/data.csv'

"""Default data chunk size"""
DEFAULT_CHUNK_SIZE = 10000.

"""Dafault rolling windows size for linear regression model"""
DEFAULT_ROLLING_WINDOW_SIZE_1ST_TYPE_ANOMALY = 3

"""Dafault rolling windows size for anomaly for average over time interval"""
DEFAULT_ROLLING_WINDOW_SIZE_2ND_TYPE_ANOMALY = 20




class AnomalyDetector(object):
    """Resource Utilization Anomaly Detection

    Implements resource utilization anomaly detection using linear regression
    model to fit and predict and compare to data to find outliers. Also time intervals with 
    high average cpu are considered as anomaly.
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
                    In this case only linear regression model is considered.
                    

                python anomaly_detection.py file-test
                    Prints outliers from the CSV file'tests/data.csv'
                    In this case two types of anomaly are considered.
                    In case of 2nd type anomaly on time interval in n seconds,
                    the last time point in interval is printed.
                    The file is processed in bulk (faster than processing from a real-time)
                    IMPORTANT: Outliers are printed at the very end.
                                Please be patient and wait until the whole file is processed

    """

   

    def get_average_outliers(
        self,
        file_name=TEST_FILE,
        rolling_window_size=DEFAULT_ROLLING_WINDOW_SIZE_2ND_TYPE_ANOMALY,
        n_sigma=3.0,
        chunksize=DEFAULT_CHUNK_SIZE,
    ):
        """Defines 2nd type anomaly when signal is high or low for relatively long time rolling window
        (default 10 s).
        Computes outliers using n-sigma deviation of rolling average from the whole chunk average.
        It is similar to assumption that instead of each second signal we have signal average signal 
        each n seconds. 
        :param file_name: Input data set file name
        :type file_name: str
        :param n_sigma: n-sigma outlier detection parameter
        :type rolling_window_size: float. Default value: 3.00
        :param chunksize: Input file reading chunk size
        :type chunksize: int
        :return: numpy array containing outliers
        :rtype: numpy array
        """
        outliers = pd.DataFrame()
        for chunk in pd.read_csv(file_name, chunksize=chunksize, date_parser=True):
            # Compute rolling mean
            chunk['cpu_usage_mean'] = chunk.cpu_usage.rolling(
                window=rolling_window_size).mean()
            chunk=chunk.dropna()
            #compute mean from rolling mean
            cpu_usage_mean_mean = chunk['cpu_usage_mean'].mean()
            #compute std for rolling mean
            cpu_usage_mean_std =  chunk['cpu_usage_mean'].std()
         # Compute anomaly detection condition
            anomaly_condition = (
                chunk.loc[abs(chunk.cpu_usage_mean - cpu_usage_mean_mean) >
                          n_sigma * cpu_usage_mean_std, ['time', 'cpu_usage_mean']]
            )
            outliers = outliers.append(anomaly_condition)
        return outliers

    def get_linear_regression_outliers_from_file(
        self,
        file_name=TEST_FILE,
        rolling_window_size=DEFAULT_ROLLING_WINDOW_SIZE_1ST_TYPE_ANOMALY,
        n_sigma=3.0,
        chunksize=DEFAULT_CHUNK_SIZE
    ):
        """Computes outliers using rolling n-sigma deviation from the rolling average
        :param file_name: Input data set file name
        :type file_name: str
        :param n_sigma: n-sigma outlier detection parameter
        :type rolling_window_size: float. Default value: 3.00
        :param chunksize: Input file reading chunk size
        :type chunksize: int
        :return: numpy array of outlier points
        :rtype: numpy array
        """
        outliers = pd.DataFrame()
        for chunk in pd.read_csv(file_name, chunksize=chunksize):
            # Set outlier indexes
            outlier_indexes, prediction_difference = self.get_linear_regression_outlier_indexes(
                input_points=chunk,
                rolling_window_size=rolling_window_size,
                n_sigma=n_sigma,
            )
            outliers = outliers.append(chunk.iloc[outlier_indexes])
        return outliers
   
    def get_linear_regression_outlier_indexes(
            self,
            input_points,
            rolling_window_size=DEFAULT_ROLLING_WINDOW_SIZE_1ST_TYPE_ANOMALY,
            n_sigma=3.0,
     ):
        """Computes outliers fitting deviation from the rolling average
        using linear regression

        :param input_points: input data frame
        :type input_points:  DataFrame
        :param n_sigma: n-sigma outlier detection parameter
        :type rolling_window_size: float. Default value: 3.00
        :return: numpy array of outlier points
        :rtype: a pair of numpy arrays (outliers, prediction_difference)
        """
        predicted_data_points = pd.DataFrame()
        cpu_usage_mean = input_points.cpu_usage.rolling(
            window=rolling_window_size).mean()
        cpu_usage_mean = cpu_usage_mean.fillna(input_points['cpu_usage'].mean())
        np_chunk = np.array(input_points['cpu_usage'])
        np_time = np.array(input_points['time'])
        chunk_size = len(np_chunk)
        chunk_size_half = int(chunk_size/2)
        np_cpu_usage_mean = np.array(cpu_usage_mean)
        mean_centered_data_points = np_chunk - np_cpu_usage_mean
        model = lm.LinearRegression()
        #fitted on half data in chunk 
        # Fit the model dividing chunk in half, fit one half, predict the other and switch
        for ar1, ar2 in zip((np_time, np.flipud(np_time)), (mean_centered_data_points, np.flipud(mean_centered_data_points))):
            model.fit(ar1[chunk_size_half:].reshape(-1,1), ar2[chunk_size_half:])
            predicted = model.predict(ar1[:chunk_size_half].reshape(-1,1))
            predicted_data_points=np.append(predicted_data_points, predicted)
        # Compute error
        if chunk_size%2 == 0:
           error = np.sqrt(mean_squared_error(mean_centered_data_points, predicted_data_points))
           #print(error)
        # Find difference between predicted and input data point
           prediction_difference = abs(predicted_data_points - mean_centered_data_points)
          # print(prediction_difference )
           outlier_indexes = np.where(prediction_difference > error*3 )
        else:
           error = np.sqrt(mean_squared_error(mean_centered_data_points[:-1], predicted_data_points))
        # Find difference between predicted and input data point
           prediction_difference = abs(predicted_data_points - mean_centered_data_points[:-1]) 
           # Compute outlier_indexes
           outlier_indexes = np.where(prediction_difference > error*3 )
        return outlier_indexes, prediction_difference
    
    def get_linear_regression_outliers_from_stream(
            self,
            input_stream=sys.stdin,
            min_rolling_window_size=DEFAULT_ROLLING_WINDOW_SIZE_1ST_TYPE_ANOMALY,
            n_sigma=3.0,
            min_chunksize=300,
            tuneup=True,
            debug=False,
    ):
        """Computes outliers using rolling n-sigma deviation from the rolling average

        :param input_stream: Input data stream
        :type input_stream: point sequence
        :param n_sigma: n-sigma outlier detection parameter
        :type min_rolling_window_size: float. Default value: 3.00
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

            # Set outlier indexes
            outlier_indexes, prediction_difference = self.get_linear_regression_outlier_indexes(
                     input_points=chunk,
                     rolling_window_size=current_rolling_window_size,
                     n_sigma=n_sigma
            )
            for timestamp, cpu_usage in self.rolling_points.iloc[outlier_indexes].reset_index().values.tolist():
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
        if len(line.split(',')) < 2: continue
        timestamp, cpu_usage = line.split(',')
        timestamp = int(timestamp)
        cpu_usage = float(cpu_usage)
        yield timestamp, cpu_usage


def file_test():
    """File processing test"""
    anomaly_detector = AnomalyDetector()
    print('1st type outliers from linear regression model')
    print (anomaly_detector.get_linear_regression_outliers_from_file())
    print('2nd type outliers, high or low cpu over chosen interval of time ' + str(DEFAULT_ROLLING_WINDOW_SIZE_2ND_TYPE_ANOMALY)+' seconds')
    print(anomaly_detector.get_average_outliers())


def stream_test():
    """Stream processing test"""
    anomaly_detector = AnomalyDetector()

    for outlier in anomaly_detector.get_linear_regression_outliers_from_stream(data_stream(open(TEST_FILE))):
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
                        Outlier = (1476400708.0, 0.182416680924528p84)
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
    
