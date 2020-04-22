#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 10:45:46 2020

@author: Yuan Lu
"""
import pandas as pd
import math
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
from datetime import datetime
import os
from boto.s3.connection import S3Connection
import boto3

def trip_analysis(run_id = 'log_1',#folder name to store all logs
                  data_path = 'university_of_rochester_projects/data/raw/rpm_profile', #path to the dataset
                  data_file = 'trip_data_for_rpm_profile_detection.csv',
                  n_fit=4,
                  bucket = 'vnomics-external-data-analytics'):
    """
    Arguments
    df: trip_data_for_rpm_profile_detection.csv
      trips: df.(df.platformId == platform) & (df.totalDistanceMiles > 10)
        mission_list: trips["missionReportId"]
    raw: {platform}_raw_port_data_for_all_trips.csv
      trips_data: raw['time','actual_torque_percent-513', 'engine_rpm-190'].dropna
        trip: all data of one mission
    """

    '''
    Create a unique ID for this hyperparameter run.
       It is a folder that all relevent files are saved to.
    '''
    conn = S3Connection()
    client = boto3.client('s3')
    

    # Pull in Trip Data Set

    #data = '/gdrive/My Drive/UR/DSC483/data/raw/rpm_profile/trip_data_for_rpm_profile_detection.csv'
    #data = 'rpm_profile/trip_data_for_rpm_profile_detection.csv'
    
    data = f'{data_path}/{data_file}'
    obj = client.get_object(Bucket=bucket, 
                        Key=data)
    df = pd.read_csv(obj['Body'],index_col = 0)

    df = df.join(pd.DataFrame([[np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]], 
                              index=df.index, 
                              columns=["HighRPM_found",
                                       "HighRPM_torque",
                                       "HighRPM_fit",
                                       "HighRPM_combo",
                                       "MaxRPM_found",
                                       "MaxRPM_torque",
                                       "n_record"])) # n_record is the number of trips within each trip (missionReportId) 

    # convert time from string to datetime
    for i in range(len(df)):
            df.startTimeGMT[i] = datetime.strptime(df.startTimeGMT[i], '%Y-%m-%d %H:%M:%S.%f')
            df.endTimeGMT[i] = datetime.strptime(df.endTimeGMT[i], '%Y-%m-%d %H:%M:%S.%f')


    # find the platform without feature of 'actual_torque_percent-513'
    #get distinct platformId
    platforms = list(df.platformId)
    platforms = list(dict.fromkeys(platforms))

    # find platform missing the feature of actual_torque_percent-513
    missing =[]
    for platform in platforms:
        rawdata = f'{data_path}/{platform}_raw_port_data_for_all_trips.csv'
        obj = client.get_object(Bucket=bucket, 
                        Key=rawdata)
        raw = pd.read_csv(obj['Body'],index_col = 0)
        if 'actual_torque_percent-513' not in raw.columns:
            missing.append(platform)
            print('[summary approach]' + f'{platform} missing feature actual_torque_percent-513!')
            print('[summary approach]' + f'{platform} missing feature actual_torque_percent-513!',
                  file=open(run_id + "/platform_miss_feature.txt", "a"))
            
    # hyperparameter
    # use the largest n_fitting actual torque and the associated rpm for fitting to find RPM at 100% torque
    #n_fitting = 3    #the largest 3 actual_torque_percent-513 and their rpm will be used for fitting
    n_fitting = n_fit

    # record all hyperparameters that might be useful to reference later
    with open(run_id + '/hyperparams.csv', 'w') as wfil:
        wfil.write("n_fitting," + str(n_fitting) + '\n')


    platforms_cleaned = list(filter(lambda x : x not in missing, platforms)) #len=48
    
    #save platforms_cleaned
    with open(run_id + '/platforms_cleaned.txt', 'w') as list_file:
        for platform in platforms_cleaned:
            list_file.write('%s\n' % platform)

    # record the time of the algorithm
    algorithm_start = datetime.now()

    for platform in platforms_cleaned:
        print('[summary approach]'+f"Analyzing platform {platform}......")
        platform_start_time = datetime.now()

        # get the trips for a platform
        # each trip has a unique missionReportId
        trips = df[(df.platformId == platform)].copy()

        # get the missionReportId for each platform
        mission_list = trips.sort_values(by='engineSpeedControlScore', ascending=True).loc[:,'missionReportId'].tolist()

        # Pull in raw port data
        rawdata = f'{data_path}/{platform}_raw_port_data_for_all_trips.csv'
        obj = client.get_object(Bucket=bucket, 
                        Key=rawdata)
        raw = pd.read_csv(obj['Body'],index_col = 0)

#        raw = pd.read_csv(rawdata)
#        del raw['Unnamed: 0']

        # Get data for missions
        # only the columns of 'time','actual_torque_percent-513', 'engine_rpm-190' are needed
        cols_keep = ['time','actual_torque_percent-513', 'engine_rpm-190']

        # rows containing any missing values will be discarded
        trips_df = raw[cols_keep].copy().dropna()

        # rows containing "actual_torque_percent-513 == 0" is considered an error and discarded
        trips_data = trips_df[(trips_df['actual_torque_percent-513'] > 0)].copy()

        # convert time from string to datetime
        for index in trips_data.index:
            trips_data.time[index] = datetime.strptime(trips_data.time[index], '%Y-%m-%d %H:%M:%S.%f')

        # remove the temporary dataframe
        del trips_df
        del raw

        # compute for each trip
        for mission in mission_list:
            print('[summary approach]'+f'- Analyzing missionReportId {mission}......')

            # find each trip's start and end time
            start = trips[trips.missionReportId == mission].startTimeGMT.iloc[0]
            end = trips[trips.missionReportId == mission].endTimeGMT.iloc[0]

            # use the trip's start and end time to group rows for each trip
            trip = trips_data[(trips_data.time >= start) & (trips_data.time <= end)]

            # initialize for each trip before analysis
            MaxRPM = None
            MaxRPM_index = None
            MaxRPM_torque = None

            HighRPM = None
            HighRPM_index = None
            HighRPM_torque = None

            sorted_fit = None
            h_index = None
            df_fit = None
            HighRPM_fit = None
            HighRPM_combo = None

            # record the number of valid rows in each trip (i.e. in each missionReportId)
            # a valid row is the row does not contain any null value
            df.loc[df.missionReportId==mission, ['n_record']] = len(trip)

            # trips with no valid record will be excluded from analysis
            if (len(trip) > 0):
                # Find HighRPM and MaxRPM

                # sort the records in each trip by descending engine_rpm-190
                sorted_trip = trip.sort_values(by="engine_rpm-190", ascending=False)

                # find the MaxRPM (the highest RPM seen in a trip and the associated actual_torque_percent-513>0)
                MaxRPM_index = sorted_trip.index[0]
                MaxRPM = sorted_trip.loc[MaxRPM_index]['engine_rpm-190']
                MaxRPM_torque = sorted_trip.loc[MaxRPM_index]['actual_torque_percent-513']

                df.loc[df.missionReportId==mission,['MaxRPM_found']] = MaxRPM
                df.loc[df.missionReportId==mission,['MaxRPM_torque']] = MaxRPM_torque

                # find the HighRPM (the highest RPM with largest actual torque)
                HighRPM_index = sorted_trip.loc[:,'actual_torque_percent-513'].idxmax(axis = 0)
                HighRPM = sorted_trip.loc[HighRPM_index]['engine_rpm-190']
                HighRPM_torque = sorted_trip.loc[HighRPM_index]['actual_torque_percent-513']

                df.loc[df.missionReportId==mission, ['HighRPM_found']] = HighRPM
                df.loc[df.missionReportId==mission, ['HighRPM_torque']] = HighRPM_torque

                #find the HighRPM by fitting the largest few data points
                # make a copy of the sorted trip for fitting
                sorted_fit = sorted_trip.copy()

                # reset the index of sorted trip
                sorted_fit.reset_index(inplace=True, drop=True)

                # find the index of the maximum actual_torque_percent-513 with the highest RPM
                h_index = sorted_fit.loc[:,'actual_torque_percent-513'].idxmax(axis = 0)

                # only keep the data points with lower rpm
                sorted_fit = sorted_fit.iloc[h_index:, ]

                if len(sorted_fit) >= n_fitting:
                    # find the largest n_fitting data points
                    df_fit = sorted_fit.nlargest(n_fitting,
                                                 'actual_torque_percent-513',
                                                 keep='first')

                    x = np.array(df_fit['actual_torque_percent-513'])
                    x = x.reshape(-1, 1)
                    y = np.array(df_fit['engine_rpm-190'])
                    lm = LinearRegression().fit(x, y)
                    HighRPM_fit = np.round_(lm.predict(np.array([[100]])))[0]

                    df.loc[df.missionReportId==mission, ['HighRPM_fit']] = HighRPM_fit

                # HighRPM_combo is the larger value of found HighRPM and HighRPM_fit
                if HighRPM is None:
                    HighRPM_combo = HighRPM_fit
                elif HighRPM_fit is None:
                    HighRPM_combo = HighRPM
                else:
                    HighRPM_combo = max(HighRPM, HighRPM_fit)

                df.loc[df.missionReportId==mission, ['HighRPM_combo']] = HighRPM_combo


        platform_end_time = datetime.now()
        with open(run_id + '/hyperparams.csv', 'a') as wfil:
            wfil.write(f'Time taken for platform {platform},' + str(platform_end_time - platform_start_time) + '\n')


    # save processed dataset
    df.to_csv(run_id + f'/processed_dataset_{run_id}.csv',
              header = True,
              index = False)

    # record algorithm time
    algorithm_end = datetime.now()
    algorithm_time = algorithm_end - algorithm_start

    with open(run_id + '/hyperparams.csv', 'a') as wfil:
        wfil.write('Algorithm started at,' + str(algorithm_start) + '\n')
        wfil.write('Algorithm ended at,' + str(algorithm_end) + '\n')
        wfil.write('Total time taken for the algorithm,' + str(algorithm_time) + '\n')

def sum_analysis(run_id = 'log_1',#folder name to store all logs
                 platforms_path = 'platforms_cleaned.txt',
                 filter_trial = 1,
                 HighRPM_threshold = 0.6,
                 HighRPM_torque_threshold = 80,
                 HighRPM_range_high = 1800,
                 HighRPM_range_low  = 1300,
                 MaxRPM_range_high = 2400,
                 MaxRPM_range_low  = 1600):
    
    # read processed datasets
    df = pd.read_csv(run_id + f'/processed_dataset_{run_id}.csv')
    
    
    #read the platforms_cleaned
    platforms_cleaned = []
    
    with open(f'{run_id}/{platforms_path}', 'r') as list_file:
        for line in list_file:
            #remove linebreak which is the last character of the string
            current_platform = line[:-1]
            
            #add item to the list
            platforms_cleaned.append(int(current_platform))
    
    #filter_trial = 1

    #HighRPM_threshold = 0.6 # the rpm corresponding to the highest actual torque has to be higher than HighRPM_threshold*observed_MaxRPM
    #HighRPM_torque_threshold = 80 # records with actual_torque higher than this threshold will be analyzed

    #HighRPM_range_high = 1800 # only HighRPM within this range will be considered
    #HighRPM_range_low  = 1300  # only HighRPM within this range will be considered

    #MaxRPM_range_high = 2400 # only HighRPM within this range will be considered
    #MaxRPM_range_low  = 1600  # only HighRPM within this range will be considered

    # record all hyperparameters that might be useful to reference later
    with open(run_id + '/hyperparams.csv', 'a') as wfil:
        wfil.write("Filtering trial," + str(filter_trial) + '\n')
        wfil.write("HighRPM_threshold," + str(HighRPM_threshold) + '\n')
        wfil.write("HighRPM_torque_threshold," + str(HighRPM_torque_threshold) + '\n')
        wfil.write("HighRPM_range_high," + str(HighRPM_range_high) + '\n')
        wfil.write("HighRPM_range_low," + str(HighRPM_range_low) + '\n')
        wfil.write("MaxRPM_range_high," + str(MaxRPM_range_high) + '\n')
        wfil.write("MaxRPM_range_low," + str(MaxRPM_range_low) + '\n')


    # create summary dataframe
    df_platform_sum = pd.DataFrame(columns = ['platformId',
                                              'HighRPM',
                                              'MaxRPM',
                                              'HighRPM_found_mean',
                                              #'HighRPM_found_mean_acc(RPM)',
                                              'HighRPM_found_max',
                                              #'HighRPM_found_max_acc(RPM)',
                                              'HighRPM_fit_mean',
                                              #'HighRPM_fit_mean_acc(RPM)',
                                              'HighRPM_fit_max',
                                              #'HighRPM_fit_max_acc(RPM)',
                                              'HighRPM_combo_mean',
                                              #'HighRPM_combo_mean_acc(RPM)',
                                              'HighRPM_combo_max',
                                              #'HighRPM_combo_max_acc(RPM)',
                                              'MaxRPM_found_mean',
                                              #'MaxRPM_found_mean_acc(RPM)',
                                              'MaxRPM_found_max',
                                              #'MaxRPM_found_max_acc(RPM)',
                                              'MaxRPM_found_max_sem'
                                              #'MaxRPM_found_max_sem_acc(RPM)'
                                             ])

    for platform in platforms_cleaned:
        print('[summary approach]'+f"Summarizing platform {platform}......")

        #get the ground truth for HighRPM and MaxRPM
        #ref_HighRPM = df.loc[(df.platformId == platform)]['HighRPM'].mean()
        #ref_MaxRPM = df.loc[(df.platformId == platform)]['MaxRPM'].mean()


        #############################################################################################
        #get the HighRPM for each platform by averaging all its trips' HighRPM_found
        HighRPM_mean = df.loc[(df.platformId == platform) & 
                              (df.HighRPM_found >= df.MaxRPM_found * HighRPM_threshold) & 
                              (df.HighRPM_torque >= HighRPM_torque_threshold) &
                              (df.HighRPM_found >= HighRPM_range_low) &
                              (df.HighRPM_found <= HighRPM_range_high)]['HighRPM_found'].mean()

        #HighRPM_mean_acc = HighRPM_mean - ref_HighRPM

        #get the HighRPM for each platform by finding the max of all its trips' HighRPM_found
        HighRPM_max = df.loc[(df.platformId == platform)& 
                             (df.HighRPM_found >= df.MaxRPM_found * HighRPM_threshold) & 
                             (df.HighRPM_torque >= HighRPM_torque_threshold) &
                             (df.HighRPM_found >= HighRPM_range_low) &
                             (df.HighRPM_found <= HighRPM_range_high)]['HighRPM_found'].max()

        #HighRPM_max_acc = HighRPM_max - ref_HighRPM
        #############################################################################################



        #############################################################################################
        #get the HighRPM for each platform by averaging all its trips' HighRPM_fit
        HighRPM_fit_mean = df.loc[(df.platformId == platform)& 
                                  (df.HighRPM_fit >= df.MaxRPM_found * HighRPM_threshold) &
                                  (df.HighRPM_fit >= HighRPM_range_low) &
                                  (df.HighRPM_fit <= HighRPM_range_high)]['HighRPM_fit'].mean()

        #HighRPM_fit_mean_acc = HighRPM_fit_mean - ref_HighRPM

        #get the HighRPM for each platform by finding the max of all its trips' HighRPM_fit
        HighRPM_fit_max = df.loc[(df.platformId == platform)& 
                                  (df.HighRPM_fit >= df.MaxRPM_found * HighRPM_threshold) &
                                  (df.HighRPM_fit >= HighRPM_range_low) &
                                  (df.HighRPM_fit <= HighRPM_range_high)]['HighRPM_fit'].max()

        #HighRPM_fit_max_acc = HighRPM_fit_max - ref_HighRPM
        #############################################################################################


        #############################################################################################
        df_HighRPM_combo = df.loc[(df.platformId == platform)].copy()
        HighRPM_combo_list = []

        for index in df_HighRPM_combo.index:
            H_combo  = df_HighRPM_combo.loc[index]['HighRPM_combo']
            H_found  = df_HighRPM_combo.loc[index]['HighRPM_found']
            H_fit    = df_HighRPM_combo.loc[index]['HighRPM_fit']
            H_torque = df_HighRPM_combo.loc[index]['HighRPM_torque']

            # HighRPM_combo is the larger value of found HighRPM and HighRPM_fit
            if (H_combo == H_found) and (H_torque is not None):
                if (H_torque >= HighRPM_torque_threshold) and (H_combo >= HighRPM_range_low) and (H_combo <= HighRPM_range_high):
                    HighRPM_combo_list.append(H_combo)
            elif H_combo is not None:
                if (H_combo >= HighRPM_range_low) and (H_combo <= HighRPM_range_high):
                    HighRPM_combo_list.append(H_combo)


        if len(HighRPM_combo_list) == 0:
            HighRPM_combo_mean = None
            #HighRPM_combo_mean_acc = None
            HighRPM_combo_max = None
            #HighRPM_combo_max_acc = None
        else:
            #get the HighRPM_combo for each platform by averaging all its trips' HighRPM_combo
            HighRPM_combo_mean = sum(HighRPM_combo_list) / len(HighRPM_combo_list)
            #HighRPM_combo_mean_acc = HighRPM_combo_mean - ref_HighRPM

            #get the HighRPM_combo for each platform by finding the max of all its trips' HighRPM_combo
            HighRPM_combo_max = max(HighRPM_combo_list)
            #HighRPM_combo_max_acc = HighRPM_combo_max - ref_HighRPM

        del df_HighRPM_combo
        #############################################################################################

        #############################################################################################
        #get the MaxRPM for each platform by averaging all its trips' MaxRPM_found
        MaxRPM_mean = df.loc[(df.platformId == platform) &
                             (df.MaxRPM_found >= MaxRPM_range_low) &
                             (df.MaxRPM_found <= MaxRPM_range_high)]['MaxRPM_found'].mean()

        #MaxRPM_mean_acc = MaxRPM_mean - ref_MaxRPM

        #get the MaxRPM for each platform by finding the max of all its trips' MaxRPM_found
        MaxRPM_max = df.loc[(df.platformId == platform)&
                            (df.MaxRPM_found >= MaxRPM_range_low) &
                            (df.MaxRPM_found <= MaxRPM_range_high)]['MaxRPM_found'].max()

        #MaxRPM_max_acc = MaxRPM_max - ref_MaxRPM
        #############################################################################################


        #############################################################################################
        # add an additional sem to MaxRPM_max as the prediction
        MaxRPM_sem = df.loc[(df.platformId == platform) &
                            (df.MaxRPM_found >= MaxRPM_range_low) &
                            (df.MaxRPM_found <= MaxRPM_range_high)]['MaxRPM_found'].sem()

        MaxRPM_max_sem = MaxRPM_max + MaxRPM_sem
        #MaxRPM_max_sem_acc = MaxRPM_max_sem - ref_MaxRPM
        #############################################################################################


        df_platform_sum = df_platform_sum.append({'platformId': platform,
                                                  #'HighRPM': ref_HighRPM,
                                                  #'MaxRPM': ref_MaxRPM,
                                                  'HighRPM_found_mean': HighRPM_mean,
                                                  #'HighRPM_found_mean_acc(RPM)': HighRPM_mean_acc,
                                                  'HighRPM_found_max': HighRPM_max,
                                                  #'HighRPM_found_max_acc(RPM)': HighRPM_max_acc,
                                                  'HighRPM_fit_mean': HighRPM_fit_mean,
                                                  #'HighRPM_fit_mean_acc(RPM)': HighRPM_fit_mean_acc,
                                                  'HighRPM_fit_max': HighRPM_fit_max,
                                                  #'HighRPM_fit_max_acc(RPM)': HighRPM_fit_max_acc,
                                                  'HighRPM_combo_mean': HighRPM_combo_mean,
                                                  #'HighRPM_combo_mean_acc(RPM)': HighRPM_combo_mean_acc,
                                                  'HighRPM_combo_max': HighRPM_combo_max,
                                                  #'HighRPM_combo_max_acc(RPM)': HighRPM_combo_max_acc,
                                                  'MaxRPM_found_mean': MaxRPM_mean,
                                                  #'MaxRPM_found_mean_acc(RPM)': MaxRPM_mean_acc,
                                                  'MaxRPM_found_max': MaxRPM_max,
                                                  #'MaxRPM_found_max_acc(RPM)': MaxRPM_max_acc,
                                                  'MaxRPM_found_max_sem': MaxRPM_max_sem
                                                  #'MaxRPM_found_max_sem_acc(RPM)': MaxRPM_max_sem_acc
                                                 },
                                                 ignore_index=True)

    df_platform_sum.to_csv(run_id + f'/summarized_result_{run_id}_{filter_trial}.csv',
                           header = True,
                           index = False)
    
def combine_sum_pool(run_id = 'log_1',
                     pool_result_path = 'pooling_result.csv',
                     sum_result_path  = 'summarized_result.csv'):
    
    pooling_test = pd.read_csv(run_id +'/' + pool_result_path)

    sum_test = pd.read_csv(run_id +'/'+ sum_result_path)

    sum_test = sum_test.join(pd.DataFrame([[np.nan, np.nan]],
                                          index=sum_test.index, 
                                          columns=["HighRPM_pool",
                                                   "HighRPM_cp_mean"]))

    for i, platform in enumerate(sum_test.platformId):
        sum_test.HighRPM_pool[i] = pooling_test[pooling_test.platform_id == platform].high_RPM.iloc[0]

        # if both combo_meand and pool exist, use the mean for prediction
        # if only one of combo_meand and pool exists, use the existing one for prediction
        if (not math.isnan(sum_test.HighRPM_pool[i])) and (not math.isnan(sum_test.HighRPM_combo_mean[i])):
            # use the mean of pool and combo for the HighRPM_cp_mean
            sum_test['HighRPM_cp_mean'][i] = (sum_test.HighRPM_combo_mean[i] + sum_test.HighRPM_pool[i])/2

        elif (math.isnan(sum_test.HighRPM_pool[i])) and (not math.isnan(sum_test.HighRPM_combo_mean[i])):
            sum_test['HighRPM_cp_mean'][i] = sum_test.HighRPM_combo_mean[i]

        elif (not math.isnan(sum_test.HighRPM_pool[i])) and (math.isnan(sum_test.HighRPM_combo_mean[i])):
            sum_test['HighRPM_cp_mean'][i] = sum_test.HighRPM_pool[i]

    sum_test.to_csv(run_id + 'sum_prediction.csv',
                    header = True,
                    index = False)