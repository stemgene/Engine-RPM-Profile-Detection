#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 10:46:43 2020

@author: Yuan Lu, Zichu Li, Haoyuan Dong
"""

import pandas as pd
import math
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression
from datetime import datetime
import os
import sys
from boto.s3.connection import S3Connection
import boto3
import matplotlib.pylab as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import warnings
import argparse
import re
import json
from get_High_RPM import *
from summary_method import *
warnings.simplefilter('ignore')

def _get_config(path):
    config = json.load(open(path,'rb'))
    return config


def main(args):
    valid_data, client = get_valid_key(args['general']['bucket_name'])
    look_up = get_summary_df(args['general']['bucket_name'], args['general']['data_path'] + '/' + args['general']['summary_name'], client)
    if not os.path.exists(args['general']['run_id']):
        os.mkdir(args['general']['run_id'])
    if not os.path.exists(args['general']['path_to_save']):
        os.mkdir(args['general']['path_to_save'])
    if args['general']['mode'] == 'train':
        
#        pass
        slope, intercept = train(valid_data, args['general']['path_to_save'],client,look_up,args['general']['bucket_name'],args['regressing'], args['general']['folder_name'])
        
        test(valid_data, args['general']['path_to_save'],args['general']['run_id'], args['general']['bucket_name'],client, args['general']['folder_name'], mode = 'train')
        
    else:
        test(valid_data,args['general']['path_to_save'], args['general']['run_id'], args['general']['bucket_name'],client, args['general']['folder_name'])
    
    """
    Code for the other approach
    """
    trip_analysis(run_id = args['general']['run_id'],
          data_path = args['general']['data_path'], 
          data_file = args['general']['summary_name'],
          n_fit=args['summary']['n_fit'],
          bucket = args['general']['bucket_name'])
    
    sum_analysis(run_id = args['general']['run_id'],
         platforms_path = args['summary']['platforms_path'],
         filter_trial = args['summary']['filter_trial'],
         HighRPM_threshold = args['summary']['HighRPM_threshold'],
         HighRPM_torque_threshold = args['summary']['HighRPM_torque_threshold'],
         HighRPM_range_high = args['summary']['HighRPM_range_high'],
         HighRPM_range_low  = args['summary']['HighRPM_range_low'],
         MaxRPM_range_high = args['summary']['MaxRPM_range_high'],
         MaxRPM_range_low  = args['summary']['MaxRPM_range_low'])
    
    combine_sum_regress(run_id = args['general']['run_id'],
             regress_result_path = args['general']['regress_result_path'],
                 sum_result_path  = args['general']['sum_result_path'])

        
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./test.json', help='path for config file')
    args = parser.parse_args()
    config = _get_config(args.config)
    
    main(config)