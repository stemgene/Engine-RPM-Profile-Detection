#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This python script is used to predict high RPM
setting the mode euqal to train or test first.

@author: Haoyuan Dong
"""

import pandas as pd
import numpy as np
import os
from boto.s3.connection import S3Connection
import boto3
import matplotlib.pylab as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import warnings
warnings.simplefilter('ignore')
import argparse
import pandas as pd
import re

def get_valid_key(bucket_name, client_only = False):
    """
    get the valid key within given bucket
    """
    conn = S3Connection()
    bucket = conn.get_bucket('vnomics-external-data-analytics')
    client = boto3.client('s3')
    if client_only: 
        return client
    resource = boto3.resource('s3')
    bucket = resource.Bucket('vnomics-external-data-analytics')
    key_set = []
    for item in bucket.objects.all():
        if '/'.join(item.key.split('/')[:4]) == 'university_of_rochester_projects/data/raw/rpm_profile':
            key_set.append(item)
    return key_set, client

    

def smooth_out(bin_size,df,quantile = [0.1,1]):
    q1 = np.quantile(df['engine_rpm-190'],q = quantile[0])
    q2 = np.quantile(df['engine_rpm-190'],q = quantile[1])
    num_bucket = (q1 + q2) // bin_size
    df = df[np.logical_and(df['engine_rpm-190']>=q1, df['engine_rpm-190'] <=q2)]
    df['bin'] = pd.cut(df['engine_rpm-190'],num_bucket)
    output = df.groupby(by = 'bin').agg({'actual_torque_percent-513':'max','engine_rpm-190':'mean'})
    x = np.array(output['engine_rpm-190'])
    y = np.array(output['actual_torque_percent-513'])
    return x, y

def cut_off(x,y,val):
    index = x>val
    x = x[index]
    y = y[index]
    return x,y

def get_top_q(torque, top_k = 0.05):
    return np.mean(torque[torque>=np.quantile(torque,1-top_k)])
                                        
def drop_col(x):
    if not np.isnan(x['actual_torque_percent-513']) and not np.isnan(x['engine_rpm-190']):
        return True
    return False

def decode(model,y,k):
    y_ = get_top_q(y,k)
    inter = model.intercept_
    slope = model.coef_[0]
    return (y_-inter)/slope

def decode_(intercept,slope,y,k):
    y_ = get_top_q(y,k)
    return (y_-intercept)/slope

def get_summary_df(bucket_name, path, client):
    
    obj = client.get_object(Bucket=bucket_name, 
                        Key=path)

    summary = pd.read_csv(obj['Body'],index_col = 0)
    look_up = {}
    for item in np.array(summary[['platformId','HighRPM']]):
        if item[0] not in look_up:
            look_up[item[0]] = item[1]
        elif look_up[item[0]] != item[1]:
            print('[pooling approach:]'+item[0], 'multiple ground truth')
    return look_up


def grid_search(df,config, key, look_up):
    q1 = config['q1']
    q2 = config['q2']
    elimination_step = config['elimination_step']
    bin_size = config['bin_size']
    metric_res = float('inf')
    res = {}

    x,y = smooth_out(bin_size,df,[q1,q2])
        
    ground_truth = look_up[key]
    for i in range(0,int(max(x)),elimination_step):


        x_,y_ = cut_off(x,y,i)

        clf = LinearRegression(fit_intercept=True)
        clf.fit(x_.reshape(-1,1),y_)

        metric = decode(clf,y,0.01)
        fit = mean_squared_error(clf.predict(x_.reshape(-1,1)),y_)
        if fit == 0:
            continue
        cur_metric_res = abs(metric-ground_truth) * fit 
        if clf.coef_[0] < 0 and cur_metric_res<metric_res:
            plt.plot(x_,clf.predict(x_.reshape(-1,1)),'r')
            print('[pooling approach:]'+'[training....] cur_metrics_res: {} best_metric_res: {} slope: {} decode: {} ' \
                  .format(cur_metric_res,metric_res,clf.coef_[0],metric))
#                             model = clf
            res['bin_size'] = bin_size
            res['elimination_step'] = elimination_step
            res['model'] = clf
            res['fit'] = fit
            metric_res = cur_metric_res
    return res

def train(valid_data,path_to_save, client, look_up, bucket_name,config, folder_name = 'round_2'):
    
    res = []
    for item in valid_data:
        if folder_name in item.key: continue
        try:
            pattern = re.compile('\d+')
            platform_id = pattern.findall(item.key.split('/')[-1])[0]
        except:
            print('[pooling approach:]'+'skip {}'.format(item.key))
            continue
        obj = client.get_object(Bucket=bucket_name, 
                        Key=item.key)
        df = pd.read_csv(obj['Body'],index_col = 0)
        if 'actual_torque_percent-513' not in df.columns or 'engine_rpm-190' not in df.columns: continue
        df = df[df.apply(drop_col,axis =1)][['actual_torque_percent-513','engine_rpm-190']]
        result = grid_search(df, config, int(platform_id), look_up)
        result['platform_id'] = platform_id
        res.append(result)
#        print(res[-1])
    mnodels = []
    for item in res:
        if 'model' not in item: 
            print('[pooling approach:]'+'[error] {} don\'t have model'.format(item['platform_id']))
            continue
        mnodels.append(item['model'])
    slope = []
    intercept = []
    for model in mnodels:
        slope.append(model.coef_[0])
        intercept.append(model.intercept_)
    with open(path_to_save +'/'+ 'slope.pickle','wb') as f:
        pickle.dump(slope,f)
        
    with open(path_to_save + '/'+'intercept.pickle','wb') as f:
        pickle.dump(intercept,f)
    return slope, intercept

def test(valid_data,path_to_load, run_id, bucket_name , client, folder_name = 'round_2', mode = 'test'):
    if not path_to_load:
        print('[pooling approach:]'+'please provide the pre-trained parameter')
    else:
        
    
        with open(path_to_load+'/' + 'intercept.pickle','rb') as f:
            intercept = pickle.load(f)
            
    
        with open(path_to_load + '/'+'slope.pickle','rb') as f:
            slope = pickle.load(f)
    
    platfor_id = []
    high_RPM = []
    for item in valid_data:
        try:
            pattern = re.compile('\d+')
            platform_id = pattern.findall(item.key.split('/')[-1])[0]
        except:
            print('[pooling approach:]'+'skip {}'.format(item.key))
            continue
        obj = None
        if mode == 'test':
            if folder_name in item.key:
                obj = client.get_object(Bucket=bucket_name, Key=item.key)
        else:
            if folder_name not in item.key:
                obj = client.get_object(Bucket=bucket_name, Key=item.key)
        if not obj: continue
        df = pd.read_csv(obj['Body'],index_col = 0)
        if 'actual_torque_percent-513' not in df.columns or 'engine_rpm-190' not in df.columns: continue
        if 'actual_torque_percent-513' not in df.columns or 'engine_rpm-190' not in df.columns: continue
        df = df[df.apply(drop_col,axis =1)][['actual_torque_percent-513','engine_rpm-190']]

        platfor_id.append(platform_id)
        score = decode_(np.quantile(intercept,0.5),np.quantile(slope,0.5),df['actual_torque_percent-513'],0)
        high_RPM.append(score)
        print('[pooling approach:]'+'[test...] platfor_id is {}; high_RPM is {}'.format(platform_id,score))
    df = pd.DataFrame({'platform_id':platfor_id,'high_RPM':high_RPM})
    df.to_csv(run_id + 'pooling_result.csv')
