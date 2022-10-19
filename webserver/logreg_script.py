#!/usr/bin/env python
import sys
import os
import shutil
import datetime
import pickle

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import pandas as pd
import numpy as np

import inspect
curr_file = inspect.getfile(inspect.currentframe()) # script filename (usually with path)

import logging
import logging.config
logging.filename    = curr_file+'.log'
logging.config.fileConfig("../config/logging.conf")
log = logging.getLogger("root")

import pymongo

import logreg
import logreg.global_vars as glbs

class TimeCheck(object):
    def __init__(self,label=None,log=None):
        self.label  = label
        self.log    = log
        self.t0     = datetime.datetime.now()
    def check(self):
        self.t1 = datetime.datetime.now()
        dt      = self.t1 - self.t0

        txt = '{sec}'.format(sec=str(dt))

        if self.label is not None:
            txt = ': '.join([self.label,txt])

        if self.log is not None:
            log.info(txt)
        else:
            print txt

log.info('Logistic Regression Trainor for <<{mstid_list}>>.'.format(mstid_list=glbs.mstid_list))

if glbs.test:
    log.info('Running in Test Mode.')

# Prepare output directories. ##################################################
logreg.prepare_output_dirs(glbs.output_dirs,clear_output_dirs=glbs.clear_output_dirs)

# Get events from mongo database. ##############################################
tc  = TimeCheck('Create Dataframe',log)
mongo               = pymongo.MongoClient()
db                  = mongo.mstid

events_from_db      = db[glbs.mstid_list].find()
data_dict           = logreg.generate_param_list(events_from_db,glbs.prm_dict)
df_raw              = pd.DataFrame.from_dict(data_dict,orient='index')

events_from_db.rewind()
metadata                = {}
metadata['mstid_list']  = glbs.mstid_list
metadata['radar']       = events_from_db[0]['radar']
metadata['sTime']       = df_raw.index.min()
metadata['eTime']       = df_raw.index.max()
metadata['slt_min']     = df_raw['slt'].min()
metadata['slt_max']     = df_raw['slt'].max()

tc.check()

# Plot histograms here. ########################################################
tc  = TimeCheck('Plot Histograms',log)
logreg.plot_histograms(df_raw,glbs.prm_dict,metadata,test=glbs.test,
        output_dir=glbs.output_dirs['search'])
tc.check()


# Choose features to base models off of. #######################################
tc  = TimeCheck('Compute Feature List',log)
feature_list_list = []
if not glbs.test:
    print 'Starting combinations...'
    import itertools
    #for L in range(len(glbs.prm_dict.keys())+1):
    for L in range(5):
        if L == 0:continue
        for subset in itertools.combinations(glbs.prm_dict.keys(),L):
            feature_list_list.append(subset)
else:
    feature_list_list.append(['orig_rti_cnt', 'orig_rti_mean', 'orig_rti_var'])
    feature_list_list.append(['orig_rti_cnt', 'total_spec', 'orig_rti_mean', 'orig_rti_var'])
    feature_list_list.append(['dom_spec','blob_track-short_list_mean_raw_mean','blob_track-short_list_var_raw_mean'])
    feature_list_list.append(['dom_spec','orig_rti_mean', 'orig_rti_var'])
    feature_list_list.append(['orig_rti_cnt', 'orig_rti_var'])
tc.check()


# Run logistic regression model. ###############################################
tc  = TimeCheck('Run Logistic Regression',log)
results_list = []
feat_nr     = 0
total_feat  = len(feature_list_list)
for feature_list in feature_list_list:
    feat_nr += 1
#    print '(%d of %d): ' % (feat_nr,total_feat),feature_list
    results_dict    = logreg.run_logit(df_raw,feature_list,glbs.dv)
    results_list.append(results_dict)
tc.check()

# Plot results. ################################################################
tc  = TimeCheck('Plot Logistic Regression Results',log)
logreg.plot_reg(results_list,metadata,sort_by=glbs.sort_by,
        output_dir=glbs.output_dirs['result'])
tc.check()
