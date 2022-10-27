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

from multiprocessing import Pool

logit_dict  = {}
metadata    = {}

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

def mylogit(feature_list):
        return logreg.run_logit(logit_dict['df_raw'],feature_list,logit_dict['dv'])

def my_plot_reg(inp):
    logreg.plot_reg_single(inp[0],metadata,output_dir=glbs.output_dirs['result'],ser_nr=inp[1])

def reduce_result(result):
    return {'rank':result[0],'feature_list':result[1]['feature_list'],'mean_pct_correct':result[1]['mean_pct_correct']}

def my_plot_ranking(inp):
    logreg.plot_ranking_single(inp[0],metadata,output_dir=glbs.output_dirs['rank'],ser_nr=inp[1])

if __name__ == '__main__':

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
    data_dict           = logreg.generate_param_list(events_from_db,glbs.prm_dict,kmin=glbs.kmin)
    df_raw              = pd.DataFrame.from_dict(data_dict,orient='index')

    events_from_db.rewind()
    metadata['mstid_list']  = glbs.mstid_list
    metadata['radar']       = events_from_db[0]['radar']
    metadata['sTime']       = df_raw.index.min()
    metadata['eTime']       = df_raw.index.max()
    metadata['slt_min']     = df_raw['slt'].min()
    metadata['slt_max']     = df_raw['slt'].max()
    metadata['n_trials']    = glbs.n_trials

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
        for L in range(4):
            if L == 0:continue
            for subset in itertools.combinations(glbs.prm_dict.keys(),L):
                feature_list_list.append(tuple(subset))
    else:
        feature_list_list.append(('orig_rti_cnt', 'orig_rti_mean', 'orig_rti_var'))
        feature_list_list.append(('orig_rti_cnt', 'total_spec', 'orig_rti_mean', 'orig_rti_var'))
        feature_list_list.append(('dom_spec','blob_track-short_list_mean_raw_mean','blob_track-short_list_var_raw_mean'))
        feature_list_list.append(('dom_spec','orig_rti_mean', 'orig_rti_var'))
        feature_list_list.append(('orig_rti_cnt', 'orig_rti_var'))
    tc.check()

    #Keep track of total number of features
    combos  = len(feature_list_list)
    metadata['combos'] = combos #Be able to send that info to plotting routines.
    rank        = range(1,combos+1)

    reduced_results = []
    tc_trl  = TimeCheck('Total Trial Time',log)
    for trl in range(glbs.n_trials):
        try:
            # Run logistic regression model. ###############################################
            tc  = TimeCheck('({trl} of {n_trials}) Log Regression and Ranking ({combos} Combinations)'.format(trl=trl,n_trials=glbs.n_trials,combos=combos),log)

            results_list = []
            
            logit_dict['dv']        = glbs.dv
            logit_dict['df_raw']    = df_raw

            pool            = Pool()
            results_list    = pool.map(mylogit,feature_list_list)

            sorted_list = sorted(results_list, key=lambda k: k[glbs.sort_by])
            if glbs.sort_by == 'r_sqrd' or glbs.sort_by == 'mean_pct_correct':
                sorted_list = sorted_list[::-1]

            res = pool.map(reduce_result,zip(rank,sorted_list))
            reduced_results.append(res)
            pool.close()
            pool.join()
            tc.check()
        except:
            continue

    tc_trl.check()

    tc  = TimeCheck('Binning, Sorting, and Ranking all trials...',log)
    # Bin Categories ############################################################### 
    ranking_dict = {}
    for feature_list in feature_list_list:
        ranking_dict[feature_list] = {}
        ranking_dict[feature_list]['feature_list']  = feature_list
        ranking_dict[feature_list]['hist']          = np.zeros(combos+1,dtype=np.int)
        ranking_dict[feature_list][glbs.sort_by]    = []

    # Sort reduced results into ranking dictionaries.
    for run_results in reduced_results:
        for res in run_results:
            ranking_dict[res['feature_list']]['hist'][res['rank']] += 1
            ranking_dict[res['feature_list']][glbs.sort_by].append(res[glbs.sort_by])

    mean_mean_key = 'mean_'+glbs.sort_by
    std_mean_key  = 'std_'+glbs.sort_by
    med_mean_key  = 'med_'+glbs.sort_by
    for key,val in ranking_dict.iteritems():
        val[mean_mean_key] = np.mean(val[glbs.sort_by])
        val[std_mean_key]  = np.std(val[glbs.sort_by])
        val[med_mean_key]  = np.median(val[glbs.sort_by])

    ranking_list = [val for val in ranking_dict.itervalues()]
    sorted_ranking_list = sorted(ranking_list, key=lambda k: k[mean_mean_key])
    if glbs.sort_by == 'r_sqrd' or glbs.sort_by == 'mean_pct_correct':
        sorted_ranking_list = sorted_ranking_list[::-1]

    rank_file   = os.path.join(glbs.output_dirs['rank'],'sorted_rank.p')
    with open(rank_file,'wb') as fl:
        pickle.dump(sorted_ranking_list,fl)

#    for item in sorted_ranking_list:
#        print item[mean_mean_key]
    
    max_plots = 50
    inp     = zip(sorted_ranking_list[:max_plots],range(1,max_plots+1))

    pool    = Pool()
    pool.map(my_plot_ranking,inp)
    pool.close()
    pool.join()

#    ser_nr  = 0
#    for item in sorted_ranking_list:
#        ser_nr += 1
#        logreg.plot_ranking_single(item,metadata,output_dir=glbs.output_dirs['rank'],ser_nr=ser_nr)
#        print item[mean_mean_key],item['feature_list']
    
    inp     = zip(sorted_list[:max_plots],range(1,max_plots+1))
    tc.check()


    # Plot results. ################################################################
    tc  = TimeCheck('Plot Logistic Regression Results',log)

    pool    = Pool()
    pool.map(my_plot_reg,inp)
    pool.close()
    pool.join()

    tc.check()
