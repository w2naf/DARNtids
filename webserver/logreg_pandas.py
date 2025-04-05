#!/usr/bin/env python
import sys
sys.path.append('/data/mstid/statistics/webserver')

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import os
import shutil
# import datetime
# import hdf5

import pandas as pd
import numpy as np

import inspect
import logging
import logging.config
curr_file = inspect.getfile(inspect.currentframe()) # script filename (usually with path)
logging.filename    = curr_file+'.log'
logging.config.fileConfig("../config/logging.conf")
log = logging.getLogger("root")

import stats_support as ssup

import pymongo
# Get events from mongo database. ##############################################
mongo         = pymongo.MongoClient()
db            = mongo.mstid

#mstid_list  = 'paper2_wal_2012'
mstid_list  = 'paper2_bks_2010_2011_0817mlt'

clear_output_dirs       = True
test                    = False
output_dirs             = {}
output_dirs['search']   = 'output/logreg/%s/search' % mstid_list
output_dirs['result']   = 'output/logreg/%s/result' % mstid_list

show_all_txt = '''
<?php
foreach (glob("*.png") as $filename) {
    echo "<img src='$filename' > ";
}
?>
'''

show_all_txt_breaks = '''
<?php
foreach (glob("*.png") as $filename) {
    echo "<img src='$filename'> <br /> ";
}
?>
'''

for value in output_dirs.values():
    if clear_output_dirs:
        try:
            shutil.rmtree(value)
        except:
            pass
    try:
        os.makedirs(value)
    except:
        pass
    with open(os.path.join(value,'0000-show_all.php'),'w') as file_obj:
        file_obj.write(show_all_txt)
    with open(os.path.join(value,'0000-show_all_breaks.php'),'w') as file_obj:
        file_obj.write(show_all_txt_breaks)

# Define category dictionary for plotting. #####################################
# You can change the number of things plotted by simply commenting out a cat_dict entry
# or by changing the number of mlt_bins.
cat_dict = {}
cat_dict['mstid']   = {'title':'MSTID', 'color':'r'}
cat_dict['quiet']   = {'title':'Quiet', 'color':'g'}
cat_dict['none']    = {'title':'None' , 'color':'b'}

# Build up possible independent variables. #####################################
prm_dict        = {}

param_code   = 'dom_spec'
prm_dict[param_code] = {}
prm_dict[param_code]['param_name']   = 'Dominant Spectrum'
prm_dict[param_code]['param_units']  = ''
prm_dict[param_code]['bins']         = 25
prm_dict[param_code]['bins_range']   = None

param_code   = 'total_spec'
prm_dict[param_code] = {}
prm_dict[param_code]['param_name']   = 'Total Spectrum'
prm_dict[param_code]['param_units']  = ''
prm_dict[param_code]['bins']         = 25
prm_dict[param_code]['bins_range']   = None

param_code   = 'spec_ratio'
prm_dict[param_code] = {}
prm_dict[param_code]['param_name']   = '(Dom FFT Bin)/(All FFT Bins)'
prm_dict[param_code]['param_units']  = ''
prm_dict[param_code]['bins']         = 25
prm_dict[param_code]['bins_range']   = (0.05,0.1)
prm_dict[param_code]['bins_range']   = None

param_code = 'orig_rti_cnt'
prm_dict[param_code] = {}
prm_dict[param_code]['param_name'] = 'Original RTI Scatter Count'
prm_dict[param_code]['param_units'] = ''
prm_dict[param_code]['bins'] = 25
prm_dict[param_code]['bins_range'] = None

param_code = 'orig_rti_mean'
prm_dict[param_code] = {}
prm_dict[param_code]['param_name'] = 'Original RTI Mean'
prm_dict[param_code]['param_units'] = '[dB]'
prm_dict[param_code]['bins'] = 25
prm_dict[param_code]['bins_range'] = None

param_code = 'orig_rti_median'
prm_dict[param_code] = {}
prm_dict[param_code]['param_name'] = 'Original RTI Median'
prm_dict[param_code]['param_units'] = '[dB]'
prm_dict[param_code]['bins'] = 25
prm_dict[param_code]['bins_range'] = None

param_code = 'orig_rti_var'
prm_dict[param_code] = {}
prm_dict[param_code]['param_name'] = 'Original RTI Variance'
prm_dict[param_code]['param_units'] = '[dB]'
prm_dict[param_code]['bins'] = 25
prm_dict[param_code]['bins_range'] = None

param_code = 'blob_track-short_list_mean_count'
prm_dict[param_code] = {}
prm_dict[param_code]['param_name'] = 'Tracked Blob Pixel Count Mean'
prm_dict[param_code]['param_units'] = 'px'
prm_dict[param_code]['bins'] = 25
prm_dict[param_code]['bins_range'] = None

param_code = 'blob_track-short_list_var_count'
prm_dict[param_code] = {}
prm_dict[param_code]['param_name'] = 'Tracked Blob Pixel Count Variance'
prm_dict[param_code]['param_units'] = 'px'
prm_dict[param_code]['bins'] = 25
prm_dict[param_code]['bins_range'] = None

param_code = 'blob_track-short_list_mean_raw_mean'
prm_dict[param_code] = {}
prm_dict[param_code]['param_name'] = 'Tracked Blob Mean Power Mean'
prm_dict[param_code]['param_units'] = 'dB'
prm_dict[param_code]['bins'] = 25
prm_dict[param_code]['bins_range'] = None

param_code = 'blob_track-short_list_var_raw_mean'
prm_dict[param_code] = {}
prm_dict[param_code]['param_name'] = 'Tracked Blob Mean Power Variance'
prm_dict[param_code]['param_units'] = 'dB'
prm_dict[param_code]['bins'] = 25
prm_dict[param_code]['bins_range'] = None

param_code = 'nr_sigs'
prm_dict[param_code] = {}
prm_dict[param_code]['param_name'] = 'Nr. of Signals'
prm_dict[param_code]['param_units'] = ''
prm_dict[param_code]['bins'] = 25
prm_dict[param_code]['bins_range'] = None

param_code = 'area'
prm_dict[param_code] = {}
prm_dict[param_code]['param_name'] = 'Area'
prm_dict[param_code]['param_units'] = 'px'
prm_dict[param_code]['bins'] = 25
prm_dict[param_code]['bins_range'] = None

param_code = 'true_max'
prm_dict[param_code] = {}
prm_dict[param_code]['param_name'] = 'MUSIC Maximum'
prm_dict[param_code]['param_units'] = ''
prm_dict[param_code]['bins'] = 25
prm_dict[param_code]['bins_range'] = (0.0015,0.0035)

param_code = 'k'
prm_dict[param_code] = {}
prm_dict[param_code]['param_name']  = 'Horiz. Wavenumber'
prm_dict[param_code]['param_units'] = ''
prm_dict[param_code]['bins']        = 50
#prm_dict[param_code]['bins_range'] = (0.0015,0.0035)

#param_code = 'blob_track-short_list_mean_box_x0'
#prm_dict[param_code] = {}
#prm_dict[param_code]['param_name'] = 'Tracked Blob x0 Mean'
#prm_dict[param_code]['param_units'] = 'px'
#prm_dict[param_code]['bins'] = 25
#prm_dict[param_code]['bins_range'] = None
#
#param_code = 'blob_track-short_list_var_box_x0'
#prm_dict[param_code] = {}
#prm_dict[param_code]['param_name'] = 'Tracked Blob x0 Variance'
#prm_dict[param_code]['param_units'] = 'px'
#prm_dict[param_code]['bins'] = 25
#prm_dict[param_code]['bins_range'] = None
#
#param_code = 'blob_track-short_list_mean_box_x1'
#prm_dict[param_code] = {}
#prm_dict[param_code]['param_name'] = 'Tracked Blob x1 Mean'
#prm_dict[param_code]['param_units'] = 'px'
#prm_dict[param_code]['bins'] = 25
#prm_dict[param_code]['bins_range'] = None
#
#param_code = 'blob_track-short_list_var_box_x1'
#prm_dict[param_code] = {}
#prm_dict[param_code]['param_name'] = 'Tracked Blob x1 Variance'
#prm_dict[param_code]['param_units'] = 'px'
#prm_dict[param_code]['bins'] = 25
#prm_dict[param_code]['bins_range'] = None
#
#param_code = 'blob_track-short_list_mean_box_y0'
#prm_dict[param_code] = {}
#prm_dict[param_code]['param_name'] = 'Tracked Blob y0 Mean'
#prm_dict[param_code]['param_units'] = 'px'
#prm_dict[param_code]['bins'] = 25
#prm_dict[param_code]['bins_range'] = None
#
#param_code = 'blob_track-short_list_var_box_y0'
#prm_dict[param_code] = {}
#prm_dict[param_code]['param_name'] = 'Tracked Blob y0 Variance'
#prm_dict[param_code]['param_units'] = 'px'
#prm_dict[param_code]['bins'] = 25
#prm_dict[param_code]['bins_range'] = None
#
#param_code = 'blob_track-short_list_mean_box_y1'
#prm_dict[param_code] = {}
#prm_dict[param_code]['param_name'] = 'Tracked Blob y1 Mean'
#prm_dict[param_code]['param_units'] = 'px'
#prm_dict[param_code]['bins'] = 25
#prm_dict[param_code]['bins_range'] = None
#
#param_code = 'blob_track-short_list_var_box_y1'
#prm_dict[param_code] = {}
#prm_dict[param_code]['param_name'] = 'Tracked Blob y1 Variance'
#prm_dict[param_code]['param_units'] = 'px'
#prm_dict[param_code]['bins'] = 25
#prm_dict[param_code]['bins_range'] = None

# Build lists of parameters. ###################################################
def generate_param_list(dayDict,param_code_dict):
    param_dict  = {}
    for item in dayDict:
        dt  = item['sDatetime']
        param_dict[dt] = {}

        
        param_dict[dt]['mstid'] = False
        param_dict[dt]['quiet'] = False
        param_dict[dt]['none']  = False

        # Keep track of category. ###################################################### 
        if 'category_manu' in item:
            if item['category_manu'] == 'mstid':
                categ = 'mstid'
                param_dict[dt]['mstid'] = True
            elif item['category_manu'] == 'quiet':
                categ = 'quiet'
                param_dict[dt]['quiet'] = True
            else:
                categ = 'none'
                param_dict[dt]['none']  = True
        else:
            categ = 'none'
            param_dict[dt]['none']  = True

        param_dict[dt]['category']  = categ
        param_dict[dt]['color']     = cat_dict[categ.lower()]['color']

        # Keep track of why things are rejected. #######################################

        #Things may not have a tracker record because either:
        #   1. The tracker has not been run on the data.
        #   2. The tracker failed gloriously because there was no data.
        #To compute tracker records with 'blob_sl' codes:
        #   First run 'blob_measure.py' to do the tracking and create a special blob only database.
        #   Then, run 'blob_epoch_plotting.py' to add it to the main MSTID database.

        try:
            param_dict[dt]['good_period']  = item['good_period']
        except:
            param_dict[dt]['good_period']  = None

        try:
            param_dict[dt]['blob_track-total_blobs']   = item['blob_track']['total_blobs'] 
        except:
            param_dict[dt]['blob_track-total_blobs']   = np.nan

        try:
            param_dict[dt]['blob_track-short_list_nr']         = item['blob_track']['short_list_nr'] 
        except:
            param_dict[dt]['blob_track-short_list_nr']         = np.nan

        try:
            param_dict[dt]['blob_track-short_list_min_hours']  = item['blob_track']['short_list_min_hours'] 
        except:
            param_dict[dt]['blob_track-short_list_min_hours']  = np.nan

        try:
            param_dict[dt]['slt']  = item['slt']
        except:
            param_dict[dt]['slt']  = np.nan
        
        try:
            param_dict[dt]['mlt']  = item['mlt']
        except:
            param_dict[dt]['mlt']  = np.nan

        # Find actual parameters of interest. ########################################## 
        for param_code,code_dict in param_code_dict.items():
            if 'blob_track' in param_code:
                base,code = param_code.split('-')
                try:
                    val = np.array(item[base][code])[0]
                except:
                    val = np.nan
            else:
                try:
                    if param_code == 'nr_sigs':
                        val = len(item['signals'])
                    elif param_code == 'orig_rti_var':
                        val = (item['signals'][0]['orig_rti_std'])**2
                    elif param_code == 'spec_ratio':
                        val = item['signals'][0]['dom_spec'] / item['signals'][0]['total_spec']
                    else:
                        val = item['signals'][0][param_code]
                except:
                    val = np.nan

            param_dict[dt][param_code] = val

    return param_dict

events_from_db      = db[mstid_list].find()
data_dict           = generate_param_list(events_from_db,prm_dict)
df_raw              = pd.DataFrame.from_dict(data_dict,orient='index')

events_from_db.rewind()
radar               = events_from_db[0]['radar']
sTime               = df_raw.index.min()
eTime               = df_raw.index.max()
slt_min             = df_raw['slt'].min()
slt_max             = df_raw['slt'].max()


# Plot histograms here. ########################################################
#param_code   = 'dom_spec'
#prm_dict[param_code] = {}
#prm_dict[param_code]['param_name']   = 'Dominant Spectrum'
#prm_dict[param_code]['param_units']  = ''
#prm_dict[param_code]['bins']         = 25
#prm_dict[param_code]['bins_range']   = None


if test:
    params_of_int       = ['orig_rti_cnt', 'total_spec', 'orig_rti_mean', 'orig_rti_var']
else:
    params_of_int       = list(prm_dict.keys())
#for param_code,param_dict in prm_dict.iteritems():
for param_code in params_of_int:
    param_dict  = prm_dict[param_code]
    for var in list(param_dict.keys()):
        locals()[var] = param_dict[var]

    fl_pfx      = '.'.join([radar,sTime.strftime('%Y%m%d'),eTime.strftime('%Y%m%d'),param_code])

    nr_xplots   = 1
    nr_yplots   = 3
    plot_nr     = 0
    figsize     = (10*nr_xplots,6*nr_yplots)
    fig         = plt.figure(figsize=figsize)

    param_code_data = {}
    for key in ['mstid','quiet','none']:
        param_code_data[key]            = {}
        df_categ                        = df_raw[df_raw[key]]
        series                          = df_categ[param_code]
        param_code_data[key]['series']  = series

        txt = []
        txt.append('Non-null events: %d' % len(series.dropna()))
        txt.append('Total events: %d' % len(series))

        nr  = len(df_categ[df_categ['good_period'] == True])
        txt.append('Good_periods: %d' % nr)

        nr  = len(df_categ['blob_track-short_list_nr'].nonzero()[0])
        txt.append('Blob Non-Zero Events: %d' % nr)

        param_code_data[key]['legend_text']  = '\n'.join(txt)


    param_code_data['max'] = np.nanmax(df_raw[param_code])
    
    #for key in cat_dict:
    for key in ['mstid','quiet','none']:

        plot_nr     = plot_nr + 1
        axis        = fig.add_subplot(nr_yplots,nr_xplots,plot_nr)

        if param_dict.get('bins_range') is not None:
            xrng    = param_dict['bins_range']
        else:
            xrng    = (0,param_code_data['max'])

        series      = param_code_data[key]['series']

        legend_text = param_code_data[key]['legend_text']
        series.hist(bins=bins,color=cat_dict[key]['color'],label=legend_text,ax=axis,range=xrng)

        if param_units != '':
            txt = '%s [%s]' % (param_name,param_units)
        else:
            txt = param_name

        axis.set_xlabel(txt)
        axis.set_xlim(xrng)
        axis.legend(loc='upper right',fontsize=8)

        axis.set_ylabel('Number of Events')
        title_pfx   = ''

        title = []
        title.append(title_pfx + 'Distribution of %s for %s Events' % (param_name,cat_dict[key]['title']))
        title = '\n'.join(title)
        axis.set_title(title)

#        txt = []
#        txt.append('%d Plotted Events' % len(values))
#        for info_key,info_items in info_dict[key].iteritems():
#            txt.append('%d %s' % (info_items,info_key.title()))

    axis.text(1.01,0,'Generated by: '+curr_file,size='xx-small',rotation=90,va='bottom',transform=axis.transAxes)
    txt = []
    txt.append(radar.upper()+sTime.strftime(' %Y %b %d - ')+eTime.strftime(' %Y %b %d')+'; %.1f-%.1f SLT' % (slt_min,slt_max))
    txt.append('[%s]' % mstid_list)
    fig.text(0.5,0.93,'\n'.join(txt),ha='center',size='xx-large')

    fig.savefig(os.path.join(output_dirs['search'],'%s_distributions.png' % fl_pfx))
    fig.clf()


################################################################################
import random
import statsmodels.api as sm

feature_list_list = []
if not test:
    print('Starting combinations...')
    import itertools
    #for L in range(len(prm_dict.keys())+1):
    for L in range(5):
        if L == 0:continue
        for subset in itertools.combinations(list(prm_dict.keys()),L):
            feature_list_list.append(subset)
else:
    feature_list_list.append(['orig_rti_cnt', 'orig_rti_mean', 'orig_rti_var'])
    feature_list_list.append(['orig_rti_cnt', 'total_spec', 'orig_rti_mean', 'orig_rti_var'])
    feature_list_list.append(['dom_spec','blob_track-short_list_mean_raw_mean','blob_track-short_list_var_raw_mean'])
    feature_list_list.append(['dom_spec','orig_rti_mean', 'orig_rti_var'])
    feature_list_list.append(['orig_rti_cnt', 'orig_rti_var'])


results_list = []

#dv  = 'none'    #Dependent variable we are going to model.
dv  = 'mstid'    #Dependent variable we are going to model.

feat_nr     = 0
total_feat  = len(feature_list_list)
for feature_list in feature_list_list:
    feat_nr += 1
    print('(%d of %d): ' % (feat_nr,total_feat),feature_list)
    #Drop all rows that have any NaNs in the feature of interest columns
    df  = df_raw.dropna(subset=feature_list,how='any')

    if dv != 'none':
        df = df.loc[df['none'] == False]

    # statsmodels requires us to add a constant column representing the intercept
    df['intercept'] = 1.0

    scaled_list = [f + '_scaled' for f in feature_list]
    for feat,scl in zip(feature_list, scaled_list):
        mx = max(df[feat])
#        print feat,mx
        df[scl] = 1.0*df[feat] / mx

    training_pct    = 0.60
    training_nr     = int(training_pct*len(df))

    rand_inx = random.sample(list(range(len(df))), training_nr)
    rand_inx.sort()

    df_training     = df.iloc[rand_inx]

    remaining_inx   = list(set(range(len(df))) - set(rand_inx))

    #cv --> Cross-validtation (I think!)
    cv_inx          = random.sample(remaining_inx,len(remaining_inx)//2)
    test_inx        = list(set(remaining_inx) - set(cv_inx))

    df_cv           = df.iloc[cv_inx]
    df_test         = df.iloc[test_inx]

    # identify the independent variables 
    scaled_list.append('intercept')
    logit           = sm.Logit(df_training[dv], df_training[scaled_list])
    result          = logit.fit()

    # get the fitted coefficients from the results
    coeff = result.params

    def pz(df,coeff):
        # compute the linear expression by multipyling the inputs by their respective coefficients.
        # note that the coefficient array has the intercept coefficient at the end
        z = 0
        for cf,name in zip(coeff,coeff._index):
            z += cf*df[name]
        return 1/(1+np.exp(-1*z))

    cv_results      = pz(df_cv,coeff)

    cv_srt_inx      = sorted(df_cv.index ,key=lambda x: df_cv.loc[x,dv])
    perfect_case    = df_cv.loc[cv_srt_inx,dv].astype(np.float)

    results_vec     = cv_results.loc[cv_srt_inx]
    sum_sqrs        = np.sum( (results_vec - perfect_case)**2 )

    #Get the probabilities of just the dependent variable case.
    cv_dv_inx       = df_cv[df_cv[dv]].index
    cv_dv_results   = cv_results[cv_dv_inx]

    cv_ndv_inx      = df_cv[~df_cv[dv]].index
    cv_ndv_results  = cv_results[cv_ndv_inx]

    cv_dv_mean      = cv_dv_results.mean()
    cv_dv_std       = cv_dv_results.std()
    cv_ndv_mean     = cv_ndv_results.mean()
    cv_ndv_std      = cv_ndv_results.std()

    stdevs      = 2.
    dv_val      = cv_dv_mean  - stdevs*cv_dv_std
    ndv_val     = cv_ndv_mean + stdevs*cv_ndv_std

    if dv_val  < 0.: dv_val  = 0.
    if ndv_val > 1.: ndv_val = 1.

    cutoff  = np.abs(dv_val - ndv_val)

    dv_correct      = float(np.sum(cv_dv_results  >  cutoff))
    ndv_correct     = float(np.sum(cv_ndv_results <= cutoff))

    dv_pct_correct  = dv_correct  / len(cv_dv_results)
    ndv_pct_correct = ndv_correct / len(cv_ndv_results)

    mean_pct_correct = np.mean([dv_pct_correct, ndv_pct_correct]) * 100.

#    print dv_pct_correct
#    print ndv_pct_correct
#    print mean_pct_correct

    results_dict = {}
    results_dict['feature_list']    = feature_list
    results_dict['df_cv']           = df_cv
    results_dict['cv_srt_inx']      = cv_srt_inx
    results_dict['df_test']         = df_test
    results_dict['result']          = result
    results_dict['cv_results']      = cv_results
    results_dict['sum_sqrs']        = sum_sqrs
    results_dict['r_sqrd']          = result.prsquared
    results_dict['aic']             = result.aic
    results_dict['cv_dv_mean']      = cv_dv_mean 
    results_dict['cv_dv_std']       = cv_dv_std  
    results_dict['cv_ndv_mean']     = cv_ndv_mean
    results_dict['cv_ndv_std']      = cv_ndv_std 
    results_dict['dv_val']          = dv_val
    results_dict['ndv_val']         = ndv_val
    results_dict['cutoff']          = cutoff
    results_dict['mean_pct_correct'] = mean_pct_correct

    results_list.append(results_dict)

def plot_reg(max_plots=50,savefig=False,clf=False):
    # Plotting results... ##########################################################

    figsize     = (10,8)
    markersize  = 30.
    ser_nr      = 0
#    sort_by     = 'sum_sqrs'
#    sort_by     = 'r_sqrd'
    sort_by     = 'mean_pct_correct'
#    sort_by     = 'aic'

    sorted_list = sorted(results_list, key=lambda k: k[sort_by])
    if sort_by == 'r_sqrd' or sort_by == 'mean_pct_correct':
        sorted_list = sorted_list[::-1]

    for results_dict in sorted_list: #resd is results_dict
        ser_nr      += 1

        feature_list    = results_dict['feature_list']
        df_cv           = results_dict['df_cv']
        cv_srt_inx      = results_dict['cv_srt_inx']
        df_test         = results_dict['df_test']
        result          = results_dict['result']
        cv_results      = results_dict['cv_results']
        sum_sqrs        = results_dict['sum_sqrs']
        aic             = results_dict['aic']
        r_sqrd          = results_dict['r_sqrd']

        cv_dv_mean      = results_dict['cv_dv_mean']
        cv_dv_std       = results_dict['cv_dv_std']
        cv_ndv_mean     = results_dict['cv_ndv_mean']
        cv_ndv_std      = results_dict['cv_ndv_std']

        dv_val          = results_dict['dv_val']
        ndv_val         = results_dict['ndv_val']
        cutoff          = results_dict['cutoff']
        mean_pct_correct = results_dict['mean_pct_correct']

#        print ('Plotting: '+str(feature_list))
        log.info('Plotting: '+str(feature_list))

        fig         = plt.figure(figsize=figsize)

        # Plot the known values here. ################################################## 
        axis        = fig.add_subplot(211)
        xvals       = np.arange(df_cv.loc[cv_srt_inx,dv].size)
        yvals       = np.array(df_cv.loc[cv_srt_inx,dv].astype(np.int))
        color       = np.array(df_cv.loc[cv_srt_inx,'color'].tolist())
        labels      = np.array(df_cv.loc[cv_srt_inx,'category'].tolist())
        axis.scatter(xvals,yvals,s=markersize,c=color,lw=0)
        axis.set_ylabel('P(\'%s\')' % dv)
        axis.set_ylim(-0.1,1.1)
        axis.yaxis.set_ticks(np.arange(0,1.1,0.2))
    #    axis.set_xlabel('Sample Index')
        axis.set_title('Cross Validation Set Known Values\nMSTID List: %s' % mstid_list)

        lgd_list    = []
        lgd_lbls    = []
        for cat_key in ['mstid','quiet','none']:
            lgd_list.append(matplotlib.patches.Circle((0,0),markersize,color=cat_dict[cat_key]['color']))

            nr_cat      = np.sum( labels == cat_key )
            lgd_lbls.append('%d %s' % (nr_cat,cat_dict[cat_key]['title']))
        axis.legend(lgd_list,lgd_lbls,loc='upper left',fontsize=10)


        # Plot the predicted values here. ############################################## 
        axis        = fig.add_subplot(212)
        yvals       = cv_results.loc[cv_srt_inx]
        axis.scatter(xvals,yvals,s=markersize,c=color,lw=0)

        hline_ls    = '--'
        axis.axhline(dv_val,  ls=hline_ls,color=cat_dict[dv]['color'])
        axis.axhline(ndv_val, ls=hline_ls,color=color[0])

        axis.axhline(cutoff, color = 'k')

        txt = []
        txt.append(str(feature_list))

        l2  = []
        l2.append('Sum Squares: %.2f' % sum_sqrs)
        l2.append('AIC : %.2f' % result.aic)
        l2.append('R2: %.3f' % r_sqrd)
        l2.append('MPC: %.1f %%' % mean_pct_correct)
        txt.append('; '.join(l2))

        axis.set_title('\n'.join(txt))
        axis.set_ylabel('P(\'%s\')' % dv)
        axis.set_ylim(0,1)
        axis.set_xlabel('Sample Index')

        outFName = '%03d-' % ser_nr +'.'.join(feature_list)+'.png'
        fig.tight_layout()
        if savefig: fig.savefig(os.path.join(output_dirs['result'],outFName),bbox_inches='tight')
        if clf: fig.clf()
        if max_plots:
            if ser_nr == max_plots:
                return

if __name__ == '__main__':
    plot_reg(savefig=True,clf=True)

