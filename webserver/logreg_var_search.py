#!/usr/bin/env python

import sys
sys.path.append('/data/mstid/statistics/webserver')

import os
import shutil
import inspect
import datetime
import pickle
from operator import itemgetter
import glob

import numpy as np
from scipy import stats as stats
from scipy.io import readsav
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import pymongo
# from bson.objectid import ObjectId

# import utils

import stats_support as ssup

curr_file = inspect.getfile(inspect.currentframe()) # script filename (usually with path)

font = { 'weight' : 'bold', 'size'   : 12}

matplotlib.rc('font', **font)

# Get events from mongo database. ##############################################
mongo         = pymongo.MongoClient()
db            = mongo.mstid

#mstid_list  = 'automated_run_1_bks_0817mlt'
mstid_list  = 'paper2_wal_2012'
output_dir  = 'output/logreg_search/%s' % mstid_list

try:
    shutil.rmtree(output_dir)
except:
    pass
os.makedirs(output_dir)

show_all_txt = '''
<?php
foreach (glob("*.png") as $filename) {
    echo "<img src='$filename' > <br />";
}
?>
'''
with open(os.path.join(output_dir,'0000-show_all.php'),'w') as file_obj:
    file_obj.write(show_all_txt)

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

param_code = 'blob_sl_mean_count'
prm_dict[param_code] = {}
prm_dict[param_code]['param_name'] = 'Tracked Blob Pixel Count Mean'
prm_dict[param_code]['param_units'] = 'px'
prm_dict[param_code]['bins'] = 25
prm_dict[param_code]['bins_range'] = None

param_code = 'blob_sl_var_count'
prm_dict[param_code] = {}
prm_dict[param_code]['param_name'] = 'Tracked Blob Pixel Count Variance'
prm_dict[param_code]['param_units'] = 'px'
prm_dict[param_code]['bins'] = 25
prm_dict[param_code]['bins_range'] = None

param_code = 'blob_sl_mean_raw_mean'
prm_dict[param_code] = {}
prm_dict[param_code]['param_name'] = 'Tracked Blob Mean Power Mean'
prm_dict[param_code]['param_units'] = 'dB'
prm_dict[param_code]['bins'] = 25
prm_dict[param_code]['bins_range'] = None

param_code = 'blob_sl_var_raw_mean'
prm_dict[param_code] = {}
prm_dict[param_code]['param_name'] = 'Tracked Blob Mean Power Variance'
prm_dict[param_code]['param_units'] = 'dB'
prm_dict[param_code]['bins'] = 25
prm_dict[param_code]['bins_range'] = None

param_code = 'blob_sl_mean_box_x0'
prm_dict[param_code] = {}
prm_dict[param_code]['param_name'] = 'Tracked Blob x0 Mean'
prm_dict[param_code]['param_units'] = 'px'
prm_dict[param_code]['bins'] = 25
prm_dict[param_code]['bins_range'] = None

param_code = 'blob_sl_var_box_x0'
prm_dict[param_code] = {}
prm_dict[param_code]['param_name'] = 'Tracked Blob x0 Variance'
prm_dict[param_code]['param_units'] = 'px'
prm_dict[param_code]['bins'] = 25
prm_dict[param_code]['bins_range'] = None

param_code = 'blob_sl_mean_box_x1'
prm_dict[param_code] = {}
prm_dict[param_code]['param_name'] = 'Tracked Blob x1 Mean'
prm_dict[param_code]['param_units'] = 'px'
prm_dict[param_code]['bins'] = 25
prm_dict[param_code]['bins_range'] = None

param_code = 'blob_sl_var_box_x1'
prm_dict[param_code] = {}
prm_dict[param_code]['param_name'] = 'Tracked Blob x1 Variance'
prm_dict[param_code]['param_units'] = 'px'
prm_dict[param_code]['bins'] = 25
prm_dict[param_code]['bins_range'] = None

param_code = 'blob_sl_mean_box_y0'
prm_dict[param_code] = {}
prm_dict[param_code]['param_name'] = 'Tracked Blob y0 Mean'
prm_dict[param_code]['param_units'] = 'px'
prm_dict[param_code]['bins'] = 25
prm_dict[param_code]['bins_range'] = None

param_code = 'blob_sl_var_box_y0'
prm_dict[param_code] = {}
prm_dict[param_code]['param_name'] = 'Tracked Blob y0 Variance'
prm_dict[param_code]['param_units'] = 'px'
prm_dict[param_code]['bins'] = 25
prm_dict[param_code]['bins_range'] = None

param_code = 'blob_sl_mean_box_y1'
prm_dict[param_code] = {}
prm_dict[param_code]['param_name'] = 'Tracked Blob y1 Mean'
prm_dict[param_code]['param_units'] = 'px'
prm_dict[param_code]['bins'] = 25
prm_dict[param_code]['bins_range'] = None

param_code = 'blob_sl_var_box_y1'
prm_dict[param_code] = {}
prm_dict[param_code]['param_name'] = 'Tracked Blob y1 Variance'
prm_dict[param_code]['param_units'] = 'px'
prm_dict[param_code]['bins'] = 25
prm_dict[param_code]['bins_range'] = None

normalize   = False

mstidDayDict,quietDayDict,noneDayDict = ssup.loadDayLists(mstid_list)
mstidDays   = [x['date'] for x in mstidDayDict]
quietDays   = [x['date'] for x in quietDayDict]
noneDays    = [x['date'] for x in noneDayDict]

mstid_mlts  = [x['mlt'] for x in mstidDayDict]
quiet_mlts  = [x['mlt'] for x in quietDayDict]
none_mlts   = [x['mlt'] for x in noneDayDict]

mstid_radar = [x['radar'] for x in mstidDayDict]
quiet_radar = [x['radar'] for x in quietDayDict]
none_radar  = [x['radar'] for x in noneDayDict]

all_events  = mstidDays + quietDays + noneDays
sTime       = min(all_events)
eTime       = max(all_events)

all_mlts    = mstid_mlts + quiet_mlts + none_mlts
min_mlt     = min(all_mlts)
max_mlt     = max(all_mlts)

radar       = '_'.join(list(np.unique(np.array(mstid_radar + quiet_radar + none_radar))))

# Build lists of parameters. ###################################################
def generate_param_list(dayDict,param_code_dict):
    param_dict  = {}

    for param_code,code_dict in param_code_dict.items():
        param_dict[param_code] = {}

        #Keep track of why things are getting rejected.
        info_dict   = {}
        info_dict['total']                          = 0
        info_dict['no_tracker_record']              = 0
        info_dict['rejected_by_tracker']            = 0
        info_dict['rejected_by_checkDataQuality']   = 0
        info_dict['matching_rejects']               = 0

        param_list  = []

        for item in dayDict:
            info_dict['total']  += 1

            #Sampling windows are innocent until proven guilty.
            reject                                  = False
            rejected_by_checkDataQuality            = False
            rejected_by_tracker                     = False
            no_tracker_record                       = False     #Things may not have a tracker record because either:
                                                                #   1. The tracker has not been run on the data.
                                                                #   2. The tracker failed gloriously because there was no data.
                                                                #To compute tracker records with 'blob_sl' codes:
                                                                #   First run 'blob_measure.py' to do the tracking and create a special blob only database.
                                                                #   Then, run 'blob_epoch_plotting.py' to add it to the main MSTID database.

            #Blob lists do not necessarily match MUSIC detected signals. Handle them separately.
            if not item['good_period']:
                info_dict['rejected_by_checkDataQuality'] += 1
                rejected_by_checkDataQuality        = True
                reject                              = True

            if 'blob_sl' in param_code:
                if param_code in item:
                    if len(item[param_code]) == 0:
                        info_dict['rejected_by_tracker'] += 1
                        rejected_by_tracker         = True
                        reject                      = True
                else:
                    info_dict['no_tracker_record']  += 1
                    no_tracker_record               = True

            if rejected_by_checkDataQuality and (rejected_by_tracker or no_tracker_record):
                info_dict['matching_rejects']       += 1

            if reject: continue

            if 'blob_sl' in param_code:
                if param_code in item:
                    for val in item[param_code]:
                        if np.isfinite(val):
                            param_list.append(val)
            else:
                if 'signals' in item:
                    if len(item['signals']) == 0: continue
                    if param_code == 'nr_sigs':
                        val = len(item['signals'])
                    elif param_code == 'orig_rti_var':
                        val = (item['signals'][0]['orig_rti_std'])**2
                    elif param_code == 'spec_ratio':
                        val = item['signals'][0]['dom_spec'] / item['signals'][0]['total_spec']
                    else:
                        val = item['signals'][0][param_code]
                    if np.isfinite(val):
                        param_list.append(val)
        param_dict[param_code]['list']    = param_list
        param_dict[param_code]['info']    = info_dict
    return param_dict

mstid_data_dict     = generate_param_list(mstidDayDict,prm_dict)
quiet_data_dict     = generate_param_list(quietDayDict,prm_dict)
none_data_dict      = generate_param_list(noneDayDict,prm_dict)

save_obj = {}
save_obj['mstid']   = mstid_data_dict
save_obj['quiet']   = quiet_data_dict
save_obj['none']    = none_data_dict
pckl_file = os.path.join(output_dir,mstid_list+'.p')

with open(pckl_file,'wb') as pckl_obj:
    pickle.dump(save_obj,pckl_obj)

for param_code,param_dict in prm_dict.items():
    for var in list(param_dict.keys()):
        locals()[var] = param_dict[var]

    fl_pfx      = '.'.join([radar,sTime.strftime('%Y%m%d'),eTime.strftime('%Y%m%d'),param_code])

    mstid_param_list    = mstid_data_dict[param_code]['list']
    mstid_info_dict     = mstid_data_dict[param_code]['info']

    quiet_param_list    = quiet_data_dict[param_code]['list']
    quiet_info_dict     = quiet_data_dict[param_code]['info']

    none_param_list     = none_data_dict[param_code]['list']
    none_info_dict      = none_data_dict[param_code]['info']

    # Define category dictionary for plotting. #####################################
    # You can change the number of things plotted by simply commenting out a cat_dict entry
    # or by changing the number of mlt_bins.
    cat_dict = {}
    cat_dict['all']     = {'title':'MSTID', 'color':'c'}
    cat_dict['mstid']   = {'title':'MSTID', 'color':'r'}
    cat_dict['quiet']   = {'title':'Quiet', 'color':'g'}
    cat_dict['none']    = {'title':'None' , 'color':'b'}

    data_dict = {}
    for key in list(cat_dict.keys()):
        data_dict[key]  = {}

    data_dict['mstid']['values']    = mstid_param_list
    data_dict['quiet']['values']    = quiet_param_list
    data_dict['none']['values']     = none_param_list
    data_dict['all']['values']      = mstid_param_list + quiet_param_list + none_param_list

    info_dict = {}
    info_dict['mstid']              = mstid_info_dict
    info_dict['quiet']              = quiet_info_dict
    info_dict['none']               = none_info_dict

    # Pre-calculate histograms #####################################################
    bins_range = prm_dict[param_code]['bins_range']  
    for key in cat_dict:
        hist,bins   = np.histogram(data_dict[key]['values'],bins=bins,range=bins_range)
        data_dict[key]['hist']  = hist.astype(np.float)
        data_dict[key]['bins']  = bins

    if bins_range == None:
        xlimits = (0,np.max(data_dict['all']['bins']))

    if normalize:
        keys = list(data_dict.keys())
        keys.remove('all')
        for key in keys:
            data_dict[key]['hist'] = data_dict[key]['hist'] / data_dict['all']['hist']

    # Begin Plotting ###############################################################

    # Plot Distribution for MSTID, Quiet, and None Days ############################
    nr_xplots   = 1
    nr_yplots   = len(list(cat_dict.keys())) - 1
    plot_nr     = 0
    figsize     = (10*nr_xplots,6*nr_yplots)
    fig         = plt.figure(figsize=figsize)

    #for key in cat_dict:
    param_units = prm_dict[param_code]['param_units'] 
    param_name = prm_dict[param_code]['param_name']
    for key in ['mstid','quiet','none']:
        if key == 'all': continue
        for data_key in list(data_dict[key].keys()):
            vars()[data_key]    = data_dict[key][data_key]

        
        plot_nr     = plot_nr + 1
        axis        = fig.add_subplot(nr_yplots,nr_xplots,plot_nr)
        width       = np.diff(bins)
        centers     = bins[:-1] + width/2.
        patches     = axis.bar(centers,hist,width=width,align='center',color=cat_dict[key]['color'])

        if param_units != '':
            txt = '%s [%s]' % (param_name,param_units)
        else:
            txt = param_name

        axis.set_xlabel(txt)

        if not normalize:
            axis.set_ylabel('Number of Events')
            title_pfx   = ''
        else:
            axis.set_ylabel('Normalized to Total Distribution')
            title_pfx   = 'Normalized '

        title = []
        title.append(title_pfx + 'Distribution of %s for %s Events' % (param_name,cat_dict[key]['title']))
    #    title.append('[%s]' % mstid_list)
        title = '\n'.join(title)
        axis.set_title(title)

        if bins_range == None:
            axis.set_xlim(xlimits)
        else:
            axis.set_xlim(bins_range)

        txt = []
        txt.append('%d Plotted Events' % len(data_dict[key]['values']))
        for info_key,info_items in info_dict[key].items():
            txt.append('%d %s' % (info_items,info_key.title()))

        axis.legend([patches[0]],['\n'.join(txt)],prop={'size':10,'weight':'normal'})

    axis.text(1.01,0,'Generated by: '+curr_file,size='xx-small',rotation=90,va='bottom',transform=axis.transAxes)
    txt = []
    txt.append(radar.upper()+sTime.strftime(' %Y %b %d - ')+eTime.strftime(' %Y %b %d')+'; %d-%d MLT' % (min_mlt,max_mlt))
    txt.append('[%s]' % mstid_list)
    fig.text(0.5,0.93,'\n'.join(txt),ha='center',size='xx-large')

    fig.savefig(os.path.join(output_dir,'%s_distributions.png' % fl_pfx))
    fig.clf()

