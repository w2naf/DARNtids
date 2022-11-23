#!/usr/bin/env python
#This script will create superposed epoch plots of blob tracks and update the mstid database with reduced versions.
import os
import shutil
import inspect
# import datetime
# import pickle
import operator

# import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import pymongo

curr_file = inspect.getfile(inspect.currentframe()) # script filename (usually with path)

font = { 'weight' : 'bold'}
matplotlib.rc('font', **font)

# Get events from mongo database. ##############################################
mongo           = pymongo.MongoClient()
db              = mongo.mstid
blob_db         = mongo.blob

#mstid_list     = 'automated_run_1_bks_0817mlt'
plot            = True
longest_only    = True #Only use the blob with the longest lifetime from each sampling window.
mstid_list      = 'paper2_wal_2012'
#mstid_list      = 'paper2_bks_2010_2011_0817mlt'
output_dir      = 'output/blob_epoch/%s' % mstid_list

if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
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

param_dict_list = []
param_dict_list.append({'param_code':'count', 'param_name':'Pixel Count'})
param_dict_list.append({'param_code':'box_x0', 'param_name':'Left Bound'})
param_dict_list.append({'param_code':'box_x1', 'param_name':'Right Bound'})
param_dict_list.append({'param_code':'box_y0', 'param_name':'Lower Bound'})
param_dict_list.append({'param_code':'box_y1', 'param_name':'Upper Bound'})
param_dict_list.append({'param_code':'raw_mean', 'param_name':'Raw Mean'})

# Get events. ##################################################################
mstid_crsr  = db[mstid_list].find({'category_manu':'mstid'}).sort('sDatetime',1)
quiet_crsr  = db[mstid_list].find({'category_manu':'quiet'}).sort('sDatetime',1)
none_crsr   = db[mstid_list].find({'$and': [{'category_manu': {'$ne':'mstid'}}, {'category_manu': {'$ne':'quiet'}}]}).sort('sDatetime',1)

cat_dict = {}
cat_dict['mstid']   = {'title':'MSTID', 'color':'r','crsr':mstid_crsr}
cat_dict['quiet']   = {'title':'Quiet', 'color':'g','crsr':quiet_crsr}
cat_dict['none']    = {'title':'None' , 'color':'b','crsr':none_crsr}

#Initialize db that will be sent to mstid MongoDB.
blob_dict_db        = {}

# Build lists of parameters. ###################################################
for param_dict in param_dict_list:
    param_name = param_dict['param_name']
    param_code = param_dict['param_code']

    # Begin Plotting ###############################################################
    nr_xplots   = 1
    nr_yplots   = 1
    plot_nr     = 1
    figsize     = (10*nr_xplots,6*nr_yplots)
    fig         = plt.figure(figsize=figsize)
    axis        = fig.add_subplot(nr_yplots,nr_xplots,plot_nr)
    counts  = {'mstid':0, 'quiet':0, 'none':0}
    for cat_key in ['mstid','none','quiet']:
        cat     = cat_dict[cat_key]
        cat['crsr'].rewind()
        for event in cat['crsr']:
            radar   = event['radar']
            sTime   = event['sDatetime']
            eTime   = event['fDatetime']

            blob_data   = blob_db[mstid_list].find_one({'radar':event['radar'],'sDatetime':event['sDatetime'],'fDatetime':event['fDatetime']})

            total_blobs             = None
            short_list_nr           = None
            short_list_min_hours    = None
    
            db_mean     = []
            db_stddev   = []
            db_var      = []

            if blob_data is not None:
                total_blobs = len(list(blob_data['blob_dict'].keys()))
                short_list_min_hours = blob_data['short_list_min_hours']

                if 'short_list' in blob_data:
                    sl  = blob_data['short_list']
                    short_list_nr   = len(list(sl.keys()))
                    if not short_list_nr == 0:
                        if longest_only:
                            sorted_x    = sorted(iter(sl.items()), key=operator.itemgetter(1))
                            sl_keys     = [sorted_x[-1][0]]
                        else:
                            sl_keys     = list(sl.keys())

                        for sl_key in sl_keys:
                            blob    = blob_data['blob_dict'][sl_key]
                            hours_list  = [(x - sTime).total_seconds()/3600. for x in blob['dt']]
                            data_list   = blob[param_code]
                            zp          = list(zip(hours_list,data_list))
                            zp.sort(key=lambda tup: tup[0])

                            hours_list  = [x[0] for x in zp]
                            data_list   = [x[1] for x in zp]

                            db_mean.append(stats.nanmean(data_list))
                            db_stddev.append(stats.nanstd(data_list))
                            db_var.append(stats.nanstd(data_list)**2)

                            axis.plot(hours_list,data_list,color=cat['color'])
                            counts[cat_key] += 1

            #Yes!!! We are actually marking something down for every event, even if it is an empty list!
            mean_name   = 'short_list_mean_%s' % param_code
            stddev_name = 'short_list_stddev_%s' % param_code
            var_name    = 'short_list_var_%s' % param_code

            _id         = event['_id'] 
            if _id not in list(blob_dict_db.keys()): 
                blob_dict_db[_id] = {}
                blob_dict_db[_id]['short_list_nr']          = short_list_nr
                blob_dict_db[_id]['short_list_min_hours']   = short_list_min_hours
                blob_dict_db[_id]['total_blobs']            = total_blobs

            blob_dict_db[_id][mean_name]    = db_mean
            blob_dict_db[_id][stddev_name]  = db_stddev
            blob_dict_db[_id][var_name]     = db_var

    if plot:
        axis.set_xlim(0,2)
        axis.set_xlabel('Hours')
        axis.set_ylabel(param_name)
        txt = 'MSTID: %d, Quiet: %d, None: %d' % (counts['mstid'], counts['quiet'], counts['none'])
        axis.set_title(txt)

        fl_pfx      = '.'.join([mstid_list,param_code])
        fig.savefig(os.path.join(output_dir,'%s.png' % fl_pfx))
        fig.clf()

#            db[mstid_list].update({'_id':event['_id']},{'$set': {'blob_track':blob_dict_db}})

for _id,blob_dict in blob_dict_db.items():
    db[mstid_list].update({'_id':_id},{'$set': {'blob_track':blob_dict}})
    
