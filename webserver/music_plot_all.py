#!/usr/bin/env python
import sys
sys.path.append('/data/mstid/statistics/webserver')

import os
import inspect
import datetime
import pickle
import shutil

import numpy as np
import matplotlib
matplotlib.use('Agg')

import pymongo
from bson.objectid import ObjectId

import music_support as msc

curr_file = inspect.getfile(inspect.currentframe()) # script filename (usually with path)

font = {'weight':'bold', 'size':22}
matplotlib.rc('font', **font)

# Get events from mongo database. ##############################################
mongo         = pymongo.MongoClient()
db            = mongo.mstid

# Runs for Paper ###############################################################
mstid_list = 'paper2_bks_2012'
#mstid_list = 'paper2_wal_2012'
#mstid_list = 'paper2_fhe_2012'
#mstid_list = 'paper2_fhw_2012'
#mstid_list = 'paper2_cve_2012'
#mstid_list = 'paper2_cvw_2012'
################################################################################

cursor  = db[mstid_list].find({'category_manu':'mstid'}).sort('sDatetime',1)

for item in cursor:
    radar       = item['radar']
    sDatetime   = item['sDatetime']
    fDatetime   = item['fDatetime']

    print(radar,sDatetime)
    if True:
        runfile_path = msc.get_pickle_name(radar,sDatetime,fDatetime,getPath=True,createPath=False,runfile=True)

        karr_file = os.path.join(os.path.dirname(runfile_path),'014_karr.png')
        if os.path.exists(karr_file):
            print('Already plotted!')
        else:
            import ipdb; ipdb.set_trace()
            msc.music_plot_all(runfile_path)
            print('   Success!!')
