#!/usr/bin/env python3
import sys
import os
import datetime
import subprocess

import matplotlib
matplotlib.use('Agg')

import multiprocessing

import mstid
from mstid import run_helper

# User-Defined Run Parameters Go Here. #########################################
radars = []
radars.append('cvw')
radars.append('cve')
radars.append('fhw')
radars.append('fhe')
radars.append('bks')
radars.append('wal')

radars.append('sas')
radars.append('pgr')
radars.append('kap')
radars.append('gbr')

db_name                     = 'mstid'
# Used for creating an SSH tunnel when running the MSTID database on a remote machine.
tunnel,mongo_port           = mstid.createTunnel() 

dct                         = {}
dct['radars']               = radars
dct['list_sDate']           = datetime.datetime(2012,11,1)
dct['list_eDate']           = datetime.datetime(2013,5,1)
#dct['list_sDate']           = datetime.datetime(2012,12,1)
#dct['list_eDate']           = datetime.datetime(2012,12,15)
dct['hanning_window_space'] = False # Set to False for MSTID Index Calculation
dct['bad_range_km']         = None  # Set to None for MSTID Index Calculation
#dct['mongo_port']           = mongo_port
dct['db_name']              = db_name
dct['data_path']            = 'mstid_data/mstid_index'
dct_list                    = run_helper.create_music_run_list(**dct)

mstid_index         = True
new_list            = True
reupdate_db         = True

music_process       = False
music_new_list      = True
music_reupdate_db   = True

nprocs              = 20
multiproc           = False

# Classification parameters go here. ###########################################
classification_path = 'mstid_data/classification'

#******************************************************************************#
# No User Input Below This Line ***********************************************#
#******************************************************************************#
if mstid_index:
    # Generate MSTID List and do rti_interp level processing.
    run_helper.get_events_and_run(dct_list,process_level='rti_interp',new_list=new_list,
            multiproc=multiproc,nprocs=nprocs)
    import ipdb; ipdb.set_trace()

    # Reload RTI Data into MongoDb. ################################################
    if reupdate_db:
        for dct in dct_list:
            mstid.updateDb_mstid_list(multiproc=multiproc,nprocs=nprocs,**dct)

    for dct in dct_list:
        mstid.classify.classify_none_events(**dct)
        mstid.classify.rcgb(classification_path=classification_path,**dct)

    # Run FFT Level processing on unclassified events.
    run_helper.get_events_and_run(dct_list,process_level='fft',category='unclassified',
            multiproc=multiproc,nprocs=nprocs)

    # Now run the real MSTID classification.
    mstid.classify.run_mstid_classification(dct_list,classification_path=classification_path,
            multiproc=multiproc,nprocs=5)

    print('Plotting calendar plot...')
    mstid.calendar_plot(dct_list,db_name=db_name,mongo_port=mongo_port)

# Run actual MUSIC Processing ##################################################
if music_process:
    for dct in dct_list:
        dct['input_mstid_list']     = dct['mstid_list']
        dct['input_db_name']        = dct['db_name']
        dct['input_mongo_port']     = dct['mongo_port']
        dct['mstid_list']           = 'music_'+dct['mstid_list']
        dct['data_path']            = 'mstid_data/music_data'
        dct['hanning_window_space'] = True
#        dct['bad_range_km']         = 500 # Set to 500 for MUSIC Calculation
        dct['bad_range_km']         = None # Set to None to match original calculations

    run_helper.get_events_and_run(dct_list,process_level='music',
            new_list=music_new_list,category=['mstid','quiet'],
            multiproc=multiproc,nprocs=nprocs)

    if music_reupdate_db:
        for dct in dct_list:
            mstid.updateDb_mstid_list(multiproc=multiproc,nprocs=nprocs,**dct)

#tunnel.kill()
print("I'm done!")
