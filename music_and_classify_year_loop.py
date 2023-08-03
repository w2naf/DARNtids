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

#years = list(range(2019,2023))
years = list(range(2012,2013))

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


#    radars.append('sto')
#    radars.append('pyk')
#    radars.append('han')

#db_name                     = 'mstid'
db_name                     = 'mstid_MUSIC'
base_dir                    = 'mstid_data'
# Used for creating an SSH tunnel when running the MSTID database on a remote machine.
#tunnel,mongo_port           = mstid.createTunnel() 

for year in years:
    dct                         = {}
    dct['radars']               = radars
    dct['list_sDate']           = datetime.datetime(year,  11,1)
    dct['list_eDate']           = datetime.datetime(year+1, 5,1)
#    dct['list_sDate']           = datetime.datetime(2012,12,1)
#    dct['list_eDate']           = datetime.datetime(2012,12,15)
    dct['hanning_window_space'] = False # Set to False for MSTID Index Calculation
    dct['bad_range_km']         = None  # Set to None for MSTID Index Calculation
    #dct['mongo_port']           = mongo_port
    dct['db_name']              = db_name
    dct['data_path']            = os.path.join(base_dir,'mstid_index')
    dct['boxcar_filter']        = False
    dct['fitacf_dir']           = '/data/sd-data'
#    dct['fitacf_dir']           = '/data/sd-data_despeck'
#    dct['fitacf_dir']           = '/data/sd-data_fitexfilter'
    dct_list                    = run_helper.create_music_run_list(**dct)

    mstid_index         = True
    new_list            = True      # Create a completely fresh list of events in MongoDB. Delete an old list if it exists.
    recompute           = False     # Recalculate all events from raw data. If False, use existing cached pickle files.
    reupdate_db         = True 

    music_process       = True
    music_new_list      = True
    music_reupdate_db   = True

    nprocs              = 60
    multiproc           = True

    # Classification parameters go here. ###########################################
    classification_path = os.path.join(base_dir,'classification')

    #******************************************************************************#
    # No User Input Below This Line ***********************************************#
    #******************************************************************************#
    # Generate MSTID List and do rti_interp level processing.
    run_helper.get_events_and_run(dct_list,process_level='rti_interp',new_list=new_list,
            recompute=recompute,multiproc=multiproc,nprocs=nprocs)

    # Reload RTI Data into MongoDb. ################################################
    if reupdate_db:
        for dct in dct_list:
            mstid.updateDb_mstid_list(multiproc=multiproc,nprocs=nprocs,**dct)

    for dct in dct_list:
        # Determine if each event is good or bad based on:
        #   1. Whether or not data is available.
        #   2. Results of pyDARNmusic.utils.checkDataQuality()
        #       (Ensures radar is operational for a minimum amount of time during the data window.
        #        Default is to require the radar to be turned off no more than 10 minutes in the
        #        data window.)
        #   3. The fraction of radar scatter points present in the data window.
        #       (Default requires minimum 67.5% data coverage.)
        #   4. The percentage of daylight in the data window.
        #       (Default requires 100% daylight in the data window.)
        mstid.classify.classify_none_events(**dct) 

        # Generate a web page and copy select figures into new directory to make it easier
        # to evaluate data and see if classification algorithm is working.
        mstid.classify.rcgb(classification_path=classification_path,**dct)

    # Run FFT Level processing on unclassified events.
    run_helper.get_events_and_run(dct_list,process_level='fft',category='unclassified',
            multiproc=multiproc,nprocs=nprocs)

    # Now run the real MSTID classification.
    mstid.classify.run_mstid_classification(dct_list,classification_path=classification_path,
            multiproc=multiproc,nprocs=5)

    print('Plotting calendar plot...')
    calendar_output_dir = os.path.join(base_dir,'calendar')
    mstid.calendar_plot(dct_list,db_name=db_name,output_dir=calendar_output_dir)

    # Run actual MUSIC Processing ##################################################
    if music_process:
        for dct in dct_list:
            dct['input_mstid_list']     = dct['mstid_list']
            dct['input_db_name']        = dct['db_name']
            dct['input_mongo_port']     = dct['mongo_port']
            dct['mstid_list']           = 'music_'+dct['mstid_list']
            dct['data_path']            = os.path.join(base_dir,'music_data')
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
