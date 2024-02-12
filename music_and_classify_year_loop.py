#!/usr/bin/env python3
"""
music_and_classify_year_loop.py
Nathaniel A. Frissell
nathaniel.frissell@scranton.edu
5 February 2024

This script will calculate the level of Medium Scale Traveling Ionospheric Disturbance (MSTID)
activity observed by SuperDARN radars via the MSTID Index described by Frissell et al. (2016)
(https://doi.org/10.1002/2015JA022168). It will also run the SuperDARN MSTID MUSIC algorithm
(https://github.com/HamSCI/pyDARNmusic) to estimate the speed, propagation direction, and 
horizontal wavelength of the observed MSTIDs.

This script will take in a directory of SuperDARN FITACF files as input data and will store
the summary output in a MongoDB. Detailed numerical output is stored in pickle files in the 
specified output directory. PNG graphical output is stored there, too.
"""
import sys
import os
import datetime
import subprocess

import matplotlib
matplotlib.use('Agg')

import multiprocessing

import mstid
from mstid import run_helper

years = list(range(2018,2019))

radars = []
## Standard North American Radars
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

db_name                     = 'mstid_GSMR_fitexfilter' # Name of output MongoDB to store results.
base_dir                    = db_name

for year in years:
    dct                             = {}
#    dct['fovModel']                 = 'HALF_SLANT' # Use a 1/2 slant range mapping equation.
    dct['fovModel']                 = 'GS' # Use the Ground Scatter Mapping Equation (Bristow et al., 1994, https://doi.org/10.1029/93JA01470, Pg. 324)
    dct['radars']                   = radars
    dct['list_sDate']               = datetime.datetime(year,  11,1)
    dct['list_eDate']               = datetime.datetime(year+1, 5,1)
    dct['hanning_window_space']     = False # Set to False for MSTID Index Calculation. This will prevent the data at the field-of-view edges from tapering to zero.
    dct['bad_range_km']             = None  # Set to None for MSTID Index Calculation
    dct['db_name']                  = db_name
    dct['data_path']                = os.path.join(base_dir,'mstid_index') # Output directory for PNG and Pickle files.

    # FITACF data used was pre-filtered using the boxcar filter written by A.J. Ribeiro.
    # See fitexfilter/ directory for binary and script for pre-filtering FITACF data.
    dct['fitacf_dir']               = '/data/sd-data_fitexfilter'

    # 'rti_fraction_threshold' is the Minimum fraction of scatter in an observational window to be considered "good" data and not discarded.
    # Note that the value was 0.675 in Frissel et al. (2016), but is reduced to 0.25 for Frissell et al. (2024) GRL because the 2018-2019 season
    # was in a lower portion of the solar cycle and had less useable ground scatter.
    dct['rti_fraction_threshold']   = 0.25 
    dct_list                        = run_helper.create_music_run_list(**dct)

    mstid_index         = True      # If True, caculate the MSTID index.
    new_list            = True      # Create a completely fresh list of events in MongoDB. Delete an old list if it exists.
    recompute           = False     # Recalculate all events from raw data. If False, use existing cached pickle files.
    reupdate_db         = True      # Re-populate the MongoDB using the cached data files on disk.

    music_process       = False     # If True, use the MUSIC algorithm to determine MSTID wavelength, propagation direction, and speed. Not used in Frissell et al. (2024) GRL
    music_new_list      = True      # Create a completely fresh list of events in MongoDB for MUSIC processing. Delete an old list if it exists.
    music_reupdate_db   = True      # Re-populate the MongoDB for MUSIC processing using the cached data files on disk.

    nprocs              = 60        # Number of threads to use with multiprocessing.
    multiproc           = True      # Enable/disable multiprocessing.

    # Classification parameters go here. ###########################################
    classification_path = os.path.join(base_dir,'classification')

    #******************************************************************************#
    # No User Input Below This Line ***********************************************#
    #******************************************************************************#
    if mstid_index:
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
