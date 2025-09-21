#!/usr/bin/env python3
import sys
import os
import datetime
import subprocess

import numpy as np

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

import mstid
from mstid import run_helper

from hdf5_api import loadMusicArrayFromHDF5


years = [2015]
radars = []
radars.append('bks')
# radars.append('pyk')

db_name                     = 'mstid_GSMR_fitexfilter_HDF5_fig3'
base_dir                    = os.path.join('mstid_data',db_name)

beam = 7

output_dir = os.path.join('output','summary_rtp',db_name)
mstid.general_lib.prepare_output_dirs({0:output_dir},clear_output_dirs=False)

for year in years:
    dct                             = {}
    dct['radars']                   = radars
    dct['list_sDate']               = datetime.datetime(year,1,1)
    dct['list_eDate']               = datetime.datetime(year,12,31)
    dct['db_name']                  = db_name
    dct['data_path']                = os.path.join(base_dir,'mstid_index')

    dct_list                        = run_helper.create_music_run_list(**dct)
  
    for dct_list_item in dct_list:
        radar              = dct_list_item['radar']
        list_sDate         = dct_list_item['list_sDate']
        list_eDate         = dct_list_item['list_eDate']
        mstid_list         = dct_list_item['mstid_list']
        mongo_port         = dct_list_item['mongo_port']
        db_name            = dct_list_item['db_name']
        data_path          = dct_list_item['data_path']
        print(f"Processing {radar} {list_sDate} to {list_eDate} from {mstid_list}")
 
        # Get events from MongoDB
        # Note that recompute=True does not actually recompute or change anything in the database.
        # It merely prevents the function from filtering out events that are already processed.
        events = mstid.mongo_tools.events_from_mongo(**dct_list_item,process_level='rti_interp',recompute=True)

        for event in events:
            sTime = event['sTime']
            eTime = event['eTime']
            # Get HDF file for event.
            hdf_file = mstid.more_music.get_hdf5_name(event['radar'],sTime,eTime,data_path=data_path,getPath=True)
            if not os.path.isfile(hdf_file):
                print(f"  WARNING: HDF5 file {hdf_file} not found.")
                continue

            musicObj = loadMusicArrayFromHDF5(hdf_file)
            ds       = musicObj.DS000_originalFit
            time_tf  = np.logical_and(ds.time >= sTime, ds.time < eTime)

            event_name = os.path.splitext(os.path.basename(hdf_file))[0]
            png_fname  = f'{event_name}.png'
            png_fpath  = os.path.join(output_dir,png_fname)

            
            fig = plt.figure()
            ax  = fig.add_subplot(2,1,1)

            ax  = fig.add_subplot(2,1,2)

            fig.tight_layout()
            fig.savefig(png_fpath,bbox_inches='tight')
            print(f'SAVED {png_fpath}')

            pass



        print(f"  END Processing {radar} {list_sDate} to {list_eDate} from {mstid_list}")


    print(year)
   
print("I'm done!")
