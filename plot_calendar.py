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
#tunnel,mongo_port           = mstid.createTunnel() 

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


group_dict  = mstid.run_helper.create_default_radar_groups_all_years()

#mstid.calendar_plot(dct_list,db_name=db_name)
png_path    = mstid.calendar_plot_lib.plot_mstid_index(sDate=dct['list_sDate'],eDate=dct['list_eDate'],driver=['mstid_reduced_index'],group_dict=group_dict,
        paper_legend=False,plot_letters=True,file_suffix='_mstid_index',dct_list=dct_list)
print(png_path)
