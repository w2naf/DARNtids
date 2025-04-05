#!/usr/bin/env python
# A version of the MSTID calendar plot with the polar vortex in a movie-strip form.
# Hopefully this will be a primary figure in MSTID paper 2.

import sys
import os
import glob
import datetime
import multiprocessing

import matplotlib
matplotlib.use('Agg')

import mstid
from mstid import run_helper
from mstid import prepare_output_dirs

# User-Defined Run Parameters Go Here. #########################################
radars = []
radars.append('cvw')
radars.append('cve')
radars.append('fhw')
radars.append('fhe')
radars.append('bks')
#radars.append('wal')

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

# User-Defined Run Parameters Go Here. #########################################
sDate               = dct['list_sDate']
eDate               = dct['list_eDate']

correlate           = False
multiproc           = False
plot_amplitude      = True
plot_direction      = False

output_dir          = os.path.join('output','driver_timeseries')
prepare_output_dirs({0:output_dir},clear_output_dirs=True)

db_name             = 'mstid'
#tunnel,mongo_port   = mstid.createTunnel()

# For MSTID amplitude plotting.
all_years           = mstid.run_helper.create_default_radar_groups_all_years()

# For MSTID Direction Plotting.
mstid_list_format   = 'music_guc_{radar}_{sDate}_{eDate}'
music_groups        = mstid.run_helper.create_default_radar_groups_all_years(mstid_format=mstid_list_format)

mstid_reduced_inx = mstid.calculate_reduced_mstid_index(all_years,
        reduction_type='mean',daily_vals=True, db_name=db_name)

drivers = []
tmp = {}
tmp['driver']   = ['mstid_reduced_inx'] 
drivers.append(tmp)

#tmp = {}
#tmp['driver']   = ['mstid_reduced_inx','smoothed_ae','omni_symh']
#drivers.append(tmp)
#
#tmp = {}
#tmp['driver']   = ['mstid_reduced_inx','neg_mbar_diff']
#drivers.append(tmp)

#tmp = {}
#tmp['driver']   = ['mstid_reduced_inx','smoothed_ae','omni_symh','neg_mbar_diff']
#drivers.append(tmp)

#tmp = {}
#tmp['driver']   = ['mstid_reduced_inx','neg_mbar_diff']
#tmp['plot_geopot_maps'] = True
#drivers.append(tmp)

for driver_dct in drivers:
    driver                  = driver_dct.get('driver','mstid_reduced_inx')
    plot_geopot_maps        = driver_dct.get('plot_geopot_maps',False)
    classification_colors   = driver_dct.get('classification_colors',True)
    section_title           = str('; '.join(driver)).upper()

    file_suffix = '_'.join(driver)

    for radar_groups,music_group in zip(all_years,music_groups):
        date_str            = run_helper.get_seDates_from_groups(radar_groups,date_fmt='%Y%m%d')
        season              = run_helper.get_seDates_from_groups(radar_groups,date_fmt='%Y')
        dates               = run_helper.get_seDates_from_groups(radar_groups,date_fmt=None)
        key_dates           = run_helper.get_key_dates(*dates)
        if season != '2012_2013': continue
        print('Calendar plot: {}; Driver: {!s}'.format(date_str,driver))

        if plot_amplitude:
            # Amplitude
            tmp = {}
            tmp['group_dict']           = radar_groups
            tmp['driver']               = driver
            tmp['db_name']              = db_name
#            tmp['mongo_port']           = mongo_port
            tmp['output_dir']           = output_dir
            tmp['mstid_reduced_inx']    = mstid_reduced_inx 
            tmp['correlate']            = correlate
            tmp['classification_colors']= classification_colors
            tmp['sDate']                = sDate
            tmp['eDate']                = eDate


            png_path    = mstid.calendar_plot_lib.plot_mstid_index(paper_legend=True,plot_letters=True,
                                                    file_suffix='_'+file_suffix,dct_list=dct_list,**tmp)
            print(png_path)

print("I'm done!")
