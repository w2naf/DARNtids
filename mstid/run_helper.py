#!/usr/bin/env python
import datetime

from .more_music import generate_initial_param_file,run_music_init_param_file

from .mongo_tools import generate_mongo_list, \
        generate_mongo_list_from_list,events_from_mongo

import itertools
import numpy as np

import multiprocessing
import subprocess

def create_music_run_list(radars,list_sDate,list_eDate,
        db_name='mstid',mongo_port=27017,
        mstid_format='guc_{radar}_{sDate}_{eDate}',
        use_input_list=False,
        input_db_name='mstid_aggregate',input_mongo_port=27017,
        input_mstid_format='guc_{radar}_{sDate}_{eDate}',
        music = False,
        **kwargs):
    """
    Generates a list of dictionaries with run parameters used by the MSTID
    MUSIC and Classification system.
    """
    if music:
        mstid_format   = 'music_guc_{radar}_{sDate}_{eDate}'

    dct_list = []
    for radar in radars:
        dct                     = {}
        dct['list_sDate']       = list_sDate
        dct['list_eDate']       = list_eDate
        dct['radar']            = radar
        date_fmt                = '%Y%m%d'
        sd_str                  = dct['list_sDate'].strftime(date_fmt)
        ed_str                  = dct['list_eDate'].strftime(date_fmt)
        dct['mstid_list']       = mstid_format.format(radar=radar,sDate=sd_str,eDate=ed_str)
        dct['db_name']          = db_name
        dct['mongo_port']       = mongo_port
        if use_input_list:
            dct['input_mstid_list'] = input_mstid_format.format(radar=radar,sDate=sd_str,eDate=ed_str)
            dct['input_db_name']    = input_db_name
            dct['input_mongo_port'] = input_mongo_port

        dct.update(kwargs)
        dct_list.append(dct)
    return dct_list

def create_group_dict(radars,list_sDate,list_eDate,group_name,group_dict={},**kwargs):
    """
    Adds a music_run_list to a dictionary.
    Each music_run_list in this dictionary is a "group".
    This makes it easy to group radars into "high latitude" and "mid latitude" groups.
    """

    dct_list                    = create_music_run_list(radars,list_sDate,list_eDate,**kwargs)

    key                         = len(list(group_dict.keys()))
    group_dict[key]             = {}
    group_dict[key]['dct_list'] = dct_list
    group_dict[key]['name']     = group_name

    return group_dict
        
def create_default_radar_groups(list_sDate=datetime.datetime(2014,11,1),list_eDate = datetime.datetime(2015,5,1),**kwargs):
    """
    Creates a radar group_dict for default sets of high latitude ('sas','pgr','kap','gbr')
    and mid latitude radars (cvw,cve,fhw,fhe,bks,wal).
    """

    # User-Defined Run Parameters Go Here. #########################################
    radar_groups = {}

    group_name      = 'High Latitude Radars'
    radars          = []
    radars.append('pgr')
    radars.append('sas')
    radars.append('kap')
    radars.append('gbr')
    radar_groups    = create_group_dict(radars,list_sDate,list_eDate,group_name,radar_groups,**kwargs)

    group_name      = 'Mid Latitude Radars'
    radars          = []
    radars.append('cvw')
    radars.append('cve')
    radars.append('fhw')
    radars.append('fhe')
    radars.append('bks')
    radars.append('wal')
    radar_groups    = create_group_dict(radars,list_sDate,list_eDate,group_name,radar_groups,**kwargs)

#    group_name      = 'West Looking'
#    radars          = []
#    radars.append('cvw')
#    radars.append('fhw')
#    radars.append('bks')
#    radar_groups    = create_group_dict(radars,list_sDate,list_eDate,group_name,radar_groups,**kwargs)
#
#    group_name      = 'East Looking'
#    radars          = []
#    radars.append('cve')
#    radars.append('fhe')
#    radars.append('wal')
#    radar_groups    = create_group_dict(radars,list_sDate,list_eDate,group_name,radar_groups,**kwargs)

    return radar_groups

def create_default_radar_groups_all_years(**kwargs):
    """
    Creates a list of default radar groups for all of the MSTID seasons I am looking at.
    """

    seDates = []
#    seDates.append( (datetime.datetime(2010,11,1),datetime.datetime(2011,5,1)) )
#    seDates.append( (datetime.datetime(2011,11,1),datetime.datetime(2012,5,1)) )
    seDates.append( (datetime.datetime(2012,11,1),datetime.datetime(2013,5,1)) )
    seDates.append( (datetime.datetime(2013,11,1),datetime.datetime(2014,5,1)) )
    seDates.append( (datetime.datetime(2014,11,1),datetime.datetime(2015,5,1)) )

    radar_group_list    = []
    for sDate,eDate in seDates:
        radar_group_list.append(create_default_radar_groups(sDate,eDate,**kwargs))

    return radar_group_list

def get_all_default_mstid_lists(**kwargs):
    """
    Pulls out the mongoDB collection names contained in a 
    create_default_radar_groups_all_years() list.
    """

    all_years   = create_default_radar_groups_all_years(**kwargs)

    mstid_lists = []
    for radar_groups in all_years:
        for key,radar_group in radar_groups.items():
            for run_dict in radar_group['dct_list']:
                mstid_lists.append(run_dict['mstid_list'])

    return mstid_lists

def run_init_file(init_file):
    """
    Launches the MUSIC script as its own process to isolate its
    memory management.
    """
    cmd = ['./run_single_event.py',init_file]
    print(' '.join(cmd))
    subprocess.check_call(cmd)

def get_events_and_run(dct_list,process_level=None,new_list=False,
        category=None,recompute=False,multiproc=True,nprocs=None,**dct):
    """
    Launch the MUSIC scripts for multiple events given a list of dictionaries
    describing which radars to use, the start and end dates of the run,
    and MUSIC script options.
    """

    events      = []
    for dct in dct_list:
        mongo_port          = dct.get('mongo_port',27017)
        mstid_list          = dct.get('mstid_list')
        db_name             = dct.get('db_name','mstid')

        input_mongo_port    = dct.get('input_mongo_port',27017)
        input_mstid_list    = dct.get('input_mstid_list')
        input_db_name       = dct.get('input_db_name','mstid_aggregate')

        # Generate Clean Mongo MSTID List
        if new_list:
            if input_mstid_list is None:
                generate_mongo_list(**dct)
            else:
                generate_mongo_list_from_list(mstid_list,db_name,mongo_port,
                        input_mstid_list,input_db_name,input_mongo_port,
                        category=category)

        if process_level:
            dct['process_level']    = process_level

        # Figure out which events need to be computed.
        these_events    = events_from_mongo(category=category,recompute=recompute,**dct)
        events += these_events

    # Prepare initial_param.json files #############################################
    init_files  = [generate_initial_param_file(event) for event in events]

    # Send events off to MUSIC for rti_interp level processing. ####################
    if multiproc:
        if len(init_files) > 0:
            pool = multiprocessing.Pool(nprocs)
            pool.map(run_init_file,init_files)
            pool.close()
            pool.join()
    else:
        for init_file in init_files:
            cmd = ['./run_single_event.py',init_file]
            print(' '.join(cmd))
            run_music_init_param_file(init_file)

def get_seDates_from_groups(radar_groups,date_fmt='%d %b %Y',sep='_'):
    dates = []
    for key,group in radar_groups.items():
        for dct in group['dct_list']:
            dates.append(dct['list_sDate'])
            dates.append(dct['list_eDate'])

    seDates = (min(dates), max(dates))

    if date_fmt is not None:
        sDate_str   = seDates[0].strftime(date_fmt)
        eDate_str   = seDates[1].strftime(date_fmt)

        seDates = sep.join([sDate_str,eDate_str])

    return seDates

def get_key_dates(sDate,eDate,key_days=[1,15]):
    """
    Returns datetime.datetime of the 1st and 15th of each month
    between sDate and eDate.
    """

    s_month = sDate.month
    # Start our month cycle on the starting month.
    months  = itertools.cycle(np.roll(np.arange(12)+1,-s_month+1))

    key_dates = []
    curr_date = sDate
    while curr_date < eDate:
        month   = next(months)
        year    = curr_date.year

        if month == 1 and len(key_dates) > 0:
            year = year + 1

        for day in key_days:
            curr_date   = datetime.datetime(year,month,day)
            if curr_date >= sDate and curr_date < eDate:
                key_dates.append(curr_date)

    return key_dates

if __name__ == "__main__":
    radar_groups    = create_default_radar_groups()
    mstid_lists     = get_all_default_mstid_lists()
