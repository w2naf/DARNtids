#!/usr/bin/env python3
"""
Script to plot summary of RTP data over multiple days for multiple radars for a single UTC time bin.
This is useful for seeing long-term trends in RTP data and understanding the raw data going into the
MSTID index.

NAF - 30 September 2025
"""

import sys
import os
import datetime
import subprocess
import pickle

import numpy as np
from scipy.interpolate import interpn
import pandas as pd

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

import mstid
from mstid import run_helper

from hdf5_api import loadMusicArrayFromHDF5

plt.rcParams['font.size'] = 16
# plt.rcParams['font.weight'] = 'bold'
# plt.rcParams['axes.grid'] = True
plt.rcParams['axes.xmargin'] = 0
plt.rcParams['axes.ymargin'] = 0
plt.rcParams['grid.linestyle'] = ':'
plt.rcParams['figure.figsize'] = (8,6)

class RtpSummary(object):
    def __init__(self,list_sDate,list_eDate,win_time_mn,win_rng_km):
        """
        Class to hold summary of RTP data over multiple days.

        Parameters
        ----------
        list_sDate : datetime.datetime
            Start date of list.
        list_eDate : datetime.datetime
            End date of list.
        win_time_mn : np.array
            1D array of time grid in minutes for an individual radar window.
        win_rng_km : np.array
            1D array of range grid in km for an individual radar window.
        """

        self.list_sDate  = list_sDate
        self.list_eDate  = list_eDate
        self.win_time_mn = win_time_mn
        self.win_rng_km  = win_rng_km

        nDays               = (list_eDate - list_sDate).days

        nTimes              = len(win_time_mn)*nDays
        nRanges             = len(win_rng_km)

        times               = np.full(nTimes, None, dtype=object)
        tfreqs              = np.full(nTimes, np.nan, dtype=float)
        rtp_summary         = np.full( (nTimes, nRanges), np.nan, dtype=float )

        self.times          = times
        self.tfreqs         = tfreqs
        self.data           = rtp_summary

    def get_date_inxs(self,date):
        """
        Get indices for data array corresponding to a specific date.

        Parameters
        ----------
        date : datetime.datetime
            Date to get indices for.

        Returns
        -------
        inxs : np.array
            Indices for data array corresponding to the input date.
        """

        nDays       = (self.list_eDate - self.list_sDate).days
        if (date < self.list_sDate) or (date >= self.list_eDate):
            raise ValueError('Input date is out of range of the RtpSummary object.')

        day_index       = (date - self.list_sDate).days
        inx_0           =  day_index*len(self.win_time_mn)
        inx_1           = (day_index+1)*len(self.win_time_mn)

        return (inx_0, inx_1)

summary_sDate = datetime.datetime(2015,12,1)
summary_eDate = datetime.datetime(2016,2,1)

radars = []
radars.append('han')
radars.append('pyk')

radars.append('cvw')
radars.append('pgr')
radars.append('cve')
radars.append('fhw')
radars.append('sas')
radars.append('fhe')
radars.append('kap')
radars.append('bks')
radars.append('wal')
radars.append('gbr')

db_name                     = 'mstid_GSMR_fitexfilter_HDF5_fig3'
base_dir                    = os.path.join('mstid_data',db_name)

beam        = 7     # Radar beam index to plot
st_bin      = 18    # UTC bin start time
plot_events = True  # If True, plot each event's raw and gridded data.

output_dir = os.path.join('output','summary_rtp',db_name)
mstid.general_lib.prepare_output_dirs({0:output_dir},clear_output_dirs=False)
for radar in radars:
    radar_output_dir = os.path.join(output_dir,radar)
    mstid.general_lib.prepare_output_dirs({0:radar_output_dir},clear_output_dirs=False)

# Define time and range grid for individual radar windows.
win_time_0_mn  = 0
win_time_1_mn  = 120
win_time_dt_mn = 1
win_time_mn    = np.arange(win_time_0_mn,win_time_1_mn,win_time_dt_mn)

win_rng_0_km  = 0
win_rng_1_km  = 1000
win_rng_dr_km = 15
win_rng_km    = np.arange(win_rng_0_km,win_rng_1_km,win_rng_dr_km)

# Build 2D query grid (ny, nx)
Tq, Rq      = np.meshgrid(win_time_mn, win_rng_km, indexing='xy')  # shapes (len(win_rng_km), len(win_time_mn))

# Get list of years to process.
years  = list(set([summary_sDate.year,summary_eDate.year]))
years.sort()

# Create dictionary to hold events to be run.
summary = {}
for radar in radars:
    summary[radar] = {}
    summary_events = summary[radar]['events'] = []
    for year in years:
        dct                             = {}
        dct['radars']                   = [radar]
        dct['list_sDate']               = datetime.datetime(year,1,1)
        dct['list_eDate']               = datetime.datetime(year,12,31)
        dct['db_name']                  = db_name
        dct['data_path']                = os.path.join(base_dir,'mstid_index')

        dct_list                        = run_helper.create_music_run_list(**dct)
    
        for dct_list_item in dct_list:
            if (dct_list_item['list_eDate'] < summary_sDate) or (dct_list_item['list_sDate'] > summary_eDate):
                continue

            radar              = dct_list_item['radar']
            list_sDate         = dct_list_item['list_sDate']
            list_eDate         = dct_list_item['list_eDate']
            mstid_list         = dct_list_item['mstid_list']
            mongo_port         = dct_list_item['mongo_port']
            db_name            = dct_list_item['db_name']
            data_path          = dct_list_item['data_path']

            print(f"Evaluating MongoDB Collection {db_name}/{mstid_list}")

            # Get events from MongoDB
            # Note that recompute=True does not actually recompute or change anything in the database.
            # It merely prevents the function from filtering out events that are already processed.
            events = mstid.mongo_tools.events_from_mongo(**dct_list_item,process_level='rti_interp',recompute=True)

            # Keep only events that are matching the requested st_bin.
            for event in events:
                sTime = event['sTime']
                eTime = event['eTime']

                if sTime < summary_sDate or sTime >= summary_eDate:
                    continue

                if sTime.hour == st_bin:
                    summary[radar]['events'].append(event)
                    print(f'     Including event {event["mstid_list"]}:{event["sTime"]}-{event["eTime"]}.')
                
    cache_name = f'rtp_summary_{radar}_{summary_sDate.strftime("%Y%m%d")}_{summary_eDate.strftime("%Y%m%d")}_st{st_bin:02d}_{db_name}.pkl'
    summary[radar]['cache_name'] = cache_name
    
    nEvents = len(summary[radar]['events'])
    print(f'Found {nEvents} events for:\n   {cache_name}')

##########
for radar, radar_summary in summary.items():
    radar_output_dir = os.path.join(output_dir,radar)
    print(f"Processing radar {radar} into {radar_output_dir}")

    cache_name = radar_summary['cache_name']
    cache_path = os.path.join(radar_output_dir,cache_name)
    if os.path.isfile(cache_path):
        print(f'  Found cache file {cache_path}. Loading and skipping processing.')
        with open(cache_path,'rb') as file_obj:
            rtp_summary = pickle.load(file_obj)
    else:
        print(f'  No cache file {cache_path}. Processing and saving to cache after processing.')

        # Pre-allocate np.array for final plot.
        rtp_summary        = RtpSummary(summary_sDate,summary_eDate,win_time_mn,win_rng_km)

        for event in radar_summary['events']:
            sTime = event['sTime']
            eTime = event['eTime']
            # Get HDF file for event.
            hdf_file = mstid.more_music.get_hdf5_name(event['radar'],sTime,eTime,data_path=data_path,getPath=True)
            event_name = os.path.splitext(os.path.basename(hdf_file))[0]

            print(f"--> Processing event {event_name}")

            if not os.path.isfile(hdf_file):
                print(f"  WARNING: HDF5 file {hdf_file} not found.")
                continue

            musicObj = loadMusicArrayFromHDF5(hdf_file)
            try:
                ds       = musicObj.DS000_originalFit
            except:
                print(f"  WARNING: HDF5 file {hdf_file} has no DS000_originalFit.")
                continue

            time_tf  = np.logical_and(ds.time >= sTime, ds.time < eTime)

            # Get range of raw data.
            my_range_km = ds.fov['slantRCenter'][beam]
            range_tf    = np.isfinite(my_range_km)
            my_range_km = my_range_km[range_tf]

            # Get time vector of raw data in minutes relative to sTime.
            my_time     = ds.time[time_tf]
            my_time_mn  = my_time - sTime
            my_time_mn  = np.array([x.total_seconds() for x in my_time_mn])/60.
            
            # Get data array of raw data.
            try:
                my_data     = ds.data[time_tf,beam,:]
                my_data     = my_data[:,range_tf]
                win_data    = (interpn((my_time_mn,my_range_km), my_data, (Tq, Rq), method='linear',bounds_error=False)).T  # shape (len(win_time_mn), len(win_rng_km))
            except:
                print(f"  WARNING: HDF5 file {hdf_file} appears to be an empty array.")
                continue

            # Get indices for this event's data in the summary array.
            dinx_0, dinx_1 = rtp_summary.get_date_inxs(sTime)

            # Calculate and store time vector.
            win_time = pd.to_timedelta(win_time_mn,unit='m') + sTime
            rtp_summary.times[dinx_0:dinx_1] = win_time

            # Get tfreq vector.
            prm_tm      = musicObj.prm['time']
            tf = np.logical_and(prm_tm >= sTime-datetime.timedelta(minutes=5), prm_tm < eTime+datetime.timedelta(minutes=5))
            prm_tm      = prm_tm[tf]
            prm_tm_dt   = prm_tm - sTime
            prm_tm_mn   = np.array([x.total_seconds() for x in prm_tm_dt])/60.

            tfreq       = musicObj.prm['tfreq']
            tfreq       = tfreq[tf]
            # Resample tfreq to win_time_mn using nearest neighbor interpolation.
            try:
                tfreqs_resampled = interpn( (prm_tm_mn,), tfreq, win_time_mn, method='nearest', bounds_error=False)
                rtp_summary.tfreqs[dinx_0:dinx_1] = tfreqs_resampled

            except:
                print(f" WARNING: Could not interpolate tfreq for event {event_name}. Setting to NaN.")

            # Insert into summary array.
            rtp_summary.data[dinx_0:dinx_1,:] = win_data

            # Plot raw and gridded data for this event.
            if plot_events:
                png_fname  = f'{event_name}.png'
                png_fpath  = os.path.join(radar_output_dir,png_fname)
                
                fig = plt.figure(figsize=(10,8))
                
                ax  = fig.add_subplot(2,1,1)
                mpbl = ax.pcolormesh(my_time_mn,my_range_km,my_data[:-1,:-1].T)
                ax.set_xlim(win_time_0_mn,win_time_1_mn)
                ax.set_ylim(win_rng_0_km,win_rng_1_km)
                ax.set_xlabel('Time [min]')
                ax.set_ylabel('GS Mapped Range [km]')
                ax.set_title(event_name+'\nRaw Data')
                fig.colorbar(mpbl,label=r'$\lambda$ Power [dB]')

                ax  = fig.add_subplot(2,1,2)
                mpbl = ax.pcolormesh(win_time_mn,win_rng_km,win_data[:-1,:-1].T)
                ax.set_xlim(win_time_0_mn,win_time_1_mn)
                ax.set_ylim(win_rng_0_km,win_rng_1_km)
                ax.set_xlabel('Time [min]')
                ax.set_ylabel('GS Mappend Range [km]')
                ax.set_title('Gridded Data')
                fig.colorbar(mpbl,label=r'$\lambda$ Power [dB]')

                fig.tight_layout()
                fig.savefig(png_fpath,bbox_inches='tight')
                print(f'SAVED {png_fpath}')
                plt.close(fig)
            
        # Save cache file.
        with open(cache_path,'wb') as file_obj:
            pickle.dump(rtp_summary,file_obj)
        print(f'  SAVED cache file {cache_path}.')
        print(f"  END Processing {radar} {list_sDate} to {list_eDate} from {mstid_list}")

        # Plot summary for this radar and year.
        png_fname  = cache_name.replace('.pkl','.png')
        png_fpath  = os.path.join(radar_output_dir,png_fname)
        fig = plt.figure(figsize=(20,4))
        ax  = fig.add_subplot(1,1,1)
        XX  = np.arange(rtp_summary.data.shape[0])
        YY  = rtp_summary.win_rng_km
        ZZ  = rtp_summary.data[:-1,:-1].T
        mpbl = ax.pcolormesh(XX,YY,ZZ)
        # ax.set_xlim(0,120*10)
        # ax.set_ylim(win_rng_0_km,win_rng_1_km)
        ax.set_xlabel('Time')
        ax.set_ylabel('GS Mapped Range [km]')
        ax.set_title(f'RTP Summary {radar} {list_sDate.strftime("%Y")}\n{db_name} st{st_bin:02d}')
        fig.colorbar(mpbl,label=r'$\lambda$ Power [dB]')

        ax2 = ax.twinx()
        ax2.plot(XX,rtp_summary.tfreqs*1e-3,'r-')
        ax2.set_ylabel('Transmit Frequency [MHz]',color='r')
        ax2.tick_params(axis='y', labelcolor='r')
        ax2.set_ylim(0,50)

        date_0 = rtp_summary.list_sDate
        date_1 = rtp_summary.list_eDate
        months = pd.date_range(date_0,date_1,freq='MS') #.strftime('%Y-%m-%d').tolist()
        xticks = months.map(lambda x: (x - date_0).days * len(rtp_summary.win_time_mn)).to_numpy()
        xlabels = months.strftime('%b\n%Y').tolist()
        ax.set_xticks(xticks)
        ax.set_xticklabels(xlabels,ha='left')

        fig.tight_layout()
        fig.savefig(png_fpath,bbox_inches='tight')
        print(f'SAVED {png_fpath}')
        plt.close(fig)


################################################################################
# Make Complete Summary Plot
png_fname   = f'rtp_summary_{summary_sDate.strftime("%Y%m%d")}_{summary_eDate.strftime("%Y%m%d")}_st{st_bin:02d}_{db_name}.png'
png_fpath   = os.path.join(output_dir,png_fname)
nRadars     = len(summary.keys())

fig = plt.figure(figsize=(20,3*nRadars))
for pinx, (radar, radar_summary) in enumerate(summary.items()):
    radar_output_dir = os.path.join(output_dir,radar)

    ax  = fig.add_subplot(nRadars,1,pinx+1)
    print(f"Plotting radar {radar} in complete summary plot.")

    cache_name = radar_summary['cache_name']
    cache_path = os.path.join(radar_output_dir,cache_name)
    with open(cache_path,'rb') as file_obj:
        rtp_summary = pickle.load(file_obj)

    # Plot summary for this radar and year.
    XX  = np.arange(rtp_summary.data.shape[0])
    YY  = rtp_summary.win_rng_km
    ZZ  = rtp_summary.data[:-1,:-1].T
    mpbl = ax.pcolormesh(XX,YY,ZZ)
    # ax.set_xlim(0,120*10)
    # ax.set_ylim(win_rng_0_km,win_rng_1_km)
    ax.set_ylabel('GS Mapped Range\n[km]')
    # ax.set_title(f'RTP Summary {radar} {list_sDate.strftime("%Y")}\n{db_name} st{st_bin:02d}')

    ax.set_title(f'{radar.upper()}: {st_bin}-{st_bin+2} UTC',loc='left')
    if pinx == 0:
        ax.set_title(f'MongoDB: {db_name}',loc='right')

    fig.colorbar(mpbl,label=r'$\lambda$ Power [dB]')

    ax2 = ax.twinx()
    ax2.plot(XX,rtp_summary.tfreqs*1e-3,'r-')
    ax2.set_ylabel('TX Freq [MHz]',color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    ax2.set_ylim(0,50)

    date_0 = rtp_summary.list_sDate
    date_1 = rtp_summary.list_eDate
    months = pd.date_range(date_0,date_1,freq='MS') #.strftime('%Y-%m-%d').tolist()
    xticks = months.map(lambda x: (x - date_0).days * len(rtp_summary.win_time_mn)).to_numpy()
    # xlabels = months.strftime('%b\n%Y').tolist()
    xlabels = months.strftime('%b %Y').tolist()

    ax.grid(True,which='both',axis='x',ls=':',color='k',alpha=0.5)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels,ha='left')

ax.set_xlabel('Time')
fig.tight_layout()
fig.savefig(png_fpath,bbox_inches='tight')
print(f'SAVED {png_fpath}')
plt.close(fig)

print("I'm done!")