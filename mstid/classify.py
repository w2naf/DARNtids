#!/usr/bin/env python

# Proposal for classification logic.
# 1. Take complete list; run good/bad logic.
# 2. Use spectral classifciation to determine MSTID or not.
# 3. Test this on at least 2 radars.
# 4. Try to rank MSTIDs comparatively between 2 radars.

import os
import copy
import datetime
import glob
import shutil
import multiprocessing
import subprocess

import matplotlib
from matplotlib import pyplot as plt

import inspect
import logging
log = logging

#import logging.config
#curr_file           = inspect.getfile(inspect.currentframe()) # script filename (usually with path)
#logging.filename    = curr_file+'.log'
#logging.config.fileConfig("logging.conf")
#log                 = logging.getLogger("root")

import numpy as np
import pandas as pd

import pymongo

#import davitpy
#import davitpy.pydarn.proc.music as music

from .general_lib import prepare_output_dirs
from . import more_music
from .more_music import get_output_path
from hdf5_api import loadMusicArrayFromHDF5, saveMusicArrayToHDF5

def mstid_classification(radar,list_sDate,list_eDate,mstid_list,
        sort_key='meanSubIntSpect_by_rtiCnt',
        data_path='music_data/music',classification_path='classification',
        db_name='mstid',mongo_port=27017,**kwargs):

    date_fmt    = '%Y%m%d%H%M'
    cmd         = []
    cmd.append('./classify_mstid_list.py')
    cmd.append(radar)
    cmd.append(list_sDate.strftime(date_fmt))
    cmd.append(list_eDate.strftime(date_fmt))
    cmd.append(mstid_list)
    cmd.append(sort_key)
    cmd.append(data_path)
    cmd.append(classification_path)
    cmd.append(db_name)
    cmd.append(str(mongo_port))
    print(('MSTID Classifying: {} {} {!s} {!s}'.format(mstid_list,radar,list_sDate,list_eDate)))
    print((' '.join(cmd)))
    try:
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError as e:
        raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))

def mstid_classification_dct(dct):
    mstid_classification(**dct)

def run_mstid_classification(dct_list,multiproc=True,nprocs=None,
        classification_path='classification'):

    for dct in dct_list:
        dct['classification_path'] = classification_path

    if multiproc:
        if len(dct_list) > 0:
            pool = multiprocessing.Pool(nprocs)
            pool.map(mstid_classification_dct,dct_list)
            pool.close()
            pool.join()
        pass
    else:
        for dct in dct_list:
            mstid_classification_dct(dct)

def copy_plot(radar,sDatetime,fDatetime,output_dir,search_string,plot_type,width=300,data_path='music_data/music',top_text=None):
    """
    Copy a figure from the original processing directory to a directory for a new view and return an html td/img tag with the 
    figure's path and image display size.
    """
    # KArr Plot ####################################################################
    musicPath   = get_output_path(radar,sDatetime,fDatetime,data_path=data_path)
    rti_path    = glob.glob(os.path.join(musicPath,search_string))

    if len(rti_path) == 1:
        rti_path = rti_path[0]

        new_fname   = os.path.basename(musicPath)+'_{plot_type}.png'.format(plot_type=plot_type)
        new_path    = os.path.join(output_dir,new_fname)
        if os.path.exists(rti_path):
            shutil.copy(rti_path,new_path)
            print(('-----> {0}'.format(new_path)))
            if top_text:
                txt = '{}<br/>'.format(top_text)
            else:
                txt = ''
            return '<td>{}<img src="{}" width="{:d}px" /></td>'.format(txt,new_fname,width)

    print(('ERROR: {0}'.format(rti_path)))
    return '<td></td>'

def rcgb(mstid_list,db_name='mstid',mongo_port=27017,
        data_path='music_data/music',classification_path='classification',**kwargs):
    """
    rcgb: RTI Checker Good Bad

    This routine does not actually check if the RTI is good or bad. Instead, it reorganizes files
    and writes out html files to help the user quickly evaluate if the current classification
    algorithms are working correctly and doing a good job.
    """
    mongo       = pymongo.MongoClient(port=mongo_port)
    db          = mongo[db_name]
    categs      = ['None','Unclassified']

    output_path = os.path.join(classification_path,'rti_checker')

    output_dirs = {}
    html_files  = {}
    base_dir    = os.path.join(output_path,'good_bad',mstid_list)

    for categ in categs:
        output_dirs[categ] = os.path.join(base_dir,categ)
        html_files[categ]  = os.path.join(base_dir,categ)

    prepare_output_dirs(output_dirs,clear_output_dirs=True,img_extra="width='50%'")

    for categ in categs:
        html_file = os.path.join(base_dir,categ,'0000_classification.html'.format(categ))
        html_files[categ]  = html_file
        with open(html_file,'w') as fl:
            fl.write('<html>\n')

    for categ in categs:
        if categ == 'Unclassified':
            crsr        = db[mstid_list].find({'category_manu':{'$exists':False}})
            nr_events   = db[mstid_list].count_documents({'category_manu':{'$exists':False}})
#            crsr        = crsr.sort('sDatetime',1)
        else:
            crsr        = db[mstid_list].find({'category_manu':categ})
            nr_events   = db[mstid_list].count_documents({'category_manu':categ})

        crsr        = crsr.sort('orig_rti_fraction',1)
        output_dir  = output_dirs[categ] 

        with open(html_files[categ],'a') as fl:
            fl.write('<h1>{1} ({0}): {2:d} Events</h1>\n'.format(mstid_list,categ.upper(),nr_events))
            fl.write('<h2>Source: {0}</h2>\n'.format(data_path))
#            fl.write('<img src="{}" />'.format(spect_order_fname))
            fl.write('    <table>\n')
        for item in crsr:
            html_txt = []
            html_txt.append('        <tr>')
        
            radar       = item['radar']
            sDatetime   = item['sDatetime']
            fDatetime   = item['fDatetime']
           
            html_txt.append('<td>')
            html_txt.append('<table border="1" width="100%">')

            if categ == 'None':
                html_txt.append('<tr><td colspan="2"><ul>')
                for msg in item['reject_message']:
                    html_txt.append('<li>{}</li>'.format(msg))
                html_txt.append('</ul</td></tr>')

            for info_key in ['terminator_fraction','orig_rti_fraction']:
                html_txt.append('<tr>')
                html_txt.append('<td>{}</td>'.format(info_key))
                rti_frac = item.get(info_key)
                if rti_frac is None: rti_frac = np.nan
                html_txt.append('<td>{:0.2f}</td>'.format(rti_frac))
                html_txt.append('</tr>')

            html_txt.append('</table>')
            html_txt.append('</td>')

            html_txt.append(copy_plot(radar,sDatetime,fDatetime,output_dir,data_path=data_path,search_string='*karr.png',plot_type='karr'))
            html_txt.append(copy_plot(radar,sDatetime,fDatetime,output_dir,data_path=data_path,search_string='*_originalFit_RTI.png',plot_type='rti'))
            html_txt.append(copy_plot(radar,sDatetime,fDatetime,output_dir,data_path=data_path,search_string='*_finalDataRTI.png',plot_type='finalRti'))
            html_txt.append(copy_plot(radar,sDatetime,fDatetime,output_dir,data_path=data_path,search_string='*_fullSpectrum.png',plot_type='spectrum'))
            html_txt.append(copy_plot(radar,sDatetime,fDatetime,output_dir,data_path=data_path,search_string='*_classification.png',plot_type='classification'))

            html_txt.append('</tr>\n')
            with open(html_files[categ],'a') as fl:
                fl.write(' '.join(html_txt))

    for categ in categs:
        with open(html_files[categ],'a') as fl:
            fl.write('    </table>\n')
            fl.write('</html>\n')

    mongo.close()

def rcss(data_dict,classification_path='classification'):
    """
    RTI Checker Spectral Sort 
    """
    mstid_list  = data_dict['mstid_list']
    categs      = data_dict['categs']
    data_path   = data_dict['data_path']

    output_path = os.path.join(classification_path,'rti_checker')

    output_dirs = {}
    html_files  = {}
    base_dir    = os.path.join(output_path,'mstid_quiet',mstid_list)

    for categ in categs:
        output_dirs[categ] = os.path.join(base_dir,categ)
        html_files[categ]  = os.path.join(base_dir,categ)

    prepare_output_dirs(output_dirs,clear_output_dirs=True,img_extra="width='50%'")

    for categ in categs:
        html_file = os.path.join(base_dir,categ,'0000_classification.html'.format(categ))
        html_files[categ]  = html_file
        with open(html_file,'w') as fl:
            fl.write('<html>\n')

    for categ in categs:
        output_dir  = output_dirs[categ] 

        cat_dct     = data_dict[categ] 
        nr_events   = cat_dct['spect_df'].shape[1]
    
        sort_df  = cat_dct['sort_df']
        series_inxs = sort_df.index

        # Summary Plot Section #########################################################
        plot_dicts = {}

        plot_key        ='intSpect'
        tmp = {}
        tmp['yvals']    = sort_df[plot_key]
        tmp['ylabel']   = '$\sum$PSD'
#        tmp['ylim']     = (0,1400)
        plot_dicts[plot_key] = tmp

        plot_key        = 'intSpect_by_rtiCnt'
        tmp = {}
        tmp['yvals']    = sort_df[plot_key]
        tmp['ylabel']   = '$\sum$PSD/rti_poss'
#        tmp['ylim']     = (0,0.08)
        plot_dicts[plot_key] = tmp

        plot_key        ='meanSubIntSpect'
        tmp = {}
        tmp['yvals']    = sort_df[plot_key]
        tmp['ylabel']   = '$\sum$PSD - $\mu$'
#        tmp['ylim']     = (-450,650)
        tmp['thresh']   = 0
        plot_dicts[plot_key] = tmp

        plot_key        ='meanSubIntSpect_by_rtiCnt'
        tmp = {}
        tmp['yvals']    = sort_df[plot_key]
        tmp['ylabel']   = '($\sum$PSD - $\mu$)/rti_poss'
#        tmp['ylim']     = (-0.025,0.025)
        tmp['thresh']   = 0
        plot_dicts[plot_key] = tmp

        for plot_key in ['orig_rti_cnt','orig_rti_possible','orig_rti_fraction']:
            tmp = {}
            tmp['yvals']    = cat_dct['orig_rti_info'][plot_key][series_inxs]
            tmp['ylabel']   = plot_key
            plot_dicts[plot_key] = tmp

        plot_keys = []
        plot_keys.append('intSpect')
        plot_keys.append('meanSubIntSpect')
        plot_keys.append('intSpect_by_rtiCnt')
        plot_keys.append('meanSubIntSpect_by_rtiCnt')
        plot_keys.append('orig_rti_cnt')
        plot_keys.append('orig_rti_fraction')
        plot_keys.append('orig_rti_possible')

        nr_ay, nr_ax, ax_nr = (4,2,0)
        fig = plt.figure(figsize=(6*nr_ax,2.5*nr_ay))
        for plot_key in plot_keys:
            plot_dict = plot_dicts[plot_key]
            ax_nr += 1
            ax  = fig.add_subplot(nr_ay, nr_ax, ax_nr)
            xvals   = sort_df['order']
            yvals   = plot_dict['yvals']
            ax.plot(xvals,yvals)
            thresh_box(yvals,ax,thresh=plot_dict.get('thresh'))
            ax.set_xlabel('Sorting Order')
            ax.set_ylabel(plot_dict.get('ylabel'))
            ax.set_ylim(plot_dict.get('ylim'))
            ax.grid()

        fig.tight_layout(h_pad=0.001)

        title = []
        title.append('{1} ({0}): {2:d} Events'.format(mstid_list,categ.upper(),nr_events))
        title.append('Source: {0}'.format(os.path.basename(data_path)))
        fig.text(0.5,1.01,'\n'.join(title),ha='center',fontsize='x-large',fontweight='bold')

        spect_order_fname = 'spect_order.png'
        fig.savefig(os.path.join(output_dir,spect_order_fname),bbox_inches='tight')
        ################################################################################

        with open(html_files[categ],'a') as fl:
            fl.write('<h1>{1} ({0}): {2:d} Events</h1>\n'.format(mstid_list,categ.upper(),nr_events))
            fl.write('<h2>Source: {0}</h2>\n'.format(data_path))
            fl.write('<img src="{}" />'.format(spect_order_fname))
            fl.write('    <table>\n')
        for series_inx in series_inxs:
            html_txt = []
            html_txt.append('        <tr>')
        
            rad_sTime_eTime = cat_dct['radar_sTime_eTime'][series_inx]
            radar       = rad_sTime_eTime[0]
            sDatetime   = rad_sTime_eTime[1]
            fDatetime   = rad_sTime_eTime[2]
           
            html_txt.append('<td>')

            html_txt.append('<table border="1" width="100%">')
            html_txt.append('<tr><th colspan="2">{!s}</th><tr>'.format(sort_df['order'][series_inx]))
            html_txt.append('<tr><td>Int PSD</td><td>{:0.1f}</td></tr>'.format(sort_df['intSpect'][series_inx]))
            html_txt.append('<tr><td>orig_rti_cnt</td><td>{!s}</td></tr>'.format(cat_dct['orig_rti_info']['orig_rti_cnt'][series_inx]))
            html_txt.append('<tr><td>orig_rti_possible</td><td>{!s}</td></tr>'.format(cat_dct['orig_rti_info']['orig_rti_possible'][series_inx]))
            html_txt.append('<tr><td>orig_rti_fraction</td><td>{:0.2f}</td></tr>'.format(cat_dct['orig_rti_info']['orig_rti_fraction'][series_inx]))
            html_txt.append('</table>')
            html_txt.append('</td>')

            html_txt.append(copy_plot(radar,sDatetime,fDatetime,output_dir,data_path=data_path,search_string='*karr.png',plot_type='karr'))
            html_txt.append(copy_plot(radar,sDatetime,fDatetime,output_dir,data_path=data_path,search_string='*_originalFit_RTI.png',plot_type='rti'))
            html_txt.append(copy_plot(radar,sDatetime,fDatetime,output_dir,data_path=data_path,search_string='*_finalDataRTI.png',plot_type='finalRti'))
            html_txt.append(copy_plot(radar,sDatetime,fDatetime,output_dir,data_path=data_path,search_string='*_fullSpectrum.png',plot_type='spectrum'))
            html_txt.append(copy_plot(radar,sDatetime,fDatetime,output_dir,data_path=data_path,search_string='*_classification.png',plot_type='classification'))

            html_txt.append('</tr>\n')
            with open(html_files[categ],'a') as fl:
                fl.write(' '.join(html_txt))

    for categ in categs:
        with open(html_files[categ],'a') as fl:
            fl.write('    </table>\n')
            fl.write('</html>\n')

def thresh_box(yvals,ax,thresh=0.):
    if thresh is None: return
    nr_good = np.count_nonzero(yvals >= 0)
    nr_bad  = yvals.size - nr_good
    text = []

    if yvals.size != 0:
        pct = nr_good/float(yvals.size)*100.
        pct = '{:0.0f}'.format(pct)
    else:
        pct = 'undef'
    txt     = '{!s}% >= {!s}'.format(pct,float(thresh))
    text.append(txt)

    if yvals.size != 0:
        pct = nr_bad/float(yvals.size)*100.
        pct = '{:0.0f}'.format(pct)
    else:
        pct = 'undef'
    txt     = '{!s}% < {!s}'.format(pct,float(thresh))
    text.append(txt)
    props = dict(boxstyle='round', facecolor='white', alpha=0.75)
    ax.text(0.0275,0.800,'\n'.join(text),transform=ax.transAxes,bbox=props,fontsize='small')
    ax.axhline(0,ls='--',color='r',lw=2.0)

def classify_none_events(mstid_list,db_name='mstid',mongo_port=27017,
        rti_fraction_threshold=0.675,terminator_fraction_threshold=0.,**kwargs):
    """
    Classify an event period as good or bad based on:
        1. Whether or not data is available.
        2. Results of pyDARNmusic.utils.checkDataQuality()
            (Ensures radar is operational for a minimum amount of time during the data window.
             Default is to require the radar to be turned off no more than 10 minutes in the
             data window.)
        3. The fraction of radar scatter points present in the data window.
            (Default requires minimum <rti_fraction_threshold> data coverage.)
        4. The percentage of daylight in the data window.
            (Default requires 100% daylight in the data window.)

    If an event is determined to be bad, 'category_manu' is set to 'None' and a 'reject_message' is
    stored in the database explaining why the event was rejected.

    Arguments:
        mstid_list:     <str> Name of MongoDB collection to operate on.
        db_name:        <str> Name of MongoDB database to use.
        mongo_port:     <27017> Port number to connect to MongoDB.
        rti_fraction_threshold:         
                        <float> Minimum fraction of data coverage within the data window.
        terminator_fraction_threshold:
                        <float> Maximum terminator (nighttime) allowed in the data window
    """

    # Clear out any old classifications.
    mongo   = pymongo.MongoClient(port=mongo_port)
    db      = mongo[db_name]
    crsr    = db[mstid_list].find()
    for event in crsr:
        _id = event['_id']
        delete_list = ['category_auto','category_manu']
        for item in delete_list:
            if item in event:
                status = db[mstid_list].update_one({'_id':_id},{'$unset': {item: 1}})

    # Run the classifier.
    crsr                = db[mstid_list].find()
    bad_counter         = 0
    no_data             = 0
    bad_period          = 0
    no_orig_rti_fract   = 0
    low_orig_rti_fract  = 0
    low_termin_fract    = 0

    for item in crsr:
        good            = True
        reject_message  = []
        if 'no_data' in item:
            if item['no_data']:
                no_data += 1
                reject_message.append('No Data')
                good    = False

        if 'good_period' in item:
            if not item['good_period']:
                bad_period += 1
                reject_message.append('Failed Quality Check')
                good    = False

        if 'orig_rti_fraction' not in item:
            no_orig_rti_fract   += 1
            reject_message.append('No RTI Fraction')
            good = False
        else:
            # Check for minimum rti_fraction coverage
            if (item['orig_rti_fraction'] < rti_fraction_threshold):
                low_orig_rti_fract  += 1
                reject_message.append('Low RTI Fraction')
                good = False

        # Check for excessive terminator coverage
        if 'terminator_fraction' not in item:
            no_orig_rti_fract   += 1
            reject_message.append('No Terminator Fraction')
            good = False
        else:
            if (item['terminator_fraction'] > terminator_fraction_threshold):
                low_termin_fract    += 1
                reject_message.append('High Terminator Fraction')
                good = False

        if not good:
            bad_counter += 1
            entry_id = db[mstid_list].update_one({'_id':item['_id']}, {'$set':{'good_period':good,'category_manu': 'None', 'reject_message':reject_message}})

    print(('{bc:d} events marked as None.'.format(bc=bad_counter)))
    print(('no_data: {!s}'.format(no_data)))
    print(('bad_period: {!s}'.format(bad_period)))
    print(('no_orig_rti_fract: {!s}'.format(no_orig_rti_fract)))
    print(('low_orig_rti_fract: {!s}'.format(low_orig_rti_fract)))
    print(('low_termin_fract: {!s}'.format(low_termin_fract)))
    mongo.close()

def load_data_dict(mstid_list,data_path,use_cache=True,cache_dir='data',read_only=False,
        test_mode=False,db_name='mstid',mongo_port=27017):
    """
    This routine:
        1. Loads the spectrum and DS000_originalFit basic statistics for every event in an MSTID list.
        2. Integrates spectra over beam and gate to give event spectra that is only a function of frequency.
        3. Interpolates all spectra to ensure every event is on an identical grid.
        4. Places all of this information into a dictionary containing pandas dataframes.
        5. hdf5s the output into a cache file.

    Note that this processing is only done on "uncategorized" events.  So, if you want to re-create
    cache files, you should start with an unclassified MSTID list and be prepared to re-classify
    everything afterward.

    If use_cache is True and the routine can find a cached hdf5 file, that file will be loaded instead of
    processing the data from scratch.

    Finally, one dataframe is create that contains all of the event spectra in a single place.  This is
    created after all other processing and even cache loading.

    * mstid_list:   Name of MSTID List/mongo collection
    * data_path:    Location of MUSIC dataObj hdf5 files.
    * use_cache:    Try loading results of a previous run of this routine stored in hdf5 files.
    * cache_dir:    Where the cached hdf5 files are located.
    * read_only:    Safety switch to prevent overwriteing of files.  Useful when you don't want to 
                    fiddle with the cache.
    * test_mode:    Drop to ipdb after anlyzing 5 events.  Useful for debugging/development.
    * db_name:      Mongo database to connect to.  No mongo connection is established if only using cache.
    * mongo_port:   Port mongo should connect on.
    """

    cache_name  = os.path.join(cache_dir,'classify_{}.{}.h5'.format(mstid_list,os.path.basename(data_path)))

    if (not os.path.exists(cache_name) or not use_cache) and (not read_only):
        mongo       = pymongo.MongoClient(port=mongo_port)
        db          = mongo[db_name]

        print(("MSTID Classification: Cache <{}> does not exist.  Creating.".format(cache_name)))
        data_dict   = {'unclassified':{'color':'blue'},
                       'mstid': {'color':'red'},
                       'quiet': {'color':'green'}}

        categs                  = ['unclassified']

        data_dict['categs']         = categs
        data_dict['mstid_list']     = mstid_list
        data_dict['db_name']        = db_name
        data_dict['mongo_port']     = mongo_port
        data_dict['data_path']      = data_path

        for categ in categs:
            if categ == 'unclassified':
                crsr        = db[mstid_list].find({'category_manu':{'$exists':False}},no_cursor_timeout=True)
                count       = db[mstid_list].count_documents({'category_manu':{'$exists':False}})
            else:
                crsr        = db[mstid_list].find({'category_manu':categ},no_cursor_timeout=True)
                count       = db[mstid_list].count_documents({'category_manu':categ})

            orig_rti_inx    = []
            orig_rti_list   = []

            # Cycle through every event window in the list, collapse the spectrum in beam and gate,
            # and append all of the spectra to a list that will become a dataframe.
            for item_inx,item in enumerate(crsr):
                radar       = item['radar']
                sDatetime   = item['sDatetime']
                fDatetime   = item['fDatetime']

                print(("MSTID Classification: Loading dataObj ({!s}/{!s}): {!s} {!s}-{!s}".format(item_inx,count,radar,sDatetime,fDatetime)))
                dataObj = more_music.get_dataObj(radar,sDatetime,fDatetime,data_path=data_path)

                if dataObj is None:
                    continue

                if not hasattr(dataObj.active,'spectrum'):
                    continue

                # Get basic statistics on dataObj.DS000_originalFit data using the ranges of dataObj.active.
                orig_rti_info   = more_music.get_orig_rti_info(dataObj,sDatetime,fDatetime)

                # Reduce spectrum by integrating over beam and gate leaving it only a function of frequency.
                fvec        = dataObj.active.freqVec
                spec        = np.abs(dataObj.active.spectrum)
                spec        = np.nansum(spec,axis=2)
                spec        = np.nansum(spec,axis=1)
                
                # Put the reduced spectrum into a pandas series object.
                series      = pd.Series(spec,fvec)

                spect_df    = data_dict[categ].get('spect_df')
                if spect_df is None:
                    # Append spectrum to dictionary keyed by a index.
                    data_dict[categ]['spect_df']        = pd.DataFrame(series)
                    orig_rti_inx.append(0)
                    
                    # Keep track of the radar and time of data in another dictionary
                    # keyed by the same index.
                    data_dict[categ]['radar_sTime_eTime']     = {}
                    data_dict[categ]['radar_sTime_eTime'][0]  = (radar,sDatetime,fDatetime)
                else:
                    #Get the index for the next spectrum.
                    series.name = max(list(spect_df.keys())) + 1

                    # Append the spectrum and radar,sDatetime,fDatetime to the appropriate dictionaries.
                    data_dict[categ]['spect_df']    = spect_df.join(series,how='outer')
                    data_dict[categ]['radar_sTime_eTime'][series.name]  = (radar,sDatetime,fDatetime)
                    orig_rti_inx.append(series.name)

                # Save statistical infor that goes along with each data window.
                orig_rti_list.append(orig_rti_info)

                # Only run through a few event windows if we are debugging/developing.
                if test_mode and item_inx == 5:
                    break

            # Dataframize the RTI statisical information.
            data_dict[categ]['orig_rti_info'] = pd.DataFrame(orig_rti_list,index=orig_rti_inx)

        # Fill in NaNs and put everything onto same frequency grid.
        # All spectra are interpolated here and finally converted into a dataFrame.
        f_ext   = []
        for categ_inx,categ in enumerate(categs):
            if 'spect_df' not in data_dict[categ].keys():
                print('No spect_df found... returning...')
                return
            spect_df = data_dict[categ]['spect_df']
            f_ext.append(spect_df.index.min())
            f_ext.append(spect_df.index.max())

        f_min       = round(np.min(f_ext),4)
        f_max       = round(np.max(f_ext),4)
        n_steps     = round((f_max - f_min)/0.00005)
        fvec_new    = np.linspace(f_min,f_max,n_steps)

        for categ_inx,categ in enumerate(categs):
            spect_df    = data_dict[categ]['spect_df']
            series      = pd.Series(np.zeros_like(fvec_new),fvec_new)
            series.name = 'tmp'
            spect_df    = spect_df.join(series,how='outer')
            spect_df    = spect_df.interpolate()
            spect_df    = spect_df.reindex(fvec_new)
            del spect_df['tmp']

            data_dict[categ]['spect_df'] = spect_df

        # Save all of that hard work to disk!
        saveMusicArrayToHDF5(data_dict, cache_name)
        mongo.close()
    else:
        print(("I'm using the cache! ({})".format(cache_name)))
        data_dict = loadMusicArrayFromHDF5(cache_name)

    data_dict['all_spect_df'] = create_all_spect_df(data_dict)
    return data_dict

def create_all_spect_df(data_dict):
    """
    Create a new, combined data frame that will let us calculate the average
    spectrum.

    * data_dict:    Dictionary generated in load_data_dict() 
    """
    categs          = data_dict['categs']
    all_spect_df    = None
    for categ_inx,categ in enumerate(categs):
        spect_df    = data_dict[categ]['spect_df']
        
        if all_spect_df is None:
            all_spect_df = spect_df.copy()
        else:
            all_spect_df = all_spect_df.join(spect_df,how='outer',rsuffix='_'+categ)

    return all_spect_df

def create_all_orig_rti_info(data_dict):
    categs          = data_dict['categs']
    all_orig_rti_info    = None
    for categ_inx,categ in enumerate(categs):
        orig_rti_info    = data_dict[categ]['orig_rti_info']
        
        if all_orig_rti_info is None:
            all_orig_rti_info = orig_rti_info.copy()
        else:
            all_orig_rti_info = pd.concat([all_orig_rti_info,orig_rti_info],ignore_index=True)

    return all_orig_rti_info

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

class MyColors(object):
    def __init__(self,scale,my_cmap='jet',truncate_cmap=None):
#        cmap        = matplotlib.cm.jet
#        cmap        = truncate_colormap(cmap,0,0.9)
#        norm        = matplotlib.colors.Normalize(vmin=scale[0],vmax=scale[1])

        cmap        = matplotlib.cm.get_cmap(my_cmap)
        if truncate_cmap is not None:
            cmap        = truncate_colormap(cmap,truncate_cmap[0],truncate_cmap[1])
        norm        = matplotlib.colors.Normalize(vmin=scale[0],vmax=scale[1])

        self.scale  = scale
        self.norm   = norm
        self.cmap   = cmap
        self.sm     = matplotlib.cm.ScalarMappable(norm=self.norm,cmap=self.cmap)

    def to_rgba(self,x):
        return self.sm.to_rgba(x)

    def create_mappable(self,ax=None):
        if ax is None:
            ax = plt.gca()
        return ax.scatter(0,0,c=1,s=1.e-9,cmap=self.cmap,norm=self.norm)

def plot_colorbars(axs):
    for ax_dct in axs:
        ax          = ax_dct.get('ax')
        pcoll       = ax_dct.get('mappable')
        cbar_label  = ax_dct.get('cbar_label')
        cbar_ticks  = ax_dct.get('cbar_ticks')

        box         = ax.get_position()
        axColor     = plt.axes([(box.x0 + box.width) * 1.04 , box.y0, 0.025, box.height])
        cbar        = plt.colorbar(pcoll,orientation='vertical',cax=axColor)
        cbar.set_label(cbar_label)
        if cbar_ticks is not None:
            cbar.set_ticks(cbar_ticks)

        labels = cbar.ax.get_yticklabels()
#        labels[-1].set_visible(False)

def spectral_plot(data_dict,output_dir='output',subtract_mean=False,plot_all_spect_mean=False,
        color_key='orig_rti_cnt',filename='spectral_plot.png',one_subplot=False,legend=True,
        my_cmap='jet',truncate_cmap=(0,0.9),plot_sup_title=True,user_title=None,save_pdf=False,
        highlight=None):

    rc  = matplotlib.rcParams
    rc['font.weight']       = 'bold'
    rc['axes.labelweight']  = 'bold'
    rc['axes.titleweight']  = 'bold'

    categs              = data_dict['categs']
    mstid_list          = data_dict['mstid_list']
    data_path           = data_dict['data_path']
    all_spect_df        = data_dict['all_spect_df']
    all_spect_mean      = np.mean(all_spect_df,axis=1)

    # Setup Colormap ###############################################################
    if color_key == 'orig_rti_cnt':
        all_orig_rti_info   = create_all_orig_rti_info(data_dict)
        scale_0             = 0.
        scale_1             = all_orig_rti_info[color_key].quantile(0.85)
        my_colors           = MyColors((scale_0, scale_1),my_cmap=my_cmap,truncate_cmap=truncate_cmap)

        for categ in categs:
            dct                 = data_dict[categ]
            dct['color_series'] = dct['orig_rti_info'][color_key]
            dct['my_colors']    = my_colors
            dct['cbar_label']   = color_key

    elif color_key == 'spectral_sort':
        for categ in categs:
            dct                 = data_dict[categ]
            dct['color_series'] = dct['sort_df']['order']
            scale_0             = dct['color_series'].min()
            scale_1             = dct['color_series'].max()
            dct['my_colors']    = MyColors((scale_0, scale_1),my_cmap=my_cmap,truncate_cmap=truncate_cmap)
            dct['cbar_label']   = 'Spectral Rank\n{}'.format(dct['sort_key'])

    elif color_key == 'meanSubIntSpect_by_rtiCnt':
        for categ in categs:
            dct                 = data_dict[categ]
            dct['color_series'] = dct['sort_df'][color_key]
            scale_0             = -0.025
            scale_1             =  0.025
            dct['my_colors']    =  MyColors((scale_0, scale_1),my_cmap=my_cmap,truncate_cmap=truncate_cmap)
            dct['cbar_label']   = 'SuperDARN MSTID Index'

    # Do some plotting!! 
    filepath    = os.path.join(output_dir,filename)

    if one_subplot:
        figsize=(8.,6.)
    else:
        figsize=(10,4.0*len(categs))

    fig             = plt.figure(figsize=figsize)
    axs             = []
    event_titles    = [] # Used for one_subplot == True

    update_leg_dict = {}
    update_leg_dict['mstid']    = {'name': 'MSTID Active', 'color':'red'}
    update_leg_dict['quiet']    = {'name': 'MSTID Quiet',  'color':'blue'}

    for categ_inx,categ in enumerate(categs):
        dct         = data_dict[categ]
        my_colors   = dct['my_colors']

        if one_subplot:
            ax      = fig.add_subplot(111)
        else:
            ax      = fig.add_subplot(len(categs),1,categ_inx+1)


        rad_se_times    = data_dict[categ]['radar_sTime_eTime']
#        if highlight is not None:
#            sTime_dct       = {}
#            for key,val in rad_se_times.iteritems():
#                sTime_dct[val[1]]   = key

        for series_inx,series in list(data_dict[categ]['spect_df'].items()):
            xvals   = series.index * 1e3
            yvals   = series
            if subtract_mean:
                yvals = yvals - all_spect_mean

            color   = data_dict[categ]['color_series'][series_inx]
            rgba    = my_colors.to_rgba(color)
            line    = ax.plot(xvals,yvals,color=rgba,alpha=0.20)

            sTime   = rad_se_times[series_inx][1]
            if highlight is not None:
                if sTime in list(highlight.keys()):
                    line    = ax.plot(xvals,yvals,color=rgba,alpha=1.,zorder=10,lw=2.5)

                    tf      = yvals.index >= 0
                    yv_pos  = yvals[tf]

                    label   = highlight[sTime]['label']
                    xy      = (yv_pos.argmax()*1.e3,yv_pos.max())
                    xytext  = (1.0*xy[0],xy[1]+1.)
                    ax.annotate(label,xy=xy,xytext=xytext,
                            ha='center',zorder=20)

        if categ_inx == 0 or not one_subplot:

            if plot_all_spect_mean:
                xvals       = all_spect_mean.index * 1e3
                yvals       = all_spect_mean
                mean_line   = ax.plot(xvals,yvals,color='k',lw=2,zorder=50,ls='--',
                        label='Mean Spectrum')
            else:
                mean_line = None

            if legend and (my_cmap != 'seismic'):
                ax.legend(loc='upper right',prop={'size':'small'})

            # Configure x-axis tick locations and labels.
            xmin, xmax  = 0,2.
            ax.set_xlim(xmin,xmax)
            delt        = 0.25
            nr_tk       = round((xmax-xmin)/delt)+1
            xts         = np.linspace(xmin,xmax,nr_tk)
            ax.set_xticks(xts)
    #        ax.set_xlim(0,xvals.max())

            xts     = ax.get_xticks()
            xtls    = []
            for xt in xts:
                per = (1./(xt/1e3)) / 60.
                
                xtl = '{:.1f}\n{:.0f}'.format(xt,per)
                xtls.append(xtl)

            ax.xaxis.set_ticklabels(xtls)
            ax.grid()

            ax.set_xlabel('Frequency [mHz]\nPeriod [min]')
            ax.set_ylabel('Integrated PSD')

        event_title = '{} ({!s} Events)'.format(categ.upper(),series_inx+1)
        update_leg_dict[categ]['events'] = '({!s} Events)'.format(series_inx+1)
        event_titles.append(event_title)

        if (not one_subplot) or (one_subplot and categ_inx == len(categs)-1):
            if user_title is None:
                title   = []
                if one_subplot:
                    event_title = ' and '.join(event_titles)

                if subtract_mean:
                    event_title = 'Mean Subtracted: {}'.format(event_title)

                title.append(event_title)
                ax.set_title('\n'.join(title))
            else:
                ax.set_title(user_title)

            ax_dct                  = {}
            ax_dct['ax']            = ax
            ax_dct['mappable']      = my_colors.create_mappable(ax)
            ax_dct['cbar_label']    = dct['cbar_label']
            axs.append(ax_dct)

    if legend and my_cmap == 'seismic':
        mstid_quiet_legend(update_leg_dict,mean_line=mean_line)

    fig.tight_layout()

    plot_colorbars(axs)

    if plot_sup_title:
        sup_title = []
        txt = 'List: {!s}'.format(mstid_list)
        sup_title.append(txt)
        txt = 'Source: {!s}'.format(data_path)
        sup_title.append(txt)
        fig.text(0.5,1.005,'\n'.join(sup_title),ha='center',fontsize='large',fontweight='bold')

    fig.savefig(filepath,bbox_inches='tight')
    if save_pdf:
        fig.savefig(filepath[:-3]+'pdf',bbox_inches='tight')
    plt.close(fig)

    return [filepath]

def mstid_quiet_legend(update_leg_dict=None,fig=None,
        loc='upper right',prop={'size':11,'weight':'bold'},title=None,bbox_to_anchor=None,
        ncol=1,mean_line=None):
    from matplotlib import pyplot as plt
    import matplotlib.patches as mpatches

    leg_dict       = {}
    leg_dict['mstid']   = {'name': 'MSTID Active', 'color':'red'}
    leg_dict['quiet']   = {'name': 'MSTID Quiet',  'color':'blue'}
    leg_dict['mean']    = {'name': 'Mean Spectrum',  'color':'k'}

    if update_leg_dict is not None:
        leg_dict.update(update_leg_dict)

    leg_keys    = []
    leg_keys.append('mstid')
    leg_keys.append('quiet')
#    leg_keys.append('mean')

    if fig is None: fig = plt.gcf() 
    ax  = fig.gca()

    handles = []
    labels  = []
    for leg_key in leg_keys:
        color = leg_dict[leg_key]['color']
        label = leg_dict[leg_key]['name']

        events = leg_dict[leg_key].get('events')
        if events is not None:
            label = ' '.join([label,events])
        handles.append(mpatches.Patch(color=color,label=label))
        labels.append(label)

    if mean_line is not None:
        handles.append(mean_line[0])
        labels.append(mean_line[0].get_label())

    fig_tmp = plt.figure()
    ax_tmp  = fig_tmp.add_subplot(111)
    ax_tmp.set_visible(False)

    legend = ax.legend(handles,labels,ncol=ncol,loc=loc,prop=prop,title=title,bbox_to_anchor=bbox_to_anchor,scatterpoints=1)
    plt.figure(fig.number) # Set current figure back to the real thing.

    return legend

def this_actually_does_the_sorting(spect_df,orig_rti_info,all_spect_mean,sort_key):
    """
    Sorts all data windows in a season by a particular spectrum parameter.

    Input Arguments:
        spect_df [shape: (number of spectral bins, number of data windows)]:
            Spectrum of each data window integrated over all radar cells, but still a function
            of spectral bin. spect_df contains only the spectral curvers from a single category
            (mstid or quiet)

        orig_rti_info [shape: (number of data windows, 6 columns)]:
            Information about the original RTI array.
                Columns:
                orig_rti_cnt: Number of radar cells in window with a backscatter measurement.
                orig_rti_fraction: orig_rti_cnt/orig_rti_possible
                orig_rti_mean:   Mean of all measured values in the data window.
                orig_rti_median: Median of all measured values in the data window.
                orig_rti_possible: Total number of possible radar cells in the data window.
                orig_rti_std: Standard deviation of all measured values in the data window.

            all_spect_mean [shape: (number of spectral bins]:
                Average seasonal power spectral density curve calculated from all data windows in
                both the mstid and quiet categories of a given radar in a given season.

            sort_key: Type of spectrum to sort by. Options are:
                intSpect_by_rtiCnt (You probably want this one. It is the one
                                    Frissell et al. (2016, https://doi.org/10.1002/2015JA022168) 
                                    sorted on and called the `MSTID Index`. See his Section 2.2.)
                meanSubIntSpect
                meanSub_spect_df
                intSpect

    meanSubIntSpect_by_rtiCnt [shape: (# data windows)]:
                

    Computed Parameters:
    intSpect [shape: (# data windows)]:
        Power Spectral Density integrated over all radar cells and spectral bins in a data window.

    meanSub_spect_df [shape: (number of spectral bins, number of data windows)]: 
        spect_df - all_spect_mean

    meanSubIntSpect [shape: (# data windows)]:
        Data window PSD curve minus radar seasonal mean PSD curve integrated over
        all spectral bins in a data window.
        
    intSpect_by_rtiCnt [shape: (# data windows)]:
       intSpect normalized by number of radar cells observing backscatter in the data window 

    meanSubIntSpect_by_rtiCnt [shape: (# data windows)]:
       meanSubIntSpect normalized by number of radar cells observing backscatter in the data window 
    """

    intSpect                = np.sum(spect_df[spect_df.index >= 0], axis=0)

    meanSub_spect_df        = spect_df.subtract(all_spect_mean,axis='index')
    meanSubIntSpect         = np.sum(meanSub_spect_df[meanSub_spect_df.index >= 0], axis=0)

    sort_df                 = pd.DataFrame({'intSpect':intSpect,'meanSubIntSpect':meanSubIntSpect})

    keys                    = ['intSpect','meanSubIntSpect']
    for key in keys:
        col_name            = '{}_by_rtiCnt'.format(key)
        sort_df[col_name]   = sort_df[key]/orig_rti_info['orig_rti_cnt']

    sort_df.sort_values(sort_key,ascending=True,inplace=True)
    sort_df['order'] = list(range(len(sort_df)))
    return sort_df 

def sort_by_spectrum(data_dict,sort_key):
    """
    Sort the spectra in a data_dict generated by load_data_dict() according to
    some parameter based on the spectra.  A good choice seems to be:

        sort_key            = 'meanSubIntSpect_by_rtiCnt'

    Which is the mean subtracted integrated spectrum divided by the number of
    ground scatter points observed.
    """
    categs          = data_dict['categs']
    all_spect_df    = data_dict['all_spect_df']
    all_spect_mean  = np.mean(all_spect_df,axis=1)

    for categ in categs:
        spect_df        = data_dict[categ]['spect_df']
        orig_rti_info   = data_dict[categ]['orig_rti_info']

        data_dict[categ]['sort_key']    = sort_key
        data_dict[categ]['sort_df']     = this_actually_does_the_sorting(spect_df,orig_rti_info,all_spect_mean,sort_key)

#    data_dict['all_spect_sort'] = this_actually_does_the_sorting(all_spect_df,orig_rti_info,all_spect_mean)
    return data_dict

def classify_mstid_events(data_dict,threshold=0.,read_only=False):
    """
    Classify good events as either MSTID or quiet using a sorted dictionary
    provided by sort_by_spectrum().

    read_only:  Only classify; don't update the mongo db.
    """

    if not read_only:
        db_name         = data_dict.get('db_name','mstid')
        mongo_port      = data_dict.get('mongo_port',27017)

        mongo           = pymongo.MongoClient(port=mongo_port)
        db              = mongo[db_name]

    mstid_list      = data_dict['mstid_list']
    unc_dct         = data_dict['unclassified']
    
    for categ in ['mstid','quiet']:
        key_0s = ['radar_sTime_eTime', 'orig_rti_info', 'spect_df', 'sort_df']
        for key_0 in key_0s:
            data_dict[categ][key_0] = copy.copy(unc_dct[key_0])

#        data_dict[categ]['radar_sTime_eTime']   = {}
#        data_dict[categ]['orig_rti_info']       = unc_dct['orig_rti_info'] 
#        data_dict[categ]['spect_df']            = []
#        data_dict[categ]['sort_df']             = []
        data_dict[categ]['sort_key']            = data_dict['unclassified']['sort_key']

    event_inxs      = list(data_dict['unclassified']['radar_sTime_eTime'].keys())
    event_inxs.sort()

    for event_inx in event_inxs:
        if unc_dct['sort_df']['meanSubIntSpect_by_rtiCnt'][event_inx] < threshold:
            categ       = 'quiet'
            del_from    = 'mstid'
        else:
            categ       = 'mstid'
            del_from    = 'quiet'
        
        del data_dict[del_from]['radar_sTime_eTime'][event_inx]

        for key_0 in ['orig_rti_info','sort_df']:
            this_df     = data_dict[del_from][key_0]
            this_df.drop(event_inx,axis=0,inplace=True)

        this_df = data_dict[del_from]['spect_df']
        this_df.drop(event_inx,axis=1,inplace=True)

        if not read_only:
            # Update mongoDb.
            radar,sDatetime,fDatetime = data_dict[categ]['radar_sTime_eTime'][event_inx]
            item    = db[mstid_list].find_one({'radar':radar,'sDatetime':sDatetime,'fDatetime':fDatetime})
            _id     = item['_id']
            
            unset_keys = ['intpsd_mean', 'intpsd_max', 'intpsd_sum']
            for unset_key in unset_keys:
                if unset_key in item:
                    status = db[mstid_list].update_one({'_id':_id},{'$unset':{unset_key:1}})

            sort_df = data_dict[categ]['sort_df']
            info        = sort_df[sort_df.index == event_inx]
            info_keys   = ['intSpect','meanSubIntSpect','intSpect_by_rtiCnt','meanSubIntSpect_by_rtiCnt']
            for info_key in info_keys:
                val     = float(info[info_key])
                status  = db[mstid_list].update_one({'_id':_id},{'$set':{info_key:val}})

            status  = db[mstid_list].update_one({'_id':item['_id']},{'$set': {'category_manu':categ}})

    if not read_only: mongo.close()

    data_dict['categs'] = ['mstid','quiet']
    return data_dict
