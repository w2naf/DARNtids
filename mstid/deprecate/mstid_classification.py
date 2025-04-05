# Proposal for classification logic.
# 1. Take complete list; run good/bad logic.
# 2. Use spectral classifciation to determine MSTID or not.
# 3. Test this on at least 2 radars.
# 4. Try to rank MSTIDs comparatively between 2 radars.

import os
import copy
from hdf5_api import loadMusicArrayFromHDF5, saveDictToHDF5, extractDataFromHDF5
import datetime
import h5py

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import inspect
import logging
import logging.config
curr_file           = inspect.getfile(inspect.currentframe()) # script filename (usually with path)
logging.filename    = curr_file+'.log'
logging.config.fileConfig("logging.conf")
log                 = logging.getLogger("root")

import numpy as np
import pandas as pd
from scipy import stats

import ephem # pip install pyephem (on Python 2)
             # pip install ephem   (on Python 3)

import pymongo

# import davitpy
# import davitpy.pydarn.proc.music as music
import handling
# from . import mongo_tools
import music_support as msc
import rti_checker_good_bad as rcgb
import rti_checker_spect_sort as rcss

mongo   = pymongo.MongoClient()
db      = mongo.mstid

def classify_none_events(mstid_list,rti_fraction_threshold=0.675,terminator_fraction_threshold=0.):
    # Clear out any old classifications.
    crsr            = db[mstid_list].find()
    for event in crsr:
        _id = event['_id']
        delete_list = ['category_auto','category_manu']
        for item in delete_list:
            if item in event:
                status = db[mstid_list].update({'_id':_id},{'$unset': {item: 1}})

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
            entry_id = db[mstid_list].update({'_id':item['_id']}, {'$set':{'category_manu': 'None', 'reject_message':reject_message}})

    print('{bc:d} events marked as None.'.format(bc=bad_counter))
    print('no_data: {!s}'.format(no_data))
    print('bad_period: {!s}'.format(bad_period))
    print('no_orig_rti_fract: {!s}'.format(no_orig_rti_fract))
    print('low_orig_rti_fract: {!s}'.format(low_orig_rti_fract))
    print('low_termin_fract: {!s}'.format(low_termin_fract))

def load_data_dict(mstid_list,data_source,use_cache=True,cache_dir='data',test_mode=False):
    cache_name  = os.path.join(cache_dir,'classify_{}.{}.h5'.format(mstid_list,os.path.basename(data_source)))

    if not os.path.exists(cache_name) or not use_cache:
        data_dict   = {'unclassified':{'color':'blue'},
                       'mstid': {'color':'red'},
                       'quiet': {'color':'green'}}

        categs                  = ['unclassified']

        data_dict['categs']         = categs
        data_dict['mstid_list']     = mstid_list
        data_dict['data_source']    = data_source

        for categ in categs:
            if categ == 'unclassified':
                crsr        = db[mstid_list].find({'category_manu':{'$exists':False}})
            else:
                crsr        = db[mstid_list].find({'category_manu':categ})

            orig_rti_inx    = []
            orig_rti_list   = []

            for item_inx,item in enumerate(crsr):
                radar       = item['radar']
                sDatetime   = item['sDatetime']
                fDatetime   = item['fDatetime']

                hdf5Path  = msc.get_hdf5_name(radar,sDatetime,fDatetime,data_path=data_source,getPath=True)
                if os.path.exists(hdf5Path):
                    dataObj = loadMusicArrayFromHDF5(hdf5Path)
        
                if not hasattr(dataObj.active,'spectrum'):
                    continue

                orig_rti_info   = get_orig_rti_info(dataObj,sDatetime,fDatetime)

                fvec        = dataObj.active.freqVec
                spec        = np.abs(dataObj.active.spectrum)
                spec        = np.nansum(spec,axis=2)
                spec        = np.nansum(spec,axis=1)

                series      = pd.Series(spec,fvec)

                spect_df    = data_dict[categ].get('spect_df')
                if spect_df is None:
                    data_dict[categ]['spect_df']        = pd.DataFrame(series)
                    orig_rti_inx.append(0)
                    
                    data_dict[categ]['radar_sTime_eTime']     = {}
                    data_dict[categ]['radar_sTime_eTime'][0]  = (radar,sDatetime,fDatetime)
                else:
                    series.name                     = list(spect_df.keys()).max() + 1
                    data_dict[categ]['spect_df']    = spect_df.join(series,how='outer')
                    data_dict[categ]['radar_sTime_eTime'][series.name]  = (radar,sDatetime,fDatetime)
                    orig_rti_inx.append(series.name)

                orig_rti_list.append(orig_rti_info)

                if test_mode and item_inx == 5:
                    break

            data_dict[categ]['orig_rti_info'] = pd.DataFrame(orig_rti_list,index=orig_rti_inx)

        # Fill in NaNs and put everything onto same frequency grid.
        f_ext   = []
        for categ_inx,categ in enumerate(categs):
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
        with h5py.File(cache_name, 'w') as fl:
            saveDictToHDF5(fl, data_dict)
    else:
        print("I'm using the cache! ({})".format(cache_name))
        with h5py.File(cache_name, 'r') as fl:
            data_dict = extractDataFromHDF5(fl)

    data_dict['all_spect_df'] = create_all_spect_df(data_dict)
    return data_dict

def create_all_spect_df(data_dict):
    categs          = data_dict['categs']
    # Create a new, combined data frame that will let us calculate the average
    # spectrum.
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
    def __init__(self,scale):
        cmap        = matplotlib.cm.jet
        cmap        = truncate_colormap(cmap,0,0.9)

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
        return ax.scatter(0,0,c=1,cmap=self.cmap,norm=self.norm)

def plot_colorbars(axs):
    for ax_dct in axs:
        ax          = ax_dct.get('ax')
        pcoll       = ax_dct.get('mappable')
        cbar_label  = ax_dct.get('cbar_label')
        cbar_ticks  = ax_dct.get('cbar_ticks')

        box         = ax.get_position()
        axColor     = plt.axes([(box.x0 + box.width) * 1.04 , box.y0, 0.01, box.height])
        cbar        = plt.colorbar(pcoll,orientation='vertical',cax=axColor)
        cbar.set_label(cbar_label)
        if cbar_ticks is not None:
            cbar.set_ticks(cbar_ticks)

        labels = cbar.ax.get_yticklabels()
#        labels[-1].set_visible(False)

def spectral_plot(data_dict,output_dir='output',subtract_mean=False,plot_all_spect_mean=False,
        color_key='orig_rti_cnt',filename='spectral_plot.png'):
    categs              = data_dict['categs']
    mstid_list          = data_dict['mstid_list']
    data_source         = data_dict['data_source']
    all_spect_df        = data_dict['all_spect_df']
    all_spect_mean      = np.mean(all_spect_df,axis=1)


    # Setup Colormap ###############################################################
    if color_key == 'orig_rti_cnt':
        all_orig_rti_info   = create_all_orig_rti_info(data_dict)
        scale_0             = 0.
        scale_1             = all_orig_rti_info[color_key].quantile(0.85)
        my_colors           = MyColors((scale_0, scale_1))

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
            dct['my_colors']    = MyColors((scale_0, scale_1))
            dct['cbar_label']   = 'Spectral Rank\n{}'.format(dct['sort_key'])

    # Do some plotting!! 
    filepath    = os.path.join(output_dir,filename)

    fig         = plt.figure(figsize=(10,4.0*len(categs)))
    axs         = []
    for categ_inx,categ in enumerate(categs):
        dct         = data_dict[categ]
        my_colors   = dct['my_colors']

        ax      = fig.add_subplot(len(categs),1,categ_inx+1)
#        color   = data_dict[categ]['color']
        for series_inx,series in data_dict[categ]['spect_df'].items():
            xvals   = series.index * 1e3
            yvals   = series
            if subtract_mean:
                yvals = yvals - all_spect_mean

            color   = data_dict[categ]['color_series'][series_inx]
            rgba    = my_colors.to_rgba(color)
            ax.plot(xvals,yvals,color=rgba)

        if plot_all_spect_mean:
            xvals   = all_spect_mean.index * 1e3
            yvals   = all_spect_mean
            ax.plot(xvals,yvals,color='k',lw=2)

        ax.set_xlabel('Frequency [mHz] / Period [min]')
        ax.set_ylabel('Integrated PSD')
        title   = []
        txt     = '{} ({!s} Events)'.format(categ.upper(),series_inx+1)
        if subtract_mean:
            txt = 'Mean Subrtacted: {}'.format(txt)
        title.append(txt)
        ax.set_title('\n'.join(title))

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

        ax_dct                  = {}
        ax_dct['ax']            = ax
        ax_dct['mappable']      = my_colors.create_mappable(ax)
        ax_dct['cbar_label']    = dct['cbar_label']
        axs.append(ax_dct)

#    ax.legend()
    fig.tight_layout()

    plot_colorbars(axs)

    sup_title = []
    txt = 'List: {!s}'.format(mstid_list)
    sup_title.append(txt)
    txt = 'Source: {!s}'.format(data_source)
    sup_title.append(txt)
    fig.text(0.5,1.005,'\n'.join(sup_title),ha='center',fontsize='large',fontweight='bold')

    fig.savefig(filepath,bbox_inches='tight')
    plt.close(fig)

    return [filepath]

def this_actually_does_the_sorting(spect_df,orig_rti_info,all_spect_mean,sort_key):
    intSpect                = np.sum(spect_df[spect_df.index >= 0], axis=0)

    meanSub_spect_df        = spect_df.subtract(all_spect_mean,axis='index')
    meanSubIntSpect         = np.sum(meanSub_spect_df[meanSub_spect_df.index >= 0], axis=0)

    sort_df                 = pd.DataFrame({'intSpect':intSpect,'meanSubIntSpect':meanSubIntSpect})

    keys                    = ['intSpect','meanSubIntSpect']
    for key in keys:
        col_name            = '{}_by_rtiCnt'.format(key)
        sort_df[col_name]   = sort_df[key]/orig_rti_info['orig_rti_cnt']

    sort_df.sort(sort_key,ascending=True,inplace=True)
    sort_df['order'] = list(range(len(sort_df)))
    return sort_df 

def sort_by_spectrum(data_dict,sort_key):
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

def classify_mstid_events(data_dict,threshold=0.):
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

        # Update mongoDb.
        radar,sDatetime,fDatetime = data_dict[categ]['radar_sTime_eTime'][event_inx]
        item    = db[mstid_list].find_one({'radar':radar,'sDatetime':sDatetime,'fDatetime':fDatetime})
        _id     = item['_id']
        
        unset_keys = ['intpsd_mean', 'intpsd_max', 'intpsd_sum']
        for unset_key in unset_keys:
            if unset_key in item:
                status = db[mstid_list].update({'_id':_id},{'$unset':{unset_key:1}})

        sort_df = data_dict[categ]['sort_df']
        info        = sort_df[sort_df.index == event_inx]
        info_keys   = ['intSpect','meanSubIntSpect','intSpect_by_rtiCnt','meanSubIntSpect_by_rtiCnt']
        for info_key in info_keys:
            val     = np.float(info[info_key])
            status  = db[mstid_list].update({'_id':_id},{'$set':{info_key:val}})

        status  = db[mstid_list].update({'_id':item['_id']},{'$set': {'category_manu':categ}})

    data_dict['categs'] = ['mstid','quiet']
    return data_dict

def main(mstid_list,data_source,radar,sDate,eDate,sort_key='meanSubIntSpect_by_rtiCnt',generate_new_list=True,**kwargs):
    # Create initial list.
    # Sort into candiate/discard.

    output_dir  = os.path.join('output',mstid_list,os.path.basename(data_source))
    handling.prepare_output_dirs({0:output_dir},clear_output_dirs=True)

    if generate_new_list:
        generate_mongo_list(mstid_list,radar,sDate,eDate)
        load_diagnostic_data(mstid_list,data_source)

        classify_none_events(mstid_list)
        rcgb.main(mstid_list,data_source)
    data_dict   = load_data_dict(mstid_list,data_source,**kwargs)
    data_dict   = sort_by_spectrum(data_dict,sort_key)
    data_dict   = classify_mstid_events(data_dict)

    rcss.main(data_dict)
    
    plot_nr     = 1
    filename    = '{:03d}_spectral_plot.png'.format(plot_nr)
    spectral_plot(data_dict,output_dir=output_dir,plot_all_spect_mean=True,filename=filename)

    plot_nr     += 1
    filename    = '{:03d}_spectral_plot_mean_subtracted.png'.format(plot_nr)
    spectral_plot(data_dict,output_dir=output_dir,filename=filename,subtract_mean=True)

    plot_nr     += 1
    filename    = '{:03d}_spectral_plot_mean_subtracted_ranked.png'.format(plot_nr)
    spectral_plot(data_dict,output_dir=output_dir,subtract_mean=True,color_key='spectral_sort',filename=filename)
