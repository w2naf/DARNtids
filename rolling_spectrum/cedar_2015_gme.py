#!/usr/bin/env python
# This is a very much hacked-up version of the rolling_spectrum script
# in order to get a nice comparison between the solar wind/GME parameters
# of one MSTID event with quiet geomagnetic conditions and one quiet event
# disturbed geomagnetic conditions.
#
# This was used for the CEDAR 2015 poster.

import sys
sys.path.append('/data/mypthon')

import os
import shutil
import datetime
import copy
import h5py
from hdf5_api import saveDictToHDF5, extractDataFromHDF5

import numpy as np
import scipy as sp
import pandas as pd

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.legend_handler import HandlerLine2D
font    = {'weight': 'bold', 'size': 12}
matplotlib.rc('font',**font)

import davitpy
import rolling_spect_lib as rsl 
import handling

class EmptyObj(object):
    pass

################################################################################
# Radar Dictionary #############################################################
radar_dict  = {}

default_beam = 7
radars = ['gbr','pgr','sas','kap','ksr','kod','sto','pyk','han']
for radar in radars:
    tmp                 = {}
    tmp['beam']         = default_beam
    radar_dict[radar]   = tmp

tmp                 = {}
tmp['beam']         = 7
tmp['rnge_lim']     = [10,35]
tmp['contrast']     = 1.15
radar_dict['bks']   = tmp

tmp                 = {}
tmp['beam']         = 2
tmp['rnge_lim']     = [ 15,35]
tmp['contrast']     = 1.15
radar_dict['fhe']   = tmp

tmp                 = {}
tmp['beam']         = 22
tmp['rnge_lim']     = [ 15,35]
tmp['contrast']     = 1.15
radar_dict['fhw']   = tmp

tmp                 = {}
tmp['beam']         = 18
tmp['rnge_lim']     = [15,40]
tmp['contrast']     = 0.65
radar_dict['cvw']   = tmp

tmp                 = {}
tmp['beam']         = 0
tmp['rnge_lim']     = [15,40]
tmp['contrast']     = 0.65
radar_dict['cve']   = tmp

tmp = {}
tmp['beam']         = 7
tmp['rnge_lim']     = [10,35]
radar_dict['wal']   = tmp

def plot_superposed_gme(gme_objs,run_times,time_prior,ax,prop_delay=datetime.timedelta(0),parameter=None,label=False):
    ylabel = str(parameter)
    nr_events = 0
    for st,et in run_times:
        for gme_obj in gme_objs:
            if parameter is None:
                series  = gme_obj.ind_df['ind_0_processed']
                ylabel  = gme_obj.plot_info['ind_0_gme_label']
                short_lbl = ylabel
            elif parameter == 'ind_0_var':
                series  = gme_obj.ind_df['ind_0_var']
                ylabel  = 'var({})'.format(gme_obj.plot_info['ind_0_gme_label'])
                short_lbl = ylabel
            elif parameter == 'ae':
                series  = gme_obj.omni_df_raw['ae']
                ylabel  = 'AE'
                short_lbl = ylabel
            elif parameter == 'symh':
                series  = gme_obj.omni_df_raw['symh']
                ylabel  = 'SYM-H'
                short_lbl = ylabel
            elif parameter == 'bx':
                series  = gme_obj.omni_df_raw['bx']
                ylabel  = 'OMNI Bx GSM [nT]'
                short_lbl = 'Bx'
            elif parameter == 'by':
                series  = gme_obj.omni_df_raw['bym']
                ylabel  = 'OMNI By GSM [nT]'
                short_lbl = 'By'
            elif parameter == 'bz':
                series  = gme_obj.omni_df_raw['bzm']
                ylabel  = 'OMNI Bz GSM [nT]'
                short_lbl = 'Bz'
            elif parameter == 'bMagAvg':
                series  = gme_obj.omni_df_raw['bMagAvg']
                ylabel  = 'OMNI |B| [nT]'
                short_lbl = '|B|'
            elif parameter == 'pDyn':
                series  = gme_obj.omni_df_raw['pDyn']
                ylabel  = 'OMNI pDyn [nPa]'
                short_lbl = 'pDyn'
            elif parameter == 'flowSpeed':
                series  = gme_obj.omni_df_raw['flowSpeed']
                ylabel  = 'OMNI Flow Speed [km/s]'
                short_lbl = 'v'
            elif parameter == 'np':
                series  = gme_obj.omni_df_raw['np']
                ylabel  = 'OMNI Np [/cc3]'
                short_lbl = 'Np'
            elif parameter == 'temp':
                series  = gme_obj.omni_df_raw['temp']
                ylabel  = 'OMNI T [K]'
                short_lbl = 'T'

            gme_st      = st-prop_delay
            gme_et      = et-prop_delay
            tf          = np.logical_and(series.index >= gme_st, series.index < gme_et)
            if np.count_nonzero(tf) == 0: continue

            gme_deltas  = series.index[tf] -gme_st -time_prior 
            xvals       = np.array([x.total_seconds() for x in gme_deltas])
            yvals       = series[tf]
            if label:
                this_label = st.strftime('%d %b %Y')
            else:
                this_label = None

            if nr_events == 0:
                color = '#660000'
            elif nr_events == 1:
                color = '#ff6600'
            else:
                color = None
            ax.plot(xvals,yvals,label=this_label,lw=2,color=color)
            nr_events += 1
            break
    nr_label    = '({!s} Events)'.format(nr_events)
#    ylabel      = '\n'.join([ylabel,nr_label])
    ax.set_ylabel(ylabel)

    ax.text(0.01,0.85,short_lbl,transform=ax.transAxes,fontsize=18,fontweight='bold')

    try:
        if parameter is None and gme_obj.gme_param == 'omni_bz_var':
            ax.set_ylim(0.,20.)

        if parameter is None and gme_obj.gme_param == 'omni_b_var':
            ax.set_ylim(0.,10.)

        if parameter is None and gme_obj.gme_param == 'omni_by_var':
            ax.set_ylim(0.,20.)
    except:
        pass

#    if parameter is None:
#        ax.set_ylim(0.,5.)

def plot_superposed_bcomps(gme_objs,run_times,time_prior,ax,prop_delay=datetime.timedelta(0)):
    keys        = ['bx','bym','bzm','bMagAvg']
    plot_dict   = {}
    for key in keys:
        plot_dict[key] = {}

    separation  = 30.

    offset      = 0.
    plot_dict['bx']['color']    = 'r'
    plot_dict['bx']['offset']   = offset
    plot_dict['bx']['label']    = 'Bx'

    offset      += separation
    plot_dict['bym']['color']   = 'g'
    plot_dict['bym']['offset']  = offset
    plot_dict['bym']['label']   = 'By'

    offset      += separation
    plot_dict['bzm']['color']   = 'b'
    plot_dict['bzm']['offset']  = offset
    plot_dict['bzm']['label']   = 'Bz'

    offset      += separation/2.
    plot_dict['bMagAvg']['color']   = '0.5'
    plot_dict['bMagAvg']['offset']  = offset
    plot_dict['bMagAvg']['label']   = '|B|'

    good = 0
    bad  = 0
    for st,et in run_times:
        for gme_obj in gme_objs:
            df          = gme_obj.omni_df

            gme_st      = st-prop_delay
            gme_et      = et-prop_delay
            tf          = np.logical_and(df.index >= gme_st, df.index < gme_et)
            if np.count_nonzero(tf) == 0: continue
            
            for key, dct in plot_dict.items():
                gme_deltas  = df.index[tf] -gme_st -time_prior 
                time        = np.array([x.total_seconds() for x in gme_deltas])

                data_vec    = np.array(gme_obj.omni_df[key][tf].tolist())
                series      = pd.Series(data_vec,time)

                data        = dct.get('data')
                if data is None:
                    dct['data'] = pd.DataFrame(series)
                else:
                    series.name = list(data.keys()).max() + 1
                    dct['data'] = data.join(series,how='outer')
            break
        
    for key in keys:
        dct     = plot_dict[key]
        data    = dct['data']
        offset  = dct['offset']
        color   = dct['color']
        label   = dct['label']
        lw      = 1.5

        for inx,series in data.items():
            xvals   = series.index
            yvals   = series + offset
            ax.plot(xvals,yvals,color=color,alpha=0.35)

        xvals       = data.index
        data_mean   = np.nanmean(data,axis=1)
        data_std    = np.nanstd(data,axis=1)

        if key == 'bMagAvg': color = 'k'
        yvals       = data_mean + offset
        ax.plot(xvals,yvals,color=color,lw=lw,label=label)
        ax.plot(xvals,yvals,color='k',lw=lw)

        yvals       = data_mean + data_std + offset
        ax.plot(xvals,yvals,color=color,lw=lw)
        ax.plot(xvals,yvals,color='k',lw=lw)

        yvals       = data_mean - data_std + offset
        ax.plot(xvals,yvals,color=color,lw=lw)
        ax.plot(xvals,yvals,color='k',lw=lw)

    ################################################################################
    yticks      = []
    yticklabels = []
    div = 3.
    for key in keys:
        offset = plot_dict[key]['offset']

        if key != 'bMagAvg':
            yticks.append(offset-separation/div)
            yticklabels.append('{:.0f}'.format(-separation/div))

        yticks.append(offset)
        yticklabels.append('0')

        yticks.append(offset+separation/div)
        yticklabels.append('{:.0f}'.format(separation/div))
    ################################################################################

    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels,fontsize='xx-small')

    ylim_min = np.min(yticks) - separation/4.
    ylim_max = np.max(yticks) + separation/2.
    ax.set_ylim( (ylim_min, ylim_max) )

    ylabel  = 'OMNI B GSM\n({!s} Events)'.format(data.shape[1])
    ax.set_ylabel(ylabel)
    ax.legend(loc='upper right',ncol=4,fontsize='xx-small',bbox_to_anchor=(0.99,1.125))

def my_plot_spect(spect_dict,ax,scale=None,db=False,cbar_label='Spectral Pwr',cbar_ticks=None,
        xlim=None,xticks=None):
    from scipy import stats
    plotZeros           = True
    max_sounding_time   = datetime.timedelta(minutes=4)

    if 'timedelta' in spect_dict:
        time_vec    = spect_dict['timedelta']
        xvec        = np.array( [x.total_seconds() for x in time_vec] )
    elif 'time' in spect_dict:
        time_vec    = spect_dict['time']
        xvec        = np.array([matplotlib.dates.date2num(x) for x in time_vec])

    freq_vec    = spect_dict['freq']*1.e3
    data_arr    = spect_dict['spect']
    ylabel      = spect_dict.get('ylabel')

    if db:
        data_arr    = 10.*np.log10(data_arr)
        cbar_label  = '{} [dB]'.format(cbar_label)

    #Set colorbar scale if not explicitly defined.
    if scale is None:
        sd          = stats.nanstd(np.abs(data_arr),axis=None)
        mean        = stats.nanmean(np.abs(data_arr),axis=None)
        scMax       = np.ceil(mean + 1.*sd)
        if np.nanmin(data_arr) < 0:
            scale   = scMax*np.array([-1.,1.])
        else:
            scale   = scMax*np.array([0.,1.])

    cmap    = matplotlib.cm.jet
    bounds  = np.linspace(scale[0],scale[1],256)
    norm    = matplotlib.colors.BoundaryNorm(bounds,cmap.N)

    verts   = []
    scan    = []
    for tm_inx in range(xvec.size-1):
        for f_inx in range(freq_vec.size-1):
            if np.isnan(data_arr[tm_inx,f_inx]): continue
            if data_arr[tm_inx,f_inx] == 0 and not plotZeros: continue

            if max_sounding_time is not None:
                if (time_vec[tm_inx+1] - time_vec[tm_inx+0]) > max_sounding_time: continue

            x1,y1 = xvec[tm_inx+0],freq_vec[f_inx+0]
            x2,y2 = xvec[tm_inx+1],freq_vec[f_inx+0]
            x3,y3 = xvec[tm_inx+1],freq_vec[f_inx+1]
            x4,y4 = xvec[tm_inx+0],freq_vec[f_inx+1]

            scan.append(data_arr[tm_inx,f_inx])
            verts.append(((x1,y1),(x2,y2),(x3,y3),(x4,y4),(x1,y1)))

    pcoll = PolyCollection(np.array(verts),edgecolors='face',linewidths=0,closed=False,cmap=cmap,norm=norm,zorder=99)
    pcoll.set_array(np.array(scan))
    ax.add_collection(pcoll,autolim=False)

    if xlim is None:
        xlim = (xvec.min(), xvec.max())

    ax.set_xlim(xlim)

    ax.set_ylim(freq_vec.min(),freq_vec.max())
    ax.set_ylabel(ylabel)

    if xticks is not None:
        ax.set_xticks(xticks)

    if 'timedelta' in spect_dict:
        xtls = format_total_second_xticklabels(ax.get_xticks())
        ax.set_xticklabels(xtls)

    if 'title' in spect_dict:
        ax.set_title(spect_dict['title'])

    yts     = ax.get_yticks()
    ytls    = []
    for yt in yts:
        per = (1./(yt/1e3)) / 60.
        
        ytl = '{:.1f}\n{:.0f}'.format(yt,per)
        ytls.append(ytl)

    ax.yaxis.set_ticklabels(ytls,rotation=90)

    cbar_info           = {}
    cbar_info['cmap']   = cmap
    cbar_info['bounds'] = bounds
    cbar_info['norm']   = norm
    cbar_info['label']  = cbar_label
    cbar_info['ticks']  = cbar_ticks
    cbar_info['mappable']  = pcoll

    return {'cbar_info': cbar_info}

def epoch_xticks(time_prior,time_post,delta=datetime.timedelta(minutes=60)):
    xticks      = []
    curr_xtk    = datetime.timedelta(0)-time_prior
    while curr_xtk <= time_post:
        xticks.append(curr_xtk.total_seconds())
        curr_xtk += delta
    xticks      = np.array(xticks)
    return xticks

def format_total_second_xticklabels(xticks):
    xtls = []
    for xtk in xticks:
        hrs  = xtk/3600.
        hr   = int(np.abs(hrs))
        mt   = int(np.round((hrs % 1) * 60))
        lbl  = '{:02d}:{:02d}'.format(hr,mt)
        if hrs < 0: lbl = '-'+lbl
        xtls.append(lbl)
    xtls = np.array(xtls)
    return xtls

def format_axis(ax,xlim=None,xticks=None,xticklabels=None,xlabel=None,axvline=None,grid=False):
    if xlim is not None:
        ax.set_xlim(xlim)
    if xticks is not None:
        ax.set_xticks(xticks)
    if xticklabels is not None:
        ax.set_xticklabels(xticklabels)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if axvline is not None:
        ax.axvline(axvline,color='g',ls='--',lw=2.)
    if grid:
        ax.grid()
    
def plot_colorbars(axs):
    for ax_dct in axs:
        ax          = ax_dct['ax']
        cbar_info   = ax_dct['cbar']
        pcoll       = cbar_info['mappable']
        cbar_label  = cbar_info['label']
        cbar_ticks  = cbar_info['ticks']

        box         = ax.get_position()
        axColor     = plt.axes([(box.x0 + box.width) * 1.04 , box.y0, 0.01, box.height])
        cbar        = plt.colorbar(pcoll,orientation='vertical',cax=axColor)
        cbar.set_label(cbar_label)
        if cbar_ticks is not None:
            cbar.set_ticks(cbar_ticks)

        labels = cbar.ax.get_yticklabels()
        labels[-1].set_visible(False)

def calc_spect_dict(sTime,eTime,radar,ds_name='fitex',output_dir='',plot=False,
        gme_objs=None,prop_delay=datetime.timedelta(0),axvline=None,
        calculate_fft=True,calculate_lsp=True):
#    ds_name = 'originalFit'
    xlim    = (sTime, eTime)
    
    beam = radar_dict[radar]['beam']
    
    date_fmt = '%Y %b %d %H%M'
    title = []
    title.append(sTime.strftime(date_fmt) + ' - ' + eTime.strftime(date_fmt))
    title.append('{}: {} B{:02d}'.format(ds_name,radar.upper(),int(beam)))
    print(': '.join(title))

    # Get data from correct radar
    load_sTime  = sTime-datetime.timedelta(hours=4)
    curr_data   = rsl.get_radar_data(load_sTime,eTime,radar,
                    use_cache = True, save = True, ds_name=ds_name)

    if curr_data is None:
        return None

    test_data   = False
    test_plot   = False
    gate_select = 'all'

    if calculate_lsp:
        curr_data.rolling_lombpd(beam=beam,gate_select=gate_select,test_data=test_data,test_plot=test_plot)
        if np.count_nonzero(np.isfinite(curr_data.lspgram_dict['spect'])) == 0: 
            return None

    if calculate_fft:
        curr_data.rolling_fft(   beam=beam,gate_select=gate_select,test_data=test_data,test_plot=test_plot)
        if np.count_nonzero(np.isfinite(curr_data.fft_dict['spect'])) == 0:
            return None

    if plot:
        ny_ax   = 1 
        if calculate_fft: ny_ax += 1
        if calculate_lsp: ny_ax += 1

        nx_ax   = 1
        ax_inx  = 0

        figsize = (8,3*ny_ax)
        fig = plt.figure(figsize=figsize)
        #####
        gme_objs = None
        if gme_objs is not None:
            ax_inx      += 1
            ax          = fig.add_subplot(ny_ax,nx_ax,ax_inx)

            for gme_obj in gme_objs:
                gme_st  = sTime-prop_delay
                gme_et  = eTime-prop_delay
                tf      = np.logical_and(gme_obj.ind_df.index >= gme_st, gme_obj.ind_df.index < gme_et)
                if np.count_nonzero(tf) == 0: continue

#                gme_deltas  = gme_obj.ind_df.index[tf] -gme_st -time_prior 
#                xvals       = np.array([x.total_seconds() for x in gme_deltas])
                xvals       = gme_obj.ind_df.index[tf]
                yvals       = gme_obj.ind_df['ind_0_processed'][tf]
                ax.plot(xvals,yvals)
                break
            
            ax.set_xlim(xlim)
            ax.grid()
            ax.set_ylabel(gme_obj.plot_info['ind_0_gme_label'])
#             Overplot Epoch Time 0
            if axvline:
                ax.axvline(axvline,ls='--',color='g')

        axs = []
        # RTI Plot
        ax_inx  += 1
        ax      = fig.add_subplot(ny_ax,nx_ax,ax_inx)

        data_obj = EmptyObj()
        data_obj.active = curr_data
        rti_ylim = curr_data.metadata.get('gateLimits')
        
        if data_obj.active.metadata['radar'] in ['cve','cvw']:
            scale   = (0,50.)
        else:
            scale   = (0,30.)

        cmap,norm,bounds = davitpy.utils.plotUtils.genCmap('p_l',scale,'lasse')
        rti = davitpy.pydarn.plotting.musicRTI(data_obj,axis=ax,plotTerminator=True,plot_info=False,
                plot_title=False, plot_cbar=False,ylim=rti_ylim,beam=beam,plot_range_limits_label=False,
                cmap_handling='matplotlib',cmap=cmap,norm=norm,bounds=bounds)
        ax.set_ylabel('Range Gate\nGS Geo. Lat.')

        # Overplot Epoch Time 0
        if axvline:
            ax.axvline(axvline,ls='--',color='g')

#        # Over-plot the gates used for the spectral calculation.
#        if calculate_fft:
#            yvals = curr_data.fft_dict['gates'] 
#            label = 'FFT Gates ({!s},{!s})'.format(np.min(yvals),np.max(yvals))
#            ax.plot(curr_data.time,yvals,zorder=500,label=label)
#
#        if calculate_lsp:
#            yvals = curr_data.lspgram_dict['gates'] 
#            label = 'Lomb Scargle Gates ({!s},{!s})'.format(np.min(yvals),np.max(yvals))
#            ax.plot(curr_data.time,yvals,zorder=500,label=label)

        ax.grid()
        ax.set_xlim(xlim)
#        lg = ax.legend(loc='upper right',fontsize='xx-small')
#        lg.set_zorder(500)

        tmp = {}
        tmp['ax']   = ax
        tmp['cbar'] = rti.cbar_info
        axs.append(tmp)

        # Spectral Displays
        spec_types = []
#        if calculate_fft:
#            spec_types.append('fft_dict')
#        if calculate_lsp:
#            spec_types.append('lspgram_dict')
#
#        for spec_type in spec_types:
#            ax_inx  += 1
#            ax          = fig.add_subplot(ny_ax,nx_ax,ax_inx)
#            spect_dict  = getattr(curr_data,spec_type)
#            info_dct    = my_plot_spect(spect_dict,ax)
#            ax.grid()
#            ax.set_xlim(xlim)

#            format_axis(ax,**xax_dict)

#            tmp = {}
#            tmp['ax']   = ax
#            tmp['cbar'] = info_dct['cbar_info']
#            axs.append(tmp)

#        title = []
#        txt = '{}: {} B{:02d}'.format(curr_data.metadata['dataSetName'],radar.upper(),int(beam))
#        title.append(txt)
#        txt = sTime.strftime('%d %b %Y %H%M UT') + ' - ' + eTime.strftime('%d %b %Y %H%M UT') 
#        title.append(txt)
#        fig.text(0.5,1,'\n'.join(title),ha='center')


        param_label = curr_data.metadata['param_label']    

        fig.autofmt_xdate()
        ax.set_xlabel('{} - Time [UT]'.format(sTime.strftime('%d %b %Y')))
        fig.tight_layout()
        ax.set_title(radar.upper(),fontdict={'weight':'bold','size':20})
#        fig.text(0.575,1,radar.upper(),ha='center',fontdict={'weight':'bold','size':28})
        plot_colorbars(axs)


        xtls = ax.get_xticklabels()
#        new_xtls = [xtl.get_text()[:-3] for xtl in xtls]
#        ax.set_xticklabels(new_xtls)
        for xtl in xtls:
            xtl.set_rotation(20)
            xtl.set_visible(True)
        
        date_fmt    = '%Y%m%d.%H%M'
        date_str    = '{}-{}'.format(sTime.strftime(date_fmt),eTime.strftime(date_fmt))
        filename    = os.path.join(output_dir,'{}_{}_spect.png'.format(date_str,radar))
        fig.savefig(filename,bbox_inches='tight')
        plt.close()
        ################################################################################

    ret_dict = {}
    if calculate_fft:
        ret_dict['fft_dict'] = curr_data.fft_dict
    if calculate_lsp:
        ret_dict['lspgram_dict']    = curr_data.lspgram_dict

    return ret_dict

def regrid_spect_dict(spect_dict,time_prior,time_post,minimum_spectrum_time=datetime.timedelta(hours=2)):
    grid_minute_vec = np.arange( (time_prior+time_post).total_seconds()/60. )
    grid_dtd_vec    = np.array( [datetime.timedelta(minutes=x) for x in grid_minute_vec] ) - time_prior

    freq_vec        = spect_dict['freq']

    # Check for data in the window.
    tf              = np.logical_and(spect_dict['time'] >= st, spect_dict['time'] < et)
    win_spec_arr    = spect_dict['spect'][tf,:]
    win_dtd_vec     = spect_dict['time'][tf] - st - time_prior

    # Make sure the spectrum values are finite.
    tf              = np.all(np.isfinite(win_spec_arr),axis=1)
    win_spec_arr    = win_spec_arr[tf,:]
    win_dtd_vec     = win_dtd_vec[tf]

    dtd_spect_dict              = spect_dict.copy()
    del dtd_spect_dict['time']
    dtd_spect_dict['timedelta'] = win_dtd_vec
    dtd_spect_dict['spect']     = win_spec_arr

    # Find where there are large gaps in the data.
    max_gap         = datetime.timedelta(minutes=4.)
    bounds          = []
    s_bound         = None
    for tm_inx,tm in enumerate(win_dtd_vec):
        if s_bound is None:
            s_bound     = tm
        elif tm-curr_bound > max_gap:
            bounds.append( (s_bound, tm) )
            s_bound = None
        elif (s_bound is not None) and (tm_inx == win_dtd_vec.size-1):
            bounds.append( (s_bound, tm) )
        curr_bound  = tm

    # Calculate total amount of time we are able to determine spectra for.
    # If less than the requested minimum_spectrum_time, reject event.
    spectrum_time   = datetime.timedelta(0)
    for s_bound,e_bound in bounds:
        spectrum_time += e_bound-s_bound

    if spectrum_time < minimum_spectrum_time:
        return None

    # Allocate gridded numpy array grid.
    grid_spec_arr       = np.zeros([grid_dtd_vec.size,spect_dict['freq'].size],dtype=np.float)
    grid_spec_arr[:]    = np.nan

    # Regrid parts of the spectrum.
    for s_bound,e_bound in bounds:
        grid_tf = np.logical_and(grid_dtd_vec >= s_bound, grid_dtd_vec < e_bound)
        win_tf  = np.logical_and(win_dtd_vec  >= s_bound, win_dtd_vec  < e_bound)

        win_x   = np.array( [x.total_seconds() for x in win_dtd_vec[win_tf]] )
        win_y   = win_spec_arr[win_tf,:]

        if len(win_x) == 1: continue

        fnew    = sp.interpolate.interp1d(win_x,win_y,axis=0,bounds_error=False)
        grid_x  = np.array( [x.total_seconds() for x in grid_dtd_vec[grid_tf]] )

        grid_spec_arr[grid_tf,:] = fnew(grid_x)

    if np.count_nonzero(np.isfinite(grid_spec_arr)) == 0:
        return None

    grid_spect_dict                 = dtd_spect_dict.copy()
    grid_spect_dict['raw_timedelta']= grid_spect_dict['timedelta'] 
    grid_spect_dict['raw_spect']    = grid_spect_dict['spect'] 
    grid_spect_dict['timedelta']    = grid_dtd_vec
    grid_spect_dict['spect']        = grid_spec_arr

    return grid_spect_dict

def plot_regrid_comparison(grid_spect_dict,st,et,prop_delay,gme_objs=None,output_dir=''):
    axs     = []
    ny_ax   = 3
    nx_ax   = 1
    ax_inx  = 0

    figsize = (10,9)
    fig = plt.figure(figsize=figsize)
    
    date_fmt    = '%Y%m%d.%H%M'
    date_str    = '{}-{}'.format(st.strftime(date_fmt),et.strftime(date_fmt))
    filename    = os.path.join(output_dir,'{}_{}_{}_regrid.png'.format(date_str,radar,grid_spect_dict['spec_type']))
    
    #####
    if gme_objs is not None:
        ax_inx      += 1
        ax          = fig.add_subplot(ny_ax,nx_ax,ax_inx)

        for gme_obj in gme_objs:
            gme_st  = st-prop_delay
            gme_et  = et-prop_delay
            tf      = np.logical_and(gme_obj.ind_df.index >= gme_st, gme_obj.ind_df.index < gme_et)
            if np.count_nonzero(tf) == 0: continue

            gme_deltas  = gme_obj.ind_df.index[tf] -gme_st -time_prior 
            xvals       = np.array([x.total_seconds() for x in gme_deltas])
            yvals       = gme_obj.ind_df['ind_0_processed'][tf]
            ax.plot(xvals,yvals)
            break

        ax.set_ylabel(gme_obj.plot_info['ind_0_gme_label'])
        format_axis(ax,**xax_dict)

    #####
    ax_inx      += 1
    ax          = fig.add_subplot(ny_ax,nx_ax,ax_inx)

    raw_spect_dict  = grid_spect_dict.copy()
    raw_spect_dict['timedelta'] = raw_spect_dict['raw_timedelta']
    raw_spect_dict['spect']     = raw_spect_dict['raw_spect']

    info_dct    = my_plot_spect(raw_spect_dict,ax)
    format_axis(ax,**xax_dict)
    ax.set_title('Original Time Vector {}'.format(grid_spect_dict['spec_type']))

    tmp = {}
    tmp['ax']   = ax
    tmp['cbar'] = info_dct['cbar_info']
    axs.append(tmp)
    
    #####
    ax_inx      += 1
    ax          = fig.add_subplot(ny_ax,nx_ax,ax_inx)
    info_dct    = my_plot_spect(grid_spect_dict,ax)
    format_axis(ax,**xax_dict)
    ax.set_title('Regridded Time Vector {}'.format(grid_spect_dict['spec_type']))

    tmp = {}
    tmp['ax']   = ax
    tmp['cbar'] = info_dct['cbar_info']
    axs.append(tmp)
    
    #####
    title = []
    title.append(filename)
#    title.append('GME Propagation Time: {:.02f} hr'.format(prop_delay.total_seconds()/3600.))
    title.append('Window Right')
    fig.text(0.5,1.,'\n'.join(title),ha='center')
    fig.tight_layout()

    plot_colorbars(axs)

    fig.savefig(filename,bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    plot_regrid     = False
    calculate_fft   = True
    calculate_lsp   = False
    recalculate     = True

    avg_plot_cbar_scale,avg_cbar_step = (0,3.5), 0.5

    radars          = ['cvw','cve','fhw','fhe','bks','wal']
    radars      += ['gbr','kap','pgr','sas']

    gme_search_periods = []
    gme_search_periods.append( [datetime.datetime(2011,11,1), datetime.datetime(2012,2,1)] )
    gme_search_periods.append( [datetime.datetime(2012,11,1), datetime.datetime(2013,2,1)] )

    # Remove unwanted events. ######################################################
    reject_list = []
    reject_list.append( (datetime.datetime(2012,11,12,13,27), datetime.datetime(2012,11,13,1,27)) )

    sTime, eTime = np.min(gme_search_periods), np.max(gme_search_periods)

    gme_dict    = {}
    gme_dict['gme_param']   = 'omni_by_var'
    gme_dict['epoch_title'] = gme_dict['gme_param'] 
    gme_dict['thresh']      = (0.,0.75)
    gme_dict['min_thresh_duration'] = datetime.timedelta(hours=1)

    time_prior  = datetime.timedelta(hours=6)
    time_post   = datetime.timedelta(hours=6)
    prop_delay  = datetime.timedelta(hours=0)
    minimum_spectrum_time = datetime.timedelta(hours=2)

    ################################################################################
    tmp = {}
    base_dir            = os.path.join('output',gme_dict['epoch_title'])
    tmp['summary']      = os.path.join(base_dir,'summary')
    tmp['calc_spect']   = os.path.join(base_dir,'calc_spect')
    tmp['included_spect']   = os.path.join(base_dir,'included_spect')
    tmp['regrid_fft']   = os.path.join(base_dir,'regrid_fft')
    tmp['regrid_lsp']   = os.path.join(base_dir,'regrid_lsp')
    output_dirs         = tmp

    try:
        shutil.rmtree(output_dirs['summary'])
    except:
        pass

    try:
        shutil.rmtree(output_dirs['included_spect'])
    except:
        pass

    handling.prepare_output_dirs(tmp,clear_output_dirs=recalculate)

    gme_event_windows   = []
    gme_objs            = []
    epoch_title         = gme_dict['epoch_title']
    for gme_search_sTime, gme_search_eTime in gme_search_periods:
        gme_obj     = rsl.GmeFilter(gme_search_sTime,gme_search_eTime,**gme_dict)
        gme_objs.append(gme_obj)

        if epoch_title in ['omni_bz']:
            arg_epoch_0 = gme_obj.find_bz_turning()
        elif epoch_title in ['omni_pdyn']:
            arg_epoch_0 = gme_obj.find_pdyn_events()
        elif epoch_title in ['omni_by_var']:
            arg_epoch_0 = gme_obj.find_endpoint()
            arg_epoch_0 = gme_obj.find_b_var_events()
        elif epoch_title in ['omni_bz_var']:
            arg_epoch_0 = gme_obj.find_endpoint()
            arg_epoch_0 = gme_obj.find_bz_var_events()
        elif epoch_title in ['omni_b_var']:
            arg_epoch_0 = gme_obj.find_endpoint()
            arg_epoch_0 = gme_obj.find_b_var_events()
        elif epoch_title in ['omni_by_pos_morning','omni_by_neg_morning','omni_by_pos_afternoon','omni_by_neg_afternoon']:
            arg_epoch_0 = gme_obj.find_midpoint()
        elif epoch_title in ['omni_pdyn']:
            arg_epoch_0 = gme_obj.find_pressure_pulses()
        else:
            arg_epoch_0 = gme_obj.find_maxima()

        arg_epoch_0 = gme_obj.check_event_quality(time_prior,time_post)
        arg_epoch_0 = gme_obj.remove_events(reject_list)

        manual_event_select = True
        if manual_event_select:
            test_events = []
            test_events.append(datetime.datetime(2011,11,7,12) + datetime.timedelta(hours=6))
            test_events.append(datetime.datetime(2012,12,5,12) + datetime.timedelta(hours=6))
            gme_obj.arg_epoch_0 = np.array(test_events)
        
        if hasattr(gme_obj,'bound_pairs'):  # Bound pairs are really kind of useless;
            del gme_obj.bound_pairs         # get rid of them here to make sure we don't have a dependence on them.

    # Remove any T_0s that are too close to each other. ############################ 
    all_arg_epoch_0 = []
    for gme_obj in gme_objs:
        all_arg_epoch_0 += gme_obj.arg_epoch_0.tolist()

    all_arg_epoch_0 = np.array(all_arg_epoch_0)
    isolation       = datetime.timedelta(hours=6)
    arg_diff        = np.diff(all_arg_epoch_0)
    iso_tf          = np.array([True] + (arg_diff > isolation).tolist())
    if all_arg_epoch_0.size > 0:
        all_arg_epoch_0 = all_arg_epoch_0[iso_tf].tolist()

    for gme_obj in gme_objs:
        arg_epoch_0 = []
        for t_0 in gme_obj.arg_epoch_0:
            if t_0 in all_arg_epoch_0:
                arg_epoch_0.append(t_0)
        gme_obj.arg_epoch_0 = np.array(arg_epoch_0)

    # Plot each GME Period and build event window list. ############################
    for gme_obj in gme_objs:
        bound_pairs = []
        for tm in gme_obj.arg_epoch_0:
            gme_st  = tm - time_prior
            gme_et  = tm + time_post
            
            bp      = (gme_st, gme_et)
            gme_event_windows.append(bp)
            bound_pairs.append(bp)

        gme_obj.bound_pairs = bound_pairs
        gme_obj.plot_gme(output_dir=output_dirs['summary'])

    gme_event_windows   = np.array(gme_event_windows)
    run_times           = gme_event_windows + prop_delay

    tmp = {}
    tmp['xlim']         = (0.-time_prior.total_seconds(),time_post.total_seconds())
    tmp['xticks']       = epoch_xticks(time_prior,time_post)
    tmp['xticklabels']  = format_total_second_xticklabels(tmp['xticks'])
    tmp['xlabel']       = 'Hours from Epoch'
    tmp['axvline']      = 0.
    tmp['grid']         = True
    xax_dict = tmp
    
    # Plot superposed epoch of GME.
    tmp = []

    tmp.append('ae')
    tmp.append('symh')

    tmp.append('pDyn')
    tmp.append('bMagAvg')

    tmp.append('flowSpeed')
    tmp.append('bx')

    tmp.append('np')
    tmp.append('by')

    tmp.append('temp')
    tmp.append('bz')
    gme_param_plot_list = tmp

    ny_ax,nx_ax,ax_inx  = len(gme_param_plot_list), 2, 0
    figsize = (15,ny_ax*2.5)
    fig = plt.figure(figsize=figsize)

    run_times = [[datetime.datetime(2011,11,7,12), datetime.datetime(2011,11,8)],
                 [datetime.datetime(2012,12,5,12), datetime.datetime(2012,12,6)]]
    run_times   = np.array(run_times)

    for gme_param_inx,gme_param in enumerate(gme_param_plot_list):
        label = True
        ax_inx      += 1
        ax          = fig.add_subplot(ny_ax,nx_ax,ax_inx)
        plot_superposed_gme(gme_objs,run_times,time_prior,ax,prop_delay,
                parameter=gme_param,label=label)
        format_axis(ax,**xax_dict)
        
        #Short-circuit things for CEDAR poster.
        new_xts = (np.arange(12) +12).tolist() + [0]
        new_xtls = ['{:02d}:00'.format(x) for x in new_xts]
        ax.set_xticklabels(new_xtls)
        ax.set_xlabel('Time [UT]')

        tkls = ax.get_xticklabels()
        for tkl in tkls:
            tkl.set_size('x-small')

    fig.tight_layout(h_pad=0.01)

    fig.legend(*ax.get_legend_handles_labels(),loc='upper right')
#    date_fmt    = '%Y%m%d.%H%M'
#    date_str    = '{}-{}'.format(sTime.strftime(date_fmt),eTime.strftime(date_fmt))
#    title = []
#    title.append('Nov, Dec, Jan')
#    title.append(date_str)
#
#    if manual_event_select:
#        if len(test_events) == 1:
#            title = []
#            date_str    = 'T_0: {}'.format(test_events[0].strftime('%Y %b %d %H%M UT'))
#            title.append(date_str)
#
#    fig.text(0.5,1,'\n'.join(title),ha='center')
#
    fig.savefig(os.path.join(output_dirs['summary'],'0002_superposed_gme.png'),bbox_inches='tight')

    ################################################################################
    ny_ax,nx_ax,ax_inx  = 1, 1, 0
    figsize = (10,4)
    fig = plt.figure(figsize=figsize)

    ax_inx      += 1
    ax          = fig.add_subplot(ny_ax,nx_ax,ax_inx)
    plot_superposed_bcomps(gme_objs,run_times,time_prior,ax,prop_delay)
    format_axis(ax,**xax_dict)

    fig.tight_layout()

    date_fmt    = '%Y%m%d.%H%M'
    date_str    = '{}-{}'.format(sTime.strftime(date_fmt),eTime.strftime(date_fmt))
    title = []
    title.append('Nov, Dec, Jan')
    title.append(date_str)
    fig.text(0.5,1,'\n'.join(title),ha='center')
    fig.savefig(os.path.join(output_dirs['summary'],'0003_superposed_b.png'),bbox_inches='tight')

#    import ipdb; ipdb.set_trace()

    all_ffts        = None
    all_lsps        = None
    good_run_times  = []
    for radar in radars:
        for st,et in run_times:
            date_fmt    = '%Y%m%d.%H%M'
            date_str    = '{}-{}'.format(st.strftime(date_fmt),et.strftime(date_fmt))
            spect_name  = os.path.join(output_dirs['calc_spect'],'{}_{}_spect.png'.format(date_str,radar))
            fft_pname   = os.path.join(output_dirs['regrid_fft'],'{}_{}_fft.h5'.format(date_str,radar))
            lsp_pname   = os.path.join(output_dirs['regrid_lsp'],'{}_{}_lsp.h5'.format(date_str,radar))

            if not recalculate:
                recalculate_this = False

            if calculate_fft:
                if not os.path.exists(fft_pname):
                    recalculate_this = True

            if calculate_lsp:
                if not os.path.exists(lsp_pname):
                    recalculate_this = True

            gme_epoch_0 = st-prop_delay+time_prior # Actual Epoch Key Time

            if recalculate_this:
                result  = calc_spect_dict(st,et,radar,plot=True,
                            gme_objs=gme_objs,prop_delay=prop_delay,
                            output_dir=output_dirs['calc_spect'],axvline=gme_epoch_0,
                            calculate_fft=calculate_fft,calculate_lsp=calculate_lsp)
                            
                if result is None: continue

                if calculate_fft:
                    grid_fft_dict   = regrid_spect_dict(result['fft_dict'],time_prior,time_post,minimum_spectrum_time=minimum_spectrum_time)
                    with h5py.File(fft_pname, 'w') as fl:
                        saveDictToHDF5(fl, grid_fft_dict)
                if calculate_lsp:
                    grid_lsp_dict   = regrid_spect_dict(result['lspgram_dict'],time_prior,time_post,minimum_spectrum_time=minimum_spectrum_time)
                    with h5py.File(lsp_pname, 'w') as fl:
                        saveDictToHDF5(fl, grid_lsp_dict)
            else:
                if calculate_fft:
                    with h5py.File(fft_pname, 'r') as fl:
                        grid_fft_dict = extractDataFromHDF5(fl)
                if calculate_lsp:
                    with h5py.File(lsp_pname, 'r') as fl:
                        grid_lsp_dict = extractDataFromHDF5(fl)

            if calculate_fft:
                # Save a good spect_dict as a template for later...
                if grid_fft_dict is None: continue

                if gme_dict.get('radar_slt'):
                    slt_0,slt_1 = gme_dict.get('radar_slt')

                    ep_0_inx = np.argmin(np.abs(grid_fft_dict['timedelta']-time_prior))
                    this_slt = grid_fft_dict['slt'][ep_0_inx]

                    if not np.isfinite(this_slt): continue
                    if slt_0:
                        if this_slt < slt_0:
                            continue 
                    if slt_1:
                        if this_slt >= slt_1:
                            continue

                grid_fft_template = grid_fft_dict

            if calculate_lsp:
                if grid_lsp_dict is None: continue

                if gme_dict.get('radar_slt'):
                    slt_0,slt_1 = gme_dict.get('radar_slt')

                    ep_0_inx = np.argmin(np.abs(grid_lsp_dict['timedelta']-time_prior))
                    this_slt = grid_lsp_dict['slt'][ep_0_inx]

                    if not np.isfinite(this_slt): continue
                    if slt_0:
                        if this_slt < slt_0:
                            continue 
                    if slt_1:
                        if this_slt >= slt_1:
                            continue

                grid_lsp_template = grid_lsp_dict

            if (st, et) not in good_run_times:
                good_run_times.append( (st, et) )

            if plot_regrid and recalculate_this:
                if calculate_fft:
                    plot_regrid_comparison(grid_fft_dict,st,et,prop_delay,gme_objs,output_dir=output_dirs['regrid_fft']) 

                if calculate_lsp:
                    plot_regrid_comparison(grid_lsp_dict,st,et,prop_delay,gme_objs,output_dir=output_dirs['regrid_lsp']) 

            if calculate_fft:
                tmp = grid_fft_dict['spect'].copy()
                tmp.shape   = (1,tmp.shape[0],tmp.shape[1])
                if all_ffts is None:
                    all_ffts    = tmp
                else:
                    all_ffts    = np.concatenate( (all_ffts,tmp), axis=0)

            if calculate_lsp:
                tmp = grid_lsp_dict['spect'].copy()
                tmp.shape   = (1,tmp.shape[0],tmp.shape[1])
                if all_lsps is None:
                    all_lsps    = tmp
                else:
                    all_lsps    = np.concatenate( (all_lsps,tmp), axis=0)

            if os.path.exists(spect_name):
                shutil.copy(spect_name,output_dirs['included_spect'])

    if calculate_fft:
        try:
            all_fft_spect_dict                 = grid_fft_template.copy()
        except:
            import ipdb; ipdb.set_trace()
        all_fft_spect_dict['spect']        = np.nanmean(all_ffts,axis=0)
        all_fft_spect_dict['nr_spect']     = all_ffts.shape[0]
        all_fft_spect_dict['roll_nr_spect'] = np.sum( np.isfinite(np.sum(all_ffts,axis=2)), axis = 0)

    if calculate_lsp:
        all_lsp_spect_dict                 = grid_lsp_template.copy()
        all_lsp_spect_dict['spect']        = np.nanmean(all_lsps,axis=0)
        all_lsp_spect_dict['nr_spect']     = all_lsps.shape[0]
        all_lsp_spect_dict['roll_nr_spect'] = np.sum( np.isfinite(np.sum(all_lsps,axis=2)), axis = 0)

    # Mean Spectra #################################################################
    axs = []
    ny_ax   = 3
    nx_ax   = 1
    ax_inx  = 0

    figsize = (10,10)
    fig = plt.figure(figsize=figsize)

    date_fmt    = '%Y%m%d.%H%M'
    date_str    = '{}-{}'.format(sTime.strftime(date_fmt),eTime.strftime(date_fmt))
    filename    = os.path.join(output_dirs['summary'],'0001_{}_{}_mean_spectra.png'.format(date_str,'_'.join(radars)))

    ax_inx      += 1
    ax          = fig.add_subplot(ny_ax,nx_ax,ax_inx)
    gme_ax      = ax

    plot_superposed_gme(gme_objs,good_run_times,time_prior,ax,prop_delay)
    format_axis(ax,**xax_dict)

    cbar_ticks  = np.arange(avg_plot_cbar_scale[0],avg_plot_cbar_scale[1]+avg_cbar_step,avg_cbar_step)
    #####
    plot_list = []
    if calculate_fft:
        all_fft_spect_dict['title'] = 'All FFTs'
        plot_list.append(all_fft_spect_dict)

    if calculate_lsp:
        all_lsp_spect_dict['title'] = 'All LSPs'
        plot_list.append(all_lsp_spect_dict)

    for plot_dict in plot_list:
        ax_inx      += 1
        ax          = fig.add_subplot(ny_ax,nx_ax,ax_inx)
        info_dct    = my_plot_spect(plot_dict,ax,scale=avg_plot_cbar_scale)
        format_axis(ax,**xax_dict)

        plot_dict['ax'] = ax

        tmp = {}
        tmp['ax']   = ax
        tmp['cbar'] = info_dct['cbar_info']
        tmp['cbar']['ticks'] = cbar_ticks
        axs.append(tmp)

    #####
    nr_spect = all_fft_spect_dict['nr_spect']

    title = []
    title.append(filename)
    title.append('Total Events: {:d}; Nr. Avged. Spectragrams: {:d}'.format(len(good_run_times),int(nr_spect)))
#    title.append('GME Propagation Time: {:.02f} hr'.format(prop_delay.total_seconds()/3600.))
    title.append('Window Right')
    fig.text(0.5,0.925,'\n'.join(title),ha='center')
    fig.tight_layout(h_pad=0.05)

    # Shorten GME Plot Height
    ax = gme_ax
    pos = list(ax.get_position().bounds)
    pos[3] = pos[3]*0.75
    ax.set_position(pos)

    # Add Colorbars
    plot_colorbars(axs)

    # Add information about how many spectra were used
    for plot_dict in plot_list:
        ax = plot_dict['ax']

        super_plot_hgt  = 0.03
        pos = list(ax.get_position().bounds)
        pos[3] = pos[3] - (super_plot_hgt)
        ax.set_position(pos)

        curr_xlim   = ax.get_xlim()
        curr_xticks = ax.get_xticks()

        pos[1] = pos[1] + pos[3]
        pos[3] = super_plot_hgt
        ax_1    = fig.add_axes(pos)
        xvec    = np.array( [x.total_seconds() for x in plot_dict['timedelta']] )
        yvec    = plot_dict['roll_nr_spect']
        ax_1.plot(xvec,yvec)
        ax_1.set_xlim(curr_xlim)
        ax_1.set_xticks(curr_xticks)
        ax_1.set_xticklabels(['']*len(curr_xticks))

        yts     = ax_1.get_yticks()
        ax_1.set_yticks([yts[-1]])
        ax_1.set_title(plot_dict['title'])
        ################################################################################

    fig.savefig(filename,bbox_inches='tight')
    plt.close()
