import os
import datetime
import copy
import string

import matplotlib
from matplotlib import pyplot as plt
from matplotlib.collections import PolyCollection
from mpl_toolkits.basemap import Basemap
import numpy as np
import scipy as sp
import scipy.stats
import pandas as pd

import pymongo

from .general_lib import prepare_output_dirs
from . import general_lib as gl
from . import polar_met
from . import drivers
from . import mongo_tools
from . import run_helper

cbar_title_fontdict     = {'weight':'bold','size':30}
cbar_ytick_fontdict     = {'size':30}
xtick_fontdict          = {'weight': 'bold', 'size':30}
ytick_major_fontdict    = {'weight': 'bold', 'size':28}
ytick_minor_fontdict    = {'weight': 'bold', 'size':24}
corr_legend_fontdict    = {'weight': 'bold', 'size':24}
keo_legend_fontdict     = {'weight': 'normal', 'size':30}
driver_xlabel_fontdict  = ytick_major_fontdict
driver_ylabel_fontdict  = ytick_major_fontdict
title_fontdict          = {'weight': 'bold', 'size':36}

def get_time_res(driver_obj):
    time_reses  = np.unique(np.diff(driver_obj.ind_times))
    if len(time_reses) > 1:
        print('Error! Time series not evenly sampled.')
        import ipdb; ipdb.set_trace()
    else:
        time_res = time_reses[0]

    return time_res

def get_x_coords(win_sDate,sDate,eDate,full=False):
    x1  = (win_sDate - sDate).total_seconds()/86400.
    if not full:
        x1  = np.floor(x1)
    return x1

def get_y_coords(ut_time,st_uts,radar,radars):
    # Find start time index.
    st_uts      = np.array(st_uts)
    st_ut_inx   = np.digitize([ut_time],st_uts)-1
    
    # Find radar index.
    radar_inx   = np.where(radar == np.array(radars))[0]
    y1          = st_ut_inx*len(radars) + radar_inx
    return y1

def get_coords(radar,win_sDate,radars,sDate,eDate,st_uts,verts=True):
    # Y-coordinate.
    x1  = get_x_coords(win_sDate,sDate,eDate)
    y1  = get_y_coords(win_sDate.hour,st_uts,radar,radars)

    if verts:
#        x1,y1   = x1+0,y1+0
        x2,y2   = x1+1,y1+0
        x3,y3   = x1+1,y1+1
        x4,y4   = x1+0,y1+1
        return ((x1,y1),(x2,y2),(x3,y3),(x4,y4),(x1,y1))
    else:
        x0      = x1 + 0.5
        y0      = y1 + 0.5
        return (x0, y0)

def my_xticks(sDate,eDate,ax,radar_ax=False,labels=True,
        fontdict=None):
    if fontdict is None:
        fontdict = xtick_fontdict
    xticks      = []
    xticklabels = []
    curr_date   = sDate
    while curr_date < eDate:
        if radar_ax:
            xpos    = get_x_coords(curr_date,sDate,eDate)
        else:
            xpos    = curr_date
        xticks.append(xpos)
        xticklabels.append('')
        curr_date += datetime.timedelta(days=1)

    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)

    # Define xtick label positions here.
    # Days of month to produce a xtick label.
    doms    = [1,15]

    curr_date   = sDate
    ytransaxes = matplotlib.transforms.blended_transform_factory(ax.transData,ax.transAxes)
    while curr_date < eDate:
        if curr_date.day in doms:
            if radar_ax:
                xpos    = get_x_coords(curr_date,sDate,eDate)
            else:
                xpos    = curr_date

            axvline = ax.axvline(xpos,-0.015,color='k')
            axvline.set_clip_on(False)

            if labels:
                ypos    = -0.025
                txt     = curr_date.strftime('%d %b\n%Y')
                ax.text(xpos,ypos,txt,transform=ytransaxes,
                        ha='left', va='top',rotation=0,
                        fontdict=fontdict)
        curr_date += datetime.timedelta(days=1)

    xmax    = (eDate - sDate).total_seconds() / (86400.)
    if radar_ax:
        ax.set_xlim(0,xmax)
    else:
        ax.set_xlim(sDate,sDate+datetime.timedelta(days=xmax))

    ax.grid(zorder=500)

def get_sDate_eDate(group_dict,sDate,eDate):
    """
    Finds the ealiest sDate and latest eDate for all radars in question,
    unless an sDate and eDate has been explicitly specified.
    """
    if (sDate is None) or (eDate is None):
        dates       = []
        for key,val in group_dict.items():
            dct_list = val['dct_list']
            for dct in dct_list:
                dates.append(dct['list_sDate'])
                dates.append(dct['list_eDate'])

    if sDate is None:
        sDate   = np.min(dates)

    if eDate is None:
        eDate   = np.max(dates)

    return sDate, eDate

def plot_calendar_panel(dct_list,sDate,eDate,scale,st_uts,val_key,ax,
        xlabels=True,db_name='mstid',mongo_port=27017,lambda_max=750.,
        highlight_ew=False,group_name=None,classification_colors=False,
        rasterized=False,**kwargs):

    mongo       = pymongo.MongoClient(port=mongo_port)
    db          = mongo[db_name]

    radar_dict  = {}
    radars      = []
    for dct in dct_list:
        radar   = dct['radar']
        radars.append(radar)

        radar_dict[radar]   = {}
        radar_dict[radar]['mstid_list'] = dct['mstid_list']

    ymax    = len(st_uts) * len(radars)

    cbar_info   = {}
    if val_key == 'music_azm':
        cbar_info['cbar_tick_fmt']  = '%.0f' 
        cbar_info['cbar_tb_vis']    = True
        cbar_info['cbar_label']     = 'MSTID\nGeog. Azim. [deg]'

        if not highlight_ew:
            cmap        = matplotlib.cm.hsv
            bounds      = np.linspace(0,360,360)
            cbar_info['cbar_ticks']     = [0, 90, 180, 270, 360]
        else:
            # Make sure red is East!!
            cmap        = matplotlib.cm.seismic_r
#            cmap        = gl.get_custom_cmap('blue_red')
            bounds      = np.linspace(90.,270.,180)
            cbar_info['cbar_ticks']     = [90, 135, 180, 225, 270]

            bounds      = np.linspace(90.,270.,180)
            cbar_ticks  = [90, 135, 180, 225, 270]

            cbar_info['cbar_ticks']     = cbar_ticks
    elif val_key == 'mstid_azm_dev':
        cbar_info['cbar_tick_fmt']  = '%.0f' 
        cbar_info['cbar_tb_vis']    = True
        cbar_info['cbar_label']     = 'Azm - $\sigma$ [deg]'

#        cmap        = gl.get_custom_cmap('blue_red')
        cmap        = matplotlib.cm.seismic
#        bounds      = np.linspace(-25,25,50)
#        cbar_info['cbar_ticks']     = [-25,-12.5,0,12.5,25]
        bounds      = np.linspace(-50,50,100)
        cbar_info['cbar_ticks']     = [-50,-25.0,0,25.0,50]
    else:
        cmap        = matplotlib.cm.jet
        bounds      = np.linspace(scale[0],scale[1],256)
        cbar_info['cbar_ticks'] = np.linspace(scale[0],scale[1],11)
        cbar_info['cbar_label'] = 'MSTID Index'

    norm    = matplotlib.colors.BoundaryNorm(bounds,cmap.N)

    if classification_colors:
        # Use colorscheme that matches MSTID Index in classification plots.
        from mstid.classify import MyColors
        scale_0             = -0.025
        scale_1             =  0.025
        my_cmap             = 'seismic'
        truncate_cmap       = (0.1, 0.9)
        my_colors           = MyColors((scale_0, scale_1),my_cmap=my_cmap,truncate_cmap=truncate_cmap)
        cmap            = my_colors.cmap
        norm            = my_colors.norm

    ################################################################################    
    current_date = sDate
    verts       = []
    vals        = []
    while current_date < eDate:
        for st_ut in st_uts:
            for radar in radars:
                win_sDate   = current_date + datetime.timedelta(hours=st_ut)
                mstid_list  = radar_dict[radar]['mstid_list']

                item        = db[mstid_list].find_one({'radar':radar,'sDatetime':win_sDate})
                if item is None: continue

                # Get the value to be plotted.
                if 'music_' in val_key:
                    sig_key = val_key.lstrip('music_')
                    
                    if highlight_ew:
                        azm_lim = (90.,270.)
                    else:
                        azm_lim = None
                    val = mongo_tools.get_mstid_value(item,sig_key,lambda_max=lambda_max,azm_lim=azm_lim)
                else:
                    if val_key not in item: continue
                    val = item.get(val_key)

                if val is None:
                    continue

                vals.append(val)
                verts.append(get_coords(radar,win_sDate,radars,sDate,eDate,st_uts))

        current_date += datetime.timedelta(days=1)

    pcoll = PolyCollection(np.array(verts),edgecolors='face',linewidths=0,closed=False,
            cmap=cmap,norm=norm,zorder=99,rasterized=rasterized)
    pcoll.set_array(np.array(vals))
    ax.add_collection(pcoll,autolim=False)

    # Make gray missing data.
    ax.set_axis_bgcolor('0.90')

    # Add radar labels.
    trans = matplotlib.transforms.blended_transform_factory(ax.transAxes,ax.transData)
    for rdr_inx,radar in enumerate(radars):
        for st_inx,st_ut in enumerate(st_uts):
            ypos = len(radars)*st_inx + rdr_inx + 0.5
            ax.text(-0.002,ypos,radar,transform=trans,ha='right',va='center')

    # Add UT Time Labels
    for st_inx,st_ut in enumerate(st_uts):
        ypos    = st_inx*len(radars)
        xpos    = -0.035
        line    = ax.hlines(ypos,xpos,1.0,transform=trans,lw=3,zorder=100)
        line.set_clip_on(False)
        
        txt     = '{:02d}-{:02d} UT'.format(int(st_ut),int(st_ut+2))
        ypos   += len(radars)/2.
        xpos    = -0.025
#        ax.text(xpos,ypos,txt,transform=trans,
#                ha='center',va='center',rotation=90,fontdict=ytick_major_fontdict)
        xpos    = -0.015
        ax.text(xpos,ypos,txt,transform=trans,
                ha='right',va='center',rotation=0,fontdict=ytick_major_fontdict)

    xpos    = -0.035
    line    = ax.hlines(1.,xpos,1.0,transform=ax.transAxes,lw=3,zorder=100)
    line.set_clip_on(False)

    ax.set_ylim(0,ymax)

    # Set xticks and yticks to every unit to make a nice grid.
    # However, do not use this for actual labeling.
    yticks = list(range(len(radars)*len(st_uts)))
    ax.set_yticks(yticks)
    ytls = ax.get_yticklabels()
    for ytl in ytls:
        ytl.set_visible(False)

    my_xticks(sDate,eDate,ax,radar_ax=True,labels=xlabels)
    
    txt = ' '.join([x.upper() for x in radars])
    if group_name is not None:
        txt = '{} ({})'.format(group_name,txt)
    ax.set_title(txt,fontdict=title_fontdict)

    ax_info         = {}
    ax_info['ax']   = ax
    ax_info['cbar_pcoll']   = pcoll
    ax_info.update(cbar_info)
    
    mongo.close()
    return ax_info

def plot_cbars(ax_list):
    for ax_info in ax_list:
        cbar_pcoll = ax_info.get('cbar_pcoll')
        if cbar_pcoll is None:
            continue

        cbar_label      = ax_info.get('cbar_label')
        cbar_ticks      = ax_info.get('cbar_ticks')
        cbar_tick_fmt   = ax_info.get('cbar_tick_fmt','%0.3f')
        cbar_tb_vis     = ax_info.get('cbar_tb_vis',False)
        ax              = ax_info.get('ax')

        box         = ax.get_position()
        axColor     = plt.axes([(box.x0 + box.width) * 1.01 , box.y0, 0.01, box.height])
        cbar        = plt.colorbar(cbar_pcoll,orientation='vertical',cax=axColor,format=cbar_tick_fmt)

        cbar.set_label(cbar_label,fontdict=cbar_title_fontdict)
        if cbar_ticks is not None:
            cbar.set_ticks(cbar_ticks)

        labels = cbar.ax.get_yticklabels()
        fontweight  = cbar_ytick_fontdict.get('weight')
        fontsize    = cbar_ytick_fontdict.get('size')
        for label in labels:
            if fontweight:
                label.set_fontweight(fontweight)
            if fontsize:
                label.set_fontsize(fontsize)

        if not cbar_tb_vis:
            for inx in [0,-1]:
                labels[inx].set_visible(False)

#def plot_cbars_grib(mappables,ax,expon=5,
#        label_fontdict=None, title_fontdict=None, ytick_fontdict=None):
#    """ This is the standard (non-paper version). """
#    if ytick_fontdict is None:
#        ytick_fontdict  = cbar_ytick_fontdict
#
#    if title_fontdict is None:
#        title_fontdict  = cbar_title_fontdict
#
#    if label_fontdict is None:
#        label_fontdict = {'weight':'normal','size':24}
#
#    box     = ax.get_position()
##    x_0     = (box.x0 + box.width) * 1.01 
#    x_0     = (box.x0 + box.width) * 1.025 
#    x_w     = 0.01
#    y_0     = box.y0
#    y_h     = (box.height/len(mappables))
#    y_scale = 0.700
##    y_scale = 0.925
#
#    mbar_levels = mappables.keys()
#    mbar_levels.sort()
#    mbar_levels = mbar_levels[::-1]
#
#    for inx,mbar_level in enumerate(mbar_levels):
#        plot_dct    = mappables[mbar_level]['plot_dct']
#        cbar_pcoll  = mappables[mbar_level]['pcoll']
#
##        cbar_label  = plot_dct['shortName'].upper() + ' [' + plot_dct['units'] + ']'
#        cbar_ticks      = None
#        cbar_tick_fmt   = '%0.2e'
#
#        y_cb        = y_0 + inx*y_h
#        axColor     = plt.axes([x_0, y_cb, x_w,y_scale*y_h])
#
#        cbar        = plt.colorbar(cbar_pcoll,orientation='vertical',cax=axColor,format=cbar_tick_fmt)
#        cbar_label  = '{!s} mb\nlevel'.format(mbar_level)
#
#        cbar.ax.text(-0.95,0.5,cbar_label,fontdict=label_fontdict,
#                rotation=90,va='center',ha='center',transform=cbar.ax.transAxes)
#
#        clim        = cbar.get_clim()
##        cbar_ticks  = np.linspace(*clim,num=3)*0.90
#        nticks  = 9
#        cbar_ticks  = np.linspace(*clim,num=9)
##        cbar_ticks  = cbar_ticks[[1,nticks/2,-2]]
#        cbar_ticks  = cbar_ticks[[1,-2]]
#        cbar.set_ticks(cbar_ticks)
#
#        cbar_ticklabels = []
#        for cbt in cbar_ticks:
##            lbl = '{:0.2f}'.format(cbt/(10**expon))
#            lbl = '{:0.1f}E{!s}'.format(cbt/(10**expon),expon)
##            lbl = '{:0.1e}'.format(cbt)
#            cbar_ticklabels.append(lbl)
#
#        cbar.ax.set_yticklabels(cbar_ticklabels,fontdict=ytick_fontdict)
#
#        labels = cbar.ax.get_yticklabels()
#        fontweight  = ytick_fontdict.get('weight')
#        fontsize    = ytick_fontdict.get('size')
#        for label in labels:
#            if fontweight:
#                label.set_fontweight(fontweight)
#            if fontsize:
#                label.set_fontsize(fontsize)
#    
##        txt = r'$\times 10^{!s}$'.format(expon) + r' [$m^2 s^{-2}$]'
#        txt = r'[$m^2 s^{-2}$]'
#        txt = plot_dct['shortName'].upper() + ' [' + plot_dct['units'] + ']'
#        ax.text(1.070,0.5,txt,rotation=90,va='center',
#                transform=ax.transAxes,fontdict=title_fontdict)
##        txt = '[' + plot_dct['units'] + ']'

def plot_cbars_grib(mappables,ax,expon=5,
        label_fontdict=None, title_fontdict=None, ytick_fontdict=None):
    if ytick_fontdict is None:
        ytick_fontdict  = cbar_ytick_fontdict

    if title_fontdict is None:
        title_fontdict  = cbar_title_fontdict

    if label_fontdict is None:
        label_fontdict = {'weight':'normal','size':24}

    box     = ax.get_position()
#    x_0     = (box.x0 + box.width) * 1.01 
    x_0     = (box.x0 + box.width) * 1.025 
    x_w     = 0.01
    y_0     = box.y0
    y_h     = (box.height/len(mappables))
    y_scale = 0.800
#    y_scale = 0.925

    mbar_levels = list(mappables.keys())
    mbar_levels.sort()
    mbar_levels = mbar_levels[::-1]

    for inx,mbar_level in enumerate(mbar_levels):
        plot_dct    = mappables[mbar_level]['plot_dct']
        cbar_pcoll  = mappables[mbar_level]['pcoll']

        cbar_ticks      = None
        cbar_tick_fmt   = '%0.2e'

        y_cb        = y_0 + inx*y_h
        axColor     = plt.axes([x_0, y_cb, x_w,y_scale*y_h])

        cbar        = plt.colorbar(cbar_pcoll,orientation='vertical',cax=axColor,format=cbar_tick_fmt)
        cbar_label  = '{!s} mb\nlevel'.format(mbar_level)

        cbar.ax.text(-0.95,0.5,cbar_label,fontdict=label_fontdict,
                rotation=90,va='center',ha='center',transform=cbar.ax.transAxes)

        clim        = cbar.get_clim()
        nticks  = 9
        cbar_ticks  = np.linspace(*clim,num=9)
#        cbar_ticks  = cbar_ticks[[1,nticks/2,-2]]
        cbar_ticks  = cbar_ticks[[1,-2]]
        cbar.set_ticks(cbar_ticks)

        cbar_ticklabels = []
        for cbt in cbar_ticks:
            lbl = '{:0.1f}E{!s}'.format(cbt/(10**expon),expon)
#            lbl = '{:0.1e}'.format(cbt)
            cbar_ticklabels.append(lbl)

        cbar.ax.set_yticklabels(cbar_ticklabels,fontdict=ytick_fontdict)

        labels = cbar.ax.get_yticklabels()
        fontweight  = ytick_fontdict.get('weight')
        fontsize    = ytick_fontdict.get('size')
        for label in labels:
            if fontweight:
                label.set_fontweight(fontweight)
            if fontsize:
                label.set_fontsize(fontsize)
    
        txt = plot_dct['shortName'].upper() + ' [' + plot_dct['units'] + ']'
        txt = r'$Z$ [$m^2 s^{-2}$]'
        ax.text(1.070,0.5,txt,rotation=90,va='center',
                transform=ax.transAxes,fontdict=title_fontdict)

def get_xmax(sDate,eDate):
    return (eDate - sDate).total_seconds() / (86400.)

def get_radar_ax_frac(this_key,group_dict):
    """
    Computes fraction of radars in one group versus all radars.
    
    This is used to figure out how much space an axis should take
    in a figure.
    """
    all_radars      = []
    these_radars    = []
    keys            = list(group_dict.keys())
    for key in keys:
        dct_list    = group_dict[key]['dct_list']
        radars      = [dct['radar'] for dct in dct_list]
        all_radars += radars

        if this_key == key:
            these_radars = radars

    frac = float(len(these_radars)) / float(len(all_radars))
    return frac

def calendar_plot(dct_list=None,group_dict=None,sDate=None,eDate=None,val_key='meanSubIntSpect_by_rtiCnt',
        scale=[-0.03,0.03], st_uts=[14, 16, 18, 20],driver=[None],
        output_dir='mstid_data/calendar',db_name='mstid',mongo_port=27017,
        fig_scale=40.,fig_scale_x=1.,fig_scale_y=0.225,
        h_pad = 0.150,
        mstid_reduced_inx=None,
        correlate=False,super_title=None,plot_radars=True,**kwargs):
    
    driver_list         = gl.get_iterable(driver)

    if dct_list is not None:
        group_dict      = {}
        group_dict[0]   = {}
        group_dict[0]['dct_list']   = dct_list

    prepare_output_dirs({0:output_dir},clear_output_dirs=False)

    sDate,eDate = get_sDate_eDate(group_dict,sDate,eDate)
    xmax        = get_xmax(sDate,eDate)

    # Plotting section. ############################################################
    ax_list = []

    ax_nr   = 0
    ax_nx   = 1
    ax_ny   = len(list(group_dict.keys()))

    ax_left             = 0.
    ax_width            = 1.0
    ax_top              = 1.0
    radar_panel_frac    = 1.0

    if driver is not None:
#        ax_ny += len(driver_list)
        ax_ny += 1
        radar_panel_frac = (ax_ny-1.) / ax_ny
        if len(driver_list) == 1:
            radar_panel_frac    = 0.6
        else:
            radar_panel_frac    = 0.5


    figsize     = (fig_scale*fig_scale_x,fig_scale*fig_scale_y*ax_ny)
    fig         = plt.figure(figsize=figsize)

    driver_panel_frac   = 1.-radar_panel_frac

    if plot_radars is False:
        driver_xlabels  = True

    driver_xlabels  = True
    ax_info = plot_the_drivers(sDate,eDate,driver_list,
            driver_panel_frac,ax_left,ax_width,ax_top,h_pad,
            output_dir,mstid_reduced_inx,correlate,xlabels=driver_xlabels)
    ax_top  = ax_info['ax_top']

    #### Plot the radar data.
    radars      = []
    if plot_radars:
        keys    = list(group_dict.keys())
        keys.sort()
        nr_groups   = len(keys)
        for key in keys:
            dct_list    = group_dict[key]['dct_list']
            radars  += [dct['radar'] for dct in dct_list]

            ax_height   = get_radar_ax_frac(key,group_dict) * (radar_panel_frac - h_pad*nr_groups)
            ax_bottom   = ax_top - ax_height - h_pad
            ax_top      = ax_bottom
            ax          = fig.add_axes((ax_left,ax_bottom,ax_width,ax_height))

            if key == max(keys):
                xlabels = True
            else:
                xlabels = False
            ax_info     = plot_calendar_panel(dct_list,sDate,eDate,scale,st_uts,val_key,ax,
                    xlabels=xlabels,db_name=db_name,mongo_port=mongo_port,**kwargs)
            ax_list.append(ax_info)

        plot_cbars(ax_list)

    if super_title is None:
        date_fmt    = '%d %b %Y'
        super_title = '{} - {}'.format(sDate.strftime(date_fmt),eDate.strftime(date_fmt))
    fig.text(0.5,1.005,super_title,ha='center',fontdict={'weight':'bold','size':48})

    date_fmt    = '%Y%m%d'
    filename    = []
    filename.append('calendar')
    filename.append(val_key)
    if driver[0] is not None:
        filename.append('_'.join(driver_list))
    filename.append('{}_{}'.format(sDate.strftime(date_fmt),eDate.strftime(date_fmt)))
    filename.append('_'.join(radars))

    filepath    = os.path.join(output_dir,'_'.join(filename))+'.png'
    fig.savefig(filepath,bbox_inches='tight')
    print('Calendar plot: {}'.format(filepath))

    return filepath

def calendar_plot_with_polar_data(dct_list=None,group_dict=None,sDate=None,eDate=None,val_key='meanSubIntSpect_by_rtiCnt',
        scale=[-0.03,0.03], st_uts=[14, 16, 18, 20],driver=None,
        output_dir='mstid_data/calendar',db_name='mstid',mongo_port=27017,
        dt=None,grib_data=None,mstid_reduced_inx=None,correlate=None,**kwargs):

    driver_list         = gl.get_iterable(driver)
    if correlate:
        driver_list = [driver_list[0],'mstid_reduced_inx','correlate']

    this_date   = dt
    filename    = '{}_vortex_calendar.png'.format(this_date.strftime('%Y%m%d_%H%M'))
    filepath    = os.path.join(output_dir,filename)
    print('Plotting: {}'.format(filepath))
    
    if dct_list is not None:
        group_dict      = {}
        group_dict[0]   = {}
        group_dict[0]['dct_list']   = dct_list

    prepare_output_dirs({0:output_dir},clear_output_dirs=False)

    sDate,eDate = get_sDate_eDate(group_dict,sDate,eDate)
    xmax        = get_xmax(sDate,eDate)

    # Plotting section. ############################################################
    ax_list = []

    ax_nr   = 0
    ax_nx   = 1
    ax_ny   = len(list(group_dict.keys()))

    ax_left             = 0.
    ax_width            = 1.0
    ax_top              = 1.0
    h_pad               = 0.050
    w_pad               = 0.025

    ax_ny += 1
    if driver is None:
        grib_panel_frac     = 0.50
        radar_panel_frac    = 0.50
    else:
        grib_panel_frac     = 0.333
        driver_panel_frac   = 0.333
        radar_panel_frac    = 0.333

    fig_scale   = 40.
    figsize     = (fig_scale*1.,fig_scale*0.225*ax_ny)
    fig         = plt.figure(figsize=figsize)

    # Geopotential Parameter #######################################################
    ax_height   = grib_panel_frac - h_pad
    ax_bottom   = ax_top - ax_height - h_pad
    ax_top      = ax_bottom

    ax_width_g  = 0.3333 - w_pad
#    geo_params  = ['residuals','raw','mean']
#    geo_params  = ['residuals','raw','next']
    geo_params  = ['raw']
    geo_prm_ax  = {}
    geo_prm_ax_grib_only  = {}
    for geo_inx,geo_param in enumerate(geo_params):
        for mbar_level,grb_d in grib_data.items():
                grb     = grb_d[geo_param]
                ax_info = geo_prm_ax.get(geo_param)
                if ax_info is None:
                    ax_left_g   = ax_left + geo_inx*(ax_width_g + w_pad)
                    ax          = fig.add_axes((ax_left_g,ax_bottom,ax_width_g,ax_height))
                    txt = grb.get('big_title')
                    ax.text(0.5,1.1,txt,ha='center',transform=ax.transAxes,
                            fontdict={'weight':'bold','size':48})

                    m       = polar_met.plot_grb_ax(grb,ax)

                    ax_info                 = {}
                    ax_info['ax']           = ax
                    ax_info['m']            = m
                    geo_prm_ax[geo_param]   = ax_info
                else:
                    ax  = ax_info.get('ax')
                    m   = ax_info.get('m')
                    polar_met.plot_grb_ax(grb,ax,m=m)

                # Stick grib data into its own figure.
                ax_info_grib_only =  geo_prm_ax_grib_only.get(geo_param)
                if ax_info_grib_only is None:
                    grb_fig = plt.figure(figsize=(15,15))
                    grb_ax  = grb_fig.add_subplot(111)

                    grb_m   = polar_met.plot_grb_ax(grb,grb_ax)

                    ax_info_grib_only                 = {}
                    ax_info_grib_only['ax']           = grb_ax
                    ax_info_grib_only['m']            = grb_m
                    geo_prm_ax_grib_only[geo_param]   = ax_info_grib_only
                else:
                    grb_ax  = ax_info_grib_only.get('ax')
                    grb_m   = ax_info_grib_only.get('m')

                    polar_met.plot_grb_ax(grb,grb_ax,m=grb_m)

                    grib_only_dir   = os.path.join(output_dir,'grib_only')
                    grib_only_fname = '{}_{}_grib.png'.format(this_date.strftime('%Y%m%d_%H%M'),geo_param)
                    grib_only_fpath = os.path.join(grib_only_dir,grib_only_fname)
                    if not os.path.exists(grib_only_dir):
                        prepare_output_dirs({0:grib_only_dir})

                    grb_fig.savefig(grib_only_fpath,bbox_inches='tight')
                    plt.close(grb_fig)


    if driver is not None:
        ax_info = plot_the_drivers(sDate,eDate,driver_list,
                driver_panel_frac,ax_left,ax_width,ax_top,h_pad,
                output_dir,mstid_reduced_inx,correlate)
        ax_top  = ax_info['ax_top']

        for ax in ax_info['axs']:
            ax.axvline(this_date,ls='--',lw=6,color='k',zorder=750)

    # Radar MSTID Data ############################################################# 
    keys    = list(group_dict.keys())
    keys.sort()
    nr_groups   = len(keys)
    radars      = []
    for key in keys:
        dct_list    = group_dict[key]['dct_list']
        radars  += [dct['radar'] for dct in dct_list]

        ax_height   = get_radar_ax_frac(key,group_dict) * (radar_panel_frac - h_pad*nr_groups)
        ax_bottom   = ax_top - ax_height - h_pad
        ax_top      = ax_bottom
        ax          = fig.add_axes((ax_left,ax_bottom,ax_width,ax_height))

        if key == max(keys):
            xlabels = True
        else:
            xlabels = False
        ax_info     = plot_calendar_panel(dct_list,sDate,eDate,scale,st_uts,val_key,ax,
                xlabels=xlabels,db_name=db_name,mongo_port=mongo_port)

        vline_xpos = get_x_coords(this_date,sDate,eDate,full=True)
        ax.axvline(vline_xpos,ls='--',lw=6,color='k',zorder=750)

        ax_list.append(ax_info)

    plot_cbars(ax_list)

    date_fmt    = '%d %b %Y'
    date_str    = '{} - {}'.format(sDate.strftime(date_fmt),eDate.strftime(date_fmt))
    fig.text(0.5,1.005,date_str,ha='center',fontdict={'weight':'bold','size':48})

    fig.savefig(filepath,bbox_inches='tight')
    plt.close(fig)

    return filepath

def calculate_reduced_mstid_azm(music_groups,val_key='music_azm',
        months=[11,12,1,2,3,4],hours=[14,16,18,20],
        reduction_type='mean',daily_vals=True,
        highlight_ew=False,lambda_max=750,
        db_name='mstid',mongo_port=27017):
    """
    Reduce the MSTID index from all radars into a single number as a function of time.
    """
    print("Calulating reduced MSTID index.")

    mongo       = pymongo.MongoClient(port=mongo_port)
    db          = mongo[db_name]

    mstid_inx_dict  = {} # Create a place to store the data.
    radars          = []

    for music_group in music_groups:
        # Work with one year at a time.
        date_str    = run_helper.get_seDates_from_groups(music_group,date_fmt='%Y%m%d')
        for radar_bank_inx, radar_bank_dict in music_group.items():
            # Work with either the high or mid latitudes
            bank_name   = radar_bank_dict['name']
            for radar_dict in radar_bank_dict['dct_list']:
                # Work with one radar at a time
                mstid_list      = radar_dict['mstid_list']
                print('Reducing MSTID Azimuth: {} - {}'.format(bank_name,mstid_list))
                events          = db[mstid_list].find()
                # Go get the actual data.
                for event in events:
                    this_sTime  = event.get('sDatetime')
                    if this_sTime.month not in months:
                        continue

                    if this_sTime.hour not in hours:
                        continue

                    this_dct = mstid_inx_dict.get(this_sTime)
                    if this_dct is None:
                        this_dct    = {}
                        mstid_inx_dict[this_sTime] = this_dct

                    # Keep track of which radars are being used.
                    radar       = event['radar']
                    if radar not in radars:
                        radars.append(radar)

#                    mstid_index = event.get(val_key,np.nan)

                    # Get the value to be plotted.
                    sig_key = val_key.lstrip('music_')
                    
                    if highlight_ew:
                        azm_lim = (90.,270.)
                    else:
                        azm_lim = None

                    val = mongo_tools.get_mstid_value(event,sig_key,lambda_max=lambda_max,azm_lim=azm_lim)
                    if val is None: val = np.nan

                    this_dct[radar] = val

    # Check to make sure all entries have all radars represented.
    for this_sTime,this_dct in mstid_inx_dict.items():
        for radar in radars:
            if radar not in this_dct:
                this_dct[radar] = np.nan

    # Put everything into a dataframe.
    df          = pd.DataFrame(mstid_inx_dict).transpose()
    
    if daily_vals:
        date_list   = np.unique([datetime.datetime(x.year,x.month,x.day) for x in df.index])

        tmp_list        = []
        n_good_radars   = []    # Set up a system for parameterizing data quality.
        for tmp_sd in date_list:
            tmp_ed      = tmp_sd + datetime.timedelta(days=1)

            tf          = np.logical_and(df.index >= tmp_sd, df.index < tmp_ed)
            tmp_df      = df[tf]
            if reduction_type == 'median':
                tmp_vals    = tmp_df.median().to_dict()
            elif reduction_type == 'mean':
                tmp_vals    = {}
                for rdr in list(tmp_df.keys()):
                    no_na   = tmp_df[rdr].dropna().values
                    val     = scipy.stats.circmean(no_na,low=0.,high=360.)
                    tmp_vals[rdr] = val

            tmp_list.append(tmp_vals)

            n_good  = np.count_nonzero(np.isfinite(tmp_df))
            n_good_radars.append(n_good)

        df = pd.DataFrame(tmp_list,index=date_list)
        n_good_df   = pd.Series(n_good_radars,df.index)
    else:
        n_good_df   = np.sum(np.isfinite(df),axis=1)

    data_arr    = np.array(df)
    if reduction_type == 'median':
        red_vals    = sp.nanmedian(data_arr,axis=1)
    elif reduction_type == 'mean':
        n_dt,n_rdr  = data_arr.shape
        red_vals    = []
        for dt_inx in range(n_dt):
            tmp_vals    = data_arr[dt_inx,:]
            tf          = np.isfinite(tmp_vals)
            tmp_vals    = tmp_vals[tf]
            red_val     = scipy.stats.circmean(tmp_vals,low=0.,high=360.)
            red_vals.append(red_val)

    ts  = pd.Series(red_vals,df.index)
    return {'red_mstid_azm':ts,'n_good_df':n_good_df}

def calculate_reduced_mstid_index(music_groups,val_key='meanSubIntSpect_by_rtiCnt',
        months=[11,12,1,2,3,4],hours=[14,16,18,20],
        reduction_type='mean',daily_vals=True,zscore=True,
        db_name='mstid',mongo_port=27017):
    """
    Reduce the MSTID index from all radars into a single number as a function of time.
    """
    print("Calulating reduced MSTID index.")

    mongo       = pymongo.MongoClient(port=mongo_port)
    db          = mongo[db_name]

    mstid_inx_dict  = {} # Create a place to store the data.
    radars          = []

    for music_group in music_groups:
        # Work with one year at a time.
        date_str    = run_helper.get_seDates_from_groups(music_group,date_fmt='%Y%m%d')
        for radar_bank_inx, radar_bank_dict in music_group.items():
            # Work with either the high or mid latitudes
            bank_name   = radar_bank_dict['name']
            for radar_dict in radar_bank_dict['dct_list']:
                # Work with one radar at a time
                mstid_list      = radar_dict['mstid_list']
                print('Reducing MSTID Index: {} - {}'.format(bank_name,mstid_list))
                events          = db[mstid_list].find()
                # Go get the actual data.
                for event in events:
                    this_sTime  = event.get('sDatetime')
                    if this_sTime.month not in months:
                        continue

                    if this_sTime.hour not in hours:
                        continue

                    this_dct = mstid_inx_dict.get(this_sTime)
                    if this_dct is None:
                        this_dct    = {}
                        mstid_inx_dict[this_sTime] = this_dct

                    # Keep track of which radars are being used.
                    radar       = event['radar']
                    if radar not in radars:
                        radars.append(radar)
                    mstid_index = event.get(val_key,np.nan)

                    this_dct[radar] = mstid_index

    # Check to make sure all entries have all radars represented.
    for this_sTime,this_dct in mstid_inx_dict.items():
        for radar in radars:
            if radar not in this_dct:
                this_dct[radar] = np.nan

    # Put everything into a dataframe.
    df          = pd.DataFrame(mstid_inx_dict).transpose()
    
    if daily_vals:
        date_list   = np.unique([datetime.datetime(x.year,x.month,x.day) for x in df.index])

        tmp_list        = []
        n_good_radars   = []    # Set up a system for parameterizing data quality.
        for tmp_sd in date_list:
            tmp_ed      = tmp_sd + datetime.timedelta(days=1)

            tf          = np.logical_and(df.index >= tmp_sd, df.index < tmp_ed)
            tmp_df      = df[tf]
            if reduction_type == 'median':
                tmp_vals    = tmp_df.median().to_dict()
            elif reduction_type == 'mean':
                tmp_vals    = tmp_df.mean().to_dict()

            tmp_list.append(tmp_vals)

            n_good  = np.count_nonzero(np.isfinite(tmp_df))
            n_good_radars.append(n_good)

        df = pd.DataFrame(tmp_list,index=date_list)
        n_good_df   = pd.Series(n_good_radars,df.index)
    else:
        n_good_df   = np.sum(np.isfinite(df),axis=1)

    data_arr    = np.array(df)
    if reduction_type == 'median':
        red_vals    = sp.nanmedian(data_arr,axis=1)
    elif reduction_type == 'mean':
        red_vals    = np.nanmean(data_arr,axis=1)

    ts  = pd.Series(red_vals,df.index)
    if zscore:
        ts  = (ts - ts.mean())/ts.std()
    return {'red_mstid_index':ts,'n_good_df':n_good_df}

def plot_the_drivers(sDate,eDate,driver_list,
            driver_panel_frac,ax_left,ax_width,ax_top,h_pad,
            output_dir,mstid_reduced_inx,correlate=False,xlabels=False,
            driver_ylabel_fontdict=driver_ylabel_fontdict,ylim_dict={}):

    smooth_win              = datetime.timedelta(days=4)
    smooth_kind             = 'mean'

    fig = plt.gcf()
    this_panel_frac     = driver_panel_frac / len(driver_list)
    driver_objs     = []
    overlay_objs    = []
    for this_driver in driver_list:
        overlay_obj = None
        if this_driver is None:
            driver_objs.append(None)
            overlay_objs.append(None)
            continue
        else:
            load_dct                        = {'var_code':this_driver}
            load_dct['sDate']               = sDate
            load_dct['eDate']               = eDate
            load_dct['smooth_win']          = smooth_win
            load_dct['smooth_kind']         = smooth_kind
            driver_obj                      = drivers.get_driver_obj(**load_dct)

        if this_driver == 'mstid_inx':
            mstid_scores    = mongo_tools.get_mstid_scores(sDate,eDate)

        driver_objs.append(driver_obj)
        overlay_objs.append(overlay_obj)
                
    axs = []
    for driver_inx,(this_driver,driver_obj,overlay_obj) in enumerate(zip(driver_list,driver_objs,overlay_objs)):
        if this_driver is None: continue

        # Driver Parameter #############################################################
        ax_height   = this_panel_frac - h_pad
        ax_bottom   = ax_top - ax_height - h_pad
        ax_top      = ax_bottom
        ax          = fig.add_axes((ax_left,ax_bottom,ax_width,ax_height))
        axs.append(ax)

        sd_0    = {'lw':4}
        sd_1    = {'color':'0.50','zorder':-1}
        if hasattr(driver_obj,'cal_plot_ts'):
            plot_dct    = driver_obj.cal_plot_ts[0]
            data_set_0  = plot_dct.get('data_set_0')
            do_0        = driver_obj.get_data_set(data_set_0)
            
            pli_0       = do_0.plot_info.copy()
            tmp         = plot_dct.get('pli_0',{})
            pli_0.update(tmp)

            tmp         = plot_dct.get('sd_0',{})
            sd_0.update(tmp)


            data_set_1  = plot_dct.get('data_set_1')
            do_1        = driver_obj.get_data_set(data_set_1)

            pli_1       = do_1.plot_info.copy()
            tmp         = plot_dct.get('pli_1',{})
            pli_1.update(tmp)

            tmp         = plot_dct.get('sd_1',{})
            sd_1.update(tmp)
        else:
            do_0        = driver_obj.active
            data_set_0  = do_0.plot_info['data_set']
            pli_0       = do_0.plot_info.copy()

            do_1        = None
            data_set_1  = None
            pli_1       = None

        lines   = []
        if do_0 is not None:
            xx          = do_0.data.index.to_pydatetime()
            yy          = do_0.data.values
            lbl_0       = pli_0.get('legend_label')
            do_0_line,  = ax.plot(xx,yy,label=lbl_0,**sd_0)
            lines.append(do_0_line)

        if do_1 is not None:
            xx          = do_1.data.index.to_pydatetime()
            yy          = do_1.data.values
            lbl_1       = pli_1.get('legend_label')
            do_1_line,  = ax.plot(xx,yy,label=lbl_1,**sd_1)
            lines.append(do_1_line)

        if this_driver == 'mstid_reduced_azm':
            mu  = do_0.data.mean()
            ax.axhline(mu,ls='--',color='r',lw=1.5)

        if this_driver == 'mstid_reduced_azm_dev':
            mu  = do_0.data.mean()
            ax.axhline(mu,ls='--',color='r',lw=1.5)

        if this_driver == 'mstid_inx':
            # Add the number of good radars.
            ax2 = ax.twinx()
            ax.set_zorder(ax2.get_zorder()+1)
            ax.patch.set_visible(False)
            n_good_radars   = getattr(driver_obj.n_good_radars,data_set_0)
            ax2_xvals       = n_good_radars.data.index.to_pydatetime()
            ax2_yvals       = n_good_radars.data.values
            ylbl            = 'n Data Points'
            tmp,            = ax2.plot(ax2_xvals,ax2_yvals,color='k',zorder=0.5,lw=3,ls='--',label=ylbl)
            lines.append(tmp)

            ytls = ax2.get_yticklabels()
            for ytl in ytls:
                ytl.set_fontsize(24)

            for tick in ax2.get_yaxis().get_major_ticks():
                tick.set_pad(10.)
                tick.label1 = tick._get_text1()

            fontdict    = driver_ylabel_fontdict
            labelpad    = 15
            txt         = n_good_radars.plot_info['ind_0_gme_label']
            ax2.set_ylabel(ylbl,fontdict=fontdict,labelpad=labelpad)

            txt         = '(Dashed Line)'
            fontdict    = {'weight': 'normal','style':'italic', 'size': 24}
            ax2.text(1.029,0.5,txt,fontdict=fontdict,transform=ax.transAxes,
                    rotation=90.,va='center')


            # Shade the plot to indicate MSTID vs Quiet Periods
            ax.axhline(0,ls='--',color='r',lw=1.5)

            verts   = []
            vals    = []
            for tmp_t0,score in mstid_scores.iterrows():
                tmp_t1  = tmp_t0 + datetime.timedelta(days=1)
                tmp_t0  = matplotlib.dates.date2num(tmp_t0)
                tmp_t1  = matplotlib.dates.date2num(tmp_t1)

                vals.append(int(score))

                x1,y1   = tmp_t0,0
                x2,y2   = tmp_t0,1
                x3,y3   = tmp_t1,1
                x4,y4   = tmp_t1,0
                verts.append( ((x1,y1),(x2,y2),(x3,y3),(x4,y4),(x1,y1)) )

#            abs_max     = np.max(np.abs(vals))
            abs_max     = 30.
#            cmap        = matplotlib.cm.seismic
            cmap        = gl.get_custom_cmap('blue_red')
            bounds      = np.linspace(-abs_max,abs_max,2*abs_max+1)
            norm        = matplotlib.colors.BoundaryNorm(bounds,cmap.N)

            ytransaxes = matplotlib.transforms.blended_transform_factory(ax.transData,ax.transAxes)
            pcoll = PolyCollection(np.array(verts),edgecolors='face',linewidths=0,closed=False,
                    cmap=cmap,norm=norm,zorder=0,transform=ytransaxes,alpha=0.3)
            pcoll.set_array(np.array(vals))
            ax.add_collection(pcoll,autolim=False)

            box         = ax.get_position()
            axColor     = plt.axes([(box.x0 + box.width) * 1.045 , box.y0, 0.030, box.height])
            cbar        = plt.colorbar(pcoll,orientation='vertical',cax=axColor)
            fontdict    = cbar_title_fontdict
            fontdict    = {'weight': 'bold', 'size': 24}
            cbar.set_label('Daily MSTID\nScore',fontdict=fontdict,labelpad=-40)

            cbar_ticks  = [-abs_max,abs_max]
            cbar.set_ticks(cbar_ticks)

            caxta       = matplotlib.transforms.blended_transform_factory(axColor.transAxes,axColor.transData)
            axColor.plot([0.,1.],[0.5,0.5],color='k',ls='--',transform=caxta,clip_on=False,lw=2)

            xpos    = 0.5
#            txts    = [(0.77,'MSTID\nDays'),
#                       (0.23,'Quiet\nDays')]

            txts    = [(0.77,'MSTID\nActive'),
                       (0.23,'MSTID\nQuiet')]
            fontdict = {'weight': 'bold', 'size': 28}
            for ypos,txt in txts:
                axColor.text(xpos,ypos,txt,ha='center',va='center',
                        rotation=90.,
                        fontdict=fontdict,transform=caxta)

            labels = cbar.ax.get_yticklabels()
            fontweight  = cbar_ytick_fontdict.get('weight')
            fontsize    = cbar_ytick_fontdict.get('size')
            for label in labels:
                if fontweight:
                    label.set_fontweight(fontweight)
                if fontsize:
                    label.set_fontsize(fontsize)

        if this_driver == 'polar_vortex':
            ax.axhline(0,ls='--',color='r',lw=1.5)

        fontdict    = driver_ylabel_fontdict
        labelpad    = 15

        ax.set_ylabel(do_0.plot_info['ind_0_gme_label'],fontdict=fontdict,labelpad=labelpad)
        ax.set_title(do_0.plot_info['title'],fontdict=title_fontdict)

        for tick in ax.get_yaxis().get_major_ticks():
            tick.set_pad(10.)
            tick.label1 = tick._get_text1()

        ytls = ax.get_yticklabels()
        for ytl in ytls:
            ytl.set_fontsize(24)

        if xlabels == True:
            nd = len(driver_list)
            if driver_inx == nd-1:
                _xlabels = True
            else:
                _xlabels = False
        else:
            _xlabels = False
        my_xticks(sDate,eDate,ax,labels=_xlabels)

        ylim = ylim_dict.get(this_driver)
        ax.set_ylim(ylim)

        ax.legend(handles=lines,loc='lower left',fontsize=20,ncol=len(lines))

    ax_info = {'axs':axs,'ax_top':ax_top}
    return ax_info

def calendar_plot_vortex_movie_strip(plot_list,frame_times=None,keograms=None,file_suffix='',save_pdf=False,
        paper_legend=False,plot_letters=True):
    letter_inx      = 0
    letter_fontdict = {'weight':'bold', 'size':48}
    letter_xpos     = -0.101
    letter_ypos     = 0.5

#    keograms = []
#    # Lon is in 2.5 deg increments from [0,357.5]
#    keograms.append({'lon_0':-107.5,'lon_1':-107.5,'lat_0':40.,'lat_1':90.,'mbar':1})
#    keograms.append({'lon_0':-107.5,'lon_1':-107.5,'lat_0':40.,'lat_1':90.,'mbar':10})

#    keograms.append({'lon_0':-152,'lon_1':-152,'lat_0':40.,'lat_1':90.,'mbar':10})

    # Pull out all of the grib data and place it in its own dict.
    plot_list_copy  = copy.deepcopy(plot_list)
    grib_data_days  = {}
    for frame in plot_list_copy:
        dt                  = frame['dt']
        grib_data           = frame.pop('grib_data')
        grib_data_days[dt]  = grib_data

    kwargs = plot_list_copy[0]

    # This function is derived from calendar_plot_with_polar_data().  The following section
    # replicates the call for that function.
    dct_list            = kwargs.pop('dct_list',None)
    group_dict          = kwargs.pop('group_dict',None)
    sDate               = kwargs.pop('sDate',None)
    eDate               = kwargs.pop('eDate',None)
    val_key             = kwargs.pop('val_key','meanSubIntSpect_by_rtiCnt')
    scale               = kwargs.pop('scale',[-0.03,0.03])
    st_uts              = kwargs.pop('st_uts',[14, 16, 18, 20])
    driver              = kwargs.pop('driver',None)
    output_dir          = kwargs.pop('output_dir','mstid_data/calendar')
    db_name             = kwargs.pop('db_name','mstid')
    mongo_port          = kwargs.pop('mongo_port',27017)
    dt                  = kwargs.pop('dt',None)
    mstid_reduced_inx   = kwargs.pop('mstid_reduced_inx',None)
    correlate           = kwargs.pop('correlate',None)
    highlight_ew        = kwargs.pop('highlight_ew',False)

    driver_list         = gl.get_iterable(driver)
    if correlate:
        driver_list = [driver_list[0],'mstid_reduced_inx','correlate']

    this_date   = dt
    filename    = '{}{}_vortex_calendar.png'.format(this_date.strftime('%Y%m%d_%H%M'),file_suffix)
    filepath    = os.path.join(output_dir,filename)
    print('Plotting: {}'.format(filepath))
    
    if dct_list is not None:
        group_dict      = {}
        group_dict[0]   = {}
        group_dict[0]['dct_list']   = dct_list

    prepare_output_dirs({0:output_dir},clear_output_dirs=False)

    sDate,eDate = get_sDate_eDate(group_dict,sDate,eDate)
    xmax        = get_xmax(sDate,eDate)

    # Plotting section. ############################################################
    ax_list = []

    ax_nr   = 0
    ax_nx   = 1
    ax_ny   = len(list(group_dict.keys()))

    ax_left             = 0.
    ax_width            = 1.0
    ax_top              = 1.0
    h_pad               = 0.050
    w_pad               = 0.025

    fig_scale_x         = 1.
    fig_scale_y         = 0.225
    sup_title_bump      = 0.

    ax_ny += 1
    if driver is None:
        grib_panel_frac     = 0.50
        radar_panel_frac    = 0.50

    if driver is not None and keograms is None:
        grib_panel_frac     = 0.200
        driver_panel_frac   = 0.350
        radar_panel_frac    = 0.450

    if driver is not None and keograms is not None:
        grib_panel_frac     = 0.150
        keogram_panel_frac  = 0.225
        driver_panel_frac   = 0.225
        radar_panel_frac    = 0.400

        h_pad               = 0.030
        fig_scale_x         = 1.
        fig_scale_y         = 0.300
        sup_title_bump      = 0.010

    fig_scale   = 40.
    figsize     = (fig_scale*fig_scale_x,fig_scale*fig_scale_y*ax_ny)
    fig         = plt.figure(figsize=figsize)

    ret_info    = plot_geopot_movie_strip(grib_data_days,sDate,eDate,frame_times,
                    grib_panel_frac,ax_left,ax_width,ax_top,h_pad,w_pad,fig,
                    keograms=keograms)

    ax_top      = ret_info['ax_top']
    ax          = ret_info['ax']
    if plot_letters:
        ax.text(letter_xpos,letter_ypos,'({})'.format(string.lowercase[letter_inx]),transform=ax.transAxes,fontdict=letter_fontdict)
        letter_inx  += 1

    if keograms is not None:
        for keogram in keograms:
            geo_param   = 'raw'
            mbar        = keogram.get('mbar')
            lon_0       = keogram.get('lon_0')
            lat_0       = keogram.get('lat_0')
            lon_1       = keogram.get('lon_1')
            lat_1       = keogram.get('lat_1')

            this_panel_frac = keogram_panel_frac / len(keograms)
            ax_height       = this_panel_frac - h_pad
            ax_bottom       = ax_top - ax_height - h_pad
            ax_top          = ax_bottom
            ax              = fig.add_axes((ax_left,ax_bottom,ax_width,ax_height))

            if plot_letters:
                ax.text(letter_xpos,letter_ypos,'({})'.format(string.lowercase[letter_inx]),transform=ax.transAxes,fontdict=letter_fontdict)
                letter_inx  += 1

            grib_times  = list(grib_data_days.keys())
            grib_times.sort()
            
            xvec,yvec,vals  = (None, None, None)
            for grib_time in grib_times:
                x_0 = get_x_coords(grib_time,sDate,eDate,full=True)

                grib_data   = grib_data_days[grib_time]
                grb_d       = grib_data[mbar]
                grb         = grb_d[geo_param]

                tf          = grb['latlon'][1] == (lon_0 % 360.)

                vals_slice  = np.array([grb['values'][tf]])
                xvec_slice  = vals_slice*0. + x_0
                yvec_slice  = np.array([grb['latlon'][0][tf]])

                if xvec is None:
                    xvec    = xvec_slice
                    yvec    = yvec_slice
                    vals    = vals_slice
                else:
                    xvec    = np.concatenate((xvec,xvec_slice))
                    yvec    = np.concatenate((yvec,yvec_slice))
                    vals    = np.concatenate((vals,vals_slice))

            vmin    = grb['scale'][0]
            vmax    = grb['scale'][1]
            pcoll   = ax.pcolor(xvec,yvec,vals,vmin=vmin,vmax=vmax)
            ax.set_ylim(lat_0,lat_1)
            for ytl in ax.get_yticklabels():
                ytl.set_fontsize(ytick_major_fontdict.get('size'))
                ytl.set_fontweight(ytick_major_fontdict.get('weight'))
            ax.set_ylabel('Lat [deg]',fontdict=driver_ylabel_fontdict)
            txt     = 'ECMFW Geopotential Keogram ({:0.1f}'.format(lon_0) + r'$^{\circ}$' \
                       + ' E lon; {!s} mb Level)'.format(mbar)
            ax.set_title(txt,fontdict=title_fontdict)


            my_xticks(sDate,eDate,ax,radar_ax=True,labels=False)

            box         = ax.get_position()
            axColor     = plt.axes([(box.x0 + box.width) * 1.01 , box.y0, 0.01, box.height])
            cbar        = plt.colorbar(pcoll,orientation='vertical',cax=axColor)

            cbar_label  = []
            cbar_label.append(grb['shortName'].upper() + ' [' + grb['units'] + ']')
            cbar_label.append('{!s} mb lvl'.format(mbar))
            cbar.set_label('\n'.join(cbar_label),fontdict=cbar_title_fontdict,labelpad=10)

            clim        = cbar.get_clim()
            nticks      = 9
            cbar_ticks  = np.linspace(*clim,num=nticks)
            cbar_ticks  = cbar_ticks[[1,nticks/2,-2]]
            cbar.set_ticks(cbar_ticks)

            expon       = 5
            cbar_ticklabels = []
            for cbt in cbar_ticks:
                lbl = '{:0.1f}E{!s}'.format(cbt/(10**expon),expon)
                cbar_ticklabels.append(lbl)
            cbar.ax.set_yticklabels(cbar_ticklabels,fontdict=cbar_ytick_fontdict)

            cbar_ytls   = cbar.ax.get_yticklabels()
            for label in cbar_ytls:
                label.set_fontweight(cbar_ytick_fontdict.get('weight'))
                label.set_fontsize(cbar_ytick_fontdict.get('size'))

    if driver is not None:
        ax_info = plot_the_drivers(sDate,eDate,driver_list,
                driver_panel_frac,ax_left,ax_width,ax_top,h_pad,
                output_dir,mstid_reduced_inx,correlate)

        ax_top  = ax_info['ax_top']

        for ax in ax_info['axs']:
            ax.axvline(this_date,ls='--',lw=6,color='k',zorder=750)

            if plot_letters:
                ax.text(letter_xpos,letter_ypos,'({})'.format(string.lowercase[letter_inx]),transform=ax.transAxes,fontdict=letter_fontdict)
                letter_inx  += 1

    # Radar MSTID Data ############################################################# 
    keys    = list(group_dict.keys())
    keys.sort()
    nr_groups   = len(keys)
    radars      = []
    for key in keys:
        dct_list    = group_dict[key]['dct_list']
        group_name  = group_dict[key].get('name')
        radars  += [dct['radar'] for dct in dct_list]

        ax_height   = get_radar_ax_frac(key,group_dict) * (radar_panel_frac - h_pad*nr_groups)
        ax_bottom   = ax_top - ax_height - h_pad
        ax_top      = ax_bottom
        ax          = fig.add_axes((ax_left,ax_bottom,ax_width,ax_height))

        if plot_letters:
            ax.text(letter_xpos,letter_ypos,'({})'.format(string.lowercase[letter_inx]),transform=ax.transAxes,fontdict=letter_fontdict)
            letter_inx  += 1

        if key == max(keys):
            xlabels = True
        else:
            xlabels = False
        ax_info     = plot_calendar_panel(dct_list,sDate,eDate,scale,st_uts,val_key,ax,
                xlabels=xlabels,db_name=db_name,mongo_port=mongo_port,group_name=group_name,
                highlight_ew=highlight_ew)

        vline_xpos = get_x_coords(this_date,sDate,eDate,full=True)
        ax.axvline(vline_xpos,ls='--',lw=6,color='k',zorder=750)

        ax_list.append(ax_info)

    plot_cbars(ax_list)

    date_fmt    = '%d %b %Y'
    date_str    = '{} - {}'.format(sDate.strftime(date_fmt),eDate.strftime(date_fmt))
    fig.text(0.5,0.970+sup_title_bump,date_str,ha='center',fontdict={'weight':'bold','size':48})

    if paper_legend:
        rect    = [0.155,-0.32,0.010,0.25]
        calendar_plot_key(fig,rect,group_dict)

    fig.savefig(filepath,bbox_inches='tight')
    if save_pdf:
        fig.savefig(filepath[:-3]+'pdf',bbox_inches='tight')
    plt.close(fig)

    return filepath

def plot_geopot_movie_strip(grib_data_days,sDate,eDate,frame_times,
        grib_panel_frac,ax_left,ax_width,ax_top,h_pad,w_pad,fig,
        keograms=None):

    # Geopotential Parameter #######################################################
    ax_height   = grib_panel_frac - h_pad
    ax_bottom   = ax_top - ax_height - h_pad
    ax_top      = ax_bottom
    ax_0        = fig.add_axes((ax_left,ax_bottom,ax_width,ax_height))
    ax_0.set_aspect('equal','box')
    ax_0.set_ylim(0,15)
    ytls    = ax_0.get_yticklabels()
    for ytl in ytls:
        ytl.set_visible(False)

    my_xticks(sDate,eDate,ax_0,radar_ax=True)

    grib_times  = list(grib_data_days.keys())
    grib_times.sort()

    td_width    = datetime.timedelta(days=15)
    for grib_time in grib_times:
        if frame_times is not None:
            if grib_time not in frame_times:
                continue
        x_0 = get_x_coords(grib_time,sDate,eDate)
        x_1 = get_x_coords(grib_time+td_width,sDate,eDate)

        d_0    = ax_0.transData.transform([x_0,0])
        d_1    = ax_0.transData.transform([x_1,15])

        inv     = fig.transFigure.inverted()
        f_0     = inv.transform(d_0)
        f_1     = inv.transform(d_1)

        xp,yp   = f_0
        wd,ht   = f_1-f_0

        ax_m    = fig.add_axes([xp,yp,wd,ht])

        grib_data   = grib_data_days[grib_time]
        geo_param   = 'raw'

        mappables   = {}
        mbar_levels = list(grib_data.keys())
        mbar_levels.sort()
        for mbar_inx,mbar_level in enumerate(mbar_levels):
            grb_d       = grib_data[mbar_level]
            grb         = grb_d[geo_param]
            if mbar_inx == 0:
                m   = None
                txt = grb.get('big_title')

#                ax_m.text(0.5,1.1,txt,ha='center',transform=ax.transAxes,
#                        fontdict={'weight':'bold','size':48})

            m,mappable  = polar_met.plot_grb_ax(grb,ax_m,m,
                    plot_colorbars=False,plot_title=False,plot_latlon_labels=False,
                    return_mappable=True)

            if keograms is not None:
                for keo_inx,keogram in enumerate(keograms):
                    lon_0   = keogram.get('lon_0')
                    lat_0   = keogram.get('lat_0')
                    lon_1   = keogram.get('lon_1')
                    lat_1   = keogram.get('lat_1')

                    keo_lbl     = '{:0.1f}'.format(lon_0) + r'$^{\circ}$' + ' E lon'
                    keo_line,   = m.drawgreatcircle(lon_0,lat_0,lon_0,lat_1,color='g',lw=5,label=keo_lbl)

                    keogram['keo_line'] = keo_line

            mappables[mbar_level] = mappable

#            # Stick grib data into its own figure.
#            ax_info_grib_only =  geo_prm_ax_grib_only.get(geo_param)
#            if ax_info_grib_only is None:
#                grb_fig = plt.figure(figsize=(15,15))
#                grb_ax  = grb_fig.add_subplot(111)
#
#                grb_m   = polar_met.plot_grb_ax(grb,grb_ax)
#
#                ax_info_grib_only                 = {}
#                ax_info_grib_only['ax']           = grb_ax
#                ax_info_grib_only['m']            = grb_m
#                geo_prm_ax_grib_only[geo_param]   = ax_info_grib_only
#            else:
#                grb_ax  = ax_info_grib_only.get('ax')
#                grb_m   = ax_info_grib_only.get('m')
#
#                polar_met.plot_grb_ax(grb,grb_ax,m=grb_m)
#
#                grib_only_dir   = os.path.join(output_dir,'grib_only')
#                grib_only_fname = '{}_{}_grib.png'.format(this_date.strftime('%Y%m%d_%H%M'),geo_param)
#                grib_only_fpath = os.path.join(grib_only_dir,grib_only_fname)
#                if not os.path.exists(grib_only_dir):
#                    prepare_output_dirs({0:grib_only_dir})
#
#                grb_fig.savefig(grib_only_fpath,bbox_inches='tight')
#                plt.close(grb_fig)

    plot_cbars_grib(mappables,ax_0)
    if keograms is not None:
        handles = []
        labels  = []
        for keogram in keograms:
            lbl = keogram['keo_line'].get_label()
            if lbl in labels: continue
            handles.append(keogram['keo_line'])
            labels.append(lbl)

        ax_0.legend(handles,labels,loc=(0.890,1.030),prop=keo_legend_fontdict)

    ax_0.set_title('ECMWF Geopotential',fontdict=title_fontdict)

#    for mbar_level in mbar_levels:
#        plot_dct    = mappables[mbar_level]['plot_dct']
#        pcoll       = mappables[mbar_level]['pcoll']
#
#        cbar        = fig.colorbar(pcoll,orientation='vertical',shrink=.55,fraction=.1,
#                ax=ax_0)
#        cbar_label  = plot_dct['shortName'].upper() + ' [' + plot_dct['units'] + ']'
#        cbar.set_label(cbar_label,fontdict={'weight':'bold','size':'x-large'})

    ret_info = {'ax_top':ax_top,'ax':ax_0}
    return ret_info

def calendar_plot_key(fig,rect,boxes,arrow_ypos,radar_groups):
    import mpl_toolkits.axes_grid.axes_size as Size
    from mpl_toolkits.axes_grid import Divider
    from matplotlib.patches import Rectangle

    ny_plots    = len(list(radar_groups.keys()))

    nr_rads = []
    for key in range(ny_plots):
        rdr_grp     = radar_groups[key]
        grp_name    = rdr_grp['name']
        radars      = [x['radar'] for x in rdr_grp['dct_list']]
        nr_rads.append(len(radars))

    v_pad       = 0.30
    horiz       = [Size.Scaled(1.0)]
    vert        = [Size.Scaled(nr_rads[1]), Size.Fixed(v_pad),Size.Scaled(nr_rads[0])]

    div         = Divider(fig,rect,horiz,vert,aspect=False)

    pos         = {}
    pos[0]      = div.new_locator( nx=0, ny=2)
    pos[1]      = div.new_locator( nx=0, ny=0)

    info_dct    = {}
    info_dct[0] = {'ylabel':'14-16\nUT','region':'High\nLat.\nRadars'}
    info_dct[1] = {'ylabel':'20-22\nUT','region':'Mid\nLat.\nRadars'}

    for key in range(ny_plots):
        rdr_grp     = radar_groups[key]
        grp_name    = rdr_grp['name']
        radars      = [x['radar'] for x in rdr_grp['dct_list']]

        ax  = fig.add_subplot(1,ny_plots,key+1)
        ax.plot([0],[0])
        
        xmax    = 1
        ax.set_xlim(0,xmax)
        ax.set_xticks(np.arange(0,xmax+1))
        
        ax.set_ylim(0,len(radars))
        ax.set_yticks(np.arange(0,len(radars)))

        ax.grid(True,lw=2,ls='-')

        for xtl in ax.get_xticklabels():
            xtl.set_visible(False)

        for ytl in ax.get_yticklabels():
            ytl.set_visible(False)

        trans   = matplotlib.transforms.blended_transform_factory(ax.transAxes,ax.transData)
        for rdr_inx,radar in enumerate(radars):
            ax.text(-0.4,rdr_inx+0.5,radar.upper(),va='center',ha='right',
                    transform=trans,fontsize=28)

        for ypos in [0.,1.]:
            xpos    = -4.00
            line    = ax.hlines(ypos,xpos,1.0,transform=ax.transAxes,lw=3,zorder=100)
            line.set_clip_on(False)

#        ylabel  = info_dct[key]['ylabel']
#        ax.text(-6.00,0.5,ylabel,fontdict={'weight':'bold','size':32},
#                transform=ax.transAxes,ha='center',va='center')

        ylabel  = info_dct[key]['region']
        ax.text(1.5,0.5,ylabel,fontdict={'weight':'bold','size':30},
                transform=ax.transAxes,ha='left',va='center')

        ax.set_axes_locator(pos[key])

    br_x0   = -0.170
    br_x1   =  0.2
    br_y0   = rect[1] - 0.025
    br_y1   = rect[1] + rect[3] + 0.025

    br_wd   = br_x1 - br_x0
    br_ht   = br_y1 - br_y0
    big_rect = [br_x0,br_y0,br_wd,br_ht]

    big_ax = fig.add_axes(big_rect,frameon=False)

    for box_dct in boxes:
        box     = box_dct['box']
        patch   = Rectangle( (box[0], box[1]), box[2], box[3], fill=False, lw=2, clip_on=False)
        big_ax.add_patch(patch)

        box_left    = box[0]                # xcoord of left edge
        box_right   = box[0]+box[2]         # xcoord of right edge
        box_midy    = box[1] + box[3]/2.
        box_dct['box_left']     = box_left
        box_dct['box_right']    = box_right
        box_dct['box_midy']     = box_midy

    lw      = 3.
    color   = 'k'

    ypos    = boxes[1]['box_midy']
    ypos    = arrow_ypos
    x0  = boxes[0]['box_right'] 
    y0  = ypos
    x1  = boxes[1]['box_left'] 
    y1  = ypos

    big_ax.annotate("",
                xy=(x0, y0), xycoords=big_ax.transAxes,
                xytext=(x1, y1), textcoords=big_ax.transAxes,
                arrowprops=dict(arrowstyle="->,head_width=2.5,head_length=2.0", #linestyle="dashed",
                                color=color,
                                lw=5,
                                shrinkA=0, shrinkB=5,
                                patchA=None,
                                patchB=None,
                                ),
                )


    big_ax.set_xlim(0,1)
    big_ax.set_ylim(0,1)

    for xtl in big_ax.get_xticklabels():
        xtl.set_visible(False)

    for ytl in big_ax.get_yticklabels():
        ytl.set_visible(False)

    big_ax.set_xticks([])
    big_ax.set_yticks([])
    
#    big_ax.spines['left'].set_visible(False)
#    big_ax.spines['right'].set_visible(False)
#    big_ax.spines['top'].set_visible(False)
#    big_ax.spines['bottom'].set_visible(False)

#    for xtl in big_ax.get_xticks():
#        xtl.set_visible(False)
#
#    for ytl in big_ax.get_yticks():
#        ytl.set_visible(False)

def calendar_plot_vortex_movie_strip_paper(plot_list,frame_times=None,file_suffix='',
        save_pdf=False,paper_legend=False,plot_letters=True,plot_geopot_maps=True,rasterized=False,dpi=400,**kwargs):
    letter_inx      = 0
    letter_fontdict = {'weight':'bold', 'size':48}
    letter_xpos     = -0.095
    letter_ypos     = 0.5

    # Pull out all of the grib data and place it in its own dict.
    plot_list_copy  = copy.deepcopy(plot_list)
    grib_data_days  = {}
    for frame in plot_list_copy:
        dt                  = frame['dt']
        grib_data           = frame.pop('grib_data')
        grib_data_days[dt]  = grib_data

    kwargs = plot_list_copy[0]

    # This function is derived from calendar_plot_with_polar_data().  The following section
    # replicates the call for that function.
    dct_list            = kwargs.pop('dct_list',None)
    group_dict          = kwargs.pop('group_dict',None)
    sDate               = kwargs.pop('sDate',None)
    eDate               = kwargs.pop('eDate',None)
    val_key             = kwargs.pop('val_key','meanSubIntSpect_by_rtiCnt')
    scale               = kwargs.pop('scale',[-0.03,0.03])
    st_uts              = kwargs.pop('st_uts',[14, 16, 18, 20])
    driver              = kwargs.pop('driver',None)
    output_dir          = kwargs.pop('output_dir','mstid_data/calendar')
    db_name             = kwargs.pop('db_name','mstid')
    mongo_port          = kwargs.pop('mongo_port',27017)
    dt                  = kwargs.pop('dt',None)
    mstid_reduced_inx   = kwargs.pop('mstid_reduced_inx',None)
    correlate           = kwargs.pop('correlate',None)
    highlight_ew        = kwargs.pop('highlight_ew',False)
    classification_colors   = kwargs.pop('classification_colors',False)

    driver_list         = gl.get_iterable(driver)
    if correlate:
        driver_list = [driver_list[0],'mstid_reduced_inx','correlate']

    this_date   = dt
    filename    = '{}{}_vortex_calendar.png'.format(this_date.strftime('%Y%m%d_%H%M'),file_suffix)
    filepath    = os.path.join(output_dir,filename)
    print('Plotting: {}'.format(filepath))
    
    if dct_list is not None:
        group_dict      = {}
        group_dict[0]   = {}
        group_dict[0]['dct_list']   = dct_list

    prepare_output_dirs({0:output_dir},clear_output_dirs=False)

    sDate,eDate = get_sDate_eDate(group_dict,sDate,eDate)
    xmax        = get_xmax(sDate,eDate)

    # Plotting section. ############################################################
    ax_list = []

    ax_nr   = 0
    ax_nx   = 1
    ax_ny   = len(list(group_dict.keys()))

    ax_left             = 0.
    ax_width            = 1.0
    ax_top              = 1.0
    h_pad               = 0.030
    w_pad               = 0.025

    fig_scale_x         = 1.
    fig_scale_y         = 0.180
    sup_title_bump      = 0.

    ax_ny += 1
    if driver is None:
        grib_panel_frac     = 0.50
        radar_panel_frac    = 0.50
    if driver == ['mstid_reduced_inx'] and not plot_geopot_maps:
        radar_panel_frac    = 0.40
        driver_panel_frac   = 0.20
        grib_panel_frac     = 0.0

        h_pad               = 0.025
        fig_scale_x         = 1.
        fig_scale_y         = 0.300
    elif len(driver) == 4 and not plot_geopot_maps:
        # Paper with 3 driver panels.
        radar_panel_frac    = 0.40
        driver_panel_frac   = 0.55
        grib_panel_frac     = 0.0

        h_pad               = 0.025
        fig_scale_x         = 1.
        fig_scale_y         = 0.300
    elif len(driver) == 3 and not plot_geopot_maps:
        # Paper with 3 driver panels.
        radar_panel_frac    = 0.40
        driver_panel_frac   = 0.45
        grib_panel_frac     = 0.0

        h_pad               = 0.025
        fig_scale_x         = 1.
        fig_scale_y         = 0.300
    elif len(driver) == 2 and plot_geopot_maps:
        # Paper with 3 driver panels.
        radar_panel_frac    = 0.40
        driver_panel_frac   = 0.30
        grib_panel_frac     = 0.15

        h_pad               = 0.025
        fig_scale_x         = 1.
        fig_scale_y         = 0.300
    else:
        radar_panel_frac    = 0.44
        driver_panel_frac   = 0.44
        grib_panel_frac     = 0.12

        h_pad               = 0.025
        fig_scale_x         = 1.
        fig_scale_y         = 0.300

    fig_scale   = 40.
    figsize     = (fig_scale*fig_scale_x,fig_scale*fig_scale_y*ax_ny)
    fig         = plt.figure(figsize=figsize)

    # Radar MSTID Data ############################################################# 
    keys    = list(group_dict.keys())
    keys.sort()
    nr_groups   = len(keys)
    radars      = []
    for key in keys:
        dct_list    = group_dict[key]['dct_list']
        group_name  = group_dict[key].get('name')
        radars  += [dct['radar'] for dct in dct_list]

        ax_height   = get_radar_ax_frac(key,group_dict) * (radar_panel_frac - h_pad*nr_groups)
        ax_bottom   = ax_top - ax_height - h_pad
        ax_top      = ax_bottom
        ax          = fig.add_axes((ax_left,ax_bottom,ax_width,ax_height))

        if plot_letters:
            ax.text(letter_xpos,letter_ypos,'({})'.format(string.lowercase[letter_inx]),transform=ax.transAxes,fontdict=letter_fontdict)
            letter_inx  += 1

        xlabels = False
        ax_info     = plot_calendar_panel(dct_list,sDate,eDate,scale,st_uts,val_key,ax,
                xlabels=xlabels,db_name=db_name,mongo_port=mongo_port,group_name=group_name,
                highlight_ew=highlight_ew,classification_colors=classification_colors,rasterized=rasterized)

        vline_xpos = get_x_coords(this_date,sDate,eDate,full=True)
        ax.axvline(vline_xpos,ls='--',lw=6,color='k',zorder=750)
        ax_list.append(ax_info)

    if driver is not None:
        ylim_dict   = {}
        ylim_dict['mstid_reduced_inx']      = (-2.5,2.5)
        ylim_dict['mstid_inx']              = (-2.5,2.5)
        ylim_dict['smoothed_ae']            = (-1400,0)
        ylim_dict['ae_proc_0']              = (-1400,0)
        ylim_dict['ae_proc_2']              = (1400,0)
        ylim_dict['omni_symh']              = (-150,100)
        ylim_dict['neg_mbar_diff']          = (-2.5,2.5)
        ylim_dict['mstid_reduced_azm']      = (270.,90.)
        ylim_dict['mstid_reduced_azm_dev']  = (-50., 50.)

        ax_info = plot_the_drivers(sDate,eDate,driver_list,
                driver_panel_frac,ax_left,ax_width,ax_top,h_pad,
                output_dir,mstid_reduced_inx,correlate,ylim_dict=ylim_dict,
                xlabels=True)

        ax_top  = ax_info['ax_top']

        for ax in ax_info['axs']:
            ax.axvline(this_date,ls='--',lw=6,color='k',zorder=750)

            if plot_letters:
                ax.text(letter_xpos,letter_ypos,'({})'.format(string.lowercase[letter_inx]),transform=ax.transAxes,fontdict=letter_fontdict)
                letter_inx  += 1

    if plot_geopot_maps:
        ret_info    = plot_geopot_movie_strip(grib_data_days,sDate,eDate,frame_times,
                        grib_panel_frac,ax_left,ax_width,ax_top,h_pad,w_pad,fig)

        ax_top      = ret_info['ax_top']
        ax          = ret_info['ax']

        if plot_letters:
            ax.text(letter_xpos,letter_ypos,'({})'.format(string.lowercase[letter_inx]),transform=ax.transAxes,fontdict=letter_fontdict)
            letter_inx  += 1

    plot_cbars(ax_list)

    date_fmt    = '%d %b %Y'
    date_str    = '{} - {}'.format(sDate.strftime(date_fmt),eDate.strftime(date_fmt))
    fig.text(0.5,1.000+sup_title_bump,date_str,ha='center',fontdict={'weight':'bold','size':48})

    if paper_legend:
        leg_scale   = 0.60
        rect        = [-0.140,0.7400,leg_scale*0.010,leg_scale*0.25]

        boxes       = []
        boxes.append({'box':[0.000,0.100,0.25,0.8]})  # Box around the key.
        boxes.append({'box':[0.425,0.195,0.05,0.6]})  # Box around actual calendar plot.
        arrow_ypos  = 0.535

        calendar_plot_key(fig,rect,boxes,arrow_ypos,group_dict)

    fig.savefig(filepath,bbox_inches='tight',dpi=dpi)
    if save_pdf:
        fig.savefig(filepath[:-3]+'pdf',bbox_inches='tight',dpi=dpi)
    plt.close(fig)

    return filepath

def polar_vortex_only(plot_list,frame_times=None,file_suffix='',
        save_pdf=False,paper_legend=False,plot_letters=True,plot_geopot_maps=True):
    letter_inx      = 0
    letter_fontdict = {'weight':'bold', 'size':48}
    letter_xpos     = -0.0385
    letter_ypos     = 0.920

    # Pull out all of the grib data and place it in its own dict.
    plot_list_copy  = copy.deepcopy(plot_list)
    grib_data_days  = {}
    for frame in plot_list_copy:
        dt                  = frame['dt']
        grib_data           = frame.pop('grib_data')
        grib_data_days[dt]  = grib_data

    kwargs = plot_list_copy[0]

    # This function is derived from calendar_plot_with_polar_data().  The following section
    # replicates the call for that function.
    dct_list            = kwargs.pop('dct_list',None)
    group_dict          = kwargs.pop('group_dict',None)
    sDate               = kwargs.pop('sDate',None)
    eDate               = kwargs.pop('eDate',None)
    val_key             = kwargs.pop('val_key','meanSubIntSpect_by_rtiCnt')
    scale               = kwargs.pop('scale',[-0.03,0.03])
    st_uts              = kwargs.pop('st_uts',[14, 16, 18, 20])
    driver              = kwargs.pop('driver',None)
    output_dir          = kwargs.pop('output_dir','mstid_data/calendar')
    db_name             = kwargs.pop('db_name','mstid')
    mongo_port          = kwargs.pop('mongo_port',27017)
    dt                  = kwargs.pop('dt',None)
    mstid_reduced_inx   = kwargs.pop('mstid_reduced_inx',None)
    correlate           = kwargs.pop('correlate',None)
    highlight_ew        = kwargs.pop('highlight_ew',False)
    classification_colors   = kwargs.pop('classification_colors',False)

    driver_list         = gl.get_iterable(driver)
    if correlate:
        driver_list = [driver_list[0],'mstid_reduced_inx','correlate']

    this_date   = dt
    filename    = '{}{}_polarvortex_timeseries.png'.format(this_date.strftime('%Y%m%d_%H%M'),file_suffix)
    filepath    = os.path.join(output_dir,filename)
    print('Plotting: {}'.format(filepath))
    
    if dct_list is not None:
        group_dict      = {}
        group_dict[0]   = {}
        group_dict[0]['dct_list']   = dct_list

    prepare_output_dirs({0:output_dir},clear_output_dirs=False)

    sDate,eDate = get_sDate_eDate(group_dict,sDate,eDate)
    xmax        = get_xmax(sDate,eDate)

    # Plotting section. ############################################################
    ax_list = []

    ax_nr   = 0
    ax_nx   = 1
    ax_ny   = len(list(group_dict.keys()))

    ax_left             = 0.
    ax_width            = 1.0
    ax_top              = 1.0
    h_pad               = 0.030
    w_pad               = 0.025

    fig_scale_x         = 1.
    fig_scale_y         = 0.180
    sup_title_bump      = 0.

    ax_ny += 1

    radar_panel_frac    = 0.00
    driver_panel_frac   = 0.15
    grib_panel_frac     = 0.165

    h_pad               = 0.025
    fig_scale_x         = 1.
    fig_scale_y         = 0.300

    fig_scale   = 40.
    figsize     = (fig_scale*fig_scale_x,fig_scale*fig_scale_y*ax_ny)
    fig         = plt.figure(figsize=figsize)

    if plot_geopot_maps:
        ret_info    = plot_geopot_movie_strip(grib_data_days,sDate,eDate,frame_times,
                        grib_panel_frac,ax_left,ax_width,ax_top,h_pad,w_pad,fig)

        ax_top      = ret_info['ax_top']
        ax          = ret_info['ax']

        if plot_letters:
            ax.text(letter_xpos,letter_ypos,'({})'.format(string.lowercase[letter_inx]),transform=ax.transAxes,fontdict=letter_fontdict)
            letter_inx  += 1

    if driver is not None:
        ylim_dict   = {}
        ylim_dict['mstid_reduced_inx']  = (-2.5,2.5)
        ylim_dict['mstid_inx']          = (-2.5,2.5)
        ylim_dict['smoothed_ae']        = (-1400,0)
        ylim_dict['ae_proc_0']          = (-1400,0)
        ylim_dict['omni_symh']          = (-250,100)
        ylim_dict['neg_mbar_diff']      = (-2.5,2.5)

        ax_info = plot_the_drivers(sDate,eDate,driver_list,
                driver_panel_frac,ax_left,ax_width,ax_top,h_pad,
                output_dir,mstid_reduced_inx,correlate,ylim_dict=ylim_dict,
                xlabels=True)

        ax_top  = ax_info['ax_top']

        for ax in ax_info['axs']:
            ax.axvline(this_date,ls='--',lw=6,color='k',zorder=750)

            if plot_letters:
                ax.text(letter_xpos,letter_ypos,'({})'.format(string.lowercase[letter_inx]),transform=ax.transAxes,fontdict=letter_fontdict)
                letter_inx  += 1

    plot_cbars(ax_list)

#    date_fmt    = '%d %b %Y'
#    date_str    = '{} - {}'.format(sDate.strftime(date_fmt),eDate.strftime(date_fmt))
#    fig.text(0.5,1.000+sup_title_bump,date_str,ha='center',fontdict={'weight':'bold','size':48})

    fig.savefig(filepath,bbox_inches='tight')
    if save_pdf:
        fig.savefig(filepath[:-3]+'pdf',bbox_inches='tight')
    plt.close(fig)

    return filepath

def drivers_only(plot_list,frame_times=None,file_suffix='',
        save_pdf=False,paper_legend=False,plot_letters=True,plot_geopot_maps=True):
    letter_inx      = 0
    letter_fontdict = {'weight':'bold', 'size':48}
    letter_xpos     = -0.090
    letter_ypos     = 0.5

    # Pull out all of the grib data and place it in its own dict.
    plot_list_copy  = copy.deepcopy(plot_list)
    grib_data_days  = {}
    for frame in plot_list_copy:
        dt                  = frame['dt']
        grib_data           = frame.pop('grib_data')
        grib_data_days[dt]  = grib_data

    kwargs = plot_list_copy[0]

    # This function is derived from calendar_plot_with_polar_data().  The following section
    # replicates the call for that function.
    dct_list            = kwargs.pop('dct_list',None)
    group_dict          = kwargs.pop('group_dict',None)
    sDate               = kwargs.pop('sDate',None)
    eDate               = kwargs.pop('eDate',None)
    val_key             = kwargs.pop('val_key','meanSubIntSpect_by_rtiCnt')
    scale               = kwargs.pop('scale',[-0.03,0.03])
    st_uts              = kwargs.pop('st_uts',[14, 16, 18, 20])
    driver              = kwargs.pop('driver',None)
    output_dir          = kwargs.pop('output_dir','mstid_data/calendar')
    db_name             = kwargs.pop('db_name','mstid')
    mongo_port          = kwargs.pop('mongo_port',27017)
    dt                  = kwargs.pop('dt',None)
    mstid_reduced_inx   = kwargs.pop('mstid_reduced_inx',None)
    correlate           = kwargs.pop('correlate',None)
    highlight_ew        = kwargs.pop('highlight_ew',False)
    classification_colors   = kwargs.pop('classification_colors',False)

    driver_list         = gl.get_iterable(driver)
    if correlate:
        driver_list = [driver_list[0],'mstid_reduced_inx','correlate']

    this_date   = dt
    filename    = '{}{}_driver_timeseries.png'.format(this_date.strftime('%Y%m%d_%H%M'),file_suffix)
    filepath    = os.path.join(output_dir,filename)
    print('Plotting: {}'.format(filepath))
    
    if dct_list is not None:
        group_dict      = {}
        group_dict[0]   = {}
        group_dict[0]['dct_list']   = dct_list

    prepare_output_dirs({0:output_dir},clear_output_dirs=False)

    sDate,eDate = get_sDate_eDate(group_dict,sDate,eDate)
    xmax        = get_xmax(sDate,eDate)

    # Plotting section. ############################################################
    ax_list = []

    ax_nr   = 0
    ax_nx   = 1
    ax_ny   = len(list(group_dict.keys()))

    ax_left             = 0.
    ax_width            = 1.0
    ax_top              = 1.0
    h_pad               = 0.030
    w_pad               = 0.025

    fig_scale_x         = 1.
    fig_scale_y         = 0.180
    sup_title_bump      = 0.

    ax_ny += 1

    radar_panel_frac    = 0.0
    driver_panel_frac   = len(driver_list) * 0.20
    grib_panel_frac     = 0.0

    h_pad               = 0.025
    fig_scale_x         = 1.
    fig_scale_y         = 0.300

    fig_scale   = 40.
    figsize     = (fig_scale*fig_scale_x,fig_scale*fig_scale_y*ax_ny)
    fig         = plt.figure(figsize=figsize)

    if plot_geopot_maps:
        ret_info    = plot_geopot_movie_strip(grib_data_days,sDate,eDate,frame_times,
                        grib_panel_frac,ax_left,ax_width,ax_top,h_pad,w_pad,fig)

        ax_top      = ret_info['ax_top']
        ax          = ret_info['ax']

        if plot_letters:
            ax.text(letter_xpos,letter_ypos,'({})'.format(string.lowercase[letter_inx]),transform=ax.transAxes,fontdict=letter_fontdict)
            letter_inx  += 1

    if driver is not None:
        ylim_dict   = {}
        ylim_dict['mstid_reduced_inx']  = (-2.5,2.5)
        ylim_dict['smoothed_ae']        = (-1400,0)
        ylim_dict['ae_proc_0']          = (-1400,0)
        ylim_dict['omni_symh']          = (-250,100)
        ylim_dict['neg_mbar_diff']      = (-2.5,2.5)

        ax_info = plot_the_drivers(sDate,eDate,driver_list,
                driver_panel_frac,ax_left,ax_width,ax_top,h_pad,
                output_dir,mstid_reduced_inx,correlate,ylim_dict=ylim_dict,
                xlabels=True)

        ax_top  = ax_info['ax_top']

        for ax in ax_info['axs']:
            ax.axvline(this_date,ls='--',lw=6,color='k',zorder=750)

            if plot_letters:
                ax.text(letter_xpos,letter_ypos,'({})'.format(string.lowercase[letter_inx]),transform=ax.transAxes,fontdict=letter_fontdict)
                letter_inx  += 1

    plot_cbars(ax_list)

#    date_fmt    = '%d %b %Y'
#    date_str    = '{} - {}'.format(sDate.strftime(date_fmt),eDate.strftime(date_fmt))
#    fig.text(0.5,1.000+sup_title_bump,date_str,ha='center',fontdict={'weight':'bold','size':48})

    fig.savefig(filepath,bbox_inches='tight')
    if save_pdf:
        fig.savefig(filepath[:-3]+'pdf',bbox_inches='tight')
    plt.close(fig)

    return filepath
