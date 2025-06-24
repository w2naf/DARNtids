#!/usr/bin/env python
import os
import glob
import sys
import datetime
import multiprocessing

import copy


import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.collections import PolyCollection
from mpl_toolkits.basemap import Basemap
from hdf5_api import saveDictToHDF5, extractDataFromHDF5

import numpy as np


from .general_lib import truncate_colormap

def gen_mean_meta(meta_list):
    dates       = [meta['dt'] for meta in meta_list]
    sDate       = min(dates)
    eDate       = max(dates)
    date_fmt    = '%Y%m%d'
    date_str    = '_'.join([x.strftime(date_fmt) for x in [sDate,eDate]])
    
    mean_meta               = meta_list[0].copy()
    del mean_meta['dt']
    mean_meta['sDate']      = sDate
    mean_meta['eDate']      = eDate
    spl                     = mean_meta['png_name'].split('_')[2:]
    mean_meta['png_name']   = '_'.join(['MEAN',date_str]+spl)
    mean_meta['big_title']  = 'MEAN'

    date_fmt    = '%Y %b %d'
    name = 'Mean {} ({} - {})'.format(mean_meta['name'].split(':')[0],
            sDate.strftime(date_fmt),eDate.strftime(date_fmt))
    mean_meta['name'] = name

    return mean_meta

def gen_resid_meta(meta_list):
    new_meta_list   = []
    for meta in meta_list:
        meta = meta.copy()
        meta['residual']    = True

        meta['name']        = 'RMS Residual {}'.format(meta['name'])
        meta['big_title']   = 'RESIDUALS'
        new_meta_list.append(meta)
    return new_meta_list

def gen_roll_meta(meta_list,roll_steps):
    tmp                 = -np.arange(1,np.abs(roll_steps)+1)
    rolled_meta         = np.roll(meta_list,roll_steps,axis=0)
    rolled_meta[tmp]    = {}
    rolled_meta         = rolled_meta.tolist()
    rolled_meta         = copy.deepcopy(rolled_meta)

    for meta_0, meta_1 in zip(meta_list,rolled_meta):
        try:
            bt_0    = meta_0.get('big_title')
            tdiff   = (meta_1['dt'] - meta_0['dt']).total_seconds() / 86400.
            td_str  = '{!s} {:+0.1f} Days'.format(bt_0,tdiff)
        except:
            td_str  = 'NONE'
        meta_1['big_title'] = td_str

    return rolled_meta

def plot_plot_list(plot_list,multiproc=True):
    print("Now plotting!!!")
    if multiproc:
        pool = multiprocessing.Pool()
        pool.map(plot_grb,plot_list)
        pool.close()
        pool.join()
    else:
        for plot_dct in plot_list:
            plot_grb(plot_dct)

def prepare_output_dirs(output_dirs={0:'output'},clear_output_dirs=False,img_extra=''):
    import os
    import shutil

    txt = []
    txt.append('<?php')
    txt.append('foreach (glob("*.png") as $filename) {')
    txt.append('    echo "<img src=\'$filename\' width=\'100%\'> ";')
    txt.append('}')
    txt.append('?>')
    show_all_txt_100 = '\n'.join(txt)

    txt = []
    txt.append('<?php')
    txt.append('foreach (glob("*.png") as $filename) {')
    txt.append('    echo "<img src=\'$filename\' {img_extra}> <br />";'.format(img_extra=img_extra))
    txt.append('}')
    txt.append('?>')
    show_all_txt = '\n'.join(txt)

    for value in output_dirs.values():
        if clear_output_dirs:
            try:
                shutil.rmtree(value)
            except:
                pass
        try:
            os.makedirs(value)
        except:
            pass
        with open(os.path.join(value,'0000-show_all_100.php'),'w') as file_obj:
            file_obj.write(show_all_txt_100)
        with open(os.path.join(value,'0000-show_all.php'),'w') as file_obj:
            file_obj.write(show_all_txt)

def get_grb_dt(grb):
    dt  = datetime.datetime(grb['year'],grb['month'],grb['day'],
            grb['hour'],grb['minute'],grb['second'])
    return dt

def gen_plot_list(values,meta_list,**kwargs):
    shape = values.shape
    if len(shape) == 2:
        values.shape = (1,shape[0],shape[1])
    plot_list = []
    for inx,meta in enumerate(meta_list):
        tmp             = {}
        tmp['values']   = values[inx,:,:]
        tmp.update(meta)
        tmp.update(kwargs)
        plot_list.append(tmp)

    return plot_list

def gen_png_name(grb):
    keys    = []
    keys.append('indicatorOfTypeOfLevel')
    keys.append('pressureUnits')
    keys.append('typeOfLevel')
#    keys.append('level')

    sfx = []
    for key in keys:
        sfx.append(str(grb[key]))
#        print key,grb[key]

    grb_sname       = grb['shortName'].upper()
    dt              = get_grb_dt(grb)
    dateFmt         = '%Y%m%d_%H%M%S'
    png_name_lst    = []
    png_name_lst.append(dt.strftime(dateFmt))
    png_name_lst.append(grb_sname)
    png_name_lst    += sfx
    png_name    = '_'.join(png_name_lst)
    return png_name

def plot_grb_ax(plot_dct,axis,m=None,grdSz=2.5,
        plot_colorbars=True,plot_title=True,plot_latlon_labels=True,
        lat_labels=[False,True,True,False], lon_labels=[True,False,False,True],
        lat_fontdict = {'weight':'bold','size':'large'},
        lon_fontdict = {'weight':'bold','size':'large'},
        return_mappable=False):
    data        = plot_dct.get('values')
    if data is None: return
    if np.count_nonzero(np.isfinite(data)) == 0:
        return

    fig         = axis.get_figure()

    dt          = plot_dct.get('dt')
    png_name    = plot_dct.get('png_name')
    residual    = plot_dct.get('residual',False)
    lats, lons  = plot_dct.get('latlon')
    scale       = plot_dct.get('scale',(250000,350000))
    cmap        = plot_dct.get('cmap',None)
    cbar_label  = plot_dct.get('cbar_label',None)
    alpha       = plot_dct.get('alpha',1.)
    patch_fill  = plot_dct.get('patch_fill',False)
    contour     = plot_dct.get('contour',False)
    contourf    = plot_dct.get('contourf',False)
    nlevels     = plot_dct.get('nlevels',10)

    if not any([patch_fill,contour,contourf]):
        patch_fill = True

    coastline_color = '0.3'

    grid_zorder     = 10
    contour_zorder  = 25
    contourf_zorder =  5

    if m is None:
        width   = 12000000
        height  = 12000000
        m = Basemap(width=width,height=height,resolution='c',projection='stere',
                            lat_0=90.,lon_0=-100.,ax=axis)
        # draw parallels and meridians.
        parallels   = np.arange(-80.,81.,20.)
        meridians   = np.arange(-180.,181.,20.)

        if plot_latlon_labels:
            for par in parallels:
                if par > 0:
                    NS = ' N'
                elif par < 0:
                    NS = ' S'
                else:
                    NS = ''

                txt = '{:.0f}'.format(np.abs(par)) + r'$^{\circ}$' + NS
                xx,yy   = m(-60.,par)
                m.ax.text(xx,yy,txt,fontdict=lat_fontdict,
                        zorder=100,clip_on=True)
        else:
            lat_labels  = [False,False,False,False]
            lon_labels  = [False,False,False,False]

        m.drawparallels(parallels,color=coastline_color,labels=lat_labels,
                zorder=grid_zorder,fontdict=lat_fontdict)

        m.drawmeridians(meridians,color=coastline_color,labels=lon_labels,
                zorder=grid_zorder,fontdict=lon_fontdict)

        m.drawcoastlines(linewidth=0.5,color=coastline_color,zorder=grid_zorder)
        m.drawmapboundary(fill_color='w',zorder=grid_zorder)

    if not residual and cmap is None:
        cmap    = matplotlib.cm.jet

    if residual:
        if plot_dct.get('mbar_level') == 1: return m
        cmap    = matplotlib.cm.cool
        contourf    = False
        contour     = False
        patch_fill  = True

    bounds  = np.linspace(scale[0],scale[1],256)
    norm    = matplotlib.colors.BoundaryNorm(bounds,cmap.N)

    if patch_fill:
        nlats,nlons = np.shape(data)
        #################################################################################
        verts = []
        scan  = []
        for lat_inx in range(nlats):
            for lon_inx in range(nlons):
                lat = lats[lat_inx,lon_inx]
                lon = lons[lat_inx,lon_inx]
                if lat >= 89.5: continue

                scan.append(data[lat_inx,lon_inx])

                x1,y1 = m(lon-grdSz/2.,lat-grdSz/2.)
                x2,y2 = m(lon+grdSz/2.,lat-grdSz/2.)
                x3,y3 = m(lon+grdSz/2.,lat+grdSz/2.)
                x4,y4 = m(lon-grdSz/2.,lat+grdSz/2.)
                verts.append(((x1,y1),(x2,y2),(x3,y3),(x4,y4),(x1,y1)))

        pcoll   = PolyCollection(np.array(verts),edgecolors='face',linewidths=0,closed=False,cmap=cmap,norm=norm)
        pcoll.set_array(np.array(scan))
        axis.add_collection(pcoll,autolim=False)

    lats,lons,data = complete_the_circle(lats,lons,data)

    levels  = np.linspace(*scale,num=nlevels)
    if contour:
        pcoll   = m.contour (lons,lats,data,levels=levels,cmap=cmap,latlon=True,zorder=contour_zorder,linewidths=3)
        pcoll   = m.contourf(lons,lats,data,levels=levels,cmap=cmap,latlon=True,zorder=0)
        ylim    = axis.get_ylim()
        axis.axhspan(ylim[0],ylim[1],color='w',zorder=1,ec=None)

    if contourf:
        pcoll   = m.contourf(lons,lats,data,levels=levels,cmap=cmap,latlon=True,zorder=contourf_zorder,alpha=alpha)

    if plot_colorbars:
        cbar    = fig.colorbar(pcoll,orientation='vertical',shrink=.55,fraction=.1)
        txt     = plot_dct['shortName'].upper() + ' [' + plot_dct['units'] + ']'
        if cbar_label is None:
            cbar_label = txt
        else:
            cbar_label = '{}: {}'.format(cbar_label,txt)
        cbar.set_label(cbar_label,fontdict={'weight':'bold','size':'x-large'})

    if plot_title:
        txt     = '{}\n{}'.format(plot_dct['name'].title(),png_name)
        axis.set_title(txt,fontdict={'size':24,'weight':'bold'})

    if return_mappable:
        mappable = {'plot_dct':plot_dct,'pcoll':pcoll}
        return m,mappable
    else:
        return m

def plot_grb(plot_dct,grdSz=2.5):
    dt          = plot_dct.get('dt')
    png_name    = plot_dct.get('png_name')
    output_dir  = plot_dct.get('output_dir')
    outFName    = os.path.join(output_dir,png_name+'.png')
    residual    = plot_dct.get('residual',False)
    print('Plotting: {}'.format(outFName))

    fig     = plt.figure(figsize=(10,7.5))
    axis    = fig.add_subplot(111)
    plot_grb_ax(plot_dct,axis,grdSz=2.5)

    fig.savefig(outFName,bbox_inches='tight')
    plt.close(fig)
    return outFName

def get_processed_grib_data(sDate=None,eDate=None,season=None,name='Geopotential',mbar_level=10,
            calculation_type='delta',
            data_dir='mstid_data/rda_111.2',cache_dir=None,use_cache=True,test_mode=False):

    import pygrib
    if season is None:
        season  = '{!s}_{!s}'.format(sDate.year,eDate.year)

    src_dir = os.path.join(data_dir,season)

    if cache_dir is None:
        cache_dir   = os.path.join(data_dir,'cache')

    cache_path      = os.path.join(cache_dir,'{}_{}_{!s}mbar.h5'.format(season,calculation_type,mbar_level))

    if (not os.path.exists(cache_path)) or (not use_cache):
        values_list     = []
        meta_list       = []
        latlon_grid     = None
        
        files = glob.glob(os.path.join(src_dir,'*'))
        files.sort()
        
        if test_mode:
            files   = [files[0]]
        for fl in files:
            # Variables in file
            # array([u'10 metre U wind component', u'10 metre V wind component',
            #       u'2 metre dewpoint temperature', u'2 metre temperature',
            #       u'Geopotential', u'Land-sea mask', u'Mean sea level pressure',
            #       u'Relative humidity', u'Soil temperature level 1',
            #       u'Surface pressure', u'Temperature', u'U component of wind',
            #       u'V component of wind', u'Vertical velocity'], 
            #      dtype='<U28')

            print('Loading RDA 111.2 Data: {}'.format(fl))
            grbs        = pygrib.open(fl)
            grb_sel     = grbs.select(name=name)
            for grb in grb_sel:
                if grb['level'] != mbar_level:
                    continue

                if latlon_grid is None:
                    latlon_grid  = grb.latlons()

                png_name    = gen_png_name(grb)
                dt          = get_grb_dt(grb)

                meta    = {}
                meta['png_name']    = png_name
                meta['name']        = '{}: {}'.format(grb['name'].title(),str(dt))
                meta['shortName']   = grb['shortName']
                meta['units']       = grb['units']
                meta['dt']          = dt
                meta['latlon']      = latlon_grid
                meta['mbar_level']  = mbar_level
                meta['big_title']   = 'RAW'
                meta_list.append(meta)

                values_list.append(grb['values'])

            grbs.close()

        values      = np.array(values_list)
        
        plot_lists  = {}

        #### Stage raw data.
        scale               = (np.percentile(values,5.), np.percentile(values,95.))
        plot_list           = gen_plot_list(values,meta_list,scale=scale)
        plot_lists['raw']   = plot_list

        if calculation_type == 'seasonal_mean':
            #### Calculate average pattern.
            mean_values         = np.mean(values,axis=0)
            mean_meta           = gen_mean_meta(meta_list)
            plot_list           = gen_plot_list(mean_values,[mean_meta],scale=scale)
            plot_lists['mean']  = plot_list

            #### Calculate residuals
            resid                   = np.sqrt((values - mean_values)**2)
            scale                   = (0, np.percentile(resid,90.))
            resid_meta              = gen_resid_meta(meta_list)
            plot_list               = gen_plot_list(resid,resid_meta,scale=scale)
            plot_lists['residuals'] = plot_list

        elif calculation_type == 'delta':
            #### Stage delta pattern
            roll_steps    = -4
            comp_vals       = np.roll(values,roll_steps,axis=0)

            # Set end stages we don't have real data for to NaN.
            tmp = -np.arange(1,np.abs(roll_steps)+1)
            comp_vals[tmp,:,:] = np.nan

            comp_meta               = gen_roll_meta(meta_list,roll_steps)

            plot_list               = gen_plot_list(comp_vals,comp_meta,scale=scale)
            plot_lists['next']      = plot_list

            #### Calculate residuals
            resid                   = np.sqrt((values - comp_vals)**2)
            scale                   = (0, np.percentile(resid,90.))
            resid_meta              = gen_resid_meta(meta_list)
            plot_list               = gen_plot_list(resid,resid_meta,scale=scale)
            plot_lists['residuals'] = plot_list

            plot_lists['meta']  = {}
            mt                  = plot_lists['meta']
            mt['nr_times']      = len(meta_list)
            mt['name']          = name
            mt['mbar_level']    = mbar_level

        if use_cache:
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir)

            with h5py.File(cache_path, 'w') as fl:
                saveDictToHDF5(fl, plot_lists)
    else:
        print('Loading CACHED RDA 111.2 file: {}'.format(cache_path))
        with h5py.File(cache_path, 'r') as fl:
            plot_lists = extractDataFromHDF5(fl)

    return plot_lists

def complete_the_circle(lats,lons,data):
    """
    Fills in the gap in GRIB data between the last longitude and 0 degrees.
    """

    extra_lons  = np.ones((lons.shape[0],1)) * 360
    extra_lats  = np.array([lats[:,0]]).transpose()
    extra_data  = np.array([data[:,0]]).transpose()

    new_lons    = np.append(lons,extra_lons,axis=1)
    new_lats    = np.append(lats,extra_lats,axis=1)
    new_data    = np.append(data,extra_data,axis=1)

    return new_lats,new_lons,new_data


def seasonal_mean_cal_plot_list(geo_pot_src,**kwargs):
    plot_list = []
    for raw_dct,resid_dct in zip(geo_pot_src['raw'],geo_pot_src['residuals']):
        tmp     = {}

        geo_pot = {}
        geo_pot['raw']          = raw_dct
        geo_pot['residuals']    = resid_dct
        geo_pot['mean']         = geo_pot_src['mean'][0]
        geo_pot['dt']           = raw_dct['dt']
        tmp['geo_pot']          = geo_pot

        tmp.update(kwargs)

        plot_list.append(tmp)
    return plot_list

def delta_cal_plot_list(grib_data,mbar_dict=None,frame_times=None,**kwargs):
    """
    Generate a list with all info needed in each entry to generate 1 frame
    of a MSTID calendar plot with polar meteorological data.

    *grb_data*: list of grib data, each entry for a different mbar_level

    *mbar_dict*: Extra parameters pertaining to specific mbar levels.

    **kwargs:   Additional parameters to be passed onto the plotting routine.
                These are held constant with every frame.
    """
    # Initialize the plot list.
    plot_list   = []

    # Figure out how many frames there are going to be.
    # Iterate through each frame.
    nr_times    = grib_data[0]['meta']['nr_times']
    for tm_inx in range(nr_times):
        # Each frame will have its own dictionary.
        tmp         = {}

        # Get the frame time.
        dt          = grib_data[0]['raw'][tm_inx]['dt']
        if frame_times is not None:
            if dt not in frame_times: continue

        tmp['dt']   = dt

        # Create a dictionary for grib data, keyed by
        # mbar_level. Grib data can have multiple layers
        # of processing.
        tmp['grib_data']    = {}
        for grb in grib_data:
            this_grb    = {}
            mbar_level  = grb['meta']['mbar_level']
            tmp['grib_data'][mbar_level] = this_grb

            grb_keys    = list(grb.keys())
            for grb_key in grb_keys:
                if grb_key == 'meta': continue
                this_grb[grb_key]   = grb[grb_key][tm_inx]

                if mbar_dict is not None:
                    if mbar_level in mbar_dict:
                        for mbar_key,mbar_item in mbar_dict[mbar_level].items():
                            this_grb[grb_key][mbar_key] = mbar_item

        tmp.update(kwargs)
        plot_list.append(tmp)
    return plot_list

if __name__ == '__main__':
    output_dir      = 'output/polar_met_test'
    prepare_output_dirs({0:output_dir},clear_output_dirs=True)
    test_mode       = True

    season      = '2012_2013'
    mbar_levels = [1,10]

    mbar_dict       = {}
    for mbar_level in mbar_levels:
        tmp                     = {}
        tmp['alpha']            = 0.5
        tmp['contour']          = True
#        tmp['contourf']         = True
        tmp['cbar_label']       = '{!s} mbar Level'.format(mbar_level)
        mbar_dict[mbar_level]   = tmp

    mbar_dict[1]['cmap']    = truncate_colormap(matplotlib.cm.Blues,0.20)
    mbar_dict[10]['cmap']   = truncate_colormap(matplotlib.cm.Reds,0.20)

    grib_data   = []
    for mbar_level in mbar_levels:
        tmp = get_processed_grib_data(season=season,mbar_level=mbar_level,test_mode=test_mode)
        grib_data.append(tmp)

    plot_list   = delta_cal_plot_list(grib_data,mbar_dict=mbar_dict)

    geo_params  = ['residuals','raw','next']
    nx_ax       = 1
    ny_ax       = len(geo_params)
#    plot_list   = plot_list[-10:-1]
    for plot_dict in plot_list:
        dt_str      = plot_dict['dt'].strftime('%Y%m%d_%H%M')
        out_fname   = '_'.join([dt_str]+geo_params)
        out_path    = os.path.join(output_dir,out_fname)
        print('Plotting: {}'.format(out_path))

        figsize     = (15.*ny_ax,15.*nx_ax)
        fig         = plt.figure(figsize=figsize)

        # Geopotential Parameter #######################################################
        geo_prm_ax  = {}
        for geo_inx,geo_param in enumerate(geo_params):
            for mbar_level,grib_data in plot_dict['grib_data'].items():
                grb     = grib_data[geo_param]
                ax_info = geo_prm_ax.get(geo_param)
                if ax_info is None:
                    ax      = fig.add_subplot(nx_ax,ny_ax,geo_inx+1)
                    txt = grb.get('big_title')
                    ax.text(0.5,1.1,txt,ha='center',transform=ax.transAxes,
                            fontdict={'weight':'bold','size':48})

                    m       = plot_grb_ax(grb,ax)

                    ax_info                 = {}
                    ax_info['ax']           = ax
                    ax_info['m']            = m
                    geo_prm_ax[geo_param]   = ax_info
                else:
                    ax  = ax_info.get('ax')
                    m   = ax_info.get('m')

                    plot_grb_ax(grb,ax,m=m)

        fig.savefig(out_path,bbox_inches='tight')
        plt.close(fig)

## After lunch to do:
#1. Make polar_met if __name__ testable.  Remove confusing other script.
#2. Make sure it can compute arbitrary mbar levels.
#3. Implement new residual calculations.
#4. Track down path of call from calendar plot.
#5. Overlay both 1 and 10 mbar settings.
#
#
### Other:
#1. Add in directional component??
