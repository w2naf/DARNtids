#!/usr/bin/env python
import os,sys,pickle,datetime
import glob
import copy

import logging
log_name    = os.path.basename(__file__)[:-2]+'log'
logging.basicConfig(filename=log_name,level=logging.DEBUG,format='%(asctime)s %(message)s')
logging.getLogger().addHandler(logging.StreamHandler())

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.collections import PolyCollection
font    = {'weight': 'bold', 'size': 12}
matplotlib.rc('font',**font)

import numpy as np
import scipy as sp
import pandas as pd

import ephem # pip install pyephem (on Python 2)
             # pip install ephem   (on Python 3)

import davitpy
from davitpy import pydarn
from davitpy import utils
from davitpy.pydarn.proc import music
import handling

class MusicFromDataSet(music.musicArray):
    def __init__(self,curr_data):

        ds_name = 'DS001_{!s}'.format(curr_data.metadata.get('dataSetName'))
        setattr(self,ds_name,curr_data)
        self.active = getattr(self,ds_name)

        if hasattr(curr_data,'prm'):
            self.prm = curr_data.prm

def my_calc_aacgm(lats,lons,year,height=250):
    shape = lats.shape

    lats    = np.ravel(lats)
    lons    = np.ravel(lons)

    tf          = np.logical_and(np.isfinite(lats), np.isfinite(lons))
    lats_in     = lats[tf].tolist()
    lons_in     = lons[tf].tolist()
    height_in   = [height]*len(lats[tf])

    result  = davitpy.models.aacgm.aacgmConvArr(lats_in,lons_in,height_in,year,0)

    mlats       = np.zeros_like(lats)
    mlons       = np.zeros_like(lats)
    mlats[:]    = np.nan
    mlons[:]    = np.nan

    mlats[tf]   = result[0]
    mlons[tf]   = result[1]

    mlats       = mlats.reshape(shape)
    mlons       = mlons.reshape(shape)
    return (mlats,mlons)

class GmeFilter(object):
    def __init__(self,sTime,eTime,thresh=None,min_thresh_duration=None,gme_param='ae',
            delay=None,win_minutes=None,win_minutes_roll_var=10, output_dir='output',ylim=None,**kwargs):
        """
        Creates a geomagnetic environment (GME) filter object for a given time and parameter. 

        The __init__ function loads in the GME data and does the initial processing of
        the GME data only.

        sTime: datetime.datetime giving start of data to be filtered.  GME data starting at a day
            earlier is loaded.

        eTime: datetime.datetime giving end of data to be filtered.  GME data ending a day
            later is loaded.

        thresh: Threshold values for GME parameter to define a filter in tuple form.
            Example:
                (None, 5): No lower bound, select GME where the value is < 5.
                (5,None):  No upper bound, select GME where the value is >= 5.
                (-5, 5):   Select data where -5 <= GME < 5.

        delay: datetime.timedelta delay between selected GME time and selected radar data time.
            This allows time for disturbances to propagate from high latitudes down to the radar
            observation site.

        win_minutes: Integer minutes to use for a moving average window for smoothing the GME 
            data.

        output_dir: Directory where plots will be saved to.
        """
        import gme

        gme_sTime   = sTime - datetime.timedelta(days=1)
        gme_eTime   = eTime + datetime.timedelta(days=1)

        self.sTime          = sTime
        self.eTime          = eTime
        self.gme_sTime      = gme_sTime
        self.gme_eTime      = gme_eTime
        self.thresh         = thresh
        self.gme_param      = gme_param
        self.delay          = delay
        self.win_minutes    = win_minutes
        self.win_minutes_roll_var    = win_minutes_roll_var
        self.ylim           = ylim
        self.output_dir     = output_dir

        if thresh is None: thresh = (None,None)
        #Convert input into explicit min/max variables.
        ind_min = thresh[0]
        ind_max = thresh[1]
        
        if 'omni' in gme_param: 
            ind_class   = gme.ind.readOmni(gme_sTime,gme_eTime,res=1)
            omni_list   = []
            omni_time   = []
            for xx in ind_class:
                tmp = {}
#                tmp['res']          = xx.res
#                tmp['timeshift']    = xx.timeshift
#                tmp['al']           = xx.al
#                tmp['au']           = xx.au
#                tmp['asyd']         = xx.asyd
#                tmp['asyh']         = xx.asyh
#                tmp['symd']         = xx.symd
#                tmp['beta']         = xx.beta
#                tmp['bye']          = xx.bye
#                tmp['bze']          = xx.bze
#                tmp['e']            = xx.e
#                tmp['flowSpeed']    = xx.flowSpeed
#                tmp['vxe']          = xx.vxe
#                tmp['vye']          = xx.vye
#                tmp['vzy']          = xx.vzy
#                tmp['machNum']      = xx.machNum
#                tmp['np']           = xx.np
#                tmp['temp']         = xx.temp

#                tmp['time']         = xx.time
                tmp['ae']           = xx.ae
                tmp['bMagAvg']      = xx.bMagAvg
                tmp['bx']           = xx.bx 
                tmp['bym']          = xx.bym
                tmp['bzm']          = xx.bzm
                tmp['pDyn']         = xx.pDyn
                tmp['symh']         = xx.symh
                tmp['flowSpeed']    = xx.flowSpeed
                tmp['np']           = xx.np
                tmp['temp']         = xx.temp
                
                omni_time.append(xx.time)
                omni_list.append(tmp)

            omni_df_raw         = pd.DataFrame(omni_list,index=omni_time)
            del omni_time
            del omni_list

            self.omni_df_raw    = omni_df_raw
            self.omni_df        = omni_df_raw.resample('T')
            self.omni_df        = self.omni_df.interpolate()

        plot_info               = {}
        plot_info['x_label']    = 'Date [UT]'
        set_roll_var_as_raw     = False
        if gme_param == 'ae':
            # Read data with DavitPy routine and place into numpy arrays.
            ind_class   = gme.ind.readAe(gme_sTime,gme_eTime,res=1)
            ind_data    = [(x.time, x.ae) for x in ind_class]

            ind_df              = pd.DataFrame(ind_data,columns=['time','ind_0_raw'])
            ind_df              = ind_df.set_index('time')

            plot_info['ind_0_gme_label']  = 'AE Index [nT]'

        elif (gme_param == 'omni_by'):
            ind_df  = pd.DataFrame(omni_df_raw['bym'])

            plot_info['ind_0_symbol']     = 'OMNI By'
            plot_info['ind_0_gme_label']  = 'OMNI By GSM [nT]'

        elif gme_param == 'omni_bz':
            ind_df  = pd.DataFrame(omni_df_raw['bzm'])

            plot_info['ind_0_symbol']     = 'OMNI Bz'
            plot_info['ind_0_gme_label']  = 'OMNI Bz GSM [nT]'

        elif gme_param == 'omni_pdyn':
            ind_df  = pd.DataFrame(omni_df_raw['pDyn'])

            plot_info['ind_0_symbol']     = 'OMNI pDyn'
            plot_info['ind_0_gme_label']  = 'OMNI pDyn [nPa]'

        elif gme_param == 'omni_by_var':
            ind_df                          = pd.DataFrame(omni_df_raw['bym'])
            set_roll_var_as_raw             = True
            plot_info['ind_0_symbol']       = 'var(OMNI By)'
            plot_info['ind_0_gme_label']    = u'$\sigma^2$ var(OMNI By GSM [nT])'

        elif gme_param == 'omni_bz_var':
            ind_df                          = pd.DataFrame(omni_df_raw['bzm'])
            set_roll_var_as_raw             = True
            plot_info['ind_0_symbol']       = 'var(OMNI Bz)'
            plot_info['ind_0_gme_label']    = u'$\sigma^2$ var(OMNI Bz GSM [nT])'

        elif gme_param == 'omni_b_var':
            ind_df                          = pd.DataFrame(omni_df_raw['bMagAvg'])
            set_roll_var_as_raw             = True
            plot_info['ind_0_symbol']       = 'var(OMNI B)'
            plot_info['ind_0_gme_label']    = u'$\sigma^2$ var(OMNI B [nT])'

        # Put into a pandas data frame to do some easy filtering and such.
        ind_df.columns  = ['ind_0_raw']
        self.ind_df_raw     = ind_df

        # Resample to 1 minute data.
        res_df              = ind_df.resample('T')
        res_df              = res_df.interpolate()
        ind_df              = res_df

        ind_df['ind_0_var'] = pd.rolling_std(ind_df['ind_0_raw'],win_minutes_roll_var,center=False)**2
        if set_roll_var_as_raw:
            ind_df['ind_0_raw'] = ind_df['ind_0_var'] 

        if win_minutes is not None:
            ind_df['ind_0_processed'] = pd.rolling_mean(ind_df['ind_0_raw'],win_minutes,center=True)
        else:
            ind_df['ind_0_processed'] = ind_df['ind_0_raw']

        filter_txt = 'GME Filter: {} {} Delay: {}'.format(gme_param.upper(),str(thresh),str(delay))
        plot_title = 'GME Filter: {} {}'.format(gme_param.upper(),str(thresh))

        if win_minutes is not None: 
            filter_txt = '{}; {} Min Roll Mean'.format(filter_txt,str(win_minutes))
            plot_title = '{}; {} Min Roll Mean'.format(plot_title,str(win_minutes))

        if win_minutes_roll_var is not None: 
            filter_txt = '{}; {} Min Roll Var'.format(filter_txt,str(win_minutes_roll_var))
            plot_title = '{}; {} Min Roll Var'.format(plot_title,str(win_minutes_roll_var))

        plot_info['title']  = plot_title

        # Use the pandas processed values for the rest of the routine.
        ind_times           = ind_df.index.to_pydatetime()
        ind_vals            = ind_df['ind_0_processed']

        # Put an actual value in for both min and max even if one is not
        # specified.
        if ind_min is None: ind_min = np.nanmin(ind_vals)
        if ind_max is None: ind_max = np.nanmax(ind_vals)+1
        
        # Find out where the index values meet the specified condition.
        tfs = []
        tfs.append(ind_vals >= ind_min)
        tfs.append(ind_vals <  ind_max)

        ind_tfs = np.logical_and.reduce(tfs)

        bound_pairs = self.__calculate_boundary_pairs__(ind_times,ind_tfs)

        if min_thresh_duration:
            for bp in bound_pairs[:]:
                if (bp[1]-bp[0]) < min_thresh_duration:
                    bound_pairs.remove(bp)

        self.filter_txt     = filter_txt

        self.ind_df         = ind_df
        self.plot_info      = plot_info
        self.ind_times      = ind_times
        self.ind_vals       = ind_vals
        self.bound_pairs    = bound_pairs

    def __calculate_boundary_pairs__(self,ind_times,ind_tfs):
        """
        Returns a list of tuples containing the times of where a corresponding
        boolean array is True.
        """

        # Find the places where it switches from one state to another.
        ind_transitions = np.diff(ind_tfs).tolist()
        # Add a true/false value to the beginning to make the indices of
        # the transitions array actually matches the time boundaries.
        ind_transitions = np.array( [ind_tfs[0]] + ind_transitions )

        # Create a list of the transition boundardies.
        bound_times     = []
        for ind_inx,ind_transition in enumerate(ind_transitions):
            if ind_transition:
                bound_times.append(ind_times[ind_inx])

        # If the length of the boundary times is odd, add one more time
        # to create an even set of boundaries.
        if len(bound_times) % 2:
            bound_times.append(ind_times[-1]+datetime.timedelta(minutes=1))

        pair_inxs   = [ (2*x, 2*x+1) for x in range(len(bound_times)/2) ]
        bound_pairs = [] 
        for inx_0, inx_1 in pair_inxs:
            bound_pairs.append( (bound_times[inx_0], bound_times[inx_1]) )

        return bound_pairs

    def __expand_boundary_pairs__(self,boundary_pairs,sTime,eTime):

        dt_list = [datetime.datetime(sTime.year,sTime.month,sTime.day,
                                     sTime.hour,sTime.minute)]

        while dt_list[-1] < eTime:
            dt_list.append(dt_list[-1] + datetime.timedelta(minutes=1))

        dt_list = np.array(dt_list)

        ret_tf = np.zeros_like(dt_list,dtype=np.bool)
        for bp in boundary_pairs:
            tf = np.logical_and(dt_list >= bp[0], dt_list < bp[1])
            ret_tf[tf] = True

        return (dt_list,ret_tf)

    def plot_gme(self,ylim=None,selected_color='r',selected_lw=1,ax=None,output_dir=None):
        """
        Plots GME parameter data that is used for the filter.  
        By default:
            * Raw data is plotted in gray.
            * Smoothed data is plotted in black.
            * Selected data is plotted in red.
        """
        if output_dir is None:
            output_dir      = self.output_dir

        gme_param           = self.gme_param
        sTime               = self.sTime
        eTime               = self.eTime

        gme_sTime           = self.gme_sTime
        gme_eTime           = self.gme_eTime
        ind_df              = self.ind_df
        plot_info           = self.plot_info
        ind_times           = self.ind_times
        ind_vals            = self.ind_vals
        if hasattr(self,'bound_pairs'):
            bound_pairs     = self.bound_pairs
        else:
            bound_pairs     = []

        if ylim is None: ylim = self.ylim

        # Plotting diagnostics. ########################################################
        if ax is None:
            figsize = (10,4)
            fig = plt.figure(figsize=figsize)
            ax  = fig.add_subplot(111)

            save_fig = True
        else:
            fig = ax.get_figure()
            save_fig = False

        x_vals = ind_df.index
        y_vals = ind_df['ind_0_raw']
        ax.plot(x_vals,y_vals,color='0.8',label='Raw')

        y_vals = ind_df['ind_0_processed']
        ax.plot(x_vals,y_vals,color='k',label='Processed')

        for tr_inx,(tm_min,tm_max) in enumerate(bound_pairs):
            if tr_inx == 0:
                label = 'Selected'
            else:
                label = None

            tf = np.logical_and(ind_df.index >= tm_min, ind_df.index < tm_max)
            x_vals  = ind_df.index[tf].to_pydatetime()
            y_vals  = np.array(ind_df.ind_0_processed[tf].tolist())
            ax.plot(x_vals,y_vals,color=selected_color,label=label,linewidth=selected_lw)

        if hasattr(self,'arg_epoch_0'):
            ax.scatter(self.arg_epoch_0,ind_df[self.maxima_level][self.arg_epoch_0],zorder=5,facecolor='r')

        ax.grid()
        ax.legend(fontsize='x-small')

        ax.set_ylim(ylim)

        gray = '0.90'
        ax.axvspan(gme_sTime,sTime,color=gray,zorder=1)
        ax.axvspan(eTime,gme_eTime,color=gray,zorder=1)
        ax.axvline(x=sTime,color='g',ls='--',lw=2,zorder=150)
        ax.axvline(x=eTime,color='g',ls='--',lw=2,zorder=150)

        ax.set_title(plot_info['title'])
        ax.set_xlabel(plot_info['x_label'])
        ax.set_ylabel(plot_info['ind_0_gme_label'])
        fig.autofmt_xdate()
        
        if save_fig:
            fig.tight_layout()
            date_fmt    = '%Y%m%d.%H%M'
            date_str    = '{}-{}'.format(gme_sTime.strftime(date_fmt),gme_eTime.strftime(date_fmt))
            filename    = os.path.join(output_dir,'{}_{}.png'.format(date_str,gme_param))
            fig.savefig(filename,bbox_inches='tight')
            plt.close()

        if gme_param == 'omni_by_dominant':
            self.plot_by_bz()

    def plot_by_bz(self,ylim=None,selected_color='r',selected_lw=1.):
        """
        Plots GME parameter data that is used for the filter.  
        By default:
            * Raw data is plotted in gray.
            * Smoothed data is plotted in black.
            * Selected data is plotted in red.
        """
        output_dir          = self.output_dir
        gme_param           = self.gme_param
        sTime               = self.sTime
        eTime               = self.eTime

        gme_sTime           = self.gme_sTime
        gme_eTime           = self.gme_eTime
        ind_df              = self.ind_df
        plot_info           = self.plot_info
        ind_times           = self.ind_times
        ind_vals            = self.ind_vals
        bound_pairs         = self.bound_pairs

        # Plotting diagnostics. ########################################################
        figsize = (10,8)
        fig = plt.figure(figsize=figsize)
        ax  = fig.add_subplot(211)

        plot_list = []

        tmp = {}
        tmp['ind']      = 'ind_0'
        tmp['color']    = 'b'
        plot_list.append(tmp)

        tmp = {}
        tmp['ind']      = 'ind_1'
        tmp['color']    = 'g'
        plot_list.append(tmp)

        for plot_dict in plot_list:
            col_name    = '{}_raw'.format(plot_dict['ind'])
            symbol      = '{}_symbol'.format(plot_dict['ind'])
            label       = plot_info[symbol]
            color       = plot_dict['color']

            rgb         = matplotlib.colors.colorConverter.to_rgb(color)
            hsv         = matplotlib.colors.rgb_to_hsv(rgb)
            new_hsv     = hsv * np.array([1.00,0.15,1.00])
            new_rgb     = matplotlib.colors.hsv_to_rgb(new_hsv)

            x_vals      = ind_df.index
            y_vals      = ind_df[col_name]
            ax.plot(x_vals,y_vals,color=new_rgb)

            col_name    = '{}_processed'.format(plot_dict['ind'])
            x_vals      = ind_df.index
            y_vals      = ind_df[col_name]
            ax.plot(x_vals,y_vals,color=color,label=label)

        ax.grid()
        ax.legend(fontsize='x-small')


        ax.set_ylim(ylim)

        gray = '0.90'
        ax.axvspan(gme_sTime,sTime,color=gray,zorder=1)
        ax.axvspan(eTime,gme_eTime,color=gray,zorder=1)
        ax.axvline(x=sTime,color='g',ls='--',lw=2,zorder=150)
        ax.axvline(x=eTime,color='g',ls='--',lw=2,zorder=150)

        ax.set_title(plot_info['title'])
        ax.set_xlabel(plot_info['x_label'])
        ax.set_ylabel(plot_info['ind_0_gme_label'])

        ################################################################################ 
        ax  = fig.add_subplot(212)

        plot_list = []

        tmp = {}
        tmp['ind']      = 'ind_0'
        tmp['color']    = 'b'
        plot_list.append(tmp)

        tmp = {}
        tmp['ind']      = 'ind_1'
        tmp['color']    = 'g'
        plot_list.append(tmp)

        for plot_dict in plot_list:
            col_name    = '{}_raw'.format(plot_dict['ind'])
            symbol      = '{}_symbol'.format(plot_dict['ind'])
            label       = 'abs({})'.format(plot_info[symbol])
            color       = plot_dict['color']

            x_vals      = ind_df.index
            y_vals      = np.abs(ind_df[col_name])
            ax.plot(x_vals,y_vals,color=color,label=label)

        ax.grid()
        ax.legend(fontsize='x-small')

        ax.set_ylim(ylim)

        gray = '0.90'
        ax.axvspan(gme_sTime,sTime,color=gray,zorder=1)
        ax.axvspan(eTime,gme_eTime,color=gray,zorder=1)
        ax.axvline(x=sTime,color='g',ls='--',lw=2,zorder=150)
        ax.axvline(x=eTime,color='g',ls='--',lw=2,zorder=150)

        ax.set_title(plot_info['title'])
        ax.set_xlabel(plot_info['x_label'])
        ax.set_ylabel(plot_info['ind_0_gme_label'])


        fig.autofmt_xdate()
        
        fig.tight_layout()
        filename = os.path.join(output_dir,'by_bz_compare.png'.format(gme_param))
        fig.savefig(filename,bbox_inches='tight')
        plt.close()

    def get_mapped_value(self,ut_datetime,level='ind_0_processed'):
        gme_param           = self.gme_param

        ind_df              = self.ind_df
        delay               = self.delay

        if delay is None:
            search_time = ut_datetime
        else:
            search_time = ut_datetime - delay
        
        inx = np.argmin(np.abs(ind_df.index.to_pydatetime() - search_time))
        val = ind_df.iloc[inx][level]
        return val


    def filter(self,curr_data):
        """
        Filters a MUSIC data set object based on the already created
        geomagnetic environment (GME) filter.
        """
        bound_pairs         = self.bound_pairs
        delay               = self.delay
        filter_txt          = self.filter_txt

        # Actually filter the data. ####################################################
        tf_arrs = []
        for tm_min,tm_max in bound_pairs:
            if delay is not None:
                tm_min = tm_min + delay
                tm_max = tm_max + delay

            tf = np.logical_and(curr_data.time >= tm_min, curr_data.time < tm_max)
            tf_arrs.append(tf)

        good_times = np.logical_or.reduce(tf_arrs)

        curr_data.time = curr_data.time[good_times]
        curr_data.data = curr_data.data[good_times,:,:]
        if hasattr(curr_data,'slt_arr'):
            curr_data.slt_arr= curr_data.slt_arr[good_times,:,:]

        curr_data.appendHistory(filter_txt)
        curr_data.setMetadata(gme_filter=filter_txt)

    def filter_time_series(self,times,vals):
        """
        Filters a time series based on the already created
        geomagnetic environment (GME) filter.

        times:  series of datetime.datetimes
        vals:   1D series of matching data that goes with times

        Times that do not meet the filter criteria will be set to NaN.
        """

        times   = np.array(times)
        vals    = np.array(vals)

        bound_pairs         = self.bound_pairs
        delay               = self.delay
        filter_txt          = self.filter_txt

        # Actually filter the data. ####################################################
        tf_arrs = []
        for tm_min,tm_max in bound_pairs:
            if delay is not None:
                tm_min = tm_min + delay
                tm_max = tm_max + delay

            tf = np.logical_and(times >= tm_min, times < tm_max)
            tf_arrs.append(tf)

        good_times  = np.logical_or.reduce(tf_arrs)
        bad_times   = np.logical_not(good_times)

        vals[bad_times] = np.nan
        return vals

    def filter_and(self,that_filter):
        """
        Generates a new GME filter object that is the union filter of two other
        GME filter objects.
        """

        my_sTime = np.min([self.gme_sTime,that_filter.gme_sTime])
        my_eTime = np.max([self.gme_eTime,that_filter.gme_eTime])

        this_times,this_tf = self.__expand_boundary_pairs__(self.bound_pairs,my_sTime,my_eTime)
        that_times,that_tf = self.__expand_boundary_pairs__(that_filter.bound_pairs,my_sTime,my_eTime)

        union_tf = np.logical_and(this_tf,that_tf)
        union_bp = self.__calculate_boundary_pairs__(this_times,union_tf)

        new_filter_obj = copy.deepcopy(self)

        new_filter_obj.bound_pairs = union_bp
        that_txt = '{} {}'.format(that_filter.gme_param.upper(),str(that_filter.thresh))
        new_filter_obj.filter_txt   = '{} | AND {}'.format(self.filter_txt,that_txt)
        new_filter_obj.plot_info['title'] = '{} | AND {}'.format(self.plot_info['title'],that_txt)

        return new_filter_obj
    
    def find_maxima(self,level='ind_0_processed',isolation=datetime.timedelta(hours=24)):
        ind_df      = self.ind_df
        bound_pairs = self.bound_pairs

        arg_epoch_0 = []
        for bp_0, bp_1 in bound_pairs:
            tf = np.logical_and(ind_df.index >= bp_0, ind_df.index < bp_1)

            mx = ind_df[tf][level].argmax()

            arg_epoch_0.append(mx)

        
        for mx in arg_epoch_0[:]:
            st = mx - isolation/2
            en = mx + isolation/2
            tf = np.logical_and(ind_df.index >= st, ind_df.index < en)
            
            if mx != ind_df[tf][level].argmax():
                print 'Removing {}!'.format(str(mx))
                arg_epoch_0.remove(mx)

        arg_epoch_0         = [x.to_datetime() for x in arg_epoch_0]
        arg_epoch_0         = np.array(arg_epoch_0)
        self.arg_epoch_0    = arg_epoch_0
        self.maxima_level   = level
        return arg_epoch_0

    def find_turning(self,level='ind_0_processed'):
        ind_df      = self.ind_df
        bound_pairs = self.bound_pairs
        
        arg_epoch_0 = []
        for bp_0, bp_1 in bound_pairs:
            arg_epoch_0.append(bp_0)

        arg_epoch_0      = np.array(arg_epoch_0)
        self.arg_epoch_0 = arg_epoch_0
        self.maxima_level = level
        return arg_epoch_0

    def find_bz_turning(self,level='ind_0_processed'):
        ind_df      = self.ind_df
        bound_pairs = self.bound_pairs
        
        tmp = ind_df[level].copy()
        tmp[tmp >= 0] = 0
        tmp[tmp <  0] = -1

        diff = np.diff(tmp)
        tf = diff == -1

        zero_inxs = ind_df.index[tf]

        frac_thresh = 0.75

        arg_epoch_0 = []
        bound_pairs = []
        for zi in zero_inxs:
            # Test Condition
            # Bz <= 5 for 1 Hour Prior --> Try to prevent being in a storm period!
            test_st     = zi - datetime.timedelta(hours=1)
            test_et     = zi
            test_tf     = np.logical_and(ind_df.index >= test_st, ind_df.index < test_et)
            test_data   = ind_df[level][test_tf]
            test_good_tf    = test_data <= 5.
            test_good_frac  = np.count_nonzero(test_good_tf) / np.float(test_good_tf.size)
            if test_good_frac < frac_thresh:
                continue

            # Test Condition
            # AE
            test_df     = self.omni_df['ae']
            test_st     = zi - datetime.timedelta(hours=6)
            test_et     = zi
            test_tf     = np.logical_and(test_df.index >= test_st, test_df.index < test_et)
            test_data   = test_df[test_tf]
            test_good_tf    = test_data <= 400.
            test_good_frac  = np.count_nonzero(test_good_tf) / np.float(test_good_tf.size)
            if test_good_frac < .99:
                continue

            # Test Condition
            # SYM-H
            test_df     = self.omni_df['symh']
            test_st     = zi - datetime.timedelta(hours=1)
            test_et     = zi
            test_tf     = np.logical_and(test_df.index >= test_st, test_df.index < test_et)
            test_data   = test_df[test_tf]
            test_good_tf    = test_data >= -10.
            test_good_frac  = np.count_nonzero(test_good_tf) / np.float(test_good_tf.size)
            if test_good_frac < frac_thresh:
                continue

            # Test Condition
            # Bz >= 0 for 1 Hour Prior
            test_st     = zi - datetime.timedelta(hours=1)
            test_et     = zi
            test_tf     = np.logical_and(ind_df.index >= test_st, ind_df.index < test_et)
            test_data   = ind_df[level][test_tf]
            test_good_tf    = test_data >= 0.
            test_good_frac  = np.count_nonzero(test_good_tf) / np.float(test_good_tf.size)
            if test_good_frac < frac_thresh:
                continue
            bp_0 = ind_df.index[test_tf].min().to_datetime()

            # Test Condition
            # Bz <= -2 for 1 Hour  Post
            test_st     = zi
            test_et     = zi + datetime.timedelta(hours=1)
            test_tf     = np.logical_and(ind_df.index >= test_st, ind_df.index < test_et)
            test_data   = ind_df[level][test_tf]
            test_good_tf    = test_data <= -2.
            test_good_frac  = np.count_nonzero(test_good_tf) / np.float(test_good_tf.size)
            if test_good_frac < frac_thresh:
                continue
            bp_1 = ind_df.index[test_tf].max().to_datetime()

            arg_epoch_0.append(zi.to_datetime())
            bound_pairs.append( (bp_0, bp_1) )

        # Remove epochs that are too close together.
        isolation   = datetime.timedelta(hours=6)
        arg_diff    = np.diff(arg_epoch_0)
        iso_tf      = np.array([True] + (arg_diff > isolation).tolist())

        arg_epoch_0         = np.array(arg_epoch_0)
        arg_epoch_0         = arg_epoch_0[iso_tf]

        bound_pairs         = np.array(bound_pairs)
        bound_pairs         = bound_pairs[iso_tf].tolist()

        self.bound_pairs    = bound_pairs
        self.arg_epoch_0    = arg_epoch_0
        self.maxima_level   = level
        return arg_epoch_0

    def find_bz_var_events(self,level='ind_0_processed',frac_thresh=0.65):
        ind_df      = self.ind_df
        test_inxs   = self.arg_epoch_0

        arg_epoch_0 = []
        bound_pairs = []
        for zi in test_inxs:
            bp_0 = zi - datetime.timedelta(hours=6)
            bp_1 = zi + datetime.timedelta(hours=6)

            tmp_cndxs = []
            tmp_cndxs.append( (datetime.timedelta(hours=6), 0.95, 0.75) )
            tmp_cndxs.append( (datetime.timedelta(hours=4), 0.95, 0.65) )
            tmp_cndxs.append( (datetime.timedelta(hours=2), 1.00, 0.60) )

            bad_event = False
            # Test Condition
            for delt,quant,thrsh in tmp_cndxs:
                test_st     = zi - delt
                test_et     = zi
                test_tf     = np.logical_and(ind_df.index >= test_st, ind_df.index < test_et)
                test_data   = ind_df[level][test_tf]
                if test_data.quantile(quant) > thrsh:
                    bad_event = True
                    break
            if bad_event: continue

            # Test Condition
            test_st     = zi
            test_et     = zi + datetime.timedelta(hours=3)
            test_tf     = np.logical_and(ind_df.index >= test_st, ind_df.index < test_et)
            test_data   = ind_df[level][test_tf]
            if test_data.quantile(0.75) < 0.5:
                continue

            arg_epoch_0.append(zi)
            bound_pairs.append( (bp_0, bp_1) )

        self.bound_pairs    = bound_pairs
        arg_epoch_0         = np.array(arg_epoch_0)
        self.arg_epoch_0    = arg_epoch_0
        self.maxima_level   = level
        return arg_epoch_0

    def find_b_var_events(self,level='ind_0_processed',frac_thresh=0.65):
        ind_df      = self.ind_df
        test_inxs   = self.arg_epoch_0

        arg_epoch_0 = []
        bound_pairs = []
        for zi in test_inxs:
            bp_0 = zi - datetime.timedelta(hours=6)
            bp_1 = zi + datetime.timedelta(hours=6)

            tmp_cndxs = []
            tmp_cndxs.append( (datetime.timedelta(hours=6), 0.95, 0.75) )
            tmp_cndxs.append( (datetime.timedelta(hours=4), 0.95, 0.65) )
            tmp_cndxs.append( (datetime.timedelta(hours=2), 1.00, 0.60) )

            bad_event = False
            # Test Condition
            for delt,quant,thrsh in tmp_cndxs:
                test_st     = zi - delt
                test_et     = zi
                test_tf     = np.logical_and(ind_df.index >= test_st, ind_df.index < test_et)
                test_data   = ind_df[level][test_tf]
                if test_data.quantile(quant) > thrsh:
                    bad_event = True
                    break
    #            test_good_tf    = test_data >= 0.
    #            test_good_frac  = np.count_nonzero(test_good_tf) / np.float(test_good_tf.size)
    #            if test_good_frac < frac_thresh:
    #                continue
            if bad_event: continue

            # Test Condition
            test_st     = zi
            test_et     = zi + datetime.timedelta(hours=3)
            test_tf     = np.logical_and(ind_df.index >= test_st, ind_df.index < test_et)
            test_data   = ind_df[level][test_tf]
            if test_data.quantile(0.75) < 0.65:
                continue
#            test_good_tf    = test_data >= 0.
#            test_good_frac  = np.count_nonzero(test_good_tf) / np.float(test_good_tf.size)
#            if test_good_frac < frac_thresh:
#                continue

            arg_epoch_0.append(zi)
            bound_pairs.append( (bp_0, bp_1) )

        self.bound_pairs    = bound_pairs
        arg_epoch_0         = np.array(arg_epoch_0)
        self.arg_epoch_0    = arg_epoch_0
        self.maxima_level   = level
        return arg_epoch_0

    def find_pdyn_events(self,level='ind_0_processed'):
        ind_df      = self.ind_df

        var_thresh_0        = 0.15
        var_thresh_0_dur    = datetime.timedelta(hours=0.75)
        tf                  = ind_df['ind_0_var'] < var_thresh_0
        bound_pairs         = self.__calculate_boundary_pairs__(ind_df.index,tf)

        for bp in bound_pairs[:]:
            if (bp[1]-bp[0]) < var_thresh_0_dur:
                bound_pairs.remove(bp)

        self.bound_pairs    = bound_pairs
        test_inxs           = self.find_endpoint()

        arg_epoch_0 = []
        for zi in test_inxs:
            bad_event   = False
            test_st     = zi - datetime.timedelta(hours=1)
            test_et     = zi + datetime.timedelta(hours=1)
            tf_0        = np.logical_and(ind_df.index >= test_st, ind_df.index < zi)
            tf_1        = np.logical_and(ind_df.index >= zi, ind_df.index < test_et)
            data_0      = ind_df[level][tf_0]
            data_1      = ind_df[level][tf_1]

            mean_0      = np.nanmean(data_0)
            mean_1      = np.nanmean(data_1)
            mean_diff   = mean_1 - mean_0

            if mean_diff < 0.75:
                bad_event = True

            tmp_cndxs = []
            tmp_cndxs.append( (datetime.timedelta(hours=6), 0.95, 8.00) )
            tmp_cndxs.append( (datetime.timedelta(hours=4), 0.95, 7.00) )
            tmp_cndxs.append( (datetime.timedelta(hours=2), 1.00, 6.00) )

            # Test Condition
            for delt,quant,thrsh in tmp_cndxs:
                test_st     = zi - delt
                test_et     = zi
                test_tf     = np.logical_and(ind_df.index >= test_st, ind_df.index < test_et)
                test_data   = ind_df[level][test_tf]
                if test_data.quantile(quant) > thrsh:
                    bad_event = True
                    break

            if bad_event: continue

            arg_epoch_0.append(zi)

        arg_epoch_0         = np.array(arg_epoch_0)
        self.arg_epoch_0    = arg_epoch_0
        self.maxima_level   = level
        return arg_epoch_0

    def find_endpoint(self,level='ind_0_processed'):
        ind_df      = self.ind_df
        bound_pairs = self.bound_pairs

        arg_epoch_0 = []
        for bp_0, bp_1 in bound_pairs:
            midp = bp_1
            inx     = np.argmin(np.abs(ind_df.index - midp))
            arg_epoch_0.append(ind_df.index[inx])

        arg_epoch_0      = [x.to_datetime() for x in arg_epoch_0]
        arg_epoch_0      = np.array(arg_epoch_0)
        self.maxima_level   = level
        self.arg_epoch_0 = arg_epoch_0
        return arg_epoch_0

    def find_midpoint(self,level='ind_0_processed'):
        ind_df      = self.ind_df
        bound_pairs = self.bound_pairs

        arg_epoch_0 = []
        for bp_0, bp_1 in bound_pairs:
            midp = bp_0 + (bp_1-bp_0)/2
            inx     = np.argmin(np.abs(ind_df.index - midp))
            arg_epoch_0.append(ind_df.index[inx])

        arg_epoch_0      = [x.to_datetime() for x in arg_epoch_0]
        arg_epoch_0      = np.array(arg_epoch_0)
        self.arg_epoch_0 = arg_epoch_0
        self.maxima_level = level
        return arg_epoch_0

    def find_pressure_pulses(self,time_before=datetime.timedelta(hours=1),
            lower_thresh=4,level='ind_0_processed'):
        ind_df      = self.ind_df
        bound_pairs = self.bound_pairs

        self.find_maxima()
        arg_epoch_0_list = self.arg_epoch_0.tolist()
        for ep_0 in arg_epoch_0_list[:]:
            tf_0 = np.logical_and(ind_df.index >= ep_0-time_before, ind_df.index < ep_0)

            vals = ind_df[level][tf_0] 
            tf_1 = vals <= lower_thresh

            cnt = np.count_nonzero(tf_1)

#            if np.float(cnt)/np.float(np.size(tf_1)) < 0.6:
#                arg_epoch_0_list.remove(ep_0)

        arg_epoch_0      = np.array(arg_epoch_0_list)
        self.arg_epoch_0 = arg_epoch_0
        self.maxima_level = level
        return arg_epoch_0

    def check_event_quality(self,time_prior,time_post,thresh=0.90,level='ind_0_raw'):
        if not hasattr(self,'arg_epoch_0'): return

        df = self.ind_df_raw

        arg_epoch_0 = self.arg_epoch_0.tolist()
        for t_0 in arg_epoch_0[:]:
            st  = t_0 - time_prior
            et  = t_0 + time_post

            tf  = np.logical_and(df.index >= st, df.index < et)

            total   = len(df[level][tf])
            good    = np.count_nonzero(np.isfinite(df[level][tf]))

            if np.float(good)/np.float(total) < thresh:
                arg_epoch_0.remove(t_0)
                
                for bp in self.bound_pairs[:]:
                    if t_0 >= bp[0] and t_0 <= bp[1]:
                        self.bound_pairs.remove(bp)

        self.arg_epoch_0 = np.array(arg_epoch_0)
        return self.arg_epoch_0

    def remove_events(self,reject_list):
        for rj in reject_list:
            for t_0 in self.arg_epoch_0[:]:
                if (t_0 >= rj[0]) and (t_0 < rj[1]):
                    arg_epoch_0 = self.arg_epoch_0.tolist()
                    arg_epoch_0.remove(t_0)
                    self.arg_epoch_0 = np.array(arg_epoch_0)

                    for bp in self.bound_pairs[:]:
                        if (t_0 >= bp[0]) and (t_0 < bp[1]):
                            self.bound_pairs.remove(bp)
        return self.arg_epoch_0

def solartime(observer, sun=ephem.Sun()):
    """
    Calculates solar local time given a PyEphem observer.
    This supports calculate_slt().
    """

    # From: http://stackoverflow.com/questions/13314626/local-solar-time-function-from-utc-and-longitude
    sun.compute(observer)
    # sidereal time == ra (right ascension) is the highest point (noon)
    hour_angle = observer.sidereal_time() - sun.ra
    ephem_slt = ephem.hours(hour_angle + ephem.hours('12:00')).norm  # norm for 24h
    # Note: ephem.hours is a float number that represents an angle in radians and converts to/from a string as "hh:mm:ss.ff".
    return ephem_slt/(2.*np.pi) * 24

def calculate_slt(ut_dtime,lats,lons,alt=250):
    """
    Calculate solar local time using PyEphem.
    ut_dtime:   UTC in datetime.datetime object.
    lats:       Array of latitudes.
    lons:       Array of longitudes.
    alt:        Altitude of of observer point in kilometers
    """

    lats = np.array(lats)
    lons = np.array(lons)
    shape = lats.shape

    lats = lats.flatten()
    lons = lons.flatten()

    lat_lons = zip(lats,lons)
    
    slts = []
    for lat,lon in lat_lons:
        if not np.isfinite(lat) or not np.isfinite(lon):
            slts.append(np.nan)
            continue
        o           = ephem.Observer()
        o.lon       = np.radians(lon)
        o.lat       = np.radians(lat)
        o.elevation = alt
        o.date      = ut_dtime
        slt         = solartime(o)
        slts.append(slt)
    slts = np.array(slts)
    slts.shape = shape
    return slts 

def calculate_mlt(ut_dtime,mlons,alt=250):
    mlons   = np.array(mlons)
    shape   = mlons.shape

    mlons   = mlons.flatten()

    tf      = np.isfinite(mlons)

    yr      = ut_dtime.year
    mo      = ut_dtime.month
    dy      = ut_dtime.day
    hr      = ut_dtime.hour
    mt      = ut_dtime.minute
    sc      = ut_dtime.second

    mlts_out    = [davitpy.models.aacgm.mltFromYmdhms(yr,mo,dy,hr,mt,sc,mlon) for mlon in mlons[tf]]

    mlts        = np.zeros_like(mlons)
    mlts[:]     = np.nan
    mlts[tf]    = mlts_out
    mlts        = mlts.reshape(shape)
    return mlts

def estimate_slt(ut_dtime,lat,lon):
    """
    Calculate solar local time (SLT) using a quick and dirty formula.
    Seems to be off from calculate_slt() on the order of 10 minutes or so
    for the one case I tested.

    ut_dtime:   UTC in datetime.datetime object.
    lat:        Array of latitudes.
    lon:        Array of longitudes.
    """
    utc = ut_dtime.hour + ut_dtime.minute/60.
    slt_est = utc + (lon/360.)*24.
    return slt_est

class ConcatenateMusic(music.musicDataObj):
    def __init__(self,radar,sTime,eTime,ds_name='DS007_detrended',base_path='',
                tselect=None,
                fov=None, fovModel='GS', fovCoords='geo', fovElevation=None,
                time_limits=None,beam_limits=None,gate_limits=(10,50),
                comment=None, parent=0, **metadata):
        """ 
        Find processed SuperDARN data in MUSIC Objects and put it all into a 
        single object for the purpose of doing statistics on each cell.
        """
        radStruct = pydarn.radar.radStruct.radar(code=radar)
        site      = pydarn.radar.radStruct.site(code=radar,dt=sTime)

        max_beam, max_gate = [-1] * 2
        curr_data_objs  = []
        datetimes       = []
        sources         = []
        gate_lims       = []
        prm             = None

        if ds_name == 'fitex':
            # Load in data and create data objects. ########################################
            try:
                myPtr       = pydarn.sdio.radDataOpen(sTime,radar,eTime=eTime,filtered=False)
                # gscat = 1: ground scatter only
                data_obj    = music.musicArray(myPtr,fovModel=fovModel,gscat=1,full_array=True)
                myPtr.close()
            except:
                return

            if not hasattr(data_obj,'active'):
                return

            gate_min = 0
            gate_max = data_obj.active.fov.gates.max()

            gate_lims.append(gate_min)
            gate_lims.append(gate_max)

            curr_data   = data_obj.active
            prm         = data_obj.prm

            mlats, mlons    = my_calc_aacgm(curr_data.fov.latCenter,curr_data.fov.lonCenter,sTime.year)
            curr_data.fov.mlatCenter  = mlats
            curr_data.fov.mlonCenter  = mlons
            fov = curr_data.fov

            curr_data.setMetadata(pkl_sTime=sTime)
            curr_data.setMetadata(pkl_eTime=eTime)

            if curr_data.fov.beams.max() > max_beam:
                max_beam = curr_data.fov.beams.max()

            if curr_data.fov.gates.max() > max_gate:
                max_gate = curr_data.fov.gates.max()
            
            datetimes   = data_obj.active.time.tolist()
            curr_data_objs.append(curr_data)

#            bad = False # Innocent until proven guilty.
#            if hasattr(dataObj,'messages'):
#                if 'No data for this time period.' in dataObj.messages:
#                    bad = True # At this point, proven guilty.
#            
#            if not bad:
#                dataObj = music.checkDataQuality(dataObj,dataSet='originalFit',sTime=sDatetime,eTime=fDatetime)
            
        else:
            # Find all of the possible directories that MUSIC radar data can be stored in.
            globs = glob.glob(os.path.join(base_path,radar,'*'))

            # If no FOV, generate a generic one. ###########################################
            if fov is None:
    #            fov       = pydarn.radar.radFov.fov(frang=myBeam.prm.frang, rsep=myBeam.prm.rsep, site=site,elevation=fovElevation,model=fovModel,coords=fovCoords)
                fov             = pydarn.radar.radFov.fov(site=site,elevation=fovElevation,model=fovModel,coords=fovCoords)
                mlats, mlons    = my_calc_aacgm(fov.latCenter,fov.lonCenter,sTime.year)
                fov.mlatCenter  = mlats
                fov.mlonCenter  = mlons

            # Choose a point for a synoptic Local Time determination.
            lt_rg = 25
            lt_bm = int(np.median(fov.beams))

            lt_lat = fov.latCenter[lt_bm,lt_rg]
            lt_lon = fov.lonCenter[lt_bm,lt_rg]

            # Find all pickle files that seem to meet the radar/date criteria, 
            # but don't load anything yet.
            pkl_paths = []
            for dr in globs:
                if not os.path.isdir(dr): continue
                basename = os.path.basename(dr)
                pkl_sTime   = datetime.datetime.strptime(basename[:13],'%Y%m%d.%H%M')
                pkl_eTime   = datetime.datetime.strptime(basename[14:],'%Y%m%d.%H%M')

                if pkl_sTime < sTime or pkl_sTime >= eTime: continue

                if (tselect is not None):
                    pkl_mTime   = pkl_sTime + (pkl_eTime - pkl_sTime)/2
                    slt         = calculate_slt(pkl_mTime,lt_lat,lt_lon)
    #                slt_est     = estimate_slt(pkl_mTime,lt_lat,lt_lon)

                    if slt < tselect[0] or slt >= tselect[1]:
                        continue

                pkl_name    = '{0}-{1}.p'.format(radar,basename)
                pkl_path    = os.path.join(dr,pkl_name)
                if os.path.exists(pkl_path):
                        pkl_paths.append( (pkl_path,pkl_sTime,pkl_eTime) )
                        logging.info('Found pkl file: {}'.format(pkl_path))
                        print('Found pkl file: {}'.format(pkl_path))
                else:
                    print('Pkl file not found: {}'.format(pkl_path))

            # Load dataSets for each pickle object.
            # Keep track of the maximum beam/gate dimensions.
            # Keep track of all of the time stamps.
            for pkl_path,pkl_sTime,pkl_eTime in pkl_paths:
                logging.info('Loading pkl file: {}'.format(pkl_path))
                with open(pkl_path,'rb') as fl:
                    data_obj    = pickle.load(fl)

                data_sets = [x for x in data_obj.get_data_sets() if ds_name in x]

                if len(data_sets) == 0:
                    logging.info('Warning: No matching data sets for {}.'.format(os.path.basename(pkl_path)))
                    continue
                elif len(data_sets) > 1:
                    logging.info('Warning: Multiple data sets found for {}; using first found.'.format(os.path.basename(pkl_path)))

                # Check for bad data (specifically, periods > 10 min
                # where the radar was not operational)
                music.checkDataQuality(data_obj,data_sets[0])
                curr_data = getattr(data_obj,data_sets[0])
                if not curr_data.metadata['good_period']: continue
                
#                gate_lims.append( data_obj.active.fov.gates.min() )
#                gate_lims.append( data_obj.active.fov.gates.max() )

                try:
                    gate_lims.append( data_obj.active.metadata['gateLimits'][0] )
                except:
                    pass

                try:
                    gate_lims.append( data_obj.active.metadata['gateLimits'][1] )
                except:
                    pass

                mlats, mlons    = my_calc_aacgm(curr_data.fov.latCenter,curr_data.fov.lonCenter,sTime.year)
                curr_data.fov.mlatCenter  = mlats
                curr_data.fov.mlonCenter  = mlons

                curr_data.setMetadata(pkl_sTime=pkl_sTime)
                curr_data.setMetadata(pkl_eTime=pkl_eTime)

                if curr_data.fov.beams.max() > max_beam:
                    max_beam = curr_data.fov.beams.max()

                if curr_data.fov.gates.max() > max_gate:
                    max_gate = curr_data.fov.gates.max()
                
                curr_data_objs.append(curr_data)
                datetimes = datetimes + curr_data.time.tolist()
                sources.append(pkl_path)

        # Create the datetimes array.  Make sure it is sorted and all entries are unique.
        datetimes.sort()
        datetimes = np.unique(datetimes)
        
        # Initialize array to hold all data.
        data_arr    = np.zeros([datetimes.size,fov.beams.max()+1,fov.gates.max()+1],dtype=np.float)
        data_arr[:] = np.nan

        slt_arr     = np.zeros_like(data_arr)
        slt_arr[:]  = np.nan

        mlt_arr     = np.zeros_like(data_arr)
        mlt_arr[:]  = np.nan

        # Populate the data_arr.
        for curr_data_inx,curr_data in enumerate(curr_data_objs):
            logging.info('Curr_data_obj {} of {}'.format(str(curr_data_inx+1),str(len(curr_data_objs))))
            for curr_tm_inx,tm in enumerate(curr_data.time):
                if tm < curr_data.metadata['pkl_sTime'] or tm >= curr_data.metadata['pkl_eTime']:
                    continue

                glbl_tm_inx = np.where(datetimes == tm)[0]
                grd_inx     = np.meshgrid(glbl_tm_inx, curr_data.fov.beams, curr_data.fov.gates,indexing='ij')

                data_arr[grd_inx] = curr_data.data[curr_tm_inx,:,:].reshape(grd_inx[1].shape)

#                slt_est = estimate_slt(tm,curr_data.fov.latCenter,curr_data.fov.lonCenter)
                logging.info('Populating array: {} {}'.format(str(curr_tm_inx),str(tm)))
                slt     = calculate_slt(tm,curr_data.fov.latCenter,curr_data.fov.lonCenter)
                slt_arr[grd_inx] = slt

                mlt     = calculate_mlt(tm,curr_data.fov.mlonCenter)
                mlt_arr[grd_inx] = mlt

        logging.info('Done populating')
        if data_arr.size != 0:
            #Make metadata block to hold information about the processing.
            # Assuming that some infomation from the very last loaded curr_data is correct for all.
            # This is not a reliable assumption!! But, it is probably true and can give some context.
            # Just don't trust it too much without double checking things!!
            param   = curr_data.metadata.get('param') 
            prm_dct = pydarn.radar.getParamDict(param)

    #        metadata['dType']     = myPtr.dType
            metadata['stid']      = curr_data.metadata.get('stid') 
            metadata['name']      = radStruct.name
            metadata['code']      = str([x for x in radStruct.code if len(x) == 3][0])
            metadata['radar']      = str([x for x in radStruct.code if len(x) == 3][0])
    #        metadata['fType']     = curr_data.metadata.get('fType') 
    #        metadata['cp']        = myPtr.cp
    #        metadata['channel']   = myPtr.channel
            metadata['sTime']     = sTime
            metadata['eTime']     = eTime
            metadata['param']     = param
            metadata['param_label'] = prm_dct['label']
            metadata['gscat']     = curr_data.metadata.get('gscat') 
            metadata['elevation'] = fovElevation
            metadata['model']     = fovModel
            metadata['coords']    = fovCoords

            metadata['dataSetName'] = ds_name
            metadata['pkl_sources'] = sources
            metadata['serial']      = 0


        if comment is None:
            dform = '%Y%m%d.%H%M'
            comment = 'Concatenated MUSICObj: {0} {1}-{2} UT'.format(radar.upper(),sTime.strftime(dform),eTime.strftime(dform))

        super(ConcatenateMusic,self).__init__(datetimes,data_arr,
                fov=fov, comment=comment, parent=parent, **metadata)

        self.slt_arr    = slt_arr
        self.mlt_arr    = mlt_arr
        self.prm        = prm
    
        if data_arr.size != 0:
            if gate_limits is None:
                if np.size(gate_lims) < 2: gate_lims = (self.fov.gates.min(), self.fov.gates.max())
                gate_limits = ( np.nanmin(gate_lims), np.nanmax(gate_lims) )

            self.define_limits(time_limits=time_limits)
            self.define_limits(gate_limits=gate_limits)
            self.define_limits(beam_limits=beam_limits)
            self.apply_limits()

    def define_limits(self,range_limits=None,gate_limits=None,beam_limits=None,time_limits=None):
        """Sets the range, gate, beam, and time limits for the chosen data set. This method only changes metadata;
        it does not create a new data set or alter the data in any way.  If you specify range_limits, they will be changed to correspond
        with the center value of the range cell.  Gate limits always override range limits.
        Use the applyLimits() method to remove data outside of the data limits.

        **Args**:
            * [**range_limits**] (iterable or None): Two-element array defining the maximum and minumum slant ranges to use. [km]
            * [**gate_limits**] (iterable or None): Two-element array defining the maximum and minumum gates to use.
            * [**beam_limits**] (iterable or None): Two-element array defining the maximum and minumum beams to use.
            * [**time_limits**] (iterable or None): Two-element array of datetime.datetime objects defining the maximum and minumum times to use.

        Written by Nathaniel A. Frissell, Fall 2013
        """
        if (range_limits is not None) or (gate_limits is not None):
            if (range_limits is not None) and (gate_limits == None):
                inx = np.where(np.logical_and(self.fov.slantRCenter >= range_limits[0],self.fov.slantRCenter <= range_limits[1]))
                gate_limits = [np.min(inx[1][:]),np.max(inx[1][:])]

            if gate_limits is not None:
                rangeMin = np.int(np.min(self.fov.slantRCenter[:,gate_limits[0]]))
                rangeMax = np.int(np.max(self.fov.slantRCenter[:,gate_limits[1]]))
                range_limits = [rangeMin,rangeMax]

            self.metadata['gateLimits']  = gate_limits
            self.metadata['rangeLimits'] = range_limits

        if beam_limits is not None:
            self.metadata['beamLimits'] = beam_limits

        if time_limits is not None:
            self.metadata['timeLimits'] = time_limits

    def apply_limits(self):
        """Removes data outside of the range_limits and gate_limits boundaries.
        """
        #Make a copy of the current data set.
        limits = self.metadata.get('timeLimits')
        if limits:
            limits  = np.array(limits)
            if limits[0] is None: limits[0] = np.min(self.time)
            if limits[1] is None: limits[1] = np.max(self.time) + datetime.timedelta(1)
            time_tf = np.logical_and(self.time >= limits[0], self.time < limits[1])
        else:
            time_tf = np.ones_like(self.time,dtype=np.bool)

        limits = self.metadata.get('beamLimits')
        if limits:
            limits  = np.array(limits)
            if limits[0] is None: limits[0] = np.min(self.fov.beams)
            if limits[1] is None: limits[1] = np.max(self.fov.beams) + datebeam.beamdelta(1)
            beam_tf         = np.logical_and(self.fov.beams >= limits[0], self.fov.beams < limits[1])
            beam_tf_full    = np.logical_and(self.fov.beams >= limits[0], self.fov.beams < limits[1]+1)
        else:
            beam_tf         = np.ones([len(self.fov.beams)],dtype=np.bool)
            beam_tf_full    = np.ones([len(self.fov.beams)+1],dtype=np.bool)

        limits = self.metadata.get('gateLimits')
        if limits:
            limits  = np.array(limits)
            if limits[0] is None: limits[0] = np.min(self.fov.gates)
            if limits[1] is None: limits[1] = np.max(self.fov.gates) + dategate.gatedelta(1)
            gate_tf         = np.logical_and(self.fov.gates >= limits[0], self.fov.gates < limits[1])
            gate_tf_full    = np.logical_and(self.fov.gates >= limits[0], self.fov.gates < limits[1]+1)
        else:
            gate_tf         = np.ones([len(self.fov.gates)],dtype=np.bool)
            gate_tf_full    = np.ones([len(self.fov.gates)+1],dtype=np.bool)

        time_inx,   = np.where(time_tf)
        beam_inx,   = np.where(beam_tf)
        gate_inx,   = np.where(gate_tf)

        beam_inx_full,  = np.where(beam_tf_full)
        gate_inx_full,  = np.where(gate_tf_full)

        tm_bm_rg                = np.meshgrid(time_inx,beam_inx,gate_inx,indexing='ij')
        bm_rg                   = np.meshgrid(beam_inx,gate_inx,indexing='ij')
        bm_rg_full              = np.meshgrid(beam_inx_full,gate_inx_full,indexing='ij')

        self.time               = self.time[time_tf]
        self.data               = self.data[tm_bm_rg]
        self.fov.beams          = self.fov.beams[beam_tf]
        self.fov.gates          = self.fov.gates[gate_tf]

        self.fov.latCenter      = self.fov.latCenter[bm_rg]
        self.fov.lonCenter      = self.fov.lonCenter[bm_rg]
        self.fov.slantRCenter   = self.fov.slantRCenter[bm_rg]

        self.fov.mlatCenter    = self.fov.mlatCenter[bm_rg] 
        self.fov.mlonCenter    = self.fov.mlonCenter[bm_rg]

        self.mlt_arr            = self.mlt_arr[tm_bm_rg]
        self.slt_arr            = self.slt_arr[tm_bm_rg]

        #Update the full FOV.
        self.fov.latFull        = self.fov.latFull[bm_rg_full]
        self.fov.lonFull        = self.fov.lonFull[bm_rg_full]
        self.fov.slantRFull     = self.fov.slantRFull[bm_rg_full]

        try:
            self.fov.relative_azm      = self.fov.relative_azm[bm_rg]
            self.fov.relative_range    = self.fov.relative_range[bm_rg]
            self.fov.relative_x        = self.fov.relative_x[bm_rg]
            self.fov.relative_y        = self.fov.relative_y[bm_rg]
        except:
            pass

    def __select_max_rg_inx__(self,beam,win_bounds=None):
        if win_bounds is not None:
            tf = np.logical_and(self.time >= win_bounds[0],self.time < win_bounds[1])
        else:
            tf  = np.ones_like(self.time,dtype=np.bool)

        try:
            input_data  = self.data[tf,beam,:]
            input_time  = self.time[tf]
        except:
            return None

        rg_sum = np.nansum(input_data,axis=0)
        if rg_sum.max() == 0:
            return None
        else:
            rg_inx  = rg_sum.argmax()
#            rg      = self.fov.gates[rg_inx]
            return rg_inx

    def rolling_lombpd(self,window=datetime.timedelta(hours=2),beam=7,gate_select='window',
            f_range = (0.00001,0.0015), nr_f=150,hanning_window=True,
            max_tfreq_diff=600,test_plot=True,test_data=False):
        """
        gate_select: 'window': Select range gate based on data from the current time window.
                     'all': Select range gate based on data from the entire data object
        """
        import scipy.signal as signal

        gates_inx_vec       = np.zeros_like(self.time)
        gates_inx_vec[:]    = np.nan
        slt_vector          = gates_inx_vec.copy()
        mlt_vector          = gates_inx_vec.copy()

        freq_vec    = np.linspace(f_range[0],f_range[1],nr_f)
        omega_vec   = 2. * np.pi * freq_vec
        spect_arr   = np.zeros([self.time.size,nr_f],dtype=np.float)
        spect_arr[:] = np.nan

        if hanning_window:
            # Calculate a window function.
            nr_sec  = int(window.total_seconds())
            win_xx  = np.arange(nr_sec)
            win_fn  = np.hanning(nr_sec)

        # Reject time periods that have data gaps longer than a certain fraction of the window length.
        max_gap = window/8

        gates_inx_vec       = np.zeros_like(self.time)
        if gate_select == 'window':
            for tm_inx,tm in enumerate(self.time):
                win_sTime = tm - window
                win_eTime = tm 

                rg_inx = self.__select_max_rg_inx__(beam,win_bounds=(win_sTime,win_eTime))
                gates_inx_vec[tm_inx] = rg_inx
        else:
            rg_inx = self.__select_max_rg_inx__(beam)
            gates_inx_vec[:] = rg_inx

        if self.prm is not None:
            prm_time    = np.array(self.prm.time)
            prm_tfreq   = np.array(self.prm.tfreq)

        for tm_inx,(tm,rg_inx) in enumerate(zip(self.time,gates_inx_vec)):
            if rg_inx is None: continue

            slt_vector[tm_inx] = self.slt_arr[tm_inx,beam,rg_inx]
            mlt_vector[tm_inx] = self.mlt_arr[tm_inx,beam,rg_inx]

            win_sTime = tm - window
            win_eTime = tm

            # Check for radar transmitting frequency discontinuities.
            if self.prm is not None:
                prm_tf      = np.logical_and(prm_time >= win_sTime, prm_time < win_eTime)
                rdr_tfreq   = prm_tfreq[prm_tf]

                if rdr_tfreq.size == 0: continue
                tfreq_diff  = np.max(rdr_tfreq) - np.min(rdr_tfreq) 
                if tfreq_diff > max_tfreq_diff: continue

            tf = np.logical_and(self.time >= win_sTime,self.time < win_eTime)
            input_data  = self.data[tf,beam,:]
            input_time  = self.time[tf]
            
            # 2012 Dec 03 2105 - 2012 Dec 04 0305: originalFit: WAL B07
            rg_data     = input_data[:,rg_inx]

            tf_good     = np.isfinite(rg_data)
            nr_good     = np.sum(tf_good)

            rg_data_good    = rg_data[tf_good]
            rg_time_good    = input_time[tf_good]
            rg_seconds      = np.array([x.total_seconds() for x in (rg_time_good-win_sTime)])

            if len(rg_time_good) < 2: continue
            gap_test_vec    = [win_sTime] + rg_time_good.tolist() + [win_eTime]
            gap_test_vec.sort()
            gaps            = np.diff(gap_test_vec)
            if gaps.max() > max_gap: continue

            if test_data:
                xx           = rg_seconds
                rg_data_good = 5.*np.sin(2*np.pi*0.0005*xx)# + 2.5*np.sin(2*np.pi*0.001*xx)

            res             = np.polyfit(rg_seconds,rg_data_good,1)
            trnd            = res[0]*rg_seconds + res[1]
            rg_data_good    = rg_data_good - trnd

            if hanning_window:
                win_pt  = np.interp(rg_seconds,win_xx,win_fn)
                rg_data_good = rg_data_good * win_pt

            # Calculate normalized periodogram.
            pgram   = signal.lombscargle(rg_seconds,rg_data_good,omega_vec)
            spec    = np.sqrt( 4.*pgram/nr_good )
            if hanning_window:
                spec = 2. * spec
            spect_arr[tm_inx,:] = spec

            if test_plot:
                time_xlim = (win_sTime, win_eTime)
                output_dir = 'output/lt_plots'
                fig = plt.figure(figsize=(10,9))

                #####
                ax  = fig.add_subplot(3,1,1)
                ax.plot(input_time,rg_data,label='Raw Data',marker='o',ls='.')
                ax.plot(rg_time_good,rg_data_good,label='Detrended Data',marker='o',ls='.')

                ax.legend(fontsize='xx-small')
                ax.set_xlabel('Time [UT]')
                ax.set_ylabel(u'$\lambda$ Power [dB]')
                ax.grid()
                ax.set_xlim(time_xlim)

                #####
                ax  = fig.add_subplot(3,1,2)
                ax.plot(rg_time_good,rg_data_good,label='Detrended Data',marker='o',ls='.')

#                if hanning_window:
#                    ax2     = ax.twinx()
#                    win_tm  = np.array( [datetime.timedelta(seconds=x) for x in win_xx] ) + win_sTime
#                    ax2.plot(win_tm,win_fn,color='g')
#                    ax2.plot(rg_time_good,win_pt,marker='o',ls='.',color='g')

                ax.legend(fontsize='xx-small')
                ax.set_xlabel('Time [UT]')
                ax.set_ylabel(u'$\lambda$ Power [dB]')
                ax.grid()
                ax.set_xlim(time_xlim)

                #####
                ax  = fig.add_subplot(3,1,3)
                xvals   = freq_vec*1.e3
                yvals   = spec
                ax.plot(xvals,yvals)

                ax.legend(fontsize='xx-small')
                ax.set_xlabel('Frequency [mHz]')
                ax.set_ylabel('Power Spectral Density')
                ax.grid()
                ax.set_xlim(0,1.5)

                #####
                radar       = self.metadata['radar']
                date_fmt    = '%Y %b %d %H%M UT'
                title = '{} - {}: {} B{:02d} G{:03d}'.format(win_sTime.strftime(date_fmt),
                    win_eTime.strftime(date_fmt),radar.upper(),int(beam),int(self.fov.gates[rg_inx]))
                fig.text(0.5,1,title,ha='center')

                fig.tight_layout()
                fig.savefig(os.path.join(output_dir,'lspgram_test.png'),bbox_inches='tight')
                plt.close(fig)

                test_plot = False

        tmp_dct            = {}
        tmp_dct['freq']    = freq_vec
        tmp_dct['spect']   = spect_arr
        tmp_dct['beam']    = beam
        tmp_dct['gates']   = np.array( [self.fov.gates[x] for x in gates_inx_vec] )
        tmp_dct['window']  = window
        tmp_dct['time']    = self.time
        tmp_dct['slt']      = slt_vector
        tmp_dct['mlt']      = mlt_vector
        tmp_dct['ylabel']   = 'LSP Frequency [mHz]\nLSP Period [min]'
        tmp_dct['spec_type'] = 'Lomb_Scargle'

        self.lspgram_dict       = tmp_dct
        return self.lspgram_dict

    def rolling_fft(self,window=datetime.timedelta(hours=2),beam=7,gate_select='window',
            f_range = (0.00001,0.0015), nr_f=4096,
            max_tfreq_diff=600,test_plot=True,test_data=False):
        import scipy.signal as signal

        freq    = np.fft.fftfreq(nr_f, 60.)
        freq_tf = np.logical_and(freq >= f_range[0], freq < f_range[1])

        gates_inx_vec       = np.zeros_like(self.time)
        gates_inx_vec[:]    = np.nan
        slt_vector          = gates_inx_vec.copy()
        mlt_vector          = gates_inx_vec.copy()

        freq_vec    = freq[freq_tf]
        omega_vec   = 2. * np.pi * freq_vec
        spect_arr   = np.zeros([self.time.size,freq_vec.size],dtype=np.float)
        spect_arr[:] = np.nan

        # Reject time periods that have data gaps longer than a certain fraction of the window length.
        max_gap = window/8

        gates_inx_vec       = np.zeros_like(self.time)
        if gate_select == 'window':
            for tm_inx,tm in enumerate(self.time):
                win_sTime = tm - window
                win_eTime = tm # + window

                rg_inx = self.__select_max_rg_inx__(beam,win_bounds=(win_sTime,win_eTime))
                gates_inx_vec[tm_inx] = rg_inx
        else:
            rg_inx = self.__select_max_rg_inx__(beam)
            gates_inx_vec[:] = rg_inx

        if self.prm is not None:
            prm_time    = np.array(self.prm.time)
            prm_tfreq   = np.array(self.prm.tfreq)

        for tm_inx,(tm,rg_inx) in enumerate(zip(self.time,gates_inx_vec)):
            if rg_inx is None:
#                logging.info('rg_inx is None: continue ({!s}, {})'.format(tm,self.metadata['radar']))
                continue

            slt_vector[tm_inx] = self.slt_arr[tm_inx,beam,rg_inx]
            mlt_vector[tm_inx] = self.mlt_arr[tm_inx,beam,rg_inx]

            win_sTime = tm - window
            win_eTime = tm #+ window

            # Check for radar transmitting frequency discontinuities.
            if self.prm is not None:
                prm_tf      = np.logical_and(prm_time >= win_sTime, prm_time < win_eTime)
                rdr_tfreq   = prm_tfreq[prm_tf]

                if rdr_tfreq.size == 0: continue
                tfreq_diff  = np.max(rdr_tfreq) - np.min(rdr_tfreq) 
                if tfreq_diff > max_tfreq_diff:
#                    logging.info('tfreq_diff > max_tfreq_diff: continue ({!s}, {})'.format(tm,self.metadata['radar']))
                    continue

            tf = np.logical_and(self.time >= win_sTime,self.time < win_eTime)
            input_data  = self.data[tf,beam,:]
            input_time  = self.time[tf]

            rg_data     = input_data[:,rg_inx]

            tf_good     = np.isfinite(rg_data)
            rg_time_good    = input_time[tf_good]

            if len(rg_time_good) < 2: 
#                logging.info('len(rg_time_good) < 2: continue ({!s}, {})'.format(tm,self.metadata['radar']))
                continue
            gap_test_vec    = [win_sTime] + rg_time_good.tolist() + [win_eTime]
            gap_test_vec.sort()
            gaps            = np.diff(gap_test_vec)
            if gaps.max() > max_gap: 
#                logging.info('gaps.max() > max_gap: continue ({!s}, {})'.format(tm,self.metadata['radar']))
                continue

            # Put things into a data frame for easier resampling.
            df  = pd.DataFrame({'p_l':self.data[:,beam,rg_inx]},index=self.time)
            df.dropna(inplace=True)

            if test_data:
                xx  = np.array([x.total_seconds() for x in df.index-win_sTime])
                df['p_l'] = 5.*np.sin(2*np.pi*0.0005*xx) #+ 2.5*np.sin(2*np.pi*0.001*xx)

            df  = df.resample('T')
            df  = df.interpolate()
            df  = df[np.logical_and(df.index >= win_sTime, df.index < win_eTime)]

            df['det'] = sp.signal.detrend(df['p_l'])
            df['det'] = df['det'] * np.hanning(len(df))

            # Hanning Window Correction Factor = 2.
            # FFT Scaling Factor = 1/sqrt(nr_f)
            spec    = 2.*np.abs(np.fft.fft(df['det'],n=nr_f)) / np.sqrt(nr_f)

            if test_plot:
                time_xlim = (win_sTime, win_eTime)
                output_dir = 'output/lt_plots'
                fig = plt.figure(figsize=(10,9))

                #####
                ax  = fig.add_subplot(3,1,1)
                ax.plot(input_time,rg_data,label='Raw Data',marker='o',ls='.')

                ax.plot(df.index,df.p_l,label='Interpolated Data')

                ax.legend(fontsize='xx-small')
                ax.set_xlabel('Time [UT]')
                ax.set_ylabel(u'$\lambda$ Power [dB]')
                ax.grid()
                ax.set_xlim(time_xlim)

                #####
                ax  = fig.add_subplot(3,1,2)
                ax.plot(df.index,df['det'],label='Detrended Data')

                ax.legend(fontsize='xx-small')
                ax.set_xlabel('Time [UT]')
                ax.set_ylabel(u'$\lambda$ Power [dB]')
                ax.grid()
                ax.set_xlim(time_xlim)

                #####
                ax  = fig.add_subplot(3,1,3)
                xvals   = freq*1.e3
                yvals   = spec
                ax.plot(xvals,yvals)

                ax.legend(fontsize='xx-small')
                ax.set_xlabel('Frequency [mHz]')
                ax.set_ylabel('Power Spectral Density')
                ax.grid()
                ax.set_xlim(0,1.5)

                #####
                radar       = self.metadata['radar']
                date_fmt    = '%Y %b %d %H%M UT'
                title = '{} - {}: {} B{:02d} G{:03d}'.format(win_sTime.strftime(date_fmt),
                    win_eTime.strftime(date_fmt),radar.upper(),int(beam),int(self.fov.gates[rg_inx]))
                fig.text(0.5,1,title,ha='center')

                fig.tight_layout()
                fig.savefig(os.path.join(output_dir,'fft_test.png'),bbox_inches='tight')
                plt.close(fig)

                test_plot = False


#            logging.info("Hey, man, we're cool. ({!s}, {})".format(tm,self.metadata['radar']))
            spect_arr[tm_inx,:] = spec[freq_tf]

        tmp_dct             = {}
        tmp_dct['freq']     = freq_vec
        tmp_dct['spect']    = spect_arr
        tmp_dct['beam']     = beam
        tmp_dct['gates']    = np.array( [self.fov.gates[x] for x in gates_inx_vec] )
        tmp_dct['window']   = window
        tmp_dct['time']     = self.time
        tmp_dct['slt']      = slt_vector
        tmp_dct['mlt']      = mlt_vector
        tmp_dct['ylabel']   = 'FFT Frequency [mHz]\nFFT Period [min]'
        tmp_dct['spec_type'] = 'FFT'

        self.fft_dict        = tmp_dct
        return self.fft_dict
            
def get_radar_data(sTime,eTime,radar,
    beams       = None,
    ds_name     = 'DS007_detrended',
    data_dir    = 'data',
    base_path   = 'music_data/paper2',
    use_cache   = True,
    save        = True
        ):
    """ Generate a data_dict containing aggregated data from many radars.

    You can even have it automatically select the boresight beam!!!
    beams       = 'boresight'
    """

    if beams is not None:
        if beams == 'boresight':
            beamstr = '_'+beams
        else:
            beams = np.array(beams)
            beamstr = '_beams'+''.join(['{0:02d}'.format(x) for x in beams])
    else:
        beamstr = ''

    # Define the filename of the datadict file.
    tstring     = '{}-{}'.format(sTime.strftime('%Y%m%d.%H%M'),eTime.strftime('%Y%m%d.%H%M'))
    filename    = 'catMusic_{time}_{dataset}_{radar}{beamstr}.p'.format(time=tstring,dataset=ds_name,radar=radar,beamstr=beamstr)
    filepath    = os.path.join(data_dir,filename)

    if use_cache:
        if os.path.exists(filepath):
            with open(filepath) as fl:
                return pickle.load(fl)

    print 'concat_music.py: {}'.format(radar)
    catMusic = ConcatenateMusic(radar,sTime,eTime,ds_name=ds_name,base_path=base_path,beams=beams)

    # Make sure there is actually data there!
    if not hasattr(catMusic,'time'):
        catMusic = None
    elif len(catMusic.time) == 0:
        catMusic = None

    if save:
        try:
            os.makedirs(data_dir)
        except:
            pass

        with open(filepath,'wb') as fl:
            pickle.dump(catMusic,fl)

    return catMusic
