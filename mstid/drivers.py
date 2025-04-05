# import os
# import datetime
# import hdf5
# import copy
# import re
# import gme
# import numpy as np
# import scipy as sp
# import pandas as pd
# from pyDARNmusic import epem
# import matplotlib
# matplotlib.use('Agg')
# from matplotlib import pyplot as plt

# #from mpl_toolkits.basemap import solar

# #import davitpy.gme as gme
# #import davitpy.pydarn.proc.signal as signal

# #from . import general_lib import prepare_output_dirs
# from . import mongo_tools

# from . import run_helper
# from . import calendar_plot

# import inspect
# curr_file = inspect.getfile(inspect.currentframe()) # script filename (usually with path)

# class Driver(object):
#     """
#     Create a container object to help us keep track of changes made to a time series.
#     """
#     def __init__(self,sTime,eTime,gme_param,plot_info={},**kwargs):

#         data_set                ='DS000_raw'
#         comment                 ='Raw Data'
#         plot_info['data_set']   = data_set 
#         plot_info['serial']     = 0
#         plot_info['gme_param']  = gme_param
#         plot_info['sTime']      = sTime
#         plot_info['eTime']      = eTime
#         plot_info['season']     = '{!s}_{!s}'.format(sTime.year,eTime.year)
#         comment_str             = '[{}] {}'.format(data_set,comment)

#         d0      = DriverObj(sTime,eTime,gme_param,plot_info=plot_info,
#                 comment=comment_str,parent=self,**kwargs)
#         setattr(self,data_set,d0)
#         d0.set_active()
#         d0.set_secondary()

#     def get_data_set(self,data_set='active'):
#         """
#         Get a data_set, even one that only partially matches the string.
#         """
#         lst = dir(self)

#         if data_set not in lst:
#             tmp = []
#             for item in lst:
#                 if data_set in item:
#                     tmp.append(item)
#             if len(tmp) == 0:
#                 data_set = 'active'
#             else:
#                 tmp.sort()
#                 data_set = tmp[-1]

#         return getattr(self,data_set)

#     def get_all_data_sets(self):
#         """
#         Return a list of all data set objects.
#         """
#         ds_names    = self.list_all_data_sets()
#         data_sets   = [getattr(self,dsn) for dsn in ds_names]
#         return data_sets
    
#     def list_all_data_sets(self):
#         """
#         Return a list of the names of all data sets associated with this object.
#         """
#         lst = dir(self)
    
#         data_sets   = []
#         for item in lst:
#             if re.match('DS[0-9]{3}_',item):
#                 data_sets.append(item)

#         return data_sets

#     def update_all_metadata(self,**kwargs):
#         """
#         Update the metadata/plot_info dictionaries of ALL attached data sets.
#         """
#         data_sets   = self.get_all_data_sets()
#         for ds in data_sets:
#             ds.metadata.update(kwargs)

#     def copy(self):
#         return copy.deepcopy(self)

#     def delete_not_active(self):
#         """
#         Delete all but the active dataset.
#         """

#         ds_names    = self.list_all_data_sets()
#         for ds_name in ds_names:
#             if getattr(self,ds_name) is not self.active:
#                 delattr(self,ds_name)
#         return self


# class DriverObj(object):
#     def __init__(self,sTime,eTime,gme_param,mstid_reduced_inx=None,
#             oversamp_T=datetime.timedelta(minutes=1),neg_mbar_diff_zscore=True,
#             plot_info=None,comment=None,parent=None):

#         # Save the input plot_info to override default data later.
#         _plot_info          = plot_info
#         plot_info           = {}
#         plot_info['sTime']  = sTime
#         plot_info['eTime']  = eTime

#         if 'omni' in gme_param: 
#             ind_class   = gme.ind.readOmni(sTime,eTime,res=1)
#             omni_list   = []
#             omni_time   = []
#             for xx in ind_class:
#                 tmp = {}

# #                tmp['res']          = xx.res
# #                tmp['timeshift']    = xx.timeshift
# #                tmp['al']           = xx.al
# #                tmp['au']           = xx.au
# #                tmp['asyd']         = xx.asyd
# #                tmp['asyh']         = xx.asyh
# #                tmp['symd']         = xx.symd
# #                tmp['beta']         = xx.beta
# #                tmp['bye']          = xx.bye
# #                tmp['bze']          = xx.bze
# #                tmp['e']            = xx.e
# #                tmp['flowSpeed']    = xx.flowSpeed
# #                tmp['vxe']          = xx.vxe
# #                tmp['vye']          = xx.vye
# #                tmp['vzy']          = xx.vzy
# #                tmp['machNum']      = xx.machNum
# #                tmp['np']           = xx.np
# #                tmp['temp']         = xx.temp
# #                tmp['time']         = xx.time

#                 tmp['ae']           = xx.ae
#                 tmp['bMagAvg']      = xx.bMagAvg
#                 tmp['bx']           = xx.bx 
#                 tmp['bym']          = xx.bym
#                 tmp['bzm']          = xx.bzm
#                 tmp['pDyn']         = xx.pDyn
#                 tmp['symh']         = xx.symh
#                 tmp['flowSpeed']    = xx.flowSpeed
#                 tmp['np']           = xx.np
#                 tmp['temp']         = xx.temp
                
#                 omni_time.append(xx.time)
#                 omni_list.append(tmp)

#             omni_df_raw         = pd.DataFrame(omni_list,index=omni_time)
#             del omni_time
#             del omni_list

#             self.omni_df_raw    = omni_df_raw
#             self.omni_df        = omni_df_raw.resample('T')
#             self.omni_df        = self.omni_df.interpolate()

#         plot_info['x_label']    = 'Date [UT]'
#         if gme_param == 'ae':
#             # Read data with DavitPy routine and place into numpy arrays.
#             ind_class   = gme.ind.readAe(sTime,eTime,res=1)
#             ind_data    = [(x.time, x.ae) for x in ind_class]

#             df_raw          = pd.DataFrame(ind_data,columns=['time','ind_0_raw'])
#             df_raw          = df_raw.set_index('time')

#             plot_info['title']              = 'Auroral Electrojet (AE) Index'
#             plot_info['ind_0_symbol']       = 'Auroral Electrojet (AE) Index'
#             plot_info['ind_0_gme_label']    = 'AE Index [nT]'

#         elif (gme_param == 'omni_by'):
#             df_raw  = pd.DataFrame(omni_df_raw['bym'])

#             plot_info['ind_0_symbol']     = 'OMNI By'
#             plot_info['ind_0_gme_label']  = 'OMNI By GSM [nT]'

#         elif gme_param == 'omni_bz':
#             df_raw  = pd.DataFrame(omni_df_raw['bzm'])

#             plot_info['ind_0_symbol']     = 'OMNI Bz'
#             plot_info['ind_0_gme_label']  = 'OMNI Bz GSM [nT]'

#         elif gme_param == 'omni_pdyn':
#             df_raw  = pd.DataFrame(omni_df_raw['pDyn'])

#             plot_info['ind_0_symbol']     = 'OMNI pDyn'
#             plot_info['ind_0_gme_label']  = 'OMNI pDyn [nPa]'

#         elif gme_param == 'omni_symh':
#             df_raw  = pd.DataFrame(omni_df_raw['symh'])

#             plot_info['title']              = 'OMNI Sym-H'
#             plot_info['ind_0_symbol']       = 'OMNI Sym-H'
#             plot_info['ind_0_gme_label']    = 'OMNI Sym-H\n[nT]'
#         elif gme_param == 'omni_bmagavg':
#             df_raw                              = pd.DataFrame(omni_df_raw['bMagAvg'])
#             plot_info['ind_0_symbol']       = 'OMNI |B|'
#             plot_info['ind_0_gme_label']    = 'OMNI |B| [nT]'
#         elif gme_param == 'mstid_score':
#             df_raw                              = mongo_tools.get_mstid_scores(sTime,eTime)
#             plot_info['ind_0_symbol']       = 'MSTID Score'
#             plot_info['ind_0_gme_label']    = 'MSTID Score'
#         elif gme_param == 'mstid_reduced_azm':
#             mstid_list_format   = 'music_guc_{radar}_{sDate}_{eDate}'
#             all_years           = run_helper.create_default_radar_groups_all_years(mstid_format=mstid_list_format)
#             db_name             = 'mstid'
#             tunnel,mongo_port   = mongo_tools.createTunnel()
#             mstid_reduced_azm   = calendar_plot.calculate_reduced_mstid_azm(all_years,
#                     reduction_type='mean',daily_vals=True,
#                     db_name=db_name,mongo_port=mongo_port)
#             df_raw  = pd.DataFrame(mstid_reduced_azm['red_mstid_azm'])

#             plot_info['title']              = 'Continental MSTID Azimuth'
#             plot_info['ind_0_symbol']       = 'Continental\nMSTID Azimuth'
#             plot_info['ind_0_gme_label']    = 'Continental\nMSTID Azimuth'
#         elif gme_param == 'mstid_reduced_azm_dev':
#             mstid_list_format   = 'music_guc_{radar}_{sDate}_{eDate}'
#             all_years           = run_helper.create_default_radar_groups_all_years(mstid_format=mstid_list_format)
#             db_name             = 'mstid'
#             tunnel,mongo_port   = mongo_tools.createTunnel()
#             mstid_reduced_azm   = calendar_plot.calculate_reduced_mstid_index(all_years,
#                     reduction_type='mean',daily_vals=True,val_key='mstid_azm_dev',zscore=False,
#                     db_name=db_name,mongo_port=mongo_port)
#             df_raw  = pd.DataFrame(mstid_reduced_azm['red_mstid_index'])

#             plot_info['title']              = 'Continental MSTID EW Dev.'
#             plot_info['ind_0_symbol']       = 'Continental\nMSTID EW Dev.'
#             plot_info['ind_0_gme_label']    = 'Continental\nMSTID EW Dev.'

#         elif gme_param == 'mstid_reduced_inx':
#             df_raw  = pd.DataFrame(mstid_reduced_inx['red_mstid_index'])
#             plot_info['title']              = 'Continental MSTID Index'
#             plot_info['ind_0_symbol']       = 'Continental\nMSTID Index'
#             plot_info['ind_0_gme_label']    = 'Continental\nMSTID Index'
#         elif gme_param == 'n_good_radars':
#             tmp = mstid_reduced_inx['n_good_df']
#             df_raw  = pd.DataFrame(tmp.values,index=tmp.index)
#             plot_info['title']              = 'Reduced SuperDARN MSTID Index - Number of Good Radars'
#             plot_info['ind_0_symbol']       = 'n Data Points'
#             plot_info['ind_0_gme_label']    = 'n Data Points'
#         elif gme_param == 'mbar_diff':
#             df_raw  = get_mbar_diff(sTime,eTime)
#             plot_info['title']              = u'Integrated ECMWF Geopotential Residual: $\Sigma$(1 mbar - 10 mbar)$^2$'
#             plot_info['ind_0_symbol']       = u'Integrated ECMWF Geopotential Residual: $\Sigma$(1 mbar - 10 mbar)$^2$'
#             plot_info['ind_0_gme_label']    = '1-10 mBar Resid'
#             plot_info['ind_0_gme_label']    = u'$\Sigma Z_r$'
#         elif gme_param == 'neg_mbar_diff':
#             df_raw  = -1 * get_mbar_diff(sTime,eTime,zscore=neg_mbar_diff_zscore)
#             plot_info['title']              = u'ECMWF Geopotential-Derived Polar Vortex Index $\zeta$'
#             plot_info['ind_0_symbol']       = 'Polar Vortex\n'+u'$\zeta$'
#             plot_info['ind_0_gme_label']    = 'Polar Vortex\n'+u'$\zeta$'
#         elif gme_param == 'mbar_corr':
#             df_raw  = get_mbar_corr(sTime,eTime)
#             plot_info['ind_0_symbol']       = 'corr(1 mBar, 10 mBar)'
#             plot_info['ind_0_gme_label']    = 'corr(1 mBar, 10 mBar)'
#         elif gme_param == 'solar_dec':
#             times   = []
#             taus    = []
#             decs    = []
            
#             curr_time   = sTime
#             while curr_time < eTime:
#                 tau, dec = epem(curr_time)

#                 times.append(curr_time)
#                 taus.append(tau)
#                 decs.append(dec)

#                 curr_time += datetime.timedelta(days=1)

#             df_raw = pd.DataFrame(decs,index=times)
#             plot_info['ind_0_symbol']       = 'Solar Dec.'
#             plot_info['ind_0_gme_label']    = 'Solar Dec.'

#         if plot_info.get('title') is None:
#             plot_info['title']  = '{}'.format(gme_param.upper())

#         if _plot_info is not None:
#             plot_info.update(_plot_info)

#         # Enforce sTime, eTime
#         tf              = np.logical_and(df_raw.index >= sTime, df_raw.index < eTime)
#         df_raw          = df_raw[tf].copy()
#         df_raw.columns  = ['ind_0_raw']

#         if parent is None:
#             # This section is for compatibility with code that only uses 
#             # the single level DriverObj.

#             # Resample data.
#             df_rsmp         = df_raw.resample(oversamp_T)
#             df_rsmp         = df_rsmp.interpolate()
#             df_rsmp.columns = ['ind_0_processed']

#             self.ind_df_raw     = df_raw
#             self.sTime          = sTime
#             self.eTime          = eTime
#             self.ind_df         = df_rsmp
#             self.ind_times      = df_rsmp.index.to_pydatetime()
#             self.ind_vals       = df_rsmp['ind_0_processed']
#         else:
#             # This section is for attributes of the new container-style
#             # Driver class.
#             self.data           = df_raw['ind_0_raw']
#             self.data.name      = gme_param

#         self.parent         = parent
#         self.gme_param      = gme_param
#         self.history        = {datetime.datetime.now():comment}
#         self.plot_info      = plot_info
#         self.metadata       = plot_info #Create alias for compatibility with other code.

#     def __sub__(self,other,data_set='subtract',comment=None,**kwargs):
#         """
#         Drop NaNs.
#         """
#         new_data    = self.data - other.data
#         if comment is None:
#             ds_0    = self.plot_info['data_set'][:5]
#             ds_1    = other.plot_info['data_set'][:5]
#             comment = '{} - {}'.format(ds_0,ds_1)
#         new_do      = self.copy(data_set,comment,new_data)
#         return new_do

#     def ds_name(self):
#         return self.plot_info.get('data_set')

#     def resample(self,dt=datetime.timedelta(minutes=1),data_set='resample',comment=None):
#         no_na_data  = self.data.dropna()
#         new_data    = no_na_data.resample(dt).interpolate(method='linear')
        
#         if comment is None: comment = 'dt = {!s}'.format(dt)
#         new_do      = self.copy(data_set,comment,new_data)
#         return new_do

# #    def interpolate(self,data_set='interpolate',comment=None):
# #        new_data    = self.data.interpolate()
# #        
# #        if comment is None: comment = 'Interpolate'
# #        new_do      = self.copy(data_set,comment,new_data)
# #        return new_do

#     def simulate(self,wave_list=None,data_set='simulate',comment=None):
#         if wave_list is None:
#             wd          = {}
#             wd['T']     = datetime.timedelta(days=5.)
#             wd['A']     = 100.
#             wave_list   = [wd]

#         t_0 = self.data.index.min().to_datetime()
#         xx  = (self.data.index - t_0).total_seconds()
#         yy  = self.data.values.copy() * 0.

#         for wd in wave_list:
#             T   = wd.get('T')
#             A   = wd.get('A',1.)
#             C   = wd.get('C',0.)

#             f_c = 1./T.total_seconds()
#             yy += A*np.sin(2*np.pi*f_c*xx) + C

#         if comment is None:
#             comment = 'Simulate'

#         new_do          = self.copy(data_set,comment)
#         new_do.data[:]  = yy

#         return new_do

#     def rolling(self,window,center=True,kind='mean',data_set=None,comment=None):
#         """
#         Apply a rolling mean.

#         window: datetime.timedelta
#         """
#         dt          = self.sample_period()
#         roll_win    = int(window.total_seconds() / dt.total_seconds())

# #        rlng        = getattr(pd,'rolling_{}'.format(kind))
# #        new_data    = rlng(self.data,roll_win,center=center)
#         new_data    = self.data.rolling(roll_win).mean()

# #        if kind == 'mean': 
# #            new_data    = pd.rolling_mean(self.data,roll_win,center=center)
# #        elif kind == 'median':
# #            new_data    = pd.rolling_median(self.data,roll_win,center=center)
# #        elif kind == 'sum':
# #            new_data    = pd.rolling_sum(self.data,roll_win,center=center)

#         if data_set is None:
#             if window < datetime.timedelta(hours=1):
#                 time_str = '{:.0f}_min'.format(window.total_seconds()/60.)
#             elif window < datetime.timedelta(days=1):
#                 time_str = '{:.0f}_hr'.format(window.total_seconds()/3600.)
#             else:
#                 time_str = '{:.0f}_day'.format(window.total_seconds()/86400.)

#             data_set = 'rolling_{}_{}'.format(time_str,kind)
        
#         if comment is None: comment = 'window = {!s}'.format(window)
#         new_do      = self.copy(data_set,comment,new_data)

#         if window < datetime.timedelta(hours=1):
#             time_str = '{:.0f} Min'.format(window.total_seconds()/60.)
#         elif window < datetime.timedelta(days=1):
#             time_str = '{:.0f} Hr'.format(window.total_seconds()/3600.)
#         else:
#             time_str = '{:.0f} Day'.format(window.total_seconds()/86400.)
#         new_do.plot_info['smoothing'] = '{} {} Smoothing'.format(time_str,kind.title())
#         return new_do

#     def dropna(self,data_set='dropna',comment='Remove NaNs',**kwargs):
#         """
#         Drop NaNs.
#         """
#         new_data    = self.data.dropna(**kwargs)
#         new_do      = self.copy(data_set,comment,new_data)
#         return new_do

#     def elbow_trend(self,elbow_date=None,elbow_dates=None,data_set='elbow_trend',comment=None,**kwargs):
#         """
#         Calculate a manual trend for a variable on the model that the season
#         starts with a constant mean and then has a linear slope at some unknown
#         breakpoint.
#         """
        
#         ### Allow user-specified elbow dates.
#         avail_dates = self.data.index.to_pydatetime()
#         sDate       = min(avail_dates)
#         eDate       = max(avail_dates)

#         test_dates  = avail_dates
#         if elbow_dates is not None:
#             new_eds = []
#             for ed_0,ed_1 in elbow_dates:
#                 tf  = np.logical_and(avail_dates >= ed_0, avail_dates < ed_1)
#                 if not np.any(tf): continue
#                 new_eds = avail_dates[tf].tolist()
#             if len(new_eds) != 0:
#                 test_dates = new_eds

#         if elbow_date is not None:
#             if elbow_date in avail_dates:
#                 test_dates  = [elbow_date]
        
#         ### Test for the best one.
#         models      = []
#         metrics     = []
#         for elbow_date in test_dates:
#             model, metric   = elbow(self.data,elbow_date,**kwargs)
#             models.append(model)
#             metrics.append(metric)

#         inx         = np.nanargmin(metrics)
#         new_data    = models[inx]

#         if comment is None:
#             comment = 'Elbow: {:%Y-%m-%d}'.format(test_dates[inx])
#         new_do      = self.copy(data_set,comment,new_data)
#         return new_do

#     def apply_filter(self,data_set='filtered',comment=None,**kwargs):
#         sig_obj = self
#         data    = self.data.copy()

#         filt        = Filter(sig_obj,**kwargs)
#         filt_data   = sp.signal.lfilter(filt.ir,[1.0],data)

#         data[:]     = filt_data

#         shift       = len(filt.ir)/2
#         t_0         = data.index.min()
#         t_1         = data.index[shift]
#         dt_shift    = (t_1-t_0)/2

#         tf          = data.index > t_1
#         data        = data[tf]
#         data.index  = data.index - dt_shift

#         new_data        = data
#         if comment is None:
#             comment = filt.comment
#         new_do          = self.copy(data_set,comment,new_data)
#         new_do.filter   = filt
#         return new_do

#     def copy(self,newsig,comment,new_data=None):
#         """Copy object.  This copies data and metadata, updates the serial number, and logs a comment in the history.  Methods such as plot are kept as a reference.

#         **Args**:
#             * **newsig** (str): Name for the new musicDataObj object.
#             * **comment** (str): Comment describing the new musicDataObj object.
#         **Returns**:
#             * **newsigobj** (:class:`musicDataObj`): Copy of the original musicDataObj with new name and history entry.

#         Written by Nathaniel A. Frissell, Fall 2013
#         """

#         if self.parent is None:
#             print('No parent object; cannot copy.')
#             return

#         all_data_sets   = self.parent.get_all_data_sets()
#         all_serials     = [x.plot_info['serial'] for x in all_data_sets]
#         serial          = max(all_serials) + 1
#         newsig          = '_'.join(['DS%03d' % serial,newsig])

#         setattr(self.parent,newsig,copy.copy(self))
#         newsigobj = getattr(self.parent,newsig)

#         newsigobj.data          = self.data.copy()
#         newsigobj.gme_param     = '{}'.format(self.gme_param)
#         newsigobj.metadata      = self.metadata.copy()
#         newsigobj.plot_info     = newsigobj.metadata
#         newsigobj.history       = self.history.copy()

#         newsigobj.metadata['data_set']  = newsig
#         newsigobj.metadata['serial']    = serial
#         newsigobj.history[datetime.datetime.now()] = '[{}] {}'.format(newsig,comment)

#         if new_data is not None:
#             newsigobj.data = new_data
        
#         newsigobj.set_active()
#         return newsigobj
  
#     def set_active(self):
#         """Sets this signal as the currently active signal.

#         Written by Nathaniel A. Frissell, Fall 2013
#         """
#         self.parent.active = self

#     def set_secondary(self):
#         """
#         Sets this signal as the secondary signal.
#         """
#         self.parent.secondary = self

#     def nyquist_frequency(self,time_vec=None,allow_mode=False):
#         """Calculate the Nyquist frequency of a vt sigStruct signal.

#         **Args**:
#             * [**time_vec**] (list of datetime.datetime): List of datetime.datetime to use instead of self.time.

#         **Returns**:
#             * **nq** (float): Nyquist frequency of the signal in Hz.

#         Written by Nathaniel A. Frissell, Fall 2013
#         """

#         dt  = self.sample_period(time_vec=time_vec,allow_mode=allow_mode)
#         nyq = float(1. / (2*dt.total_seconds()))
#         return nyq

#     def is_evenly_sampled(self,time_vec=None):
#         if time_vec is None:
#             time_vec = self.data.index.to_pydatetime()

#         diffs       = np.diff(time_vec)
#         diffs_unq   = np.unique(diffs)

#         if len(diffs_unq) == 1:
#             return True
#         else:
#             return False

#     def sample_period(self,time_vec=None,allow_mode=False):
#         """Calculate the sample period of a vt sigStruct signal.

#         **Args**:
#             * [**time_vec**] (list of datetime.datetime): List of datetime.datetime to use instead of self.time.

#         **Returns**:
#             * **samplePeriod** (float): samplePeriod: sample period of signal in seconds.

#         Written by Nathaniel A. Frissell, Fall 2013
#         """
        
#         if time_vec is None: time_vec = self.data.index.to_pydatetime()

#         diffs       = np.diff(time_vec)
#         diffs_unq   = np.unique(diffs)
#         self.diffs  = diffs_unq

#         if len(diffs_unq) == 1:
#             samplePeriod = diffs[0].total_seconds()
#         else:
#             diffs_sec   = np.array([x.total_seconds() for x in diffs])
#             maxDt       = np.max(diffs_sec)
#             avg         = np.mean(diffs_sec)
#             mode        = sp.stats.mode(diffs_sec).mode[0]

#             md          = self.metadata
#             warn        = 'WARNING'
#             if md.has_key('title'): warn = ' '.join([warn,'FOR','"'+md['title']+'"'])
#             print(warn + ':')
#             print('   Date time vector is not regularly sampled!')
#             print('   Maximum difference in sampling rates is ' + str(maxDt) + ' sec.')
#             samplePeriod = mode

#             if not allow_mode:
#                 raise()
#             else:
#                 print('   Using mode sampling period of ' + str(mode) + ' sec.')
        
#         smp = datetime.timedelta(seconds=samplePeriod)
#         return smp

#     def set_metadata(self,**metadata):
#         """Adds information to the current musicDataObj's metadata dictionary.
#         Metadata affects various plotting parameters and signal processing routinges.

#         **Args**:
#             * **metadata** (**kwArgs): keywords sent to matplot lib, etc.

#         Written by Nathaniel A. Frissell, Fall 2013
#         """
#         self.metadata = dict(self.metadata.items() + metadata.items())

#     def print_metadata(self):
#         """Nicely print all of the metadata associated with the current musicDataObj object.

#         Written by Nathaniel A. Frissell, Fall 2013
#         """
#         keys = self.metadata.keys()
#         keys.sort()
#         for key in keys:
#             print(key+':',self.metadata[key])

#     def append_history(self,comment):
#         """Add an entry to the processing history dictionary of the current musicDataObj object.

#         **Args**:
#             * **comment** (string): Infomation to add to history dictionary.

#         Written by Nathaniel A. Frissell, Fall 2013
#         """
#         self.history[datetime.datetime.now()] = '['+self.metadata['dataSetName']+'] '+comment

#     def print_history(self):
#         """Nicely print all of the processing history associated with the current musicDataObj object.

#         Written by Nathaniel A. Frissell, Fall 2013
#         """
#         keys = self.history.keys()
#         keys.sort()
#         for key in keys:
#             print(key,self.history[key])

#     def get_short_name(self,sep='-'):
#         """
#         Returns the DS Number and GME Param for filenames, etc.
#         """
#         ds_num  = self.metadata['data_set'][:5]
#         param   = self.metadata.get('processing_code')
#         if param is None:
#             param   = self.metadata['gme_param']

#         sn      = sep.join([ds_num,param])
#         return sn

#     def plot_fft(self,ax=None,label=None,plot_target_nyq=False,
#             xlim=None,xticks=None,xscale='log',phase=False,plot_legend=True,**kwargs):
#         T_max   = kwargs.pop('T_max',datetime.timedelta(days=0.5))
#         f_max   = 1./T_max.total_seconds()

#         data    = self.data.copy()

#         # Handle NaNs just for FFT purposes.
#         if data.hasnans:
#             data    = data.interpolate()
#             data    = data.dropna()

#         data    = data - data.mean()

#         smp     = self.sample_period(data.index.to_pydatetime())
#         nqf     = self.nyquist_frequency(data.index.to_pydatetime())

#         hann    = np.hanning(data.size)
#         data    = data * hann

#         n_fft   = 2**(data.size).bit_length()
#         sf      = sp.fftpack.fft(data,n=n_fft)

#         freq    = sp.fftpack.fftfreq(n_fft,smp.total_seconds())
#         T_s     = 1./freq
#         T_d     = T_s / (24. * 60. * 60.)

# #        if label is not None:
# #            txt     = 'df = {!s} Hz'.format(freq[1])
# #            label   = '\n'.join([label,txt])

#         if ax is None:
#             ax = plt.gca()

#         xx      = freq[1:n_fft/2]
#         yy      = sf[1:n_fft/2]
        
#         if phase is not True:
#             yy      = np.abs(yy)
#             yy      = yy/yy.max()
#             ylabel  = 'FFT |S(f)|'
#         else:
#             yy      = np.angle(yy,deg=True)
#             ylabel  = 'FFT Phase [deg]'

#         ax.plot(xx,yy,marker='.',label=label,**kwargs)

#         #### Plot Nyquist Line for Target Sampling Rate (1 Day)
#         if plot_target_nyq:
#             dt_min  = datetime.timedelta(days=1)
#             nyq_min = 1./(2.*dt_min.total_seconds())

#             label   = '{!s}'.format(2*dt_min)
#             ax.axvline(nyq_min,ls='--',color='g',label=label)

#         #### Define xticks
# #        set_spectrum_xaxis(f_max,ax=ax)
#         if xscale is not None:
#             ax.set_xscale(xscale)
#         if xlim is not None:
#             ax.set_xlim(xlim)
#         if xticks is not None:
#             ax.set_xticks(xticks)

#         xts_hz  = ax.get_xticks()
#         xts_d   = (1./xts_hz) / (24.*60.*60.)

#         xtl     = []
#         for xts_hz,xts_d in zip(xts_hz,xts_d):
#             if np.isinf(xts_d):
#                 xtl.append('{:0.3g}\nInf'.format(xts_hz))
#             else:
#                 xtl.append('{:0.3g}\n{:.1f}'.format(xts_hz,xts_d))
#         ax.set_xticklabels(xtl)
#         ax.set_xlabel('Frequency [Hz]\nPeriod [days]')

#         #### A little more plot cleanup
#         ax.set_ylabel(ylabel)
#         if plot_legend:
#             ax.legend(loc='upper right',fontsize='small')
#         ax.grid(True)

#     def plot_lsp(self,ax=None,n_freq=2**10,label=None,**kwargs):
#         T_max   = kwargs.pop('T_max',datetime.timedelta(days=0.5))
#         f_max   = 1./T_max.total_seconds()

#         data    = self.data.copy()
#         data    = data.dropna()

#         data    = data - data.mean()

#         smp     = self.sample_period(data.index.to_pydatetime(),allow_mode=True)
#         nqf     = self.nyquist_frequency(data.index.to_pydatetime(),allow_mode=True)

#         hann    = np.hanning(data.size)
#         data    = data * hann

#         t_0     = data.index.min().to_pydatetime()
#         smp_vec = (data.index - t_0).total_seconds()
        
#         freq    = np.arange(n_freq)/(n_freq-1.) * f_max
#         freq    = freq[1:]
#         omega   = 2.*np.pi*freq

#         T_s     = 2.*np.pi/freq
#         T_d     = T_s / (24. * 60. * 60.)

#         lsp     = sp.signal.lombscargle(smp_vec,data.values,omega)

#         if ax is None:
#             ax = plt.gca()

#         xx      = freq
#         yy      = lsp
#         yy      = 2.*np.sqrt(4.*(lsp/n_freq))
#         ax.plot(xx,yy,marker='.',label=label,**kwargs)

#         #### Define xticks
#         set_spectrum_xaxis(f_max,ax=ax)


#         ax.set_ylabel('LSP S(f)')
#         ax.grid(True)

# def elbow(dta,elbow_date,model_0='mean'):
#     """
#     Calculate the mean of the beginning of a season and a linear trend
#     in the second part of the season.
#     """
#     new_data    = dta.copy()

#     # Define a total_seconds index.
#     td_arr      = dta.index.to_pydatetime() - dta.index.min()
#     sec_arr     = np.array([x.total_seconds() for x in td_arr])

#     #Calculate regression for second part of season.
#     tf              = dta.index >= elbow_date
#     xx              = sec_arr[tf]
#     yy_0            = dta.values[tf]
#     m,b,r,p,ste     = sp.stats.linregress(xx,yy_0)
#     yy_1            = m*xx + b
#     new_data[tf]    = yy_1
#     bound_r1_0      = yy_1[0]

#     #Calculate mean for first part of season.
#     tf          = dta.index < elbow_date
#     if model_0 == 'mean':
#         mu              = dta[tf].mean()
#         new_data[tf]    = mu
#         bound_r0_1      = mu
#     elif model_0 == 'linear':
#         xx              = sec_arr[tf]
#         yy_0            = dta.values[tf]
#         m,b,r,p,ste     = sp.stats.linregress(xx,yy_0)
#         yy_1            = m*xx + b
#         new_data[tf]    = yy_1
#         if yy_1.size == 0:
#             bound_r0_1  = np.nan
#         else:
#             bound_r0_1  = yy_1[-1]
#     elif model_0 == 'bound_r1_0':
#         new_data[tf]    = bound_r1_0
#         bound_r0_1      = bound_r1_0

#     # Calculate a metric to help find the smallest gap between
#     # the mu and the yy_1[0]. In other words, make the transition
#     # as smooth as possible.
#     metric      = np.abs(bound_r1_0 - bound_r0_1)

#     return new_data,metric

# def set_spectrum_xaxis(f_max,ax=None):
#     if ax is None:
#         ax = plt.gca()
#     ax.set_xlim(xmax=f_max)
#     xt_d    = np.arange(2.5,0,-0.5)
#     xt_d    = np.array([50,5,2,1.5,1,0.7,0.5])
#     xticks  = 1./(xt_d*24.*60.*60.)
#     ax.set_xticks(xticks)

#     #### Plot Nyquist Line for Target Sampling Rate (1 Day)
#     dt_min  = datetime.timedelta(days=1)
#     nyq_min = 1./(2.*dt_min.total_seconds())

#     label   = 'Target Nyquist'
#     label   = None
#     ax.axvline(nyq_min,ls='--',color='r',label=label)

#     xtl     = []
#     for xt_0,xt_1 in zip(xticks,xt_d):
#         if np.isinf(xt_1):
#             xtl.append('Inf')
#         else:
#             xtl.append('{:.1f}'.format(xt_1))
#     ax.set_xticklabels(xtl)
#     ax.set_xlabel('Period [days]')

# def get_mbar_diff(sTime,eTime,mbars=[1,10],lat_min=65,zscore=True,cache=True):
#     season  = '{}_{}'.format(sTime.strftime('%Y'),eTime.strftime('%Y'))

#     base_path   = os.path.join('mstid_data','rda_111.2')
#     input_path  = os.path.join(base_path,'cache')
#     output_path = os.path.join(input_path,'mbar_diff')
    
#     cache_fname = '{}_{!s}_{!s}_mbar_diff.h5'.format(season,mbars[0],mbars[1])
#     cache_fpath = os.path.join(output_path,cache_fname)
    
#     if (not cache) or (not os.path.exists(cache_fpath)):
#         date_list_out   = []
#         data_list_out   = []

#         files_in    = []
#         data_in     = []
#         for mbar in mbars:
#             fname   = '{}_delta_{!s}mbar.h5'.format(season,mbar)
#             fpath   = os.path.join(input_path,fname)

#             data = loadMusicArrayFromHDF5(fpath)

#             files_in.append(fname)
#             data_in.append(data)

#         data_0 = data_in[0]['raw']
#         data_1 = data_in[1]['raw']

#         for dct_0, dct_1 in zip(data_0,data_1):
#             dt_0    = dct_0['dt']
#             dt_1    = dct_1['dt']

#             if dt_0 != dt_1:
#                 print('Bad!  dt_0 != dt_1!!!')
#                 import ipdb; ipdb.set_trace()
#                 continue

#             print('mBar Residual Calculation: {!s}'.format(dt_0))
#             lats,lons = dct_0['latlon']

#             lat_tf  = lats >= lat_min
#             vals_0  = dct_0['values'][lat_tf]
#             vals_1  = dct_1['values'][lat_tf]
#             lat_scale   = np.sin( np.deg2rad(90.-np.abs(lats)) )[lat_tf]

# #            lat_min, lat_max    = ( 60., 85.)
# #            lon_min, lon_max    = (210.,315.)
# #            lats_tf = np.logical_and(lats >= lat_min, lats < lat_max)
# #            lons_tf = np.logical_and(lons >= lon_min, lons < lon_max)
# #            vals_tf = np.logical_and(lats_tf,lons_tf)
# #            vals_0  = dct_0['values'][vals_tf]
# #            vals_1  = dct_1['values'][vals_tf]
# #            lat_scale   = 1.
# ##            lat_scale   = np.sin( np.deg2rad(90.-np.abs(lats)) )[vals_tf]
            
#             resid       = np.sum( lat_scale * (vals_1 - vals_0)**2 )
#             date_list_out.append(dt_0)
#             data_list_out.append(resid)

#         df = pd.DataFrame(data_list_out,index=date_list_out)

#         if cache:
#             if not os.path.exists(output_path): 
#                 os.makedirs(output_path)

#             saveMusicArrayToHDF5(df, cache_fpath)
#     else:
#         df = loadMusicArrayFromHDF5(cache_fpath)

#     # Write out some statistics about the original data.
#     txt = []
#     txt.append( ('sTime', '{}'.format(sTime.strftime('%Y-%m-%d')), 12) )
#     txt.append( ('eTime', '{}'.format(eTime.strftime('%Y-%m-%d')), 12) )
#     txt.append( ('mean',  '{:>.4g}'.format(float(df.mean())),       12) )
#     txt.append( ('std',   '{:>.4g}'.format(float(df.std())),        12) )

# #    fname = 'mbar_diff_rpt.txt'
# #    if not os.path.exists(fname):
# #        with open(fname,'w') as fl:
# #            fl.write('This file was generated by: {}\n'.format(curr_file))
# #            for key, val, width in txt:
# #                fl.write('{0:{width}}'.format(key,width=width))
# #            fl.write('\n')
# #
# #    with open(fname,'a') as fl:
# #        for key, val, width in txt:
# #            fl.write('{0:{width}}'.format(val,width=width))
# #        fl.write('\n')

#     # Feature Scaling / Standardization
#     if zscore:
#         df = (df - df.mean())/df.std()
    
#     return df

# def get_mbar_corr(sTime,eTime,mbars=[1,10],lat_min=50,cache=True):
#     season  = '{}_{}'.format(sTime.strftime('%Y'),eTime.strftime('%Y'))

#     base_path   = os.path.join('mstid_data','rda_111.2')
#     input_path  = os.path.join(base_path,'cache')
#     output_path = os.path.join(input_path,'mbar_corr')
    
#     cache_fname = '{}_{!s}_{!s}_mbar_diff.h5'.format(season,mbars[0],mbars[1])
#     cache_fpath = os.path.join(output_path,cache_fname)

#     if (not cache) or (not os.path.exists(cache_fpath)):
#         date_list_out   = []
#         data_list_out   = []

#         files_in    = []
#         data_in     = []
#         for mbar in mbars:
#             fname   = '{}_delta_{!s}mbar.h5'.format(season,mbar)
#             fpath   = os.path.join(input_path,fname)

#             data = loadMusicArrayFromHDF5(fpath)

#             files_in.append(fname)
#             data_in.append(data)

#         data_0 = data_in[0]['raw']
#         data_1 = data_in[1]['raw']

#         for dct_0, dct_1 in zip(data_0,data_1):
#             dt_0    = dct_0['dt']
#             dt_1    = dct_1['dt']

#             if dt_0 != dt_1:
#                 print('Bad!  dt_0 != dt_1!!!')
#                 # import ipdb; ipdb.set_trace()
#                 continue

#             print('mBar Residual Calculation: {!s}'.format(dt_0))
#             lats,lons = dct_0['latlon']

#             lat_tf  = lats >= lat_min

#             vals_0  = dct_0['values'][lat_tf]
#             vals_1  = dct_1['values'][lat_tf]

#             corr, p_val= sp.stats.pearsonr(np.array(vals_0),np.array(vals_1))

#             date_list_out.append(dt_0)
#             data_list_out.append(corr)

#         df = pd.DataFrame(data_list_out,index=date_list_out)

#         if cache:
#             if not os.path.exists(output_path): 
#                 os.makedirs(output_path)

#             saveMusicArrayToHDF5(df, cache_fpath)
#     else:
#         df = loadMusicArrayFromHDF5(cache_fpath)

#     return df

# class Filter(object):
#     def __init__(self, sig_obj, numtaps=100, cutoff_low=None, cutoff_high=None, width=None, 
#           window='blackman', pass_zero=True, scale=True,newSigName='filtered'):
#         """Filter a VT sig/sigStruct object and define a FIR filter object.
#         If only cutoff_low is defined, this is a high pass filter.
#         If only cutoff_high is defined, this is a low pass filter.
#         If both cutoff_low and cutoff_high is defined, this is a band pass filter.

#         Uses scipy.signal.firwin()
#         High pass and band pass filters inspired by Matti Pastell's page:
#           http://mpastell.com/2010/01/18/fir-with-scipy/

#         Metadata keys:
#           'filter_cutoff_low' --> cutoff_low
#           'filter_cutoff_high' --> cutoff_high
#           'filter_cutoff_numtaps' --> cutoff_numtaps

#         numtaps : int
#           Length of the filter (number of coefficients, i.e. the filter
#           order + 1).  `numtaps` must be even if a passband includes the
#           Nyquist frequency.

#         cutoff_low: float or 1D array_like
#             High pass cutoff frequency of filter (expressed in the same units as `nyq`)
#             OR an array of cutoff frequencies (that is, band edges). In the
#             latter case, the frequencies in `cutoff` should be positive and
#             monotonically increasing between 0 and `nyq`.  The values 0 and
#             `nyq` must not be included in `cutoff`.

#         cutoff_high: float or 1D array_like
#             Like cutoff_low, but this is the low pass cutoff frequency of the filter.

#         width : float or None
#             If `width` is not None, then assume it is the approximate width
#             of the transition region (expressed in the same units as `nyq`)
#             for use in Kaiser FIR filter design.  In this case, the `window`
#             argument is ignored.

#         window : string or tuple of string and parameter values
#             Desired window to use. See `scipy.signal.get_window` for a list
#             of windows and required parameters.

#         pass_zero : bool
#             If True, the gain at the frequency 0 (i.e. the "DC gain") is 1.
#             Otherwise the DC gain is 0.

#         scale : bool
#             Set to True to scale the coefficients so that the frequency
#             response is exactly unity at a certain frequency.
#             That frequency is either:
#                       0 (DC) if the first passband starts at 0 (i.e. pass_zero is True);
#                       `nyq` (the Nyquist rate) if the first passband ends at
#                           `nyq` (i.e the filter is a single band highpass filter);
#                       center of first passband otherwise.

#         nyq : float
#             Nyquist frequency.  Each frequency in `cutoff` must be between 0
#             and `nyq`.

#         :returns: filter object
#         """
        
#         nyq     = sig_obj.nyquist_frequency()

#         if   cutoff_high != None:    #Low pass
#             lp = sp.signal.firwin(numtaps=numtaps, cutoff=cutoff_high, width=width,
#                     window=window, pass_zero=pass_zero, scale=scale, nyq=nyq)
#             d = lp

#         if   cutoff_low != None:    #High pass
#             hp = -sp.signal.firwin(numtaps=numtaps, cutoff=cutoff_low, width=width,
#                     window=window, pass_zero=pass_zero, scale=scale, nyq=nyq)
#             hp[numtaps/2] = hp[numtaps/2] + 1
#             d = hp

#         if cutoff_high != None and cutoff_low != None:
#             d = -(lp+hp)
#             d[numtaps/2] = d[numtaps/2] + 1

#         if cutoff_high == None and cutoff_low == None:
#             print("WARNING!! You must define cutoff frequencies!")
#             return
        
#         self.comment    = ' '.join(['Filter:',window+',','Nyquist:',str(nyq),'Hz,','Cuttoff:','['+str(cutoff_low)+', '+str(cutoff_high)+']','Hz'])
#         self.nyq        = nyq
#         self.ir         = d

# #        self.filter(sigObj,newSigName=newSigName)

#     def __str__(self):
#         return self.comment

#     def plot_tf_magnitude(self,f_n=2048,scale=1.,
#             label='Filter TF',style_dict={},ax=None):

#         if ax is None:
#             ax = plt.gca()

#         f_min,f_max = ax.get_xlim()
#         nyq     = self.nyq

#         f_vec   = np.linspace(f_min,f_max,f_n,endpoint=True)

#         worN    = (f_vec/nyq) * np.pi
#         omega,h = sp.signal.freqz(self.ir,1,worN=worN)
        
#         xx      = nyq*(omega/np.pi)
#         yy      = np.abs(h)*scale

#         def_style   = {'ls':'--','lw':1,'color':'m'}
#         def_style.update(style_dict)
#         ax.plot(xx,yy,label=label,**def_style)

#     #Plot step and impulse response
#     def plotImpulseResponse(self,xmin=None,xmax=None,ymin_imp=None,ymax_imp=None,ymin_step=None,ymax_step=None):
#         """Plot the frequency and phase response of the filter object.

#         :param xmin: Minimum value for x-axis.
#         :param xmax: Maximum value for x-axis.
#         :param ymin_imp: Minimum value for y-axis for the impulse response plot.
#         :param ymax_imp: Maximum value for y-axis for the impulse response plot.
#         :param ymin_step: Minimum value for y-axis for the step response plot.
#         :param ymax_step: Maximum value for y-axis for the step response plot.
#         """

#         l = len(self.ir)
#         impulse = np.repeat(0.,l); impulse[0] =1.
#         x = np.arange(0,l)
#         response = sp.signal.lfilter(self.ir,1,impulse)
#         mp.subplot(211)
#         mp.stem(x, response)
#         mp.ylabel('Amplitude')
#         mp.xlabel(r'n (samples)')
#         mp.title(r'Impulse response')
#         mp.subplot(212)

#         step = np.cumsum(response)
#         mp.stem(x, step)
#         mp.ylabel('Amplitude')
#         mp.xlabel(r'n (samples)')
#         mp.title(r'Step response')
#         mp.subplots_adjust(hspace=0.5)
#         mp.show()

#     def filter(self,vtsig,newSigName='filtered'):
#         """Apply the filter to a vtsig object.

#         :param vtsig: vtsig object
#         :param xmax: Maximum value for x-axis.
#         :param ymin_imp: Minimum value for y-axis for the impulse response plot.
#         :param ymax_imp: Maximum value for y-axis for the impulse response plot.
#         :param ymin_step: Minimum value for y-axis for the step response plot.
#         :param ymax_step: Maximum value for y-axis for the step response plot.
#         """
        
#         sigobj = prepForProc(vtsig)
#         vtsig  = sigobj.parent

#         #Apply filter
#         filt_data = sp.signal.lfilter(self.ir,[1.0],sigobj.data)

#         #Filter causes a delay in the signal and also doesn't get the tail end of the signal...  Shift signal around, provide info about where the signal is valid.
#         shift = np.int32(-np.floor(len(self.ir)/2.))

#         start_line = np.zeros(len(filt_data))
#         start_line[0] = 1

#         filt_data  = np.roll(filt_data,shift)
#         start_line = np.roll(start_line,shift)
        
#         tinx0 = abs(shift)
#         tinx1 = np.where(start_line == 1)[0][0]

#         val_tm0 = sigobj.dtv[tinx0]
#         val_tm1 = sigobj.dtv[tinx1]

#         #Create new signal object.
#         newsigobj = sigobj.copy(newSigName,self.comment)
#         #Put in the filtered data.
#         newsigobj.data = copy.copy(filt_data)
#         newsigobj.dtv = copy.copy(sigobj.dtv)

#         #Clear out ymin and ymax from metadata; make sure meta data block exists.
#         #If not, create it.
#         if hasattr(newsigobj,'metadata'):
#             delMeta = ['ymin','ymax']
#             for key in delMeta:
#                 if newsigobj.metadata.has_key(key):
#                     del newsigobj.metadata[key]
#         else:
#             newsigobj.metadata = {}

#         newsigobj.updateValidTimes([val_tm0,val_tm1])

#         key = 'title'
#         if newsigobj.metadata.has_key(key):
#             newsigobj.metadata[key] = ' '.join(['Filtered',newsigobj.metadata[key]])
#         elif vtsig.metadata.has_key(key):
#             newsigobj.metadata[key] = ' '.join(['Filtered',vtsig.metadata[key]])
#         else:
#             newsigobj.metadata[key] = 'Filtered'

#         setattr(vtsig,'active',newsigobj)

# ################################################################################
# ## The following functions are wrappers for getting the data objects to easily
# ## allow for different processing schemes. They were developed while revising
# ## the cross correlation procedures, but now need to be made accessible to 
# ## the other processing routines in this project.
# ## NAF - 28 Jan 2016
# def neg_ae(driver_obj,data_set='active'):
#     ds_0        = driver_obj.get_data_set(data_set)
#     new_data    = -1*ds_0.data
#     ds_1        = ds_0.copy('neg_ae','Negative AE',new_data)

#     pli                     = ds_1.plot_info
#     pli['ind_0_gme_label']  = '-AE Index\n[nT]'
#     pli['legend_label']     = '-AE Index [nT]'
#     pli['ind_0_symbol']     = 'Negative Auroral Electrojet (AE) Index'
#     pli['title']            = 'Negative Auroral Electrojet (AE) Index'
#     return ds_1

# def get_mstid_data(this_driver,sDate,eDate,smooth_win,smooth_kind,
#         zscore=True,db_name='mstid'):
# #    tunnel,mongo_port   = mongo_tools.createTunnel()

#     # For MSTID amplitude plotting.
#     all_years           = run_helper.create_default_radar_groups_all_years()
#     mstid_reduced_inx   = calendar_plot.calculate_reduced_mstid_index(all_years,
#                 reduction_type='mean',daily_vals=True,zscore=zscore,
#                 db_name=db_name)

#     driver_obj          = Driver(sDate,eDate,this_driver,mstid_reduced_inx=mstid_reduced_inx)
#     pli                 = driver_obj.active.plot_info
#     pli['legend_label'] = pli['ind_0_gme_label']
#     driver_obj.active.dropna()

#     smp = driver_obj.active.sample_period(allow_mode=True)
#     driver_obj.active.resample(smp)
# #    driver_obj.active.interpolate()

#     ds  = driver_obj.active.rolling(smooth_win,kind=smooth_kind)
#     smoothed_name   = ds.plot_info['data_set']

#     # Choose which data sets to plot.
#     tmp = []
#     tmp.append({'data_set_0':smoothed_name,'pli_0':{'legend_label':'Smoothed'},
#         'data_set_1':'DS000_raw','pli_1':{'legend_label':'Raw'},'sd_1':{'color':'k'}})
#     driver_obj.cal_plot_ts  = tmp

#     tmp = []
#     tmp.append({'data_set_1':'DS000_raw', 'sd_1':{'ms':10,'ls':'-'},
#         'data_set_0':'DS003_interpolate','sd_0':{'ms':5.5,'ls':'-'}})

#     tmp.append({'data_set_1':'DS003_interpolate', 'sd_1':{'ms':7.5,'ls':'-'},
#         'data_set_0':smoothed_name,'sd_0':{'ms':7.5,'ls':'-'}})

#     driver_obj.ts_spect_plots = tmp
#     return driver_obj

# def get_mstid_data_elbow_detrend(this_driver,sDate,eDate,smooth_win,smooth_kind,
#         elbow_dict={},zscore=True,
#         db_name             = 'mstid'):
#     tunnel,mongo_port   = mongo_tools.createTunnel()

#     # For MSTID amplitude plotting.
#     all_years           = run_helper.create_default_radar_groups_all_years()
#     mstid_reduced_inx   = calendar_plot.calculate_reduced_mstid_index(all_years,
#             reduction_type='mean',daily_vals=True,zscore=zscore,
#             db_name=db_name,mongo_port=mongo_port)

#     driver_obj          = Driver(sDate,eDate,this_driver,mstid_reduced_inx=mstid_reduced_inx)
#     pli                 = driver_obj.active.plot_info
#     pli['legend_label'] = pli['ind_0_gme_label']

#     no_na   = driver_obj.active.dropna()

#     smp     = driver_obj.active.sample_period(allow_mode=True)
#     resamp  = driver_obj.active.resample(smp)
#     interp  = driver_obj.active.interpolate()

#     ebt     = driver_obj.active.elbow_trend(**elbow_dict)
#     sub     = interp - ebt

#     smooth  = driver_obj.active.rolling(smooth_win,kind=smooth_kind)

#     # Choose which data sets to plot.
#     tmp = []
#     tmp.append({'data_set_0':smooth.ds_name(),'pli_0':{'legend_label':'Smoothed'},
#         'data_set_1':'DS000_raw','pli_1':{'legend_label':'Raw'},'sd_1':{'color':'k'}})
#     driver_obj.cal_plot_ts  = tmp

#     tmp = []
#     tmp.append({'data_set_1':'DS000_raw',      'sd_1':{'ms':10,'ls':'-'},
#                 'data_set_0':interp.ds_name(), 'sd_0':{'ms':5.5,'ls':'-'}})

#     tmp.append({'data_set_1':interp.ds_name(), 'sd_1':{'ms':7.5,'ls':'-'},
#                 'data_set_0':ebt.ds_name(),    'sd_0':{'ms':5.5,'ls':'-'}})

#     tmp.append({'data_set_1':sub.ds_name(),    'sd_1':{'ms':7.5,'ls':'-'},
#                 'data_set_0':smooth.ds_name(), 'sd_0':{'ms':7.5,'ls':'-'}})

#     driver_obj.ts_spect_plots = tmp
#     return driver_obj

# def get_driver_obj(var_code,sDate,eDate,
#         smooth_win          = datetime.timedelta(days=4),
#         smooth_kind         = 'mean',
#         resample_time       = datetime.timedelta(days=1),
#         plot_info           = None,
#         elbow_dict          = {},
#         mstid_zscore        = True,
#         polar_vortex_zscore = True,
#         output_dir          = 'output'):

#     test_code   = var_code
#     if re.match('[0-9]{3}_',var_code):
#         test_code = var_code[4:]

#     if 'mstid_inx' == test_code or 'mstid_reduced_inx' == test_code:
#         driver_obj                  = get_mstid_data('mstid_reduced_inx'  ,sDate,eDate,smooth_win,smooth_kind,zscore=mstid_zscore)
#         driver_obj.n_good_radars    = get_mstid_data('n_good_radars'      ,sDate,eDate,smooth_win,smooth_kind)
#     elif 'mstid_inx_elbow_detrend' == test_code:
#         driver_obj                  = get_mstid_data_elbow_detrend('mstid_reduced_inx'  ,sDate,eDate,smooth_win,smooth_kind,elbow_dict=elbow_dict,zscore=mstid_zscore)
#         driver_obj.n_good_radars    = get_mstid_data('n_good_radars'      ,sDate,eDate,smooth_win,smooth_kind)
#     elif 'n_good_radars' == test_code:
#         driver_obj = get_mstid_data('n_good_radars',sDate,eDate,smooth_win,smooth_kind)
#     elif 'mstid_reduced_azm' == test_code:
#         driver_obj          = Driver(sDate,eDate,'mstid_reduced_azm')
#         pli                 = driver_obj.active.plot_info
#         pli['legend_label'] = pli['ind_0_gme_label']
#         driver_obj.active.dropna()

#         smp = driver_obj.active.sample_period(allow_mode=True)
#         driver_obj.active.resample(smp)
#         driver_obj.active.interpolate()

#         ds  = driver_obj.active.rolling(smooth_win,kind=smooth_kind)
#         smoothed_name   = ds.plot_info['data_set']

#         # Choose which data sets to plot.
#         tmp = []
#         tmp.append({'data_set_0':smoothed_name,'pli_0':{'legend_label':'Smoothed'},
#             'data_set_1':'DS000_raw','pli_1':{'legend_label':'Raw'},'sd_1':{'color':'k'}})
#         driver_obj.cal_plot_ts  = tmp

#         tmp = []
#         tmp.append({'data_set_1':'DS000_raw', 'sd_1':{'ms':10,'ls':'-'},
#             'data_set_0':'DS003_interpolate','sd_0':{'ms':5.5,'ls':'-'}})

#         tmp.append({'data_set_1':'DS003_interpolate', 'sd_1':{'ms':7.5,'ls':'-'},
#             'data_set_0':smoothed_name,'sd_0':{'ms':7.5,'ls':'-'}})

#         driver_obj.ts_spect_plots = tmp
#     elif 'mstid_reduced_azm_dev' == test_code:
#         driver_obj          = Driver(sDate,eDate,'mstid_reduced_azm_dev')
#         pli                 = driver_obj.active.plot_info
#         pli['legend_label'] = pli['ind_0_gme_label']
#         driver_obj.active.dropna()

#         smp = driver_obj.active.sample_period(allow_mode=True)
#         driver_obj.active.resample(smp)
#         driver_obj.active.interpolate()

#         ds  = driver_obj.active.rolling(smooth_win,kind=smooth_kind)
#         smoothed_name   = ds.plot_info['data_set']

#         # Choose which data sets to plot.
#         tmp = []
#         tmp.append({'data_set_0':smoothed_name,'pli_0':{'legend_label':'Smoothed'},
#             'data_set_1':'DS000_raw','pli_1':{'legend_label':'Raw'},'sd_1':{'color':'k'}})
#         driver_obj.cal_plot_ts  = tmp

#         tmp = []
#         tmp.append({'data_set_1':'DS000_raw', 'sd_1':{'ms':10,'ls':'-'},
#             'data_set_0':'DS003_interpolate','sd_0':{'ms':5.5,'ls':'-'}})

#         tmp.append({'data_set_1':'DS003_interpolate', 'sd_1':{'ms':7.5,'ls':'-'},
#             'data_set_0':smoothed_name,'sd_0':{'ms':7.5,'ls':'-'}})

#         driver_obj.ts_spect_plots = tmp
#     elif 'ae_proc_0' == test_code:
#         driver_obj  = Driver(sDate,eDate,'ae')

#         driver_obj.active.dropna()
#         neg_ae(driver_obj)
#         pli                     = driver_obj.active.plot_info
#         pli['legend_label']     = '-AE Index [nT] (Proc 0)'
#         driver_obj.active.set_secondary()

#         ds = driver_obj.active.rolling(smooth_win,kind=smooth_kind)
#         smoothed_name   = ds.plot_info['data_set']
#         driver_obj.active.resample(resample_time)
#         driver_obj.active.interpolate()

#         tmp = []
#         tmp.append({'data_set_0':smoothed_name,'pli_0':{'legend_label':'Smoothed'},
#                     'data_set_1':'DS002_neg_ae','pli_1':{'legend_label':'Raw'}})
#         driver_obj.cal_plot_ts  = tmp

#         tmp = []
#         tmp.append({'data_set_1':'DS002_neg_ae',
#                     'data_set_0':smoothed_name})

#         tmp.append({'data_set_1':smoothed_name, 
#                     'data_set_0':'DS005_interpolate','sd_0':{'ms':7.5}})
#         driver_obj.ts_spect_plots = tmp

#     elif 'ae_proc_1' == test_code:
#         driver_obj  = Driver(sDate,eDate,'ae')

#         driver_obj.active.dropna()
#         neg_ae(driver_obj)
#         pli                     = driver_obj.active.plot_info
#         pli['legend_label']     = '-AE Index [nT] (Proc 1)'
#         driver_obj.active.set_secondary()

# #        cutoff_low  = 3.e-7
#         cutoff_low  = None
#         cutoff_high = 1./datetime.timedelta(days=2).total_seconds()
#         driver_obj.active.apply_filter(numtaps=50001,
#                 cutoff_low=cutoff_low,cutoff_high=cutoff_high)

#         driver_obj.active.resample(resample_time)
#         driver_obj.active.interpolate()

#         tmp = []
#         tmp.append({'data_set_1':'DS002_neg_ae',
#                     'data_set_0':'DS003_filtered'})

#         tmp.append({'data_set_1':'DS003_filtered', 
#                     'data_set_0':'DS005_interpolate','sd_0':{'ms':7.5}})

#         tmp.append({'data_set_1':'DS002_neg_ae', 
#                     'data_set_0':'DS005_interpolate','sd_0':{'ms':7.5}})
#         driver_obj.ts_spect_plots = tmp
#     elif 'ae_proc_2' == test_code:
#         # This one is like ae_proc_0, but we are getting rid of the negative.
#         # This might be the one that goes in the paper.
#         driver_obj  = Driver(sDate,eDate,'ae')

#         driver_obj.active.dropna()
#         pli                     = driver_obj.active.plot_info
#         pli['legend_label']     = 'AE [nT]'
#         driver_obj.active.set_secondary()

#         ds = driver_obj.active.rolling(smooth_win,kind=smooth_kind)
#         smoothed_name   = ds.plot_info['data_set']
#         driver_obj.active.resample(resample_time)
#         driver_obj.active.interpolate()

#         tmp = []
#         tmp.append({'data_set_0':smoothed_name,'pli_0':{'legend_label':'Smoothed'},
#                     'data_set_1':'DS000_raw','pli_1':{'legend_label':'Raw'}})
#         driver_obj.cal_plot_ts  = tmp

#         tmp = []
#         tmp.append({'data_set_1':'DS000_raw',
#                     'data_set_0':smoothed_name})

#         tmp.append({'data_set_1':smoothed_name, 
#                     'data_set_0':'DS005_interpolate','sd_0':{'ms':7.5}})
#         driver_obj.ts_spect_plots = tmp
#     elif 'omni_symh' == test_code:
#         driver_obj          = Driver(sDate,eDate,'omni_symh')
#         pli                 = driver_obj.active.plot_info
#         pli['legend_label'] = 'OMNI Sym-H [nT]'
#         driver_obj.active.interpolate()

#         ds  = driver_obj.active.rolling(smooth_win,kind=smooth_kind)
#         smooth_name = ds.plot_info['data_set']
#         driver_obj.active.resample(resample_time)
#         driver_obj.active.interpolate()

#         tmp = []
#         tmp.append({'data_set_0':smooth_name,'pli_0':{'legend_label':'Smoothed'},
#                     'data_set_1':'DS000_raw','pli_1':{'legend_label':'Raw'}})
#         driver_obj.cal_plot_ts  = tmp

#         # Choose which data sets to plot.
#         tmp = []
# #        tmp.append({'data_set_1':'DS000_raw',          'sd_1':{'ms':10, 'ls':'-'},
# #                    'data_set_0':'DS001_interpolate',  'sd_0':{'ms':5.5,'ls':'-'}})

#         tmp.append({'data_set_1':'DS000_raw',          
#                     'data_set_0':'DS001_interpolate'})

#         tmp.append({'data_set_1':'DS000_raw',          'sd_1':{'ms':10, 'ls':'-'},
#                     'data_set_0':smooth_name, 'sd_0':{'ms':5.5,'ls':'-'}})

#         tmp.append({'data_set_1':smooth_name, 'sd_1':{'ms':7.5,'ls':'-'},
#                     'data_set_0':'DS004_interpolate',     'sd_0':{'ms':7.5,'ls':'-'}})
#         driver_obj.ts_spect_plots = tmp

#     elif 'polar_vortex' == test_code:
#         driver_obj          = Driver(sDate,eDate,'neg_mbar_diff',neg_mbar_diff_zscore=polar_vortex_zscore)
#         pli                 = driver_obj.active.plot_info
#         pli['legend_label'] = u'Polar Vortex $\zeta$'

#         ds  = driver_obj.active.rolling(smooth_win,kind=smooth_kind)
#         smooth_name = ds.plot_info['data_set']
#         driver_obj.active.resample(resample_time)
#         driver_obj.active.interpolate()

#         # Choose which data sets to plot.
#         tmp = []
#         tmp.append({'data_set_0':smooth_name,'pli_0':{'legend_label':'Smoothed'},
#             'data_set_1':'DS000_raw','pli_1':{'legend_label':'Raw'},'sd_1':{'color':'k'}})
#         driver_obj.cal_plot_ts  = tmp

#         tmp = []
#         tmp.append({'data_set_1':'DS000_raw',          'sd_1':{'ms':10, 'ls':'-'},
#                     'data_set_0':smooth_name, 'sd_0':{'ms':5.5,'ls':'-'}})

#         tmp.append({'data_set_1':smooth_name, 'sd_1':{'ms':7.5,'ls':'-'},
#                     'data_set_0':'DS002_interpolate',  'sd_0':{'ms':7.5,'ls':'-'}})
#         driver_obj.ts_spect_plots = tmp

#     elif 'polar_vortex_elbow_detrend' == test_code:
#         driver_obj          = Driver(sDate,eDate,'neg_mbar_diff',neg_mbar_diff_zscore=polar_vortex_zscore)
#         pli                 = driver_obj.active.plot_info
#         pli['legend_label'] = u'Polar Vortex $\zeta$'

#         raw     = driver_obj.active
#         ebt     = driver_obj.active.elbow_trend(**elbow_dict)
#         sub     = raw - ebt

#         smooth  = driver_obj.active.rolling(smooth_win,kind=smooth_kind)
#         resamp  = driver_obj.active.resample(resample_time)
#         interp  = driver_obj.active.interpolate()

#         # Choose which data sets to plot.
#         tmp = []
#         tmp.append({'data_set_0':smooth.ds_name(),'pli_0':{'legend_label':'Smoothed'},
#                     'data_set_1':'DS000_raw','pli_1':{'legend_label':'Raw'},'sd_1':{'color':'k'}})
#         driver_obj.cal_plot_ts  = tmp

#         tmp = []
#         tmp.append({'data_set_1':raw.ds_name(),    'sd_1':{'ms':10, 'ls':'-'},
#                     'data_set_0':ebt.ds_name(),    'sd_0':{'ms':5.5,'ls':'-'}})

#         tmp.append({'data_set_1':sub.ds_name(),    'sd_1':{'ms':7.5,'ls':'-'},
#                     'data_set_0':interp.ds_name(), 'sd_0':{'ms':5.5,'ls':'-'}})

#         tmp.append({'data_set_1':smooth.ds_name(), 'sd_1':{'ms':7.5,'ls':'-'},
#                     'data_set_0':interp.ds_name(), 'sd_0':{'ms':7.5,'ls':'-'}})
#         driver_obj.ts_spect_plots = tmp
    
#     driver_obj.update_all_metadata(processing_code=var_code)

#     if plot_info is not None:
#         driver_obj.update_all_metadata(**plot_info)
#     return driver_obj
