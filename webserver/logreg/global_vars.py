#mstid_list  = 'paper2_wal_2012'
mstid_list  = 'paper2_bks_2010_2011_0817mlt'

clear_output_dirs       = True
test                    = False

#dv                      = 'none'
dv                      = 'mstid'

kmin                    = 0.009
n_trials                = 1000

# How do you want to sort the final plots?
#sort_by     = 'sum_sqrs'
#sort_by     = 'r_sqrd'
#sort_by     = 'aic'
sort_by     = 'mean_pct_correct'

output_dirs             = {}
output_dirs['search']   = 'output/logreg/%s/search' % mstid_list
output_dirs['result']   = 'output/logreg/%s/result' % mstid_list
output_dirs['rank']     = 'output/logreg/%s/rank'   % mstid_list

# Define category dictionary for plotting. #####################################
# You can change the number of things plotted by simply commenting out a cat_dict entry
# or by changing the number of mlt_bins.
cat_dict = {}
cat_dict['mstid']   = {'title':'MSTID', 'color':'r'}
cat_dict['quiet']   = {'title':'Quiet', 'color':'g'}
cat_dict['none']    = {'title':'None' , 'color':'b'}

# Build up possible independent variables. #####################################
prm_dict        = {}

param_code   = 'dom_spec'
prm_dict[param_code] = {}
prm_dict[param_code]['param_name']   = 'Dominant Spectrum'
prm_dict[param_code]['param_units']  = ''
prm_dict[param_code]['bins']         = 25
prm_dict[param_code]['bins_range']   = None

param_code   = 'total_spec'
prm_dict[param_code] = {}
prm_dict[param_code]['param_name']   = 'Total Spectrum'
prm_dict[param_code]['param_units']  = ''
prm_dict[param_code]['bins']         = 25
prm_dict[param_code]['bins_range']   = None

param_code   = 'spec_ratio'
prm_dict[param_code] = {}
prm_dict[param_code]['param_name']   = '(Dom FFT Bin)/(All FFT Bins)'
prm_dict[param_code]['param_units']  = ''
prm_dict[param_code]['bins']         = 25
prm_dict[param_code]['bins_range']   = (0.05,0.1)
prm_dict[param_code]['bins_range']   = None

param_code = 'orig_rti_cnt'
prm_dict[param_code] = {}
prm_dict[param_code]['param_name'] = 'Original RTI Scatter Count'
prm_dict[param_code]['param_units'] = ''
prm_dict[param_code]['bins'] = 25
prm_dict[param_code]['bins_range'] = None

param_code = 'orig_rti_mean'
prm_dict[param_code] = {}
prm_dict[param_code]['param_name'] = 'Original RTI Mean'
prm_dict[param_code]['param_units'] = '[dB]'
prm_dict[param_code]['bins'] = 25
prm_dict[param_code]['bins_range'] = None

param_code = 'orig_rti_median'
prm_dict[param_code] = {}
prm_dict[param_code]['param_name'] = 'Original RTI Median'
prm_dict[param_code]['param_units'] = '[dB]'
prm_dict[param_code]['bins'] = 25
prm_dict[param_code]['bins_range'] = None

param_code = 'orig_rti_var'
prm_dict[param_code] = {}
prm_dict[param_code]['param_name'] = 'Original RTI Variance'
prm_dict[param_code]['param_units'] = '[dB]'
prm_dict[param_code]['bins'] = 25
prm_dict[param_code]['bins_range'] = None

param_code = 'blob_track-short_list_mean_count'
prm_dict[param_code] = {}
prm_dict[param_code]['param_name'] = 'Tracked Blob Pixel Count Mean'
prm_dict[param_code]['param_units'] = 'px'
prm_dict[param_code]['bins'] = 25
prm_dict[param_code]['bins_range'] = None

param_code = 'blob_track-short_list_var_count'
prm_dict[param_code] = {}
prm_dict[param_code]['param_name'] = 'Tracked Blob Pixel Count Variance'
prm_dict[param_code]['param_units'] = 'px'
prm_dict[param_code]['bins'] = 25
prm_dict[param_code]['bins_range'] = None

param_code = 'blob_track-short_list_mean_raw_mean'
prm_dict[param_code] = {}
prm_dict[param_code]['param_name'] = 'Tracked Blob Mean Power Mean'
prm_dict[param_code]['param_units'] = 'dB'
prm_dict[param_code]['bins'] = 25
prm_dict[param_code]['bins_range'] = None

param_code = 'blob_track-short_list_var_raw_mean'
prm_dict[param_code] = {}
prm_dict[param_code]['param_name'] = 'Tracked Blob Mean Power Variance'
prm_dict[param_code]['param_units'] = 'dB'
prm_dict[param_code]['bins'] = 25
prm_dict[param_code]['bins_range'] = None

param_code = 'nr_sigs'
prm_dict[param_code] = {}
prm_dict[param_code]['param_name'] = 'Nr. of Signals'
prm_dict[param_code]['param_units'] = ''
prm_dict[param_code]['bins'] = 25
prm_dict[param_code]['bins_range'] = None

param_code = 'area'
prm_dict[param_code] = {}
prm_dict[param_code]['param_name'] = 'Area'
prm_dict[param_code]['param_units'] = 'px'
prm_dict[param_code]['bins'] = 25
prm_dict[param_code]['bins_range'] = None

param_code = 'true_max'
prm_dict[param_code] = {}
prm_dict[param_code]['param_name'] = 'MUSIC Maximum'
prm_dict[param_code]['param_units'] = ''
prm_dict[param_code]['bins'] = 25
prm_dict[param_code]['bins_range'] = (0.0015,0.0035)

param_code = 'k'
prm_dict[param_code] = {}
prm_dict[param_code]['param_name']  = 'Horiz. Wavenumber'
prm_dict[param_code]['param_units'] = ''
prm_dict[param_code]['bins']        = 50
#prm_dict[param_code]['bins_range'] = (0.0015,0.0035)

#param_code = 'blob_track-short_list_mean_box_x0'
#prm_dict[param_code] = {}
#prm_dict[param_code]['param_name'] = 'Tracked Blob x0 Mean'
#prm_dict[param_code]['param_units'] = 'px'
#prm_dict[param_code]['bins'] = 25
#prm_dict[param_code]['bins_range'] = None
#
#param_code = 'blob_track-short_list_var_box_x0'
#prm_dict[param_code] = {}
#prm_dict[param_code]['param_name'] = 'Tracked Blob x0 Variance'
#prm_dict[param_code]['param_units'] = 'px'
#prm_dict[param_code]['bins'] = 25
#prm_dict[param_code]['bins_range'] = None
#
#param_code = 'blob_track-short_list_mean_box_x1'
#prm_dict[param_code] = {}
#prm_dict[param_code]['param_name'] = 'Tracked Blob x1 Mean'
#prm_dict[param_code]['param_units'] = 'px'
#prm_dict[param_code]['bins'] = 25
#prm_dict[param_code]['bins_range'] = None
#
#param_code = 'blob_track-short_list_var_box_x1'
#prm_dict[param_code] = {}
#prm_dict[param_code]['param_name'] = 'Tracked Blob x1 Variance'
#prm_dict[param_code]['param_units'] = 'px'
#prm_dict[param_code]['bins'] = 25
#prm_dict[param_code]['bins_range'] = None
#
#param_code = 'blob_track-short_list_mean_box_y0'
#prm_dict[param_code] = {}
#prm_dict[param_code]['param_name'] = 'Tracked Blob y0 Mean'
#prm_dict[param_code]['param_units'] = 'px'
#prm_dict[param_code]['bins'] = 25
#prm_dict[param_code]['bins_range'] = None
#
#param_code = 'blob_track-short_list_var_box_y0'
#prm_dict[param_code] = {}
#prm_dict[param_code]['param_name'] = 'Tracked Blob y0 Variance'
#prm_dict[param_code]['param_units'] = 'px'
#prm_dict[param_code]['bins'] = 25
#prm_dict[param_code]['bins_range'] = None
#
#param_code = 'blob_track-short_list_mean_box_y1'
#prm_dict[param_code] = {}
#prm_dict[param_code]['param_name'] = 'Tracked Blob y1 Mean'
#prm_dict[param_code]['param_units'] = 'px'
#prm_dict[param_code]['bins'] = 25
#prm_dict[param_code]['bins_range'] = None
#
#param_code = 'blob_track-short_list_var_box_y1'
#prm_dict[param_code] = {}
#prm_dict[param_code]['param_name'] = 'Tracked Blob y1 Variance'
#prm_dict[param_code]['param_units'] = 'px'
#prm_dict[param_code]['bins'] = 25
#prm_dict[param_code]['bins_range'] = None
