#!/usr/bin/env python
import matplotlib
matplotlib.use('Agg')

import multi_radar_music_support as msc
runfile_path    = '/data/mstid/statistics/webserver/static/multi_radar_music/fhw_fhe/20121103.1400-20121103.1600/fhw_fhe-20121103.1400-20121103.1600.runfile.p'
process_level   = 'all'
msc.music_plot_all(runfile_path,process_level=process_level)
