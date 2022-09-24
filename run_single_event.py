#!/usr/bin/env python
import sys
import matplotlib
matplotlib.use('Agg')

import mstid

init_param_file = sys.argv[1] 
print(init_param_file)
mstid.run_music_init_param_file(init_param_file)
sys.exit()
