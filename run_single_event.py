#!/usr/bin/env python
"""
This script will launch the MSTID/MUSIC processing for a single event given
the path to an initialization file with the event parameters. This allows each
event to be launched as an independent process. By running each event as an
independent system process, it makes very easy to:
    1. Stop/prevent memory leaks.
    2. Run events in parallel.
"""
import sys
import os
import datetime
import matplotlib
matplotlib.use('Agg')

import logging
import mstid

logger = logging.getLogger()
class LogFilter(logging.Filter):
    def __init__(self,message):
        self.message = message
    def filter(self, record):
        return not record.getMessage().startswith(self.message)
logger.addFilter(LogFilter('An error occured while defining limits.  No limits set.  Check your input values.'))

init_param_file = sys.argv[1] 
print(init_param_file)

# Set up logging.
log_dir = 'log'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
log_path = os.path.join(log_dir,os.path.basename(init_param_file)+'.log')
with open(log_path,'w') as fl:
    fl.write('{!s}: Processing {!s}\n'.format(datetime.datetime.now(),init_param_file))
logging.basicConfig(filename=log_path, encoding='utf-8', level=logging.WARN)

# Run MSTID/MUSIC processing.
mstid.run_music_init_param_file(init_param_file)

# Delete log file if no real messages.
with open(log_path,'r') as fl:
    log = fl.readlines()
if len(log) == 1:
    os.remove(log_path)

# Exit the program.
sys.exit()
