#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# enable debugging
import cgitb
cgitb.enable()

import sys
import os
sys.path.insert(0,os.path.dirname(__file__))

from voacap import app as application
