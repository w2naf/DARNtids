#!/usr/bin/env python

from distutils.core import setup

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(name='mstid',
      version='0.1',
      description='SuperDARN Traveling Ionospheric Disturbance Analysis Toolkit',
      author='Nathaniel A. Frissell',
      author_email='nathaniel.frissell@scranton.edu',
      url='https://hamsci.org',
      packages=['mstid'],
      install_requires=requirements
     )
