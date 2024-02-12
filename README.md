# DARNtids
SuperDARN Traveling Ionospheric Disturbance Analysis Toolkit

This package will calculate the level of Medium Scale Traveling Ionospheric Disturbance (MSTID)
activity observed by SuperDARN radars via the MSTID Index described by Frissell et al. (2016)
(https://doi.org/10.1002/2015JA022168). It will also run the SuperDARN MSTID MUSIC algorithm
(https://github.com/HamSCI/pyDARNmusic) to estimate the speed, propagation direction, and 
horizontal wavelength of the observed MSTIDs.

To run this code, see the script ```music_and_classify_year_loop.py```.

This code was used to calculated the MSTID index in the Frissell et al. (2024) manuscript submitted
to Geophysical Research Letters.
