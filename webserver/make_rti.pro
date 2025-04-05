radar = 'bks'
param = 'power'
sDate = 20110101
eDate = 20110102

RAD_FIT_READ,sDate,radar

file=DIR('idl_rti.ps',/PS_OPEN)
RAD_FIT_PLOT_RTI,/SUN,COORDS='gs_geog'
PS_CLOSE
END
