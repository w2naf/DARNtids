# from pyDARNmusic import load_fitacf,musicRTP
# from pyDARNmusic.music import musicArray
# from matplotlib.figure import Figure
# import datetime
# radar = "bks"
# stime   = datetime.datetime(2012,12,1,16)
# etime   = datetime.datetime(2012,12,1,18)
# fitacf  = load_fitacf(radar,stime,etime)
# dataObj = musicArray(fitacf,sTime=stime,eTime=etime,fovModel='GS')
# fig = Figure(figsize=(20,10))

# musicRTP(dataObj,axis=ax)
# d = 'webserver/static/rti'
# param = "powerti12"
# outputFile = d+'/'+stime.strftime('%Y%m%d')+'.'+radar+'.'+param+'rti.png'
# fig.savefig(outputFile)