import pickle

filename    = 'static/music/bks/20100601.2000-20100601.2200/bks-20100601.2000-20100601.2200.p'

with open(filename,'rb') as fl:
    dataObj = pickle.load(fl)


