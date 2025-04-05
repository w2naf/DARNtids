#!/usr/bin/env python
import os
from hdf5_api import loadMusicArrayFromHDF5
import time

import numpy as np
import scipy as sp
import cv2

import matplotlib
import matplotlib.pyplot as plt

import wx
import wx.gizmos as gizmos

from auto_range import *

def colorize(img,mode='bgr',cmin=None,cmax=None):
    byt     = sp.misc.bytescale(img,cmin=cmin,cmax=cmax)
    result  = matplotlib.cm.jet(byt,bytes=True)
    result  = cv2.cvtColor(result,cv2.COLOR_RGBA2RGB)
    if mode == 'bgr':
        result  = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    return result

def draw_flow(img, flow, step=16):
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1)
    fx, fy = flow[y,x].T
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.polylines(vis, lines, 0, (0, 255, 0))
    for (x1, y1), (x2, y2) in lines:
        cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
    return vis

def draw_hsv(flow):
    h, w = flow.shape[:2]
    fx, fy = flow[:,:,0], flow[:,:,1]
    ang = np.arctan2(fy, fx) + np.pi
    v = np.sqrt(fx*fx+fy*fy)
    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[...,0] = ang*(180/np.pi/2)
    hsv[...,1] = 255
    hsv[...,2] = np.minimum(v*4, 255)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr

def gray2rgb(img):
    yy,xx       = img.shape
    rgb         = np.zeros([yy,xx,3],dtype=np.uint8)
    inx         = np.nonzero(img)
    vals        = img[inx[0],inx[1]]
    rgb[inx]    = np.tile(vals,(3,1)).T
    return rgb

def locked_aspect(dims0,dims1):
    width_0     = dims0[0]
    height_0    = dims0[1]

    width_1     = dims1[0]
    height_1    = dims1[1]

    aspect_0    = float(width_0) / float(height_0)
    aspect_1    = float(width_1) / float(height_1)

    if aspect_0 > aspect_1:
        new_height  = int(round((width_1 * height_0) / float(width_0)))
        new_width   = width_1
    elif aspect_0 < aspect_1:
        new_width   = int(round((height_1 * width_0) / float(height_0)))
        new_height  = height_1
    else:
        new_width   = width_1
        new_height  = height_1

    return (new_width,new_height)

def nothing(var):
    pass

def overlap(img_0,img_1):
    """Create a matrix where value is 1 if binary input matrices overlap, 0 where they don't."""
    bin_0   = np.zeros_like(img_0)
    bin_1   = np.zeros_like(img_1)

    bin_0[img_0.nonzero()]  = 1
    bin_1[img_1.nonzero()]  = 1

    add             = bin_0 + bin_1
    new             = np.zeros_like(add)
    new[add == 2 ]  = 255
    return new

def threshold(gray):
    flat            = gray.flatten()
    flat            = flat[flat != 0]
    flat.sort()
    top             = np.round(0.85 * flat.size)
    trs             = np.round(0.75 * np.median(flat[top:]))
    ret, thresh     = cv2.threshold(gray,trs,255,cv2.THRESH_BINARY_INV)
    return thresh

def warp_flow(img, flow):
    h, w = flow.shape[:2]
    flow = -flow
    flow[:,:,0] += np.arange(w)
    flow[:,:,1] += np.arange(h)[:,np.newaxis]
    res = cv2.remap(img, flow, None, cv2.INTER_LINEAR)
    return res

class FrameTrack(object):
    def __init__(self):
        pass

class DiscreetColors(object):
    def __init__(self):
        mp_color_convert    = matplotlib.colors.ColorConverter()

        color_wheel_list = []
        color_wheel_list.append('Red')
        color_wheel_list.append('Lime')
        color_wheel_list.append('OrangeRed')
        color_wheel_list.append('Blue')
        color_wheel_list.append('Gold')
        color_wheel_list.append('Turquoise')
        color_wheel_list.append('Fuchsia')
        color_wheel_list.append('Cyan')

        self.nr_colors  = len(color_wheel_list)
        self.color_wheel = {}
        key = 0
        for item in color_wheel_list:
            rgb = np.array(mp_color_convert.to_rgb(item))
            byt = sp.misc.bytescale(rgb,cmin=0,cmax=1,low=0,high=255)
            self.color_wheel[key] = {'name': item, 'rgb': byt}
            key += 1

disc_colors = DiscreetColors()

class Markers(object):
    def __init__(self,markers,gray=None):
        self.markers        = markers
        self.gry            = gray
        self.ids            = np.unique(self.markers)
        self.ids            = self.ids[self.ids != 0]
        self.ids.sort()
        self.calc_info()
        self.markers_rgb   = self.colorize()

    def subset(self,mrk_id,colorize=False):
        #Pull out only one of the current labels.
        tmp             = np.zeros_like(self.markers)
        tmp[self.markers == mrk_id] = 1

        if colorize:
            tmp = self.colorize(markers=tmp)
        return tmp

    def subset_bgr(self,mrk_id):
        return self.subset(mrk_id,colorize=True)

    def get_color(self,mrk_id):
        if not hasattr(self,'colors'): self.colors = disc_colors
        clr_inx = (mrk_id-1) % self.colors.nr_colors
        return self.colors.color_wheel[clr_inx]

    def colorize(self,markers=None):
        if markers is None: markers = self.markers
        yy,xx   = markers.shape
        rgb     = np.zeros([yy,xx,3],dtype=np.uint8)
        unq     = np.unique(markers)

        for mrk in unq:
            if mrk == 0: continue
            byt     = self.get_color(mrk)['rgb']
            inx     = np.where(markers == mrk)
            rgb[inx[0],inx[1],:] = byt
            
        for mrk in unq:
            if mrk == 0: continue
            byt     = self.get_color(mrk)['rgb']
            inx     = np.where(markers == mrk)
            minx    = np.min(inx[1])
            maxx    = np.max(inx[1])
            miny    = np.min(inx[0])
            maxy    = np.max(inx[0])

            for item in self.info:
                if item['id'] == mrk:
                    item['box'] = {'x0':minx, 'y0':miny, 'x1':maxx, 'y1': maxy}

            rgb     = cv2.rectangle(rgb,(minx,miny),(maxx,maxy),[255,0,0])

        return rgb
     
    def count(self,mrk_id):
        """Return the pixel count of a single ID."""
        return np.sum(self.markers == mrk_id)

    def counts(self):
        """Return pixel count of all marked regions in a sorted (largest first) list of dictionaries."""
        result  = []
        if self.ids.size != 0:
            result  = sorted(self.info,key=lambda k: k['count'])
            result  = result[::-1]  # Return largest value first.
        return result

    def calc_info(self):
        info = []
        if self.ids.size != 0:
            for mrk_id in self.ids:
                inx     = np.where(self.markers == mrk_id)

                tmp_dict            = {}
                tmp_dict['id']      = mrk_id
                tmp_dict['count']   = self.count(mrk_id)
                tmp_dict['color']   = self.get_color(mrk_id)['name']

                if self.gry is not None:
                    tmp_dict['hist']        = self.masked_histogram(self.gry,mrk_id)
                    tmp_dict['raw_mean']    = np.mean(tmp_dict['hist'])
                

                info.append(tmp_dict)

        info = sorted(info,key=lambda k: k['id'])
        self.info = info
        return self.info

    def __str__(self):
        lst = []
        for item in self.info:
            txt = 'id: %02d; mean: %.3f; count: %d' % (item['id'], item['raw_mean'], item['count'])
            lst.append(txt)
        result  = '\n'.join(lst)
        return result

    def masked_histogram(self,img,mrk_id):
        "Generate a 1d vector of an image masked with the region of a single ID."
        hist    = (img * self.subset(mrk_id)).flatten()
        return hist

class Segment(Markers):
    def __init__(self,img,gray=None):
        ret, markers   = cv2.connectedComponents(img)
        if gray == None: gray = img
        super(Segment,self).__init__(markers,gray=gray)

class MstidTracker(object):
    def __init__(self,filename='../data/wal-20121101.1400-20121101.1600.h5'):

        settings= {}
        self.settings           = settings
        settings['filename']    = filename

        settings['gate_min']    = 10
        settings['gate_max']    = 40
        settings['gate_min']    = None
        settings['gate_max']    = None
        settings['beam_min']    = None
        settings['beam_max']    = None

        settings['pyr_scale']   = 50
        settings['levels']      = 3
        settings['winsize']     = 15
        settings['iterations']  = 3
        settings['poly_n']      = 5
        settings['poly_sigma']  = 12

        settings['contrast']    = 1.
        settings['brightness']  = 9.5
        settings['cmin']        = 0.
        settings['cmax']        = 30.
        settings['scale_factor'] = 3
        settings['cv_interp']   = cv2.INTER_LINEAR # or cv2.INTER_NEAREST

        dataObj = loadMusicArrayFromHDF5(settings['filename'])

        self.currentData    = dataObj.DS000_originalFit

        # Get chosen start-stop times... not definative.
        settings['sTime']   = min(dataObj.active.time)
        settings['eTime']   = max(dataObj.active.time)

#        settings['gate_min'], settings['gate_max'] = auto_range(dataObj)

        if settings['gate_min'] == None: settings['gate_min'] = self.currentData.fov.gates.min()
        if settings['gate_max'] == None: settings['gate_max'] = self.currentData.fov.gates.max()
        if settings['beam_min'] == None: settings['beam_min'] = self.currentData.fov.beams.min()
        if settings['beam_max'] == None: settings['beam_max'] = self.currentData.fov.beams.max()

        self.tot_frames     = self.currentData.time.size
        self.frame          = -1

        self.frames         = {}

        res = True
        while res:
            res = self.track()

    def read_and_process(self):
        if self.frame+1 in list(self.frames.keys()):
            self.frame += 1
            dd  = self.frames[str(self.frame)]
        else:
            dd = FrameTrack()
            dd.frame_nr, dd.dt, dd.raw = self.read()
            dd.img      = self.raw2img(dd.raw)
            dd.img      = self.resize(dd.img)
            dd.raw      = self.resize(dd.raw)
            dd.gry      = cv2.cvtColor(dd.img, cv2.COLOR_RGB2GRAY)
            dd.gry_rgb  = gray2rgb(dd.gry)
            dd.thr      = threshold(dd.gry)
            dd.thr_rgb  = gray2rgb(dd.thr)
            dd.mrk      = Segment(dd.thr,gray=dd.raw)
        
            if self.frame == 0:
                dd.mrk_tracked  = dd.mrk
                self.frames['0']  = dd
                self.labels     = dd.mrk_tracked.markers.max()

        return dd

    def read(self):
        s   = self.settings
        if not hasattr(self,'frame'): self.frame = -1
        if self.frame >= self.tot_frames-1 or self.frame < -1: self.frame = -1
        self.frame += 1

        dt          = self.currentData.time[self.frame]
        data        = np.transpose(self.currentData.data[self.frame, s['beam_min']:s['beam_max'], s['gate_min']:s['gate_max']])

        data        = np.nan_to_num(data)
        data        = np.array(data)

        return self.frame, dt, data

    def raw2img(self,data):
        s           = self.settings

        data        = s['contrast'] * data + s['brightness']
        data        = np.clip(data,s['cmin'],s['cmax'])

        data        = sp.misc.bytescale(data, cmin=s['cmin'], cmax=s['cmax'], high=255, low=0)
#        shp         = data.shape
#        data        = np.zeros([shp[0],shp[1],4],dtype=np.uint8)
        data        = matplotlib.cm.rainbow(data,bytes=True,alpha=None)                   # Apply the mapping
#        data        = self.cm(data,bytes=True,alpha=None)                   # Apply the mapping
        data        = cv2.cvtColor(data,cv2.COLOR_RGBA2RGB)
        return data

    def resize(self,data):
        s           = self.settings
        data        = cv2.flip(data,0)
        data        = cv2.resize(data,(0,0),fx=s['scale_factor'],fy=s['scale_factor'],interpolation=s['cv_interp'])
        return data

    def track(self,verbose=False):
#        print '(%03d) %d' % (self.tot_frames,self.frame)
        if self.frame == self.tot_frames-1: return False
        if self.frame == -1:
            self.t0             = self.read_and_process()
            self.t1             = self.read_and_process()
            self.frames['0']      = self.t0
        elif self.frame == self.t1.frame_nr:
            self.t0     = self.t1
            self.t1     = self.read_and_process()
        else:
            self.t0     = self.read_and_process()
            self.t1     = self.read_and_process()

        self.status_text    = []
        txt = self.t0.dt.strftime('=== TRACKING: %Y %b %m %H%M to ') + self.t1.dt.strftime('%H%M UT ===')
        if verbose: print(txt)
        self.status_text.append(txt)

#        s   = self.settings
#        self.flow = cv2.calcOpticalFlowFarneback(self.t0.gry, self.t1.gry, 
#                flow        = None, 
#                pyr_scale   = s['pyr_scale'],
#                levels      = s['levels'],
#                winsize     = s['winsize'],
#                iterations  = s['iterations'],
#                poly_n      = s['poly_n'],
#                poly_sigma  = s['poly_sigma'],
#                flags       = 0)

        mrk_0               = self.t0.mrk_tracked
        mrk_1               = self.t1.mrk

        mrk_1_new_markers   = np.zeros_like(mrk_1.markers,dtype=np.uint32)
        found_markers       = []
        if len(mrk_0.info) > 0 and len(mrk_1.info) > 0:
            for info_0 in mrk_0.info:
                id_0            = info_0['id']

                sub_0           = mrk_0.subset(id_0)

                dists   = []
                for info_1 in mrk_1.info:
                    id_1    = info_1['id']
                    if 'corr' not in info_1: info_1['corr'] = {}
                    h0      = info_0['hist']
                    h1      = info_1['hist']

                    dist    = sp.spatial.distance.correlation(h0,h1)
                    dists.append({'id_0':id_0,'id_1':id_1,'corr_dist':dist})
                    corr    = np.correlate(info_0['hist'],info_1['hist'])

                    info_1['corr']      = {id_0: corr}
                    info_1['corr_dist'] = {id_0: dist}

                if False:
                    nx_plots    = 1
                    ny_plots    = len(mrk_1.info)
                    plot_nr     = 1
                    fig         = plt.figure()
                    for info_1 in mrk_1.info:
                        ax      = fig.add_subplot(ny_plots,nx_plots,plot_nr)
                        corr    = info_1['corr'][id_0]
                        ax.bar(np.arange(corr.size),corr,lw=0)
                        title   = []
                        title.append('mrk_0 Blob %d, mrk_1 Blob %d' % (id_0,info_1['id']))
                        title.append('Distance %.03f' % info_1['corr_dist'][id_0])
                        ax.set_title('\n'.join(title))
                        plot_nr += 1
                    fig.tight_layout()
                    fig.show()

                #sort distances
                dists       = sorted(dists,key=lambda k: k['corr_dist'])
                min_dist_dct= dists[0]

                min_dist    = min_dist_dct['corr_dist']
                id_1        = min_dist_dct['id_1']
                color_0     = (item for item in mrk_0.info if item['id'] == id_0).next()['color']
                color_1     = (item for item in mrk_1.info if item['id'] == id_1).next()['color']
                if min_dist <= 0.45:
                    mrk_1_new_markers[mrk_1.markers == id_1] = id_0
                    found_markers.append(id_1)   #Keep track of found markers
                    txt = 'TRACKED: mrk0.%d (%s) -- mrk1.%d (%s) (dist: %.3f) --> trk.%d (%s)' % (id_0,color_0,id_1,color_1,min_dist,id_0,color_0)
                    if verbose: print(txt)
                    self.status_text.append(txt)
                else:
                    txt = 'NO MATCH: mrk0.%d (%s) -- mrk1.%d (%s) (dist: %.3f)' % (id_0,color_0,id_1,color_1,min_dist)
                    if verbose: print(txt)
                    self.status_text.append(txt)

        for mrk in mrk_1.ids:
            if mrk not in found_markers:
                self.labels += 1
                mrk_1_new_markers[mrk_1.markers == mrk] = self.labels

        self.t1.mrk_tracked = Markers(mrk_1_new_markers,gray=self.t1.raw)
        self.t1.status      = self.status_text
        self.frames[str(self.t1.frame_nr)] = self.t1

        return True

class AdjustSettings(wx.Frame):
    def __init__(self,parent,id):
        self.data   = parent.data

        wx.Frame.__init__(self, parent, id, 'Settings',(0,0))
        box = wx.BoxSizer(wx.HORIZONTAL)
        fgs = wx.FlexGridSizer(2,2,2,2)
    
        contrast_lbl    = wx.StaticText(self,label='Contrast\nx 0.01')
        brightness_lbl  = wx.StaticText(self,label='Brightness\nx 0.1')

        self.contrast   = wx.Slider(self,0,0,0,200,wx.DefaultPosition,wx.DefaultSize,wx.SL_AUTOTICKS | wx.SL_HORIZONTAL | wx.SL_LABELS)
        self.brightness = wx.Slider(self,-1,0,0,400,wx.DefaultPosition,wx.DefaultSize,wx.SL_AUTOTICKS | wx.SL_HORIZONTAL | wx.SL_LABELS)
        fgs.AddMany([(contrast_lbl,0,wx.ALIGN_CENTER_VERTICAL),   (self.contrast,1,wx.EXPAND),
                     (brightness_lbl,0,wx.ALIGN_CENTER_VERTICAL), (self.brightness,1,wx.EXPAND)])
        fgs.AddGrowableCol(1, 1)

        box.Add(fgs,1, wx.EXPAND)

        self.SetSizer(box)

        self.contrast.SetValue(self.data.settings['contrast'] * 100)
        self.brightness.SetValue(self.data.settings['brightness']*10)
        self.Bind(wx.EVT_SCROLL,self.set_contrast,id=self.contrast.GetId())
        self.Bind(wx.EVT_SCROLL,self.set_brightness,id=self.brightness.GetId())

    def set_contrast(self,event):
        self.data.settings['contrast'] = event.GetPosition()/100.

    def set_brightness(self,event):
        self.data.settings['brightness'] = event.GetPosition()/10.

class ShowData(wx.Frame):
    def __init__(self, parent, id, title, fps=15,filename='../data/wal-20121101.1400-20121101.1600.h5'):
        self.path       = filename
        self.initialize = True

        wx.Frame.__init__(self, parent, id, title)

        self.statusbar  = self.CreateStatusBar()
#        t0  = datetime.datetime.now()
        self.data       = MstidTracker(filename=self.path)


#        saveMusicArrayToHDF5(self.data, 'output/trackerObj.h5')


#        t1  = datetime.datetime.now()
#        delt    = t1 - t0
#        print delt
        self.SetTitle(os.path.basename(self.data.settings['filename']))

        self.fps        = fps
        self.timer      = wx.Timer(self)
        self.frame      = 0

        self.NextFrame(None,track=False)

        self.Bind(wx.EVT_SIZE,self.OnSize)
        self.Bind(wx.EVT_PAINT, self.OnPaint)
        self.Bind(wx.EVT_TIMER, self.NextFrame)
        self.Bind(wx.EVT_CLOSE, self.OnQuit)


    def NextFrame(self, event,track=False):
        self.frm_0  = self.data.frames[str(self.frame)]
        self.frm_1  = self.data.frames[str(self.frame+1)]

        if self.frm_1.frame_nr == self.data.tot_frames-1: 
            self.pause()
#            self.final_plot()
            return
        
        if self.initialize:
            self.imgs    = {}
            self.imgs[0] = {'name':'img0'}
            self.imgs[1] = {'name':'img1'}
            self.imgs[2] = {'name':'gry0'}
            self.imgs[3] = {'name':'gry1'}
            self.imgs[4] = {'name':'thr0'}
            self.imgs[5] = {'name':'thr1'}
            self.imgs[6] = {'name':'mrk0'}
            self.imgs[7] = {'name':'mrk1'}
            self.imgs[8] = {'name':'Tracked'}

        if track:
            self.data.track()

        imgs            = self.imgs
        imgs[0]['img']  = self.frm_0.img
        imgs[1]['img']  = self.frm_1.img
        imgs[2]['img']  = self.frm_0.gry_rgb
        imgs[3]['img']  = self.frm_1.gry_rgb
        imgs[4]['img']  = self.frm_0.thr_rgb
        imgs[5]['img']  = self.frm_1.thr_rgb
        imgs[6]['img']  = self.frm_0.mrk_tracked.markers_rgb
        imgs[7]['img']  = self.frm_1.mrk.markers_rgb
        imgs[8]['img']  = self.frm_1.mrk_tracked.markers_rgb


        if self.initialize:
            vbox    = wx.BoxSizer(wx.VERTICAL)
            vbox.Add(self.toolbar(),0,border=5)

            self.bmp_height = imgs[0]['img'].shape[0]
            self.bmp_width  = imgs[0]['img'].shape[1]
            self.bmp_pad    = 3
            self.nr_imgs    = len(imgs)
            self.frm_width  = self.nr_imgs * (self.bmp_pad + self.bmp_width) + 2*self.bmp_pad
            self.frm_height = self.bmp_height + 2*self.bmp_pad

            hbox = wx.BoxSizer(wx.HORIZONTAL)
            for key in range(len(list(imgs.keys()))):
                img     = imgs[key]
                pnl1    = wx.Panel(self, -1, style=wx.SIMPLE_BORDER)

                hbox.Add(pnl1, 1, wx.EXPAND | wx.ALL, self.bmp_pad)
                height, width   = img['img'].shape[:2]
                img['panel']    = pnl1

            vbox.Add(hbox,1,wx.EXPAND)

            #Controls on the bottom.
            vbox2   = wx.BoxSizer(wx.VERTICAL)
            vbox2.Add(self.timebar(),0,wx.EXPAND)
            vbox2.Add(self.mediabar(),0,wx.CENTER)

            hbox2   = wx.BoxSizer(wx.HORIZONTAL)
            hbox2.Add(vbox2,2)

            self.running_info = wx.TextCtrl(self,-1,style = wx.TE_MULTILINE|wx.TE_READONLY|wx.TE_AUTO_URL)
            
            hbox2.Add(self.running_info,2,wx.EXPAND)

            vbox.Add(hbox2,0,wx.EXPAND)

            self.SetSizer(vbox)

            zoom    = 2
            self.frm_width  = self.frm_width  * zoom
            self.frm_height = self.frm_height * zoom
            self.SetSize((self.frm_width,self.frm_height))

            self.initialize = False
        else:
            for key in range(len(list(imgs.keys()))):
                img = imgs[key]
                img['bmp'].CopyFromBuffer(self.img_prep(img['img']))
                self.Refresh()

        self.running_info.AppendText('\n'+'\n'.join(self.frm_1.status)+'\n')
        self.statusbar.SetStatusText(self.frm_0.dt.strftime('%Y %b %m %H:%M UT'))
        self.led_nr.SetValue('%03d' % self.frm_0.frame_nr)
        self.led.SetValue(self.frm_0.dt.strftime('%H%M'))
        self.time_slider.SetValue(self.frm_0.frame_nr)
        self.frame += 1

    def toolbar(self):
        toolbar     = wx.ToolBar(self,-1,style=wx.TB_HORIZONTAL|wx.NO_BORDER)
        toolbar.AddSimpleTool(2, wx.Image('icons/stock_open.png',wx.BITMAP_TYPE_PNG).ConvertToBitmap(),'Open','')
        toolbar.AddSimpleTool(3, wx.Image('icons/preferences-desktop-2.png',wx.BITMAP_TYPE_PNG).ConvertToBitmap(),'Settings','')
        toolbar.AddSeparator()
        toolbar.AddSimpleTool(4, wx.Image('icons/stock_exit.png',wx.BITMAP_TYPE_PNG).ConvertToBitmap(),'Exit','')
        toolbar.Realize()

        self.Bind(wx.EVT_TOOL,self.OnOpen,id=2)
        self.Bind(wx.EVT_TOOL,self.OnSettings,id=3)
        self.Bind(wx.EVT_TOOL,self.OnQuit,id=4)
        return toolbar

    def mediabar(self):
        self.play_bmp       = wx.Image('icons/media-playback-start-7.png',wx.BITMAP_TYPE_PNG).ConvertToBitmap()
        self.pause_bmp      = wx.Image('icons/media-playback-pause-7.png',wx.BITMAP_TYPE_PNG).ConvertToBitmap()
        self.play_id        = 23

        toolbar             = wx.ToolBar(self,-1,style=wx.TB_HORIZONTAL|wx.NO_BORDER)
        toolbar.AddSimpleTool(21, wx.Image('icons/media-skip-backward-7.png',wx.BITMAP_TYPE_PNG).ConvertToBitmap(),'Rewind','')
        toolbar.AddSimpleTool(22, wx.Image('icons/media-seek-backward-7.png',wx.BITMAP_TYPE_PNG).ConvertToBitmap(),'Step Back','')
        toolbar.AddSimpleTool(self.play_id, self.play_bmp,'Play','')
        toolbar.AddSimpleTool(24, wx.Image('icons/media-seek-forward-7.png',wx.BITMAP_TYPE_PNG).ConvertToBitmap(),'Step Forward','')
        toolbar.AddSimpleTool(25, wx.Image('icons/media-skip-forward-7.png',wx.BITMAP_TYPE_PNG).ConvertToBitmap(),'Jump to End','')
        toolbar.Realize()
        
        toolbar.EnableTool(25,False)
        self.mediabar       = toolbar

        self.Bind(wx.EVT_TOOL,self.OnRewind,id=21)
        self.Bind(wx.EVT_TOOL,self.OnBack,id=22)
        self.Bind(wx.EVT_TOOL,self.OnPlay,id=self.play_id)
        self.Bind(wx.EVT_TOOL,self.OnForward,id=24)

        return toolbar

    def timebar(self):
        box = wx.BoxSizer(wx.HORIZONTAL)
        pnl         = wx.Panel(self,-1)
        self.led_nr    = gizmos.LEDNumberCtrl(pnl, -1,size=(60,35),style=gizmos.LED_ALIGN_CENTER)
        self.led_nr.SetValue('000')
        box.Add(pnl,0)

        pnl         = wx.Panel(self,-1)
        self.led    = gizmos.LEDNumberCtrl(pnl, -1,size=(80,35),style=gizmos.LED_ALIGN_CENTER)
        self.led.SetValue('0000')
        box.Add(pnl,0)

        self.time_slider    = wx.Slider(self,-1,0,0,self.data.tot_frames-2)
        self.time_slider.Disable()
        box.Add(self.time_slider,1)
        return box

    def OnPaint(self,event):
        for key in range(len(list(self.imgs.keys()))):
            img = self.imgs[key]
            dc = wx.BufferedPaintDC(img['panel'])
            dc.Clear()

            pnl_width, pnl_height = img['panel'].GetSize()
            xPos    = (pnl_width  - self.bmp_width)/2
            yPos    = (pnl_height - self.bmp_height)/2

            dc.DrawBitmap(img['bmp'], xPos, yPos)

            wx.StaticText(img['panel'],-1,img['name'],(0,0),style=wx.ALIGN_CENTRE)

    def OnPlay(self,event):
        if not self.timer.IsRunning():
            self.play()
        else:
            self.pause()

    def play(self,event=None):
        if not self.timer.IsRunning():
            self.mediabar.SetToolNormalBitmap(self.play_id,self.pause_bmp)
            self.mediabar.SetToolShortHelp(self.play_id,'Pause')
            self.timer.Start(1000./self.fps)

    def pause(self,event=None):
        if self.timer.IsRunning():
            self.mediabar.SetToolNormalBitmap(self.play_id,self.play_bmp)
            self.mediabar.SetToolShortHelp(self.play_id,'Play')
            self.timer.Stop()

    def OnForward(self,event):
        self.pause()
        self.NextFrame(None)

    def OnBack(self,event):
        self.pause()
        self.frame -= 2
        if self.frame < 1: self.frame = 0
        self.NextFrame(None)

    def OnRewind(self,event):
        self.pause()
        self.running_info.Clear()
        self.data   = MstidTracker(filename=self.path)
        self.frame  = 0
        self.NextFrame(None)

    def OnSize(self,event):
        self.Layout()
        for key in range(len(list(self.imgs.keys()))):
            img = self.imgs[key]
            sz = img['panel'].GetSize()
            sz = locked_aspect((self.bmp_width,self.bmp_height),sz)

            self.bmp_width  = sz[0]
            self.bmp_height = sz[1]

            if 'bmp' in img: img['bmp'].Destroy()
            img['bmp']  = self.img2bmp(img['img'])

    def OnOpen(self,event):
        dlg = wx.FileDialog(self, "Choose a file", os.getcwd(), "", "*.*", wx.OPEN)
        if dlg.ShowModal() == wx.ID_OK:
            self.pause()
            self.path   = dlg.GetPath()
            mypath      = os.path.basename(self.path)
            self.running_info.Clear()
            self.frame  = 0
            self.data   = MstidTracker(filename=self.path)
            self.NextFrame(None,track=False)
            self.SetStatusText("You selected: %s" % mypath)
            self.SetTitle(os.path.basename(self.data.settings['filename']))
            self.time_slider.SetRange(0,self.data.tot_frames)
             
        dlg.Destroy()

    def OnQuit(self,event):
        self.Destroy()

    def OnSettings(self,event):
        self.settings_frame = AdjustSettings(self,-1)
        self.settings_frame.Show()
    
    def img_prep(self,img):
        if not hasattr(self,'bmp_height'): self.bmp_height = img.shape[0]
        if not hasattr(self,'bmp_width'):  self.bmp_width  = img.shape[1]

        if self.bmp_height != img.shape[0] or self.bmp_width != img.shape[1]:
#            img       = cv2.resize(img,(self.bmp_width,self.bmp_height),interpolation=cv2.INTER_LINEAR)
            img       = cv2.resize(img,(self.bmp_width,self.bmp_height),interpolation=cv2.INTER_NEAREST)
        return img

    def img2bmp(self,img):
        img = self.img_prep(img)
        bmp = wx.BitmapFromBuffer(self.bmp_width, self.bmp_height, img)
        return bmp

    def final_plot(self):
        plt.ion()
        fig     = plt.figure()
        axis    = fig.add_subplot(111)
        axis.plot(np.sin(list(range(100))))


class MyApp(wx.App):
    def OnInit(self):
        frm = ShowData(None,-1,'MSTID Data')
        frm.Show()
        
        return True

if __name__ == '__main__':
    app     = MyApp(0)
    app.MainLoop()
