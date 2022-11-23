class TimeCheck(object):
    import datetime
    #import inspect
    #curr_file = inspect.getfile(inspect.currentframe()) # script filename (usually with path)
    #import logging
    #import logging.config
    #logging.filename    = curr_file+'.log'
    #logging.config.fileConfig("logging.conf")
    #log = logging.getLogger("root")

    def __init__(self,label=None,log=None):
        self.label  = label
        self.log    = log
        self.t0     = datetime.datetime.now()
    def check(self):
        self.t1 = datetime.datetime.now()
        dt      = self.t1 - self.t0

        txt = '{sec}'.format(sec=str(dt))

        if self.label is not None:
            txt = ': '.join([self.label,txt])

        if self.log is not None:
            log.info(txt)
        else:
            print(txt)

def prepare_output_dirs(output_dirs={0:'output'},clear_output_dirs=False,width_100=False,img_extra=''):
    import os
    import shutil

    if width_100:
        img_extra = "width='100%'"

    txt = []
    txt.append('<?php')
    txt.append('foreach (glob("*.png") as $filename) {')
    txt.append('    echo "<img src=\'$filename\' {img_extra}> ";'.format(img_extra=img_extra))
    txt.append('}')
    txt.append('?>')
    show_all_txt = '\n'.join(txt)

    txt = []
    txt.append('<?php')
    txt.append('foreach (glob("*.png") as $filename) {')
    txt.append('    echo "<img src=\'$filename\' {img_extra}> <br />";'.format(img_extra=img_extra))
    txt.append('}')
    txt.append('?>')
    show_all_txt_breaks = '\n'.join(txt)

    for value in output_dirs.values():
        if clear_output_dirs:
            try:
                shutil.rmtree(value)
            except:
                pass
        try:
            os.makedirs(value)
        except:
            pass

        with open(os.path.join(value,'0000-show_all.php'),'w') as file_obj:
            file_obj.write(show_all_txt)
        with open(os.path.join(value,'0000-show_all_breaks.php'),'w') as file_obj:
            file_obj.write(show_all_txt_breaks)
