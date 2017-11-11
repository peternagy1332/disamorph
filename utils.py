import copyreg
import datetime
import sys
import time
import os

class Logger(object):
    def __init__(self, model_directory, prefix):
        self.terminal = sys.stdout
        st = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H:%M:%S')
        self.run_log_file = open(os.path.join(model_directory, prefix + '-' + st) + '.log', 'w', encoding='utf8')

    def write(self, message):
        #st = datetime.datetime.fromtimestamp(time.time()).strftime('%H:%M:%S')
        self.terminal.write(message)

        #if '\r' not in message:
        if '\r' not in message:
            self.run_log_file.write(message)

        if '\r' in message and message.count('=') == 60:
            self.run_log_file.write(message + os.linesep)

    def flush(self):
        #self.run_log_file.close()
        pass

class Utils(object):
    def __init__(self, model_configuration):
        self.__config = model_configuration

    @staticmethod
    def update_progress(count, total, suffix=''):
        bar_len = 30
        filled_len = int(round(bar_len * count / float(total)))

        percents = round(100.0 * count / float(total), 1)
        bar = '=' * filled_len + '-' * (bar_len - filled_len)

        sys.stdout.write('\t[%s] %s%s | %s\r' % (bar, percents, '%', suffix))
        sys.stdout.flush()

    def redirect_stdout(self, prefix):
        sys.stdout = Logger(self.__config.model_directory, prefix)

    def start_stopwatch(self):
        self.start_time = time.time()

    def stop_stopwatch_and_print_running_time(self):
        stop_time = time.time()

        diff = stop_time - self.start_time

        diffD = (((diff / 365) / 24) / 60)
        Days = int(diffD)

        diffH = (diffD - Days) * 365
        Hours = int(diffH)

        diffM = (diffH - Hours) * 24
        Minutes = int(diffM)

        diffS = (diffM - Minutes) * 60
        Seconds = int(diffS)

        print("Running took ", Days, "days;", str(Hours)+":"+str(Minutes)+":"+str(Seconds))

    @staticmethod
    def fullvars(obj):
        cls = type(obj)
        try:
            slotnames = cls.__dict__['__slotnames__']
        except (KeyError, AttributeError):
            slotnames = copyreg._slotnames(cls)
        try:
            d = vars(obj).copy()
        except TypeError:
            d = {}
        for name in slotnames:
            try:
                d[name] = getattr(obj, name)
            except AttributeError:
                pass
        return d