import os.path
import sys, time
import sys,os
import root_config
import threading
import root_config
from misc_util import  your_datetime
def _get_logFilePath(dir_:str,log_file_prefix: str):
    pid = os.getpid()
    th = threading.currentThread()
    th_name = th.getName()  
    log_file_name = f"{log_file_prefix}{os.path.basename(sys.argv[0])[:-3]}--{your_datetime():%m.%d-%H:%M:%S}--{pid}:{th_name}"

    def name2fp(name):
        full_path = os.path.join(
            dir_,
            f"{name}.log"
        )
        return full_path
    os.makedirs(dir_,exist_ok=True)
    while (os.path.exists(name2fp(log_file_name))):
        log_file_name += f"_"
    # print(f"redirect stdout to:\n\"{name2fp(log_file_name)}\"")
    # print(f"redirect stdout to:\n{name2fp(log_file_name)}")
    print(f"\nredirect to:\n{name2fp(log_file_name)}:0\n")
    # print(f"redirect stdout to:\n<{name2fp(log_file_name)}>")
    # print(f"redirect stdout to:\n<{name2fp(log_file_name)}>:0")
    return name2fp(log_file_name)
def redirectA(dir_:str,log_file_prefix: str):
    logFilePath=_get_logFilePath(dir_,log_file_prefix)
    sys.stdout = open(logFilePath, 'w')





class Tee:#cursor
    FORCE_TO_FLUSH_INTERVAL = 10
    def __init__(self, *files):
        self.files = []
        for file in files:
            if isinstance(file, str):
                self.files.append(open(file, 'w'))
            else:
                self.files.append(file)
        self.last_flush_time = time.time()
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            if time.time() - self.last_flush_time > self.FORCE_TO_FLUSH_INTERVAL:
                f.flush() # Force flush every FORCE_TO_FLUSH_INTERVAL seconds
                self.last_flush_time = time.time()
    def flush(self) :
        for f in self.files:
            f.flush()
class RedirectorB:
    def __init__(self,log_file_prefix: str,
                 dir_:str=root_config.logPath,
                 redirect_stderr=True,also_to_screen=True):
        logFilePath = _get_logFilePath(dir_, log_file_prefix)
        if also_to_screen:
            f=open(logFilePath,'w')
            sys.stdout = Tee(sys.stdout,f)
            if redirect_stderr:
                sys.stderr = Tee(sys.stderr,f)
        else:
            tee = Tee( logFilePath)
            sys.stdout=tee
            if redirect_stderr:
                sys.stderr = tee

if __name__=='__main__':
    _=RedirectorB('./tmp/','ttt435',
                  redirect_stderr=1,
                  also_to_screen=1)
    print('aaaaaaaabbbb')
    print('h4t93qht0')
    exit(0)




class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


class HiddenSpecified_OutAndErr:
    class FilterOut(object):# from https://stackoverflow.com/questions/34904946/how-to-filter-stdout-in-python-logging
        def __init__(self, stream, l__filter_the_line_that_contains):
            self.stream = stream
            self.l__filter_the_line_that_contains = l__filter_the_line_that_contains
            # self.pattern = re.compile(re_pattern) if isinstance(re_pattern, str) else re_pattern
        def __getattr__(self, attr_name):
            return getattr(self.stream, attr_name)
        def write(self, data):
            for string in self.l__filter_the_line_that_contains:
                if  string in data:
                    return
            self.stream.write(data)
            # self.stream.flush()
        def flush(self):
            self.stream.flush()
    def __init__(self,  l__filter_the_line_that_contains ):
        self.l__filter_the_line_that_contains = l__filter_the_line_that_contains
    def __enter__(self):
        self._original_stdout =sys.stdout
        self._original_stderr =sys.stderr
        sys.stdout = self.FilterOut(sys.stdout , self.l__filter_the_line_that_contains)
        sys.stderr = self.FilterOut(sys.stderr , self.l__filter_the_line_that_contains)

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr

        