# from rich import print
import rich
__primitive_print=print
def print(*args,
          use_primitive=1,
          **kw):
    if use_primitive:
        return __primitive_print(*args,**kw)
    else:
        
        return rich.print(*[arg.replace('[',r'\[') if isinstance(arg,str) else arg for arg in args ],**kw, )
from redirect_util import *
# ------------------------------------------------------
EXP_fp_and_lineNo="f'{os.path.abspath(__file__)}:{inspect.currentframe().f_lineno}'"
EXP_print_fp_and_lineNo="eval(f'{os.path.abspath(__file__)}:{inspect.currentframe().f_lineno}')"
#------------------ log_Util ---------------------------- V2022CFG230119
from logging import debug as DEBUG
ddd=DEBUG
from logging import info as INFO
from logging import warning as WARNING
from logging import error as ERROR
def pDEBUG(*args):
    DEBUG(" ".join([str(arg) for arg in args]))
def pINFO(*args):
    INFO(" ".join([str(arg) for arg in args]))
def pWARNING(*args):
    WARNING(" ".join([str(arg) for arg in args]))
def pERROR(*args):
    ERROR(" ".join([str(arg) for arg in args]))
import logging
## create logger with 'spam_application'
# logger = logging.getLogger("My_app")
logger = logging.root
class _CustomFormatter(logging.Formatter):
    blue = "\x1b[1;34m"
    light_blue = "\x1b[1;36m"
    purple = "\x1b[1;35m"
    normal="\x1b[0;20m"
    white="\x1b[37;20m"
    cyan= "\x1b[36;20m"#
    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    green = "\x1b[32;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    white_on_red_bg="\x1b[41;20m"
    reset = "\x1b[0m"
    '''
    asctime	%(asctime)s	将日志的时间构造成可读的形式，默认情况下是精确到毫秒，如 2018-10-13 23:24:57,832，可以额外指定 datefmt 参数来指定该变量的格式
    name	%(name)	日志对象的名称
    filename	%(filename)s	不包含路径的文件名
    pathname	%(pathname)s	包含路径的文件名
    funcName	%(funcName)s	日志记录所在的函数名
    levelname	%(levelname)s	日志的级别名称
    message	%(message)s	具体的日志信息
    lineno	%(lineno)d	日志记录所在的行号
    pathname	%(pathname)s	完整路径
    process	%(process)d	当前进程ID
    processName	%(processName)s	当前进程名称
    thread	%(thread)d	当前线程ID
    threadName	%threadName)s	当前线程名称
    '''
    
    
    # format = "%(asctime)s%(filename)s-%(lineno)d-%(funcName)s [%(levelname)s] %(message)s"
    format = "%(asctime)s %(filename)s:%(lineno)d %(funcName)s [%(levelname)s] %(message)s"

    FORMATS = {
        logging.DEBUG: cyan + format + reset,
        # logging.INFO: normal + format + reset,
        logging.INFO: white + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: white_on_red_bg + format + reset,
        logging.CRITICAL: white_on_red_bg + format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(fmt=log_fmt,datefmt="%H:%M:%S")
        return formatter.format(record)




def _configure_logging(level=logging.DEBUG):
    print("_configure_logging")
    # logging.basicConfig(
    
    
    
    
    
    # logging.basicConfig(datefmt='%M:%S')
    logger.setLevel(level)


    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    ch.setFormatter(_CustomFormatter())

    logger.addHandler(ch)
    print("_configure_logging over")

_configure_logging(level=logging.INFO)

if(__name__=="__main__"):
    INFO("INFO")
    ERROR("ERROR")
    DEBUG("DEBUG")
    WARNING("WARNING")