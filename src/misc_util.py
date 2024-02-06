
import os,time,root_config
import numpy as np
from pathlib import Path
import pprint
class ch_cwd_to_this_file:
    def __init__(self, _code_file_path):
        self._code_file_path = _code_file_path
    def __enter__(self):
        self._old_dir = os.getcwd()
        cwd=os.path.dirname(os.path.abspath(self._code_file_path))
        os.chdir(cwd)
    def __exit__(self, exc_type, exc_val, exc_tb):
        os.chdir(self._old_dir)
# def img_2_img_full_path(img,format='jpg',original_name_or_path=''):
#     """
#     thread safe
#     """
#     assert isinstance(img,np.ndarray)
#     assert img.shape[2]==3 or img.shape[2]==4
#     original_img_name_without_dir=os.path.basename(original_name_or_path)
#     full_path = os.path.join(root_config.path_root, f'./tmp_images/[{root_config.DATASET}][{tmp_cate_or_obj}][{sequence_name}]{img_name_without_suffix}.jpg')
#     if not os.path.exists(os.path.dirname(full_path)):
#         os.makedirs(os.path.dirname(full_path))
#     print("get_data path:", full_path)
#     img.save(full_path)
#     return img_full_path

import datetime
import pytz
def your_datetime()->datetime.datetime:
    """
    """
    
    local_tz = datetime.datetime.now(datetime.timezone.utc).astimezone().tzinfo
    
    
    now = datetime.datetime.now()
    
    local_time = now.astimezone(local_tz)
    
    return local_time
def get_datetime_str( os_is_windows=False)->str:

    ret= f"{your_datetime():%m.%d-%H:%M:%S}"
    if os_is_windows:
        ret=ret.replace(':','-')
    return ret






import json
import numpy
from torch import Tensor


def to_list_to_primitive(obj):
    if isinstance(obj, numpy.ndarray):
        return obj.tolist()
    if isinstance(obj, Tensor):
        return obj.cpu().data.numpy().tolist()
    if isinstance(obj, list):
        return [to_list_to_primitive(i) for i in obj]
    # if isinstance(obj, DataFrame):
    #     return obj.values.tolist()
    elif (isinstance(obj, numpy.int32) or
          isinstance(obj, numpy.int64) or
          isinstance(obj, numpy.float32) or
          isinstance(obj, numpy.float64)):
        return obj.item()
    elif (isinstance(obj, int) or
          isinstance(obj, float)
          ):
        return obj
    else:
        assert 0


class MyJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()
        if isinstance(obj, Tensor):
            return obj.cpu().data.numpy().tolist()
        elif (isinstance(obj, numpy.int32) or
              isinstance(obj, numpy.int64) or
              isinstance(obj, numpy.float32) or
              isinstance(obj, numpy.float64)):
            return obj.item()
        return json.JSONEncoder.default(self, obj)

def truncate_str(string:str,MAX_LEN:int,suffix_if_truncate="......")->str:
    assert isinstance(string,str)
    if len(string)>  MAX_LEN:
        string=string[:MAX_LEN]+suffix_if_truncate
    return string
def map_string_to_int(string,MIN,MAX):
    """
    """
    assert isinstance(MIN,int)
    assert isinstance(MAX,int)
    assert MAX-MIN>=2
    
    sum = 0
    for char in string:
        sum += ord(char)
    # print("sum", sum)
    ret=2**sum
    ret += sum 
    ret=ret%(MAX-MIN)
    ret+=MIN
    return ret


def print_optimizer(optimizer):
    state_dict=optimizer.state_dict()
    param_groups=state_dict['param_groups']
    # for i,param_group in enumerate(param_groups):
    pprint.pprint(param_groups)
    
