
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
