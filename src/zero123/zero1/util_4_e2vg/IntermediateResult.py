import numpy as np
import json, math


class IntermediateResult:
    def __init__(s, ):
        s.data = {}

    def append(s, i, K, pose):
        s.data[i] = {
            "K": K,
            "pose": pose
        }

    def load(s, path):
        with open(path, "r") as f:
            s.data = json.load(f)
        
        for i in s.data:
            for key in s.data[i]:
                s.data[i][key] = np.array(s.data[i][key])

    def dump(self, path):
        
        import json
        import numpy
        from torch import Tensor

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

        with    open(path, "w") as f:
            json.dump(self.data, f, cls=MyJSONEncoder)
