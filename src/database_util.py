"""
没放dataset_util里是因为4paper里很多本地运行的.py要导入dataset_util,但本地导入Database很多依赖缺失
"""
import functools
#-----------------------Database--------------------------------------------
from Dataset.gso import GsoDatabase
from Dataset.co3dv2 import Co3dv2Database
from Dataset.idpose import IdposeOmniDatabase
from Dataset.omni import OMNIDatabase
from Dataset.navi import NaviDatabase
def datasetName_cate_seq__2__database(datasetName,cate, seq):
    if datasetName == "co3d":
        return Co3dv2Database(cate, seq)
    elif datasetName == "idposeOmni":
        return IdposeOmniDatabase(cate)
    elif datasetName == 'gso':
        return GsoDatabase(cate)
    elif datasetName == 'omni':
        return OMNIDatabase(cate)
    elif datasetName == 'navi':
        return NaviDatabase(cate)
    else:
        raise NotImplementedError
@functools.cache
def datasetName_cate_seq__2__database__cached(*args,**kw):
    return datasetName_cate_seq__2__database(*args,**kw)