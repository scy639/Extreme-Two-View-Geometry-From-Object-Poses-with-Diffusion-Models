import functools
#-----------------------Database--------------------------------------------
from Dataset.gso import GsoDatabase
from Dataset.navi import NaviDatabase
def datasetName_cate_seq__2__database(datasetName,cate, seq):
    if datasetName == 'gso':
        return GsoDatabase(cate)
    elif datasetName == 'navi':
        return NaviDatabase(cate)
    else:
        raise NotImplementedError
@functools.cache
def datasetName_cate_seq__2__database__cached(*args,**kw):
    return datasetName_cate_seq__2__database(*args,**kw)