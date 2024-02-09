
        
import os,sys
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from imports import *
from infer_pair import *
import sys
import gen6d.Gen6D.pipeline as pipeline
from miscellaneous.EvalResult import EvalResult
from evaluate.eval_on_an_obj import eval_on_an_obj





#----------------------------------------------

def run(l_datasetName,model_name ="E2VG"):
    l__datasetName_cate_seq_Q0INDEX=get__l__datasetName_cate_seq_Q0INDEX(datasetNames=l_datasetName,datasetName_2_s=MyTestset.datasetName_2_s)
    #
    SUFFIX=""
    for datasetName,cate,seq, q0 in l__datasetName_cate_seq_Q0INDEX:
        if datasetName=='navi':
            root_config.tmp_batch_image__SUFFIX='.jpg' #save disk space
        assert seq==""# only co3d has the seq level
        root_config.DATASET = datasetName
        for Q0INDEX in [q0]:
            root_config.Q0INDEX = Q0INDEX
            root_config.refIdSuffix = f"+{Q0INDEX}"
            if root_config.Q0Sipr:
                SUFFIX='-rotated'
            root_config.idSuffix = f"(testset{SUFFIX}){root_config.SEED}+{Q0INDEX}"
            root_config.refIdSuffix += SUFFIX
            eval_on_an_obj(
                category=cate,
                model_name=model_name,
                vis_include_theOther=0,
            )
    EvalResult.AllAcc.dump_average_acc(SUFFIX )
