import torch.cuda

#-------
# DATASET="gso"
# DATASET="navi"
DATASET=None
#-------
SAMPLE_BATCH_SIZE=16
FORCE_zero123_render_even_img_exist=0
SAMPLE_BATCH_B_SIZE=4
#-------gen6d
NUM_REF=64
#-------
SKIP_EVAL_SEQ_IF_EVAL_RESULT_EXIST=1
MAX_PAIRS=20
CONSIDER_IPR=False  # in-plane rotation of q0
Q0Sipr:bool=False
Q0Sipr_range:int=45
Q1Sipr:bool=False
Q1Sipr_range:int=45
#-------
SEED=0    
Q0INDEX:int=None

idSuffix=f"{SEED}+{Q0INDEX}"
refIdSuffix=f"+{Q0INDEX}"
# refIdSuffix=f"{SEED}"
# refIdSuffix=f"{SEED}+"
#-------misc
tmp_batch_image__SUFFIX='.png' 
SHARE_tmp_batch_images=False #False or folder name  
NO_CARVEKIT:bool=True #CARVEKIT is a bg remover. if input img is masked, then no need to remove bg
VIS=True ##  visualize result;  to save time, you can let it be 0.     VIS:  bool/int(0 or 1)  /fp(0.0-1.0,vis ratio).  
#-------

#-------path
import os
path_root=os.path.dirname(os.path.abspath(__file__)) #path of src
path_4debug=os.path.join(path_root,"4debug")
projPath_gen6d=os.path.join(path_root,"gen6d/Gen6D")
projPath_zero123=os.path.join(path_root,"zero123/zero1")
dataPath_zero123=os.path.join(path_root,"zero123/zero1/output_im")
dataPath_gen6d=os.path.join(path_root,projPath_gen6d,"data/zero123")
from path_configuration import *
weightPath_zero123=os.path.join(path_root,"../weight/105000.ckpt")
weightPath_gen6d='../weight/weight_gen6d'
weightPath_selector=os.path.join(  weightPath_gen6d ,"selector_pretrain/model_best.pth")
weightPath_refiner=os.path.join(  weightPath_gen6d ,"refiner_pretrain/model_best.pth")
weightPath_loftr=os.path.join(path_root  ,"../weight/indoor_ds_new.ckpt")
evalResultPath_co3d=os.path.join(path_root,"result/eval_result")
evalVisPath=os.path.join(path_root,"result/visual")
logPath=os.path.join(path_root,"log") 
os.makedirs(path_4debug,exist_ok=True)
os.makedirs(logPath,exist_ok=True)
os.makedirs(evalResultPath_co3d,exist_ok=True)
os.makedirs(evalVisPath,exist_ok=True)


















class RefIdWhenNormal:
    @staticmethod
    def get_id(cate,seq,refIdSuffix_):
        return f"{cate}--{seq}--{refIdSuffix_}"
    
    
#-----------------------unused conf--------------------
#-------debug
NO_TRY=0
class ForDebug:
    class forceIPRtoBe:
        enable:bool=False
        IPR:int=None 
class Cheat:
    force_elev=None#
#-------GPU
GPU_INDEX=0
DEVICE=f"cuda:{GPU_INDEX}" if torch.cuda.is_available() else "cpu"
#-------zero123 l_xyz
ZERO123_MULTI_INPUT_IMAGE = 0
_Z = 0
#-------zero123 NUM_SAMPLE
USE_ALL_SAMPLE=0
NUM_SAMPLE=1
#
USE_white_bg_Detector=1
Q0_MIN_ELEV=0 
ELEV_RANGE=None   # if not None: in degree. lower and upper bound of absolute elev. eg. (-0.1,40)
USE_CONFIDENCE=0
REFINE_ITER:int=3
#-------check,geometry
LOOK_AT_CROP_OUTSIDE_GEN6D=1
# 1:call gen6d_imgPaths2relativeRt_B(where perspective trans is performed); 0:give detection_outputs to gen6d so that perspective trans in refiner
IGNORE_EXCEPTION=    0     
ONLY_GEN_DO_NOT_MEASURE=0
LOG_WHEN_SAMPLING=0
one_SEQ_mul_Q0__one_Q0_mul_Q1=1
class CONF_one_SEQ_mul_Q0__one_Q0_mul_Q1:
    ONLY_CHECK_BASENAME=False
FOR_PAPER=False
LOAD_BY_IPC=False  
MARGIN_in_LOOK_AT=0.05
#-------4 ablation
MASK_ABLATION=None # None/'EROSION'/'DILATION'
ABLATE_REFINE_ITER:int=None# None or int (when int, it can be 0, so must use 'if root_config.ABLATE_REFINE_ITER is (not) None:' instead of 'if (not) root_config.ABLATE_REFINE_ITER:')
#-------4 val
VALing=0
SKIP_GEN_REF_IF_REF_FOLDER_EXIST=False