
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '2'
# import redirect_util,root_config
# _ = redirect_util.RedirectorB(  
#     log_file_prefix='',
#     redirect_stderr=1,
#     also_to_screen=1,
# )
import functools
import root_config
from imports import *
import json
import math
import os,cv2
import os.path as osp
import numpy as np
from vis.vis_rel_pose import *
import torch
from skimage.io import imsave, imread
# from vis.vis_rel_pose import vis_w2cRelPoses,vis_w2cPoses
from gen6d.Gen6D.pipeline import Estimator4Co3dEval
from pose_util import *
import pandas as pd
import os, shutil, sys, json
# from miscellaneous.EvalResult import EvalResult
from infer_pair import gen6d_imgPaths2relativeRt_B
from pose_util import R_t_2_pose
from collections import namedtuple
from PIL import Image
from image_util import *
from dataclasses import dataclass
from import_util import import_relposepp_evaluate_pairwise
# evaluate_pairwise=import_relposepp_evaluate_pairwise()



# ImageData=namedtuple('ImageData',['path','K','bbox'])
@dataclass
class ImageData:
    path:str
    K:np.ndarray=None
    bbox:list=None
    
def infer_pairs(
        q0ImageData:ImageData,
        l_q1ImageData:list[ImageData],
        refId:str,  # will determine where to put refDatabase folder
        #
        auto_K=False,
        auto_bbox=False,
        #
        vis_result_folder=None,
        #
        **kw,
):
    """
    q0 ie 'reference image' in paper
    q1 ie 'query image' in paper

    ret pytorch3d
    """
    model_name='E2VG'
    #-------------------------------------------------
    @functools.cache
    def path_2_imgArr(path):
        path=str(path)
        ret=imread(path)
        return ret
    all_imageData:list[ImageData]=[q0ImageData]+l_q1ImageData
    for i,imageData in enumerate(all_imageData):
        imageData:ImageData
        path,K,bbox=        imageData.path,imageData.K,imageData.bbox
        path=Path(path)
        if K is None:
            if auto_K:
                img = Image.open(path)
                image_size = img.size  # w,h
                img.close()
                f = np.sqrt(image_size[0] ** 2 + image_size[1] ** 2)
                fx, fy = f, f
                all_imageData[i].K=np.array([[fx, 0, image_size[0]/2],
                           [0, fy, image_size[1]/2],
                           [0, 0, 1]])
            else:
                assert 0
        if bbox is None:
            if auto_bbox:
                bbox = imgArr_2_objXminYminXmaxYmax(
                    path_2_imgArr(path),
                    bg_color=(255, 255, 255),
                    THRES=5,
                )
                all_imageData[i].bbox=bbox
            else:
                assert 0
        x0,y0,x1,y1=bbox
        assert x0<x1
        assert y0<y1
        if all([isinstance(i,int) for i in bbox]):
            pass
        else:
            assert all(  [isinstance(i,float) for i in bbox]  )
            assert all(  [i>=0 and i<=1 for i in bbox]  )
            img = Image.open(path)
            assert img.mode=='RGB',f"{img.mode=}"
            w,h = img.size
            bbox=[int(x0*w),int(y0*h),int(x1*w),int(y1*h),]#TODOlp check in boundary 
            all_imageData[i].bbox=bbox
        if 1:
            
            imgArr=path_2_imgArr(path)
            imgArr=draw_bbox(imgArr,bbox,bbox_type='x0y0x1y1')
            debug_imsave(Path('infer_pairs')/f'{refId}'/path.name,imgArr)
    # return
    # if (model_name == "relpose++"):
    #     from baseline.relpose_plus_plus_main.relpose.models.util import get_relposepp_model_Wrap
    #     from Dataset.custom import CustomDataset
    #     #----------dataset
    #     #image_paths;bboxes
    #     bboxes=[]
    #     image_paths=[]
    #     for i, imageData in enumerate(all_imageData):
    #         image_paths.append(imageData.path)
    #         bboxes.append(imageData.bbox)
    #     #
    #     relposepp_customDataset = CustomDataset(
    #         image_paths=image_paths,
    #         mask_dir=None,
    #         bboxes=bboxes,
    #         mask_images=False,
    #     )
    #     #ckpt
    #     checkpoint_path = '/sharedata/home/suncaiyi/space/cv/baseline/relpose_plus_plus_main/weights/relposepp'
    #     device = root_config.DEVICE if torch.cuda.is_available() else "cpu"
    #     relposepp_model, _ = get_relposepp_model_Wrap(model_dir=checkpoint_path, device=device)
    model_estimator_instance = None
    image0_path=q0ImageData.path
    l_R_T_pose=[]
    for j in range(len(l_q1ImageData)):
        q1ImageData: ImageData=l_q1ImageData[j]
        image1_path=q1ImageData.path
        print(f"\n--------evaluating--{j}/{len(l_q1ImageData)}--------------")
        """
        # if q0ImageData.K!=q1ImageData.K:
        if np.any(q0ImageData.K!=q1ImageData.K):
            if 'do_not_check_K_equivalent' in kw and kw['do_not_check_K_equivalent']:
                print('q0ImageData.K!=q1ImageData.K.   But: do_not_check_K_equivalent')
            else:
                raise NotImplementedError
        """
        print("ref img", image0_path)
        print("query img", image1_path)
        if (model_name == "E2VG"):
            if model_estimator_instance is None:
                model_estimator_instance = Estimator4Co3dEval(
                    refIdWhenNormal=refId,
                )
            Global.intermediate["E2VG"]["inter_img"].l = []
            R_pred_rel, T31_pred_rel, relative_pose, inter = gen6d_imgPaths2relativeRt_B(
                estimator=model_estimator_instance,
                K0=q0ImageData.K,
                K1=q1ImageData.K,
                image0_path=image0_path,
                image1_path=image1_path,
                bbox0=q0ImageData.bbox,
                bbox1=q1ImageData.bbox,
                input_image_eleRadian=None,
            )
            gen6d_pose0_w2opencv_leftMul = inter["pose0"]
            gen6d_pose0_w2pytorch3d_leftMul = opencv_2_pytorch3d__leftMulW2cpose(gen6d_pose0_w2opencv_leftMul)
            gen6d_pose1_w2opencv_leftMul = inter["pose1"]
            gen6d_pose1_w2pytorch3d_leftMul = opencv_2_pytorch3d__leftMulW2cpose(gen6d_pose1_w2opencv_leftMul)
            relative_pose = opencv_2_pytorch3d__leftMulRelPose(relative_pose)
            R_pred_rel = relative_pose[:3, :3]
            T31_pred_rel = relative_pose[:3, 3:]
            if vis_result_folder:
                assert model_name=='E2VG'
                vis_result_folder=Path(vis_result_folder)
                assert vis_result_folder.exists()
                def vis_pose():
                    title = f"j={j} image0_path={image0_path} image1_path={image1_path}"
                    w2c = R_t_2_pose(R=R_pred_rel, t=[0, 0, 0])
                    l_w2c = [w2c @ gen6d_pose0_w2pytorch3d_leftMul, ]
                    l_color = ['b', ]
                    tmp_zero123Input_info = model_estimator_instance.sparseViewPoseEstimator.ref_database.get_zero123Input_info()
                    tmp_eleRadian, zero123_input_img_path = tmp_zero123Input_info["elevRadian"], \
                        tmp_zero123Input_info["img_path__in_ref"]
                    l_w2c.append(opencv_2_pytorch3d__leftMulW2cpose(
                        # eleRadian_2_base_w2c(tmp_eleRadian)
                        np.array(tmp_zero123Input_info["pose"])  # the same as expression above
                    ))
                    l_color.append("white")
                    l_w2c.append(gen6d_pose0_w2pytorch3d_leftMul)
                    l_color.append("grey")
                    title += f".elev={tmp_eleRadian * 180 / math.pi:.2f}°\nzero123 base img={zero123_input_img_path}"  # azim=-60, elev=30 by default
                    for w2c, color in zip(l_w2c, l_color):
                        Global.poseVisualizer1.append(
                            w2c,
                            color=color,
                        )
                    param = dict(
                        l_w2c=l_w2c,
                        l_color=l_color,
                        do_not_show_base=1,
                    )
                    view0 = vis_w2cPoses(**param, title=title)
                    view1 = vis_w2cPoses(**param, no_margin=1, kw_view_init=dict(elev=30, azim=60))
                    view2 = vis_w2cPoses(**param, no_margin=1, kw_view_init=dict(elev=15, azim=180))
                    view3 = vis_w2cPoses(**param, no_margin=1, kw_view_init=dict(elev=45, azim=240))
                    # pose_save_path_format=osp.join(root_config.evalResultPath_co3d, f"[{model_name}-{root_config.idSuffix}]{category}-{sequence_name}-{key_frames[0]},{key_frames[1]} {{}}.npy")
                    zero123_input_img = imread(zero123_input_img_path)
                    ttt_w = 30
                    ttt_w2 = view0.shape[1] - (view1.shape[1] + view2.shape[1] + view3.shape[1])
                    ttt_w = max(ttt_w, ttt_w2)
                    ttt_h = ttt_w * zero123_input_img.shape[0] // zero123_input_img.shape[1]
                    zero123_input_img = cv2.resize(zero123_input_img, (ttt_w, ttt_h))
                    ret = cv2_util.concat_images_list(
                        view0,
                        cv2_util.concat_images_list(
                            view1, view2, view3,
                            zero123_input_img,
                            vert=0
                        ),
                        vert=1,
                    )
                    return ret
                vis_w2cRelPoses_img = vis_pose()
                def tmpGet__vis_w2cRelPoses_img_containing_poseWhenRefine():
                    param = dict(
                        do_not_show_base=1,
                        title=f"",
                        no_margin=1,
                    )
                    view0 = Global.poseVisualizer1.get_img(**param)
                    view1 = Global.poseVisualizer1.get_img(**param, kw_view_init=dict(elev=30, azim=60))
                    view2 = Global.poseVisualizer1.get_img(**param, kw_view_init=dict(elev=15, azim=180))
                    view3 = Global.poseVisualizer1.get_img(**param, kw_view_init=dict(elev=45, azim=240))
                    vis_w2cRelPoses_img_containing_poseWhenRefine = cv2_util.concat_images_list(view0, view1,
                                                                                                view2, view3,
                                                                                                vert=0)
                    return vis_w2cRelPoses_img_containing_poseWhenRefine

                if root_config.one_SEQ_mul_Q0__one_Q0_mul_Q1:
                    vis_img = cv2_util.concat_images_list(
                        cv2_util.concat_images_list(
                            imread(image0_path),
                        ),
                        cv2_util.concat_images_list(
                            imread(image1_path),
                            Global.intermediate["E2VG"]["inter_img"].l[0],
                            vert=0,
                            max_h=Global.intermediate["E2VG"]["inter_img"].l[0].shape[0]
                        ),
                        vis_w2cRelPoses_img,
                        tmpGet__vis_w2cRelPoses_img_containing_poseWhenRefine(),
                        vert=True
                    )
                else:
                    vis_img = cv2_util.concat_images_list(
                        cv2_util.concat_images_list(
                            imread(image0_path),
                            Global.intermediate["E2VG"]["inter_img"].l[0],
                            vert=0,
                            max_h=Global.intermediate["E2VG"]["inter_img"].l[0].shape[0]
                        ),
                        cv2_util.concat_images_list(
                            imread(image1_path),
                            Global.intermediate["E2VG"]["inter_img"].l[1],
                            vert=0,
                            max_h=Global.intermediate["E2VG"]["inter_img"].l[1].shape[0]
                        ),
                        vis_w2cRelPoses_img,
                        tmpGet__vis_w2cRelPoses_img_containing_poseWhenRefine(),
                        vert=True
                    )
                cv2_util.putText(
                    vis_img, Global.poseVisualizer1.get_pose_str_A(),
                    (vis_img.shape[1] - 1000, vis_img.shape[0] * 3 // 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (180, 180, 180),
                )
                tmp_save_path = vis_result_folder/f'{refId}--{Path(image0_path).name}--{Path(image1_path).name}.jpg'
                imsave(tmp_save_path, vis_img)
                print(f"visual result: {tmp_save_path}")
                Global.poseVisualizer1.clear()
                """#-------------------------------
                vis1.run(l_R, l_color, l_note, l_T=l_T, elev_degree=35)"""

        l_R_T_pose. append ((R_pred_rel, T31_pred_rel, relative_pose))
    return l_R_T_pose




def infer_pairs_wrapper(
    referenceImage_path_bbox,queryImages_path_bbox,
    refId,cameraConvention='opencv'
    ):
    """
    :param referenceImage_path_bbox: 
        (path of the reference image,  bbox of object in the reference image)
    :param queryImages_path_bbox: 
        queryImages_path_bbox=[
            (path of query image 1,        bbox of object in this image),
            (path of query image 2,        bbox),
            (path of query image 3,        bbox),
            ...
        ]
        You should provide at least one query image
    :param refId: 
        refId indentify the building result of a reference image. 
        If {refId} has been built before, then the program will reuse the building result to save time (building from a reference image takes >1min on a single 3090 GPU)
    :param cameraConvention:
        'opencv' or 'pytorch3d'
    :return: 
        relativePoses=[
            relative pose from reference image to query image 1,
            relative pose from reference image to query image 2,
            relative pose from reference image to query image 3,
            ...
        ]
        X-query_i = relativePoses[i] @ X-reference. X means point in the coordinate system of camera. 
        The camera follows {cameraConvention} convention
    """
 
    root_config.VIS=1 #  visualize result;  to save time, you can let it be 0
    root_config.NO_CARVEKIT=0   #  CARVEKIT is a bg remover. if input img is masked, then no need to remove bg;else enable CARVEKIT, meaning let NO_CARVEKIT=0
    
    
    l_path_bbox=[referenceImage_path_bbox]+queryImages_path_bbox
    l_path_bbox=[(str(path),bbox) for path,bbox in l_path_bbox]
    vis_result_folder=Path(root_config.evalVisPath)/'_custom_data'/refId
    os.makedirs(vis_result_folder, exist_ok=True)
    l__R_T_pose= infer_pairs(
        q0ImageData=ImageData(path=l_path_bbox[0][0], K=None, bbox=l_path_bbox[0][1]),
        l_q1ImageData=[ImageData(path=path, K=None, bbox=bbox) for path,bbox in l_path_bbox[1:]],
        refId=refId,
        #
        auto_K=True,
        auto_bbox=False,
        #
        vis_result_folder=vis_result_folder,
    )
    ret=[]
    for R,t,pose_4x4_pytorch3dConvention in l__R_T_pose:
        if cameraConvention=='opencv':
            pose_4x4=pytorch3d_2_opencv__leftMulRelPose(pose_4x4_pytorch3dConvention)
        else:
            pose_4x4=pose_4x4_pytorch3dConvention
        ret.append(pose_4x4)
    return ret
        
if __name__ == "__main__":
    
    root_config.NO_TRY = 1
    root_config.NUM_REF=128
    
    root_config.SAMPLE_BATCH_SIZE = 32
    root_config.SAMPLE_BATCH_B_SIZE = 4
    root_config.DATASET = "navi"
    root_config.FORCE_zero123_render_even_img_exist = 0
    root_config.SKIP_EVAL_SEQ_IF_EVAL_RESULT_EXIST = 1
    root_config.SKIP_GEN_REF_IF_REF_FOLDER_EXIST=1    
    root_config.CONF_one_SEQ_mul_Q0__one_Q0_mul_Q1.ONLY_CHECK_BASENAME=1
    root_config.MAX_PAIRS=20 
    root_config.USE_CONFIDENCE=0
    root_config.CONSIDER_IPR=0
    root_config.Q0Sipr=0
    root_config.Q0Sipr_range=45
    # root_config.Q0Sipr_range=180
    root_config.tmp_batch_image__SUFFIX=".png"
    root_config.LOAD_BY_IPC=1
    # root_config.SHARE_tmp_batch_images='_val'
    
    
    
    
    
    
    # root_config.USE_CONFIDENCE=True
    def _f(refId,l_path_bbox):
        vis_result_folder=Path(root_config.evalVisPath)/'_infer_custom'/refId
        os.makedirs(vis_result_folder, exist_ok=True)
        infer_pairs(
            q0ImageData=ImageData(path=l_path_bbox[0][0], K=None, bbox=l_path_bbox[0][1]),
            l_q1ImageData=[ImageData(path=path, K=None, bbox=bbox) for path,bbox in l_path_bbox[1:]],
            refId=refId,
            #
            auto_K=True,
            auto_bbox=False,
            #
            vis_result_folder=vis_result_folder,
        )
    

    
    
    refId='cup'
    l_path_bbox=[
        ('/sharedata/home/suncaiyi/space/cv/custom_data/cup/0.jpg',( 115,685,633,1265  ),),
        ('/sharedata/home/suncaiyi/space/cv/custom_data/cup/1.jpg',(  1058,520,1653,1146 ),),
        ('/sharedata/home/suncaiyi/space/cv/custom_data/cup/2.jpg',(  41,411,745,1080 ),),
    ]
    _f(refId,l_path_bbox) 
    
    
    
    
    
    refId='toy_pig'
    l_path_bbox=[
        ('/sharedata/home/suncaiyi/space/cv/custom_data/toy_pig/0.jpg',(   0.6450381679389313,0.4557477110885046,
                                                                           0.9389312977099237,0.9440488301119023,          ),),
        ('/sharedata/home/suncaiyi/space/cv/custom_data/toy_pig/1.jpg',(   0.6229007633587786,0.5513733468972533,
                                                                           0.9015267175572519,0.8748728382502543,          ),),
        ('/sharedata/home/suncaiyi/space/cv/custom_data/toy_pig/2.jpg',(   0.030534351145038167,0.5178026449643948,
                                                                           0.37251908396946565,0.9888097660223805,          ),),
    ]
    _f(refId,l_path_bbox) 
    
    
    
    
