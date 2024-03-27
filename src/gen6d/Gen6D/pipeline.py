import sys,os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from elev_util import eleRadian_2_baseXyz_lXyz,imgPath2elevRadian,eleRadian_2_base_w2c
from oee.utils.elev_est_api import imgPath2IPRbyOEE
from imports import *
from misc_util import *
from .scy import Config
from .utils.draw_utils import pts_range_to_bbox_pts, draw_bbox_3d, draw_bbox_3d_dpt
import argparse
import subprocess
from pathlib import Path
import os
import numpy as np
from skimage.io import imsave, imread
import json
from .dataset.database import parse_database_name, get_ref_point_cloud
from .estimator import name2estimator, Gen6DEstimator
from .eval import visualize_intermediate_results
from .utils.base_utils import load_cfg, project_points
from .scy.DebugUtil import *
from .scy.gen6dGlobal import gen6dGlobal
from .scy.IntermediateResult import IntermediateResult
from .scy.MyJSONEncoder import MyJSONEncoder

# if(not Config.LOCAL):
#     from .scy import ElevationUtil


def import_run4gen6d():
    # global zero1_PATH, run4gen6d
    global zero1_PATH

    import sys


            
    zero1_PATH = root_config.projPath_zero123
    import sys
    sys.path.append(zero1_PATH)
    # print("sys.path=", sys.path)
    # sys.path.insert(0, zero1_PATH)
    import run4gen6d
    sys.path.pop()
    # sys.path.pop(0)


    return run4gen6d

class run4gen6d_Getter:
    core=None
    @staticmethod
    def get():
        if run4gen6d_Getter.core:
            pass
        else:
            run4gen6d_Getter.core=import_run4gen6d()
        return run4gen6d_Getter.core


# if(not Config.LOCAL):
#     run4gen6d = import_run4gen6d()  # ?rm run4gen6d


def run4gen6d_main(
        id_,
        input_image_path,
        output_dir,
        num_samples,
        l_xyz,
        base_xyz,
        K,
        ddim_steps=50,
        **kw,
):
    run4gen6d=run4gen6d_Getter.get()
    # cwd
    cwd = os.getcwd()
    os.chdir(zero1_PATH)
    ret=run4gen6d.main(id_, input_image_path,
                   output_dir,
                   num_samples=num_samples,
                   l_xyz=l_xyz,
                   base_xyz=base_xyz,
                   ddim_steps=ddim_steps,
                   K="弃用。从pipeline __run_zero123 一路传进来的这个K应该是zero123生成图的K（crop修改后，最终被用作gen6d里ref database的K(被写进intermediateResult.json然后在gen6d ref database里被读取)）",
                   **kw
                   )
    os.chdir(cwd)
    return ret
def run4gen6d__sample_model_batchB_wrapper(
        *args,
        **kw,
):
    run4gen6d=run4gen6d_Getter.get()
    # cwd
    cwd = os.getcwd()
    os.chdir(zero1_PATH)
    ret = run4gen6d.sample_model_batchB_wrapper(
        *args, **kw
    )
    os.chdir(cwd)
    return ret

class SparseViewPoseEstimator():
    def __init__(s, database_name,_vis_dir ):
        s.database_name = database_name
        # cwd backup and ch
        cwd = os.getcwd()
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        """
        database_name 决定了ref_dir
            ref_dir str: f'data/{database_name}/ref'
        ref_dir content: {int}.jpg+...+intermediateResult.json
            eg: data/zero123/myCarB-do_not_crop-margin0.1/ref


        """
        if (_vis_dir and root_config.VIS!=0):
            gen6dGlobal._vis_dir = _vis_dir
            _vis_dir_Path = Path(_vis_dir)
            _vis_dir_Path.mkdir(exist_ok=True, parents=True)
            (_vis_dir_Path / 'images_out').mkdir(exist_ok=True, parents=True)
            (_vis_dir_Path / 'images_inter').mkdir(exist_ok=True, parents=True)
        # print cwd
        # print("cwd=", os.getcwd())
        database_type_and_name: str = f"zero123/{database_name}"
        ref_database = parse_database_name(database_type_and_name)
        # self
        s.estimator=None
        s.ref_database = ref_database
        s.database_type_and_name = database_type_and_name
        s.database_name = database_name
        s._vis_dir = _vis_dir
        os.chdir(cwd)  # cwd recover
    def _init_and_build_estimator(self):
        with ch_cwd_to_this_file(__file__):
            # print("[SparseViewPoseEstimator] lazy load gen6d estimator")
            cfg = load_cfg('configs/gen6d_pretrain.yaml')
            self.estimator: Gen6DEstimator = name2estimator[cfg['type']](cfg)   
            self.estimator.build(self.ref_database, split_type='all')
            print("build from reference image over")


    def estimate(
            s, K,q_img_path, pose_init=None, SCALE_DOWN=1,detection_outputs=None,
            # vis
            SHOW_GT_BBOX=False,
    ):
        if(  s.estimator is None  ):
            s._init_and_build_estimator()
        # cwd backup and ch
        cwd = os.getcwd()
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        estimator = s.estimator
        _vis_dir = s._vis_dir  

        img_path = q_img_path
        img = imread(img_path)
        # generate a pseudo K
        h, w, _ = img.shape
        
        # print("raw query img h,w=", h, w)
        h, w = int(h / SCALE_DOWN), int(w / SCALE_DOWN)
        img = cv2.resize(img, (w, h))
        f = np.sqrt(h ** 2 + w ** 2)  # 
        # 
        # K = np.asarray([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], np.float32)
        if pose_init is not None:
            raise Exception
            # we only refine one time after initialization
            estimator.cfg['refine_iter'] = 1
        if (not Config.PR_AS_INIT):  
            pose_init = None
        pose_pr, inter_results = estimator.predict(# pose_pr:3,4
            img, K,
            pose_init=pose_init,
            detection_outputs=detection_outputs,
        )

        if (_vis_dir and root_config.VIS!=0):
            # if (0):
            # print("_vis_dir", _vis_dir)
            q_img_name_without_dir = os.path.basename(img_path)
            q_img_name_without_suffix = os.path.splitext(q_img_name_without_dir)[0]
            que_id = q_img_name_without_suffix
            ref_database = s.ref_database
            object_pts = get_ref_point_cloud(ref_database)  
            object_bbox_3d = pts_range_to_bbox_pts(
                np.max(object_pts, 0), np.min(object_pts, 0))

            hist_pts = []
            if (SHOW_GT_BBOX):
                intermediate_result = IntermediateResult()
                intermediate_result.load(os.path.join(
                    'data', s.database_type_and_name, 'ref', 'intermediateResult.json'))
            pts, dpt = project_points(object_bbox_3d, pose_pr, K)
            # bbox_img = draw_bbox_3d(img, pts, (0, 0, 255))
            bbox_img = draw_bbox_3d_dpt(img, pts, dpt, (0, 0, 255))

            SHOW_GT_BBOX = 0
            if (SHOW_GT_BBOX):
                gt_img_path = os.path.join(
                    'data', s.database_type_and_name, 'ref', os.path.basename(img_path))
                if (os.path.exists(gt_img_path)):
                    i_ref = que_id
                    
                    gt_pose = intermediate_result.data[str(i_ref)]["pose"]
                    gt_K = intermediate_result.data[str(i_ref)]["K"]
                    gt_pts, gt_dpt = project_points(object_bbox_3d, gt_pose, gt_K)
                    # bbox_img = draw_bbox_3d(
                    bbox_img = draw_bbox_3d_dpt(
                        bbox_img, gt_pts, gt_dpt, (0, 255, 0), thickness=1)
                    object_center = ref_database.center
                    pose_gt = gt_pose
            else:
                object_center = None
                pose_gt = None
            # imsave(f'{str(_vis_dir)}/images_out/{que_id}-bbox.jpg', bbox_img)
            # print("i,pose_pr=", que_id, pose_pr)
            # np.save(f'{str(_vis_dir)}/images_out/{que_id}-pose.npy', pose_pr)
            inter_img = visualize_intermediate_results(img, K, inter_results, estimator.ref_info, object_bbox_3d,
                                                     object_center=object_center, pose_gt=pose_gt,  # 
                                                     )
            Global.intermediate["E2VG"]["inter_img"].append(inter_img)
            # imsave(f'{str(_vis_dir)}/images_inter/{que_id}.jpg',inter_img)

            # 
            inter_results_ = inter_results.copy()
            ref_info_ = estimator.ref_info.copy()
            if ("det_que_img" in inter_results_):
                del inter_results_['det_que_img']
            del ref_info_['imgs']
            del ref_info_['ref_imgs']
            del ref_info_['masks']
            GEN6D_INTER_RESULT_FILENAME = "gen6d_inter_result.json"
            gen6d_inter_result = {}
            gen6d_inter_result[que_id] = {
                "inter_results": inter_results_,
                "ref_info": ref_info_,
                "pose_gt": pose_gt,
                "pose_pr": pose_pr,
            }

            hist_pts.append(pts)
            args_num = 5
            args_std = 2.5


            for relative_path in [
                GEN6D_INTER_RESULT_FILENAME,
                f"images_inter/{GEN6D_INTER_RESULT_FILENAME}"
            ]:
                with open(f'{str(_vis_dir)}/{relative_path}', 'w') as f:
                    json.dump(gen6d_inter_result, f,
                              indent=4,
                              cls=MyJSONEncoder)
        # cwd recover
        os.chdir(cwd)
        pose_pr=np.concatenate([pose_pr, np.array([[0, 0, 0, 1]])], axis=0)
        return pose_pr

import  torch
class Estimator4Co3dEval():
    """
    s._ref_database_folder(".../ref/") 下目录结构：  eg./sharedata/home/suncaiyi/space/cv/gen6d/Gen6D/data/zero123/GSO_alarm----+8(elevV2)/ref/
        zero123Input/  #saved by s.__save_zero123Input_img_and_info
            info.json
            'img_path__origin'  #eg. "[gso][GSO_alarm][]008_warp.jpg"
        ref_relaToBase_info.json  
        vis/
            images_inter
            images_out
            gen6d_inter_result.json
        ref_folder_build_OVER 
        intermediateResult.json  
        '(0-127).jpg'
        ...
        
    """
    def __init__(
            s,
            refIdWhenNormal,#decide refDatabase folder
    ):
        id_ =refIdWhenNormal
        s.id_ = id_
        s._ref_database_folder=os.path.join(root_config.dataPath_gen6d, f'{id_}/ref')
        s.sparseViewPoseEstimator:SparseViewPoseEstimator = None
    def __get__zero123Input_info__ifExistElseNone(self):
        if(  not hasattr(self,'__zero123Input_info')     ):
            if (not self.sparseViewPoseEstimator  ):
                return None
            self.__zero123Input_info=self.sparseViewPoseEstimator.ref_database.get_zero123Input_info()
        return self.__zero123Input_info
    def get__zero123Input_info(self,K,image0_path):
        info=self.__get__zero123Input_info__ifExistElseNone()
        if(info is None):
            self.__build_ref_and_init_sparseViewPoseEstimator(K,image0_path)
            info = self.__get__zero123Input_info__ifExistElseNone()
        if root_config.one_SEQ_mul_Q0__one_Q0_mul_Q1:
            
            #     info["img_path__origin___beforeIPR"]=info["img_path__origin"]
            tmp1=os.path.relpath(info["img_path__origin___beforeIPR"],root_config.path_root)
            tmp2=os.path.relpath(image0_path,root_config.path_root)
            if root_config.CONF_one_SEQ_mul_Q0__one_Q0_mul_Q1.ONLY_CHECK_BASENAME:
                tmp1=os.path.basename(tmp1)
                tmp2=os.path.basename(tmp2)
            assert tmp1==tmp2,f"{tmp1} should == {tmp2}"
        return info

    def __save_zero123Input_img_and_info(
        self, img_path__origin, elevRadian,
        # IPR
        IPR_degree_counterClockwise, img_path__origin___beforeIPR,
    ):
        dir_zero123Input = os.path.join(
            self._ref_database_folder,  
            'zero123Input'
        )
        img_path__in_ref=os.path.join(dir_zero123Input,os.path.basename(img_path__origin) )
        '''
        #relToRoot
        img_path_relToRoot__origin=os.path.relpath(root_config.path_root,img_path__origin)
        img_path_relToRoot__in_ref=os.path.relpath(root_config.path_root,img_path__in_ref)
        info={ 
            "img_path_relToRoot__origin":img_path_relToRoot__origin,
            "img_path_relToRoot__in_ref":img_path_relToRoot__in_ref,
        '''
        info={# both in absolute path 
            "img_path__origin___beforeIPR":img_path__origin___beforeIPR,
            "img_path__origin":img_path__origin,
            "img_path__in_ref":img_path__in_ref,
            "elevRadian":elevRadian,
            "elevDegree":180*elevRadian/np.pi,
            "IPR_degree_counterClockwise":IPR_degree_counterClockwise,
            "pose":eleRadian_2_base_w2c(elevRadian),
        }
        os.makedirs(dir_zero123Input,exist_ok=True)
        shutil.copy(img_path__origin,dir_zero123Input)
        with open(os.path.join(dir_zero123Input,"info.json"),"w")as f:
            class MyJSONEncoder(json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, np.ndarray):
                        return obj.tolist()
                    if isinstance(obj, torch.Tensor):
                        return obj.cpu().data.numpy().tolist()
                    elif (isinstance(obj, np.int32) or
                          isinstance(obj, np.int64) or
                          isinstance(obj, np.float32) or
                          isinstance(obj, np.float64)):
                        return obj.item()
                    return json.JSONEncoder.default(self, obj)
            json.dump(info,f,cls=MyJSONEncoder)
            print("reference info=",json.dumps(info,indent=4,cls=MyJSONEncoder))
    def tmp_gen_zero123Input_info_for_old_ref_database__save_zero123Input_img_and_info(self,img_path__origin,elevRadian):
        return self.__save_zero123Input_img_and_info(img_path__origin.replace("/root/autodl-tmp/cv",root_config.path_root+"/"),elevRadian)
    def __run_zero123(s,K, input_image_path, input_image_eleRadian: float, id_):
        if (input_image_eleRadian == None):
            base_xyz, l_xyz = None, None
        else:
            base_xyz, l_xyz = eleRadian_2_baseXyz_lXyz(input_image_eleRadian)
        def save__ref_relaToBase_info(base_xyz, l_xyz ):
            def get_spherical_distance(azim_a, elev_a, azim_b, elev_b):
                #to radian
                azim_a = np.radians(azim_a)
                elev_a = np.radians(elev_a)
                azim_b = np.radians(azim_b)
                elev_b = np.radians(elev_b)
                
                x_a = np.cos(azim_a) * np.cos(elev_a)
                y_a = np.sin(azim_a) * np.cos(elev_a)
                z_a = np.sin(elev_a)

                x_b = np.cos(azim_b) * np.cos(elev_b)
                y_b = np.sin(azim_b) * np.cos(elev_b)
                z_b = np.sin(elev_b)

                
                distance = np.sqrt((x_b - x_a) ** 2 + (y_b - y_a) ** 2 + (z_b - z_a) ** 2)

                
                # distance_a = np.sqrt(x_a ** 2 + y_a ** 2 + z_a ** 2)
                # distance_b = np.sqrt(x_b ** 2 + y_b ** 2 + z_b ** 2)
                distance_a,distance_b=1.0,1.0

                
                cos_c = (distance_a**2+distance_b**2-distance**2)/(2*distance_a*distance_b)
                angle = np.arccos(cos_c)

                
                degree = np.degrees(angle)
                # degree=degree.item()unnecessary
                return degree

            l=[]
            x0 = base_xyz[0]
            y0 = base_xyz[1]
            def _f(degree):
                degree=degree%360
                if degree>180:
                    degree-=360
                # elif degree<-180:# unnecessary  (python xxx % int => [0,int)
                #     degree+=180
                return degree
            for x_relToBase,y_relToBase,_ in l_xyz:
                y_relToBase=_f(y_relToBase)
                delta_azim=y_relToBase
                delta_elev=(-x_relToBase)
                delta_angle=get_spherical_distance(y0,-x0,y0+y_relToBase,(-x0)-x_relToBase)
                assert abs(delta_azim)<180.01,f"delta_azim={delta_azim}"
                assert abs(delta_elev)<(90-root_config.Q0_MIN_ELEV)+0.01,f"delta_elev={delta_elev}"
                assert delta_angle>-0.01,f"delta_angle={delta_angle}"
                assert delta_angle<180.01,f"delta_angle={delta_angle}"
                l.append([delta_azim,delta_elev,delta_angle])
            json_path=os.path.join(
                s._ref_database_folder,
                'ref_relaToBase_info.json'
            )
            with open(json_path,"w") as f:
                json.dump(l,f,)
        save__ref_relaToBase_info(base_xyz, l_xyz)

        run4gen6d_main(
            id_,
            input_image_path,
            output_dir=s._ref_database_folder,
            num_samples=root_config.NUM_SAMPLE,
            l_xyz=l_xyz,
            base_xyz=base_xyz,
            K=K,
        )
    def __build_ref_and_init_sparseViewPoseEstimator(s,K,q_img_path0,q_img_eleRadian=None, ):
        
        zero123_input_img_path = q_img_path0
        id_ = s.id_
        assert zero123_input_img_path is not None
        if root_config.SKIP_GEN_REF_IF_REF_FOLDER_EXIST:
            assert root_config.FORCE_zero123_render_even_img_exist==False,'如果你要FORCE..,那就得把SKIP_GEN_REF..关了'
        os.makedirs(s._ref_database_folder,exist_ok=1)
        ref_folder_build_OVER=Path(s._ref_database_folder)/"ref_folder_build_OVER"
        # if not ref_folder_build_OVER.exists():#temp
        #     print('if not ref_folder_build_OVER.exists():#temp')
        #     ref_folder_build_OVER.touch()
        if root_config.SKIP_GEN_REF_IF_REF_FOLDER_EXIST and ref_folder_build_OVER.exists():
            pass
        else:
            if ref_folder_build_OVER.exists():
                print(f"[warning] ref_folder_build_OVER already exists: {str(ref_folder_build_OVER)}")
            if root_config.CONSIDER_IPR:
                # img = Image.open(zero123_input_img_path)
                
                degree_counterClockwise, path_rot = imgPath2IPRbyOEE(
                    K, zero123_input_img_path, run4gen6d_main,run4gen6d__sample_model_batchB_wrapper,
                    id_=id_,
                )
                assert os.path.dirname(path_rot)==os.path.dirname(zero123_input_img_path)
                
                img_path__origin___beforeIPR=zero123_input_img_path
                zero123_input_img_path=path_rot
                """
                img_ipr = img.rotate(degree_counterClockwise, fillcolor=(255, 255, 255))
                del img
                img0_warp_path = img0_warp_path.replace(".jpg", "_IPR.jpg")
                img_ipr.save(img0_warp_path)
                """
                # print(f"after IPR= {zero123_input_img_path}")
            else:
                degree_counterClockwise=None
                # img_path__origin___beforeIPR="!root_config.CONSIDER_IPR==False"
                img_path__origin___beforeIPR=zero123_input_img_path
            if (q_img_eleRadian == None):
                q_img_eleRadian = imgPath2elevRadian(\
                    K,
                    zero123_input_img_path,
                    run4gen6d_main=run4gen6d_main,
                    id_=id_,
                )
            s.__save_zero123Input_img_and_info(zero123_input_img_path,q_img_eleRadian,degree_counterClockwise,img_path__origin___beforeIPR)
            s.__run_zero123(K,zero123_input_img_path, input_image_eleRadian=q_img_eleRadian, id_=id_)
            ref_folder_build_OVER.touch(exist_ok=True)
        sparseViewPoseEstimator = SparseViewPoseEstimator(
            database_name=id_,
            _vis_dir=os.path.join(s._ref_database_folder,'vis'),
        )
        s.sparseViewPoseEstimator:SparseViewPoseEstimator = sparseViewPoseEstimator
    def estimate(
            s,
            K,
            q_img_path=None,   
            q_img_eleRadian=None,
            #
            pose_init=None, SCALE_DOWN=1,detection_outputs=None,
            # vis
            SHOW_GT_BBOX=False,
    ):
        assert s.sparseViewPoseEstimator is not None
        # if(root_config.GEN_TRAINING_DATA):
        #     return
        SCALE_DOWN = 1
        return s.sparseViewPoseEstimator.estimate(
            K=K,
            q_img_path=q_img_path, pose_init=pose_init,detection_outputs=detection_outputs,
            SCALE_DOWN=SCALE_DOWN,

            SHOW_GT_BBOX=SHOW_GT_BBOX,
        )

