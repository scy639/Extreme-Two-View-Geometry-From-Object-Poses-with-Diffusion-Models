import math
from PIL import Image
import image_util
import matplotlib.pyplot as plt
from imports import Global
import numpy as np
import cv2
import os,json
import imageio
from copy import deepcopy
import glob
import loguru
import torch
from pathlib import Path
import root_config
from miscellaneous.Zero123_BatchB_Input import  Zero123_BatchB_Input
# from oee.models.loftr import LoFTR, default_cfg
# from oee.utils.utils3d import rect_to_img, canonical_to_camera, calc_pose
from ..models.loftr import LoFTR, default_cfg
from ..utils.utils3d import rect_to_img, canonical_to_camera, calc_pose

class IPR_Delta_Elev_Azim:
    """
    NOTE elev_azim是zero123 的x,y(elev不是angle to horizental plane)
    """
    l_elev_azim=(
        (5,0),
        (10,0),
        (-5,0),
        (-10,0),
        (0,5),
        (0,10),
        (0,15),
        (0, -5),
        (0, -10),
        (0, -15),
    )
    l_elev_azim=(# D  (remove comment is C)
        # (5,0),
        (10,0),
        # (-5,0),
        (-10,0),
        # (0,5),
        (0,10),
        (0,30),
        (0,45),
        # (0, -5),
        (0, -10),
        (0, -30),
        (0, -45),
    )
    # l_elev_azim=(  #B (   '(tmp4testR_IPR-CONSIDER_IPR-B)'
    #     (10,0),
    #     (-10,0),
    #     (0,10),
    #     (0, -10),
    # )
    l_elev_azim=( # faster configuration
        (10,0),
        (-10,0),
        (0,10),
        (0, -10),
    )
    l_elev=[e[0] for e in l_elev_azim]
    l_azim=[e[1] for e in l_elev_azim]
    l_elev=tuple(l_elev)
    l_azim=tuple(l_azim)
class ElevEstHelper:
    _feature_matcher = None

    @classmethod
    def get_feature_matcher(cls):
        if cls._feature_matcher is None:
            loguru.logger.info("Loading feature matcher...")
            _default_cfg = deepcopy(default_cfg)
            _default_cfg['coarse']['temp_bug_fix'] = True  # set to False when using the old ckpt
            matcher = LoFTR(config=_default_cfg)
            current_dir = os.path.dirname(os.path.abspath(__file__))
            ckpt_path=root_config.weightPath_loftr
            matcher.load_state_dict(torch.load(ckpt_path)['state_dict'])
            matcher = matcher.eval().cuda()
            cls._feature_matcher = matcher
        return cls._feature_matcher
    DELTA:int=None # set in: def getFourNearImagePaths

def mask_out_bkgd(img_path, dbg=False):
    # img = imageio.imread_v2(img_path)
    img = imageio.imread(img_path)
    if img.shape[-1] == 4:
        fg_mask = img[:, :, :3]
    else:
        # loguru.logger.info("Image has no alpha channel, using thresholding to mask out background")
        fg_mask = ~(img > 245).all(axis=-1)
    return fg_mask


def get_feature_matching(img_paths, dbg=False,for_IPR=False):
    if for_IPR:
        N=len(IPR_Delta_Elev_Azim.l_elev_azim)
    else:
        N=4
    assert len(img_paths)==N
    matcher = ElevEstHelper.get_feature_matcher()
    feature_matching = {}
    masks = []
    for i in range(N):
        mask = mask_out_bkgd(img_paths[i], dbg=dbg)
        masks.append(mask)
    for i in range(0, N):
        for j in range(i + 1, N):
            img0_pth = img_paths[i]
            img1_pth = img_paths[j]
            mask0 = masks[i]
            mask1 = masks[j]
            img0_raw = cv2.imread(img0_pth, cv2.IMREAD_GRAYSCALE)
            img1_raw = cv2.imread(img1_pth, cv2.IMREAD_GRAYSCALE)
            original_shape = img0_raw.shape
            img0_raw_resized = cv2.resize(img0_raw, (480, 480))
            img1_raw_resized = cv2.resize(img1_raw, (480, 480))

            img0 = torch.from_numpy(img0_raw_resized)[None][None].cuda() / 255.
            img1 = torch.from_numpy(img1_raw_resized)[None][None].cuda() / 255.
            batch = {'image0': img0, 'image1': img1}

            # Inference with LoFTR and get prediction
            with torch.no_grad():
                matcher(batch)#infer(will change batch)
                mkpts0 = batch['mkpts0_f'].cpu().numpy()
                mkpts1 = batch['mkpts1_f'].cpu().numpy()
                mconf = batch['mconf'].cpu().numpy()
            mkpts0[:, 0] = mkpts0[:, 0] * original_shape[1] / 480
            mkpts0[:, 1] = mkpts0[:, 1] * original_shape[0] / 480
            mkpts1[:, 0] = mkpts1[:, 0] * original_shape[1] / 480
            mkpts1[:, 1] = mkpts1[:, 1] * original_shape[0] / 480
            keep0 = mask0[mkpts0[:, 1].astype(int), mkpts1[:, 0].astype(int)]
            keep1 = mask1[mkpts1[:, 1].astype(int), mkpts1[:, 0].astype(int)]
            keep = np.logical_and(keep0, keep1)
            mkpts0 = mkpts0[keep]
            mkpts1 = mkpts1[keep]
            mconf = mconf[keep]
            feature_matching[f"{i}_{j}"] = np.concatenate([mkpts0, mkpts1, mconf[:, None]], axis=1)

    return feature_matching


def gen_pose_hypothesis(center_elevation,for_IPR=False):
    if for_IPR:
        elevations=list(IPR_Delta_Elev_Azim.l_elev)
        azimuths = list(IPR_Delta_Elev_Azim.l_azim)
        elevations = [0]+elevations
        azimuths = [0]+azimuths
        elevations = [center_elevation+i for i in elevations]
        azimuths =  [30+i for i in azimuths]
        elevations = np.radians( elevations)
        azimuths = np.radians( azimuths)
        # #print(f"[gen_pose_hypothesis]elevations={elevations},azimuths={azimuths}")
    else:
        elevations = np.radians(
            [center_elevation, center_elevation - 10, center_elevation + 10, center_elevation,
             center_elevation])  # 45~120
        azimuths = np.radians([30, 30, 30, 20, 40])
    input_poses = calc_pose(elevations, azimuths, len(azimuths))
    input_poses = input_poses[1:]
    input_poses[..., 1] *= -1
    input_poses[..., 2] *= -1
    return input_poses


def ba_error_general(K, matches, poses):
    projmat0 = K @ poses[0].inverse()[:3, :4]
    projmat1 = K @ poses[1].inverse()[:3, :4]
    match_01 = matches[0]
    pts0 = match_01[:, :2]
    pts1 = match_01[:, 2:4]
    Xref = cv2.triangulatePoints(projmat0.cpu().numpy(), projmat1.cpu().numpy(),
                                 pts0.cpu().numpy().T, pts1.cpu().numpy().T)
    Xref = Xref[:3] / Xref[3:]
    Xref = Xref.T
    Xref = torch.from_numpy(Xref).float()
    reproj_error = 0
    for j,(match, cp) in enumerate(zip(matches[1:], poses[2:])):
        # #print(f"\nmatches[{j+1}]")
        dist = (torch.norm(match_01[:, :2][:, None, :] - match[:, :2][None, :, :], dim=-1))
        # #print(f"dist.numel()={ dist.numel()}")
        if dist.numel() > 0:
            # #print("dist.shape", dist.shape)
            m0to2_index = dist.argmin(1)
            keep = dist[torch.arange(match_01.shape[0]), m0to2_index] < 1
            # #print(f"keep.sum()={keep.sum()}")
            if keep.sum() > 0:
                xref_in2 = rect_to_img(K, canonical_to_camera(Xref, cp.inverse()))
                reproj_error2 = torch.norm(match[m0to2_index][keep][:, 2:4] - xref_in2[keep], dim=-1)
                conf02 = match[m0to2_index][keep][:, -1]
                err=(reproj_error2 * conf02).sum() / (conf02.sum())
                # #print(f"==>> err: {err}")
                reproj_error += err

    return reproj_error


def find_optim_elev(elevs, nimgs, matches, K, dbg=False,for_IPR=False):
    errs = []
    for elev in elevs:
        err = 0
        cam_poses = gen_pose_hypothesis(elev,for_IPR=for_IPR)
        for start in range(nimgs - 1):
            # #print(f"\n\nstart={start}")
            batch_matches, batch_poses = [], []
            for i in range(start, nimgs + start):
                ci = i % nimgs
                batch_poses.append(cam_poses[ci])
            for j in range(nimgs - 1):
                key = f"{start}_{(start + j + 1) % nimgs}"
                match = matches[key]
                
                batch_matches.append(match)
            err += ba_error_general(K, batch_matches, batch_poses)
        # #print(f"for this elev={elev}, err={err}")
        errs.append(err)
    errs = torch.tensor(errs)
    if dbg:
        plt.plot(elevs, errs)
        plt.show()
    min_err=min(errs).item()
    # #print(f"[find_optim_elev]min_err={min_err}")
    optim_elev = elevs[torch.argmin(errs)].item()
    if '4_ipr_ex1__l_tuples' in   Global.anything:
        Global.anything['4_ipr_ex1__l_tuples'][-1][-1]+=[min_err,90-optim_elev] 
    return optim_elev,min_err


def get_elev_est(feature_matching, min_elev=30, max_elev=150, K=None, for_IPR=False):
    flag = True
    matches = {}
    if for_IPR:
        N=len(IPR_Delta_Elev_Azim.l_elev_azim)
    else:
        N=4
    for i in range(N):
        for j in range(i + 1, N):
            match_ij = feature_matching[f"{i}_{j}"]
            if len(match_ij) == 0:
                flag = False
            match_ji = np.concatenate([match_ij[:, 2:4], match_ij[:, 0:2], match_ij[:, 4:5]], axis=1)
            matches[f"{i}_{j}"] = torch.from_numpy(match_ij).float()
            matches[f"{j}_{i}"] = torch.from_numpy(match_ji).float()
    if not flag:
        loguru.logger.info("0 matches, could not estimate elevation")
        if for_IPR:
            return math.inf
        else:
            return None
    interval = 10
    elevs = np.arange(min_elev, max_elev, interval)
    optim_elev1 ,_= find_optim_elev(elevs, N, matches, K,for_IPR=for_IPR)

    elevs = np.arange(optim_elev1 - 10, optim_elev1 + 10, 1)
    optim_elev2,min_err = find_optim_elev(elevs, N, matches, K,for_IPR=for_IPR)
    # #print(f"optim_elev2,min_err= {optim_elev2,min_err}")
    if for_IPR:
        return min_err
    else:
        return optim_elev2


def elev_est_api(img_paths, 
                #  min_elev=30, max_elev=150, 
                 min_elev, max_elev, 
                 K=None, dbg=False):
    feature_matching = get_feature_matching(img_paths, dbg=dbg)
    if K is None:
        loguru.logger.warning("K is not provided, using default K")
        K = np.array([[280.0, 0, 128.0],
                      [0, 280.0, 128.0],
                      [0, 0, 1]])
    K = torch.from_numpy(K).float() 
    elev = get_elev_est(feature_matching, min_elev, max_elev, K,  )
    if elev is None:
        # return 45 #temp
        raise Exception('0 matches')
    elev=90-elev
    return elev

#--------------------------- IPRbyOEE ---------------------------

def imgPath2IPRbyOEE(
        K, input_image_path,
        run4gen6d_main, run4gen6d__sample_model_batchB_wrapper,
        id_,
):
    """
    融合、改自 imgPath2elevRadian,_f2,并将其中call elev_est_api在此展开
    """
    K = np.array([[280.0, 0, 128.0],
                  [0, 280.0, 128.0],
                  [0, 0, 1]])
    if K is None:
        assert 0
        loguru.logger.warning("K is not provided, using default K")
        K = np.array([[280.0, 0, 128.0],
                      [0, 280.0, 128.0],
                      [0, 0, 1]])
    K = torch.from_numpy(K).float()  #  this line is from ID-pose instead of
    def nearImagePaths_2_min_err(nearImagePaths):
        
        feature_matching = get_feature_matching(nearImagePaths, for_IPR=True, )
        min_err = get_elev_est(feature_matching,
                               90 - 79, 90 - 0,
                               K, for_IPR=True)
        return min_err
    def get_min_err(path_rot,batchB=False):
        """
        when batchB,
            path_rot is actually list of path_rot
            return l_min_err
        """
        nonlocal K
        # id2 = f"4IPRbyOEE-{id_}-{os.path.basename(path_rot)}"
        def getNearImagePaths(path_rot:str):
            """
            when batchB, return l_xyz,...
            in zero123: path_output_ims = os.path.join(root_config.dataPath_zero123, parentFolderName_output_ims,id)
            """
            assert isinstance(path_rot,str) or isinstance(path_rot,Path) 
            parentFolderName_output_ims = '_' + f"4IPRbyOEE-{id_}"
            id4outputim=f"{os.path.basename(path_rot)}"
            l_xyz=[(elev_azim[0],elev_azim[1],0) for i,elev_azim in enumerate(IPR_Delta_Elev_Azim.l_elev_azim)]
            if batchB:
                return l_xyz,id4outputim,parentFolderName_output_ims
            l__path_output_im=run4gen6d_main(
                id4outputim,# id2,
                path_rot,
                output_dir=None,
                num_samples=1,
                l_xyz=l_xyz,
                base_xyz=(0, 0, 0),
                ddim_steps=50,
                K=None,
                only_gen=True,# dont crop,re center etc
                parentFolderName_output_ims=parentFolderName_output_ims,
            )
            assert len(l_xyz)==len(l__path_output_im)
            return l__path_output_im

        '''
        def debug3523():
            img_dir ="/sharedata/home/suncaiyi/space/One-2-3-45/exp/00318_warp copy9(min_elev=1, max_elev=89)/stage2_8"
            img_paths = []
            for i in range(4):
                img_paths.append(f"{img_dir}/0_{i}.png")
            return img_paths
        fourNearImagePaths=debug3523()
        '''

        if batchB:
            assert isinstance(path_rot,list)
            l__zero123_BatchB_Input=[]
            for i in path_rot:
                l_xyz,id4outputim,parentFolderName_output_ims=getNearImagePaths(i)
                zero123_BatchB_Input=Zero123_BatchB_Input(
                    id_=id4outputim,folder_outputIms=parentFolderName_output_ims,input_image_path=i,l_xyz=l_xyz,
                )
                l__zero123_BatchB_Input.append(zero123_BatchB_Input)
            l__zero123_BatchB_Input=run4gen6d__sample_model_batchB_wrapper(
                l__zero123_BatchB_Input,
                ddim_steps=50,
            )
            l_min_err=[]
            for zero123_BatchB_Input in l__zero123_BatchB_Input:
                nearImagePaths=zero123_BatchB_Input.outputims
                min_err = nearImagePaths_2_min_err(nearImagePaths)
                l_min_err.append(min_err)
            return l_min_err
        else:
            nearImagePaths = getNearImagePaths(path_rot)
            min_err = nearImagePaths_2_min_err(nearImagePaths)
            return min_err

    def _f3(l_angle_in_degree):
        assert 0,'没法rotate_C'
        l_min_err = []
        l_path_rot = []
        img0 = Image.open(input_image_path)
        for rot_angle_in_degree in l_angle_in_degree:
            #print(f"\n\n\nrot_angle_in_degree={rot_angle_in_degree}")
            # img0_rot = img0.rotate(rot_angle_in_degree, fillcolor=(255, 255, 255),
            #                        resample=Image.BICUBIC,
            #                        )
            # img0_rot=image_util.rotate_B(rot_angle_in_degree,img0)
            path0_rot = f"{input_image_path}.rot{rot_angle_in_degree}.png"
            img0_rot.save(path0_rot)
            if root_config.ForDebug.forceIPRtoBe.enable is True:
                min_err = -639
            else:
                min_err = get_min_err(path0_rot)
            l_min_err.append(min_err)
            l_path_rot.append(path0_rot)
        assert len(l_min_err) == len(l_angle_in_degree)
        assert len(l_path_rot) == len(l_angle_in_degree)
        index=l_min_err.index(min(l_min_err))
        #print('\nzip(l_angle_in_degree,l_min_err,l_path_rot):\n',json.dumps(list(zip(l_angle_in_degree,l_min_err,l_path_rot)),indent=4))
        deg = l_angle_in_degree[index]
        path_rot = l_path_rot[index]
        return deg,path_rot
    def _f3_batchB(l_angle_in_degree):
        l_path_rot = []
        img0 = Image.open(input_image_path)
        l_rot_img=image_util.rotate_C(img0,l_angle_in_degree,)
        for img0_rot,rot_angle_in_degree in zip(l_rot_img,l_angle_in_degree):
            #print(f"\n\n\nrot_angle_in_degree={rot_angle_in_degree}")
            # img0_rot = img0.rotate(rot_angle_in_degree, fillcolor=(255, 255, 255),
            #                        resample=Image.BICUBIC,
            #                        )
            # img0_rot=image_util.rotate_B(rot_angle_in_degree,img0)
            path0_rot = f"{input_image_path}.rot{rot_angle_in_degree}.png"
            img0_rot.save(path0_rot)
            l_path_rot.append(path0_rot)
        #print(f"{l_path_rot=}")
        if root_config.ForDebug.forceIPRtoBe.enable is True:
            assert len(l_angle_in_degree)==1
            l_min_err = [-639]
        else:
            l_min_err=get_min_err(l_path_rot,batchB=True)
        assert len(l_min_err) == len(l_angle_in_degree)
        assert len(l_path_rot) == len(l_angle_in_degree)
        index=l_min_err.index(min(l_min_err))
        #print('\nzip(l_angle_in_degree,l_min_err,l_path_rot):\n',json.dumps(list(zip(l_angle_in_degree,l_min_err,l_path_rot)),indent=4))
        deg = l_angle_in_degree[index]
        path_rot = l_path_rot[index]
        return deg,path_rot
    def normalize_deg(x):
        """
        to 0,360
        eg. -2%360=358
        """
        assert isinstance(x,int)
        return x%360
    if root_config.ForDebug.forceIPRtoBe.enable is True:
        l_angle_in_degree =[root_config.ForDebug.forceIPRtoBe.IPR]
    else:
        if 1:
            _f3_=_f3_batchB
        else:
            _f3_=_f3
        if 0:
            # l_angle_in_degree = range(-45,45+1,5)
            l_angle_in_degree = range(-40,40+1,5) 
            l_angle_in_degree=[normalize_deg(x) for x in l_angle_in_degree]
            coarse_deg,_=_f3_(l_angle_in_degree)
            l_angle_in_degree = range(coarse_deg-5,coarse_deg+5+1,
                                    #   1)
                                    2)
            # l_angle_in_degree=(-4,-2,0,2,4,) coarse_deg+=
        else:#faster configuration
            l_angle_in_degree = (-40,-30,-20,-10,0,10,20,30,40,)#len=9
            l_angle_in_degree=[normalize_deg(x) for x in l_angle_in_degree]
            coarse_deg,_=_f3_(l_angle_in_degree)
            l_angle_in_degree = (-8,-6,-4,-2,0,2,4,6,8)#len=9-1
            l_angle_in_degree=[coarse_deg+i for i in l_angle_in_degree]
            """ more faster than conf above, about several seconds('faster configuration' reported in paper). but I dont know why Racc30 decrease 0.02-0.03 on rotated gso compared to 2024.1 
            l_angle_in_degree = ( -30, -10, 10, 30, ) #len=4
            l_angle_in_degree=[normalize_deg(x) for x in l_angle_in_degree]
            coarse_deg1,_=_f3_(l_angle_in_degree)
            l_angle_in_degree = (-14,-7,0,7,14,)#len=4
            l_angle_in_degree=[coarse_deg1+i for i in l_angle_in_degree]
            l_angle_in_degree=[normalize_deg(x) for x in l_angle_in_degree]
            coarse_deg2,_=_f3_(l_angle_in_degree)
            l_angle_in_degree = (-4,-2,0,2,4,)#len=4
            l_angle_in_degree=[coarse_deg2+i for i in l_angle_in_degree]
            """
    l_angle_in_degree=[normalize_deg(x) for x in l_angle_in_degree]
    fine_deg ,path_rot= _f3_(l_angle_in_degree)
    # raise Exception('tmp4iprExB')
    # raise Exception('tmp4test_batchB')
    return fine_deg,path_rot
