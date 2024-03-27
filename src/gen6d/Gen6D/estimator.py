from skimage.io import imsave
from .scy.Config import *
from .scy.DebugUtil import *
from .scy.Zero123Detector import Zero123Detector
from .scy.gen6dGlobal import gen6dGlobal
import  root_config
import cv2
import numpy as np
import torch
import functools
from .dataset.database import BaseDatabase, get_database_split, get_object_vert, get_object_center,Zero123CustomDatabase

from . import network
from .network import name2network
from .utils.base_utils import load_cfg, transformation_offset_2d, transformation_scale_2d, \
    transformation_compose_2d, transformation_crop, transformation_rotation_2d
from .utils.database_utils import select_reference_img_ids_fps, normalize_reference_views
from .utils.pose_utils import estimate_pose_from_similarity_transform_compose
from imports import Global

def compute_similarity_transform(pts0, pts1):
    """
    @param pts0:
    @param pts1:
    @return: sR @ p0 + t = p1
    """
    ref_c = np.mean(pts0, 0)
    que_c = np.mean(pts1, 0)
    ref_d = pts0 - ref_c[None, :]
    que_d = pts1 - que_c[None, :]
    scale = np.mean(np.linalg.norm(que_d, 2, 1)) / np.mean(np.linalg.norm(ref_d, 2, 1))
    ref_d_ = ref_d * scale
    U, S, VT = np.linalg.svd(ref_d_.T @ que_d)
    rotation = VT.T @ U.T
    offset = - scale * (rotation @ ref_c) + que_c
    return scale, rotation, offset


def compute_similarity_transform_batch(pts0, pts1):
    """
    @param pts0:
    @param pts1:
    @return: sR @ p0 + t = p1
    """
    c0 = np.mean(pts0, 1)  # n, 2
    c1 = np.mean(pts1, 1)  # n, 2
    d0 = pts0 - c0[:, None, :]
    d1 = pts1 - c1[:, None, :]
    scale = np.mean(np.linalg.norm(d1, 2, 2, keepdims=True), 1, keepdims=True) / \
            np.mean(np.linalg.norm(d0, 2, 2, keepdims=True), 1, keepdims=True)  # n,1,1
    d0_ = d0 * scale  # n,k,2
    U, S, VT = np.linalg.svd(d0_.transpose([0, 2, 1]) @ d1)  # n,2,2
    rotation = VT.transpose([0, 2, 1]) @ U.transpose([0, 2, 1])  # n,2,2
    offset = - scale * (rotation @ c0[:, :, None]) + c1[:, :, None]
    return scale, rotation, offset  # [n,1,1] [n,2,2] [n,2,1]


def compute_inlier_mask(scale, rotation, offset, corr, thresh):
    x0 = corr[None, :, :2]  # [1,k,2]
    x1 = corr[None, :, 2:]  # [1,k,2]
    x1_ = scale * (x0 @ rotation.transpose([0, 2, 1])) + offset.transpose([0, 2, 1])
    mask = np.linalg.norm(x1 - x1_, 2, 2) < thresh
    return mask


def ransac_similarity_transform(corr):
    n, _ = corr.shape
    batch_size = 4096
    bad_seed_thresh = 4
    inlier_thresh = 5
    best_inlier, best_mask = 0, None
    iter_num = 0
    confidence = 0.99
    while True:
        idx = np.random.randint(0, n, [batch_size, 2])
        seed0 = corr[idx[:, 0]]  # b,4
        seed1 = corr[idx[:, 1]]  # b,4
        bad_mask = np.linalg.norm(seed0 - seed1, 2, 1) < bad_seed_thresh
        seed0 = seed0[~bad_mask]
        seed1 = seed1[~bad_mask]
        seed = np.stack([seed0, seed1], 1)
        scale, rotation, offset = compute_similarity_transform_batch(seed[:, :, :2], seed[:, :, 2:])  #
        mask = compute_inlier_mask(scale, rotation, offset, corr, inlier_thresh)  # b,n
        inlier_num = np.sum(mask, 1)
        if np.max(inlier_num) >= best_inlier:
            best_mask = mask[np.argmax(inlier_num)]
        iter_num += seed.shape[0]
        inlier_ratio = np.mean(best_mask)
        if 1 - (1 - inlier_ratio ** 2) ** iter_num > confidence:
            break

    inlier_corr = corr[best_mask]
    scale, rotation, offset = compute_similarity_transform_batch(inlier_corr[None, :, :2], inlier_corr[None, :, 2:])
    scale, rotation, offset = scale[0, 0, 0], rotation[0], offset[0, :, 0]
    return scale, rotation, offset, best_mask


def compose_similarity_transform(scale, rotation, offset):
    M = transformation_scale_2d(scale)
    M = transformation_compose_2d(M, np.concatenate([rotation, np.zeros([2, 1])], 1).astype(np.float32))
    M = transformation_compose_2d(M, transformation_offset_2d(offset[0], offset[1]))
    return M


class Gen6DEstimator:
    default_cfg = {
        'ref_resolution': 128,
        "ref_view_num": 64,
        "det_ref_view_num": 32,

        'selector': None,
        'detector': None,
        'refiner': None,

        'refine_iter': 3,
    }

    def __init__(self, cfg):
        self.cfg = {**self.default_cfg, **cfg}
        """
        cfg={
            'name': 'gen6d_pretrain', 
            'type': 'gen6d', 
            'detector': 'configs/detector/detector_pretrain.yaml', 
            'selector': 'configs/selector/selector_pretrain.yaml', 
            'refiner': 'configs/refiner/refiner_pretrain.yaml', 
            'ref_resolution': 128, 
            'ref_view_num': -1, 
            'det_ref_view_num': 32, 
            'refine_iter': 3
        }
        """
        self.ref_info = {}

        # print("gen6dGlobal.USE_Zero123Detector:",gen6dGlobal.USE_Zero123Detector)
        if (gen6dGlobal.USE_Zero123Detector):
            self.detector = Zero123Detector()
        else:
            if (DO_NOT_load_model):
                self.detector = None
            else:
                self.detector = self._load_module(self.cfg['detector'])
        if (DO_NOT_load_model):
            self.selector: network.ViewpointSelector = None
        else:
            """
            cfg['selector'] is path
            load_cfg(  cfg['selector']  )= {'name': 'selector_pretrain', 'network': 'selector', 'selector_angle_num': 5}
            """
            self.selector: network.ViewpointSelector = self._load_module(self.cfg['selector'])
        if ((self.cfg['refiner'] is not None) and (not DO_NOT_load_model)):
            self.refiner = self._load_module(self.cfg['refiner'])
        else:
            self.refiner = None

    @staticmethod
    @functools.cache
    def _load_module(cfg):
        refiner_cfg = load_cfg(cfg)
        if refiner_cfg['network']=='selector':
            pth_path=root_config.weightPath_selector
            del refiner_cfg["name"]
        elif refiner_cfg['network']=='refiner':
            pth_path=root_config.weightPath_refiner
            del refiner_cfg["name"]
        else:
            pth_path=f'data/model/{refiner_cfg["name"]}/model_best.pth'
        refiner:network.VolumeRefiner = name2network[refiner_cfg['network']](refiner_cfg)
        print(f'loading from {pth_path}')
        # gpu_util.occumpy_mem_for_ckpt( weight_path=  )
        state_dict = torch.load(pth_path,map_location=torch.device(root_config.DEVICE))
        if 'step' in state_dict:
            print(f"step={state_dict['step']}")
        refiner.load_state_dict(state_dict['network_state_dict'])
        refiner.cuda().eval()
        return refiner

    # def _check(self, ref_point_cloud, ref_imgs, ref_poses, ref_Ks, ref_ids, database):
    #     rfn = ref_imgs.shape[0]
    #     output_imgs = []
    #     for rfi in range(rfn):
    #         pts2d, _ = project_points(ref_point_cloud, ref_poses[rfi], ref_Ks[rfi])
    #         kps_img = draw_keypoints(ref_imgs[rfi],pts2d)//2+ref_imgs[rfi]//2
    #         img_raw = database.get_image(ref_ids[rfi])
    #         output_imgs.append(concat_images_list(img_raw,kps_img,vert=True))
    #
    #     imsave(f'data/vis_val/check.jpg',concat_images_list(*output_imgs))
    #     import ipdb; ipdb.set_trace()

    def build(self,
              database: Zero123CustomDatabase,#BaseDatabase,
              split_type: str):
        ref_relaToBase_info:np.ndarray=database.get_ref_relaToBase_info()
        object_center = get_object_center(database)  
        object_vert = get_object_vert(database)  # [1,0,0]
        ref_ids_all, _ = get_database_split(database, split_type)
        """
        # use fps to select reference images for detection and selection
        # ref_view_num=self.cfg['ref_view_num']
        ref_view_num=root_config.NUM_REF
        ref_ids = select_reference_img_ids_fps(database, ref_ids_all, ref_view_num)
        """
        ref_ids=np.asarray(ref_ids_all)
        # print(f'ref_ids=', ref_ids.tolist()[:5]+ref_ids.tolist()[-5:])
        
        
        
        ref_imgs, ref_masks, ref_Ks, ref_poses, ref_Hs = \
            normalize_reference_views(database, ref_ids, self.cfg['ref_resolution'],
                                      margin=0.05,
                                      # margin=0.4,

                                      save_normalized_ref_imgs=0,
                                      )  

        # in-plane rotation for viewpoint selection
        rfn, h, w, _ = ref_imgs.shape

        
        # for i in range(rfn):
        #     imsave(
        #         f'4debug/after normalize_reference_views/ref_img[id={ref_ids[i]},rfn={i}].jpg', ref_imgs[i])

        ref_imgs_rots = []
        angles = [-np.pi / 2, -np.pi / 4, 0, np.pi / 4, np.pi / 2]
        for angle in angles:
            M = transformation_offset_2d(-w / 2, -h / 2)
            M = transformation_compose_2d(M, transformation_rotation_2d(angle))
            M = transformation_compose_2d(M, transformation_offset_2d(w / 2, h / 2))
            H_ = np.identity(3).astype(np.float32)
            H_[:2, :3] = M
            ref_imgs_rot = []
            for rfi in range(rfn):
                H_new = H_ @ ref_Hs[rfi]
                ref_imgs_rot.append(
                    cv2.warpPerspective(database.get_image(ref_ids[rfi]), H_new, (w, h), flags=cv2.INTER_LINEAR))
            ref_imgs_rots.append(np.stack(ref_imgs_rot, 0))
        ref_imgs_rots = np.stack(ref_imgs_rots, 0)  

        self.detector.load_ref_imgs(ref_imgs[:self.cfg['det_ref_view_num']])
        self.selector.load_ref_imgs(ref_imgs_rots, ref_poses, object_center, object_vert,ref_relaToBase_info=ref_relaToBase_info)
        #imgs:  rfn, h, w, 3;  'ref_imgs':an==5,rfn==127, h==128, w==128, 3; 
        self.ref_info = {'imgs': ref_imgs, 'ref_imgs': ref_imgs_rots, 'masks': ref_masks, 'Ks': ref_Ks,
                         'poses': ref_poses, 'center': object_center}

        if self.refiner is not None:
            self.refiner.load_ref_imgs(database, ref_ids_all)

    def predict(self, que_img, que_K, pose_init=None,
                detection_outputs=None,
                ):
        inter_results = {}

        if pose_init is None:
            # stage 1: detection
            with torch.no_grad():
                if (detection_outputs is None):
                    detection_outputs = self.detector.detect_que_imgs(que_img[None])
                else:
                    assert  len(detection_outputs['positions'])==1
                    h, w, _ = que_img.shape
                    # h, w=self.cfg['ref_resolution'],self.cfg['ref_resolution']
                    relative_positions=detection_outputs['positions'][0]
                    detection_outputs['positions'][0]=[int(relative_positions[0]*w),int(relative_positions[1]*h)]
                position = detection_outputs['positions'][0]
                scale_r2q = detection_outputs['scales'][0]

            # crop the image according to the detected scale and the detected position
            que_img_, _ = transformation_crop(que_img, position, 1 / scale_r2q, 0, self.cfg['ref_resolution'])  # h,w,3
            inter_results['det_position'] = position
            inter_results['det_scale_r2q'] = scale_r2q
            inter_results['det_que_img'] = que_img_

            # stage 2: viewpoint selection
            with torch.no_grad():
                selection_results = self.selector.select_que_imgs(que_img_[None])#qn, qn, [qn,rfn]

            
            ref_idx = selection_results['ref_idx'][0]
            angle_r2q = selection_results['angles'][0]
            scores = selection_results['scores'][0]

            inter_results['sel_angle_r2q'] = angle_r2q
            inter_results['sel_scores'] = scores
            inter_results['sel_ref_idx'] = ref_idx

            # stage 3: solve for pose from detection/selected viewpoint/in-plane rotation
            ref_pose = self.ref_info['poses'][ref_idx]
            ref_K = self.ref_info['Ks'][ref_idx]
            pose_pr = estimate_pose_from_similarity_transform_compose(
                position, scale_r2q, angle_r2q, ref_pose, ref_K, que_K, self.ref_info['center'])
        else:
            pose_pr = pose_init

        # stage 4: refine pose
        if self.refiner is not None:
            refine_poses = [pose_pr]
            # for k in range(self.cfg['refine_iter']):
            for k in range(  root_config.REFINE_ITER  ):
                pose_pr = self.refiner.refine_que_imgs(que_img, que_K, pose_pr, size=128, ref_num=6, ref_even=True)
                refine_poses.append(pose_pr)  
            inter_results['refine_poses'] = refine_poses
            Global.RefinerInterPoses.set(refine_poses)
        return pose_pr, inter_results


name2estimator = {
    'gen6d': Gen6DEstimator,
}
