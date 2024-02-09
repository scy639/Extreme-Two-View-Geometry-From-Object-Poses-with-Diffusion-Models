import sys,os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

# NAME="mouse"
# DATABASE_TYPE="custom"
# SHOW_GT_BBOX = 0
# QUERY = "colmap/images"
# FROM_VIDEO = 1
# video_path = f"data/{DATABASE_TYPE}/mouse-test.mp4"

# NAME = "myCar"
# NAME = "myCar-do_not_crop"
# NAME = "myCarB-do_not_crop"
# NAME = "myCarB-do_not_crop-margin0.1"

# NAME = "myCarC"
# NAME = "myCarC-calib"
# NAME = "parer"

# NAME = "parer-filtered"
# DATABASE_TYPE = "zero123"
# SHOW_GT_BBOX = 1
# QUERY = "query"
# FROM_VIDEO = 1
# frame_interval=1
# VIDEO_NAME="1"
# SCALE_DOWN = 3

# NAME = "skateboard--207_21896_45453--run_query"
# DATABASE_TYPE = "zero123"
# SHOW_GT_BBOX = 1
# QUERY = "query"
# FROM_VIDEO = 0
# frame_interval=1
# VIDEO_NAME="1"
# SCALE_DOWN = 1

NAME="mouse"
DATABASE_TYPE="custom"
SHOW_GT_BBOX = 0
QUERY = "colmap/images"
FROM_VIDEO = 1
video_path = f"data/{DATABASE_TYPE}/mouse-test.mp4"



import  scy.Config as Config
from utils.draw_utils import pts_range_to_bbox_pts, draw_bbox_3d, draw_bbox_3d_dpt
import argparse
import subprocess
from pathlib import Path
import os
import numpy as np
from skimage.io import imsave, imread
from tqdm import tqdm
import json
from dataset.database import parse_database_name, get_ref_point_cloud
from estimator import name2estimator, Gen6DEstimator
from eval import visualize_intermediate_results
from prepare import video2image
from utils.base_utils import load_cfg, project_points
from utils.pose_utils import pnp
from scy.DebugUtil import *
from scy.gen6dGlobal import gen6dGlobal

if(DATABASE_TYPE == "zero123"):
    if(FROM_VIDEO):
        gen6dGlobal.USE_Zero123Detector=0
    video_dir = f"data/{DATABASE_TYPE}/{NAME}/query_video"
    video_path = f"{video_dir}/{VIDEO_NAME}.mp4"
    video_out_dir = f"{video_dir}/{VIDEO_NAME}"
    if(FROM_VIDEO):
        SHOW_GT_BBOX = 0
    if(FROM_VIDEO):
        output_dir = f"data/{DATABASE_TYPE}/{NAME}/test/{VIDEO_NAME}"
    else:
        output_dir = f"data/{DATABASE_TYPE}/{NAME}/test"
else:
    output_dir = ""
GEN6D_INTER_RESULT_FILENAME = "gen6d_inter_result.json"
gen6d_inter_result = {}


def weighted_pts(pts_list, weight_num=10, std_inv=10):
    weights = np.exp(-(np.arange(weight_num) / std_inv) ** 2)[::-1]  # wn
    pose_num = len(pts_list)
    if pose_num < weight_num:
        weights = weights[-pose_num:]
    else:
        pts_list = pts_list[-weight_num:]
    pts = np.sum(np.asarray(pts_list) *
                 weights[:, None, None], 0) / np.sum(weights)
    return pts


def main(args):
    global SHOW_GT_BBOX
    cfg = load_cfg(args.cfg)
    output_dir = Path(args.output)
    gen6dGlobal.output_dir = output_dir
    ref_database = parse_database_name(args.database)
    estimator: Gen6DEstimator = name2estimator[cfg['type']](
        cfg)   
    estimator.build(ref_database, split_type='all')

    object_pts = get_ref_point_cloud(ref_database)  
    object_bbox_3d = pts_range_to_bbox_pts(
        np.max(object_pts, 0), np.min(object_pts, 0))

    output_dir.mkdir(exist_ok=True, parents=True)

    (output_dir / 'images_raw').mkdir(exist_ok=True, parents=True)
    (output_dir / 'images_out').mkdir(exist_ok=True, parents=True)
    (output_dir / 'images_inter').mkdir(exist_ok=True, parents=True)
    (output_dir / 'images_out_smooth').mkdir(exist_ok=True, parents=True)

    if (FROM_VIDEO):
    # que_num = video2image(args.video, output_dir / 'images_raw', 1, args.resolution, args.transpose)
        que_num = video2image(args.video, Path(video_out_dir) /
                              'images_raw', frame_interval, args.resolution, args.transpose)
        # if que_num > 60:
        #     que_num = 60
    else:
        que_num = 128
    pose_init = None
    hist_pts = []
    if (SHOW_GT_BBOX):
        from scy.IntermediateResult import IntermediateResult
        intermediate_result = IntermediateResult()
        intermediate_result.load(os.path.join(
            'data', args.database, 'ref', 'intermediateResult.json'))
    if(FROM_VIDEO):
        img_dir = Path(video_out_dir) / 'images_raw'
        img_list = os.listdir(img_dir)
        img_list = [img_dir/img for img in img_list if img.endswith('.jpg')]
    else:
        img_list = [os.path.join(
            'data', args.database, QUERY, f'{que_id}.jpg') for que_id in range(que_num)]
    print("img_list", img_list)
    for que_id in tqdm(range(que_num)):
        img_path = img_list[que_id]
        img=imread(img_path)
        # generate a pseudo K
        h, w, _ = img.shape

        
        print("raw query img h,w=",h,w)
        h,w=int(h/SCALE_DOWN),int(w/SCALE_DOWN)
        img=cv2.resize(img,(w,h))

        f = np.sqrt(h ** 2 + w ** 2)  #  
        # 
        K = np.asarray([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], np.float32)

        if pose_init is not None:
            # we only refine one time after initialization
            estimator.cfg['refine_iter'] = 1
        if(not Config.PR_AS_INIT):
            pose_init=None
        pose_pr, inter_results = estimator.predict(
            img, K,
            pose_init=pose_init
        )
        pose_init = pose_pr

        pts, dpt = project_points(object_bbox_3d, pose_pr, K)
        # bbox_img = draw_bbox_3d(img, pts, (0, 0, 255))
        bbox_img = draw_bbox_3d_dpt(img, pts, dpt, (0, 0, 255))

        # SHOW_GT_BBOX = (que_id % 2 == 0)  # debug(all debug are marked by 
        if (SHOW_GT_BBOX):
            gt_img_path = os.path.join(
                'data', args.database, 'ref', os.path.basename(img_path))
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
        imsave(f'{str(output_dir)}/images_out/{que_id}-bbox.jpg', bbox_img)
        print("i,pose_pr=", que_id, pose_pr)
        np.save(f'{str(output_dir)}/images_out/{que_id}-pose.npy', pose_pr)
        imsave(f'{str(output_dir)}/images_inter/{que_id}.jpg',
               visualize_intermediate_results(img, K, inter_results, estimator.ref_info, object_bbox_3d,
                                              object_center=object_center, pose_gt=pose_gt,  # 
                                              ))

        # 
        inter_results_ = inter_results.copy()
        ref_info_ = estimator.ref_info.copy()
        if ("det_que_img" in inter_results_):
            del inter_results_['det_que_img']
        del ref_info_['imgs']
        del ref_info_['ref_imgs']
        del ref_info_['masks']
        gen6d_inter_result[que_id] = {
            "inter_results": inter_results_,
            "ref_info": ref_info_,
            "pose_gt": pose_gt,
            "pose_pr": pose_pr,
        }

        hist_pts.append(pts)
        pts_ = weighted_pts(hist_pts, weight_num=args.num, std_inv=args.std)
        pose_ = pnp(object_bbox_3d, pts_, K)
        pts__, _ = project_points(object_bbox_3d, pose_, K)
        bbox_img_ = draw_bbox_3d(img, pts__, (0, 0, 255))
        imsave(f'{str(output_dir)}/images_out_smooth/{que_id}-bbox.jpg', bbox_img_)

    #   save gen6d_inter_result
    for relative_path in [
        GEN6D_INTER_RESULT_FILENAME,
        f"images_inter/{GEN6D_INTER_RESULT_FILENAME}"
    ]:
        with open(f'{str(output_dir)}/{relative_path}', 'w') as f:
            from scy.MyJSONEncoder import MyJSONEncoder
            json.dump(gen6d_inter_result, f,
                      indent=4,
                      cls=MyJSONEncoder)

    cmd = [args.ffmpeg, '-y', '-framerate', '30', '-r', '30',
           '-i', f'{output_dir}/images_out_smooth/%d-bbox.jpg',
           '-c:v', 'libx264', '-pix_fmt', 'yuv420p', f'{output_dir}/video.mp4']
    subprocess.run(cmd)


if __name__ == "__main__":

    
    import os

    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str,
                        default=f'configs/gen6d_pretrain.yaml')
    parser.add_argument('--database', type=str,
                        default=f"{DATABASE_TYPE}/{NAME}")
    parser.add_argument('--output', type=str,
                        default=output_dir)

    # input video process
    parser.add_argument('--video', type=str,
                        # default=f"data/{DATABASE_TYPE}/video/mouse-test.mp4")
                        # default=f"data/{DATABASE_TYPE}/mouse-test.mp4")
                        default=video_path)
    parser.add_argument('--resolution', type=int, default=960)
    parser.add_argument('--transpose', action='store_true',
                        dest='transpose', default=False)

    # smooth poses
    parser.add_argument('--num', type=int, default=5)
    parser.add_argument('--std', type=float, default=2.5)

    parser.add_argument('--ffmpeg', type=str, default='ffmpeg')
    args = parser.parse_args()
    gen6dGlobal.args = args
    main(args)
