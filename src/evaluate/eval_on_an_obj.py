import root_config
from gen6d.Gen6D.pipeline import Estimator4Co3dEval
import pandas as pd
import os, shutil, sys, json
from miscellaneous.EvalResult import EvalResult
import random

from imports import *
import json
import math
import os,cv2
import os.path as osp
from Dataset.gso import GsoDataset
from Dataset.navi import NaviDataset
import numpy as np
import torch
from tqdm.auto import tqdm
from skimage.io import imsave, imread


from vis.vis_rel_pose import vis_w2cRelPoses,vis_w2cPoses




def get_permutations(num_images, eval_time=False):
    if not eval_time:
        permutations = []
        for i in range(1, num_images):
            for j in range(num_images - 1):
                if i > j:
                    permutations.append((j, i))
    else:
        permutations = []
        for i in range(0, num_images):
            for j in range(0, num_images):
                if i != j:
                    permutations.append((j, i))

    return permutations
def eval_on_an_obj(
        category,
        model_name:str='e2vg',
        **kwargs,
):
    # print(root_config.SPLIT)
    evalResult = EvalResult(model_name=model_name, idSuffix=root_config.idSuffix, category=category)
    if (kwargs['vis_include_theOther']):  # theOther: E2VG<->relpose++
        theOther = "relpose++" if model_name == "E2VG" else "E2VG"
        theOther_evalResult = EvalResult(model_name=theOther, idSuffix=kwargs['theOther_idSuffix'], category=category)

    device = root_config.DEVICE if torch.cuda.is_available() else "cpu"
    if (root_config.DATASET == "co3d"):
        # from dataset.co3d_v2 import Co3dDataset
        
        #     split=root_config.SPLIT,
        #     category=category,
        #     eval_time=1,
        #     normalize_cameras=True,
        #     first_camera_transform=False,
        # )
        dataset = Co3dv2Dataset(category=category, )
        # with open(ORDER_PATH.format(sample_num=sample_num, category=category)) as f:
        #     order = json.load(f)
    elif root_config.DATASET == "linemod":
        dataset = LinemodDataset(category=category)
    elif root_config.DATASET == "omni":
        dataset = OmniDataset(category=category)
    elif root_config.DATASET == "idposeAbo":
        dataset = IdposeAboDataset(category=category)
    elif root_config.DATASET == "idposeOmni":
        dataset = IdposeOmniDataset(category=category)
    elif root_config.DATASET == "gso":
        dataset = GsoDataset(category=category)
    elif root_config.DATASET == "navi":
        dataset = NaviDataset(category=category)
    else:
        raise NotImplementedError
    if (model_name == "relpose++"):
        checkpoint_path='/sharedata/home/suncaiyi/space/cv/baseline/relpose_plus_plus_main/weights/relposepp'
        relposepp_model, _ = get_relposepp_model_Wrap(model_dir=checkpoint_path, device=device)

    # iterable = tqdm(dataset) if use_pbar else dataset
    # seqAndPair2err = {}

    def user_confirm(seq_name):
        return 1
    # pINFO("len(iterable) (seq num):",len(iterable))
    for sequence_name in dataset.sequence_list:
        if (not user_confirm(sequence_name)):
            continue
        if root_config.Q0Sipr:
            
            q0ipr=map_string_to_int(category+sequence_name+str(root_config.Q0INDEX),
                                    # -45,45)
                                    -root_config.Q0Sipr_range,root_config.Q0Sipr_range)
            print(f"add  inplane rotation  {q0ipr} to reference image")
        else:
            q0ipr=0
        l_i_frame = dataset.seq_2_l_index(sequence_name)#l_i_frame is actually _img_ids
            
        # print("len(l_i_frame)", len(l_i_frame))

        def get_pair(l_i_frame: list):
            pairs = get_permutations(len(l_i_frame), eval_time=True)
            MAX_PAIRS = root_config.MAX_PAIRS  
            np.random.seed(root_config.SEED)
            np.random.shuffle(pairs)  
            ret = []
            i_pair = 0
            while (len(ret) < MAX_PAIRS and i_pair < len(pairs)):
                pair = pairs[i_pair]
                pair = tuple(pair)
                if (root_config.Q0INDEX is not None):
                    Q0INDEX = root_config.Q0INDEX
                    assert isinstance(Q0INDEX, int)
                    pair = (Q0INDEX, pair[1])
                if (pair not in ret):
                    ret.append(pair)
                i_pair += 1
            # # ------- 4 debug ---------------------------------
            # Q1INDEXs = [80, 107, 103, 88, 45, 81, 125, 84]
            # while (1):
            #     print(f"\n\n【】are you sure to force q1 indexs={Q1INDEXs}?  (y/n)" )
            #     user_input = input()
            #     if (user_input == "y"):
            #         break
            #     elif (user_input == "n"):
            #         exit(0)
            #     else:
            #         print("invalid input")
            # ret=[]
            # for Q1INDEX in Q1INDEXs:
            #     pair = (root_config.Q0INDEX, Q1INDEX)
            #     ret.append(pair)
            # #-------- END ------------------------------------
            # print("pairs[:<=10]", ret[:10] if len(ret) > 10 else ret)
            return ret

        pairs = get_pair(l_i_frame)
        if (evalResult.exist_pairs(sequence_name, pairs) and root_config.SKIP_EVAL_SEQ_IF_EVAL_RESULT_EXIST):
            print(f"eval result exists, skip eval this seq. {category=} {sequence_name=}")
            continue
        model_estimator_instance = None
        for i_pair, (i, j) in enumerate(pairs):
            def eval_pair():
                nonlocal model_estimator_instance
                print(f"\n--------evaluating---{i_pair}/{len(pairs)}---(i, j)=({i},{j})---seq={sequence_name}---")
                if (evalResult.exist_pair(sequence_name, i, j) and root_config.SKIP_EVAL_SEQ_IF_EVAL_RESULT_EXIST):
                    print(f"eval result exists, skip eval this pair. {i_pair=}")
                    # continue
                    return
                key_frames = [l_i_frame[i], l_i_frame[j]]
                if (model_name == "relpose++"):
                    pass
                elif (model_name == "E2VG"):
                    if model_estimator_instance is None:
                        model_estimator_instance = Estimator4Co3dEval(
                            refIdWhenNormal=root_config.RefIdWhenNormal.get_id(category, sequence_name,
                                                                               root_config.refIdSuffix)
                            # output_dir=f'./co3d_SparseViewPoseEstimator_output/{id_}',
                        )
                Global.intermediate["E2VG"]["inter_img"].l = []
                if (root_config.ONLY_GEN_DO_NOT_MEASURE and i_pair > 0):  
                    # continue
                    return
                print(f"evaluating {sequence_name} {i_pair}/{len(pairs)}")
                if root_config.Q1Sipr:
                    q1ipr=map_string_to_int(category+sequence_name+str(key_frames[1]),
                                            -root_config.Q1Sipr_range,root_config.Q1Sipr_range)
                    print(f"add  inplane rotation  {q1ipr} to query image")
                else:
                    q1ipr=0
                if (model_name == "relpose++"):
                    batch = dataset.get_data(sequence_name=sequence_name, index0=key_frames[0], index1=key_frames[1],q0ipr=q0ipr,q1ipr=q1ipr,)
                    crop_params = batch["crop_params"].to(device).unsqueeze(0)
                elif (model_name == "E2VG"):
                    batch = dataset.get_data_4gen6d(sequence_name=sequence_name, ids=key_frames,q0ipr=q0ipr,q1ipr=q1ipr,)
                    Ks = batch["K"]
                    assert Ks.shape == (2, 3, 3)
                    
                    K = Ks[0]
                elif (model_name == "loftr") or model_name.startswith("mapfree"):
                    batch = dataset.get_data_4loftr(sequence_name=sequence_name, index0=key_frames[0], index1=key_frames[1],
                                                    q0ipr=q0ipr,q1ipr=q1ipr,
                                                    max_size=320,
                                                    )
                else:
                    raise  NotImplementedError
                # del q1ipr
                # Load GT
                """
                rotations = batch["relative_rotation"].to(device).unsqueeze(0)
                t31s = batch["relative_t31"].to(device).unsqueeze(0)
                """
                pose44_gt_rel = batch["relative_pose44"].to(device)
                R_gt_rel = pose44_gt_rel[:3, :3]
                T31_gt_rel = pose44_gt_rel[:3, 3:]
                assert R_gt_rel.shape == (3, 3)
                assert T31_gt_rel.shape == (3, 1)
                R_gt_rel = R_gt_rel.detach().cpu().numpy().reshape((3, 3))
                T31_gt_rel = T31_gt_rel.detach().cpu().numpy().reshape((3, 1))

                if model_name == "relpose++":
                    EVAL_FN_MAP = {
                        "pairwise": evaluate_pairwise,
                        "coordinate_ascent": evaluate_coordinate_ascent,
                    }
                    images_transformed = batch["images_transformed"].to(device).unsqueeze(0)
                    with torch.no_grad():
                        R_pred_rel, _ = EVAL_FN_MAP[mode](
                            relposepp_model,
                            images_transformed,
                            crop_params,
                            # 
                            category=category,
                            sequence_name=sequence_name
                        )
                        assert R_pred_rel.shape == (3, 3)
                    with torch.no_grad():
                        _, _, T_pred = relposepp_model(
                            images=images_transformed,
                            crop_params=crop_params,
                        )
                        assert T_pred.shape == (2, 3,)
                        t0 = T_pred[0].cpu().numpy()
                        t1 = T_pred[1].cpu().numpy()
                        # print(f"{t0=}")
                        # print(f"{t1=}")
                    T31_pred_rel=-R_pred_rel@t0+t1
                    T31_pred_rel=T31_pred_rel.reshape((3,1))
                    assert T31_pred_rel.shape==(3,1)
                elif model_name == "E2VG":
                    from infer_pair import  gen6d_imgPaths2relativeRt_B
                    def get_eleRadian(R, t):
                        camera_center = -R.T @ t
                        eleRadian = math.atan(
                            camera_center[1] / math.sqrt(camera_center[0] ** 2 + camera_center[2] ** 2))
                        return eleRadian

                    image0_path = batch["image_not_transformed_full_path"][0]
                    image1_path = batch["image_not_transformed_full_path"][1]
                    print("reference image", image0_path)
                    print("query image", image1_path)
                    # input_image_eleRadian_use_original=get_eleRadian(R=batch["R_original"][0],t=batch["T_original"][0])
                    # input_image_eleRadian_afterNorm=get_eleRadian(R=batch["R"][0],t=batch["T"][0])
                    
                    if not root_config.LOOK_AT_CROP_OUTSIDE_GEN6D:
                        raise Exception
                        R_pred_rel, T31_pred_rel, inter = gen6d_imgPaths2relativeRt(
                            estimator=model_estimator_instance,
                            K=K,
                            image0_path=image0_path,
                            image1_path=image1_path,

                            # input_image_eleRadian=  input_image_eleRadian_use_original ,
                            input_image_eleRadian=None,
                            detection_outputs=batch["detection_outputs"],
                        )
                    else:
                        if root_config.ABLATE_REFINE_ITER is not None:
                            Global.RefinerInterPoses.load_pair( sequence_name, i, j   )
                        R_pred_rel, T31_pred_rel, relative_pose, inter = gen6d_imgPaths2relativeRt_B(
                            estimator=model_estimator_instance,
                            K0=K,
                            K1=K,
                            image0_path=image0_path,
                            image1_path=image1_path,
                            bbox0=batch["bbox"][0],
                            bbox1=batch["bbox"][1],
                            input_image_eleRadian=None,
                        )
                    gen6d_pose0_w2opencv_leftMul = inter["pose0"]
                    gen6d_pose0_w2pytorch3d_leftMul = opencv_2_pytorch3d__leftMulW2cpose(gen6d_pose0_w2opencv_leftMul)
                    gen6d_pose1_w2opencv_leftMul = inter["pose1"]
                    gen6d_pose1_w2pytorch3d_leftMul = opencv_2_pytorch3d__leftMulW2cpose(gen6d_pose1_w2opencv_leftMul)
                    # R_pred_rel = opencv_2_pytorch3d__leftMulRelR(R_pred_rel)
                    relative_pose = opencv_2_pytorch3d__leftMulRelPose(relative_pose)
                    R_pred_rel = relative_pose[:3, :3]
                    T31_pred_rel = relative_pose[:3, 3:]
                elif model_name=='loftr':
                    from . import loftr
                    R_pred_rel,T31_pred_rel=loftr.Runner.run(
                        root_config.DATASET,category,sequence_name,key_frames[0],key_frames[1],
                        batch["image_full_paths"],batch["Ks"],
                        pose44_gt_rel_opencv=pytorch3d_2_opencv__leftMulRelPose(pose44_gt_rel.detach().cpu().numpy())
                    )
                    # return #stage 1
                elif    model_name.startswith("mapfree"):
                    from . import mapfree
                    R_pred_rel,T31_pred_rel=mapfree.MapfreeRunner.run(
                        root_config.DATASET,category,sequence_name,key_frames[0],key_frames[1],
                        batch["image_full_paths"],batch["Ks"],
                        pose44_gt_rel_opencv=pytorch3d_2_opencv__leftMulRelPose(pose44_gt_rel.detach().cpu().numpy()),
                        q0ipr=q0ipr,
                        q1ipr=q1ipr,
                    )
                    # return #stage 1
                else:
                    raise NotImplementedError
                R_error = compute_angular_error(R_pred_rel, R_gt_rel)
                T_error = compute_translation_error(T31_pred_rel, T31_gt_rel)
                if model_name=='E2VG':
                    _kw={
                        'RefinerInterPoses':Global.RefinerInterPoses.to_dicValue_and_clear(),
                    }
                else:
                    _kw={}
                evalResult.append_pair(
                    sequence_name, i, j,
                    R_pred_rel, R_gt_rel, R_error,
                    T31_pred_rel, T31_gt_rel, T_error,
                    key_frames,
                    #**kw
                    **_kw,
                )

                def _vis():
                    def vis_pose():
                        w2c = R_t_2_pose(R=R_pred_rel, t=[0, 0, 0])  
                        gt_w2c = R_t_2_pose(R=R_gt_rel, t=[0, 0, 0])
                        title = f"{sequence_name} i,j={i},{j} img_id0,1=={key_frames[0]},{key_frames[1]}"
                        if model_name == "E2VG":
                            l_w2c = [w2c @ gen6d_pose0_w2pytorch3d_leftMul, gt_w2c @ gen6d_pose0_w2pytorch3d_leftMul]
                            l_color = ['b', 'g']
                            if (kwargs['vis_include_theOther'] and theOther_evalResult.exist_pair(sequence_name, i, j)):
                                theOther_R_pred_rel = theOther_evalResult.get_pair__in_dic(sequence_name, i, j)[
                                    "R_pred_rel"]
                                l_w2c.append(
                                    R_t_2_pose(R=theOther_R_pred_rel, t=[0, 0, 0]) @ gen6d_pose0_w2pytorch3d_leftMul)
                                l_color.append('y')

                            # def tmp_get_elev():
                            #     refIdWhen4Elev = root_config.RefIdWhen4Elev.find_id_basedOn_cate_seq_refIdSuffix(category,
                            #                                                                                         sequence_name,
                            #                                                                                         root_config.refIdSuffix)
                            #     _dir = root_config.RefIdUtil.id2idDir(refIdWhen4Elev)
                            #     file = os.path.join(_dir, "ref", "elev.json")
                            #     with open(file, "r") as f:
                            #         dic = json.load(f)
                            #         elev = dic['rad']
                            #         zero123_input_img_path = dic['input_image_path']
                            #     return elev, zero123_input_img_path
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
                            title += f".elev={tmp_eleRadian * 180 / math.pi:.2f}°{q0ipr=}{q1ipr=}\nzero123 base img={Path(zero123_input_img_path).relative_to(root_config.path_root)}"  # azim=-60, elev=30 by default
                        elif model_name == "relpose++":  
                            l_w2c = [w2c, gt_w2c, np.eye(4)]
                            l_color = ["b", "g", "white"]
                        elif model_name == "loftr" or model_name.startswith("mapfree"):
                            l_w2c = [w2c, gt_w2c, np.eye(4)]
                            l_color = ["b", "g", "white"]
                        else:
                            raise NotImplementedError
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
                        # np.save(pose_save_path_format.format("w2c")        , w2c)
                        # np.save(pose_save_path_format.format("gt_w2c")     , gt_w2c)
                        if model_name == "E2VG":
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
                        elif model_name == "relpose++":
                            ret = cv2_util.concat_images_list(
                                view0,
                                cv2_util.concat_images_list(
                                    view1, view2, view3,
                                    vert=0
                                ),
                                vert=1,
                            )
                        elif model_name == "loftr" or model_name.startswith("mapfree"):
                            ret = cv2_util.concat_images_list(
                                view0,
                                cv2_util.concat_images_list(
                                    view1, view2, view3,
                                    vert=0
                                ),
                                vert=1,
                            )
                        else:
                            raise NotImplementedError
                        return ret

                    vis_w2cRelPoses_img = vis_pose()
                    cv2.putText(
                        vis_w2cRelPoses_img, f"R_error={str(R_error)}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 0),
                        # thickness,
                        # lineType
                    )
                    cv2_util.putText(
                        vis_w2cRelPoses_img,
                        f"[{model_name}]R_pred_rel=\n{R_pred_rel}\nR_gt_rel=\n{R_gt_rel}",
                        (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 0),
                    )
                    if (kwargs['vis_include_theOther'] and theOther_evalResult.exist_pair(sequence_name, i, j)):
                        (
                            theOther_R_pred_rel, theOther_R_gt_rel, theOther_R_error,
                            theOther_T31_pred_rel, theOther_T31_gt_rel, theOther_T_error,
                            theOther_key_frames
                        ) = theOther_evalResult.get_pair__in_tuple(
                            sequence_name, i, j)
                        cv2.putText(
                            vis_w2cRelPoses_img, f"[{theOther}]R_error={str(theOther_R_error)}",
                            (vis_w2cRelPoses_img.shape[1] - 500, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 0, 0),
                        )
                        cv2_util.putText(
                            vis_w2cRelPoses_img,
                            f"[{theOther}]R_pred_rel=\n{theOther_R_pred_rel}\nR_gt_rel=\n{theOther_R_gt_rel}",
                            (vis_w2cRelPoses_img.shape[1] - 500, 80),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 0, 0),
                        )
                    if (model_name == "relpose++"):
                        vis_img = vis_w2cRelPoses_img
                    elif (model_name == "E2VG"):
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
                        def get_resized_image0(image0_path):
                            ret=imread(image0_path)
                            MAX_W = 300
                            old_w = ret.shape[1]
                            w = min(MAX_W, old_w)
                            tmp_scale = w / old_w
                            ret = cv2.resize(ret, dsize=None, fx=tmp_scale, fy=tmp_scale)
                            return ret
                        if root_config.one_SEQ_mul_Q0__one_Q0_mul_Q1:
                            vis_img = cv2_util.concat_images_list(
                                cv2_util.concat_images_list(
                                    get_resized_image0(image0_path),
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
                                    get_resized_image0(image0_path),
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
                    elif model_name == "loftr" or model_name.startswith("mapfree"):
                        vis_img = vis_w2cRelPoses_img
                    else:
                        raise NotImplementedError
                    tmp_save_dir = osp.join(root_config.evalVisPath,
                                            f"[{model_name}-{root_config.idSuffix}]{category}-{sequence_name}")
                    os.makedirs(tmp_save_dir, exist_ok=True)
                    tmp_save_path = osp.join(tmp_save_dir,
                                             f"[{model_name}-{root_config.idSuffix}]{category}-{sequence_name}-{key_frames[0]},{key_frames[1]}.jpg")
                    imsave(tmp_save_path, vis_img)
                    print(f"visual result: {tmp_save_path}")
                    Global.poseVisualizer1.clear()
                def vis_gate():
                    cfg_vis=root_config.VIS
                    if isinstance(cfg_vis,bool):
                        return cfg_vis
                    if isinstance(cfg_vis,int):
                        assert cfg_vis==0 or cfg_vis==1
                        return cfg_vis
                    if isinstance(cfg_vis,float):
                        assert 0<=cfg_vis<=1
                        if random.random()<cfg_vis:
                            return True
                        else:
                            return False
                    raise ValueError
                if vis_gate():
                    _vis()

            eval_pair()
        evalResult.dump()
        evalResult.dump_acc()
        if (model_name == "E2VG"):
            del model_estimator_instance  
    EvalResult.AllAcc.append_acc_path(evalResult.category_acc_json,evalResult.category,evalResult.idSuffix)

