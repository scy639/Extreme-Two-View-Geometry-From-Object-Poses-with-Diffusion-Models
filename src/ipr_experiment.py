from infer_pairs import *
import torch
if __name__ == "__main__":
    
    root_config.USE_CONFIDENCE = 0
    root_config.SKIP_GEN_REF_IF_REF_FOLDER_EXIST = 0
    also_infer_and_vis_otherModel = None

    Global.anything['tmp_4_ipr_ex1'] = '任意'


    def _f(refId, l_path_bbox):
        vis_result_folder = Path(root_config.evalVisPath) / '_infer_custom' / refId
        os.makedirs(vis_result_folder, exist_ok=True)
        infer_pairs(
            q0ImageData=ImageData(path=l_path_bbox[0][0], K=None, bbox=l_path_bbox[0][1]),
            l_q1ImageData=[ImageData(path=path, K=None, bbox=bbox) for path, bbox in l_path_bbox[1:]],
            refId=refId,
            #
            auto_K=True,
            auto_bbox=True,
            #
            vis_result_folder=vis_result_folder,
            #
            also_infer_and_vis_otherModel=also_infer_and_vis_otherModel,
            # **kw
            do_not_check_K_equivalent=True
        )


    """refId = 'chicken_racer'
    l_path_bbox = [
        (
            '/sharedata/home/suncaiyi/space/cv/custom_data/4_ipr_ex/chicken_racer/0.jpg',
            (0.06205493387589013, 0.028484231943031537, 0.9216683621566633, 0.9796541200406917,),
        ),
        (
            '/sharedata/home/suncaiyi/space/cv/custom_data/4_ipr_ex/chicken_racer/1.jpg',
            (0.04679552390640895, 0.029501525940996948, 0.9298067141403866, 0.970498474059003,),
        ),
    ]
    _f(refId, l_path_bbox)"""
    Global.anything['4_ipr_ex1__l_tuples'] = []


    def _f2(path0, path1, _base_refId):
        # l_angle_in_degree = [0, 45, 90, 180, 270, 315]
        l_angle_in_degree = [1, 45, 90, 180, 270, 315]
        # l_angle_in_degree = [0, 5, 10, 15, 345,350, 355]
        Global.anything['4_ipr_ex1__l_tuples'].append([])
        img0 = Image.open(path0)
        for rot_angle_in_degree in l_angle_in_degree:
            print(f"\n\n\nrot_angle_in_degree={rot_angle_in_degree}")
            img0_rot = img0.rotate(rot_angle_in_degree, fillcolor=(255, 255, 255))
            path0_rot = f"{path0}.rot{rot_angle_in_degree}.jpg"
            img0_rot.save(path0_rot)
            refId = f'{_base_refId}-{rot_angle_in_degree}'
            l_path_bbox = [
                (path0_rot, None),
                (path1, None),
            ]
            Global.anything['4_ipr_ex1__l_tuples'][-1].append([])
            try:
                _f(refId, l_path_bbox)
            except torch.linalg.LinAlgError as e:
                print(f"torch.linalg.LinAlgError= {e}")
                Global.anything['4_ipr_ex1__l_tuples'][-1][-1]+=['torch.linalg.LinAlgError',path0_rot]
                continue
            except Exception as e:
                if 'tmp_4_ipr_ex1' in e.args:
                    Global.anything['4_ipr_ex1__l_tuples'][-1][-1]+=[path0_rot]
                    continue
                else:
                    print(f"e= {e}")
                    if "unsupported operand type(s) for -: 'int' and 'NoneType'" in str(e):
                        Global.anything['4_ipr_ex1__l_tuples'][-1][-1]+=["unsupported operand type(s) for -: 'int' and 'NoneType'",path0_rot]
                        continue
                    raise e
        print(f"_base_refId={_base_refId}:\n",json.dumps(Global.anything['4_ipr_ex1__l_tuples'],indent=4 ) ,
              scy_use_primitive=True,flush=1)


    """_f2('/sharedata/home/suncaiyi/space/cv/custom_data/4_ipr_ex/chicken_racer/0.jpg',
        '/sharedata/home/suncaiyi/space/cv/custom_data/4_ipr_ex/chicken_racer/1.jpg',
        # 'chicken_racer')
        # 'chicken_racer-DELTA30')
        'chicken_racer-DELTA20')"""
    """_f2('/sharedata/home/suncaiyi/space/cv/custom_data/4_ipr_ex/tractor/0.jpg',
    '/sharedata/home/suncaiyi/space/cv/custom_data/4_ipr_ex/tractor/1.jpg',
    'tractor')
    # 'tractor-DELTA30')
    # 'tractor-DELTA20')"""

    l__base_refId__path0 = (
        ("gso_alarm",
         "/sharedata/home/suncaiyi/space/cv/custom_data/4_ipr_ex/many/[gso][GSO_alarm][]008_warp.jpg",),
        ("gso_grandfather",
         "/sharedata/home/suncaiyi/space/cv/custom_data/4_ipr_ex/many/[gso][GSO_grandfather][]097_warp.jpg",),
        ("schleich_lion",
         "/sharedata/home/suncaiyi/space/cv/custom_data/4_ipr_ex/many/[navi][schleich_lion_action_figure][]009_warp.jpg",),
        ("schleich_lion2",
         "/sharedata/home/suncaiyi/space/cv/custom_data/4_ipr_ex/many/[navi][schleich_lion_action_figure][]014_warp.jpg",),
        ("soldier_wood",
         "/sharedata/home/suncaiyi/space/cv/custom_data/4_ipr_ex/many/[navi][soldier_wood_showpiece][]000_warp.jpg",),
        ("water_gun_y",
         "/sharedata/home/suncaiyi/space/cv/custom_data/4_ipr_ex/many/[navi][water_gun_toy_yellow][]012_warp.jpg",),
    )
    PATH1 = '/sharedata/home/suncaiyi/space/cv/custom_data/4_ipr_ex/many/1_arbitrary.jpg'
    for _base_refId,path0 in  l__base_refId__path0:
        # _base_refId+='-DELTA45'
        _base_refId+='-DELTA30(bugFixed)2'
        # _base_refId+='-DELTA30(bugFixed)2(repeat2)'
        # _base_refId+='-DELTA45(bugFixed)2'
        _f2(path0,PATH1,_base_refId,)

