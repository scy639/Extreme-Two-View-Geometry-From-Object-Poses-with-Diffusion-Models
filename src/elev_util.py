import os.path
import glob
from imports import *
import json
if 1:
    from import_util import is_in_sysPath
    # if(is_in_sysPath(path=os.path.abspath(os.path.join(os.path.dirname(__file__),os.path.pardir)))):
    #     from oee.utils.elev_est_api import elev_est_api
    # else:
    #     from ..oee.utils.elev_est_api import elev_est_api
    from oee.utils.elev_est_api import elev_est_api,ElevEstHelper
import  numpy  as np
def _sample_sphere(num_samples, begin_elevation = 0):
    """ sample angles from the sphere
    reference: https://zhuanlan.zhihu.com/p/25988652?group_id=828963677192491008
    """
    ratio = (begin_elevation + 90) / 180
    num_points = int(num_samples // (1 - ratio))
    phi = (np.sqrt(5) - 1.0) / 2.
    azimuths = []
    elevations = []
    for n in range(num_points - num_samples, num_points):
        z = 2. * n / num_points - 1.
        azimuths.append(2 * np.pi * n * phi % (2 * np.pi))
        elevations.append(np.arcsin(z))
    return np.array(azimuths), np.array(elevations)
def _get_l_ele_azimuth_inRadian(num_samples, begin_elevation = 0):
    azimuths,elevations=_sample_sphere(num_samples, begin_elevation)
    l_ele_azimuth_inRadian=np.stack([elevations,azimuths],axis=1)
    return l_ele_azimuth_inRadian
import math
def eleRadian_2_baseXyz_lXyz(eleRadian:float):#xyz is in degree!
    eleDegree=eleRadian*180/math.pi
    base_xyz=(-eleDegree,0,0)
    l_xyz=[]
    l_ele_azimuth_inRadian=_get_l_ele_azimuth_inRadian(
        # num_samples=128
        num_samples=root_config.NUM_REF
    )
    if root_config.ELEV_RANGE:#only keep elev in ELEV_RANGE
        assert len(root_config.ELEV_RANGE)==2
        # l_ele_azimuth_inRadian=np.array(l_ele_azimuth_inRadian)
        # to degree
        l_ele_azimuth_inDeg=np.rad2deg(l_ele_azimuth_inRadian)
        l_ele_azimuth_inDeg=l_ele_azimuth_inDeg[l_ele_azimuth_inDeg[:,0]>=root_config.ELEV_RANGE[0]]
        l_ele_azimuth_inDeg=l_ele_azimuth_inDeg[l_ele_azimuth_inDeg[:,0]<=root_config.ELEV_RANGE[1]]
        # to radian
        l_ele_azimuth_inRadian=np.deg2rad(l_ele_azimuth_inDeg)
        del l_ele_azimuth_inDeg
        # l_ele_azimuth_inRadian=to_list_to_primitive(l_ele_azimuth_inRadian)
    #
    for ele_azimuth_inRadian in l_ele_azimuth_inRadian:
        x0=base_xyz[0]
        y0=base_xyz[1]
        x1=-ele_azimuth_inRadian[0]
        y1=ele_azimuth_inRadian[1]
        x1=x1*180/math.pi
        y1=y1*180/math.pi
        l_xyz.append((x1-x0,y1-y0,0))
    return base_xyz,l_xyz
#------------one2345-----------------------



def imgPath2elevRadian(K,input_image_path,run4gen6d_main,id_):
    id2 = f"4elev-{id_}-{os.path.basename(input_image_path)}"
    
    output_dir = os.path.join(root_config.dataPath_gen6d, f'{id2}/ref')
    def getFourNearImagePaths( ):
        # delta_x_2 = [-10, 10, 0, 0]
        # delta_y_2 = [0, 0, -10, 10]
        DELTA=10
        if 'tmp_4_ipr_ex1' not in Global.anything:
            assert DELTA==10
        delta_x_2 = [-DELTA, DELTA, 0, 0]
        delta_y_2 = [0, 0, -DELTA, DELTA]
        ElevEstHelper.DELTA=DELTA
        
        l_xyz=[(delta_x_2[i],delta_y_2[i],0) for i in range(4)]
        l__path_output_im=run4gen6d_main(
            id2,
            input_image_path,
            # output_dir=output_dir,
            output_dir=None,
            num_samples=1,
            l_xyz=l_xyz,
            base_xyz=(0,0,0),
            ddim_steps=75,
            K=K,
            only_gen=True,  # dont crop,re center etc
        )
        # ret=[os.path.join(output_dir,f"{i}.jpg") for i in range(4)]
        assert len(l_xyz)==len(l__path_output_im)
        return l__path_output_im
    if root_config.Cheat.force_elev:
        print(f"[warning] You enable {root_config.Cheat.force_elev=}")
        fourNearImagePaths='(Cheat.force_elev)'
        elev=root_config.Cheat.force_elev
    else:
        fourNearImagePaths=getFourNearImagePaths()
        elev = elev_est_api( fourNearImagePaths, 
                            # min_elev=30, max_elev=150,
                            # min_elev=20, max_elev=160,
                            min_elev=90-79, max_elev=90-0, 
                            # min_elev=1, max_elev=160,
                            )
    elev_deg:int=elev
    elev = np.deg2rad(elev)
    #info: rad,degree,output_dir,output_imgs,output_json
    os.makedirs(output_dir,exist_ok=1)
    output_json=os.path.join(output_dir,"elev.json")
    info={
        "rad":elev,
        "degree":elev_deg, 
        "input_image_path":input_image_path,
        "output_dir":output_dir,
        "output_imgs":fourNearImagePaths,
        "output_json":output_json,
    }
    print("[imgPath2elevRadian]",json.dumps(info,indent=4))
    if 'tmp_4_ipr_ex1' in Global.anything:
        raise Exception("tmp_4_ipr_ex1",)
    with open(output_json,"w") as f:
        json.dump(info,f,indent=4)
    #tmp4SecondTimeDebugElev
    if not hasattr(Global,'tmp4SecondTimeDebugElev'):
        Global.tmp4SecondTimeDebugElev=[]
    Global.tmp4SecondTimeDebugElev.append(info)
    return elev
def eleRadian_2_base_w2c(eleRadian):
    base_xyz, useless__l_xyz = eleRadian_2_baseXyz_lXyz(eleRadian=eleRadian)
    pose=xyz2pose(*base_xyz)
    assert pose.shape==(3,4)
    pose=np.concatenate([pose,np.array([[0,0,0,1]])],axis=0)
    return pose