import root_config
from .dataset_imports import *
from .navi_util.data_util import load_scene_data,camera_matrices_from_annotation

class NaviDatabase(BaseDatabase) :
    """
    """
    def __init__(self, obj_with_scene :str,):
        """
        obj_with_scene: 'obj(scene)'
        """
        assert obj_with_scene.endswith(")")
        assert obj_with_scene.count('(')==1
        assert obj_with_scene.count(')')==1
        #
        obj,scene = obj_with_scene.split('(')
        scene = scene.replace(')','')
        #
        DATASET_ROOT = root_config.dataPath_navi
        self._obj_with_scene = obj_with_scene  # bed001,bed002,....
        self.obj = obj  # bed001,bed002,....
        self.scene = scene
        self._dir = Path(f'{DATASET_ROOT}/{self.obj}/{self.scene}')
        #
        folder__images_after_exif_transpose=self._dir/'scy_images_after_exif_transpose'
        ttt355=folder__images_after_exif_transpose
        if   os.path.exists(folder__images_after_exif_transpose):
            ttt355=None
        else:
            os.mkdir(folder__images_after_exif_transpose)
        #
        annotations, _, image_names = load_scene_data(
            obj, scene, DATASET_ROOT, max_num_images=None,
            folder__images_after_exif_transpose=ttt355,
        )
        del ttt355
        assert len(self._imgFullPaths_2_img_ids__A(   list(folder__images_after_exif_transpose.glob('*.jpg')),SUFFIX='.jpg'  ))==len(image_names),'可能是创建folder__images_after_exif_transpose时还没创建完就被终止了，导致有文件夹但里面文件数目不对'
        self.imageFullpaths=[ str(folder__images_after_exif_transpose/i) for i in image_names]
        self.maskFullpaths=[ str(self._dir/'masks'/(i.replace('.jpg','.png'))) for i in image_names]
        self.poses=[]
        self.Ks=[]
        for i,image_name in enumerate(image_names):
            annotation=annotations[i]
            assert image_name==annotation['filename']
            object_to_world, K = camera_matrices_from_annotation(annotation)
            object_to_world=object_to_world[:3,:]
            self.poses.append(object_to_world)
            self.Ks.append(K)
        del image_names
        self._img_ids = self._imgFullPaths_2_img_ids__A(self.imageFullpaths,check=True,SUFFIX=".jpg")
        def scy_debug_check_pose_in_database():
            _INTERVAL = 3
            # if(obj=="bed_002"):
            #     l_key =list(range(26,173,_INTERVAL))
            # else:
            #     l_key=list(range(0,len(self._img_ids),_INTERVAL))
            #     TMP_N=200
            #     if(len(l_key)>TMP_N):
            #         l_key=l_key[:TMP_N]
            l_key = list(range(0, len(self._img_ids), _INTERVAL))
            l_key=[self._img_ids[k]for k in l_key]
            TMP_N = 20
            if (len(l_key) > TMP_N):
                l_key = l_key[:TMP_N]
            l_t=[self.get_pose(k)[:,3] for k in l_key]
            # print("l_t",l_t)
            l_w2c = [np.concatenate([self.get_pose(k), np.array([[0, 0, 0, 1]])], axis=0) for k in l_key]
            l_w2c_i = [np.linalg.inv(w2c) for w2c in l_w2c]#c2w
            l_t2=[w2c_i[:,3] for w2c_i in l_w2c_i]
            l__cameraX_inW=[w2c[0,:3] for w2c in l_w2c]
            l__cameraY_inW=[w2c[1,:3] for w2c in l_w2c]
            l__cameraZ_inW=[w2c[2,:3] for w2c in l_w2c]
            from vis.vis_rel_pose import vis_w2cPoses
            param = dict(
                l_w2c=l_w2c,
                y_is_vertical=0,
            )
            view0 = vis_w2cPoses(**param, no_margin=1, )
            view1 = vis_w2cPoses(**param, no_margin=1, kw_view_init=dict(elev=30, azim=60))
            view2 = vis_w2cPoses(**param, no_margin=1, kw_view_init=dict(elev=15, azim=180))
            view3 = vis_w2cPoses(**param, no_margin=1, kw_view_init=dict(elev=45, azim=240))
            vis_img = cv2_util.concat_images_list(
                view0,
                view1,
                view2,
                view3,
                vert=0
            )
            cv2_util.putText(
                vis_img,
                f"{l_key}",
                (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (100, 100, 100),
            )
            debug_imsave(root_config.path_4debug + f"/{self.__class__.__name__}-{obj_with_scene}/poses.jpg", vis_img)
            def ttt436(l_key):
                ret=[  ]
                MAX_W = 200
                for k in l_key:
                    ttt_img=self.get_whiteBg_maskedImage(k)
                    old_w=ttt_img.shape[1]
                    w=min(MAX_W,old_w)
                    tmp_scale=w/old_w
                    ttt_img=cv2.resize(ttt_img, dsize=None, fx=tmp_scale, fy=tmp_scale)
                    ret.append(ttt_img)
                return ret
            l_whiteBg_maskedImage=ttt436(l_key)
            img_num_per_row=int(math.sqrt(len(l_whiteBg_maskedImage)))
            vis_img2=cv2_util.concat_images_list(*l_whiteBg_maskedImage,vert=0,img_num_per_row=img_num_per_row)
            debug_imsave(root_config.path_4debug + f"/{self.__class__.__name__}-{obj_with_scene}/images.jpg", vis_img2)
            print(1)
        # scy_debug_check_pose_in_database()

    def get_K(self, img_id):
        return self.Ks[img_id]
    def get_pose(self, img_id):
        return self.poses[img_id]
    def get_img_ids(self):
        return self._img_ids.copy()
    def get_image_full_path(self, img_id):
        return self.imageFullpaths[img_id]
    def get_mask_full_path(self, img_id):
        return self.maskFullpaths[img_id]




class NaviDataset(LinemodDataset):
    def __init__(self, category: str):
        # super().__init__(category)
        self.sequence_list = [""]
        self.database = NaviDatabase(obj_with_scene=category, )
