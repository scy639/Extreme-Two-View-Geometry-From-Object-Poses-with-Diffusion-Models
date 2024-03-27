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
        DATASET_ROOT =  os.path.abspath  ( root_config.dataPath_navi)
        self._obj_with_scene = obj_with_scene  # bed001,bed002,....
        self.obj = obj  # bed001,bed002,....
        self.scene = scene
        self._dir = Path(f'{DATASET_ROOT}/{self.obj}/{self.scene}')
        assert self._dir.exists(),f"{str(self._dir)} does not exist"
        #
        folder__images_after_exif_transpose=self._dir/'_images_after_exif_transpose'
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
