
import sys,os
os.environ["CUDA_VISIBLE_DEVICES"] = '5'
cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(cur_dir, "src"))
from pathlib import Path
cur_dir=Path(cur_dir)
import root_config
from infer_pairs import infer_pairs_wrapper



#------------------configs----------------------
root_config.CONSIDER_IPR=False # IPR means inplane rotation. if the object in reference image is not oriented correctly,  set CONSIDER_IPR=True to enable inplane rotation predictor; if oriented correctly, set set CONSIDER_IPR=0 to skip inplane rotation estimation to save time
# If GPU out of memory, decrease the following values:
root_config.SAMPLE_BATCH_SIZE = 32
root_config.SAMPLE_BATCH_B_SIZE = 9









referenceImage_path_bbox = (cur_dir/"media/example_custom_data/0.png", (27, 92, 192, 193),)
queryImages_path_bbox = [
    (cur_dir/"media/example_custom_data/1.png", (84, 79, 169, 202),),
    (cur_dir/"media/example_custom_data/2.png", (54, 36, 179, 218),),
    (cur_dir/"media/example_custom_data/3.png", (31, 69, 217, 194),),
    (cur_dir/"media/example_custom_data/4.png", (48, 89, 178, 202),),
    (cur_dir/"media/example_custom_data/5.png", (73, 59, 176, 229),),
]
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
    bbox=(x0,y0,x1,y1), in pixel , or relative to the image size. 
    You should provide at least one query image
:param refId: 
    refId indentify the building result of a reference image. 
    If {refId} has been built before, then the program will reuse the building result to save time (building from a reference image takes >1min on a single 3090 GPU)
:return: 
    relativePoses=[
        relative pose from reference image to query image 1,
        relative pose from reference image to query image 2,
        relative pose from reference image to query image 3,
        ...
    ]
    X-query_i = relativePoses[i] @ X-reference. X means point in the coordinate system of camera. 
    The camera follows {cameraConvention} convention, cameraConvention is 'opencv' by default
"""
relativePoses = infer_pairs_wrapper(
    referenceImage_path_bbox, queryImages_path_bbox,
    refId='lion',
)
print(relativePoses)