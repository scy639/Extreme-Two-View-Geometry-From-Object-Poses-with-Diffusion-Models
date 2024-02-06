import root_config,os,sys,shutil
from skimage.io import imread, imsave
import numpy as np
import os,sys,math,functools,inspect
from pathlib import Path
import PIL
def debug_imsave(path__rel_to__path_4debug,arr):
    #
    if isinstance(path__rel_to__path_4debug,Path):
        path__rel_to__path_4debug=str(path__rel_to__path_4debug)
    assert isinstance(path__rel_to__path_4debug,str)
    if not(path__rel_to__path_4debug.endswith('.jpg') or
        path__rel_to__path_4debug.endswith('.png') ):
        print('[warning] incorrect image format')
    #
    if isinstance(arr,PIL.Image.Image):
        # Convert PIL image to NumPy array
        arr = np.asarray(arr)
    assert isinstance(arr,np.ndarray)
    #
    full_path=os.path.join(root_config.path_4debug,path__rel_to__path_4debug)
    os.makedirs(os.path.dirname(full_path),exist_ok=1)
    print(f"[debug_imsave]saving...",end=" ",flush = True)
    imsave(full_path,arr)
    print(f"save to \"{full_path}\"")
