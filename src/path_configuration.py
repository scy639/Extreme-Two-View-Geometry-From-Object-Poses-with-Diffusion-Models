
import os
path_root=os.path.dirname(os.path.abspath(__file__))


# the parent folder of GSO objects folders (GSO_alarm,GSO_backpack,...)
dataPath_gso=os.path.join(path_root,"../gso-renderings")
# the parent folder of GSO objects folders (GSO_alarm,GSO_backpack,...)
dataPath_navi=os.path.join(path_root,"../NAVI/v1")

weightPath_zero123=os.path.join(path_root,"../105000.ckpt")
weightPath_selector=os.path.join(  path_root ,"gen6d/Gen6D/data/model/selector_pretrain/model_best.pth")
weightPath_loftr=os.path.join(path_root  ,"../indoor_ds_new.ckpt")

