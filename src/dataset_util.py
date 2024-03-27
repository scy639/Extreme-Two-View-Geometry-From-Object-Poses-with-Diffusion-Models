import json
from misc_util import truncate_str

def _process_s(s:str,I_TASK=0,NUM_TASK=1):
    """
        s1.remove space (lstrip)
        s2.ignore empty line
        s2.ignore #
    """
    # ------------------s
    assert isinstance(s,str)
    _s = s.split("\n")
    s = []
    for string in _s:
        string = string.lstrip()
        if string == "":
            continue
        if string.startswith("#"):
            continue
        assert "#" not in string
        assert string.count(':') <= 1
        assert string.count('/') <= 1
        s.append(string)
    # ------------------TASK
    I_START = len(s) // NUM_TASK * I_TASK
    I_END = len(s) // NUM_TASK * (I_TASK + 1) if I_TASK != NUM_TASK - 1 else len(s)
    s = s[I_START:I_END]
    return s
def get___l__category_seq_q0(s:str,I_TASK=0,NUM_TASK=1,log=True):
    assert isinstance(s,str)
    s=_process_s(s,I_TASK,NUM_TASK)
    # ------------------category_2_seq_q0
    l__category_seq_q0 = []
    for string in s:
        if ":" in string:
            category, q0 = string.split(":")
            q0 = int(q0)
        else:
            category = string
            q0 = 0
        if "/" in category:
            category, seq = string.split("/")
        else:
            category=category
            seq = ''
        l__category_seq_q0.append((category, seq, q0))
    if log:
        to_log=str(l__category_seq_q0)  .replace('),', '),\n')
        to_log=truncate_str(to_log,100)
        print(
            "l__category_seq_q0=",
            #   l__category_seq_q0
            #   json.dumps(l__category_seq_q0,indent=4)
            to_log,
        )
    return l__category_seq_q0

def get_category2q0(s:str,I_TASK=0,NUM_TASK=1,log=True):
    s=_process_s(s,I_TASK,NUM_TASK)
    category2q0 = {}
    for string in s:
        if string.startswith("#"):
            continue
        assert "#" not in string
        if ":" in string:
            category, q0 = string.split(":")
            q0 = int(q0)
        else:
            category = string
            q0 = 0
        category2q0[category] = q0

    # ------------------category_2_seq_q0
    category2q0 = {}
    for string in s:
        assert '/' not in string
        if ":" in string:
            category, q0 = string.split(":")
            q0 = int(q0)
        else:
            category = string
            q0 = 0
        category2q0[category] = q0
    if log:
        print(
            "category2q0=",
            #   category2q0
            json.dumps(category2q0,indent=4)
        )
    return category2q0
def get__datasetName_2_l_cate_seq_Q0INDEX(datasetNames:list,datasetName_2_s:dict,log=True)-> dict:
    assert isinstance(datasetNames,list)
    assert isinstance(datasetName_2_s,dict)
    ret={}
    for datasetName in datasetNames:
        s=datasetName_2_s[datasetName]
        l_cate_seq_Q0INDEX=get___l__category_seq_q0(s=s,log=False)
        ret[datasetName]=l_cate_seq_Q0INDEX
    if log:
        print(f"datasetName_2_l_cate_seq_Q0INDEX= {ret}")
    return ret
def get__l__datasetName_cate_seq_Q0INDEX(datasetNames:list,datasetName_2_s:dict,
                                        #  log=True,
                                         log=False,
                                         )-> list:
    assert isinstance(datasetNames,list)
    assert isinstance(datasetName_2_s,dict)
    dic=get__datasetName_2_l_cate_seq_Q0INDEX(datasetNames,datasetName_2_s,log=False)
    ret=[]
    for datasetName,l_cate_seq_Q0INDEX in dic.items():
        for cate,seq,Q0INDEX in l_cate_seq_Q0INDEX:
            item = (datasetName, cate, seq, Q0INDEX)
            if item not in ret:  # Check if the item already exists in ret
                ret.append(item)
            else:
                assert 0,item
    if log:
        to_log=ret
        to_log=truncate_str(str(to_log),100)
        print(f"l__datasetName_cate_seq_Q0INDEX= {to_log}")
    return ret

class MyTestset:
    datasetName_2_s={
        'navi':""" 
                    chicken_racer(multiview_12_pixel_5)
                    tractor_green_showpiece(multiview_03_pixel_6pro)
                    schleich_spinosaurus_action_figure(multiview_10_ipad_5)
                    garbage_truck_green_toy_s(multiview_02_pixel_6pro)
                    can_kernel_corn(multiview_01_pixel_4a)
                    circo_fish_toothbrush_holder_14995988(multiview_14_pixel_5)
                    pumpkin_showpiece_s(multiview_00_pixel_4a)
                    school_bus(multiview_10_canon_t4i)
                    paper_weight_flowers_showpiece(multiview_03_pixel_6pro)
                    remote_control_toy_car_s(multiview_04_pixel_6pro)
                    schleich_hereford_bull(multiview_06_ipad_5)
                    duck_bath_yellow_s(multiview_02_pixel_6pro)
                    fire_engine_toy_red_yellow_s(multiview_00_pixel_4a)
                    schleich_african_black_rhino(multiview_09_canon_t4i)
                    soldier_wood_showpiece(multiview_04_pixel_6pro)
                    dino_5(multiview_09_pixel_5)
                    bunny_racer(multiview_00_pixel_5)
                    schleich_lion_action_figure(multiview_07_pixel_5):9
                    water_gun_toy_yellow(multiview_00_pixel_4a):3
                    keywest_showpiece_s(multiview_02_pixel_6pro):9
                    well_with_leaf_roof_showpiece(multiview_03_pixel_6pro):0
                    steps_small_showpiece(multiview_04_pixel_6pro):3
                    welcome_sign_mushrooms(multiview_00_pixel_4a):12
                    schleich_bald_eagle(multiview_07_pixel_5):0
                    3d_dollhouse_sink(multiview_04_pixel_4xl):27
                    dino_4(multiview_10_pixel_4xl):18
                    water_gun_toy_green(multiview_01_pixel_4a):9
        """,
        'gso':
            """GSO_alarm:8
               GSO_backpack:8
               GSO_blocks:63
               GSO_chicken
               GSO_cream:89
               GSO_elephant:6
               GSO_grandfather:63
               GSO_grandmother
               GSO_lion:47
               GSO_lunch_bag:8
               GSO_mario:55
               GSO_oil:3
               GSO_school_bus1:63
               GSO_school_bus2:8
               GSO_shoe:71
               GSO_soap:46
               GSO_sofa:50
               GSO_sorter:46
               GSO_sorting_board:73
               GSO_teapot:87
               GSO_toaster:86
               GSO_train:63
               GSO_turtle:91""",
    }
    l__datasetName_cate_seq_Q0INDEX=get__l__datasetName_cate_seq_Q0INDEX(datasetNames=list(datasetName_2_s.keys()),datasetName_2_s=datasetName_2_s)
    