import json
from misc_util import truncate_str

def _process_s(s:str,I_TASK=0,NUM_TASK=1):
    """
        s1.去除行首all空格
        s2.忽略空行
        s2.忽略#行
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
    """
    适用于无seq的（co3dv2外的
    """
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
        'navi':"""#navi_B delete 重复物体所得
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
        # 'omni':
        #     """backpack_015:83
        #       chair_003:23
        #       bread_162:0
        #       brush_004:0
        #       cabinet_008:16
        #       calculator_003:14
        #       chair_015:18
        #       dustbin_007:31
        #       dustbin_006:99
        #       scissor_012:0
        #       toy_truck_024:86
        #       toy_plane_029:1""",
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
    
    
class TempCo3dtestset:#tmp for  check relpose++
    datasetName_2_s={
        'co3d':"""
               # hydrant/185_19986_38630:24 1103发现没gen

               hydrant/157_17287_33549

               # couch/175_18976_35151 1103发现没gen
               # couch/193_20822_43319 1103发现没gen
               # couch/215_22688_47261 1103发现没gen

               hydrant/244_25997_52016
               hydrant/106_12698_26785
               hydrant/194_20922_42215
               hydrant/194_20925_42241
               hydrant/250_26744_53526
               hydrant/194_20956_44543
               hydrant/106_12648_23157
               hydrant/106_12660_22718
               keyboard/153_16970_32014
               keyboard/76_7706_16174
               keyboard/191_20631_39408
               laptop/62_4324_11087
               laptop/112_13277_23636
               laptop/241_25545_51811
               laptop/62_4317_10781
               laptop/62_4341_11248

               # toyplane/77_7885_16197 测试集有这个cate
               # toyplane/77_7901_16266
               # toyplane/121_14150_27596
               # toyplane/190_20485_38424
               # toyplane/190_20488_38923
               # toyplane/199_21386_43613
               # toyplane/255_27516_55384
               # toyplane/264_28179_53215
               # toyplane/264_28180_53406
               # toyplane/309_32622_59952
               # toyplane/373_41650_83139
               # toyplane/373_41664_83296
               # toyplane/373_41781_83422
               suitcase/31_1262_4177
               suitcase/48_2717_7806
               suitcase/48_2730_7942
               suitcase/50_2928_8645
               suitcase/50_2945_8929
               bicycle/62_4318_10726
               bicycle/62_4323_10695
               bicycle/62_4324_10701
               bicycle/62_4327_11291
               bicycle/108_12855_23322
               bicycle/127_14750_29938
               bicycle/136_15656_31168
               bicycle/196_21125_43621
               bicycle/252_27048_54124
               bicycle/350_36865_69259
               bicycle/372_40981_81625
               bicycle/373_41633_83104
               bicycle/373_41840_83504

               #1105 generated
               book/20_688_1353
               book/20_690_1412
               book/20_712_1422
               book/20_752_1509
               book/20_784_1988
               book/28_936_2384
               book/30_1202_3579
               book/30_1241_3644
               book/31_1268_3819
               book/119_13962_28926
               book/150_16670_31845
               cellphone/76_7610_15980
               cellphone/112_13288_23942
               cup/12_100_593
               cup/20_685_1352
               cup/20_689_1264
               cup/31_1243_3785
               handbag/396_49461_97546
               handbag/396_49751_97984
               microwave/48_2735_7961
               microwave/48_2739_8223
               microwave/48_2753_8230
               microwave/414_56888_110026
               microwave/426_59669_115581
               microwave/428_60178_117220
               microwave/436_62147_122347
               microwave/504_72519_140728
               microwave/506_72906_141733
               microwave/506_72924_141766
               microwave/569_82871_163873
               motorcycle/185_19993_39343
               motorcycle/216_22798_47409
               motorcycle/352_37168_70976
               motorcycle/359_37731_71933
               motorcycle/362_38239_73050
               motorcycle/362_38275_75311
               motorcycle/362_38295_75300
               mouse/14_170_904
               mouse/30_1102_3037
               mouse/93_10162_19392
               mouse/117_13753_28132
               mouse/117_13764_29495
               mouse/158_17387_32896
               mouse/158_17447_33339
               mouse/207_21910_46023
               mouse/217_22911_48738
               mouse/217_22933_49699
               mouse/236_24792_52205
               mouse/236_24794_52259
               plant/40_1818_5584
               remote/65_4647_11854
               remote/68_5129_12126
               remote/68_5248_12347
               remote/68_5249_12348
               remote/68_5251_12350
               remote/68_5281_12391
               remote/186_20089_37053
               remote/195_20990_41694
               remote/195_20991_41590
               remote/195_20993_41693
               remote/195_20994_41695
               toilet/105_12567_23172
               toilet/105_12596_24925
               toilet/115_13552_28913
               toilet/124_14451_29937
               toilet/165_18077_34348
               toilet/184_19890_38332
               toilet/193_20825_42667
               toilet/215_22725_49626
               toilet/267_28299_56126
               toilet/267_28306_55651
               vase/58_3364_10284
               """
    }
    l__datasetName_cate_seq_Q0INDEX=get__l__datasetName_cate_seq_Q0INDEX(datasetNames=list(datasetName_2_s.keys()),datasetName_2_s=datasetName_2_s)
    
    
class TempOmnitestset:
    datasetName_2_s={
        'omni':"""book_001
                    book_003
                    book_007
                    book_008
                    book_009
                    book_014
                    book_020
                    book_021
                    sandwich_003
                    sandwich_004
                    sandwich_005
                    sandwich_006
                    sandwich_007
                    sandwich_013
                    sandwich_022
                    sandwich_023
                    sandwich_024
                    sandwich_026
                    sandwich_028
                    skateboard_002
                    # suitcase_001 assert np.array_equal(unique_values, np.array([0, 1]))不通过
                    suitcase_004
                    suitcase_006
                    suitcase_007
                    suitcase_008""",
    }
    l__datasetName_cate_seq_Q0INDEX=get__l__datasetName_cate_seq_Q0INDEX(datasetNames=list(datasetName_2_s.keys()),datasetName_2_s=datasetName_2_s)


class TempOmnitestset_B:
    datasetName_2_s={
        'omni':"""book_001
                  book_003
                  book_007
                  book_009
                  book_014
                  book_020
                  book_021
                  sandwich_003
                  sandwich_004:44
                  sandwich_005
                  sandwich_006
                  sandwich_007:61
                  sandwich_013:60
                  sandwich_022
                  sandwich_023
                  sandwich_024
                  sandwich_026:60
                  sandwich_028
                  skateboard_002:51
                  suitcase_004:72
                  suitcase_006:61
                  # suitcase_007
                  # suitcase_008""",
    }
    l__datasetName_cate_seq_Q0INDEX=get__l__datasetName_cate_seq_Q0INDEX(datasetNames=list(datasetName_2_s.keys()),datasetName_2_s=datasetName_2_s)

class Navi_A:
    datasetName_2_s={
        'navi':
                """
                #-----I.-------------
                    schleich_lion_action_figure(multiview_10_pixel_5)
                    chicken_racer(multiview_08_ipad_4)
                    chicken_racer(multiview_12_pixel_5)
                    chicken_racer(multiview_03_pixel_5)
                    chicken_racer(multiview_10_ipad_5)
                    tractor_green_showpiece(multiview_03_pixel_6pro)
                    tractor_green_showpiece(multiview_02_pixel_6pro)
                    schleich_spinosaurus_action_figure(multiview_10_canon_t4i)
                    schleich_spinosaurus_action_figure(multiview_07_ipad_5)
                    schleich_spinosaurus_action_figure(multiview_01_pixel_5)
                    schleich_spinosaurus_action_figure(multiview_10_ipad_5)
                    garbage_truck_green_toy_s(multiview_02_pixel_6pro)
                    garbage_truck_green_toy_s(multiview_04_pixel_6pro)
                    keywest_showpiece_s(multiview_04_pixel_6pro)
                    # keywest_showpiece_s #
                    can_kernel_corn(multiview_01_pixel_4a)
                    can_kernel_corn(multiview_03_pixel_6pro)
                    # circo_fish_toothbrush_holder_14995988
                    circo_fish_toothbrush_holder_14995988(multiview_14_pixel_5)
                    pumpkin_showpiece_s(multiview_00_pixel_4a)
                    school_bus(multiview_12_ipad_5)
                    school_bus(multiview_10_canon_t4i)
                    paper_weight_flowers_showpiece(multiview_00_pixel_4a)
                    paper_weight_flowers_showpiece(multiview_03_pixel_6pro)
                    remote_control_toy_car_s(multiview_04_pixel_6pro)
                    schleich_hereford_bull(multiview_06_ipad_5)
                    schleich_hereford_bull(multiview_02_pixel_5)
                    duck_bath_yellow_s(multiview_02_pixel_6pro)
                    duck_bath_yellow_s(multiview_04_pixel_6pro)
                    fire_engine_toy_red_yellow_s(multiview_00_pixel_4a)
                    fire_engine_toy_red_yellow_s(multiview_03_pixel_6pro)
                    schleich_african_black_rhino(multiview_09_canon_t4i)
                    soldier_wood_showpiece(multiview_04_pixel_6pro)
                    dino_5(multiview_09_pixel_5)
                    bunny_racer(multiview_00_pixel_5)
                #--------------------------II. those q0!=0 (B.py)----------------------------------------------
                    schleich_lion_action_figure(multiview_03_pixel_5):3
                    schleich_lion_action_figure(multiview_07_pixel_5):9
                    water_gun_toy_yellow(multiview_00_pixel_4a):3
                    keywest_showpiece_s(multiview_00_pixel_6pro):15
                    keywest_showpiece_s(multiview_02_pixel_6pro):9
                    well_with_leaf_roof_showpiece(multiview_03_pixel_6pro):0
                    steps_small_showpiece(multiview_03_pixel_6pro):12
                    steps_small_showpiece(multiview_04_pixel_6pro):3
                    welcome_sign_mushrooms(multiview_00_pixel_4a):12
                    #-------------10-20
                    schleich_bald_eagle(multiview_07_pixel_5):0
                    schleich_bald_eagle(multiview_11_pixel_7):42
                    schleich_bald_eagle(multiview_08_canon_t4i):15
                    3d_dollhouse_sink(multiview_05_pixel_5):27
                    3d_dollhouse_sink(multiview_04_pixel_4xl):27
                    3d_dollhouse_sink(multiview_04_canon_t4i):27
                    dino_4(multiview_10_pixel_4xl):18
                    water_gun_toy_green(multiview_01_pixel_4a):9
                    #---2023-12-09晚，还剩下20-30没弄""",
    }
    l__datasetName_cate_seq_Q0INDEX=get__l__datasetName_cate_seq_Q0INDEX(datasetNames=list(datasetName_2_s.keys()),datasetName_2_s=datasetName_2_s)
    
    
class Navi_B:#based on Navi_A.  注释掉的是Navi_A中有而这里不用的
    s="""
                #-----I.-------------
                    # schleich_lion_action_figure(multiview_10_pixel_5)
                    # chicken_racer(multiview_08_ipad_4)
                    chicken_racer(multiview_12_pixel_5)
                    # chicken_racer(multiview_03_pixel_5)
                    # chicken_racer(multiview_10_ipad_5)
                    tractor_green_showpiece(multiview_03_pixel_6pro)
                    tractor_green_showpiece(multiview_02_pixel_6pro)
                    # schleich_spinosaurus_action_figure(multiview_10_canon_t4i)
                    # schleich_spinosaurus_action_figure(multiview_07_ipad_5)
                    # schleich_spinosaurus_action_figure(multiview_01_pixel_5)
                    schleich_spinosaurus_action_figure(multiview_10_ipad_5)
                    garbage_truck_green_toy_s(multiview_02_pixel_6pro)
                    # garbage_truck_green_toy_s(multiview_04_pixel_6pro)
                    # keywest_showpiece_s(multiview_04_pixel_6pro)
                    can_kernel_corn(multiview_01_pixel_4a)
                    can_kernel_corn(multiview_03_pixel_6pro)
                    circo_fish_toothbrush_holder_14995988(multiview_14_pixel_5)
                    pumpkin_showpiece_s(multiview_00_pixel_4a)
                    # school_bus(multiview_12_ipad_5)
                    school_bus(multiview_10_canon_t4i)
                    # paper_weight_flowers_showpiece(multiview_00_pixel_4a)
                    paper_weight_flowers_showpiece(multiview_03_pixel_6pro)
                    remote_control_toy_car_s(multiview_04_pixel_6pro)
                    schleich_hereford_bull(multiview_06_ipad_5)
                    # schleich_hereford_bull(multiview_02_pixel_5)
                    duck_bath_yellow_s(multiview_02_pixel_6pro)
                    # duck_bath_yellow_s(multiview_04_pixel_6pro)
                    fire_engine_toy_red_yellow_s(multiview_00_pixel_4a)
                    # fire_engine_toy_red_yellow_s(multiview_03_pixel_6pro)
                    schleich_african_black_rhino(multiview_09_canon_t4i)
                    soldier_wood_showpiece(multiview_04_pixel_6pro)
                    dino_5(multiview_09_pixel_5)
                    bunny_racer(multiview_00_pixel_5)
                #--------------------------II. those q0!=0 (B.py)----------------------------------------------
                    # schleich_lion_action_figure(multiview_03_pixel_5):3
                    schleich_lion_action_figure(multiview_07_pixel_5):9
                    water_gun_toy_yellow(multiview_00_pixel_4a):3
                    # keywest_showpiece_s(multiview_00_pixel_6pro):15
                    keywest_showpiece_s(multiview_02_pixel_6pro):9
                    well_with_leaf_roof_showpiece(multiview_03_pixel_6pro):0
                    # steps_small_showpiece(multiview_03_pixel_6pro):12
                    steps_small_showpiece(multiview_04_pixel_6pro):3
                    welcome_sign_mushrooms(multiview_00_pixel_4a):12
                    #-------------10-20
                    schleich_bald_eagle(multiview_07_pixel_5):0
                    # schleich_bald_eagle(multiview_11_pixel_7):42
                    # schleich_bald_eagle(multiview_08_canon_t4i):15
                    # 3d_dollhouse_sink(multiview_05_pixel_5):27
                    3d_dollhouse_sink(multiview_04_pixel_4xl):27
                    # 3d_dollhouse_sink(multiview_04_canon_t4i):27
                    dino_4(multiview_10_pixel_4xl):18
                    water_gun_toy_green(multiview_01_pixel_4a):9
                    #---2023-12-09晚，还剩下20-30没弄"""
    datasetName_2_s={'navi':s}
    l__datasetName_cate_seq_Q0INDEX=get__l__datasetName_cate_seq_Q0INDEX(datasetNames=list(datasetName_2_s.keys()),datasetName_2_s=datasetName_2_s)
    
class TempAGsoTrainset:# 截止至-I用时还叫GsoTrainset; 后3个物体不应入选set，没有经过我的手工筛选q0
    s="""gso_adiZero_Slide_2_SC
         gso_ASICS_GELAce_Pro_Pearl_WhitePink
         gso_ASICS_GELBlur33_20_GS_BlackWhiteSafety_Orange
         gso_Black_Decker_Stainless_Steel_Toaster_4_Slice
         gso_BlackBlack_Nintendo_3DSXL
         gso_Blue_Jasmine_Includes_Digital_Copy_UltraViolet_DVD"""
    datasetName_2_s={'gso':s}
    l__datasetName_cate_seq_Q0INDEX=get__l__datasetName_cate_seq_Q0INDEX(datasetNames=list(datasetName_2_s.keys()),datasetName_2_s=datasetName_2_s)
class GsoTemp4IprTest_A: 
    s="""
    gso_Crayola_Crayons_24_count
    gso_D_ROSE_773_II_hvInJwJ5HUD
    gso_Lenovo_Yoga_2_11
    gso_Marvel_Avengers_Titan_Hero_Series_Doctor_Doom:22
        """
    datasetName_2_s={'gso':s}
    l__datasetName_cate_seq_Q0INDEX=get__l__datasetName_cate_seq_Q0INDEX(datasetNames=list(datasetName_2_s.keys()),datasetName_2_s=datasetName_2_s)
        
class GsoTemp4IprTest_B: 
    s="""
    gso_Crayola_Crayons_24_count
    gso_D_ROSE_773_II_hvInJwJ5HUD
    gso_Lenovo_Yoga_2_11
    gso_Marvel_Avengers_Titan_Hero_Series_Doctor_Doom:22
    
    
    gso_BREAKFAST_MENU
    gso_Breyer_Horse_Of_The_Year_2015:6
    gso_BUILD_A_ZOO:12
    gso_California_Navy_Tieks_Italian_Leather_Ballet_Flats
    gso_Canon_Pixma_Ink_Cartridge_8_Green:4
    gso_CAR_CARRIER_TRAIN
    gso_Chelsea_lo_fl_rdheel_nQ0LPNF1oMw
    gso_Cole_Hardware_Hammer_Black
    # # ----------below are not test by (tmp4testR_IPR-CONSIDER_IPR-D-testRefactor-BICUBIC)
    # gso_Crayola_Crayons_Washable_24_crayons:25
    # gso_Dino_3
    # gso_Dino_4:4
    # gso_Dog
    # gso_Epson_LabelWorks_LC5WBN9_Tape_reel_labels_071_x_295_Roll_Black_on_White:4
    # gso_Focus_8643_Lime_Squeezer_10x35x188_Enamelled_Aluminum_Light
    # gso_GEARS_PUZZLES_STANDARD_gcYxhNHhKlI
    # gso_GEOMETRIC_PEG_BOARD:14
    # gso_Granimals_20_Wooden_ABC_Blocks_Wagon
    # gso_Granimals_20_Wooden_ABC_Blocks_Wagon_g2TinmUGGHI:33
    # gso_Great_Dinos_Triceratops_Toy:35
        """
    datasetName_2_s={'gso':s}
    l__datasetName_cate_seq_Q0INDEX=get__l__datasetName_cate_seq_Q0INDEX(datasetNames=list(datasetName_2_s.keys()),datasetName_2_s=datasetName_2_s)
            
class GsoTrainset:
    s="""
    gso_adiZero_Slide_2_SC
    
    # to 调整q0(2024.1.4 完成调整)
         gso_ASICS_GELAce_Pro_Pearl_WhitePink:37
         gso_ASICS_GELBlur33_20_GS_BlackWhiteSafety_Orange:52
        #  gso_Black_Decker_Stainless_Steel_Toaster_4_Slice:52 testset
         gso_BlackBlack_Nintendo_3DSXL:8
         gso_Blue_Jasmine_Includes_Digital_Copy_UltraViolet_DVD
        
        
        
        gso_BUNNY_RATTLE:8
        gso_Crayola_Crayons_24_count
        gso_Dell_Series_9_Color_Ink_Cartridge_MK993_High_Yield:33
        gso_D_ROSE_773_II_hvInJwJ5HUD
        # gso_Lenovo_Yoga_2_11  trun
        gso_Marvel_Avengers_Titan_Hero_Series_Doctor_Doom:22
        gso_Melissa_Doug_Pound_and_Roll
        gso_Mens_Billfish_Slip_On_in_Coffee_e8bPKE9Lfgo
        gso_MINI_ROLLER:25
        gso_My_First_Rolling_Lion:10
        gso_Nickelodeon_Teenage_Mutant_Ninja_Turtles_Michelangelo
        gso_Olive_Kids_Game_On_Pack_n_Snack:25
        gso_Olive_Kids_Mermaids_Pack_n_Snack_Backpack:35
        gso_Olive_Kids_Trains_Planes_Trucks_Bogo_Backpack:25
        
        
        
        # gso_3D_Dollhouse_Sofa:14 testset
        # gso_3D_Dollhouse_Swing  trun
        gso_60_CONSTRUCTION_SET
        gso_adizero_5Tool_25
        gso_Air_Hogs_Wind_Flyers_Set_Airplane_Red:20
        # gso_Android_Figure_Chrome   symmetrical
        gso_Animal_Planet_Foam_2Headed_Dragon
        # gso_Avengers_Thor_PLlrpYniaeB  trun
        gso_BREAKFAST_MENU
        gso_Breyer_Horse_Of_The_Year_2015:6
        gso_BUILD_A_ZOO:12
        gso_California_Navy_Tieks_Italian_Leather_Ballet_Flats
        gso_Canon_Pixma_Ink_Cartridge_8_Green:4
        gso_CAR_CARRIER_TRAIN
        # gso_Chelsea_lo_fl_rdheel_nQ0LPNF1oMw    #R_acc30<0.2
        gso_Cole_Hardware_Hammer_Black
        gso_Crayola_Crayons_Washable_24_crayons:25
        # gso_Dino_3 与  navi test 重合
        # gso_Dino_4:4 与  navi test 重合
        gso_Dog
        gso_Epson_LabelWorks_LC5WBN9_Tape_reel_labels_071_x_295_Roll_Black_on_White:4
        # gso_Focus_8643_Lime_Squeezer_10x35x188_Enamelled_Aluminum_Light    #R_acc30<0.2
        # gso_GEARS_PUZZLES_STANDARD_gcYxhNHhKlI    #R_acc30<0.2
        gso_GEOMETRIC_PEG_BOARD:14
        gso_Granimals_20_Wooden_ABC_Blocks_Wagon
        gso_Granimals_20_Wooden_ABC_Blocks_Wagon_g2TinmUGGHI:33
        gso_Great_Dinos_Triceratops_Toy:35
        
        
        # (被注释掉的是不合适的。（但还是被传到服务器上来了，不用管))
        gso_HAPPY_ENGINE
        gso_JA_Henckels_International_Premio_Cutlery_Block_Set_14Piece
        gso_Jawbone_UP24_Wireless_Activity_Tracker_Pink_Coral_L
        # gso_Kanex_MultiSync_Wireless_Keyboard
        gso_KID_ROOM_FURNITURE_SET_1
        gso_Kingston_DT4000MR_G2_Management_Ready_USB_64GB
        gso_Magnifying_Glassassrt
        gso_Markings_Letter_Holder:8
        gso_Melissa_Doug_Cart_Turtle_Block:25
        gso_Mens_Billfish_Slip_On_in_Coffee_nK6AJJAHOae:25
        gso_My_Little_Pony_Princess_Celestia
        gso_Nickelodeon_Teenage_Mutant_Ninja_Turtles_Leonardo:4
        gso_Nickelodeon_Teenage_Mutant_Ninja_Turtles_Raphael:17
        gso_Nickelodeon_The_Spongebob_Movie_PopAPart_Spongebob
        gso_Now_Designs_Dish_Towel_Mojave_18_x_28
        gso_Now_Designs_Snack_Bags_Bicycle_2_count
        gso_Ortho_Forward_Facing:30
        gso_Ortho_Forward_Facing_3Q6J2oKJD92
        gso_OWL_SORTER:4
        gso_OXO_Cookie_Spatula:17
        gso_Playmates_Industrial_CoSplinter_Teenage_Mutant_Ninja_Turtle_Action_Figure
        gso_Playmates_nickelodeon_teenage_mutant_ninja_turtles_shredder:4
        gso_Pony_C_Clamp_1440
        gso_PUNCH_DROP_TjicLPMqLvz
        gso_Racoon:4
        # gso_Reebok_COMFORT_REEFRESH_FLIP    #R_acc30<0.2
        gso_Reebok_ZIGSTORM:16
        gso_Schleich_African_Black_Rhino
        gso_Spectrum_Wall_Mount
        gso_Squirrel:4
        gso_Squirtin_Barnyard_Friends_4pk
        gso_The_Scooper_Hooper
        gso_Thomas_Friends_Woodan_Railway_Henry:24
        gso_Thomas_Friends_Wooden_Railway_Deluxe_Track_Accessory_Pack
        gso_Thomas_Friends_Wooden_Railway_Talking_Thomas_z7yi7UFHJRj:17
        # gso_Threshold_Porcelain_Spoon_Rest_White
        # gso_Threshold_Porcelain_Teapot_White
        gso_Weisshai_Great_White_Shark:4
        
        
        #-----------------packed in 2024.1.4
        gso_3D_Dollhouse_Happy_Brother
        gso_Android_Lego:6
        gso_Avengers_Gamma_Green_Smash_Fists:4
        gso_BABY_CAR:12
        # gso_BALANCING_CACTUS:27    #R_acc30<0.2
        gso_Black_Decker_CM2035B_12Cup_Thermal_Coffeemaker:27
        gso_Chefmate_8_Frypan
        gso_COAST_GUARD_BOAT:12
        # gso_Craftsman_Grip_Screwdriver_Phillips_Cushion    #R_acc30<0.2
        gso_DANCING_ALLIGATOR_zoWBjc0jbTs
        # gso_Diamond_Visions_Scissors_Red:38    #R_acc30<0.2
        gso_FIRE_ENGINE:6
        gso_FIRE_TRUCK:25
        gso_GARDEN_SWING:14
        gso_Guardians_of_the_Galaxy_Galactic_Battlers_Rocket_Raccoon_Figure
        gso_Imaginext_Castle_Ogre:14
        # KITCHEN_SET_CLASSIC_40HwCHfeG0H   trun
        gso_LADYBUG_BEAD:33
        gso_Lalaloopsy_Peanut_Big_Top_Tricycle:14
        gso_Markings_Desk_Caddy:23
        gso_MINI_EXCAVATOR:29
        gso_MINI_FIRE_ENGINE:27
        gso_My_First_Wiggle_Crocodile:25
        gso_Nintendo_Yoshi_Action_Figure:32
        gso_Ocedar_Snap_On_Dust_Pan_And_Brush_1_ct:17
        gso_RedBlack_Nintendo_3DSXL:6
        gso_Remington_TStudio_Silk_Ceramic_Hair_Straightener_2_Inch_Floating_Plates
        gso_Rubbermaid_Large_Drainer
        # Schleich_Allosaurus # there are similar obj in testset
        gso_Schleich_Hereford_Bull
        gso_Schleich_S_Bayala_Unicorn_70432:3
        gso_Shark
        gso_Simon_Swipe_Game:24
        gso_Smith_Hawken_Woven_BasketTray_Organizer_with_3_Compartments_95_x_9_x_13:12
        gso_SNAIL_MEASURING_TAPE:14
        gso_SpiderMan_Titan_Hero_12Inch_Action_Figure_5Hnn4mtkFsP:33
        gso_SpiderMan_Titan_Hero_12Inch_Action_Figure_oo1qph4wwiW:6
        gso_Squirt_Strain_Fruit_Basket
        # gso_Super_Mario_3D_World_Deluxe_Set    #R_acc30<0.2
        gso_Teenage_Mutant_Ninja_Turtles_Rahzar_Action_Figure:7
        gso_Thomas_Friends_Wooden_Railway_Porter_5JzRhMm3a9o
        gso_Tory_Burch_Kiernan_Riding_Boot:6
        gso_Transformers_Age_of_Extinction_Stomp_and_Chomp_Grimlock_Figure
        gso_TriStar_Products_PPC_Power_Pressure_Cooker_XL_in_Black:25
        gso_TURBOPROP_AIRPLANE_WITH_PILOT
        # Victor_Reversible_Bookend  trun
        gso_Womens_Suede_Bahama_in_Graphite_Suede_p1KUwoWbw7R
        gso_Wooden_ABC_123_Blocks_50_pack:25
        
        """
    datasetName_2_s={'gso':s}
    l__datasetName_cate_seq_Q0INDEX=get__l__datasetName_cate_seq_Q0INDEX(datasetNames=list(datasetName_2_s.keys()),datasetName_2_s=datasetName_2_s)
    
    
class Co3dTrainset:#就是Nov时用的那个s，只不过从 finetune_dataset.py 挪过来了
    s="""
    # hydrant/185_19986_38630:24 1103发现没gen

    hydrant/157_17287_33549

    # couch/175_18976_35151 1103发现没gen
    # couch/193_20822_43319 1103发现没gen
    # couch/215_22688_47261 1103发现没gen

    hydrant/244_25997_52016
    hydrant/106_12698_26785
    hydrant/194_20922_42215
    hydrant/194_20925_42241
    hydrant/250_26744_53526
    hydrant/194_20956_44543
    hydrant/106_12648_23157
    hydrant/106_12660_22718
    keyboard/153_16970_32014
    keyboard/76_7706_16174
    keyboard/191_20631_39408
    laptop/62_4324_11087
    laptop/112_13277_23636
    laptop/241_25545_51811
    laptop/62_4317_10781
    laptop/62_4341_11248

    # toyplane/77_7885_16197 测试集有这个cate
    # toyplane/77_7901_16266
    # toyplane/121_14150_27596
    # toyplane/190_20485_38424
    # toyplane/190_20488_38923
    # toyplane/199_21386_43613
    # toyplane/255_27516_55384
    # toyplane/264_28179_53215
    # toyplane/264_28180_53406
    # toyplane/309_32622_59952
    # toyplane/373_41650_83139
    # toyplane/373_41664_83296
    # toyplane/373_41781_83422
    suitcase/31_1262_4177
    suitcase/48_2717_7806
    suitcase/48_2730_7942
    suitcase/50_2928_8645
    suitcase/50_2945_8929
    bicycle/62_4318_10726
    bicycle/62_4323_10695
    bicycle/62_4324_10701
    bicycle/62_4327_11291
    bicycle/108_12855_23322
    bicycle/127_14750_29938
    bicycle/136_15656_31168
    bicycle/196_21125_43621
    bicycle/252_27048_54124
    bicycle/350_36865_69259
    bicycle/372_40981_81625
    bicycle/373_41633_83104
    bicycle/373_41840_83504

    #1105 generated
    book/20_688_1353
    book/20_690_1412
    book/20_712_1422
    book/20_752_1509
    book/20_784_1988
    book/28_936_2384
    book/30_1202_3579
    book/30_1241_3644
    book/31_1268_3819
    book/119_13962_28926
    book/150_16670_31845
    cellphone/76_7610_15980
    cellphone/112_13288_23942
    cup/12_100_593
    cup/20_685_1352
    cup/20_689_1264
    cup/31_1243_3785
    handbag/396_49461_97546
    handbag/396_49751_97984
    microwave/48_2735_7961
    microwave/48_2739_8223
    microwave/48_2753_8230
    microwave/414_56888_110026
    microwave/426_59669_115581
    microwave/428_60178_117220
    microwave/436_62147_122347
    microwave/504_72519_140728
    microwave/506_72906_141733
    microwave/506_72924_141766
    microwave/569_82871_163873
    motorcycle/185_19993_39343
    motorcycle/216_22798_47409
    motorcycle/352_37168_70976
    motorcycle/359_37731_71933
    motorcycle/362_38239_73050
    motorcycle/362_38275_75311
    motorcycle/362_38295_75300
    mouse/14_170_904
    mouse/30_1102_3037
    mouse/93_10162_19392
    mouse/117_13753_28132
    mouse/117_13764_29495
    mouse/158_17387_32896
    mouse/158_17447_33339
    mouse/207_21910_46023
    mouse/217_22911_48738
    mouse/217_22933_49699
    mouse/236_24792_52205
    mouse/236_24794_52259
    plant/40_1818_5584
    remote/65_4647_11854
    remote/68_5129_12126
    remote/68_5248_12347
    remote/68_5249_12348
    remote/68_5251_12350
    remote/68_5281_12391
    remote/186_20089_37053
    remote/195_20990_41694
    remote/195_20991_41590
    remote/195_20993_41693
    remote/195_20994_41695
    toilet/105_12567_23172
    toilet/105_12596_24925
    toilet/115_13552_28913
    toilet/124_14451_29937
    toilet/165_18077_34348
    toilet/184_19890_38332
    toilet/193_20825_42667
    toilet/215_22725_49626
    toilet/267_28299_56126
    toilet/267_28306_55651
    vase/58_3364_10284
    """
    datasetName_2_s={'co3d':s}
    l__datasetName_cate_seq_Q0INDEX=get__l__datasetName_cate_seq_Q0INDEX(datasetNames=list(datasetName_2_s.keys()),datasetName_2_s=datasetName_2_s)
    
class TempCo3dTrainset_A:#tmp 4 test wether model can overfit on co3d to check correctness of co3d trainset
    s="""
    hydrant/157_17287_33549
    keyboard/153_16970_32014
    laptop/62_4324_11087
    suitcase/31_1262_4177
    bicycle/62_4318_10726
    book/20_688_1353
    cellphone/76_7610_15980
    cup/12_100_593
    handbag/396_49461_97546
    microwave/48_2735_7961
    motorcycle/185_19993_39343
    mouse/14_170_904
    plant/40_1818_5584
    remote/65_4647_11854
    toilet/105_12567_23172
    vase/58_3364_10284
    """
    datasetName_2_s={'co3d':s}
    l__datasetName_cate_seq_Q0INDEX=get__l__datasetName_cate_seq_Q0INDEX(datasetNames=list(datasetName_2_s.keys()),datasetName_2_s=datasetName_2_s)
    
class TempCo3dTrainset_B:#meaning same as TempCo3dTrainset_B
    s="""
    hydrant/157_17287_33549
    keyboard/153_16970_32014
    laptop/62_4324_11087
    suitcase/31_1262_4177
    """
    datasetName_2_s={'co3d':s}
    l__datasetName_cate_seq_Q0INDEX=get__l__datasetName_cate_seq_Q0INDEX(datasetNames=list(datasetName_2_s.keys()),datasetName_2_s=datasetName_2_s)

class Co3dTestset_A:# for suppl. relpose++ paper 里的10个testset cate,each cate 选了1-2seq
    s="""
    suitcase/48_2700_7931
    suitcase/31_1262_4177
    skateboard/63_4348_11198
    skateboard/168_18360_34837
    couch/133_15357_30726
    couch/338_34912_63680
    book/20_688_1353
    book/150_16670_31845
    remote/65_4647_11854
    remote/195_20994_41695
    handbag/396_49461_97546
    handbag/396_49751_97984
    
    kite/380_44993_90001
    kite/422_58671_113667
    sandwich/198_21285_41285
    # sandwich/210_22200_45659  KeyError  19
    
    frisbee/68_5188_12072
    ball/113_13374_24686
    """