def genIntermediateResult(K,path=None, 
                          path_save=None, #eg. /baseline/relpose_plus_plus_main/relpose/../../../gen6d/Gen6D/./data/zero123/GSO_alarm----+8/ref'
                          calib_xy=(0,0) ,base_xyz=(0,0,0),called_by_run4gen6d=False ):
    ASK=0
    CHOOSE_J=False
    MOVE_OBJ_TO_CENTER=False
    import sys
    import os




    #----------------------------------------------
    path0 = path
    if CHOOSE_J:
        import choose_j
        path1 = os.path.join(path, 'after_choose_j')
        choose_j.main(read_path=path0, save_path=path1, ask=ASK)
    else:
        path1 = path0

    #----------------------------------------------
    if MOVE_OBJ_TO_CENTER:
        import move_obj_to_center
        path2 = os.path.join(path, 'after_move_obj_to_center')
        move_obj_to_center.main(read_path=path1, save_path=path2)
    else:
        path2 = path1

    #----------------------------------------------
    import crop
    intermediateResult = crop.crop(
        read_path=path2, save_path=path_save,
        calib_xy=calib_xy,
        base_xyz=base_xyz,
        K=K,

        # **kw
        ask=ASK,
        save_image=1,
        do_not_crop=1,
        norm_obj_by_z=1,

        margin_percent=0,
        # margin_percent=0.1,

        DRAW_cropped_img=0,  
    )
    path_intermediateResult =os.path.join(path_save, "intermediateResult.json")
    intermediateResult.dump(path_intermediateResult)
    #print("intermediateResult saved to:", os.path.abspath(path_intermediateResult))

if (__name__ == "__main__"):
    genIntermediateResult( called_by_run4gen6d=False)