
import os,sys
def is_in_sysPath(path):
    # print(os.getcwd())
    # print(sys.path)
    tmp=[file for file in sys.path if os.path.exists(file)]
    return any([os.path.samefile(path,file)  for file in tmp ])
def can_not_relative_import(file_path):
    path=os.path.abspath(os.path.join(os.path.dirname(file_path),os.path.pardir))
    if(is_in_sysPath(path=path)): 
        return 1
    path=os.path.abspath(os.path.join(os.path.dirname(file_path) ))
    if(is_in_sysPath(path=path)): 
        return 1
    return 0
def import_relposepp_evaluate_pairwise():
    import  sys,root_config
    sys.path.append(  root_config.projPath_relposepp  )

    from eval.eval_rotation_util import evaluate_pairwise

    for i,p in enumerate(sys.path):
        if p==root_config.projPath_relposepp:
            del sys.path[i]
            break
    print("sys.path=",sys.path)
    return evaluate_pairwise