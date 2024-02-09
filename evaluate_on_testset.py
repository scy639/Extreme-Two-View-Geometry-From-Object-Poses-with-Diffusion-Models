

import sys,os
cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(cur_dir, "src"))











import root_config
from evaluate.eval_test_set import run

def main(datasetName:str,rotate=False):
    #------------------configs----------------------
    root_config.VIS=0 # do not  visualize result to save time. when debugging, you can let it be True
    root_config.SKIP_EVAL_SEQ_IF_EVAL_RESULT_EXIST = 1 # skip to eval a category if its eval result exists
    # when GPU out of memory, decrease the following values:
    root_config.SAMPLE_BATCH_SIZE = 32
    root_config.SAMPLE_BATCH_B_SIZE = 9
    
    
    
    if rotate:#add inplane rotation to images
        root_config.CONSIDER_IPR=True
        root_config.Q0Sipr=True
        root_config.Q1Sipr=True
    run(  [datasetName] )
if __name__=='__main__':
    main(datasetName='gso',)#gso testset
    main(datasetName='navi',)#navi testset
    main(datasetName='gso',rotate=True)#rotated gso testset
    main(datasetName='navi',rotate=True)#rotated navi testset