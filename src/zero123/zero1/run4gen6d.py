import argparse, os, sys, datetime, glob
import numpy as np
import time
import torch


def get_parser(**parser_kwargs):

    parser = argparse.ArgumentParser(**parser_kwargs)

    parser.add_argument(
        "--id",
        type=str,
    )
    parser.add_argument(
        "--input_image_path",
        type=str,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=4,
    )

    return parser
from run_ import sample_model_,sample_model_batchB_wrapper
from util_4_e2vg.genIntermediateResult import genIntermediateResult
from util_4_e2vg.Util import OutputIm_Name_Parser
def main(
    id,
    input_image_path,
    output_dir,
    num_samples,
    K,
    ddim_steps=50,
    base_xyz=(0,0,0),
    **kw,
):
    parser = get_parser()
    args=parser.parse_args()
    args.id=id
    args.input_image_path=input_image_path
    args.output_dir=output_dir
    args.num_samples=num_samples


    #args: output_dir, input_image_path, num_samples
    folder_output_ims:str=sample_model_( input_image_path=input_image_path, num_samples=num_samples, id=id,
                                  batch_sample=True,
                                  ddim_steps=ddim_steps,
                                  **kw)
    if 'only_gen' in kw and kw['only_gen']:
        l__path_output_im:list=OutputIm_Name_Parser.parse_B(folder_output_ims)
        return l__path_output_im



    import sys,os
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.join(cur_dir, "util_4_e2vg"))
    
    
    genIntermediateResult(path=folder_output_ims, path_save=args.output_dir, calib_xy=(0,0), base_xyz=base_xyz,called_by_run4gen6d=True,K=K,
                        #   id=args.id
                          )
    sys.path.pop()

