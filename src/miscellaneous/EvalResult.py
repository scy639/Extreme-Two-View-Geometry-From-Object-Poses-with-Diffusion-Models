import root_config
from imports import *
import json
import math
import os,cv2
import os.path as osp
# from Dataset.co3dv2 import Co3dv2Dataset
# from Dataset.linemod import LinemodDataset
# from Dataset.omni import OmniDataset
# from Dataset.idpose import IdposeAboDataset,IdposeOmniDataset
from Dataset.gso import GsoDataset
from Dataset.navi import NaviDataset
import numpy as np
import torch
from skimage.io import imsave, imread
import pandas as pd
import os, shutil, sys, json
class EvalResult():  
    """
    model_name, idSuffix, category <--> a EvalResult
    a EvalResult can contains several seq's result
    核心数据是 seqAndPair2err;category_acc只是衍生品，依赖于seqAndPair2err,可根据seqAndPair2err生成
    """
    all_acc_path=[]
    all_category=[]
    all_idSuffix=[]
    class AllAcc:
        @staticmethod
        def append_acc_path(path,category,idSuffix):
            EvalResult.all_acc_path.append(path)
            EvalResult.all_category.append(category)
            EvalResult.all_idSuffix.append(idSuffix)
        @staticmethod
        def dump_average_acc(SUFFIX="",only_log_these_cates__and__only_print_do_not_dump=None):
            if only_log_these_cates__and__only_print_do_not_dump:
                assert isinstance(only_log_these_cates__and__only_print_do_not_dump[0],str)
            """
            dic_avg=
            {'R_acc5': 0.06608695652173913, 'R_acc10': 0.2426086956521739, 'R_acc15': 0.3391304347826086, 'R_acc30': 0.49391304347826087,
            'T_acc5': 0.11217391304347828, 'T_acc10': 0.26173913043478264, 'T_acc15': 0.37391304347826093, 'T_acc30': 0.5695652173913043}

            """
            def report(fullPaths,):
                cate__2__idSuffix_2_dic = {}
                for i,fullPath in enumerate(fullPaths):
                    if fullPath.startswith("#"):
                        print("this line is comment:", fullPath)
                        continue
                    if fullPath == "":
                        print("this line is empty")
                        continue
                    cate = EvalResult.all_category[i]
                    print(cate)
                    if only_log_these_cates__and__only_print_do_not_dump is not None:
                        if cate not in only_log_these_cates__and__only_print_do_not_dump:
                            continue
                    idSuffix = EvalResult.all_idSuffix[i]
                    with  open(fullPath, "r") as f:
                        dic = json.load(f)
                        dic_R = dic["R"]["all"]
                        dic_T = dic["T"]["all"]
                        dic_R = {f"R_{k}": v for k, v in dic_R.items()}
                        dic_T = {f"T_{k}": v for k, v in dic_T.items()}
                        dic = {
                            **dic_R,
                            **dic_T,
                        }
                    if (cate not in cate__2__idSuffix_2_dic):
                        cate__2__idSuffix_2_dic[cate] = {}
                    cate__2__idSuffix_2_dic[cate][idSuffix] = dic

                def sort_by_key(dic):
                    assert isinstance(dic, dict)
                    sorted_dict = dict(sorted(dic.items()))
                    return sorted_dict
                cate__2__idSuffix_2_dic = sort_by_key(cate__2__idSuffix_2_dic)
                l_dic_ = []
                dfIndex0 = []
                dfIndex1 = []
                for cate, idSuffix_2_dic in cate__2__idSuffix_2_dic.items():
                    idSuffix_2_dic = sort_by_key(idSuffix_2_dic)
                    l_dic_ += list(idSuffix_2_dic.values())
                    dfIndex0 += [cate] * len(idSuffix_2_dic)
                    dfIndex1 += list(idSuffix_2_dic.keys())
                df = pd.DataFrame(l_dic_, index=[dfIndex0, dfIndex1])
                # df = df.applymap(lambda x: '{:.4f}'.format(x))
                
                # df = df.applymap(lambda x: '{:.2%}'.format(x))

                average_row = df.sum()  
                average_row = average_row / df.shape[0]  
                dic_avg = average_row.to_dict()
                l_dic_withAvg = l_dic_ + [dic_avg]
                dfIndex0.append("Average")
                dfIndex1.append(" ")
                df = pd.DataFrame(l_dic_withAvg, index=[dfIndex0, dfIndex1])
                pd.set_option('display.max_rows',1000)
                pd.set_option('display.max_columns',1000)
                print(df)
                # if only_log_these_cates__and__only_print_do_not_dump:
                if 1:
                    # print(f"only_log_these_cates__and__only_print_do_not_dump={only_log_these_cates__and__only_print_do_not_dump},df is shown above. avg R_... are shown below:")
                    R_acc15,R_acc30,T_acc15,T_acc30=dic_avg['R_acc15'],dic_avg['R_acc30'],dic_avg['T_acc15'],dic_avg['T_acc30']
                    print(f"avg R_acc15,R_acc30,T_acc15,T_acc30=")
                    latex=''
                    for _ in [R_acc15,R_acc30,T_acc15,T_acc30]:
                        _=f'{_:.2%}'
                        _=_.replace('%','')
                        print(_)
                        latex+=f'& {_} '
                    # print(f"latex: {latex}")
                    print("--------------------------\n\n")
                if only_log_these_cates__and__only_print_do_not_dump:
                    return
                
                # content = df.to_string(index=False)
                
                # import pyperclip
                # pyperclip.copy(content)
                
                
                OUT_NAME=f"[EvalResult.AllAcc{SUFFIX}]refIdSuffix={root_config.refIdSuffix}.idSuffix={root_config.idSuffix}.MAX_PAIRS={root_config.MAX_PAIRS}.{f'{your_datetime():%m%d-%H-%M-%S}'}"
                while( os.path.exists(  os.path.join(root_config.evalResultPath_co3d,f'{OUT_NAME}.xlsx')  )):
                    OUT_NAME+="_"
                df.to_excel(os.path.join(root_config.evalResultPath_co3d,f'{OUT_NAME}.xlsx'))
                print("save to:\n",f'{OUT_NAME}.xlsx')
                if root_config.VALing:
                    path=os.path.join(root_config.path_4val,f'dic_avg{SUFFIX}.json')
                    with open(path, "w") as f:
                        json.dump(dic_avg, f)
                        print(f"[EvalResult.AllAcc] Saved to {path}")
                    exit(0)
            report(EvalResult.all_acc_path)

    def __init__(self, model_name, idSuffix, category) -> None:
        self.model_name = model_name
        self.idSuffix = idSuffix
        self.category = category
        #
        self.seqAndPair2err_json = osp.join(root_config.evalResultPath_co3d,
                                            f"[{model_name}-{idSuffix}]{category}-seqAndPair2err.json")
        if (os.path.exists(self.seqAndPair2err_json)):
            with open(self.seqAndPair2err_json) as f:
                self.seqAndPair2err = json.load(f)
        else:
            self.seqAndPair2err = {}
        #
        self.category_acc_json = osp.join(root_config.evalResultPath_co3d,
                                f"[{self.model_name}-{self.idSuffix}]{self.category}-acc.json")
    @staticmethod
    def seq_i_j_2_str(sequence_name, i, j):
        assert isinstance(sequence_name, str)
        assert isinstance(i, int)
        assert isinstance(j, int)
        return f"{sequence_name}_{i}-{j}"

    @staticmethod
    def str_2_seq_i_j(str):  # like 134_15451_31119_190-76. note that there are '_' in seq
        assert "__" not in str
        assert "-" in str
        assert "_" in str
        sequence_name_and_i, j = str.split("-")
        # last _ to split sequence_name and i
        sequence_name, i = sequence_name_and_i.rsplit("_", 1)
        i = int(i)
        j = int(j)
        return sequence_name, i, j

    def get_pair__in_dic(self, sequence_name, i, j):
        assert self.exist_pair(sequence_name, i, j)
        return self.seqAndPair2err[EvalResult.seq_i_j_2_str(sequence_name, i, j)]

    def get_pair__in_tuple(self, sequence_name, i, j):
        assert self.exist_pair(sequence_name, i, j)
        return (
            np.array(self.seqAndPair2err[EvalResult.seq_i_j_2_str(sequence_name, i, j)]["R_pred_rel"]),
            np.array(self.seqAndPair2err[EvalResult.seq_i_j_2_str(sequence_name, i, j)]["R_gt_rel"]),
            np.array(self.seqAndPair2err[EvalResult.seq_i_j_2_str(sequence_name, i, j)]["R_error"]),
            #
            np.array(self.seqAndPair2err[EvalResult.seq_i_j_2_str(sequence_name, i, j)]["T31_pred_rel"]),
            np.array(self.seqAndPair2err[EvalResult.seq_i_j_2_str(sequence_name, i, j)]["T31_gt_rel"]),
            np.array(self.seqAndPair2err[EvalResult.seq_i_j_2_str(sequence_name, i, j)]["T_error"]),
            #
            self.seqAndPair2err[EvalResult.seq_i_j_2_str(sequence_name, i, j)]["key_frames"],
        )

    def append_pair(self, sequence_name, i, j,
                    R_pred_rel, R_gt_rel, R_error,
                    T31_pred_rel, T31_gt_rel, T_error,
                    key_frames,
                    **kw,
                    ):
        assert np.isscalar(R_error)
        assert np.isscalar(T_error)
        assert len(key_frames) == 2
        self.seqAndPair2err[EvalResult.seq_i_j_2_str(sequence_name, i, j)] = {
            "R_pred_rel": R_pred_rel.tolist(),
            "R_gt_rel": R_gt_rel.tolist(),
            "R_error": R_error,
            #
            "T31_pred_rel": T31_pred_rel.tolist(),
            "T31_gt_rel": T31_gt_rel.tolist(),
            "T_error": T_error,
            #
            "key_frames": key_frames,
            **kw,
        }

    def exist_pair(self, sequence_name, i, j):
        return EvalResult.seq_i_j_2_str(sequence_name, i, j) in self.seqAndPair2err
    def check_no_other_pair(self,sequence_name, pairs):
        """
        load的json里不能有不包含在pairs里的pair(所谓other_pair。理论上只有seed改变才会出现pairs不同的情况
        pairs: eg [(0, 146), (0, 31), (0, 29),...
        """
        l_str=[EvalResult.seq_i_j_2_str(sequence_name, i, j)  for i, j in pairs]
        for str_ in self.seqAndPair2err:#str_ eg 157_17287_33549_0-146
            seq,i,j=self.str_2_seq_i_j(str=str_)
            if seq==sequence_name:
                assert str_ in l_str
                assert (i,j) in pairs
    def exist_pairs(self, sequence_name, pairs):
        self.check_no_other_pair(sequence_name, pairs)
        return all([self.exist_pair(sequence_name, i, j) for i, j in pairs])

    def dump(self):  # should be called after each seq
        with open(self.seqAndPair2err_json, "w") as f:
            json.dump(self.seqAndPair2err, f)
            INFO(f"[EvalResult] Saved to {self.seqAndPair2err_json}")

    def __get_l_sequence_name(self):
        return list(set([EvalResult.str_2_seq_i_j(key)[0] for key in self.seqAndPair2err.keys()]))

    def dump_acc(self, ):
        def _get_acc_dic(KEY: str):
            def __sequence_name__2__flat_angularErr(sequence_name):
                flat_angularErr = [
                    val[KEY] for key, val in self.seqAndPair2err.items() if
                    EvalResult.str_2_seq_i_j(key)[0] == sequence_name
                ]
                # assert len(flat_angularErr) == 0 or np.array(flat_angularErr[0]).shape[0] == 2
                flat_angularErr = np.array(flat_angularErr).flatten()
                return flat_angularErr

            dic = {
                "all": EvalResult.l_angularErr__2__accDic(
                    [item[KEY] for item in self.seqAndPair2err.values()]),
                **{
                    sequence_name:
                        EvalResult.l_angularErr__2__accDic(
                            __sequence_name__2__flat_angularErr(sequence_name)
                        )
                    for sequence_name in self.__get_l_sequence_name()
                }
            }
            return dic

        dic_R = _get_acc_dic("R_error")
        dic_T = _get_acc_dic("T_error")
        dic = {
            "R": dic_R,
            "T": dic_T,
        }
        with open(self.category_acc_json, "w") as f:
            json.dump(dic, f, indent=4)
            INFO(f"[EvalResult] Saved to {self.category_acc_json},acc={json.dumps(dic, indent=4)}")
    @staticmethod
    def l_angularErr__2__accDic(flat_angularErr: list):  # one dimension list
        
        acc5 = np.mean(np.array(flat_angularErr) < 5).item()
        acc10 = np.mean(np.array(flat_angularErr) < 10).item()
        acc15 = np.mean(np.array(flat_angularErr) < 15).item()
        acc30 = np.mean(np.array(flat_angularErr) < 30).item()
        return {
            "acc5": acc5,
            "acc10": acc10,
            "acc15": acc15,
            "acc30": acc30,
        }