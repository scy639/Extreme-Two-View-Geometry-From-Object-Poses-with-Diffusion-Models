class Zero123_BatchB_Input:
    def __init__(self,
                 id_:str,
                 folder_outputIms:str,
                 input_image_path:str,
                 l_xyz:list,
                 ):
        """
        folder_outputIms:
            1. initially, means: where to put Ig. name of folder, not path
            2. after sample_model_batchB_wrapper ,its meaning turn from name to path
        input_image_path:
            1. initially, means: full path of input image
            2. after sample_model_batchB_wrapper , its meaning turn from path to tensor
        outputims:
            1. None
            2. list of output image path
        """
        self.id_=id_
        self.folder_outputIms=folder_outputIms
        self.input_image_path=input_image_path
        self.l_xyz=l_xyz
        self.outputims:list=None
    def __len__(self):
        return len(self.l_xyz)