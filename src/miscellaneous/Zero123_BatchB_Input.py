class Zero123_BatchB_Input:
    def __init__(self,
                 id_:str,
                 folder_outputIms:str,
                 input_image_path:str,
                 l_xyz:list,
                 ):
        """
        folder_outputIms:
            1. initially, means: zero123渲染图放到哪个文件夹,name of folder, not path
            2. 在sample_model_batchB_wrapper中某一步后,its meaning 会从name变成path
        input_image_path:
            1. initially, means: full path of input image
            2. 在sample_model_batchB_wrapper中某一步后,its meaning 会从path变成tensor
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