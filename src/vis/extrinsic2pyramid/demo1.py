import numpy as np
from util.camera_pose_visualizer import CameraPoseVisualizer

if __name__ == '__main__':
    # argument : the minimum/maximum value of x, y, z
    visualizer = CameraPoseVisualizer([-50, 50], [-50, 50], [0, 50])
    
    # w2c=np.eye(4)
    w2c = np.array(
        # [
        #     [1, 0, 0, 0],
        #     [0, 1, 0, 0],
        #     [0, 0, 1, 20],
        #     [0, 0, 0, 1],
        # ]
        [
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [-1, 0, 0, 20],
            [0, 0, 0, 1],
        ]
    )
    extrinsic=w2c.T
    # argument : extrinsic matrix, color, scaled focal length(z-axis length of frame body of camera
    visualizer.extrinsic2pyramid(extrinsic, 'c', 10)

    visualizer.show()
