import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def camera_pose_from_matrix(matrix):
    R = matrix[:3, :3]
    t = matrix[:3, 3]
    pose = np.eye(4)
    pose[:3, :3] = R.T
    pose[:3, 3] = -R.T @ t
    return pose


# def vis_c2wPose(pose, save_path=None):
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')

#     # Define the camera points
#     camera_points = np.array([
#         [0.1, 0, 0],
#         [0, 0.1, 0],
#         [0, 0, 0.1],
#         [0, 0, 0],
#     ])

#     # Transform the camera points by the pose matrix
#     camera_points = pose @ np.vstack((camera_points.T, np.ones((1, 4))))
#     camera_points = camera_points[:3, :].T

#     # Plot the camera points
#     ax.scatter(camera_points[:, 0], camera_points[:, 1], camera_points[:, 2], c='r', marker='o')

#     # Set the plot limits and labels
#     ax.set_xlim([-1, 1])
#     ax.set_ylim([-1, 1])
#     ax.set_zlim([-1, 1])
#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     ax.set_zlabel('Z')

#     # Save the plot if a save path is provided
#     if save_path is not None:
#         plt.savefig(save_path)

#     # Show the plot
#     plt.show()


def vis_c2wPose(pose,xlim, ylim, zlim, save_path=None,):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Define the camera cone
    # camera_base = np.array([
    #     [0.1, 0, 0],
    #     [0, 0.1, 0],
    #     [-0.1, 0, 0],
    #     [0, -0.1, 0],
    # ])
    # camera_apex = np.array([0, 0, 0.2])
    F = 0.4
    X_HALF_LEN = 0.15
    Y_HALF_LEN = X_HALF_LEN/2
    camera_base = np.array([
        [X_HALF_LEN, Y_HALF_LEN, F],
        [X_HALF_LEN, -Y_HALF_LEN, F],
        [-X_HALF_LEN, -Y_HALF_LEN, F],
        [-X_HALF_LEN, Y_HALF_LEN, F],
    ])
    camera_apex = np.array([0, 0, 0])

    camera_points = np.vstack((camera_base, camera_apex))

    # Transform the camera points by the pose matrix
    camera_points = pose @ np.vstack((camera_points.T, np.ones((1, 5))))
    camera_points = camera_points[:3, :].T

    # Plot the camera cone
    camera_triangles = [
        [camera_points[0], camera_points[1], camera_points[4]],
        [camera_points[1], camera_points[2], camera_points[4]],
        [camera_points[2], camera_points[3], camera_points[4]],
        [camera_points[3], camera_points[0], camera_points[4]],
    ]
    camera_collection = Poly3DCollection(camera_triangles, alpha=0.25, facecolor='r', edgecolor='k')
    ax.add_collection(camera_collection)

    # Set the plot limits and labels
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Save the plot if a save path is provided
    if save_path is not None:
        plt.savefig(save_path)

    # Show the plot
    plt.show()


if(__name__ == '__main__'):
    # pose = np.eye(4)
    # pose[:3, :3] = np.array(
    #     [
    #         [1, 0, 0],
    #         [0, 1, 0],
    #         [0, 0, 1],
    #     ]
    #     # [
    #     #     [0, 0, 1],
    #     #     [0, 1, 0],
    #     #     [-1, 0, 0],
    #     # ]
    # )
    pose = np.array(
        [
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [-1, 0, 0, 20],
            [0, 0, 0, 1],
        ]
    )
    vis_c2wPose(pose, [-50, 50], [-50, 50], [0, 50],save_path='camera.png')
