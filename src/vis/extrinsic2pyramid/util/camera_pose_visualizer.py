import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import PIL
import io
# from exception_util import *
import root_config




def set_axes_equal(ax):#https://stackoverflow.com/questions/13685386/how-to-set-the-equal-aspect-ratio-for-all-axes-x-y-z
    # Use this for matplotlib prior to 3.3.0 only.
    # ax.set_aspect("equal'")
    #
    # Use this for matplotlib 3.3.0 and later.
    # https://github.com/matplotlib/matplotlib/pull/17515
    ax.set_box_aspect([1.0, 1.0, 1.0])
    """
    Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    """

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])





class CameraPoseVisualizer:
    def __init__(self, xlim, ylim, zlim,camera_style='A',):
        self.camera_style=camera_style
        self.fig = plt.figure(figsize=(18, 7))
        try:
            self.ax = self.fig.gca(projection='3d')
        except:
            self.ax = self.fig.add_subplot(projection='3d')
        # self.ax.set_aspect("auto")
        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)
        self.ax.set_zlim(zlim)
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')
        self.ax.set_zlabel('z')
        set_axes_equal(self.ax)
        # print('initialize camera pose visualizer')

        

    def w2cColumnRt_2_pyramid(self, R,t, color='r', focal_len_scaled=5, aspect_ratio=0.3,y_is_vertical=True,opacity=0.85):
        """
        R,t:w2c;column
        :param R:
        :param t: 3,1
        :return:
        """
        extrinsic=np.eye(4)
        extrinsic[:3,:3]=R
        extrinsic[:3,3]=t
        extrinsic=np.linalg.inv(extrinsic)
        return self.extrinsic2pyramid(extrinsic,color,focal_len_scaled,aspect_ratio,y_is_vertical,opacity)

    def w2cColumnExtrinsic_2_pyramid(self, extrinsic, color='r', focal_len_scaled=5, aspect_ratio=0.3,y_is_vertical=True,opacity=0.85,**kw):
        """
        extrinsic:w2c;column
        """
        new_extrinsic=np.linalg.inv(extrinsic)#w2c to c2w(or say: leftMul to rightMul)
        return self.extrinsic2pyramid(new_extrinsic,color,focal_len_scaled,aspect_ratio,y_is_vertical=y_is_vertical,opacity=opacity,**kw)
    def extrinsic2pyramid(
            self, extrinsic, color='r', focal_len_scaled=5, aspect_ratio=0.3,
            # img=None,
            y_is_vertical=True, opacity=0.85,
            colorWho='mesh',
    ):
        # vertex_std = np.array(
        #     # [
        #     #     [0, 0, 0, 1],
        #     #     [focal_len_scaled * aspect_ratio, -focal_len_scaled * aspect_ratio, focal_len_scaled, 1],
        #     #     [focal_len_scaled * aspect_ratio, focal_len_scaled * aspect_ratio, focal_len_scaled, 1],
        #     #     [-focal_len_scaled * aspect_ratio, focal_len_scaled * aspect_ratio, focal_len_scaled, 1],
        #     #     [-focal_len_scaled * aspect_ratio, -focal_len_scaled * aspect_ratio, focal_len_scaled, 1],
        #     # ]
        #     # [
        #     #     [0, 0, 0, 1],
        #     #     [focal_len_scaled * aspect_ratio, -focal_len_scaled * aspect_ratio, focal_len_scaled, 1],
        #     #     [0.5, focal_len_scaled * aspect_ratio /2, focal_len_scaled, 1],
        #     #     [-0.5, focal_len_scaled * aspect_ratio /2, focal_len_scaled, 1],
        #     #     [-focal_len_scaled * aspect_ratio, -focal_len_scaled * aspect_ratio, focal_len_scaled, 1],
        #     # ]
        #     [
        #         [0, 0, 0, 1],
        #         [focal_len_scaled * aspect_ratio, -focal_len_scaled * aspect_ratio, focal_len_scaled, 1],
        #         [focal_len_scaled * aspect_ratio *0.8, focal_len_scaled * aspect_ratio / 4, focal_len_scaled, 1],
        #         [-focal_len_scaled * aspect_ratio *0.8, focal_len_scaled * aspect_ratio / 4, focal_len_scaled, 1],
        #         [-focal_len_scaled * aspect_ratio, -focal_len_scaled * aspect_ratio, focal_len_scaled, 1],
        #     ]
        # )
        _Y=focal_len_scaled * aspect_ratio / 4
        # print("y_is_vertical",y_is_vertical)
        if(not y_is_vertical):
            _Y=-_Y
        if self.camera_style=='A':
            _N=0.6
            vertex_std = np.array(
                [
                    [0, 0, 0, 1],
                    [focal_len_scaled * aspect_ratio, -_Y, focal_len_scaled, 1],
                    [focal_len_scaled * aspect_ratio * _N, _Y, focal_len_scaled, 1],
                    [-focal_len_scaled * aspect_ratio * _N, _Y, focal_len_scaled, 1],
                    [-focal_len_scaled * aspect_ratio, -_Y, focal_len_scaled, 1],
                ]
            )
        elif self.camera_style=='B':
            _N=1.0
            _Y*=3
            _Z=focal_len_scaled#4paper-11.09.19:00
            _Z=4
            if 0:
                aspect_ratio*=1.3
                _Y*=1
            vertex_std = np.array(
                [
                    [0, 0, 0, 1],
                    [focal_len_scaled * aspect_ratio,       -_Y,    _Z, 1],
                    [focal_len_scaled * aspect_ratio * _N,   _Y,    _Z, 1],
                    [-focal_len_scaled * aspect_ratio * _N,  _Y,    _Z, 1],
                    [-focal_len_scaled * aspect_ratio,       -_Y,   _Z, 1],
                ]
            )
            opacity=0.4
        else:
            raise NotImplementedError
        vertex_transformed = vertex_std @ extrinsic.T

        meshes = [[vertex_transformed[0, :-1], vertex_transformed[1][:-1], vertex_transformed[2, :-1]],
                            [vertex_transformed[0, :-1], vertex_transformed[2, :-1], vertex_transformed[3, :-1]],
                            [vertex_transformed[0, :-1], vertex_transformed[3, :-1], vertex_transformed[4, :-1]],
                            [vertex_transformed[0, :-1], vertex_transformed[4, :-1], vertex_transformed[1, :-1]],
                            [vertex_transformed[1, :-1], vertex_transformed[2, :-1], vertex_transformed[3, :-1], vertex_transformed[4, :-1]]]
        img=r"C:\Users\YiLucky\Desktop\test.png"
        img=None
        if(img):# show img in meshes[-1]
            if(isinstance(img,str)):
                img=plt.imread(img)
            else:
                assert 0
            assert  img.shape[2]==3
            #img resize to (focal_len_scaled * aspect_ratio * _N,_Y)
            import cv2
            # img=img.transpose(1,0,2)
            print("img.shape",img.shape)
            # new_w_h=(int(focal_len_scaled * aspect_ratio * _N),int(_Y))
            new_w_h=(640,480)
            print("new_w_h",new_w_h)
            img=cv2.resize(img,new_w_h)#h, w, 3
            # img=img.transpose(1,0,2)
            from vis.py_matplotlib_helper import plotImage
            R=extrinsic[:3,:3]
            t=extrinsic[:3,3]
            plotImage(self.ax, img, R, t, size=np.array((1, img.shape[0] / img.shape[1])))

        if colorWho=='mesh':
            facecolors = color
            edgecolors = "black"
            linestyle='dotted'
            # linestyle='dashed'
            linewidths=2
        else:
            facecolors = (1,1,1,0)
            edgecolors = color
            linestyle='solid'
            linewidths=1.5
        self.ax.add_collection3d(
            Poly3DCollection(
                meshes,
                facecolors=facecolors,
                linewidths=linewidths,
                edgecolors=edgecolors,
                alpha=opacity,
                linestyle=linestyle,
            ))


    def customize_legend(self, list_label,list_color=None):
        list_handle = []
        for idx, label in enumerate(list_label):
            if list_color is None:
                color = plt.cm.rainbow(idx / len(list_label))
            else:
                color=list_color[idx]
            patch = Patch(color=color, label=label)
            list_handle.append(patch)
        plt.legend(loc='right',
                   bbox_to_anchor=(1,0),#https://stackoverflow.com/questions/25068384/bbox-to-anchor-and-loc-in-matplotlib
                   handles=list_handle)

    def colorbar(self, max_frame_length):
        cmap = mpl.cm.rainbow
        norm = mpl.colors.Normalize(vmin=0, vmax=max_frame_length)
        self.fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), orientation='vertical', label='Frame Number')

    def show(self,title='Extrinsic Parameters',kw_view_init={"elev":None, "azim":None}):
        self.ax.view_init(**kw_view_init)
        # self.customize_legend(['a','bbbb'])

        plt.title(title)
        # plt.show()
        # plt.draw()
        # plt.waitforbuttonpress(0) # this will wait for indefinite time
        # plt.close(self.fig)

        def close_figure(event):
            if event.key == 'escape':
                plt.close(event.canvas.figure)
        plt.gcf().canvas.mpl_connect('key_press_event', close_figure)
        plt.show()

    def get_img(self,no_margin=False,format_='jpg'):
        """
        :return: numpy.ndarray
        """
        # return self.fig.canvas.tostring_rgb()
        kw={}
        if(no_margin):
            kw={
                "bbox_inches":'tight',
                "pad_inches":0
            }
        """
        while(1):
            try:
                self.fig.savefig('tmp4get_img.jpg',**kw)
                img=plt.imread('tmp4get_img.jpg')
                break
            except PIL.UnidentifiedImageError as e:
                handle_exception(e)
                continue
        """
        def fig2arr(fig):
            with io.BytesIO() as buff:#from https://stackoverflow.com/questions/7821518/save-plot-to-numpy-array
                fig.savefig(buff, format=format_,**kw)
                buff.seek(0)
                # im = plt.imread(buff)# by default format is png => rgba
                im = plt.imread(buff,format=format_)
            # im=im[...,:3]
            return im
        img=fig2arr(self.fig)
        return img