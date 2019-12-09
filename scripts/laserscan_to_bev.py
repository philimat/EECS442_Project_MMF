import os
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy
import scipy.interpolate
from laserscan import SemLaserScan
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D
import yaml
from occupancygrid import BEVOccupancyGrid
import pickle

# Function to get equal sized axis for 3D scatter plot to view Lidar points
def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

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

def point_cloud_2_birdseye(points,
                           res=0.1,
                           side_range=(-10., 10.),  # left-most to right-most
                           fwd_range = (-10., 10.), # back-most to forward-most
                           height_range=(-2., 2.),  # bottom-most to upper-most
                           ):
    """ Creates an 2D birds eye view representation of the point cloud data.

    Args:
        points:     (numpy array)
                    N rows of points data
                    Each point should be specified by at least 3 elements x,y,z
        res:        (float)
                    Desired resolution in metres to use. Each output pixel will
                    represent an square region res x res in size.
        side_range: (tuple of two floats)
                    (-left, right) in metres
                    left and right limits of rectangle to look at.
        fwd_range:  (tuple of two floats)
                    (-behind, front) in metres
                    back and front limits of rectangle to look at.
        height_range: (tuple of two floats)
                    (min, max) heights (in metres) relative to the origin.
                    All height values will be clipped to this min and max value,
                    such that anything below min will be truncated to min, and
                    the same for values above max.
    Returns:
        2D numpy array representing an image of the birds eye view.
    """
    # EXTRACT THE POINTS FOR EACH AXIS
    x_points = points[:, 0]
    y_points = points[:, 1]
    z_points = points[:, 2]

    # FILTER - To return only indices of points within desired cube
    # Three filters for: Front-to-back, side-to-side, and height ranges
    # Note left side is positive y axis in LIDAR coordinates
    f_filt = np.logical_and((x_points > fwd_range[0]), (x_points < fwd_range[1]))
    s_filt = np.logical_and((y_points > -side_range[1]), (y_points < -side_range[0]))
    filter = np.logical_and(f_filt, s_filt)
    indices = np.argwhere(filter).flatten()

    # KEEPERS
    x_points = x_points[indices]
    y_points = y_points[indices]
    z_points = z_points[indices]

    # CONVERT TO PIXEL POSITION VALUES - Based on resolution
    x_img = (-y_points / res).astype(np.int32)  # x axis is -y in LIDAR
    y_img = (-x_points / res).astype(np.int32)  # y axis is -x in LIDAR

    # SHIFT PIXELS TO HAVE MINIMUM BE (0,0)
    # floor & ceil used to prevent anything being rounded to below 0 after shift
    x_img -= int(np.floor(side_range[0] / res))
    y_img += int(np.ceil(fwd_range[1] / res))

    # CLIP HEIGHT VALUES - to between min and max heights
    pixel_values = np.clip(a=z_points,
                           a_min=height_range[0],
                           a_max=height_range[1])

    # RESCALE THE HEIGHT VALUES - to be between the range 0-255
    pixel_values = scale_to_255(pixel_values,
                                min=height_range[0],
                                max=height_range[1])

    # INITIALIZE EMPTY ARRAY - of the dimensions we want
    x_max = 1 + int((side_range[1] - side_range[0]) / res)
    y_max = 1 + int((fwd_range[1] - fwd_range[0]) / res)
    im = np.zeros([y_max, x_max], dtype=np.uint8)

    # FILL PIXEL VALUES IN IMAGE ARRAY
    im[y_img, x_img] = pixel_values

    return im

if __name__ == '__main__':
    seq_directory = '../../KITTI/KITTI_odometry/dataset/sequences/'
    sequences = os.listdir(seq_directory)
    current_sequence = sequences[0]
    pc_dir = os.path.join(seq_directory,current_sequence,'velodyne')
    label_dir = os.path.join(seq_directory,current_sequence,'labels')
    config_file = '../config/semantic-kitti.yaml'
    CFG = yaml.safe_load(open(config_file, 'r'))
    class_strings = CFG["labels"]
    color_dict = CFG["color_map"]
    nclasses = len(color_dict)
    ground_labels = [40,44,48,49,60,72]

    i = 0
    for lidar_file in tqdm(glob(os.path.join(pc_dir,'*.bin'))):

        if i == 0:
            name = os.path.split(lidar_file)[-1]
            img_ext = '.png'
            bev_dir = os.path.join(seq_directory,current_sequence,'bev')
            label_file = os.path.join(label_dir,os.path.splitext(name)[0] + '.label')
            scan = SemLaserScan(nclasses,color_dict)
            scan.open_scan(lidar_file)
            scan.open_label(label_file)
            ground_indices = np.zeros(scan.sem_label.shape)

            # Find all lidar points associated with ground labels in semantic-kitti dataset
            for label in ground_labels:
                ground_indices += (scan.sem_label == label)

            ground_points = scan.points[ground_indices.astype(np.bool)]
            grid = BEVOccupancyGrid(scan.points, n_x=448, n_y=512, n_z=32, side_range=(-40,40),fwd_range=(0,70.4),height_range=(-3,1), filter_external=True)
            grid.compute()
            
            # best-fit quadratic curve (2nd-order)
            A = np.c_[np.ones(ground_points.shape[0]), ground_points[:,:2], np.prod(ground_points[:,:2], axis=1), ground_points[:,:2]**2]
            C,_,_,_ = scipy.linalg.lstsq(A, ground_points[:,2])

            # evaluate it on a grid (not in BEV space)
            Y, X = np.meshgrid(np.flip(grid.midsegments[0]),np.flip(grid.midsegments[1]))
            XX = X.flatten()
            YY = Y.flatten()
            Z = np.dot(np.c_[np.ones(XX.shape), XX, YY, XX*YY, XX**2, YY**2], C).reshape(X.shape)

            # np.save('height',Z)
            # scipy.sparse.save_npz('sparse_grid',grid.grid)
            # ###Load into file
            # with open("myfile.pkl","wb") as f:
            #     pickle.dump(grid.grid,f)

            # ###Extract from file
            # with open("myfile.pkl","rb") as f:
            #     grid_temp = pickle.load(f)

            # fig = plt.figure()
            # ax = fig.add_subplot(111, projection='3d')
            # # only show small number of points for processing issues
            # plot_idx = np.random.choice(ground_points.shape[0],size=1000,replace=False)
            # d = np.linalg.norm(ground_points[plot_idx,:],axis=1)
            # ax.scatter(ground_points[plot_idx,0],ground_points[plot_idx,1],ground_points[plot_idx,2],c=d,cmap='jet')
            # set_axes_equal(ax)
            # plt.show()
            # plt.close()

            # # ground estimation image. Will have to get coordinate transformation right when doing the subtraction
            # plt.contourf(Y,X,Z)
            # plt.colorbar()
            # plt.title('Ground Estimation Depth')
            # plt.show()
            # plt.close()
            # sum up the BEV occupancy grid along the channels to see what it looks like from above
            bev_img = np.sum(grid.grid,axis=2)
            cmap = cm.get_cmap("jet")
            plt.imshow(bev_img,cmap=cmap)
            plt.axis('off')
            plt.title('Sum along channels of BEV tensor')
            plt.show()
            plt.close()

            break
        i += 1

