import numpy as np

class BEVOccupancyGrid(object):
    def __init__(self, points, n_x, n_y, n_z, side_range, fwd_range, height_range, filter_external=True):
        """Grid of voxels with support for different build methods.
        Parameters
        ----------
        points: (N, 3) numpy.array
        n_x, n_y, n_z :  int
            The number of segments in which each axis will be divided.
            Ignored if corresponding size_x, size_y or size_z is not None.
        side_range, fwd_range, height_range: tuples (min, max) define the walls of the occupancy grid
        filter_external: boolean
            Whether or not to remove points from the cloud or have them accumulate at the edges
        """
        self.points = points
        self.x_y_z = [n_x, n_y, n_z]
        self.xyzmin = [side_range[0], fwd_range[0], height_range[0]]
        self.xyzmax = [side_range[1], fwd_range[1], height_range[1]]

        if filter_external:
            # FILTER - To return only indices of points within desired box
            # Three filters for: Front-to-back (y), side-to-side (x), and height (z) ranges
            # Note left side is positive y axis in LIDAR coordinates
            f_filt = np.logical_and((points[:,0] > fwd_range[0]), (points[:,0] < fwd_range[1]))
            s_filt = np.logical_and((points[:,1] > side_range[0]), (points[:,1] < side_range[1]))
            h_filt = np.logical_and((points[:,2] > height_range[0]), (points[:,2] < height_range[1]))
            filter_xy = np.logical_and(f_filt, s_filt)
            filter = np.logical_and(filter_xy, h_filt)
            indices = np.argwhere(filter).flatten()
            self.points = points[indices,:]
        
        else:
            self.points = points


    def compute(self):
        segments = []
        shape = []

        for i in range(3):
            # note the +1 in num
            s, step = np.linspace(self.xyzmin[i], self.xyzmax[i], num=self.x_y_z[i]+1, retstep=True)
            segments.append(s)
            shape.append(step)

        self.segments = segments
        self.shape = shape

        self.n_voxels = self.x_y_z[0] * self.x_y_z[1] * self.x_y_z[2]

        # find where each point lies in corresponding segmented axis
        self.voxel_x = np.clip(segments[0].size - np.searchsorted(segments[0],  self.points[:,1], side = "right") - 1, 0, self.x_y_z[0] - 1)
        self.voxel_y = np.clip(segments[1].size - np.searchsorted(segments[1],  self.points[:,0],  side = "right") - 1, 0, self.x_y_z[1] - 1)
        self.voxel_z = np.clip(segments[2].size - np.searchsorted(segments[2],  self.points[:,2],  side = "right") - 1, 0, self.x_y_z[2] - 1)
        self.voxel_n = np.ravel_multi_index([self.voxel_x, self.voxel_y, self.voxel_z], self.x_y_z)

        # compute center of each voxel
        self.midsegments = [(self.segments[i][1:] + self.segments[i][:-1]) / 2 for i in range(3)]
        self.voxel_centers = cartesian(self.midsegments).astype(np.float32)

        self.grid = np.zeros((self.x_y_z[1],self.x_y_z[0],self.x_y_z[2]))
        self.grid[self.voxel_y,self.voxel_x,self.voxel_z] = 1

def cartesian(arrays, out=None):
    """Generate a cartesian product of input arrays.
    Parameters
    ----------
    arrays : list of array-like
        1-D arrays to form the cartesian product of.
    out : ndarray
        Array to place the cartesian product in.
    Returns
    -------
    out : ndarray
        2-D array of shape (M, len(arrays)) containing cartesian products
        formed of input arrays.
    Examples
    --------
    >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
    array([[1, 4, 6],
           [1, 4, 7],
           [1, 5, 6],
           [1, 5, 7],
           [2, 4, 6],
           [2, 4, 7],
           [2, 5, 6],
           [2, 5, 7],
           [3, 4, 6],
           [3, 4, 7],
           [3, 5, 6],
           [3, 5, 7]])
    """
    arrays = [np.asarray(x) for x in arrays]
    shape = (len(x) for x in arrays)
    dtype = arrays[0].dtype

    ix = np.indices(shape)
    ix = ix.reshape(len(arrays), -1).T

    if out is None:
        out = np.empty_like(ix, dtype=dtype)

    for n, arr in enumerate(arrays):
        out[:, n] = arrays[n][ix[:, n]]

    return out