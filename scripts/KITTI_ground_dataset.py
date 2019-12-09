import numpy as np
import os
from glob import glob
import torch
from tqdm import tqdm
from torch.utils.data.dataset import Dataset
import scipy.interpolate
from laserscan import SemLaserScan
from occupancygrid import BEVOccupancyGrid
import yaml

class KITTIGroundDataset(Dataset):
    def __init__(self, data_dir, sequences):
        assert(np.isin(sequences,np.arange(0,11)).all())
        self.sequences = ["{:02}".format(seq) for seq in sequences]
        self.dataset = []
        self.ground_labels = [40,44,48,49,60,72]
        self.n_x = 448
        self.n_y = 512
        self.n_z = 32
        self.fwd_range = (0,70.4)
        self.side_range = (-40,40)
        self.height_range = (-3,1)
        self.sem_kitti_config_file = '../config/semantic-kitti.yaml'

        CFG = yaml.safe_load(open(self.sem_kitti_config_file, 'r'))
        self.color_dict = CFG["color_map"]
        self.nclasses = len(self.color_dict)

        print("Loading dataset...")
        for sequence in tqdm(self.sequences):
            pc_dir = os.path.join(data_dir,sequence,'velodyne')
            for lidar_file in glob(os.path.join(pc_dir,'*.bin')):
                label_file = lidar_file.replace('velodyne','labels').replace('.bin','.label')
                self.dataset.append((lidar_file,label_file))
        print("Load dataset done!")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        lidar_file, label_file = self.dataset[index]

        scan = SemLaserScan(self.nclasses,self.color_dict)
        scan.open_scan(lidar_file)
        scan.open_label(label_file)
        ground_indices = np.zeros(scan.sem_label.shape)

        # Find all lidar points associated with ground labels in semantic-kitti dataset
        for label in self.ground_labels:
            ground_indices += (scan.sem_label == label)

        ground_points = scan.points[ground_indices.astype(np.bool)]
        grid = BEVOccupancyGrid(scan.points, 
                                n_x=self.n_x, 
                                n_y=self.n_y, 
                                n_z=self.n_z, 
                                side_range = self.side_range,
                                fwd_range = self.fwd_range,
                                height_range = self.height_range, 
                                filter_external = True
                                )
        grid.compute()
        
        # best-fit quadratic curve (2nd-order)
        A = np.c_[np.ones(ground_points.shape[0]), ground_points[:,:2], np.prod(ground_points[:,:2], axis=1), ground_points[:,:2]**2]
        C,_,_,_ = scipy.linalg.lstsq(A, ground_points[:,2])

        # evaluate it on a grid
        Y, X = np.meshgrid(np.flip(grid.midsegments[0]),np.flip(grid.midsegments[1]))
        XX = X.flatten()
        YY = Y.flatten()
        Z = np.dot(np.c_[np.ones(XX.shape), XX, YY, XX*YY, XX**2, YY**2], C).reshape(X.shape)

        return torch.FloatTensor(grid.grid).permute(2,0,1), torch.FloatTensor(Z).unsqueeze(dim=0)