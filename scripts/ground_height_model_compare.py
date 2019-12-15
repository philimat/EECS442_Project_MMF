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
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from unet import Unet
from KITTI_ground_dataset import KITTIGroundDataset

def test(testloader, net, criterion, device):
    '''
    Function for testing.
    '''
    losses = []
    with torch.no_grad():
        net = net.eval()
        for images, labels in tqdm(testloader):
            images = images.to(device)
            labels = labels.to(device)
            output = net(images)
            point_mask = torch.sum(images,axis=1,keepdim=True).type(torch.BoolTensor)
            loss = criterion(output[point_mask], labels[point_mask])
            losses.append(loss.item())
    print(np.mean(np.array(losses)))
    return (np.array(losses))

def MSE(z1,z2):
    n = z1.shape[0]
    return (1/n) * np.sum((z1 - z2)**2, axis=0)

if __name__ == '__main__':
    seq_directory = '../../KITTI/KITTI_odometry/dataset/sequences/'
    sequences = os.listdir(seq_directory)
    test_sequence_idx = 9
    current_sequence = sequences[test_sequence_idx]
    pc_dir = os.path.join(seq_directory,current_sequence,'velodyne')
    label_dir = os.path.join(seq_directory,current_sequence,'labels')
    lidar_files = glob(os.path.join(pc_dir,'*.bin'))
    config_file = '../config/semantic-kitti.yaml'
    CFG = yaml.safe_load(open(config_file, 'r'))
    class_strings = CFG["labels"]
    color_dict = CFG["color_map"]
    nclasses = len(color_dict)
    ground_labels = [40,44,48,49,60,72]

    batch_size = 8
    data_dir = '../../KITTI/KITTI_odometry/dataset/sequences/'
    test_data = KITTIGroundDataset(data_dir, np.arange(9,10))
    test_loader = DataLoader(test_data, batch_size=batch_size)
    name = 'ground_estimation_net_v2'

    net = Unet(32,1,5,32,concat=False)
    net = nn.DataParallel(net)
    net.load_state_dict(torch.load('./models/model_{}.pth'.format(name)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)
    criterion = nn.MSELoss()
    unet_losses = test(test_loader, net, criterion, device)

    print("U-Net MSE Average", np.mean(unet_losses))
    print("U-Net MSE Standard deviation", np.std(unet_losses))

    lidar_height_model_MSE = np.zeros(len(lidar_files))
    lidar_height = -1.73

    i = 0
    for lidar_file in tqdm(lidar_files):

        name = os.path.split(lidar_file)[-1]
        label_file = os.path.join(label_dir,os.path.splitext(name)[0] + '.label')
        scan = SemLaserScan(nclasses,color_dict)
        scan.open_scan(lidar_file)
        scan.open_label(label_file)
        ground_indices = np.zeros(scan.sem_label.shape)

        # Find all lidar points associated with ground labels in semantic-kitti dataset
        for label in ground_labels:
            ground_indices += (scan.sem_label == label)

        ground_points = scan.points[ground_indices.astype(np.bool)]
        n, _ = ground_points.shape
        ground_height = np.mean(ground_points[:,2])
        lidar_height_model_MSE[i] = MSE(np.repeat(lidar_height,n), ground_points[:,2])
    
        i += 1
    
    print("Lidar Height MSE", np.mean(lidar_height_model_MSE))
    print("Lidar Standard Deviation", np.std(lidar_height_model_MSE))