from models import *
import numpy as np
import glob
import argparse

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision.models import vgg
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.utils.data import Dataset
from skimage.transform import resize

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

from torch_lib.Dataset import *
from library.Math import *
from library.Plotting import *
from torch_lib import Model

import os

import warnings
warnings.filterwarnings("ignore")


def plot_regressed_3d_bbox(img, cam_to_img, box_2d, dimensions, alpha, theta_ray):    
    location, X = calc_location(dimensions, cam_to_img, box_2d, alpha, theta_ray)
    orient = alpha + theta_ray

    plot_3d_box(img, cam_to_img, orient, dimensions, location) # 3d boxes

    return location

class ImageFolder(Dataset):
    def __init__(self, folder_path, img_size=416):
        self.files = sorted(glob.glob('%s/*.*' % folder_path))
        self.img_shape = (img_size, img_size)

    def __getitem__(self, index):
        img_path = self.files[index % len(self.files)]
        # Extract image
        img = np.array(Image.open(img_path))
        h, w, _ = img.shape
        dim_diff = np.abs(h - w)
        pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
        pad = ((pad1, pad2), (0, 0), (0, 0)) if h <= w else ((0, 0), (pad1, pad2), (0, 0))
        input_img = np.pad(img, pad, 'constant', constant_values=127.5) / 255.
        input_img = resize(input_img, (*self.img_shape, 3), mode='reflect')
        input_img = np.transpose(input_img, (2, 0, 1))
        input_img = torch.from_numpy(input_img).float()
        return img_path, input_img

    def __len__(self):
        return len(self.files)

def load_classes(path):
    fp = open(path, "r")
    names = fp.read().split("\n")[:-1]
    return names

def bbox_iou(box1, box2, x1y1x2y2=True):
    if not x1y1x2y2:
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(
        inter_rect_y2 - inter_rect_y1 + 1, min=0
    )
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou

def non_max_suppression(prediction, num_classes, conf_thres=0.5, nms_thres=0.4):
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    output = [None for _ in range(len(prediction))]
    for image_i, image_pred in enumerate(prediction):
        conf_mask = (image_pred[:, 4] >= conf_thres).squeeze()
        image_pred = image_pred[conf_mask]
        if not image_pred.size(0):
            continue
        class_conf, class_pred = torch.max(image_pred[:, 5 : 5 + num_classes], 1, keepdim=True)
        detections = torch.cat((image_pred[:, :5], class_conf.float(), class_pred.float()), 1)
        unique_labels = detections[:, -1].cpu().unique()
        if prediction.is_cuda:
            unique_labels = unique_labels.cuda()
        for c in unique_labels:
            detections_class = detections[detections[:, -1] == c]
            _, conf_sort_index = torch.sort(detections_class[:, 4], descending=True)
            detections_class = detections_class[conf_sort_index]
            max_detections = []
            while detections_class.size(0):
                max_detections.append(detections_class[0].unsqueeze(0))
                if len(detections_class) == 1:
                    break
                ious = bbox_iou(max_detections[-1], detections_class[1:])
                detections_class = detections_class[1:][ious < nms_thres]

            max_detections = torch.cat(max_detections).data
            output[image_i] = (
                max_detections if output[image_i] is None else torch.cat((output[image_i], max_detections))
            )
    return output


parser = argparse.ArgumentParser()
parser.add_argument('--image_folder', type=str, default='images/')
parser.add_argument('--config_path', type=str, default='config/yolov3-kitti.cfg')
parser.add_argument('--weights_path', type=str, default='yolov3-kitti.weights')
parser.add_argument('--class_path', type=str, default='data/kitti.names')
parser.add_argument('--conf_thres', type=float, default=0.8)
parser.add_argument('--nms_thres', type=float, default=0.4)
parser.add_argument('--img_size', type=int, default=416)
parser.add_argument('--use_cuda', type=bool, default=True)
args = parser.parse_args()

cuda = torch.cuda.is_available() and args.use_cuda

model = Darknet(args.config_path, img_size=args.img_size)
model.load_weights(args.weights_path)

if cuda:
    model.cuda()

model.eval()
dataloader = DataLoader(ImageFolder(args.image_folder, img_size=args.img_size),
                        batch_size=1, shuffle=False)
classes = load_classes(args.class_path) # Extracts class labels from file
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

my_vgg = vgg.vgg19_bn(pretrained=True)
model_3d = Model.Model(features=my_vgg.features, bins=2).cuda()
checkpoint = torch.load("./3d_info_weights/best.pkl")
model_3d.load_state_dict(checkpoint['model_state_dict'])
model_3d.eval()
calib_file = "./camera_cal/" + "calib_cam_to_cam.txt"
angle_bins = generate_bins(2)

imgs = []          
img_detections = []

total = len(dataloader)
for batch_i, (img_paths, input_imgs) in enumerate(dataloader):

    input_imgs = input_imgs.cuda()
    with torch.no_grad():
        detections = model(input_imgs)
        detections = non_max_suppression(detections, 80, args.conf_thres, args.nms_thres)
    
    print ('Solving Image %d / %d ' % (batch_i+1, total))

    img = np.array(Image.open(img_paths[0]))
    img_for3d = img

    img_dpi = 100
    xsize = img.shape[1] / img_dpi
    ysize = img.shape[0] / img_dpi
    plt.figure(figsize=[xsize,ysize], dpi=img_dpi)
    fig, ax = plt.subplots(1, figsize=[xsize,ysize], dpi=img_dpi)

    ax.imshow(img)
    image_size = args.img_size

    pad_x = max(img.shape[0] - img.shape[1], 0) * (image_size / max(img.shape))
    pad_y = max(img.shape[1] - img.shape[0], 0) * (image_size / max(img.shape))
    unpad_h = image_size - pad_y
    unpad_w = image_size - pad_x

    if detections[0] is not None:
        unique_labels = detections[0][:,-1].cpu().unique()
        for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections[0]:
            box_h = int(((y2 - y1) / unpad_h) * (img.shape[0]))
            box_w = int(((x2 - x1) / unpad_w) * (img.shape[1]))
            y1 = int(((y1 - pad_y // 2) / unpad_h) * (img.shape[0]))
            x1 = int(((x1 - pad_x // 2) / unpad_w) * (img.shape[1]))
            
            if int(cls_pred) == 0:
                bbx_color = 'r'
            else:
                bbx_color = 'b'
            bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=1, edgecolor=bbx_color, facecolor='none')
            plt.text(x1, y1-30, s=classes[int(cls_pred)]+' '+ str('%.2f'%cls_conf.item()), color='white', verticalalignment='top',
                    bbox={'color': bbx_color, 'pad': 0})
            ax.add_patch(bbox)


            det_class = None
            avg = [0, 0, 0]
            if int(cls_pred) == 0:
                det_class = "car"
                avg = [1.5, 1.6, 4.0]
            if int(cls_pred) == 1:
                det_class = "van"
                avg = [2.5, 2, 5.0]
            if int(cls_pred) == 2:
                det_class = "truck"
                avg = [3.6, 2.5, 7]
            if int(cls_pred) == 3:
                det_class = "pedestrian"
                avg = [1.8, 0.5, 1]
            if int(cls_pred) == 4:
                det_class = "person_sitting"
                avg = [1.2, 0.55, 0.75]
            if int(cls_pred) == 5:
                det_class = "cyclist"
                avg = [1.8, 0.55, 1.2]
            if int(cls_pred) == 6:
                det_class = "tram"
                avg = [3.6, 2.8, 6]
            if int(cls_pred) == 7:
                det_class = "misc"
                avg = [0, 0, 0]
            # 3d!!!!!
            if x1<0:
                x1 = 0
            if y1<0:
                y1 = 0
            detectedObject = DetectedObject(img_for3d, det_class, [(x1,y1), (x1+box_w,y1+box_h)], calib_file)
            theta_ray = detectedObject.theta_ray
            input_img = detectedObject.img
            proj_matrix = detectedObject.proj_matrix

            box_2d = [(x1,y1), (x1+box_w,y1+box_h)]
            input_tensor = torch.zeros([1,3,224,224]).cuda()
            input_tensor[0,:,:,:] = input_img
            [orient, conf_3d, dim] = model_3d(input_tensor)
            orient = orient.cpu().data.numpy()[0, :, :]
            conf_3d = conf_3d.cpu().data.numpy()[0, :]
            dim = dim.cpu().data.numpy()[0, :]

            dim = dim + avg
            argmax = np.argmax(conf_3d)
            orient = orient[argmax, :]
            cos = orient[0]
            sin = orient[1]
            alpha = np.arctan2(sin, cos)
            alpha += angle_bins[argmax]
            alpha -= np.pi

            location = plot_regressed_3d_bbox(img_for3d, proj_matrix, box_2d, dim, alpha, theta_ray)
    
    plt.axis('off')
    plt.gca().xaxis.set_major_locator(NullLocator())
    plt.gca().yaxis.set_major_locator(NullLocator())
    plt.savefig("output/" + (img_paths[0])[7:], dpi=img_dpi)
    plt.close()
    plt.figure(figsize=[xsize,ysize], dpi=img_dpi)
    fig, ax = plt.subplots(1, figsize=[xsize,ysize], dpi=img_dpi)

    plt.figure(figsize=[xsize,ysize], dpi=img_dpi)
    fig2, ax2 = plt.subplots(1, figsize=[xsize,ysize], dpi=img_dpi)
    ax2.imshow(img_for3d)
    plt.axis('off')
    plt.savefig("output_3d/" + (img_paths[0])[7:], dpi=img_dpi)
    plt.close()