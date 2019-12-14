from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *
import os
import sys

import argparse
import tqdm

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim

import warnings

warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=11)
parser.add_argument("--image_folder", type=str, default="data/samples")
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--model_config_path", type=str, default="config/yolov3-kitti.cfg")
parser.add_argument("--data_config_path", type=str, default="config/kitti.data")
parser.add_argument("--weights_path", type=str, default="kitti_best.weights")
parser.add_argument("--class_path", type=str, default="data/kitti.names")
parser.add_argument("--iou_thres", type=float, default=0.5)
parser.add_argument("--conf_thres", type=float, default=0.8)
parser.add_argument("--nms_thres", type=float, default=0.4)
parser.add_argument("--img_size", type=int, default=416)
parser.add_argument("--checkpoint_interval", type=int, default=2)
parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
parser.add_argument("--use_cuda", type=bool, default=True)
args = parser.parse_args()

my_dataset = args.data_config_path

cuda = torch.cuda.is_available() and args.use_cuda

classes = load_classes(args.class_path)

data_config = parse_data_config(args.data_config_path)
train_path = data_config["train"]

hyperparams = parse_model_config(args.model_config_path)[0]
learning_rate = float(hyperparams["learning_rate"])
momentum = float(hyperparams["momentum"])
decay = float(hyperparams["decay"])
burn_in = int(hyperparams["burn_in"])
test_path = data_config["valid"]
num_classes = int(data_config["classes"])

model = Darknet(args.model_config_path)
model.load_weights(args.weights_path)

if cuda:
    model = model.cuda()

model.train()

ListDataset(train_path)

dataloader = torch.utils.data.DataLoader(
    ListDataset(train_path), batch_size=args.batch_size, shuffle=True)

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))

losses_x = losses_y = losses_w = losses_h = losses_conf = losses_cls = losses_recall = losses_precision = batch_loss= 0.0
accumulated_batches = 4
best_mAP = 0.0

print("start traing")

test_dataset = ListDataset(test_path)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False) 

for epoch in range(args.epochs):
    losses_x = losses_y = losses_w = losses_h = losses_conf = losses_cls = losses_recall = losses_precision = batch_loss= 0.0

    optimizer.zero_grad()   

    for batch_i, (_, imgs, targets) in enumerate(dataloader):
        print("Training on epoch %d / %d, Batch %d / %d" % (epoch, args.epochs-1, batch_i, len(dataloader)))
        imgs = imgs.cuda()
        targets = targets.cuda()
        loss = model(imgs, targets)
        loss.backward()
        if ((batch_i + 1) % accumulated_batches == 0) or (batch_i == len(dataloader) - 1):
            optimizer.step()
            optimizer.zero_grad()
            
        losses_x += model.losses["x"]
        losses_y += model.losses["y"]
        losses_w += model.losses["w"]
        losses_h += model.losses["h"]
        losses_conf += model.losses["conf"]
        losses_cls += model.losses["cls"]
        losses_recall += model.losses["recall"]
        losses_precision += model.losses["precision"]
        batch_loss += loss.item()

        loss_data = "%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\n"% (
                model.losses["x"] ,
                model.losses["y"] ,
                model.losses["w"] ,
                model.losses["h"] ,
                model.losses["conf"] ,
                model.losses["cls"] ,
                loss.item(),
                model.losses["recall"] ,
                model.losses["precision"],
            )
        
        model.seen += imgs.size(0)
        torch.cuda.empty_cache()
	
    print("[Losses: x %f, y %f, w %f, h %f, conf %f, cls %f, total %f, recall: %.5f, precision: %.5f]"
            % (
                model.losses["x"],
                model.losses["y"],
                model.losses["w"],
                model.losses["h"],
                model.losses["conf"],
                model.losses["cls"],
                loss.item(),
                model.losses["recall"],
                model.losses["precision"],
            )
        )
    loss_file = open('loss.txt', 'a')
    loss_file.write(str(epoch)+'\t' + str(loss.item())+'\t'+str(model.losses["recall"])+'\t'+str(model.losses["precision"])+'\n')
    loss_file.close()

    if epoch % args.checkpoint_interval == 0:
        model.save_weights("%s/%d.weights" % (args.checkpoint_dir, epoch))
        
    print("Compute %d Epoch mAP..." % epoch)

    all_detections = []
    all_annotations = []

    for batch_i, (_, imgs, targets) in enumerate(tqdm.tqdm(test_dataloader, desc="Detecting objects")):

        imgs = Variable(imgs.type(Tensor))

        with torch.no_grad():
            outputs = model(imgs)
            outputs = non_max_suppression(outputs, 80, conf_thres=args.conf_thres, nms_thres=args.nms_thres)

        for output, annotations in zip(outputs, targets):

            all_detections.append([np.array([]) for _ in range(num_classes)])
            if output is not None:
                pred_boxes = output[:, :5].cpu().numpy()
                scores = output[:, 4].cpu().numpy()
                pred_labels = output[:, -1].cpu().numpy()

                sort_i = np.argsort(scores)
                pred_labels = pred_labels[sort_i]
                pred_boxes = pred_boxes[sort_i]

                for label in range(num_classes):
                    all_detections[-1][label] = pred_boxes[pred_labels == label]

            all_annotations.append([np.array([]) for _ in range(num_classes)])
            if any(annotations[:, -1] > 0):

                annotation_labels = annotations[annotations[:, -1] > 0, 0].numpy()
                _annotation_boxes = annotations[annotations[:, -1] > 0, 1:]

                # Reformat to x1, y1, x2, y2 and rescale to image dimensions
                annotation_boxes = np.empty_like(_annotation_boxes)
                annotation_boxes[:, 0] = _annotation_boxes[:, 0] - _annotation_boxes[:, 2] / 2
                annotation_boxes[:, 1] = _annotation_boxes[:, 1] - _annotation_boxes[:, 3] / 2
                annotation_boxes[:, 2] = _annotation_boxes[:, 0] + _annotation_boxes[:, 2] / 2
                annotation_boxes[:, 3] = _annotation_boxes[:, 1] + _annotation_boxes[:, 3] / 2
                annotation_boxes *= args.img_size

                for label in range(num_classes):
                    all_annotations[-1][label] = annotation_boxes[annotation_labels == label, :]

    average_precisions = {}
    for label in range(num_classes):
        true_positives = []
        scores = []
        num_annotations = 0

        for i in tqdm.tqdm(range(len(all_annotations))):
            detections = all_detections[i][label]
            annotations = all_annotations[i][label]

            num_annotations += annotations.shape[0]
            detected_annotations = []

            for *bbox, score in detections:
                scores.append(score)

                if annotations.shape[0] == 0:
                    true_positives.append(0)
                    continue

                overlaps = bbox_iou_numpy(np.expand_dims(bbox, axis=0), annotations)
                assigned_annotation = np.argmax(overlaps, axis=1)
                max_overlap = overlaps[0, assigned_annotation]

                if max_overlap >= args.iou_thres and assigned_annotation not in detected_annotations:
                    true_positives.append(1)
                    detected_annotations.append(assigned_annotation)
                else:
                    true_positives.append(0)

        if num_annotations == 0:
            average_precisions[label] = 0
            continue

        true_positives = np.array(true_positives)
        false_positives = np.ones_like(true_positives) - true_positives
        indices = np.argsort(-np.array(scores))
        false_positives = false_positives[indices]
        true_positives = true_positives[indices]

        false_positives = np.cumsum(false_positives)
        true_positives = np.cumsum(true_positives)

        recall = true_positives / num_annotations
        precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)

        average_precision = compute_ap(recall, precision)
        average_precisions[label] = average_precision

    map_file = open('map.txt', 'a')

    print("Average Precisions:")
    for c, ap in average_precisions.items():
        print("Class: " + str(c) + "  -AP: ", str(ap))
        map_file.write(str(ap) + '\t')
    mAP = np.mean(list(average_precisions.values()))
    print("mAP: " + str(mAP))

    map_file.write(str(mAP) + '\n')
    map_file.close()
    
    if(mAP > best_mAP):
        best_mAP = mAP
        print("new best model!")
        model.save_weights("%s/kitti_best.weights" % (args.checkpoint_dir))
