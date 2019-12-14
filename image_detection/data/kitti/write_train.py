import sys
import numpy as nu
import os

f = open("/home/jiajunh/camera_detection/myYOLO/data/kitti/train.txt", 'w')
for i in range(1000):
    f.write("/home/jiajunh/camera_detection/myYOLO/train_data/images/train/" + "{:06d}".format(i) + ".png"+ '\n')
f.close()

f = open("/home/jiajunh/camera_detection/myYOLO/data/kitti/val.txt", 'w')
for i in range(100):
    f.write("/home/jiajunh/camera_detection/myYOLO/train_data/images/test/" + "{:06d}".format(i) + ".png"+ '\n')
f.close()
