from pfmfile import *
import numpy as np
import os
import sys
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
    import cv2
from open3d import *
import csv

def main():
  leftdir = "./2011_09_26/2011_09_26_drive_0009_sync/image_02/data/"
  rightdir = "./2011_09_26/2011_09_26_drive_0009_sync/image_03/data/"
  sgmdir = "./sgm_map/"
  pcddir = "./pcd/"
  dispdir = "./disparity"
  start_num = 0
  num = 1
  for i in range(start_num, start_num+num):
      print("Image Number: ", i, "/", num)
      #print("{:04d}".format(i))
      leftimg = cv2.imread(leftdir + "000000" +"{:04d}".format(i) + ".png")
      rightimg = cv2.imread(rightdir + "000000" +"{:04d}".format(i) + ".png")
    
      bl = 0.05956621+0.473105
      focal = 721.5377
      height, width, chan = leftimg.shape

      gray0 = cv2.cvtColor(leftimg, cv2.COLOR_BGR2GRAY)
      gray1 = cv2.cvtColor(rightimg, cv2.COLOR_BGR2GRAY)
      
      window_size = 9
      min_disp = 0
      num_disp = 320

      stereo = cv2.StereoSGBM_create(minDisparity = min_disp,
              numDisparities = 192,#num_disp,
              blockSize = window_size,
              uniquenessRatio = 5,#10,
              speckleWindowSize = 100,
              speckleRange = 2,
              disp12MaxDiff = 1,
              P1 = 8*3*window_size**2,
              P2 = 32*3*window_size**2
              )
      #print(np.min(stereo.compute(gray0, gray1)))
      #print(stereo.compute(gray0, gray1).astype(np.float32))
      disp = stereo.compute(gray0, gray1).astype(np.float32) / 16.0
      writePFM(sgmdir + "000000" + "{:04d}".format(i) + ".pfm", disp)
      cv2.imwrite(dispdir+"/000000" + "{:04d}".format(i) + ".png", disp)

      print(disp.shape)
      cloud = np.zeros((width * height, 3))
      colors = np.zeros((width * height, 3))

      for k in range(width):
          for j in range(height):
              u = k - (width/2)
              v = j - (height/2)
              d = disp[j,k]

              if d > 16:
                  z = focal * bl / d
                  x = u * z / focal
                  y = v * z / focal
                  cloud[k*height+j] = [x,y,z]
                  color = leftimg[j,k]/255
                  colors[k*height+j] = [color[2],color[1],color[0]]

      #Clear out array rows that are empty
      #print(cloud.shape)
      cloud = cloud[~np.all(cloud==0, axis = 1)]
      #print(cloud.shape)
      colors = colors[~np.all(colors==0, axis = 1)]
      pcd = PointCloud()
      pcd.points = Vector3dVector(cloud)
      pcd.colors = Vector3dVector(colors)
      write_point_cloud(pcddir + "000000" + "{:04d}".format(i) + ".ply", pcd)
      
if __name__ == '__main__':
  main()