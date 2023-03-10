import os

import numpy as np

from sliding_window1 import *
import time
import cv2
os.chdir("/Users/datle/Desktop/nhan_dien_xe")
params=load_classifier('car_detect.p')
os.chdir("/Users/datle/Desktop/license_plate_detection/dataset_test/images")
img   = cv2.imread('Cars7.png', cv2.IMREAD_COLOR)
img   = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img1  = img.copy()
img2  = img.copy()
start= time.time()
bbox, bbox_nms= find_car_multi_scale(img,params, win_size)
end= time.time()
print(f'time is: {end-start}')
heatmap=draw_heatmap(bbox, img)
heatmap_thresh= apply_threshhold(heatmap, thresh=win_size['thresh'])
bbox_heatmap= get_labeled(heatmap_thresh)

heatmap_thresh, heatmap= product_heat_and_label_pic(heatmap, heatmap_thresh)

img   =draw(img, bbox)
img1  =draw(img1, bbox_nms)
img2  =draw(img2, bbox_heatmap)
i= np.concatenate((img,img1,img2),axis=0)
i1= np.concatenate((heatmap, heatmap_thresh), axis=1)
i1= cv2.resize(i1, (600,300))
cv2.imshow('i',i)
cv2.imshow('i1',i1)
cv2.waitKey(0)