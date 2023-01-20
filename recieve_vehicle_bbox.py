import os
import numpy as np
from sliding_window1 import *
import time
import cv2
from setting import win_size, win_size1
def bbox_vehicle(img_path):
    os.chdir("/Users/datle/Desktop/nhan_dien_xe")
    params=load_classifier('car_detect.p')
    os.chdir("/Users/datle/Desktop/license_plate_detection/dataset_test/images")
    img   = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img   = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    bbox, bbox_nms = find_car_multi_scale(img, params, win_size)
    heatmap=draw_heatmap(bbox, img)
    heatmap_thresh= apply_threshhold(heatmap, thresh=win_size['thresh'])
    bbox_heatmap= get_labeled(heatmap_thresh)
    show_crop_image(img, bbox_heatmap)
    return bbox_heatmap
def show_crop_image(img, bbox):
    imgs=[]
    for x in bbox:
        if (x[3]- x[1]) <15 and (x[2]- x[0]) <15:
            continue
        imgs.append(img[x[1]:x[3],x[0]:x[2],:])
    for img in imgs:
        os.chdir("/Users/datle/Desktop/nhan_dien_xe")
        cv2.imwrite('crop_image.jpg', img)
        # cv2.imshow('img',img)
        # cv2.waitKey(0)
img_path="Cars7.png"
print(bbox_vehicle(img_path))
def lp_detect():
    img= cv2.imread('crop_image.jpg', cv2.IMREAD_COLOR)
    img= cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img1= img.copy()
    img2= img.copy()
    os.chdir("/Users/datle/Desktop/nhan_dien_xe")
    params= load_classifier('lp_detect_special.p')
    print(params)
    bbox, bbox_nms = find_car_multi_scale(img, params, win_size1)
    heatmap = draw_heatmap(bbox, img)
    heatmap_thresh = apply_threshhold(heatmap, thresh=win_size1['thresh'])
    bbox_heatmap = get_labeled(heatmap_thresh)

    heatmap_thresh, heatmap = product_heat_and_label_pic(heatmap, heatmap_thresh)

    img = draw(img, bbox)
    img1 = draw(img1, bbox_nms)
    img2 = draw(img2, bbox_heatmap)
    i = np.concatenate((img, img1, img2), axis=0)
    i1 = np.concatenate((heatmap, heatmap_thresh), axis=1)
    i1 = cv2.resize(i1, (600, 300))
    cv2.imshow('i', i)
    cv2.imshow('i1', i1)
    cv2.waitKey(0)

lp_detect()
