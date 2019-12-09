'''
Created on Jun 18, 2019

@author: eljurros
'''
import h5py
import os
import numpy as np
import matplotlib.pyplot as plt
from pandas.compat.numpy import np_array_datetime64_compat
from Label_Estimate_Helper_Functions import draw_filter, get__mask

f = h5py.File('/home/eljurros/spare-workplace/Multi_Organ_Segmentation/DataSet/SegTHOR_H5/val.h5', "r")
dest_path = '/home/eljurros/spare-workplace/Multi_Organ_Segmentation/DataSet/images/val'
im_path = os.path.join(dest_path, 'img')
gt_path = os.path.join(dest_path, 'gt')
bb_path = os.path.join(dest_path, 'bb')
if os.path.exists(im_path) is False:
    os.mkdir(im_path)
if os.path.exists(gt_path) is False:
    os.mkdir(gt_path)
if os.path.exists(bb_path) is False:
    os.mkdir(bb_path)
print f.keys()
img = f['img']
label = f['label']
bbox_coord = f['bboxes']



print img.shape
import cv2 as cv
print img.shape, img.dtype


for i in range(497):
    print f['img_id'][i]
    plt.imsave(os.path.join(im_path, f['img_id'][i][0]), img[i])
    plt.imsave(os.path.join(gt_path, f['img_id'][i][0]), label[i])
    coord = get__mask(label[i], 'bbox')
    plt.imsave(os.path.join(bb_path, f['img_id'][i][0]), draw_filter(coord, 'bbox'))
    