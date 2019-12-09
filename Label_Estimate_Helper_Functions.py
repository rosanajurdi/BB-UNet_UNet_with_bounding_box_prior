'''
Created on Apr 10, 2019

@author: eljurros

'''
import os
import cv2
import random as rng
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import cv2 as cv
from Dataset_Helpers import class2one_hot
from torchvision import transforms
t = transforms.ToTensor()
import torch
def draw_filter(bboxes_coord, typ):
    if typ == 'circular':
        noisy_mask = circ_mask((512, 512), bboxes_coord)
    else:
        noisy_mask = rect_mask((512,512), bboxes_coord)
    return noisy_mask

def get__mask(gt, typ ='bbox'):
    bboxes = []
    gt_one_hot = np.squeeze(class2one_hot(t(gt), 4))
    for mask_id in range(4):
        bbox = extract_bbox(gt_one_hot[mask_id])

        if typ == 'circular':
            c_props = get_circle_attributes_from_bb(bbox)
            bboxes.append(c_props)
        else:
            bboxes.append(bbox)

    return bboxes


def circ_mask(shape, bboxes):
    concat_circ = []
    color = (255, 0, 255)
    for attr in bboxes:
        label_img = np.zeros(shape, np.uint8)
        r, c1, c2 = attr[0], attr[1], attr[2]
        cv.circle(label_img, (int(c1), int(c2)), int(np.ceil(r/2)), color, cv.FILLED)
        concat_circ.append(label_img)
    return np.array(concat_circ)

def rect_mask(shape, bboxes, mode = 'norm'):
    """Given a bbox and a shape, creates a mask (white rectangle foreground, black background)
    Param:
        shape: shape (H,W) or (H,W,1)
        bbox: bbox numpy array [y1, x1, y2, x2]
    Returns:
        mask
    """
    concat_mask = []
    bboxes = bboxes    
    for bbox in list(bboxes):
        mask = np.zeros((512, 512), np.uint8)
        if len(np.unique(bbox)) != 1:
            mask[np.int(bbox[0]):np.int(bbox[2]),
                       np.int(bbox[1]):np.int(bbox[3])] = 255
        concat_mask.append(mask)
    return np.array(concat_mask)

def extract_bbox(mask, order='y1x1y2x2'):
    """Compute bounding box from a mask.
    Param:
        mask: [height, width]. Mask pixels are either >0 or 0.
        order: ['y1x1y2x2' | ]
    Returns:
        bbox numpy array [y1, x1, y2, x2] or tuple x1, y1, x2, y2.
    """
    horizontal_indicies = np.where(np.any(mask, axis=0))[0]
    vertical_indicies = np.where(np.any(mask, axis=1))[0]
    if horizontal_indicies.shape[0]:
        x1, x2 = horizontal_indicies[[0, -1]]
        y1, y2 = vertical_indicies[[0, -1]]
        x2 += 1
        y2 += 1
    else:
        x1, x2, y1, y2 = 0, 0, 0, 0
    if order == 'x1y1x2y2':
        return x1, y1, x2, y2
    else:
        return ([int(y1), int(x1), int(y2), int(x2)])


def get_circle_attributes_from_bb(bbox):
    x1 = bbox[1]
    x2 = bbox[3]
    y1 = bbox[0]
    y2 = bbox[2]
    # r = np.max([np.abs(x2 - x1), np.abs(y2 - y1)])
    r = np.sqrt((x2-x1)**2 + (y2-y1)**2)
    cx = np.divide((x1 + x2), 2)
    cy = np.divide(y1 + y2, 2)
    return [int(r/2), int(cx), int(cy)]

def draw_bbox(image, mask):
    img = image.copy()
    bbox = extract_bbox(mask)
    cv.rectangle(img, (bbox[1], bbox[0]), (bbox[3], bbox[2]), (255, 0, 0), 2)
    return img, bbox



def circular_label_estimate(image,mask, mode='conv'):
    """
    """

    bbox = extract_bbox(mask)
    r, c = get_circle_attributes_from_bb(bbox)

    label_img = np.zeros(image.shape, np.uint8)
    color = (255, 0, 255)
    cv.circle(label_img, (int(c[0]), int(c[1])), int(np.ceil(r)), color, cv.FILLED)

    if mode == 'conv':
        clabel_mask = image*label_img
    else: 
        clabel_mask = label_img

    return clabel_mask



    
