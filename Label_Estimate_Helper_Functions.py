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
from Dataset_Helpers import class2one_hot, sset
from torchvision import transforms
t = transforms.ToTensor()
import torch
from skimage import measure

def simplex(t, axis=1) -> bool:
    _sum = t.sum(axis).type(torch.float32)
    _ones = torch.ones_like(_sum, dtype=torch.float32)
    return torch.allclose(_sum, _ones)
def create_seed(shape=(512, 512), bboxes = None, seed_div = 10, n_organ = 4):
    """Given a bbox and a shape, creates a seed (white rectangle foreground,
    black background)
    Param:
        shape: shape (H,W) or (H,W,1)
        bbox: bbox numpy array [y1, x1, y2, x2]
    Returns:
        mask
    """
    assert bboxes is not None
    if n_organ > 1:
        concat_mask = []
        for bbox in list(bboxes):
            mask = np.zeros(shape, np.uint8)
            if len(np.unique(bbox)) != 1:
                x1 = bbox[1]
                x2 = bbox[3]
                y1 = bbox[0]
                y2 = bbox[2]
                sidex = np.absolute(x1 - x2)
                sidey = np.absolute(y1 - y2)
                smallest_side = np.min([sidex, sidey])/seed_div
                d  = get_circle_attributes_from_bb(bbox)
                r, c0, c1 = d[0], d[1], d[2]
                mask[np.int(c1 - smallest_side):np.int(c1 + smallest_side), np.int(c0 - smallest_side):np.int(c0 + smallest_side)] = 1
            concat_mask.append(mask)
        x = np.array(concat_mask)
    else:
        mask = np.zeros(shape, np.uint8)
        x1 = bboxes[1]
        x2 = bboxes[3]
        y1 = bboxes[0]
        y2 = bboxes[2]
        sidex = np.absolute(x1 - x2)
        sidey = np.absolute(y1 - y2)
        smallest_side = np.min([sidex, sidey])/seed_div
        d  = get_circle_attributes_from_bb(bboxes)
        r,c0,c1 = d[0], d[1], d[2]
        mask[np.int(c1 - smallest_side):np.int(c1 + smallest_side), np.int(c0 - smallest_side):np.int(c0 + smallest_side)] = 1
        x = mask

    return x

def Variate_Bbox(shape=(512, 512), bboxes = None, bbox_range = 70, n_organ = 4):
    """Given a bbox and a shape, creates a seed (white rectangle foreground,
    black background)
    Param:
        shape: shape (H,W) or (H,W,1)
        bbox: bbox numpy array [y1, x1, y2, x2]
    Returns:
        mask
    """
    assert bboxes is not None
    if n_organ > 1:
        concat_mask = []
        for bbox in list(bboxes):
            mask = np.zeros(shape, np.uint8)
            if len(np.unique(bbox)) != 1:
                x1 = bbox[1]
                x2 = bbox[3]
                y1 = bbox[0]
                y2 = bbox[2]
                sidex = np.absolute(x1 - x2)
                sidey = np.absolute(y1 - y2)
                s_x, s_y = sidex/bbox_range, sidey/bbox_range
                d  = get_circle_attributes_from_bb(bbox)
                r, c0, c1 = d[0], d[1], d[2]
                mask[np.int(x1 - s_x):np.int(x2 + s_x), np.int(y1 - s_y):np.int(y2 + s_y)] = 255
            concat_mask.append(mask)
        x = np.array(concat_mask)
    else:
        mask = np.zeros(shape, np.uint8)
        x1 = bboxes[1]
        x2 = bboxes[3]
        y1 = bboxes[0]
        y2 = bboxes[2]
        sidex = np.absolute(x1 - x2)
        sidey = np.absolute(y1 - y2)
        smallest_side = np.min([sidex, sidey])/seed_div
        d  = get_circle_attributes_from_bb(bboxes)
        r,c0,c1 = d[0], d[1], d[2]
        mask[np.int(c1 - smallest_side):np.int(c1 + smallest_side), np.int(c0 - smallest_side):np.int(c0 + smallest_side)] = 1
        x = mask

    return x
   
def draw_filter(bboxes_coord, typ, n_organ, img_size = 512):
    if typ == 'circular':
        noisy_mask = circ_mask((img_size, img_size), bboxes_coord, n_organ)
    else:
        
        noisy_mask = rect_mask((img_size,
                                img_size),
                                bboxes_coord,
                                n_organ)
    
    return noisy_mask

def get__mask(gt, typ ='bbox', organ_n = 2):
    bboxes = []
    gt_one_hot = np.squeeze(class2one_hot(t(gt), organ_n))
    assert gt_one_hot.shape[0] == organ_n + 1
    if gt_one_hot.shape[0] > 2:
        for mask_id in range(1, gt_one_hot.shape[0]):
            bbox = extract_bbox(gt_one_hot[mask_id])
            if typ == 'circular':
                c_props = get_circle_attributes_from_bb(bbox)
                bboxes.append(c_props)
            else:
                bboxes.append(bbox)
    else:
        bboxes = extract_bbox(gt)
        if typ == 'circular':
            c_props = get_circle_attributes_from_bb(bboxes)
            bboxes = c_props


    return gt_one_hot, bboxes


def circ_mask(shape, c_attr, n_organ):
    concat_circ = []
    color = (255, 0, 255)
    if n_organ > 1:
        for attr in c_attr:
            label_img = np.zeros(shape, np.uint8)
            r, c1, c2 = attr[0], attr[1], attr[2]
            cv.circle(label_img, (int(c1), int(c2)), int(np.ceil(r/2)), color, cv.FILLED)
            concat_circ.append(label_img)
        x = np.array(concat_circ)
    else:
        label_img = np.zeros(shape, np.uint8)
        r, c1, c2 = c_attr[0], c_attr[1], c_attr[2]
        cv.circle(label_img, (int(c1), int(c2)), int(np.ceil(r/2)), color, cv.FILLED)
        x = label_img
    return x

def rect_mask(shape, bboxes, n_organs):
    """Given a bbox and a shape, creates a mask (white rectangle foreground, black background)
    Param:
        shape: shape (H,W) or (H,W,1)
        bbox: bbox numpy array [y1, x1, y2, x2]
    Returns:
        mask
    """
    concat_mask = []
    bboxes = bboxes
    if n_organs > 1:
        if len(bboxes[0]) != 1:
            for bbox in list(bboxes):
                mask = np.zeros(shape, np.uint8)
                if len(np.unique(bbox)) != 1:
                    mask[np.int(bbox[0]):np.int(bbox[2]),
                        np.int(bbox[1]):np.int(bbox[3])] = 255
                concat_mask.append(mask)
        x = np.array(concat_mask)
    else:
        mask = np.zeros(shape, np.uint8)
        mask[np.int(bboxes[0]):np.int(bboxes[2]),
                    np.int(bboxes[1]):np.int(bboxes[3])] = 255
        x = mask
    return x

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
    d = np.sqrt((x2-x1)**2 + (y2-y1)**2)
    cx = np.divide((x1 + x2), 2)
    cy = np.divide(y1 + y2, 2)
    return [int(d/2), int(cx), int(cy)]

def draw_bbox(bbox, size):
    mask = np.zeros((size, size), np.uint8)
    mask[np.int(bbox[0]):np.int(bbox[2]), np.int(bbox[1]):np.int(bbox[3])] = 255
    return mask

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

from scipy.spatial.distance import cdist

def Get_Upper_Lower_boundaries(mask, order='y1x1y2x2', size = 512):
    """Compute bounding box from a mask.
    Param:
        mask: [height, width]. Mask pixels are either >0 or 0.
        order: ['y1x1y2x2' | ]
    Returns:
        bbox numpy array [y1, x1, y2, x2] or tuple x1, y1, x2, y2.
    """
    contours, _ = cv.findContours(np.array(mask), cv2.RETR_CCOMP,
                                      cv2.CHAIN_APPROX_SIMPLE)
    inner = np.zeros((size,size), np.uint8)
    outer = np.zeros((size,size), np.uint8)
    if contours != []:
        con = np.array(mask)
        cv2.drawContours(con, contours, contourIdx = -1, color = (255, 0, 0), thickness = 2)
        bbox = extract_bbox(mask)
        bb = draw_bbox(bbox, mask.shape[-1])
        c = get_circle_attributes_from_bb(bbox)
        circle = np.zeros((size,size), np.uint8)
        
        cv.circle(circle, (int(c[1]), int(c[2])), int(np.ceil(c[0])), (255, 0, 255) , cv.FILLED)
        d1 = cdist(np.array(contours[0]).reshape(-1,2),np.array([c[1], c[2]]).reshape(-1,2))
        d2 = cdist(np.array(contours[1]).reshape(-1,2),np.array([c[1], c[2]]).reshape(-1,2))
        d = np.concatenate((d1, d2))
        d_min = np.min(d)
        d_max = np.max(d)
        inner = np.zeros((size,size), np.uint8)
        cv.circle(inner, (int(c[1]), int(c[2])), int(np.ceil(d_min)), (255, 0, 255) , cv.FILLED)
        outer = np.zeros((size,size),np.uint8 )
        cv.circle(outer, (int(c[1]), int(c[2])), int(np.ceil(d_max)), (255, 0, 255) , cv.FILLED)


    return inner, outer, contours


def Get_contour_characteristics(image):
    contours = measure.find_contours(image, 0.5)
    listt = []
    for n, contour in enumerate(contours):
        summ = 0
        for y,x in zip(contour[:, 1], contour[:, 0]):
            summ += image[np.int(x)][np.int(y)]
        listt.append(summ)
    return listt, contours
