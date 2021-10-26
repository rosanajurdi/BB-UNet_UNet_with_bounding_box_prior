'''
Created on Mar 22, 2019

@author: ROsanaEL Jurdi

'''
import torch
import numpy as np
import cv2 as cv
from skimage.transform import resize
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt as distance

def uniq(a):
    return set(torch.unique(a.cpu()).numpy())
def sset(a, sub):
    return uniq(a).issubset(sub)
def one_hot(t, axis=1) :
    '''
    makes sure that the matrice adds up to ones nd that classes50 and 1 ] are a subset of it
    '''
    return simplex(t, axis) and sset(t, [0, 1])
def class2one_hot(seg, C) :
    '''
    stacks the tensors of the one hot representations of the classes + the background class 
    '''
    if len(seg.shape) == 2:  # Only w, h, used by the dataloader
        seg = seg.unsqueeze(dim=0)
    assert sset(seg, list(range(0,C + 1)))
    b, w, h = seg.shape  # type: Tuple[int, int, int]
    res = torch.stack([seg == c for c in range(C+1)], dim=1).type(torch.int32)
    assert res.shape == (b, C+1, w, h)
#     assert one_hot(res)
    return res

def simplex(t, axis=1):
    '''
    @description: finds out if segmentation tensor of background (0, size, size) and forground ( 1, size, size) adds up to one. 
    '''
    _sum = t.sum(axis).type(torch.float32)
    _ones = torch.ones_like(_sum, dtype=torch.float32)
    return torch.allclose(_sum, _ones)

def soft_dist(seg):
    res = np.zeros_like(seg)
    C= 2
    for c in range(C):
        posmask = seg[c].astype(np.bool)
        if posmask.any():
            negmask = ~posmask   # Negation of backgorund matrix
            res[c] = distance(negmask) * negmask - (distance(posmask) - 1) * posmask
    return res
def one_hot2dist(seg) :
    assert one_hot(torch.Tensor(seg), axis=0)
    C= len(seg)
    res = np.zeros_like(seg)
    for c in range(C):
        posmask = seg[c].astype(np.bool)
        if posmask.any():
            negmask = ~posmask   # Negation of backgorund matrix
            res[c] = distance(negmask) * negmask - (distance(posmask) - 1) * posmask
    return res


def one_hot_my(a): # one hot encoder for seg masks
    g1=a[:,:]==0
    g2=a[:,:]==1
    return torch.stack((g1,g2),0)


def Get_DistMap(mask, C):
    '''
    for now this gets the distance maps for only one organ. 
    takes as input a binary mask of the organ and provides a distance map and a onehot vector
    '''
    mask_tensor = torch.tensor(mask, dtype=torch.int64)  ###################################
    mask_onehot = class2one_hot(mask_tensor, C)  # because the res is bchw maybe it is abinary segmentation
    mask_distmap = one_hot2dist(np.squeeze(mask_onehot.cpu().numpy()))
    mask_distmap = torch.tensor(mask_distmap, dtype=torch.int64)
    return mask_distmap, mask_onehot


def squarify(M,val):
    (a, b) = M.shape
    a = 512 - M.shape[0]
    b = 512 - M.shape[1]
    if a % 2 == 0:
        padd_1 = (a/2,a/2)
    else: 
        padd_1 = (a/2 + 1, a/2)
    if b % 2 == 0:
        padd_2 = (b/2,b/2)
    else:
        padd_2 = (b/2 +1, b/2)
    padding = (padd_1, padd_2)
    padded_img = np.pad(M,padding,mode='constant',constant_values=val)
    assert padded_img.shape == (512,512)
    return padded_img


def Preprocess_Prostate(img1, img2, gt_img):
    norm_inputt2 = None
    inputt1 = clip(img1, -1000, 3000)
    norm_inputt1 = normalize(inputt1)
    if img2 is not None:
        inputt2 = clip(img2, -1000, 3000)
        norm_inputt2 = normalize(inputt2)
        
    return norm_inputt1, norm_inputt2, gt_img

import cv2
def Preprocess_Hippocampur(img1, img2, gt, size):
    norm_inputt2 = None
    inputt1 = clip(img1, -1000, 3000)
    norm_inputt1 = normalize(inputt1)
    resize_inputt1 = cv2.resize(np.array(img1), (size, size), interpolation = cv2.INTER_NEAREST)
    gt = cv2.resize(np.array(gt), (size, size), interpolation = cv2.INTER_NEAREST)
    if img2 is not None:
        inputt2 = clip(img2, -1000, 3000)
        norm_inputt2 = normalize(inputt2)
        norm_inputt2 = cv2.resize(img2, (size, size), interpolation = cv2.INTER_NEAREST)
        
    return resize_inputt1, norm_inputt2, gt
def clip(inputt, lower, upper):
    # clip the image
    inputt[np.where(inputt < -1000)] = -1000
    inputt[np.where(inputt > 3000)] = 3000
    return inputt
def normalize(inputt):
    # normalize:
    if len(np.unique(inputt)) != 1:
        mean = np.mean(inputt)
        std = np.std(inputt)
        assert std != 0
        inputt = np.divide((inputt - mean), std)

    return inputt
def Preprocess_SegTHOR(img, gt_img, contours):
    '''
    Gets an input image as well as tecountour and preprocesses it 
    returns preprocessed image
    '''
    try:
        inputt = np.tile(-1000, (img.shape[0], img.shape[0]))
        inputt[contours == np.max(contours)] = img[contours == np.max(contours)]
        inputt = clip(inputt, -1000, 3000)
        norm_inputt = normalize(inputt)

        return norm_inputt, gt_img
    except:
        pass


