'''
Created on Jun 17, 2019

@author: eljurros
'''
'''
Created on Mar 22, 2019

@author: ROsanaEL Jurdi
def Preprocess_Image(img, contours, gt_img):
    
    Gets an input image as well as tecountour and preprocesses it 
    returns preprocessed image
    
    try:
        inputt = np.tile(-1000, (img.shape[0], img.shape[0]))
        inputt[contours == np.max(contours)] = img[contours == np.max(contours)]
    
        # clip the image
        inputt[np.where(inputt < -1000)] = -1000
        inputt[np.where(inputt > 3000)] = 3000
    
        # Now crop
        (x, y) = np.where(contours == np.max(contours))
        (topx, topy) = (np.min(x), np.min(y))
        (bottomx, bottomy) = (np.max(x), np.max(y))
        inputt = inputt[topx:bottomx+1, topy:bottomy+1]
        gt = gt_img[topx:bottomx+1, topy:bottomy+1]
        contours = contours[topx:bottomx+1, topy:bottomy+1]
    
        # normalize:
        mean = np.mean(inputt)
        std = np.std(inputt)
        norm_inputt = np.divide((inputt - mean), std)
        
        inputt = squarify(norm_inputt.copy(), -1000)
        gt = squarify(gt.copy(), 0)
        contours = squarify(contours, 0)

        return norm_inputt, gt_img, contours
    except:
        pass

'''
import torch
import numpy as np
import cv2 as cv
from skimage.transform import resize
import matplotlib.pyplot as plt
def class2one_hot(seg, C) :
    '''
    stacks the tensors of the one hot representations of the classes
    '''
    if len(seg.shape) == 2:  # Only w, h, used by the dataloader
        seg = seg.unsqueeze(dim=0)

    res = torch.stack([seg == c for c in range(C)], dim=1).type(torch.int32)
    return res

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


def Preprocess_Image(img, contours, gt_img):
    '''
    Gets an input image as well as tecountour and preprocesses it 
    returns preprocessed image
    '''
    try:
        inputt = np.tile(-1000, (img.shape[0], img.shape[0]))
        inputt[contours == np.max(contours)] = img[contours == np.max(contours)]

        # clip the image
        inputt[np.where(inputt < -1000)] = -1000
        inputt[np.where(inputt > 3000)] = 3000


        # normalize:
        mean = np.mean(inputt)
        std = np.std(inputt)
        norm_inputt = np.divide((inputt - mean), std)

        return norm_inputt, gt_img, contours
    except:
        pass
