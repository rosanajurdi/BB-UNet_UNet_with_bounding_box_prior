'''
Created on Jul 6, 2019

@author: eljurros
'''
import torch
import numpy as np
from torch import einsum
from torch.autograd import Variable
from Dataset_Helpers import simplex, one_hot, one_hot2dist, soft_dist
import torch.nn.functional as F
import matplotlib.pyplot as plt
from skimage.measure import label
import torch, torch.nn as nn, numpy as np, matplotlib.pyplot as plt
# see full source for setup of problem
# X, y, and beta_ols are created
from torch.autograd import Variable

class TopLoss(nn.Module):
    def __init__(self, size):
        super(TopLoss, self).__init__()
        self.pdfn = LevelSetLayer2D(size=size,  sublevel=False)
        self.topfn = PartialSumBarcodeLengths(dim=1, skip=1) # penalize more than 1 hole
        self.topfn2 = SumBarcodeLengths(dim=0) # penalize more than 1 max

    def forward(self, beta):
        dgminfo = self.pdfn(beta)
        return self.topfn(dgminfo) + self.topfn2(dgminfo)



def threshold(array):
    array = (array > 0.91) * 1.0
    return array

def contour(x, thresh_width=10):
    '''
    Differenciable aproximation of morphological skelitonization operaton
    thresh_width - maximal expected width of vessel
    '''   
    min_pool_x = torch.nn.functional.max_pool2d(x*-1, (3, 3), 1, 1)*-1
    contour = torch.nn.functional.relu(torch.nn.functional.max_pool2d(min_pool_x, (3, 3), 1, 1) - min_pool_x)
    return contour.sum(axis=(1,2,3))
class Contour_loss():
    '''
    inputs shape  (batch, channel, height, width).
    calculate clDice loss
    Because pred and target at moment of loss calculation will be a torch tensors
    it is preferable to calculate target_skeleton on the step of batch forming,
    when it will be in numpy array format by means of opencv
    '''
    def __init__(self, **kwargs):
        print(f"Initialized ")
    def __call__(self, probs, target):
        pc = probs.type(torch.float32)
        tc = target.type(torch.float32)
        b, _, w, h = pc.shape
        cl_pred = contour(pc)
        target_skeleton = contour(tc)
        big_pen = (cl_pred - target_skeleton) ** 2
        contour_loss = big_pen / (w * h)
    
        return contour_loss.mean()




def CSME_CALC(gt_1):
    c = 0
    b = gt_1.clone()
    for i in range(0, gt_1.shape[1]):
        for j in range(0, gt_1.shape[2]):
            if i == 0 and j != 0:
                if gt_1[:, i,j] == 1:
                    if gt_1[:, i,j -1] != 1:
                        c = c+1
                    b[:, i,j] = c 
            if i != 0 and j !=0:
                if gt_1[:, i,j] == 1:
                    if gt_1[:,i,j-1] != 1 and gt_1[:,i-1,j] != 1:
                        c = c +1
                        b[:, i,j] = c
                    elif gt_1[:,i,j -1] == 1 and gt_1[:,i-1,j] != 1:
                        b[:, i,j] =b[:,i,j -1]
                    elif gt_1[:,i,j-1] != 1 and gt_1[:,i-1,j] == 1:
                        b[:, i,j] = b[:,i-1,j]
                    elif gt_1[:,i,j-1] == 1 and gt_1[:,i-1,j] == 1 and b[:,i,j-1] == b[:,i-1,j]:
                        b[:, i,j] = b[:,i-1,j]
                    elif gt_1[:,i,j-1] == 1 and gt_1[:,i-1,j] == 1 and b[:,i,j-1] != b[:,i-1,j]:
                        b[:, i,j] = b[:,i,j-1]
                        b[b == b[0][i-1][j]] = b[:,i,j-1]
    return torch.tensor(len(torch.unique(b))-1, dtype=torch.float, requires_grad = True)



class CMSE(torch.nn.HingeEmbeddingLoss):
    def __init__(self,margin=1.0, size_average=True, reduce=True):
        super(CMSE, self).__init__()
        self.pred_con =0 
        
    def forward(self, gt, predicted):
        
        gt_2 = gt

        for p in predicted:
            self.pred_con += (CSME_CALC(p)-1)**2

        return self.pred_con



class F_LOSS(torch.nn.HingeEmbeddingLoss):
    def __init__(self,margin=1.0, size_average=True, reduce=True):
        super(F_LOSS, self).__init__(margin, size_average, reduce)
        
    def forward(self, cs, ct):
        D = 1.0 # constant value
        x = ((torch.abs(cs - ct))/(2*D))**2 #formula
        hinge_loss = torch.nn.HingeEmbeddingLoss()
        y = hinge_loss(x, ct) # sorry, couldn't write the whole correct equation here
        return y



def DIST_Pen(gt, dist_bbox, predicted, gt_samples, thresh = 100):
    '''
    novel loss function that takes into consideration the connectedness of elements.
    h: rounded prediction
    g: distance map of the prediction
    v: the error map between distance maps.
    mse : the error value.
    
    '''
    dist_b = dist_bbox[:, 1:, :]
    error = []
    for i in range(0, len(predicted)):
        h = np.array(predicted[i].round())
        g = soft_dist(h)
        a = torch.tensor(np.squeeze(dist_b[i])).float()
        predicted_distmap = torch.tensor(g).float()
        v = torch.abs(a - predicted_distmap[-1])**2/thresh
        #v_norm = (v + 10e-7)/(v.max() - v.min()+10e-7)
        mse = (1- torch.exp(torch.tensor(-torch.clamp(v.mean(), -10,10))))/(1+torch.exp(torch.tensor(-torch.clamp(v.mean(), -10,10))))
        error.append(mse) 
    mse_error = torch.tensor(np.array(np.stack(error))).float()
    return torch.div(Variable(mse_error.mean(),requires_grad=True), predicted.shape[0]), v, a, predicted_distmap[-1]

def DIST_Pen2(gt, dist_bbox, predicted, gt_samples, thresh = 100):
    '''
    novel loss function that takes into consideration the connectedness of elements.
    h: rounded prediction
    g: distance map of the prediction
    v: the error map between distance maps.
    mse : the error value.
    
    '''
    dist_b = dist_bbox[:, 1:, :]
    error = []
    for i in range(0, len(predicted)):
        diff = np.array(gt[i]) - predicted[i][1]
        h = np.array(diff.round())
        g = soft_dist(h)
        a = torch.tensor(np.squeeze(dist_b[i])).float()
        predicted_distmap = torch.tensor(g).float()
        v = torch.abs(a - predicted_distmap[-1])**2/thresh
        #v_norm = (v + 10e-7)/(v.max() - v.min()+10e-7)
        mse = (1- torch.exp(torch.tensor(-torch.clamp(v.mean(), -10,10))))/(1+torch.exp(torch.tensor(-torch.clamp(v.mean(), -10,10))))
        error.append(mse) 
    mse_error = torch.tensor(np.array(np.stack(error))).float()
    return torch.div(Variable(mse_error.mean(),requires_grad=True), predicted.shape[0]), v, a, predicted_distmap[-1]


class SurfaceLoss():
    def __init__(self, **kwargs):
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        self.idc = kwargs["idc"]
        print(self.__class__.__name__,kwargs)

    def __call__(self, probs, dist_maps) :
        try:
            assert simplex(probs)
            assert not one_hot(dist_maps)

            pc = probs.type(torch.float32)[:, self.idc, ...]
            dc = 10*dist_maps.type(torch.float32)[:, self.idc, ...]
            
            #multipled = einsum("bcwh,bcwh->bcwh", pc, dc)
            multipled = pc*dc
            loss = multipled.mean()
        except:
            print(probs)
            print(probs.shape)
            print(loss)

        return loss
def Constraint_UpperB_loss(lambdaa, preds, b):
    '''
    the constraint penalty loss for multi-label segmentation 
    lamdaa is the weighting factor of the constraint loss relative to the dice loss
    a is the lower bound
    b is the upper bound
    Dimensions::::
        preds = (4,512,512)
        a = (4,1)
        b = (4,1)
    1048576 is the 512x512 normalizations per image size
    '''
    S = torch.sum(preds,dim=3)
    Vs = torch.sum(S, dim=2)
    C = (Vs - b)**2/1048576
    Constrait_loss_per_batch = torch.sum(C, dim=0)
    # return Variable(torch.sum(torch.tensor(C)),requires_grad=True)
    # Normalizing by the batch size
    return lambdaa*torch.div(Variable(torch.tensor(Constrait_loss_per_batch).float(),
                                      requires_grad=True), preds.shape[0])
class CrossEntropy():
    def __init__(self, **kwargs):
        print(self.__class__.__name__,kwargs)
    
    def __call__(self, probs, target):
        #assert simplex(probs)
         #assert simplex(target)

        log_p = (torch.clamp(probs, 10e-4).type(torch.float32) +10e-7).log()
        mask = target.type(torch.float32)

        #loss = - einsum("bcwh,bcwh->", mask, log_p)
        loss = mask*log_p
        loss = - loss.sum()
        loss /= mask.sum() +10e-7

        return loss

def Constraint_loss(lambdaa, preds, a,b):
    '''
    the constraint penalty loss for multi-label segmentation 
    lamdaa is the weighting factor of the constraint loss relative to the dice loss
    a is the lower bound
    b is the upper bound
    Dimensions::::
        preds = (4,512,512)
        a = (4,1)
        b = (4,1)
    1048576 is the 512x512 normalizations per image size
    
    '''
    S = torch.sum(preds,dim=3)
    Vs = torch.sum(S, dim=2)
    C = torch.zeros((preds.shape[0], preds.shape[1]))
    comp_val_a = Vs.le(torch.tensor(a).float())
    comp_val_b = Vs.ge(torch.tensor(b).float())

    for i, vals in enumerate(zip(np.squeeze(np.array(comp_val_a)),
                                 np.squeeze(np.array(comp_val_b)))):

        for j, val in enumerate(zip(vals[0], vals[1])):
            if val[0] == 1:                     # if the condition the Vs<a is true
                C[i][j] = (Vs[i][j] - a[j])*(Vs[i][j] - a[j])/(1048576*1048576)
            elif val[1] == 1:
                C[i][j] = (Vs[i][j] - b[j])*(Vs[i][j] - b[j])/(1048576*1048576)
        Constrait_loss_per_batch = torch.sum(C,dim=0)
    # return Variable(torch.sum(torch.tensor(C)),requires_grad=True)
    # Normalizing by the batch size
    return lambdaa*torch.div(Variable(torch.tensor(Constrait_loss_per_batch).float(),
                                      requires_grad=True), preds.shape[0])

def Constraint_loss_single_organ_MB(lambdaa, preds, a,b):
    '''
    the constraint penalty loss for single-label segmentation with multiple bounds
    lamdaa is the weighting factor of the constraint loss relative to the dice loss
    a is the lower bound
    b is the upper bound
    1048576 is the 512x512 normalizations per image size
    
    '''
    S = torch.sum(preds,dim=3)
    Vs = torch.sum(S, dim=2)
    C = torch.zeros((preds.shape[0], preds.shape[1]))
    comp_val_a = Vs.reshape((-1,)).le(torch.tensor(a).float()).reshape((-1,1))
    comp_val_b = Vs.reshape((-1,)).ge(torch.tensor(b).float()).reshape((-1,1))

    for i, vals in enumerate(zip(np.squeeze(np.array(comp_val_a)),
                                 np.squeeze(np.array(comp_val_b)))):
        if a[i] == 0 and b[i] == 0:
            C[i] = 0
        else:
            if vals[0] == 1:                     # if the condition the Vs<a is true
                C[i] = (Vs[i] - a[i])*(Vs[i] - a[i])/(1048576*1048576)
            elif vals[1] == 1:
                C[i] = (Vs[i] - b[i])*(Vs[i] - b[i])/(1048576*1048576)
    return lambdaa*Variable(torch.tensor(C).float(), requires_grad=True)

    '''
    the constraint penalty loss for single-label segmentation with single bounds
    suitable for batch training and per image training first if else 
    lamdaa is the weighting factor of the constraint loss relative to the dice loss
    a is the lower bound
    b is the upper bound
    1048576 is the 512x512 normalizations per image size
    
    '''
    S = torch.sum(preds,dim=3)
    Vs = torch.sum(S, dim=2)
    C = torch.zeros((preds.shape[0], preds.shape[1]))
    comp_val_a = Vs.le(torch.tensor(a).float())
    comp_val_b = Vs.ge(torch.tensor(b).float())
    if preds.shape[0] >= 2:
        for i, vals in enumerate(zip(np.squeeze(np.array(comp_val_a)),
                                     np.squeeze(np.array(comp_val_b)))):
    
            if vals[0] == 1:                     # if the condition the Vs<a is true
                C[i] = (Vs[i] - a)*(Vs[i] - a)/(1048576*1048576)
            elif vals[1] == 1:
                C[i] = (Vs[i] - b)*(Vs[i] - b)/(1048576*1048576)
    else:
        if comp_val_a == 1:                     # if the condition the Vs<a is true
            C = (Vs - a)*(Vs - a)/(1048576*1048576)
        elif comp_val_b == 1:
            C = (Vs - b)*(Vs - b)/(1048576*1048576)
    Constrait_loss_per_batch = torch.sum(C,dim=0)
    # return Variable(torch.sum(torch.tensor(C)),requires_grad=True)
    return lambdaa*torch.div(Variable(torch.tensor(Constrait_loss_per_batch).float(),
                                      requires_grad=True), preds.shape[0])



def dice_loss(input,target):
    """
    input is a torch variable of size BatchxnclassesxHxW representing log probabilities for each class
    target is a 1-hot representation of the groundtruth, shoud have same size as the input
    returns non normalized dice for batch
    """
    assert input.size() == target.size(), "Input sizes must be equal."
    assert input.dim() == 4, "Input must be a 4D Tensor."
    # uniques=np.unique(target.numpy())
    # assert set(list(uniques))<=set([0,1]), "target must only contain zeros and ones"

    probs = input
    num=probs*target#b,c,h,w--p*g
    num=torch.sum(num,dim=3)#b,c,h
    num=torch.sum(num,dim=2)+10e-7

    den1=probs*probs#--p^2
    den1=torch.sum(den1,dim=3)#b,c,h
    den1=torch.sum(den1,dim=2)

    den2=target*target#--g^2
    den2=torch.sum(den2,dim=3)#b,c,h
    den2=torch.sum(den2,dim=2)#b,c
    
    dice=(num/(den1+den2+10e-7))
    
    return -dice

def dice_star(input,target, lambdaa = 1):
    """
    input is a torch variable of size BatchxnclassesxHxW representing log probabilities for each class
    target is a 1-hot representation of the groundtruth, shoud have same size as the input
    returns non normalized dice for batch
    """
    assert input.size() == target.size(), "Input sizes must be equal."
    assert input.dim() == 4, "Input must be a 4D Tensor."
    # uniques=np.unique(target.numpy())
    # assert set(list(uniques))<=set([0,1]), "target must only contain zeros and ones"

    probs = input
    num=probs*target#b,c,h,w--p*g
    num=torch.sum(num,dim=3)#b,c,h
    num=torch.sum(num,dim=2)+10e-7

    den1=probs*probs#--p^2
    den1=torch.sum(den1,dim=3)#b,c,h
    den1=torch.sum(den1,dim=2)

    den2=target*target#--g^2
    den2=torch.sum(den2,dim=3)#b,c,h
    den2=torch.sum(den2,dim=2)#b,c
    
    dice = num - lambdaa*(den1+den2)
    
    return -dice

def dice_loss_constraint_dynamic(input,target, size):
    """

    """
    d_loss = dice_loss(input, target)
    c_loss = Constraint_loss_single_organ_MB(0.7, input, np.multiply(0.9,size), np.multiply(1.1, size))
    loss_perbatch = torch.sum(d_loss+c_loss)
    return loss_perbatch/input.shape[0]

def dice_loss_constraint_static(input,target, a,b):
    d_loss = dice_loss(input, target)
    c_loss = Constraint_loss_single_organ_SB(0.7, input, a,b)
    loss_perbatch = torch.sum(d_loss+c_loss)
    return loss_perbatch/input.shape[0]


def Constrained_Boundary_Loss(preds, dist_L, dist_U, eps_1 = 0.01, eps_2 = 0.01):
    surface_loss = SurfaceLoss(idc=[1])
    s1 = surface_loss(preds, dist_L)/100.00
    s2 = surface_loss(preds, dist_U)/100.00

    if s1 > eps_1:
        loss = s2
    elif s2 > eps_2:
        loss = s1
    elif s1 > eps_1 and s2 > eps_2:
        loss = (s1 + s2)/2
    else:
        loss = 0
    
    return loss
