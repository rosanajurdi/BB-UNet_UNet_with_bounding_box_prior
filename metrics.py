from collections import defaultdict

from scipy import spatial
import numpy as np
import torch.nn.functional as F
import torch
from torch import einsum
def dice_score_2(prediction, groundtruth):
    prediction = torch.tensor(prediction)
    groundtruth = torch.tensor(groundtruth)

    prediction = prediction.clone().detach()
    groundtruth = groundtruth.clone().detach()

    inter = prediction*groundtruth
    union = prediction*prediction + groundtruth*groundtruth
    iflat = inter.flatten().sum()
    uflat = union.flatten().sum()
    if (uflat != 0):
        d = 200*(np.float(iflat)/uflat)
    if (uflat == 0):
        d = 100
    return d
def threshold_predictions(predictions, thr=0.999):
    thresholded_preds = predictions[:]
    low_values_indices = thresholded_preds < thr
    thresholded_preds[low_values_indices] = 0
    low_values_indices = thresholded_preds >= thr
    thresholded_preds[low_values_indices] = 1
    return thresholded_preds

def intersection(a, b):
    assert a.shape == b.shape
    assert sset(a, [0, 1])
    assert sset(b, [0, 1])
    return a & b


def union(a, b):
    assert a.shape == b.shape
    assert sset(a, [0, 1])
    assert sset(b, [0, 1])
    return a | b

def probs2class(probs):
    b, _, w, h = probs.shape  # type: Tuple[int, int, int, int]
    assert simplex(probs)

    res = probs.argmax(dim=1)
    assert res.shape == (b, w, h)

    return res



def class2one_hot(seg, C):
    if len(seg.shape) == 2:  # Only w, h, used by the dataloader
        seg = seg.unsqueeze(dim=0)
    assert sset(seg, list(range(C)))

    b, w, h = seg.shape  # type: Tuple[int, int, int]

    res = torch.stack([seg == c for c in range(C)], dim=1).type(torch.int32)
    assert res.shape == (b, C, w, h)
    assert one_hot(res)

    return res

def probs2one_hot(probs):
    _, C, _, _ = probs.shape
    assert simplex(probs)

    res = class2one_hot(probs2class(probs), C)
    assert res.shape == probs.shape
    assert one_hot(res)

    return res


def sset(a, sub):
    #print(uniq(a))
    #print(sub)
    return uniq(a).issubset(sub)


def eq(a, b):
    return torch.eq(a, b).all()


def simplex(t, axis=1):
    _sum = t.sum(axis).type(torch.float32)
    #print(_sum.sum())
    _ones = torch.ones_like(_sum, dtype=torch.float32)
    #print(_ones.sum())
    return _ones.sum() == _sum.sum()

def uniq(a):
    return set(torch.unique(a.cpu()).numpy())

def one_hot(t, axis=1):
    #print(simplex(t, axis))
    #print(sset(t, [0, 1]))
    return simplex(t, axis) and sset(t, [0, 1])

def meta_dice(sum_str, label, pred, smooth = 1e-8):
    assert label.shape == pred.shape
    assert one_hot(pred.detach())

    inter_size: Tensor = einsum(sum_str, [intersection(label, pred)]).type(torch.float32)
    sum_sizes: Tensor = (einsum(sum_str, [label]) + einsum(sum_str, [pred])).type(torch.float32)

    dices: Tensor = (2 * inter_size + smooth) / (sum_sizes + smooth)

    return dices



class MetricManager(object):
    def __init__(self, metric_fns):
        self.metric_fns = metric_fns
        self.result_dict = defaultdict(float)
        self.num_samples = 0 
    
    def __call__(self, prediction, ground_truth):
        self.num_samples += len(prediction)
        for metric_fn in self.metric_fns:
            for p, gt in zip(prediction, ground_truth):
                res = metric_fn(p, gt)
                dict_key = metric_fn.__name__
                self.result_dict[dict_key] += res
            
    def get_results(self):
        res_dict = {}
        for key, val in self.result_dict.items():
            res_dict[key] = val / self.num_samples
        return res_dict
    
    def reset(self):
        self.num_samples = 0
        self.result_dict = defaultdict(float)
        

def numeric_score(prediction, groundtruth):
    """Computation of statistical numerical scores:

    * FP = False Positives
    * FN = False Negatives
    * TP = True Positives
    * TN = True Negatives

    return: tuple (FP, FN, TP, TN)
    """
    FP = np.float(np.sum((prediction == 1) & (groundtruth == 0)))
    FN = np.float(np.sum((prediction == 0) & (groundtruth == 1)))
    TP = np.float(np.sum((prediction == 1) & (groundtruth == 1)))
    TN = np.float(np.sum((prediction == 0) & (groundtruth == 0)))
    return FP, FN, TP, TN

def dice_per_organ(prediction, groundtruth, thresh):
    pred = prediction.detach().numpy()
    tr = groundtruth.detach().numpy()
    num = np.sum(pred*tr)
    s = np.sum(pred)
    '''
    if s < thresh:
        s = 0
        num = 0
    '''
    denum = s + np.sum(tr)
    if denum == 0 and num == 0:
        d = 1
    else:
        d = 2*(num/denum)
    return d

def jaccard_score(prediction, groundtruth):
    pflat = prediction.flatten()
    gflat = groundtruth.flatten()
    return (1 - spatial.distance.jaccard(pflat, gflat)) * 100.0


def hausdorff_score(prediction, groundtruth):
    return spatial.distance.directed_hausdorff(prediction, groundtruth)[0]


def precision_score(prediction, groundtruth):
    # PPV
    FP, FN, TP, TN = numeric_score(prediction, groundtruth)
    if (TP + FP) <= 0.0:
        return 0.0

    precision = np.divide(TP, TP + FP)
    return precision * 100.0

def dice_metric(input,target):
    """
    input is a torch variable of size BatchxnclassesxHxW representing log probabilities for each class
    target is a 1-hot representation of the groundtruth, shoud have same size as the input
    """
    assert input.size() == target.size(), "Input sizes must be equal."
    uniques=np.unique(target.numpy())
    assert set(list(uniques))<=set([0,1]), "target must only contain zeros and ones"

    probs = input
    num=probs*target#b,c,h,w--p*g
    num=torch.sum(num,dim=3)#b,c,h
    num=torch.sum(num,dim=2)

    den1=probs*probs#--p^2
    den1=torch.sum(den1,dim=3)#b,c,h
    den1=torch.sum(den1,dim=2)

    den2=target*target#--g^2
    den2=torch.sum(den2,dim=3)#b,c,h
    den2=torch.sum(den2,dim=2)#b,c
    
    dice=np.squeeze(2*(num/(den1+den2)))
    
    if input.shape[1] != 1:
        for i, d in enumerate(dice):
            if np.isnan(d) == 1:
                dice[i] = 1
    else:
        if np.isnan(dice) == 1:
                dice = 1
                dice = torch.tensor(dice)

    # return dice.detach().numpy()[1:]
    return dice.detach().numpy()

def recall_score(prediction, groundtruth):
    # TPR, sensitivity
    FP, FN, TP, TN = numeric_score(prediction, groundtruth)
    if (TP + FN) <= 0.0:
        return 0.0
    TPR = np.divide(TP, TP + FN)
    return TPR * 100.0


def specificity_score(prediction, groundtruth):
    FP, FN, TP, TN = numeric_score(prediction, groundtruth)
    if (TN + FP) <= 0.0:
        return 0.0
    TNR = np.divide(TN, TN + FP)
    return TNR * 100.0


def intersection_over_union(prediction, groundtruth):
    FP, FN, TP, TN = numeric_score(prediction, groundtruth)
    if (TP + FP + FN) <= 0.0:
        return 0.0
    return TP / (TP + FP + FN) * 100.0


def accuracy_score(prediction, groundtruth):
    pred_thresh = threshold_predictions(prediction)
    FP, FN, TP, TN = numeric_score(pred_thresh, groundtruth)
    N = FP + FN + TP + TN
    accuracy = np.divide(TP + TN, N)
    return accuracy * 100.0

def compute_stats(metric, mean_prev, var_prev, n):
    '''
    computes the moving average of a metric 
    @parameters:
        @metric = newest vaue to add
        @mean_prev = previous mean 
        @var_prev = previous variance
        @n = current total number of samples
    '''
    if n > 1:
        mean_n = (1.00/n)*(metric + (n-1)*mean_prev)
        var_n = (np.float(n-2)/(n-1))*(var_prev) + (1.00/n)*(metric - mean_prev)**2
    else:
        mean_n = metric
        var_n = 0
    
    return mean_n, var_n
    
    
    
    
    
    
