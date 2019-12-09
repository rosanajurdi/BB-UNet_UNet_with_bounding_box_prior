'''
Created on Jun 17, 2019

@author: eljurros
'''
'''
Created on Mar 20, 2019

@author: eljurros
@inspired by medical torch
'''
import torch.nn.functional as F
from PIL import Image
import torch
import cv2 as cv
import os
import sys
import glob
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
sys.path.append('/home/eljurros/spare-workplace/Multi_Organ_Segmentation/DataSet_Functions')
from DataSet_Functions.Dataset_Helpers import Preprocess_Image
from PIL import Image
import random
import h5py
import numpy as np
from Multi_Organ_Seg.loss_func import simplex
from Multi_Organ_Seg.metrics import compute_stats
from Dataset_Helpers import class2one_hot
from Label_Estimate_Helper_Functions import draw_filter
def uniq(a):
    return set(torch.unique(a.cpu()).numpy())
def sset(a, sub):
    return uniq(a).issubset(sub)

class SampleMetadata(object):
    def __init__(self, d=None):
        self.metadata = {} or d

    def __setitem__(self, key, value):
        self.metadata[key] = value

    def __getitem__(self, key):
        return self.metadata[key]

    def __contains__(self, key):
        return key in self.metadata

    def keys(self):
        return self.metadata.keys()


class SegThorDS(Dataset):
    """Segthor dataset.

    :param root_dir: the directory containing the training dataset.
    just a 
    """

    def __init__(self, root_dir):
        '''
        just got the input/GT pairs present within a dataset per patient input_patient/GT_paitent
        '''
        self.root_dir = root_dir
        self.filename_pairs = []
        for patient_path, _, files in os.walk(self.root_dir, topdown=False):
            if len(files) > 1:
                input_filename = self._build_train_input_filename(patient_path, 'img')
                gt_filename = self._build_train_input_filename(patient_path, 'mask')
                contour_filename = self._build_train_input_filename(patient_path, 'contour')
                input_filename = os.path.join(patient_path, input_filename)
                gt_filename = os.path.join(patient_path, gt_filename)
                contour_filename = os.path.join(patient_path, contour_filename)
                self.filename_pairs.append((input_filename, gt_filename,
                                            contour_filename))

    @staticmethod
    def _build_train_input_filename(patient_path, im_type='img'):
        '''
        gets the img, gt names
        '''
        basename = os.path.basename(patient_path)
        if im_type == 'img':
            return "{}.nii.gz".format(basename)
        elif im_type == 'mask':
            return "GT.nii.gz"
        elif im_type == 'contour':
            return "CONTOUR.nii.gz"


class SegmentationPair2D(object):
    """
    This class is used to build 2D segmentation datasets. It represents
    a pair of two data volumes (the input data and the ground truth data).

    :param input_filename: the input filename (supported by nibabel).
    :param gt_filename: the ground-truth filename.
    """

    def __init__(self, input_filename, gt_filename, contour_filename):
        self.input_filename = input_filename
        self.gt_filename = gt_filename
        self.contour_filename = contour_filename
        self.input_handle = nib.load(self.input_filename)
        self.gt_handle = nib.load(self.gt_filename)
        self.contour_handle = nib.load(self.contour_filename)
        self.cache = True

        if len(self.input_handle.shape) > 3:
            raise RuntimeError("4-dimensional volumes not supported.")

    def filter_empty_slices(self, input_array, gt_array, contour_array):
        organ_id = 2
        new_input_array = []
        new_gt_array = []
        new_contour_array = []
        for index, pair in enumerate(zip(input_array, gt_array, contour_array)):
            if organ_id in np.unique(pair[1]):
                gt = np.isin(pair[1], organ_id).astype(int)
                new_input_array.append(pair[0])
                new_gt_array.append(gt)
                new_contour_array.append(pair[2])

        return np.array(new_input_array), np.array(new_gt_array), np.array(new_contour_array)
    
    def get_one_to_all_slices(self, input_array, gt_array, organ_id):
        '''
        function that gets heart organ slices, negative samples where the heart does not exist 
        and returns whether the heart is existent or not in a separate array.
        '''
        new_input_array = []
        new_gt_array = []
        heart_exists = []
        for index, pair in enumerate(zip(input_array, gt_array)):
            gt = np.isin(pair[1], organ_id).astype(int)
            new_input_array.append(pair[0])
            new_gt_array.append(gt)
            if organ_id in np.unique(pair[1]):
                heart_exists.append(0)
            else:
                heart_exists.append(-1)

    def all_four_organs(self, input_array, gt_array):
        new_input_array = []
        new_gt_array = []
        for index, pair in enumerate(zip(input_array, gt_array)):
            if len(np.unique(pair[1])) == 5:
                new_input_array.append(pair[0])
                new_gt_array.append(pair[1])

        return np.array(new_input_array), np.array(new_gt_array)
    
    def get_slices(self, input_array, gt_array, contour):
        new_input_array = []
        contour_array = []
        kept_slices = []
        multi_gt_array = []
        count = [0, 0, 0, 0, 0]
        size = [0, 0, 0, 0]
        min = [1000,1000,1000,1000]
        max = [0,0,0,0]
        for index, pair in enumerate(zip(input_array, gt_array, contour)):
            p = np.round(pair[1]).astype(int)
            if len(np.unique(p)) > 1 :
                new_input_array.append(pair[0])
                kept_slices.append(index)
                multi_gt_array.append(p)
                contour_array.append(pair[2])

                if 0 in np.unique(p):
                    # compute slices having class 1 in it
                    count[0] += 1
                    # compute organ size
                if 1 in np.unique(p):
                    count[1] += 1
                    tmp = list(p.flatten()).count(1)
                    size[0] += tmp
                    if min[0] > tmp:
                        min[0] = tmp
                    if max[0] < tmp:
                        max[0] = tmp

                if 2 in np.unique(p):
                    count[2] += 1
                    tmp = list(p.flatten()).count(2)
                    size[1] += tmp
                    if min[1] > tmp:
                        min[1] = tmp
                    if max[1] < tmp:
                        max[1] = tmp
                if 3 in np.unique(p):
                    count[3] += 1
                    tmp = list(p.flatten()).count(3)
                    size[2] += tmp
                    if min[2] > tmp:
                        min[2] = tmp
                    if max[2] < tmp:
                        max[2] = tmp

                if 4 in np.unique(p):
                    count[4] += 1
                    tmp = list(p.flatten()).count(4)
                    size[3] += tmp
                    if min[3] > tmp:
                        min[3] = tmp
                    if max[3] < tmp:
                        max[3] = tmp

        mean = np.divide(size, count[1:])
        # average_variance = np.array(organ_size)/np.array(count)

        return new_input_array, multi_gt_array, contour_array, kept_slices, count, mean, min, max


    def get_pair_shapes(self):
        """Return the tuple (input, ground truth) representing both the input
        and ground truth shapes."""
        input_shape = self.input_handle.header.get_data_shape()

        # Handle unlabeled data
        if self.gt_handle is None:
            gt_shape = None
        else:
            gt_shape = self.gt_handle.header.get_data_shape()

        return input_shape, gt_shape

    def Preprocess_Data_Per_Patient(self, input_array, contour_array, gt_array):
        '''
        takes as input the arrays of data relative to a patient, 
        iterates over them to get corresponding image/gt/contour triplets 
        invokes processing per image 
        returns an array of arrays of the data per patient
        '''

        new_input_array = []
        new_gt_array = []
        new_contour_data = []
        for _, pair in enumerate(zip(input_array, contour_array, gt_array)):
            cropped_input, cropped_gt, contours = Preprocess_Image(pair[0],
                                                                  pair[1],
                                                                  pair[2])

            new_input_array.append(cropped_input)
            new_gt_array.append(cropped_gt)
            new_contour_data.append(contours)

        return np.array(new_input_array), np.array(new_gt_array), np.array(new_contour_data)
 
    def get_pair_data(self):
        """Return the tuple (input, ground truth) with the data content in
        numpy array.
        applies filtering for empty slices and crops from contours
        """

        cache_mode = 'fill' if self.cache else 'unchanged'
        input_data = self.input_handle.get_fdata(cache_mode, dtype=np.float32)
        contour_data = self.contour_handle.get_fdata(cache_mode, 
                                                     dtype=np.float32)

        gt_data = self.gt_handle.get_fdata(cache_mode, dtype=np.float32)

        input = np.transpose(input_data)
        gt_data = np.transpose(gt_data)
        cntr = np.transpose(contour_data)

        n_input_arr, gt_arr, cntr, kept_slices, cnt, mean, min, max = self.get_slices(input,
                                                                      gt_data,
                                                                      cntr)

        input_d, gt_d, contour_d = self.Preprocess_Data_Per_Patient(n_input_arr,
                                                                    cntr,
                                                                    gt_arr)

        return input_d, gt_d, contour_d, kept_slices, cnt,mean, min, max

    def get_pair_slice(self, slice_index, slice_axis=2):
        """Return the specified slice from (input, ground truth).

        :param slice_index: the slice number.
        :param slice_axis: axis to make the slicing.
        """
        if self.cache:
            input_dataobj, gt_dataobj = self.get_pair_data()
        else:
            # use dataobj to avoid caching
            input_dataobj = self.input_handle.dataobj

            if self.gt_handle is None:
                gt_dataobj = None
            else:
                gt_dataobj = self.gt_handle.dataobj

        if slice_axis not in [0, 1, 2]:
            raise RuntimeError("Invalid axis, must be between 0 and 2.")

        if slice_axis == 2:
            input_slice = np.asarray(input_dataobj[..., slice_index],
                                     dtype=np.float32)
        elif slice_axis == 1:
            input_slice = np.asarray(input_dataobj[:, slice_index, ...],
                                     dtype=np.float32)
        elif slice_axis == 0:
            input_slice = np.asarray(input_dataobj[slice_index, ...],
                                     dtype=np.float32)

        # Handle the case for unlabeled data
        gt_meta_dict = None
        if self.gt_handle is None:
            gt_slice = None
        else:
            if slice_axis == 2:
                gt_slice = np.asarray(gt_dataobj[..., slice_index],
                                      dtype=np.float32)
            elif slice_axis == 1:
                gt_slice = np.asarray(gt_dataobj[:, slice_index, ...],
                                      dtype=np.float32)
            elif slice_axis == 0:
                gt_slice = np.asarray(gt_dataobj[slice_index, ...],
                                      dtype=np.float32)

            gt_meta_dict = SampleMetadata({
                "zooms": self.gt_handle.header.get_zooms()[:2],
                "data_shape": self.gt_handle.header.get_data_shape()[:2],
            })

        input_meta_dict = SampleMetadata({
            "zooms": self.input_handle.header.get_zooms()[:2],
            "data_shape": self.input_handle.header.get_data_shape()[:2],
        })

        dreturn = {
            "input": input_slice,
            "gt": gt_slice,
            "input_metadata": input_meta_dict,
            "gt_metadata": gt_meta_dict,
        }

        return dreturn


class TwoD_pair_class(object):
    def __init__(self, input_img, gt_img):
        self.input_img = input_img
        self.gt_img = gt_img

        self.input_handle = self.input_img
        self.gt_handle = self.gt_img

    def get_pair_slice(self):
        dictt = {
            "input": self.input_handle,
            "gt": self.gt_handle,
        }
        return dictt
from scipy.ndimage import distance_transform_edt as distance
def one_hot_my(a): # one hot encoder for seg masks
    g1=a[:,:]==0
    g2=a[:,:]==1
    return torch.stack((g1,g2),0)

def one_hot(t, axis=1) :
    '''
    makes sure that the matrice adds up to ones nd that classes50 and 1 ] are a subset of it
    '''
    return simplex(t, axis) and sset(t, [0, 1])
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

class SegTHor_2D_TrainDS(Dataset):
    def __init__(self, root_dir, transform=None, out_type='mask', 
                 ds_mode='train', gt_mode='norm', RGB=False, size=400, model_type='Unet'):
        '''
        SEgthor dataset class 
        @parameter : 
            @root_dir: the root directory for the dataset default: ../Segthor
            @transform transformations to be implemented on the input:output or ds params, 
            @out_type on which labels we want to train our model on :
                - mask:
                - grb
                - ancillary -cc
                c_labels 
            @filter_type: the type of filter we would like to merge our predictions with as in ancillary training:
                -bb: bounding boxes
                -cc: circular shapes
            @ds_mode: what type of ds are we laoding 
                -val,
                train
                train_ancillary
                -test, 
            @gt_mode= 
            train ds_mode : the labels we are training on do we want them convolved with the output or no
            train_ancillary in ds_mode: the ancillary filters are they convolved with images or no 
            @RGB: Rgb ilage or no,
            @size=400)
            
        @configurations: 
            @ancillary_CC_Conv: 
            model_type = 'BB_Unet'
            filter_type = 'cc'
            out_type_selection = 'mask'
            gt_mode = 'conv'
            ds_mode = 'train_ancillary'
            dice_total = []
            root_dir = '../Segthor'
        '''
        h5_file = '{}.h5'
        self.root_dir = root_dir
        self.handlers = []
        self.indexes = []
        self.gt_mode = gt_mode
        self.RGB = RGB
        self.out_type = out_type
        self.size = size
        self.model_type = model_type
        self.transform = transform
        
        if ds_mode == 'train':
            self.train_dir = os.path.join(root_dir, h5_file.format('train'))
        elif ds_mode == 'val':
            self.val_dir = os.path.join(root_dir, h5_file.format('val'))
        elif ds_mode == 'test':
            self.test_dir = os.path.join(root_dir, h5_file.format('test'))
        elif ds_mode == 'train_ancillary':
            self.train_dir = os.path.join(root_dir, h5_file.format('train_ancillary'))
        elif ds_mode == 'train_Primary':
            self.train_dir = os.path.join(root_dir, h5_file.format('train_Primary'))


        if ds_mode in ['train', 'train_ancillary', 'train_Primary']:
            self.dir = self.train_dir
        if ds_mode== 'val':
            self.dir = self.val_dir
        if ds_mode== 'test':
            self.dir = self.test_dir

    def __len__(self):
        self.f = h5py.File(self.dir, "r")
        return self.f['img'].shape[0]

    def __getitem__(self, index):
        # input directory
        with h5py.File(self.dir, "r") as self.f:
            self.img_dir = self.f['img']
            # label directory
            if self.out_type == 'mask':
                self.label_dir = self.f['label']
            elif self.out_type == 'BB':
                self.label_dir = self.f['BB']
            elif self.out_type == 'BBCONV':
                self.label_dir = self.f['BBCONV']
            elif self.out_type == 'CC':
                self.label_dir = self.f['CC']
            elif self.out_type == 'CCONV':
                self.label_dir = self.f['CCONV']

            img = np.array(self.f['img'][index])
            gt = np.array(self.label_dir[index])
            gt = torch.tensor(gt)
            name = str(self.f['img_id'][index])
            gt_onehot = class2one_hot(gt, 5)[0]
            mask_distmap = one_hot2dist(gt_onehot.cpu().numpy())
            bbox_mask = draw_filter(np.array(self.f['bboxes'][index]), 'bbox')
            bbox_conv = bbox_mask*img
            circ_mask = draw_filter(np.array(self.f['circular'][index]), 'circular')
            circ_conv = circ_mask*img
            if self.transform:
                input_img = img

            if self.model_type == 'Unet':
                data_dict = {
                    'input': input_img,
                    'gt': gt_onehot[1:5],
                    'name': name,
                    'mask_distmap' : mask_distmap}

            elif self.model_type == 'BB_Unet':
                data_dict = {'input': input_img,
                             'gt': gt_onehot[1:5],
                             'name': name,
                             'bbox':bbox_mask,
                             'bbconv': bbox_conv,
                             'cc': circ_mask,
                             'cconv': circ_conv,
                             'mask_distmap' : mask_distmap}

            self.f.close()
            return data_dict

                
        
        




