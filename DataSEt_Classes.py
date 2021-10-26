'''
Created on Jun 17, 2019

@author: eljurros
'''
'''
Created on Mar 20, 2019

@author: eljurros
@inspired by medical torch
'''
import sys
sys.path.append('/home/eljurros/spare-workplace/Multi_Organ_Segmentation')
sys.path.append('/home/eljurros/spare-workplace/Multi_Organ_Segmentation/DataSet')
sys.path.append('/home/eljurros/spare-workplace/Multi_Organ_Segmentation/DataSet_Functions')

sys.path.append('/home/eljurros/spare-workplace/Multi_Organ_Segmentation/Multi_Organ_Seg')
sys.path.append('/home/eljurros/spare-workplace/Multi_Organ_Segmentation/Common_Scripts')
import torch.nn.functional as F
import torch
import os
import nibabel as nib
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from PIL import Image
import random
import h5py
import numpy as np
from Dataset_Helpers import simplex, Get_DistMap, Preprocess_Prostate, Preprocess_Hippocampur, Preprocess_SegTHOR
from metrics import compute_stats
from Dataset_Helpers import class2one_hot
from Label_Estimate_Helper_Functions import draw_filter, create_seed, Get_Upper_Lower_boundaries, Variate_Bbox
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

class Prostate(Dataset):
    '''
    get sthe prostate dataset
    '''
    def __init__(self, root_dir, typ=None):
        '''
        just got the input/GT pairs present within a dataset per patient input_patient/GT_paitent
        '''
        self.root_dir = root_dir
        self.filename_pairs = []
        #assert typ != None
        for root_path, _, files in os.walk(os.path.join(self.root_dir, typ, 'imagesTr'), topdown=False):
            if len(files) > 1:
                for file in files:
                    patient_path = os.path.join(root_path, file)
                    input_filename = self._build_train_input_filename(root_dir, patient_path, 'img')
                    gt_filename = self._build_train_input_filename(root_dir, patient_path, 'mask')
                    self.filename_pairs.append((input_filename, gt_filename))
                    print(input_filename, gt_filename)

    @staticmethod
    def _build_train_input_filename(root_path, patient_path, im_type='img'):
        '''
        gets the img, gt names and locations
        '''
        basename = os.path.basename(patient_path)
        
        base_img_path = os.path.join(root_path,patient_path.split('/')[-3],'imagesTr')
        base_gt_path = os.path.join( root_path, patient_path.split('/')[-3],'labelsTr')
        if im_type == 'img':
            return os.path.join(base_img_path, basename)
        elif im_type == 'mask':
            return os.path.join(base_gt_path, basename)


class SegmentationPair2D_Prostate(object):
    """
    This class is used to build 2D segmentation datasets. It represents
    a pair of two data volumes (the input data and the ground truth data).

    :param input_filename: the input filename (supported by nibabel).
    :param gt_filename: the ground-truth filename.
    """

    def __init__(self, input_filename, gt_filename, contour_filename = None):
        self.input_filename = input_filename
        self.gt_filename = gt_filename
        self.contour_filename = contour_filename
        self.input_handle = nib.load(self.input_filename)
        self.gt_handle = nib.load(self.gt_filename)
        if contour_filename is not None:
            self.contour_handle = nib.load(self.contour_filename)
        self.cache = True
        
        if len(self.get_pair_shapes()[0]) == 3:
            self.Medical_Image_XD = 3
            print('3D Medical images of shape: ', str(self.get_pair_shapes()[0]))
        elif len(self.get_pair_shapes()[0]) == 4:
            self.Medical_Image_XD = 4
            print('4D Medical Images')

        #if len(self.input_handle.shape) > 3:
            #raise RuntimeError("4-dimensional volumes not supported.")
    

    def get_slices(self, input_array, gt_array):
        input_array_ch1 = []
        input_array_ch2 = []
        kept_slices = []
        multi_gt_array = []
        tp = self.organ_n+1
        count = [0]*tp
        size = [0]*self.organ_n
        min = [1000]*self.organ_n
        max = [0]*self.organ_n
        mean = [0]*self.organ_n
        if self.Medical_Image_XD == 4:
            for index, pair in enumerate(zip(input_array[0], input_array[1], gt_array)):
                p = np.round(pair[-1]).astype(int)
                if len(np.unique(p)) > 1:
                    input_array_ch1.append(pair[0])
                    input_array_ch2.append(pair[1])
                    kept_slices.append(index)
                    multi_gt_array.append(p)
                    for i in range(0, self.organ_n+1):
                        if i in np.unique(p):
                            # compute slices having class 1 in it
                            count[i] += 1
                            # compute organ size
                            if i != 0:
                                count[i -1] += 1
                                tmp = list(p.flatten()).count(i)
                                size[i -1] += tmp
                                if min[i -1] > tmp:
                                    min[i -1] = tmp
                                if max[i -1] < tmp:
                                    max[i -1] = tmp
                                
                # average_variance = np.array(organ_size)/np.array(count)
    
        elif self.Medical_Image_XD == 3:
            for index, pair in enumerate(zip(input_array,gt_array)):
                p = np.round(pair[-1]).astype(int)
                if len(np.unique(p)) > 1:
                    input_array_ch1.append(pair[0])
                    kept_slices.append(index)
                    multi_gt_array.append(p)
                    for i in range(0, self.organ_n+1):
                        if i in np.unique(p):
                            # compute slices having class 1 in it
                            count[i] += 1
                            # compute organ size
                            if i != 0:
                                tmp = list(p.flatten()).count(i)
                                size[i -1] += tmp
                                if min[i -1] > tmp:
                                    min[i -1] = tmp
                                if max[i -1] < tmp:
                                    max[i -1] = tmp
                                mean[i-1] = mean[i-1] + tmp 
    
                # average_variance = np.array(organ_size)/np.array(count)
    
        return input_array_ch1,input_array_ch2, multi_gt_array, kept_slices, count, min, max, np.array(mean)/len(kept_slices)
    
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

    def Preprocess_Data_Per_Patient(self, input_array_ch1,input_array_ch2, multi_gt_array, multi_contour = None):
        '''
        takes as input the arrays of data relative to a patient, 
        iterates over them to get corresponding image/gt/contour triplets 
        invokes processing per image 
        returns an array of arrays of the data per patient
        '''

        input_ch1 = []
        input_ch2  = []
        gt = []
        if multi_contour is None:
            if input_array_ch2 is not None:
                input_array_ch2  = np.array(input_array_ch2)
                for _, pair in enumerate(zip(input_array_ch1,input_array_ch2, multi_gt_array)):
                    pre_input1,pre_input2, pre_gt = Preprocess_Prostate(pair[0],pair[1],pair[2])
        
                    input_ch1.append(pre_input1)
                    input_ch2.append(pre_input2)
                    gt.append(pre_gt)
                
            else:
                for _, pair in enumerate(zip(input_array_ch1, multi_gt_array)):
                    pre_input1,_,pre_gt = Preprocess_Prostate(pair[0],None,pair[-1])
    
                    input_ch1.append(pre_input1)
                    gt.append(pre_gt)
        if multi_contour is not None:
            for _, pair in enumerate(zip(input_array_ch1, multi_gt_array, multi_contour)):
                pre_input1,pre_gt = Preprocess_SegTHOR(pair[0],pair[1], pair[-1])
    
                input_ch1.append(pre_input1)
                gt.append(pre_gt)
        return np.array(input_ch1), np.array(input_ch2), np.array(gt)

        
 
    
    def get_pair_data(self):
        """Return the tuple (input, ground truth) with the data content in
        numpy array.
        applies filtering for empty slices and crops from contours
        """

        cache_mode = 'fill' if self.cache else 'unchanged'
        input_data = self.input_handle.get_fdata(cache_mode, dtype=np.float32)
        if self.contour_filename is not None:
            contour_data = self.contour_handle.get_fdata(cache_mode, 
                                                         dtype=np.float32)

        gt_data = self.gt_handle.get_fdata(cache_mode, dtype=np.float32)

        input = np.transpose(input_data)
        gt_data = np.transpose(gt_data)
        #self.organ_n = len(np.unique(gt_data))-1
        self.organ_n = len(np.unique(gt_data))-1
        #print(self.organ_n)
        #print('dataset has', str(self.organ_n),' = ',str(np.unique(gt_data)), 'organs')
        if self.contour_filename is not None:
            cntr = np.transpose(contour_data)
        else:
            cntr = None
        # here is where I need to specify the method of extractions
        if self.Medical_Image_XD == 4:
            input_array_ch1,input_array_ch2, multi_gt_array, kept_slices, count, min, max = self.get_slices(input,gt_data)
        
            input_ch1, input_ch2, gt= self.Preprocess_Data_Per_Patient(input_array_ch1,input_array_ch2, multi_gt_array)
            return input_ch1,input_ch2, gt, kept_slices, min, max
        elif self.Medical_Image_XD == 3:
            input_array_ch1,_, multi_gt_array, kept_slices, count, min, max, mean = self.get_slices(input,gt_data)
            input_ch1,_, gt= self.Preprocess_Data_Per_Patient(input_array_ch1,None, multi_gt_array, cntr)

            return input_ch1, gt, kept_slices, min, max, mean
        

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

def Normalize_Image(img):
    '''
    Gets an input image as well as tecountour and preprocesses it 
    returns preprocessed image
    '''
    # normalize:
    maxx = np.max(img)
    minn = np.min(img)
    assert min != 0
    norm_inputt = np.divide((img - minn), maxx - minn)

    return norm_inputt


class SegTHor_2D_TrainDS(Dataset):
    def __init__(self, root_dir, transform=None, ds_mode='train', size=400, seed_div = 4, organ_n = 2):
        '''
        SEgthor dataset class 
        @parameter : 
            @root_dir: the root directory for the dataset default: ../Segthor
            @transform transformations to be implemented on the input:output or ds params, 
            @out_type (Definition depraceated from prev iterative learning) on which labels we want to train our model on :
                - mask:
                - grb
                - ancillary -cc
                c_labels
                new defenition: 
                Seeds or fully supervised masks  
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
            gt_mode = 'conv'
            ds_mode = 'train_ancillary'
            dice_total = []
            root_dir = '../Segthor'
        '''
        h5_file = '{}.h5'
        self.root_dir = root_dir
        self.handlers = []
        self.indexes = []
        self.size = size
        self.transform = transform
        self.seed_div = seed_div
        self.organ_n = organ_n

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
        if ds_mode == 'val':
            self.dir = self.val_dir
        if ds_mode == 'test':
            self.dir = self.test_dir

    def __len__(self):
        self.f = h5py.File(self.dir, "r")
        return self.f['img'].shape[0]

    def Get_fully_supervised_label(self, index, organ_n):
        self.label_dir = self.f['label']
        gt = np.array(self.label_dir[index])
        
        gt = torch.tensor(gt)
        gt = class2one_hot(gt, np.int(organ_n))[0]
        return gt

    def __getitem__(self, index):
        # input directory
        with h5py.File(self.dir, "r") as self.f:
            self.img_dir = self.f['img']
            # label directory
            self.bboxes_coord = self.f['bboxes'][index]
            gt_onehot = self.Get_fully_supervised_label(index, self.organ_n)
            gt = gt_onehot[1:self.organ_n+1]
            img = np.array(self.f['img'][index])
            name = str(self.f['img_id'][index])
            ground_truth = torch.tensor(gt_onehot)
            bbox = draw_filter(np.array(self.bboxes_coord), 'bbox',
                                    self.organ_n+1, img.shape[1])
            bbox_v = Variate_Bbox(img.shape, np.array(self.bboxes_coord),1000, self.organ_n+1)
            bbox_img = np.concatenate((img.reshape((1,320,320)), bbox), axis=0)
            #print("diff in size is {}", np.float(list(bbox_v.flatten()).count(255))/np.float(list(bbox.flatten()).count(255)))
            bbox_conv = bbox*img
            circ_mask = draw_filter(np.array(self.f['circular'][index]), 'circular', self.organ_n+1, img.shape[1])
            circ_conv = circ_mask*img

            '''
            for i in range(0,4):
                plt.imsave('gt{}.png'.format(i),ground_truth[i])
                plt.imsave('seed{}.png'.format(i),seed[i])
                plt.imsave('bbox{}.png'.format(i),bbox_mask[i])
                plt.imsave('circle{}.png'.format(i), circ_mask[i])
                                         'seed': seed,
                         'inner': inner, 
                         'outer': outer,
                         'Bound_U': np.array(Bound_U), 
                          'Bound_L': np.array(Bound_L)
            
                
            '''

            if self.transform:
                input_img = img

            data_dict = {'input': img,
                         'bbox_img':bbox_img,
                         'gt': gt_onehot, 
                         'name': name,
                         'bbox': bbox,
                         'seed': bbox_v,
                         'bbconv': bbox_conv,
                         'cc': circ_mask, 
                         'cconv': circ_conv,
                         'index': index, 
                         
	
                         }

            self.f.close()
            return data_dict



