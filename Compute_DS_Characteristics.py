'''
Created on Mar 20, 2019

@author: eljurros
'''
import os
import sys
from operator import add
from DataSEt_Classes import SegThorDS, SegmentationPair2D
import imageio
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from Multi_Organ_Seg.metrics import compute_stats
sys.path.append('../medicaltorch-0.2')

gray_scale = True


def get_ds_characteristics(typ):
    '''
    iterates over each patient, generates a
    ssegmentation_pair instance per patient
    '''
    root_path = '/home/eljurros/spare-workplace/Multi_Organ_Segmentation/DataSet/CT_ROOT'
    training_dataset_path = os.path.join(root_path, typ)
    Segthor_ds = SegThorDS(root_dir=training_dataset_path)
    ds_size = 0
    organ_count = [0,0,0,0,0]
    organ_size = [0,0,0,0]
    pat_mean = [0, 0, 0, 0]
    pat_min = [10000,10000,10000,10000]
    pat_max = [0,0,0,0]
    k = 0

    for _, patient_path in enumerate(Segthor_ds.filename_pairs):
        patient_name = os.path.basename(patient_path[0])
        input_filename, gt_filename, contour_filename = patient_path[0], \
                                                        patient_path[1], \
                                                        patient_path[2]
        Slicer = SegmentationPair2D(input_filename,
                                    gt_filename,
                                    contour_filename)

        input_data, gt_array, contour, kept_slices, count, mean, min, max = Slicer.get_pair_data() 
        k += 1

        organ_count = list(map(add, organ_count, count))
        
        assert(np.array(input_data).shape[0] == len(kept_slices))
        ds_size += np.array(input_data).shape[0]

        print ('patient name : ' + str(patient_name) +
               '\n length of input data ' + str(len(input_data)) +
               ' \n current augmented size of dataset' + str(ds_size) + 
               '\n patients organ size dist : ' + str(mean) +
               '\n patients min size dist : ' + str(min) +
               '\n patients max size dist : ' + str(max) +
               '\n organ count' + str(count))

        for i in range(4):
            if min[i] < pat_min[i]:
                pat_min[i] = min[i]
            if max[i] > pat_max[i]:
                pat_max[i] = max[i]
            pat_mean[i] += mean[i]

    return ds_size, k, pat_min, pat_max,pat_mean, organ_count


ds_size, k, pat_min,pat_max,mean, organ_count = get_ds_characteristics('train_ancillary')
print ('dataset size = ' + str(ds_size))
print('minimum organ size = ' + str(pat_min))
print('max organ size = ' + str(pat_max))
print('average organ size = ' + str(np.divide(mean, k)))
print('organ slices ' + str(organ_count))




      
