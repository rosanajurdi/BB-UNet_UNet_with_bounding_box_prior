'''
Created on Mar 20, 2019

@author: eljurros
'''
from DataSEt_Classes import SegThorDS, SegmentationPair2D
from Label_Estimate_Helper_Functions import extract_bbox, rect_mask, get__mask
from torchvision import transforms
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
import imageio
import h5py
import sys
import os

sys.path.append('../medicaltorch-0.2')
sys.path.append('/home/eljurros/spare-workplace/Multi_Organ_Segmentation/DataSet_Functions')

gray_scale = True
typ = 'val'

i = 0
total_slices = 0
filtered_slices = 0
root_path = '/home/eljurros/spare-workplace/Multi_Organ_Segmentation/DataSet/CT_ROOT'
training_dataset_path = os.path.join(root_path, typ)

Segthor_ds = SegThorDS(root_dir=training_dataset_path)
total = len(Segthor_ds.filename_pairs)
size = 497
file = h5py.File('/home/eljurros/spare-workplace/Multi_Organ_Segmentation/DataSet/SegTHOR_H5/{}.h5'.format(typ), "w")
img_dataset = file.create_dataset('img', (size, 512, 512))
label_dataset = file.create_dataset('label', (size, 512, 512))
slice_name = file.create_dataset('img_id', (size,1), dtype="S10")
bbox_im_dir = file.create_dataset('bboxes', (size, 4, 4))
circular = file.create_dataset('circular', (size, 4, 3))
j = 0

def get_convolution(img, mask):
    return img*mask


for i, patient_path in enumerate(Segthor_ds.filename_pairs):
    patient_name = os.path.basename(patient_path[0])
    input_filename, gt_filename, contour_filename = patient_path[0], \
                                                    patient_path[1],\
                                                    patient_path[2]

    Slicer = SegmentationPair2D(input_filename,
                                gt_filename,
                                contour_filename)

    input_data, gt_array, contour, kept_slices, count,_,_,_ = Slicer.get_pair_data()
    print (patient_name)
    for _, triple in enumerate(zip(input_data, gt_array,
                                   contour, kept_slices)):
        img_slice, gt, cnt = triple[0], triple[1], triple[2]
        # print('saving onto index :{}'.format(i+j))
        img = img_slice
        gt_data = gt
        rectmask = np.array(get__mask(triple[1], 'bboxes'))
        circmask = np.array(get__mask(triple[1], 'circular'))
        file['img'][j] = img
        file['label'][j] = gt_data
        file['bboxes'][j] = rectmask
        file['circular'][j] = circmask
        # file['circular_conv'][j] = get_convolution(circmask, img)
        # file['bbox_conv'][j] = get_convolution(rectmask, img)
        patient_name = os.path.basename(patient_path[0]).split('.nii')[0]
        sn = '{}_{}'.format(patient_name, triple[3])
        file['img_id'][j] = sn

        j += 1
        if j == size:
            file.close()
            break
