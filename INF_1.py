import sys
sys.path.append('/home/eljurros/spare-workplace/Multi_Organ_Segmentation')
sys.path.append('/home/eljurros/spare-workplace/Multi_Organ_Segmentation/DataSet')
sys.path.append('/home/eljurros/spare-workplace/Multi_Organ_Segmentation/DataSet_Functions')

sys.path.append('/home/eljurros/spare-workplace/Multi_Organ_Segmentation/Multi_Organ_Seg')
sys.path.append('/home/eljurros/spare-workplace/Multi_Organ_Segmentation/Common_Scripts')
import os
import numpy as np
import matplotlib.pyplot as plt
import DataSEt_Classes
import metrics 
from torchvision import transforms
import torch.nn.functional as F
import torch
import warnings
import models
import csv
import loss_func
from torch.autograd import Variable
import gc
from skimage.morphology import closing
from skimage.measure import label, regionprops
from scipy.spatial.distance import directed_hausdorff

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
from torch.utils.data import DataLoader
from matplotlib import cm
gt_mode = 'norm'
size = 320
model_type = 'BB_Unet'
bb_state = 'bbox'
out_type_selection = 'mask'
fieldnames = ['SLICE_ID', 'dice', 
              'Post_processed', 'dice_connected' ,'bb_filtered',  
              'haus', 'gt_connected','pred_connected', 'c_error']


def threshold(array):
    array = (array > 0.51) * 1.0
    return array


ckpt_path = '/media/eljurros/Transcend/Decathlone/Task02_Heart/nifty/FOLD2/Saved_models/checkpnt_BB_bb_dsc_loss.ckpt'

fold = os.path.basename(ckpt_path.split('/Saved_models')[0])
ROOT = ckpt_path.split('/Saved_models')[0].split('/{}'.format(fold))[0]
typ = 'test'
#ckpt_path = os.path.join(ROOT, 'saved_models', 'Full_DDOT_PR.ckpt_metric')

if os.path.exists(os.path.join(ROOT,fold, 'CSV_Results')) is False: 
    os.mkdir(os.path.join(ROOT,fold, 'CSV_Results'))

path = os.path.join(ROOT,fold, 'CSV_Results', '{}_{}.csv'.format(os.path.basename(ckpt_path).split('ckpt')[0], typ))

#model = models.Unet(drop_rate=0.4, bn_momentum=0.1)
model = torch.load(ckpt_path).module
model = model.to('cpu')
transform = transforms.Compose([
        transforms.Resize((size, size)),
])

dataset = DataSEt_Classes.SegTHor_2D_TrainDS(ROOT, transform=transform,
                                                         ds_mode=typ, size=size, seed_div = 4, organ_n = 1)

with open(os.path.join(path), 'w') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        loader = DataLoader(dataset, batch_size=1, shuffle=False,
                                pin_memory=False,
                                num_workers=1)
        model.eval()

        for i, batch in enumerate(loader):
            print(i)
            #input_samples, gt_samples = batch["input"], batch["gt"][:,0].reshape(-1, 1, size, size)
            input_samples, gt_samples = batch["input"], batch["gt"].reshape(-1, 1, size, size)
            var_input = input_samples.reshape(-1, 1, size, size)
            var_input = torch.tensor(var_input, dtype=torch.float)
            var_input = Variable(var_input, requires_grad=True)
            if len(np.unique(gt_samples)) == 2:
                #b = batch["inner"][:, 1, :, ].reshape(-1, 1, size, size)
                #b = (b == 255) * 1.0
                #b = b.float()
                #print(batch['name'][0])
                
                if model_type == 'Unet':
                    preds = model(var_input)
                    #preds = F.sigmoid(preds)
                    preds = F.softmax(preds)
                else:
                    #var_bbox = batch[bb_state].reshape(-1, 1, size, size)
                    var_bbox = torch.ones((1, 1, size, size))
                    var_bbox = torch.tensor(var_bbox, dtype=torch.float)
                    var_bbox = Variable(var_bbox, requires_grad=True)
                    preds = model(var_input, var_bbox)
                    preds = F.softmax(preds, dim=1)
                
                #predicted_output = threshold(preds[:,1,:,])
                predicted_output = threshold(preds[0][1])
                #plt.(input_samples[0].detach().numpy(), alpha=0.25)
                #plt.imshow(predicted_output)
                predicted_output = predicted_output.type(torch.float32)
                
                bb_filtered = predicted_output.type(torch.float32)*torch.tensor(np.squeeze((batch['bbox'] == 255) * 1.0)).type(torch.float32)
                
                post_processed = torch.tensor(closing(np.array(predicted_output)))
                #plt.imshow(post_processed)
                # plt.imsave(os.path.join('/home/eljurros/Desktop/pics/gt', '{}.png'.format(str(i))), np.array(np.squeeze(gt_samples)))
                # plt.imsave(os.path.join('/home/eljurros/Desktop/pics/preds', '{}.png'.format(str(i))), np.array(np.squeeze(preds.detach().numpy()[0][1])))
                
                gt_samples = gt_samples.type(torch.float32)
                dice = metrics.dice_score_2(predicted_output.reshape(1,1,size, size), gt_samples)

                dice_closing = metrics.dice_score_2(post_processed.reshape(1,1,size, size), gt_samples)
                pred_label = label(np.array(predicted_output))
                bb_dice = metrics.dice_score_2(bb_filtered.reshape(1,1,size, size), gt_samples)
                pred_connected = len(np.unique(pred_label))
                
                m = max(directed_hausdorff(np.squeeze(gt_samples), np.squeeze(predicted_output))[0],
                        directed_hausdorff(np.squeeze(predicted_output), np.squeeze(gt_samples))[0])
                print('"""""""""""""""' , str(pred_connected), '""""""""""""""""""""')
                if pred_connected == 2:
                    pass
                try:
                    gt_label = label(np.array(np.squeeze(gt_samples)))
                    gt_connected = len(np.unique(gt_label))
                except:
                    pass
                area = []
                for i in range(0, pred_connected-1):
                    area.append(regionprops(label(np.array(predicted_output)))[i]['area'])
                index = []
                try:
                    for j in range(gt_connected-1):
                        index.append(np.argmax(area))
                        area.pop(index[-1])
                except:
                    pass
                import torch
                if len(np.unique(pred_label)) != 1:
                    connect_postprocess = torch.stack([torch.tensor(pred_label) == c+1 for c in index], dim=0).type(torch.int32)
                    connect_postprocess = torch.sum(connect_postprocess,dim=0).type(torch.float32)
                    dice_connected = metrics.dice_score_2(connect_postprocess.reshape(1,1,size, size), gt_samples)
                else:
                    dice_connected = np.nan
                #mse_loss, v, sr, p = loss_func.DIST_Pen(batch['mask_distmap'],preds.detach().numpy(), gt_samples,100)
                #mse_loss_100, v, sr, p = loss_func.DIST_Pen(batch['bbox_dist'],preds.detach().numpy(), gt_samples,100)
                #mse_loss_200, v, sr, p = loss_func.DIST_Pen(batch['bbox_dist'],preds.detach().numpy(), gt_samples,200)
                #mse_loss_300, v, sr, p = loss_func.DIST_Pen(batch['bbox_dist'],preds.detach().numpy(), gt_samples,300)
                #mse_loss_400, v, sr, p = loss_func.DIST_Pen(batch['bbox_dist'],preds.detach().numpy(), gt_samples,400)
                #mse_loss_500, v, sr, p = loss_func.DIST_Pen(batch['bbox_dist'],preds.detach().numpy(), gt_samples,500)
                print(batch['name'], dice, dice_closing, dice_connected)
                
                writer.writerow({'SLICE_ID': batch['name'], 'dice':np.float(dice),
                                 
                                 'Post_processed': np.float(dice_closing), 
                                 
                                 'dice_connected' : np.float(dice_connected), 'bb_filtered':np.float(bb_dice),
                                 'pred_connected': pred_connected, 'gt_connected': gt_connected, 'haus':np.float(m), 
                                 'c_error': np.float(np.abs(pred_connected - gt_connected))})
                
                
                '''
                writer.writerow({'SLICE_ID': batch['name'],'dice metric': dice*100, 'dist loss': mse_loss.item(), 
                                 'softdist loss_100': mse_loss_100,'softdist loss_200': mse_loss_200, 
                                 'softdist loss_300': mse_loss_300, 'softdist loss_400': mse_loss_400,
                                 'softdist loss_500': mse_loss_500})  
                '''
            


    
    

