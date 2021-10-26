import sys
import DataSEt_Classes
import models_2 as models
import loss_func
from metrics import dice_score_2
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torch import optim
import numpy as np
#import nonechucks as nc
import os
import time
import torch
from torch import nn
import warnings
import matplotlib.pyplot as plt
import torch.nn.functional as F

# SAVING RESULTS
dirr = '/home/eljurros/spare-workplace/JSTP_BBUNET/JSTP_BBUNET/h5_files/FOLD_1/h5_files'                  # the directory where the .h5 files are present
txt_file_path ='/home/eljurros/spare-workplace/JSTP_BBUNET/JSTP_BBUNET/h5_files/FOLD_1/text.txt'       # where the evolution text file should be saved
chkpt_path = '/home/eljurros/spare-workplace/JSTP_BBUNET/JSTP_BBUNET/h5_files/FOLD_1'  # where the checkpoint should be saved
name = 'model.ckpt'   # name of checkpoint
# parameters
size = 320     # dimension of image
initial_lr = 0.001     # learning rate
num_epochs = 200           # number of training epochs
opt_loss = 1000              # optimum loss variable
loss_type = 'dice'
out_type_selection = 'mask'
gt_mode = 'norm'
model_type = 'Unet'
bb_state = "bbox"
n_organs = 1
opt_loss = 1000
init = 0
d_prev = d_metric = 0
modality = 2 

def Get_SoftMax(matrix):
    soft_max = nn.Softmax(dim=1)
    return soft_max(matrix)


if model_type == 'Unet':
    model = models.Unet(in_dim = modality, drop_rate=0.4, bn_momentum=0.1,n_organs=n_organs+1)

elif model_type == 'BB_Unet':
    model = models.Unet(drop_rate=0.4, bn_momentum=0.1,n_organs=n_organs+1)
    # model = model.to('cuda')

transform = transforms.Compose([
        transforms.ToTensor(),
])

optimizer = optim.Adam(model.parameters(), lr=initial_lr)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)
model = nn.DataParallel(model)


for epoch in range(0, num_epochs+1):
    start_time = time.time()
    
    for phase in ['train', 'val']:
        if phase == 'val': 
            pass
        dataset = DataSEt_Classes.SegTHor_2D_TrainDS(dirr, transform=transform,organ_n = n_organs,ds_mode=phase,size=400)
        #dataset = nc.SafeDataset(dataset)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            loader = DataLoader(dataset, batch_size=2, shuffle=True,
                                pin_memory=False,
                                num_workers=1)
            # loader = nc.SafeDataLoader(dataset, batch_size=10, shuffle=True )
            if phase in ['train', 'train_ancillary', 'train_Primary']:
                #model = model.to('cuda')
                scheduler.step()
                lr = scheduler.get_lr()[0]
                train_loss_total = 0.0
                tr_dice_total = 0
                tr_len = len(loader)
            elif phase == 'val':
                model.eval()
                val_loss_total = 0
                val_dice_total = 0
                val_len = len(loader)
            for k, batch in enumerate(loader):
                input_samples, gt_samples = batch["bbox_img"], batch["gt"]
                var_input = input_samples.reshape(-1, modality, size, size)
                var_input = torch.tensor(var_input, dtype=torch.float)
                var_input = Variable(var_input, requires_grad=True)
                gt_samples = torch.tensor(gt_samples, dtype=torch.float)
                with torch.set_grad_enabled(phase in ['train', 'train_ancillary', 'train_Primary']):
                    if model_type == 'Unet':
                        #model = model.to('cuda')
                        #preds = model(var_input.to('cuda'))
                        preds = model(var_input)
                        preds = F.softmax(preds,dim=1)
                    elif model_type == 'BB_Unet':
                        var_bbox = batch[bb_state].reshape(-1, 1, size, size)
                        var_bbox = torch.tensor(var_bbox, dtype=torch.float)
                        var_bbox = Variable(var_bbox, requires_grad=True)
                        preds = model(var_input.to('cuda'))
                        preds = F.softmax(preds, dim=1)

                    #loss = loss_func.dice_loss(preds, gt_samples.to('cuda')).mean() 
                    loss = loss_func.dice_loss(preds, gt_samples).mean() 


                    predicted_output = np.argmax(preds.to('cpu').detach(), axis=1)
                    predicted_output = predicted_output.type(torch.float32)
                    if n_organs == 1:
                        d_metric = dice_score_2(predicted_output[0].to('cpu'), gt_samples[:,1,:,])
                    else:
                        d_metric = []
                        for i in range(2):
                            d_metric.append(dice_score_2(predicted_output[:,i,:,].to('cpu'), gt_samples[:,i,:,]))
                            
                    if phase in ['train_ancillary','train']: 
                        if n_organs == 1:
                            if len(np.unique(gt_samples[:,1,:,])) == 2: 
                                train_loss_total += loss.item()
                                optimizer.zero_grad()
                                loss.backward()
                                optimizer.step()
                                print ('training on batch{} on epoch {} with loss {} with evaluation : {}'.format(np.float(k), epoch,loss, d_metric))
                        else:
                            train_loss_total += loss.item()
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()
                            print ('training on batch{} on epoch {} with loss {} with evaluation : {}'.format(np.float(k), epoch,loss, d_metric))

                    else:
                        val_loss_total += loss.item()
                        print('validating on batch{} on epoch {} with loss {} with evaluation : {}'.format(np.float(k), epoch,loss, d_metric))

    average_val_loss = np.float(val_loss_total)/val_len
    average_tr_loss = np.float(train_loss_total)/tr_len
    
    with open(txt_file_path, 'a') as the_file:
        the_file.write('{}###{}###{}###{}'.format(epoch, average_tr_loss,  average_val_loss,d_metric))
        the_file.write('\n') 
    
    print('{}###{}###{}'.format(epoch, average_tr_loss,  average_val_loss))
    
    if opt_loss > average_val_loss:
        opt_loss = average_val_loss
        
        if os.path.exists(chkpt_path) is False:
            os.mkdir(chkpt_path)

        save_path = os.path.join(chkpt_path, name)
        torch.save(model.to('cpu'), save_path)
        print ('model saved-{}-{}'.format(epoch, opt_loss))


