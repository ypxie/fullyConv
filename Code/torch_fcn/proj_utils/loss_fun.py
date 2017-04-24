import numpy as np
import torch
import torch.nn.functional as F

def weighted_loss( base='mse'):
    '''
    The y_pred_mask has the mask in the last dimension, but the y_true does not have that
    y_true is (batch, channel, row, col), we need to permite the dimensio first
    '''
    def loss(y_true_mask, y_pred):
        assert len(y_pred.size()) <= 5, 'dimension larger than 5 is not implemented yet!!'
        if len(y_pred.size()) == 3:
            y_true = y_true_mask[:,0:-1, :].permute(0,2,1)  #torch.transpose(y_true_mask[:,0:-1, :],(0,2,1))
            y_pred = y_pred.permute(0,2,1)       #torch.transpose(y_pred,(0,2,1))
            y_mask = y_true_mask[:,-1,:]
        elif len(y_pred.size()) == 4:
            y_true = y_true_mask[:,0:-1,:,:].permute(0,2,3,1) #torch.transpose(y_true_mask[:,0:-1,:,:],(0,2,3,1))
            y_pred = y_pred.permute(0,2,3,1) #torch.transpose(y_pred, (0,2,3,1))
            y_mask = y_true_mask[:,-1,:,:]
        elif len(y_pred.size()) == 5:
            y_true = y_true_mask[:,0:-1,:,:,:].permute(0,2,3,4,1) #torch.transpose(y_true_mask[:,0:-1,:,:,:], (0,2,3,4,1))
            y_pred = y_pred.permute(0,2,3,4,1)  #torch.transpose(y_pred, (0,2,3,4,1))
            y_mask = y_true_mask[:,-1,:,:,:]
        #print(y_true.size())
        #print(y_pred.size())
        naive_loss = get(base)(y_true,y_pred)
        masked =  naive_loss * y_mask
        return torch.mean(masked)

    return loss


def fcn_loss( base='mse'):
    '''
    The y_pred_mask has the mask in the last dimension, but the y_true does not have that
    y_true is (batch, channel, row, col),we need to permite the dimensio first
    '''
    def loss(y_true, y_pred):
        assert len(y_pred.size()) <= 5, 'dimension larger than 5 is not implemented yet!!'
        if len(y_pred.size()) == 3:
            y_true = y_true.permute(0,2,1)  # K.permute_dimensions(y_true,(0,2,1))
            y_pred = y_pred.permute(0,2,1)  #K.permute_dimensions(y_pred,(0,2,1))

        elif len(y_pred.size()) == 4:
            y_true = y_true.permute(0,2,3,1) #K.permute_dimensions(y_true,(0,2,3,1))
            y_pred = y_pred.permute(0,2,3,1) #K.permute_dimensions(y_pred, (0,2,3,1))

        elif len(y_pred.size()) == 5:
            y_true = y_true.permute(0,2,3,4,1)   #K.permute_dimensions(y_true, (0,2,3,4,1))
            y_pred = y_pred.permute(0,2,3,4,1)  #K.permute_dimensions(y_pred, (0,2,3,4,1))

        naive_loss = get(base)(y_true,y_pred)
        return naive_loss.mean()
        
    return loss

smooth = 1.

def dice_coef_loss(y_true, y_pred):
    y_true_f = y_true.view(-1)
    y_pred_f = y_pred.view(-1)
    intersection = torch.sum(y_true_f * y_pred_f)
    return 1- (2. * intersection + smooth) / (torch.sum(y_true_f*y_true_f) + torch.sum(y_pred_f*y_pred_f) + smooth)

def mean_squared_error(y_true, y_pred):
    diff = y_pred - y_true
    last_dim = len(y_pred.size()) - 1
    return torch.mean(diff*diff, dim=last_dim)

def binary_crossentropy(y_true, y_pred):
    #y_pred should be pre_softmax.
    sumit = -y_true*F.logsigmoid(y_pred) - (1-y_true) * F.log(1-F.sigmoid(y_pred))
    last_dim = len(sumit.size()) - 1
    return torch.mean(sumit, dim=last_dim)

# aliases
mse = MSE = mean_squared_error
dice = dice_coef_loss
#mae = MAE = mean_absolute_error

from .generic_utils import get_from_module
def get(identifier):
    return get_from_module(identifier, globals(), 'objective')

