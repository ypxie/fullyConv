import torch
import torch.nn as nn
import torch.nn.functional as F
from .proj_utils.torch_utils import to_device
from .proj_utils.local_utils import Indexflow
import numpy as np

def passthrough(x, **kwargs):
    return x

def convAct(elu, nchan):
    if elu:
        return nn.ELU(inplace=True)
    else:
        return nn.PReLU(nchan)

# normalization between sub-volumes is necessary
# for good performance
class ContBatchNorm2d(nn.modules.batchnorm._BatchNorm):
    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))
        super(ContBatchNorm2d, self)._check_input_dim(input)

    def forward(self, input):
        self._check_input_dim(input)
        return F.batch_norm(
            input, self.running_mean, self.running_var, self.weight, self.bias,
            True, self.momentum, self.eps)

class LUConv(nn.Module):
    def __init__(self, nchan, elu, inChans=None):
        super(LUConv, self).__init__()
        if inChans is None:
            inChans = nchan
        self.act = convAct(elu, inChans)
        self.conv = nn.Conv2d(inChans, nchan, kernel_size=3, padding=1)
        self.bn = ContBatchNorm2d(nchan)

    def forward(self, x):
        out = self.act(self.bn(self.conv(x)))
        return out

def _make_nConv(nchan, depth, elu):
    layers = []
    if depth >=0:
        for _ in range(depth):
            layers.append(LUConv(nchan, elu))
        return nn.Sequential(*layers)
    else:
        return passthrough

class InputTransition(nn.Module):
    def __init__(self,inputChans, outChans, elu):
        self.outChans = outChans
        self.inputChans = inputChans
        super(InputTransition, self).__init__()
        self.conv = nn.Conv2d(inputChans, outChans, kernel_size=3, padding=1)
        self.bn = ContBatchNorm2d(outChans)
        self.relu = convAct(elu, outChans)

    def forward(self, x):
        # do we want a PRELU here as well?
        out = self.bn(self.conv(x))
        # split input in to 16 channels
        if self.inputChans == 1:
            x_aug = torch.cat([x]*self.outChans, 0)
            out = self.relu(torch.add(out, x_aug))
        else:  
            out = self.relu(out)
        return out

class DownTransition(nn.Module):
    def __init__(self, inChans, outChans, nConvs, elu, dropout=False):
        super(DownTransition, self).__init__()
        #outChans = 2*inChans
        self.down_conv = nn.Conv2d(inChans, outChans, kernel_size=3, padding=1, stride=2)
        self.bn1 = ContBatchNorm2d(outChans)
        self.do1 = passthrough
        self.relu1 = convAct(elu, outChans)
        self.relu2 = convAct(elu, outChans)
        if dropout:
            self.do1 = nn.Dropout2d()
        
        self.ops = _make_nConv(outChans, nConvs, elu)

    def forward(self, x):
        down = self.relu1(self.bn1(self.down_conv(x)))
        out = self.do1(down)
        out = self.ops(out)
        out = self.relu2(out+down)
        return out

def match_tensor(out, refer_shape):
    skiprow,skipcol = refer_shape
    row, col = out.size()[2], out.size()[3]
    if skipcol >= col:
        pad_col = skipcol - col        
        left_pad_col  = pad_col // 2
        right_pad_col = pad_col - left_pad_col      
        out = F.pad(out, (left_pad_col, right_pad_col, 0,0))
    else:
        crop_col = col - skipcol
        left_crop_col  = crop_col // 2
        right_col = left_crop_col + skipcol
        out = out[:,:,:, left_crop_col:right_col]
    
    if skiprow >= row:
        pad_row = skiprow - row
        left_pad_row  = pad_row // 2
        right_pad_row = pad_row - left_pad_row
        out = F.pad(out, (0,0, left_pad_row,right_pad_row))
    else:
        crop_row = row - skiprow   
        left_crop_row  = crop_row // 2
        
        right_row = left_crop_row + skiprow
        
        out = out[:,:,left_crop_row:right_row, :]
        
    return out


class UpConcat(nn.Module):
    def __init__(self, inChans, hidChans, outChans, nConvs, elu, dropout=False,stride=2):
        # remeber inChans is mapped to hidChans, then concate together with skipx, the mixed channel =outChans
        super(UpConcat, self).__init__()
        #hidChans = outChans // 2
        self.up_conv = nn.ConvTranspose2d(inChans, hidChans, kernel_size=3, 
                                          padding=1, stride=stride, output_padding=1)
        self.bn1 = ContBatchNorm2d(hidChans)
        self.do1 = passthrough
        self.do2 = nn.Dropout2d()
        self.relu1 = convAct(elu, hidChans)
        self.relu2 = convAct(elu, outChans)
        if dropout:
            self.do1 = nn.Dropout2d()
        
        self.ops = _make_nConv(outChans, nConvs, elu)

    def forward(self, x, skipx):
        out = self.do1(x)
        skipxdo = self.do2(skipx)
        out = self.relu1(self.bn1(self.up_conv(out)))
        out = match_tensor(out, skipxdo.size()[2:])

        xcat = torch.cat([out, skipxdo], 1)
        out  = self.ops(xcat)
        out  = self.relu2(out + xcat)
        return out

class UpConv(nn.Module):
    def __init__(self, inChans, outChans, nConvs, elu, dropout=False, stride = 2):
        super(UpConv, self).__init__()
        #hidChans = outChans // 2
        self.up_conv = nn.ConvTranspose2d(inChans, outChans, kernel_size=3, 
                                          padding=1, stride = stride, output_padding=1)
        self.bn1 = ContBatchNorm2d(outChans)
        self.do1 = passthrough
        self.relu1 = convAct(elu, outChans)
        if dropout:
            self.do1 = nn.Dropout2d()

    def forward(self, x, dest_size):
        '''
        dest_size should be (row, col)
        '''
        out = self.do1(x)
        out = self.relu1(self.bn1(self.up_conv(out)))
        out = match_tensor(out, dest_size)
        return out

class OutputTransition(nn.Module):
    def __init__(self, inChans,outChans=1,hidChans=2, elu=True):
        super(OutputTransition, self).__init__()
        self.conv1 = nn.Conv2d(inChans, hidChans, kernel_size=5, padding=2)
        self.bn1 = ContBatchNorm2d(hidChans)
        self.relu1 = convAct(elu, hidChans)
        self.conv2 = nn.Conv2d(hidChans, outChans, kernel_size=1)

    def forward(self, x):
        # convolve 32 down to 2 channels
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.conv2(out)
        # make channels the last axis
        #out = out.permute(0, 2, 3, 1).contiguous()
        # flatten
        #out = out.view(out.numel() // 2, 2)
        #out = self.final(out)
        #treat channel 0 as the predicted output
        return out

class VNet(nn.Module):
    # the number of convolutions in each layer corresponds
    # to what is in the actual prototxt, not the intent
    def __init__(self, elu=True, nll=False):
        super(VNet, self).__init__()
        self.register_buffer('device_id', torch.zeros(1))
        self.in_tr_100     = InputTransition(3, 16, elu) 
        self.down_tr32_50  = DownTransition(16, 32, 1, elu)
        self.down_tr64_25  = DownTransition(32, 64, 2, elu)
        self.down_tr128_12 = DownTransition(64, 128, 2, elu, dropout=True)
        self.down_tr256_6  = DownTransition(128, 256,  2, elu, dropout=True)
        self.up_tr256_12   = UpConcat(256, 128, 256, 2, elu, dropout=True)
        self.up_tr128_25   = UpConcat(256, 64, 128, 2, elu, dropout=True)
        self.up_tr64_50    = UpConcat(128, 32, 64, 1, elu)
        self.up_tr32_100   = UpConcat(64,  16, 32, 1, elu)
        self.out_tr        = OutputTransition(32, 1,2, elu)

    def forward(self, x):
        x = to_device(x,self.device_id)
        out16 = self.in_tr_100(x)
        out32 = self.down_tr32_50(out16)
        out64 = self.down_tr64_25(out32)
        out128 = self.down_tr128_12(out64)
        out256 = self.down_tr256_6(out128)
        out = self.up_tr256_12(out256, out128)
        out = self.up_tr128_25(out, out64)
        out = self.up_tr64_50(out, out32)
        out = self.up_tr32_100(out, out16)
        out = self.out_tr(out)
        return out
    def predict(self, x):
        self.eval()
        x = to_device(x,self.device_id)
        return self.forward(x)

class MultiContex(nn.Module):
    # the number of convolutions in each layer corresponds
    # to what is in the actual prototxt, not the intent
    def __init__(self, elu=True, nll=False):
        super(MultiContex, self).__init__()
        self.register_buffer('device_id', torch.zeros(1))
        self.in_tr_100     = InputTransition(3, 16, elu) 
        self.down_tr32_50  = DownTransition(16, 32, 1, elu)
        self.down_tr64_25  = DownTransition(32, 64, 2, elu)
        self.down_tr128_12 = DownTransition(64, 128, 2, elu, dropout=True)
        self.down_tr256_6  = DownTransition(128, 256,  2, elu, dropout=True)

        self.up_tr256_12   = UpConcat(256, 128, 256, 2, elu, dropout=True)
        self.up_tr128_25   = UpConcat(256, 64, 128, 2, elu, dropout=True)
        self.up_tr64_50    = UpConcat(128, 32, 64, 1, elu)
        self.up_tr32_100   = UpConcat(64,  16, 32, 1, elu)
       
        self.up_12_100   = UpConv(256, 32, 2, elu, stride = 8)
        self.up_25_100   = UpConv(128,  32, 2, elu, stride = 4)
        
        self.out_tr      = OutputTransition(32*3, 1, 2, elu)

    def forward(self, x):
        x = to_device(x,self.device_id)
        out16 = self.in_tr_100(x)
        out32 = self.down_tr32_50(out16)
        out64 = self.down_tr64_25(out32)
        out128 = self.down_tr128_12(out64)
        out256 = self.down_tr256_6(out128)
        out_up_12 = self.up_tr256_12(out256, out128)
        out_up_25 = self.up_tr128_25(out_up_12, out64)
        out_up_50 = self.up_tr64_50(out_up_25, out32)
        out_up_50_100 = self.up_tr32_100(out_up_50, out16)

        out_up_12_100 = self.up_12_100(out_up_12, x.size()[2:])
        out_up_25_100 = self.up_25_100(out_up_25, x.size()[2:])

        out = torch.cat([out_up_50_100,out_up_12_100, out_up_25_100 ], 1)

        out = self.out_tr(out)
        return out

    def predict(self, x, batch_size=None):
        self.eval()
        x = to_device(x,self.device_id).float()
        total_num = x.size()[0]
        if batch_size is None or batch_size <= total_num:
            return self.forward(x).cpu().data.numpy()
        else:
            results = []
            for ind in Indexflow(total_num, batch_size, False):
                devInd = to_device(torch.from_numpy(ind), self.device_id, False)
                data = x[devInd]
                results.append(self.forward(data).cpu().data.numpy())
            return np.concatenate(results,axis=0)

class MultiContex_seg(nn.Module):
    # the number of convolutions in each layer corresponds
    # to what is in the actual prototxt, not the intent
    def __init__(self, elu=True, nll=False):
        super(MultiContex_seg, self).__init__()
        self.register_buffer('device_id', torch.zeros(1))
        self.in_tr_100     = InputTransition(3, 16, elu) 
        self.down_tr32_50  = DownTransition(16, 32, 1, elu)
        self.down_tr64_25  = DownTransition(32, 64, 2, elu)
        self.down_tr128_12 = DownTransition(64, 128, 2, elu, dropout=True)
        self.down_tr256_6  = DownTransition(128, 128,  2, elu, dropout=True)

        self.up_tr256_12   = UpConcat(128, 128, 256, 2, elu, dropout=True)
        self.up_tr128_25   = UpConcat(256, 64, 128, 2, elu, dropout=True)
        self.up_tr64_50    = UpConcat(128, 32, 64, 1, elu)
        self.up_tr32_100   = UpConcat(64,  16, 32, 1, elu)

        self.up_12_100   = UpConv(256, 32, 2, elu, stride = 8)
        self.up_25_100   = UpConv(128,  32, 2, elu, stride = 4)

        self.det_tran  = LUConv(32*2, elu, inChans=32*3)
        self.det_hid   = _make_nConv(32*2, 2, elu)
        self.det_out   = OutputTransition(32*2, 1, 2, elu)
        
        self.seg_tran  = LUConv(32*2, elu, inChans=32*3)
        self.seg_hid   = _make_nConv(32*2, 2, elu)
        self.seg_out   = OutputTransition(32*2, 1, 2, elu)

        self.adv_tran  = LUConv(32*2, elu, inChans=32*3)
        self.adv_hid   = _make_nConv(32*2, 2, elu)
        self.adv_out   = OutputTransition(32*2, 1, 2, elu)

        #self.dense_mean = nn.Linear(32*3, 128)
        #self.dense_adv  = nn.Linear(128, 1)
        #self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x, adv=True):
        x = to_device(x,self.device_id)
        out16 = self.in_tr_100(x)
        out32_50 = self.down_tr32_50(out16)
        out64_25 = self.down_tr64_25(out32_50)
        out128_12 = self.down_tr128_12(out64_25)
        out256_6 = self.down_tr256_6(out128_12)

        out_up_12 = self.up_tr256_12(out256_6, out128_12)
        out_up_25 = self.up_tr128_25(out_up_12, out64_25)
        out_up_50 = self.up_tr64_50(out_up_25, out32_50)

        out_up_50_100 = self.up_tr32_100(out_up_50, out16)
        out_up_12_100 = self.up_12_100(out_up_12, x.size()[2:])
        out_up_25_100 = self.up_25_100(out_up_25, x.size()[2:])

        concat_out = torch.cat([out_up_50_100,out_up_25_100, out_up_12_100 ], 1)

        det_tran = self.det_tran(concat_out)
        det_hid  = self.det_hid(det_tran)
        det_out  = self.det_out(det_hid)

        seg_tran = self.seg_tran(concat_out)
        seg_hid  = self.seg_hid(seg_tran)
        seg_out  = F.sigmoid(self.seg_out(seg_hid))
        
        adv_out = None
        if adv:
            #adv_cat = torch.cat([concat_out, det_out, seg_out], 1)
            adv_tran = self.adv_tran(concat_out)
            adv_hid  = self.adv_hid(adv_tran)
            adv_out  = F.relu(self.seg_out(adv_hid))
            #mean_map = F.avg_pool2d(concat_out, (concat_out.size()[2], concat_out.size()[3]))
            #mean_map = mean_map.view(-1, mean_map.size()[1])
            #adv = self.dense_adv(self.leaky_relu(self.dense_mean(mean_map)))

        return det_out, seg_out, adv_out


    def predict(self, x, batch_size=None):
        self.eval()
        x = to_device(x,self.device_id).float()
        total_num = x.size()[0]
        if batch_size is None or batch_size <= total_num:
            det, seg, _ = self.forward(x, adv=False)
            return det.cpu().data.numpy(),seg.cpu().data.numpy() #,adv.cpu().data.numpy(),
        else:
            det_results = []
            seg_results = []
            #advs = []
            for ind in Indexflow(total_num, batch_size, False):
                devInd = to_device(torch.from_numpy(ind), self.device_id, False)
                data = x[devInd]
                det, seg, adv = self.forward(data)
                det_results.append(det.cpu().data.numpy())
                seg_results.append(seg.cpu().data.numpy())
                #advs.append(adv.cpu().data.numpy())
            return np.concatenate(det_results,axis=0), np.concatenate(seg_results,axis=0)
