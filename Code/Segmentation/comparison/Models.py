import torch
import torch.nn as nn
import torch.nn.functional as F
from .proj_utils.torch_utils import to_device
from .proj_utils.local_utils import Indexflow
import numpy as np

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
    def __init__(self, elu=True, nll=False, multi_contex=True):
        super(MultiContex_seg, self).__init__()
        self.multi_contex = multi_contex
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

        if self.multi_contex:
            self.up_12_100   = UpConv(256, 32, 2, elu, stride = 8)
            self.up_25_100   = UpConv(128,  32, 2, elu, stride = 4)
            
            self.det_tran  = LUConv(32*2, elu, inChans=32*3)
            self.seg_tran  = LUConv(32*2, elu, inChans=32*3)
            self.adv_tran  = LUConv(32*2, elu, inChans=32*3)
        else:
            self.det_tran = LUConv(32 * 2, elu, inChans=32)
            self.seg_tran = LUConv(32 * 2, elu, inChans=32)
            self.adv_tran = LUConv(32 * 2, elu, inChans=32)

        self.det_hid = _make_nConv(32 * 2, 2, elu)
        self.det_out = OutputTransition(32 * 2, 1, 2, elu)

        self.seg_hid = _make_nConv(32 * 2, 2, elu)
        self.seg_out = OutputTransition(32 * 2, 1, 2, elu)

        self.adv_hid = _make_nConv(32 * 2, 2, elu)
        self.adv_out = OutputTransition(32 * 2, 1, 2, elu)

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

        if self.multi_contex:
            out_up_12_100 = self.up_12_100(out_up_12, x.size()[2:])
            out_up_25_100 = self.up_25_100(out_up_25, x.size()[2:])
            concat_out = torch.cat([out_up_50_100,out_up_25_100, out_up_12_100 ], 1)
        else:
            concat_out = out_up_50_100

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
            return [det.cpu().data.numpy(),seg.cpu().data.numpy()] #,adv.cpu().data.numpy(),
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
            return [np.concatenate(det_results,axis=0), np.concatenate(seg_results,axis=0)]

