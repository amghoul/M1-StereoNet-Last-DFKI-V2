import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.backends.cudnn as cudnn
from .factorizer import Factorizer
from .spatioTemporalConv_General import SpatioTemporalConv
#from .spatioTemporalConv import SpatioTemporalConv
###############
import time
###############
#from .pytorch_memlab import LineProfiler,profile, set_target_gpu, MemReporter
#torch.backends.cudnn.benchmark = False  
#from .memory import log_mem, log_mem_amp, log_mem_amp_cp, log_mem_cp
#from .plot import plot_mem, pp
#import pandas as pd
##############
cuda1 = torch.device('cuda:0')
repetitions=5
rep = 0
num_components = 8
comp_names={"fe":0,"cv":1,"cf":2,"st0":3,"st1":4,"st2":5,"st3":6,"ft":7}
com_fullNames=["Feature_Extraction", "Cost_Volume", "Cost_Filtering","Stage0","Stage1","Stage2","Stage3","Full_time"]
timings_components=[]
for i in range(num_components):
    timings_components.append(np.zeros((repetitions,1)))
#print(len(timings_components[0]))
##############

def convbn(in_channel, out_channel, kernel_size, stride, pad, dilation, model_bn=1):
    
    if model_bn ==1:
        return nn.Sequential(
        nn.Conv2d(
            in_channel,
            out_channel,
            kernel_size=kernel_size,
            stride=stride,
            padding=dilation if dilation>1 else pad,
            dilation=dilation),
       nn.BatchNorm2d(out_channel))
    else: # model_bn=0
        return nn.Sequential(
        nn.Conv2d(
            in_channel,
            out_channel,
            kernel_size=kernel_size,
            stride=stride,
            padding=dilation if dilation>1 else pad,
            dilation=dilation))
    
def soft_argmin(cost_volume):
    """Remove single-dimensional entries from the shape of an array."""
    # cost_volume_D_squeeze = torch.squeeze(cost_volume, dim=1)

    softmax = nn.Softmax(dim=1)
    disparity_softmax = softmax(-cost_volume)

    d_grid = torch.arange(cost_volume.shape[1], dtype=torch.float)
    d_grid = d_grid.reshape(-1, 1, 1)
    d_grid = d_grid.repeat((cost_volume.shape[0], 1, cost_volume.shape[2], cost_volume.shape[3])) # [batchSize, 1, h, w]
    d_grid = d_grid.to('cuda')

    tmp = disparity_softmax*d_grid
    arg_soft_min = torch.sum(tmp, dim=1, keepdim=True)

    return arg_soft_min

class BasicBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride, downsample, pad, dilation,model_bn):
        super().__init__()
        self.conv1 = nn.Sequential(
            convbn(in_channel, out_channel, 3, stride, pad, dilation,model_bn),
            nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.conv2 = convbn(out_channel, out_channel, 3, 1, pad, dilation,model_bn)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)

        # out = self.conv2(out)
        if self.downsample is not None:
            x = self.downsample(x)
        ### bug?
        out = x + out
        return out

class FeatureExtraction(nn.Module):
    def __init__(self, k,model_bn):
        super().__init__()
        self.k = k
        self.downsample = nn.ModuleList()
        in_channel = 3
        out_channel = 32
        for _ in range(k):
            self.downsample.append(
                nn.Conv2d(
                    in_channel,
                    out_channel,
                    kernel_size=5,
                    stride=2,
                    padding=2))
            in_channel = out_channel
            out_channel = 32
        self.residual_blocks = nn.ModuleList()
        for _ in range(6):
            self.residual_blocks.append(
                BasicBlock(
                    32, 32, stride=1, downsample=None, pad=1, dilation=1,model_bn=model_bn))
        self.conv_alone = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
    def forward(self, rgb_img):
        output = rgb_img
        for i in range(self.k):
            output = self.downsample[i](output)
        for block in self.residual_blocks:
            output = block(output)
        return self.conv_alone(output)

class EdgeAwareRefinement(nn.Module):
    def __init__(self, in_channel,model_bn):
        super().__init__()
        self.conv2d_feature = nn.Sequential(
            convbn(in_channel, 32, kernel_size=3, stride=1, pad=1, dilation=1,model_bn=model_bn),
            nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.residual_astrous_blocks = nn.ModuleList()
        astrous_list = [1, 2, 4, 8 , 1 , 1]
        for di in astrous_list:
            self.residual_astrous_blocks.append(
                BasicBlock(
                    32, 32, stride=1, downsample=None, pad=1, dilation=di,model_bn=model_bn))
                
        self.conv2d_out = nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, low_disparity, corresponding_rgb):
        output = torch.unsqueeze(low_disparity, dim=1)
        twice_disparity = F.interpolate(
            output,
            size = corresponding_rgb.size()[-2:],
            mode='bilinear',
            align_corners=False)
        '''
        if corresponding_rgb.size()[-1]/ low_disparity.size()[-1] >= 1.5:
            twice_disparity *= 8
        '''
        output = self.conv2d_feature(
            torch.cat([twice_disparity, corresponding_rgb], dim=1))
        for astrous_block in self.residual_astrous_blocks:
            output = astrous_block(output)
        
        return nn.ReLU(inplace=True)(torch.squeeze(
            twice_disparity + self.conv2d_out(output), dim=1))
        
class disparityregression(nn.Module):
    def __init__(self, maxdisp):
        super().__init__()
        self.disp = torch.cuda.FloatTensor(
            np.reshape(np.array(range(maxdisp)), [1, maxdisp, 1, 1]))

    def forward(self, x):
        disp = self.disp.repeat(x.size()[0], 1, x.size()[2], x.size()[3]) #[1,24,68,120]
        out = torch.sum(x * disp, 1)   #[1,68,120]
        return out

class CostVolumeFiltering(nn.Module):
    def __init__(self, is_filter1_differ,filter1_kernels,fact_kernels,ch_in, ch_out, subspace_scale, stream_axes,BN_1D,BN_2D,BN_1D_last,model_bn,use_l1_norm=1):
        super().__init__()
        assert (is_filter1_differ == 1 or is_filter1_differ ==0), "is_filter1_differ argument must have value 0 or 1"
        no_streams = len(stream_axes)
        self.filterBlocks = nn.ModuleList()
        if is_filter1_differ==1:
            factorizer_filter1 = Factorizer(filter1_kernels, ch_in, ch_out, subspace_scale, stream_axes)
            for i, stream in enumerate(factorizer_filter1.streams):
                for _ in range(1): 
                    self.filterBlocks.append(SpatioTemporalConv(stream,ch_in,ch_out,model_bn,False,BN_1D,BN_2D,BN_1D_last))
                    if use_l1_norm == 1:
                        ch_in == ch_out
        ####
        factorizer = Factorizer(fact_kernels, ch_in, ch_out, subspace_scale, stream_axes)
        for i, stream in enumerate(factorizer.streams):
            rem_blocks = 4-is_filter1_differ
            if rem_blocks == 3: # if is_filter1_differ == 1 and use_l1_norm==1
                for i in range(rem_blocks): #3
                    self.filterBlocks.append(SpatioTemporalConv(stream,ch_in,ch_out,model_bn,False,BN_1D,BN_2D,BN_1D_last))
            else:  #if is_filter1_differ == 0
                if use_l1_norm == 1:
                    for i in range(rem_blocks): #4
                        if i == 0 :
                            self.filterBlocks.append(SpatioTemporalConv(stream,ch_in,ch_out,model_bn,False,BN_1D,BN_2D,BN_1D_last))
                        else:
                            ch_in = ch_out
                            self.filterBlocks.append(SpatioTemporalConv(stream,ch_in,ch_out,model_bn,False,BN_1D,BN_2D,BN_1D_last))
                else:
                     for i in range(rem_blocks): #4
                        self.filterBlocks.append(SpatioTemporalConv(stream,ch_in,ch_out,model_bn,False,BN_1D,BN_2D,BN_1D_last))
            self.filterBlocks.append(SpatioTemporalConv(stream,ch_in,1,model_bn,True))
            
    def forward(self, x):
        #i =0
        for f in self.filterBlocks:
            #print("x size for block i= ", i, x.size())
            x= f(x)
            #i = i+1
        return x

class StereoNet(nn.Module):
    def __init__(self, k, r, is_filter1_differ,filter1_kernels,fact_kernels,BN_1D,BN_2D,BN_1D_last,model_bn,use_l1_norm=0,maxdisp=192):
        super().__init__()
        self.use_l1_norm = use_l1_norm
        self.maxdisp = maxdisp
        self.k = k
        self.r = r
        self.feature_extraction = FeatureExtraction(k,model_bn)
        if self.use_l1_norm == 1:
            self.filter = CostVolumeFiltering(is_filter1_differ,filter1_kernels,fact_kernels,1, 32, 1, [1],BN_1D,BN_2D,BN_1D_last,model_bn)
        else:
            self.filter = CostVolumeFiltering(is_filter1_differ,filter1_kernels,fact_kernels,32, 32, 1, [1],BN_1D,BN_2D,BN_1D_last,model_bn)
        
        self.edge_aware_refinements = nn.ModuleList()
        for _ in range(self.r): ### my change from 1 to 4
            self.edge_aware_refinements.append(EdgeAwareRefinement(4,model_bn))
    #@profile
    def forward(self, left, right):
        disp = (self.maxdisp + 1) // pow(2, self.k)
        refimg_feature = self.feature_extraction(left)
        targetimg_feature = self.feature_extraction(right)

        cost = torch.cuda.FloatTensor(refimg_feature.size()[0], refimg_feature.size()[1], 
            disp,refimg_feature.size()[2],refimg_feature.size()[3]).zero_()

        for i in range(disp):
            if i > 0:
                cost[:, :, i, :, i:] = refimg_feature[ :, :, :, i:] - targetimg_feature[:, :, :, :-i]
            else:
                cost[:, :, i, :, :] = refimg_feature - targetimg_feature
        
        if self.use_l1_norm == 1:
            cost = (torch.sum(torch.abs(cost),1)).unsqueeze(1)

        cost = cost.contiguous() # cost here is [1, 32, 24, 46, 154] 
        #print("cost size: ", cost.size())
        cost = self.filter(cost)  # [1, 1, 24, 46, 154]
        cost = torch.squeeze(cost, 1) # [1, 24, 46, 154]
        pred =soft_argmin(cost)
        #pred = F.softmax(cost, dim=1) #[1, 24, 46, 154]
        pred = disparityregression(disp)(pred) # [1, 46, 154]
        
        #img_pyramid_list = [left]
        img_pyramid_list = []
        for i in range(self.r):
            img_pyramid_list.append( F.interpolate(left,scale_factor=1/pow(2,i), mode='bilinear', align_corners=False))
        
        img_pyramid_list.reverse() # [1,3,135,240] [1,3,270,480] [1,3,540,960]
        
        pred_pyramid_list= [pred] # [1,68,120] unrefined

        for i in range(self.r):
            pred_pyramid_list.append(self.edge_aware_refinements[i](pred_pyramid_list[i], img_pyramid_list[i]))

        length_all = len(pred_pyramid_list)

        for i in range(length_all): # my change from 1 to 4
            #if i ==0:
                #pred_pyramid_list[i] = pred_pyramid_list[i]* (left.size()[-1] / pred_pyramid_list[i].size()[-1])
            pred_pyramid_list[i] = pred_pyramid_list[i] * 8 ## because we take cost volume 1/8
            
            pred_pyramid_list[i] = torch.squeeze(
            F.interpolate(torch.unsqueeze(pred_pyramid_list[i], dim=1),size=left.size()[-2:], mode='bilinear', align_corners=False),dim=1)

        return pred_pyramid_list

#########################for Prof
class StereoNetProf(nn.Module):
    def __init__(self, k, r, is_filter1_differ,filter1_kernels,fact_kernels,BN_1D,BN_2D,BN_1D_last,model_bn,maxdisp=192):
        super().__init__()
        self.maxdisp = maxdisp
        self.k = k
        self.r = r
        self.feature_extraction = FeatureExtraction(k,model_bn)
        self.filter = CostVolumeFiltering(is_filter1_differ,filter1_kernels,fact_kernels,32, 32, 1, [1],BN_1D,BN_2D,BN_1D_last,model_bn)
        
        self.edge_aware_refinements = nn.ModuleList()
        for _ in range(self.r): ### my change from 1 to 4
            self.edge_aware_refinements.append(EdgeAwareRefinement(4,model_bn))
    #@profile
    def forward(self, left, right):
        disp = (self.maxdisp + 1) // pow(2, self.k)
        refimg_feature = self.feature_extraction(left)
        targetimg_feature = self.feature_extraction(right)

        cost = torch.cuda.FloatTensor(refimg_feature.size()[0], refimg_feature.size()[1], 
            disp,refimg_feature.size()[2],refimg_feature.size()[3]).zero_()

        for i in range(disp):
            if i > 0:
                cost[:, :, i, :, i:] = refimg_feature[ :, :, :, i:] - targetimg_feature[:, :, :, :-i]
            else:
                cost[:, :, i, :, :] = refimg_feature - targetimg_feature
        cost = cost.contiguous() # cost here is [1, 32, 24, 46, 154] 
        cost = self.filter(cost)  # [1, 1, 24, 46, 154]
        cost = torch.squeeze(cost, 1) # [1, 24, 46, 154]
        
        pred =soft_argmin(cost)
        #pred = F.softmax(cost, dim=1) #[1, 24, 46, 154]
        pred = disparityregression(disp)(pred) # [1, 46, 154]
        
        #img_pyramid_list = [left]
        img_pyramid_list = []
        for i in range(self.r):
            img_pyramid_list.append( F.interpolate(left,scale_factor=1/pow(2,i), mode='bilinear', align_corners=False))
        
        img_pyramid_list.reverse() # [1,3,135,240] [1,3,270,480] [1,3,540,960]
        
        pred_pyramid_list= [pred] # [1,68,120] unrefined

        for i in range(self.r):
            pred_pyramid_list.append(self.edge_aware_refinements[i](pred_pyramid_list[i], img_pyramid_list[i]))

        length_all = len(pred_pyramid_list)
        
        for i in range(length_all): # my change from 1 to 4
            pred_pyramid_list[i] = pred_pyramid_list[i] * 8
            
            pred_pyramid_list[i] = torch.squeeze(
            F.interpolate(torch.unsqueeze(pred_pyramid_list[i], dim=1),size=left.size()[-2:], mode='bilinear', align_corners=False),dim=1)

        return pred_pyramid_list

############StereoNet for timinng
class StereoNetTime(nn.Module):
    def __init__(self, k, r, is_filter1_differ,filter1_kernels,fact_kernels, BN_1D,BN_2D,BN_1D_last,model_bn,maxdisp=192):
        super().__init__()
        self.maxdisp = maxdisp
        self.k = k
        self.r = r
        self.feature_extraction = FeatureExtraction(k,model_bn)
        self.filter = CostVolumeFiltering(is_filter1_differ,filter1_kernels,fact_kernels,32, 32,  1, [1],BN_1D,BN_2D,BN_1D_last,model_bn)
        
        self.edge_aware_refinements = nn.ModuleList()
        for _ in range(self.r): ### my change from 1 to 4
            self.edge_aware_refinements.append(EdgeAwareRefinement(4,model_bn))
    #@profile
    def forward(self, input):
        return self.forward2(input,input)
    
    #@profile
    def forward2(self, left, right):
        torch.cuda.synchronize()
        starter_ft= time.perf_counter()
        disp = (self.maxdisp + 1) // pow(2, self.k)
        #Feature Extraction
        starter_fs = time.perf_counter()
        
        refimg_feature = self.feature_extraction(left)
        targetimg_feature = self.feature_extraction(right)
        torch.cuda.synchronize()
        ender_fs = time.perf_counter()
        curr_time = ender_fs-starter_fs
        timings_components[comp_names["fe"]][rep] = curr_time

        # cost volume
        starter_cv= time.perf_counter()
        
        cost = torch.cuda.FloatTensor(refimg_feature.size()[0], refimg_feature.size()[1], 
            disp,refimg_feature.size()[2],refimg_feature.size()[3]).zero_()
        for i in range(disp):
            if i > 0:
                cost[:, :, i, :, i:] = refimg_feature[ :, :, :, i:] - targetimg_feature[:, :, :, :-i]
            else:
                cost[:, :, i, :, :] = refimg_feature - targetimg_feature
        cost = cost.contiguous()   #torch.Size([1, 32, 24, 46, 154]) [batchSize,Ch, D, H ,W]
        
        torch.cuda.synchronize()
        ender_cv= time.perf_counter()
        curr_time = ender_cv-starter_cv
        timings_components[comp_names["cv"]][rep]= curr_time

        #Volume filtering
        starter_cf= time.perf_counter()
        cost = self.filter(cost)
        #print("cost filter before squeeze shape is: ", cost.size())
        cost = torch.squeeze(cost, 1)
        #print("cost filter shape is: ", cost.size())
        #exit()
        #print("cost shape is: ",cost.shape)
        pred =soft_argmin(cost)
        #pred = F.softmax(cost, dim=1) #[1,24,68,120]
        
        torch.cuda.synchronize()
        ender_cf= time.perf_counter()
        curr_time = ender_cf - starter_cf
        timings_components[comp_names["cf"]][rep]= curr_time
        
        starter_st0= time.perf_counter()
        
        pred = disparityregression(disp)(pred) # [1,68,120]
        
        torch.cuda.synchronize()
        ender_st0= time.perf_counter()
        curr_time = ender_st0-starter_st0
        timings_components[comp_names["st0"]][rep]= curr_time
        #print(left.shape)
        #exit()

        #img_pyramid_list = [left]
        img_pyramid_list = []
        for i in range(self.r):
            img_pyramid_list.append( F.interpolate(left,scale_factor=1/pow(2,i), mode='bilinear', align_corners=False))
            
        img_pyramid_list.reverse() # [1,3,135,240] [1,3,270,480] [1,3,540,960]
        pred_pyramid_list= [pred] # [1,68,120] unrefined
        #torch.cuda.synchronize()
        for i in range(self.r):
            starter_st1= time.perf_counter()
            pred_pyramid_list.append(self.edge_aware_refinements[i](pred_pyramid_list[i], img_pyramid_list[i]))
            ender_st1= time.perf_counter()
            curr_time = ender_st1-starter_st1
            if i==0:
                timings_components[comp_names["st1"]][rep]= curr_time
            if i==1:
                timings_components[comp_names["st2"]][rep]= curr_time
            if i==2:
                timings_components[comp_names["st3"]][rep]= curr_time

        length_all = len(pred_pyramid_list)
        diff=0.0
        for i in range(length_all): # my change from 1 to 4
            #start_pyr= time.perf_counter()
            pred_pyramid_list[i] = pred_pyramid_list[i] * 8
            pred_pyramid_list[i] = torch.squeeze(
            F.interpolate(torch.unsqueeze(pred_pyramid_list[i], dim=1),size=left.size()[-2:], mode='bilinear', align_corners=False),dim=1)
            #end_pyr= time.perf_counter()
            #diff +=(end_pyr-start_pyr)
        #torch.cuda.synchronize()
        ender_ft= time.perf_counter()
        curr_time = ender_ft - starter_ft
        timings_components[comp_names["ft"]][rep]= curr_time
        return pred_pyramid_list




