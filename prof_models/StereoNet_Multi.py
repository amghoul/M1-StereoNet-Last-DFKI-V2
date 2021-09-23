import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.backends.cudnn as cudnn
###############
import time
###############

from pytorch_memlab import LineProfiler,profile, set_target_gpu, MemReporter
#torch.backends.cudnn.benchmark = False  

from memory import log_mem, log_mem_amp, log_mem_amp_cp, log_mem_cp
from plot import plot_mem, pp
import pandas as pd

##############
cuda1 = torch.device('cuda')
repetitions=300

comp_names={"fe":0,"cv":1,"cf":2,"st0":3,"st1":4,"st2":5,"st3":6,"ft":7}
com_fullNames=["Feature_Extraction", "Cost_Volume", "Cost_Filtering","Stage0","Stage1","Stage2","Stage3","Full_time"]
num_components = len(com_fullNames)
timings_components=[]
for i in range(num_components):
    timings_components.append(0)
#print(len(timings_components[0]))
##############

def convbn(in_channel, out_channel, kernel_size, stride, pad, dilation):
    
    return nn.Sequential(
        nn.Conv2d(
            in_channel,
            out_channel,
            kernel_size=kernel_size,
            stride=stride,
            padding=dilation if dilation>1 else pad,
            dilation=dilation),
       nn.BatchNorm2d(out_channel))

def convbn_3d(in_channel, out_channel, kernel_size, stride, pad):

    return nn.Sequential(
        nn.Conv3d(
            in_channel,
            out_channel,
            kernel_size=kernel_size,
            padding=pad,
            stride=stride),
       nn.BatchNorm3d(out_channel))

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
    def __init__(self, in_channel, out_channel, stride, downsample, pad, dilation):
        super().__init__()
        self.conv1 = nn.Sequential(
            convbn(in_channel, out_channel, 3, stride, pad, dilation),
            nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.conv2 = convbn(out_channel, out_channel, 3, 1, pad, dilation)
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
    def __init__(self, k):
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
                    32, 32, stride=1, downsample=None, pad=1, dilation=1))
        self.conv_alone = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
    def forward(self, rgb_img):
        output = rgb_img
        for i in range(self.k):
            output = self.downsample[i](output)
        for block in self.residual_blocks:
            output = block(output)
        return self.conv_alone(output)

class EdgeAwareRefinement(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.conv2d_feature = nn.Sequential(
            convbn(in_channel, 32, kernel_size=3, stride=1, pad=1, dilation=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.residual_astrous_blocks = nn.ModuleList()
        astrous_list = [1, 2, 4, 8 , 1 , 1]
        for di in astrous_list:
            self.residual_astrous_blocks.append(
                BasicBlock(
                    32, 32, stride=1, downsample=None, pad=1, dilation=di))
                
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
        #print(out)
        #print(out.shape)
        #exit()
        return out


class StereoNet(nn.Module):
    def __init__(self, k, r, maxdisp=192):
        super().__init__()
        self.maxdisp = maxdisp
        self.k = k
        self.r = r
        self.feature_extraction = FeatureExtraction(k)
        self.filter = nn.ModuleList()
        for _ in range(4):
            self.filter.append(
                nn.Sequential(
                    convbn_3d(32, 32, kernel_size=3, stride=1, pad=1),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True)))
        self.conv3d_alone = nn.Conv3d(
            32, 1, kernel_size=3, stride=1, padding=1)
        
        self.edge_aware_refinements = nn.ModuleList()
        for _ in range(self.r): ### my change from 1 to 4
            self.edge_aware_refinements.append(EdgeAwareRefinement(4))
    def forward(self, left, right):
        disp = (self.maxdisp + 1) // pow(2, self.k)
        refimg_feature = self.feature_extraction(left)
        targetimg_feature = self.feature_extraction(right)

        # matching
        cost = torch.cuda.FloatTensor(refimg_feature.size()[0],
                                 refimg_feature.size()[1],
                                 disp,
                                 refimg_feature.size()[2],
                                 refimg_feature.size()[3]).zero_()
        for i in range(disp):
            if i > 0:
                cost[:, :, i, :, i:] = refimg_feature[ :, :, :, i:] - targetimg_feature[:, :, :, :-i]
            else:
                cost[:, :, i, :, :] = refimg_feature - targetimg_feature
        cost = cost.contiguous()
        for f in self.filter:
            cost = f(cost)
        cost = self.conv3d_alone(cost)
        cost = torch.squeeze(cost, 1)
        pred =soft_argmin(cost)
        #pred = F.softmax(cost, dim=1) #[1,24,68,120]
        pred = disparityregression(disp)(pred) # [1,68,120]

        #img_pyramid_list = [left]
        img_pyramid_list = []
        for i in range(self.r):
            img_pyramid_list.append(
                F.interpolate(
                left,
                scale_factor=1/pow(2,i),
                mode='bilinear',
                align_corners=False))
        
        img_pyramid_list.reverse() # [1,3,135,240] [1,3,270,480] [1,3,540,960]
        pred_pyramid_list= [pred] # [1,68,120] unrefined
        for i in range(self.r):
            pred_pyramid_list.append(self.edge_aware_refinements[i](
                    pred_pyramid_list[i], img_pyramid_list[i]))

        length_all = len(pred_pyramid_list)
        
        for i in range(length_all): # my change from 1 to 4
            pred_pyramid_list[i] = pred_pyramid_list[i]* (
                left.size()[-1] / pred_pyramid_list[i].size()[-1])
            
            pred_pyramid_list[i] = torch.squeeze(
            F.interpolate(
                torch.unsqueeze(pred_pyramid_list[i], dim=1),
                size=left.size()[-2:],
                mode='bilinear',
                align_corners=False),
            dim=1)    
        return pred_pyramid_list
#########################for profiling
class StereoNetProf(nn.Module):
    def __init__(self, k, r, maxdisp=192):
        super().__init__()
        self.maxdisp = maxdisp
        self.k = k
        self.r = r
        self.feature_extraction = FeatureExtraction(k)
        self.filter = nn.ModuleList()
        for _ in range(4):
            self.filter.append(
                nn.Sequential(
                    convbn_3d(32, 32, kernel_size=3, stride=1, pad=1),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True)))
        self.conv3d_alone = nn.Conv3d(
            32, 1, kernel_size=3, stride=1, padding=1)
        
        self.edge_aware_refinements = nn.ModuleList()
        for _ in range(self.r): ### my change from 1 to 4
            self.edge_aware_refinements.append(EdgeAwareRefinement(4))
    
    @profile
    def forward(self, left, right):
        disp = (self.maxdisp + 1) // pow(2, self.k)
        refimg_feature = self.feature_extraction(left)
        targetimg_feature = self.feature_extraction(right)

        # matching
        cost = torch.cuda.FloatTensor(refimg_feature.size()[0],
                                 refimg_feature.size()[1],
                                 disp,
                                 refimg_feature.size()[2],
                                 refimg_feature.size()[3]).zero_()
        for i in range(disp):
            if i > 0:
                cost[:, :, i, :, i:] = refimg_feature[ :, :, :, i:] - targetimg_feature[:, :, :, :-i]
            else:
                cost[:, :, i, :, :] = refimg_feature - targetimg_feature
        cost = cost.contiguous()
        for f in self.filter:
            cost = f(cost)
        cost = self.conv3d_alone(cost)
        cost = torch.squeeze(cost, 1)
        pred =soft_argmin(cost)
        #pred = F.softmax(cost, dim=1) #[1,24,68,120]
        pred = disparityregression(disp)(pred) # [1,68,120]

        #img_pyramid_list = [left]
        img_pyramid_list = []
        for i in range(self.r):
            img_pyramid_list.append(
                F.interpolate(
                left,
                scale_factor=1/pow(2,i),
                mode='bilinear',
                align_corners=False))
        
        img_pyramid_list.reverse() # [1,3,135,240] [1,3,270,480] [1,3,540,960]
        pred_pyramid_list= [pred] # [1,68,120] unrefined
        pred_pyramid_list.append(self.edge_aware_refinements[0](pred_pyramid_list[0], img_pyramid_list[0]))
        pred_pyramid_list.append(self.edge_aware_refinements[1](pred_pyramid_list[1], img_pyramid_list[1]))
        pred_pyramid_list.append(self.edge_aware_refinements[2](pred_pyramid_list[2], img_pyramid_list[2]))    
        length_all = len(pred_pyramid_list)
        pred_pyramid_list[0] = pred_pyramid_list[0]* (left.size()[-1] / pred_pyramid_list[0].size()[-1])
        pred_pyramid_list[0] = torch.squeeze( F.interpolate(torch.unsqueeze(pred_pyramid_list[0], dim=1), size=left.size()[-2:], mode='bilinear', align_corners=False),dim=1)
        pred_pyramid_list[1] = pred_pyramid_list[1]* (left.size()[-1] / pred_pyramid_list[1].size()[-1])
        pred_pyramid_list[1] = torch.squeeze( F.interpolate(torch.unsqueeze(pred_pyramid_list[1], dim=1), size=left.size()[-2:], mode='bilinear', align_corners=False),dim=1)
        pred_pyramid_list[2] = pred_pyramid_list[2]* (left.size()[-1] / pred_pyramid_list[2].size()[-1])
        pred_pyramid_list[2] = torch.squeeze( F.interpolate(torch.unsqueeze(pred_pyramid_list[2], dim=1), size=left.size()[-2:], mode='bilinear', align_corners=False),dim=1)
        pred_pyramid_list[3] = pred_pyramid_list[3]* (left.size()[-1] / pred_pyramid_list[3].size()[-1])
        pred_pyramid_list[3] = torch.squeeze( F.interpolate(torch.unsqueeze(pred_pyramid_list[3], dim=1), size=left.size()[-2:], mode='bilinear', align_corners=False),dim=1)
        return pred_pyramid_list

############################Multi ok to change
class StereoNetTime(nn.Module):
    def __init__(self, k, r, maxdisp=192):
        super().__init__()
        self.maxdisp = maxdisp
        self.k = k
        self.r = r
        self.feature_extraction = FeatureExtraction(k)
        self.filter = nn.ModuleList()
        for _ in range(4):
            self.filter.append(nn.Sequential(convbn_3d(32, 32, kernel_size=3, stride=1, pad=1),nn.LeakyReLU(negative_slope=0.2, inplace=True)))        
        self.conv3d_alone = nn.Conv3d(32, 1, kernel_size=3, stride=1, padding=1)
        
        self.edge_aware_refinements = nn.ModuleList()
        for _ in range(self.r): ### my change from 1 to 4
            self.edge_aware_refinements.append(EdgeAwareRefinement(4))
    
    def forward(self, input):
        return self.forward2(input,input)

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
        timings_components[comp_names["fe"]] += curr_time

        # cost volume
        starter_cv= time.perf_counter()

        cost = torch.cuda.FloatTensor(refimg_feature.size()[0], refimg_feature.size()[1],disp,refimg_feature.size()[2],refimg_feature.size()[3]).zero_()
        for i in range(disp):
            if i > 0:
                cost[:, :, i, :, i:] = refimg_feature[ :, :, :, i:] - targetimg_feature[:, :, :, :-i]
            else:
                cost[:, :, i, :, :] = refimg_feature - targetimg_feature
        cost = cost.contiguous()
        
        torch.cuda.synchronize()
        ender_cv= time.perf_counter()
        curr_time = ender_cv-starter_cv
        timings_components[comp_names["cv"]]+= curr_time
        
        #Volume filtering
        starter_cf= time.perf_counter()
        for f in self.filter:
            cost = f(cost)
        cost = self.conv3d_alone(cost)
        cost = torch.squeeze(cost, 1)
        pred = F.softmax(cost, dim=1) #[1,24,68,120]
        pred =soft_argmin(cost)
        torch.cuda.synchronize()
        ender_cf= time.perf_counter()
        curr_time = ender_cf - starter_cf
        timings_components[comp_names["cf"]]+= curr_time
        
        starter_st0= time.perf_counter()
        
        pred = disparityregression(disp)(pred) # [1,68,120]

        torch.cuda.synchronize()
        ender_st0= time.perf_counter()
        curr_time = ender_st0-starter_st0
        timings_components[comp_names["st0"]]+= curr_time
        #img_pyramid_list = [left]
        img_pyramid_list = []
        for i in range(self.r):
            img_pyramid_list.append( F.interpolate(left,scale_factor=1/pow(2,i), mode='bilinear', align_corners=False))
        
        img_pyramid_list.reverse() # [1,3,135,240] [1,3,270,480] [1,3,540,960]
        pred_pyramid_list= [pred] # [1,68,120] unrefined
        for i in range(self.r):
            starter_st1= time.perf_counter()
            pred_pyramid_list.append(self.edge_aware_refinements[i](pred_pyramid_list[i], img_pyramid_list[i]))
            ender_st1= time.perf_counter()
            curr_time = ender_st1-starter_st1
            if i==0:
                timings_components[comp_names["st1"]]+= curr_time
            if i==1:
                timings_components[comp_names["st2"]]+= curr_time
            if i==2:
                timings_components[comp_names["st3"]]+= curr_time

        length_all = len(pred_pyramid_list)
        diff=0.0
        for i in range(length_all): # my change from 1 to 4
            #start_pyr= time.perf_counter()
            pred_pyramid_list[i] = pred_pyramid_list[i]* ( 1.0 * left.size()[-1] / pred_pyramid_list[i].size()[-1])
            pred_pyramid_list[i] = torch.squeeze(
            F.interpolate(torch.unsqueeze(pred_pyramid_list[i], dim=1),size=left.size()[-2:], mode='bilinear', align_corners=False),dim=1)    
            #end_pyr= time.perf_counter()
            #diff +=(end_pyr-start_pyr)
        #torch.cuda.synchronize()
        ender_ft= time.perf_counter()
        curr_time = ender_ft - starter_ft
        timings_components[comp_names["ft"]]+= curr_time
        return pred_pyramid_list

########Profoling 
# preparing required packages:

def profile_network(model, input_channels, height, width):
    # preparing input:
    
    input = torch.randn(1, input_channels, height, width).cuda(cuda1)
    input_size = (input_channels, height, width)
    # preparing model:
    model = model.cuda(cuda1)
    model = model.eval()
    # profiling:
    print("+++++++++++++++++++++++++++++")
    print("+++ INFO : PRINTING MODEL +++")
    print("+++++++++++++++++++++++++++++")
    print(model)
    print("+++++++++++++++++++++++++++++++")
    print("+++ INFO : PRINTING SUMMARY +++")
    print("+++++++++++++++++++++++++++++++")
    summary(model, input_size)
    print("+++++++++++++++++++++++++++++++++++++++++")
    print("+++ INFO : PRINTING MACS & PARAMETERS +++")
    print("+++++++++++++++++++++++++++++++++++++++++")
    macs, params = get_model_complexity_info(
        model, input_size, as_strings=True, print_per_layer_stat=True, verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    print("++++++++++++++++++++++++++++++++++++++")
    print("+++ INFO : PRINTING TIMING DETAILS +++")
    print("++++++++++++++++++++++++++++++++++++++")
    
    with torchprof.Profile(model, use_cuda=True ) as prof:
        model(input)
    print(prof.display(show_events=False))
    
    #exit()
    trace, event_lists_dict = prof.raw()
    print(len(trace))
    #print(event_lists_dict[trace[0].path][0])
    
    print("+++++++++++++++++++++++++++++++++++++++++++")
    print("+++ INFO : PRINTING PARAMETER PROFILING +++")
    print("+++++++++++++++++++++++++++++++++++++++++++")
    ts_params = Texttable()
    t_rows_params = [[]]
    for name, param in model.named_parameters():
        if param.requires_grad:
            t_rows_params.append([name, param.data.numel()])
    ts_params.add_rows(t_rows_params)
    ts_params.set_chars(["","","",""])
    ts_params.header(['Name', 'Total Number'])
    print(ts_params.draw())
    #######
    t = Texttable()
    t_rows = [[]]
    for name, param in model.named_parameters():
        if param.requires_grad:
            t_rows.append([name, param.data.numel(), list(param.data.shape), torch.min(
                param.data), torch.max(param.data), torch.mean(param.data), torch.std(param.data)])
    t.add_rows(t_rows)
    t.header(['Name', 'Total Number', 'Shape', 'Min', 'Max', 'Mean', 'Std'])
    print(t.draw())
    '''
    print("+++++++++++++++++++++++++++++++++++++++++++")
    print("+++ INFO : EXPORTING MODEL GRAPH AS PDF +++")
    print("+++++++++++++++++++++++++++++++++++++++++++")
    hl_graph = hl.build_graph(model, input)
    hl_graph.save("graph")
    '''

def test_profiling(model,dummy_input,device):
    B,C,H,W = dummy_input.size()
    for _ in range(10):
        _ = model(dummy_input)
        
    profile_network(model, C, H, W)


def prof_model(model,model_time,model_memlab,x, dummy_input,toProfile,modeltype):
    if toProfile == "noProfiling":
        print("######################## Printing th model ###########################")
        #model = StereoNet(k=3, r=x-1).to(device)
        print(model)
        out = model(dummy_input, dummy_input)
        torch.cuda.synchronize()
    if toProfile == "timeOnly":
        ############# for timing using perf_counter
        print("######################## measure timing ###########################")
        #model = StereoNetTime(k=3, r=x-1 ).to(device)
        #GPU-WARM-UP
        for _ in range(10):
            _ = model_time(dummy_input)
        
        for i in range(num_components):
            timings_components[i]=0.0

        with torch.no_grad():
            for rep in range(repetitions):
                _ = model_time(dummy_input)

        mean_tim_comp=[]
        for i in range(num_components):
            mean_tim_comp.append(0.0)
        
        for i in range(num_components):
            mean_tim_comp[i]= timings_components[i]/ repetitions
        
        for i in range(num_components):
            if i == 7:
                print("Other_Time",": ", "{:.3f}".format((mean_tim_comp[i]-sum(mean_tim_comp[:i]))*1000)," ms" , "perc: ", "{:.3f}".format((mean_tim_comp[i]-sum(mean_tim_comp[:i])) * 100/mean_tim_comp[7]),"%")
                print("Actual_Time",": ", "{:.3f}".format(sum(mean_tim_comp[:i]) * 1000)," ms" )
                print(com_fullNames[i],": ", "{:.3f}".format(mean_tim_comp[i]*1000), " ms")
            else:
                print(com_fullNames[i],": ", "{:.3f}".format(mean_tim_comp[i]*1000), " ms", "perc: ", "{:.3f}".format(mean_tim_comp[i] * 100/mean_tim_comp[7]),"%")
        torch.cuda.synchronize()
        
        ########################
    
    if toProfile == "logMem":
        ####################
        print("######################## using log_mem ###########################")
        #model = StereoNetTime(k=3, r=x-1 ).to(device)
        bs = 1
        input = dummy_input
        base_dir = '.'
        # %% Analysis baseline
        mem_log = []
        try:
            mem_log.extend(log_mem(model_time, input, exp='baseline'))
        except Exception as e:
            print(f'log_mem failed because of {e}')
        
        df = pd.DataFrame(mem_log)

        plot_mem(df, exps=['baseline'], output_file=f'{base_dir}/baseline_memory_plot_{modeltype}.png')
        #plot_mem(df, exps=['baseline'], output_file=f'{base_dir}/baseline_memory_plot_.png')
        pd.set_option("display.max_rows", None, "display.max_columns", None)
        print(df)
        torch.cuda.synchronize()
    
    if toProfile == "profilingAll":
        ###########################
        print("######################## Allprofiling ###########################")
        #model = StereoNetTime(k=3, r=x-1 ).to(device)
        test_profiling(model_time,dummy_input,device)
        torch.cuda.synchronize()
    
    if toProfile == "memlab":
        print("######################## memlab memory only###########################")
        #model = StereoNetProf(k=3, r=x-1).to(device)
        out = model_memlab(dummy_input, dummy_input)
        torch.cuda.synchronize()
    
if __name__ == '__main__':
    #CUDA_VISIBLE_DEVICES=0 python
    
    import torch
    from torchsummary import summary
    from ptflops.flops_counter import get_model_complexity_info
    import torchprof
    import hiddenlayer as hl
    from texttable import Texttable
    import time
    import datetime

    BN_1D=1
    BN_2D=0
    nosubspace=1
        
    #stages=[1,2,3,4]
    stages=[4]
    prof_list=["memlab"]
    #prof_list=["profilingAll"]
    #prof_list=["noProfiling","timeOnly","logMem","profilingAll","memlab"]

    device = torch.device('cuda')
    dummy_input = torch.randn(1, 3,540, 960,dtype=torch.float).to(device)
    
    print("The current used GPU is: ", torch.cuda.get_device_name(torch.cuda.current_device()))
    B,C,H,W = dummy_input.size()
    model_name = "org"
    
    for x in stages:
        modeltype = 'dmodel_'+model_name+"_stages_"+str(x)
        print("!!!!!!!! Profiling stage: " , str(x))
        model = StereoNet(k=3, r=x-1).to(device)
        model_time = StereoNetTime(k=3, r=x-1 ).to(device)
        model_memlab = StereoNetProf(k=3, r=x-1).to(device)
        for toProfile in prof_list:
            prof_model(model, model_time, model_memlab, x, dummy_input,toProfile,modeltype)
    

        

