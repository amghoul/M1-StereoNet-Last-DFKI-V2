from __future__ import print_function
import torch
import torch.nn.parallel
import torch.utils.data
import copy
from qtorch import FixedPoint, FloatingPoint
from qtorch.auto_low import sequential_lower, lower
from .FinalQuant import *

def save_best_lossess(args,return_avg_losses,return_sum_stages_losses,minLoss,epoch,savefilename):
    if return_sum_stages_losses < minLoss['sum_losses_stages']:
        minLoss['loss0'] = return_avg_losses[0]
        if args.stages >=2:
            minLoss['loss1'] = return_avg_losses[1]
        if args.stages >=3:
            minLoss['loss2'] = return_avg_losses[2]
        if args.stages ==4:
            minLoss['loss3'] = return_avg_losses[3]    
        minLoss['checkpointPath'] = savefilename
        minLoss['epoch'] = epoch
        minLoss['sum_losses_stages'] = return_sum_stages_losses
    return minLoss

def save_chckpoint(args, model,return_avg_losses,epoch,optimizer,scheduler,minLoss,savefilename):
    torch.save({
        'state_dict': model.state_dict(),
        'avg_train_loss_stage': return_avg_losses,
        'epoch': epoch,
        'optimizer_state_dict': optimizer.state_dict(),
        'sheduler' : scheduler.state_dict(),
        'current_learning_rate':optimizer.param_groups[0]['lr'],
        'best_losses_stages': minLoss,
        'saved_args': args
        }, savefilename)

def load_checkpoint(checkpoint_file,model,optimizer,scheduler):
    checkpoint = torch.load(checkpoint_file)
    model.load_state_dict(checkpoint['state_dict'])
    avg_train_loss_stage = checkpoint['avg_train_loss_stage']
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['sheduler'])
    epoch_start = checkpoint['epoch']+1
    current_lr=checkpoint['current_learning_rate']
    minLoss=checkpoint['best_losses_stages']
    saved_args=checkpoint['saved_args']
    return model, avg_train_loss_stage, optimizer, scheduler, epoch_start, current_lr,minLoss,saved_args

def load_quantized_model(args,model):
        forward_num = FixedPoint(wl=args.quantWL, fl=args.quantFL, clamp=True, symmetric=False)
        backward_num = FloatingPoint(exp=4, man=4)
        #layerTypes=['conv','linear','pool','pad','activation','normalization','dropout','loss']
        layerTypes=['activation','loss']
        model = sequential_lower(model, layer_types=layerTypes, forward_number=forward_num)#, backward_number=backward_num)
        float_model_dict = copy.deepcopy(model.state_dict())
        quant_model_dict = copy.deepcopy(model.state_dict())
        quant_model_dict=quantize_model(quant_model_dict,args.quantWL,args.quantWL -args.quantFL) # loop for quantizing weight parameters
        model.load_state_dict(quant_model_dict)
        return model, float_model_dict

def save_losses(optimizer,path_file_losses, epoch,stages,losses_All_Stages,test_train_losses,test_train_EPEs,losses_sum_All_Stages,
    test_train_sum_stages_losses,test_train_outliers_sumary,epoch_train_end_time,epoch_train_start_time,epoch_test_end_time,epoch_test_start_time,scheduler):
    with open(path_file_losses, 'a') as f:
        f.write("%d:" % (epoch))
        for i in range(4):
            for x in range(stages):
                if i ==0:
                    f.write("{0:.3f}:".format(losses_All_Stages[epoch][x]))
                if i ==1:
                    f.write("{0:.3f}:".format(test_train_losses[epoch][0][x]))
                if i ==2:
                    f.write("{0:.3f}:".format(test_train_EPEs[epoch][0][x]))
                if i ==3:
                    f.write("{0:.3f}:".format(test_train_outliers_sumary[epoch][0][x]))## return only the D1-allw
        f.write("{0:.3f}:".format(losses_sum_All_Stages[epoch]))
        f.write("{0:.3f}:".format(test_train_sum_stages_losses[epoch][0]))## return only the D1-allw
        f.write("{0:.3f}:".format(epoch_train_end_time-epoch_train_start_time))
        f.write("{0:.3f}:".format(epoch_test_end_time-epoch_test_start_time))
        f.write("{0:.10f}".format(optimizer.param_groups[0]['lr']))
        f.write("\n")
        f.close()

def save_GL(path_file_GL,avgTrEr,minTrErr,avgValErr,minValErr,GL_Tr,GL_Val):
    with open(path_file_GL, 'a') as f:
        f.write("{0:.3f}:".format(avgTrEr))
        f.write("{0:.3f}:".format(minTrErr))
        f.write("{0:.3f}:".format(avgValErr))
        f.write("{0:.3f}:".format(minValErr))
        f.write("{0:.3f}:".format(GL_Tr))
        f.write("{0:.3f}".format(GL_Val))
        f.write("\n")
        f.close()

def load_dataset(args):
    if args.dataset == "kitti":
        if args.datatype == '2015':
            from dataloader import KITTIloader2015 as ls
            from dataloader import KITTILoader as DA
            train_left_img, train_right_img, train_left_disp,train_left_disp_noc, test_left_img, test_right_img, test_left_disp,test_left_disp_noc,train_mask_obj_map,test_mask_obj_map = ls.dataloader(
            args.datapath)
            
        else:# args.datatype == '2012':
            from dataloader import KITTIloader2012 as ls
            from dataloader import KITTILoader1 as DA
            train_left_img, train_right_img, train_left_disp,train_left_disp_noc, test_left_img, test_right_img, test_left_disp,test_left_disp_noc = ls.dataloader(
            args.datapath)
            
    else: ##sceneflow dataset
        from dataloader import listflowfile as lt  ## change import fie for scenflow dataset
        #from dataloader import SecenFlowLoader1 as DA
        from dataloader import SecenFlowLoaderMy as DA
        
        train_left_img, train_right_img, train_left_disp, train_left_disp_occ ,test_left_img, test_right_img, test_left_disp, test_left_disp_occ= lt.dataloader(
            args.datapath,data_range_train= 21818,data_range_Val=4248) #200 4248
        '''
        train_left_img, train_right_img, train_left_disp, train_left_disp_occ ,test_left_img, test_right_img, test_left_disp, test_left_disp_occ = lt.dataloader(
            args.datapath,data_range_train= 7,data_range_Val=3) #200 4248
        '''
    train_left_img.sort()
    train_right_img.sort()
    train_left_disp.sort()
    if args.dataset == "kitti":
        train_left_disp_noc.sort()
        if args.datatype == '2015': 
            train_mask_obj_map.sort()
    else:
        train_left_disp_occ.sort()
    
    test_left_img.sort()
    test_right_img.sort()
    test_left_disp.sort()
    if args.dataset == "kitti":
        test_left_disp_noc.sort()
        if args.datatype == '2015': 
            test_mask_obj_map.sort()
    else:
        test_left_disp_occ.sort()
    
    #__normalize = {'mean': [0.0, 0.0, 0.0], 'std': [1.0, 1.0, 1.0]}
    __normalize = {'mean': [0.5, 0.5, 0.5], 'std': [0.5, 0.5, 0.5]}
    if args.dataset == "kitti":
        if args.datatype == '2015':
            TrainImgLoader = torch.utils.data.DataLoader(
                DA.myImageFloder(train_left_img, train_right_img, train_left_disp,train_left_disp_noc,train_mask_obj_map, True,args.flip_vertical),
                batch_size=args.train_bsize, shuffle=True, num_workers=12, drop_last=False) ## org shuffle = False
            
            #else: #mode= test
            TestImgLoader = torch.utils.data.DataLoader(
                DA.myImageFloder(test_left_img, test_right_img, test_left_disp,test_left_disp_noc,test_mask_obj_map, False,args.flip_vertical),
                batch_size=args.test_bsize, shuffle=False, num_workers=4, drop_last=False)
        else: #args.datatype == '2012':
            TrainImgLoader = torch.utils.data.DataLoader(
                DA.myImageFloder(train_left_img, train_right_img, train_left_disp,train_left_disp_noc,True,args.flip_vertical),
                batch_size=args.train_bsize, shuffle=True, num_workers=12, drop_last=False) ## org shuffle = False
            #else: #mode= test
            TestImgLoader = torch.utils.data.DataLoader(
                DA.myImageFloder(test_left_img, test_right_img, test_left_disp,test_left_disp_noc,False,args.flip_vertical),
                batch_size=args.test_bsize, shuffle=False, num_workers=4, drop_last=False)

    else: # args.dataset== "sceneflow"
        #if args.mode == "train":
        TrainImgLoader = torch.utils.data.DataLoader(
            DA.myImageFloder(train_left_img, train_right_img, train_left_disp,train_left_disp_occ, True, normalize=__normalize),
            batch_size=args.train_bsize, shuffle=True, num_workers=12, drop_last=False)
        #else: #mode= test
        TestImgLoader = torch.utils.data.DataLoader(
            DA.myImageFloder(test_left_img, test_right_img, test_left_disp,test_left_disp_occ, False, normalize=__normalize),
            batch_size=args.test_bsize, shuffle=False, num_workers=4, drop_last=False)#4
    return TrainImgLoader,TestImgLoader

