import torch
import torch.nn.functional as F
import cv2 as cv
import torch.nn as nn

def GERF_loss(pred, GT, args):
    # mask = (GT < args.maxdisp) & (GT >= 0)
    if args.dataset == "kitti":
        mask = (GT < args.maxdisp) & (GT > 0)
    else:
        GT = -1 * GT
        mask = (GT < args.maxdisp) & (GT > 0)
        #mask = GT < args.maxdisp
    mask.detach_()
    #print(mask.size(), GT.size(), pred.size())
    count = len(torch.nonzero(mask))
    if count == 0:
        count = 1
    return torch.sum(torch.sqrt(torch.pow(pred[mask] - GT[mask], 2) + 4) /2 - 1) / count

def GERF_loss_mask(pred, GT,mask, args):
    mask.detach_()
    count = len(torch.nonzero(mask))
    if count == 0:
        count = 1
    return torch.sum(torch.sqrt(torch.pow(pred[mask] - GT[mask], 2) + 4) /2 - 1) / count

def smooth_L1_loss(pred, GT, args):
    if args.dataset == "kitti":
        mask = (GT < args.maxdisp) & (GT > 0)
    else:
        GT = -1 * GT
        mask = (GT < args.maxdisp) & (GT > 0)
        #mask = GT < args.maxdisp
    mask.detach_()
    loss = F.smooth_l1_loss(pred[mask], GT[mask], size_average=True)
    #loss = (pred[mask] - GT[mask]).abs().mean()
    return loss

def smooth_L1_loss_mask(pred, GT,mask, args):
    mask.detach_()
    loss = F.smooth_l1_loss(pred[mask], GT[mask], size_average=True)
    #loss = (pred[mask] - GT[mask]).abs().mean()
    return loss

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        # xavier(m.weight.data)
        m.weight.data.normal_(0, 0.01)
        if m.weight.data.shape == torch.Size([1, 5, 1, 1]):
            # for new_score_weight
            torch.nn.init.constant_(m.weight, 0.2)
        if m.bias is not None:
            m.bias.data.zero_()
    if isinstance(m, nn.Conv3d):
        # xavier(m.weight.data)
        m.weight.data.normal_(0, 0.01)
        if m.weight.data.shape == torch.Size([1, 5, 1, 1]):
            # for new_score_weight
            torch.nn.init.constant_(m.weight, 0.2)
        if m.bias is not None:
            m.bias.data.zero_()

class AverageMeter(object):
    """Compute and stores the average and current value"""

    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val= 0
        self.avg= 0
        self.sum= 0
        self.count= 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
def Outliers(mask,pred, gt,ABS_THRESH=3.0,REL_THRESH=0.05,use_REL_THRESH=True):
    gt = gt #*255.0
    Outlier=[]
    #print(pred.min(), pred.max(), pred.dtype)
    #print(gt.min(), gt.max(), gt.dtype)
    #ABS_THRESH = 3.0
    #REL_THRESH = 0.05
    #mask = (gt > 0)
    if use_REL_THRESH:
        Outlier=((abs(gt[mask]-pred[mask]) > ABS_THRESH) * (abs(gt[mask]-pred[mask])/abs(gt[mask]) > REL_THRESH)).sum().float() / mask.sum()
    else:
        Outlier=((abs(gt[mask]-pred[mask]) > ABS_THRESH) ).sum().float() / mask.sum()
    return Outlier

if __name__ == '__main__':

    # import matplotlib.pyplot as plt
    # image = cv.imread('/media/lxy/sdd1/ActiveStereoNet/StereoNet_pytorch/results/forvideo/iter-122.jpg')

    im_gray = cv.imread('/media/lxy/sdd1/ActiveStereoNet/StereoNet_pytorch/results/forvideo/iter-133.jpg', cv.IMREAD_GRAYSCALE)
    # print(im_gray.shape)
    im_color = cv.applyColorMap(im_gray*2, cv.COLORMAP_JET)
    # cv.imshow('test', im_color)
    # cv.waitKey(0)
    cv.imwrite('test.png',im_color)
    # print(image.shape)
    # plt.figure('Image')
    # sc =plt.imshow(image)
    # sc.set_cmap('hsv')
    # plt.colorbar()
    # plt.axis('off')
    # plt.show()
    # print('end')
    # image[:,:,0].save('/media/lxy/sdd1/ActiveStereoNet/StereoNet_pytorch/results/pretrained_StereoNet_single/it1er-151.jpg')
