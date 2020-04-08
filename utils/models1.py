import math
import numpy as np
from os.path import join, dirname, isfile
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet34
import tqdm
from .miscellaneous import outlier1d

DEBUG = 0

if DEBUG>0:
    import matplotlib.pyplot as plt
    import sys
    import time

class QuantizationLayer(nn.Module):
    def __init__(self, dim, device):
        nn.Module.__init__(self)
        self.dim = dim
        self.mode = 0 # 0:Training 1:Validation 
        self.device = device
        self.startIdx = 4
        self.endBias = 10
        self.blurFilterKernel = torch.tensor([[[
                                    .03125, .03125, .01562, .03125, .03125,
                                    .03125, .04406, .03125, .04406, .03125,
                                    .04250, .06250, .14502, .06250, .04250,
                                    .03125, .04406, .03125, .04406, .03125,
                                    .03125, .03125, .01562, .03125, .03125
                                ]]])
        # self.blurFilterKernel = torch.tensor([[[
        #                             .02000, .02500, .05000, .02500, .02000,
        #                             .02500, .03000, .10000, .03000, .02500,
        #                             .01000, .04000, .20000, .04000, .01000,
        #                             .02500, .03000, .10000, .03000, .02500,
        #                             .02000, .02500, .05000, .02500, .02000
        #                         ]]])
        assert math.isclose( self.blurFilterKernel.sum(), 1., rel_tol=1e-05), 'blurFilterKernel value error'
        self.blurFilterKernelSize = int( math.sqrt( len(self.blurFilterKernel[0,0])))

    def setMode(self, mode):
        self.mode = mode

    def forward(self, events):
        # points is a list, since events can have any size
        B = len(events)
        # separate events into S segments with evently-divided number of events
        S = 48
        # obtain the height & width
        H, W = self.dim

        # obtain class instance variables
        device = self.device
        blurFiltSize = self.blurFilterKernelSize
        blurFiltPad = blurFiltSize//2
        blurFilt = self.blurFilterKernel.expand(1, B, -1)
        blurFilt = blurFilt.view(B, 1, blurFiltSize, blurFiltSize).to(device)
        sIdx = self.startIdx
        eIdx = S - self.endBias

        num_alongX = int( S * W * B)
        num_alongY = int( S * H * B)
        alongX = torch.zeros(num_alongX, dtype=torch.int32, device=device)
        alongY = torch.zeros(num_alongY, dtype=torch.int32, device=device)
        segmentLen_batch = []
        for bi in range(B):
            events[bi] = torch.from_numpy(events[bi]).to(device).squeeze(0)
            x, y, t, p, b = events[bi].t()
            segmentLen = len(x)//S
            segmentLen_batch.append(segmentLen)
            chunks = torch.arange(S, dtype=torch.int32, device=device)
            chunks = chunks.unsqueeze(-1).expand(-1, segmentLen).reshape(-1)
            resultLen = len(chunks)
            ix = x[:resultLen] + W*chunks + W*S*bi
            iy = y[:resultLen] + H*chunks + H*S*bi
            ones = torch.ones(resultLen, dtype=torch.int32, device=device)
            alongX.put_(ix.long(), ones, accumulate=True)
            alongY.put_(iy.long(), ones, accumulate=True)

        segmentLen_batch = torch.FloatTensor(segmentLen_batch).to(device)

        alongX = alongX.view(1, -1, S, W).float()
        alongX = F.conv2d(alongX, blurFilt, padding=blurFiltPad, groups=B)
        alongX = alongX.squeeze()
        alongDim = torch.arange(W, dtype=torch.float32, device=device)
        alongDim = alongDim.expand(B, -1).unsqueeze(-1)
        meanX = torch.bmm(alongX, alongDim).squeeze(-1) / segmentLen_batch.view(-1,1)
        start_seg_x = meanX[:, sIdx].unsqueeze(-1)
        start_seg_x_dis_to_center = W//2 - start_seg_x
        # align and centralize the image along x-axis
        alignedX = meanX - start_seg_x - start_seg_x_dis_to_center
        alignedX = alignedX.round().int()

        alongY = alongY.view(1, -1, S, H).float()
        alongY = F.conv2d(alongY, blurFilt, padding=blurFiltPad, groups=B)
        alongY = alongY.squeeze()
        alongDim = torch.arange(H, dtype=torch.float32, device=device)
        alongDim = alongDim.expand(B, -1).unsqueeze(-1)
        meanY = torch.bmm(alongY, alongDim).squeeze(-1) / segmentLen_batch.view(-1,1)
        start_seg_y = meanY[:, sIdx].unsqueeze(-1)
        start_seg_y_dis_to_center = H//2 - start_seg_y
        # align and centralize the image along y-axis
        alignedY = meanY - start_seg_y - start_seg_y_dis_to_center
        alignedY = alignedY.round().int()

        meanX = meanX.cpu().numpy()
        meanY = meanY.cpu().numpy()
        container_batch = []
        for bi in range(B):
            segmentLen = int(segmentLen_batch[bi].item())
            usableEventsLen = segmentLen*S
            
            x, y, t, p, b = events[bi].t()
            x = x.int()
            y = y.int()
            shiftedX = alignedX[bi].unsqueeze(-1).expand(-1, segmentLen).reshape(-1)
            x = x[:usableEventsLen]
            x -= shiftedX
            x = torch.clamp(x, 0, W-1)
            shiftedY = alignedY[bi].unsqueeze(-1).expand(-1, segmentLen).reshape(-1)
            y = y[:usableEventsLen]
            y -= shiftedY
            y = torch.clamp(y, 0, H-1)
            
            idx_in_container = x + W*y
            idx_in_container = idx_in_container.long()
            idx_in_container = torch.chunk(idx_in_container, S)
            ones = torch.ones(segmentLen, dtype=torch.int32, device=device)
            onesBool = torch.ones(segmentLen, dtype=torch.bool, device=device)
            container = torch.zeros(W*H, dtype=torch.int32, device=device)
            container.put_(idx_in_container[sIdx], ones, accumulate=True)
            verifier_old = []
            verifier_old.append( torch.zeros_like(container, dtype=torch.bool, device=device))
            verifier_old[0].put_(idx_in_container[sIdx], onesBool, accumulate=False)
            verifier_old.append( torch.ones_like(verifier_old[0]))
            confidentPixs = torch.zeros_like(container)
            for si in range(sIdx+1, eIdx):
                isXoutlier = outlier1d(meanX[bi, si:si+10], thresh=3)[0]
                isYoutlier = outlier1d(meanY[bi, si:si+10], thresh=3)[0]
                if isXoutlier or isYoutlier:
                    continue
                verifier_new = torch.zeros_like(verifier_old[0])
                verifier_new.put_(idx_in_container[si], onesBool, accumulate=False)
                confidentPixs += verifier_new & verifier_old[0] & verifier_old[1]
                verifier_new = verifier_new | verifier_old[0]
                verifier_new_cnt = verifier_new.sum().float()
                new_info_cnt = torch.logical_xor(verifier_new, verifier_old[0]).sum().float()
                if new_info_cnt/verifier_new_cnt < 0.01:
                    if DEBUG>=8:
                        print("%d: %d" % (bi, si))
                    break
                container.put_(idx_in_container[si], ones, accumulate=True)
                verifier_old[1] = verifier_old[0]
                verifier_old[0] = verifier_new
            
            container = container.float()
            mean = container.mean()
            std = container.std()
            clampVal = mean + 3*std
            container = torch.clamp(container, 0, clampVal)
            container /= clampVal
            container_batch.append(container)
            
            confidentPixs = confidentPixs.float()
            mean = confidentPixs.mean()
            std = confidentPixs.std()
            clampVal = mean + 3*std
            confidentPixs = torch.clamp(confidentPixs, 0, clampVal)
            confidentPixs /= clampVal
            container_batch.append(confidentPixs)
        
        containers = torch.stack(container_batch).view(-1, 2, H, W)

        if DEBUG==9:
            container_img = containers.cpu().numpy()
            for bi in range(B):
                fig = plt.figure(figsize=(15,15))
                ax = []
                rows = 4
                columns = 1
                for i in range(rows * columns):
                    ax.append( fig.add_subplot(rows, columns, i+1))
                    if i==0:
                        ax[-1].set_title('x-t graph')
                        plt.imshow(alongX[bi], cmap='gray')
                    elif i==1:
                        ax[-1].set_title('y-t graph')
                        plt.imshow(alongY[bi], cmap='gray')
                    elif i==2:
                        ax[-1].set_title('x-y graph')
                        plt.imshow(container_img[bi][0], cmap='gray', vmin=0, vmax=1)
                    elif i==3:
                        ax[-1].set_title('x-y graph 2')
                        plt.imshow(container_img[bi][1], cmap='gray', vmin=0, vmax=1)
                plt.tight_layout()
                plt.show()
            sys.exit(0)

        vox = containers

        return vox

class Classifier(nn.Module):
    def __init__(self,
                 voxel_dimension=(180,240),
                 crop_dimension=(224, 224),  # dimension of crop before it goes into classifier
                 num_classes=101,
                 pretrained=True,
                 device='cpu'):

        nn.Module.__init__(self)
        self.mode = 0   # 0:Training 1:Validation 
        self.quantization_layer = QuantizationLayer(voxel_dimension, device)
        self.crop_dimension = crop_dimension
        self.num_classes = num_classes
        self.in_channels = 2

        classifierSelect = 'resnet34'
        if classifierSelect == 'resnet34':
            self.modelChildren = ['avgpool', 'layer4', 'layer3', 'layer2', 'layer1', 'maxpool', 'relu', 'bn1']
            classifierSelect_sub = ''
            if classifierSelect_sub == 'wide':
                self.classifier = torch.hub.load('pytorch/vision:v0.5.0', 'wide_resnet50_2', pretrained=pretrained)
            else:
                self.classifier = resnet34(pretrained=pretrained)
            # replace fc layer and first convolutional layer
            self.classifier.conv1 = nn.Conv2d(self.in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.classifier.fc = nn.Linear(self.classifier.fc.in_features, num_classes)
        elif classifierSelect == 'densenet201':
            self.classifier = torch.hub.load('pytorch/vision:v0.5.0', 'densenet201', pretrained=pretrained)
            # replace fc layer and first convolutional layer
            self.classifier.features.conv0 = nn.Conv2d(self.in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.classifier.classifier = nn.Linear(self.classifier.classifier.in_features, num_classes)

    def setMode(self, mode):
        self.mode = mode
        self.quantization_layer.setMode(mode)

    def freezeUnfreeze(self):
        unfreeze_list = self.modelChildren[:1] + ['conv1', 'fc']
        print(unfreeze_list)
        try:
            self.modelChildren.pop(0)
            for name, child in self.classifier.named_children():
                if name in unfreeze_list:
                    print(name + ' is unfrozen | ', end='')
                    for param in child.parameters():
                        param.requires_grad = True
                else:
                    print(name + ' is frozen | ', end='')
                    for param in child.parameters():
                        param.requires_grad = False
        except:
            for name, child in self.classifier.named_children():
                print(name + ' is unfrozen | ', end='')
                for param in child.parameters():
                    param.requires_grad = True
        print()

    def crop_and_resize_to_resolution(self, x, output_resolution=(224, 224)):
        B, C, H, W = x.shape
        if H > W:
            h = H // 2
            x = x[:, :, h - W // 2:h + W // 2, :]
        else:
            h = W // 2
            x = x[:, :, :, h - H // 2:h + H // 2]
        x = F.interpolate(x, size=output_resolution)
        return x

    def forward(self, x):
        vox = self.quantization_layer.forward(x)
        vox_cropped = self.crop_and_resize_to_resolution(vox, self.crop_dimension)
        pred = self.classifier.forward(vox_cropped)
        return pred, vox
