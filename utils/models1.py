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
        self.segments = 48
        self.startIdx = 3
        self.endBias = 10
        self.blurFilterKernel = torch.tensor([[[
                            .04, .04, .04, .04, .04,
                            .04, .04, .04, .04, .04,
                            .04, .04, .04, .04, .04,
                            .04, .04, .04, .04, .04,
                            .04, .04, .04, .04, .04
                        ]]])
        # self.blurFilterKernel = torch.tensor([[[
        #                             .03125, .03125, .01562, .03125, .03125,
        #                             .03125, .04406, .03125, .04406, .03125,
        #                             .04250, .06250, .14502, .06250, .04250,
        #                             .03125, .04406, .03125, .04406, .03125,
        #                             .03125, .03125, .01562, .03125, .03125
        #                         ]]])
        # self.blurFilterKernel = torch.tensor([[[
        #                             .04000, .04000, .00000, .04000, .04000,
        #                             .04000, .08000, .00000, .08000, .04000,
        #                             .00000, .00000, .20000, .00000, .00000,
        #                             .04000, .08000, .00000, .08000, .04000,
        #                             .04000, .04000, .00000, .04000, .04000
        #                         ]]])
        assert math.isclose( self.blurFilterKernel.sum(), 1., rel_tol=1e-05), 'blurFilterKernel value error'
        self.blurFilterKernelSize = int( math.sqrt( len(self.blurFilterKernel[0,0])))

    def setMode(self, mode):
        self.mode = mode

    def forward(self, events):
        # points is a list, since events can have any size
        B = len(events)
        # separate events into S segments with evently-divided number of events
        S = self.segments
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

        alongX = alongX.view(-1, S*W).float()
        mean = alongX.mean(dim=-1)
        std = alongX.std(dim=-1)
        clampVal = mean + 3*std
        for bi in range(B):
            alongX[bi] = torch.clamp(alongX[bi], 0, clampVal[bi])
        alongX = alongX.view(1, -1, S, W)
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

        alongY = alongY.view(-1, S*H).float()
        mean = alongY.mean(dim=-1)
        std = alongY.std(dim=-1)
        clampVal = mean + 3*std
        for bi in range(B):
            alongY[bi] = torch.clamp(alongY[bi], 0, clampVal[bi])
        alongY = alongY.view(1, -1, S, H)
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
        vox_batch = []
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
            idx_in_verifier = (x//2) + (W//2)*(y//2)
            idx_in_verifier = idx_in_verifier.long()
            idx_in_verifier = torch.chunk(idx_in_verifier, S)
            ones = torch.ones(segmentLen, dtype=torch.int32, device=device)
            onesBool = torch.ones(segmentLen, dtype=torch.bool, device=device)
            container = torch.zeros(W*H, dtype=torch.int32, device=device)
            container.put_(idx_in_container[sIdx], ones, accumulate=True)
            verifier_old = torch.zeros( (W//2)*(H//2), dtype=torch.bool, device=device)
            verifier_old.put_(idx_in_verifier[sIdx], onesBool, accumulate=False)
            for si in range(sIdx+1, eIdx):
                isXoutlier = outlier1d(meanX[bi, si:si+10], thresh=2)[0]
                isYoutlier = outlier1d(meanY[bi, si:si+10], thresh=2)[0]
                if isXoutlier or isYoutlier:
                    continue
                verifier_new = verifier_old.detach().clone()
                verifier_new.put_(idx_in_verifier[si], onesBool, accumulate=False)
                verifier_new_cnt = verifier_new.sum().float()
                new_info_cnt = torch.logical_xor(verifier_new, verifier_old).sum().float()
                if new_info_cnt/verifier_new_cnt < 0.1:
                    break
                container.put_(idx_in_container[si], ones, accumulate=True)
                verifier_old = verifier_new
            container = container.float()
            mean = container.mean()
            std = container.std()
            clampVal = mean + 3*std
            container = torch.clamp(container, 0, clampVal)
            container /= clampVal
            if self.mode==0 and random.random()>0.5:
                noise_density = random.random()/2
                noise_level = 2 + random.random()*3
                noises = torch.randn_like(container)
                noises = torch.clamp(noises, noise_density, noise_level)/noise_level
                container += noises
            container_batch.append(container)
        
        containers = torch.stack(container_batch).view(-1, 1, H, W)

        if DEBUG==9:
            container_img = containers.cpu().numpy()
            for bi in range(B):
                fig = plt.figure(figsize=(15,15))
                ax = []
                rows = 3
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
                plt.tight_layout()
                plt.show()
            sys.exit(0)

        return containers

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
        self.in_channels = 1

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
            self.classifier.fc = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(self.classifier.fc.in_features, num_classes)
            )
        elif classifierSelect == 'densenet201':
            self.classifier = torch.hub.load('pytorch/vision:v0.5.0', 'densenet201', pretrained=pretrained)
            # replace fc layer and first convolutional layer
            self.classifier.features.conv0 = nn.Conv2d(self.in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.classifier.classifier = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(self.classifier.classifier.in_features, num_classes)
            )
            
    def setMode(self, mode):
        self.mode = mode
        self.quantization_layer.setMode(mode)

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
        frame = self.quantization_layer.forward(x)
        # frame_cropped = self.crop_and_resize_to_resolution(frame, self.crop_dimension)
        # pred = self.classifier.forward(frame_cropped)
        pred = self.classifier.forward(frame)
        return pred, frame
