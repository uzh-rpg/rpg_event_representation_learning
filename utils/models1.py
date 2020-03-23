import math
import random
import torch.nn as nn
from os.path import join, dirname, isfile
import torch
import torch.nn.functional as F
import numpy as np
from torchvision.models.resnet import resnet34
import tqdm

DEBUG = 0

if DEBUG>0:
    import matplotlib.pyplot as plt
    import time

class QuantizationLayer(nn.Module):
    def __init__(self, dim):
        nn.Module.__init__(self)
        self.dim = dim
        self.mode = 0 # 0:Training 1:Validation 

    def setMode(self, mode):
        self.mode = mode

    def forward(self, events):
        # points is a list, since events can have any size
        B = int((1+events[-1,-1]).item())

        # separate events into S segments with evently-divided timestamps
        S = 16

        # obtain the height & width
        H, W = self.dim

        num_diluted = int( np.prod(self.dim) * S * B)
        num_container = int( np.prod(self.dim) * B)

        dilution = torch.zeros(num_diluted, dtype=torch.bool, device=events.device)

        # get values for each channel
        # x, y in the form of +x axis with -y axis
        x, y, t, p, b = events.t()

        # normalizing timestamps
        fast_norm = True
        if fast_norm:
            t /= t.max()*1.001
        else:
            for bi in range(B):
                t[events[:,-1] == bi] /= t[events[:,-1] == bi].max()*1.001

        row_cnt = len(events)
        num_events_ones = torch.ones(row_cnt, dtype=torch.bool, device=events.device)
        time_separator = (t * S).int()
        idx_in_dilution = x + W*y + W*H*time_separator + W*H*S*b
        dilution.put_(idx_in_dilution.long(), num_events_ones, accumulate=False)
        dilution = dilution.view(-1, S, H, W).permute(1,0,2,3).float()
        dilution = dilution*2 - 1
        erode_w = torch.ones( (B, 1, 3, 3), dtype=torch.float32, device=events.device)
        erode_w[:,:,1,1] = -1
        erode_w /= 16
        sum = torch.zeros( (S-1, B), dtype=torch.float32, device=events.device)
        for i in range(0, S-1):
            dilution[i] = dilution[i] + dilution[i+1]*8
            dilution[i] = F.conv2d( dilution[i].unsqueeze(0), erode_w, padding=1, stride=1, groups=B)
            dilution[i] = F.relu(dilution[i])
            sum[i] = dilution[i].view(B, -1).sum(dim=-1)
        max_idx = sum.max(dim=0)[1]

        combine_frame = 4
        fragment_size = S//combine_frame
        concentrate = torch.zeros(3 * num_container, dtype=torch.float32, device=events.device)
        concentrate = concentrate.view(-1,B,H,W)
        for bi in range(B):
            concentrate[0][bi] = dilution[ max_idx[bi].long()][bi]
        dilution = dilution > 0
        rand = random.randint(0, S-fragment_size)
        for i in range(rand, rand+fragment_size):
            concentrate[1] += dilution[i]
        for i in range(0, S-1):
            concentrate[2] += dilution[i]
        concentrate = concentrate.permute(1,0,2,3)

        # centralize the image
        pad = lambda n: (n, 0) if n>=0 else (0, -n)
        for bi in range(B):
            # concentrate[bi] /= concentrate[bi].max()
            x_mean = torch.mean(x[events[:,-1] == bi]).item()
            y_mean = torch.mean(y[events[:,-1] == bi]).item()
            x_diff_from_center = int( math.floor(W//2 - x_mean))
            y_diff_from_center = int( math.floor(H//2 - y_mean))
            padding = pad(x_diff_from_center) + pad(y_diff_from_center)
            for i in range(3):
                padded = F.pad(concentrate[bi][i], padding)
                concentrate[bi][i] = padded[padding[3]:H+padding[3],padding[1]:W+padding[1]]

        if DEBUG==9:
            dilution = dilution.permute(1,0,2,3)
            for bi in range(B):
                visualization = plt.figure()
                fig0 = visualization.add_subplot(221)
                fig1 = visualization.add_subplot(222)
                fig2 = visualization.add_subplot(223)
                fig3 = visualization.add_subplot(224)
                img0 = concentrate[bi][0].numpy()
                # img0 = img0 / np.max(img0)
                # img0 = np.where(img0>0, 255, 0)
                img1 = concentrate[bi][1].numpy()
                # img1 = img1 / np.max(img1)
                # img1 = np.where(img1>0, 1, 0)
                img2 = concentrate[bi][2].numpy()
                # img2 = img2 / np.max(img2)
                # img2 = np.where(img1>0, 255, 0)
                img3 = dilution[bi][S//2].numpy()
                # img3 = img3 / np.max(img3)
                # img3 = np.where(img3>0, 255, 0)
                fig0.imshow(img0, cmap='gray', vmin=0, vmax=10)
                fig1.imshow(img1, cmap='gray', vmin=0, vmax=10)
                fig2.imshow(img2, cmap='gray', vmin=0, vmax=10)
                fig3.imshow(img3, cmap='gray', vmin=0, vmax=10)
                plt.show()
            while True:
                pass

        vox = concentrate

        return vox

class Classifier(nn.Module):
    def __init__(self,
                 voxel_dimension=(180,240),  # dimension of voxel will be C x 2 x H x W
                 crop_dimension=(224, 224),  # dimension of crop before it goes into classifier
                 num_classes=101,
                 pretrained=True):

        nn.Module.__init__(self)
        self.mode = 0 # 0:Training 1:Validation 
        self.quantization_layer = QuantizationLayer(voxel_dimension)
        self.crop_dimension = crop_dimension
        self.num_classes = num_classes

        use_resnet = True
        use_wide_resnet = False
        if use_resnet:
            self.modelChildren = ['avgpool', 'layer4', 'layer3', 'layer2', 'layer1', 'maxpool', 'relu', 'bn1']
            if use_wide_resnet:
                self.classifier = torch.hub.load('pytorch/vision:v0.5.0', 'wide_resnet50_2', pretrained=pretrained)
            else:
                self.classifier = resnet34(pretrained=pretrained)
            # replace fc layer and first convolutional layer
            self.classifier.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.classifier.fc = nn.Linear(self.classifier.fc.in_features, num_classes)
        else:
            self.classifier = torch.hub.load('pytorch/vision:v0.5.0', 'densenet201', pretrained=pretrained)
            self.classifier.features.conv0 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
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
