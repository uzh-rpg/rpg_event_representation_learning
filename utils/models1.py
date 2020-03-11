import math
import random
import torch.nn as nn
from os.path import join, dirname, isfile
import torch
import torch.nn.functional as F
import numpy as np
from torchvision.models.resnet import resnet34
import tqdm

DEBUG = 9

if DEBUG>0:
    import matplotlib.pyplot as plt

class QuantizationLayer(nn.Module):
    def __init__(self, dim,
                 mlp_layers=[1, 100, 100, 1],
                 activation=nn.LeakyReLU(negative_slope=0.1)):
        nn.Module.__init__(self)
        self.dim = dim

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

        concentrate = torch.zeros(num_container, dtype=torch.int32, device=events.device)
        concentrate = concentrate.view(-1, H, W)

        # get values for each channel
        # x, y in the form of +x axis with -y axis
        x, y, t, p, b = events.t()

        # normalizing timestamps
        for bi in range(B):
            t[events[:,-1] == bi] /= t[events[:,-1] == bi].max()

        p = (p+1)/2  # maps polarity to 0, 1

        row_cnt = len(events)
        num_events_ones = torch.ones(row_cnt, dtype=torch.bool, device=events.device)
        time_separator = ( t / (1/S)).int()
        time_separator[time_separator == S] = S-1
        idx_in_dilution = x + W*y + W*H*time_separator + W*H*S*b
        dilution.put_(idx_in_dilution.long(), num_events_ones, accumulate=False)
        dilution = dilution.view(-1, S, H, W)
        mixture = dilution.clone().detach().float()
        mixture = mixture.permute(1,0,2,3)
        erode_w = torch.ones( (B, 1, 3, 3), dtype=torch.float32, device=events.device)
        erode_w[:,:,1,1] = 0
        erode_w /= 8
        for i in range(1, S):
            # mixture[i] *= 0.5 + (S-i)/S
            mixture[i] += mixture[i-1]
            mixture[i] = F.conv2d( mixture[i].unsqueeze(0), erode_w, padding=1, stride=1, groups=B) - 0.5
            mixture[i] = F.relu(mixture[i])
        mixture_bin = mixture > 0
        for i in range(S):
            concentrate += mixture_bin[i]

        # normalizing pixels to range 0-1
        concentrate = concentrate.view(-1,1,H,W).float()
        for bi in range(B):
            concentrate[bi] /= concentrate[bi].max()

        if DEBUG==9:
            dilution = dilution.view(-1,S,H,W)
            for bi in range(B):
                visualization = plt.figure()
                fig0 = visualization.add_subplot(221)
                fig1 = visualization.add_subplot(222)
                fig2 = visualization.add_subplot(223)
                fig3 = visualization.add_subplot(224)
                img0 = concentrate[bi][0].numpy()
                # img0 = img0 / np.max(img0)
                # img0 = np.where(img0>0, 255, 0)
                img1 = dilution[bi][1].numpy()
                # img1 = img1 / np.max(img1)
                # img1 = np.where(img1>0, 1, 0)
                img2 = dilution[bi][2].numpy()
                # img2 = img2 / np.max(img2)
                # img2 = np.where(img1>0, 255, 0)
                img3 = dilution[bi][3].numpy()
                # img3 = img3 / np.max(img3)
                # img3 = np.where(img3>0, 255, 0)
                fig0.imshow(img0, cmap='gray', vmin=0, vmax=1)
                fig1.imshow(img1, cmap='gray', vmin=0, vmax=1)
                fig2.imshow(img2, cmap='gray', vmin=0, vmax=1)
                fig3.imshow(img3, cmap='gray', vmin=0, vmax=1)
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
                 mlp_layers=[1, 30, 30, 1],
                 activation=nn.LeakyReLU(negative_slope=0.1),
                 pretrained=True):

        nn.Module.__init__(self)
        self.quantization_layer = QuantizationLayer(voxel_dimension, mlp_layers, activation)
        self.classifier = resnet34(pretrained=pretrained)

        self.crop_dimension = crop_dimension

        # replace fc layer and first convolutional layer
        self.classifier.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.classifier.fc = nn.Linear(self.classifier.fc.in_features, num_classes)

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


