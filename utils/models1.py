import math
import random
import torch.nn as nn
from os.path import join, dirname, isfile
import torch
import torch.nn.functional as F
import numpy as np
from torchvision.models.resnet import resnet34
import tqdm
from .torch_percentile import percentile

DEBUG = 0

if DEBUG>0:
    import matplotlib.pyplot as plt
    import time

def phase_correlation(a, b):
        B, H, W = a.size()
        a = a.unsqueeze(dim=-1).expand(B, H, W, 2)
        b = b.unsqueeze(dim=-1).expand(B, H, W, 2)
        G_a = torch.fft(a, signal_ndim=2)
        G_b = torch.fft(b, signal_ndim=2)
        conj_b = torch.conj(G_b)
        R = G_a * conj_b
        R /= torch.abs(R)
        r = torch.ifft(R, signal_ndim=2)
        r = torch.split(r, 1, dim=-1)[0].squeeze(-1)
        shift = r.view(B, -1).argmax(dim=1)
        shift = torch.cat(((shift / W).view(-1, 1), (shift % W).view(-1, 1)), dim=1)
        return shift

def fftshift(image):
    # Original with size (B, H, W, 2)
    image = image.permute(3,0,1,2)
    real, imag = image[0], image[1]
    for dim in range(1, len(real.size())):
        real = torch.fft.roll_n(real, axis=dim, n=real.size(dim)//2)
        imag = torch.fft.roll_n(imag, axis=dim, n=imag.size(dim)//2)
    image = torch.stack([real, imag], dim=0).permute(1,2,3,0)
    return image

def ifftshift(image):
    # Original with size (B, H, W, 2)
    image = image.permute(3,0,1,2)
    real, imag = image[0], image[1]
    for dim in range(len(real.size()) - 1, 0, -1):
        real = fft.roll_n(real, axis=dim, n=real.size(dim)//2)
        imag = fft.roll_n(imag, axis=dim, n=imag.size(dim)//2)
    image = torch.stack([real, imag], dim=0).permute(1,2,3,0)
    return image

class QuantizationLayer(nn.Module):
    def __init__(self, dim, device):
        nn.Module.__init__(self)
        self.dim = dim
        self.mode = 0 # 0:Training 1:Validation 
        self.device = device
        self.k_noise = 2
        self.startIdx = 2

    def setMode(self, mode):
        self.mode = mode

    def forward(self, events):
        # points is a list, since events can have any size
        B = len(events)
        # separate events into S segments with evently-divided number of events
        S = 32
        # obtain the height & width
        H, W = self.dim

        num_alongX = int( S * W * B)
        num_alongY = int( S * H * B)
        alongX = torch.zeros(num_alongX, dtype=torch.int32, device=self.device)
        alongY = torch.zeros(num_alongY, dtype=torch.int32, device=self.device)
        segmentLen_batch = []
        for bi in range(B):
            events[bi] = torch.from_numpy(events[bi]).to(self.device).squeeze(0)
            x, y, t, p, b = events[bi].t()
            segmentLen = len(x)//S
            segmentLen_batch.append(segmentLen)
            chunks = torch.arange(S, dtype=torch.int32, device=self.device)
            chunks = chunks.unsqueeze(-1).expand(-1, segmentLen).reshape(-1)
            resultLen = len(chunks)
            ix = x[:resultLen] + W*chunks + W*S*bi
            iy = y[:resultLen] + H*chunks + H*S*bi
            ones = torch.ones(resultLen, dtype=torch.int32, device=self.device)
            alongX.put_(ix.long(), ones, accumulate=True)
            alongY.put_(iy.long(), ones, accumulate=True)

        segmentLen_batch = torch.FloatTensor(segmentLen_batch).to(self.device)

        alongX = alongX.view(-1, S*W)
        noise_loc = torch.topk(alongX, self.k_noise, dim=-1, largest=True)[1]
        noise_loc = torch.cat(((noise_loc / W).view(-1, 1), (noise_loc % W).view(-1, 1)), dim=1)
        noise_loc = noise_loc.view(-1, self.k_noise, 2)
        alongX = alongX.view(-1, S, W)
        for bi in range(B):
            for nx in noise_loc[bi, :, 1]:
                alongX[bi, :, nx] = 0.
        alongDim = torch.arange(W, dtype=torch.float32, device=self.device)
        alongDim = alongDim.expand(B, -1).unsqueeze(-1)
        meanX = torch.bmm(alongX.float(), alongDim).squeeze(-1) / segmentLen_batch.view(-1,1)
        start_seg_x = meanX[:, self.startIdx].unsqueeze(-1)
        alignedX = meanX - start_seg_x
        alignedX = alignedX.round().int()

        alongY = alongY.view(-1, S*H)
        noise_loc = torch.topk(alongY, self.k_noise, dim=-1, largest=True)[1]
        noise_loc = torch.cat(((noise_loc / H).view(-1, 1), (noise_loc % H).view(-1, 1)), dim=1)
        noise_loc = noise_loc.view(-1, self.k_noise, 2)
        alongY = alongY.view(-1, S, H)
        for bi in range(B):
            for ny in noise_loc[bi, :, 1]:
                alongY[bi, :, ny] = 0.
        alongDim = torch.arange(H, dtype=torch.float32, device=self.device)
        alongDim = alongDim.expand(B, -1).unsqueeze(-1)
        meanY = torch.bmm(alongY.float(), alongDim).squeeze(-1) / segmentLen_batch.view(-1,1)
        start_seg_y = meanY[:, self.startIdx].unsqueeze(-1)
        alignedY = meanY - start_seg_y
        alignedY = alignedY.round().int()

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
            ones = torch.ones(segmentLen, dtype=torch.int32, device=self.device)
            onesBool = torch.ones(segmentLen, dtype=torch.bool, device=self.device)
            container = torch.zeros(W*H, dtype=torch.int32, device=self.device)
            container.put_(idx_in_container[self.startIdx], ones, accumulate=True)
            verifier_old = torch.zeros( (W//2)*(H//2), dtype=torch.bool, device=self.device)
            verifier_old.put_(idx_in_verifier[self.startIdx], onesBool, accumulate=False)
            for si in range(self.startIdx+1, S):
                verifier_new = verifier_old.detach().clone()
                verifier_new.put_(idx_in_verifier[si], onesBool, accumulate=False)
                verifier_new_cnt = verifier_new.sum().float()
                new_info_cnt = torch.logical_xor(verifier_new, verifier_old).sum().float()
                if new_info_cnt/verifier_new_cnt < 0.1:
                    break
                container.put_(idx_in_container[si], ones, accumulate=True)
                verifier_old = verifier_new
            container_batch.append(container)
        containers = torch.stack(container_batch).view(-1, H, W).float()

        # centralize the image
        pad = lambda n: (n, 0) if n>=0 else (0, -n)
        for bi in range(B):
            x, y, t, p, b = events[bi].t()
            x_mean = torch.mean(x).item()
            y_mean = torch.mean(y).item()
            x_diff_from_center = int( math.floor(W//2 - x_mean))
            y_diff_from_center = int( math.floor(H//2 - y_mean))
            padding = pad(x_diff_from_center) + pad(y_diff_from_center)
            padded = F.pad(containers[bi], padding)
            containers[bi] = padded[padding[3]:H+padding[3],padding[1]:W+padding[1]]
        
        containers = containers.unsqueeze(dim=1)

        if DEBUG==9:
            container_img = containers.numpy()
            for bi in range(B):
                fig = plt.figure(figsize=(15,15))
                ax = []
                rows = 3
                columns = 1
                for i in range(rows * columns):
                    ax.append( fig.add_subplot(rows, columns, i+1))
                    if i==0:
                        ax[-1].set_title('x-y graph')
                        plt.imshow(container_img[bi][0], cmap='gray', vmin=0, vmax=4)
                    elif i==1:
                        ax[-1].set_title('x-t graph')
                        plt.imshow(alongX[bi], cmap='gray', vmin=0, vmax=80)
                    elif i==2:
                        ax[-1].set_title('y-t graph')
                        plt.imshow(alongY[bi], cmap='gray', vmin=0, vmax=80)
                plt.tight_layout()
                plt.show()
            while True:
                pass

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

        classifierSelect = 'resnet34'
        if classifierSelect == 'resnet34':
            self.modelChildren = ['avgpool', 'layer4', 'layer3', 'layer2', 'layer1', 'maxpool', 'relu', 'bn1']
            classifierSelect_sub = ''
            if classifierSelect_sub == 'wide':
                self.classifier = torch.hub.load('pytorch/vision:v0.5.0', 'wide_resnet50_2', pretrained=pretrained)
            else:
                self.classifier = resnet34(pretrained=pretrained)
            # replace fc layer and first convolutional layer
            self.classifier.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.classifier.fc = nn.Linear(self.classifier.fc.in_features, num_classes)
        elif classifierSelect == 'densenet201':
            self.classifier = torch.hub.load('pytorch/vision:v0.5.0', 'densenet201', pretrained=pretrained)
            # replace fc layer and first convolutional layer
            self.classifier.features.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
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
