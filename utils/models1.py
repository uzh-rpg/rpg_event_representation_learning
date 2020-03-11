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

class ValueLayer(nn.Module):
    def __init__(self, mlp_layers, activation=nn.ReLU(), num_channels=9):
        assert mlp_layers[-1] == 1, "Last layer of the mlp must have 1 input channel."
        assert mlp_layers[0] == 1, "First layer of the mlp must have 1 output channel"

        nn.Module.__init__(self)
        self.mlp = nn.ModuleList()
        self.activation = activation

        # create mlp
        in_channels = 1
        for out_channels in mlp_layers[1:]:
            self.mlp.append(nn.Linear(in_channels, out_channels))
            in_channels = out_channels

        # init with trilinear kernel
        path = join(dirname(__file__), "quantization_layer_init", "trilinear_init.pth")
        if isfile(path):
            state_dict = torch.load(path)
            self.load_state_dict(state_dict)
        else:
            self.init_kernel(num_channels)

    def forward(self, x):
        # create sample of batchsize 1 and input channels 1
        x = x[None,...,None]

        # apply mlp convolution
        for i in range(len(self.mlp[:-1])):
            x = self.activation(self.mlp[i](x))

        x = self.mlp[-1](x)
        x = x.squeeze()

        return x

    def init_kernel(self, num_channels):
        ts = torch.zeros((1, 2000))
        optim = torch.optim.Adam(self.parameters(), lr=1e-2)

        torch.manual_seed(1)

        for _ in tqdm.tqdm(range(1000)):  # converges in a reasonable time
            optim.zero_grad()

            ts.uniform_(-1, 1)

            # gt
            gt_values = self.trilinear_kernel(ts, num_channels)

            # pred
            values = self.forward(ts)

            # optimize
            loss = (values - gt_values).pow(2).sum()

            loss.backward()
            optim.step()


    def trilinear_kernel(self, ts, num_channels):
        gt_values = torch.zeros_like(ts)

        gt_values[ts > 0] = (1 - (num_channels-1) * ts)[ts > 0]
        gt_values[ts < 0] = ((num_channels-1) * ts + 1)[ts < 0]

        gt_values[ts < -1.0 / (num_channels-1)] = 0
        gt_values[ts > 1.0 / (num_channels-1)] = 0

        return gt_values


class QuantizationLayer(nn.Module):
    def __init__(self, dim,
                 mlp_layers=[1, 100, 100, 1],
                 activation=nn.LeakyReLU(negative_slope=0.1)):
        nn.Module.__init__(self)
        self.value_layer = ValueLayer(mlp_layers,
                                      activation=activation,
                                      num_channels=dim[0])
        self.dim = dim
        # vox_dim = [ i//2 for i in self.dim]
        self.vox_dim = tuple(self.dim)

    def forward(self, events):
        # points is a list, since events can have any size
        B = int((1+events[-1,-1]).item())

        # separate events into S segments with evently-divided timestamps
        # choose M random segments as input channel
        S = 16
        M = 4


        # obtain the height & width
        H, W = self.dim

        num_diluted = int( np.prod(self.dim) * S * B)
        num_container = int( np.prod(self.dim) * B)
        num_combiner = int( np.prod(self.dim) * M * B)
        # num_counter = int( np.prod(self.dim) * B)

        dilution = torch.zeros(num_diluted, dtype=torch.bool, device=events.device)

        concentrate = torch.zeros(num_container, dtype=torch.int32, device=events.device)
        concentrate = concentrate.view(-1, H, W)

        combiner = torch.zeros(num_combiner, dtype=torch.bool, device=events.device)

        # container = torch.zeros(num_container, dtype=torch.int32, device=events.device)

        # counter = torch.zeros(num_counter, dtype=torch.int32, device=events.device)

        # timer = torch.zeros(num_counter, dtype=torch.float32, device=events.device)

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
            mixture[i] *= 0.5 + (S-i)/S
            mixture[i] += mixture[i-1] * i/S
            mixture[i] = F.conv2d( mixture[i].unsqueeze(0), erode_w, padding=1, stride=1, groups=B) - 0.25
        mixture_bin = mixture > 0
        for i in range(S):
            concentrate += mixture_bin[i]

        range_limit = concentrate.view(B,-1)
        range_limit = range_limit > 0
        dilution = dilution.view(B,S,-1)
        combiner = combiner.view(B,M,-1)
        min_pixel_required = range_limit.sum(dim=-1) // S
        rand_idxs = random.sample( range(0,S), M)
        for i in range(M):
            idx = rand_idxs[i]
            for bi in range(B):
                combiner[bi][i] = dilution[bi][idx] & range_limit[bi]
                cnt = 0
                limit = S//M
                while cnt<limit and combiner[bi][i].sum(dim=-1)<min_pixel_required[bi]:
                    cnt += 1
                    if idx < M//2:
                        combiner[bi][i] |= dilution[bi][idx+cnt] & range_limit[bi]
                    else:
                        combiner[bi][i] |= dilution[bi][idx-cnt] & range_limit[bi]
        combiner = combiner.view(-1,M,H,W).float()

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

        # idx_in_container = x + W*y + W*H*b
        # # num_events_ones = torch.ones_like(idx_in_container, dtype=torch.int32)
        # container.put_(idx_in_container.long(), num_events_ones, accumulate=True)
        # container = container.view(-1, H, W)

        # idx_in_counter = x//2 + W//2*(y//2) + W*H*b//4
        # counter.put_(idx_in_counter.long(), num_events_ones, accumulate=True)
        # counter = counter.float()
        # timerDivider = counter.clone()
        # timerDivider[timerDivider==0] = 1

        # timer.put_(idx_in_counter.long(), t, accumulate=True)

        # timer = timer/timerDivider
        
        # counter = counter.view(-1, H//2, W//2)
        # timer = timer.view(-1, H//2, W//2)

        # diff_y = container[:,::2] - container[:,1::2]
        # diff_y = diff_y[:,:,::2] + diff_y[:,:,1::2]
        # diff_y = diff_y.float()

        # diff_x = container[:,:,::2] - container[:,:,1::2]
        # diff_x = diff_x[:,1::2] + diff_x[:,::2]
        # diff_x = diff_x.float()

        # diff_x = diff_x.unsqueeze(dim=1)
        # diff_y = diff_y.unsqueeze(dim=1)
        # timer = timer.unsqueeze(dim=1)
        # counter = counter.unsqueeze(dim=1)
        # best_dilution = best_dilution.unsqueeze(dim=1)
        # best_dilution = F.interpolate(best_dilution, scale_factor=0.5)

        vox = torch.cat([concentrate, combiner], dim=1)

        if DEBUG==9:
            IMG = 0
            print(vox.size())
            # visualization = plt.figure()
            # fig0 = visualization.add_subplot(221)
            # fig1 = visualization.add_subplot(222)
            # fig2 = visualization.add_subplot(223)
            # fig3 = visualization.add_subplot(224)
            # img0 = timer[IMG][0].numpy()
            # img0 = np.where(img0>0, 255, 0)
            # img1 = combined_img[IMG][0].numpy()
            # img1 = np.where(img1>0, 255, 0)
            # img2 = combiner[IMG][1].numpy()
            # img2 = np.where(img2>0, 255, 0)
            # img3 = combiner[IMG][9].numpy()
            # img3 = np.where(img3>0, 255, 0)
            # fig0.imshow(img0, cmap='gray', vmin=0, vmax=255)
            # fig1.imshow(img1, cmap='gray', vmin=0, vmax=255)
            # fig2.imshow(img2, cmap='gray', vmin=0, vmax=255)
            # fig3.imshow(img3, cmap='gray', vmin=0, vmax=255)
            # plt.show(block=False)
            # plt.pause(10)

            # fig0 = visualization.add_subplot(221)
            # fig1 = visualization.add_subplot(222)
            # fig2 = visualization.add_subplot(223)
            # fig3 = visualization.add_subplot(224)
            # img0 = timer[IMG][0].numpy()
            # img1 = counter[IMG][0].numpy()
            # img1 = np.where(img1>0, 255, 0)
            # img2 = diff_y[IMG][0].numpy()
            # img2 = np.where(img2>0, 255, 0)
            # img3 = container[IMG].numpy()
            # img3 = np.where(img3>0, 255, 0)
            # fig0.imshow(img0, cmap='gray', vmin=0, vmax=1)
            # fig1.imshow(img1, cmap='gray', vmin=0, vmax=255)
            # fig2.imshow(img2, cmap='gray', vmin=0, vmax=255)
            # fig3.imshow(img3, cmap='gray', vmin=0, vmax=255)
            # plt.show(block=False)
            # plt.pause(10)

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
        # 5 channel
        self.classifier.conv1 = nn.Conv2d(5, 64, kernel_size=7, stride=2, padding=3, bias=False)
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


