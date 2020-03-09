import math
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
        vox_dim = [ i//2 for i in self.dim]
        self.vox_dim = tuple(vox_dim)

    def forward(self, events):
        # points is a list, since events can have any size
        B = int((1+events[-1,-1]).item())

        # separate events into S segments with evently-divided timestamps
        S = 16

        # obtain the height & width
        H, W = self.dim

        num_container = int( np.prod(self.dim) * B)
        container = torch.zeros(num_container, dtype=torch.int32, device=events.device)

        num_combiner = int( np.prod(self.dim) * S * B)
        combiner = torch.zeros(num_combiner, dtype=torch.int32, device=events.device)
        combined_img = torch.zeros(num_container, dtype=torch.float32, device=events.device)
        combined_img = combined_img.view(-1, H, W)

        num_counter = int( np.prod(self.vox_dim) * B)
        counter = torch.zeros(num_counter, dtype=torch.int32, device=events.device)

        timer = torch.zeros(num_counter, dtype=torch.float32, device=events.device)

        # get values for each channel
        # x, y in the form of +x axis with -y axis
        x, y, t, p, b = events.t()

        # normalizing timestamps
        for bi in range(B):
            t[events[:,-1] == bi] /= t[events[:,-1] == bi].max()

        p = (p+1)/2  # maps polarity to 0, 1

        idx_in_container = x + W*y + W*H*b
        num_events_ones = torch.ones_like(idx_in_container, dtype=torch.int32)
        container.put_(idx_in_container.long(), num_events_ones, accumulate=True)
        container = container.view(-1, H, W)

        row_cnt = len(events)
        time_segment = torch.zeros(row_cnt, dtype=torch.int32, device=events.device)
        for i in range(1, S):
            time_segment[t >= i/S] += 1
        idx_in_combiner = x + W*y + W*H*time_segment + W*H*S*b
        combiner.put_(idx_in_combiner.long(), time_segment, accumulate=True)
        combiner = combiner.view(-1, S, H, W).float()
        pad = lambda n: (n, 0) if n>=0 else (0, -n)
        erode_w = torch.ones((1, 1, 3, 3), dtype=torch.float32, device=events.device)
        for bi in range(B):
            segment = combiner[bi][0].clone().detach()
            time_segment_bi = time_segment[events[:,-1] == bi]
            x_bi = x[events[:,-1] == bi]
            y_bi = y[events[:,-1] == bi]
            x_mean = torch.mean(x_bi[time_segment_bi == 0]).item()
            y_mean = torch.mean(y_bi[time_segment_bi == 0]).item()
            pts = len(time_segment_bi[time_segment_bi == 0])
            for i in range(1, S):
                x_mean_new = torch.mean(x_bi[time_segment_bi == i]).item()
                y_mean_new = torch.mean(y_bi[time_segment_bi == i]).item()
                pts_new = len(time_segment_bi[time_segment_bi == i])
                if pts_new > pts:
                    tmp_segment = segment
                    segment = combiner[bi][i].clone().detach()
                    x_diff = int( math.floor(x_mean_new - x_mean))
                    y_diff = int( math.floor(y_mean_new - y_mean))
                    x_mean = x_mean_new
                    y_mean = y_mean_new
                else:
                    tmp_segment = combiner[bi][i].clone().detach()
                    x_diff = int( math.floor(x_mean - x_mean_new))
                    y_diff = int( math.floor(y_mean - y_mean_new))

                padding = pad(x_diff) + pad(y_diff)
                padded_segment = F.pad(tmp_segment, padding)
                padded_segment = padded_segment[:H,:W]
                segment += padded_segment
            combined_img[bi] = F.relu(segment - S)

        idx_in_counter = x//2 + W//2*(y//2) + W*H*b//4
        counter.put_(idx_in_counter.long(), num_events_ones, accumulate=True)
        counter = counter.float()
        timerDivider = counter.clone()
        timerDivider[timerDivider==0] = 1

        timer.put_(idx_in_counter.long(), t, accumulate=True)

        timer = timer/timerDivider
        
        counter = counter.view(-1, H//2, W//2)
        timer = timer.view(-1, H//2, W//2)

        diff_y = container[:,::2] - container[:,1::2]
        diff_y = diff_y[:,:,::2] + diff_y[:,:,1::2]
        diff_y = diff_y.float()

        diff_x = container[:,:,::2] - container[:,:,1::2]
        diff_x = diff_x[:,1::2] + diff_x[:,::2]
        diff_x = diff_x.float()

        diff_x = diff_x.unsqueeze(dim=1)
        diff_y = diff_y.unsqueeze(dim=1)
        timer = timer.unsqueeze(dim=1)
        counter = counter.unsqueeze(dim=1)
        combined_img = combined_img.unsqueeze(dim=1)
        combined_img = F.interpolate(combined_img, scale_factor=0.5)

        vox = torch.cat([diff_x, diff_y, timer, counter, combined_img], dim=1)

        if DEBUG==9:
            IMG = 0
            print(vox.size())
            visualization = plt.figure()
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
        # 5 channels, combined_events_representation & x_vector_representation & y_vector_representation & count_representation & time_representation
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


