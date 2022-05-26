import torch.nn as nn
from os.path import join, dirname, isfile
import torch
import torch.nn.functional as F
import numpy as np
from torchvision.models.resnet import resnet34
import tqdm


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

    def forward(self, events, test=False):
        if test:
            events = events.reshape([-1, 5])
            events = events[events[..., -1] != -1]

            # points is a list, since events can have any size
            B = events[:, -1].unique().cpu().numpy().astype(int)
            num_voxels = int(2 * np.prod(self.dim) * len(B))
            vox = torch.full([num_voxels, ], fill_value=0,
                            dtype=events.dtype, device=events.device)
            C, H, W = self.dim

            # get values for each channel
            x, y, t, p, b = events.t()

            # normalizing timestamps
            for i, bi in enumerate(B):
                t[events[:, -1] == bi] /= t[events[:, -1] == bi].max()
                b[b == bi] = i

            p = (p+1)/2  # maps polarity to 0, 1

            idx_before_bins = x \
                + W * y \
                + 0 \
                + W * H * C * p \
                + W * H * C * 2 * b

            for i_bin in range(C):
                values = t * self.value_layer.forward(t-i_bin/(C-1))

                # draw in voxel grid
                idx = idx_before_bins + W * H * i_bin
                vox.put_(idx.long(), values, accumulate=True)

            vox = vox.view(-1, 2, C, H, W)
            vox = torch.cat([vox[:, 0, ...], vox[:, 1, ...]], 1)

            return vox


        events = events.reshape([-1, 5])
        events = events[events[..., -1] != -1]
        events[:, -1] -= events[:, -1].min()

        # points is a list, since events can have any size
        B = events[:,-1].unique().cpu().numpy().astype(int)
        C, H, W = self.dim
        vox = torch.full([len(B), 2, C, H, W], fill_value=0, dtype=torch.float32, device=events.device)

        # get values for each channel
        x, y, t, p, b = events.t()

        # normalizing timestamps
        for i, bi in enumerate(B):
            t[b == bi] /= t[b == bi].max()
            b[b == bi] = i

        p = (p+1)/2  # maps polarity to 0, 1

        x = x.long()
        y = y.long()
        p = p.long()
        b = b.long()

        times = t.unique()

        for i_bin in range(C):
            values = times * self.value_layer.forward(times-i_bin/(C-1))
            indices = (times[None, :] == t[:, None]).nonzero()[:,1]
            values_all = values[indices]

            # draw in voxel grid
            bin = i_bin * torch.ones_like(x, dtype=torch.long).to(vox.device)
            vox.index_put_((b, p, bin, y, x), values_all, accumulate=True)

        vox = torch.cat([vox[:, 0, ...], vox[:, 1, ...]], 1)

        return vox


class Classifier(nn.Module):
    def __init__(self,
                 voxel_dimension=(9,180,240),  # dimension of voxel will be C x 2 x H x W
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
        input_channels = 2*voxel_dimension[0]
        self.classifier.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
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

    def forward(self, x, test=False):
        vox = self.quantization_layer.forward(x, True)
        vox_cropped = self.crop_and_resize_to_resolution(vox, self.crop_dimension)
        pred = self.classifier.forward(vox_cropped)
        return pred, vox


