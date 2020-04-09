import numpy as np
import torch
from .torch_percentile import percentile

# Obtained and modified from https://github.com/MingyeXu/GS-Net
def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx

# Obtained and modified from https://github.com/MingyeXu/GS-Net
def get_graph_feature(x, k=20, idx=None, device='cpu'):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)   # (batch_size, num_points, k)

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2)
  
    return feature

# Obtained and modified from https://stackoverflow.com/questions/11882393/matplotlib-disregard-outliers-when-plotting/11886564#11886564
def outlier1d(points, thresh=3.5):
    """
    Returns a boolean array with True if points are outliers and False 
    otherwise.

    Parameters:
    -----------
        points : An numobservations by numdimensions array of observations
        thresh : The modified z-score to use as a threshold. Observations with
            a modified z-score (based on the median absolute deviation) greater
            than this value will be classified as outliers.

    Returns:
    --------
        mask : A numobservations-length boolean array.

    References:
    ----------
        Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
        Handle Outliers", The ASQC Basic References in Quality Control:
        Statistical Techniques, Edward F. Mykytka, Ph.D., Editor. 
    """
    if len(points.shape) == 1:
        points = points[:,None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score > thresh


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
        real = torch.fft.roll_n(real, axis=dim, n=real.size(dim)//2)
        imag = torch.fft.roll_n(imag, axis=dim, n=imag.size(dim)//2)
    image = torch.stack([real, imag], dim=0).permute(1,2,3,0)
    return image