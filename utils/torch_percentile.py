# Obtained and modified from https://gist.github.com/spezold/42a451682422beb42bc43ad0c0967a30

from typing import Union

import torch
import numpy as np


def percentile(t: torch.tensor, q: float) -> Union[int, float]:
    """
    Return the ``q``-th percentile of the flattened input tensor's data.
    
    CAUTION:
     * Needs PyTorch >= 1.1.0, as ``torch.kthvalue()`` is used.
     * Values are not interpolated, which corresponds to
       ``numpy.percentile(..., interpolation="nearest")``.
       
    :param t: Input tensor.
    :param q: Percentile to compute, which must be between 0 and 100 inclusive.
    :return: Resulting value (scalar).
    """
    # Note that ``kthvalue()`` works one-based, i.e. the first sorted value
    # indeed corresponds to k=1, not k=0! Use float(q) instead of q directly,
    # so that ``round()`` returns an integer, even if q is a np.float32.
    k = 1 + round(.01 * float(q) * (t.numel() - 1))
    result = t.view(-1).kthvalue(k).values.item()
    return result

if __name__ == "__main__":
    
    q_s = np.r_[0, 100 * np.random.uniform(size=8), 100.]
    
    a = np.random.uniform(size=(3, 4, 5))
    t = torch.from_numpy(a)
    
    for q in q_s:
        p_t = percentile(t, q)
        p_a = np.percentile(a, q, interpolation="nearest")
        print("q={}, PyTorch result: {}".format(q, p_t))
        print("q={}, NumPy result:   {}".format(q, p_a))
        assert p_t == p_a