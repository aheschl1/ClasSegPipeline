import numpy as np
import warnings

# convert to bit representation of the discrete label mask, where channel 0 is the least significant bit
def label_to_bitmask(label):
    label = label.astype(np.uint8)
    
    n_bits = np.floor(np.log2(label.max())).astype(int) + 1
    n_bits = max(n_bits, 1)

    bitmask = np.zeros((n_bits, label.shape[1], label.shape[2]), dtype=np.uint8)
    for i in range(n_bits):
        bitmask[i, ...] = (label[0]//2**i)%2
    return bitmask

def bitmask_to_label(bitmask: np.array):
    if not len(np.unique(bitmask)) == 2:
        warnings.warn(f"not binary! {np.unique(bitmask)}")
    label = np.zeros((bitmask.shape[0], 1, bitmask.shape[-2], bitmask.shape[-1]), dtype=np.uint8)
    for i in range(bitmask.shape[0] if len(bitmask.shape) == 3 else bitmask.shape[1]):
        if len(bitmask.shape) == 4:
            s = bitmask[:, i: i+1, ...]
        else:
            s = bitmask[i:i+1, ...]
        label += (s*2**i).astype(np.uint8)
    return label