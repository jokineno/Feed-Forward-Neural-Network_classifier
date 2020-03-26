import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def plot_weights(conv_layer):
    weights = conv_layer.weight.data

    num_kernels = weights.shape[0]
    num_cols = 12
    num_rows = 6

    fig = plt.figure(figsize=(num_cols,num_rows))
    plt.subplots_adjust(bottom=0.3)

    for i in range(num_kernels):
        ax1 = fig.add_subplot(num_rows,num_cols,i+1)
        npimg = np.array(weights[i].numpy(), np.float32)
        npimg = np.minimum(1, np.maximum(0, (npimg + 0.5)))
        npimg = npimg.transpose((1, 2, 0))
        ax1.imshow(npimg)
        ax1.axis('off')
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])

    plt.savefig('filters.png', dpi=100)

