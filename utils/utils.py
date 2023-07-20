from scipy.signal import medfilt
import numpy as np
import math
import matplotlib.pyplot as plt

def saveContourPlots(array_of_contours, file_path, list_of_strings, num_cols):
    # save loss history to a chart

    num_steps = array_of_contours.shape[0]
    step_array = np.arange(num_steps)

    num_contours = array_of_contours.shape[1]
    num_rows = math.ceil(num_contours/num_cols)
    dims = math.ceil(math.sqrt(num_contours))

    x_label = list_of_strings[0]
    y_label = list_of_strings[1]
    labels = list_of_strings[2:]

    plt.figure()

    for i in range(num_contours):
        plt.subplot(num_rows, num_cols, i+1)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        #plt.ylim(0,1)
        #plt.yticks(np.arange(0, 1, step=0.2))
        plt.plot(step_array, array_of_contours[:,i], 'r--',label=labels[i])
        plt.legend()

    plt.show()
    plt.savefig(file_path)

#from model_vc import Generator
import torch

def setup_gen(config, Generator):
    G = Generator(config.dim_neck, config.dim_emb, config.dim_pre, config.freq)
    g_optimizer = torch.optim.Adam(G.parameters(), config.adam_init)
    g_checkpoint = torch.load(config.autovc_ckpt, map_location='cpu')
    G.load_state_dict(g_checkpoint['model_state_dict'])
    g_optimizer.load_state_dict(g_checkpoint['optimizer_state_dict'])
    # fixes tensors on different devices error
    # https://github.com/pytorch/pytorch/issues/2830
    for state in g_optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.cuda(config.which_cuda)
    previous_ckpt_iters = g_checkpoint['iteration']

    G.to(config.device)
    return G

from vte_model import Vt_Embedder
from collections import OrderedDict

def setup_vte(config, spmel_params):
    vte =  Vt_Embedder(config, spmel_params)
    for param in vte.parameters():
        param.requires_grad = False
    vte_optimizer = torch.optim.Adam(vte.parameters(), 0.0001)
    vte_checkpoint = torch.load(config.vte_ckpt)
    new_state_dict = OrderedDict()
    for i, (key, val) in enumerate(vte_checkpoint['model_state_dict'].items()):
#            if key.startswith('class_layer'):
#                continue
        new_state_dict[key] = val
    vte.load_state_dict(new_state_dict)

    for state in vte_optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.cuda(which_cuda)

    vte.to(config.device)
    vte.eval()
    return vte
