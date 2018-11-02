import pickle
import numpy as np
import torch
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt


def gpu_helper(gpu):
    if gpu >= -1:
        def to_gpu(x):
            x = x.cuda()
            return x
        return to_gpu
    else:
        def no_op(x):
            return x
        return no_op

def save_weights(module,file):
    weights = []
    for p in module.parameters():
        weights += [p.cpu().data.numpy()]
    pickle.dump(weights,open(file,'wb'))

def load_weights(module,file):
    weights = pickle.load(open(file,'rb'))
    for p,w in zip(module.parameters(),weights):
        p.data = torch.from_numpy(w)
    return weights

def detach_all(a):
    detached = []
    for ai in a:
        if isinstance(ai, list) or isinstance(ai, tuple):
            detached += [detach_all(ai)]
        else:
            detached += [ai.detach()]
    return detached

def simple_plot(data_1d, xlabel, ylabel, title, filename):
    fig = plt.figure()
    ax = plt.subplot(111)
    plt.plot(range(len(data_1d)),np.array(data_1d))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.title(title)
    fig.savefig(params['saveto']+filename)
