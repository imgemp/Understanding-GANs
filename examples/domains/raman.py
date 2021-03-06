import os
import zipfile
import random

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms

from sklearn.model_selection import KFold

from ugans.core import Data, Net
from ugans.utils import load_url

import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import seaborn as sns



class Raman(Data):
    def __init__(self, batch_size=128, **kwargs):
        super(Raman, self).__init__()
        self.batch_size = batch_size
        self.load_raman()
        # Number of workers for dataloader
        workers = 2
        # Create the dataloader
        self.dataloader = torch.utils.data.DataLoader(torch.from_numpy(self.x_att), batch_size=batch_size,
                                                      shuffle=True, num_workers=workers,
                                                      drop_last=True)
        self.dataiterator = iter(self.dataloader)
        print('Number of batches: {}'.format(self.x_att.shape[0] // batch_size), flush=True)


    def zero_one(self,x):
        x -= np.min(x,axis=1)[:,None]
        return (x.T/np.ptp(x,axis=1)).T


    def get_endmembers(self,dataset='examples/domains/data/raman.pkl.gz'):
        x, y, waves, majors = load_url('http://www-anw.cs.umass.edu/public_data/untapped/raman.pkl.gz',dataset)
        x = self.zero_one(x)
        endmems = np.zeros((y.shape[1],x.shape[1]))
        for i in range(y.shape[1]):
            samples = (y[:,i] == 1.)
            endmems[i] = np.mean(x[samples],axis=0)
        return endmems


    def load_raman(self):
        xy, ux, waves, names, colors = self.load_process_data(DropLastDim=False)
        self.x_dim = xy[-2].shape[1]
        self.att_dim = xy[-1].shape[1]
        self.x_att = np.hstack((xy[-2],xy[-1])).astype('float32')
        self.attribute_names = names
        self.waves = waves


    def load_process_data(self,dataset='examples/domains/data/raman.pkl.gz',trial=0,n_folds=2,
                          remove_mean=False,log_x=False,DropLastDim=True):
        x, y, waves, majors = load_url('http://www-anw.cs.umass.edu/public_data/untapped/raman.pkl.gz',dataset)
        x = self.zero_one(x)

        # last column of y is dummy column (all zeros, major 'name' is None)
        y = y[:,:-1]
        majors = majors[:-1]

        ux = x.mean(axis=0)
        if remove_mean:
            # mean zero
            x -= ux

        if DropLastDim:
            # remove last column (for simplex)
            y = y[:,:-1]

        # select train-validation split
        cv = KFold(n_splits=n_folds,shuffle=True,random_state=0).split(X=x,y=y)
        if trial is not None:
            for num,split in enumerate(cv):
                if num >= trial:
                    break
        # print('Only '+str(num)+' splits available. Using last split.')
        
        train_idx, valid_idx = split

        x_train = x[train_idx]
        y_train = y[train_idx]

        x_valid = x[valid_idx]
        y_valid = y[valid_idx]

        x_unsup = x
        y_unsup = y

        xy = (x_train, y_train, x_valid, y_valid, x_unsup, y_unsup)
        xy_names = ('x_train', 'y_train', 'x_valid', 'y_valid', 'x_unsup', 'y_unsup')
        print('Data Shapes:')
        for name, d in zip(xy_names,xy):
            print(name,d.shape)

        names = [major.decode('UTF-8') for major in majors]

        colors = cm.rainbow(np.linspace(0,1,len(majors)))

        return xy, ux, waves, names, colors
        
    def plot_current(self, train, params, i, ylim=[0,1], force_ylim=True, fs=24, fs_tick=18):
        samples = train.m.get_fake(64, params['z_dim']).cpu().data.numpy()
        plt.plot(self.waves,samples.T)
        plt.title('Generated Spectra', fontsize=fs)
        plt.xlabel('Channels', fontsize=fs)
        plt.ylabel('Intensities', fontsize=fs)
        plt.tick_params(axis='both', which='major', labelsize=fs_tick)
        if force_ylim:
            plt.gca().set_ylim(ylim)
        plt.savefig(params['saveto']+'samples/samples_{}.png'.format(i))
        plt.close()

    def plot_series(self, np_samples, params, ylim=[0,1], force_ylim=True, fs=24, fs_tick=18):
        np_samples_ = np.array(np_samples)
        cols = len(np_samples_)
        fig = plt.figure(figsize=(2*cols, 2*params['n_viz']))
        for i, samps in enumerate(np_samples_):
            for j, samp in enumerate(samps):
                if j == 0:
                    ax = plt.subplot(params['n_viz'],cols,i+1)
                    plt.title('step %d'%(i*params['viz_every']))
                else:
                    plt.subplot(params['n_viz'],cols,i+j*cols+1, sharex=ax, sharey=ax)
                plt.plot(self.waves, samp)
                if force_ylim:
                    plt.gca().set_ylim(ylim)
                plt.xticks([]); plt.yticks([])
        plt.gcf().tight_layout()
        fig.savefig(params['saveto']+'series.pdf')
        plt.close()

    def plot_real(self, params, ylim=[0,1], force_ylim=True, fs=24, fs_tick=18):
        samples = self.sample(batch_size=self.batch_size)[:64].cpu().data.numpy()
        plt.plot(self.waves,samples.T)
        plt.title('Generated Spectra', fontsize=fs)
        plt.xlabel('Channels', fontsize=fs)
        plt.ylabel('Intensities', fontsize=fs)
        plt.tick_params(axis='both', which='major', labelsize=fs_tick)
        if force_ylim:
            plt.gca().set_ylim(ylim)
        plt.savefig(params['saveto']+'samples_real.png')
        plt.close()

    def sample(self, batch_size, dim=64):
        try:
            samples, atts = next(self.dataiterator).split([self.x_dim, self.att_dim], dim=1)
        except:
            self.dataiterator = iter(self.dataloader)
            samples, atts = next(self.dataiterator).split([self.x_dim, self.att_dim], dim=1)
        return samples.reshape(batch_size, -1)

    def sample_att(self, batch_size, dim=64):
        try:
            samples_atts = next(self.dataiterator)
        except:
            self.dataiterator = iter(self.dataloader)
            samples_atts = next(self.dataiterator)
        return samples_atts


class Generator(Net):
    def __init__(self, input_dim, output_dim, n_hidden=128, n_layer=2, nonlin='leaky_relu'):
        super(Generator, self).__init__()
        
        hidden_fcs = []
        in_dim = input_dim
        for l in range(n_layer):
            hidden_fcs += [nn.Linear(in_dim, n_hidden)]
            in_dim = n_hidden
        self.hidden_fcs = nn.ModuleList(hidden_fcs)
        self.final_fc = nn.Linear(in_dim, output_dim)
        self.layers = hidden_fcs + [self.final_fc]
        if nonlin == 'leaky_relu':
            self.nonlin = F.leaky_relu
        elif nonlin == 'relu':
            self.nonlin = F.relu
        elif nonlin == 'tanh':
            self.nonlin = F.tanh
        elif nonlin == 'sigmoid':
            self.nonlin = torch.sigmoid
        else:
            self.nonlin = lambda x: x

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_hidden = n_hidden

        self.first_forward = True

    def forward(self, x):
        h = x
        if self.first_forward: print('\nGenerator output shapes:', flush=True)
        for hfc in self.hidden_fcs:
            h = self.nonlin(hfc(h))
            if self.first_forward: print(h.shape, flush=True)
        output = self.final_fc(h)
        if self.first_forward: print(output.shape, flush=True)
        self.first_forward = False
        return output


class AttExtractor(Net):
    def __init__(self, input_dim, output_dim, n_hidden=128, n_layer=2, nonlin='leaky_relu'):
        super(AttExtractor, self).__init__()
        
        hidden_fcs = []
        in_dim = input_dim
        for l in range(n_layer):
            hidden_fcs += [nn.Linear(in_dim, n_hidden)]
            in_dim = n_hidden
        self.hidden_fcs = nn.ModuleList(hidden_fcs)
        self.final_fc = nn.Linear(in_dim, output_dim)
        self.layers = hidden_fcs + [self.final_fc]
        if nonlin == 'leaky_relu':
            self.nonlin = F.leaky_relu
        elif nonlin == 'relu':
            self.nonlin = F.relu
        elif nonlin == 'tanh':
            self.nonlin = F.tanh
        elif nonlin == 'sigmoid':
            self.nonlin = torch.sigmoid
        else:
            self.nonlin = lambda x: x

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_hidden = n_hidden

        self.first_forward = True

    def forward(self, x):
        h = x
        if self.first_forward: print('\nAttExtractor output shapes:', flush=True)
        for hfc in self.hidden_fcs:
            h = self.nonlin(hfc(h))
            if self.first_forward: print(h.shape, flush=True)
        output = self.final_fc(h)
        if self.first_forward: print(output.shape, flush=True)
        self.first_forward = False
        return output


class LatExtractor(Net):
    def __init__(self, input_dim, output_dim, n_hidden=128, n_layer=2, nonlin='leaky_relu'):
        super(LatExtractor, self).__init__()
        
        hidden_fcs = []
        in_dim = input_dim
        for l in range(n_layer):
            hidden_fcs += [nn.Linear(in_dim, n_hidden)]
            in_dim = n_hidden
        self.hidden_fcs = nn.ModuleList(hidden_fcs)
        self.final_fc = nn.Linear(in_dim, output_dim)
        self.layers = hidden_fcs + [self.final_fc]
        if nonlin == 'leaky_relu':
            self.nonlin = F.leaky_relu
        elif nonlin == 'relu':
            self.nonlin = F.relu
        elif nonlin == 'tanh':
            self.nonlin = F.tanh
        elif nonlin == 'sigmoid':
            self.nonlin = torch.sigmoid
        else:
            self.nonlin = lambda x: x

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_hidden = n_hidden

        self.first_forward = True

    def forward(self, x):
        h = x
        if self.first_forward: print('\nLatExtractor output shapes:', flush=True)
        for hfc in self.hidden_fcs:
            h = self.nonlin(hfc(h))
            if self.first_forward: print(h.shape, flush=True)
        output = self.final_fc(h)
        if self.first_forward: print(output.shape, flush=True)
        self.first_forward = False
        return output

    def init_weights(self):
        for layer in self.layers:
            nn.init.orthogonal_(layer.weight.data, gain=0.8)


class Discriminator(Net):
    # assumption: output_dim = 1
    def __init__(self, input_dim, n_hidden=128, n_layer=1, nonlin='leaky_relu', quad=False):
        super(Discriminator, self).__init__()
        hidden_fcs = []
        in_dim = input_dim
        for l in range(n_layer):
            hidden_fcs += [nn.Linear(in_dim, n_hidden)]
            in_dim = n_hidden
        self.hidden_fcs = nn.ModuleList(hidden_fcs)
        self.quad = quad
        if quad:
            self.final_fc = nn.Linear(in_dim, in_dim)  #W z + b
        else:
            self.final_fc = nn.Linear(in_dim, 1)
        self.layers = hidden_fcs + [self.final_fc]
        if nonlin == 'leaky_relu':
            self.nonlin = F.leaky_relu
        elif nonlin == 'relu':
            self.nonlin = F.relu
        elif nonlin == 'tanh':
            self.nonlin = F.tanh
        elif nonlin == 'sigmoid':
            self.nonlin = torch.sigmoid
        else:
            self.nonlin = lambda x: x

        self.first_forward = True

    def forward(self,x):
        h = x
        if self.first_forward: print('\nDiscriminator output shapes:', flush=True)
        for hfc in self.hidden_fcs:
            h = self.nonlin(hfc(h))
            if self.first_forward: print(h.shape, flush=True)
        if self.quad:
            output = torch.sum(h*self.final_fc(h),dim=1)
        else:
            output = self.final_fc(h)
        if self.first_forward: print(output.shape, flush=True)
        self.first_forward = False
        return output

    def init_weights(self):
        for layer in self.layers:
            nn.init.orthogonal_(layer.weight.data, gain=0.8)


class Disentangler(Net):
    def __init__(self, input_dim, output_dim, n_hidden=128, n_layer=1, nonlin='leaky_relu'):
        super(Disentangler, self).__init__()
        hidden_fcs = []
        in_dim = input_dim
        for l in range(n_layer):
            hidden_fcs += [nn.Linear(in_dim, n_hidden)]
            in_dim = n_hidden
        self.hidden_fcs = nn.ModuleList(hidden_fcs)
        self.output_dim = output_dim
        self.final_fc = nn.Linear(in_dim, output_dim)
        self.layers = hidden_fcs + [self.final_fc]
        if nonlin == 'leaky_relu':
            self.nonlin = F.leaky_relu
        elif nonlin == 'relu':
            self.nonlin = F.relu
        elif nonlin == 'tanh':
            self.nonlin = F.tanh
        elif nonlin == 'sigmoid':
            self.nonlin = torch.sigmoid
        else:
            self.nonlin = lambda x: x

        self.first_forward = True

    def forward(self,x):
        h = x
        if self.first_forward: print('\nDisentangler output shapes:', flush=True)
        for hfc in self.hidden_fcs:
            h = self.nonlin(hfc(h))
            if self.first_forward: print(h.shape, flush=True)
        output = torch.sigmoid(self.final_fc(h))
        if self.first_forward: print(output.shape, flush=True)
        self.first_forward = False
        return output

    def init_weights(self):
        for layer in self.layers:
            nn.init.orthogonal_(layer.weight.data, gain=0.8)
            layer.bias.data.zero_()
