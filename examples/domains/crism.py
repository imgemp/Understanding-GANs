import os
import zipfile
import random

import spectral.io.envi as envi

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

datasets = ['./examples/domains/data/CRISM_data_summPar_1/FRT000097E2/FRT000097E2_07_IF166L_TRR3_atcr_sabcondv4_1_Lib11123_1_4_5_l1_gadmm_a_v2_ca_ice_b200_MS',
            './examples/domains/data/CRISM_data_summPar_1/HRL000040FF/HRL000040FF_07_IF183L_TRR3_atcr_sabcondv4_1_Lib11123_1_4_5_l1_gadmm_a_v2_ca_ice_b200_MS']
labelsets = [d+'_2014params' for d in datasets]

# srun -p m40-long --gres=gpu:1 examples/run.sh "examples/args/crism/con/00.txt"


class CRISM(Data):
    def __init__(self, batch_size=128, **kwargs):
        super(CRISM, self).__init__()
        self.batch_size = batch_size
        self.load_crism()
        # Number of workers for dataloader
        workers = 2
        # Create the dataloader
        self.dataloader = torch.utils.data.DataLoader(torch.from_numpy(self.x_att), batch_size=batch_size,
                                                      shuffle=True, num_workers=workers,
                                                      drop_last=True)
        self.dataiterator = iter(self.dataloader)
        print('Number of batches: {}'.format(self.x_att.shape[0] // batch_size), flush=True)


    def load_crism(self):
        x, goodrows = get_np_image(datasets)
        y = get_np_labels(labelsets, goodrows)
        waves = np.linspace(1.02, 2.6, x.shape[1])
        self.x_dim = x.shape[1]
        self.att_dim = y.shape[1]
        self.x_att = np.hstack((x,y)).astype('float32')
        self.waves = waves


    def get_np_image(self, datasets):
        channels = []
        goodrows = []
        xs = []
        for dataset in datasets:
            img = envi.open(dataset+'.hdr', dataset+'.img')
            img_np = np.asarray(img.asarray())
            x = img_np.reshape((-1,img_np.shape[-1]))
            channels += [x.shape[1]]
            nanrows = np.all(np.isnan(x), axis=1)
            print('Removing {:0.2f}% of {:d} rows (all NaN) from {:s}.'.format(nanrows.sum()/x.shape[0]*100,x.shape[0],dataset))
            x = x[~nanrows]
            goodrows += [~nanrows]
            nancols = np.all(np.isnan(x), axis=0)
            start = np.argmin(nancols)
            end = len(nancols)-1-np.argmin(nancols[::-1])
            print('Selecting good channels ({:d}-{:d}) from {:d} total of {:s}.'.format(start,end,x.shape[1],dataset))
            xs += [x[:,start:end+1]]
        if len(set(channels)) > 1:
            channel_str = ','.join([str(ch) for ch in channels])
            raise ValueError('Dataset channels do not align: [{:s}].'.format(channel_str))
        x_joined = np.vstack(xs)
        goodrows = np.hstack(goodrows)
        nanrows = np.any(np.isnan(x_joined), axis=1)
        print('Removing {:0.2f}% of {:d} rows (any NaN) from joined dataset.'.format(nanrows.sum()/x_joined.shape[0]*100,x_joined.shape[0]))
        x_joined = x_joined[~nanrows]
        goodrows[goodrows] = ~nanrows
        return x_joined, goodrows


    def get_np_labels(self, labelsets, goodrows):
        labels = []
        ys = []
        for label in labels:
            img = envi.open(labelset+'.hdr', labelset+'.img')
            img_np = np.asarray(img.asarray())
            y = img_np.reshape((-1,img_np.shape[-1]))
            labels += [y.shape[1]]
            ys += [y]
        if len(set(labels)) > 1:
            label_str = ','.join([str(l) for l in labels])
            raise ValueError('# of labels do not align: [{:s}].'.format(label_str))
        y_joined = np.vstack(ys)
        print('Removing {:0.2f}% of rows (any NaN) from joined label.'.format((1-goodrows.sum()/y_joined.shape[0])*100))
        y_joined = y_joined[goodrows]
        return y_joined


    def plot_current(self, train, params, i, ylim=1, force_ylim=True, fs=24, fs_tick=18):
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

    def plot_series(self, np_samples, params, ylim=1, force_ylim=True, fs=24, fs_tick=18):
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

    def plot_real(self, params, ylim=1, force_ylim=True, fs=24, fs_tick=18):
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
