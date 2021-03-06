import os
import zipfile
import random

import h5py
import spectral.io.envi as envi

import numpy as np

from scipy.spatial.distance import pdist, squareform

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

from ugans.core import Data, Net
from ugans.utils import load_url

import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import seaborn as sns

# # originals FRT000097E2, HRL000040FF
# # ids = [('FRT000047A3','66'), ('FRT00005850','67'), ('FRT000097E2','66'), ('FRT0000CBE5','66')] #, ('FRT00013EBC','66'), ('FRT000161EF','67'), ('HRL000040FF','83'), ('HRL0000C0BA','83')]
# ids = [('FRT000097E2','66'), ('HRL000040FF','83')]
# prefixes = ['./examples/domains/data/CRISM_data_summPar_1/'+direc+'/'+direc+'_07_IF1'+num+'L_TRR3_atcr_sabcondv4_1_Lib11123_1_4_5_l1_gadmm_a_v2_ca_ice_b200_MS' for direc, num in ids]
# # postfix = '_CR'
# postfix = ''
# datasets = [p+postfix for p in prefixes]
# labelsets = [p+'_2014params' for p in prefixes]

datasets = ['./examples/domains/data/store_Composite_summParam.h5']
labelsets = datasets

# srun -p m40-long --gres=gpu:1 examples/run.sh "examples/args/crism/con/00.txt"


class CRISM(Data):
    def __init__(self, batch_size=128, num_labels=None, slice_size=int(506898), binarize_y=True, perc=95., **kwargs):
        super(CRISM, self).__init__()
        self.batch_size = batch_size
        self.num_labels = num_labels
        self.slice_size = slice_size
        self.slice_idx = 0
        self.binarize_y = binarize_y
        self.perc = perc
        self.loaded_once = False
        self.load_crism(num_labels)
        self.loaded_once = True
        # Number of workers for dataloader
        workers = 2
        # Create the dataloader
        self.dataloader = torch.utils.data.DataLoader(torch.from_numpy(self.x_att), batch_size=batch_size,
                                                      shuffle=True, drop_last=True)
        self.dataiterator = iter(self.dataloader)
        self.F_att_eval = None
        self.mica_library = None
        print('Number of batches: {}'.format(self.x_att.shape[0] // batch_size), flush=True)

    def reload(self):
        del self.dataloader
        del self.dataiterator
        del self.y_real
        del self.x_att
        torch.cuda.empty_cache()
        self.load_crism(self.num_labels)
        # Number of workers for dataloader
        workers = 2
        # Create the dataloader
        self.dataloader = torch.utils.data.DataLoader(torch.from_numpy(self.x_att), batch_size=self.batch_size,
                                                      shuffle=True, drop_last=True)
        self.dataiterator = iter(self.dataloader)
        return

    def load_crism(self, num_labels, normalize=True):
        x, goodrows = self.get_np_image(datasets)
        x = self.zero_one_x_ind(x)  # if switch back to zero_one_x_ind, remember to add sigmoid back to generator
        # x -= 0.5
        y, names, new_goodrows = self.get_np_labels(labelsets, goodrows, normalize)
        x = x[new_goodrows]
        if not self.binarize_y and normalize:
            scaler = StandardScaler()
            y = scaler.fit_transform(y)
            # y = self.zero_one_y(y)
        self.y_real = np.array(y).astype('float32')

        self.slice_idx += self.slice_size

        if not os.path.isfile('./examples/masks/crism_arun/att_means.npy'):
            np.save('./examples/masks/crism_arun/att_means.npy', self.y_real.mean(axis=0))
        if isinstance(num_labels, int):
            y = y[:,:num_labels]
        waves = np.linspace(1.02, 2.6, x.shape[1])
        x_dim = x.shape[1]
        if x_dim != 240:
            x = x[:,:240]
            waves = waves[:240]
            if not self.loaded_once: print('Temporary hack for even dims for conv, xdims: {:d}-->{:d}.'.format(x_dim, x.shape[1]))
        self.x_dim = x.shape[1]
        self.att_dim = y.shape[1]
        self.x_att = np.hstack((x,y)).astype('float32')
        self.waves = waves
        self.att_names = names

    def zero_one_x(self,x):
        mn, mx = np.min(x), np.max(x)
        if mx > mn:
            return (x-mn)/(mx-mn)
        else:
            return x

    def zero_one_x_ind(self,x):
        '''expecting normalization along columns'''
        x -= np.min(x,axis=1)[:,None]
        return (x.T/np.ptp(x,axis=1)).T

    def max_one_x_ind(self,x):
        '''expecting normalization along columns'''
        return 1e3 * x / np.max(x, axis=1, keepdims=True)

    def zero_one_y(self,y):
        y -= np.min(y,axis=0)[None]
        return (y/np.ptp(y,axis=0))

    def fnScaleMICAEM(self, mica_data):
        'First subtract 1 and set everything at 1 to 0'
        mica_data_shft = mica_data - 1
        'divide by the minimum in each row'
        mica_data_min = mica_data_shft.min(axis=1) 
        
        mica_data_scale = np.zeros(mica_data.shape)

        'Scale each endmember and create plots to see what it looks like'
        for ii in range(mica_data.shape[0]):
            temp = mica_data_shft[ii,:]/mica_data_min[ii]
            mica_data_scale[ii,:] = temp * -0.02
            
        return (mica_data_scale + 1)

    def get_np_image(self, datasets, scale_mica=True):
        channels = []
        shared_range = [0,np.infty]
        goodrows = []
        xs = []
        start_row = self.slice_idx
        end_row = self.slice_idx + self.slice_size
        for dataset in datasets:
            if 'h5' not in dataset:
                img = envi.open(dataset+'.hdr', dataset+'.img')
                img_np = np.asarray(img.asarray())
                x = img_np.reshape((-1,img_np.shape[-1]))
            else:
                with h5py.File(dataset, 'r') as f:
                    table = f['CRISM_MS']['table'].value
                    # x = np.stack([s[1] for s in table]).astype('float32')
                    # HACK - ONLY WORKS FOR store_Composite_summParam.h5 dataset
                    inds = np.load('./examples/domains/data/shuffled_inds.npy')
                    if start_row >= inds.shape[0]:
                        start_row = self.slice_idx = 0
                        end_row = self.slice_idx + self.slice_size
                    end_row = min(end_row, inds.shape[0])
                    x = np.stack([table[inds[i]][1] for i in range(start_row, end_row)]).astype('float32')
                    # x = x[inds]
            channels += [x.shape[1]]
            nanrows = np.all(np.isnan(x), axis=1)
            if not self.loaded_once: print('Removing {:0.2f}% of {:d} rows (all NaN) from {:s}.'.format(nanrows.sum()/x.shape[0]*100,x.shape[0],dataset))
            x = x[~nanrows]
            goodrows += [~nanrows]
            nancols = np.all(np.isnan(x), axis=0)
            start = np.argmin(nancols)
            end = len(nancols)-1-np.argmin(nancols[::-1])
            if not self.loaded_once: print('Selecting good channels ({:d}-{:d}) from {:d} total of {:s}.'.format(start,end,x.shape[1],dataset))
            shared_range[0] = max(shared_range[0], start)
            shared_range[1] = min(shared_range[1], end)
            xs += [x]
        for i in range(len(xs)):
            xs[i] = xs[i][:,shared_range[0]:shared_range[1]+1]
        if len(set(channels)) > 1:
            channel_str = ','.join([str(ch) for ch in channels])
            raise ValueError('Dataset channels do not align: [{:s}].'.format(channel_str))

        x_joined = np.vstack(xs)
        goodrows = np.hstack(goodrows)
        nanrows = np.any(np.isnan(x_joined), axis=1)
        if not self.loaded_once: print('Removing {:0.2f}% of {:d} rows (any NaN) from joined dataset.'.format(nanrows.sum()/x_joined.shape[0]*100,x_joined.shape[0]))
        x_joined = x_joined[~nanrows]
        goodrows[goodrows] = ~nanrows
        # call fnScaleMICAEM on x_joined here
        if scale_mica:
            x_joined = self.fnScaleMICAEM(x_joined)

        # if start_row >= x_joined.shape[0]:
        #     start_row = self.slice_idx = 0
        #     end_row = self.slice_idx + self.slice_size
        # end_row = min(end_row, x_joined.shape[0])
        # x_joined = x_joined[start_row:end_row,:]

        return x_joined, goodrows

    def get_np_labels(self, labelsets, goodrows, normalize=True):
        labels = []
        ys = []
        names = []
        start_row = self.slice_idx
        end_row = self.slice_idx + self.slice_size
        for labelset in labelsets:
            if 'h5' not in labelset:
                img = envi.open(labelset+'.hdr', labelset+'.img')
                img_np = np.asarray(img.asarray())
                y = img_np.reshape((-1,img_np.shape[-1]))
                names = img.metadata['band names']
            else:
                with h5py.File(labelset, 'r') as f:
                    table = f['CRISM_summParam']['table'].value
                    # y = np.stack([s[1] for s in table]).astype('float32')
                    # HACK - ONLY WORKS FOR store_Composite_summParam.h5 dataset
                    inds = np.load('./examples/domains/data/shuffled_inds.npy')
                    # y = y[inds]
                    if start_row >= inds.shape[0]:
                        start_row = self.slice_idx = 0
                        end_row = self.slice_idx + self.slice_size
                    end_row = min(end_row, inds.shape[0])
                    y = np.stack([table[inds[i]][1] for i in range(start_row, end_row)]).astype('float32')
                    bad_atts = np.any(y==65535, axis=0)
                    if not self.loaded_once: print('Removing {:d} columns (65535) from joined labelset.'.format(np.sum(bad_atts)))
                    y = y[:, ~bad_atts]
                # names = [str(_) for _ in range(y.shape[1])]  # hack for now, where are the names?
                # names = np.load('./examples/domains/data/att_names_first_26.npy')  # what are the last 20?
                names = ["BDI1000IR", "OLINDEX3", "R1330", "BD1300", "LCPINDEX2", "HCPINDEX2", "VAR", "ISLOPE1",
                         "BD1400", "BD1435", "BD1500_2", "ICER1_2", "BD1750_2", "BD1900_2", "BD1900R2", "BDI2000",
                         "BD2100_2", "BD2165", "BD2190", "MIN2200", "BD2210_2", "D2200", "BD2230", "BD2250",
                         "MIN2250", "BD2265", "BD2290", "D2300", "BD2355", "SINDEX2", "ICER2_2", "MIN2295_2480",
                         "MIN2345_2537", "BD2500_2", "BD3000", "BD3100", "BD3200", "BD3400_2", "CINDEX2", "BD2600",
                         "IRR2", "IRR3", "R1080", "R1506", "R2529", "R3920"]
                names = list(np.array(names)[~bad_atts])
            labels += [y.shape[1]]
            ys += [y]
        if len(set(labels)) > 1:
            label_str = ','.join([str(l) for l in labels])
            raise ValueError('# of labels do not align: [{:s}].'.format(label_str))
        y_joined = np.vstack(ys)
        if not self.loaded_once: print('Removing {:0.2f}% of rows (any NaN) from joined labelset.'.format((1-goodrows.sum()/y_joined.shape[0])*100))
        y_joined = y_joined[goodrows]
        y_joined = np.clip(y_joined, 0., np.inf)

        if self.binarize_y:
            y_joined = (y > np.percentile(y, self.perc, axis=0)[None]).astype(float)
            self.prep_hist_binary(y_joined)
            new_goodrows = goodrows
        else:
            new_goodrows = self.prep_hist(y_joined, normalize)
            y_joined = y_joined[new_goodrows]

        # if start_row >= y_joined.shape[0]:
        #     start_row = self.slice_idx = 0
        #     end_row = self.slice_idx + self.slice_size
        # end_row = min(end_row, y_joined.shape[0])
        # y_joined = y_joined[start_row:end_row,:]

        return y_joined, names, new_goodrows

    def prep_hist(self, y, normalize=True):
        perc = 0  # 95
        print('percentile', flush=True)
        print(np.percentile(y, perc, axis=0)[None], flush=True)
        print('precentile.shape', flush=True)
        p = np.percentile(y, perc, axis=0)[None]
        print(p.shape, flush=True)
        insig = np.all(y <= np.percentile(y, perc, axis=0)[None], axis=1)
        valid = y <= np.percentile(y, perc, axis=0)[None]
        print('valid.shape', flush=True)
        print(valid.shape, flush=True)
        print('insig.shape', flush=True)
        print(insig.shape, flush=True)
        print('insig', flush=True)
        print(insig, flush=True)
        print('number no significant atts', flush=True)
        print(np.sum(insig), flush=True)
        print('percentage no significant atts', flush=True)
        print(100. * np.mean(insig), flush=True)
        print('total', flush=True)
        print(y.shape[0], flush=True)
        y = y[~insig]

        if normalize:
            scaler = StandardScaler()
            y = scaler.fit_transform(y)
        counts = []
        bins = []
        powerfits = []
        stds = np.std(y,axis=0)
        mins = np.min(y,axis=0)
        maxs = np.max(y,axis=0)
        for col in range(y.shape[1]):
            co, bi = np.histogram(y[:,col], bins=50, density=1)
            # note sum of co = bins
            counts += [co]
            bins += [bi]
            # fit power law
            X = np.stack([bi[:-1], np.ones(len(bi)-1)], axis=1)
            positive = (co > 0)
            X = X[positive]
            co = co[positive]
            # w = np.dot(np.linalg.pinv(np.dot(X.T, X) + 1e-8*np.eye(2)), np.dot(X.T, np.log(co)))
            # print(w, np.dot(X.T, X), np.linalg.pinv(np.dot(X.T, X) + 1e-8*np.eye(2)))
            w = np.linalg.lstsq(X, np.log(co))[0]
            powerfits += [w]
            print(1./co.min(), flush=True)
        # print(len(powerfits))
        # label_stats = np.load('./examples/domains/data/label_stats.npz')
        np.savez_compressed('./examples/domains/data/label_stats_thresholded.npz',
            counts=counts, bins=bins, mins=mins, maxs=maxs, stds=stds, powerfits=powerfits)
        self.ycounts = counts
        self.ybins = bins
        self.ymins = mins
        self.ymaxs = maxs
        self.ystds = stds
        self.powerfits = np.stack(powerfits, axis=0)
        # self.ycounts = label_stats['counts']
        # self.ybins = label_stats['bins']
        # self.ymins = label_stats['mins']
        # self.ymaxs = label_stats['maxs']
        # self.ystds = label_stats['stds']
        return ~insig

    def prep_hist_binary(self, y):
        counts = []
        bins = []
        powerfits = []
        stds = np.std(y,axis=0)
        mins = np.min(y,axis=0)
        maxs = np.max(y,axis=0)
        for col in range(y.shape[1]):
            co, bi = np.histogram(y[:,col], bins=2, density=1)
            # note sum of co = bins
            counts += [co/2.]
            bins += [bi]
        # label_stats = np.load('./examples/domains/data/label_stats.npz')
        np.savez_compressed('./examples/domains/data/label_stats_thresholded_binary.npz',
            counts=counts, bins=bins, mins=mins, maxs=maxs, stds=stds)
        self.ycounts = counts
        self.ybins = bins
        self.ymins = mins
        self.ymaxs = maxs
        self.ystds = stds
        # self.ycounts = label_stats['counts']
        # self.ybins = label_stats['bins']
        # self.ymins = label_stats['mins']
        # self.ymaxs = label_stats['maxs']
        # self.ystds = label_stats['stds']
        return None

    def plot_att_hists2(self, params, i=0, y2=None):
        if y2 is not None:
            stds2 = np.std(y2,axis=0)
        # TODO(imgemp): create hist of real data at __init__ (plot memory < actual data)
        plt.clf()
        fig, ax = plt.subplots(10,4, figsize=(22,10))
        if self.binarize_y:
            log = False
        else:
            log = True
        for r in range(10):
            for c in range(4):
                if r*4+c < len(self.ybins):
                    n, bins, _ = ax[r,c].hist(self.ybins[r*4+c][:-1], bins=self.ybins[r*4+c], weights=self.ycounts[r*4+c], log=log, color='b', alpha=0.5)
                    if not self.binarize_y:
                        ax[r,c].plot(self.ybins[r*4+c][:-1], np.exp(self.powerfits[r*4+c][0]*self.ybins[r*4+c][:-1]+self.powerfits[r*4+c][1]), color='b', lw=4)
                    if y2 is not None:
                        if self.binarize_y:
                            p_pos = y2[:,r*4+c].mean()
                            ax[r,c].bar(x=[0.25, 0.75], height=[1.-p_pos, p_pos], width=0.5, color='r', alpha=0.5)
                        else:
                            ax[r,c].hist(y2[:,r*4+c], bins=bins, density=1, log=log, color='r', alpha=0.5)
                    ax[r,c].set_ylabel(str(r*4+c))
                    title = r'{:s}: {:.3f}$\sigma$'.format(self.att_names[r*4+c], self.ystds[r*4+c])
                    if self.binarize_y and y2 is not None:
                        title += ' ptrue={:.2f},p={:.2f}'.format(self.ycounts[r*4+c][1], y2[:,r*4+c].mean())
                    ax[r,c].set_title(title)
                    ax[r,c].tick_params(left=False,bottom=True,right=False,top=False)
                    if self.binarize_y:
                        mn = 0.
                        mx = 1.
                    else:
                        mn = min(self.ymins[r*4+c],y2[:,r*4+c].min())
                        mx = max(self.ymaxs[r*4+c],y2[:,r*4+c].max())
                    ax[r,c].set_xticks([mn,mx])
                    ax[r,c].set_xticklabels([mn,mx])
                    ymax = np.max(self.ycounts[r*4+c])
                    ax[r,c].set_yticks([ymax])
                    ax[r,c].set_yticklabels([np.around(ymax, decimals=1)])
                    # ax[r,c].set_yticklabels([])
        plt.title('Blue is Real, Hists are Log scale')
        fig.tight_layout()
        plt.savefig(params['saveto']+'hists/att_hist_{}.png'.format(i))
        plt.close()

    def plot_att_hists(self, params, i=0, y2=None):
        y = self.y_real
        stds = np.std(y,axis=0)
        if y2 is not None:
            if y2.shape[1] != y.shape[1]:
                print(y2.shape)
                print(y.shape)
                return
            stds2 = np.std(y2,axis=0)
        # TODO(imgemp): create hist of real data at __init__ (plot memory < actual data)
        plt.clf()
        fig, ax = plt.subplots(10,4, figsize=(22,10))
        for r in range(10):
            for c in range(4):
                if r*4+c < y.shape[1]:
                    n, bins, _ = ax[r,c].hist(y[:,r*4+c], bins=50, density=1, color='b', alpha=0.5)
                    if y2 is not None:
                        ax[r,c].hist(y2[:,r*4+c], bins=bins, density=1, color='r', alpha=0.5)
                    ax[r,c].set_ylabel(str(r*4+c))
                    ax[r,c].set_title(r'{:s}: {:.3f}$\sigma$'.format(self.att_names[r*4+c], stds[r*4+c]))
                    ax[r,c].tick_params(left=False,bottom=True,right=False,top=False)
                    mn = min(y[:,r*4+c].min(),y2[:,r*4+c].min())
                    mx = max(y[:,r*4+c].max(),y2[:,r*4+c].max())
                    ax[r,c].set_xticks([mn,mx])
                    ax[r,c].set_xticklabels([mn,mx])
                    ax[r,c].set_yticklabels([])
        fig.tight_layout()
        plt.savefig(params['saveto']+'hists/att_hist_{}.png'.format(i))
        plt.close()

    def plot_current(self, train, params, i, ylim=[0,1], force_ylim=True, fs=24, fs_tick=18, opt=False):
        if opt:
            samples = torch.cat([train.m.get_fake(64, params['z_dim']) for i in range(10)], dim=0)
            atts = train.m.F_att(samples).cpu().data.numpy()
            dists = squareform(pdist(atts, 'cityblock'))
            dists[range(640),range(640)] = 1.
            sorted_degrees = np.argsort((1/dists).mean(axis=1))
            samples = samples.cpu().data.numpy()[sorted_degrees[:64]]
        else:
            samples = train.m.get_fake(64, params['z_dim']).cpu().data.numpy()
        plt.plot(self.waves,samples.T)
        plt.title('Generated Spectra', fontsize=fs)
        plt.xlabel('Channels', fontsize=fs)
        plt.ylabel('Intensities', fontsize=fs)
        plt.tick_params(axis='both', which='major', labelsize=fs_tick)
        plt.grid(True, axis='x')
        if force_ylim:
            plt.gca().set_ylim(ylim)
        plt.savefig(params['saveto']+'samples/samples_{}.png'.format(i))
        plt.close()
        samples = train.m.get_fake(10000, params['z_dim'])
        if self.F_att_eval is None:
            atts = train.m.F_att(samples).cpu().data.numpy()
        else:
            atts = self.F_att_eval(samples).cpu().data.numpy()
        self.plot_att_hists2(params, i=i, y2=atts)
        self.plot_grouped_by_mica(train, params, i=i, force_ylim=force_ylim)
        self.plot_training_hist(train, params, i=i)

    def plot_series(self, np_samples, params, ylim=[0,1], force_ylim=True, fs=24, fs_tick=18, filename='series'):
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
                plt.grid(True, axis='x')
                plt.xticks([]); plt.yticks([])
        plt.gcf().tight_layout()
        fig.savefig(params['saveto']+filename+'.pdf')
        plt.close()

    def plot_real(self, params, ylim=[0,1], force_ylim=True, fs=24, fs_tick=18, opt=False):
        if opt:
            samples = []
            atts = []
            for i in range(10):
                samps, ats = torch.split(self.sample_att(self.batch_size), [params['x_dim'], params['att_dim']], dim=1)
                samples += [samps[:64].cpu().data.numpy()]
                atts += [ats[:64].cpu().data.numpy()]
            atts = np.vstack(atts)
            samples = np.vstack(samples)
            dists = squareform(pdist(atts, 'cityblock'))
            dists[range(640),range(640)] = 1.
            sorted_degrees = np.argsort((1/dists).mean(axis=1))
            samples = samples[sorted_degrees[:64]]
        else:
            samples = self.sample(batch_size=self.batch_size)[:64].cpu().data.numpy()
        plt.plot(self.waves,samples.T)
        plt.title('Generated Spectra', fontsize=fs)
        plt.xlabel('Channels', fontsize=fs)
        plt.ylabel('Intensities', fontsize=fs)
        plt.tick_params(axis='both', which='major', labelsize=fs_tick)
        plt.grid(True, axis='x')
        if force_ylim:
            plt.gca().set_ylim(ylim)
        plt.savefig(params['saveto']+'samples_real.png')
        plt.close()
        self.plot_series(np.split(samples, 8), {'n_viz':8, 'viz_every':1, 'saveto':params['saveto']}, filename='grid_real', force_ylim=force_ylim)

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

    def load_mica_library(self, train):
        if self.mica_library is None:
            # sliName = './examples/domains/data/CRISM_data_summPar_1/UMass_redMICA_CR_enhanced.sli'
            # sliHdrName = './examples/domains/data/CRISM_data_summPar_1/UMass_redMICA_CR_enhanced.sli.hdr'
            # subset = (4,244)
            sliName = './examples/domains/data/CRISM_data_summPar_1/yukiMicaNum.sli'
            sliHdrName = './examples/domains/data/CRISM_data_summPar_1/yukiMicaNum.hdr'
            sliHdr = envi.read_envi_header(sliHdrName)
            endMem_Name = sliHdr['spectra names']
            not_hematite = np.array([(name != 'HEMATITE') for name in endMem_Name])
            self.mica_names = list(np.array(endMem_Name)[not_hematite])
            subset = (0,240)
            micaSLI = envi.open(sliHdrName, sliName)
            mica_dataRed = micaSLI.spectra[not_hematite, :]
            mica_dataRed = self.fnScaleMICAEM(mica_dataRed[:, subset[0]:subset[1]]).astype('float32')
            mica_dataRed = self.zero_one_x_ind(mica_dataRed)
            # mica_dataRed -= 0.5
            self.mica_library = train.m.to_gpu(torch.from_numpy(mica_dataRed))

    def plot_grouped_by_mica(self, train, params, i, ylim=[0,1], force_ylim=True, fs=24, fs_tick=18):
        # if mica_library is not loaded, load it
        self.load_mica_library(train)
        # compute mica features
        mica_attributes = train.m.F_att(self.mica_library)
        mica_latents = train.m.F_lat(self.mica_library)
        self.mica_features = torch.cat([mica_latents, mica_attributes], dim=1).cpu().data.numpy()
        # generate samples and compute their features
        samples = torch.cat([train.m.get_fake(64, params['z_dim']) for i in range(10)], dim=0)
        attributes = train.m.F_att(samples)
        latents = train.m.F_lat(samples)
        features = torch.cat([latents, attributes], dim=1).cpu().data.numpy()
        # compute cosine similarity
        similarity = cosine_similarity(features, self.mica_features)
        # matches = argmax cosine similarity
        matches = np.argmax(similarity, axis=1)
        # group spectra by matches
        groups = {}
        for idx, match in enumerate(matches):
            if match not in groups:
                groups[match] = set([(idx, similarity[idx][match])])
            else:
                groups[match].add((idx, similarity[idx][match]))
        # for each class in generated spectra:
        # plot spectra and save plot with filename as mica match
        samples = samples.cpu().data.numpy()
        # sim_min = np.cos(np.pi/4.)
        sim_min = max(np.cos(np.pi/4.), np.min(np.max(similarity, axis=1)))
        sim_max = np.max(similarity)
        for endmember, sample_idxs in groups.items():
            n = 0
            avg_sim = 0.
            for idx, sim in list(sample_idxs):
                if sim >= sim_min:
                    # alpha = np.clip(0.5 * (sim + 1.), 0., 1.)
                    sim_norm = (sim - sim_min) / (sim_max - sim_min)
                    alpha = np.clip(sim_norm, 0., 1.)
                    plt.plot(self.waves, samples[idx], '--', color=cm.jet(alpha))
                    avg_sim += sim
                    n += 1
            if n >= 1:
                avg_sim /= float(n)
                plt.plot(self.waves, self.mica_library[endmember].cpu().data.numpy(), 'k-')
                plt.title('{:s}: avg_sim={:1.4f} global_min/max={:1.4f}/{:1.4f}'.format(self.mica_names[endmember], avg_sim, sim_min, sim_max), fontsize=fs_tick-6)
                plt.xlabel('Channels', fontsize=fs)
                plt.ylabel('Intensities', fontsize=fs)
                plt.tick_params(axis='both', which='major', labelsize=fs_tick)
                plt.grid(True, axis='x')
                if force_ylim:
                    plt.gca().set_ylim(ylim)
                filename = ''.join([c for c in self.mica_names[endmember] if c.isalpha() or c.isdigit()]).rstrip()
                plt.savefig(params['saveto']+'mica/{}_{}.png'.format(i, filename))
                plt.close()

    def plot_training_hist(self, train, params, i, ylim=[0,1], force_ylim=True, fs=24, fs_tick=18):
        # if mica_library is not loaded, load it
        self.load_mica_library(train)
        # compute mica features
        mica_attributes = train.m.F_att(self.mica_library)
        mica_latents = train.m.F_lat(self.mica_library)
        self.mica_features = torch.cat([mica_latents, mica_attributes], dim=1).cpu().data.numpy()
        # generate samples and compute their features
        if not self.loaded_once: print('computing features for training set...')
        # n_batches = self.x_att.shape[0] // params['batch_size']
        n_batches = 400
        samples = torch.cat([train.m.get_real(params['batch_size']) for i in range(n_batches)], dim=0)
        attributes = train.m.F_att(samples)
        latents = train.m.F_lat(samples)
        features = torch.cat([latents, attributes], dim=1).cpu().data.numpy()
        # compute cosine similarity
        if not self.loaded_once: print('computing cosine similarity...')
        similarity = cosine_similarity(features, self.mica_features)
        # matches = argmax cosine similarity
        matches = np.argmax(similarity, axis=1)
        num_matches = len(set(list(matches)))
        if not self.loaded_once: print('done computing.')

        fig, ax = plt.subplots()
        bins = np.arange(0, len(self.mica_names)+1)
        n, _, _ = plt.hist(matches, bins=bins, density=True, color='b', alpha=1.)
        ax.set_ylabel('counts')
        ax.set_title('training endmember histogram ({:1.0f}k samples, {:d}/{:d} classes)'.format(matches.shape[0]/1000, num_matches, len(self.mica_names)))
        ax.set_xticks(np.arange(0,len(self.mica_names)))
        labels = []
        for i, name in enumerate(self.mica_names):
            labels += ['('+str(np.sum(matches==i))+') '+name]
        ax.set_xticklabels(labels, rotation=90)
        fig.tight_layout()
        plt.savefig(params['saveto']+'train_hists/train_hist_{}.png'.format(i))
        plt.close()


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


class AttExtractorBin(Net):
    def __init__(self, input_dim, output_dim, n_hidden=128, n_layer=2, nonlin='leaky_relu'):
        super(AttExtractorBin, self).__init__()

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
        output = torch.sigmoid(self.final_fc(h))
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

def shuffle_inds(dataset):
    f = h5py.File(dataset, 'r')
    table = f['CRISM_chosenPix']['table'].value
    keys = set()
    for i, row in enumerate(table):
        image = row[1][0].decode("utf-8")
        keys.add(image)
    images = dict(zip(keys, [[] for _ in range(len(keys))]))  # 25 images for store_Composite_summParam.h5
    for i, row in enumerate(temp):
        image = row[1][0].decode("utf-8")
        images[image].append(i)
    quarters = [[], [], [], []]
    for k, v in images.items():
        vcopy = list(v)
        np.random.shuffle(vcopy)
        splits = np.array_split(vcopy, 4)
        for i in range(len(splits)):
            quarters[i] = quarters[i] + list(splits[i])
    shuffled_inds = []
    for quarter in quarters:
        shuffled_inds += quarter
    return shuffled_inds
