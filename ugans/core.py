import functools

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Uniform, Normal

from ugans.utils import detach_all, JSD

from IPython import embed

 
class Data(object):
    '''
    Data Generator object. Main functionality is contained in `sample' method.
    '''
    def __init__(self):
        return None

    '''
    Takes batch size as input and returns torch tensor containing batch_size rows and
    num_feature columns
    '''
    def sample(self, batch_size):
        return None

    def sample_att(self, batch_size):
        return None

    def plot_current(self):
        return None

    def plot_series(self):
        return None


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layers = None

    def forward(self,x):
        return None

    def init_weights(self):
        for layer in self.layers:
            nn.init.orthogonal_(layer.weight.data, gain=0.8)
            layer.bias.data.zero_()

    def get_param_data(self):
        return [p.data.detach() for p in self.parameters()]

    def set_param_data(self, data, req_grad=False):
        for p,d in zip(self.parameters(), data):
            if req_grad:
                p.data = d
            else:
                p.data = d.detach()

    def accumulate_gradient(self, grad):
        for p,g in zip(self.parameters(),grad):
            if p.grad is None:
                p.grad = g.detach()
            else:
                p.grad += g.detach()

    def multiply_gradient(self, scalar):
        for p in self.parameters():
            if p.grad is not None:
                p.grad *= scalar

    def zero_grad(self):  # overwriting zero_grad of super because it has a bug
        r"""Sets gradients of all model parameters to zero."""
        for p in self.parameters():
            if p.grad is not None:
                p.grad.detach()
                p.grad.zero_()


class Manager(object):
    def __init__(self, data, G, F_att, F_lat, D, D_dis, params, to_gpu, logger=None):
        self.data = data
        self.G = G
        self.F_att = F_att
        self.F_lat = F_lat
        self.D = D
        self.D_dis = D_dis
        self.mods = (G, F_att, F_lat, D, D_dis)
        self.mod_names = ('G','F_att','F_lat','D','D_dis')
        self.params = params
        self.to_gpu = to_gpu
        if params['pz']:  # True = Uniform
            self.z_rand = Uniform(to_gpu(torch.tensor(0.0)), to_gpu(torch.tensor(1.0)))
        else:  # False = Normal
            self.z_rand = Normal(to_gpu(torch.tensor(0.0)), to_gpu(torch.tensor(1.0)))
        if params['divergence'] == 'JS' or params['divergence'] == 'standard':
            loss = nn.BCEWithLogitsLoss()
            self.criterion = lambda dec, label: -loss(dec, label)
        elif params['divergence'] == 'Wasserstein':
            self.criterion = lambda dec, label: torch.mean(dec*(2.*label-1.))  #loss(dec, label) #torch.sum(dec)  #torch.sum(dec*(2.*label-1.))
        if params['att_type'] == 0:
            self.att_loss = lambda pred, true: torch.mean(-(true*torch.log(pred+1e-10) + (1-true)*torch.log(1-pred+1e-10)))
        elif params['att_type'] == 1:
            m = data.powerfits[:,0]
            b = data.powerfits[:,1]
            self.att_loss = lambda pred, true: torch.mean(np.exp(-m*true-b) * (pred-true)**2.)
        elif params['att_type'] == 2:
            self.att_loss = lambda pred, true: torch.mean(torch.abs(pred-true))
        else:
            self.att_loss = lambda pred, true: 0*torch.mean(pred)
        self.logger = logger
        self.feature_mask = self.load_feature_mask()
        self.feature_means = self.load_feature_means()

    def get_real(self, batch_size):
        return self.to_gpu(self.data.sample(batch_size))

    def get_z(self, batch_size, z_dim):
        z = self.z_rand.sample((batch_size, z_dim))
        return z

    def get_fake(self, batch_size, z_dim):
        return self.G(self.get_z(batch_size, z_dim))

    def get_outputs(self, data):
        outputs = []
        for datum in data:
            if isinstance(datum,np.ndarray):
                datum = self.to_gpu(Variable(torch.from_numpy(datum).float()))
            attributes = self.F_att(datum)
            latents = self.F_lat(datum)
            features = torch.cat([latents, attributes], dim=1)
            if self.feature_means is not None and self.feature_mask is not None:
                features = (1-self.feature_mask)*self.feature_means + self.feature_mask*features
            d_probs = self.D(features).squeeze()
            d_dis_preds = self.D_dis(latents)
            outputs += [(latents, attributes, d_dis_preds, d_probs)]
        return outputs

    def get_decisions(self, data):
        decisions = []
        for datum in data:
            if isinstance(datum,np.ndarray):
                datum = self.to_gpu(Variable(torch.from_numpy(datum).float()))
            features = torch.cat([self.F_z(datum), self.F_att(datum)], dim=1)
            decisions += [self.D(features).squeeze()]
        return decisions

    def get_V(self, batch_size, real_dec=None, fake_dec=None):
        res = []
        if real_dec is not None:
            V_real = self.criterion(real_dec, self.to_gpu(Variable(torch.ones(real_dec.shape[0]))))  # ones = true
            res += [V_real]
        if fake_dec is not None:
            V_fake = self.criterion(fake_dec, self.to_gpu(Variable(torch.zeros(fake_dec.shape[0]))))  # zeros = fake
            res += [V_fake]
            if self.params['divergence'] == 'standard':
                V_fake_mod = -self.criterion(fake_dec, self.to_gpu(Variable(torch.ones(fake_dec.shape[0]))))  # we want to fool, so pretend it's all genuine
                res += [V_fake_mod]
            elif self.params['divergence'] == 'JS' or self.params['divergence'] == 'Wasserstein':
                res += [V_fake]
            else:
                raise NotImplementedError(self.params['divergence'])
        return res

    def get_data_with_atts(self, batch_size):
        if self.params['images']:
            data, atts = torch.split(self.data.sample_att(batch_size), [self.params['c_dim']*self.params['x_dim']**2, self.params['att_dim']], dim=1)
        else:
            data, atts = torch.split(self.data.sample_att(batch_size), [self.params['x_dim'], self.params['att_dim']], dim=1)
        return self.to_gpu(data), self.to_gpu(atts)

    def get_predictions(self, data):
        predictions = []
        for datum in data:
            if isinstance(datum,np.ndarray):
                datum = self.to_gpu(Variable(torch.from_numpy(datum).float()))
            predictions += [self.D_dis(self.F_z(datum))]
        return predictions

    def load_feature_means(self, filepath='', batches=100):
        self.feature_means = None
        if filepath == '':
            filepath = self.params['feature_means']
        if filepath == '':
            return None
        try:
            feature_means = np.load(filepath)
            # handle small feature means (just for attributes, not latents)
            rem = self.params['lat_dim']+self.params['att_dim'] - feature_means.shape[0]
            if rem > 0:
                feature_means = np.concatenate((np.zeros(rem),feature_means))
            feature_means = self.to_gpu(torch.from_numpy(feature_means).float())
        except:
            try:
                print('Computing feature means...', flush=True)
                feature_means = self.compute_feature_means(batches)
                np.save(self.params['feature_means'], feature_means.cpu().data.numpy())
            except:
                print('Process failed. Check feature mean filepath.', flush=True)
                feature_means = None
        return feature_means

    def compute_feature_means(self, batches):
        with torch.no_grad():
            feature_means = self.to_gpu(torch.zeros(self.params['lat_dim']+self.params['att_dim']))
            for b in range(batches):
                real_data = self.get_real(self.params['batch_size'])
                fake_data = self.G(self.get_z(self.params['batch_size'], self.params['z_dim']))
                real_outputs, fake_outputs = self.get_outputs([real_data, fake_data])
                real_feats = torch.cat([real_outputs[0], real_outputs[1]], dim=1)
                fake_feats = torch.cat([fake_outputs[0], fake_outputs[1]], dim=1)
                feats = torch.cat([real_feats, fake_feats], dim=0)
                feature_means += torch.mean(feats, dim=0) / batches
        return feature_means

    def load_feature_mask(self, filepath=''):
        if filepath == '':
            filepath = self.params['feature_mask']
        try:
            feature_mask = np.load(filepath)
            # handle small feature mask (just for attributes, not latents)
            rem = self.params['lat_dim']+self.params['att_dim'] - feature_mask.shape[0]
            if rem > 0:
                feature_mask = np.concatenate((np.ones(rem),feature_mask))
            feature_mask = self.to_gpu(torch.from_numpy(feature_mask).float())
        except:
            feature_mask = self.to_gpu(torch.ones(self.params['lat_dim']+self.params['att_dim']))
        return feature_mask


class Train(object):
    def __init__(self, manager):
        self.m = manager

        optimizers = []
        weights = [mod.parameters() for mod in self.m.mods]
        for net, ps in zip(['gen','att','lat','disc','dis'], weights):
            if self.m.params[net+'_optim'] == 'Adam':
                optimizers += [optim.Adam(ps, lr=self.m.params[net+'_learning_rate'], betas=self.m.params['betas'], eps=self.m.params['epsilon'], weight_decay=self.m.params[net+'_weight_decay'])]
            if self.m.params[net+'_optim'] == 'RMSProp':
                optimizers += [optim.RMSprop(ps, lr=self.m.params[net+'_learning_rate'], weight_decay=self.m.params[net+'_weight_decay'], eps=1e-10, alpha=0.9)]
            if self.m.params[net+'_optim'] == 'SGD':
                optimizers += [optim.SGD(ps, lr=self.m.params[net+'_learning_rate'], momentum=self.m.params[net+'_momentum'], weight_decay=self.m.params[net+'_weight_decay'])]
        self.optimizers = optimizers

        # Note: SimgGD should always be last in maps list
        self.maps = [mp(manager).map for mp in self.m.params['maps']]
        self.cmap = self.compose(*self.maps)  # [f,g] becomes f(g(x))

    def train_op(self, it):
        self.m.G.zero_grad()
        self.m.F_att.zero_grad()
        self.m.F_lat.zero_grad()
        self.m.D.zero_grad()
        self.m.D_dis.zero_grad()

        # 1. Get real data and samples from p(z) to pass to generator
        real_data = self.m.get_real(self.m.params['batch_size'])
        fake_z = self.m.get_z(self.m.params['batch_size'], self.m.params['z_dim'])
        data_att, atts = self.m.get_data_with_atts(self.m.params['batch_size'])

        # 2. Evaluate Map
        mps, losses, norms = detach_all(self.cmap([real_data, self.m.G(fake_z), data_att, atts, it]))
        map_g, map_att, map_lat_gan, map_lat_dis, map_d, map_dis = mps
        map_lat = [a+b for a, b in zip(map_lat_gan, map_lat_dis)]
        losses = [loss.item() for loss in losses]
        norm_g, norm_att, norm_lat_gan, norm_lat_dis, norm_d, norm_dis = norms
        norm_lat = norm_lat_gan + norm_lat_dis
        norms = [norm.item() for norm in (norm_g, norm_att, norm_lat, norm_d, norm_dis)]

        # 3. Accumulate F(x_k)
        self.m.G.accumulate_gradient(map_g) # compute/store map, but don't change params
        self.m.F_att.accumulate_gradient(map_att)
        self.m.F_lat.accumulate_gradient(map_lat)
        self.m.D.accumulate_gradient(map_d)
        self.m.D_dis.accumulate_gradient(map_dis)

        # 4. Update network parameters
        for optimizer in self.optimizers:
            optimizer.step()

        return losses, norms

    @staticmethod
    def compose(*functions):
        '''
        https://mathieularose.com/function-composition-in-python/
        '''
        return functools.reduce(lambda f, g: lambda x: f(g(*x)), functions, lambda x: x)
        

class Map(object):
    def __init__(self, manager):
        self.m = manager

    def map(self, F=None):
        # return [d_map, g_map, V]
        return None, None, None
