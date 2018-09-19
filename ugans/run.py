import os
import psutil
import shutil
import argparse
import datetime
import pickle
import numpy as np
import torch
import time
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import sys
sys.path.append('../')

from ugans.core import Manager
from ugans.utils import gpu_helper, save_weights

from tqdm import tqdm

from IPython import embed


process = psutil.Process(os.getpid())

def parse_params():
    parser = argparse.ArgumentParser(description='Understanding GANs in PyTorch')
    parser.add_argument('-dom','--domain', type=str, default='circles', help='domain to run', required=False)
    parser.add_argument('-desc','--description', type=str, default='', help='description for the experiment', required=False)
    parser.add_argument('-bs','--batch_size', type=int, default=512, help='batch_size for training', required=False)
    parser.add_argument('-div','--divergence', type=str, default='JS', help='divergence measure, i.e. V, for training', required=False)
    
    parser.add_argument('-g_opt','--gen_optim', type=str, default='RMSProp', help='generator training algorithm', required=False)
    parser.add_argument('-g_lr','--gen_learning_rate', type=float, default=1e-4, help='generator learning rate', required=False)
    parser.add_argument('-g_l2','--gen_weight_decay', type=float, default=0., help='generator weight decay', required=False)
    parser.add_argument('-g_nh','--gen_n_hidden', type=int, default=128, help='# of hidden units for generator', required=False)
    parser.add_argument('-g_nl','--gen_n_layer', type=int, default=2, help='# of hidden layers for generator', required=False)
    parser.add_argument('-g_nonlin','--gen_nonlinearity', type=str, default='relu', help='type of nonlinearity for generator', required=False)
    
    parser.add_argument('-att_opt','--att_optim', type=str, default='RMSProp', help='attribute extractor training algorithm', required=False)
    parser.add_argument('-att_lr','--att_learning_rate', type=float, default=1e-4, help='attribute extractor learning rate', required=False)
    parser.add_argument('-att_l2','--att_weight_decay', type=float, default=0., help='attribute extractor weight decay', required=False)
    parser.add_argument('-att_nh','--att_n_hidden', type=int, default=128, help='# of hidden units for attribute extractor', required=False)
    parser.add_argument('-att_nl','--att_n_layer', type=int, default=1, help='# of hidden layers for attribute extractor', required=False)
    parser.add_argument('-att_nonlin','--att_nonlinearity', type=str, default='relu', help='type of nonlinearity for attribute extractor', required=False)

    parser.add_argument('-lat_opt','--lat_optim', type=str, default='RMSProp', help='latent extractor training algorithm', required=False)
    parser.add_argument('-lat_lr','--lat_learning_rate', type=float, default=1e-4, help='latent extractor learning rate', required=False)
    parser.add_argument('-lat_l2','--lat_weight_decay', type=float, default=0., help='latent extractor weight decay', required=False)
    parser.add_argument('-lat_nh','--lat_n_hidden', type=int, default=128, help='# of hidden units for latent extractor', required=False)
    parser.add_argument('-lat_nl','--lat_n_layer', type=int, default=1, help='# of hidden layers for latent extractor', required=False)
    parser.add_argument('-lat_nonlin','--lat_nonlinearity', type=str, default='relu', help='type of nonlinearity for latent extractor', required=False)

    parser.add_argument('-d_opt','--disc_optim', type=str, default='RMSProp', help='discriminator training algorithm', required=False)
    parser.add_argument('-d_lr','--disc_learning_rate', type=float, default=1e-4, help='discriminator learning rate', required=False)
    parser.add_argument('-d_l2','--disc_weight_decay', type=float, default=0., help='discriminator weight decay', required=False)
    parser.add_argument('-d_nh','--disc_n_hidden', type=int, default=128, help='# of hidden units for discriminator', required=False)
    parser.add_argument('-d_nl','--disc_n_layer', type=int, default=1, help='# of hidden layers for discriminator', required=False)
    parser.add_argument('-d_nonlin','--disc_nonlinearity', type=str, default='relu', help='type of nonlinearity for discriminator', required=False)
    parser.add_argument('-d_quad','--disc_quadratic_layer', type=lambda x: (str(x).lower() == 'true'), default=False, help='whether to use a quadratic final layer', required=False)

    parser.add_argument('-dis_opt','--dis_optim', type=str, default='RMSProp', help='disentangler training algorithm', required=False)
    parser.add_argument('-dis_lr','--dis_learning_rate', type=float, default=1e-4, help='disentangler learning rate', required=False)
    parser.add_argument('-dis_l2','--dis_weight_decay', type=float, default=0., help='disentangler weight decay', required=False)
    parser.add_argument('-dis_nh','--dis_n_hidden', type=int, default=128, help='# of hidden units for disentangler', required=False)
    parser.add_argument('-dis_nl','--dis_n_layer', type=int, default=1, help='# of hidden layers for disentangler', required=False)
    parser.add_argument('-dis_nonlin','--dis_nonlinearity', type=str, default='relu', help='type of nonlinearity for disentangler', required=False)

    parser.add_argument('-betas','--betas', type=float, nargs=2, default=(0.5,0.999), help='beta params for Adam', required=False)
    parser.add_argument('-eps','--epsilon', type=float, default=1e-8, help='epsilon param for Adam', required=False)
    parser.add_argument('-mx_it','--max_iter', type=int, default=100001, help='max # of training iterations', required=False)
    parser.add_argument('-viz_every','--viz_every', type=int, default=1000, help='skip viz_every iterations between plotting current results', required=False)
    parser.add_argument('-series_every','--series_every', type=int, default=25000, help='skip series_every iterations between plotting series plot', required=False)
    parser.add_argument('-w_every','--weights_every', type=int, default=25000, help='skip weights_every iterations between saving weights', required=False)
    parser.add_argument('-n_viz','--n_viz', type=int, default=8, help='number of samples for series plot', required=False)
    
    parser.add_argument('-zdim','--z_dim', type=int, default=256, help='dimensionality of p(z) - unit normal', required=False)
    parser.add_argument('-xdim','--x_dim', type=int, default=2, help='dimensionality of p(x) - data distribution', required=False)
    parser.add_argument('-latdim','--lat_dim', type=int, default=2, help='dimensionality of latent feature extractor', required=False)
    parser.add_argument('-attdim','--att_dim', type=int, default=2, help='dimensionality of attribute feature extractor', required=False)
    
    parser.add_argument('-maps','--map_strings', type=str, nargs='+', default=[], help='string names of optimizers to use for generator and discriminator', required=False)
    parser.add_argument('-gam','--gamma', type=float, default=1., help='gamma parameter for consensus, reg, reg_alt, and cc', required=False)
    parser.add_argument('-saveto','--saveto', type=str, default='', help='path prefix for saving results', required=False)
    parser.add_argument('-gpu','--gpu', type=int, default=-2, help='if/which gpu to use (-1: all, -2: None)', required=False)
    parser.add_argument('-verb','--verbose', type=lambda x: (str(x).lower() == 'true'), default=False, help='whether to print progress to stdout', required=False)
    args = vars(parser.parse_args())

    if args['domain'] == 'circles':
        from examples.domains.circles import Circles as Domain
        from examples.domains.circles import Generator, AttExtractor, LatExtractor, Discriminator, Disentangler
    else:
        raise NotImplementedError(args['domain'])

    from ugans.core import Train
    args['maps'] = []
    for mp in args['map_strings']:
        if mp.lower() == 'consensus':
            from ugans.train_ops.consensus import Consensus
            args['maps'] += [Consensus]
        else:
            raise NotImplementedError(mp)
    from ugans.train_ops.simgd import SimGD
    args['maps'] += [SimGD]

    if args['saveto'] == '':
        args['saveto'] = 'examples/results/' + args['domain'] + '/'*(len(args['map_strings'])>0) + '-'.join(args['map_strings']) + '/' + args['description']

    if args['description'] == '':
        args['description'] = args['domain'] + '-'*(len(args['map_strings'])>0) + '-'.join(args['map_strings'])
    elif args['description'].isdigit():
        args['description'] = args['domain'] + '-'*(len(args['map_strings'])>0) + '-'.join(args['map_strings']) + '-' + args['description']

    saveto = args['saveto'] + '/' + datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S/{}').format('')
    if not os.path.exists(saveto):
        os.makedirs(saveto)
        os.makedirs(saveto+'/samples')
        os.makedirs(saveto+'/weights')
    shutil.copy(os.path.realpath('ugans/run.py'), os.path.join(saveto, 'run.py'))
    shutil.copy(os.path.realpath('ugans/core.py'), os.path.join(saveto, 'core.py'))
    shutil.copy(os.path.realpath('examples/domains/'+args['domain']+'.py'), os.path.join(saveto, args['domain']+'.py'))
    for mp in args['map_strings']:
        train_file = mp+'.py'
        shutil.copy(os.path.realpath('ugans/train_ops/'+train_file), os.path.join(saveto, train_file))
    with open(saveto+'args.txt', 'w') as file:
        for key, val in args.items():
            file.write('--'+str(key)+' '+str(val)+'\n')
    args['saveto'] = saveto

    cuda_available = torch.cuda.is_available()
    if args['gpu'] >= -1 and cuda_available:
        torch.cuda.device(args['gpu'])
        args['description'] += ' (gpu'+str(torch.cuda.current_device())+')'
    else:
        args['description'] += ' (cpu)'

    return Train, Domain, Generator, AttExtractor, LatExtractor, Discriminator, Disentangler, args


def run_experiment(Train, Domain, Generator, AttExtractor, LatExtractor, Discriminator, Disentangler, params):
    to_gpu = gpu_helper(params['gpu'])

    data = Domain()
    G = Generator(input_dim=params['z_dim'],output_dim=params['x_dim'],n_hidden=params['gen_n_hidden'],
                  n_layer=params['gen_n_layer'],nonlin=params['gen_nonlinearity'])
    F_att = AttExtractor(input_dim=params['x_dim'],output_dim=params['att_dim'],n_hidden=params['att_n_hidden'],
                         n_layer=params['att_n_layer'],nonlin=params['att_nonlinearity'])
    F_lat = LatExtractor(input_dim=params['x_dim'],output_dim=params['lat_dim'],n_hidden=params['lat_n_hidden'],
                         n_layer=params['lat_n_layer'],nonlin=params['lat_nonlinearity'])
    D = Discriminator(input_dim=params['att_dim']+params['lat_dim'],n_hidden=params['disc_n_hidden'],n_layer=params['disc_n_layer'],
                      nonlin=params['disc_nonlinearity'],quad=params['disc_quadratic_layer'])
    D_dis = Disentangler(input_dim=params['lat_dim'],output_dim=params['att_dim'],n_hidden=params['dis_n_hidden'],
                         n_layer=params['dis_n_layer'],nonlin=params['dis_nonlinearity'])
    for mod in [G, F_att, F_lat, D, D_dis]:
        mod.init_weights()
    G, F_att, F_lat, D, D_dis = [to_gpu(mod) for mod in [G, F_att, F_lat, D, D_dis]]

    m = Manager(data, G, F_att, F_lat, D, D_dis, params, to_gpu)

    train = Train(manager=m)

    fs = []
    frames = []
    np_samples = []
    ds = []
    gs = []
    ls = []
    viz_every = params['viz_every']

    iterations = range(params['max_iter'])
    if params['verbose']:
        iterations = tqdm(iterations,desc=params['description'])

    for i in iterations:
        
        d, g, f = train.train_op(i)
        
        if params['verbose']:
            iterations.set_postfix({'||F_D||^2':d,'||F_G||^2':g,'V':f, 'Mem': process.memory_info().rss})

        fs.append(f)
        ds.append(d)
        gs.append(g)

        if viz_every > 0 and i % viz_every == 0:
            if params['n_viz'] > 0:
                np.save(params['saveto']+'samples/'+str(i), train.m.get_fake(params['n_viz'], params['z_dim']).cpu().data.numpy())
            data.plot_current(train, params, i)

        if params['weights_every'] > 0 and i % params['weights_every'] == 0:
            save_weights(m.D,params['saveto']+'D_'+str(i)+'.pkl')
            save_weights(m.G,params['saveto']+'G_'+str(i)+'.pkl')
                 

    np.savetxt(params['saveto']+'d_norm.out',np.array(ds))
    np.savetxt(params['saveto']+'g_norm.out',np.array(gs))
    np.savetxt(params['saveto']+'loss.out',np.array(fs))

    print('Plotting gradient norms...')
    fig = plt.figure()
    ax = plt.subplot(111)
    plt.plot(range(len(ds)), ds)
    ax.set_ylabel('Discriminator Gradient L2 Norm')
    ax.set_xlabel('Iteration')
    plt.title('final loss='+str(ds[-1]))
    fig.savefig(params['saveto']+'d_norm.pdf')

    fig = plt.figure()
    ax = plt.subplot(111)
    plt.plot(range(len(gs)), gs)
    ax.set_ylabel('Generator Gradient L2 Norm')
    ax.set_xlabel('Iteration')
    plt.title('final loss='+str(gs[-1]))
    fig.savefig(params['saveto']+'g_norm.pdf')

    print('Plotting loss...')
    fig = plt.figure()
    ax = plt.subplot(111)
    plt.plot(range(len(fs)),np.array(fs))
    ax.set_ylabel('Loss')
    ax.set_xlabel('Iteration')
    plt.title('final loss='+str(fs[-1]))
    fig.savefig(params['saveto']+'loss.pdf')

    print('Plotting sample series over epochs...')
    if params['n_viz'] > 0:
        np_samples = []
        for viz_i in range(0,params['max_iter'],viz_every):
            np_samples.append(np.load(params['saveto']+'samples/'+str(viz_i)+'.npy'))
        data.plot_series(np_samples, params)

    print('Complete.')


if __name__ == '__main__':
    Train, Domain, Generator, AttExtractor, LatExtractor, Discriminator, Disentangler, params = parse_params()
    run_experiment(Train, Domain, Generator, AttExtractor, LatExtractor, Discriminator, Disentangler, params)
