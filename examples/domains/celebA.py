import os
import zipfile

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms

from ugans.core import Data, Net
from ugans.utils import download_file_from_google_drive

import matplotlib.pyplot as plt
import seaborn as sns

# Taken from DCGAN
class CelebA(Data):
    def __init__(self, batch_size=128):
        super(CelebA, self).__init__()
        self.download_celeb_a()
        # Root directory for dataset
        dataroot = './examples/domains/data/celebA_img'
        # Number of workers for dataloader
        workers = 2
        # Batch size during training
        # batch_size = 128
        # Spatial size of training images. All images will be resized to this
        #   size using a transformer.
        image_size = 64
        dataset = dset.ImageFolder(root=dataroot,
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
        # Create the dataloader
        self.dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                      shuffle=True, num_workers=workers)
        self.dataiterator = iter(self.dataloader)
        atts = np.load('./examples/domains/data/celebA_att.npz')
        self.attribute_names = atts['names']
        self.attributes = atts['attributes']

    def download_celeb_a(self, dirpath='./examples/domains/data'):
        data_dir = 'celebA_img'
        if os.path.exists(os.path.join(dirpath, data_dir)):
            print('Found Celeb-A - skip')
            return
        # url = 'https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AADIKlz8PR9zr6Y20qbkunrba/Img/img_align_celeba.zip?dl=1&pv=1'
        # url_img = 'https://drive.google.com/file/d/0B7EVK8r0v71pblRyaVFSWGxPY0U/view?usp=sharing'
        filepath = os.path.join(dirpath, 'img_align_celeba.zip')
        download_file_from_google_drive(id='0B7EVK8r0v71pZjFTYXZWM3FlRnM', destination=filepath)
        zip_dir = ''
        with zipfile.ZipFile(filepath) as zf:
            zip_dir = zf.namelist()[0]
            zf.extractall(dirpath)
        os.remove(filepath)
        os.rename(os.path.join(dirpath, zip_dir), os.path.join(dirpath, data_dir))
        # may need to create a folder for each image
        img_dir = os.path.join(dirpath, data_dir)
        for file in os.listdir(img_dir):
            file_folder = os.path.join(img_dir, file.strip('.jpg'))
            os.mkdir(file_folder)
            os.rename(os.path.join(dirpath, file), os.path.join(file_folder, file))
        # url_att = 'https://drive.google.com/file/d/0B7EVK8r0v71pZjFTYXZWM3FlRnM/view?usp=sharing'
        newfilepath = os.path.join(dirpath, 'celebA_att.txt')
        download_file_from_google_drive(id='0B7EVK8r0v71pblRyaVFSWGxPY0U', destination=newfilepath)
        # os.rename(os.path.join(dirpath, os.path.basename(filepath)), newfilepath)
        with open(newfilepath) as file:
            num_labels = file.readline()
            attribute_names = file.readline()
        attributes = np.loadtxt(newfilepath, usecols=range(1,41))
        np.savez_compressed(os.path.join(dirpath, 'celebA_att.npz'), names=attribute_names,attributes=attributes)
        
    def plot_current(self, train, params, i):
        images = train.m.get_fake(64, params['z_dim']).detach().view(-1, 1, 64, 64)
        img = torchvision.utils.make_grid(images)
        # img = img / 2 + 0.5     # unnormalize
        plt.imshow(np.transpose(img.cpu().numpy(), (1, 2, 0)))
        plt.xticks([]); plt.yticks([])
        plt.savefig(params['saveto']+'samples_{}.png'.format(i))
        plt.close()

    def plot_series(self, np_samples, params):
        np_samples_ = np.array(np_samples) / 2 + 0.5
        cols = len(np_samples_)
        fig = plt.figure(figsize=(2*cols, 2*params['n_viz']))
        for i, samps in enumerate(np_samples_):
            if i == 0:
                ax = plt.subplot(1,cols,1)
            else:
                plt.subplot(1,cols,i+1, sharex=ax, sharey=ax)
            thissamp = samps.reshape((params['n_viz'],1,64,64)).transpose((0,2,3,1))
            ax2 = plt.imshow(thissamp.reshape(-1, 64), cmap='gray')
            plt.xticks([]); plt.yticks([])
            plt.title('step %d'%(i*params['viz_every']))
        plt.gcf().tight_layout()
        fig.savefig(params['saveto']+'series.pdf')
        plt.close()

    def plot_real(self, params):
        images = self.sample(batch_size=64).view(-1, 1, 64, 64)
        img = torchvision.utils.make_grid(images)
        # img = img / 2 + 0.5     # unnormalize
        plt.imshow(np.transpose(img.cpu().numpy(), (1, 2, 0)))
        plt.xticks([]); plt.yticks([])
        plt.savefig(params['saveto']+'samples_real.png')
        plt.close()

    def sample(self, batch_size, dim=64):
        try:
            images, ids = next(self.dataiterator)
        except:
            self.dataiterator = iter(self.dataloader)
            images, ids = next(self.dataiterator)
        return images.reshape(batch_size, -1)

    def sample_att(self, batch_size, dim=64):
        try:
            images, ids = next(self.dataiterator)
            attributes = torch.from_numpy(self.attributes[ids.data.numpy()].astype('float32'))
        except:
            self.dataiterator = iter(self.dataloader)
            images, ids = next(self.dataiterator)
            attributes = torch.from_numpy(self.attributes[ids.data.numpy()].astype('float32'))
        return torch.cat((images.reshape(batch_size, -1), attributes), dim=1)


def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)


class Generator(Net):
    def __init__(self, input_dim, n_hidden=128, output_dim=64, n_layer=None, nonlin=None):
        super(Generator, self).__init__()

        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(     input_dim, output_dim * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(output_dim * 8),
            nn.ReLU(True),
            # state size. (output_dim*8) x 4 x 4
            nn.ConvTranspose2d(output_dim * 8, output_dim * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(output_dim * 4),
            nn.ReLU(True),
            # state size. (output_dim*4) x 8 x 8
            nn.ConvTranspose2d(output_dim * 4, output_dim * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(output_dim * 2),
            nn.ReLU(True),
            # state size. (output_dim*2) x 16 x 16
            nn.ConvTranspose2d(output_dim * 2,     output_dim, 4, 2, 1, bias=False),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(True),
            # state size. (output_dim) x 32 x 32
            nn.ConvTranspose2d(    output_dim,      1, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (1 channel) x 64 x 64
        )
        
        self.input_dim = input_dim
        self.n_hidden = n_hidden
        self.output_dim = output_dim

        self.first_forward = True

    def forward(self, x):
        output = x.view(-1, self.input_dim, 1, 1)
        output = self.main(output) / 2. + 0.5
        return output.view(-1, self.output_dim**2)

    def init_weights(self):
        self.apply(weights_init)


class AttExtractor(Net):
    # assumption: output_dim = 1
    def __init__(self, input_dim, output_dim, image_dim=64, n_hidden=128, n_layer=None, nonlin=None, quad=False):
        super(AttExtractor, self).__init__()
        
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(1, image_dim, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (image_dim) x 32 x 32
            nn.Conv2d(image_dim, image_dim * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(image_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (image_dim*2) x 16 x 16
            nn.Conv2d(image_dim * 2, image_dim * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(image_dim * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (image_dim*4) x 8 x 8
            nn.Conv2d(image_dim * 4, image_dim * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(image_dim * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (image_dim*8) x 4 x 4
            # nn.Conv2d(image_dim * 8, 1, 4, 1, 0, bias=False),
            # nn.Sigmoid()
        )

        output = nn.Linear(image_dim*8 * 4 * 4, output_dim)

        self.output = output
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.image_dim = image_dim
        self.n_hidden = n_hidden

        self.first_forward = True

    def forward(self, x):
        x = x.view(-1, 1, self.image_dim, self.image_dim)
        output = self.main(x)
        output = output.view(-1, self.image_dim*8*4*4)
        output = self.output(output)
        temp = output.view(-1, self.output_dim)
        return output.view(-1, self.output_dim)

    def init_weights(self):
        self.apply(weights_init)


class LatExtractor(Net):
    # assumption: output_dim = 1
    def __init__(self, input_dim, output_dim, image_dim=64, n_hidden=128, n_layer=None, nonlin=None, quad=False):
        super(LatExtractor, self).__init__()
        
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(1, image_dim, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (image_dim) x 32 x 32
            nn.Conv2d(image_dim, image_dim * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(image_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (image_dim*2) x 16 x 16
            nn.Conv2d(image_dim * 2, image_dim * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(image_dim * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (image_dim*4) x 8 x 8
            nn.Conv2d(image_dim * 4, image_dim * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(image_dim * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (image_dim*8) x 4 x 4
            # nn.Conv2d(image_dim * 8, 1, 4, 1, 0, bias=False),
            # nn.Sigmoid()
        )

        output = nn.Linear(image_dim*8 * 4 * 4, output_dim)

        self.output = output
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.image_dim = image_dim
        self.n_hidden = n_hidden

        self.first_forward = True

    def forward(self, x):
        x = x.view(-1, 1, self.image_dim, self.image_dim)
        output = self.main(x)
        output = output.view(-1, self.image_dim*8*4*4)
        output = self.output(output)
        return output.view(-1, self.output_dim)

    def init_weights(self):
        self.apply(weights_init)


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
            self.nonlin = F.sigmoid
        else:
            self.nonlin = lambda x: x

        self.first_forward = True

    def forward(self,x):
        h = x
        if self.first_forward: print('Discriminator output shapes:')
        for hfc in self.hidden_fcs:
            h = self.nonlin(hfc(h))
            if self.first_forward: print(h.shape)
        self.first_forward = False
        if self.quad:
            return torch.sum(h*self.final_fc(h),dim=1)
        else:
            return self.final_fc(h)

    def init_weights(self):
        for layer in self.layers:
            nn.init.orthogonal_(layer.weight.data, gain=0.8)
            layer.bias.data.zero_()


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
            self.nonlin = F.sigmoid
        else:
            self.nonlin = lambda x: x

        self.first_forward = True

    def forward(self,x):
        h = x
        if self.first_forward: print('Disentangler output shapes:')
        for hfc in self.hidden_fcs:
            h = self.nonlin(hfc(h))
            if self.first_forward: print(h.shape)
        self.first_forward = False
        return self.final_fc(h)

    def init_weights(self):
        for layer in self.layers:
            nn.init.orthogonal_(layer.weight.data, gain=0.8)
            layer.bias.data.zero_()
