import torch
import torch.nn as nn
import torch.nn.functional as F

from ugans.core import Data, Net
from IPython import embed

def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.bias.data.fill_(1.)
        elif classname.find('Linear') != -1:
            m.bias.data.fill_(1.)
        

class View(nn.Module):
    def __init__(self, *shape):
        super(View, self).__init__()
        self.shape = shape
    def forward(self, input):
        return input.view(*self.shape)

class Generator(Net):
    def __init__(self, input_dim, output_dim, **kwargs):
        super(Generator, self).__init__()
        # input is bs x 50
        self.main = nn.Sequential(
            # input is Z, going into a dense layer
            nn.Linear(input_dim, 15 * 250, bias=True),
            nn.ReLU(True),
            nn.BatchNorm1d(15 * 250, momentum=0.9),
            View((-1, 250, 15)),
            # state size. bs x 250 x 15
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv1d(250, 250//2, kernel_size=11, stride=1, padding=5, bias=True),
            nn.BatchNorm1d(250//2),
            nn.ReLU(True),
            # state size. bs x 125 x 30
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv1d(250//2, 250//4, kernel_size=11, stride=1, padding=5, bias=True),
            nn.BatchNorm1d(250//4),
            nn.ReLU(True),
            # state size. bs x 62 x 60
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv1d(250//4, 250//8, kernel_size=11, stride=1, padding=5, bias=True),
            nn.BatchNorm1d(250//8),
            nn.ReLU(True),
            # state size. bs x 31 x 120
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv1d(250//8, 250//16, kernel_size=11, stride=1, padding=5, bias=True),
            nn.BatchNorm1d(250//16),
            nn.ReLU(True),
            # # state size. bs x 15 x 240
            nn.Conv1d(250//16, 1, kernel_size=11, stride=1, padding=5, bias=True),
            nn.Sigmoid(),
            View((-1, output_dim))
        )
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_hidden = 6

        self.first_forward = True

    def forward(self, x):
        if self.first_forward: print('\nGenerator output shape:', flush=True)
        output = self.main(x)
        if self.first_forward: print(output.shape, flush=True)
        self.first_forward = False
        return output

    def init_weights(self):
        self.apply(weights_init)


class LatExtractor(Net):
    def __init__(self, input_dim, output_dim, **kwargs):
        super(LatExtractor, self).__init__()
        # input is bs x 240
        self.main = nn.Sequential(
            # input is X, going into a convolutional layer
            View((-1, 1, 240)),
            nn.Conv1d(1, 20, kernel_size=11, stride=2, padding=5, bias=True),
            nn.ReLU(True),
            # state size. bs x 20 x 120
            nn.Conv1d(20, 20*2, kernel_size=11, stride=2, padding=5, bias=True),
            nn.ReLU(True),
            # state size. bs x 40 x 60
            nn.Conv1d(20*2, 20*4, kernel_size=11, stride=2, padding=5, bias=True),
            nn.ReLU(True),
            # state size. bs x 80 x 30
            nn.Conv1d(20*4, 20*8, kernel_size=11, stride=2, padding=5, bias=True),
            nn.ReLU(True),
            # # state size. bs x 160 x 15
            # View((-1, 160*15)),
            # nn.Linear(160*15, 1)
        )

        # #'LAYER -1'
        # #'In: 240 X 1 X 1, depth =1'
        # #'Out: 120 X 1 X 1, depth = 25'
        # discriminator.add(Conv1D(filters=self.disFilters, 
        #     kernel_size=self.filterSize ,strides=2,
        #     input_shape=(self.img_rows, 1), bias_initializer=Constant(0.1), 
        #     activation= 'relu', padding ='same'))
        # discriminator.add(Dropout(self.dropout))

        # #'LAYER -2'
        # #'In: 120 X 1 X 1, depth =25'
        # #'Out: 60 X 1 X 1, depth = 50'
        # discriminator.add(Conv1D(filters=self.disFilters*2, 
        #     kernel_size=self.filterSize, strides=2,
        #     bias_initializer=Constant(0.1), activation= 'relu', 
        #     padding ='same'))
        # discriminator.add(Dropout(self.dropout))

        # #'LAYER -3'
        # #'In: 60 X 1 X 1, depth =50'
        # #'Out: 30 X 1 X 1, depth = 75'
        # discriminator.add(Conv1D(filters=self.disFilters*4, 
        #     kernel_size=self.filterSize ,strides=2,
        #     bias_initializer=Constant(0.1), activation= 'relu', 
        #     padding ='same'))
        # discriminator.add(Dropout(self.dropout))

        # #'LAYER -4'
        # #'In: 30 X 1 X 1, depth =50'
        # #'Out: 15 X 1 X 1, depth = 100'
        # discriminator.add(Conv1D(filters=self.disFilters*8, 
        #     kernel_size=self.filterSize ,strides=2,
        #     bias_initializer=Constant(0.1), activation= 'relu',
        #     padding ='same'))
        # discriminator.add(Dropout(self.dropout))

        # # Output Layer
        # discriminator.add(Flatten())
        # discriminator.add(Dense(1))

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_hidden = 6

        self.first_forward = True

    def forward(self, x):
        if self.first_forward: print('\nLatExtractor output shape:', flush=True)
        output = self.main(x)
        if self.first_forward: print(output.shape, flush=True)
        self.first_forward = False
        embed()
        assert False
        return output

    def init_weights(self):
        self.apply(weights_init)
