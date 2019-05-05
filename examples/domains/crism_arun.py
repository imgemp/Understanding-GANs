import torch
import torch.nn as nn
import torch.nn.functional as F

from ugans.core import Data, Net


def weights_init(m):
        # classname = m.__class__.__name__
        # if classname.find('Conv') != -1:
        #     m.weight.data.normal_(0.0, 0.02)
        # elif classname.find('BatchNorm') != -1:
        #     m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(1.)

class View(nn.Module):
    def __init__(self, *shape):
        super(View, self).__init__()
        self.shape = shape
    def forward(self, input):
        return input.view(*shape)

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
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv1d(250, 250//2, kernel_size=11, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(250//2),
            nn.ReLU(True),
            # state size. bs x 125 x 30
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv1d(250//2, 250//4, kernel_size=11, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(250//4),
            nn.ReLU(True),
            # state size. bs x 62 x 60
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv1d(250//4, 250//8, kernel_size=11, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(250//8),
            nn.ReLU(True),
            # state size. bs x 31 x 120
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv1d(250//8, 250//16, kernel_size=11, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(250//16),
            nn.ReLU(True),
            # state size. bs x 1 x 240
            nn.Conv1d(1, output_dim, kernel_size=11, stride=1, padding=1, bias=True)
        )
        
        self.input_dim = input_dim
        self.n_hidden = n_hidden
        self.output_dim = output_dim
        
        # 'Convolutional Layer 1'
        # 'In: 50 X 1, depth =1'
        # 'Out: 15 X 400, depth = 25'
        # generator = Sequential()
        # generator.add(Dense(15 * self.genFilters, input_dim=self.input_dim, 
        #     activation= 'relu' ,bias_initializer=Constant(0.1)))
        # generator.add(BatchNormalization(momentum=0.9))
        # generator.add(Reshape((15, self.genFilters)))
        # generator.add(Dropout(self.dropout))

        # 'LAYER -2'
        # 'In: 15 X 1 X1, depth = 400'
        # 'Out: 30 X 1 X 1, depth = 200'
        # generator.add(UpSampling1D(size=2))
        # generator.add(Conv1D(int(self.genFilters/2), self.filterSize, 
        #     activation= 'relu', padding='same', bias_initializer=Constant(0.1)))
        # generator.add(BatchNormalization(momentum=0.9))
        # generator.add(Dropout(self.dropout))

        # 'LAYER -3'
        # 'In: 30 X 1 X1, depth = 400'
        # 'Out: 60 X 1 X 1, depth = 200'
        # generator.add(UpSampling1D(size=2))
        # generator.add(Conv1D(int(self.genFilters/4), self.filterSize,
        #     activation= 'relu', padding='same', bias_initializer=Constant(0.1)))
        # generator.add(BatchNormalization(momentum=0.9))
        # generator.add(Dropout(self.dropout))

        # 'LAYER -4'
        # 'In: 60 X 1 X1, depth = 400'
        # 'Out: 120 X 1 X 1, depth = 200'
        # generator.add(UpSampling1D(size=2))
        # generator.add(Conv1D(int(self.genFilters/8), self.filterSize,
        #     activation= 'relu', padding='same', bias_initializer=Constant(0.1)))
        # generator.add(BatchNormalization(momentum=0.9))
        # generator.add(Dropout(self.dropout))

        # 'LAYER -5'
        # 'In: 120 X 1 X1, depth = 400'
        # 'Out: 240 X 1 X 1, depth = 200'
        # generator.add(UpSampling1D(size=2))
        # generator.add(Conv1D(int(self.genFilters/16), self.filterSize,
        #     activation= 'relu', padding='same', bias_initializer=Constant(0.1)))
        # generator.add(BatchNormalization(momentum=0.9))
        # generator.add(Dropout(self.dropout))

        # 'OUTPUT LAYER'
        # 'In: 240 X 1 X 1, depth=25'
        # 'Out: 240 X 1 X 1, depth =1' 
        # generator.add(Conv1D(1, self.filterSize, padding='same', 
        #     bias_initializer=Constant(0.1)))
        # #generator.add(Flatten())
        # generator.add(Activation('sigmoid'))

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.first_forward = True

    def forward(self, x):
        output = self.main(x)
        return output

    def init_weights(self):
        self.apply(weights_init)
