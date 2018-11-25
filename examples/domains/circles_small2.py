import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from ugans.core import Data, Net

from skimage.draw import polygon as draw_polygon, circle as draw_circle
from skimage._shared.utils import warn
from skimage.draw._random_shapes import _generate_rectangle_mask, _generate_triangle_mask, _generate_random_colors
from scipy.stats import poisson

import matplotlib.pyplot as plt
import seaborn as sns


class Circles(Data):
    def __init__(self):
        super(Circles, self).__init__()

    def plot_current(self, train, params, i):
        images = train.m.get_fake(64, params['z_dim']).detach().view(-1, 1, 64, 64)
        img = torchvision.utils.make_grid(images)
        img = img / 2 + 0.5     # unnormalize
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
        img = img / 2 + 0.5     # unnormalize
        plt.imshow(np.transpose(img.cpu().numpy(), (1, 2, 0)))
        plt.xticks([]); plt.yticks([])
        plt.savefig(params['saveto']+'samples_real.png')
        plt.close()

    def sample(self, batch_size, dim=64):
        samples = []
        for b in range(batch_size):
            image = random_shapes_distr((dim, dim), max_shapes=1, shape='circle', min_size=20,
                                        max_size=30,multichannel=False)[0]
            samples += [((255-image)/255.).astype('float32').flatten()]
        return torch.from_numpy(np.vstack(samples))

    def sample_att(self, batch_size, dim=64, min_size=20, max_size=30):
        samples = []
        for b in range(batch_size):
            result = random_shapes_distr((dim, dim), max_shapes=1, shape='circle', min_size=min_size,
                                         max_size=max_size, multichannel=False)
            image, label = result  # label = ('circle', (px, py, radius))
            image = (255-image)/255.
            px, py, radius = np.array(label[0][1])
            px = (px-dim-1)/float(dim)
            py = (py-dim-1)/float(dim)
            radius = 0.5*(2*radius-min_size)/(max_size-min_size)
            label = np.array([px,py,radius])
            samples += [np.concatenate([image.flatten(), label]).astype('float32')]
        return torch.from_numpy(np.vstack(samples))


def _generate_circle_mask_new(point, image, shape, random):
    """Generate a mask for a filled circle shape.
    The radius of the circle is generated randomly.
    Parameters
    ----------
    point : tuple
        The row and column of the top left corner of the rectangle.
    image : tuple
        The height, width and depth of the image into which the shape is placed.
    shape : tuple
        The minimum and maximum size and color of the shape to fit.
    random : np.random.RandomState
        The random state to use for random sampling.
    Raises
    ------
    ArithmeticError
        When a shape cannot be fit into the image with the given starting
        coordinates. This usually means the image dimensions are too small or
        shape dimensions too large.
    Returns
    -------
    label : tuple
        A (category, ((r0, r1), (c0, c1))) tuple specifying the category and
        bounding box coordinates of the shape.
    indices : 2-D array
        A mask of indices that the shape fills.
    """
    if shape[0] == 1 or shape[1] == 1:
        raise ValueError('size must be > 1 for circles')
    min_radius = shape[0] / 2.0
    max_radius = shape[1] / 2.0
    left = point[1]
    right = image[1] - point[1]
    top = point[0]
    bottom = image[0] - point[0]
    available_radius = min(left, right, top, bottom, max_radius)
    if available_radius < min_radius:
        raise ArithmeticError('cannot fit shape to image')
    radius = random.randint(min_radius, available_radius + 1)
    circle = draw_circle(point[0], point[1], radius)
    
    #label = ('circle', ((point[0] - radius + 1, point[0] + radius),
    #                    (point[1] - radius + 1, point[1] + radius)))
    label = ('circle', (point[0], point[1], radius))
    
    return circle, label

def random_shapes_distr(image_shape,
                  max_shapes,
                  min_shapes=1,
                  min_size=2,
                  max_size=None,
                  multichannel=True,
                  num_channels=3,
                  shape=None,
                  intensity_range=None,
                  allow_overlap=False,
                  num_trials=100,
                  random_seed=None):
    """Generate an image with random shapes, labeled with bounding boxes.
    The image is populated with random shapes with random sizes, random
    locations, and random colors, with or without overlap.
    Shapes have random (row, col) starting coordinates and random sizes bounded
    by `min_size` and `max_size`. It can occur that a randomly generated shape
    will not fit the image at all. In that case, the algorithm will try again
    with new starting coordinates a certain number of times. However, it also
    means that some shapes may be skipped altogether. In that case, this
    function will generate fewer shapes than requested.
    Parameters
    ----------
    image_shape : tuple
        The number of rows and columns of the image to generate.
    max_shapes : int
        The maximum number of shapes to (attempt to) fit into the shape.
    min_shapes : int, optional
        The minimum number of shapes to (attempt to) fit into the shape.
    min_size : int, optional
        The minimum dimension of each shape to fit into the image.
    max_size : int, optional
        The maximum dimension of each shape to fit into the image.
    multichannel : bool, optional
        If True, the generated image has ``num_channels`` color channels,
        otherwise generates grayscale image.
    num_channels : int, optional
        Number of channels in the generated image. If 1, generate monochrome
        images, else color images with multiple channels. Ignored if
        ``multichannel`` is set to False.
    shape : {rectangle, circle, triangle, None} str, optional
        The name of the shape to generate or `None` to pick random ones.
    intensity_range : {tuple of tuples of uint8, tuple of uint8}, optional
        The range of values to sample pixel values from. For grayscale images
        the format is (min, max). For multichannel - ((min, max),) if the
        ranges are equal across the channels, and ((min_0, max_0), ... (min_N, max_N))
        if they differ. As the function supports generation of uint8 arrays only,
        the maximum range is (0, 255). If None, set to (0, 254) for each
        channel reserving color of intensity = 255 for background.
    allow_overlap : bool, optional
        If `True`, allow shapes to overlap.
    num_trials : int, optional
        How often to attempt to fit a shape into the image before skipping it.
    seed : int, optional
        Seed to initialize the random number generator.
        If `None`, a random seed from the operating system is used.
    Returns
    -------
    image : uint8 array
        An image with the fitted shapes.
    labels : list
        A list of labels, one per shape in the image. Each label is a
        (category, ((r0, r1), (c0, c1))) tuple specifying the category and
        bounding box coordinates of the shape.
    Examples
    --------
    >>> import skimage.draw
    >>> image, labels = skimage.draw.random_shapes((32, 32), max_shapes=3)
    >>> image # doctest: +SKIP
    array([
       [[255, 255, 255],
        [255, 255, 255],
        [255, 255, 255],
        ...,
        [255, 255, 255],
        [255, 255, 255],
        [255, 255, 255]]], dtype=uint8)
    >>> labels # doctest: +SKIP
    [('circle', ((22, 18), (25, 21))),
     ('triangle', ((5, 6), (13, 13)))]
    """
    if min_size > image_shape[0] or min_size > image_shape[1]:
        raise ValueError('Minimum dimension must be less than ncols and nrows')
    max_size = max_size or max(image_shape[0], image_shape[1])

    if not multichannel:
        num_channels = 1

    if intensity_range is None:
        intensity_range = (0, 254) if num_channels == 1 else ((0, 254), )
    else:
        tmp = (intensity_range, ) if num_channels == 1 else intensity_range
        for intensity_pair in tmp:
            for intensity in intensity_pair:
                if not (0 <= intensity <= 255):
                    msg = 'Intensity range must lie within (0, 255) interval'
                    raise ValueError(msg)

    random = np.random.RandomState(random_seed)
    user_shape = shape
    image_shape = (image_shape[0], image_shape[1], num_channels)
    image = np.ones(image_shape, dtype=np.uint8) * 255
    filled = np.zeros(image_shape, dtype=bool)
    labels = []

    num_shapes = random.randint(min_shapes, max_shapes + 1)
    colors = _generate_random_colors(num_shapes, num_channels,
                                     intensity_range, random)
    for shape_idx in range(num_shapes):
        if user_shape is None:
            shape_generator = random.choice(SHAPE_CHOICES)
        else:
            shape_generator = SHAPE_GENERATORS[user_shape]
        shape = (min_size, max_size)
        for _ in range(num_trials):
            # Pick start coordinates.
            # mu0 = 0.65*image_shape[0]
            # loc0 = -0.43*image_shape[0]
            # mu1 = 0.65*image_shape[1]
            # loc1 = -0.43*image_shape[1]
            mu0 = 0.10*image_shape[0]
            loc0 = 0.20*image_shape[0]
            mu1 = 0.10*image_shape[1]
            loc1 = 0.20*image_shape[1]
            # row = poisson.rvs(mu0, loc=loc0)
            # column = poisson.rvs(mu1, loc=loc1)
            row = loc0 + mu0*poisson.rvs(1)
            column = loc1 + mu1*poisson.rvs(1)
            #column = random.randint(image_shape[1])
            #row = random.randint(image_shape[0])
            point = (row, column)
            try:
                indices, label = shape_generator(point, image_shape, shape,
                                                 random)
            except ArithmeticError:
                # Couldn't fit the shape, skip it.
                continue
            # Check if there is an overlap where the mask is nonzero.
            if allow_overlap or not filled[indices].any():
                image[indices] = colors[shape_idx]
                filled[indices] = True
                labels.append(label)
                break
        else:
            warn('Could not fit any shapes to image, '
                 'consider reducing the minimum dimension')

    if not multichannel:
        image = np.squeeze(image, axis=2)
    return image, labels

SHAPE_GENERATORS = dict(
    rectangle=_generate_rectangle_mask,
    circle=_generate_circle_mask_new,
    triangle=_generate_triangle_mask)
SHAPE_CHOICES = list(SHAPE_GENERATORS.values())


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
            # input is Z, going into a convolution: (c*hin - kernel)/stride + 1, c=4,stride=kernel=2 --> 2*hin
            # or (2*hin + 2*padding - kernel)/stride + 1, 2*hin + 2*padding-kernel+1, padding=1, kernel=3
            nn.Upsample(scale_factor = 3, mode='bilinear'),
            # nn.ReflectionPad2d(1),
            nn.Conv2d(input_dim, output_dim * 8, kernel_size=2, stride=1, padding=1),
            nn.BatchNorm2d(output_dim * 8),
            nn.ReLU(True),
            # state size. (output_dim*8) x 4 x 4
            nn.Upsample(scale_factor = 2, mode='bilinear'),
            # nn.ReflectionPad2d(1),
            nn.Conv2d(output_dim * 8, output_dim * 4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(output_dim * 4),
            nn.ReLU(True), # hin - kernel + 1 + 2*padding (assumes stride=1)   3*hin - 2 + 1 + 2 = 
            # state size. (output_dim*4) x 8 x 8
            nn.Upsample(scale_factor = 2, mode='bilinear'),
            # nn.ReflectionPad2d(1),
            nn.Conv2d(output_dim * 4, output_dim * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(output_dim * 2),
            nn.ReLU(True),
            # state size. (output_dim*2) x 16 x 16
            nn.Upsample(scale_factor = 2, mode='bilinear'),
            # nn.ReflectionPad2d(1),
            nn.Conv2d(output_dim * 2,     output_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(True),
            # state size. (output_dim) x 32 x 32
            nn.Upsample(scale_factor = 2, mode='bilinear'),
            # nn.ReflectionPad2d(1),
            nn.Conv2d(    output_dim,      1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Tanh()
            # state size. (1 channel) x 64 x 64
        )
        
        self.input_dim = input_dim
        self.n_hidden = n_hidden
        self.output_dim = output_dim

        self.first_forward = True

    def forward(self, x):
        output = x.view(-1, self.input_dim, 1, 1)
        output = self.main(output)
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
