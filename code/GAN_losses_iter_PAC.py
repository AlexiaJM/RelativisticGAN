#!/usr/bin/env python3

# To get TensorBoard output, use the python command: tensorboard --logdir /home/alexia/Output/DCGAN
# TensorBoard disabled for now.

# To get CIFAR10
# wget http://pjreddie.com/media/files/cifar.tgz
# tar xzf cifar.tgz


## Parameters

# thanks https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
def strToBool(str):
	return str.lower() in ('true', 'yes', 'on', 't', '1')

import argparse
parser = argparse.ArgumentParser()
parser.register('type', 'bool', strToBool)
parser.add_argument('--image_size', type=int, default=64)
parser.add_argument('--batch_size', type=int, default=32) # DCGAN paper original value used 128 (32 is generally better to prevent vanishing gradients with SGAN and LSGAN, not important with relativistic GANs)
parser.add_argument('--n_colors', type=int, default=3)
parser.add_argument('--z_size', type=int, default=128)
parser.add_argument('--G_h_size', type=int, default=128, help='Number of hidden nodes in the Generator. Used only in arch=0. Too small leads to bad results, too big blows up the GPU RAM.') # DCGAN paper original value
parser.add_argument('--D_h_size', type=int, default=128, help='Number of hidden nodes in the Discriminator. Used only in arch=0. Too small leads to bad results, too big blows up the GPU RAM.') # DCGAN paper original value
parser.add_argument('--lr_D', type=float, default=.0001, help='Discriminator learning rate')
parser.add_argument('--lr_G', type=float, default=.0001, help='Generator learning rate')
parser.add_argument('--n_iter', type=int, default=100000, help='Number of iteration cycles')
parser.add_argument('--beta1', type=float, default=0.5, help='Adam betas[0], DCGAN paper recommends .50 instead of the usual .90')
parser.add_argument('--beta2', type=float, default=0.999, help='Adam betas[1]')
parser.add_argument('--decay', type=float, default=0, help='Decay to apply to lr each cycle. decay^n_iter gives the final lr. Ex: .00002 will lead to .13 of lr after 100k cycles')
parser.add_argument('--SELU', type='bool', default=False, help='Using scaled exponential linear units (SELU) which are self-normalizing instead of ReLU with BatchNorm. Used only in arch=0. This improves stability.')
parser.add_argument("--NN_conv", type='bool', default=False, help="This approach minimize checkerboard artifacts during training. Used only by arch=0. Uses nearest-neighbor resized convolutions instead of strided convolutions (https://distill.pub/2016/deconv-checkerboard/ and github.com/abhiskk/fast-neural-style).")
parser.add_argument('--seed', type=int)
parser.add_argument('--input_folder', default='/home/alexia/Datasets/Meow_64x64', help='input folder')
parser.add_argument('--output_folder', default='/home/alexia/Dropbox/Ubuntu_ML/Output/GANlosses', help='output folder')
parser.add_argument('--inception_folder', default='/home/alexia/Inception', help='Inception model folder (path must exists already, model will be downloaded automatically)')
parser.add_argument('--load', default=None, help='Full path to network state to load (ex: /home/output_folder/run-5/models/state_11.pth)')
parser.add_argument('--cuda', type='bool', default=True, help='enables cuda')
parser.add_argument('--n_gpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--loss_D', type=int, default=1, help='Loss of D, see code for details (1=GAN, 2=LSGAN, 3=WGAN-GP, 4=HingeGAN, 5=RSGAN, 6=RaSGAN, 7=RaLSGAN, 8=RaHingeGAN)')
parser.add_argument('--Diters', type=int, default=1, help='Number of iterations of D')
parser.add_argument('--Giters', type=int, default=1, help='Number of iterations of G.')
parser.add_argument('--penalty', type=float, default=10, help='Gradient penalty parameter for WGAN-GP')
parser.add_argument('--spectral', type='bool', default=False, help='If True, use spectral normalization to make the discriminator Lipschitz. This Will also remove batch norm in the discriminator.')
parser.add_argument('--spectral_G', type='bool', default=False, help='If True, use spectral normalization to make the generator Lipschitz (Generally only D is spectral, not G). This Will also remove batch norm in the discriminator.')
parser.add_argument('--weight_decay', type=float, default=0, help='L2 regularization weight. Helps convergence but leads to artifacts in images, not recommended.')
parser.add_argument('--gen_extra_images', type=int, default=50000, help='Generate additional images with random fake cats in calculating FID (Recommended to use the same amount as the size of the dataset; for CIFAR-10 we use 50k, but most people use 10k) It must be a multiple of 100.')
parser.add_argument('--gen_every', type=int, default=100000, help='Generate additional images with random fake cats every x iterations. Used in calculating FID.')
parser.add_argument('--extra_folder', default='/home/alexia/Output/Extra', help='Folder for extra photos (different so that my dropbox does not get overwhelmed with 50k pictures)')
parser.add_argument('--show_graph', type='bool', default=False, help='If True, show gradients graph. Really neat for debugging.')
parser.add_argument('--no_batch_norm_G', type='bool', default=False, help='If True, no batch norm in G.')
parser.add_argument('--no_batch_norm_D', type='bool', default=False, help='If True, no batch norm in D.')
parser.add_argument('--Tanh_GD', type='bool', default=False, help='If True, tanh everywhere.')
parser.add_argument('--grad_penalty', type='bool', default=False, help='If True, use gradient penalty of WGAN-GP but with whichever loss_D chosen. No need to set this true with WGAN-GP.')
parser.add_argument('--arch', type=int, default=0, help='1: standard CNN  for 32x32 images from the Spectral GAN paper, 0:DCGAN with number of layers adjusted based on image size. Some options may be ignored by some architectures.')
parser.add_argument('--print_every', type=int, default=1000, help='Generate a mini-batch of images at every x iterations (to see how the training progress, you can do it often).')
parser.add_argument('--save', type='bool', default=True, help='Do we save models, yes or no? It will be saved in extra_folder')
parser.add_argument('--CIFAR10', type='bool', default=False, help='If True, use CIFAR-10 instead of your own dataset. Make sure image_size is set to 32!')
parser.add_argument('--CIFAR10_input_folder', default='/home/alexia/Datasets/CIFAR10', help='input folder (automatically downloaded)')
#parser.add_argument('--CIFAR10_input_folder_images', default='/home/alexia/Datasets/CIFAR10_images', help='input folder (to download on http://pjreddie.com/media/files/cifar.tgz and extract)')
param = parser.parse_args()

## Imports

# Time
import time
start = time.time()

# Setting the title for the file saved
if param.loss_D == 1:
	title = 'GAN_'
if param.loss_D == 2:
	title = 'LSGAN_'
if param.loss_D == 3:
	title = 'WGANGP_'
if param.loss_D == 4:
	title = 'HingeGAN_'
if param.loss_D == 5:
	title = 'RSGAN_'
if param.loss_D == 6:
	title = 'RaSGAN_'
if param.loss_D == 7:
	title = 'RaLSGAN_'
if param.loss_D == 8:
	title = 'RaHingeGAN_'

if param.seed is not None:
	title = title + 'seed%i' % param.seed


# Check folder run-i for all i=0,1,... until it finds run-j which does not exists, then creates a new folder run-j
import os
run = 0
base_dir = f"{param.output_folder}/{title}-{run}"
while os.path.exists(base_dir):
	run += 1
	base_dir = f"{param.output_folder}/{title}-{run}"
os.mkdir(base_dir)
logs_dir = f"{base_dir}/logs"
os.mkdir(logs_dir)
os.mkdir(f"{base_dir}/images")
if param.gen_extra_images > 0 and not os.path.exists(f"{param.extra_folder}"):
	os.mkdir(f"{param.extra_folder}")

# where we save the output
log_output = open(f"{logs_dir}/log.txt", 'w')
print(param)
print(param, file=log_output)

import numpy
import torch
import torch.autograd as autograd
from torch.autograd import Variable

# For plotting the Loss of D and G using tensorboard
# To fix later, not compatible with using tensorflow
#from tensorboard_logger import configure, log_value
#configure(logs_dir, flush_secs=5)

import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transf
import torchvision.models as models
import torchvision.utils as vutils
import torch.nn.utils.spectral_norm as spectral_norm

if param.cuda:
	import torch.backends.cudnn as cudnn
	cudnn.deterministic = True
	cudnn.benchmark = True

# To see images
from IPython.display import Image
to_img = transf.ToPILImage()

import pytorch_visualize as pv

import math

torch.utils.backcompat.broadcast_warning.enabled=True

from fid import calculate_fid_given_paths as calc_fid
#from inception import get_inception_score
#from inception import load_images

## Setting seed
import random
if param.seed is None:
	param.seed = random.randint(1, 10000)
print(f"Random Seed: {param.seed}")
print(f"Random Seed: {param.seed}", file=log_output)
random.seed(param.seed)
numpy.random.seed(param.seed)
torch.manual_seed(param.seed)
if param.cuda:
	torch.cuda.manual_seed_all(param.seed)

## Transforming images
trans = transf.Compose([
	transf.Resize((param.image_size, param.image_size)),
	# This makes it into [0,1]
	transf.ToTensor(),
	# This makes it into [-1,1]
	transf.Normalize(mean = [0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5])
])

## Importing dataset
data = dset.ImageFolder(root=param.input_folder, transform=trans)
if param.CIFAR10:
	data = dset.CIFAR10(root=param.CIFAR10_input_folder, train=True, download=True, transform=trans)

# Loading data randomly
def generate_random_sample():
	while True:
		random_indexes = numpy.random.choice(data.__len__(), size=param.batch_size*2, replace=False)
		batch = [data[i][0] for i in random_indexes]
		yield torch.stack(batch, 0)
random_sample = generate_random_sample()

## Models

if param.arch == 1:
	title = title + '_CNN_'

	class DCGAN_G(torch.nn.Module):
		def __init__(self):
			super(DCGAN_G, self).__init__()

			self.dense = torch.nn.Linear(param.z_size, 512 * 4 * 4)

			if param.spectral_G:
				model = [spectral_norm(torch.nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=True))]
				model += [torch.nn.ReLU(True),
					spectral_norm(torch.nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=True))]
				model += [torch.nn.ReLU(True),
					spectral_norm(torch.nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=True))]
				model += [torch.nn.ReLU(True),
					spectral_norm(torch.nn.Conv2d(64, param.n_colors, kernel_size=3, stride=1, padding=1, bias=True)),
					torch.nn.Tanh()]
			else:
				model = [torch.nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=True)]
				if not param.no_batch_norm_G:
					model += [torch.nn.BatchNorm2d(256)]
				if param.Tanh_GD:
					model += [torch.nn.Tanh()]
				else:
					model += [torch.nn.ReLU(True)]
				model += [torch.nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=True)]
				if not param.no_batch_norm_G:
					model += [torch.nn.BatchNorm2d(128)]
				if param.Tanh_GD:
					model += [torch.nn.Tanh()]
				else:
					model += [torch.nn.ReLU(True)]
				model += [torch.nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=True)]
				if not param.no_batch_norm_G:
					model += [torch.nn.BatchNorm2d(64)]
				if param.Tanh_GD:
					model += [torch.nn.Tanh()]
				else:
					model += [torch.nn.ReLU(True)]
				model += [torch.nn.Conv2d(64, param.n_colors, kernel_size=3, stride=1, padding=1, bias=True),
					torch.nn.Tanh()]
			self.model = torch.nn.Sequential(*model)

		def forward(self, input):
			if isinstance(input.data, torch.cuda.FloatTensor) and param.n_gpu > 1:
				output = torch.nn.parallel.data_parallel(self.model(self.dense(input.view(-1, param.z_size)).view(-1, 512, 4, 4)), input, range(param.n_gpu))
			else:
				output = self.model(self.dense(input.view(-1, param.z_size)).view(-1, 512, 4, 4))
			#print(output.size())
			return output

	class DCGAN_D(torch.nn.Module):
		def __init__(self):
			super(DCGAN_D, self).__init__()

			self.dense = torch.nn.Linear(512 * 4 * 4, 1)

			if param.spectral:
				model = [spectral_norm(torch.nn.Conv2d(param.n_colors*2, 64, kernel_size=3, stride=1, padding=1, bias=True)),
					torch.nn.LeakyReLU(0.1, inplace=True),
					spectral_norm(torch.nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1, bias=True)),
					torch.nn.LeakyReLU(0.1, inplace=True),

					spectral_norm(torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=True)),
					torch.nn.LeakyReLU(0.1, inplace=True),
					spectral_norm(torch.nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1, bias=True)),
					torch.nn.LeakyReLU(0.1, inplace=True),

					spectral_norm(torch.nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=True)),
					torch.nn.LeakyReLU(0.1, inplace=True),
					spectral_norm(torch.nn.Conv2d(256, 256, kernel_size=4, stride=2, padding=1, bias=True)),
					torch.nn.LeakyReLU(0.1, inplace=True),

					spectral_norm(torch.nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=True)),
					torch.nn.LeakyReLU(0.1, inplace=True)]
			else:
				model = [torch.nn.Conv2d(param.n_colors*2, 64, kernel_size=3, stride=1, padding=1, bias=True)]
				if not param.no_batch_norm_D:
					model += [torch.nn.BatchNorm2d(64)]
				if param.Tanh_GD:
					model += [torch.nn.Tanh()]
				else:
					model += [torch.nn.LeakyReLU(0.1, inplace=True)]
				model += [torch.nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1, bias=True)]
				if not param.no_batch_norm_D:
					model += [torch.nn.BatchNorm2d(64)]
				if param.Tanh_GD:
					model += [torch.nn.Tanh()]
				else:
					model += [torch.nn.LeakyReLU(0.1, inplace=True)]
				model += [torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=True)]
				if not param.no_batch_norm_D:
					model += [torch.nn.BatchNorm2d(128)]
				if param.Tanh_GD:
					model += [torch.nn.Tanh()]
				else:
					model += [torch.nn.LeakyReLU(0.1, inplace=True)]
				model += [torch.nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1, bias=True)]
				if not param.no_batch_norm_D:
					model += [torch.nn.BatchNorm2d(128)]
				if param.Tanh_GD:
					model += [torch.nn.Tanh()]
				else:
					model += [torch.nn.LeakyReLU(0.1, inplace=True)]
				model += [torch.nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=True)]
				if not param.no_batch_norm_D:
					model += [torch.nn.BatchNorm2d(256)]
				if param.Tanh_GD:
					model += [torch.nn.Tanh()]
				else:
					model += [torch.nn.LeakyReLU(0.1, inplace=True)]
				model += [torch.nn.Conv2d(256, 256, kernel_size=4, stride=2, padding=1, bias=True)]
				if not param.no_batch_norm_D:
					model += [torch.nn.BatchNorm2d(256)]
				if param.Tanh_GD:
					model += [torch.nn.Tanh()]
				else:
					model += [torch.nn.LeakyReLU(0.1, inplace=True)]
				model += [torch.nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=True)]
				if param.Tanh_GD:
					model += [torch.nn.Tanh()]
				else:
					model += [torch.nn.LeakyReLU(0.1, inplace=True)]
			self.model = torch.nn.Sequential(*model)

			self.sig = torch.nn.Sigmoid()

		def forward(self, input):
			if isinstance(input.data, torch.cuda.FloatTensor) and param.n_gpu > 1:
				output = torch.nn.parallel.data_parallel(self.dense(self.model(input).view(-1, 512 * 4 * 4)).view(-1), input, range(param.n_gpu))
			else:
				output = self.dense(self.model(input).view(-1, 512 * 4 * 4)).view(-1)
			if param.loss_D in [1]:
				output = self.sig(output)
			#print(output.size())
			return output

if param.arch == 0:
	# DCGAN generator
	class DCGAN_G(torch.nn.Module):
		def __init__(self):
			super(DCGAN_G, self).__init__()
			main = torch.nn.Sequential()

			# We need to know how many layers we will use at the beginning
			mult = param.image_size // 8

			### Start block
			# Z_size random numbers
			if param.spectral_G:
				main.add_module('Start-SpectralConvTranspose2d', torch.nn.utils.spectral_norm(torch.nn.ConvTranspose2d(param.z_size, param.G_h_size * mult, kernel_size=4, stride=1, padding=0, bias=False)))
			else:
				main.add_module('Start-ConvTranspose2d', torch.nn.ConvTranspose2d(param.z_size, param.G_h_size * mult, kernel_size=4, stride=1, padding=0, bias=False))
			if param.SELU:
				main.add_module('Start-SELU', torch.nn.SELU(inplace=True))
			else:
				if not param.no_batch_norm_G and not param.spectral_G:
					main.add_module('Start-BatchNorm2d', torch.nn.BatchNorm2d(param.G_h_size * mult))
				if param.Tanh_GD:
					main.add_module('Start-Tanh', torch.nn.Tanh())
				else:
					main.add_module('Start-ReLU', torch.nn.ReLU())
			# Size = (G_h_size * mult) x 4 x 4

			### Middle block (Done until we reach ? x image_size/2 x image_size/2)
			i = 1
			while mult > 1:
				if param.NN_conv:
					main.add_module('Middle-UpSample [%d]' % i, torch.nn.Upsample(scale_factor=2))
					if param.spectral_G:
						main.add_module('Middle-SpectralConv2d [%d]' % i, torch.nn.utils.spectral_norm(torch.nn.Conv2d(param.G_h_size * mult, param.G_h_size * (mult//2), kernel_size=3, stride=1, padding=1)))
					else:
						main.add_module('Middle-Conv2d [%d]' % i, torch.nn.Conv2d(param.G_h_size * mult, param.G_h_size * (mult//2), kernel_size=3, stride=1, padding=1))
				else:
					if param.spectral_G:
						main.add_module('Middle-SpectralConvTranspose2d [%d]' % i, torch.nn.utils.spectral_norm(torch.nn.ConvTranspose2d(param.G_h_size * mult, param.G_h_size * (mult//2), kernel_size=4, stride=2, padding=1, bias=False)))
					else:
						main.add_module('Middle-ConvTranspose2d [%d]' % i, torch.nn.ConvTranspose2d(param.G_h_size * mult, param.G_h_size * (mult//2), kernel_size=4, stride=2, padding=1, bias=False))
				if param.SELU:
					main.add_module('Middle-SELU [%d]' % i, torch.nn.SELU(inplace=True))
				else:
					if not param.no_batch_norm_G and not param.spectral_G:
						main.add_module('Middle-BatchNorm2d [%d]' % i, torch.nn.BatchNorm2d(param.G_h_size * (mult//2)))
					if param.Tanh_GD:
						main.add_module('Middle-Tanh [%d]' % i, torch.nn.Tanh())
					else:
						main.add_module('Middle-ReLU [%d]' % i, torch.nn.ReLU())
				# Size = (G_h_size * (mult/(2*i))) x 8 x 8
				mult = mult // 2
				i += 1

			### End block
			# Size = G_h_size x image_size/2 x image_size/2
			if param.NN_conv:
				main.add_module('End-UpSample', torch.nn.Upsample(scale_factor=2))
				if param.spectral_G:
					main.add_module('End-SpectralConv2d', torch.nn.utils.spectral_norm(torch.nn.Conv2d(param.G_h_size, param.n_colors, kernel_size=3, stride=1, padding=1)))
				else:
					main.add_module('End-Conv2d', torch.nn.Conv2d(param.G_h_size, param.n_colors, kernel_size=3, stride=1, padding=1))
			else:
				if param.spectral_G:
					main.add_module('End-SpectralConvTranspose2d', torch.nn.utils.spectral_norm(torch.nn.ConvTranspose2d(param.G_h_size, param.n_colors, kernel_size=4, stride=2, padding=1, bias=False)))
				else:
					main.add_module('End-ConvTranspose2d', torch.nn.ConvTranspose2d(param.G_h_size, param.n_colors, kernel_size=4, stride=2, padding=1, bias=False))
			main.add_module('End-Tanh', torch.nn.Tanh())
			# Size = n_colors x image_size x image_size
			self.main = main

		def forward(self, input):
			if isinstance(input.data, torch.cuda.FloatTensor) and param.n_gpu > 1:
				output = torch.nn.parallel.data_parallel(self.main, input, range(param.n_gpu))
			else:
				output = self.main(input)
			return output

	# DCGAN discriminator (using somewhat the reverse of the generator)
	class DCGAN_D(torch.nn.Module):
		def __init__(self):
			super(DCGAN_D, self).__init__()
			main = torch.nn.Sequential()

			### Start block
			# Size = n_colors x image_size x image_size
			if param.spectral:
				main.add_module('Start-SpectralConv2d', torch.nn.utils.spectral_norm(torch.nn.Conv2d(param.n_colors*2, param.D_h_size, kernel_size=4, stride=2, padding=1, bias=False)))
			else:
				main.add_module('Start-Conv2d', torch.nn.Conv2d(param.n_colors*2, param.D_h_size, kernel_size=4, stride=2, padding=1, bias=False))
			if param.SELU:
				main.add_module('Start-SELU', torch.nn.SELU(inplace=True))
			else:
				if param.Tanh_GD:
					main.add_module('Start-Tanh', torch.nn.Tanh())
				else:
					main.add_module('Start-LeakyReLU', torch.nn.LeakyReLU(0.2, inplace=True))
			image_size_new = param.image_size // 2
			# Size = D_h_size x image_size/2 x image_size/2

			### Middle block (Done until we reach ? x 4 x 4)
			mult = 1
			i = 0
			while image_size_new > 4:
				if param.spectral:
					main.add_module('Middle-SpectralConv2d [%d]' % i, torch.nn.utils.spectral_norm(torch.nn.Conv2d(param.D_h_size * mult, param.D_h_size * (2*mult), kernel_size=4, stride=2, padding=1, bias=False)))
				else:
					main.add_module('Middle-Conv2d [%d]' % i, torch.nn.Conv2d(param.D_h_size * mult, param.D_h_size * (2*mult), kernel_size=4, stride=2, padding=1, bias=False))
				if param.SELU:
					main.add_module('Middle-SELU [%d]' % i, torch.nn.SELU(inplace=True))
				else:
					if not param.no_batch_norm_D and not param.spectral:
						main.add_module('Middle-BatchNorm2d [%d]' % i, torch.nn.BatchNorm2d(param.D_h_size * (2*mult)))
					if param.Tanh_GD:
						main.add_module('Start-Tanh [%d]' % i, torch.nn.Tanh())
					else:
						main.add_module('Middle-LeakyReLU [%d]' % i, torch.nn.LeakyReLU(0.2, inplace=True))
				# Size = (D_h_size*(2*i)) x image_size/(2*i) x image_size/(2*i)
				image_size_new = image_size_new // 2
				mult *= 2
				i += 1

			### End block
			# Size = (D_h_size * mult) x 4 x 4
			if param.spectral:
				main.add_module('End-SpectralConv2d', torch.nn.utils.spectral_norm(torch.nn.Conv2d(param.D_h_size * mult, 1, kernel_size=4, stride=1, padding=0, bias=False)))
			else:
				main.add_module('End-Conv2d', torch.nn.Conv2d(param.D_h_size * mult, 1, kernel_size=4, stride=1, padding=0, bias=False))
			if param.loss_D in [1]:
				main.add_module('End-Sigmoid', torch.nn.Sigmoid())
			# Size = 1 x 1 x 1 (Is a real cat or not?)
			self.main = main

		def forward(self, input):
			if isinstance(input.data, torch.cuda.FloatTensor) and param.n_gpu > 1:
				output = torch.nn.parallel.data_parallel(self.main, input, range(param.n_gpu))
			else:
				output = self.main(input)
			# Convert from 1 x 1 x 1 to 1 so that we can compare to given label (cat or not?)
			return output.view(-1)

## Initialization
G = DCGAN_G()
D = DCGAN_D()

# Initialize weights
def weights_init(m):
	classname = m.__class__.__name__
	if classname.find('Conv') != -1:
		m.weight.data.normal_(0.0, 0.02)
	elif classname.find('BatchNorm') != -1:
		# Estimated variance, must be around 1
		m.weight.data.normal_(1.0, 0.02)
		# Estimated mean, must be around 0
		m.bias.data.fill_(0)
G.apply(weights_init)
D.apply(weights_init)
print("Initialized weights")
print("Initialized weights", file=log_output)

# Criterion
criterion = torch.nn.BCELoss()
BCE_stable = torch.nn.BCEWithLogitsLoss()
BCE_stable_noreduce = torch.nn.BCEWithLogitsLoss(reduce=False)

# Soon to be variables
x = torch.FloatTensor(param.batch_size, param.n_colors*2, param.image_size, param.image_size)
x_fake = torch.FloatTensor(param.batch_size, param.n_colors*2, param.image_size, param.image_size)
y = torch.FloatTensor(param.batch_size)
y2 = torch.FloatTensor(param.batch_size)
# Weighted sum of fake and real image, for gradient penalty
x_both = torch.FloatTensor(param.batch_size, param.n_colors*2, param.image_size, param.image_size)
z = torch.FloatTensor(param.batch_size, param.z_size, 1, 1)
# Uniform weight
u = torch.FloatTensor(param.batch_size, 1, 1, 1)
# This is to see during training, size and values won't change
z_test = torch.FloatTensor(param.batch_size, param.z_size, 1, 1).normal_(0, 1)
# For the gradients, we need to specify which one we want and want them all
grad_outputs = torch.ones(param.batch_size)

# Everything cuda
if param.cuda:
	G = G.cuda()
	D = D.cuda()
	criterion = criterion.cuda()
	BCE_stable.cuda()
	BCE_stable_noreduce.cuda()
	x = x.cuda()
	x_fake = x_fake.cuda()
	x_both = x_both.cuda()
	y = y.cuda()
	y2 = y2.cuda()
	u = u.cuda()
	z = z.cuda()
	z_test = z_test.cuda()
	grad_outputs = grad_outputs.cuda()

# Now Variables
x = Variable(x)
x_fake = Variable(x_fake)
y = Variable(y)
y2 = Variable(y2)
z = Variable(z)
z_test = Variable(z_test)

# Based on DCGAN paper, they found using betas[0]=.50 better.
# betas[0] represent is the weight given to the previous mean of the gradient
# betas[1] is the weight given to the previous variance of the gradient
optimizerD = torch.optim.Adam(D.parameters(), lr=param.lr_D, betas=(param.beta1, param.beta2), weight_decay=param.weight_decay)
optimizerG = torch.optim.Adam(G.parameters(), lr=param.lr_G, betas=(param.beta1, param.beta2), weight_decay=param.weight_decay)

# exponential weight decay on lr
decayD = torch.optim.lr_scheduler.ExponentialLR(optimizerD, gamma=1-param.decay)
decayG = torch.optim.lr_scheduler.ExponentialLR(optimizerG, gamma=1-param.decay)

# Load existing models
if param.load:
	checkpoint = torch.load(param.load)
	current_set_images = checkpoint['current_set_images']
	iter_offset = checkpoint['i']
	G.load_state_dict(checkpoint['G_state'])
	D.load_state_dict(checkpoint['D_state'])
	optimizerG.load_state_dict(checkpoint['G_optimizer'])
	optimizerD.load_state_dict(checkpoint['D_optimizer'])
	decayG.load_state_dict(checkpoint['G_scheduler'])
	decayD.load_state_dict(checkpoint['D_scheduler'])
	z_test.copy_(checkpoint['z_test'])
	del checkpoint
	print(f'Resumed from iteration {current_set_images*param.gen_every}.')
else:
	current_set_images = 0
	iter_offset = 0

print(G)
print(G, file=log_output)
print(D)
print(D, file=log_output)

## Fitting model
for i in range(iter_offset, param.n_iter):

	# Fake images saved
	if i % param.print_every == 0:
		fake_test = G(z_test)
		vutils.save_image(fake_test.data, '%s/images/fake_samples_iter%05d.png' % (base_dir, i), normalize=True)

	for p in D.parameters():
		p.requires_grad = True

	for t in range(param.Diters):

		########################
		# (1) Update D network #
		########################

		D.zero_grad()
		images = random_sample.__next__()
		current_batch_size = param.batch_size
		if param.cuda:
			images = images.cuda()
		# Transfer batch of images to x
		x.data.copy_(torch.cat([images[0:(current_batch_size),:],images[current_batch_size:(current_batch_size*2),:]],1))
		del images
		y_pred = D(x)

		if param.show_graph and i == 0:
			# Visualization of the autograd graph
			d = pv.make_dot(y_pred, D.state_dict())
			d.view()

		if param.loss_D in [1,2,3,4]:
			# Train with real data
			y.data.resize_(current_batch_size).fill_(1)
			if param.loss_D == 1:
				errD_real = criterion(y_pred, y)
			if param.loss_D == 2:
				errD_real = torch.mean((y_pred - y) ** 2)
				#a = torch.abs(y_pred - y)
				#errD_real = torch.mean(a**(1+torch.log(1+a**4)))
			if param.loss_D == 3:
				errD_real = -torch.mean(y_pred)
			if param.loss_D == 4:
				errD_real = torch.mean(torch.nn.ReLU()(1.0 - y_pred))
			errD_real.backward()

			# Train with fake data
			z.data.resize_(current_batch_size*2, param.z_size, 1, 1).normal_(0, 1)
			fake = G(z)
			x_fake.data.copy_(torch.cat([fake[0:(current_batch_size),:],fake[current_batch_size:(current_batch_size*2),:]],1))
			y.data.resize_(current_batch_size).fill_(0)
			# Detach y_pred from the neural network G and put it inside D
			y_pred_fake = D(x_fake.detach())
			if param.loss_D == 1:
				errD_fake = criterion(y_pred_fake, y)
			if param.loss_D == 2:
				errD_fake = torch.mean((y_pred_fake) ** 2)
				#a = torch.abs(y_pred_fake - y)
				#errD_fake = torch.mean(a**(1+torch.log(1+a**2)))
			if param.loss_D == 3:
				errD_fake = torch.mean(y_pred_fake)
			if param.loss_D == 4:
				errD_fake = torch.mean(torch.nn.ReLU()(1.0 + y_pred_fake))
			errD_fake.backward()
			errD = errD_real + errD_fake
			#print(errD)
		else:
			y.data.resize_(current_batch_size).fill_(1)
			y2.data.resize_(current_batch_size).fill_(0)
			z.data.resize_(current_batch_size*2, param.z_size, 1, 1).normal_(0, 1)
			fake = G(z)
			x_fake.data.copy_(torch.cat([fake[0:(current_batch_size),:],fake[current_batch_size:(current_batch_size*2),:]],1))
			y_pred_fake = D(x_fake.detach())
			if param.loss_D == 5:
				errD = BCE_stable(y_pred - y_pred_fake, y)
			if param.loss_D == 6:
				errD = (BCE_stable(y_pred - torch.mean(y_pred_fake), y) + BCE_stable(y_pred_fake - torch.mean(y_pred), y2))/2
			if param.loss_D == 7: # (y_hat-1)^2 + (y_hat+1)^2
				errD = (torch.mean((y_pred - torch.mean(y_pred_fake) - y) ** 2) + torch.mean((y_pred_fake - torch.mean(y_pred) + y) ** 2))/2
			if param.loss_D == 8:
				errD = (torch.mean(torch.nn.ReLU()(1.0 - (y_pred - torch.mean(y_pred_fake)))) + torch.mean(torch.nn.ReLU()(1.0 + (y_pred_fake - torch.mean(y_pred)))))/2
			errD_real = errD
			errD_fake = errD
			errD.backward()

		if (param.loss_D in [3] or param.grad_penalty):
			# Gradient penalty
			u.data.resize_(current_batch_size, 1, 1, 1)
			u.uniform_(0, 1)
			x_both = x.data*u + x_fake.data*(1-u)
			if param.cuda:
				x_both = x_both.cuda()
			# We only want the gradients with respect to x_both
			x_both = Variable(x_both, requires_grad=True)
			grad = torch.autograd.grad(outputs=D(x_both), inputs=x_both, grad_outputs=grad_outputs, retain_graph=True, create_graph=True, only_inputs=True)[0]
			# We need to norm 3 times (over n_colors x image_size x image_size) to get only a vector of size "batch_size"
			grad_penalty = param.penalty*((grad.norm(2, 1).norm(2,1).norm(2,1) - 1) ** 2).mean()
			grad_penalty.backward()
		optimizerD.step()


	########################
	# (2) Update G network #
	########################

	# Make it a tiny bit faster
	for p in D.parameters():
		p.requires_grad = False

	for t in range(param.Giters):

		G.zero_grad()
		y.data.resize_(current_batch_size).fill_(1)
		z.data.resize_(current_batch_size*2, param.z_size, 1, 1).normal_(0, 1)
		fake = torch.cat([fake[0:(current_batch_size),:],fake[current_batch_size:(current_batch_size*2),:]],1)
		y_pred_fake = D(fake)

		if param.loss_D not in [1, 2, 3, 4]:
			images = random_sample.__next__()
			current_batch_size = param.batch_size
			if param.cuda:
				images = images.cuda()
			x.data.copy_(torch.cat([images[0:(current_batch_size),:],images[current_batch_size:(current_batch_size*2),:]],1))
			del images

		if (param.loss_D == 1):
			errG = criterion(y_pred_fake, y)
		if param.loss_D == 2:
			errG = torch.mean((y_pred_fake - y) ** 2)
		if param.loss_D == 3:
			errG = -torch.mean(y_pred_fake)
		if param.loss_D == 4:
			errG = -torch.mean(y_pred_fake)
		if param.loss_D == 5:
			y_pred = D(x)
			# Non-saturating
			errG = BCE_stable(y_pred_fake - y_pred, y)
		if param.loss_D == 6:
			y_pred = D(x)
			# Non-saturating
			y2.data.resize_(current_batch_size).fill_(0)
			errG = (BCE_stable(y_pred - torch.mean(y_pred_fake), y2) + BCE_stable(y_pred_fake - torch.mean(y_pred), y))/2
		if param.loss_D == 7:
			y_pred = D(x)
			errG = (torch.mean((y_pred - torch.mean(y_pred_fake) + y) ** 2) + torch.mean((y_pred_fake - torch.mean(y_pred) - y) ** 2))/2
		if param.loss_D == 8:
			y_pred = D(x)
			# Non-saturating
			errG = (torch.mean(torch.nn.ReLU()(1.0 + (y_pred - torch.mean(y_pred_fake)))) + torch.mean(torch.nn.ReLU()(1.0 - (y_pred_fake - torch.mean(y_pred)))))/2
		errG.backward()
		D_G = y_pred_fake.data.mean()
		optimizerG.step()
	decayD.step()
	decayG.step()

	# Log results so we can see them in TensorBoard after
	#log_value('Diff', -(errD.data.item()+errG.data.item()), i)
	#log_value('errD', errD.data.item(), i)
	#log_value('errG', errG.data.item(), i)

	if (i+1) % param.print_every == 0:
		end = time.time()
		fmt = '[%d] Diff: %.4f param.loss_D: %.4f param.loss_G: %.4f time:%.4f'
		s = fmt % (i, -errD.data.item()+errG.data.item(), errD.data.item(), errG.data.item(), end - start)
		print(s)
		print(s, file=log_output)

	# Evaluation metrics
	if (i+1) % param.gen_every == 0:

		current_set_images += 1

		# Save models
		if param.save:
			if not os.path.exists('%s/models/' % (param.extra_folder)):
				os.mkdir('%s/models/' % (param.extra_folder))
			torch.save({
				'i': i + 1,
				'current_set_images': current_set_images,
				'G_state': G.state_dict(),
				'D_state': D.state_dict(),
				'G_optimizer': optimizerG.state_dict(),
				'D_optimizer': optimizerD.state_dict(),
				'G_scheduler': decayG.state_dict(),
				'D_scheduler': decayD.state_dict(),
				'z_test': z_test,
			}, '%s/models/state_%02d.pth' % (param.extra_folder, current_set_images))
			s = 'Models saved'
			print(s)
			print(s, file=log_output)

		# Delete previously existing images
		if os.path.exists('%s/%01d/' % (param.extra_folder, current_set_images)):
			for root, dirs, files in os.walk('%s/%01d/' % (param.extra_folder, current_set_images)):
				for f in files:
					os.unlink(os.path.join(root, f))
		else:
			os.mkdir('%s/%01d/' % (param.extra_folder, current_set_images))

		# Generate 50k images for FID/Inception to be calculated later (not on this script, since running both tensorflow and pytorch at the same time cause issues)
		ext_curr = 0
		z_extra = torch.FloatTensor(100, param.z_size, 1, 1)
		if param.cuda:
			z_extra = z_extra.cuda()
		for ext in range(int(param.gen_extra_images/100)):
			fake_test = G(Variable(z_extra.normal_(0, 1)))
			for ext_i in range(100):
				vutils.save_image((fake_test[ext_i].data*.50)+.50, '%s/%01d/fake_samples_%05d.png' % (param.extra_folder, current_set_images,ext_curr), normalize=False, padding=0)
				ext_curr += 1
		del z_extra
		del fake_test
		# Later use this command to get FID of first set:
		# python fid.py "/home/alexia/Output/Extra/01" "/home/alexia/Datasets/fid_stats_cifar10_train.npz" -i "/home/alexia/Inception" --gpu "0"
