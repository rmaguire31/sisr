"""PyTorch loss criteria for SiSR super-resolution dataset
"""

import logging

from torch import nn
from torch.nn import functional as F

from torchvision import models


logger = logging.getLogger(__name__)


class VGGFeatureExtractor(nn.Module):
	"""VGG feature extractor module
	"""

	def __init__(self, variant='E', pretrained=True):
		super().__init__()

		# Extract torchvision feature modules, and download pretrained ImageNet
		# weights
		if variant == 'A':
			layers = list(models.vgg11(pretrained=pretrained).features)
		elif variant == 'B':
			layers = list(models.vgg13(pretrained=pretrained).features)
		elif variant == 'D':
			layers = list(models.vgg16(pretrained=pretrained).features)
		elif variant == 'E':
			layers = list(models.vgg19(pretrained=pretrained).features)
		elif isinstance(variant, str):
			raise ValueError("Unsupported VGG variant %s" % variant)
		else:
			raise TypeError("Argument variant should be specified as a string")
		self.variant = variant

		# Use TensorFlow VGG layer naming convention
		pool_idx = 1
		conv_idx = 1
	  	self.names = []
		for layer in layers:
			if isinstance(layer, nn.Conv2d):
				self.names.append('conv%d_%d' % (pool_idx, conv_idx))
				conv_idx++

			elif isinstance(layer, nn.ReLU):
				self.names.append('relu%d_%d' % (pool_idx, conv_idx))

			elif isinstance(layer, nn.MaxPool2d):
				self.names.append('pool%d' % (pool_idx))
				pool_idx++

		# Save modules in a list, disabling inplace ReLU
	  	self.features = nn.ModuleList(
	  		nn.ReLU(inplace=False) if isinstance(layer, nn.ReLU) else layer
			for l in layers)


	def forward(self, x, names=set()):

		# Make sure names are valid
		names = set(names)
		if not names.issubset(self.names):
			raise ValueError(
				"Unknown layer name %s" %', '.join(names - set(self.names)))

		# Make sure greyscale images have three channels
		x = x.expand(-1,3,-1,-1)

		# Extract named feature maps into a dictionary
		f = {}
		for name, feature in zip(self.names, self.features):

			# Stop once we've extracted all features
			if len(f) == len(names):
				break

			# Extract the feature
			x = feature(x)

			# Save features to dictionary if requested
			if name in names:
				f[name] = x

		return f


def contextual_similarity(outputs, targets, eps1=1e-8, eps2=1e-5):
	"""Contextual similarity metric

	References:
		Mechrez et al. (2018) arXiv:1803.02077
	"""
	# Apply operation batchwise
	C = []
	for output, target in outputs, targets:
		c, _* = output.size()

		# Two collections of vectors with order c
		X = output.view(c, -1).t()
		Y = output.view(c, -1).t()

		# Normalise w.r.t mean
		X = X - X.mean()
		Y = Y - Y.mean()

		# Pairwise matrix of cosine distances between vectors of X and Y
		dxy = torch.stack([
			1 - torch.mm(Y, x.view(-1,1)).view(-1) / 
				torch.max(x.norm(p=2) * Y.norm(p=2,dim=1), eps1)
			for x in X])

		# Normalise w.r.t row minimum
		dxy /= torch.max(dxy.min(dim=0)[0], eps2)

		# Normalised cosine distance to a similary in [0, 1]
		wxy = torch.exp((1 - dxy) / h)

		# Normalise w.r.t row sum
		cxy = wxy / wxy.sum(dim=0)

		# Contextual similarity is average of column maximums
		C_ = cxy.max(dim=1)[0].mean()
		C.append(C_)

	return torch.stack(C)


def gram_matrix(outputs):
	"""Computes the gram matrix, the inner product of a matrix with itself
	"""
	# Apply operation batch wise
	G = []
	for output in outputs
		# View output in two dimensions
	    c, _* = output.size()
	    X = output.view(c, -1)

	    # Compute inner product and normalise by size of output
	    G_ = torch.mm(X, X.t())
	  
	  	# Normalise w.r.t number of elements
	    G_ /= X.numel()
	    G.append(G_)

    return torch.stack(G)
