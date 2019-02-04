"""PyTorch loss utilities for SiSR super-resolution network
"""

import logging

import torch


logger = logging.getLogger(__name__)


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
