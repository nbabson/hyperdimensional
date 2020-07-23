from __future__ import annotations

import random

import numpy as np
from scipy.spatial.distance import hamming
from functools import wraps

random.seed()


# TODO: Add cosine similarity
# TODO: Add function argument for method override
# TODO: Record-based encoding


def add_method(cls):
	def decorator(func):
		@wraps(func)
		def wrapper(self, *args, **kwargs):
			return func(*args, **kwargs)

		setattr(cls, func.__name__, wrapper)
		# Note we are not binding func, but wrapper which accepts self but does exactly the same as func
		return func  # returning func means func can still be used normally

	return decorator


# noinspection PyMethodParameters
class Hypervector:

	# define: INIT, ADD, MUL, and DIST operations depending on representation

	# representation template
	def __init_xxx(dim, enc):
		"""
		Create a new (random) hyperdimensional vector and return it
		"""
		return None

	def __add_xxx(x, y):
		"""
		Create a new hyperdimensional vector, z, that is the result of the addition (bundling) of two hyperdimensional vectors x and y
		"""
		# TODO: Figure out majority rule addition
		return None

	def __mul_xxx(x, y):
		"""
		Create a new hyperdimensional vector, z, that is the result of the multoplication (binding) of two hyperdimensional vectors x and y
		"""
		return None

	def __dist_xxx(x, y):
		"""
		Return the distance between the two hyperdimensional vectors x and y
		"""
		return None

	# Binary Spatter Code
	def init_bsc(self) -> np.ndarray:
		if self.enc is None:
			return np.random.randint(2, size=self.dim)
		elif self.enc['type'] == 'record':
			H = np.zeros(self.dim)
			id_vectors = np.random.randint(2, size=(self.enc['N'], self.dim))
			level_vectors = np.random.randint(2, size=(self.enc['M'], self.dim))
			bins = np.linspace(self.enc['range'][0], self.enc['range'][1], self.enc['M']+1)
			level_dict = {(x, y): l for x, y, l in zip(bins, bins[1:], level_vectors)}

			encoded = np.empty((self.enc['N'], self.dim))
			for num, feature in enumerate(self.features):
				# encoded[num] = [self.mul_bsc(level_dict[(low, high)], id_vectors[num]) for (low, high) in level_dict if (feature >= low) and (feature <= high)][0]
				for (low, high) in level_dict:
					if (feature >= low) and (feature <= high):
						encoded[num] = self.mul_bsc(level_dict[(low, high)], id_vectors[num])
						break

			for vector in encoded:
				H = self.add_bsc(H, vector)

			return H

	def add_bsc(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
		z = x + y
		z[z == 1] = np.random.randint(2, size=len(z[z == 1]))
		z[z == 2] = np.ones(len(z[z == 2]))
		return z

	def mul_bsc(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
		z = np.bitwise_xor(x, y)
		return z

	def dist_bsc(self, x: np.ndarray, y: np.ndarray) -> float:
		return hamming(x, y)

	# Bipolar
	def __init_bipolar(dim: int, enc: dict, features: list = None) -> np.ndarray:
		return np.random.choice([-1.0, 1.0], size=dim)

	def __add_bipolar(x: np.ndarray, y: np.ndarray) -> np.ndarray:
		z = np.clip(x + y, a_min=-1.0, a_max=1.0)
		z[z == 0] = np.random.choice([-1.0, 1.0], size=len(z[z == 0]))
		return z

	def __mul_bipolar(x: np.ndarray, y: np.ndarray) -> np.ndarray:
		return x * y

	def __dist_bipolar(x: np.ndarray, y: np.ndarray) -> float:
		# TODO: Check replacing by np.round(1 - (np.count_nonzero(a + b) / float(len(a))), int(np.log10(dim))), is slower
		return (len(x) - np.dot(x, y)) / (2 * float(len(x)))

	# Binary Sparse
	def __init_bsd(dim: int, enc: dict, features: list = None) -> np.ndarray:
		# TODO make probability as a param
		# sparsity << 0.5
		sparsity = 0.2
		return np.random.choice([0, 1], size=dim, p=[1 - sparsity, sparsity])

	def __add_bsd(x: np.ndarray, y: np.ndarray) -> np.ndarray:
		# bundling in BSD is nothing but a fancy binding, role-filler scheme - use same code
		# TODO refactor to explicitly reuse the same function

		z0 = np.bitwise_or(x, y)
		# permutation factor
		k = 8

		zk = np.zeros((k, x.shape[0]), dtype=int)
		for i in range(0, k):
			zk[i] = np.random.permutation(z0)
		z = np.bitwise_or.reduce(zk)

		return np.bitwise_and(z, z0)

	def __mul_bsd(x: np.ndarray, y: np.ndarray) -> np.ndarray:
		z0 = np.bitwise_or(x, y)
		# permutation factor
		k = 8
		# zk = np.fromfunction(lambda i, j: np.random.permutation(z0), (k, 1), dtype=int)
		# z = np.bitwise_or.reduce(zk)

		zk = np.zeros((k, x.shape[0]), dtype=int)
		for i in range(0, k):
			zk[i] = np.random.permutation(z0)
		z = np.bitwise_or.reduce(zk)

		return np.bitwise_and(z, z0)

	def __dist_bsd(x: np.ndarray, y: np.ndarray) -> float:
		d = 1 - np.sum(np.bitwise_and(x, y)) / np.sqrt(np.sum(x) * np.sum(y))
		return d

	# operations list
	__OPERATIONS = {
		'bsc': {
			'init': init_bsc,
			'add': add_bsc,
			'mul': mul_bsc,
			'dist': dist_bsc
		},
		'bsd': {
			'init': __init_bsd,
			'add': __add_bsd,
			'mul': __mul_bsd,
			'dist': __dist_bsd
		},
		'bipolar': {
			'init': __init_bipolar,
			'add': __add_bipolar,
			'mul': __mul_bipolar,
			'dist': __dist_bipolar
		},
	}

	# Initialize random HD vector
	def __init__(self, dim: int, rep: str, enc: dict, features: list = None) -> None:
		self.rep = rep
		self.dim = dim
		self.enc = enc
		self.features = features
		self.value = self.__OPERATIONS['bsc']['init'](self)

	# Print vector
	def __repr__(self):
		return np.array2string(self.value)

	# Print vector
	def __str__(self):
		return np.array2string(self.value)

	# Addition
	def __add__(self, a: Hypervector) -> Hypervector:
		b = Hypervector(self.dim, self.rep, self.enc)
		b.value = self.__OPERATIONS[self.rep]['add'](self.value, a.value)
		return b

	# Multiplication
	def __mul__(self, a: Hypervector) -> Hypervector:
		b = Hypervector(self.dim, self.rep, self.enc)
		b.value = self.__OPERATIONS[self.rep]['mul'](self.value, a.value)
		return b

	# Distance
	def dist(self, a: Hypervector) -> float:
		return self.__OPERATIONS[self.rep]['dist'](self.value, a.value)

	@classmethod
	def add_repr(cls, repr):
		cls.__OPERATIONS['abc'] = repr


class Space:

	def __init__(self, dim: int = 1000, rep: str = 'bsc', enc: dict = None) -> None:
		self.dim = dim
		self.rep = rep
		self.enc = enc
		self.vectors = {}

	def _random_name(self):
		return ''.join(random.choice('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ') for i in range(8))

	def __repr__(self):
		return ''.join("'%s' , %s\n" % (v, self.vectors[v]) for v in self.vectors)

	def __getitem__(self, name: str):
		return self.vectors[name]

	def add(self, name: str = None, features: list = None) -> Hypervector:
		if self.enc is not None and self.enc['type'] == 'record':
			if features is None:
				raise ValueError('Record-based encoded vectors must supply features to be added')

		if name is None:
			name = self._random_name()

		v = Hypervector(self.dim, self.rep, self.enc, features)

		self.vectors[name] = v
		return v

	def insert(self, v: Hypervector, name: str = None) -> str:
		if name is None:
			name = self._random_name()

		self.vectors[name] = v

		return name

	def find(self, x: Hypervector) -> (Hypervector, float):
		d = 1.0
		match = None

		for v in self.vectors:
			if self.vectors[v].dist(x) < d:
				match = v
				d = self.vectors[v].dist(x)

		# print d
		return match, d
