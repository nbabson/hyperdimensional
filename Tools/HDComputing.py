from __future__ import annotations

import random
import numpy as np
from scipy.spatial.distance import hamming
from sklearn.metrics.pairwise import cosine_similarity
from functools import wraps
from typing import Type, Tuple, Union
from mypy_extensions import TypedDict

random.seed()

# TODO: Add cosine similarity, permutation
# TODO: Add function argument for method override => class inheritance
# TODO: Record-based encoding: lower orthogonality on closer level vectors
# TODO: n-gram


def add_method(cls):
	def decorator(func):
		@wraps(func)
		def wrapper(self, *args, **kwargs):
			return func(*args, **kwargs)

		setattr(cls, func.__name__, wrapper)
		# Note we are not binding func, but wrapper which accepts self but does exactly the same as func
		return func  # returning func means func can still be used normally

	return decorator


class Vector:
	def __init__(self, dim: int, value: np.ndarray):
		self.dim = dim
		self.value = value

	def __add__(self, other: Vector) -> Vector:
		raise NotImplementedError("Bundling operation '+' must be implemented")

	def __mul__(self, other: Vector) -> Vector:
		raise NotImplementedError("Binding operation '*' must be implemented")

	def __getitem__(self, item) -> Vector:
		return type(self)(self.dim, self.value[item])


class BSCVector(Vector):
	def __init__(self, dim: Union[Tuple[int, int], int], value: np.ndarray = None):
		"""
		:param dim: dimension value or tuple of (number of vectors, dimension)
		:param value: value(s) to be assigned as BSCVector
		"""
		if value is None:
			value = np.random.randint(2, size=dim)
		super().__init__(dim, value)

	def __add__(self, x: BSCVector) -> BSCVector:
		z = self.value + x.value
		z[z == 1] = np.random.randint(2, size=len(z[z == 1]))
		z[z == 2] = np.ones(len(z[z == 2]))
		return BSCVector(self.dim, z)

	def __mul__(self, y: BSCVector) -> BSCVector:
		z = np.bitwise_xor(self.value, y.value)
		return BSCVector(self.dim, z)

	# def __getitem__(self, item):
	# 	return BSCVector(self.dim, self.value[item])

	def __invert__(self):
		# TODO: permute
		pass

	def __or__(self, y: BSCVector) -> float:
		# TODO: distance
		return hamming(self.value, y.value)

	def __eq__(self, y: BSCVector) -> bool:
		return np.array_equal(self.value, y.value)

	def __ne__(self, y: BSCVector) -> bool:
		return not np.array_equal(self.value, y.value)

	def __repr__(self) -> str:
		return np.array2string(self.value)

	def dist(x: BSCVector, y: BSCVector) -> float:
		return hamming(x.value, y.value)

	def similar(x: BSCVector, y: BSCVector) -> float:
		return cosine_similarity(x.value, y.value)


class BipolarVector(Vector):
	def __init__(self, dim: Union[Tuple[int, int], int], value: np.ndarray = None):
		"""
		:param dim: dimension value or tuple of (number of vectors, dimension)
		:param value: value(s) to be assigned as BipolarVector
		"""
		if value is None:
			value = np.random.choice([-1.0, 1.0], size=dim)
		super().__init__(dim, value)

	def __add__(self, y: BipolarVector) -> BipolarVector:
		z = np.clip(self.value + y.value, a_min=-1.0, a_max=1.0)
		z[z == 0] = np.random.choice([-1.0, 1.0], size=len(z[z == 0]))
		return BipolarVector(self.dim, z)

	def __mul__(self, y: BipolarVector) -> BipolarVector:
		return BipolarVector(self.dim, self.value * y.value)

	# def __getitem__(self, item):
	# 	return BipolarVector(self.dim, self.value[item])

	def __invert__(self):
		# TODO: permute
		pass

	def __or__(self, y: BipolarVector) -> float:
		# TODO: distance
		return (len(self.value) - np.dot(self.value, y.value)) / (2 * float(len(self.value)))

	def __eq__(self, y: BipolarVector) -> bool:
		return np.array_equal(self.value, y.value)

	def __ne__(self, y: BipolarVector) -> bool:
		return not np.array_equal(self.value, y.value)

	def __repr__(self) -> str:
		return np.array2string(self.value)

	def dist(x: BipolarVector, y: BipolarVector) -> float:
		# TODO: Check replacing by np.round(1 - (np.count_nonzero(a + b) / float(len(a))), int(np.log10(dim))), is slower
		return (len(x.value) - np.dot(x.value, y.value)) / (2 * float(len(x.value)))

	def similar(x: BipolarVector, y: BipolarVector) -> float:
		# TODO: Bipolar similarity
		pass


class BSDVector(Vector):
	def __init__(self, dim: Union[Tuple[int, int], int], value: np.ndarray = None):
		"""
		:param dim: dimension value or tuple of (number of vectors, dimension)
		:param value: value(s) to be assigned as BSDVector
		"""
		# TODO make probability as a param
		if value is None:
			# sparsity << 0.5
			sparsity = 0.2
			value = np.random.choice([0, 1], size=dim, p=[1 - sparsity, sparsity])
		super().__init__(dim, value)

	def __add__(self, y: BSDVector) -> BSDVector:
		z0 = np.bitwise_or(self.value, y.value)
		# permutation factor
		k = 8
		# zk = np.fromfunction(lambda i, j: np.random.permutation(z0), (k, 1), dtype=int)
		# z = np.bitwise_or.reduce(zk)

		zk = np.zeros((k, self.value.shape[0]), dtype=int)
		for i in range(0, k):
			zk[i] = np.random.permutation(z0)
		z = np.bitwise_or.reduce(zk)

		return BSDVector(self.dim, np.bitwise_and(z, z0))

	def __mul__(self, y: BSDVector) -> BSDVector:
		z0 = np.bitwise_or(self.value, y.value)
		# permutation factor
		k = 8
		# zk = np.fromfunction(lambda i, j: np.random.permutation(z0), (k, 1), dtype=int)
		# z = np.bitwise_or.reduce(zk)

		zk = np.zeros((k, self.value.shape[0]), dtype=int)
		for i in range(0, k):
			zk[i] = np.random.permutation(z0)
		z = np.bitwise_or.reduce(zk)

		return BSDVector(self.dim, np.bitwise_and(z, z0))

	# def __getitem__(self, item):
	# 	return BSDVector(self.dim, self.value[item])

	def __invert__(self):
		# TODO: permute
		pass

	def __or__(self, y: BSDVector) -> float:
		# TODO: distance
		d = 1 - np.sum(np.bitwise_and(self.value, y.value)) / np.sqrt(np.sum(self.value) * np.sum(y.value))
		return d

	def __eq__(self, y: BSDVector) -> bool:
		return np.array_equal(self.value, y.value)

	def __ne__(self, y: BSDVector) -> bool:
		return not np.array_equal(self.value, y.value)

	def __repr__(self) -> str:
		return np.array2string(self.value)

	def dist(x: BSDVector, y: BSDVector) -> float:
		d = 1 - np.sum(np.bitwise_and(x.value, y.value)) / np.sqrt(np.sum(x.value) * np.sum(y.value))
		return d

	def similar(x: BSDVector, y: BSDVector) -> float:
		# TODO: Bipolar similarity
		pass


class RecordEncodingDict(TypedDict):
	"""
	N: number of features to be encoded

	M: number of levels

	range: tuple of [low, high] range for levels
	"""
	N: int
	M: int
	range: Tuple[int, int]


def record_encode(dim: int, rep: Type[Vector], enc: RecordEncodingDict, features: np.ndarray, id_vectors: Type[Vector] = None, level_vectors: Type[Vector] = None) -> Tuple[Type[Vector], Type[Vector], Type[Vector]]:
	"""
	:param dim: dimensions of hypervector
	:param rep: representation as an inherited Vector object
	:param enc: RecordEncodingDict wth encoding parameters
	:param features: 1D ndarray of features to be encoded
	:param id_vectors: ID vectors to be used
	:param level_vectors: Level vectors to be used
	:return: id_vectors, level_vectors, encoded hypervector
	"""

	# Create empty vector of given representation
	S = rep(dim=dim, value=np.zeros(dim))

	if id_vectors is None:
		# id_vectors = np.random.randint(2, size=(enc['N'], dim))
		# level_vectors = np.random.randint(2, size=(enc['M'], dim))
		id_vectors = rep(dim=(enc['N'], dim))
		level_vectors = rep(dim=(enc['M'], dim))

	bins = np.linspace(enc['range'][0], enc['range'][1], enc['M'] + 1)
	level_dict = {(x, y): l for x, y, l in zip(bins, bins[1:], level_vectors)}

	encoded = np.empty((enc['N'], dim))
	for num, feature in enumerate(features):
		# encoded[num] = [self.mul_bsc(level_dict[(low, high)], id_vectors[num]) for (low, high) in level_dict if (feature >= low) and (feature <= high)][0]
		for (low, high) in level_dict:
			if (feature >= low) and (feature <= high):
				v1 = level_dict[(low, high)]
				v2 = id_vectors[num]
				val = v1 * v2
				encoded[num] = val.value
				break

	for vector in encoded:
		S = S + rep(dim=dim, value=vector)

	return id_vectors, level_vectors, S


class Hyperspace:
	def __init__(self, dim: int = 1000, rep: Type[Vector] = BSCVector, enc: str = None) -> None:
		self.dim = dim
		self.rep = rep
		self.enc = enc
		self.vectors = {}

		if self.enc is not None and self.enc == 'record':
			self.id_vectors = None
			self.level_vectors = None

	def __random_name(self):
		return ''.join(random.choice('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ') for i in range(8))

	def __repr__(self):
		return ''.join("'%s' , %s\n" % (v, self.vectors[v]) for v in self.vectors)

	def __getitem__(self, name: str):
		return self.vectors[name]

	def add(self, name: str = None, features: np.ndarray = None) -> Vector:
		if self.enc is not None and self.enc == 'record':
			if features is None:
				raise ValueError('Record-based encoded vectors must supply features to be added')
			elif self.id_vectors is None:
				self.id_vectors, self.level_vectors, _ = record_encode(self.dim, self.rep, self.enc)
			else:
				_, _, _ = record_encode(self.dim, self.rep, self.enc)

		if name is None:
			name = self.__random_name()

		v = self.rep(dim=self.dim)

		self.vectors[name] = v
		return v

	def insert(self, v: Vector, name: str = None) -> str:
		if name is None:
			name = self.__random_name()

		self.vectors[name] = v

		return name

	def find(self, x: Type[Vector]) -> (Type[Vector], float):
		d = 1.0
		match = None

		for v in self.vectors:
			if self.vectors[v].dist(x) < d:
				match = v
				d = self.vectors[v].dist(x)

		# print d
		return match, d
