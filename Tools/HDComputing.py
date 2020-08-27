from __future__ import annotations

import random
import numpy as np
from scipy.spatial.distance import hamming, cosine
from functools import wraps
from typing import Type, Tuple, Union
from mypy_extensions import TypedDict
from abc import ABC, abstractmethod

random.seed()

# TODO: Add MIT license
# TODO: Cosine similarity operator
# TODO: Majority-rule additions
# TODO: n-gram
# TODO: Overload iterator for Vector
# TODO: Add empty value init for all

# VERIFY: Verify permutation


def add_method(cls):
	def decorator(func):
		@wraps(func)
		def wrapper(self, *args, **kwargs):
			return func(*args, **kwargs)

		setattr(cls, func.__name__, wrapper)
		return func

	return decorator


class Vector(ABC):
	def __init_subclass__(cls, **kwargs):
		if not hasattr(cls, 'permutation_indices'):
			raise TypeError("Can't instantiate class " + cls.__name__ + " with abstract variable permutation_indices")
		return super().__init_subclass__(**kwargs)

	@abstractmethod
	def __init__(self, dim: Union[Tuple[int, int], int], value: np.ndarray):
		"""
		:param dim: dimension or tuple of (number of vectors, dimension)
		:param value: value(s) to be assigned as BSCVector, randomly initialized if empty
		"""
		if dim is tuple:
			self.dim = dim[1]
		else:
			self.dim = dim
		self.value = value

		# Add permutation indices to static dictionary for first occurrence of dimension
		if self.dim not in self.__class__.permutation_indices:
			self.__class__.permutation_indices[self.dim] = np.random.permutation(self.dim)

	@abstractmethod
	def __add__(self, other: Vector) -> Vector:
		"""
		Bundling operator
		"""
		# raise NotImplementedError("Bundling operation '+' must be implemented")
		pass

	@abstractmethod
	def __mul__(self, other: Vector) -> Vector:
		"""
		Binding operator
		"""
		# raise NotImplementedError("Binding operation '*' must be implemented")
		pass

	def __invert__(self):
		"""
		Permutation operator
		"""
		permuted_value = self.value[self.__class__.permutation_indices[self.dim]]
		return self.__class__(dim=self.dim, value=permuted_value)

	def __or__(self, other: Vector) -> Vector:
		"""
		Distance operator
		"""
		pass

	def __eq__(self, other: Vector) -> bool:
		"""
		Returns np.array_equal on hypervector values
		"""
		return np.array_equal(self.value, other.value)

	def __ne__(self, other: Vector) -> bool:
		"""
		Returns not np.array_equal on hypervector values
		"""
		return not np.array_equal(self.value, other.value)

	def __getitem__(self, item) -> Vector:
		"""
		Indexing operator for hypervectors
		"""
		return type(self)(self.dim, self.value[item])

	def __setitem__(self, key, value) -> None:
		"""
		Set value in multi-valued hypervectors
		"""
		if type(value) is np.ndarray:
			self.value[key] = value
		elif isinstance(value, Vector):
			self.value[key] = value.value
		else:
			raise TypeError('Value must be an np.ndarray or object of a Vector sub-class')


class BSCVector(Vector):
	# Static dictionary to maintain common permutation indices across objects
	permutation_indices = {}

	def __init__(self, dim: Union[Tuple[int, int], int], value: np.ndarray = None, empty: bool = False):
		if value is None:
			if empty:
				value = np.zeros(shape=dim, dtype=np.int)
			else:
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

	def __or__(self, y: BSCVector) -> float:
		return hamming(self.value, y.value)

	def __repr__(self) -> str:
		return np.array2string(self.value)

	def dist(x: BSCVector, y: BSCVector) -> float:
		return hamming(x.value, y.value)

	def similar(x: BSCVector, y: BSCVector) -> float:
		return cosine(x.value, y.value)


class BipolarVector(Vector):
	# Static dictionary to maintain common permutation indices across objects
	permutation_indices = {}

	def __init__(self, dim: Union[Tuple[int, int], int], value: np.ndarray = None):
		if value is None:
			value = np.random.choice([-1.0, 1.0], size=dim)
		super().__init__(dim, value)

	def __add__(self, y: BipolarVector) -> BipolarVector:
		z = np.clip(self.value + y.value, a_min=-1.0, a_max=1.0)
		z[z == 0] = np.random.choice([-1.0, 1.0], size=len(z[z == 0]))
		return BipolarVector(self.dim, z)

	def __mul__(self, y: BipolarVector) -> BipolarVector:
		return BipolarVector(self.dim, self.value * y.value)

	def __or__(self, y: BipolarVector) -> float:
		return (len(self.value) - np.dot(self.value, y.value)) / (2 * float(len(self.value)))

	def __repr__(self) -> str:
		return np.array2string(self.value)

	def dist(x: BipolarVector, y: BipolarVector) -> float:
		# VERIFY: Replacing by np.round(1 - (np.count_nonzero(a + b) / float(len(a))), int(np.log10(dim))), is slower
		return (len(x.value) - np.dot(x.value, y.value)) / (2 * float(len(x.value)))

	def similar(x: BipolarVector, y: BipolarVector) -> float:
		# TODO: Bipolar similarity
		pass


class BSDVector(Vector):
	# Static dictionary to maintain common permutation indices across objects
	permutation_indices = {}

	def __init__(self, dim: Union[Tuple[int, int], int], value: np.ndarray = None):
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

	def __or__(self, y: BSDVector) -> float:
		d = 1 - np.sum(np.bitwise_and(self.value, y.value)) / np.sqrt(np.sum(self.value) * np.sum(y.value))
		return d

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
	# VERIFY: Infer N from dimensions of features
	# Create empty vector of given representation
	S = rep(dim=dim, empty=True)

	if id_vectors is None:
		# id_vectors = np.random.randint(2, size=(enc['N'], dim))
		# level_vectors = np.random.randint(2, size=(enc['M'], dim))
		id_vectors = rep(dim=(enc['N'], dim))

		num_flip = int(dim / (enc['M'] - 1))
		l_vectors = np.zeros(shape=(enc['M'], dim), dtype=np.int)
		l_vectors[0] = np.random.randint(2, size=dim, dtype=np.int)

		# TODO: Fix flipping for reps other than {0,1} values
		for index in range(1, enc['M']):
			l_vectors[index] = l_vectors[index - 1]
			flip_indices = np.random.randint(0, dim, size=num_flip)
			l_vectors[index][flip_indices] = 1 - l_vectors[index - 1][flip_indices]

		level_vectors = rep(dim=(enc['M'], dim), value=l_vectors)

	bins = np.linspace(enc['range'][0], enc['range'][1], enc['M'] + 1)
	level_dict = {(x, y): l for x, y, l in zip(bins, bins[1:], level_vectors)}

	encoded = rep(dim=dim, value=np.zeros((enc['N'], dim)))
	for num, feature in enumerate(features):
		# encoded[num] = [self.mul_bsc(level_dict[(low, high)], id_vectors[num]) for (low, high) in level_dict if (feature >= low) and (feature <= high)][0]
		# VERIFY: Biased towards higher bins, randomly choose?
		for (low, high), level_vector in level_dict.items():
			if low <= feature < high:
				# v1 = level_vector
				# v2 = id_vectors[num]
				# val = v1 * v2
				# encoded[num] = level_vector * id_vectors[num]
				S = S + (level_vector * id_vectors[num])

				# print('Encoded ID vector ' + str(num) + ' at level (' + str(low) + ', ' + str(high) + ')')
				break

	# for vector in encoded:
	# 	S = S + vector

	S.value = S.value.astype(dtype=np.int)

	return id_vectors, level_vectors, S


class Hyperspace:
	def __init__(self, dim: int = 1000, rep: Type[Vector] = BSCVector, enc: dict = None) -> None:
		""" Hyperspace to contain hypervectors of defined dimensions, representation and encoding method

		:param dim: dimensions of hypervector
		:param rep: representation as a subclass of Vector
		:param enc: dictionary with keys for 'record' or 'n-gram'
		"""
		self.dim = dim
		self.rep = rep
		self.enc = enc
		self.vectors = {}

		# Create ID and level vectors for record-based encoding
		if self.enc is not None and 'record' in self.enc:
			self.id_vectors = None
			self.level_vectors = None

	def __random_name(self):
		return ''.join(random.choice('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ') for i in range(8))

	def __repr__(self):
		return ''.join("'%s' , %s\n" % (v, self.vectors[v]) for v in self.vectors)

	def __getitem__(self, name: str) -> Type[Vector]:
		return self.vectors[name]

	def add(self, name: str = None, features: np.ndarray = None) -> Type[Vector]:
		""" Add new hypervector to space

		:param name: Name for hypervector, randomly generated if empty
		:param features: ndarray of features for record-based or n-gram encoding
		:param replace: replace vector with existing name, else adds to it
		:return: Vector subclass defined in representation
		"""
		if name is None:
			name = self.__random_name()

		# TODO: Add option to specify number of vectors generated, return as list
		if self.enc is not None and 'record' in self.enc:
			if features is None:
				raise ValueError('Record-based encoded vectors must supply features to be added')
			elif self.id_vectors is None:
				self.id_vectors, self.level_vectors, v = record_encode(self.dim, self.rep, self.enc['record'], features)
			else:
				_, _, v = record_encode(self.dim, self.rep, self.enc['record'], features)

			# Build class hypervectors
			if name not in self.vectors:
				self.vectors[name] = v
			else:
				self.vectors[name] = self.vectors[name] + v

		else:
			v = self.rep(dim=self.dim)
			self.vectors[name] = v

		return v

	def insert(self, v: Type[Vector], name: str = None) -> None:
		""" Insert hypervector in space

		:param v: Vector subclass object
		:param name: Name of hypervector
		"""
		if name is None:
			name = self.__random_name()

		self.vectors[name] = v

	def query(self, x: Type[Vector]) -> Tuple[Type[Vector], float]:
		""" Find hypervector in space closest to input hypervector

		:param x: Hypervector to measure distance against
		:return: (closest hypervector, distance)
		"""
		d = 1.0
		match = None

		for v in self.vectors:
			if self.vectors[v].dist(x) < d:
				match = v
				d = self.vectors[v].dist(x)

		return match, d
