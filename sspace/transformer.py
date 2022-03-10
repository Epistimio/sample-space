# -*- coding: utf-8 -*-
# pylint: disable=too-many-lines
"""
Perform transformations on Dimensions
=====================================

Used to lower high level dimensions to support more algorithms

"""

from dataclasses import dataclass
import logging
from typing import Optional

from sspace.orion.transformer import Enumerate
from sspace.space import Dimension, Categorical, Space


def needs_flatten(shape):
    if isinstance(shape, (list, tuple)):
        return True

    return shape > 1


@dataclass
class PriorDomain:
    type: str
    linear: bool = True
    shape: Optional[tuple] = None


class Transform:
    """Interface to a dimension/sample transformer"""

    def apply(self, dim: Dimension, sample):
        """Apply the transforms to a given sample"""
        return sample

    def unapply(self, dim: Dimension, sample):
        """Remove transforms to a given sample"""
        return sample

    def transform(self, dim: Dimension) -> Dimension:
        """Returns the transformed dimensions"""
        return dim


class Transformer(Transform):
    """Collection of transforms that can be apply/unapplied on samples and dimensions"""
    def __init__(self) -> None:
        self.transforms = []

    # Transformer API
    # ===============

    def apply(self, dim: Dimension, sample):
        """Apply the transforms to a given sample"""
        for transform in self.transforms:
            sample = transform.apply(sample)
        return sample

    def unapply(self, dim: Dimension, sample):
        """Remove transforms to a given sample"""
        for transform in self.transforms:
            sample = transform.unapply(sample)
        return sample

    def transform(self, dim: Dimension) -> Dimension:
        """Returns the transformed dimensions"""
        for transform in self.transforms:
            sample = transform.transform(dim)
        return sample

    def _guard(self, dim: Dimension, method: str) -> bool:
        """Apply a transform recursively of the dimension is a space"""
        if isinstance(dim, Space):
            for _, dim in dim.space_tree.items():
                getattr(dim.transformer(), method)(dim)

            return False

        return True

    # Transform builders
    # ==================

    # Categorical
    def enumerate(self, dim: Dimension):
        """Transform a categorical dimension into discrete integers [0-n]"""
        if self._guard(dim, 'enumerate'):
            return self

        if isinstance(dim, Categorical):
            self.transforms.append(Enumerate)

        return self

    def onehot(self, dim: Dimension):
        """Transform a categorical dimension into discrete integers"""
        if self._guard(dim, 'onehot'):
            return self

        if isinstance(dim, Categorical):
            both = True
            for t in self.transforms:
                if isinstance(t, Enumerate):
                    both = False

            if both:
                self.transforms.extend([Enumerate, OneHotEncode])

        return self

    # Scalars
    def quantize(self, dim: Dimension):
        """Round not discrete dimension to integers"""
        return self.round(dim)

    def round(self, dim: Dimension):
        """Round not discrete dimension to integers"""
        if hasattr(dim, 'discrete') and not dim.discrete:
            self.transforms.extend([Round])

        return self

    def decimal(self, dim: Dimension):
        """Convert integer to decimal"""

        if hasattr(dim, 'discrete') and dim.discrete:
            self.transforms.extend([Decimal])

        return self

    def precision(self, dim: Dimension):
        """Limit scalar's range"""
        if hasattr(dim, 'precision') and dim.precision:
            self.transforms.extend([Precision])

        return self

    # Shape Requirement
    def reshape(self, dim: Dimension):
        """Flatten the given dimensions if dim.shape > 1"""
        return self.flatten(dim)

    def flatten(self, dim: Dimension):
        """Flatten the given dimensions if dim.shape > 1"""
        if self._guard(dim, 'flatten'):
            return self

        if hasattr(dim, 'shape') and needs_flatten(dim.shape):
            self.transforms.append(Flatten)

        return self

    # Dist Requirepment
    def linear(self, dim: Dimension):
        """Linearize all logarithmic prior"""
        if self._guard(dim, 'flatten'):
            return self

        if hasattr(dim, 'log') and dim.log:
            self.transforms.append(Linearize)

        return self

    # Type Requirement
    def numerical(self, dim: Dimension):
        if isinstance(dim, Categorical):
            self.enumerate(dim)

    def integer(self, dim: Dimension):
        if isinstance(dim, Categorical):
            self.enumerate(dim)
        else:
            self.round(dim)

    def real(self, dim: Dimension):
        if isinstance(dim, Categorical):
            # so the value will be a vector
            # we can find the category by checking the max
            self.onehot(dim)

        else:
            self.floats()


Identity = Transform
