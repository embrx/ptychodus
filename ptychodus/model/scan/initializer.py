from __future__ import annotations
from abc import ABC, abstractmethod, abstractproperty
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import overload, Union
import logging

import numpy

from ...api.observer import Observable, Observer
from ...api.scan import ScanPoint, ScanPointSequence, ScanPointTransform
from .settings import ScanSettings

logger = logging.getLogger(__name__)


@dataclass
class ScanInitializerParameters:
    rng: numpy.random.Generator
    transform: ScanPointTransform = ScanPointTransform.PXPY
    jitterRadiusInMeters: Decimal = Decimal()
    centroid: ScanPoint = ScanPoint(Decimal(), Decimal())

    @classmethod
    def createFromSettings(cls, rng: numpy.random.Generator,
                           settings: ScanSettings) -> ScanInitializerParameters:
        transform = ScanPointTransform.fromSimpleName(settings.transform.value)
        jitterRadiusInMeters = settings.jitterRadiusInMeters.value
        centroid = ScanPoint(settings.centroidXInMeters.value, settings.centroidYInMeters.value)
        return cls(rng, transform, jitterRadiusInMeters, centroid)

    def syncToSettings(self, settings: ScanSettings) -> None:
        settings.transform.value = self.transform.simpleName
        settings.jitterRadiusInMeters.value = self.jitterRadiusInMeters
        settings.centroidXInMeters.value = self.centroid.x
        settings.centroidYInMeters.value = self.centroid.y


class ScanInitializer(ScanPointSequence, Observable):
    '''ABC for plugins that can initialize scan sequences'''

    def __init__(self, parameters: ScanInitializerParameters) -> None:
        super().__init__()
        self._parameters = parameters

    @abstractproperty
    def nameHint(self) -> str:
        '''returns a name hint'''
        pass

    @abstractproperty
    def category(self) -> str:
        '''returns a unique category for organizing scan positions'''
        pass

    @abstractproperty
    def variant(self) -> str:
        '''returns a unique variant name'''
        pass

    @abstractproperty
    def canActivate(self) -> bool:
        pass

    @abstractmethod
    def _getPoint(self, index: int) -> ScanPoint:
        '''returns the scan point'''
        pass

    def _getJitteredAndTransformedPoint(self, index: int) -> ScanPoint:
        '''returns the jittered and transformed scan point'''
        point = self._getPoint(index)

        if self._parameters.jitterRadiusInMeters > Decimal():
            rad = Decimal(repr(self._parameters.rng.uniform()))
            dirX = Decimal(repr(self._parameters.rng.normal()))
            dirY = Decimal(repr(self._parameters.rng.normal()))

            scalar = self._parameters.jitterRadiusInMeters * (rad / (dirX**2 + dirY**2)).sqrt()
            point = ScanPoint(point.x + scalar * dirX, point.y + scalar * dirY)

        point = self._parameters.transform(point)

        return ScanPoint(
            self._parameters.centroid.x + point.x,
            self._parameters.centroid.y + point.y,
        )

    def syncToSettings(self, settings: ScanSettings) -> None:
        '''synchronizes parameters to settings'''
        self._parameters.syncToSettings(settings)
        settings.initializer.value = self.variant
        settings.activeScan.value = self.nameHint

    def getTransformNameList(self) -> list[str]:
        return [transform.displayName for transform in ScanPointTransform]

    def getTransformName(self) -> str:
        return self._parameters.transform.displayName

    def setTransformByName(self, name: str) -> None:
        nameLower = name.casefold()

        for transform in ScanPointTransform:
            if nameLower == transform.displayName.casefold():
                self.setTransform(transform)
                return

        logger.error(f'Unknown scan point transform \"{name}\"!')

    def getTransform(self) -> ScanPointTransform:
        '''gets the scan point transform'''
        return self._parameters.transform

    def setTransform(self, transform: ScanPointTransform) -> None:
        '''sets the scan point transform'''
        if self._parameters.transform != transform:
            self._parameters.transform = transform
            self.notifyObservers()

    def getJitterRadiusInMeters(self) -> Decimal:
        '''gets the jitter radius'''
        return self._parameters.jitterRadiusInMeters

    def setJitterRadiusInMeters(self, jitterRadiusInMeters: Decimal) -> None:
        '''sets the jitter radius'''
        if self._parameters.jitterRadiusInMeters != jitterRadiusInMeters:
            self._parameters.jitterRadiusInMeters = jitterRadiusInMeters
            self.notifyObservers()

    def getCentroidXInMeters(self) -> Decimal:
        '''gets the x centroid'''
        return self._parameters.centroid.x

    def setCentroidXInMeters(self, value: Decimal) -> None:
        '''sets the x centroid'''
        if self._parameters.centroid.x != value:
            self._parameters.centroid = ScanPoint(value, self._parameters.centroid.y)
            self.notifyObservers()

    def getCentroidYInMeters(self) -> Decimal:
        '''gets the y centroid'''
        return self._parameters.centroid.y

    def setCentroidYInMeters(self, value: Decimal) -> None:
        '''sets the y centroid'''
        if self._parameters.centroid.y != value:
            self._parameters.centroid = ScanPoint(self._parameters.centroid.x, value)
            self.notifyObservers()

    @overload
    def __getitem__(self, index: int) -> ScanPoint:
        ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[ScanPoint]:
        ...

    def __getitem__(self, index: Union[int, slice]) -> Union[ScanPoint, Sequence[ScanPoint]]:
        if isinstance(index, slice):
            return [
                self._getJitteredAndTransformedPoint(idx)
                for idx in range(index.start, index.stop, index.step)
            ]
        else:
            return self._getJitteredAndTransformedPoint(index)