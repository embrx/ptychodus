from pathlib import Path
from typing import Final
import logging

import h5py
import numpy

from ptychodus.api.data import (DiffractionDataset, DiffractionFileReader, DiffractionMetadata,
                                SimpleDiffractionDataset)
from ptychodus.api.image import ImageExtent
from ptychodus.api.plugins import PluginRegistry
from ptychodus.api.scan import Scan, ScanFileReader, ScanPoint, TabularScan

from .h5DiffractionFile import H5DiffractionPatternArray, H5DiffractionFileTreeBuilder

logger = logging.getLogger(__name__)


class SLAC_H5DiffractionFileReader(DiffractionFileReader):

    def __init__(self) -> None:
        self._dataPath = '/jungfrau1M/ROI_0_area'
        self._treeBuilder = H5DiffractionFileTreeBuilder()

    def read(self, filePath: Path) -> DiffractionDataset:
        dataset = SimpleDiffractionDataset.createNullInstance(filePath)

        try:
            with h5py.File(filePath, 'r') as h5File:
                metadata = DiffractionMetadata.createNullInstance(filePath)
                contentsTree = self._treeBuilder.build(h5File)

                try:
                    data = h5File[self._dataPath]
                except KeyError:
                    logger.debug('Unable to find data.')
                else:
                    numberOfPatterns, detectorHeight, detectorWidth = data.shape

                    metadata = DiffractionMetadata(
                        numberOfPatternsPerArray=numberOfPatterns,
                        numberOfPatternsTotal=numberOfPatterns,
                        patternDataType=data.dtype,
                        detectorExtentInPixels=ImageExtent(detectorWidth, detectorHeight),
                        filePath=filePath,
                    )

                array = H5DiffractionPatternArray(
                    label=filePath.stem,
                    index=0,
                    filePath=filePath,
                    dataPath=self._dataPath,
                )

                dataset = SimpleDiffractionDataset(metadata, contentsTree, [array])
        except OSError:
            logger.debug(f'Unable to read file \"{filePath}\".')

        return dataset


class SLAC_H5ScanFileReader(ScanFileReader):
    MICRONS_TO_METERS: Final[float] = 1e-6

    def __init__(self, angleInRadians: float = 0.) -> None:
        self._angleInRadians = angleInRadians

    def read(self, filePath: Path) -> Scan:
        pointMap: dict[int, ScanPoint] = dict()

        with h5py.File(filePath, 'r') as h5File:
            try:
                # piezo stage positions are in microns
                pi_x = h5File['/lmc/ch03'][()]
                pi_y = h5File['/lmc/ch04'][()]
                pi_z = h5File['/lmc/ch05'][()]
            except KeyError:
                logger.exception('Unable to load scan.')
            else:
                # vertical coordinate is always pi_z
                ycoords = -pi_z * self.MICRONS_TO_METERS

                # horizontal coordinate may be a combination of pi_x and pi_y
                cosAngle = numpy.cos(self._angleInRadians)
                sinAngle = numpy.sin(self._angleInRadians)
                xcoords = (cosAngle * pi_x + sinAngle * pi_y) * self.MICRONS_TO_METERS

                for index, (x, y) in enumerate(zip(xcoords, ycoords)):
                    if numpy.isfinite(x) and numpy.isfinite(y):
                        pointMap[index] = ScanPoint(x, y)

        return TabularScan(pointMap)


def registerPlugins(registry: PluginRegistry) -> None:
    SIMPLE_NAME: Final[str] = 'SLAC_H5'
    DISPLAY_NAME: Final[str] = 'SLAC Hierarchical Data Format 5 Files (*.h5 *.hdf5)'

    registry.diffractionFileReaders.registerPlugin(
        SLAC_H5DiffractionFileReader(),
        simpleName=SIMPLE_NAME,
        displayName=DISPLAY_NAME,
    )
    registry.scanFileReaders.registerPlugin(
        SLAC_H5ScanFileReader(),
        simpleName=SIMPLE_NAME,
        displayName=DISPLAY_NAME,
    )
