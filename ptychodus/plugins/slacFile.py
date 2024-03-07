from pathlib import Path
from typing import Final
import logging

import numpy

from ptychodus.api.data import (DiffractionDataset, DiffractionFileReader, DiffractionMetadata,
                                DiffractionPatternState, SimpleDiffractionDataset,
                                SimpleDiffractionPatternArray)
from ptychodus.api.image import ImageExtent
from ptychodus.api.object import Object, ObjectFileReader
from ptychodus.api.plugins import PluginRegistry
from ptychodus.api.probe import Probe, ProbeFileReader
from ptychodus.api.scan import Scan, ScanFileReader, ScanPoint, TabularScan
from ptychodus.api.tree import SimpleTreeNode

logger = logging.getLogger(__name__)


class SLACDiffractionFileReader(DiffractionFileReader):

    def read(self, filePath: Path) -> DiffractionDataset:

        try:
            npz = numpy.load(filePath)
        except OSError:
            logger.warning(f'Unable to read file \"{filePath}\".')
            return SimpleDiffractionDataset.createNullInstance(filePath)

        try:
            patterns = npz['diffraction']
        except KeyError:
            logger.warning(f'No diffraction patterns in \"{filePath}\"!')
            return SimpleDiffractionDataset.createNullInstance(filePath)

        numberOfPatterns, detectorHeight, detectorWidth = patterns.shape

        metadata = DiffractionMetadata(
            numberOfPatternsPerArray=numberOfPatterns,
            numberOfPatternsTotal=numberOfPatterns,
            patternDataType=patterns.dtype,
            detectorExtentInPixels=ImageExtent(detectorWidth, detectorHeight),
            filePath=filePath,
        )

        contentsTree = SimpleTreeNode.createRoot(['Name', 'Type', 'Details'])
        contentsTree.createChild(
            [filePath.stem,
             type(patterns).__name__, f'{patterns.dtype}{patterns.shape}'])

        array = SimpleDiffractionPatternArray(
            label=filePath.stem,
            index=0,
            data=patterns,
            state=DiffractionPatternState.FOUND,
        )

        return SimpleDiffractionDataset(metadata, contentsTree, [array])


class SLACScanFileReader(ScanFileReader):

    def read(self, filePath: Path) -> Scan:
        pointMap: dict[int, ScanPoint] = dict()

        try:
            npz = numpy.load(filePath)
        except OSError:
            logger.warning(f'Unable to read file \"{filePath}\".')
            return TabularScan(pointMap)

        try:
            positionXInMeters = npz['xcoords']
            positionYInMeters = npz['ycoords']
        except KeyError:
            logger.warning(f'No scan positions in \"{filePath}\"!')
            return TabularScan(pointMap)

        for index, (x, y) in enumerate(zip(positionXInMeters, positionYInMeters)):
            pointMap[index] = ScanPoint(x, y)

        return TabularScan(pointMap)


class SLACProbeFileReader(ProbeFileReader):

    def read(self, filePath: Path) -> Probe:
        try:
            npz = numpy.load(filePath)
        except OSError:
            logger.warning(f'Unable to read file \"{filePath}\".')
            return Probe()

        try:
            array = npz['probeGuess']
        except KeyError:
            logger.warning(f'No probe guess in \"{filePath}\"!')
            return Probe()

        return Probe(array)


class SLACObjectFileReader(ObjectFileReader):

    def read(self, filePath: Path) -> Object:
        try:
            npz = numpy.load(filePath)
        except OSError:
            logger.warning(f'Unable to read file \"{filePath}\".')
            return Object()

        try:
            array = npz['objectGuess']
        except KeyError:
            logger.warning(f'No object guess in \"{filePath}\"!')
            return Object()

        return Object(array)


def registerPlugins(registry: PluginRegistry) -> None:
    SIMPLE_NAME: Final[str] = 'SLAC'
    DISPLAY_NAME: Final[str] = 'SLAC NumPy Zipped Archive (*.npz)'

    registry.diffractionFileReaders.registerPlugin(
        SLACDiffractionFileReader(),
        simpleName=SIMPLE_NAME,
        displayName=DISPLAY_NAME,
    )
    registry.scanFileReaders.registerPlugin(
        SLACScanFileReader(),
        simpleName=SIMPLE_NAME,
        displayName=DISPLAY_NAME,
    )
    registry.probeFileReaders.registerPlugin(
        SLACProbeFileReader(),
        simpleName=SIMPLE_NAME,
        displayName=DISPLAY_NAME,
    )
    registry.objectFileReaders.registerPlugin(
        SLACObjectFileReader(),
        simpleName=SIMPLE_NAME,
        displayName=DISPLAY_NAME,
    )
