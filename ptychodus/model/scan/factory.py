from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Optional, TypeAlias
import logging

import numpy

from ...api.plugins import PluginChooser
from ...api.scan import Scan, ScanFileReader
from .cartesian import CartesianScanInitializer
from .concentric import ConcentricScanInitializer
from .file import FromFileScanInitializer
from .lissajous import LissajousScanInitializer
from .repository import ScanRepositoryItem
from .settings import ScanSettings
from .spiral import SpiralScanInitializer

logger = logging.getLogger(__name__)

InitializerFactory: TypeAlias = Callable[[], Optional[ScanRepositoryItem]]


class ScanRepositoryItemFactory:

    def __init__(self, rng: numpy.random.Generator, settings: ScanSettings,
                 fileReaderChooser: PluginChooser[ScanFileReader]) -> None:
        self._rng = rng
        self._settings = settings
        self._fileReaderChooser = fileReaderChooser
        self._initializersBySimpleName: Mapping[str, InitializerFactory] = {
            FromFileScanInitializer.SIMPLE_NAME: self.createItemFromFile,
            'Raster': self.createRasterItem,
            'Snake': self.createSnakeItem,
            'CenteredRaster': self.createCenteredRasterItem,
            'CenteredSnake': self.createCenteredSnakeItem,
            ConcentricScanInitializer.SIMPLE_NAME: self.createConcentricItem,
            SpiralScanInitializer.SIMPLE_NAME: self.createSpiralItem,
            LissajousScanInitializer.SIMPLE_NAME: self.createLissajousItem,
        }
        self._initializersByDisplayName: Mapping[str, InitializerFactory] = {
            FromFileScanInitializer.DISPLAY_NAME: self.createItemFromFile,
            'Raster': self.createRasterItem,
            'Snake': self.createSnakeItem,
            'Centered Raster': self.createCenteredRasterItem,
            'Centered Snake': self.createCenteredSnakeItem,
            ConcentricScanInitializer.DISPLAY_NAME: self.createConcentricItem,
            SpiralScanInitializer.DISPLAY_NAME: self.createSpiralItem,
            LissajousScanInitializer.DISPLAY_NAME: self.createLissajousItem,
        }

    def getOpenFileFilterList(self) -> Sequence[str]:
        return self._fileReaderChooser.getDisplayNameList()

    def getOpenFileFilter(self) -> str:
        return self._fileReaderChooser.getCurrentDisplayName()

    def createFileInitializer(self,
                              filePath: Path,
                              *,
                              simpleFileType: str = '',
                              displayFileType: str = '') -> Optional[FromFileScanInitializer]:
        if simpleFileType:
            self._fileReaderChooser.setFromSimpleName(simpleFileType)
        elif displayFileType:
            self._fileReaderChooser.setFromDisplayName(displayFileType)
        else:
            logger.error('Refusing to create file initializer without file type!')
            return None

        fileReader = self._fileReaderChooser.getCurrentStrategy()
        return FromFileScanInitializer(filePath, fileReader)

    def openItemFromFile(self,
                         filePath: Path,
                         *,
                         simpleFileType: str = '',
                         displayFileType: str = '') -> Optional[ScanRepositoryItem]:
        item: Optional[ScanRepositoryItem] = None

        if filePath.is_file():
            initializer = self.createFileInitializer(filePath,
                                                     simpleFileType=simpleFileType,
                                                     displayFileType=displayFileType)

            if initializer is None:
                logger.error('Refusing to create item without initializer!')
            else:
                item = ScanRepositoryItem(self._rng, filePath.stem)
                item.setInitializer(initializer)
        else:
            logger.debug(f'Refusing to create item with invalid file path \"{filePath}\"')

        return item

    def createItemFromScan(self,
                           nameHint: str,
                           scan: Scan,
                           *,
                           filePath: Optional[Path] = None,
                           simpleFileType: str = '',
                           displayFileType: str = '') -> ScanRepositoryItem:
        item = ScanRepositoryItem(self._rng, nameHint, scan)

        if filePath is not None:
            if filePath.is_file():
                initializer = self.createFileInitializer(filePath,
                                                         simpleFileType=simpleFileType,
                                                         displayFileType=displayFileType)

                if initializer is None:
                    logger.error('Refusing to add null initializer!')
                else:
                    item.setInitializer(initializer)
            else:
                logger.debug(f'Refusing to add initializer with invalid file path \"{filePath}\"')

        return item

    def createItemFromFile(self) -> Optional[ScanRepositoryItem]:
        filePath = self._settings.inputFilePath.value
        fileType = self._settings.inputFileType.value
        return self.openItemFromFile(filePath, simpleFileType=fileType)

    def createRasterItem(self) -> ScanRepositoryItem:
        initializer = CartesianScanInitializer(snake=False, centered=False)
        item = ScanRepositoryItem(self._rng, initializer.simpleName)
        item.setInitializer(initializer)
        return item

    def createSnakeItem(self) -> ScanRepositoryItem:
        initializer = CartesianScanInitializer(snake=True, centered=False)
        item = ScanRepositoryItem(self._rng, initializer.simpleName)
        item.setInitializer(initializer)
        return item

    def createCenteredRasterItem(self) -> ScanRepositoryItem:
        initializer = CartesianScanInitializer(snake=False, centered=True)
        item = ScanRepositoryItem(self._rng, initializer.simpleName)
        item.setInitializer(initializer)
        return item

    def createCenteredSnakeItem(self) -> ScanRepositoryItem:
        initializer = CartesianScanInitializer(snake=True, centered=True)
        item = ScanRepositoryItem(self._rng, initializer.simpleName)
        item.setInitializer(initializer)
        return item

    def createConcentricItem(self) -> ScanRepositoryItem:
        initializer = ConcentricScanInitializer()
        item = ScanRepositoryItem(self._rng, initializer.simpleName)
        item.setInitializer(initializer)
        return item

    def createSpiralItem(self) -> ScanRepositoryItem:
        initializer = SpiralScanInitializer()
        item = ScanRepositoryItem(self._rng, initializer.simpleName)
        item.setInitializer(initializer)
        return item

    def createLissajousItem(self) -> ScanRepositoryItem:
        initializer = LissajousScanInitializer()
        item = ScanRepositoryItem(self._rng, initializer.simpleName)
        item.setInitializer(initializer)
        return item

    def getInitializerDisplayNameList(self) -> Sequence[str]:
        return [initializerName for initializerName in self._initializersByDisplayName]

    def createItemFromSimpleName(self, name: str) -> Optional[ScanRepositoryItem]:
        item: Optional[ScanRepositoryItem] = None

        try:
            itemFactory = self._initializersBySimpleName[name]
        except KeyError:
            logger.error(f'Unknown scan initializer \"{name}\"!')
        else:
            item = itemFactory()

        return item

    def createItemFromDisplayName(self, name: str) -> Optional[ScanRepositoryItem]:
        item: Optional[ScanRepositoryItem] = None

        try:
            itemFactory = self._initializersByDisplayName[name]
        except KeyError:
            logger.error(f'Unknown scan initializer \"{name}\"!')
        else:
            item = itemFactory()

        return item