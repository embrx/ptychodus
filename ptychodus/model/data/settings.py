from __future__ import annotations
from pathlib import Path

from ...api.observer import Observable, Observer
from ...api.settings import SettingsRegistry, SettingsGroup


class DiffractionDatasetSettings(Observable, Observer):

    def __init__(self, settingsGroup: SettingsGroup) -> None:
        super().__init__()
        self._settingsGroup = settingsGroup
        self.fileType = settingsGroup.createStringEntry('FileType', 'HDF5')
        self.filePath = settingsGroup.createPathEntry('FilePath', Path('/path/to/data.h5'))
        self.scratchDirectory = settingsGroup.createPathEntry('ScratchDirectory',
                                                              Path('/dev/null'))
        self.watchForFiles = settingsGroup.createBooleanEntry('WatchForFiles', False)

    @classmethod
    def createInstance(cls, settingsRegistry: SettingsRegistry) -> DiffractionDatasetSettings:
        settings = cls(settingsRegistry.createGroup('Diffraction Dataset'))
        settings._settingsGroup.addObserver(settings)
        return settings

    def update(self, observable: Observable) -> None:
        if observable is self._settingsGroup:
            self.notifyObservers()


class DiffractionPatternSettings(Observable, Observer):

    def __init__(self, settingsGroup: SettingsGroup) -> None:
        super().__init__()
        self._settingsGroup = settingsGroup
        self.numberOfDataThreads = settingsGroup.createIntegerEntry('NumberOfDataThreads', 8)
        self.cropEnabled = settingsGroup.createBooleanEntry('CropEnabled', True)
        self.cropCenterXInPixels = settingsGroup.createIntegerEntry('CropCenterXInPixels', 32)
        self.cropCenterYInPixels = settingsGroup.createIntegerEntry('CropCenterYInPixels', 32)
        self.cropExtentXInPixels = settingsGroup.createIntegerEntry('CropExtentXInPixels', 64)
        self.cropExtentYInPixels = settingsGroup.createIntegerEntry('CropExtentYInPixels', 64)
        self.flipXEnabled = settingsGroup.createBooleanEntry('FlipXEnabled', False)
        self.flipYEnabled = settingsGroup.createBooleanEntry('FlipYEnabled', False)
        self.threshold = settingsGroup.createIntegerEntry('Threshold', 0)

    @classmethod
    def createInstance(cls, settingsRegistry: SettingsRegistry) -> DiffractionPatternSettings:
        settings = cls(settingsRegistry.createGroup('Diffraction Pattern'))
        settings._settingsGroup.addObserver(settings)
        return settings

    def update(self, observable: Observable) -> None:
        if observable is self._settingsGroup:
            self.notifyObservers()