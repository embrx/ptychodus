from __future__ import annotations
from pathlib import Path

from ptychodus.api.observer import Observable, Observer
from ptychodus.api.settings import SettingsRegistry, SettingsGroup

class PositionPredictionSettings(Observable, Observer):

    def __init__(self, settingsGroup: SettingsGroup) -> None:
        super().__init__()
        self._settingsGroup = settingsGroup
        self.stateFilePath = settingsGroup.createPathEntry('ReconstructionImagePath',
                                                           Path('/path/to/best_model.tiff'))
        self.Method = settingsGroup.createStringEntry('Method', 'Serial')
        self.RandomSeed = settingsGroup.createIntegerEntry('RandomSeed', 2)
        self.ProbePositionList = settingsGroup.createPathEntry('ReconstructionImagePath',
                                                           Path('/path/to/best_model.csv'))
        self.CentralCrop = settingsGroup.createIntegerEntry('CentralCrop', 'None')
        self.NumNeighborhoodsCollective = settingsGroup.createIntegerEntry(
            'NumNeighborsCollective', 4)
        self.RegistrationParams = settingsGroup.createPathEntry('RegistrationParams', Path('/path/to/best_model.py'))
        self.RegistrationMethod = settingsGroup.createStringEntry('RegistrationMethod', 'hybrid')

    @classmethod
    def createInstance(cls, settingsRegistry: SettingsRegistry) -> PositionPredictionSettings:
        settingsGroup = settingsRegistry.createGroup('PtychoNN')
        settings = cls(settingsGroup)
        settingsGroup.addObserver(settings)
        return settings

    def update(self, observable: Observable) -> None:
        if observable is self._settingsGroup:
            self.notifyObservers()