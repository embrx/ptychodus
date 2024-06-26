from __future__ import annotations

from ptychodus.api.observer import Observable, Observer
from ptychodus.api.settings import SettingsRegistry, SettingsGroup


class TikeSettings(Observable, Observer):

    def __init__(self, settingsGroup: SettingsGroup) -> None:
        super().__init__()
        self._settingsGroup = settingsGroup
        self.numGpus = settingsGroup.createStringEntry('NumGpus', '1')
        self.noiseModel = settingsGroup.createStringEntry('NoiseModel', 'gaussian')
        self.numBatch = settingsGroup.createIntegerEntry('NumBatch', 10)
        self.batchMethod = settingsGroup.createStringEntry('BatchMethod', 'wobbly_center')
        self.numIter = settingsGroup.createIntegerEntry('NumIter', 1)
        self.convergenceWindow = settingsGroup.createIntegerEntry('ConvergenceWindow', 0)
        self.alpha = settingsGroup.createRealEntry('Alpha', '0.05')
        self.stepLength = settingsGroup.createRealEntry('StepLength', '1')

    @classmethod
    def createInstance(cls, settingsRegistry: SettingsRegistry) -> TikeSettings:
        settings = cls(settingsRegistry.createGroup('Tike'))
        settings._settingsGroup.addObserver(settings)
        return settings

    def update(self, observable: Observable) -> None:
        if observable is self._settingsGroup:
            self.notifyObservers()
