from __future__ import annotations
from pathlib import Path

from ...api.observer import Observable, Observer
from ...api.settings import SettingsRegistry, SettingsGroup


class ProbeSettings(Observable, Observer):

    def __init__(self, settingsGroup: SettingsGroup) -> None:
        super().__init__()
        self._settingsGroup = settingsGroup
        self.initializer = settingsGroup.createStringEntry('Initializer', 'SuperGaussian')
        self.inputFileType = settingsGroup.createStringEntry('InputFileType', 'NPY')
        self.inputFilePath = settingsGroup.createPathEntry('InputFilePath',
                                                           Path('/path/to/probe.npy'))
        self.automaticProbeSizeEnabled = settingsGroup.createBooleanEntry(
            'AutomaticProbeSizeEnabled', True)
        self.probeSize = settingsGroup.createIntegerEntry('ProbeSize', 64)
        self.probeEnergyInElectronVolts = settingsGroup.createRealEntry(
            'ProbeEnergyInElectronVolts', '10000')
        self.numberOfProbeModes = settingsGroup.createIntegerEntry('NumberOfProbeModes', 1)

        self.sgAnnularRadiusInMeters = settingsGroup.createRealEntry(
            'SuperGaussianAnnularRadiusInMeters', '0')
        self.sgProbeWidthInMeters = settingsGroup.createRealEntry(
            'SuperGaussianProbeWidthInMeters', '400e-6')
        self.sgOrderParameter = settingsGroup.createRealEntry('SuperGaussianOrderParameter', '1')

        self.zonePlateRadiusInMeters = settingsGroup.createRealEntry('ZonePlateRadiusInMeters',
                                                                     '90e-6')
        self.outermostZoneWidthInMeters = settingsGroup.createRealEntry(
            'OutermostZoneWidthInMeters', '50e-9')
        self.centralBeamstopDiameterInMeters = settingsGroup.createRealEntry(
            'CentralBeamstopDiameterInMeters', '60e-6')
        self.defocusDistanceInMeters = settingsGroup.createRealEntry('DefocusDistanceInMeters',
                                                                     '800e-6')

    @classmethod
    def createInstance(cls, settingsRegistry: SettingsRegistry) -> ProbeSettings:
        settings = cls(settingsRegistry.createGroup('Probe'))
        settings._settingsGroup.addObserver(settings)
        return settings

    def update(self, observable: Observable) -> None:
        if observable is self._settingsGroup:
            self.notifyObservers()