from __future__ import annotations
from pathlib import Path

from ...api.observer import Observable, Observer
from ...api.settings import SettingsRegistry, SettingsGroup


class PtychoPINNModelSettings(Observable, Observer):

    def __init__(self, settingsGroup: SettingsGroup) -> None:
        super().__init__()
        self._settingsGroup = settingsGroup
        self.learningRate = settingsGroup.createRealEntry('LearningRate', '1e-3')
        self.N = settingsGroup.createIntegerEntry('N', 64)
        self.offset = settingsGroup.createIntegerEntry('Offset', 4)
        self.gridsize = settingsGroup.createIntegerEntry('Gridsize', 2)
        self.batchSize = settingsGroup.createIntegerEntry('BatchSize', 16)
        self.nFiltersScale = settingsGroup.createIntegerEntry('NFiltersScale', 2)
        self.nphotons = settingsGroup.createRealEntry('NPhotons', '1e9')
        self.probeTrainable = settingsGroup.createBooleanEntry('ProbeTrainable', False)
        self.intensityScaleTrainable = settingsGroup.createBooleanEntry(
            'IntensityScaleTrainable', False)
        self.objectBig = settingsGroup.createBooleanEntry('ObjectBig', True)
        self.probeBig = settingsGroup.createBooleanEntry('ProbeBig', False)
        self.probeScale = settingsGroup.createRealEntry('ProbeScale', '10.')
        self.probeMask = settingsGroup.createBooleanEntry('ProbeMask', True)
        self.modelType = settingsGroup.createStringEntry('ModelType', 'pinn')
        self.size = settingsGroup.createIntegerEntry('Size', 392)
        self.ampActivation = settingsGroup.createStringEntry('AmpActivation', 'sigmoid')

    @classmethod
    def createInstance(cls, settingsRegistry: SettingsRegistry) -> PtychoPINNModelSettings:
        settingsGroup = settingsRegistry.createGroup('PtychoPINN')
        settings = cls(settingsGroup)
        settingsGroup.addObserver(settings)
        return settings

    def update(self, observable: Observable) -> None:
        if observable is self._settingsGroup:
            self.notifyObservers()


class PtychoPINNTrainingSettings(Observable, Observer):

    def __init__(self, settingsGroup: SettingsGroup) -> None:
        super().__init__()
        self._settingsGroup = settingsGroup
        self.maeWeight = settingsGroup.createRealEntry('MAEWeight', '0.')
        self.nllWeight = settingsGroup.createRealEntry('NLLWeight', '1.')
        self.tvWeight = settingsGroup.createRealEntry('TVWeight', '0.')
        self.realspaceMAEWeight = settingsGroup.createRealEntry('RealspaceMAEWeight', '0.')
        self.realspaceWeight = settingsGroup.createRealEntry('RealspaceWeight', '0.')

        # generic settings shared with ptychonn
        self.maximumTrainingDatasetSize = settingsGroup.createIntegerEntry(
            'MaximumTrainingDatasetSize', 100000)
        self.validationSetFractionalSize = settingsGroup.createRealEntry(
            'ValidationSetFractionalSize', '0.1')
        self.optimizationEpochsPerHalfCycle = settingsGroup.createIntegerEntry(
            'OptimizationEpochsPerHalfCycle', 6)
        self.maximumLearningRate = settingsGroup.createRealEntry('MaximumLearningRate', '1e-3')
        self.minimumLearningRate = settingsGroup.createRealEntry('MinimumLearningRate', '1e-4')
        self.trainingEpochs = settingsGroup.createIntegerEntry('TrainingEpochs', 50)
        self.saveTrainingArtifacts = settingsGroup.createBooleanEntry(
            'SaveTrainingArtifacts', False)
        self.outputPath = settingsGroup.createPathEntry('OutputPath', Path('/path/to/output'))
        self.outputSuffix = settingsGroup.createStringEntry('OutputSuffix', 'suffix')

    @classmethod
    def createInstance(cls, settingsRegistry: SettingsRegistry) -> PtychoPINNTrainingSettings:
        settingsGroup = settingsRegistry.createGroup('PtychoPINNTraining')
        settings = cls(settingsGroup)
        settingsGroup.addObserver(settings)
        return settings

    def update(self, observable: Observable) -> None:
        if observable is self._settingsGroup:
            self.notifyObservers()
