from __future__ import annotations
from collections.abc import Iterator, Sequence
from decimal import Decimal
from typing import Final
import logging

from ptychodus.api.geometry import Interval
from ptychodus.api.observer import Observable, Observer
from ptychodus.api.reconstructor import NullReconstructor, Reconstructor, ReconstructorLibrary
from ptychodus.api.settings import SettingsRegistry

from .multigrid import TikeMultigridPresenter, TikeMultigridSettings
from .objectCorrection import TikeObjectCorrectionPresenter, TikeObjectCorrectionSettings
from .positionCorrection import TikePositionCorrectionPresenter, TikePositionCorrectionSettings
from .probeCorrection import TikeProbeCorrectionPresenter, TikeProbeCorrectionSettings
from .settings import TikeSettings

logger = logging.getLogger(__name__)


class TikePresenter(Observable, Observer):
    MAX_INT: Final[int] = 0x7FFFFFFF

    def __init__(self, settings: TikeSettings) -> None:
        super().__init__()
        self._settings = settings
        self._logger = logging.getLogger('tike')

    @classmethod
    def createInstance(cls, settings: TikeSettings) -> TikePresenter:
        presenter = cls(settings)
        settings.addObserver(presenter)
        return presenter

    def getNumGpus(self) -> str:
        return self._settings.numGpus.value

    def setNumGpus(self, value: str) -> None:
        self._settings.numGpus.value = value

    def getNoiseModelList(self) -> Sequence[str]:
        return ['poisson', 'gaussian']

    def getNoiseModel(self) -> str:
        return self._settings.noiseModel.value

    def setNoiseModel(self, name: str) -> None:
        self._settings.noiseModel.value = name

    def getNumBatchLimits(self) -> Interval[int]:
        return Interval[int](1, self.MAX_INT)

    def getNumBatch(self) -> int:
        limits = self.getNumBatchLimits()
        return limits.clamp(self._settings.numBatch.value)

    def setNumBatch(self, value: int) -> None:
        self._settings.numBatch.value = value

    def getBatchMethodList(self) -> Sequence[str]:
        return ['wobbly_center', 'wobbly_center_random_bootstrap', 'compact']

    def getBatchMethod(self) -> str:
        return self._settings.batchMethod.value

    def setBatchMethod(self, name: str) -> None:
        self._settings.batchMethod.value = name

    def getNumIterLimits(self) -> Interval[int]:
        return Interval[int](1, self.MAX_INT)

    def getNumIter(self) -> int:
        limits = self.getNumIterLimits()
        return limits.clamp(self._settings.numIter.value)

    def setNumIter(self, value: int) -> None:
        self._settings.numIter.value = value

    def getConvergenceWindowLimits(self) -> Interval[int]:
        return Interval[int](0, self.MAX_INT)

    def getConvergenceWindow(self) -> int:
        limits = self.getConvergenceWindowLimits()
        return limits.clamp(self._settings.convergenceWindow.value)

    def setConvergenceWindow(self, value: int) -> None:
        self._settings.convergenceWindow.value = value

    def getAlphaLimits(self) -> Interval[Decimal]:
        return Interval[Decimal](Decimal(0), Decimal(1))

    def getAlpha(self) -> Decimal:
        limits = self.getAlphaLimits()
        return limits.clamp(self._settings.alpha.value)

    def setAlpha(self, value: Decimal) -> None:
        self._settings.alpha.value = value

    def getStepLengthLimits(self) -> Interval[Decimal]:
        return Interval[Decimal](Decimal(0), Decimal(1))

    def getStepLength(self) -> Decimal:
        limits = self.getStepLengthLimits()
        return limits.clamp(self._settings.stepLength.value)

    def setStepLength(self, value: Decimal) -> None:
        self._settings.stepLength.value = value

    def getLogLevelList(self) -> Sequence[str]:
        return ['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG']

    def getLogLevel(self) -> str:
        level = self._logger.getEffectiveLevel()
        return logging.getLevelName(level)

    def setLogLevel(self, name: str) -> None:
        nameBefore = self.getLogLevel()

        if name == nameBefore:
            return

        try:
            self._logger.setLevel(name)
        except ValueError:
            logger.error(f'Bad log level \"{name}\".')

        nameAfter = self.getLogLevel()
        logger.info(f'Changed Tike log level {nameBefore} -> {nameAfter}')
        self.notifyObservers()

    def update(self, observable: Observable) -> None:
        if observable is self._settings:
            self.notifyObservers()


class TikeReconstructorLibrary(ReconstructorLibrary):

    def __init__(self, settingsRegistry: SettingsRegistry) -> None:
        super().__init__()
        self._settings = TikeSettings.createInstance(settingsRegistry)
        self._multigridSettings = TikeMultigridSettings.createInstance(settingsRegistry)
        self._positionCorrectionSettings = TikePositionCorrectionSettings.createInstance(
            settingsRegistry)
        self._probeCorrectionSettings = TikeProbeCorrectionSettings.createInstance(
            settingsRegistry)
        self._objectCorrectionSettings = TikeObjectCorrectionSettings.createInstance(
            settingsRegistry)

        self.presenter = TikePresenter.createInstance(self._settings)
        self.multigridPresenter = TikeMultigridPresenter.createInstance(self._multigridSettings)
        self.positionCorrectionPresenter = TikePositionCorrectionPresenter.createInstance(
            self._positionCorrectionSettings)
        self.probeCorrectionPresenter = TikeProbeCorrectionPresenter.createInstance(
            self._probeCorrectionSettings)
        self.objectCorrectionPresenter = TikeObjectCorrectionPresenter.createInstance(
            self._objectCorrectionSettings)

        self.reconstructorList: list[Reconstructor] = list()

    @classmethod
    def createInstance(cls, settingsRegistry: SettingsRegistry,
                       isDeveloperModeEnabled: bool) -> TikeReconstructorLibrary:
        core = cls(settingsRegistry)

        try:
            from .reconstructor import DifferenceMapReconstructor
            from .reconstructor import IterativeLeastSquaresReconstructor
            from .reconstructor import RegularizedPIEReconstructor
            from .reconstructor import TikeReconstructor
        except ModuleNotFoundError:
            logger.info('Tike not found.')

            if isDeveloperModeEnabled:
                core.reconstructorList.append(NullReconstructor('rpie'))
                core.reconstructorList.append(NullReconstructor('lstsq_grad'))
                core.reconstructorList.append(NullReconstructor('dm'))
        else:
            tikeReconstructor = TikeReconstructor(core._settings, core._multigridSettings,
                                                  core._positionCorrectionSettings,
                                                  core._probeCorrectionSettings,
                                                  core._objectCorrectionSettings)
            core.reconstructorList.append(RegularizedPIEReconstructor(tikeReconstructor))
            core.reconstructorList.append(IterativeLeastSquaresReconstructor(tikeReconstructor))
            core.reconstructorList.append(DifferenceMapReconstructor(tikeReconstructor))

        return core

    @property
    def name(self) -> str:
        return 'Tike'

    def __iter__(self) -> Iterator[Reconstructor]:
        return iter(self.reconstructorList)
