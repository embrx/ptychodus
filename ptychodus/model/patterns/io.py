from __future__ import annotations
from collections.abc import Sequence
from pathlib import Path
import logging

from ptychodus.api.observer import Observable, Observer

from .active import ActiveDiffractionDataset
from .api import PatternsAPI
from .settings import PatternSettings

logger = logging.getLogger(__name__)


class DiffractionDatasetInputOutputPresenter(Observable, Observer):

    def __init__(self, settings: PatternSettings, dataset: ActiveDiffractionDataset,
                 patternsAPI: PatternsAPI, reinitObservable: Observable) -> None:
        super().__init__()
        self._settings = settings
        self._dataset = dataset
        self._patternsAPI = patternsAPI
        self._reinitObservable = reinitObservable

    @classmethod
    def createInstance(cls, settings: PatternSettings, dataset: ActiveDiffractionDataset,
                       patternsAPI: PatternsAPI,
                       reinitObservable: Observable) -> DiffractionDatasetInputOutputPresenter:
        presenter = cls(settings, dataset, patternsAPI, reinitObservable)
        reinitObservable.addObserver(presenter)
        return presenter

    def getOpenFileFilterList(self) -> Sequence[str]:
        return self._patternsAPI.getOpenFileFilterList()

    def getOpenFileFilter(self) -> str:
        return self._patternsAPI.getOpenFileFilter()

    def openDiffractionFile(self, filePath: Path, fileFilter: str) -> None:
        try:
            fileType = self._patternsAPI.openPatterns(filePath=filePath,
                                                      fileType=fileFilter,
                                                      assemble=False)
        except Exception:
            logger.exception('Failed to load diffraction dataset.')
            return

        if fileType is None:
            logger.error('Failed to load diffraction dataset.')
        else:
            self._settings.fileType.value = fileType
            self._settings.filePath.value = filePath

        self.notifyObservers()

    def _openDiffractionFileFromSettings(self) -> None:
        self._patternsAPI.openPatterns(filePath=self._settings.filePath.value,
                                       fileType=self._settings.fileType.value,
                                       assemble=True)

    def startAssemblingDiffractionPatterns(self) -> None:
        self._patternsAPI.startAssemblingDiffractionPatterns()

    def stopAssemblingDiffractionPatterns(self, finishAssembling: bool) -> None:
        self._patternsAPI.stopAssemblingDiffractionPatterns(finishAssembling)

    def getSaveFileFilterList(self) -> Sequence[str]:
        return self._patternsAPI.getSaveFileFilterList()

    def getSaveFileFilter(self) -> str:
        return self._patternsAPI.getSaveFileFilter()

    def saveDiffractionFile(self, filePath: Path, fileFilter: str) -> None:
        self._patternsAPI.savePatterns(filePath, fileFilter)

    def update(self, observable: Observable) -> None:
        if observable is self._reinitObservable:
            self._openDiffractionFileFromSettings()
