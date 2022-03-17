from __future__ import annotations
import logging

import numpy

from .crop import *
from .data_file import *
from .detector import *
from .h5tree import *
from .image import *
from .object import *
from .observer import *
from .probe import *
from .ptychonn import PtychoNNBackend
from .ptychopy import PtychoPyBackend
from .reconstructor import *
from .scan import *
from .settings import *
from .tike import TikeBackend
from .velociprobe import *

logger = logging.getLogger(__name__)


class ModelCore:
    def __init__(self, isDeveloperModeEnabled: bool) -> None:
        self.rng = numpy.random.default_rng()

        self.settingsRegistry = SettingsRegistry()
        self._detectorSettings = DetectorSettings.createInstance(self.settingsRegistry)
        self._cropSettings = CropSettings.createInstance(self.settingsRegistry)
        self._scanSettings = ScanSettings.createInstance(self.settingsRegistry)
        self._probeSettings = ProbeSettings.createInstance(self.settingsRegistry)
        self._objectSettings = ObjectSettings.createInstance(self.settingsRegistry)
        self._reconstructorSettings = ReconstructorSettings.createInstance(self.settingsRegistry)

        self._selectableScanSequence = SelectableScanSequence.createInstance(self._scanSettings)
        self._scanSequence = TransformedScanSequence.createInstance(self._scanSettings,
                                                                    self._selectableScanSequence)
        self._detector = Detector.createInstance(self._detectorSettings)
        self._cropSizer = CropSizer.createInstance(self._cropSettings, self._detector)
        self._probeSizer = ProbeSizer.createInstance(self._probeSettings, self._cropSizer)
        self._objectSizer = ObjectSizer.createInstance(self._scanSequence, self._detector,
                                                       self._cropSizer, self._probeSizer)

        self._probe = Probe.createInstance(self._probeSettings)
        self._object = Object.createInstance(self._objectSettings)

        self.h5FileTreeReader = H5FileTreeReader()
        self._velociprobeReader = VelociprobeReader()
        self._velociprobeImageSequence = VelociprobeImageSequence.createInstance(
            self._velociprobeReader)
        self._croppedImageSequence = CroppedImageSequence.createInstance(
            self._cropSizer, self._velociprobeImageSequence)
        self._dataDirectoryWatcher = DataDirectoryWatcher()

        self.ptychopyBackend = PtychoPyBackend.createInstance(self.settingsRegistry,
                                                              isDeveloperModeEnabled)
        self.tikeBackend = TikeBackend.createInstance(self.settingsRegistry, self._cropSizer,
                                                      self._velociprobeReader, self._scanSequence,
                                                      self._probeSizer, self._probe,
                                                      self._objectSizer, self._object,
                                                      isDeveloperModeEnabled)
        self.ptychonnBackend = PtychoNNBackend.createInstance(self.settingsRegistry,
                                                              isDeveloperModeEnabled)
        self._selectableReconstructor = SelectableReconstructor.createInstance(
            self._reconstructorSettings, self.ptychopyBackend.reconstructorList +
            self.tikeBackend.reconstructorList + self.ptychonnBackend.reconstructorList)

        self.dataFilePresenter = DataFilePresenter()
        self.settingsPresenter = SettingsPresenter.createInstance(self.settingsRegistry)
        self.detectorPresenter = DetectorPresenter.createInstance(self._detectorSettings)
        self.cropPresenter = CropPresenter.createInstance(self._cropSettings, self._cropSizer)
        self.detectorImagePresenter = DetectorImagePresenter.createInstance(
            self._croppedImageSequence)
        self.velociprobePresenter = VelociprobePresenter.createInstance(
            self._velociprobeReader, self._detectorSettings, self._cropSettings,
            self._probeSettings)
        self.probePresenter = ProbePresenter.createInstance(self._detectorSettings,
                                                            self._probeSettings, self._probeSizer,
                                                            self._probe)
        self.scanPresenter = ScanPresenter.createInstance(self._scanSettings,
                                                          self._selectableScanSequence,
                                                          self._scanSequence)
        self.objectPresenter = ObjectPresenter.createInstance(self.rng, self._objectSettings,
                                                              self._objectSizer, self._object)
        self.reconstructorPresenter = ReconstructorPresenter.createInstance(
            self._reconstructorSettings, self._selectableReconstructor)

    @classmethod
    def createInstance(cls, isDeveloperModeEnabled: bool = False) -> ModelCore:
        model = cls(isDeveloperModeEnabled)
        model.dataFilePresenter.addReader(model.h5FileTreeReader)
        model.dataFilePresenter.addReader(model._velociprobeReader)
        return model

    def start(self) -> None:
        self._dataDirectoryWatcher.start()

    def stop(self) -> None:
        self._dataDirectoryWatcher.stop()
        self._dataDirectoryWatcher.join()
