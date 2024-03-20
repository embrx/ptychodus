from pathlib import Path
from ptychodus.api.image import ImageExtent
from ptychodus.api.object import ObjectFileReader, ObjectPhaseCenteringStrategy
from ptychodus.api.object import ObjectPhaseCenteringStrategy, ObjectFileReader, ObjectFileWriter
from ptychodus.api.observer import Observable
from ptychodus.api.plot import Plot2D
from ptychodus.api.plugins import PluginChooser
from ptychodus.api.reconstructor import ReconstructInput, ReconstructOutput
from ptychodus.api.reconstructor import ReconstructInput, ReconstructOutput, Plot2D
from ptychodus.api.scan import ScanPoint, Scan
from ptychodus.api.scan import TabularScan
from ptychodus.api.settings import SettingsRegistry, SettingsGroup
from ptychodus.model.data.settings import DiffractionPatternSettings
from ptychodus.model.data.sizer import DiffractionPatternSizer
from ptychodus.model.detector import Detector, DetectorSettings
from ptychodus.model.itemRepository import SelectedRepositoryItem
from ptychodus.model.itemRepository import SelectedRepositoryItem, RepositoryItemSettingsDelegate
from ptychodus.model.itemRepository import SelectedRepositoryItem, RepositoryItemSettingsDelegate, ItemRepository, Observable
from ptychodus.model.object.api import ObjectAPI
from ptychodus.model.object.core import ObjectCore
from ptychodus.model.object.factory import ObjectRepositoryItemFactory
from ptychodus.model.object.interpolator import ObjectInterpolatorFactory
from ptychodus.model.object.repository import ObjectRepository, ObjectRepositoryItem  # Add ObjectRepositoryItem import
from ptychodus.model.object.selected import SelectedObject, ObjectRepositoryItemSettingsDelegate
from ptychodus.model.object.settings import ObjectSettings
from ptychodus.model.object.sizer import ObjectSizer
from ptychodus.model.probe import Apparatus, ProbeSizer, ProbeSettings
from ptychodus.model.probe import ProbeSettings
from ptychodus.model.ptychopinn.reconstructor import PtychoPINNTrainableReconstructor
from ptychodus.model.ptychopinn.reconstructor import PtychoPINNTrainableReconstructor, create_ptycho_data_container
from ptychodus.model.ptychopinn.settings import PtychoPINNModelSettings, PtychoPINNTrainingSettings
from ptychodus.model.scan import ScanSizer
from ptychodus.model.scan.settings import ScanSettings
from ptychodus.plugins.slacFile import SLACDiffractionFileReader, SLACScanFileReader, SLACProbeFileReader, SLACObjectFileReader

import numpy as np

## Define ScanRepositoryItem class
#class ScanRepositoryItem(object):
#    def __init__(self, scan: Scan):
#        self._scan = scan
#
#    @property
#    def nameHint(self) -> str:
#        return 'Scan'
#
#    def getScan(self) -> Scan:
#        return self._scan

# Create instances of the file readers
diffraction_reader = SLACDiffractionFileReader()
scan_reader = SLACScanFileReader()
probe_reader = SLACProbeFileReader()
object_reader = SLACObjectFileReader()

# Assume the file paths are stored in these variables
diffraction_file_path = Path('path/to/diffraction_file.npz')
scan_file_path = Path('path/to/scan_file.npz')
probe_file_path = Path('path/to/probe_file.npz')
object_file_path = Path('path/to/object_file.npz')

# Read the data
diffraction_dataset = diffraction_reader.read(diffraction_file_path)
scan = scan_reader.read(scan_file_path)
probe = probe_reader.read(probe_file_path)
object_ = object_reader.read(object_file_path)

# Create dummy data and objects for initializing PtychoPINNTrainableReconstructor
rng = np.random.default_rng(42)  # Random number generator with seed

settings_registry = SettingsRegistry('ptychodus')
ptychopinn_model_settings = PtychoPINNModelSettings.createInstance(settings_registry)
ptychopinn_training_settings = PtychoPINNTrainingSettings.createInstance(settings_registry)

detector_settings = DetectorSettings.createInstance(settings_registry)
detector = Detector(detector_settings)
diffraction_pattern_settings = DiffractionPatternSettings.createInstance(settings_registry)
diffraction_pattern_sizer = DiffractionPatternSizer.createInstance(diffraction_pattern_settings, detector)
probe_settings_group = SettingsGroup('probesettings')
probe_settings = ProbeSettings(probe_settings_group)
apparatus = Apparatus(detector, diffraction_pattern_sizer, probe_settings)
probe_sizer = ProbeSizer(diffraction_pattern_sizer)
scan_settings = ScanSettings.createInstance(settings_registry)
scan_sizer = ScanSizer(scan_settings, None)

phase_centering_strategy_chooser = PluginChooser[ObjectPhaseCenteringStrategy]()
file_reader_chooser = PluginChooser[ObjectFileReader]()
file_writer_chooser = PluginChooser[ObjectFileWriter]()

object_core = ObjectCore(
    rng, settings_registry, apparatus, scan_sizer, probe_sizer,
    phase_centering_strategy_chooser, file_reader_chooser, file_writer_chooser
)

object_api = object_core.objectAPI

# Instantiate PtychoPINNTrainableReconstructor
reconstructor = PtychoPINNTrainableReconstructor(
    ptychopinn_model_settings, ptychopinn_training_settings, object_api
)

object_repository = ObjectRepository()
obj_group = SettingsGroup('Object')
object_settings = ObjectSettings(obj_group)
object_sizer = ObjectSizer(object_settings, apparatus, scan_sizer, probe_sizer)
object_factory = ObjectRepositoryItemFactory(rng, object_settings, object_sizer, ObjectRepository(), file_reader_chooser)
settings_delegate = ObjectRepositoryItemSettingsDelegate(object_settings, object_factory, object_repository)
reinit_observable = Observable()
selected_object = SelectedObject(object_repository, settings_delegate, reinit_observable)

# Create ObjectRepositoryItem from the loaded object
object_repository_item = ObjectRepositoryItem('Loaded Object')
object_repository_item.setObject(object_)

# Insert the item into the ObjectRepository
object_repository.insertItem(object_repository_item)

# Create ReconstructInput for training
train_input = ReconstructInput(
    diffractionPatternArray=diffraction_dataset[0].getData(),
    probeArray=probe.getArray(),
    objectInterpolator=object_api.getSelectedObjectInterpolator(),
    scan=scan,
)

# Call the train method
reconstructor.ingestTrainingData(train_input)
train_plot = reconstructor.train()

# Basic sanity checks on train output
assert isinstance(train_plot, Plot2D)
assert len(train_plot.axisX.series) > 0
assert len(train_plot.axisY.series) > 0

# Create ReconstructInput for reconstruction
reconstruct_input = ReconstructInput(
    diffractionPatternArray=diffraction_dataset[0].getData(),
    probeArray=probe.getArray(),
    objectInterpolator=object_api.getSelectedObjectInterpolator(),
    scan=scan,
)

# Call the reconstruct method
reconstruct_output = reconstructor.reconstruct(reconstruct_input)

# Basic sanity checks on reconstruction output
assert isinstance(reconstruct_output, ReconstructOutput)
assert reconstruct_output.objectArray is not None
assert reconstruct_output.objectArray.shape == object_.getArray().shape
assert reconstruct_output.result == 0

print("All tests passed!")
