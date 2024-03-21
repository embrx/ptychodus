from pathlib import Path
import numpy as np
from ptychodus.api.plugins import PluginChooser, PluginRegistry
from ptychodus.api.observer import Observable
from ptychodus.api.object import ObjectPhaseCenteringStrategy, ObjectFileReader, ObjectFileWriter
from ptychodus.api.reconstructor import ReconstructInput, ReconstructOutput
from ptychodus.api.settings import SettingsRegistry, SettingsGroup
from ptychodus.api.plot import Plot2D
from ptychodus.model.data.settings import DiffractionPatternSettings
from ptychodus.model.data.sizer import DiffractionPatternSizer
from ptychodus.model.detector import Detector, DetectorSettings
from ptychodus.model.itemRepository import SelectedRepositoryItem
from ptychodus.model.object.api import ObjectAPI
from ptychodus.model.object.factory import ObjectRepositoryItemFactory
from ptychodus.model.object.interpolator import ObjectInterpolatorFactory
from ptychodus.model.object.repository import ObjectRepository, ObjectRepositoryItem
from ptychodus.model.object.selected import ObjectRepositoryItemSettingsDelegate, SelectedObject
from ptychodus.model.object.settings import ObjectSettings
from ptychodus.model.object.sizer import ObjectSizer
from ptychodus.model.probe import Apparatus, ProbeSizer, ProbeSettings
from ptychodus.model.ptychopinn.reconstructor import PtychoPINNTrainableReconstructor
from ptychodus.model.ptychopinn.settings import PtychoPINNModelSettings, PtychoPINNTrainingSettings
from ptychodus.model.scan import ScanSizer
from ptychodus.model.scan.repository import ScanRepository, ScanRepositoryItem
from ptychodus.model.scan.factory import ScanRepositoryItemFactory
from ptychodus.model.scan.selected import ScanRepositoryItemSettingsDelegate, SelectedScan
from ptychodus.model.scan.settings import ScanSettings
from ptychodus.plugins.slacFile import SLACDiffractionFileReader, SLACScanFileReader, SLACProbeFileReader, SLACObjectFileReader
from ptychodus.plugins.objectPhaseCentering import IdentityPhaseCenteringStrategy, CenterBoxMeanPhaseCenteringStrategy

# Assume the file path is stored in this variable
data_file_path = Path('/home/ollie/Documents/scratch/ptycho/ptycho/datasets/Run1084_recon3_postPC_shrunk_3.npz')

# Read the data from the NPZ file
with np.load(data_file_path) as data:
    diffraction_patterns = data['diffraction']
    probe_array = data['probeGuess']
    object_array = data['objectGuess']
    xcoords_start = data['xcoords_start']
    ycoords_start = data['ycoords_start']

# Split the data into separate files
diffraction_file_path = data_file_path.with_name('diffraction_file.npz')
np.savez(diffraction_file_path, diffraction=diffraction_patterns)

probe_file_path = data_file_path.with_name('probe_file.npz')
np.savez(probe_file_path, probeGuess=probe_array)

object_file_path = data_file_path.with_name('object_file.npz')
np.savez(object_file_path, objectGuess=object_array)

scan_file_path = data_file_path.with_name('scan_file.npz')
np.savez(scan_file_path, xcoords_start=xcoords_start, ycoords_start=ycoords_start)

# Create instances of the file readers
diffraction_reader = SLACDiffractionFileReader()
scan_reader = SLACScanFileReader()
probe_reader = SLACProbeFileReader()
object_reader = SLACObjectFileReader()

# Read the data using the SLAC file readers
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

# Register phase centering strategy plugins
plugin_registry = PluginRegistry()
plugin_registry.objectPhaseCenteringStrategies.registerPlugin(IdentityPhaseCenteringStrategy(), simpleName='Identity')
plugin_registry.objectPhaseCenteringStrategies.registerPlugin(CenterBoxMeanPhaseCenteringStrategy(), simpleName='CenterBoxMean')

# Create SelectedScan instance
scan_repository = ScanRepository()
scan_file_reader_chooser = PluginChooser[SLACScanFileReader]()
scan_item_factory = ScanRepositoryItemFactory(rng, scan_settings, scan_file_reader_chooser)
scan_settings_delegate = ScanRepositoryItemSettingsDelegate(scan_settings, scan_item_factory, scan_repository)
selected_scan = SelectedScan.createInstance(scan_repository, scan_settings_delegate, settings_registry)

scan_sizer = ScanSizer.createInstance(scan_settings, selected_scan)

phase_centering_strategy_chooser = plugin_registry.objectPhaseCenteringStrategies
file_writer_chooser = PluginChooser[ObjectFileWriter]()

object_settings_group = SettingsGroup('objectsettings')
object_settings = ObjectSettings(object_settings_group)
object_sizer = ObjectSizer(object_settings, apparatus, scan_sizer, probe_sizer)

reinit_observable = Observable()
interpolator_factory = ObjectInterpolatorFactory.createInstance(
    object_settings,
    object_sizer,
    phase_centering_strategy_chooser,
    reinit_observable
)

object_repository = ObjectRepository()
object_file_reader_chooser = PluginChooser[SLACObjectFileReader]()
object_factory = ObjectRepositoryItemFactory(rng, object_settings, object_sizer, object_repository, object_file_reader_chooser)
settings_delegate = ObjectRepositoryItemSettingsDelegate(object_settings, object_factory, object_repository)
selected_object = SelectedObject(object_repository, settings_delegate, reinit_observable)

object_api = ObjectAPI(object_factory, object_repository, selected_object, object_sizer, interpolator_factory)

# Instantiate PtychoPINNTrainableReconstructor
reconstructor = PtychoPINNTrainableReconstructor(
    ptychopinn_model_settings, ptychopinn_training_settings, object_api
)

# Create ObjectRepositoryItem from the loaded object
object_repository_item = ObjectRepositoryItem('Loaded Object')
object_repository_item.setObject(object_)

# Insert the item into the ObjectRepository
object_repository.insertItem(object_repository_item)

selected_object.selectItem(object_repository_item.nameHint)

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
