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

# Split the data into train and reconstruct subsets 
train_size = 512

train_diffraction_file_path = data_file_path.with_name('train_diffraction_file.npz')
np.savez(train_diffraction_file_path, diffraction=diffraction_patterns[:, :, :train_size])

train_probe_file_path = data_file_path.with_name('train_probe_file.npz') 
np.savez(train_probe_file_path, probeGuess=probe_array)

train_object_file_path = data_file_path.with_name('train_object_file.npz')
np.savez(train_object_file_path, objectGuess=object_array)

train_scan_file_path = data_file_path.with_name('train_scan_file.npz')
np.savez(train_scan_file_path, xcoords_start=xcoords_start[:train_size], ycoords_start=ycoords_start[:train_size])

reconstruct_diffraction_file_path = data_file_path.with_name('reconstruct_diffraction_file.npz')
np.savez(reconstruct_diffraction_file_path, diffraction=diffraction_patterns)

reconstruct_probe_file_path = data_file_path.with_name('reconstruct_probe_file.npz')
np.savez(reconstruct_probe_file_path, probeGuess=probe_array)

reconstruct_object_file_path = data_file_path.with_name('reconstruct_object_file.npz') 
np.savez(reconstruct_object_file_path, objectGuess=object_array)

reconstruct_scan_file_path = data_file_path.with_name('reconstruct_scan_file.npz')
np.savez(reconstruct_scan_file_path, xcoords_start=xcoords_start, ycoords_start=ycoords_start)

# Create instances of the file readers
diffraction_reader = SLACDiffractionFileReader()
scan_reader = SLACScanFileReader()
probe_reader = SLACProbeFileReader()
object_reader = SLACObjectFileReader()

# Read the train data using the SLAC file readers
train_diffraction_dataset = diffraction_reader.read(train_diffraction_file_path)
train_scan = scan_reader.read(train_scan_file_path)
train_probe = probe_reader.read(train_probe_file_path)
train_object = object_reader.read(train_object_file_path)

# Read the reconstruct data using the SLAC file readers  
reconstruct_diffraction_dataset = diffraction_reader.read(reconstruct_diffraction_file_path)
reconstruct_scan = scan_reader.read(reconstruct_scan_file_path)  
reconstruct_probe = probe_reader.read(reconstruct_probe_file_path)
reconstruct_object = object_reader.read(reconstruct_object_file_path)

# Create dummy data and objects for initializing PtychoPINNTrainableReconstructor
rng = np.random.default_rng(42)  # Random number generator with seed

settings_registry = SettingsRegistry('ptychodus')
ptychopinn_model_settings = PtychoPINNModelSettings.createInstance(settings_registry)
ptychopinn_training_settings = PtychoPINNTrainingSettings.createInstance(settings_registry)

# Setting reconstruction parameters
ptychopinn_model_settings.gridsize.value = 1
ptychopinn_model_settings.probeScale.value = 5.

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

# Create ObjectRepositoryItem from the loaded train object
train_object_repository_item = ObjectRepositoryItem('Loaded Train Object')
train_object_repository_item.setObject(train_object)

# Insert the item into the ObjectRepository
object_repository.insertItem(train_object_repository_item)

selected_object.selectItem(train_object_repository_item.nameHint)

# Create ReconstructInput for training
train_input = ReconstructInput(
    diffractionPatternArray=train_diffraction_dataset[0].getData(),
    probeArray=train_probe.getArray(),
    objectInterpolator=object_api.getSelectedObjectInterpolator(),
    scan=train_scan,
)

# Call the train method
reconstructor.ingestTrainingData(train_input)  
train_plot = reconstructor.train()

# Basic sanity checks on train output
assert isinstance(train_plot, Plot2D)
assert len(train_plot.axisX.series) > 0
assert len(train_plot.axisY.series) > 0

# Create ObjectRepositoryItem from the loaded reconstruct object
reconstruct_object_repository_item = ObjectRepositoryItem('Loaded Reconstruct Object')
reconstruct_object_repository_item.setObject(reconstruct_object)

# Insert the item into the ObjectRepository
object_repository.insertItem(reconstruct_object_repository_item)

selected_object.selectItem(reconstruct_object_repository_item.nameHint)  

# Create ReconstructInput for reconstruction
reconstruct_input = ReconstructInput( 
    diffractionPatternArray=reconstruct_diffraction_dataset[0].getData(),
    probeArray=reconstruct_probe.getArray(),
    objectInterpolator=object_api.getSelectedObjectInterpolator(),
    scan=reconstruct_scan,
)

# Call the reconstruct method
reconstruct_output = reconstructor.reconstruct(reconstruct_input)

# Basic sanity checks on reconstruction output
assert isinstance(reconstruct_output, ReconstructOutput)
assert reconstruct_output.objectArray is not None  
assert reconstruct_output.objectArray.shape == reconstruct_object.getArray().shape
assert reconstruct_output.result == 0

print("All tests passed!")
