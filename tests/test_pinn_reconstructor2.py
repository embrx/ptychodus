import numpy as np
from ptychodus.model.ptychopinn.reconstructor import PtychoPINNTrainableReconstructor
from ptychodus.model.ptychopinn.settings import PtychoPINNModelSettings, PtychoPINNTrainingSettings
from ptychodus.model.object.api import ObjectAPI
from ptychodus.api.reconstructor import ReconstructInput, ReconstructOutput
from ptychodus.api.scan import TabularScan
from ptychodus.api.settings import SettingsRegistry, SettingsGroup
from ptychodus.model.object.factory import ObjectRepositoryItemFactory
from ptychodus.model.object.repository import ObjectRepository
from ptychodus.model.object.selected import SelectedObject, ObjectRepositoryItemSettingsDelegate
from ptychodus.model.object.sizer import ObjectSizer
from ptychodus.model.object.interpolator import ObjectInterpolatorFactory
from ptychodus.model.probe import Apparatus, ProbeSizer, Detector, DiffractionPatternSizer, ProbeSettings
from ptychodus.model.scan import ScanSizer, ScanSettings, SelectedScan
from ptychodus.api.plugins import PluginChooser
from ptychodus.api.object import ObjectFileReader, ObjectPhaseCenteringStrategy
from ptychodus.api.observer import Observable
from ptychodus.api.plot import Plot2D
from ptychodus.model.object.settings import ObjectSettings

def test_ptychopinn_trainable_reconstructor():
    # Settings Initialization
    settings_registry = SettingsRegistry(replacementPathPrefix="/path/to/prefix")
    model_settings_group = SettingsGroup("PtychoPINNModelSettings")
    model_settings = PtychoPINNModelSettings(model_settings_group)
    training_settings_group = SettingsGroup("PtychoPINNTrainingSettings")
    training_settings = PtychoPINNTrainingSettings(training_settings_group)
    object_settings = ObjectSettings(settings_registry.createGroup('Object'))

    # Dependency Initialization
    rng = np.random.default_rng()
    detector = Detector()
    diffraction_pattern_sizer = DiffractionPatternSizer()
    probe_settings = ProbeSettings()
    apparatus = Apparatus(detector, diffraction_pattern_sizer, probe_settings)
    scan_settings = ScanSettings()
    selected_scan = SelectedScan()
    scan_sizer = ScanSizer(scan_settings, selected_scan)
    probe_sizer = ProbeSizer(diffraction_pattern_sizer)
    object_sizer = ObjectSizer(object_settings, apparatus, scan_sizer, probe_sizer)

    # ObjectAPI Initialization
    file_reader_chooser = PluginChooser[ObjectFileReader]()
    object_factory = ObjectRepositoryItemFactory(rng, object_settings, object_sizer, ObjectRepository(), file_reader_chooser)
    object_repository = ObjectRepository()
    settings_delegate = ObjectRepositoryItemSettingsDelegate(object_settings, object_factory, object_repository)
    reinit_observable = Observable()
    selected_object = SelectedObject(object_repository, settings_delegate, reinit_observable)
    phase_centering_strategy_chooser = PluginChooser[ObjectPhaseCenteringStrategy]()
    interpolator_factory = ObjectInterpolatorFactory(object_settings, object_sizer, phase_centering_strategy_chooser, reinit_observable)
    object_api = ObjectAPI(object_factory, object_repository, selected_object, object_sizer, interpolator_factory)

    # PtychoPINNTrainableReconstructor Initialization
    reconstructor = PtychoPINNTrainableReconstructor(model_settings, training_settings, object_api)

    # Dummy Data Generation
    num_patterns = 100
    pattern_size = 64
    object_size = 128

    diffraction_patterns = np.random.randint(0, 255, (num_patterns, pattern_size, pattern_size), dtype=np.uint8)
    scan_coordinates = np.random.rand(num_patterns, 2).astype(np.float32)
    probe_guess = np.random.rand(1, pattern_size, pattern_size).astype(np.complex64)
    object_guess = np.random.rand(1, object_size, object_size).astype(np.complex64)

    # ReconstructInput Creation
    reconstruct_input = ReconstructInput(
        diffractionPatternArray=diffraction_patterns,
        scan=TabularScan.createFromPointIterable(scan_coordinates),
        probeArray=probe_guess,
        objectInterpolator=object_api.getSelectedObjectInterpolator()
    )

    # Test the ingestTrainingData() method
    reconstructor.ingestTrainingData(reconstruct_input)

    # Test the train() method
    train_plot = reconstructor.train()

    # Assert that the train_plot is of type Plot2D
    assert isinstance(train_plot, Plot2D)

    # Assert that the train_plot has the expected axes labels
    assert train_plot.axisX.label == 'Epoch'
    assert train_plot.axisY.label == 'Loss'

    # Assert that the train_plot has the expected series labels
    assert len(train_plot.axisX.series) == 1
    assert len(train_plot.axisY.series) == 2
    assert train_plot.axisX.series[0].label == 'Epoch'
    assert train_plot.axisY.series[0].label == 'Training Loss'
    assert train_plot.axisY.series[1].label == 'Validation Loss'

    # Test the reconstruct() method
    reconstruct_output = reconstructor.reconstruct(reconstruct_input)

    # Assert that the reconstruct_output is of type ReconstructOutput
    assert isinstance(reconstruct_output, ReconstructOutput)

    # Assert that the reconstructed object array has the expected shape
    reconstructed_object = reconstruct_output.objectArray
    if reconstructed_object is not None:
        assert reconstructed_object.shape == (object_size, object_size)

        # Assert that the reconstructed object array is of complex data type
        assert np.iscomplexobj(reconstructed_object)
    else:
        print("Reconstructed object is None")

    # Test the saveTrainingData() method
    save_file_path = "/path/to/save/file.npz"
    reconstructor.saveTrainingData(save_file_path)

    # Test the clearTrainingData() method
    reconstructor.clearTrainingData()

    print("All tests passed!")

if __name__ == "__main__":
    test_ptychopinn_trainable_reconstructor()
