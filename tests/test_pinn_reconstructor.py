import numpy as np
from ptychodus.model.ptychopinn.reconstructor import PtychoPINNTrainableReconstructor
from ptychodus.model.ptychopinn.settings import PtychoPINNModelSettings, PtychoPINNTrainingSettings
from ptychodus.model.object.api import ObjectAPI
from ptychodus.api.reconstructor import ReconstructInput, ReconstructOutput
from ptychodus.api.scan import TabularScan
from ptychodus.api.settings import SettingsRegistry
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

def main():
    # Settings Initialization
    settings_registry = SettingsRegistry(replacementPathPrefix="/path/to/prefix")
    model_settings = PtychoPINNModelSettings.createInstance(settings_registry)
    training_settings = PtychoPINNTrainingSettings.createInstance(settings_registry)

    # Dependency Initialization
    rng = np.random.default_rng()
    detector = Detector()  # Provide a valid Detector instance
    diffraction_pattern_sizer = DiffractionPatternSizer()  # Provide a valid DiffractionPatternSizer instance
    probe_settings = ProbeSettings()  # Provide a valid ProbeSettings instance
    apparatus = Apparatus(detector, diffraction_pattern_sizer, probe_settings)
    scan_settings = ScanSettings()  # Provide a valid ScanSettings instance
    selected_scan = SelectedScan()  # Provide a valid SelectedScan instance
    scan_sizer = ScanSizer(scan_settings, selected_scan)
    probe_sizer = ProbeSizer(diffraction_pattern_sizer)
    object_settings = ObjectSettings(settings_registry.createGroup('Object'))
    object_sizer = ObjectSizer(object_settings, apparatus, scan_sizer, probe_sizer)

    # ObjectAPI Initialization
    file_reader_chooser = PluginChooser[ObjectFileReader]()  # Provide a valid PluginChooser[ObjectFileReader] instance
    object_factory = ObjectRepositoryItemFactory(rng, object_settings, object_sizer, ObjectRepository(), file_reader_chooser)
    object_repository = ObjectRepository()
    settings_delegate = ObjectRepositoryItemSettingsDelegate(object_settings, object_factory, object_repository)
    reinit_observable = Observable()  # Provide a valid Observable instance
    selected_object = SelectedObject(object_repository, settings_delegate, reinit_observable)
    phase_centering_strategy_chooser = PluginChooser[ObjectPhaseCenteringStrategy]()  # Provide a valid PluginChooser[ObjectPhaseCenteringStrategy] instance
    interpolator_factory = ObjectInterpolatorFactory(object_settings, object_sizer, phase_centering_strategy_chooser, reinit_observable)
    object_api = ObjectAPI(object_factory, object_repository, selected_object, object_sizer, interpolator_factory)

    # PtychoPINNTrainableReconstructor Initialization
    reconstructor = PtychoPINNTrainableReconstructor(model_settings, training_settings, object_api)

    # Dummy Data Generation
    num_patterns = 100  # Number of diffraction patterns
    pattern_size = 64  # Size of each diffraction pattern (assumes square patterns)
    object_size = 128  # Size of the object (assumes square object)

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

    # Training
    reconstructor.ingestTrainingData(reconstruct_input)
    train_plot = reconstructor.train()

    # Reconstruction
    reconstruct_output = reconstructor.reconstruct(reconstruct_input)

    # Output and Assertions
    print("Training Plot:")
    print(train_plot)

    reconstructed_object = reconstruct_output.objectArray
    if reconstructed_object is not None:
        print("Reconstructed Object Shape:", reconstructed_object.shape)
        assert reconstructed_object.shape == (object_size, object_size)
        assert np.iscomplexobj(reconstructed_object)
    else:
        print("Reconstructed object is None")

    # Cleanup
    reconstructor.clearTrainingData()

def test_ptychopinn_trainable_reconstructor():
    # Settings Initialization
    settings_registry = SettingsRegistry(replacementPathPrefix="/path/to/prefix")
    model_settings = PtychoPINNModelSettings.createInstance(settings_registry)
    training_settings = PtychoPINNTrainingSettings.createInstance(settings_registry)
    object_settings = ObjectSettings(settings_registry.createGroup('Object'))

    # Dependency Initialization
    rng = np.random.default_rng()
    detector = Detector()  # Provide a valid Detector instance
    diffraction_pattern_sizer = DiffractionPatternSizer()  # Provide a valid DiffractionPatternSizer instance
    probe_settings = ProbeSettings()  # Provide a valid ProbeSettings instance
    apparatus = Apparatus(detector, diffraction_pattern_sizer, probe_settings)
    scan_settings = ScanSettings()  # Provide a valid ScanSettings instance
    selected_scan = SelectedScan()  # Provide a valid SelectedScan instance
    scan_sizer = ScanSizer(scan_settings, selected_scan)
    probe_sizer = ProbeSizer(diffraction_pattern_sizer)
    object_sizer = ObjectSizer(object_settings, apparatus, scan_sizer, probe_sizer)

    # ObjectAPI Initialization
    file_reader_chooser = PluginChooser[ObjectFileReader]()  # Provide a valid PluginChooser[ObjectFileReader] instance
    object_factory = ObjectRepositoryItemFactory(rng, object_settings, object_sizer, ObjectRepository(), file_reader_chooser)
    object_repository = ObjectRepository()
    settings_delegate = ObjectRepositoryItemSettingsDelegate(object_settings, object_factory, object_repository)
    reinit_observable = Observable()  # Provide a valid Observable instance
    selected_object = SelectedObject(object_repository, settings_delegate, reinit_observable)
    phase_centering_strategy_chooser = PluginChooser[ObjectPhaseCenteringStrategy]()  # Provide a valid PluginChooser[ObjectPhaseCenteringStrategy] instance
    interpolator_factory = ObjectInterpolatorFactory(object_settings, object_sizer, phase_centering_strategy_chooser, reinit_observable)
    object_api = ObjectAPI(object_factory, object_repository, selected_object, object_sizer, interpolator_factory)

    # PtychoPINNTrainableReconstructor Initialization
    reconstructor = PtychoPINNTrainableReconstructor(model_settings, training_settings, object_api)

    # Dummy Data Generation
    num_patterns = 100  # Number of diffraction patterns
    pattern_size = 64  # Size of each diffraction pattern (assumes square patterns)
    object_size = 128  # Size of the object (assumes square object)

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

    # Add more assertions based on your specific requirements and expected outputs

    print("All tests passed!")

if __name__ == "__main__":
    main()
    test_ptychopinn_trainable_reconstructor()
