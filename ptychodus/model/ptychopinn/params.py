"""
Stores global variables for data generation and model configuration
"""

from typing import Any

from .settings import PtychoPINNModelSettings, PtychoPINNTrainingSettings

# TODO naming convention for different types of parameters
# TODO what default value and initialization for the probe scale?

cfg: dict[str, Any] = {
    'N': 64,
    'offset': 4,
    'gridsize': 2,
    'outer_offset_train': None,
    'outer_offset_test': None,
    'batch_size': 16,
    'nepochs': 60,
    'n_filters_scale': 2,
    'output_prefix': 'outputs',
    'big_gridsize': 10,
    'max_position_jitter': 10,
    'sim_jitter_scale': 0.,
    'default_probe_scale': 0.7,
    'mae_weight': 0.,
    'nll_weight': 1.,
    'tv_weight': 0.,
    'realspace_mae_weight': 0.,
    'realspace_weight': 0.,
    'nphotons': 1e9,
    'nimgs_train': 9,
    'nimgs_test': 3,
    'data_source': 'lines',
    'probe.trainable': False,
    'intensity_scale.trainable': False,
    'positions.provided': False,
    'object.big': True,
    'probe.big': False,
    'probe_scale': 10.,
    'set_phi': False,
    'probe.mask': True,
    'model_type': 'pinn',
    'label': '',
    'size': 392,
    'amp_activation': 'sigmoid',
}

# TODO parameter description
# probe.big: if True, increase the real space solution from 32x32 to 64x64


# TODO bigoffset should be a derived quantity, at least for simulation
def get_bigN():
    N = cfg['N']
    gridsize = cfg['gridsize']
    offset = cfg['offset']
    return N + (gridsize - 1) * offset


def get_padding_size():
    buffer = cfg['max_position_jitter']
    gridsize = cfg['gridsize']
    offset = cfg['offset']
    return (gridsize - 1) * offset + buffer


def get_padded_size():
    bigN = get_bigN()
    buffer = cfg['max_position_jitter']
    return bigN + buffer


def params():
    d = {k: v for k, v in cfg.items()}
    d['bigN'] = get_bigN()
    return d


def update_cfg_from_settings(modelSettings: PtychoPINNModelSettings,
                             trainingSettings: PtychoPINNTrainingSettings) -> None:
    cfg['learning_rate'] = float(modelSettings.learningRate.value)
    cfg['N'] = modelSettings.N.value
    cfg['offset'] = modelSettings.offset.value
    cfg['gridsize'] = modelSettings.gridsize.value
    cfg['batch_size'] = modelSettings.batchSize.value
    cfg['n_filters_scale'] = modelSettings.nFiltersScale.value
    cfg['nphotons'] = float(modelSettings.nphotons.value)
    cfg['probe.trainable'] = modelSettings.probeTrainable.value
    cfg['intensity_scale.trainable'] = modelSettings.intensityScaleTrainable.value
    cfg['object.big'] = modelSettings.objectBig.value
    cfg['probe.big'] = modelSettings.probeBig.value
    cfg['probe_scale'] = float(modelSettings.probeScale.value)
    cfg['probe.mask'] = modelSettings.probeMask.value
    cfg['model_type'] = modelSettings.modelType.value
    cfg['size'] = modelSettings.size.value
    cfg['amp_activation'] = modelSettings.ampActivation.value

    cfg['mae_weight'] = float(trainingSettings.maeWeight.value)
    cfg['nll_weight'] = float(trainingSettings.nllWeight.value)
    cfg['tv_weight'] = float(trainingSettings.tvWeight.value)
    cfg['realspace_mae_weight'] = float(trainingSettings.realspaceMAEWeight.value)
    cfg['realspace_weight'] = float(trainingSettings.realspaceWeight.value)
    cfg['output_path'] = trainingSettings.outputPath.value
    cfg['output_suffix'] = trainingSettings.outputSuffix.value

    # Update derived values
    cfg['bigN'] = get_bigN()
    cfg['padded_size'] = get_padded_size()
    cfg['padding_size'] = get_padding_size()


# TODO refactor
def validate() -> bool:
    valid_data_sources = [
        'lines', 'grf', 'experimental', 'points', 'testimg', 'diagonals', 'xpp', 'V', 'generic'
    ]
    # FIXME asserts
    assert cfg['data_source'] in valid_data_sources, \
        f"Invalid data source: {cfg['data_source']}. Must be one of {valid_data_sources}."
    if cfg['realspace_mae_weight'] > 0.:
        assert cfg['realspace_weight'] > 0
    #assert cfg['bigoffset'] % 4 == 0
    # TODO
    return True


def set(key, value):
    cfg[key] = value
    assert validate()  # FIXME


def get(key):
    if key == 'bigN':
        cfg['bigN'] = get_bigN()
        return cfg['bigN']
    return cfg[key]
