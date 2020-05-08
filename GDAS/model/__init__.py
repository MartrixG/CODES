__all__ = ['get_cell_based_tiny_net']


def get_cell_based_tiny_net(config):
    super_type = getattr(config, 'super_type', 'basic')
    if super_type == 'nasnet-super':
        from model.search_model_gdas_nasnet import NASNetWorkGDAS
        return NASNetWorkGDAS(config.C, config.N, config.steps, config.multiplier, \
                              config.stem_multiplier, config.num_classes, config.space, config.affine,
                              config.track_running_stats)
