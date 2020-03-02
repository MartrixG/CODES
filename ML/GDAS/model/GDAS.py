import sys
from pathlib import Path
from lib.util.starts import prepare_seed, prepare_logger
from lib.util.configure_util import load_config
from lib.dataset.get_dataset_with_transform import get_dataset
import torch
from torch.backends import cudnn



def train(xargs):
    lib_dir = (Path(__file__).parent / '..' / '..' / 'lib').resolve()
    if str(lib_dir) not in sys.path:
        sys.path.insert(0, str(lib_dir))

    assert (torch.cuda.is_available(), 'CUDA is not available.')
    # start cudnn
    cudnn.enabled = True
    # make each conv is the same
    cudnn.benchmark = False
    # make sure the same seed has the same result
    cudnn.deterministic = True
    torch.set_num_threads(xargs.workers)
    prepare_seed(xargs.rand_seed)
    logger = prepare_logger(xargs)
    train_data, valid_data, xshape, class_num = get_dataset(xargs.dataset, xargs.data_path, -1)
    config = load_config(xargs.config_path, {'class_num': class_num, 'xshape': xshape}, logger)
