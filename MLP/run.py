import argparse
import sys
import random
from pathlib import Path
from types import SimpleNamespace

import torch

from model.model_search import Classifier
from model.model_train import Network
from utils.data_process import get_src_dataset, get_search_loader
from utils.util import load_config, count_parameters_in_MB

if __name__ == '__main__':
    lib_dir = (Path(__file__).parent / '..' / '..' / 'lib').resolve()
    if str(lib_dir) not in sys.path:
        sys.path.insert(0, str(lib_dir))
    parser = argparse.ArgumentParser("MLP")
    parser.add_argument('--track_running_stats', default=True, type=int,
                        help='Whether use track_running_stats or not in the BN layer.')
    parser.add_argument('--search_config', default='config/search-config/UCI-config.json', type=str,
                        help='The path of the configuration.')
    parser.add_argument('--train_config', default='config/train-config/UCI-config.json', type=str,
                        help='The path of the training configuration.')
    parser.add_argument('--workers', type=int, default=4, help='number of data loading workers (default: 4)')
    parser.add_argument('--save_dir', type=str, default='log/', help='Folder to save checkpoints and log.')
    parser.add_argument('--print_frequency', type=int, default=5, help='print frequency (default: 200)')
    parser.add_argument('--rand_seed', type=int, default=1, help='manual seed')
    parser.add_argument('--evaluate', type=str, default='search', help='choose train or test')

    parse = vars(parser.parse_args())
    configs = load_config(parse['train_config'], True)
    for k in configs:
        parse[k] = configs[k]
    args = SimpleNamespace(**parse)




    train_data, test_data, x_shape, class_num = get_src_dataset(args.data_path, args.name)
    search_loader, train_loader, valid_loader, test_loader = get_search_loader(
        train_data, test_data, args.name, args.split, args.workers, args.batch_size)
    model = Network(args.name, x_shape, class_num, args.premodel, args.genotype, args)
    model = model.cuda()
    for (feature, label) in train_loader:
        feature = feature.cuda()
        res = model(feature)
        print(res)
        print(res.shape)
        break
    # print(count_parameters_in_MB(model))
    # model = Classifier(8, 2, 561, 12)
    # model.genotype(args.genotype)
    # test_data = torch.rand((4, 561))
    # out = model(test_data)
    # print(out.shape)
