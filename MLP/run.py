import argparse
import os
import sys
from types import SimpleNamespace

from utils.util import load_config
from model.train import main

if __name__ == '__main__':
    curPath = os.path.abspath(os.path.dirname(__file__))
    rootPath = os.path.split(curPath)[0]
    PathProject = os.path.split(rootPath)[0]
    sys.path.append(rootPath)
    sys.path.append(PathProject)
    parser = argparse.ArgumentParser("MLP")
    parser.add_argument('--track_running_stats', default=True, type=int,
                        help='Whether use track_running_stats or not in the BN layer.')
    parser.add_argument('--config', default='config/search-config/UCI-config.json', type=str,
                        help='The path of the configuration.')
    parser.add_argument('--data_path', default='data/', type=str)
    parser.add_argument('--split', default='config/split-file/', type=str)
    parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
    parser.add_argument('--workers', type=int, default=4, help='number of data loading workers (default: 4)')
    parser.add_argument('--print_frequency', type=int, default=50, help='print frequency (default: 200)')
    parser.add_argument('--seed', type=int, default=1, help='manual seed')
    parser.add_argument('--gpu', type=int, default=0, help='gpu device id')

    parse = vars(parser.parse_args())
    configs = load_config(parse['config'], True)
    for k in configs:
        parse[k] = configs[k]
    args = SimpleNamespace(**parse)
    main(args)
