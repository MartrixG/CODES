import argparse
import sys
import random
from pathlib import Path

if __name__ == '__main__':
    lib_dir = (Path(__file__).parent / '..' / '..' / 'lib').resolve()
    if str(lib_dir) not in sys.path:
        sys.path.insert(0, str(lib_dir))
    parser = argparse.ArgumentParser("MyGDAS")
    # data set
    parser.add_argument('--data_path', type=str, default='data/', help='Path to dataset')
    parser.add_argument('--dataset', type=str, default='HAPT', choices=['cifar10', 'cifar100', 'HAPT'],
                        help='Choose between Cifar10/100 and HAPT.')

    parser.add_argument('--search_space_name', default='HAPT', type=str, help='The search space name.')
    parser.add_argument('--track_running_stats', default=1, type=int, choices=[0, 1],
                        help='Whether use track_running_stats or not in the BN layer.')
    parser.add_argument('--opt_config', default='config/MyGDAS-opt.config', type=str, help='The path of the '
                                                                                            'configuration.')
    parser.add_argument('--arch_config', default='config/MyGDAS-arch.config', type=str,
                        help='The path of the model configuration. When this arg is set, it will cover max_nodes / '
                             'channels / num_cells.')
    # architecture learning rate
    parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
    parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
    parser.add_argument('--tau_min', type=float, default=0.1, help='The minimum tau for Gumbel')
    parser.add_argument('--tau_max', type=float, default=10, help='The maximum tau for Gumbel')
    # log
    parser.add_argument('--arch_nas_dataset', type=str,
                        help='The path to load the architecture dataset (tiny-nas-benchmark).')

    parser.add_argument('--workers', type=int, default=4, help='number of data loading workers (default: 2)')
    parser.add_argument('--save_dir', type=str, default='log/', help='Folder to save checkpoints and log.')
    parser.add_argument('--print_freq', type=int, default=200, help='print frequency (default: 200)')
    parser.add_argument('--rand_seed', type=int, default=-1, help='manual seed')
    args = parser.parse_args()
    if args.rand_seed is None or args.rand_seed < 0:
        args.rand_seed = random.randint(1, 100000)
