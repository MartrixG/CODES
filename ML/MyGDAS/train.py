import argparse
import sys
import random
from pathlib import Path
from model import MyGDAS, evaluate

if __name__ == '__main__':
    lib_dir = (Path(__file__).parent / '..' / '..' / 'lib').resolve()
    if str(lib_dir) not in sys.path:
        sys.path.insert(0, str(lib_dir))
    parser = argparse.ArgumentParser("MyGDAS")
    # data set
    parser.add_argument('--data_path', type=str, default='data/HAPT/', help='Path to dataset')
    parser.add_argument('--dataset', type=str, default='HAPT', choices=['cifar10', 'cifar100', 'HAPT'],
                        help='Choose between Cifar10/100 and HAPT.')
    # search model
    parser.add_argument('--search_space_name', default='HAPT', type=str, help='The search space name.')
    parser.add_argument('--track_running_stats', default=1, type=int, choices=[0, 1],
                        help='Whether use track_running_stats or not in the BN layer.')
    parser.add_argument('--opt_config', default='config/MyGDAS-opt.config', type=str, help='The path of the '
                                                                                            'configuration.')
    parser.add_argument('--arch_config', default='config/MyGDAS-arch.config', type=str,
                        help='The path of the model configuration.')
    # evaluate config
    parser.add_argument("--eva_config", default='config/evaluate.config', type=str,
                        help='The path of the evaluate models\' configuration.')
    # architecture learning rate
    parser.add_argument('--arch_learning_rate', type=float, default=0.003, help='learning rate for arch encoding')
    parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
    parser.add_argument('--tau_min', type=float, default=0.1, help='The minimum tau for Gumbel')
    parser.add_argument('--tau_max', type=float, default=10, help='The maximum tau for Gumbel')

    # opt learning rate
    parser.add_argument('--epochs', type=int, help='training epoch')
    parser.add_argument('--batch_size', type=int, help='batch_size')
    parser.add_argument('--opt_learning_rate', type=float, help='opt learning rate')
    # log
    parser.add_argument('--record_file', type=str, default='Records.txt',
                        help='The path to log the record for each epoch.')
    parser.add_argument('--genotype_file', type=str, default='genotype.txt')

    parser.add_argument('--workers', type=int, default=4, help='number of data loading workers (default: 4)')
    parser.add_argument('--save_dir', type=str, default='log/', help='Folder to save checkpoints and log.')
    parser.add_argument('--print_frequency', type=int, default=4, help='print frequency (default: 200)')
    parser.add_argument('--rand_seed', type=int, default=-1, help='manual seed')
    parser.add_argument('--evaluate', type=str, default='train', help='choose train or test')
    args = parser.parse_args()
    if args.rand_seed is None or args.rand_seed < 0:
        args.rand_seed = random.randint(1, 100000)
    if args.evaluate == 'evaluate':
        evaluate.evaluate(args)
    else:
        MyGDAS.train(args)
