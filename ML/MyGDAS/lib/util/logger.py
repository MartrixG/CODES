import sys
import time
from pathlib import Path


class Logger(object):

    def __init__(self, log_dir, seed, create_model_dir=True):
        """Create a summary writer logging to log_dir."""
        self.seed = int(seed)
        self.log_dir = Path(log_dir)
        self.model_dir = Path(log_dir) / 'checkpoint'
        self.log_dir.mkdir(parents=True, exist_ok=True)
        if create_model_dir:
            self.model_dir.mkdir(parents=True, exist_ok=True)
        # make log dir
        self.logger_path = self.log_dir / 'seed-{:}-T-{:}.log'.format(self.seed, time.strftime('%d-%h-at-%H-%M-%S',
                                                                                               time.gmtime(
                                                                                                   time.time())))
        self.logger_file = open(self.logger_path, 'w')

    def __repr__(self):
        return '{name}(dir={log_dir})'.format(name=self.__class__.__name__, **self.__dict__)

    def path(self, mode):
        # get all path
        valid = ('model', 'best', 'info', 'log')
        if mode == 'model':
            return self.model_dir / 'seed-{:}-basic.pth'.format(self.seed)
        elif mode == 'best':
            return self.model_dir / 'seed-{:}-best.pth'.format(self.seed)
        elif mode == 'info':
            return self.log_dir / 'seed-{:}-last-info.pth'.format(self.seed)
        elif mode == 'log':
            return self.log_dir
        else:
            raise TypeError('Unknown mode = {:}, valid modes = {:}'.format(mode, valid))

    def extract_log(self):
        return self.logger_file

    def close(self):
        self.logger_file.close()
        if self.writer is not None:
            self.writer.close()

    def log(self, string, save=True, stdout=False):
        if stdout:
            sys.stdout.write(string)
            sys.stdout.flush()
        else:
            print(string)
        if save:
            self.logger_file.write('{:}\n'.format(string))
            self.logger_file.flush()
