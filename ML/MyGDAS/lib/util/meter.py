class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.count = 0.0
        self.sum = 0.0
        self.val = 0.0
        self.avg = 0.0
        self.reset()

    def reset(self):
        pass

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __repr__(self):
        return '{name}(val={val}, avg={avg}, count={count})'.format(name=self.__class__.__name__, **self.__dict__)
