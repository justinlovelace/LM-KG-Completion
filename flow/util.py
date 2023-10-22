import os
import time

class AverageMeter(object):
    """Computes and stores the average and current value.

    Adapted from: https://github.com/pytorch/examples/blob/master/imagenet/train.py
    """
    def __init__(self):
        self.val = 0.
        self.avg = 0.
        self.sum = 0.
        self.count = 0.

    def reset(self):
        self.val = 0.
        self.avg = 0.
        self.sum = 0.
        self.count = 0.

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def set_model_dir(args):
    model_dir = f'{args.dataset}/{time.time()}'
    args.output_dir = os.path.abspath(os.path.join(args.save_dir, model_dir))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    print(f'Will save model to {args.output_dir}')