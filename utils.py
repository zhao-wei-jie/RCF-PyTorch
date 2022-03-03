import imp
import os
import logging
import numpy as np
import torch
import torch.nn.functional as F
from models import RCF,NextRCF
def select_model(args):
    if args.model=='rcf':
        # print(fuse)
        return RCF(pretrained='vgg16convs.mat',dataflag=args.dataflag,fuse=args.fuse_num,short_cat=args.short_cat).cuda()
    elif args.model=='convnext':
        return NextRCF().cuda()


class EvalMax:
    def __init__(self) -> None:
        self.max_eval={}
        self.updated=False
    def __call__(self, ret,i=None):
        for k in ret.keys():#通过键，批量对比大小
                if 'pl'  in k and ret[k]>self.max_eval.setdefault(k,0):
                    self.updated=True
                    self.max_eval[k]=ret[k]
                    if i:
                        self.max_eval[k+'_epo']=i
        return self.max_eval
    def hasupdate(self):
        if self.updated:
            self.updated=False
            return True

class Logger(object):
    def __init__(self, path='log.txt'):
        self.logger = logging.getLogger('Logger')
        self.file_handler = logging.FileHandler(path, 'w')
        self.stdout_handler = logging.StreamHandler()
        self.logger.addHandler(self.file_handler)
        self.logger.addHandler(self.stdout_handler)
        self.stdout_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
        self.file_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
        self.logger.setLevel(logging.INFO)

    def info(self, txt):
        self.logger.info(txt)

    def close(self):
        self.file_handler.close()
        self.stdout_handler.close()


class Averagvalue(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def Cross_entropy_loss(prediction, label):
    mask = label.clone()
    num_positive = torch.sum((mask == 1).float()).float()
    num_negative = torch.sum((mask == 0).float()).float()

    mask[mask == 1] = 1.0 * num_negative / (num_positive + num_negative)
    mask[mask == 0] = 1.1 * num_positive / (num_positive + num_negative)
    mask[mask == 2] = 0
    cost = F.binary_cross_entropy_with_logits(prediction, label, weight=mask
        # , reduce=False
        # ,reduction='sum'
        )
    return torch.sum(cost)
