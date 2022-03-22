import imp
import os
import logging
import numpy as np
import torch
import torch.nn.functional as F
from models import RCF, NextRCF
from other_models.unet_model import UNet
import argparse
from other_models.config import config,update_config
import other_models.hr_models
from other_models.semseg.models import Lawin
import mmcv

def argsF():
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('--batch-size', default=1, type=int, help='batch size')
    parser.add_argument('--opt', default='adamw', type=str, help='opt')
    parser.add_argument('--model', default='rcf', type=str, help='rcf')
    parser.add_argument('--lr', default=1e-6, type=float,
                        help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--weight-decay', default=2e-4,
                        type=float, help='weight decay')
    parser.add_argument('--stepsize', default=3, type=int,
                        help='learning rate step size')
    parser.add_argument('--gamma', default=0.1, type=float,
                        help='learning rate decay rate')
    parser.add_argument('--max-epoch', default=10, type=int,
                        help='the number of training epochs')
    parser.add_argument('--iter-size', default=10, type=int, help='iter size')
    parser.add_argument('--start-epoch', default=0,
                        type=int, help='manual epoch number')
    parser.add_argument('--print-freq', default=200,
                        type=int, help='print frequency')
    parser.add_argument('--gpu', default='0', type=str, help='GPU ID')
    parser.add_argument('--resume', default=None, type=str,
                        help='path to latest checkpoint')
    parser.add_argument('--pretrain', default=None, type=str,
                        help='path to latest checkpoint')
    parser.add_argument('--save-dir', help='output folder', default='results/')
    parser.add_argument(
        '--dataset', help='root folder of dataset', default=None)
    parser.add_argument('--dataflag', default='color',
                        help='color or grayscale')
    parser.add_argument('--amp', default='O0', help='O0~O3')
    parser.add_argument('--aug', default=False,
                        type=bool, help='true or false')
    parser.add_argument('--fuse_num', default=5, help='5')
    parser.add_argument('--short_cat', default=0, type=int)
    parser.add_argument('--scale', default=False, type=bool)
    parser.add_argument('--augs', default=[], type=str, nargs='*')
    parser.add_argument('--LRLP', default=False, type=bool,
                        help='low resolution lable processor')
    parser.add_argument('--msg', default=None, type=str,
                        help='a training msg')
    parser.add_argument('--norm', default=False, type=bool,
                        help='data norm')
    parser.add_argument('--is_photo_distor', default=False, type=bool,
                        help='photo_distor')
    parser.add_argument('--norm_mode', default=1, type=int)
    return parser


def tensor2numpy(img):
    return img.cpu().numpy().transpose((1, 2, 0))


def data_scale(img, lab, r, LRLP=False):
    if r == 100:
        return img, lab
    N, C, H, W = img.shape
    scaled_imgs = []

    # print(H,W)
    if LRLP:
        scaled_labs = lab
    else:
        scaled_labs = []
    # print(r/100)
    for i, l in zip(img, lab):
        scaled_img = mmcv.imrescale(tensor2numpy(
            i), r/100, interpolation='bilinear')
        if C == 1:
            scaled_img = scaled_img[np.newaxis, :, :]
        if C == 3:
            scaled_img = scaled_img.transpose((2, 0, 1))
        scaled_imgs.append(scaled_img)
        if not LRLP:
            scaled_lab = mmcv.imrescale(l.cpu().numpy().transpose(
                (1, 2, 0)), r/100, interpolation='nearest')
            scaled_labs.append(scaled_lab[np.newaxis, :, :])
    scaled_imgs = np.array(scaled_imgs)
    if LRLP:
        if r < 50:  # 如果下采样比例过小，则分两次下采样
            scaled_labs = F.fractional_max_pool2d(
                scaled_labs, output_ratio=0.5, kernel_size=2)
        scaled_labs = F.fractional_max_pool2d(
            scaled_labs, output_size=(scaled_imgs.shape[-2:]), kernel_size=2)
    # print(scaled_imgs.shape)

    # print(scaled_labs.shape)
    # scaled_labs=np.array(scaled_labs)
    # print(torch.tensor(scaled_imgs).shape)
    if not isinstance(scaled_labs, torch.Tensor):
        scaled_labs = np.array(scaled_labs)
        scaled_labs = torch.tensor(scaled_labs)
    return torch.tensor(scaled_imgs), scaled_labs


def data_rotate(img,angle):
    imgs = []
    for i in img:
        imgs.append(mmcv.imrotate((tensor2numpy(i),angle)))
        # print(imgs[0].shape)
    imgs = np.array(imgs).transpose((0, 3, 1, 2))
    return torch.tensor(imgs)


def data_flip(img):
    imgs = []
    for i in img:
        imgs.append(mmcv.imflip(tensor2numpy(i)))
        # print(imgs[0].shape)
    imgs = np.array(imgs).transpose((0, 3, 1, 2))
    return torch.tensor(imgs)


def select_model(args):
    if args.model == 'rcf':
        # print(fuse)
        return RCF(pretrained='vgg16convs.mat', dataflag=args.dataflag, fuse=args.fuse_num, short_cat=args.short_cat).cuda()
    elif args.model == 'convnext':
        return NextRCF().cuda()
    elif args.model == 'unet':
        if args.dataflag == "color":
            inc = 3
        else:
            inc = 1
        return UNet(inc, 1).cuda()
    elif args.model == 'hrnet_ocr':
        args.cfg = 'other_models/hr_models/seg_hrnet_ocr_w48_trainval_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484.yaml'
        # args.opts = None
        update_config(config, args)
        model = eval('other_models.hr_models.'+config.MODEL.NAME +
                     '.get_seg_model')(config)
        # model = FullModel(model, criterion)
    elif args.model == 'lawin':
        model = Lawin(backbone = 'MiT-B5', num_classes=1)
        model.init_pretrained('pretrained_models/mit_b5.pth')
        return model.cuda()


class EvalMax:
    def __init__(self) -> None:
        self.max_eval = {}
        self.updated = False

    def __call__(self, ret, i=None):
        for k in ret.keys():  # 通过键，批量对比大小
            if 'pl' in k and ret[k] > self.max_eval.setdefault(k, 0):
                self.updated = True
                self.max_eval[k] = ret[k]
                if i:
                    self.max_eval[k+'_epo'] = i
        return self.max_eval

    def hasupdate(self):
        if self.updated:
            self.updated = False
            return True


class Logger(object):
    def __init__(self, path='log.txt'):
        self.logger = logging.getLogger('Logger')
        self.file_handler = logging.FileHandler(path, 'w')
        self.stdout_handler = logging.StreamHandler()
        self.logger.addHandler(self.file_handler)
        self.logger.addHandler(self.stdout_handler)
        self.stdout_handler.setFormatter(logging.Formatter(
            '%(asctime)s %(levelname)s %(message)s'))
        self.file_handler.setFormatter(logging.Formatter(
            '%(asctime)s %(levelname)s %(message)s'))
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
