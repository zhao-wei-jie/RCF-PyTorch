import torch
import os.path as osp
pth='results/LAWIN20220321_2044-bs-2-lr-0.0001-dataflag-color-aug-False/checkpoint_epoch232.pth'
basedir=osp.dirname(pth)
ckpoint=torch.load(pth)
torch.save(ckpoint['state_dict'],basedir+'/'+osp.basename(pth).replace('checkpoint_epoch','model'))