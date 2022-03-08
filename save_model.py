import torch
import os.path as osp
pth='results/RCF20220225_0949-bs-4-lr-0.002-iter_size-1-opt-adamw/checkpoint_epoch642.pth'
basedir=osp.dirname(pth)
ckpoint=torch.load(pth)
torch.save(ckpoint['state_dict'],basedir+'/'+osp.basename(pth).replace('checkpoint_epoch','model'))