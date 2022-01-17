import os
import numpy as np
import os.path as osp
import cv2
import argparse
from scipy.ndimage.measurements import label
import torch
from torch.utils.data import DataLoader
import torchvision
from dataset import BSDS_Dataset,TTPLA_Dataset
from models import RCF

from tqdm import tqdm
from glob import glob
from PIL import Image
import logging
from torch.utils.data import Dataset
from torchvision import transforms
import sys
class inferImage(Dataset):
    def __init__(self, img_dir, cnum=None, aug=False,transform=None, target_transform=None):
        self.infer_list=glob(img_dir)
        self.file_list=self.infer_list
        self.transforms=torchvision.transforms.Compose([
        torchvision.transforms.Resize(512)        
        ])

    def __len__(self):
        return len(self.infer_list)

    def __getitem__(self, idx):
        # print(idx)
        img_path = self.infer_list[idx]
        # image = read_image(img_path)
        img=self.transforms(Image.open(img_path).convert("RGB"))
        image = np.asarray(img)
        image=np.einsum('ijk->kij',image)
        image=torch.tensor(image)
        return image.float()

def single_scale_test(model, test_loader, test_list, save_dir,eval,save_img=True):
    model.eval()
    eval_res=[]
    eval_label=[]
    all_preci=0.0
    all_acc=0.0
    if not osp.isdir(save_dir):
        os.makedirs(save_dir)
    for idx, data in enumerate(test_loader):
        image,label=data
        image = image.cuda()
        label=label.cuda()
        _, _, H, W = image.shape
        results = model(image)
        all_res = torch.zeros((len(results), 1, H, W))
        for i in range(len(results)):
          all_res[i, 0, :, :] = results[i]
        filename = osp.splitext(test_list[idx])[0]
        if save_img:
            torchvision.utils.save_image(1 - all_res, osp.join(save_dir, '%s.jpg' % filename))
        fuse_res = torch.squeeze(results[-1].detach())
        temp_res=torch.zeros_like(fuse_res)        
        temp_res[fuse_res>0.5]=1
        label=label.squeeze()
        TP=((temp_res==1)==(label==1)).sum()
        T_FP=(temp_res==1).sum()
        all_acc+=(temp_res==label).sum()/label.sum()
        all_preci+=TP/T_FP
        fuse_res = fuse_res.cpu().numpy()
        label=label.cpu().numpy()

        eval_label.append(label)
        eval_res.append(temp_res.cpu().numpy())
        # print(np.sum(eval_res==1),np.sum(eval_res==0),np.sum(label==1),np.sum(label==0))       
        # sys.exit()
        fuse_res = ((1 - fuse_res) * 255).astype(np.uint8)
        if save_img:
            cv2.imwrite(osp.join(save_dir, '%s_ss.png' % filename), fuse_res)
        #print('\rRunning single-scale test [%d/%d]' % (idx + 1, len(test_loader)), end='')
        
    ret=eval(results=eval_res,gt_seg_maps=eval_label)
    ret['preci']=all_preci/(idx+1)
    ret['recall']=all_acc/(idx+1)
    print('Running single-scale test done')
    return ret


def multi_scale_test(model, test_loader, test_list, save_dir):
    model.eval()
    if not osp.isdir(save_dir):
        os.makedirs(save_dir)
    scale = [0.5, 1, 1.5]
    for idx, image in enumerate(test_loader):
        in_ = image[0].numpy().transpose((1, 2, 0))
        _, _, H, W = image.shape
        ms_fuse = np.zeros((H, W), np.float32)
        for k in range(len(scale)):
            im_ = cv2.resize(in_, None, fx=scale[k], fy=scale[k], interpolation=cv2.INTER_LINEAR)
            im_ = im_.transpose((2, 0, 1))
            results = model(torch.unsqueeze(torch.from_numpy(im_).cuda(), 0))
            fuse_res = torch.squeeze(results[-1].detach()).cpu().numpy()
            fuse_res = cv2.resize(fuse_res, (W, H), interpolation=cv2.INTER_LINEAR)
            ms_fuse += fuse_res
        ms_fuse = ms_fuse / len(scale)
        ### rescale trick
        # ms_fuse = (ms_fuse - ms_fuse.min()) / (ms_fuse.max() - ms_fuse.min())
        filename = osp.splitext(test_list[idx])[0]
        ms_fuse = ((1 - ms_fuse) * 255).astype(np.uint8)
        cv2.imwrite(osp.join(save_dir, '%s_ms.png' % filename), ms_fuse)
        #print('\rRunning multi-scale test [%d/%d]' % (idx + 1, len(test_loader)), end='')
    print('Running multi-scale test done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Testing')
    parser.add_argument('--gpu', default='0', type=str, help='GPU ID')
    parser.add_argument('--checkpoint', default=None, type=str, help='path to latest checkpoint')
    parser.add_argument('--save-dir', help='output folder', default='results/RCF')
    parser.add_argument('--dataset', help='root folder of dataset', default='data/HED-BSDS')
    args = parser.parse_args()
    
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    
    if args.save_dir =='results/RCF':
        args.save_dir=osp.join(osp.splitext(args.checkpoint)[0])
    if not osp.isdir(args.save_dir):
        os.makedirs(args.save_dir)
  
    # test_dataset  = BSDS_Dataset(root=args.dataset, split='test')
    test_dataset=TTPLA_Dataset(split='eval')
    # test_dataset  = inferImage(img_dir=args.dataset)
    test_loader   = DataLoader(test_dataset, batch_size=1, num_workers=2, drop_last=False, shuffle=False)
    test_list = [osp.split(i.rstrip())[1] for i in test_dataset.file_list]
    assert len(test_list) == len(test_loader)

    model = RCF().cuda()

    if osp.isfile(args.checkpoint):
        print("=> loading checkpoint from '{}'".format(args.checkpoint))
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint['state_dict'])
        print("=> checkpoint loaded")
    else:
        print("=> no checkpoint found at '{}'".format(args.checkpoint))
    
    if osp.isdir(args.checkpoint):
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        fh = logging.FileHandler(args.checkpoint+'/test_log.txt', mode='w')
        fh.setLevel(logging.INFO)
        logger.addHandler(fh)
        
        max_eval={}
        max_eval['acc']=0
        max_eval['acc_epo']=''
        max_eval['iou']=0
        max_eval['iou_epo']=''
        max_eval['preci']=0
        max_eval['preci_epo']=''
        max_eval['recall']=0
        max_eval['recall_epo']=''

        pth_list=glob(args.checkpoint+'/*.pth')
        
        print('Performing %d testing...'%(len(pth_list)))
        for i in tqdm(pth_list):
            checkpoint = torch.load(i)
            model.load_state_dict(checkpoint['state_dict'])
            ret=single_scale_test(model, test_loader, test_list,   args.save_dir,test_dataset.evaluate,False)
            if ret['IoU.pl']>max_eval['iou']:
                max_eval['iou']=ret['IoU.pl']
                max_eval['iou_epo']=i
            if ret['Acc.pl']>max_eval['acc']:
                max_eval['acc']=ret['Acc.pl']
                max_eval['acc_epo']=i
            if ret['preci']>max_eval['preci']:
                max_eval['preci']=ret['preci']
                max_eval['preci_epo']=i
            if ret['recall']>max_eval['recall']:
                max_eval['recall']=ret['recall']
                max_eval['recall_epo']=i 
            # print()
            logger.info(ret)
            logger.info(max_eval)
    else:
        print('Performing the testing...')
        single_scale_test(model, test_loader, test_list, args.save_dir,test_dataset.evaluate)
        # multi_scale_test(model, test_loader, test_list, args.save_dir)