
from datetime import date
import os
os.environ['CUDA_VISIBLE_DEVICES']='2'#指定训练gpu
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
from utils import EvalMax,select_model
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
        img=Image.open(img_path)
        # img=self.transforms(Image.open(img_path).convert("RGB"))
        image = np.asarray(img)
        image = image[np.newaxis, :, :]
        # image=np.einsum('ijk->kij',image)
        image=torch.tensor(image)
        return image.float()

def single_scale_test(model, test_loader, test_list, save_dir,eval=None,save_img=True,use_amp=False):
    model.eval()
    eval_res=[]
    eval_label=[]

    if save_dir and not osp.isdir(save_dir):
        os.makedirs(save_dir)
    for idx, data in enumerate(test_loader):
        if eval:
            image,label=data
            label=label.cuda()
            label=label.squeeze()
            label=label.cpu().numpy()
            eval_label.append(label)
        else:
            image=data
        image = image.cuda()
        
        _, _, H, W = image.shape
        with torch.cuda.amp.autocast(enabled=use_amp):
            results = model(image)
        results = [torch.sigmoid(r) for r in results]
        # print(results.shape)
        
        filename = osp.splitext(test_list[idx])[0]
        if save_img:
            all_res = torch.zeros((len(results), 1, H, W))
            for i in range(len(results)):
                all_res[i, 0, :, :] = results[i]
            torchvision.utils.save_image(1 - all_res, osp.join(save_dir, '%s.jpg' % filename))
        fuse_res = torch.squeeze(results[-1].detach())
        temp_res=torch.zeros_like(fuse_res)        
        temp_res[fuse_res>0.5]=1
        
        fuse_res = fuse_res.cpu().numpy()       
        eval_res.append(temp_res.cpu().numpy())
        # print(np.sum(eval_res==1),np.sum(eval_res==0),np.sum(label==1),np.sum(label==0))       
        # sys.exit()
        fuse_res = ((1 - fuse_res) * 255).astype(np.uint8)
        if save_img:
            cv2.imwrite(osp.join(save_dir, '%s_ss.png' % filename), fuse_res)
        #print('\rRunning single-scale test [%d/%d]' % (idx + 1, len(test_loader)), end='')
    print('Running single-scale test done')
    if eval: 
        # eval_res.   
        ret=eval(results=eval_res,gt_seg_maps=eval_label,metric=['mIoU', 'mFscore'])
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
    parser.add_argument('--dataset', help='root folder of dataset', default=None)
    parser.add_argument('--model', default='rcf', type=str, help='rcf')
    parser.add_argument('--dataflag', default='color',help='color or grayscale')
    args = parser.parse_args()
    
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    
    if args.save_dir =='results/RCF':
        args.save_dir=osp.join(osp.splitext(args.checkpoint)[0])
    if not osp.isdir(args.save_dir):
        os.makedirs(args.save_dir)
    if args.dataset:
    #   test_dataset  = BSDS_Dataset(root=args.dataset, split='test')
      test_dataset  = inferImage(img_dir=args.dataset)
    else:
        test_dataset=TTPLA_Dataset(split='eval',dataflag=args.dataflag)
    
    test_loader   = DataLoader(test_dataset, batch_size=1, num_workers=1, drop_last=False, shuffle=False)
    test_list = [osp.split(i.rstrip())[1] for i in test_dataset.file_list]
    assert len(test_list) == len(test_loader)

    # model = RCF().cuda()
    model = select_model(args.model,args.dataflag)
    if osp.isfile(args.checkpoint):
        print("=> loading checkpoint from '{}'".format(args.checkpoint))
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint['state_dict'])
        print("=> checkpoint loaded")
    else:
        print("=> no checkpoint found at '{}'".format(args.checkpoint))
    
    if osp.isdir(args.checkpoint):
        logging.basicConfig(level=logging.NOTSET)
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        fh = logging.FileHandler(args.checkpoint+'/test.log', mode='w')
        fh.setLevel(logging.INFO)
        logger.addHandler(fh)
        
        
        # max_eval['acc']=0
        # max_eval['acc_epo']=''
        # max_eval['iou']=0
        # max_eval['iou_epo']=''
        # max_eval['preci']=0
        # max_eval['preci_epo']=''
        # max_eval['recall']=0
        # max_eval['recall_epo']=''
        max_eval=EvalMax()
        pth_list=glob(args.checkpoint+'/*.pth')
        
        print('Performing %d testing...'%(len(pth_list)))
        for i in tqdm(pth_list):
            checkpoint = torch.load(i)
            try:
                model.load_state_dict(checkpoint['state_dict'])
            except RuntimeError as err:
                print(i,err)
                continue

            ret=single_scale_test(model, test_loader, test_list,   args.save_dir,test_dataset.evaluate,False)
            logger.info(ret)
            logger.info(max_eval(ret,i))
    else:
        print('Performing the testing...')
        #使用getattr获取函数，可在函数不存在时返回none       
        single_scale_test(model, test_loader, test_list, args.save_dir,getattr(test_dataset,'evaluate',None))        
        # multi_scale_test(model, test_loader, test_list, args.save_dir)
