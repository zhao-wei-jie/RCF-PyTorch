
from datetime import date
import os
# os.environ['']='0'#指定训练gpu
import numpy as np
import os.path as osp
import cv2
import argparse
import torch
from torch.utils.data import DataLoader
import torchvision
from dataset import BSDS_Dataset, TTPLA_Dataset
from models import RCF
from other_models.unet_model import UNet
import mmcv
from tqdm import tqdm
from glob import glob
from PIL import Image
import logging
from torch.utils.data import Dataset
from torchvision import transforms
import sys
from utils import EvalMax, select_model, argsF,data_scale
from other_models.hr_models.seg_hrnet_ocr import HighResolutionNet
import torch.nn.functional as F


class inferImage(Dataset):
    def __init__(self, img_dir, args):
        self.infer_list = glob(img_dir)
        self.file_list = self.infer_list
        self.dataflag = args.dataflag
        self.norm = args.norm
        self.norm_mode = args.norm_mode
        self.transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(512)
        ])
        self.mean = np.array([62.364, 61.025, 56.462], dtype=np.float32)
        if self.norm:
            self.std = np.array([25.631, 25.461, 25.943], dtype=np.float32)

    def __len__(self):
        return len(self.infer_list)

    def __getitem__(self, idx):
        # print(idx)
        img_path = self.infer_list[idx]
        # image = read_image(img_path)
        img = mmcv.imread(img_path, self.dataflag)
        if self.dataflag == 'color':
            # img=self.transforms(Image.open(img_path).convert("RGB"))
            # image = np.asarray(img)
            if self.norm:
                if self.norm_mode == 2:
                    img = img/255.0
                else:
                # gray_weight = np.array([0.299, 0.587, 0.114], dtype=np.float32)
                # img = mmcv.imnormalize(img, (self.mean*gray_weight).sum(), (self.std*gray_weight).sum())
                    img = mmcv.imnormalize(img, self.mean.mean(), np.ones(1))
            img = np.einsum('ijk->kij', img)
        else:
            # img=Image.open(img_path)
            # image = np.asarray(img)
            if self.norm:
                if self.norm_mode == 2:
                    img = img/255.0
                else:
                    gray_weight = np.array([0.299, 0.587, 0.114], dtype=np.float32)
                    img = mmcv.imnormalize(img, (self.mean*gray_weight).sum(), (self.std*gray_weight).sum())
            img = img[np.newaxis, :, :]

        image = torch.tensor(img)
        return image.float(), osp.basename(osp.dirname(img_path))


def single_scale_test(model, test_loader, test_list, save_dir, args, eval=None, save_img=False, use_amp=False):
    # print(save_img)
    model.eval()
    eval_res = []
    eval_label = []
    test_scale = [100]
    if  args.scale:  # 添加两个尺度
        test_scale.append(50)
        test_scale.append(25)
    if save_dir and not osp.isdir(save_dir):
        os.makedirs(save_dir)
    timer = mmcv.Timer()

    for scale in test_scale:
        print(scale, 'scale')
        per_time = 0.0
        for idx, data in enumerate(test_loader):
            if eval:
                image, label = data
                image, label = data_scale(image, label, scale ,True)
                label = label.cuda()
                label = label.squeeze()
                label = label.cpu().numpy()
                eval_label.append(label)
                name = ['ttpla']
            else:
                image, name = data
            image = image.cuda()

            H, W = image.shape[-2:]
            with torch.cuda.amp.autocast(enabled=use_amp):
                timer.since_last_check()
                results = model(image)

            if isinstance(model, UNet):
                results = torch.sigmoid(results)
                fuse_res = torch.squeeze(results.detach())
            else:
                results = [torch.sigmoid(r) for r in results]
                fuse_res = torch.squeeze(results[-1].detach())
            per_time += timer.since_last_check()
            temp_res = torch.zeros_like(fuse_res)
            temp_res[fuse_res > 0.5] = 1

            # print(results.shape)

            filename = osp.splitext(test_list[idx])[0]
            if save_img is True:
                image = image[0]
                if image.shape[0] == 1:
                    image = torch.cat([image]*3, dim=0)
                # print(image.shape)               
                image[:, fuse_res > 0.5] /= 2
                image[2, fuse_res > 0.5] += 128
                if args.norm_mode == 2:
                    image[2, fuse_res > 0.5] /= 255.0
                    image *= 255.0
                # fuse_res = fuse_res.cpu().numpy()
                # fuse_res = ((1 - fuse_res) * 255).astype(np.uint8)

                mmcv.imwrite(image.cpu().numpy().transpose((1, 2, 0)), osp.join(
                    save_dir, str(scale),'ss', name[0] , '%s_ss.png' % filename))
            if save_img is True:
                all_res = torch.zeros((len(results), 1, H, W))
                for i in range(len(results)):
                    all_res[i, 0, :, :] = results[i]
                if os.path.exists(osp.join(save_dir, str(scale), name[0])) is not True:
                    os.mkdir(osp.join(save_dir, str(scale), name[0], ))
                torchvision.utils.save_image(
                    1 - all_res, osp.join(save_dir, str(scale), name[0], '%s.jpg' % filename))

            # print(temp_res.shape,label.shape)

            eval_res.append(temp_res.cpu().numpy())
            # print(np.sum(eval_res==1),np.sum(eval_res==0),np.sum(label==1),np.sum(label==0))
            # sys.exit()
            #print('\rRunning single-scale test [%d/%d]' % (idx + 1, len(test_loader)), end='')
        print('fps', 1/(per_time/len(test_list)), per_time /
            len(test_list), per_time, len(test_list))
    print('Running single-scale test done')
    if eval:
        # eval_res.
        ret = eval(results=eval_res, gt_seg_maps=eval_label,
                   metric=['mIoU', 'mFscore'])
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
            im_ = cv2.resize(
                in_, None, fx=scale[k], fy=scale[k], interpolation=cv2.INTER_LINEAR)
            im_ = im_.transpose((2, 0, 1))
            results = model(torch.unsqueeze(torch.from_numpy(im_).cuda(), 0))
            fuse_res = torch.squeeze(results[-1].detach()).cpu().numpy()
            fuse_res = cv2.resize(
                fuse_res, (W, H), interpolation=cv2.INTER_LINEAR)
            ms_fuse += fuse_res
        ms_fuse = ms_fuse / len(scale)
        # rescale trick
        # ms_fuse = (ms_fuse - ms_fuse.min()) / (ms_fuse.max() - ms_fuse.min())
        filename = osp.splitext(test_list[idx])[0]
        ms_fuse = ((1 - ms_fuse) * 255).astype(np.uint8)
        cv2.imwrite(osp.join(save_dir, '%s_ms.png' % filename), ms_fuse)
        #print('\rRunning multi-scale test [%d/%d]' % (idx + 1, len(test_loader)), end='')
    print('Running multi-scale test done')


if __name__ == '__main__':

    parser = argsF()
    parser.add_argument('--checkpoint', default=None,
                        type=str, help='path to latest checkpoint')
    args = parser.parse_args()
    if osp.isfile(args.checkpoint):
        print("=> loading checkpoint from '{}'".format(args.checkpoint))
        checkpoint = torch.load(args.checkpoint)
        args_ = checkpoint['args']
        model = select_model(args_)
        model.load_state_dict(checkpoint['state_dict'])
        print("=> checkpoint loaded")
    else:
        print("=> no checkpoint found at '{}'".format(args.checkpoint))
    print('Called with args:')
    for (key, value) in vars(args_).items():
        print('{0:15} | {1}'.format(key, value))
    args.save_dir = osp.join(osp.splitext(args.checkpoint)[0])

    if not osp.isdir(args.save_dir):
        os.makedirs(args.save_dir)
    if args.dataset:
        #   test_dataset  = BSDS_Dataset(root=args.dataset, split='test')
        test_dataset = inferImage(img_dir=args.dataset, args=args_)
    else:
        test_dataset = TTPLA_Dataset(split='eval', args=args_)

    test_loader = DataLoader(test_dataset, batch_size=1,
                             num_workers=1, drop_last=False, shuffle=False)
    test_list = [osp.split(i.rstrip())[1] for i in test_dataset.file_list]
    assert len(test_list) == len(test_loader)

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
        max_eval = EvalMax()
        pth_list = glob(args.checkpoint+'/*.pth')

        print('Performing %d testing...' % (len(pth_list)))
        for i in pth_list:
            checkpoint = torch.load(i)
            try:
                model.load_state_dict(checkpoint['state_dict'])
            except RuntimeError as err:
                print(i, err)
                continue

            ret = single_scale_test(model, test_loader, test_list, args.save_dir,
                                    save_img=False, use_amp=True, eval=test_dataset.evaluate, args=args_)
            logger.info(ret)
            logger.info(max_eval(ret, i))
    else:
        print('Performing the testing...')
        # 使用getattr获取函数，可在函数不存在时返回none
        single_scale_test(model, test_loader, test_list, args.save_dir, eval=getattr(
            test_dataset, 'evaluate', None), save_img=True, use_amp=True, args=args_)
        # multi_scale_test(model, test_loader, test_list, args.save_dir)
