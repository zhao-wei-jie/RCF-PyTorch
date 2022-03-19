import os
# os.environ['CUDA_VISIBLE_DEVICES']='0'#指定训练gpu
import numpy as np
import os.path as osp
import cv2
import time
import torch
from torch.utils.data import DataLoader
import torchvision
from dataset import BSDS_Dataset, TTPLA_Dataset
from other_models.unet_model import UNet
from other_models.hr_models.seg_hrnet_ocr import HighResolutionNet
from utils import Logger, Averagvalue, Cross_entropy_loss, EvalMax, select_model, argsF, data_scale,tensor2numpy
from test import single_scale_test
import random
# from apex import amp
import torch.nn.functional as F
import sys
import mmcv


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def mosaic(img):
    N, C, H, W = img.shape
    mosaic_img = torch.zeros((int(N//2), C, H, int(W//2)))
    for i in range(0, N, 2):  # 拼接样本，缩小电线,只用于灰度图
        cat_img = torch.cat((img[i, 0], img[i+1, 0]), dim=0)
        # 只接受前2维为h*w,故输入在前需要换维
        cat_img = mmcv.imrescale(cat_img.cpu().numpy(), 0.5)
        # print(cat_img.shape)
        mosaic_img[int(i//2), 0, :, :] = torch.from_numpy(cat_img)[:, :]
    return mosaic_img


def data_flip(img):
    imgs = []
    for i in img:
        imgs.append(mmcv.imflip(tensor2numpy(i)))
        # print(imgs[0].shape)
    imgs = np.array(imgs).transpose((0, 3, 1, 2))
    return torch.tensor(imgs)


def data_aug(img, label, augs=['mosaic']):

    if 'mosaic' in augs:
        mosaic_img = mosaic(img).cuda()
        mosaic_label = mosaic(label).cuda()

        img = torch.cat(torch.chunk(img, 2, dim=3), dim=0)  # 将图片分给两边，视为两个样本
        label = torch.cat(torch.chunk(label, 2, dim=3), dim=0)

        img = torch.cat((img, mosaic_img), dim=0)
        label = torch.cat((label, mosaic_label), dim=0)
    if 'scale' in augs:
        r = random.randrange(40, 80)
        img, label = data_scale(img, label, r)
    # print(mosaic_img.shape)

        # torch.
    # print(mosaic_img.shape)

    return img, label


def train(args, model, train_loader, optimizer, epoch, logger, scaler, use_amp):
    batch_time = Averagvalue()
    losses = Averagvalue()
    model.train()
    end = time.time()
    counter = 0
    for i, (image, label) in enumerate(train_loader):
        # image, label = image.cuda(), label.cuda()
        def training(image_, label_, end, counter):
            with torch.cuda.amp.autocast(enabled=use_amp):
                # 混合精度输入不可以是双精度,会无法转换
                outputs = model(image_)
                loss = torch.zeros(1).cuda()

                if isinstance(model, UNet):
                    loss += Cross_entropy_loss(outputs, label_)
                if isinstance(model, HighResolutionNet):
                    label_=  F.fractional_max_pool2d(
                                                    label_, output_size=(outputs[0].shape[-2:]), kernel_size=2)
                    loss += Cross_entropy_loss(outputs[0], label_)
                if hasattr(model, 'short_cat'):
                    if model.short_cat == 2:
                        loss += Cross_entropy_loss(outputs[-1], label_)
                    else:
                        for o in outputs:
                            loss = loss + Cross_entropy_loss(o, label_)
                counter += 1
                loss = loss / args.iter_size
            scaler.scale(loss).backward()
            # loss.backward()
            if counter == args.iter_size:
                scaler.step(optimizer)
                # optimizer.step()
                scaler.update()
                optimizer.zero_grad()
                counter = 0
            # measure accuracy and record loss
            losses.update(loss.item(), image.size(0))
            batch_time.update(time.time() - end)
            end = time.time()

        if args.aug:
            image, label = data_aug(image, label)
        # print(image.shape)
        # sys.exit(0)
        train_scale = [100]
        if args.scale:  # 添加两个尺度
            train_scale.append(50)
            train_scale.append(25)
        for s in train_scale:
            # logger.info('training scale %d'%(s))
            image_, label_ = data_scale(image, label, s, args.LRLP)
            image_, label_ = image_.cuda(), label_.cuda()
            training(image_, label_, end, counter)
            for aug in args.augs:
                if aug == 'flip':
                    # logger.info('training flip')
                    image__ = data_flip(image_).cuda()
                    label__ = data_flip(label_).cuda()
                    training(image__, label__, end, counter)

    logger.info('Epoch: [{0}/{1}][{2}/{3}] '.format(epoch + 1, args.max_epoch, i, len(train_loader)) +
                'Time {batch_time.val:.3f} (avg: {batch_time.avg:.3f}) '.format(batch_time=batch_time) +
                'Loss {loss.val:f} (avg: {loss.avg:f}) '.format(loss=losses) +
                'lr %e' % (get_lr(optimizer)))


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
    logger.info('Running multi-scale test done')


if __name__ == '__main__':

    parser = argsF()
    args = parser.parse_args()
    if args.model == 'unet':
        args.short_cat = None
        args.fuse_num = None
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    # os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    ex_time = time.strftime('%Y%m%d_%H%M', time.localtime())
    args.save_dir = args.save_dir+str.upper(args.model)+ex_time
    # by Accurate, Large Minibatch SGD:Training ImageNet in 1 Hour,lr=base_lr*gpus*bs/256
    # args.lr=args.lr*(args.batch_size*args.iter_size/256)
    save_dict = dict(bs=args.batch_size, lr=args.lr,
                     dataflag=args.dataflag, aug=args.aug)
    # print(save_dict)
    for k, v in save_dict.items():
        args.save_dir += '-'+k+'-'+str(v)
    if not osp.isdir(args.save_dir):
        os.makedirs(args.save_dir)

    logger = Logger(osp.join(args.save_dir, 'train.log'))

    # train_dataset = BSDS_Dataset(root=args.dataset, split='train')
    # test_dataset  = BSDS_Dataset(root=osp.join(args.dataset, 'HED-BSDS'), split='test')

    train_dataset = TTPLA_Dataset(
        split='train', args=args)
    test_dataset = TTPLA_Dataset(
        split='eval', args=args)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, num_workers=8, drop_last=True, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             num_workers=8, drop_last=False, shuffle=False)
    test_list = [i for i in test_dataset.file_list]
    # assert len(test_list) == len(test_loader) #,print(len(test_list) ,len(test_loader),len(train_loader))
    model = select_model(args)
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.stepsize, gamma=args.gamma)
    if args.opt == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    if args.opt == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr)
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.stepsize, gamma=args.gamma)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=args.gamma, threshold_mode='abs')
    use_amp = False
    if args.amp == 'O1':
        logger.info('use amp')
        use_amp = True
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    def load_w(pth, mode):
        if osp.isfile(pth):
            logger.info("=> loading checkpoint from '{}'".format(pth))
            checkpoint = torch.load(pth)
            model.load_state_dict(checkpoint['state_dict'])
            if mode == 'resume':
                optimizer.load_state_dict(checkpoint['optimizer'])
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
                args.start_epoch = checkpoint['epoch'] + 1
                if 'scaler' in checkpoint.keys():
                    logger.info("=>  loaded scaler")
                    scaler.load_state_dict(checkpoint['scaler'])
            logger.info("=> checkpoint loaded")
        else:
            logger.info("=> no checkpoint found at '{}'".format(pth))
    if args.resume is not None:
        load_w(args.resume, 'resume')
    if args.pretrain:
        load_w(args.pretrain, 'pretrain')
    # else:
    #     model.load_state_dict(torch.load('bsds500_pascal_model.pth'))
    logger.info('Called with args:')
    for (key, value) in vars(args).items():
        logger.info('{0:15} | {1}'.format(key, value))
    max_eval = EvalMax()
    for epoch in range(args.start_epoch, args.max_epoch):
        logger.info('training...')
        train(args, model, train_loader, optimizer,
              epoch, logger, scaler, use_amp)
        # save_dir = osp.join(args.save_dir, 'epoch%d-test' % (epoch + 1))
        logger.info('testing...')
        ret = single_scale_test(
            model, test_loader, test_list, None, args,
            test_dataset.evaluate, False, use_amp)
        logger.info(ret)
        logger.info(max_eval(ret, epoch+1))
        # multi_scale_test(model, test_loader, test_list, save_dir)
        # Save checkpoint
        save_file = osp.join(
            args.save_dir, 'checkpoint_epoch{}.pth'.format(epoch + 1))
        if max_eval.hasupdate():
            torch.save({
                'epoch': epoch,
                'args': args,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'scaler': scaler.state_dict()
            }, save_file)
        if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            lr_scheduler.step(ret['IoU.pl'])
        else:
            lr_scheduler.step()  # will adjust learning rate

    logger.close()
