import torch
import cv2
import numpy as np
import os.path as osp
import PIL.Image as Image
from torch import serialization
import mmcv
from mmcv.utils import print_log
from prettytable import PrettyTable
from torch.utils.data import Dataset
from collections import OrderedDict
import torchvision
from random import randint
import sys

from mmseg.core import eval_metrics, intersect_and_union, pre_eval_to_metrics

class BSDS_Dataset(torch.utils.data.Dataset):
    def __init__(self, root='data/HED-BSDS', split='test', transform=False):
        super(BSDS_Dataset, self).__init__()
        self.root = root
        self.split = split
        self.transform = transform
        if self.split == 'train':
            self.file_list = osp.join(self.root, 'bsds_pascal_train_pair.lst')
        elif self.split == 'test':
            self.file_list = osp.join(self.root, 'test.lst')
        else:
            raise ValueError('Invalid split type!')
        with open(self.file_list, 'r') as f:
            self.file_list = f.readlines()
        self.mean = np.array([104.00698793, 116.66876762, 122.67891434], dtype=np.float32)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        if self.split == 'train':
            img_file, label_file = self.file_list[index].split()
            label = cv2.imread(osp.join(self.root, label_file), 0)
            label = np.array(label, dtype=np.float32)
            label = label[np.newaxis, :, :]
            label[label == 0] = 0
            label[np.logical_and(label > 0, label < 127.5)] = 2
            label[label >= 127.5] = 1
        else:
            img_file = self.file_list[index].rstrip()

        img = cv2.imread(osp.join(self.root, img_file))
        img = np.array(img, dtype=np.float32)
        img = (img - self.mean).transpose((2, 0, 1))

        if self.split == 'train':
            return img, label
        else:
            return img

class TTPLA_Dataset(torch.utils.data.Dataset):
    def __init__(self, root='../ttpla/', split='test',dataflag='color'):
        super(TTPLA_Dataset, self).__init__()
        self.root = root
        self.split = split
        self.dataflag=dataflag
        self.transform =torchvision.transforms.Compose([
            torchvision.transforms.Resize(384)        
            ])
        self.CLASSES=['bg','pl']
        if self.split == 'train':
            self.file_list = osp.join(self.root, 'ttpla_train.txt')
        elif self.split in ['test','eval']:
            self.file_list = osp.join(self.root, 'ttpla_val.txt')
            
        else:
            raise ValueError('Invalid split type!')
        with open(self.file_list, 'r') as f:
            self.file_list = f.readlines()
            print(self.split,len(self.file_list))
        self.mean = np.array([62.364 ,61.025 ,56.462], dtype=np.float32)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        r=randint(96,384)
 
            
        if self.split in ['train','eval']:
            img_file= self.file_list[index]
            label = mmcv.imread(osp.join(self.root,'annpng_powerline', img_file.strip('\n')+'.png'),backend='pillow',flag='unchanged')
            # print(label.size,(label==1).sum(),(label==0).sum())
            label=self.transf(label,r)#缩放至最大训练尺寸
            
            # print(2,label.shape)
            label = label[np.newaxis, :, :]
            
                        
            # label[label == 0] = 0
            # label[np.logical_and(label > 0, label < 127.5)] = 2
            # label[label >= 127.5] = 1
            # print(label.shape,label.dtype)
            # sys.exit(0)
            label = np.array(label, dtype=np.float32)
        else:
            img_file = self.file_list[index]

        img =  mmcv.imread(osp.join(self.root, img_file.strip('\n')+'.jpg'),self.dataflag)
        img=self.transf(img,r)
        # img=rrisize(img)
        # img = np.array(img, dtype=np.float32)
        if self.dataflag=='color':
            img = (img - self.mean).transpose((2, 0, 1))
        if self.dataflag=='grayscale':
            img = img - self.mean.mean()
            img = img[np.newaxis, :, :]
        
        if self.split in ['train','eval']:
            return img, label
        else:
            return img
    
    def transf(self,img , r):
        h, w = img.shape[:2]
        scale=h/540
        # print(w//scale)        
        img=mmcv.imresize(img,(int(w//scale),540))#保持比例将高度控制在540

        # W,H=img.size#获取尺寸信息
        # # print(1,label.size)            
        # img=rrisize(img)#随机缩放
        # img = np.array(img, dtype=np.float32)#转numpy
        # # print(2,label.shape)
        # img=mmcv.impad(img,shape=(H,W))#填充至最大训练尺寸
        return img
    
    def evaluate(self,
                 results,
                 metric='mIoU',
                 logger=None,
                 gt_seg_maps=None,
                 **kwargs):
        """Evaluate the dataset.

        Args:
            results (list[tuple[torch.Tensor]] | list[str]): per image pre_eval
                 results or predict segmentation map for computing evaluation
                 metric.
            metric (str | list[str]): Metrics to be evaluated. 'mIoU',
                'mDice' and 'mFscore' are supported.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.
            gt_seg_maps (generator[ndarray]): Custom gt seg maps as input,
                used in ConcatDataset

        Returns:
            dict[str, float]: Default metrics.
        """
        if isinstance(metric, str):
            metric = [metric]
        allowed_metrics = ['mIoU', 'mDice', 'mFscore']
        if not set(metric).issubset(set(allowed_metrics)):
            raise KeyError('metric {} is not supported'.format(metric))

        eval_results = {}
        self.ignore_index=255
        # test a list of files
        if mmcv.is_list_of(results, np.ndarray) or mmcv.is_list_of(
                results, str) or mmcv.is_list_of(results, torch.Tensor):
            if gt_seg_maps is None:
                gt_seg_maps = self.get_gt_seg_maps()
            num_classes = 2
            ret_metrics = eval_metrics(
                results,
                gt_seg_maps,
                num_classes,
                self.ignore_index,
                metric,
                # label_map=self.label_map,
                # reduce_zero_label=self.reduce_zero_label
                )
        # test a list of pre_eval_results
        else:
            ret_metrics = pre_eval_to_metrics(results, metric)

        # Because dataset.CLASSES is required for per-eval.
        if self.CLASSES is None:
            class_names = tuple(range(num_classes))
        else:
            class_names = self.CLASSES

        # summary table
        ret_metrics_summary = OrderedDict({
            ret_metric: np.round(np.nanmean(ret_metric_value) * 100, 2)
            for ret_metric, ret_metric_value in ret_metrics.items()
        })

        # each class table
        ret_metrics.pop('aAcc', None)
        ret_metrics_class = OrderedDict({
            ret_metric: np.round(ret_metric_value * 100, 2)
            for ret_metric, ret_metric_value in ret_metrics.items()
        })
        ret_metrics_class.update({'Class': class_names})
        ret_metrics_class.move_to_end('Class', last=False)

        # for logger
        class_table_data = PrettyTable()
        for key, val in ret_metrics_class.items():
            class_table_data.add_column(key, val)

        summary_table_data = PrettyTable()
        for key, val in ret_metrics_summary.items():
            if key == 'aAcc':
                summary_table_data.add_column(key, [val])
            else:
                summary_table_data.add_column('m' + key, [val])

        print_log('per class results:', logger)
        print_log('\n' + class_table_data.get_string(), logger=logger)
        print_log('Summary:', logger)
        print_log('\n' + summary_table_data.get_string(), logger=logger)

        # each metric dict
        for key, value in ret_metrics_summary.items():
            if key == 'aAcc':
                eval_results[key] = value / 100.0
            else:
                eval_results['m' + key] = value / 100.0

        ret_metrics_class.pop('Class', None)
        for key, value in ret_metrics_class.items():
            eval_results.update({
                key + '.' + str(name): value[idx] / 100.0
                for idx, name in enumerate(class_names)
            })

        return eval_results
