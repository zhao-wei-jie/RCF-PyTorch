import io
from scipy.io import savemat
from glob import glob
import cv2,numpy as np
import json
import matplotlib.pyplot as plt
import os.path as osp
# alist=glob('results/RCF20220112_1910/epoch1-test/*ss.png')
# print(len(alist))
# key = "result"
# for i in alist:
#     image=cv2.imread(i,cv2.IMREAD_GRAYSCALE).astype(np.float_)/255
#     print(image.shape)
#     savemat(i.replace('.png','.mat'), {key: image})

def log2dict(line):
    eval={}
    alist=line.split(',')
    for idx,i in enumerate(alist):
        
        i=i.split(':')
        try:
            eval[i[0].strip(' ').strip('\'')]=float(i[1])
        except ValueError:
            continue
            eval[i[0].strip(' ').strip('\'')]=i[1].strip(' ').strip('\'')
    return eval

def old_log(f,evallist):
        # print(idx)
    for idx,line in enumerate(f):
        if(idx%2==1): 
            # print(idx)
            continue
        
        line = line[:-1]
        # a=json.loads(res)
        line=line.strip('{').strip('}')

        eval=log2dict(line)
        # print(line      
        # print(eval)
        evallist.append(eval)
    # break
def new_log(f,evallist):
    start=True #是否找出第一个epoch的标志，用于继续训练时找到中断的epoch
    for idx,line in enumerate(f):
            eval=None
            if 'aAcc' in line:
                line=line.split('{')[-1].strip('}')
                eval=log2dict(line)
            if 'Epoch' in line:
                lr=line.split('lr ')[-1].strip('\n')
                eval={}
                eval['lr']=float(lr)
                if start:
                    epoch=line.split('/')[0].split('[')[-1]
                    start=False
                    print('start_epoch',epoch)
                    eval['epoch']=int(epoch)
            if eval:
                evallist.append(eval)

def py_log(paths,mode='list'):
    if isinstance(paths,str):
        mode='list'
        paths=[paths]
    plt.figure(figsize=(10, 5),dpi=200)
    ax = plt.gca()
    # ax.set_xticks(plot_epochs)
    # label = legend[i * num_metrics + j]
    # if metric in ['mIoU', 'mAcc', 'aAcc']:
    
    for path in paths:
        if mode=='list':     
            plt.figure(figsize=(10, 5),dpi=200)
            ax = plt.gca()
        evallist=[]

        with open(path) as f:
            if osp.basename(path)=='train.log':
                print('new')
                new_log(f,evallist)
            elif osp.basename(path)=='test.log':
                print('old')
                old_log(f,evallist)
        print(path)
        iou_values,f1_values,lr_v,star_epoch=draw_log(evallist,path,ax)
        label=osp.dirname(path).split('bs-8')[-1].strip('-').strip('optadamw')
        ioulable='iou-'+label
        lrlable='lr-'+label
        #只记录到性能最高点后10个
        max_iou_pos=np.argmax(iou_values)
        iou_values=iou_values[:max_iou_pos+10]
        f1_values=f1_values[:max_iou_pos+10]
        lr_v=lr_v[:max_iou_pos+10]
        plot_epochs=range(star_epoch,max_iou_pos+10+star_epoch)
        
        # iou_values.sort()
        # f1_values.sort()
        def plt_combo():
            plt.legend()
            plt.xticks(rotation=45)#x坐标数字倾斜角度
            plt.grid()#画出网格
        
        # plt.xlabel('epoch')
        ax1=plt.subplot(2,1,1)
        plt.plot(plot_epochs, iou_values, label=ioulable)
        if mode=='list':  
            plt.plot(plot_epochs, f1_values, label='f1')
            plt.text(plot_epochs[np.argmax(f1_values)], max(f1_values), '%.4f'%(max(f1_values))+'-'+str(np.argmax(f1_values)+star_epoch))#写出最高点值
        
        
        # print(max_iou_pos,iou_values)
        plt.text(plot_epochs[max_iou_pos], max(iou_values),'%.4f'%(max(iou_values))+'-'+str(max_iou_pos+star_epoch))#写出最高点值
        
        x=[star_epoch]
        maxi=0
        for idx,i in enumerate(iou_values):
            if i>maxi:
                maxi=i
                if idx+star_epoch-x[-1]>=10 and max_iou_pos-idx>=10:       
                    x.append(idx+star_epoch)
        x.append(max_iou_pos+star_epoch)
        plt.xticks(x)
        plt_combo()
        plt.ylabel('score')
        # print(int(min(iou_values)*10),int(max(f1_values)*10)+1)
        plt.yticks([y/10 for y in range(int(min(iou_values)*10),int(max(f1_values)*10)+1)])#按iou最小值到f1最大值动态划分
        if osp.basename(path)=='train.log':
            assert len(iou_values)==len(lr_v),(len(iou_values),len(lr_v))
            plt.subplot(212,sharex=ax1)
            plt.plot(plot_epochs, lr_v, label=lrlable)
            plt.ylabel('lr')
            plt_combo()
        if mode=='list':            
            out=osp.dirname(path)
            print(f'save curve to: {out}')
            plt.savefig(osp.join(out,'fig'))
            plt.cla()
    if mode == 'sum':
        out='results'
        print(f'save curve to: {out}')
        plt.savefig(osp.join(out,'sum'))
        plt.cla()


    
# else:
#     plt.xlabel('iter')
#     plt.plot(plot_iters, plot_values, label=label, linewidth=0.5)
#     plt.text(plot_iters[np.argmax(plot_values)], max(plot_values), max(plot_values))#写出最高点值
#     ax.set_xticks(plot_iters)#x坐标显示的数值
#     # plt.xticks(range(0,max(plot_iters),8000))
#     plt.legend()
#     # plt.subplots_adjust(bottom=0.2) #设置底部宽度
#     plt.xticks(rotation=45)#x坐标数字倾斜角度
#     plt.grid()#画出网格
# if args.title is not None:
#     plt.title(args.title)
def draw_log(evallist,path,ax):
    iou_values=[]
    f1_values=[]
    lr_v=[]
    start_epoch=0
    # print(len(evallist))
    for i in evallist:
        # tmp=[i['IoU.pl'],i['Fscore.pl']]
        # print(i)
        if 'IoU.pl' in i.keys():
            iou_values.append(i['IoU.pl'])
        if 'Fscore.pl' in i.keys():
            f1_values.append(i['Fscore.pl'])
        elif 'f1' in i.keys():
            f1_values.append(i['f1'])
        if 'lr' in i.keys():
            lr_v.append(i['lr'])
        if 'epoch' in i.keys():
            start_epoch=i['epoch']
    return iou_values,f1_values,lr_v,start_epoch
import argparse
parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--pos', default=-1, type=int, help='pos')
args = parser.parse_args()
alist=glob('results/*_*-bs-*/*.log')
alist.sort()
print(alist)
py_log(alist[args.pos:],mode='list')
# py_log('results/RCF20220122_1759-bs-8-lr-0.002-iter_size-1-opt-adamw/train.log')
# alist=glob('results/RCF20220119_2123-bs-8-lr-0.03125-iter_size-10-opt-adamw/*.pth')
# for i in alist:
#     print(i)